"""
Orchestrates the full Task 4 pipeline:
  1. Build CIFAR-10 90/10 split (seed 6304)
  2. Train Vanilla (or load cached checkpoint)
  3. Fit Mahalanobis stats from unaugmented CIFAR-10 training features
  4. Cache Vanilla outputs on val/test/near/far
  5. Table 1: MSP/MLS/Energy/Mahalanobis on frozen Vanilla -- near/far/all AUROC
     + validation-calibrated rejection (95th percentile threshold)
  6. Train GCSC, cache outputs, evaluate with MLS
  7. Train PROSER (from the Vanilla checkpoint), cache outputs, evaluate with
     MLS (known-class logits only) AND the placeholder-based score
  8. Table 2: Vanilla/GCSC/PROSER comparison (CSA + near/far OSR)
  9. Failure analysis: incorrectly-accepted near/far unknowns under the
     Vanilla MLS threshold
  10. Save task4_summary.json

Run with:  python -m task4.evaluate_osr --cifar10_root /path --cifar100_root /path
"""

import argparse
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import Subset

from task4.config import load_config
from task4.data.cifar10 import (
    load_cifar10_datasets, EVAL_TRANSFORM, TRAIN_TRANSFORM, CIFAR10_CLASSES, build_gcsc_transform,
)
from task4.data.cifar100_unknowns import load_near_far_unknowns, NEAR_UNKNOWN_CLASSES, FAR_UNKNOWN_CLASSES
from task4.data.make_splits import build_and_save_splits
from task4.models.resnet_cifar import CIFARResNet18
from task4.methods.vanilla import train_vanilla_or_gcsc
from task4.methods.proser import train_proser
from task4.extract_outputs import cache_outputs
from task4.scores.msp import msp_score
from task4.scores.mls import mls_score
from task4.scores.energy import energy_score
from task4.scores.mahalanobis import fit_mahalanobis_stats, mahalanobis_score
from task4.evaluation.metrics import auroc, closed_set_accuracy
from task4.evaluation.thresholds import calibrate_threshold, acceptance_rate, fpr_at_95_tpr
from task4.evaluation.failure_analysis import find_incorrectly_accepted

RESULTS_DIR = "task4/results"
CACHE_DIR = "task4/cache"
CONFIGS_DIR = "task4/configs"
NUM_CLASSES = 10


def _stage(n: int, total: int, title: str):
    print(f"\n{'=' * 60}\n[Stage {n}/{total}] {title}\n{'=' * 60}", flush=True)


def get_or_train_vanilla(root, device, config_path):
    ckpt_path = os.path.join(RESULTS_DIR, "vanilla_checkpoint.pt")
    if os.path.exists(ckpt_path):
        print(f"[task4] Found cached Vanilla checkpoint at {ckpt_path}, loading.")
        model = CIFARResNet18(num_classes=NUM_CLASSES).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        return model, ckpt_path

    cfg = load_config(config_path)
    print(f"[task4:vanilla] Loaded config from {config_path}:\n{cfg}")
    opt_cfg = cfg.get("optimizer", {})

    train_idx, val_idx = build_and_save_splits(root)
    train_ds, train_ds_eval_view, _ = load_cifar10_datasets(root, transform_train=TRAIN_TRANSFORM)
    print("\n" + "#" * 60 + "\n[1/3] Training Vanilla\n" + "#" * 60)
    model, best_val_acc, ckpt_path = train_vanilla_or_gcsc(
        train_ds, train_ds_eval_view, train_idx, val_idx, "vanilla", device=device,
        epochs=cfg.get("epochs", 100), batch_size=cfg.get("batch_size", 128),
        lr=opt_cfg.get("lr", 0.1), momentum=opt_cfg.get("momentum", 0.9),
        weight_decay=opt_cfg.get("weight_decay", 5e-4), seed=cfg.get("seed", 6304),
        out_dir=RESULTS_DIR,
    )
    return model, ckpt_path


def get_or_train_gcsc(root, device, config_path):
    ckpt_path = os.path.join(RESULTS_DIR, "gcsc_checkpoint.pt")
    if os.path.exists(ckpt_path):
        print(f"[task4] Found cached GCSC checkpoint at {ckpt_path}, loading.")
        model = CIFARResNet18(num_classes=NUM_CLASSES).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        return model

    cfg = load_config(config_path)
    print(f"[task4:gcsc] Loaded config from {config_path}:\n{cfg}")
    opt_cfg = cfg.get("optimizer", {})
    ra_cfg = cfg.get("randaugment", {})
    gcsc_transform = build_gcsc_transform(num_ops=ra_cfg.get("num_ops", 2),
                                           magnitude=ra_cfg.get("magnitude", 9))

    train_idx, val_idx = build_and_save_splits(root)
    train_ds, train_ds_eval_view, _ = load_cifar10_datasets(root, transform_train=gcsc_transform)
    print("\n" + "#" * 60 + "\n[2/3] Training GCSC (RandAugment)\n" + "#" * 60)
    model, best_val_acc, ckpt_path = train_vanilla_or_gcsc(
        train_ds, train_ds_eval_view, train_idx, val_idx, "gcsc", device=device,
        epochs=cfg.get("epochs", 100), batch_size=cfg.get("batch_size", 128),
        lr=opt_cfg.get("lr", 0.1), momentum=opt_cfg.get("momentum", 0.9),
        weight_decay=opt_cfg.get("weight_decay", 5e-4), seed=cfg.get("seed", 6304),
        out_dir=RESULTS_DIR,
    )
    return model


def get_or_train_proser(root, device, config_path, vanilla_ckpt_path):
    ckpt_path = os.path.join(RESULTS_DIR, "proser_checkpoint.pt")
    if os.path.exists(ckpt_path):
        print(f"[task4] Found cached PROSER checkpoint at {ckpt_path}, loading.")
        model = CIFARResNet18(num_classes=NUM_CLASSES, num_dummy_classifiers=5).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        return model

    cfg = load_config(config_path)
    print(f"[task4:proser] Loaded config from {config_path}:\n{cfg}")
    opt_cfg = cfg.get("optimizer", {})

    train_idx, val_idx = build_and_save_splits(root)
    train_ds, train_ds_eval_view, _ = load_cifar10_datasets(root, transform_train=TRAIN_TRANSFORM)
    print("\n" + "#" * 60 + "\n[3/3] Fine-tuning PROSER (classifier + data placeholders)\n" + "#" * 60)
    model, best_val_acc, ckpt_path = train_proser(
        train_ds, train_ds_eval_view, train_idx, val_idx, vanilla_ckpt_path,
        device=device, epochs=cfg.get("epochs", 50), batch_size=cfg.get("batch_size", 128),
        num_known=cfg.get("num_known", 10), num_dummy=cfg.get("num_dummy", 5),
        beta=cfg.get("beta", 1.0), gamma=cfg.get("gamma", 0.1),
        lr=opt_cfg.get("lr", 1e-3), momentum=opt_cfg.get("momentum", 0.9),
        weight_decay=opt_cfg.get("weight_decay", 5e-4), seed=cfg.get("seed", 6304),
        out_dir=RESULTS_DIR,
    )
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cifar10_root", required=True)
    parser.add_argument("--cifar100_root", required=True)
    parser.add_argument("--device", default="mps" if torch.mps.is_available() else "cpu")
    parser.add_argument("--vanilla_config", default=os.path.join(CONFIGS_DIR, "vanilla.yaml"))
    parser.add_argument("--gcsc_config", default=os.path.join(CONFIGS_DIR, "gcsc.yaml"))
    parser.add_argument("--proser_config", default=os.path.join(CONFIGS_DIR, "proser.yaml"))
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    device = args.device
    TOTAL_STAGES = 8
    t_start = time.time()

    _stage(1, TOTAL_STAGES, "Data prep (CIFAR-10 split + CIFAR-100 near/far unknowns)")
    train_idx, val_idx = build_and_save_splits(args.cifar10_root)
    _, train_ds_eval_view, test_ds = load_cifar10_datasets(args.cifar10_root)
    val_ds = Subset(train_ds_eval_view, val_idx)
    train_unaugmented_ds = Subset(train_ds_eval_view, train_idx)  # for Mahalanobis stats

    print("Loading CIFAR-100 near/far unknown evaluation sets (test-only, fixed groups) ...")
    near_ds, far_ds = load_near_far_unknowns(args.cifar100_root)
    print(f"Near unknowns: {len(near_ds)} images ({NEAR_UNKNOWN_CLASSES})")
    print(f"Far unknowns: {len(far_ds)} images ({FAR_UNKNOWN_CLASSES})")

    # ================= 1. Vanilla =================
    _stage(2, TOTAL_STAGES, "Vanilla: train/load, Mahalanobis fit, output caching")
    vanilla_model, vanilla_ckpt = get_or_train_vanilla(args.cifar10_root, device, args.vanilla_config)

    print("\n[task4] Fitting Mahalanobis class-mean/covariance stats on unaugmented CIFAR-10 training features ...")
    train_outputs = cache_outputs(vanilla_model, train_unaugmented_ds, device,
                                   os.path.join(CACHE_DIR, "vanilla_train.pt"),
                                   desc="Vanilla features (train, for Mahalanobis fit)")
    class_means, inv_diag_cov = fit_mahalanobis_stats(train_outputs["features"], train_outputs["labels"])

    print("\n[task4] Caching Vanilla outputs on val/test/near/far ...")
    vanilla_val = cache_outputs(vanilla_model, val_ds, device, os.path.join(CACHE_DIR, "vanilla_val.pt"),
                                 desc="Vanilla outputs (val)")
    vanilla_test = cache_outputs(vanilla_model, test_ds, device, os.path.join(CACHE_DIR, "vanilla_test.pt"),
                                  desc="Vanilla outputs (test)")
    vanilla_near = cache_outputs(vanilla_model, near_ds, device, os.path.join(CACHE_DIR, "vanilla_near.pt"),
                                  desc="Vanilla outputs (near unknown)")
    vanilla_far = cache_outputs(vanilla_model, far_ds, device, os.path.join(CACHE_DIR, "vanilla_far.pt"),
                                 desc="Vanilla outputs (far unknown)")

    def compute_all_scores(outputs):
        logits, feats = outputs["logits"], outputs["features"]
        return {
            "msp": msp_score(logits).numpy(),
            "mls": mls_score(logits).numpy(),
            "energy": energy_score(logits).numpy(),
            "mahalanobis": mahalanobis_score(feats, class_means, inv_diag_cov).numpy(),
        }

    print("\n[task4] Computing MSP/MLS/Energy/Mahalanobis on val/test/near/far ...")
    val_scores = compute_all_scores(vanilla_val)
    test_scores = compute_all_scores(vanilla_test)
    near_scores = compute_all_scores(vanilla_near)
    far_scores = compute_all_scores(vanilla_far)

    test_preds = vanilla_test["logits"].argmax(dim=1).numpy()
    test_labels = vanilla_test["labels"].numpy()
    vanilla_csa = closed_set_accuracy(test_preds, test_labels)
    print(f"[task4] Vanilla closed-set test accuracy: {vanilla_csa:.4f}")

    # ---- Table 1: score comparison on frozen Vanilla ----
    _stage(3, TOTAL_STAGES, "Table 1: MSP/MLS/Energy/Mahalanobis comparison on frozen Vanilla")
    score_comparison = {}
    for score_name in ["msp", "mls", "energy", "mahalanobis"]:
        threshold = calibrate_threshold(val_scores[score_name])
        row = {
            "auroc_near": auroc(test_scores[score_name], near_scores[score_name]),
            "auroc_far": auroc(test_scores[score_name], far_scores[score_name]),
            "auroc_all": auroc(test_scores[score_name],
                                np.concatenate([near_scores[score_name], far_scores[score_name]])),
            "threshold": threshold,
            "test_acceptance_rate": acceptance_rate(test_scores[score_name], threshold),
            "fpr95_near": fpr_at_95_tpr(near_scores[score_name], threshold),
            "fpr95_far": fpr_at_95_tpr(far_scores[score_name], threshold),
        }
        score_comparison[score_name] = row
        print(f"  [{score_name}] AUROC(near)={row['auroc_near']:.4f}  AUROC(far)={row['auroc_far']:.4f}  "
              f"AUROC(all)={row['auroc_all']:.4f}")

    # ================= 2. GCSC =================
    _stage(4, TOTAL_STAGES, "GCSC: train/load (RandAugment) + output caching")
    gcsc_model = get_or_train_gcsc(args.cifar10_root, device, args.gcsc_config)
    gcsc_val = cache_outputs(gcsc_model, val_ds, device, os.path.join(CACHE_DIR, "gcsc_val.pt"),
                              desc="GCSC outputs (val)")
    gcsc_test = cache_outputs(gcsc_model, test_ds, device, os.path.join(CACHE_DIR, "gcsc_test.pt"),
                               desc="GCSC outputs (test)")
    gcsc_near = cache_outputs(gcsc_model, near_ds, device, os.path.join(CACHE_DIR, "gcsc_near.pt"),
                               desc="GCSC outputs (near unknown)")
    gcsc_far = cache_outputs(gcsc_model, far_ds, device, os.path.join(CACHE_DIR, "gcsc_far.pt"),
                              desc="GCSC outputs (far unknown)")

    gcsc_val_mls = mls_score(gcsc_val["logits"]).numpy()
    gcsc_test_mls = mls_score(gcsc_test["logits"]).numpy()
    gcsc_near_mls = mls_score(gcsc_near["logits"]).numpy()
    gcsc_far_mls = mls_score(gcsc_far["logits"]).numpy()
    gcsc_threshold = calibrate_threshold(gcsc_val_mls)
    gcsc_preds = gcsc_test["logits"].argmax(dim=1).numpy()
    gcsc_csa = closed_set_accuracy(gcsc_preds, gcsc_test["labels"].numpy())

    # ================= 3. PROSER =================
    _stage(5, TOTAL_STAGES, "PROSER: fine-tune from Vanilla checkpoint + output caching")
    proser_model = get_or_train_proser(args.cifar10_root, device, args.proser_config, vanilla_ckpt)
    proser_val = cache_outputs(proser_model, val_ds, device, os.path.join(CACHE_DIR, "proser_val.pt"),
                                has_dummy=True, desc="PROSER outputs (val)")
    proser_test = cache_outputs(proser_model, test_ds, device, os.path.join(CACHE_DIR, "proser_test.pt"),
                                 has_dummy=True, desc="PROSER outputs (test)")
    proser_near = cache_outputs(proser_model, near_ds, device, os.path.join(CACHE_DIR, "proser_near.pt"),
                                 has_dummy=True, desc="PROSER outputs (near unknown)")
    proser_far = cache_outputs(proser_model, far_ds, device, os.path.join(CACHE_DIR, "proser_far.pt"),
                                has_dummy=True, desc="PROSER outputs (far unknown)")

    # PROSER + MLS (known-class logits only, for direct comparability)
    proser_val_mls = mls_score(proser_val["logits"]).numpy()
    proser_test_mls = mls_score(proser_test["logits"]).numpy()
    proser_near_mls = mls_score(proser_near["logits"]).numpy()
    proser_far_mls = mls_score(proser_far["logits"]).numpy()
    proser_mls_threshold = calibrate_threshold(proser_val_mls)
    proser_preds = proser_test["logits"].argmax(dim=1).numpy()
    proser_csa = closed_set_accuracy(proser_preds, proser_test["labels"].numpy())  # known-logits only

    # PROSER + placeholder-based score
    def full_probs_unknown_score(outputs, num_known=NUM_CLASSES):
        full_logits = torch.cat([outputs["logits"], outputs["dummy_logits"]], dim=1)
        probs = torch.softmax(full_logits, dim=1)
        return (1.0 - probs[:, :num_known].sum(dim=1)).numpy()

    proser_val_ph = full_probs_unknown_score(proser_val)
    proser_test_ph = full_probs_unknown_score(proser_test)
    proser_near_ph = full_probs_unknown_score(proser_near)
    proser_far_ph = full_probs_unknown_score(proser_far)
    proser_ph_threshold = calibrate_threshold(proser_val_ph)

    # ---- Table 2: Vanilla vs GCSC vs PROSER(MLS) vs PROSER(placeholder) ----
    _stage(6, TOTAL_STAGES, "Table 2: Vanilla / GCSC / PROSER comparison (CSA + near/far OSR)")
    def osr_row(test_s, near_s, far_s, threshold):
        return {
            "auroc_near": auroc(test_s, near_s),
            "auroc_far": auroc(test_s, far_s),
            "auroc_all": auroc(test_s, np.concatenate([near_s, far_s])),
            "threshold": threshold,
            "test_acceptance_rate": acceptance_rate(test_s, threshold),
            "fpr95_near": fpr_at_95_tpr(near_s, threshold),
            "fpr95_far": fpr_at_95_tpr(far_s, threshold),
        }

    model_comparison = {
        "vanilla_mls": {"csa": vanilla_csa,
                        **osr_row(test_scores["mls"], near_scores["mls"], far_scores["mls"],
                                  score_comparison["mls"]["threshold"])},
        "gcsc_mls": {"csa": gcsc_csa,
                     **osr_row(gcsc_test_mls, gcsc_near_mls, gcsc_far_mls, gcsc_threshold)},
        "proser_mls": {"csa": proser_csa,
                       **osr_row(proser_test_mls, proser_near_mls, proser_far_mls, proser_mls_threshold)},
        "proser_placeholder": {"csa": proser_csa,
                               **osr_row(proser_test_ph, proser_near_ph, proser_far_ph, proser_ph_threshold)},
    }
    print("\n[task4] Model comparison (CSA + near/far OSR):")
    for name, row in model_comparison.items():
        print(f"  [{name}] CSA={row['csa']:.4f}  AUROC(all)={row['auroc_all']:.4f}")

    # ---- Failure analysis: vanilla MLS threshold, near + far ----
    _stage(7, TOTAL_STAGES, "Failure analysis (incorrectly accepted unknowns)")
    print("Running failure analysis (Vanilla MLS threshold) ...")
    near_labels_fine = [near_ds.dataset.targets[i] for i in near_ds.indices]
    far_labels_fine = [far_ds.dataset.targets[i] for i in far_ds.indices]
    near_fine_to_local = sorted(set(near_labels_fine))
    far_fine_to_local = sorted(set(far_labels_fine))
    near_local_labels = [near_fine_to_local.index(l) for l in near_labels_fine]
    far_local_labels = [far_fine_to_local.index(l) for l in far_labels_fine]
    near_class_names_ordered = [near_ds.dataset.classes[i] for i in near_fine_to_local]
    far_class_names_ordered = [far_ds.dataset.classes[i] for i in far_fine_to_local]

    near_preds_on_unknown = vanilla_near["logits"].argmax(dim=1).numpy()
    far_preds_on_unknown = vanilla_far["logits"].argmax(dim=1).numpy()

    failures_near = find_incorrectly_accepted(
        near_scores["mls"], near_preds_on_unknown, near_local_labels,
        score_comparison["mls"]["threshold"], CIFAR10_CLASSES, near_class_names_ordered)
    failures_far = find_incorrectly_accepted(
        far_scores["mls"], far_preds_on_unknown, far_local_labels,
        score_comparison["mls"]["threshold"], CIFAR10_CLASSES, far_class_names_ordered)

    summary = {
        "vanilla_score_comparison": score_comparison,
        "model_comparison": model_comparison,
        "failure_analysis": {"near": failures_near, "far": failures_far},
    }
    _stage(8, TOTAL_STAGES, "Saving results")
    with open(os.path.join(RESULTS_DIR, "task4_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    elapsed = time.time() - t_start
    print(f"\n{'=' * 60}\nDone in {elapsed / 60:.1f} min. "
          f"Summary written to {RESULTS_DIR}/task4_summary.json\n{'=' * 60}")


if __name__ == "__main__":
    main()
