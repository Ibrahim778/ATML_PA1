"""
Loads Sketch ONLY in this script, after every Task 3 checkpoint/setting has
been fixed via source-side validation. Produces the ERM/DAN-DG/SAM
comparison table, source-domain separability, sharpness proxy, and
per-class Sketch changes vs. ERM.

Run with:  python -m task3.evaluate_sketch --root /path/to/PACS \
             --erm_checkpoint task2/results/source_only_checkpoint.pt
"""

import argparse
import json
import os

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from shared.pacs import load_pacs_domain, PACS_CLASSES
from shared.pacs_protocol import build_source_splits, EVAL_TRANSFORM, SEED
from task2.models.backbone import ResNet18Backbone
from task2.evaluation.metrics import evaluate_domain, accuracy, macro_f1
from task2.evaluation.class_analysis import per_class_accuracy_change, dominant_confusions
from task3.methods.erm import load_erm_baseline
from task3.train import train_task3_method, load_config, result_path_for, SOURCE_DOMAINS, NUM_CLASSES
from task3.evaluation.source_domain_separability import source_domain_separability_score
from task3.evaluation.sharpness import build_fixed_sharpness_batch, compute_sharpness

RESULTS_DIR = "task3/results"


def collect_source_val_feats(backbone, root, splits, device):
    feats_by_domain = {}
    for d in SOURCE_DOMAINS:
        ds = load_pacs_domain(root, d, transform=EVAL_TRANSFORM)
        loader = DataLoader(Subset(ds, splits[d]["val_idx"]), batch_size=64, shuffle=False)
        _, _, feats = evaluate_domain(backbone, loader, device)
        feats_by_domain[d] = feats.numpy()
    return feats_by_domain


def train_or_load(method_name: str, cfg: dict, ckpt_path: str, run_fn, device: str):
    """If a checkpoint already exists at ckpt_path, load it straight into a
    fresh backbone (skipping training entirely), and load the matching
    {method_name}_result.json (saved by task3/train.py) for its training
    curve/summary instead of leaving that None. Otherwise, train via
    run_fn(cfg) as before."""
    if os.path.isfile(ckpt_path):
        print(f"[task3:{method_name}] Found existing checkpoint at {ckpt_path} -- loading, skipping training.")
        backbone = ResNet18Backbone(num_classes=NUM_CLASSES).to(device)
        backbone.load_state_dict(torch.load(ckpt_path, map_location=device))

        result_path = result_path_for(method_name, RESULTS_DIR)
        result = None
        if os.path.isfile(result_path):
            with open(result_path) as f:
                result = json.load(f)
            print(f"[task3:{method_name}] Loaded saved training result from {result_path}.")
        else:
            print(f"[task3:{method_name}] No saved result at {result_path} -- "
                  f"training_curve will be empty for this run.")
        return backbone, result
    return run_fn(cfg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="task2/configs/base.yaml", help="Path to base.yaml (shared with Task 2)")
    parser.add_argument("--configs_dir", default="task3/configs",
                         help="Dir containing dan_dg.yaml and sam.yaml")
    parser.add_argument("--erm_checkpoint", required=True,
                         help="Path to the Task 2 Source-only checkpoint")
    parser.add_argument("--device", default="mps" if torch.mps.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    base_cfg = yaml.safe_load(open(args.base))
    root = base_cfg["data_root"]

    dandg_cfg = load_config(args.base, os.path.join(args.configs_dir, "dan_dg.yaml"))
    sam_cfg = load_config(args.base, os.path.join(args.configs_dir, "sam.yaml"))

    def _run(cfg):
        method_kwargs = {k: v for k, v in cfg.items() if k in ("lambda_dg", "rho")}
        return train_task3_method(
            cfg["method"], cfg["data_root"], device=args.device,
            max_epochs=cfg.get("epochs", 30), patience=cfg.get("patience", 5),
            batch_per_source=cfg.get("batch_size_per_source", 8),
            seed=cfg.get("seed", SEED), out_dir=RESULTS_DIR,
            lr=cfg.get("lr", 1e-4), weight_decay=cfg.get("weight_decay", 1e-4),
            method_kwargs=method_kwargs,
        )

    # ---- Train DAN-DG and SAM (source-side only), reusing existing checkpoints if present ----
    dandg_ckpt = os.path.join(RESULTS_DIR, "dan_dg_checkpoint.pt")
    sam_ckpt = os.path.join(RESULTS_DIR, "sam_checkpoint.pt")

    print("\n" + "#" * 60 + "\n[1/2] DAN-DG (source-side only, no Sketch)\n" + "#" * 60)
    dandg_backbone, dandg_result = train_or_load("dan_dg", dandg_cfg, dandg_ckpt, _run, args.device)

    print("\n" + "#" * 60 + "\n[2/2] SAM (source-side only, no Sketch)\n" + "#" * 60)
    sam_backbone, sam_result = train_or_load("sam", sam_cfg, sam_ckpt, _run, args.device)

    print(f"\nLoading ERM baseline from {args.erm_checkpoint} (Task 2 Source-only checkpoint, unchanged) ...")
    erm_backbone = load_erm_baseline(args.erm_checkpoint, num_classes=NUM_CLASSES, device=args.device)

    backbones = {"erm": erm_backbone, "dan_dg": dandg_backbone, "sam": sam_backbone}

    # ---- NOW load Sketch, for the first time in Task 3 ----
    print("\nAll Task 3 checkpoints and settings are frozen -- loading Sketch for final evaluation ...")
    sketch_ds = load_pacs_domain(root, "sketch", transform=EVAL_TRANSFORM)
    sketch_loader = DataLoader(sketch_ds, batch_size=64, shuffle=False)

    splits = build_source_splits(SOURCE_DOMAINS, root,
                                  out_path="shared/splits/pacs_sketch_seed6304.json")

    comparison = {}
    sketch_outputs = {}
    for name, backbone in backbones.items():
        print(f"\nEvaluating [{name}] on source-val, Sketch, separability, and sharpness ...")
        val_summary_domains = {}
        for d in SOURCE_DOMAINS:
            ds = load_pacs_domain(root, d, transform=EVAL_TRANSFORM)
            loader = DataLoader(Subset(ds, splits[d]["val_idx"]), batch_size=64, shuffle=False)
            preds, labels, _ = evaluate_domain(backbone, loader, args.device)
            val_summary_domains[d] = {"accuracy": accuracy(preds, labels), "macro_f1": macro_f1(preds, labels)}

        mean_acc = sum(v["accuracy"] for v in val_summary_domains.values()) / len(SOURCE_DOMAINS)
        worst_acc = min(v["accuracy"] for v in val_summary_domains.values())
        mean_f1 = sum(v["macro_f1"] for v in val_summary_domains.values()) / len(SOURCE_DOMAINS)
        worst_f1 = min(v["macro_f1"] for v in val_summary_domains.values())

        sketch_preds, sketch_labels, sketch_feats = evaluate_domain(backbone, sketch_loader, args.device)
        sketch_outputs[name] = (sketch_preds, sketch_labels, sketch_feats)

        source_feats = collect_source_val_feats(backbone, root, splits, args.device)
        sep_score = source_domain_separability_score(source_feats)

        sharpness_batch_x, sharpness_batch_y = build_fixed_sharpness_batch(
            {d: load_pacs_domain(root, d, transform=EVAL_TRANSFORM) for d in SOURCE_DOMAINS}
        )
        sharpness = compute_sharpness(backbone, sharpness_batch_x, sharpness_batch_y, args.device)

        comparison[name] = {
            "per_domain_val": val_summary_domains,
            "mean_source_accuracy": mean_acc,
            "worst_source_accuracy": worst_acc,
            "mean_source_f1": mean_f1,
            "worst_source_f1": worst_f1,
            "sketch_accuracy": accuracy(sketch_preds, sketch_labels),
            "sketch_macro_f1": macro_f1(sketch_preds, sketch_labels),
            "source_domain_separability": sep_score,
            "sharpness_proxy": sharpness,
        }

    erm_sketch_acc = comparison["erm"]["sketch_accuracy"]
    for name in comparison:
        comparison[name]["sketch_accuracy_change_vs_erm"] = comparison[name]["sketch_accuracy"] - erm_sketch_acc

    # ---- Per-class Sketch changes vs ERM ----
    erm_preds, erm_labels, _ = sketch_outputs["erm"]
    class_analysis = {}
    for name in ["dan_dg", "sam"]:
        m_preds, m_labels, _ = sketch_outputs[name]
        class_analysis[name] = {
            "per_class_change": per_class_accuracy_change(
                erm_preds, m_preds, m_labels, NUM_CLASSES, class_names=PACS_CLASSES),
            "dominant_confusions": dominant_confusions(
                m_preds, m_labels, NUM_CLASSES, class_names=PACS_CLASSES),
        }

    summary = {
        "comparison_table": comparison,
        "class_analysis_vs_erm": class_analysis,
        # training_curve is [] for any method loaded from an existing checkpoint
        # rather than retrained this run.
        "training_curves": {
            "dan_dg": dandg_result["training_curve"] if dandg_result else [],
            "sam": sam_result["training_curve"] if sam_result else [],
        },
    }
    with open(os.path.join(RESULTS_DIR, "task3_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved {RESULTS_DIR}/task3_summary.json")


if __name__ == "__main__":
    main()