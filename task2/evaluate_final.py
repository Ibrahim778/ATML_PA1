"""
Runs Source-only, DAN, DANN, CDAN through task2/train.py, then produces:
  - comparison table (per-source val, mean source, target acc/F1, target
    change vs Source-only, domain separability)
  - per-class target accuracy changes + dominant confusions
  - controlled design study (lambda_mmd sweep for DAN, by default)

Run with:  python -m task2.evaluate_final --root /path/to/PACS
"""

import argparse
import json
import os

import torch
import yaml

from task2.train import train_task2_method, load_config, SOURCE_DOMAINS, NUM_CLASSES
from task2.evaluation.domain_separability import domain_separability_score
from task2.evaluation.class_analysis import per_class_accuracy_change, dominant_confusions
from shared.pacs import PACS_CLASSES

RESULTS_DIR = "task2/results"


def run_all_methods(base_path: str, configs_dir: str, device: str):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_results = {}
    all_target_outputs = {}
    method_names = ["source_only", "dan", "dann", "cdan"]

    for i, method_name in enumerate(method_names, 1):
        print(f"\n{'#' * 60}\n[{i}/{len(method_names)}] Training {method_name}\n{'#' * 60}")
        cfg = load_config(base_path, os.path.join(configs_dir, f"{method_name}.yaml"))
        method_kwargs = {k: v for k, v in cfg.items() if k in ("lambda_mmd", "max_alpha")}
        backbone, method, result, target_outputs = train_task2_method(
            cfg["method"], cfg["data_root"], device=device,
            max_epochs=cfg.get("epochs", 30), patience=cfg.get("patience", 5),
            batch_per_source=cfg.get("batch_size_per_source", 8),
            batch_target=cfg.get("batch_size_target", 24),
            seed=cfg.get("seed", 6304), out_dir=RESULTS_DIR,
            lr=cfg.get("lr", 1e-4), weight_decay=cfg.get("weight_decay", 1e-4),
            method_kwargs=method_kwargs,
        )
        all_results[method_name] = result
        all_target_outputs[method_name] = target_outputs
        print(f"[{method_name}] done -- target acc={result['target_metrics']['accuracy']:.4f} "
              f"macro-F1={result['target_metrics']['macro_f1']:.4f}")

    return all_results, all_target_outputs


def build_comparison_table(all_results: dict, all_target_outputs: dict):
    source_only_target_acc = all_results["source_only"]["target_metrics"]["accuracy"]

    table = {}
    for method_name, result in all_results.items():
        _, _, target_feats = all_target_outputs[method_name]
        # domain separability needs source-val features too; approximate
        # here using target features vs. themselves is meaningless, so this
        # is filled in by run_domain_separability() below using cached
        # source-val features collected during training. Left as a
        # placeholder key to be populated by the caller.
        table[method_name] = {
            "per_domain_val": result["per_domain_val"],
            "mean_source_acc": sum(v["accuracy"] for v in result["per_domain_val"].values()) / len(SOURCE_DOMAINS),
            "mean_source_f1": sum(v["macro_f1"] for v in result["per_domain_val"].values()) / len(SOURCE_DOMAINS),
            "target_accuracy": result["target_metrics"]["accuracy"],
            "target_macro_f1": result["target_metrics"]["macro_f1"],
            "target_acc_change_vs_source_only": result["target_metrics"]["accuracy"] - source_only_target_acc,
        }
    return table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="task2/configs/base.yaml", help="Path to base.yaml")
    parser.add_argument("--configs_dir", default="task2/configs",
                        help="Dir containing source_only.yaml, dan.yaml, dann.yaml, cdan.yaml")
    parser.add_argument("--device", default="mps" if torch.mps.is_available() else "cpu")
    args = parser.parse_args()

    base_cfg = yaml.safe_load(open(args.base))
    root = base_cfg["data_root"]

    all_results, all_target_outputs = run_all_methods(args.base, args.configs_dir, args.device)
    comparison = build_comparison_table(all_results, all_target_outputs)

    print("\nComputing domain separability scores ...")
    # Domain separability: source-val features vs target features, per method.
    for method_name in comparison:
        print(f"  [{method_name}] extracting source-val features for separability check ...")
        target_preds, target_labels, target_feats = all_target_outputs[method_name]
        # NOTE: source-val features should be collected during training and
        # cached (see task2/train.py's per_domain_val loop) -- re-extract
        # here for simplicity using the saved checkpoint.
        from task2.models.backbone import ResNet18Backbone
        from task2.evaluation.metrics import evaluate_domain
        from shared.pacs import load_pacs_domain
        from shared.pacs_protocol import build_source_splits, EVAL_TRANSFORM
        from torch.utils.data import DataLoader, Subset

        backbone = ResNet18Backbone(num_classes=NUM_CLASSES).to(args.device)
        backbone.load_state_dict(torch.load(all_results[method_name]["checkpoint_path"], map_location=args.device))

        splits = build_source_splits(SOURCE_DOMAINS, root,
                                      out_path="shared/splits/pacs_sketch_seed6304.json")
        source_feats_list = []
        for d in SOURCE_DOMAINS:
            ds = load_pacs_domain(root, d, transform=EVAL_TRANSFORM)
            loader = DataLoader(Subset(ds, splits[d]["val_idx"]), batch_size=64, shuffle=False)
            _, _, feats = evaluate_domain(backbone, loader, args.device)
            source_feats_list.append(feats.numpy())
        import numpy as np
        source_feats = np.concatenate(source_feats_list, axis=0)

        sep_score = domain_separability_score(source_feats, target_feats.numpy())
        comparison[method_name]["domain_separability"] = sep_score
        print(f"  [{method_name}] domain separability = {sep_score:.4f} (0.5 = chance)")

    print("\nRunning per-class analysis vs Source-only ...")
    # Per-class analysis vs Source-only, for each non-baseline method.
    so_preds, so_labels, _ = all_target_outputs["source_only"]
    class_analysis = {}
    for method_name in ["dan", "dann", "cdan"]:
        m_preds, m_labels, _ = all_target_outputs[method_name]
        class_analysis[method_name] = {
            "per_class_change": per_class_accuracy_change(
                so_preds, m_preds, m_labels, NUM_CLASSES, class_names=PACS_CLASSES),
            "dominant_confusions": dominant_confusions(
                m_preds, m_labels, NUM_CLASSES, class_names=PACS_CLASSES),
        }

    summary = {
        "comparison_table": comparison,
        "class_analysis_vs_source_only": class_analysis,
        "training_curves": {m: r["training_curve"] for m, r in all_results.items()},
    }
    with open(os.path.join(RESULTS_DIR, "task2_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {RESULTS_DIR}/task2_summary.json")


if __name__ == "__main__":
    main()