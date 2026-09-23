"""
Bounded controlled study (spec Step 5): vary EITHER lambda_dg in {0.1, 1, 10}
for DAN-DG, OR rho in {0.01, 0.05, 0.1} for SAM. The main comparison keeps
lambda_dg=1 / rho=0.05; Sketch results here are for analysis only and must
not be used to pick a different "winning" setting post hoc.

Run with:
    python -m task3.controlled_study --root /path/to/PACS --study dan_dg
    python -m task3.controlled_study --root /path/to/PACS --study sam
"""

import argparse
import json
import os

import torch
import yaml
from torch.utils.data import DataLoader

from shared.pacs import load_pacs_domain
from task2.evaluation.metrics import evaluate_domain, accuracy, macro_f1
from task3.train import train_task3_method, load_config

RESULTS_DIR = "task3/results"


def run_sweep(root: str, device: str, method_name: str, param_name: str, values, cfg: dict):
    rows = []
    sketch_ds = load_pacs_domain(root, "sketch", transform=None)  # transform applied inside loader below
    from shared.pacs_protocol import EVAL_TRANSFORM
    sketch_ds = load_pacs_domain(root, "sketch", transform=EVAL_TRANSFORM)
    sketch_loader = DataLoader(sketch_ds, batch_size=64, shuffle=False)

    for val in values:
        print(f"\n===== {method_name} {param_name}={val} =====")
        backbone, result = train_task3_method(
            method_name, root, device=device, out_dir=RESULTS_DIR,
            max_epochs=cfg.get("epochs", 30), patience=cfg.get("patience", 5),
            batch_per_source=cfg.get("batch_size_per_source", 8),
            seed=cfg.get("seed", 6304),
            lr=cfg.get("lr", 1e-4), weight_decay=cfg.get("weight_decay", 1e-4),
            method_kwargs={param_name: val},
        )
        preds, labels, _ = evaluate_domain(backbone, sketch_loader, device)
        rows.append({
            param_name: val,
            "mean_source_val_f1": result["best_mean_source_val_f1"],
            "sketch_accuracy": accuracy(preds, labels),
            "sketch_macro_f1": macro_f1(preds, labels),
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="task2/configs/base.yaml", help="Path to base.yaml (shared with Task 2)")
    parser.add_argument("--configs_dir", default="task3/configs",
                         help="Dir containing dan_dg.yaml and sam.yaml")
    parser.add_argument("--study", choices=["dan_dg", "sam"], required=True)
    parser.add_argument("--device", default="mps" if torch.mps.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    cfg = load_config(args.base, os.path.join(args.configs_dir, f"{args.study}.yaml"))
    root = cfg["data_root"]

    if args.study == "dan_dg":
        rows = run_sweep(root, args.device, "dan_dg", "lambda_dg", [0.1, 1.0, 10.0], cfg)
    else:
        rows = run_sweep(root, args.device, "sam", "rho", [0.01, 0.05, 0.1], cfg)

    out_path = os.path.join(RESULTS_DIR, f"controlled_study_{args.study}.json")
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()