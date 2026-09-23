"""
Bounded controlled study (spec Step 6): vary EITHER lambda_mmd in {0.1, 1, 10}
for DAN, OR the max gradient-reversal strength in {0.25, 0.5, 1} for DANN,
keeping every other setting fixed. The main comparison in evaluate_final.py
still uses lambda_mmd=1 / max GRL strength=1; this script is for analysis
only and must not be used to pick a different "winning" setting post hoc.

Run with:
    python -m task2.controlled_study --root /path/to/PACS --study dan
    python -m task2.controlled_study --root /path/to/PACS --study dann
"""

import argparse
import json
import os

import torch

from task2.train import train_task2_method

RESULTS_DIR = "task2/results"


def run_dan_lambda_sweep(root: str, device: str, values=(0.1, 1.0, 10.0)):
    rows = []
    for lam in values:
        print(f"\n===== DAN lambda_mmd={lam} =====")
        _, _, result, _ = train_task2_method(
            "dan", root, device=device, out_dir=RESULTS_DIR,
            method_kwargs={"lambda_mmd": lam},
        )
        rows.append({
            "lambda_mmd": lam,
            "mean_source_val_f1": result["best_mean_source_val_f1"],
            "target_accuracy": result["target_metrics"]["accuracy"],
            "target_macro_f1": result["target_metrics"]["macro_f1"],
        })
    return rows


class _DANNWithMaxAlpha:
    """Wraps DANN so the GRL alpha schedule is capped at `max_alpha` instead
    of the default 1.0, for the controlled GRL-strength study."""
    def __init__(self, feature_dim, num_classes, device, max_alpha=1.0):
        from task2.methods.dann import DANN
        self._inner = DANN(feature_dim, num_classes, device)
        self.max_alpha = max_alpha
        self.name = "dann"

    def extra_parameters(self):
        return self._inner.extra_parameters()

    def compute_loss(self, backbone, source_batches, target_batch, progress_p):
        from shared.grl import grl_alpha_schedule
        self._inner.grl.alpha = min(grl_alpha_schedule(progress_p), self.max_alpha)
        return self._inner.compute_loss(backbone, source_batches, target_batch, progress_p)


def run_dann_grl_sweep(root: str, device: str, values=(0.25, 0.5, 1.0)):
    import task2.train as train_module
    rows = []
    for max_alpha in values:
        print(f"\n===== DANN max GRL alpha={max_alpha} =====")
        # Temporarily register the capped-alpha variant under the "dann" key.
        original_cls = train_module.METHOD_REGISTRY["dann"]
        train_module.METHOD_REGISTRY["dann"] = lambda fd, nc, dev, **kw: _DANNWithMaxAlpha(
            fd, nc, dev, max_alpha=max_alpha)
        try:
            _, _, result, _ = train_task2_method(
                "dann", root, device=device, out_dir=RESULTS_DIR,
            )
        finally:
            train_module.METHOD_REGISTRY["dann"] = original_cls

        rows.append({
            "max_grl_alpha": max_alpha,
            "mean_source_val_f1": result["best_mean_source_val_f1"],
            "target_accuracy": result["target_metrics"]["accuracy"],
            "target_macro_f1": result["target_metrics"]["macro_f1"],
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--study", choices=["dan", "dann"], required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    if args.study == "dan":
        rows = run_dan_lambda_sweep(args.root, args.device)
    else:
        rows = run_dann_grl_sweep(args.root, args.device)

    out_path = os.path.join(RESULTS_DIR, f"controlled_study_{args.study}.json")
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
