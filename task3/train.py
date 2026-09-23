"""
Trains DAN-DG and SAM using the identical source-only protocol from Task 2
(same splits, backbone init, augmentation, optimizer settings, budget,
seed) but WITHOUT any Sketch access. ERM is not retrained here -- it's the
Task 2 Source-only checkpoint, loaded via task3/methods/erm.py.
"""

import json
import os
import torch
import yaml
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from shared.pacs import load_pacs_domain
from shared.pacs_protocol import build_source_splits, make_cycling_loader, TRAIN_TRANSFORM, EVAL_TRANSFORM, SEED
from task2.models.backbone import ResNet18Backbone
from task2.evaluation.metrics import evaluate_domain, accuracy, macro_f1
from task3.methods.dan_dg import DANDG
from task3.methods.sam import SAM, sam_training_step
from task3.selection.source_validation import evaluate_source_domains

SOURCE_DOMAINS = ["photo", "art_painting", "cartoon"]
NUM_CLASSES = 7


def load_config(base_path: str, method_path: str) -> dict:
    """Load base.yaml, then overlay the method-specific yaml on top of it."""
    cfg = yaml.safe_load(open(base_path)) or {}
    cfg.update(yaml.safe_load(open(method_path)) or {})
    return cfg


def result_path_for(method_name: str, out_dir: str) -> str:
    return os.path.join(out_dir, f"{method_name}_result.json")


def train_task3_method(method_name: str, root: str, device: str = "mps",
                        max_epochs: int = 30, patience: int = 5,
                        batch_per_source: int = 8, seed: int = SEED,
                        out_dir: str = "task3/results",
                        lr: float = 1e-4, weight_decay: float = 1e-4,
                        method_kwargs: dict = None):
    assert method_name in ("dan_dg", "sam"), "ERM is loaded via task3/methods/erm.py, not trained here."
    method_kwargs = method_kwargs or {}
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(seed)

    print(f"[task3:{method_name}] Building source splits and loading PACS domains (no Sketch access) ...")
    splits = build_source_splits(SOURCE_DOMAINS, root,
                                  out_path="shared/splits/pacs_sketch_seed6304.json", seed=seed)

    train_datasets = {d: load_pacs_domain(root, d, transform=TRAIN_TRANSFORM) for d in SOURCE_DOMAINS}
    eval_datasets = {d: load_pacs_domain(root, d, transform=EVAL_TRANSFORM) for d in SOURCE_DOMAINS}

    source_train_loaders = {
        d: make_cycling_loader(train_datasets[d], splits[d]["train_idx"], batch_per_source, seed=seed)
        for d in SOURCE_DOMAINS
    }
    val_loaders = {
        d: DataLoader(Subset(eval_datasets[d], splits[d]["val_idx"]), batch_size=64, shuffle=False)
        for d in SOURCE_DOMAINS
    }

    backbone = ResNet18Backbone(num_classes=NUM_CLASSES).to(device)

    if method_name == "dan_dg":
        method = DANDG(backbone.feature_dim, NUM_CLASSES, device, **method_kwargs)
        optimizer = torch.optim.AdamW(
            list(backbone.parameters()) + method.extra_parameters(), lr=lr, weight_decay=weight_decay
        )
    else:  # sam
        rho = method_kwargs.get("rho", 0.05)
        optimizer = SAM(backbone.parameters(), torch.optim.AdamW, rho=rho, lr=lr, weight_decay=weight_decay)

    best_mean_f1 = -1.0
    best_state = None
    epochs_without_improvement = 0
    steps_per_epoch = 50
    curves = []

    print(f"[task3:{method_name}] Starting training: max_epochs={max_epochs}, "
          f"patience={patience}, steps_per_epoch={steps_per_epoch}")
    epoch_bar = tqdm(range(max_epochs), desc=f"[task3:{method_name}] epochs", unit="epoch")
    for epoch in epoch_bar:
        backbone.train()
        backbone.freeze_bn_running_stats()

        logs = {}
        step_desc = "SAM (2 passes/step)" if method_name == "sam" else "steps"
        step_bar = tqdm(range(steps_per_epoch), desc=f"epoch {epoch} {step_desc}", unit="step", leave=False)
        for _ in step_bar:
            source_batches = {d: next(source_train_loaders[d]) for d in SOURCE_DOMAINS}
            source_batches = {d: (x.to(device), y.to(device)) for d, (x, y) in source_batches.items()}

            if method_name == "dan_dg":
                loss, logs = method.compute_loss(backbone, source_batches, None, 0.0)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:  # sam
                cls_loss = sam_training_step(backbone, source_batches, optimizer)
                logs = {"cls_loss": cls_loss}
            step_bar.set_postfix({k: f"{v:.4f}" if isinstance(v, float) else v for k, v in logs.items()})

        curves.append({"epoch": epoch, **logs})

        val_summary = evaluate_source_domains(backbone, val_loaders, device)
        mean_f1 = val_summary["mean_macro_f1"]
        epoch_bar.set_postfix(mean_val_f1=f"{mean_f1:.4f}", best=f"{best_mean_f1:.4f}",
                              no_improve=epochs_without_improvement)
        tqdm.write(f"[task3:{method_name}] epoch {epoch}: mean source val macro-F1 = {mean_f1:.4f}")

        if mean_f1 > best_mean_f1:
            best_mean_f1 = mean_f1
            best_state = {k: v.clone() for k, v in backbone.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            tqdm.write(f"[task3:{method_name}] early stopping at epoch {epoch} "
                       f"(no improvement for {patience} epochs)")
            break

    backbone.load_state_dict(best_state)
    ckpt_path = os.path.join(out_dir, f"{method_name}_checkpoint.pt")
    torch.save(backbone.state_dict(), ckpt_path)
    print(f"[task3:{method_name}] Saved checkpoint to {ckpt_path}")

    final_val = evaluate_source_domains(backbone, val_loaders, device)
    result = {
        "method": method_name,
        "checkpoint_path": ckpt_path,
        "best_mean_source_val_f1": best_mean_f1,
        "source_validation": final_val,
        "training_curve": curves,
    }

    result_path = result_path_for(method_name, out_dir)
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[task3:{method_name}] Saved training result/curve to {result_path}")

    return backbone, result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="task2/configs/base.yaml", help="Path to base.yaml (shared with Task 2)")
    parser.add_argument("--config", required=True, help="Path to method-specific yaml (dan_dg.yaml or sam.yaml)")
    parser.add_argument("--device", default="mps" if torch.mps.is_available() else "cpu")
    parser.add_argument("--out_dir", default="task3/results")
    args = parser.parse_args()

    cfg = load_config(args.base, args.config)
    method_kwargs = {k: v for k, v in cfg.items() if k in ("lambda_dg", "rho")}

    _, result = train_task3_method(
        cfg["method"], cfg["data_root"], device=args.device,
        max_epochs=cfg.get("epochs", 30), patience=cfg.get("patience", 5),
        batch_per_source=cfg.get("batch_size_per_source", 8),
        seed=cfg.get("seed", SEED), out_dir=args.out_dir,
        lr=cfg.get("lr", 1e-4), weight_decay=cfg.get("weight_decay", 1e-4),
        method_kwargs=method_kwargs,
    )
    print(result["source_validation"])