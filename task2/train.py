"""
One training loop, parameterized by method name, so Source-only/DAN/DANN/
CDAN share identical initialization, source sampling, augmentation,
optimizer, and training budget -- only the loss computation differs.
"""

import os
import torch
import yaml
from tqdm import tqdm

from shared.pacs import load_pacs_domain
from shared.pacs_protocol import (
    build_source_splits, make_cycling_loader,
    TRAIN_TRANSFORM, EVAL_TRANSFORM, SEED,
)
from task2.models.backbone import ResNet18Backbone
from task2.methods.source_only import SourceOnly
from task2.methods.dan import DAN
from task2.methods.dann import DANN
from task2.methods.cdan import CDAN
from task2.evaluation.metrics import evaluate_domain, accuracy, macro_f1

SOURCE_DOMAINS = ["photo", "art_painting", "cartoon"]
TARGET_DOMAIN = "sketch"
NUM_CLASSES = 7

METHOD_REGISTRY = {"source_only": SourceOnly, "dan": DAN, "dann": DANN, "cdan": CDAN}


def build_method(name: str, feature_dim: int, device: str, **kwargs):
    cls = METHOD_REGISTRY[name]
    return cls(feature_dim, NUM_CLASSES, device, **kwargs)


def load_config(base_path: str, method_path: str) -> dict:
    """Load base.yaml, then overlay the method-specific yaml on top of it."""
    cfg = yaml.safe_load(open(base_path)) or {}
    cfg.update(yaml.safe_load(open(method_path)) or {})
    return cfg


def train_task2_method(method_name: str, root: str, device: str = "mps",
                        max_epochs: int = 30, patience: int = 5,
                        batch_per_source: int = 8, batch_target: int = 24,
                        seed: int = SEED, out_dir: str = "task2/results",
                        lr: float = 1e-4, weight_decay: float = 1e-4,
                        method_kwargs: dict = None):
    method_kwargs = method_kwargs or {}
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(seed)

    print(f"[task2:{method_name}] Building source splits and loading PACS domains ...")
    # ---- Data ----
    splits = build_source_splits(SOURCE_DOMAINS, root,
                                  out_path="shared/splits/pacs_sketch_seed6304.json", seed=seed)

    train_datasets = {d: load_pacs_domain(root, d, transform=TRAIN_TRANSFORM) for d in SOURCE_DOMAINS}
    eval_datasets = {d: load_pacs_domain(root, d, transform=EVAL_TRANSFORM) for d in SOURCE_DOMAINS}
    target_train_ds = load_pacs_domain(root, TARGET_DOMAIN, transform=TRAIN_TRANSFORM)  # unlabeled use only
    target_eval_ds = load_pacs_domain(root, TARGET_DOMAIN, transform=EVAL_TRANSFORM)

    source_train_loaders = {
        d: make_cycling_loader(train_datasets[d], splits[d]["train_idx"], batch_per_source, seed=seed)
        for d in SOURCE_DOMAINS
    }
    # All Sketch images may be used (unlabeled) during Task 2 adaptation.
    target_indices = list(range(len(target_train_ds)))
    target_loader = make_cycling_loader(target_train_ds, target_indices, batch_target, seed=seed)

    from torch.utils.data import DataLoader, Subset
    val_loaders = {
        d: DataLoader(Subset(eval_datasets[d], splits[d]["val_idx"]), batch_size=64, shuffle=False)
        for d in SOURCE_DOMAINS
    }
    target_eval_loader = DataLoader(target_eval_ds, batch_size=64, shuffle=False)

    # ---- Model + method ----
    print(f"[task2:{method_name}] Building ResNet-18 backbone (ImageNet1K_V1 pretrained) ...")
    backbone = ResNet18Backbone(num_classes=NUM_CLASSES).to(device)
    method = build_method(method_name, backbone.feature_dim, device, **method_kwargs)

    optimizer = torch.optim.AdamW(
        list(backbone.parameters()) + method.extra_parameters(), lr=lr, weight_decay=weight_decay
    )
    print(f"[task2:{method_name}] Starting training: max_epochs={max_epochs}, "
          f"patience={patience}, steps_per_epoch=50")

    best_mean_f1 = -1.0
    best_state = None
    epochs_without_improvement = 0
    steps_per_epoch = 50  # fixed inner-loop length per "epoch" over cycling loaders
    total_steps = max_epochs * steps_per_epoch
    step = 0

    curves = []

    epoch_bar = tqdm(range(max_epochs), desc=f"[task2:{method_name}] epochs", unit="epoch")
    for epoch in epoch_bar:
        backbone.train()
        backbone.freeze_bn_running_stats()

        logs = {}
        step_bar = tqdm(range(steps_per_epoch), desc=f"epoch {epoch} steps", unit="step", leave=False)
        for _ in step_bar:
            source_batches = {d: next(source_train_loaders[d]) for d in SOURCE_DOMAINS}
            source_batches = {d: (x.to(device), y.to(device)) for d, (x, y) in source_batches.items()}
            x_t, _ = next(target_loader)
            target_batch = (x_t.to(device), None)

            progress_p = step / max(total_steps, 1)
            loss, logs = method.compute_loss(backbone, source_batches, target_batch, progress_p)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            step_bar.set_postfix({k: f"{v:.4f}" if isinstance(v, float) else v for k, v in logs.items()})

        curves.append({"epoch": epoch, **logs})

        # ---- Validation (mean macro-F1 across the three source domains) ----
        f1s = []
        for d in SOURCE_DOMAINS:
            preds, labels, _ = evaluate_domain(backbone, val_loaders[d], device)
            f1s.append(macro_f1(preds, labels))
        mean_f1 = sum(f1s) / len(f1s)

        epoch_bar.set_postfix(mean_val_f1=f"{mean_f1:.4f}", best=f"{best_mean_f1:.4f}",
                              no_improve=epochs_without_improvement)
        tqdm.write(f"[task2:{method_name}] epoch {epoch}: mean source val macro-F1 = {mean_f1:.4f}")

        if mean_f1 > best_mean_f1:
            best_mean_f1 = mean_f1
            best_state = {k: v.clone() for k, v in backbone.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            tqdm.write(f"[task2:{method_name}] early stopping at epoch {epoch} "
                       f"(no improvement for {patience} epochs)")
            break

    backbone.load_state_dict(best_state)
    ckpt_path = os.path.join(out_dir, f"{method_name}_checkpoint.pt")
    torch.save(backbone.state_dict(), ckpt_path)
    print(f"[task2:{method_name}] Saved checkpoint to {ckpt_path}")

    # ---- Final reporting ----
    print(f"[task2:{method_name}] Running final evaluation on source-val and target ...")
    per_domain_val = {}
    for d in SOURCE_DOMAINS:
        preds, labels, _ = evaluate_domain(backbone, val_loaders[d], device)
        per_domain_val[d] = {"accuracy": accuracy(preds, labels), "macro_f1": macro_f1(preds, labels)}

    target_preds, target_labels, target_feats = evaluate_domain(backbone, target_eval_loader, device)
    target_metrics = {"accuracy": accuracy(target_preds, target_labels),
                       "macro_f1": macro_f1(target_preds, target_labels)}

    result = {
        "method": method_name,
        "checkpoint_path": ckpt_path,
        "best_mean_source_val_f1": best_mean_f1,
        "per_domain_val": per_domain_val,
        "target_metrics": target_metrics,
        "training_curve": curves,
    }
    return backbone, method, result, (target_preds, target_labels, target_feats)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="task2/configs/base.yaml", help="Path to base.yaml")
    parser.add_argument("--config", required=True, help="Path to method-specific yaml")
    parser.add_argument("--device", default="mps" if torch.mps.is_available() else "cpu")
    parser.add_argument("--out_dir", default="task2/results")
    args = parser.parse_args()

    cfg = load_config(args.base, args.config)
    method_kwargs = {k: v for k, v in cfg.items() if k in ("lambda_mmd", "max_alpha")}

    _, _, result, _ = train_task2_method(
        cfg["method"], cfg["data_root"], device=args.device,
        max_epochs=cfg.get("epochs", 30), patience=cfg.get("patience", 5),
        batch_per_source=cfg.get("batch_size_per_source", 8),
        batch_target=cfg.get("batch_size_target", 24),
        seed=cfg.get("seed", SEED), out_dir=args.out_dir,
        lr=cfg.get("lr", 1e-4), weight_decay=cfg.get("weight_decay", 1e-4),
        method_kwargs=method_kwargs,
    )
    print(result["target_metrics"])