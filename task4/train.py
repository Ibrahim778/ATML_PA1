"""
Run with:
    python -m task4.train --variant vanilla --root /path/to/cifar
    python -m task4.train --variant gcsc --root /path/to/cifar

By default reads task4/configs/<variant>.yaml for all hyperparameters
(epochs, batch size, optimizer settings, RandAugment settings for GCSC).
Pass --config to use a different YAML file, e.g. for an ablation.
"""

import argparse
import torch

from task4.config import load_config
from task4.data.cifar10 import load_cifar10_datasets, TRAIN_TRANSFORM, EVAL_TRANSFORM, build_gcsc_transform
from task4.data.make_splits import build_and_save_splits
from task4.methods.vanilla import train_vanilla_or_gcsc
from task4.methods.proser import train_proser


def _train_vanilla_or_gcsc(args, cfg):
    train_idx, val_idx = build_and_save_splits(args.root)

    if args.variant == "gcsc":
        ra_cfg = cfg.get("randaugment", {})
        train_transform = build_gcsc_transform(
            num_ops=ra_cfg.get("num_ops", 2), magnitude=ra_cfg.get("magnitude", 9)
        )
        print(f"[task4:gcsc] RandAugment num_ops={ra_cfg.get('num_ops', 2)} "
              f"magnitude={ra_cfg.get('magnitude', 9)}")
    else:
        train_transform = TRAIN_TRANSFORM

    train_ds, train_ds_eval_view, _ = load_cifar10_datasets(
        args.root, transform_train=train_transform, transform_eval=EVAL_TRANSFORM
    )

    opt_cfg = cfg.get("optimizer", {})
    model, best_val_acc, ckpt_path = train_vanilla_or_gcsc(
        train_ds, train_ds_eval_view, train_idx, val_idx,
        variant_name=args.variant, device=args.device,
        epochs=cfg.get("epochs", 100),
        batch_size=cfg.get("batch_size", 128),
        lr=opt_cfg.get("lr", 0.1),
        momentum=opt_cfg.get("momentum", 0.9),
        weight_decay=opt_cfg.get("weight_decay", 5e-4),
        seed=cfg.get("seed", 6304),
    )
    print(f"Done. {args.variant} best val acc = {best_val_acc:.4f}, checkpoint at {ckpt_path}")


def _train_proser(args, cfg):
    train_idx, val_idx = build_and_save_splits(args.root)
    train_ds, train_ds_eval_view, _ = load_cifar10_datasets(
        args.root, transform_train=TRAIN_TRANSFORM, transform_eval=EVAL_TRANSFORM
    )

    vanilla_ckpt = args.vanilla_checkpoint or cfg.get("vanilla_checkpoint")
    if not vanilla_ckpt:
        raise ValueError(
            "PROSER needs a trained Vanilla checkpoint. Either train Vanilla first "
            "(python -m task4.train --variant vanilla --root ...) and set "
            "vanilla_checkpoint in task4/configs/proser.yaml, or pass --vanilla_checkpoint."
        )

    opt_cfg = cfg.get("optimizer", {})
    model, best_val_acc, ckpt_path = train_proser(
        train_ds, train_ds_eval_view, train_idx, val_idx, vanilla_ckpt,
        device=args.device,
        epochs=cfg.get("epochs", 50),
        batch_size=cfg.get("batch_size", 128),
        num_known=cfg.get("num_known", 10),
        num_dummy=cfg.get("num_dummy", 5),
        beta=cfg.get("beta", 1.0),
        gamma=cfg.get("gamma", 0.1),
        lr=opt_cfg.get("lr", 1e-3),
        momentum=opt_cfg.get("momentum", 0.9),
        weight_decay=opt_cfg.get("weight_decay", 5e-4),
        seed=cfg.get("seed", 6304),
    )
    print(f"Done. proser best val acc (known-class CSA) = {best_val_acc:.4f}, checkpoint at {ckpt_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["vanilla", "gcsc", "proser"], required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--config", default=None,
                         help="Path to a YAML config (defaults to task4/configs/<variant>.yaml)")
    parser.add_argument("--vanilla_checkpoint", default=None,
                         help="proser only: overrides vanilla_checkpoint in proser.yaml")
    parser.add_argument("--device", default="mps" if torch.mps.is_available() else "cpu")
    args = parser.parse_args()

    config_path = args.config or f"task4/configs/{args.variant}.yaml"
    cfg = load_config(config_path)
    print(f"[task4:{args.variant}] Loaded config from {config_path}:\n{cfg}")

    if args.variant in ("vanilla", "gcsc"):
        _train_vanilla_or_gcsc(args, cfg)
    else:
        _train_proser(args, cfg)


if __name__ == "__main__":
    main()
