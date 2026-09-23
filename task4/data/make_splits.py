import json
import os

from task4.data.cifar10 import load_cifar10_train_val_split, SEED


def build_and_save_splits(root: str, out_path: str = "task4/data/cifar10_split_seed6304.json"):
    if os.path.exists(out_path):
        print(f"[task4] Found cached split at {out_path}, reusing.")
        with open(out_path) as f:
            d = json.load(f)
        return d["train_idx"], d["val_idx"]

    print("[task4] Building stratified 90/10 CIFAR-10 split (seed 6304) ...")
    train_idx, val_idx = load_cifar10_train_val_split(root, seed=SEED)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"train_idx": train_idx, "val_idx": val_idx}, f)
    print(f"[task4] Train: {len(train_idx)}  Val: {len(val_idx)}. Saved to {out_path}")
    return train_idx, val_idx


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    build_and_save_splits(args.root)
