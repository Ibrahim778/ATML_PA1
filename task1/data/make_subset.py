"""
Builds:
  1. Stratified 80/20 train/val split of STL-10 'train' partition (seed 6304).
     Used only for training the linear classifier heads.
  2. Class-balanced 500-image subset of STL-10 'test' partition (seed 6304).
     This is the fixed evaluation set every intervention in Task 1 runs on.

Both are saved to disk as JSON so every later script reloads the *same*
indices instead of resampling.
"""

import json
import os
import numpy as np
from sklearn.model_selection import train_test_split
from torchvision.datasets import STL10

SEED = 6304
STL10_CLASSES = [
    "airplane", "bird", "car", "cat", "deer",
    "dog", "horse", "monkey", "ship", "truck",
]


def make_train_val_split(root: str):
    ds = STL10(root=root, split="train", download=True)
    labels = np.array(ds.labels)
    idx = np.arange(len(labels))

    train_idx, val_idx = train_test_split(
        idx, test_size=0.2, stratify=labels, random_state=SEED
    )
    return train_idx, val_idx, labels


def make_balanced_eval_subset(root: str, n_total: int = 500):
    ds = STL10(root=root, split="test", download=True)
    labels = np.array(ds.labels)
    classes = np.unique(labels)
    n_per_class = n_total // len(classes)

    rng = np.random.RandomState(SEED)
    selected = []
    imbalance_notes = {}
    for c in classes:
        c_idx = np.where(labels == c)[0]
        if len(c_idx) < n_per_class:
            imbalance_notes[int(c)] = len(c_idx)
            chosen = c_idx
        else:
            chosen = rng.choice(c_idx, size=n_per_class, replace=False)
        selected.extend(chosen.tolist())

    selected = sorted(selected)
    return selected, imbalance_notes


def main(root: str = "./data", out_dir: str = "task1/data"):
    os.makedirs(out_dir, exist_ok=True)

    train_idx, val_idx, _ = make_train_val_split(root)
    eval_idx, imbalance = make_balanced_eval_subset(root)

    with open(os.path.join(out_dir, "train_val_split_seed6304.json"), "w") as f:
        json.dump({"train_idx": train_idx.tolist(), "val_idx": val_idx.tolist()}, f)

    with open(os.path.join(out_dir, "eval_subset_seed6304.json"), "w") as f:
        json.dump({"eval_idx": eval_idx, "imbalance": imbalance}, f)

    print(f"Train: {len(train_idx)}  Val: {len(val_idx)}  Eval subset: {len(eval_idx)}")
    if imbalance:
        print(f"Imbalanced classes (fewer than target/class): {imbalance}")


if __name__ == "__main__":
    main()
