"""
Shared across Task 2 and Task 3, per the assignment's instruction to reuse
one PACS protocol for both tasks. Provides:
  - stratified 80/20 train/val split per source domain (seed 6304)
  - domain-balanced batch iteration (N examples per domain per step, with
    per-domain loaders cycling independently so exhausted domains restart
    without blocking the others)
"""

import itertools
import json
import os

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import transforms as T

SEED = 6304

TRAIN_TRANSFORM = T.Compose([
    T.Resize((256, 256)),
    T.RandomCrop(224),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet1K_V1 stats
])

EVAL_TRANSFORM = T.Compose([
    T.Resize((256, 256)),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def stratified_split(labels, test_size: float = 0.2, seed: int = SEED):
    idx = np.arange(len(labels))
    train_idx, val_idx = train_test_split(idx, test_size=test_size, stratify=labels, random_state=seed)
    return train_idx, val_idx


def build_source_splits(source_domains, root: str, out_path: str = None, seed: int = SEED):
    """Returns {domain: {"train_idx": [...], "val_idx": [...]}}, built from
    an unaugmented view of each domain just to read labels. Cached to
    out_path (JSON) so re-runs are exactly reproducible."""
    from shared.pacs import load_pacs_domain

    if out_path and os.path.exists(out_path):
        with open(out_path) as f:
            return json.load(f)

    splits = {}
    for domain in source_domains:
        ds = load_pacs_domain(root, domain, transform=None)
        labels = np.array([s[1] for s in ds.samples])
        train_idx, val_idx = stratified_split(labels, seed=seed)
        splits[domain] = {"train_idx": train_idx.tolist(), "val_idx": val_idx.tolist()}

    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(splits, f, indent=2)
    return splits


def make_cycling_loader(dataset, indices, batch_size: int, seed: int = SEED):
    """An infinite iterator over (x, y) batches of exactly `batch_size`,
    reshuffled deterministically each pass. drop_last=True so every yielded
    batch is exactly batch_size, which domain-balanced training requires."""
    subset = Subset(dataset, indices)
    g = torch.Generator().manual_seed(seed)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=True,
                         generator=g, drop_last=True, num_workers=2)
    return itertools.cycle(loader)


class DomainBalancedBatcher:
    """Wraps several infinite per-domain loaders and yields one dict of
    {domain_name: (x, y)} per step, each already the requested per-domain
    batch size (8 for sources, 24 for target in Task 2's adaptation step)."""

    def __init__(self, per_domain_loaders: dict):
        self.loaders = per_domain_loaders

    def __next__(self):
        return {name: next(loader) for name, loader in self.loaders.items()}

    def __iter__(self):
        return self
