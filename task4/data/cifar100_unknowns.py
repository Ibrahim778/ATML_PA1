"""
Fixed, spec-mandated near/far CIFAR-100 unknown groups. CIFAR-100 training
images must never be used for training/selection/thresholds -- only these
800 test images per group, loaded exclusively at final OSR evaluation time.
"""

import numpy as np
from torch.utils.data import Subset
from torchvision.datasets import CIFAR100

from task4.data.cifar10 import EVAL_TRANSFORM

NEAR_UNKNOWN_CLASSES = ["bus", "pickup_truck", "motorcycle", "tractor", "wolf", "fox", "leopard", "camel"]
FAR_UNKNOWN_CLASSES = ["bottle", "bowl", "chair", "clock", "keyboard", "mushroom", "sunflower", "wardrobe"]


def load_cifar100_unknown_subset(root: str, class_names: list, n_per_class: int = 100,
                                  transform=EVAL_TRANSFORM):
    """CIFAR-100 test set has 100 images/class; the spec's groups (8 classes x
    100 = 800 per group) use all of them, so n_per_class=100 takes the full
    per-class allotment rather than a further subsample."""
    ds = CIFAR100(root=root, train=False, download=True, transform=transform)
    fine_to_idx = {name: i for i, name in enumerate(ds.classes)}

    target_class_ids = set()
    for name in class_names:
        assert name in fine_to_idx, f"CIFAR-100 fine class not found: {name}"
        target_class_ids.add(fine_to_idx[name])

    labels = np.array(ds.targets)
    selected = []
    for cid in target_class_ids:
        cls_idx = np.where(labels == cid)[0]
        selected.extend(cls_idx[:n_per_class].tolist())

    return Subset(ds, selected)


def load_near_far_unknowns(root: str, transform=EVAL_TRANSFORM):
    near = load_cifar100_unknown_subset(root, NEAR_UNKNOWN_CLASSES, transform=transform)
    far = load_cifar100_unknown_subset(root, FAR_UNKNOWN_CLASSES, transform=transform)
    return near, far
