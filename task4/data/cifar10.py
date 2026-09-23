import numpy as np
from sklearn.model_selection import train_test_split
from torchvision import transforms as T
from torchvision.datasets import CIFAR10

SEED = 6304
CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                    "dog", "frog", "horse", "ship", "truck"]

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)

TRAIN_TRANSFORM = T.Compose([
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize(CIFAR_MEAN, CIFAR_STD),
])


def build_gcsc_transform(num_ops: int = 2, magnitude: int = 9):
    """GCSC's transform, parameterized so task4/configs/gcsc.yaml's
    randaugment.num_ops / randaugment.magnitude actually control it,
    rather than being fixed at import time."""
    return T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.RandAugment(num_ops=num_ops, magnitude=magnitude),
        T.ToTensor(),
        T.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])


GCSC_TRAIN_TRANSFORM = build_gcsc_transform()  # default (num_ops=2, magnitude=9), per spec

EVAL_TRANSFORM = T.Compose([
    T.ToTensor(),
    T.Normalize(CIFAR_MEAN, CIFAR_STD),
])


def load_cifar10_train_val_split(root: str, seed: int = SEED):
    ds = CIFAR10(root=root, train=True, download=True)
    labels = np.array(ds.targets)
    idx = np.arange(len(labels))
    train_idx, val_idx = train_test_split(idx, test_size=0.1, stratify=labels, random_state=seed)
    return train_idx.tolist(), val_idx.tolist()


def load_cifar10_datasets(root: str, transform_train=TRAIN_TRANSFORM, transform_eval=EVAL_TRANSFORM):
    train_ds = CIFAR10(root=root, train=True, download=True, transform=transform_train)
    train_ds_eval_view = CIFAR10(root=root, train=True, download=True, transform=transform_eval)
    test_ds = CIFAR10(root=root, train=False, download=True, transform=transform_eval)
    return train_ds, train_ds_eval_view, test_ds
