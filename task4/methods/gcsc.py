"""
GCSC = Vanilla training recipe (task4/methods/vanilla.py) with RandAugment
inserted before normalization, per spec. This file exists mainly to keep
the method boundary explicit in the repo structure and to document that
the ONLY difference from Vanilla is the transform passed into
train_vanilla_or_gcsc's `train_ds` (see task4/data/cifar10.py's
GCSC_TRAIN_TRANSFORM). Evaluated with MLS, same as Vanilla, so the
comparison isolates the effect of stronger augmentation on OSR.
"""

from task4.methods.vanilla import train_vanilla_or_gcsc


def train_gcsc(train_ds, val_ds, train_idx, val_idx, device="cuda", **kwargs):
    """train_ds must already use GCSC_TRAIN_TRANSFORM (RandAugment applied
    after crop/flip, before ToTensor/Normalize)."""
    return train_vanilla_or_gcsc(train_ds, val_ds, train_idx, val_idx,
                                  variant_name="gcsc", device=device, **kwargs)
