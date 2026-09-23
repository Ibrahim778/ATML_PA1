"""
Manifold mixup, applied after layer2 and before layer3 of the CIFAR
ResNet-18, per spec. Mixes representations from two DIFFERENT classes so
the interpolated feature lies between their learned regions, and is then
trained toward the dummy classifiers as a proxy unknown.
"""

import torch


def sample_mixup_pairs(labels: torch.Tensor):
    """For a batch of labels, returns an index permutation such that
    labels[i] != labels[perm[i]] for every i where a valid partner exists.
    Falls back to a random permutation for any index where no different-
    class partner could be found (extremely unlikely for batch_size >= 2
    with multiple classes present)."""
    n = labels.size(0)
    device = labels.device
    perm = torch.randperm(n, device=device)
    same_class = labels[perm] == labels
    if same_class.any():
        # Try a bounded number of re-shuffles to resolve same-class pairs.
        for _ in range(10):
            if not same_class.any():
                break
            idx_to_fix = same_class.nonzero(as_tuple=True)[0]
            new_perm = torch.randperm(n, device=device)
            perm[idx_to_fix] = new_perm[idx_to_fix]
            same_class = labels[perm] == labels
    return perm


def manifold_mixup(feat_a: torch.Tensor, feat_b: torch.Tensor, alpha: float = 2.0):
    """Sample lambda ~ Beta(alpha, alpha) and mix two batches of
    intermediate features (post-layer2). Returns (mixed_feat, lam)."""
    beta_dist = torch.distributions.Beta(alpha, alpha)
    lam = beta_dist.sample().item()
    mixed = lam * feat_a + (1 - lam) * feat_b
    return mixed, lam
