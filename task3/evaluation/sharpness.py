"""
Delta_sharp = L(theta + epsilon) - L(theta), epsilon = 0.05 * grad / ||grad||_2,
evaluated on ONE fixed validation batch (32 examples per source domain,
seed 6304) so ERM, DAN-DG, and SAM are compared under an identical
perturbation and identical data. This is a standardized LOCAL diagnostic,
not a claim about the global loss landscape.
"""

import copy

import torch
import torch.nn as nn

SEED = 6304
RADIUS = 0.05


def build_fixed_sharpness_batch(val_datasets: dict, n_per_domain: int = 32, seed: int = SEED):
    """val_datasets: {domain: Dataset (already using EVAL_TRANSFORM)}. Returns
    a single (x, y) batch concatenated across domains, sampled deterministically."""
    import numpy as np
    from torch.utils.data import Subset, DataLoader

    rng = np.random.RandomState(seed)
    xs, ys = [], []
    for domain, ds in val_datasets.items():
        idx = rng.choice(len(ds), n_per_domain, replace=False)
        loader = DataLoader(Subset(ds, idx.tolist()), batch_size=n_per_domain, shuffle=False)
        x, y = next(iter(loader))
        xs.append(x)
        ys.append(y)
    return torch.cat(xs, dim=0), torch.cat(ys, dim=0)


def compute_sharpness(backbone, x: torch.Tensor, y: torch.Tensor, device: str, radius: float = RADIUS) -> float:
    """backbone is placed in eval() for this computation (per spec) and its
    original mode/state is restored afterward."""
    was_training = backbone.training
    backbone.eval()

    x, y = x.to(device), y.to(device)
    criterion = nn.CrossEntropyLoss()

    # L(theta)
    backbone.zero_grad()
    logits = backbone(x)
    loss_theta = criterion(logits, y)
    loss_theta.backward()

    grad_vec = torch.cat([p.grad.detach().flatten() for p in backbone.parameters() if p.grad is not None])
    grad_norm = grad_vec.norm(p=2).clamp(min=1e-12)

    # Save original params, apply epsilon = radius * grad / ||grad||.
    original_params = [p.detach().clone() for p in backbone.parameters()]
    with torch.no_grad():
        for p in backbone.parameters():
            if p.grad is None:
                continue
            eps = radius * p.grad / grad_norm
            p.add_(eps)

    backbone.zero_grad()
    with torch.no_grad():
        logits_perturbed = backbone(x)
        loss_perturbed = criterion(logits_perturbed, y)

    # Restore original parameters.
    with torch.no_grad():
        for p, orig in zip(backbone.parameters(), original_params):
            p.copy_(orig)

    if was_training:
        backbone.train()

    return float((loss_perturbed - loss_theta).item())
