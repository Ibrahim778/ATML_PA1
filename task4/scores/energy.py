import torch


def energy_score(logits: torch.Tensor) -> torch.Tensor:
    """u_Energy(x) = -log sum_k exp(z_k(x)). Larger = more novel."""
    return -torch.logsumexp(logits, dim=1)
