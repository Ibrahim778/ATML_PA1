import torch


def mls_score(logits: torch.Tensor) -> torch.Tensor:
    """u_MLS(x) = -max_k z_k(x). Larger = more novel."""
    return -logits.max(dim=1).values
