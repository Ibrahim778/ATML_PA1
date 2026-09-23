import torch


def msp_score(logits: torch.Tensor) -> torch.Tensor:
    """u_MSP(x) = 1 - max_k p_k(x). Larger = more novel."""
    probs = torch.softmax(logits, dim=1)
    return 1.0 - probs.max(dim=1).values
