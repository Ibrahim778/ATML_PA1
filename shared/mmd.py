"""
Sum-of-three-RBF-kernel MMD^2, bandwidths = {0.5, 1, 2} x the median pairwise
squared distance in the current combined batch, per the assignment spec.
Used identically by DAN (Task 2, source vs. target) and DAN-DG (Task 3,
every pair of source domains), so the discrepancy measure itself is held
fixed between the two tasks -- only what it's applied to differs.
"""

import torch


def _pairwise_sq_dists(x: torch.Tensor) -> torch.Tensor:
    sq = (x ** 2).sum(dim=1, keepdim=True)
    dist = sq + sq.T - 2 * x @ x.T
    return dist.clamp(min=0)


def multi_kernel_mmd(feats_a: torch.Tensor, feats_b: torch.Tensor,
                      bandwidth_mults=(0.5, 1.0, 2.0)) -> torch.Tensor:
    """MMD^2 between two feature sets using a sum of 3 RBF kernels whose
    bandwidths are `bandwidth_mults` x the median pairwise squared distance
    in the combined (a + b) batch. Returns a scalar tensor (differentiable)."""
    combined = torch.cat([feats_a, feats_b], dim=0)
    n = combined.size(0)
    sq_dists = _pairwise_sq_dists(combined)

    off_diag_mask = ~torch.eye(n, dtype=torch.bool, device=combined.device)
    median_sq_dist = sq_dists[off_diag_mask].median().clamp(min=1e-8)

    na = feats_a.size(0)

    def kernel_term(bandwidth):
        K = torch.exp(-sq_dists / (2.0 * bandwidth))
        Kaa = K[:na, :na].mean()
        Kbb = K[na:, na:].mean()
        Kab = K[:na, na:].mean()
        return Kaa + Kbb - 2.0 * Kab

    mmd = sum(kernel_term(median_sq_dist * m) for m in bandwidth_mults)
    return mmd


def pairwise_source_mmd(feats_by_domain: dict, bandwidth_mults=(0.5, 1.0, 2.0)) -> torch.Tensor:
    """Average MMD^2 over all unordered pairs of domains -- used by Task 3's
    DAN-DG, which aligns every pair of OBSERVED source domains (no target)."""
    domains = list(feats_by_domain.keys())
    pairs = [(domains[i], domains[j]) for i in range(len(domains)) for j in range(i + 1, len(domains))]
    total = sum(multi_kernel_mmd(feats_by_domain[a], feats_by_domain[b], bandwidth_mults) for a, b in pairs)
    return total / len(pairs)
