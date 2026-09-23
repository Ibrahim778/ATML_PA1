"""
u_Mah(x) = min_c (f(x) - mu_c)^T Sigma^-1 (f(x) - mu_c), with class means
mu_c and one SHARED DIAGONAL covariance Sigma estimated from unaugmented
CIFAR-10 training features (a 1e-6 epsilon added to every diagonal entry
per spec, to keep the inverse numerically stable).
"""

import torch


def fit_mahalanobis_stats(train_feats: torch.Tensor, train_labels: torch.Tensor, num_classes: int = 10):
    """Returns (class_means: (C, D), inv_diag_cov: (D,)) -- the diagonal of
    Sigma^-1, since Sigma is diagonal, matching a diagonal covariance
    assumption (spec: "one shared diagonal covariance")."""
    d = train_feats.size(1)
    class_means = torch.zeros(num_classes, d)
    for c in range(num_classes):
        mask = train_labels == c
        class_means[c] = train_feats[mask].mean(dim=0)

    # Shared covariance: pool centered features across all classes.
    centered = train_feats - class_means[train_labels]
    var = centered.var(dim=0, unbiased=True) + 1e-6
    inv_diag_cov = 1.0 / var
    return class_means, inv_diag_cov


def mahalanobis_score(feats: torch.Tensor, class_means: torch.Tensor, inv_diag_cov: torch.Tensor) -> torch.Tensor:
    """feats: (N, D). Returns (N,) min-over-classes squared Mahalanobis
    distance under the shared diagonal covariance."""
    # (N, C, D) squared diffs weighted by inv_diag_cov, summed over D.
    diffs = feats.unsqueeze(1) - class_means.unsqueeze(0)  # (N, C, D)
    weighted_sq = diffs ** 2 * inv_diag_cov.view(1, 1, -1)
    dists = weighted_sq.sum(dim=2)  # (N, C)
    return dists.min(dim=1).values
