"""
Threshold calibrated on CIFAR-10 VALIDATION scores only (never on CIFAR-100
unknowns): tau = 95th percentile of unknownness on val, so ~95% of known
validation examples are accepted (u(x) <= tau). Applied unchanged to the
test-set known and unknown scores.
"""

import numpy as np


def calibrate_threshold(val_unknownness_scores: np.ndarray, target_tpr: float = 0.95) -> float:
    return float(np.percentile(val_unknownness_scores, target_tpr * 100))


def acceptance_rate(scores: np.ndarray, threshold: float) -> float:
    """Fraction of examples with u(x) <= threshold, i.e. accepted as known."""
    return float((scores <= threshold).mean())


def fpr_at_95_tpr(unknown_scores: np.ndarray, threshold: float) -> float:
    """Fraction of UNKNOWN examples incorrectly accepted (u(x) <= threshold)
    under the convention that the threshold targets 95% TPR on known data."""
    return float((unknown_scores <= threshold).mean())
