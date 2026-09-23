"""
Inspects unknown examples that were incorrectly ACCEPTED (u(x) <= threshold)
under the vanilla-MLS threshold, recording the unknown class, the known
class it was mistaken for, its score, and the threshold -- per spec.
"""

import numpy as np


def find_incorrectly_accepted(scores: np.ndarray, preds: np.ndarray, true_unknown_labels: list,
                               threshold: float, class_names_known: list, class_names_unknown: list,
                               top_k: int = 3):
    """scores/preds are over the UNKNOWN evaluation set only. true_unknown_labels
    gives each example's original CIFAR-100 fine-class index into
    class_names_unknown. Returns up to top_k lowest-score (most confidently
    accepted) incorrectly-accepted examples."""
    accepted_mask = scores <= threshold
    accepted_idx = np.where(accepted_mask)[0]
    if len(accepted_idx) == 0:
        return []

    # Sort by score ascending: the most confidently accepted (lowest
    # unknownness) failures are the most informative.
    accepted_idx = accepted_idx[np.argsort(scores[accepted_idx])]

    rows = []
    for i in accepted_idx[:top_k]:
        rows.append({
            "unknown_class": class_names_unknown[true_unknown_labels[i]],
            "predicted_known_class": class_names_known[preds[i]],
            "score": float(scores[i]),
            "threshold": float(threshold),
        })
    return rows
