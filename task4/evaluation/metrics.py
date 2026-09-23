import numpy as np
from sklearn.metrics import roc_auc_score, f1_score


def auroc(known_scores: np.ndarray, unknown_scores: np.ndarray) -> float:
    """AUROC for known-vs-unknown, where higher score = more novel/unknown.
    Labels: 0 = known, 1 = unknown."""
    y_true = np.concatenate([np.zeros_like(known_scores), np.ones_like(unknown_scores)])
    y_score = np.concatenate([known_scores, unknown_scores])
    return float(roc_auc_score(y_true, y_score))


def closed_set_accuracy(preds: np.ndarray, labels: np.ndarray) -> float:
    return float((preds == labels).mean())


def closed_set_macro_f1(preds: np.ndarray, labels: np.ndarray) -> float:
    return float(f1_score(labels, preds, average="macro"))
