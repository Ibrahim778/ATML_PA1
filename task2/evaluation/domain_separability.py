"""
Freezes the backbone, collects equal numbers of source-validation and
target features, does a seeded 70/30 split, and trains a balanced logistic
regression to distinguish source from target. Held-out accuracy is the
domain-separability score; 50% = chance (domains indistinguishable).
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

SEED = 6304


def domain_separability_score(source_feats: np.ndarray, target_feats: np.ndarray, seed: int = SEED) -> float:
    n = min(len(source_feats), len(target_feats))
    rng = np.random.RandomState(seed)
    src_idx = rng.choice(len(source_feats), n, replace=False)
    tgt_idx = rng.choice(len(target_feats), n, replace=False)

    X = np.concatenate([source_feats[src_idx], target_feats[tgt_idx]], axis=0)
    y = np.concatenate([np.zeros(n), np.ones(n)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )
    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000)
    clf.fit(X_train, y_train)
    return float(clf.score(X_test, y_test))
