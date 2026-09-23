"""
Freezes the backbone, collects balanced features from the three source
validation sets, does a seeded 70/30 split, and trains a multinomial
logistic regression to predict which of Photo/Art/Cartoon a feature came
from. Held-out accuracy is the source-domain separability score; chance
is 33.3%. Lower = stronger invariance ACROSS THE OBSERVED SOURCES -- this
says nothing about Sketch, which the classifier never sees.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

SEED = 6304


def source_domain_separability_score(feats_by_domain: dict, seed: int = SEED) -> float:
    """feats_by_domain: {"photo": (N,D) array, "art_painting": ..., "cartoon": ...}."""
    domains = list(feats_by_domain.keys())
    n_per_domain = min(len(feats_by_domain[d]) for d in domains)

    rng = np.random.RandomState(seed)
    X_parts, y_parts = [], []
    for i, d in enumerate(domains):
        idx = rng.choice(len(feats_by_domain[d]), n_per_domain, replace=False)
        X_parts.append(feats_by_domain[d][idx])
        y_parts.append(np.full(n_per_domain, i))

    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )
    clf = LogisticRegression(C=1.0, max_iter=1000)
    clf.fit(X_train, y_train)
    return float(clf.score(X_test, y_test))
