"""
256-unit hidden layer, ReLU, dropout 0.5, two-class output -- identical
architecture for DANN (input = 512-d feature) and CDAN (input =
feature_dim * num_classes, the flattened f (x) p outer product), per spec.
"""

import torch.nn as nn


class DomainDiscriminator(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256, dropout: float = 0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )

    def forward(self, x):
        return self.net(x)


def cdan_multilinear_map(feat, probs):
    """g(x) = vec(f (x) p): outer product of the 512-d feature and the
    C-class softmax probability vector, flattened to (N, feature_dim*C).
    No detachment of f or p, per spec."""
    n, d = feat.shape
    c = probs.shape[1]
    return (feat.unsqueeze(2) @ probs.unsqueeze(1)).reshape(n, d * c)
