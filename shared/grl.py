"""
Gradient-reversal layer (Ganin et al., 2016): identity on the forward pass,
negated (and scaled by alpha) gradient on the backward pass. Shared by
DANN and CDAN (Task 2) -- the discriminator's gradient is reversed before
reaching the backbone, so the backbone is pushed toward domain confusion
while the discriminator is pushed toward domain separation.
"""

import numpy as np
import torch
from torch.autograd import Function


class _GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


class GradientReversalLayer(torch.nn.Module):
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return _GradientReversalFunction.apply(x, self.alpha)


def grl_alpha_schedule(p: float) -> float:
    """alpha(p) = 2 / (1 + exp(-10p)) - 1, p in [0, 1] = training progress."""
    return float(2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0)
