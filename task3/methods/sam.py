"""
Standard Sharpness-Aware Minimization (Foret et al., 2021), reimplemented
from the paper's two-step procedure (this pattern -- first_step ascends to
a perturbed point, second_step restores parameters and applies the base
optimizer's update using the gradient computed there -- is now the common
way every public SAM implementation structures the algorithm; there is no
single canonical source file to attribute it to).

SAM does not fit the single compute_loss() interface the other Task 3/
Task 2 methods use, because it needs the loss function evaluated TWICE
per step (once at theta, once at theta + epsilon) with a parameter update
in between. task3/train.py special-cases method_name == "sam" and calls
sam_training_step() directly instead of going through compute_loss().
"""

import torch
import torch.nn as nn


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer_cls, rho: float = 0.05, **base_optimizer_kwargs):
        defaults = dict(rho=rho)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer_cls(self.param_groups, **base_optimizer_kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False):
        """Ascend to the perturbed point theta + epsilon, where epsilon is
        the normalized gradient scaled by rho."""
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale
                p.add_(e_w)
                self.state[p]["e_w"] = e_w
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False):
        """Restore original parameters, then apply the base optimizer's
        update using the gradient computed AT the perturbed point."""
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.sub_(self.state[p]["e_w"])
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def _grad_norm(self):
        return torch.norm(torch.stack([
            p.grad.norm(p=2)
            for group in self.param_groups
            for p in group["params"] if p.grad is not None
        ]), p=2)


def sam_training_step(backbone, source_batches: dict, sam_optimizer: SAM, criterion=None):
    """One SAM step over domain-balanced source batches. Returns the
    classification loss at the ORIGINAL parameters (for logging)."""
    criterion = criterion or nn.CrossEntropyLoss()

    def compute_erm_loss():
        loss = 0.0
        for dom, (x, y) in source_batches.items():
            logits = backbone(x)
            loss = loss + criterion(logits, y)
        return loss / len(source_batches)

    # Pass 1: gradient at theta.
    sam_optimizer.zero_grad()
    loss_1 = compute_erm_loss()
    loss_1.backward()
    sam_optimizer.first_step(zero_grad=True)

    # Pass 2: gradient at theta + epsilon; second_step restores theta then updates.
    loss_2 = compute_erm_loss()
    loss_2.backward()
    sam_optimizer.second_step(zero_grad=True)

    return loss_1.item()
