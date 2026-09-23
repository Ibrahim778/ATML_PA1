"""
DAN-DG: ERM classification loss plus the average MMD discrepancy over the
three unordered pairs of SOURCE domains -- never touches Sketch. Uses the
same MMD implementation/kernel construction as Task 2's DAN so the
discrepancy measure itself is held fixed; only what it's applied to
(source-source pairs, not source-target) differs.
"""

import torch
import torch.nn as nn

from shared.mmd import pairwise_source_mmd


class DANDG:
    name = "dan_dg"

    def __init__(self, feature_dim: int, num_classes: int, device: str, lambda_dg: float = 1.0):
        self.criterion = nn.CrossEntropyLoss()
        self.lambda_dg = lambda_dg

    def extra_parameters(self):
        return []

    def compute_loss(self, backbone, source_batches: dict, target_batch, progress_p: float):
        """target_batch is always None/ignored here -- Task 3 has no target access."""
        cls_loss = 0.0
        feats_by_domain = {}
        for dom, (x, y) in source_batches.items():
            logits, feat = backbone(x, return_features=True)
            cls_loss = cls_loss + self.criterion(logits, y)
            feats_by_domain[dom] = feat
        cls_loss = cls_loss / len(source_batches)

        mmd = pairwise_source_mmd(feats_by_domain)
        loss = cls_loss + self.lambda_dg * mmd
        return loss, {"cls_loss": cls_loss.item(), "pairwise_mmd": mmd.item()}
