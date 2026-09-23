"""
DAN-style alignment: classification loss over labeled sources plus a
lambda-weighted MMD penalty between pooled source features and unlabeled
target features, applied to the 512-d feature before the classifier head.
"""

import torch
import torch.nn as nn

from shared.mmd import multi_kernel_mmd


class DAN:
    name = "dan"

    def __init__(self, feature_dim: int, num_classes: int, device: str, lambda_mmd: float = 1.0, **kwargs):
        """**kwargs absorbs config keys meant for other methods (e.g.
        max_alpha), which train.py passes through unfiltered since
        base.yaml sets them as universal defaults."""
        self.criterion = nn.CrossEntropyLoss()
        self.lambda_mmd = lambda_mmd

    def extra_parameters(self):
        return []

    def compute_loss(self, backbone, source_batches: dict, target_batch, progress_p: float):
        x_t, _ = target_batch  # target labels never used
        cls_loss = 0.0
        source_feats = []
        for dom, (x, y) in source_batches.items():
            logits, feat = backbone(x, return_features=True)
            cls_loss = cls_loss + self.criterion(logits, y)
            source_feats.append(feat)
        cls_loss = cls_loss / len(source_batches)
        source_feats = torch.cat(source_feats, dim=0)

        _, target_feats = backbone(x_t, return_features=True)
        mmd = multi_kernel_mmd(source_feats, target_feats)

        loss = cls_loss + self.lambda_mmd * mmd
        return loss, {"cls_loss": cls_loss.item(), "mmd": mmd.item()}