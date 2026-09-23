"""
Cross-entropy over the three labeled source domains, domain-balanced
batches. This checkpoint is saved and reused UNCHANGED as the Task 3 ERM
baseline -- do not retrain it separately under a different configuration.
"""

import torch.nn as nn


class SourceOnly:
    name = "source_only"

    def __init__(self, feature_dim: int, num_classes: int, device: str, **kwargs):
        """**kwargs absorbs config keys meant for other methods (e.g.
        lambda_mmd, max_alpha), which train.py passes through unfiltered
        since base.yaml sets them as universal defaults."""
        self.criterion = nn.CrossEntropyLoss()

    def extra_parameters(self):
        return []

    def compute_loss(self, backbone, source_batches: dict, target_batch, progress_p: float):
        """source_batches: {domain: (x, y)}. target_batch is ignored (no
        target access in source-only training)."""
        cls_loss = 0.0
        for dom, (x, y) in source_batches.items():
            logits = backbone(x)
            cls_loss = cls_loss + self.criterion(logits, y)
        cls_loss = cls_loss / len(source_batches)
        return cls_loss, {"cls_loss": cls_loss.item()}