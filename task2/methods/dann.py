"""
DANN: a binary domain discriminator tries to distinguish pooled source
features from target features; a gradient-reversal layer sends the
opposite gradient back to the backbone, pushing it toward domain confusion.
Only source examples contribute to the classification loss; both source
and target contribute to the domain loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from shared.grl import GradientReversalLayer, grl_alpha_schedule
from task2.models.domain_discriminator import DomainDiscriminator


class DANN:
    name = "dann"

    def __init__(self, feature_dim: int, num_classes: int, device: str, max_alpha: float = 1.0, **kwargs):
        """**kwargs absorbs config keys meant for other methods (e.g.
        lambda_mmd), which train.py passes through unfiltered since
        base.yaml sets them as universal defaults."""
        self.criterion = nn.CrossEntropyLoss()
        self.domain_criterion = nn.CrossEntropyLoss()
        self.discriminator = DomainDiscriminator(feature_dim).to(device)
        self.grl = GradientReversalLayer()
        self.device = device
        self.max_alpha = max_alpha

    def extra_parameters(self):
        return list(self.discriminator.parameters())

    def compute_loss(self, backbone, source_batches: dict, target_batch, progress_p: float):
        self.grl.alpha = self.max_alpha * grl_alpha_schedule(progress_p)

        x_t, _ = target_batch
        cls_loss = 0.0
        source_feats = []
        for dom, (x, y) in source_batches.items():
            logits, feat = backbone(x, return_features=True)
            cls_loss = cls_loss + self.criterion(logits, y)
            source_feats.append(feat)
        cls_loss = cls_loss / len(source_batches)
        source_feats = torch.cat(source_feats, dim=0)

        _, target_feats = backbone(x_t, return_features=True)

        all_feats = torch.cat([source_feats, target_feats], dim=0)
        # L2-normalize before the GRL/discriminator: unbounded feature norms
        # let the classifier and discriminator inflate each other's
        # gradients (this is what was driving cls_loss/domain_loss to blow
        # up over epochs). Classification logits above are computed from
        # the un-normalized feat inside backbone(), so this only affects
        # the adversarial branch.
        all_feats = F.normalize(all_feats, p=2, dim=1)

        domain_labels = torch.cat([
            torch.zeros(source_feats.size(0), dtype=torch.long),
            torch.ones(target_feats.size(0), dtype=torch.long),
        ]).to(self.device)

        reversed_feats = self.grl(all_feats)
        domain_logits = self.discriminator(reversed_feats)
        domain_loss = self.domain_criterion(domain_logits, domain_labels)

        loss = cls_loss + domain_loss
        return loss, {"cls_loss": cls_loss.item(), "domain_loss": domain_loss.item(),
                       "grl_alpha": self.grl.alpha}