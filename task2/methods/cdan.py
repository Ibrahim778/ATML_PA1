"""
CDAN: the domain discriminator sees g(x) = vec(f (x) p) -- the outer
product of feature and predicted class-probability vector -- instead of
just the feature. Same discriminator architecture, GRL schedule, and loss
weight as DANN. No entropy conditioning; f and p are not detached, per spec.
"""

import torch
import torch.nn as nn

from shared.grl import GradientReversalLayer, grl_alpha_schedule
from task2.models.domain_discriminator import DomainDiscriminator, cdan_multilinear_map


class CDAN:
    name = "cdan"

    def __init__(self, feature_dim: int, num_classes: int, device: str, max_alpha: float = 1.0, **kwargs):
        """**kwargs absorbs config keys meant for other methods (e.g.
        lambda_mmd), which train.py passes through unfiltered since
        base.yaml sets them as universal defaults."""
        self.criterion = nn.CrossEntropyLoss()
        self.domain_criterion = nn.CrossEntropyLoss()
        self.discriminator = DomainDiscriminator(feature_dim * num_classes).to(device)
        self.grl = GradientReversalLayer()
        self.device = device
        self.num_classes = num_classes
        self.max_alpha = max_alpha

    def extra_parameters(self):
        return list(self.discriminator.parameters())

    def compute_loss(self, backbone, source_batches: dict, target_batch, progress_p: float):
        self.grl.alpha = self.max_alpha * grl_alpha_schedule(progress_p)

        x_t, _ = target_batch
        cls_loss = 0.0
        source_feats, source_logits_list = [], []
        for dom, (x, y) in source_batches.items():
            logits, feat = backbone(x, return_features=True)
            cls_loss = cls_loss + self.criterion(logits, y)
            source_feats.append(feat)
            source_logits_list.append(logits)
        cls_loss = cls_loss / len(source_batches)
        source_feats = torch.cat(source_feats, dim=0)
        source_logits = torch.cat(source_logits_list, dim=0)

        target_logits, target_feats = backbone(x_t, return_features=True)

        all_feats = torch.cat([source_feats, target_feats], dim=0)
        all_logits = torch.cat([source_logits, target_logits], dim=0)
        all_probs = torch.softmax(all_logits, dim=1)  # not detached, per spec

        g = cdan_multilinear_map(all_feats, all_probs)
        reversed_g = self.grl(g)
        domain_logits = self.discriminator(reversed_g)

        domain_labels = torch.cat([
            torch.zeros(source_feats.size(0), dtype=torch.long),
            torch.ones(target_feats.size(0), dtype=torch.long),
        ]).to(self.device)
        domain_loss = self.domain_criterion(domain_logits, domain_labels)

        loss = cls_loss + domain_loss
        return loss, {"cls_loss": cls_loss.item(), "domain_loss": domain_loss.item(),
                       "grl_alpha": self.grl.alpha}