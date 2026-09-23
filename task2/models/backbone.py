"""
ResNet-18 backbone + 7-class head, with the BatchNorm freezing policy
mandated by the spec: running_mean/running_var stay at their pretrained
ImageNet values throughout Tasks 2 and 3 (so BN statistics never depend on
the source-target mixture); the BN scale/bias (gamma/beta) remain trainable.
"""

import torch
import torch.nn as nn
import torchvision.models as tvm


class ResNet18Backbone(nn.Module):
    def __init__(self, num_classes: int = 7):
        super().__init__()
        weights = tvm.ResNet18_Weights.IMAGENET1K_V1
        base = tvm.resnet18(weights=weights)
        self.feature_dim = base.fc.in_features  # 512
        self.features = nn.Sequential(*list(base.children())[:-1])  # everything up to avgpool
        self.fc = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        feat = self.features(x).flatten(1)
        logits = self.fc(feat)
        if return_features:
            return logits, feat
        return logits

    def freeze_bn_running_stats(self):
        """Call every step AFTER model.train(). Puts only BatchNorm modules
        into eval mode (so running_mean/var aren't updated) while leaving
        everything else -- including BN's own gamma/beta -- trainable.
        Do NOT call model.eval() on the whole network; that would also
        disable dropout/etc. elsewhere and stop gradient flow assumptions
        this training loop relies on."""
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
