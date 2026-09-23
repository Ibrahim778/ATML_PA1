"""
torchvision's ResNet-18 with the ImageNet 7x7/stride-2 stem replaced by a
3x3/stride-1 conv and the initial max-pool removed, so it operates
correctly on 32x32 CIFAR images instead of immediately downsampling them
to near-nothing. Trained from random initialization (not ImageNet
pretrained), per spec.
"""

import torch
import torch.nn as nn
import torchvision.models as tvm


class CIFARResNet18(nn.Module):
    def __init__(self, num_classes: int = 10, num_dummy_classifiers: int = 0):
        super().__init__()
        base = tvm.resnet18(weights=None)
        base.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        base.maxpool = nn.Identity()

        self.feature_dim = base.fc.in_features  # 512
        # Keep individual layer references so PROSER can hook in after layer2.
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = base.avgpool
        self.fc = nn.Linear(self.feature_dim, num_classes)

        self.num_dummy_classifiers = num_dummy_classifiers
        if num_dummy_classifiers > 0:
            self.dummy_fc = nn.Linear(self.feature_dim, num_dummy_classifiers)
        else:
            self.dummy_fc = None

    def forward_up_to_layer2(self, x):
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x); x = self.maxpool(x)
        x = self.layer1(x); x = self.layer2(x)
        return x

    def forward_from_layer3(self, h):
        x = self.layer3(h); x = self.layer4(x)
        x = self.avgpool(x)
        feat = torch.flatten(x, 1)
        logits = self.fc(feat)
        dummy_logits = self.dummy_fc(feat) if self.dummy_fc is not None else None
        return logits, dummy_logits, feat

    def forward(self, x, return_features: bool = False, return_dummy: bool = False):
        h = self.forward_up_to_layer2(x)
        logits, dummy_logits, feat = self.forward_from_layer3(h)
        if return_features and return_dummy:
            return logits, dummy_logits, feat
        if return_features:
            return logits, feat
        if return_dummy:
            return logits, dummy_logits
        return logits

    def add_dummy_classifiers(self, num_dummy_classifiers: int):
        """Attach randomly-initialized dummy classifiers for PROSER,
        initialized from an already-trained Vanilla checkpoint."""
        self.num_dummy_classifiers = num_dummy_classifiers
        self.dummy_fc = nn.Linear(self.feature_dim, num_dummy_classifiers)
