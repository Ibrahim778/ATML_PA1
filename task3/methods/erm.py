"""
The Task 3 ERM baseline is the Task 2 Source-only checkpoint, reused
UNCHANGED (same splits, same backbone init, same training budget). This
file only loads it -- it does not retrain anything.
"""

import torch

from task2.models.backbone import ResNet18Backbone


def load_erm_baseline(checkpoint_path: str, num_classes: int = 7, device: str = "cuda") -> ResNet18Backbone:
    backbone = ResNet18Backbone(num_classes=num_classes).to(device)
    backbone.load_state_dict(torch.load(checkpoint_path, map_location=device))
    backbone.eval()
    return backbone
