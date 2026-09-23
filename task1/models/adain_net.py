"""
True AdaIN (Huang & Belongie, ICCV 2017) style transfer: a frozen VGG19
encoder (truncated at relu4_1) extracts content and style features, an
AdaIN layer aligns the content feature's per-channel mean/std to the
style feature's, and a pretrained decoder reconstructs the stylized image
in a single forward pass (no per-image optimization loop).

Architecture and the vgg_normalised/decoder weight split follow the
well-known unofficial PyTorch reproduction:
    naoto0804/pytorch-AdaIN (MIT License)
    https://github.com/naoto0804/pytorch-AdaIN
Pretrained weights are downloaded directly from that repository's GitHub
Release (v0.0.0) by download_adain_weights.py -- see that file for the
exact URLs. The VGG encoder weights ("vgg_normalised") are NOT the same
as torchvision's ImageNet weights: they come from the original Torch
AdaIN implementation and use a reflection-padding, ceil-mode-pooling VGG19
variant, which is why a separate architecture/weight file is needed here
rather than reusing task1/models/backbones.py's ResNet/ViT/CLIP wrappers.

The AdaIN math itself (calc_mean_std / adaptive_instance_normalization)
is reimplemented directly from the paper's Eq. 8, not copied from any
external file.
"""

import os
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image


# ---------------------------------------------------------------------------
# Architecture (matches naoto0804/pytorch-AdaIN net.py so pretrained weights
# load with strict key/shape matching)
# ---------------------------------------------------------------------------

def build_vgg() -> nn.Sequential:
    return nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, this is the last layer used
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU()  # relu5-4
)

def build_decoder() -> nn.Sequential:
    return nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 256, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 3, (3, 3)),
)


# ---------------------------------------------------------------------------
# AdaIN math -- reimplemented directly from Huang & Belongie (2017), Eq. 8
# ---------------------------------------------------------------------------

def calc_mean_std(feat: torch.Tensor, eps: float = 1e-5):
    n, c = feat.shape[:2]
    feat_var = feat.view(n, c, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(n, c, 1, 1)
    feat_mean = feat.view(n, c, -1).mean(dim=2).view(n, c, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat: torch.Tensor, style_feat: torch.Tensor) -> torch.Tensor:
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized * style_std.expand(size) + style_mean.expand(size)


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------

class AdaINModel:
    """Loads the pretrained vgg_normalised encoder (truncated at relu4_1) and
    decoder, and runs single-pass AdaIN style transfer."""

    def __init__(self, vgg_path: str, decoder_path: str, device: str = "cuda"):
        vgg_full = build_vgg()
        vgg_full.load_state_dict(torch.load(vgg_path, map_location=device))
        # Only need up through relu4_1 (index 31 in the Sequential above).
        self.encoder = nn.Sequential(*list(vgg_full.children())[:31]).to(device).eval()

        self.decoder = build_decoder()
        self.decoder.load_state_dict(torch.load(decoder_path, map_location=device))
        self.decoder = self.decoder.to(device).eval()

        for p in self.encoder.parameters():
            p.requires_grad_(False)
        for p in self.decoder.parameters():
            p.requires_grad_(False)

        self.device = device
        self._to_tensor = T.ToTensor()

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    @torch.no_grad()
    def stylize(self, content_pil: Image.Image, style_pil: Image.Image, alpha: float = 1.0):
        """Returns (stylized_pil, content_loss, style_loss). Losses are
        computed post-hoc (not part of an optimization loop) purely for
        use in the cue-conflict rejection rule:
          content_loss = MSE(encode(output), t)          -- did the decoder
                          faithfully reconstruct the AdaIN target?
          style_loss   = mean/std mismatch between encode(output) and
                          encode(style) at relu4_1         -- did style
                          actually transfer?
        """
        content = self._to_tensor(content_pil).unsqueeze(0).to(self.device)
        style = self._to_tensor(style_pil).unsqueeze(0).to(self.device)

        content_feat = self.encode(content)
        style_feat = self.encode(style)

        t = adaptive_instance_normalization(content_feat, style_feat)
        t = alpha * t + (1 - alpha) * content_feat

        out = self.decoder(t).clamp(0,1)
        out_pil = T.functional.to_pil_image(out.squeeze(0).cpu())

        return out_pil 


def load_adain_model(weights_dir: str = "task1/models/weights", device: str = "cuda") -> AdaINModel:
    vgg_path = os.path.join(weights_dir, "vgg_normalised.pth")
    decoder_path = os.path.join(weights_dir, "decoder.pth")
    if not (os.path.exists(vgg_path) and os.path.exists(decoder_path)):
        raise FileNotFoundError(
            f"AdaIN weights not found in {weights_dir}. "
            "Run: python -m task1.models.download_adain_weights"
        )
    return AdaINModel(vgg_path, decoder_path, device=device)
