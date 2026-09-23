"""
Common interface: preprocess() applies ONLY this backbone's normalization
(no resize/crop -- that happens once, upstream, identically for all models).
extract_features() is a frozen forward pass to the penultimate representation.
"""

import torch
import torch.nn as nn
import torchvision.models as tvm
import torchvision.transforms as T
import open_clip


class BackboneWrapper:
    name: str
    feature_dim: int

    def preprocess(self, pil_img):
        raise NotImplementedError

    @torch.no_grad()
    def extract_features(self, batch: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ResNet50Backbone(BackboneWrapper):
    name = "resnet50"
    feature_dim = 2048

    def __init__(self, device="cuda"):
        weights = tvm.ResNet50_Weights.IMAGENET1K_V2
        model = tvm.resnet50(weights=weights)
        self.body = nn.Sequential(*list(model.children())[:-1]).to(device).eval()
        for p in self.body.parameters():
            p.requires_grad_(False)
        # weights.meta has NO "mean"/"std" keys in current torchvision --
        # the normalization stats live on the transforms() object instead.
        preset_transform = weights.transforms()
        self._normalize = T.Normalize(mean=preset_transform.mean, std=preset_transform.std)
        self.device = device

    def preprocess(self, pil_img):
        t = T.functional.to_tensor(pil_img)
        return self._normalize(t)

    @torch.no_grad()
    def extract_features(self, batch: torch.Tensor) -> torch.Tensor:
        batch = batch.to(self.device)
        feats = self.body(batch)
        return feats.flatten(1)


class ViTB16Backbone(BackboneWrapper):
    name = "vit_b_16"
    feature_dim = 768

    def __init__(self, device="cuda"):
        weights = tvm.ViT_B_16_Weights.IMAGENET1K_V1
        self.model = tvm.vit_b_16(weights=weights).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        preset_transform = weights.transforms()
        self._normalize = T.Normalize(mean=preset_transform.mean, std=preset_transform.std)
        self.device = device

    def preprocess(self, pil_img):
        t = T.functional.to_tensor(pil_img)
        return self._normalize(t)

    @torch.no_grad()
    def extract_features(self, batch: torch.Tensor) -> torch.Tensor:
        # torchvision's ViT has no public hook for the CLS token, so we
        # replicate its forward() up to (but not including) the classifier
        # head. Sanity-check this against model(x) / model.heads(cls_token)
        # after installing -- internal method names can shift across
        # torchvision versions.
        batch = batch.to(self.device)
        x = self.model._process_input(batch)
        n = x.shape[0]
        cls_token = self.model.class_token.expand(n, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = self.model.encoder(x)
        return x[:, 0]


class CLIPBackbone(BackboneWrapper):
    name = "clip_vit_b_32"
    feature_dim = 512

    def __init__(self, device="cuda"):
        model, _, preprocess_fn = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        self.model = model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")
        # Pull only the Normalize transform out of OpenCLIP's pipeline so we
        # don't re-run its Resize/CenterCrop on an already-224x224,
        # already-intervened image.
        self._normalize = next(t for t in preprocess_fn.transforms if isinstance(t, T.Normalize))
        self.device = device

    def preprocess(self, pil_img):
        t = T.functional.to_tensor(pil_img)
        return self._normalize(t)

    @torch.no_grad()
    def extract_features(self, batch: torch.Tensor) -> torch.Tensor:
        batch = batch.to(self.device)
        feats = self.model.encode_image(batch)
        return feats / feats.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def zero_shot_logits(self, batch: torch.Tensor, class_names: list) -> torch.Tensor:
        prompts = [f"a photo of a {c}." for c in class_names]
        tokens = self.tokenizer(prompts).to(self.device)
        text_feats = self.model.encode_text(tokens)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
        img_feats = self.extract_features(batch)
        logit_scale = self.model.logit_scale.exp()
        return logit_scale * img_feats @ text_feats.T


def load_all_backbones(device="cuda"):
    return {
        "resnet50": ResNet50Backbone(device=device),
        "vit_b_16": ViTB16Backbone(device=device),
        "clip_vit_b_32": CLIPBackbone(device=device),
    }
