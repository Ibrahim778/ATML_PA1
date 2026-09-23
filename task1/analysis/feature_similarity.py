"""
Representation stability: for each backbone, pair every transformed image
with its clean counterpart and measure cosine similarity between features,
averaged over the eval subset (I_T in the assignment spec).
"""

import torch


@torch.no_grad()
def extract_features_batch(backbone, pil_images, batch_size=64):
    feats = []
    for i in range(0, len(pil_images), batch_size):
        batch_imgs = pil_images[i:i + batch_size]
        tensors = torch.stack([backbone.preprocess(im) for im in batch_imgs])
        feats.append(backbone.extract_features(tensors).cpu())
    return torch.cat(feats, dim=0)


def cosine_stability(clean_feats: torch.Tensor, transformed_feats: torch.Tensor) -> float:
    """I_T = mean cosine similarity between paired clean/transformed features."""
    clean_n = torch.nn.functional.normalize(clean_feats, dim=1)
    trans_n = torch.nn.functional.normalize(transformed_feats, dim=1)
    sims = (clean_n * trans_n).sum(dim=1)
    return float(sims.mean().item())


def run_representation_stability(models, clean_images, transformed_images_by_condition):
    """
    transformed_images_by_condition: {"grayscale": [...], "cue_conflict": [...],
    "translation_32": [...], "patch_shuffle": [...]} -- images must be paired
    index-for-index with clean_images (except cue_conflict, which pairs with
    its own matched content images -- see run_task1.py for how that's assembled).

    Returns {model_name: {condition: I_T}}.
    """
    results = {name: {} for name in models}
    for name, (backbone, _head) in models.items():
        clean_feats = extract_features_batch(backbone, clean_images)
        for condition, images in transformed_images_by_condition.items():
            trans_feats = extract_features_batch(backbone, images)
            results[name][condition] = cosine_stability(clean_feats, trans_feats)
    return results
