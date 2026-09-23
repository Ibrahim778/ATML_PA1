"""
Evaluation routines for Task 1. Every function here assumes:
  - a fixed list of 224x224 clean PIL images + integer labels for the eval subset
  - a dict of {backbone_name: (backbone, head)} for the three trained heads
  - CLIP additionally evaluated zero-shot via backbone.zero_shot_logits()

Interventions are generated once (upstream, in transforms.py / make_cue_conflicts.py)
and passed in as PIL images so every model sees identical inputs.
"""

import numpy as np
import torch
from sklearn.metrics import f1_score
from tqdm import tqdm

from task1.data.make_subset import STL10_CLASSES


@torch.no_grad()
def predict(backbone, head, pil_images, batch_size=64, show_progress=False, desc=None):
    """Returns (pred_labels, confidences) for a list of PIL images using a
    trained linear head on frozen features. show_progress=True is useful for
    large eval sets; left off by default since predict() is called many
    times per intervention and a bar per call gets noisy."""
    preds, confs = [], []
    batch_starts = range(0, len(pil_images), batch_size)
    if show_progress:
        batch_starts = tqdm(list(batch_starts), desc=desc or "Predicting", unit="batch")
    for i in batch_starts:
        batch_imgs = pil_images[i:i + batch_size]
        tensors = torch.stack([backbone.preprocess(im) for im in batch_imgs])
        feats = backbone.extract_features(tensors)
        logits = head(feats.to(next(head.parameters()).device))
        probs = torch.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)
        preds.extend(pred.cpu().tolist())
        confs.extend(conf.cpu().tolist())
    return np.array(preds), np.array(confs)


@torch.no_grad()
def predict_clip_zero_shot(clip_backbone, pil_images, class_names=STL10_CLASSES, batch_size=64):
    preds, confs = [], []
    for i in range(0, len(pil_images), batch_size):
        batch_imgs = pil_images[i:i + batch_size]
        tensors = torch.stack([clip_backbone.preprocess(im) for im in batch_imgs])
        logits = clip_backbone.zero_shot_logits(tensors, class_names)
        probs = torch.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)
        preds.extend(pred.cpu().tolist())
        confs.extend(conf.cpu().tolist())
    return np.array(preds), np.array(confs)


def accuracy(preds, labels):
    return float((preds == labels).mean())


def macro_f1(preds, labels):
    return float(f1_score(labels, preds, average="macro"))


def consistency(preds_a, preds_b):
    """Fraction of images whose prediction is unchanged between two conditions."""
    return float((preds_a == preds_b).mean())


# ---------------------------------------------------------------------------
# 1. Clean baseline
# ---------------------------------------------------------------------------

def run_clean_baseline(models, clean_images, labels):
    """models: {name: (backbone, head_or_None)}. CLIP zero-shot handled
    separately via its own key convention 'clip_zero_shot'."""
    results = {}
    for name, (backbone, head) in models.items():
        preds, confs = predict(backbone, head, clean_images)
        results[name] = {
            "accuracy": accuracy(preds, labels),
            "macro_f1": macro_f1(preds, labels),
            "mean_max_conf": float(confs.mean()),
            "preds": preds,
        }
    return results


def run_clip_zero_shot_baseline(clip_backbone, clean_images, labels):
    preds, confs = predict_clip_zero_shot(clip_backbone, clean_images)
    return {
        "accuracy": accuracy(preds, labels),
        "macro_f1": macro_f1(preds, labels),
        "mean_max_conf": float(confs.mean()),
        "preds": preds,
    }


# ---------------------------------------------------------------------------
# 2. Color bias
# ---------------------------------------------------------------------------

def run_color_intervention(models, transformed_images, labels, clean_preds_by_model):
    """clean_preds_by_model: {name: clean_preds_array} for consistency calc."""
    results = {}
    for name, (backbone, head) in models.items():
        preds, confs = predict(backbone, head, transformed_images)
        results[name] = {
            "accuracy": accuracy(preds, labels),
            "consistency_vs_clean": consistency(preds, clean_preds_by_model[name]),
            "mean_max_conf": float(confs.mean()),
        }
    return results


# ---------------------------------------------------------------------------
# 3. Shape vs. texture (cue conflicts)
# ---------------------------------------------------------------------------

def classify_prediction(pred_label, shape_class, texture_class):
    if pred_label == shape_class:
        return "shape"
    elif pred_label == texture_class:
        return "texture"
    else:
        return "other"


def run_shape_texture_bias(models, cue_conflict_images, metadata):
    """metadata: list of dicts with 'shape_class' and 'texture_class' per image,
    aligned index-for-index with cue_conflict_images."""
    results = {}
    for name, (backbone, head) in models.items():
        preds, _ = predict(backbone, head, cue_conflict_images)

        n_shape = n_texture = n_other = 0
        per_image = []
        for pred, meta in zip(preds, metadata):
            cat = classify_prediction(pred, meta["shape_class"], meta["texture_class"])
            per_image.append(cat)
            if cat == "shape":
                n_shape += 1
            elif cat == "texture":
                n_texture += 1
            else:
                n_other += 1

        n_total = len(metadata)
        shape_bias = 100.0 * n_shape / max(n_shape + n_texture, 1)
        coverage = 100.0 * (n_shape + n_texture) / max(n_total, 1)

        results[name] = {
            "n_shape": n_shape, "n_texture": n_texture, "n_other": n_other,
            "shape_bias_pct": shape_bias, "coverage_pct": coverage,
            "per_image_category": per_image,
        }
    return results


# ---------------------------------------------------------------------------
# 4. Translation
# ---------------------------------------------------------------------------

def run_translation_curve(models, clean_images, labels, magnitudes=(0, 8, 16, 32)):
    """For each magnitude, translate every image in all 4 cardinal directions,
    predict on each, then average accuracy/consistency across directions.
    Returns {model_name: {magnitude: {"accuracy": ..., "consistency": ...}}}."""
    from task1.data.transforms import translate_all_directions
    from tqdm import tqdm

    clean_preds_by_model = {}
    for name, (backbone, head) in models.items():
        clean_preds, _ = predict(backbone, head, clean_images)
        clean_preds_by_model[name] = clean_preds

    curves = {name: {} for name in models}
    total_calls = len(magnitudes) * len(models) * 4  # 4 cardinal directions
    pbar = tqdm(total=total_calls, desc="Translation curve", unit="eval")

    for mag in magnitudes:
        tqdm.write(f"[translation] magnitude={mag}px: building shifted images ...")
        # Build per-direction image sets once, reused across all models.
        per_direction_images = [translate_all_directions(im, mag) for im in clean_images]
        directions = list(per_direction_images[0].keys())

        for name, (backbone, head) in models.items():
            dir_accs, dir_cons = [], []
            for d in directions:
                dir_images = [per_direction_images[i][d] for i in range(len(clean_images))]
                preds, _ = predict(backbone, head, dir_images)
                dir_accs.append(accuracy(preds, labels))
                dir_cons.append(consistency(preds, clean_preds_by_model[name]))
                pbar.update(1)
            curves[name][mag] = {
                "accuracy": float(np.mean(dir_accs)),
                "consistency": float(np.mean(dir_cons)),
            }
            tqdm.write(f"  [{name}] mag={mag}: acc={curves[name][mag]['accuracy']:.4f} "
                       f"consistency={curves[name][mag]['consistency']:.4f}")
    pbar.close()
    return curves


# ---------------------------------------------------------------------------
# 5. Patch shuffle
# ---------------------------------------------------------------------------

def run_patch_shuffle(models, shuffled_images, labels, clean_preds_by_model):
    results = {}
    for name, (backbone, head) in models.items():
        preds, _ = predict(backbone, head, shuffled_images)
        results[name] = {
            "accuracy": accuracy(preds, labels),
            "consistency_vs_clean": consistency(preds, clean_preds_by_model[name]),
        }
    return results
