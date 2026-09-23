"""
Orchestrates the full Task 1 pipeline:
  1. Data prep (splits + eval subset)
  2. Backbone feature caching + linear head training
  3. Clean baseline
  4. Color bias (grayscale + hue rotation)
  5. Shape vs texture (cue conflicts)
  6. Translation curve
  7. Patch shuffle
  8. Representation stability (cosine) + UMAP plots
  9. Save everything to task1/results/

Run with:  python -m task1.scripts.run_task1
"""

import json
import os
import time

import numpy as np
import torch
from PIL import Image
from torchvision.datasets import STL10
from tqdm import tqdm

from common.seed import set_global_seed
from task1.data.make_subset import make_train_val_split, make_balanced_eval_subset, STL10_CLASSES
from task1.data.transforms import Grayscale, HueRotate, PatchShuffle, translate_all_directions
from task1.data.make_cue_conflicts import generate_cue_conflicts
from task1.models.download_adain_weights import download_adain_weights
from task1.models.backbones import load_all_backbones
from task1.models.extract_cache import extract_and_cache_features
from task1.models.heads import train_linear_head
from task1.analysis.evaluate_bias import (
    run_clean_baseline, run_clip_zero_shot_baseline, run_color_intervention,
    run_shape_texture_bias, run_translation_curve, run_patch_shuffle, predict,
)
from task1.analysis.feature_similarity import extract_features_batch, cosine_stability
from task1.analysis.representation import run_all_umap_plots

SEED = 6304
ROOT = "./data"
IMAGE_SIZE = 224
RESULTS_DIR = "task1/results"
DEVICE = "mps" if torch.cuda.is_available() else "cpu"


def _stage(n: int, total: int, title: str):
    """Prints a visible banner marking the start of a pipeline stage, so
    long unattended runs show what's currently happening."""
    print(f"\n{'=' * 60}\n[Stage {n}/{total}] {title}\n{'=' * 60}", flush=True)

def load_or_extract_features(backbone, idx, root, split, cache_path, desc):
    if os.path.isfile(cache_path):
        cached = torch.load(cache_path, weights_only=False)
        if cached.get("indices") == list(idx):
            tqdm.write(f"  found cached features at {cache_path}, loading instead of extracting")
            return cached["features"], cached["labels"]
        tqdm.write(f"  cache at {cache_path} has mismatched indices, re-extracting")
    return extract_and_cache_features(
        backbone, idx, root, split, cache_path, device=DEVICE, desc=desc)

def load_pil_images(split, indices, image_size=IMAGE_SIZE):
    ds = STL10(root=ROOT, split=split, download=True)
    images, labels = [], []
    for idx in indices:
        img, label = ds[idx]
        img = img.convert("RGB").resize((image_size, image_size), Image.BICUBIC)
        images.append(img)
        labels.append(label)
    return images, np.array(labels)


def main():
    t_start = time.time()
    set_global_seed(SEED)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    TOTAL_STAGES = 9

    # ---------------- 1. Data prep ----------------
    _stage(1, TOTAL_STAGES, "Data prep (splits + eval subset)")
    train_idx, val_idx, _ = make_train_val_split(ROOT)
    eval_idx, imbalance = make_balanced_eval_subset(ROOT)
    print(f"Train/val/eval sizes: {len(train_idx)}/{len(val_idx)}/{len(eval_idx)}")

    print("Loading and resizing the 500-image eval subset ...")
    clean_images, labels = load_pil_images("test", eval_idx)
    print(f"Eval subset ready ({len(clean_images)} images).")

        # ---------------- 2. Backbones + heads ----------------
    _stage(2, TOTAL_STAGES, "Backbone feature caching + linear head training")
    print("Loading pretrained backbones (ResNet-50, ViT-B/16, CLIP ViT-B/32) ...")
    backbones = load_all_backbones(device=DEVICE)
    models = {}  # name -> (backbone, head)

    for name, backbone in tqdm(backbones.items(), desc="Backbones", unit="backbone"):
        cache_dir = os.path.join(RESULTS_DIR, "feature_cache")
        os.makedirs(cache_dir, exist_ok=True)

        train_path = os.path.join(cache_dir, f"{name}_train.pt")
        val_path = os.path.join(cache_dir, f"{name}_val.pt")

        train_feats, train_labels = load_or_extract_features(
            backbone, train_idx, ROOT, "train", train_path,
            desc=f"[{name}] extracting train features")
        val_feats, val_labels = load_or_extract_features(
            backbone, val_idx, ROOT, "train", val_path,
            desc=f"[{name}] extracting val features")

        tqdm.write(f"[{name}] training linear head ...")
        head, best_val_acc = train_linear_head(
            train_feats, train_labels, val_feats, val_labels,
            feature_dim=backbone.feature_dim, device=DEVICE, seed=SEED,
        )
        tqdm.write(f"[{name}] linear head best val acc: {best_val_acc:.4f}")
        models[name] = (backbone, head)

    # ---------------- 3. Clean baseline ----------------
    _stage(3, TOTAL_STAGES, "Clean baseline evaluation")
    clean_results = run_clean_baseline(models, clean_images, labels)
    clip_zs_results = run_clip_zero_shot_baseline(backbones["clip_vit_b_32"], clean_images, labels)
    for name, res in clean_results.items():
        print(f"  [{name}] clean acc={res['accuracy']:.4f}  macro-F1={res['macro_f1']:.4f}")
    print(f"  [clip_zero_shot] acc={clip_zs_results['accuracy']:.4f}  macro-F1={clip_zs_results['macro_f1']:.4f}")

    clean_preds_by_model = {name: np.array(res["preds"]) for name, res in clean_results.items()}

    # ---------------- 4. Color bias ----------------
    _stage(4, TOTAL_STAGES, "Color bias (grayscale + hue rotation)")
    print("Generating grayscale images ...")
    gray_images = [Grayscale()(im) for im in clean_images]
    print("Generating hue-rotated images ...")
    hue_images = [HueRotate(degrees=90.0)(im) for im in clean_images]

    print("Evaluating grayscale intervention ...")
    gray_results = run_color_intervention(models, gray_images, labels, clean_preds_by_model)
    print("Evaluating hue-rotation intervention ...")
    hue_results = run_color_intervention(models, hue_images, labels, clean_preds_by_model)
    color_results = {"grayscale": gray_results, "hue_rotate_90": hue_results}

    # ---------------- 5. Shape vs texture (cue conflicts) ----------------
    _stage(5, TOTAL_STAGES, "Shape vs. texture (AdaIN cue conflicts)")
    # NOTE: generated from the STL-10 TRAIN partition (separate from the eval
    # subset), per make_cue_conflicts.py. Uses true AdaIN (Huang & Belongie,
    # 2017) via pretrained weights from naoto0804/pytorch-AdaIN -- see
    # task1/models/adain_net.py for attribution. Run the pilot in
    # make_cue_conflicts.run_pilot() and recalibrate CONTENT_LOSS_MAX /
    # STYLE_LOSS_MAX BEFORE trusting the full run below.
    download_adain_weights()
    cue_meta_path = "task1/data/cue_conflicts/cue_conflict_metadata.json"
    if not os.path.exists(cue_meta_path):
        print("No cached cue conflicts found -- generating now (this is the slowest stage) ...")
        generate_cue_conflicts(root=ROOT, device=DEVICE)
    else:
        print(f"Found cached cue conflicts at {cue_meta_path}, skipping generation.")

    with open(cue_meta_path) as f:
        cue_data = json.load(f)
    cue_metadata = cue_data["metadata"]
    cue_conflict_dir = "task1/data/cue_conflicts"
    print(f"Loading {len(cue_metadata)} cue-conflict images ...")
    cue_images = [Image.open(os.path.join(cue_conflict_dir, m["file"])).convert("RGB")
                  for m in tqdm(cue_metadata, desc="Loading cue conflicts", unit="img")]

    print("Evaluating shape/texture bias across models ...")
    shape_texture_results = run_shape_texture_bias(models, cue_images, cue_metadata)
    for name, res in shape_texture_results.items():
        print(f"  [{name}] shape_bias={res['shape_bias_pct']:.1f}%  coverage={res['coverage_pct']:.1f}%")

    # ---------------- 6. Translation ----------------
    _stage(6, TOTAL_STAGES, "Translation curve (0/8/16/32 px, 4 directions)")
    translation_results = run_translation_curve(models, clean_images, labels,
                                                  magnitudes=(0, 8, 16, 32))

    # ---------------- 7. Patch shuffle ----------------
    _stage(7, TOTAL_STAGES, "Patch shuffle (4x4 grid)")
    print("Generating shuffled images ...")
    patch_shuffler = PatchShuffle(grid_size=4, seed=SEED)
    shuffled_images = [patch_shuffler(im, image_index=i)
                        for i, im in enumerate(tqdm(clean_images, desc="Patch shuffle", unit="img"))]
    print("Evaluating patch-shuffle intervention ...")
    patch_shuffle_results = run_patch_shuffle(models, shuffled_images, labels, clean_preds_by_model)
    for name, res in patch_shuffle_results.items():
        print(f"  [{name}] acc={res['accuracy']:.4f}  consistency={res['consistency_vs_clean']:.4f}")

    # ---------------- 8. Representation stability + UMAP ----------------
    _stage(8, TOTAL_STAGES, "Representation stability (cosine) + UMAP")
    # Translation uses magnitude-32, "right" direction as the representative
    # transformed condition for representation analysis (document this choice).
    translate_32_images = [translate_all_directions(im, 32)["right"] for im in clean_images]

    rep_stability = {}
    clean_feats_by_model = {}
    trans_feats_by_condition_by_model = {}

    for name, (backbone, _head) in tqdm(models.items(), desc="Extracting features for representation analysis", unit="backbone"):
        clean_feats = extract_features_batch(backbone, clean_images)
        clean_feats_by_model[name] = clean_feats.numpy()

        conditions = {
            "grayscale": gray_images,
            "translation_32": translate_32_images,
            "patch_shuffle": shuffled_images,
        }
        rep_stability[name] = {}
        trans_feats_by_condition_by_model[name] = {}
        for cond_name, images in conditions.items():
            trans_feats = extract_features_batch(backbone, images)
            rep_stability[name][cond_name] = cosine_stability(clean_feats, trans_feats)
            trans_feats_by_condition_by_model[name][cond_name] = trans_feats.numpy()
            tqdm.write(f"  [{name}] {cond_name}: I_T={rep_stability[name][cond_name]:.4f}")

        # Cue-conflict representation stability pairs against the content
        # (shape) source image, not the fixed eval subset -- handled
        # separately since it's a different image set with its own indices.

    print("Fitting UMAP projections and saving plots ...")
    run_all_umap_plots(models, clean_feats_by_model, trans_feats_by_condition_by_model, labels,
                        out_dir=os.path.join(RESULTS_DIR, "umap"))

    # ---------------- 9. Save everything ----------------
    _stage(9, TOTAL_STAGES, "Saving results")
    def strip_preds(d):
        """Drop raw prediction arrays before JSON serialization."""
        out = {}
        for k, v in d.items():
            if isinstance(v, dict):
                v = {kk: vv for kk, vv in v.items() if kk != "preds"}
            out[k] = v
        return out

    summary = {
        "clean_baseline": strip_preds(clean_results),
        "clip_zero_shot_baseline": {k: v for k, v in clip_zs_results.items() if k != "preds"},
        "color_bias": color_results,
        "shape_texture_bias": {
            name: {k: v for k, v in res.items() if k != "per_image_category"}
            for name, res in shape_texture_results.items()
        },
        "translation_curve": translation_results,
        "patch_shuffle": patch_shuffle_results,
        "representation_stability": rep_stability,
        "cue_conflict_counts": {"accepted": cue_data["n_accepted"], "rejected": cue_data["n_rejected"]},
        "eval_subset_imbalance": imbalance,
    }

    with open(os.path.join(RESULTS_DIR, "task1_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    elapsed = time.time() - t_start
    print(f"\n{'=' * 60}\nDone in {elapsed / 60:.1f} min. "
          f"Summary written to {RESULTS_DIR}/task1_summary.json\n{'=' * 60}")


if __name__ == "__main__":
    main()
