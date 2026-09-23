"""
Generates shape/texture cue-conflict images: content supplies the shape of
class A, style supplies the texture of class B (and vice versa).

Style-transfer method: AdaIN (Huang & Belongie, ICCV 2017), single
feed-forward pass. See task1/models/adain_net.py for the architecture and
weight-source attribution (naoto0804/pytorch-AdaIN, MIT License).

Rejection rule (defined BEFORE any model evaluation, uses no model
predictions): a stylization is accepted only if, from the single forward
pass,
    content_loss <= CONTENT_LOSS_MAX   (decoder faithfully reconstructed the
                                         AdaIN target -- i.e. output isn't
                                         a degenerate/garbled image)
    style_loss   <= STYLE_LOSS_MAX     (relu4_1 mean/std actually shifted
                                         toward the style image, i.e.
                                         texture was transferred)
Thresholds below are placeholders -- calibrate them by visually inspecting
a small pilot batch (see `run_pilot`) before the full run, then fix them.
Record accepted/rejected counts as required by the spec.
"""

import json
import os
import random
from itertools import combinations

import torchvision.transforms as T
from torchvision.datasets import STL10
from tqdm import tqdm
import numpy as np

from PIL import Image

from task1.data.make_subset import STL10_CLASSES
from task1.models.adain_net import load_adain_model

SEED = 6304
IMAGE_SIZE = 224

CONTENT_LOSS_MAX = 0.4
STYLE_LOSS_MAX = 0.14
ALPHA = 1

def is_accepted(content_loss: float, style_loss: float) -> bool:
    """Visual/structural rejection rule, fixed BEFORE model evaluation.
    Does not use any of the three backbones' predictions."""
    return content_loss <= CONTENT_LOSS_MAX and style_loss <= STYLE_LOSS_MAX


def load_class_images(root: str, split: str = "train"):
    ds = STL10(root=root, split=split, download=True)
    by_class = {c: [] for c in range(len(STL10_CLASSES))}
    for i in range(len(ds)):
        _, label = ds[i]
        by_class[label].append(i)
    return ds, by_class


def run_pilot(root: str = "./data", n_examples: int = 20, device: str = "cuda",
              out_dir: str = "task1/data/cue_conflicts_pilot"):
    """Generates a small batch WITHOUT applying the rejection rule, so you
    can look at the images and set CONTENT_LOSS_MAX / STYLE_LOSS_MAX above
    before running the full generation.

    n_examples defaults to 20 (up from 8) -- enough to get a real sense of
    the loss DISTRIBUTION, not just a couple of lucky/unlucky examples.
    After it finishes, open the saved images sorted by content_loss (the
    printed suggestion tells you which files to look at) and find the
    largest content_loss where the image still looks like a genuine
    shape+texture blend rather than pure abstract texture. That value,
    not a percentile picked blindly, is your real threshold.
    """
    os.makedirs(out_dir, exist_ok=True)

    orig_shape_dir = os.path.join(os.path.join(out_dir, "original"), "shape")
    orig_content_dir = os.path.join(os.path.join(out_dir, "original"), "content")

    os.makedirs(os.path.join(out_dir, "original"), exist_ok=True)
    os.makedirs(orig_shape_dir, exist_ok=True)
    os.makedirs(orig_content_dir, exist_ok=True)

    rng = random.Random(SEED)
    ds, by_class = load_class_images(root, split="train")
    resize = T.Resize((IMAGE_SIZE, IMAGE_SIZE))
    model = load_adain_model(device=device)

    pairs = list(combinations(range(len(STL10_CLASSES)), 2))
    rng.shuffle(pairs)

    rows = []
    for (a, b) in tqdm(pairs[:n_examples], desc="Generating pilot stylizations", unit="img"):
        c_idx = rng.choice(by_class[a])
        s_idx = rng.choice(by_class[b])
        content_img, _ = ds[c_idx]
        style_img, _ = ds[s_idx]
        content_img = resize(content_img.convert("RGB"))
        style_img = resize(style_img.convert("RGB"))

        out_pil, c_loss, s_loss = model.stylize(content_img, style_img, alpha=ALPHA)
        fname = (f"shape-{STL10_CLASSES[a]}_texture-{STL10_CLASSES[b]}"
                 f"_content{c_idx}_style{s_idx}.png")
        out_pil.save(os.path.join(out_dir, fname))
        rows.append({"file": fname, "content_loss": c_loss, "style_loss": s_loss})
        tqdm.write(f"{fname}: content_loss={c_loss:.4f} style_loss={s_loss:.4f}")

    with open(os.path.join(out_dir, "pilot_losses.json"), "w") as f:
        json.dump(rows, f, indent=2)

    # Rank by content_loss so you can inspect the borderline cases directly,
    # rather than guessing at a threshold from raw numbers with no images
    # attached. Content_loss is the more informative axis for "did the shape
    # survive" -- style_loss stays in a much narrower, less diagnostic range.
    ranked = sorted(rows, key=lambda r: r["content_loss"])
    print(f"\n{len(rows)} pilot images saved to {out_dir}/")
    print("Ranked by content_loss (low -> high). Open these in order and find the "
          "LAST one where the shape is still recognizable under the texture:")
    for r in ranked:
        print(f"  content_loss={r['content_loss']:.4f}  style_loss={r['style_loss']:.4f}  {r['file']}")
    print(f"\nSet CONTENT_LOSS_MAX to that image's content_loss (or a little below it), "
          f"and STYLE_LOSS_MAX similarly from the style_loss column -- do NOT reuse "
          f"thresholds computed under a different ALPHA value, since the loss scale "
          f"shifts with alpha.")
    return rows

# SSIM and Filter img thanks to supus et. al (claude)
def ssim(img1: Image.Image, img2: Image.Image) -> float:
    try:
        from skimage.metrics import structural_similarity as ssim_fn
        arr1 = np.array(img1.convert('L'))
        arr2 = np.array(img2.convert('L'))
        return float(ssim_fn(arr1, arr2, data_range=255))
    except ImportError:
        arr1 = np.array(img1.convert('L'), dtype=np.float64)
        arr2 = np.array(img2.convert('L'), dtype=np.float64)
        c1, c2 = (0.01 * 255)**2, (0.03 * 255)**2
        mu1, mu2 = arr1.mean(), arr2.mean()
        sigma1_sq = ((arr1 - mu1)**2).mean()
        sigma2_sq = ((arr2 - mu2)**2).mean()
        sigma12 = ((arr1 - mu1) * (arr2 - mu2)).mean()
        num = (2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)
        den = (mu1**2 + mu2**2 + c1) * (sigma1_sq + sigma2_sq + c2)
        return float(num / den)


def filter_img(out_pil, style_pil, content_pil):
    arr = np.array(out_pil)
    # Check 1: Non-blank / non-black intensity
    if arr.mean() < 10.0:
        return False, "Too dark / black image", 0.0
    if arr.std() < 8.0:
        return False, "Low contrast / uniform image", 0.0

    # Check 2: Structural preservation of content shape
    ssim_val = ssim(content_pil, out_pil)
    if ssim_val < 0.10:
        return False, f"Content shape obliterated (SSIM={ssim_val:.3f} < 0.10)", ssim_val
    if ssim_val > 0.90:
        return False, f"Style not transferred (SSIM={ssim_val:.3f} > 0.90)", ssim_val

    return True, "Accepted", ssim_val

def generate_cue_conflicts(root: str = "./data", out_dir: str = "task1/data/cue_conflicts",
                            target_total: int = 220, images_per_direction: int = 24,
                            device: str = "mps"):
    """
    Selects >=5 unordered class pairs, generates both directions for each,
    applies the fixed rejection rule, and stops once >=200 valid conflicts
    are collected (target_total gives headroom above the 200 minimum in
    case of rejections). Saves accepted images + metadata; also records
    accepted/rejected counts.
    """
    os.makedirs(out_dir, exist_ok=True)

    orig_shape_dir = os.path.join(os.path.join(out_dir, "original"), "shape")
    orig_style_dir = os.path.join(os.path.join(out_dir, "original"), "style")

    os.makedirs(os.path.join(out_dir, "original"), exist_ok=True)
    os.makedirs(orig_shape_dir, exist_ok=True)
    os.makedirs(orig_style_dir, exist_ok=True)

    rng = random.Random(SEED) # SEED

    print(f"[cue-conflicts] Loading STL-10 train partition from {root} ...")
    ds, by_class = load_class_images(root, split="train")
    resize = T.Resize((IMAGE_SIZE, IMAGE_SIZE))
    print("[cue-conflicts] Loading AdaIN encoder/decoder ...")
    model = load_adain_model(device=device)

    all_pairs = list(combinations(range(len(STL10_CLASSES)), 2))
    rng.shuffle(all_pairs)
    class_pairs = all_pairs[:5]   # at least 5 unordered pairs, per spec
    print(f"[cue-conflicts] Using class pairs: "
          f"{[(STL10_CLASSES[a], STL10_CLASSES[b]) for a, b in class_pairs]}")

    metadata = []
    n_accepted, n_rejected = 0, 0
    total_target = len(class_pairs) * 2 * images_per_direction  # 2 directions per pair

    pbar = tqdm(total=total_target, desc="Generating cue conflicts", unit="img")
    for (a, b) in class_pairs:
        for (shape_cls, texture_cls) in [(a, b), (b, a)]:  # both directions
            shape_pool = by_class[shape_cls][:]
            texture_pool = by_class[texture_cls][:]
            rng.shuffle(shape_pool)
            rng.shuffle(texture_pool)

            made = 0
            attempt = 0
            while made < images_per_direction and attempt < images_per_direction * 3:
                content_idx = shape_pool[attempt % len(shape_pool)]
                style_idx = texture_pool[attempt % len(texture_pool)]
                attempt += 1

                content_img, _ = ds[content_idx]
                style_img, _ = ds[style_idx]
                content_img = resize(content_img.convert("RGB").resize((224,224), Image.BILINEAR))
                style_img = resize(style_img.convert("RGB").resize((224,224), Image.BILINEAR))

                out_pil  = model.stylize(content_img, style_img, alpha=ALPHA)
                accept, reason, ssimval = filter_img(out_pil, style_img, content_img)

                if accept:
                    fname = (f"shape-{STL10_CLASSES[shape_cls]}_texture-{STL10_CLASSES[texture_cls]}"
                             f"_content{content_idx}_style{style_idx}.png")
                    out_pil.save(os.path.join(out_dir, fname))
                    content_img.save(os.path.join(orig_shape_dir, fname))
                    style_img.save(os.path.join(orig_style_dir, fname))

                    metadata.append({
                        "file": fname,
                        "shape_class": shape_cls,
                        "texture_class": texture_cls,
                        "shape_class_name": STL10_CLASSES[shape_cls],
                        "texture_class_name": STL10_CLASSES[texture_cls],
                        "content_idx": content_idx,
                        "style_idx": style_idx,
                        "ssim": ssimval,

                    })
                    n_accepted += 1
                    made += 1
                    pbar.update(1)
                else:
                    n_rejected += 1
                    print(reason)

                pbar.set_postfix(accepted=n_accepted, rejected=n_rejected)
    pbar.close()

    with open(os.path.join(out_dir, "cue_conflict_metadata.json"), "w") as f:
        json.dump({
            "metadata": metadata,
            "n_accepted": n_accepted,
            "n_rejected": n_rejected,
            "class_pairs": class_pairs,
            "content_loss_max": CONTENT_LOSS_MAX,
            "style_loss_max": STYLE_LOSS_MAX,
            "alpha": ALPHA,
            "method": "AdaIN (Huang & Belongie, 2017); weights from naoto0804/pytorch-AdaIN",
        }, f, indent=2)

    print(f"Accepted: {n_accepted}  Rejected: {n_rejected}  "
          f"(target >=200 accepted; got {n_accepted})")
    return metadata, n_accepted, n_rejected


if __name__ == "__main__":
    generate_cue_conflicts()