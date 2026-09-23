# PA1 -- Beyond IID

## Setup

```bash
pip install -e .
pip install -r requirements.txt
```

Every long-running script (`run_task1.py`, `task2/train.py`, `task3/train.py`,
`task4/evaluate_osr.py`, etc.) prints stage banners and uses `tqdm` progress
bars for epochs, batches, and per-image loops, so you can tell what's
running and roughly how long is left at every point.

## Task 1: Inductive Biases and Feature Representations

Dataset: STL-10. Backbones: torchvision ResNet-50 (`IMAGENET1K_V2`), torchvision
ViT-B/16 (`IMAGENET1K_V1`), OpenCLIP ViT-B/32 (`pretrained='openai'`). All frozen;
one linear head trained per backbone on cached features (seed 6304 throughout).

Run the full pipeline:

```bash
python -m task1.scripts.run_task1
```

This will:
1. Build the stratified 80/20 train/val split and the class-balanced 500-image
   eval subset (both seed 6304, saved under `task1/data/`).
2. Cache backbone features and train linear heads.
3. Evaluate the clean baseline (+ CLIP zero-shot).
4. Run color-bias (grayscale, hue rotation), shape/texture (cue-conflict),
   translation, and patch-shuffle interventions.
5. Compute cosine representation stability and fit UMAP projections
   (`task1/results/umap/*.png`).
6. Write `task1/results/task1_summary.json`.

### Design choices to document in the report
- **Additional color transform:** fixed 90-degree hue rotation (chosen over
  palette transfer / class-swapped stats for simplicity and interpretability).
- **Cue-conflict method:** true AdaIN (Huang & Belongie, ICCV 2017), single
  feed-forward pass -- a frozen VGG19 encoder (truncated at relu4_1) extracts
  content/style features, an AdaIN layer aligns their channel-wise
  mean/variance, and a pretrained decoder reconstructs the stylized image.
  Architecture and pretrained encoder/decoder weights are from
  **naoto0804/pytorch-AdaIN** (MIT License), downloaded directly from that
  repository's GitHub Release v0.0.0 by
  `task1/models/download_adain_weights.py`. The decoder was trained by that
  repo's author on MSCOCO + WikiArt. See `task1/models/adain_net.py` for the
  full attribution note; the AdaIN math itself (`calc_mean_std`,
  `adaptive_instance_normalization`) is reimplemented directly from the
  paper's Eq. 8, not copied from the external repo.
- **Cue-conflict rejection rule:** accept only if the post-hoc content loss
  (does the decoder's reconstruction match the AdaIN target?) and style loss
  (did relu4_1 mean/std actually shift toward the style image?) fall under
  fixed thresholds. Run `make_cue_conflicts.run_pilot()` and visually inspect
  a small batch to calibrate `CONTENT_LOSS_MAX` / `STYLE_LOSS_MAX` *before*
  the full run -- the values in the file are placeholders. See
  `task1/data/make_cue_conflicts.py` for the exact thresholds used.
- **Patch-shuffle seeding:** per-image permutation seed = `6304 + image_index`.
- **Representation-analysis conditions:** grayscale, translation (32px,
  rightward direction used as the representative magnitude), and patch
  shuffle. Cue-conflict representation stability (if included) pairs each
  stylized image against its own content/shape source image, not the fixed
  eval subset.
- **Dimensionality reduction:** UMAP (`n_neighbors=15`, `min_dist=0.1`,
  `metric='cosine'`, seed 6304), fit jointly on clean + transformed features
  per backbone per condition.

### External code / assistance attribution
- LLM (Claude) assistance used to scaffold and implement the code in this
  repository, per the course's Coding Assistance policy. All code has been
  reviewed and is understood by the author.
- AdaIN encoder/decoder architecture and pretrained weights: reused from
  **naoto0804/pytorch-AdaIN** (MIT License), an unofficial PyTorch
  implementation of Huang & Belongie (ICCV 2017), "Arbitrary Style Transfer
  in Real-time with Adaptive Instance Normalization." Weights downloaded
  from that repository's GitHub Release v0.0.0:
  https://github.com/naoto0804/pytorch-AdaIN/releases/tag/v0.0.0.
  The decoder was trained by that repository's author on MSCOCO + WikiArt --
  not trained as part of this assignment. The AdaIN normalization math is
  reimplemented directly from the paper, not copied from the external repo.

## Task 2: Unsupervised Domain Adaptation

Dataset: PACS (download separately -- see `shared/pacs.py`; not distributed
via torchvision). Source domains: Photo, Art Painting, Cartoon. Target:
Sketch (unlabeled, used during adaptation). Backbone: torchvision ResNet-18
(`IMAGENET1K_V1`), BatchNorm running statistics frozen at pretrained values
throughout (see `task2/models/backbone.py`).

Run all four methods (Source-only, DAN, DANN, CDAN) and produce the
comparison table:

```bash
python -m task2.evaluate_final --root /path/to/PACS
```

Controlled design study (lambda_mmd sweep for DAN, or GRL-strength sweep
for DANN):

```bash
python -m task2.controlled_study --root /path/to/PACS --study dan
python -m task2.controlled_study --root /path/to/PACS --study dann
```

Outputs: `task2/results/task2_summary.json`, per-method checkpoints.

## Task 3: Domain Generalization

Reuses the Task 2 PACS protocol and Source-only checkpoint as the ERM
baseline (loaded, not retrained). DAN-DG aligns pairs of OBSERVED source
domains only (no Sketch access); SAM uses standard two-pass sharpness-aware
minimization. Sketch is loaded only in the final evaluation script, after
every Task 3 setting is frozen.

```bash
python -m task3.evaluate_sketch --root /path/to/PACS \
    --erm_checkpoint task2/results/source_only_checkpoint.pt
```

Controlled design study (lambda_dg sweep for DAN-DG, or rho sweep for SAM):

```bash
python -m task3.controlled_study --root /path/to/PACS --study dan_dg
python -m task3.controlled_study --root /path/to/PACS --study sam
```

Outputs: `task3/results/task3_summary.json`.

## Task 4: Open-Set Recognition

Known classes: CIFAR-10 (all 10). Unknowns: fixed CIFAR-100 near/far groups
(see `task4/data/cifar100_unknowns.py`), evaluation-only. CIFAR-appropriate
ResNet-18 (3x3 stride-1 stem, no initial maxpool), trained from random
initialization.

```bash
python -m task4.evaluate_osr --cifar10_root /path/to/cifar10 \
    --cifar100_root /path/to/cifar100
```

This trains Vanilla, GCSC (RandAugment), and PROSER (classifier + data
placeholders, fine-tuned from the Vanilla checkpoint) in sequence, caches
all logits/features/dummy-logits, computes MSP/MLS/Energy/Mahalanobis on
the frozen Vanilla model, builds the Vanilla/GCSC/PROSER comparison table
(CSA + near/far AUROC, both MLS and PROSER's placeholder-based score), and
runs failure analysis on incorrectly-accepted unknowns.

**IMPORTANT -- verify before trusting for your report:** PROSER's
classifier-placeholder and data-placeholder losses
(`task4/methods/proser.py`) are reimplemented from the assignment spec's
mechanism description, not transcribed from Zhou et al. (2021)'s exact
equations. Cross-check against the paper before treating results as a
faithful reproduction.

Outputs: `task4/results/task4_summary.json`, `task4/cache/*.pt` (cached
logits/features per model per eval set).

## Cross-task synthesis

Section 5 of the assignment (visual cues vs. distribution shift,
invariance vs. discriminability, target-domain access, recognition vs.
rejection) draws on results from all four `*_summary.json` files above --
no additional code is needed for that section, only the written analysis.
