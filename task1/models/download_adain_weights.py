"""
Downloads the pretrained VGG19 encoder ('vgg_normalised.pth') and decoder
('decoder.pth') used for AdaIN style transfer.

Source (MIT License): naoto0804/pytorch-AdaIN, GitHub Release v0.0.0
    https://github.com/naoto0804/pytorch-AdaIN/releases/tag/v0.0.0
The decoder was trained by that repository's author on MSCOCO + WikiArt,
reproducing Huang & Belongie (2017), "Arbitrary Style Transfer in Real-time
with Adaptive Instance Normalization." Cite this source in your README's
external-code attribution section.

Run once before generating cue conflicts:
    python -m task1.models.download_adain_weights
"""

import os
import urllib.request

RELEASE_BASE = "https://github.com/naoto0804/pytorch-AdaIN/releases/download/v0.0.0"
FILES = {
    "vgg_normalised.pth": f"{RELEASE_BASE}/vgg_normalised.pth",
    "decoder.pth": f"{RELEASE_BASE}/decoder.pth",
}


def download_adain_weights(out_dir: str = "task1/models/weights"):
    os.makedirs(out_dir, exist_ok=True)
    for fname, url in FILES.items():
        out_path = os.path.join(out_dir, fname)
        if os.path.exists(out_path):
            print(f"Already present: {out_path}")
            continue
        print(f"Downloading {url} -> {out_path}")
        urllib.request.urlretrieve(url, out_path)
    print("AdaIN weights ready.")


if __name__ == "__main__":
    download_adain_weights()
