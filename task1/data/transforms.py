"""
All transforms take a 224x224 RGB PIL image and return a 224x224 RGB PIL
image. They know nothing about which backbone will consume the result --
per-model normalization is applied later, in the backbone wrapper's
preprocess() step. This guarantees every model sees pixel-identical
intervened images.
"""

import numpy as np
import torchvision.transforms as T
import matplotlib.colors as mcolors
from PIL import Image


# ---------------------------------------------------------------------------
# Color interventions
# ---------------------------------------------------------------------------

class Grayscale:
    """Removes chromatic info, preserves geometry. 3-channel output."""

    def __init__(self):
        self._t = T.Grayscale(num_output_channels=3)

    def __call__(self, pil_img: Image.Image) -> Image.Image:
        return self._t(pil_img)


class HueRotate:
    """Fixed hue rotation by `degrees`. Preserves saturation, luminance,
    and geometry -- tests sensitivity to *changed* (not removed) color."""

    def __init__(self, degrees: float = 90.0):
        self.degrees = degrees

    def __call__(self, pil_img: Image.Image) -> Image.Image:
        arr = np.asarray(pil_img.convert("RGB"), dtype=np.float32) / 255.0
        hsv = mcolors.rgb_to_hsv(arr)
        hsv[..., 0] = (hsv[..., 0] + self.degrees / 360.0) % 1.0
        rgb = mcolors.hsv_to_rgb(hsv)
        out = (rgb * 255).clip(0, 255).astype(np.uint8)
        return Image.fromarray(out, mode="RGB")


# ---------------------------------------------------------------------------
# Translation -- deterministic, all 4 directions per magnitude
# ---------------------------------------------------------------------------

class Translate:
    """Shifts content by (dx, dy) pixels via reflection pad + crop back to
    the original size."""

    def __init__(self, dx: int, dy: int):
        self.dx, self.dy = dx, dy

    def __call__(self, pil_img: Image.Image) -> Image.Image:
        if self.dx == 0 and self.dy == 0:
            return pil_img
        arr = np.asarray(pil_img.convert("RGB"))
        h, w, _ = arr.shape
        pad = max(abs(self.dx), abs(self.dy))
        padded = np.pad(arr, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
        top = pad - self.dy
        left = pad - self.dx
        cropped = padded[top:top + h, left:left + w]
        return Image.fromarray(cropped, mode="RGB")


TRANSLATION_DIRECTIONS = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}


def translate_all_directions(pil_img: Image.Image, magnitude: int) -> dict:
    """Returns {direction: transformed_image} for one magnitude across all
    four cardinal directions (required so results are averaged over
    direction, per the spec, rather than sampled randomly)."""
    return {
        name: Translate(sx * magnitude, sy * magnitude)(pil_img)
        for name, (sx, sy) in TRANSLATION_DIRECTIONS.items()
    }


# ---------------------------------------------------------------------------
# Patch shuffle -- deterministic per image, non-identity, reused across models
# ---------------------------------------------------------------------------

class PatchShuffle:
    """One fixed non-identity grid_size x grid_size permutation per image,
    derived reproducibly from `seed` + the image's index. Calling this
    twice on the same image_index always returns the identical shuffle."""

    def __init__(self, grid_size: int = 4, seed: int = 6304):
        self.grid_size = grid_size
        self.seed = seed

    def __call__(self, pil_img: Image.Image, image_index: int) -> Image.Image:
        arr = np.asarray(pil_img.convert("RGB"))
        h, w, _ = arr.shape
        g = self.grid_size
        assert h % g == 0 and w % g == 0, f"{h}x{w} not divisible by grid {g}"
        ph, pw = h // g, w // g

        patches = [arr[i * ph:(i + 1) * ph, j * pw:(j + 1) * pw] for i in range(g) for j in range(g)]
        n = g * g

        rng = np.random.RandomState(self.seed + image_index)
        perm = rng.permutation(n)
        while np.array_equal(perm, np.arange(n)):
            perm = rng.permutation(n)

        shuffled = np.zeros_like(arr)
        for new_pos in range(n):
            old_pos = perm[new_pos]
            gi, gj = divmod(new_pos, g)
            shuffled[gi * ph:(gi + 1) * ph, gj * pw:(gj + 1) * pw] = patches[old_pos]

        return Image.fromarray(shuffled, mode="RGB")
