"""
PACS is not distributed via torchvision. Download it separately (e.g. from
https://domaingeneralization.github.io/#data) and point `root` at a folder
with the standard layout:

    <root>/
        photo/<class_name>/*.jpg
        art_painting/<class_name>/*.jpg
        cartoon/<class_name>/*.jpg
        sketch/<class_name>/*.jpg

Class folder names must be consistent across domains; PACS_CLASSES below
lists the seven official classes for reference/validation.
"""

import os
from torchvision.datasets import ImageFolder

PACS_DOMAINS = ["photo", "art_painting", "cartoon", "sketch"]
PACS_CLASSES = ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"]


def load_pacs_domain(root: str, domain: str, transform=None) -> ImageFolder:
    assert domain in PACS_DOMAINS, f"Unknown PACS domain: {domain}"
    domain_dir = os.path.join(root, domain)
    if not os.path.isdir(domain_dir):
        raise FileNotFoundError(
            f"{domain_dir} not found. Download PACS and set `root` to the "
            f"folder containing photo/, art_painting/, cartoon/, sketch/."
        )
    ds = ImageFolder(domain_dir, transform=transform)
    assert sorted(ds.classes) == sorted(PACS_CLASSES), (
        f"Unexpected class folders in {domain_dir}: {ds.classes}"
    )
    return ds
