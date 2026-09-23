import torch
from PIL import Image
from torchvision.datasets import STL10
from tqdm import tqdm


def extract_and_cache_features(backbone, indices, root, split, out_path,
                                image_size=224, batch_size=64, device="cuda",
                                desc: str = None):
    """Extract backbone features for a fixed set of STL-10 indices and cache
    (features, labels, indices) to disk so linear-head training or
    intervention evaluation never re-touches raw images through this path."""
    ds = STL10(root=root, split=split, download=True)

    all_feats, all_labels = [], []
    batch_imgs, batch_lbls = [], []

    def flush():
        if not batch_imgs:
            return
        tensors = torch.stack([backbone.preprocess(img) for img in batch_imgs])
        feats = backbone.extract_features(tensors)
        all_feats.append(feats.cpu())
        all_labels.extend(batch_lbls)
        batch_imgs.clear()
        batch_lbls.clear()

    resize = lambda pil_img: pil_img.resize((image_size, image_size), Image.BILINEAR)

    bar_desc = desc or f"Extracting {backbone.name} features ({split})"
    for idx in tqdm(indices, desc=bar_desc, unit="img"):
        pil_img, label = ds[idx]
        pil_img = resize(pil_img.convert("RGB"))
        batch_imgs.append(pil_img)
        batch_lbls.append(label)
        if len(batch_imgs) == batch_size:
            flush()
    flush()

    feats = torch.cat(all_feats, dim=0)
    labels_t = torch.tensor(all_labels)
    torch.save({"features": feats, "labels": labels_t, "indices": list(indices)}, out_path)
    return feats, labels_t
