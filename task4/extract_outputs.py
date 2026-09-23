"""
Runs a frozen, trained model over a DataLoader and caches logits, dummy
logits (if any), features, and labels to disk -- so every OSR score in
task4/scores/ reads from exactly the same saved outputs (per spec: "All
four scores must use exactly the same saved logits and features").
"""

import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


@torch.no_grad()
def extract_outputs(model, dataset, device: str, batch_size: int = 256,
                     has_dummy: bool = False, desc: str = "Extracting outputs"):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    all_logits, all_dummy, all_feats, all_labels = [], [], [], []
    for x, y in tqdm(loader, desc=desc, unit="batch"):
        x = x.to(device)
        if has_dummy:
            logits, dummy_logits, feat = model.forward_from_layer3(model.forward_up_to_layer2(x))
            all_dummy.append(dummy_logits.cpu())
        else:
            logits, feat = model(x, return_features=True)
        all_logits.append(logits.cpu())
        all_feats.append(feat.cpu())
        all_labels.append(y)

    out = {
        "logits": torch.cat(all_logits, dim=0),
        "features": torch.cat(all_feats, dim=0),
        "labels": torch.cat(all_labels, dim=0),
    }
    if has_dummy:
        out["dummy_logits"] = torch.cat(all_dummy, dim=0)
    return out


def cache_outputs(model, dataset, device: str, out_path: str, has_dummy: bool = False, desc: str = None):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    outputs = extract_outputs(model, dataset, device, has_dummy=has_dummy,
                               desc=desc or f"Extracting -> {os.path.basename(out_path)}")
    torch.save(outputs, out_path)
    print(f"[task4] Cached outputs to {out_path}")
    return outputs


def load_cached_outputs(path: str):
    return torch.load(path, map_location="cpu")
