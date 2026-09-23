import torch
from sklearn.metrics import f1_score


@torch.no_grad()
def evaluate_domain(backbone, loader, device):
    """Runs the backbone over a full (non-cycling) DataLoader and returns
    predictions, labels, and features for every example."""
    backbone.eval()
    all_preds, all_labels, all_feats = [], [], []
    for x, y in loader:
        x = x.to(device)
        logits, feat = backbone(x, return_features=True)
        all_preds.append(logits.argmax(dim=1).cpu())
        all_labels.append(y)
        all_feats.append(feat.cpu())
    backbone.train()
    return (torch.cat(all_preds), torch.cat(all_labels), torch.cat(all_feats))


def accuracy(preds, labels) -> float:
    return float((preds == labels).float().mean().item())


def macro_f1(preds, labels) -> float:
    return float(f1_score(labels.numpy(), preds.numpy(), average="macro"))


def per_class_accuracy(preds, labels, num_classes: int) -> dict:
    out = {}
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() == 0:
            out[c] = None
            continue
        out[c] = float((preds[mask] == labels[mask]).float().mean().item())
    return out
