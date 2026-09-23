import numpy as np
from sklearn.metrics import confusion_matrix


def per_class_accuracy_change(source_only_preds, method_preds, labels, num_classes, class_names=None):
    """Returns list of {class, source_only_acc, method_acc, delta} sorted by
    delta descending, to surface the largest improvements/degradations."""
    rows = []
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() == 0:
            continue
        so_acc = float((source_only_preds[mask] == labels[mask]).float().mean())
        m_acc = float((method_preds[mask] == labels[mask]).float().mean())
        rows.append({
            "class": c,
            "class_name": class_names[c] if class_names else str(c),
            "source_only_acc": so_acc,
            "method_acc": m_acc,
            "delta": m_acc - so_acc,
        })
    return sorted(rows, key=lambda r: r["delta"], reverse=True)


def dominant_confusions(preds, labels, num_classes, class_names=None, top_k=5):
    """Top-k (true, predicted) confusion pairs excluding the diagonal."""
    cm = confusion_matrix(labels.numpy(), preds.numpy(), labels=list(range(num_classes)))
    np.fill_diagonal(cm, 0)
    flat = [(i, j, cm[i, j]) for i in range(num_classes) for j in range(num_classes) if cm[i, j] > 0]
    flat.sort(key=lambda t: t[2], reverse=True)
    out = []
    for i, j, count in flat[:top_k]:
        out.append({
            "true_class": class_names[i] if class_names else str(i),
            "predicted_class": class_names[j] if class_names else str(j),
            "count": int(count),
        })
    return out
