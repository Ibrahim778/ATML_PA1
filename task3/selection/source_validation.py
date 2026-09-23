from task2.evaluation.metrics import evaluate_domain, accuracy, macro_f1


def evaluate_source_domains(backbone, val_loaders: dict, device: str):
    """val_loaders: {domain: DataLoader}. Returns per-domain metrics plus
    mean and worst-domain macro-F1 -- checkpoint selection uses mean
    macro-F1; worst-domain is reported alongside as a diagnostic."""
    per_domain = {}
    for d, loader in val_loaders.items():
        preds, labels, _ = evaluate_domain(backbone, loader, device)
        per_domain[d] = {"accuracy": accuracy(preds, labels), "macro_f1": macro_f1(preds, labels)}

    f1s = [v["macro_f1"] for v in per_domain.values()]
    accs = [v["accuracy"] for v in per_domain.values()]
    return {
        "per_domain": per_domain,
        "mean_macro_f1": sum(f1s) / len(f1s),
        "worst_macro_f1": min(f1s),
        "mean_accuracy": sum(accs) / len(accs),
        "worst_accuracy": min(accs),
    }
