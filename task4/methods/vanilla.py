"""
Shared training loop for Vanilla and GCSC (identical recipe; GCSC just adds
RandAugment to the transform, applied upstream in task4/data/cifar10.py).
SGD, lr=0.1, momentum=0.9, weight_decay=5e-4, cosine decay, batch 128,
100 epochs, seed 6304. Checkpoint = highest CIFAR-10 validation accuracy.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from task4.models.resnet_cifar import CIFARResNet18

SEED = 6304


def train_vanilla_or_gcsc(train_ds, val_ds, train_idx, val_idx, variant_name: str,
                           device: str = "cuda", epochs: int = 100, batch_size: int = 128,
                           lr: float = 0.1, momentum: float = 0.9, weight_decay: float = 5e-4,
                           out_dir: str = "task4/results", seed: int = SEED):
    """variant_name: "vanilla" or "gcsc" -- purely a label for logging/
    checkpoint naming; the transform difference (RandAugment) must already
    be baked into `train_ds`. lr/momentum/weight_decay default to the
    spec's fixed recipe but are overridable -- see task4/configs/*.yaml
    and task4/train.py, which read these from config files rather than
    hardcoding them here."""
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(seed)

    train_loader = DataLoader(Subset(train_ds, train_idx), batch_size=batch_size,
                               shuffle=True, num_workers=2, drop_last=True)
    val_loader = DataLoader(Subset(val_ds, val_idx), batch_size=256, shuffle=False, num_workers=2)

    model = CIFARResNet18(num_classes=10).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    best_state = None

    print(f"[task4:{variant_name}] Starting training: epochs={epochs}, batch_size={batch_size}")
    epoch_bar = tqdm(range(epochs), desc=f"[task4:{variant_name}] epochs", unit="epoch")
    for epoch in epoch_bar:
        model.train()
        total_loss, n_batches = 0.0, 0
        batch_bar = tqdm(train_loader, desc=f"epoch {epoch}", unit="batch", leave=False)
        for x, y in batch_bar:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            batch_bar.set_postfix(loss=f"{loss.item():.4f}")
        scheduler.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                preds = model(x).argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        val_acc = correct / total

        epoch_bar.set_postfix(train_loss=f"{total_loss / n_batches:.4f}",
                              val_acc=f"{val_acc:.4f}", best=f"{best_val_acc:.4f}")
        tqdm.write(f"[task4:{variant_name}] epoch {epoch}: train_loss={total_loss / n_batches:.4f} "
                   f"val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    ckpt_path = os.path.join(out_dir, f"{variant_name}_checkpoint.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"[task4:{variant_name}] Saved checkpoint to {ckpt_path} (val_acc={best_val_acc:.4f})")

    return model, best_val_acc, ckpt_path
