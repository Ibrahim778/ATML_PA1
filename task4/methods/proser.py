"""
PROSER (Zhou et al., 2021) fine-tuning from a Vanilla checkpoint, with 5
dummy classifiers appended.

IMPORTANT -- verify against the paper before trusting this for your report:
the classifier-placeholder and data-placeholder losses below are
reimplemented from the MECHANISM description in the assignment spec
("encourages one of the dummy classifiers to become the strongest
remaining response once the correct class is excluded"), not transcribed
from the paper's exact equations (I don't have reliable verbatim access to
Zhou et al.'s notation, and getting an index or a sign wrong would be worse
than flagging this clearly). The concrete choice made here:

  Classifier-placeholder loss (weight beta):
    L1a = CE(full_logits, y)                          -- normal classification
    L1b = CE(logits_with_y_masked_to_-inf, target)     -- target = whichever
          dummy classifier currently scores highest for this example, so
          gradient pushes THAT dummy above every other non-y known class.
    L_classifier = L1a + beta * L1b

  Data-placeholder loss (weight gamma), on manifold-mixed features between
  two DIFFERENT known classes (mixed after layer2, before layer3):
    L_data = CE(logits_mixed, target)                   -- target = whichever
             dummy classifier currently scores highest for the mixed sample.

  L_total = L_classifier (first half of batch) + gamma * L_data (second half)

Cross-check this against Zhou et al. (2021), Sec. 3, before treating the
numbers as a faithful PROSER reproduction in your report.
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from task4.methods.manifold_mixup import sample_mixup_pairs, manifold_mixup
from task4.models.resnet_cifar import CIFARResNet18

SEED = 6304
NUM_DUMMY = 5
BETA = 1.0
GAMMA = 0.1


def classifier_placeholder_loss(logits: torch.Tensor, dummy_logits: torch.Tensor, y: torch.Tensor,
                                 num_known: int, beta: float = BETA):
    full_logits = torch.cat([logits, dummy_logits], dim=1)  # (N, K+C)

    l1a = F.cross_entropy(full_logits, y)

    masked = full_logits.clone()
    masked.scatter_(1, y.unsqueeze(1), float("-inf"))
    dummy_argmax = dummy_logits.argmax(dim=1) + num_known  # index into the concatenated vector
    l1b = F.cross_entropy(masked, dummy_argmax)

    return l1a + beta * l1b, {"l1a": l1a.item(), "l1b": l1b.item()}


def data_placeholder_loss(logits_mixed: torch.Tensor, dummy_logits_mixed: torch.Tensor, num_known: int):
    full_logits = torch.cat([logits_mixed, dummy_logits_mixed], dim=1)
    dummy_argmax = dummy_logits_mixed.argmax(dim=1) + num_known
    return F.cross_entropy(full_logits, dummy_argmax)


def train_proser(train_ds, val_ds, train_idx, val_idx, vanilla_checkpoint_path: str,
                  device: str = "cuda", epochs: int = 50, batch_size: int = 128,
                  num_known: int = 10, num_dummy: int = NUM_DUMMY,
                  beta: float = BETA, gamma: float = GAMMA,
                  lr: float = 1e-3, momentum: float = 0.9, weight_decay: float = 5e-4,
                  out_dir: str = "task4/results", seed: int = SEED):
    """lr/momentum/weight_decay default to the spec's fixed recipe but are
    overridable -- see task4/configs/proser.yaml and task4/train.py."""
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(seed)

    print(f"[task4:proser] Loading Vanilla checkpoint from {vanilla_checkpoint_path} ...")
    model = CIFARResNet18(num_classes=num_known).to(device)
    model.load_state_dict(torch.load(vanilla_checkpoint_path, map_location=device))
    model.add_dummy_classifiers(num_dummy)
    model.dummy_fc = model.dummy_fc.to(device)

    train_loader = DataLoader(Subset(train_ds, train_idx), batch_size=batch_size,
                               shuffle=True, num_workers=2, drop_last=True)
    val_loader = DataLoader(Subset(val_ds, val_idx), batch_size=256, shuffle=False, num_workers=2)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = -1.0
    best_state = None

    print(f"[task4:proser] Starting fine-tuning: epochs={epochs}, "
          f"num_dummy={num_dummy}, beta={beta}, gamma={gamma}")
    epoch_bar = tqdm(range(epochs), desc="[task4:proser] epochs", unit="epoch")
    for epoch in epoch_bar:
        model.train()
        total_loss, n_batches = 0.0, 0
        batch_bar = tqdm(train_loader, desc=f"epoch {epoch}", unit="batch", leave=False)
        for x, y in batch_bar:
            x, y = x.to(device), y.to(device)
            n = x.size(0)
            half = n // 2
            x1, y1 = x[:half], y[:half]
            x2, y2 = x[half:], y[half:]

            # ---- Half 1: classifier placeholders ----
            logits1, dummy1, _ = model.forward_from_layer3(model.forward_up_to_layer2(x1))
            cls_loss, cls_logs = classifier_placeholder_loss(logits1, dummy1, y1, num_known, beta)

            # ---- Half 2: manifold-mixup data placeholders ----
            if x2.size(0) >= 2:
                h2 = model.forward_up_to_layer2(x2)
                perm = sample_mixup_pairs(y2)
                h2_mixed, lam = manifold_mixup(h2, h2[perm])
                logits2_mixed, dummy2_mixed, _ = model.forward_from_layer3(h2_mixed)
                data_loss = data_placeholder_loss(logits2_mixed, dummy2_mixed, num_known)
            else:
                data_loss = torch.tensor(0.0, device=device)

            loss = cls_loss + gamma * data_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            batch_bar.set_postfix(cls_loss=f"{cls_loss.item():.4f}",
                                  data_loss=f"{data_loss.item():.4f}")
        scheduler.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                preds = model(x).argmax(dim=1)  # known-class logits only, for CSA
                correct += (preds == y).sum().item()
                total += y.size(0)
        val_acc = correct / total

        epoch_bar.set_postfix(train_loss=f"{total_loss / n_batches:.4f}",
                              val_acc=f"{val_acc:.4f}", best=f"{best_val_acc:.4f}")
        tqdm.write(f"[task4:proser] epoch {epoch}: train_loss={total_loss / n_batches:.4f} "
                   f"val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    ckpt_path = os.path.join(out_dir, "proser_checkpoint.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"[task4:proser] Saved checkpoint to {ckpt_path} (val_acc={best_val_acc:.4f})")

    return model, best_val_acc, ckpt_path


@torch.no_grad()
def proser_placeholder_score(model, x: torch.Tensor, num_known: int) -> torch.Tensor:
    """Placeholder-based unknownness score for evaluation: combines the
    strongest dummy response with the known-class responses. Larger =
    more novel. Calibrate the rejection threshold on CIFAR-10 validation
    data only (see task4/evaluation/thresholds.py), per spec."""
    logits, dummy_logits, _ = model.forward_from_layer3(model.forward_up_to_layer2(x))
    full_logits = torch.cat([logits, dummy_logits], dim=1)
    probs = torch.softmax(full_logits, dim=1)
    known_prob_mass = probs[:, :num_known].sum(dim=1)
    return 1.0 - known_prob_mass
