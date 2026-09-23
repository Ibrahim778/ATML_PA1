import torch
import torch.nn as nn


class LinearHead(nn.Module):
    def __init__(self, feature_dim: int, num_classes: int = 10):
        super().__init__()
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


def train_linear_head(train_feats, train_labels, val_feats, val_labels,
                       feature_dim, num_classes=10, device="cuda",
                       max_epochs=50, patience=5, seed=6304, batch_size=128):
    torch.manual_seed(seed)

    head = LinearHead(feature_dim, num_classes).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    train_feats, train_labels = train_feats.to(device), train_labels.to(device)
    val_feats, val_labels = val_feats.to(device), val_labels.to(device)

    best_val_acc = -1.0
    best_state = None
    epochs_without_improvement = 0
    g = torch.Generator().manual_seed(seed)

    for epoch in range(max_epochs):
        head.train()
        perm = torch.randperm(train_feats.size(0), generator=g)
        for i in range(0, len(perm), batch_size):
            idx = perm[i:i + batch_size]
            optimizer.zero_grad()
            loss = criterion(head(train_feats[idx]), train_labels[idx])
            loss.backward()
            optimizer.step()

        head.eval()
        with torch.no_grad():
            val_acc = (head(val_feats).argmax(dim=1) == val_labels).float().mean().item()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in head.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            break

    head.load_state_dict(best_state)
    return head, best_val_acc
