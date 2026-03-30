import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from pathlib import Path

FEATURES = r'C:\Users\HP\Downloads\Foss_hack\features'
OUT_DIR  = r'C:\Users\HP\Downloads\Foss_hack\features'
EPOCHS, BATCH, LR, PATIENCE = 50, 32, 1e-3, 10

CLASSES = [
    'Duttaphrynus_melanostictus',
    'Euphlyctis_cyanophlyctis',
    'Hoplobatrachus_tigerinus',
    'Microhyla_ornata',
    'Polypedates_maculatus',
]

# ── Dataset ───────────────────────────────────────────────────────────
class FrogDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]

# ── CNN model ─────────────────────────────────────────────────────────
class FrogCNN(nn.Module):
    def __init__(self, n_classes=5):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
            # Block 2
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
            # Block 3
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, n_classes),
        )
    def forward(self, x): return self.classifier(self.features(x))

# ── Train / eval helpers ──────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer=None, device='cpu'):
    training = optimizer is not None
    model.train() if training else model.eval()
    total_loss, correct, n = 0, 0, 0
    with torch.set_grad_enabled(training):
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            out  = model(X_b)
            loss = criterion(out, y_b)
            if training:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item() * len(y_b)
            correct    += (out.argmax(1) == y_b).sum().item()
            n          += len(y_b)
    return total_loss / n, correct / n

# ── Main ──────────────────────────────────────────────────────────────
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    X = np.load(f"{FEATURES}/X.npy")
    y = np.load(f"{FEATURES}/y.npy")
    print(f"Loaded X{X.shape}  y{y.shape}")

    # stratified split 70/15/15
    idx = np.arange(len(y))
    idx_tr, idx_tmp = train_test_split(idx, test_size=0.30, stratify=y, random_state=42)
    idx_val, idx_te = train_test_split(idx_tmp, test_size=0.50, stratify=y[idx_tmp], random_state=42)

    ds = FrogDataset(X, y)
    tr_loader  = DataLoader(Subset(ds, idx_tr),  batch_size=BATCH, shuffle=True)
    val_loader = DataLoader(Subset(ds, idx_val), batch_size=BATCH)
    te_loader  = DataLoader(Subset(ds, idx_te),  batch_size=BATCH)

    model     = FrogCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_acc, patience_count = 0.0, 0
    print(f"\n{'Epoch':>5}  {'Tr loss':>8}  {'Tr acc':>7}  {'Val loss':>9}  {'Val acc':>8}")
    print("-" * 50)

    for epoch in range(1, EPOCHS + 1):
        tr_loss,  tr_acc  = run_epoch(model, tr_loader,  criterion, optimizer, device)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device=device)
        scheduler.step(val_loss)

        print(f"{epoch:>5}  {tr_loss:>8.4f}  {tr_acc:>7.3f}  {val_loss:>9.4f}  {val_acc:>8.3f}", end="")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"{OUT_DIR}/best_model.pth")
            patience_count = 0
            print("  ← saved")
        else:
            patience_count += 1
            print()
            if patience_count >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    # ── Test set evaluation
    model.load_state_dict(torch.load(f"{OUT_DIR}/best_model.pth"))
    te_loss, te_acc = run_epoch(model, te_loader, criterion, device=device)
    print(f"\nTest accuracy: {te_acc:.3f}  |  Test loss: {te_loss:.4f}")
    print(f"Best val acc:  {best_val_acc:.3f}")
    print(f"\nModel saved to {OUT_DIR}/best_model.pth")
    np.save(f"{OUT_DIR}/test_indices.npy", idx_te)

if __name__ == "__main__":
    main()