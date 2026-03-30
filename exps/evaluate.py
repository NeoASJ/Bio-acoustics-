import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from train import FrogCNN, FrogDataset

FEATURES = r'C:\Users\HP\Downloads\Foss_hack\features'
CLASSES  = [
    'D. melanostictus',
    'E. cyanophlyctis',
    'H. tigerinus',
    'M. ornata',
    'P. maculatus',
]

def evaluate():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    X       = np.load(f"{FEATURES}/X.npy")
    y       = np.load(f"{FEATURES}/y.npy")
    idx_te  = np.load(f"{FEATURES}/test_indices.npy")

    ds        = FrogDataset(X, y)
    te_loader = DataLoader(Subset(ds, idx_te), batch_size=32)

    model = FrogCNN()
    model.load_state_dict(torch.load(f"{FEATURES}/best_model.pth", map_location=device))
    model.eval().to(device)

    all_preds, all_true = [], []
    with torch.no_grad():
        for X_b, y_b in te_loader:
            preds = model(X_b.to(device)).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_true.extend(y_b.numpy())

    print("\nClassification Report:")
    print(classification_report(all_true, all_preds, target_names=CLASSES))

    # Confusion matrix
    cm = confusion_matrix(all_true, all_preds)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix — Test Set')
    plt.tight_layout()
    plt.savefig(f"{FEATURES}/confusion_matrix.png", dpi=150)
    print(f"\nConfusion matrix saved to {FEATURES}/confusion_matrix.png")

if __name__ == "__main__":
    evaluate()