"""
model_2_transfer.py
───────────────────
Transfer learning using EfficientNet-B0 pretrained on ImageNet,
fine-tuned on mel spectrograms for 4-class frog call classification.

Classes:
  0 — Duttaphrynus_melanostictus
  1 — Euphlyctis_cyanophlyctis
  2 — Hoplobatrachus_tigerinus
  3 — Microhyla_ornata

Usage:
  pip install torchvision torchaudio seaborn
  python model_2_transfer.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torchaudio
import torchaudio.transforms as AT
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import warnings
import os
warnings.filterwarnings('ignore')

# ── CONFIG ────────────────────────────────────────────────────────────
# Update these paths to match your data location
CLEAN_DIR = r'C:\Users\HP\Downloads\Foss_hack\Data-Pool-Bio-Acoustics-\exps\cleaned\clean'
OUT_DIR = r'C:\Users\HP\Downloads\Foss_hack\Data-Pool-Bio-Acoustics-\exps\model_outputs'

# Audio parameters
SR = 22050
DURATION = 3  # seconds
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 512

# Model parameters
IMG_SIZE = 224  # EfficientNet expects 224x224
BATCH_SIZE = 16  # Reduce if you get CUDA out of memory errors
EPOCHS_PHASE1 = 10  # Train head only
EPOCHS_PHASE2 = 20  # Fine-tune entire network
LR_HEAD = 1e-3  # Learning rate for new classifier head
LR_BODY = 1e-4  # Lower lr for pretrained backbone
WEIGHT_DECAY = 1e-4

# Device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Classes
CLASSES = [
    'Duttaphrynus_melanostictus',
    'Euphlyctis_cyanophlyctis',
    'Hoplobatrachus_tigerinus',
    'Microhyla_ornata',
]
SHORT = ['D.melan', 'E.cyan', 'H.tiger', 'M.orn']

# Create output directory
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
print(f"Device: {DEVICE}")
print(f"Output directory: {OUT_DIR}")


# ── Dataset Class ─────────────────────────────────────────────────────
class FrogDataset(Dataset):
    """Dataset for frog call classification with mel spectrograms"""
    
    def __init__(self, file_label_pairs, transform=None, augment=False):
        self.pairs = file_label_pairs
        self.transform = transform
        self.augment = augment
        self.mel_tf = AT.MelSpectrogram(
            sample_rate=SR, 
            n_fft=N_FFT,
            hop_length=HOP_LENGTH, 
            n_mels=N_MELS
        )
        self.db_tf = AT.AmplitudeToDB(top_db=80)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        path, label = self.pairs[idx]
        
        try:
            # Load audio
            waveform, sr = torchaudio.load(str(path))

            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            # Resample if needed
            if sr != SR:
                waveform = AT.Resample(sr, SR)(waveform)

            # Pad or truncate to fixed duration
            target_len = SR * DURATION
            if waveform.shape[1] > target_len:
                waveform = waveform[:, :target_len]
            else:
                pad = target_len - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, pad))

            # Data augmentation (only during training)
            if self.augment:
                # Time shift
                shift = int(np.random.uniform(-0.1, 0.1) * SR)
                waveform = torch.roll(waveform, shift, dims=1)
                # Add small noise
                waveform += torch.randn_like(waveform) * 0.005
                # Small volume change
                waveform *= np.random.uniform(0.8, 1.2)

            # Extract mel spectrogram
            mel = self.mel_tf(waveform)
            mel = self.db_tf(mel)
            
            # Normalize to [0, 1]
            mel = (mel - mel.min()) / (mel.max() - mel.min() + 1e-8)
            
            # Resize to match EfficientNet input size
            mel = transforms.Resize((IMG_SIZE, IMG_SIZE))(mel)
            
            # Convert to 3-channel (EfficientNet expects 3 channels)
            mel = mel.repeat(3, 1, 1)

            # Apply normalization (ImageNet stats)
            if self.transform:
                mel = self.transform(mel)

            return mel, label
            
        except Exception as e:
            print(f"Error loading {path}: {e}")
            # Return a dummy spectrogram in case of error
            dummy = torch.zeros(3, IMG_SIZE, IMG_SIZE)
            return dummy, label


# ── Model: EfficientNet-B0 with custom head ───────────────────────────
def build_efficientnet(num_classes=4, freeze_backbone=False):
    """Build EfficientNet-B0 with custom classification head"""
    
    # Load pretrained EfficientNet-B0
    model = models.efficientnet_b0(weights='IMAGENET1K_V1')
    
    # Freeze backbone if specified
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    
    # Replace classifier head
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(0.3),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, num_classes),
    )
    
    return model


# ── Training Functions ─────────────────────────────────────────────────
def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # Print progress every 50 batches
        if (batch_idx + 1) % 50 == 0:
            print(f'  Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}')
    
    avg_loss = total_loss / total
    accuracy = 100. * correct / total
    return avg_loss, accuracy

def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    avg_loss = total_loss / total
    accuracy = 100. * correct / total
    return avg_loss, accuracy


# ── Data Loading ──────────────────────────────────────────────────────
print("\n" + "="*60)
print("SCANNING DATASET")
print("="*60)

all_pairs = []
for label, cls in enumerate(CLASSES):
    cls_path = Path(CLEAN_DIR) / cls
    if not cls_path.exists():
        print(f"Warning: {cls_path} does not exist!")
        continue
    
    files = sorted(cls_path.glob("*.wav"))
    for f in files:
        all_pairs.append((f, label))
    print(f"  {cls}: {len(files)} files")

if len(all_pairs) == 0:
    print("\nERROR: No audio files found! Please check CLEAN_DIR path.")
    print(f"Current CLEAN_DIR: {CLEAN_DIR}")
    exit(1)

# Stratified split
labels_only = [p[1] for p in all_pairs]
train_pairs, temp_pairs = train_test_split(
    all_pairs, test_size=0.2, stratify=labels_only, random_state=42
)

temp_labels = [p[1] for p in temp_pairs]
val_pairs, test_pairs = train_test_split(
    temp_pairs, test_size=0.5, stratify=temp_labels, random_state=42
)

print(f"\nSplit sizes:")
print(f"  Training: {len(train_pairs)} files")
print(f"  Validation: {len(val_pairs)} files")
print(f"  Test: {len(test_pairs)} files")

# Create datasets with normalization
norm = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

train_ds = FrogDataset(train_pairs, transform=norm, augment=True)
val_ds = FrogDataset(val_pairs, transform=norm, augment=False)
test_ds = FrogDataset(test_pairs, transform=norm, augment=False)

# Create dataloaders
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

print(f"\nDataloaders created:")
print(f"  Train batches: {len(train_dl)}")
print(f"  Val batches: {len(val_dl)}")
print(f"  Test batches: {len(test_dl)}")


# ── Training ──────────────────────────────────────────────────────────
print("\n" + "="*60)
print("TRAINING PHASE 1: Training head only (backbone frozen)")
print("="*60)

# Build model with frozen backbone
model = build_efficientnet(num_classes=4, freeze_backbone=True).to(DEVICE)
criterion = nn.CrossEntropyLoss()

# Only train the classifier head
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), 
    lr=LR_HEAD,
    weight_decay=WEIGHT_DECAY
)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS_PHASE1)

# Training history
history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
best_val_acc = 0.0

print(f"{'Epoch':>6} {'Train Loss':>11} {'Train Acc':>10} {'Val Loss':>9} {'Val Acc':>8}")
print("-" * 52)

for epoch in range(1, EPOCHS_PHASE1 + 1):
    train_loss, train_acc = train_epoch(model, train_dl, optimizer, criterion, DEVICE)
    val_loss, val_acc = validate_epoch(model, val_dl, criterion, DEVICE)
    scheduler.step()
    
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_acc)
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), f"{OUT_DIR}/transfer_best_phase1.pt")
        print(f"  ✓ Saved best model (val_acc: {val_acc:.2f}%)")
    
    print(f"{epoch:>6} {train_loss:>11.4f} {train_acc:>9.2f}% {val_loss:>9.4f} {val_acc:>7.2f}%")


print("\n" + "="*60)
print("TRAINING PHASE 2: Fine-tuning full network")
print("="*60)

# Unfreeze all layers
for param in model.parameters():
    param.requires_grad = True

# Use different learning rates for backbone and classifier
optimizer = optim.Adam([
    {'params': model.features.parameters(), 'lr': LR_BODY},
    {'params': model.classifier.parameters(), 'lr': LR_HEAD},
], weight_decay=WEIGHT_DECAY)

# FIX: Remove verbose parameter (deprecated in newer PyTorch versions)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3
)

print(f"{'Epoch':>6} {'Train Loss':>11} {'Train Acc':>10} {'Val Loss':>9} {'Val Acc':>8}")
print("-" * 52)

for epoch in range(1, EPOCHS_PHASE2 + 1):
    train_loss, train_acc = train_epoch(model, train_dl, optimizer, criterion, DEVICE)
    val_loss, val_acc = validate_epoch(model, val_dl, criterion, DEVICE)
    scheduler.step(val_loss)  # ReduceLROnPlateau expects the validation loss
    
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_acc)
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), f"{OUT_DIR}/transfer_best.pt")
        print(f"  ✓ Saved best model (val_acc: {val_acc:.2f}%)")
    
    # Print learning rate info every 5 epochs
    current_lr = optimizer.param_groups[0]['lr']
    lr_info = f", LR: {current_lr:.2e}" if epoch % 5 == 0 else ""
    print(f"{epoch:>6} {train_loss:>11.4f} {train_acc:>9.2f}% {val_loss:>9.4f} {val_acc:>7.2f}%{lr_info}")

print(f"\nBest validation accuracy: {best_val_acc:.2f}%")


# ── Test Evaluation ───────────────────────────────────────────────────
print("\n" + "="*60)
print("TEST EVALUATION")
print("="*60)

# Load best model
best_model_path = f"{OUT_DIR}/transfer_best.pt"
if os.path.exists(best_model_path):
    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
else:
    print(f"Warning: {best_model_path} not found, using current model")
    
model.eval()

all_preds = []
all_true = []
all_probs = []

with torch.no_grad():
    for xb, yb in test_dl:
        xb = xb.to(DEVICE)
        outputs = model(xb)
        probs = torch.softmax(outputs, dim=1)
        preds = outputs.argmax(1).cpu().numpy()
        
        all_preds.extend(preds)
        all_true.extend(yb.numpy())
        all_probs.extend(probs.cpu().numpy())

# Calculate metrics
test_acc = accuracy_score(all_true, all_preds)
print(f"\nTest Accuracy: {test_acc:.2%}")

print("\nClassification Report:")
print(classification_report(all_true, all_preds, target_names=SHORT))

# Save results
np.save(f"{OUT_DIR}/transfer_test_acc.npy", test_acc)
np.save(f"{OUT_DIR}/transfer_preds.npy", np.array(all_preds))
np.save(f"{OUT_DIR}/transfer_true.npy", np.array(all_true))
np.save(f"{OUT_DIR}/transfer_probs.npy", np.array(all_probs))

# Save class names
with open(f"{OUT_DIR}/classes.txt", 'w') as f:
    for cls in CLASSES:
        f.write(f"{cls}\n")


# ── Visualization ─────────────────────────────────────────────────────
print("\n" + "="*60)
print("GENERATING VISUALIZATIONS")
print("="*60)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Transfer Learning (EfficientNet-B0) - Training Results", fontsize=16, fontweight='bold')

# Loss curves
axes[0, 0].plot(history['train_loss'], label='Train', linewidth=2)
axes[0, 0].plot(history['val_loss'], label='Validation', linewidth=2)
axes[0, 0].axvline(EPOCHS_PHASE1 - 0.5, color='gray', linestyle='--', linewidth=1, 
                   label='Unfreeze Backbone')
axes[0, 0].set_title("Loss Curves", fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel("Epoch")
axes[0, 0].set_ylabel("Loss")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Accuracy curves
axes[0, 1].plot(history['train_acc'], label='Train', linewidth=2)
axes[0, 1].plot(history['val_acc'], label='Validation', linewidth=2)
axes[0, 1].axvline(EPOCHS_PHASE1 - 0.5, color='gray', linestyle='--', linewidth=1,
                   label='Unfreeze Backbone')
axes[0, 1].set_title("Accuracy Curves", fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel("Epoch")
axes[0, 1].set_ylabel("Accuracy (%)")
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Confusion matrix
cm = confusion_matrix(all_true, all_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=axes[0, 2],
            xticklabels=SHORT, yticklabels=SHORT, cbar_kws={'label': 'Count'})
axes[0, 2].set_title(f"Confusion Matrix\n(Test Acc: {test_acc:.2%})", fontsize=12, fontweight='bold')
axes[0, 2].set_xlabel("Predicted")
axes[0, 2].set_ylabel("True")

# Class distribution
class_counts = [len([p for p in all_pairs if p[1] == i]) for i in range(4)]
axes[1, 0].bar(CLASSES, class_counts, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
axes[1, 0].set_title("Class Distribution", fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel("Species")
axes[1, 0].set_ylabel("Number of Files")
axes[1, 0].tick_params(axis='x', rotation=45)

# Per-class accuracy
class_acc = []
for i in range(4):
    mask = np.array(all_true) == i
    if mask.sum() > 0:
        acc = np.mean(np.array(all_preds)[mask] == i)
        class_acc.append(acc)
    else:
        class_acc.append(0)

bars = axes[1, 1].bar(SHORT, class_acc, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
axes[1, 1].set_title("Per-Class Accuracy", fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel("Species")
axes[1, 1].set_ylabel("Accuracy")
axes[1, 1].set_ylim([0, 1.05])
axes[1, 1].axhline(y=test_acc, color='red', linestyle='--', label=f'Overall: {test_acc:.2%}')
axes[1, 1].legend()

# Add value labels on bars
for bar, acc in zip(bars, class_acc):
    height = bar.get_height()
    axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{acc:.2%}', ha='center', va='bottom')

# Sample spectrograms (optional)
axes[1, 2].axis('off')
sample_idx = np.random.randint(0, len(test_ds))
sample_mel, sample_label = test_ds[sample_idx]
sample_mel_vis = sample_mel.mean(dim=0).numpy()  # Average over channels
im = axes[1, 2].imshow(sample_mel_vis, aspect='auto', origin='lower', cmap='viridis')
axes[1, 2].set_title(f"Sample Spectrogram\nTrue: {SHORT[sample_label]}", fontsize=12, fontweight='bold')
axes[1, 2].set_xlabel("Time Frame")
axes[1, 2].set_ylabel("Mel Band")
plt.colorbar(im, ax=axes[1, 2], label='dB')

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/transfer_results.png", dpi=150, bbox_inches='tight')
plt.close()

print(f"\n✓ Plots saved to: {OUT_DIR}/transfer_results.png")
print(f"✓ Best model saved to: {OUT_DIR}/transfer_best.pt")
print(f"✓ Results saved to: {OUT_DIR}/transfer_test_acc.npy")

# Print summary
print("\n" + "="*60)
print("TRAINING COMPLETE - SUMMARY")
print("="*60)
print(f"Dataset: {len(all_pairs)} total files")
print(f"Training: {len(train_pairs)} | Validation: {len(val_pairs)} | Test: {len(test_pairs)}")
print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
print(f"Test Accuracy: {test_acc:.2%}")
print(f"Model saved at: {OUT_DIR}/transfer_best.pt")
print("="*60)