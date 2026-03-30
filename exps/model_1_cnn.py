import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchaudio
import torchaudio.transforms as T
import librosa
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ==================== CONFIGURATION ====================
class Config:
    # Audio parameters
    SAMPLE_RATE = 22050
    DURATION = 5  # seconds
    N_MELS = 128
    N_MFCC = 40
    N_FFT = 2048
    HOP_LENGTH = 512
    
    # Model parameters
    BATCH_SIZE = 16
    EPOCHS = 50
    LEARNING_RATE = 0.001
    NUM_CLASSES = 4
    
    # Data paths
    DATA_ROOT = "path/to/your/audio/files"  # Update this path
    
    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================== DATASET CLASS ====================
class AudioDataset(Dataset):
    def __init__(self, file_paths, labels, feature_type='spectrogram', transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.feature_type = feature_type
        self.transform = transform
        
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        # Load audio
        audio_path = self.file_paths[idx]
        try:
            waveform, sr = librosa.load(audio_path, sr=Config.SAMPLE_RATE, 
                                        duration=Config.DURATION)
            
            # Pad or truncate to fixed length
            target_length = Config.SAMPLE_RATE * Config.DURATION
            if len(waveform) < target_length:
                waveform = np.pad(waveform, (0, target_length - len(waveform)))
            else:
                waveform = waveform[:target_length]
            
            # Extract features
            if self.feature_type == 'spectrogram':
                # Convert to mel spectrogram
                mel_spec = librosa.feature.melspectrogram(
                    y=waveform, sr=Config.SAMPLE_RATE, 
                    n_mels=Config.N_MELS, n_fft=Config.N_FFT,
                    hop_length=Config.HOP_LENGTH
                )
                features = librosa.power_to_db(mel_spec, ref=np.max)
                features = torch.FloatTensor(features).unsqueeze(0)  # Add channel dimension
                
            elif self.feature_type == 'mfcc':
                # Extract MFCC features
                mfcc = librosa.feature.mfcc(
                    y=waveform, sr=Config.SAMPLE_RATE, 
                    n_mfcc=Config.N_MFCC, n_fft=Config.N_FFT,
                    hop_length=Config.HOP_LENGTH
                )
                features = torch.FloatTensor(mfcc).unsqueeze(0)  # Add channel dimension
                
            elif self.feature_type == 'combined':
                # Combine MFCC and spectrogram
                mel_spec = librosa.feature.melspectrogram(
                    y=waveform, sr=Config.SAMPLE_RATE, 
                    n_mels=Config.N_MELS, n_fft=Config.N_FFT,
                    hop_length=Config.HOP_LENGTH
                )
                mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
                mfcc = librosa.feature.mfcc(
                    y=waveform, sr=Config.SAMPLE_RATE, 
                    n_mfcc=Config.N_MFCC, n_fft=Config.N_FFT,
                    hop_length=Config.HOP_LENGTH
                )
                features = np.vstack([mel_spec_db, mfcc])
                features = torch.FloatTensor(features).unsqueeze(0)
            
            label = torch.LongTensor([self.labels[idx]])[0]
            
            return features, label
            
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            # Return dummy data in case of error
            dummy_shape = (1, Config.N_MELS, 87) if self.feature_type == 'spectrogram' else (1, Config.N_MFCC, 87)
            return torch.zeros(dummy_shape), torch.LongTensor([0])[0]

# ==================== CNN MODEL ====================
class AudioCNN(nn.Module):
    def __init__(self, num_classes, input_channels=1):
        super(AudioCNN, self).__init__()
        
        self.conv_layers = nn.Sequential(
            # Conv Block 1
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Conv Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Conv Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Conv Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        self.fc_layers = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

# ==================== DATA LOADING AND STRATIFICATION ====================
def load_audio_files(data_root):
    """Load audio files and their corresponding species labels"""
    species_folders = {
        'Duttaphrynus_melanostictus': 'Duttaphrynus_melanostictus',
        'Euphlyctis_cyanophlyctis': 'Euphlyctis_cyanophlyctis',
        'Hoplobatrachus_tigerinus': 'Hoplobatrachus_tigerinus',
        'Microhyla_ornata': 'Microhyla_ornata'
    }
    
    file_paths = []
    species_labels = []
    
    for species_name, folder_name in species_folders.items():
        species_path = os.path.join(data_root, folder_name)
        if os.path.exists(species_path):
            for audio_file in os.listdir(species_path):
                if audio_file.endswith(('.wav', '.mp3', '.flac')):
                    file_paths.append(os.path.join(species_path, audio_file))
                    species_labels.append(species_name)
    
    return file_paths, species_labels

def stratify_split(file_paths, labels, test_size=0.2, val_size=0.1):
    """Perform stratified split maintaining class distribution"""
    # Encode labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    
    # First split: train+val vs test
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        file_paths, encoded_labels, test_size=test_size, 
        stratify=encoded_labels, random_state=42
    )
    
    # Second split: train vs val (from train+val)
    val_ratio = val_size / (1 - test_size)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths, train_val_labels, 
        test_size=val_ratio, stratify=train_val_labels, random_state=42
    )
    
    # Decode labels back to species names
    train_labels_names = label_encoder.inverse_transform(train_labels)
    val_labels_names = label_encoder.inverse_transform(val_labels)
    test_labels_names = label_encoder.inverse_transform(test_labels)
    
    return (train_paths, train_labels_names), \
           (val_paths, val_labels_names), \
           (test_paths, test_labels_names), \
           label_encoder

# ==================== TRAINING FUNCTIONS ====================
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    return running_loss / len(dataloader), 100 * correct / total

def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return running_loss / len(dataloader), 100 * correct / total

def train_model(model, train_loader, val_loader, config):
    """Main training loop"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    best_val_acc = 0.0
    
    for epoch in range(config.EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, config.DEVICE)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, config.DEVICE)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        scheduler.step(val_loss)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_audio_classifier.pth')
            print(f"Saved best model with validation accuracy: {val_acc:.2f}%")
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{config.EPOCHS}]')
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            print('-' * 50)
    
    return train_losses, val_losses, train_accs, val_accs

# ==================== MAIN EXECUTION ====================
def main():
    print(f"Using device: {Config.DEVICE}")
    
    # Load audio files
    print("Loading audio files...")
    file_paths, species_labels = load_audio_files(Config.DATA_ROOT)
    
    print(f"\nDataset statistics:")
    species_counts = Counter(species_labels)
    for species, count in species_counts.items():
        print(f"{species}: {count} files")
    
    # Stratified split
    print("\nPerforming stratified split...")
    (train_paths, train_labels), (val_paths, val_labels), \
    (test_paths, test_labels), label_encoder = stratify_split(file_paths, species_labels)
    
    print(f"\nSplit sizes:")
    print(f"Training: {len(train_paths)} files")
    print(f"Validation: {len(val_paths)} files")
    print(f"Test: {len(test_paths)} files")
    
    # Encode labels for training
    train_labels_encoded = label_encoder.transform(train_labels)
    val_labels_encoded = label_encoder.transform(val_labels)
    test_labels_encoded = label_encoder.transform(test_labels)
    
    # Create datasets
    print("\nCreating datasets with spectrogram features...")
    train_dataset = AudioDataset(train_paths, train_labels_encoded, feature_type='spectrogram')
    val_dataset = AudioDataset(val_paths, val_labels_encoded, feature_type='spectrogram')
    test_dataset = AudioDataset(test_paths, test_labels_encoded, feature_type='spectrogram')
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    
    # Create model
    print("\nInitializing model...")
    model = AudioCNN(num_classes=Config.NUM_CLASSES).to(Config.DEVICE)
    print(model)
    
    # Train model
    print("\nStarting training...")
    train_losses, val_losses, train_accs, val_accs = train_model(model, train_loader, val_loader, Config)
    
    # Test final model
    print("\nEvaluating on test set...")
    test_loss, test_acc = validate_epoch(model, test_loader, nn.CrossEntropyLoss(), Config.DEVICE)
    print(f"Test Accuracy: {test_acc:.2f}%")
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()
    
    print("\nTraining completed!")
    print("Best model saved as 'best_audio_classifier.pth'")

if __name__ == "__main__":
    # Update the data root path before running
    Config.DATA_ROOT = r"C:\Users\HP\Downloads\Foss_hack\Data-Pool-Bio-Acoustics-\exps\cleaned\clean"  
    main()