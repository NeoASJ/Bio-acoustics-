"""
classify_audio.py
─────────────────
Inference script for frog call classification using the trained EfficientNet model.
Supports single file, batch processing, and real-time microphone input.
"""

import torch
import torch.nn as nn
from torchvision import transforms
import torchaudio
import torchaudio.transforms as AT
import numpy as np
import librosa
import sounddevice as sd
import soundfile as sf
from pathlib import Path
import time
import argparse
import warnings
warnings.filterwarnings('ignore')

# ── CONFIGURATION ────────────────────────────────────────────────────
class Config:
    # Audio parameters (must match training)
    SAMPLE_RATE = 22050
    DURATION = 3  # seconds
    N_MELS = 128
    N_FFT = 1024
    HOP_LENGTH = 512
    IMG_SIZE = 224
    
    # Model path
    MODEL_PATH = r'C:\Users\HP\Downloads\Foss_hack\Data-Pool-Bio-Acoustics-\exps\model_outputs\transfer_best.pt'
    
    # Class names (must match training order)
    CLASSES = [
        'Duttaphrynus_melanostictus',
        'Euphlyctis_cyanophlyctis',
        'Hoplobatrachus_tigerinus',
        'Microhyla_ornata'
    ]
    SHORT_NAMES = ['D. melanostictus', 'E. cyanophlyctis', 'H. tigerinus', 'M. ornata']
    
    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── MODEL ARCHITECTURE (must match training) ─────────────────────────
def build_efficientnet(num_classes=4):
    """Build the same model architecture used in training"""
    from torchvision import models
    
    model = models.efficientnet_b0(weights=None)  # No pretrained weights needed
    
    # Replace classifier head (same as training)
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

# ── AUDIO PROCESSING ─────────────────────────────────────────────────
class AudioProcessor:
    def __init__(self):
        self.mel_tf = AT.MelSpectrogram(
            sample_rate=Config.SAMPLE_RATE,
            n_fft=Config.N_FFT,
            hop_length=Config.HOP_LENGTH,
            n_mels=Config.N_MELS
        )
        self.db_tf = AT.AmplitudeToDB(top_db=80)
        
        # ImageNet normalization (same as training)
        self.norm = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    
    def load_audio(self, audio_path):
        """Load audio file and convert to tensor"""
        try:
            waveform, sr = torchaudio.load(audio_path)
            
            # Convert to mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            # Resample if needed
            if sr != Config.SAMPLE_RATE:
                resampler = AT.Resample(sr, Config.SAMPLE_RATE)
                waveform = resampler(waveform)
            
            return waveform
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            return None
    
    def process_waveform(self, waveform):
        """Convert waveform to mel spectrogram ready for model"""
        # Pad or truncate to fixed duration
        target_len = Config.SAMPLE_RATE * Config.DURATION
        if waveform.shape[1] > target_len:
            waveform = waveform[:, :target_len]
        else:
            pad = target_len - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad))
        
        # Extract mel spectrogram
        mel = self.mel_tf(waveform)
        mel = self.db_tf(mel)
        
        # Normalize
        mel = (mel - mel.min()) / (mel.max() - mel.min() + 1e-8)
        
        # Resize and convert to 3 channels
        mel = transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE))(mel)
        mel = mel.repeat(3, 1, 1)
        
        # Apply normalization
        mel = self.norm(mel)
        
        # Add batch dimension
        return mel.unsqueeze(0)
    
    def process_audio_file(self, audio_path):
        """Process audio file and return model-ready tensor"""
        waveform = self.load_audio(audio_path)
        if waveform is None:
            return None
        return self.process_waveform(waveform)
    
    def record_audio(self, duration=3, sample_rate=22050):
        """Record audio from microphone"""
        print(f"\n🎤 Recording for {duration} seconds...")
        recording = sd.rec(int(duration * sample_rate), 
                           samplerate=sample_rate, 
                           channels=1, 
                           dtype='float32')
        sd.wait()  # Wait until recording is finished
        print("✓ Recording complete!")
        
        # Convert to tensor
        waveform = torch.from_numpy(recording).float().T
        return waveform

# ── CLASSIFIER ───────────────────────────────────────────────────────
class FrogCallClassifier:
    def __init__(self, model_path=None):
        self.device = Config.DEVICE
        self.processor = AudioProcessor()
        
        # Load model
        self.model = build_efficientnet(num_classes=len(Config.CLASSES))
        
        if model_path and Path(model_path).exists():
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"✓ Model loaded from {model_path}")
        else:
            print(f"⚠ Using default model (no weights loaded)")
        
        self.model.to(self.device)
        self.model.eval()
    
    def classify(self, audio_input, return_probs=False):
        """
        Classify audio input (file path, waveform tensor, or numpy array)
        
        Args:
            audio_input: Can be:
                - str: path to audio file
                - torch.Tensor: waveform tensor
                - numpy.ndarray: waveform array
            return_probs: If True, return all class probabilities
        
        Returns:
            tuple: (predicted_class, confidence, [probabilities])
        """
        # Handle different input types
        if isinstance(audio_input, str):
            # File path
            spectrogram = self.processor.process_audio_file(audio_input)
            if spectrogram is None:
                return None, None, None
        elif isinstance(audio_input, torch.Tensor):
            # Already a tensor, process directly
            spectrogram = self.processor.process_waveform(audio_input)
        elif isinstance(audio_input, np.ndarray):
            # Convert numpy array to tensor
            waveform = torch.from_numpy(audio_input).float()
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)  # Add channel dimension
            spectrogram = self.processor.process_waveform(waveform)
        else:
            raise ValueError(f"Unsupported input type: {type(audio_input)}")
        
        # Predict
        with torch.no_grad():
            spectrogram = spectrogram.to(self.device)
            outputs = self.model(spectrogram)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        predicted_class = Config.CLASSES[predicted.item()]
        confidence_score = confidence.item()
        all_probs = probabilities.cpu().numpy()[0]
        
        if return_probs:
            return predicted_class, confidence_score, all_probs
        return predicted_class, confidence_score
    
    def classify_batch(self, audio_files):
        """Classify multiple audio files"""
        results = []
        for file_path in audio_files:
            try:
                pred_class, confidence = self.classify(file_path)
                results.append({
                    'file': file_path,
                    'prediction': pred_class,
                    'confidence': confidence
                })
                print(f"  {Path(file_path).name}: {pred_class} ({confidence:.2%})")
            except Exception as e:
                print(f"  Error processing {file_path}: {e}")
                results.append({
                    'file': file_path,
                    'prediction': 'ERROR',
                    'confidence': 0,
                    'error': str(e)
                })
        return results

# ── VISUALIZATION ────────────────────────────────────────────────────
def display_prediction(predicted_class, confidence, all_probs=None):
    """Display prediction results nicely"""
    print("\n" + "="*50)
    print("🔊 CLASSIFICATION RESULT")
    print("="*50)
    print(f"Species: {predicted_class}")
    print(f"Confidence: {confidence:.2%}")
    
    if all_probs is not None:
        print("\nAll class probabilities:")
        for i, prob in enumerate(all_probs):
            bar_length = int(prob * 30)
            bar = "█" * bar_length + "░" * (30 - bar_length)
            print(f"  {Config.SHORT_NAMES[i]:20s} {bar} {prob:.2%}")
    print("="*50)

# ── LIVE RECORDING ───────────────────────────────────────────────────
def live_classification(classifier, duration=3):
    """Record and classify audio in real-time"""
    print("\n🎙 LIVE CLASSIFICATION MODE")
    print(f"Recording duration: {duration} seconds")
    print("Press Ctrl+C to stop\n")
    
    while True:
        try:
            # Record audio
            waveform = classifier.processor.record_audio(duration, Config.SAMPLE_RATE)
            
            # Classify
            predicted_class, confidence, probs = classifier.classify(waveform, return_probs=True)
            
            # Display result
            print(f"\n🎯 Prediction: {predicted_class}")
            print(f"   Confidence: {confidence:.2%}")
            
            # Simple progress bar for probabilities
            max_width = 40
            for i, prob in enumerate(probs):
                width = int(prob * max_width)
                bar = "█" * width + "░" * (max_width - width)
                print(f"   {Config.SHORT_NAMES[i]:20s} {bar} {prob:.2%}")
            
            print("-" * 50)
            
        except KeyboardInterrupt:
            print("\n\n✓ Stopped live classification")
            break
        except Exception as e:
            print(f"Error: {e}")
            break

# ── MAIN FUNCTION ────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Frog Call Classification')
    parser.add_argument('--file', '-f', type=str, help='Audio file to classify')
    parser.add_argument('--folder', '-d', type=str, help='Folder with audio files to classify')
    parser.add_argument('--live', '-l', action='store_true', help='Use microphone for live classification')
    parser.add_argument('--duration', '-t', type=int, default=3, help='Recording duration in seconds (default: 3)')
    parser.add_argument('--model', '-m', type=str, default=Config.MODEL_PATH, help='Path to model file')
    
    args = parser.parse_args()
    
    # Update model path if provided
    if args.model:
        Config.MODEL_PATH = args.model
    
    # Initialize classifier
    print(f"Device: {Config.DEVICE}")
    classifier = FrogCallClassifier(Config.MODEL_PATH)
    
    # Handle different modes
    if args.live:
        # Live classification mode
        live_classification(classifier, args.duration)
        
    elif args.folder:
        # Batch classification of folder
        folder_path = Path(args.folder)
        if not folder_path.exists():
            print(f"Folder not found: {folder_path}")
            return
        
        audio_files = list(folder_path.glob("*.wav")) + list(folder_path.glob("*.mp3"))
        print(f"\n📁 Found {len(audio_files)} audio files in {folder_path}")
        
        results = classifier.classify_batch(audio_files)
        
        # Summary
        print("\n" + "="*50)
        print("BATCH CLASSIFICATION SUMMARY")
        print("="*50)
        from collections import Counter
        predictions = Counter([r['prediction'] for r in results])
        for species, count in predictions.items():
            print(f"  {species}: {count} files")
            
    elif args.file:
        # Single file classification
        if not Path(args.file).exists():
            print(f"File not found: {args.file}")
            return
        
        print(f"\n📁 Processing: {args.file}")
        predicted_class, confidence, probs = classifier.classify(args.file, return_probs=True)
        display_prediction(predicted_class, confidence, probs)
        
    else:
        # Interactive mode
        print("\n" + "="*50)
        print("FROG CALL CLASSIFIER - INTERACTIVE MODE")
        print("="*50)
        print("\nOptions:")
        print("  1. Classify a single file")
        print("  2. Classify all files in a folder")
        print("  3. Live classification (microphone)")
        print("  4. Exit")
        
        while True:
            choice = input("\nSelect option (1-4): ").strip()
            
            if choice == '1':
                file_path = input("Enter audio file path: ").strip()
                if Path(file_path).exists():
                    predicted_class, confidence, probs = classifier.classify(file_path, return_probs=True)
                    display_prediction(predicted_class, confidence, probs)
                else:
                    print("File not found!")
                    
            elif choice == '2':
                folder_path = input("Enter folder path: ").strip()
                folder = Path(folder_path)
                if folder.exists():
                    audio_files = list(folder.glob("*.wav")) + list(folder.glob("*.mp3"))
                    print(f"\nFound {len(audio_files)} audio files")
                    classifier.classify_batch(audio_files)
                else:
                    print("Folder not found!")
                    
            elif choice == '3':
                live_classification(classifier, args.duration)
                
            elif choice == '4':
                print("Goodbye!")
                break
            else:
                print("Invalid choice!")

if __name__ == "__main__":
    main()

# python classify_audio.py --folder "path/to/folder/with/audio/files"
# python classify_audio.py --live --duration 3