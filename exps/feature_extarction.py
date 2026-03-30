import numpy as np
import librosa
from pathlib import Path

DATASET = r'C:\Users\HP\Downloads\Foss_hack\Data-Pool-Bio-Acoustics-\exps\DATASET'
OUT_DIR = r'C:\Users\HP\Downloads\Foss_hack\features'
SR       = 22050
N_MELS   = 128
N_FFT    = 2048
HOP      = 512
CLIP_DUR = 3.0
# fixed time frames for 3s @ 22050Hz with hop=512
FIXED_T  = 1 + int(np.floor(CLIP_DUR * SR / HOP))  # = 130

CLASSES = [
    'Duttaphrynus_melanostictus',
    'Euphlyctis_cyanophlyctis',
    'Hoplobatrachus_tigerinus',
    'Microhyla_ornata',
    'Polypedates_maculatus',
]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

def fix_length(arr, target_t):
    """Pad or trim the time axis (last axis) to exactly target_t frames."""
    t = arr.shape[-1]
    if t < target_t:
        pad = target_t - t
        arr = np.pad(arr, ((0, 0), (0, pad)), mode='constant')
    elif t > target_t:
        arr = arr[:, :target_t]
    return arr

def extract(path):
    y, _ = librosa.load(str(path), sr=SR, mono=True)

    # ensure audio is exactly CLIP_DUR seconds
    clip_len = int(CLIP_DUR * SR)
    if len(y) < clip_len:
        y = np.pad(y, (0, clip_len - len(y)))
    else:
        y = y[:clip_len]

    mel    = librosa.power_to_db(
                librosa.feature.melspectrogram(
                    y=y, sr=SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP),
                ref=np.max)
    delta  = librosa.feature.delta(mel)
    delta2 = librosa.feature.delta(mel, order=2)

    # guarantee fixed time dimension
    mel    = fix_length(mel,    FIXED_T)
    delta  = fix_length(delta,  FIXED_T)
    delta2 = fix_length(delta2, FIXED_T)

    feat = np.stack([mel, delta, delta2], axis=0)  # (3, 128, 130)

    # normalise each channel independently to [0, 1]
    for c in range(3):
        mn, mx = feat[c].min(), feat[c].max()
        feat[c] = (feat[c] - mn) / (mx - mn + 1e-6)

    return feat.astype(np.float32)

X, y = [], []
print("Extracting features...")
for cls in CLASSES:
    for wav in sorted(Path(DATASET, cls).glob("*.wav")):
        try:
            feat = extract(wav)
            assert feat.shape == (3, N_MELS, FIXED_T), \
                f"Shape mismatch: {feat.shape}"
            X.append(feat)
            y.append(CLASS_TO_IDX[cls])
        except Exception as e:
            print(f"  [!] {wav.name}: {e}")
    print(f"  {cls}: done")

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int64)

Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
np.save(f"{OUT_DIR}/X.npy", X)
np.save(f"{OUT_DIR}/y.npy", y)

print(f"\nX shape : {X.shape}   expected (500, 3, 128, {FIXED_T})")
print(f"y shape : {y.shape}")
print(f"Saved to {OUT_DIR}")