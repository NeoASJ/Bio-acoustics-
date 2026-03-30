import os
import random
import shutil
import librosa
import numpy as np
import soundfile as sf
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────
SOURCE   = r'C:\Users\HP\Downloads\Foss_hack\Data-Pool-Bio-Acoustics-\exps\dataset_ready'
OUTPUT   = r'C:\Users\HP\Downloads\Foss_hack\Data-Pool-Bio-Acoustics-\exps\DATASET'
SR       = 22050
CLIP_DUR = 3.0
CLIP_LEN = int(CLIP_DUR * SR)   # 66150 samples
TARGET   = 100
SEED     = 42
SUPPORTED = {'.wav', '.mp3', '.m4a', '.mpga'}

random.seed(SEED)
np.random.seed(SEED)

CLASSES = [
    'Duttaphrynus_melanostictus',
    'Euphlyctis_cyanophlyctis',
    'Hoplobatrachus_tigerinus',
    'Microhyla_ornata',
    'Polypedates_maculatus',
]

# ── Augmentation helpers ──────────────────────────────────────────────
def aug_stretch(y):
    return librosa.effects.time_stretch(y, rate=np.random.uniform(0.85, 1.15))

def aug_pitch(y):
    return librosa.effects.pitch_shift(y, sr=SR, n_steps=np.random.uniform(-2, 2))

def aug_noise(y):
    return np.clip(y + np.random.randn(len(y)) * 0.004, -1.0, 1.0)

AUGS = [aug_stretch, aug_pitch, aug_noise]

def make_exactly(y, length):
    """Pad or trim array to exact length."""
    if len(y) < length:
        y = np.pad(y, (0, length - len(y)))
    return y[:length]

# ── Main ──────────────────────────────────────────────────────────────
print("=" * 60)
print("  BUILD DATASET — slice + balance → 3s clips")
print("=" * 60)

for cls in CLASSES:
    src_dir = Path(SOURCE) / cls
    dst_dir = Path(OUTPUT) / cls
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    dst_dir.mkdir(parents=True)

    files = [f for f in src_dir.iterdir() if f.suffix.lower() in SUPPORTED]

    # ── Step 1: slice all files into 3s clips
    clips = []
    for f in sorted(files):
        try:
            y, _ = librosa.load(str(f), sr=SR, mono=True)
            # slice into non-overlapping 3s chunks
            for start in range(0, len(y) - CLIP_LEN + 1, CLIP_LEN):
                clips.append(y[start:start + CLIP_LEN].copy())
        except Exception as e:
            print(f"  [!] {f.name}: {e}")

    print(f"\n  {cls}")
    print(f"  Files: {len(files)}  →  raw 3s clips: {len(clips)}")

    # ── Step 2: augment if below target
    aug_added = 0
    attempts  = 0
    while len(clips) < TARGET and attempts < TARGET * 10:
        attempts += 1
        try:
            src   = random.choice(clips).copy()
            aug   = random.choice(AUGS)(src)
            clips.append(make_exactly(aug, CLIP_LEN))
            aug_added += 1
        except:
            pass

    if aug_added:
        print(f"  Augmented clips added: {aug_added}")

    # ── Step 3: subsample to exactly TARGET
    selected = random.sample(clips, min(TARGET, len(clips)))
    print(f"  Final clips saved: {len(selected)}")

    # ── Step 4: save as wav — verify each is exactly 3.0s
    for i, clip in enumerate(selected):
        clip = make_exactly(clip, CLIP_LEN)   # guarantee exact length
        out  = dst_dir / f"{cls}_clip_{i:04d}.wav"
        sf.write(str(out), clip, SR)

# ── Verify ────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  VERIFICATION")
print("=" * 60)
all_ok = True
for cls in CLASSES:
    files = list((Path(OUTPUT) / cls).glob("*.wav"))
    durations = [librosa.get_duration(path=str(f)) for f in files]
    bad = [d for d in durations if abs(d - 3.0) > 0.05]
    status = "OK" if not bad else f"[!] {len(bad)} bad files"
    print(f"  {cls:<40} {len(files)} files  min={min(durations):.2f}s  max={max(durations):.2f}s  {status}")
    if bad:
        all_ok = False

print()
if all_ok:
    print("  All 500 clips are exactly 3.0s — ready for feature extraction!")
else:
    print("  Some files still have wrong duration — check above.")
print(f"  Output: {OUTPUT}")
print("=" * 60)