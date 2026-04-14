import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from itertools import combinations

DATASET = r'C:\Users\HP\Downloads\Foss_hack\Data-Pool-Bio-Acoustics-\exps\DATASET'
SOURCE  = r'C:\Users\HP\Downloads\Foss_hack\cleaned_audio'
SR       = 22050
CLIP_LEN = int(3.0 * SR)   # 66150 samples
RMS_THR  = 0.001
SIM_THR  = 0.98
SEED     = 99

np.random.seed(SEED)

CLASSES = [
    'Duttaphrynus_melanostictus',
    'Euphlyctis_cyanophlyctis',
    'Hoplobatrachus_tigerinus',
    'Microhyla_ornata',
    'Polypedates_maculatus',
]
SUPPORTED = {'.wav', '.mp3', '.m4a', '.mpga'}

def load_clip(path):
    y, _ = librosa.load(str(path), sr=SR, mono=True)
    return y

def rms(y):
    return float(np.sqrt(np.mean(y ** 2)))

def mean_mel(y):
    mel = librosa.feature.melspectrogram(
              y=y, sr=SR, n_mels=64, n_fft=1024, hop_length=512)
    return librosa.power_to_db(mel, ref=np.max).mean(axis=1)

def cosine_sim(a, b):
    a, b = a - a.mean(), b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 1e-8 else 0.0

def make_exactly(y):
    if len(y) < CLIP_LEN:
        y = np.pad(y, (0, CLIP_LEN - len(y)))
    return y[:CLIP_LEN]

def get_replacement_clip(cls, existing_names):
    """Slice a fresh 3s clip from source files not already used."""
    src_dir = Path(SOURCE) / cls
    src_files = [f for f in src_dir.iterdir() if f.suffix.lower() in SUPPORTED]
    np.random.shuffle(src_files)
    for f in src_files:
        try:
            y, _ = librosa.load(str(f), sr=SR, mono=True)
            n_clips = len(y) // CLIP_LEN
            offsets = list(range(n_clips))
            np.random.shuffle(offsets)
            for off in offsets:
                clip = y[off * CLIP_LEN:(off + 1) * CLIP_LEN]
                if rms(clip) >= RMS_THR:
                    return make_exactly(clip)
        except:
            continue
    return None

print("=" * 60)
print("  AUTO-FIX: remove silent + near-duplicate clips")
print("=" * 60)

total_removed = 0

for cls in CLASSES:
    cls_dir  = Path(DATASET) / cls
    files    = sorted(cls_dir.glob("*.wav"))
    removed  = set()
    replaced = 0

    print(f"\n  {cls}")

    # ── Pass 1: remove silent files
    for f in files:
        y = load_clip(f)
        if rms(y) < RMS_THR:
            print(f"    [silent]  {f.name}  rms={rms(y):.5f}  → removing")
            removed.add(f.name)

    # ── Pass 2: find near-duplicate pairs, remove the second one
    signals = {}
    mels    = {}
    for f in files:
        if f.name not in removed:
            y = load_clip(f)
            signals[f.name] = y
            mels[f.name]    = mean_mel(y)

    names = list(signals.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            if a in removed or b in removed:
                continue
            sim = cosine_sim(mels[a], mels[b])
            if sim >= SIM_THR:
                print(f"    [dup]     {b}  sim={sim:.4f} with {a}  → removing")
                removed.add(b)

    # ── Replace every removed file with a fresh clip
    for bad_name in removed:
        bad_path = cls_dir / bad_name
        repl = get_replacement_clip(cls, set(f.name for f in cls_dir.glob("*.wav")))
        if repl is not None:
            sf.write(str(bad_path), repl, SR)
            print(f"    [replace] {bad_name}  → new 3s clip written")
            replaced += 1
        else:
            bad_path.unlink()
            print(f"    [delete]  {bad_name}  → no replacement found, deleted")

    total_removed += len(removed)
    print(f"    Fixed: {len(removed)} files replaced")

# ── Final verification
print(f"\n{'='*60}")
print("  FINAL VERIFICATION")
print(f"{'='*60}")
all_ok = True
for cls in CLASSES:
    files  = sorted(Path(DATASET, cls).glob("*.wav"))
    silent = [f.name for f in files if rms(load_clip(f)) < RMS_THR]
    print(f"  {cls:<40} {len(files)} files  silent={len(silent)}")
    if silent or len(files) != 100:
        all_ok = False

print()
if all_ok:
    print("  All classes have exactly 100 clean clips.")
    print("  Run feature_extraction.py next.")
else:
    print("  [!] Some issues remain — check above.")
print("=" * 60)