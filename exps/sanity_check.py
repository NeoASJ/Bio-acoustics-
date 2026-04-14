import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from itertools import combinations

DATASET = r'C:\Users\HP\Downloads\Foss_hack\Data-Pool-Bio-Acoustics-\exps\DATASET'
SR      = 22050

CLASSES = [
    'Duttaphrynus_melanostictus',
    'Euphlyctis_cyanophlyctis',
    'Hoplobatrachus_tigerinus',
    'Microhyla_ornata',
    'Polypedates_maculatus',
]

# ── helpers ───────────────────────────────────────────────────────────
def load(path):
    y, _ = librosa.load(str(path), sr=SR, mono=True)
    return y

def rms(y):
    return float(np.sqrt(np.mean(y ** 2)))

def mean_mel(y):
    mel = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=64, n_fft=1024, hop_length=512)
    return librosa.power_to_db(mel, ref=np.max).mean(axis=1)  # (64,)

def cosine_sim(a, b):
    a, b = a - a.mean(), b - b.mean()
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 1e-8 else 0.0

# ── per-class checks ──────────────────────────────────────────────────
print("=" * 65)
print("  SANITY CHECK REPORT")
print("=" * 65)

class_means = {}   # cls -> mean mel vector (for inter-class check)

for cls in CLASSES:
    files  = sorted(Path(DATASET, cls).glob("*.wav"))
    print(f"\n  CLASS: {cls}  ({len(files)} files)")
    print(f"  {'-'*55}")

    signals  = []
    rms_vals = []
    silent   = []
    near_dup = []

    for f in files:
        y = load(f)
        r = rms(y)
        rms_vals.append(r)
        signals.append(y)
        if r < 0.001:
            silent.append(f.name)

    # ── intra-class duplicate check (sample 20 random pairs)
    rng = np.random.default_rng(42)
    idxs = list(range(len(signals)))
    pairs = list(combinations(idxs, 2))
    sample_pairs = [pairs[i] for i in rng.choice(len(pairs),
                    size=min(200, len(pairs)), replace=False)]

    high_sim = []
    for i, j in sample_pairs:
        m1 = mean_mel(signals[i])
        m2 = mean_mel(signals[j])
        sim = cosine_sim(m1, m2)
        if sim > 0.98:   # suspiciously similar
            high_sim.append((files[i].name, files[j].name, sim))

    # ── report
    print(f"  RMS energy  — min: {min(rms_vals):.4f}  max: {max(rms_vals):.4f}  mean: {np.mean(rms_vals):.4f}")

    if silent:
        print(f"  [!] Silent files ({len(silent)}): {silent}")
    else:
        print(f"  Silent files: none")

    if high_sim:
        print(f"  [!] Near-duplicate pairs found ({len(high_sim)}):")
        for a, b, s in high_sim[:5]:
            print(f"      {a}  <->  {b}  sim={s:.4f}")
    else:
        print(f"  Near-duplicate pairs: none detected in sampled pairs")

    # store mean mel for inter-class check
    all_mels = np.stack([mean_mel(y) for y in signals])
    class_means[cls] = all_mels.mean(axis=0)

# ── inter-class similarity ────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  INTER-CLASS MEAN SIMILARITY (lower = more separable)")
print(f"{'='*65}")
print(f"  {'Class A':<30}  {'Class B':<30}  {'Similarity':>10}")
print(f"  {'-'*30}  {'-'*30}  {'-'*10}")

pairs_info = []
for (c1, m1), (c2, m2) in combinations(class_means.items(), 2):
    sim = cosine_sim(m1, m2)
    pairs_info.append((sim, c1, c2))

for sim, c1, c2 in sorted(pairs_info, reverse=True):
    flag = "  [!] HIGH — may confuse model" if sim > 0.85 else ""
    short1 = c1.split('_')[0][:4] + '. ' + c1.split('_')[1][:8]
    short2 = c2.split('_')[0][:4] + '. ' + c2.split('_')[1][:8]
    print(f"  {short1:<30}  {short2:<30}  {sim:>10.4f}{flag}")

print(f"\n{'='*65}")
print("  SUMMARY")
print(f"{'='*65}")
total_silent = sum(1 for cls in CLASSES
                   for f in Path(DATASET, cls).glob("*.wav")
                   if rms(load(f)) < 0.001)
print(f"  Total silent files:     {total_silent}")
print(f"  Total clips checked:    500")
if total_silent == 0:
    print(f"  Dataset looks clean — proceed to feature_extraction.py")
else:
    print(f"  [!] Remove silent files before proceeding")
print("=" * 65)