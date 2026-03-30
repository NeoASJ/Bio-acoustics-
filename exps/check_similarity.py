import numpy as np
import librosa
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

DATASET = r'C:\Users\HP\Downloads\Foss_hack\Data-Pool-Bio-Acoustics-\exps\DATASET'
OUT_DIR = r'C:\Users\HP\Downloads\Foss_hack\Data-Pool-Bio-Acoustics-\exps\similarity_reports'
SR      = 22050

CLASSES = [
    'Duttaphrynus_melanostictus',
    'Euphlyctis_cyanophlyctis',
    'Hoplobatrachus_tigerinus',
    'Microhyla_ornata',
    'Polypedates_maculatus',
]
SHORT = ['D.melan', 'E.cyan', 'H.tiger', 'M.orn', 'P.mac']

Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

# ── Feature: mean mel vector per file ────────────────────────────────
def get_mel_vector(path):
    y, _ = librosa.load(str(path), sr=SR, mono=True)
    mel  = librosa.feature.melspectrogram(
               y=y, sr=SR, n_mels=64, n_fft=1024, hop_length=512)
    db   = librosa.power_to_db(mel, ref=np.max)
    return db.mean(axis=1)   # shape (64,)

def cosine_sim(a, b):
    a, b = a - a.mean(), b - b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d > 1e-8 else 0.0

# ── Load all vectors ──────────────────────────────────────────────────
print("Loading mel vectors for all files...")
class_vecs  = {}   # cls -> list of (filename, vector)
class_means = {}   # cls -> mean vector

for cls in CLASSES:
    files = sorted(Path(DATASET, cls).glob("*.wav"))
    vecs  = []
    for f in files:
        try:
            v = get_mel_vector(f)
            vecs.append((f.name, v))
        except Exception as e:
            print(f"  [!] {f.name}: {e}")
    class_vecs[cls]  = vecs
    class_means[cls] = np.mean([v for _, v in vecs], axis=0)
    print(f"  {cls}: {len(vecs)} vectors loaded")

# ════════════════════════════════════════════════════════════════════
# PLOT 1 — Intra-class similarity heatmap per class
# Each file vs every other file within the same class
# ════════════════════════════════════════════════════════════════════
print("\nGenerating intra-class similarity heatmaps...")

fig, axes = plt.subplots(1, 5, figsize=(28, 6))
fig.suptitle("Intra-class similarity — each file vs every other file in same class\n"
             "(dark = similar, light = different — outliers show as light rows/cols)",
             fontsize=13, y=1.02)

outlier_report = {}   # cls -> list of (filename, mean_sim)

for ax, cls, short in zip(axes, CLASSES, SHORT):
    vecs  = class_vecs[cls]
    n     = len(vecs)
    names = [v[0].replace(cls + '_clip_', '') for v, _ in
             [(x, x) for x in vecs]]
    names = [v[0] for v, _ in [(x, x) for x in vecs]]

    # build similarity matrix
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            mat[i, j] = cosine_sim(vecs[i][1], vecs[j][1])

    im = ax.imshow(mat, vmin=-0.5, vmax=1.0, cmap='Blues', aspect='auto')
    ax.set_title(short, fontsize=11, fontweight='bold')
    ax.set_xlabel("File index")
    ax.set_ylabel("File index")
    ax.tick_params(labelsize=7)

    # find outliers: files whose mean similarity to class is low
    mean_sims = mat.mean(axis=1)
    threshold = mean_sims.mean() - 1.5 * mean_sims.std()
    outliers  = [(vecs[i][0], float(mean_sims[i]))
                 for i in range(n) if mean_sims[i] < threshold]
    outlier_report[cls] = outliers

plt.colorbar(im, ax=axes[-1], fraction=0.046, label='Cosine similarity')
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/1_intra_class_heatmaps.png", dpi=120, bbox_inches='tight')
plt.close()
print(f"  Saved: 1_intra_class_heatmaps.png")

# ════════════════════════════════════════════════════════════════════
# PLOT 2 — Per-file similarity to own class vs other classes
# For every file: bar showing sim to own class (should be high)
# and sim to each other class (should be low)
# ════════════════════════════════════════════════════════════════════
print("Generating per-file class similarity profiles...")

fig, axes = plt.subplots(5, 1, figsize=(20, 22))
fig.suptitle("Per-file similarity profile — own class (green) vs other classes (red)\n"
             "Good files: high green bar, low red bars. Suspicious: low green or high red.",
             fontsize=13)

colors_other = ['#E24B4A', '#BA7517', '#534AB7', '#1D9E75']

for ax, cls, short in zip(axes, CLASSES, SHORT):
    vecs      = class_vecs[cls]
    n         = len(vecs)
    own_mean  = class_means[cls]

    # sim of each file to its own class mean
    own_sims  = [cosine_sim(v, own_mean) for _, v in vecs]

    # sim of each file to other class means
    other_cls = [c for c in CLASSES if c != cls]
    other_sims = {c: [cosine_sim(v, class_means[c]) for _, v in vecs]
                  for c in other_cls}

    x = np.arange(n)
    ax.bar(x, own_sims, color='#2d8a4e', alpha=0.85, label='Own class', width=0.6)
    for i, (oc, col) in enumerate(zip(other_cls, colors_other)):
        ax.plot(x, other_sims[oc], color=col, alpha=0.7,
                linewidth=1.2, label=oc.split('_')[0][:6])

    # mark suspicious files (own sim < 0.5 OR any other sim > own sim)
    for i, (fname, _) in enumerate(vecs):
        own = own_sims[i]
        max_other = max(other_sims[c][i] for c in other_cls)
        if own < 0.5 or max_other > own:
            ax.axvline(x=i, color='red', alpha=0.3, linewidth=1.5)
            ax.text(i, ax.get_ylim()[1] * 0.92 if ax.get_ylim()[1] > 0 else 0.9,
                    '!', ha='center', fontsize=7, color='red', fontweight='bold')

    ax.set_title(f"{short}  ({n} files)", fontsize=11, fontweight='bold')
    ax.set_xlabel("File index (clip number)")
    ax.set_ylabel("Cosine similarity")
    ax.set_xlim(-1, n)
    ax.axhline(0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.legend(fontsize=8, loc='lower right', ncol=3)
    ax.tick_params(labelsize=8)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/2_per_file_class_profile.png", dpi=120, bbox_inches='tight')
plt.close()
print(f"  Saved: 2_per_file_class_profile.png")

# ════════════════════════════════════════════════════════════════════
# PLOT 3 — Inter-class mean similarity matrix (5x5)
# ════════════════════════════════════════════════════════════════════
print("Generating inter-class similarity matrix...")

mat = np.zeros((5, 5))
for i, c1 in enumerate(CLASSES):
    for j, c2 in enumerate(CLASSES):
        mat[i, j] = cosine_sim(class_means[c1], class_means[c2])

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(mat, vmin=-1, vmax=1, cmap='RdYlGn')
ax.set_xticks(range(5)); ax.set_xticklabels(SHORT, rotation=30, ha='right')
ax.set_yticks(range(5)); ax.set_yticklabels(SHORT)
for i in range(5):
    for j in range(5):
        ax.text(j, i, f"{mat[i,j]:.2f}", ha='center', va='center',
                fontsize=11, fontweight='bold',
                color='white' if abs(mat[i,j]) > 0.6 else 'black')
ax.set_title("Inter-class mean similarity\n(green=similar, red=different)", fontsize=12)
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/3_inter_class_matrix.png", dpi=120, bbox_inches='tight')
plt.close()
print(f"  Saved: 3_inter_class_matrix.png")

# ════════════════════════════════════════════════════════════════════
# TEXT REPORT — suspicious files per class
# ════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  SUSPICIOUS FILE REPORT")
print("=" * 65)

total_suspicious = 0
for cls in CLASSES:
    vecs     = class_vecs[cls]
    own_mean = class_means[cls]
    other_cls = [c for c in CLASSES if c != cls]

    suspicious = []
    for fname, v in vecs:
        own_sim   = cosine_sim(v, own_mean)
        other_max = max(cosine_sim(v, class_means[c]) for c in other_cls)
        other_cls_name = max(other_cls, key=lambda c: cosine_sim(v, class_means[c]))
        if own_sim < 0.5 or other_max > own_sim:
            suspicious.append((fname, own_sim, other_max, other_cls_name))

    print(f"\n  {cls}")
    if suspicious:
        print(f"  {'File':<50} {'Own sim':>8}  {'Max other':>10}  {'Closest other'}")
        print(f"  {'-'*50}  {'-'*8}  {'-'*10}  {'-'*20}")
        for fname, os_, mo, oc in sorted(suspicious, key=lambda x: x[1]):
            flag = " ← REMOVE" if os_ < 0.3 or mo > os_ + 0.2 else " ← REVIEW"
            print(f"  {fname:<50} {os_:>8.3f}  {mo:>10.3f}  {oc.split('_')[0]}{flag}")
        total_suspicious += len(suspicious)
    else:
        print(f"  No suspicious files found")

print(f"\n{'='*65}")
print(f"  Total suspicious files: {total_suspicious}")
print(f"  Reports saved to: {OUT_DIR}")
print(f"{'='*65}")