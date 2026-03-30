"""
clean_dataset.py
────────────────
Step 1: Audit and clean your bioacoustics dataset before CNN training.

What this script does:
  • Scans each class folder for bad/outlier audio files
  • REMOVES files where own_sim < 0.3 OR max_other_sim > own_sim + 0.2
  • QUARANTINES files where own_sim < 0.5 OR max_other_sim > own_sim
  • Saves a full CSV audit log so you can review every decision
  • Prints a before/after count per class

Folder layout expected:
  DATASET/
    Duttaphrynus_melanostictus/  *.wav
    Euphlyctis_cyanophlyctis/    *.wav
    ...
"""

import numpy as np
import librosa
import shutil
import csv
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ── CONFIG — edit these two lines ────────────────────────────────────
DATASET   = r'C:\Users\HP\Downloads\Foss_hack\Data-Pool-Bio-Acoustics-\exps\DATASET'
OUT_DIR   = r'C:\Users\HP\Downloads\Foss_hack\Data-Pool-Bio-Acoustics-\exps\cleaned'
# ─────────────────────────────────────────────────────────────────────

SR = 22050
CLASSES = [
    'Duttaphrynus_melanostictus',
    'Euphlyctis_cyanophlyctis',
    'Hoplobatrachus_tigerinus',
    'Microhyla_ornata',
    'Polypedates_maculatus',
]

# Thresholds — tune these if you want to be more/less aggressive
REMOVE_OWN_SIM      = 0.30   # own similarity below this → REMOVE
REMOVE_OTHER_MARGIN = 0.20   # other class beats own by this margin → REMOVE
REVIEW_OWN_SIM      = 0.50   # own similarity below this → QUARANTINE
REVIEW_OTHER_EQUAL  = 0.00   # any other class beats own → QUARANTINE


# ── Helpers ──────────────────────────────────────────────────────────
def get_mel_vector(path):
    y, _ = librosa.load(str(path), sr=SR, mono=True)
    mel  = librosa.feature.melspectrogram(
               y=y, sr=SR, n_mels=64, n_fft=1024, hop_length=512)
    db   = librosa.power_to_db(mel, ref=np.max)
    return db.mean(axis=1)

def cosine_sim(a, b):
    a, b = a - a.mean(), b - b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d > 1e-8 else 0.0


# ── Step 1: Load all vectors & class means ───────────────────────────
print("=" * 60)
print("  PHASE 1 — Loading mel vectors")
print("=" * 60)

class_vecs  = {}
class_means = {}

for cls in CLASSES:
    files = sorted(Path(DATASET, cls).glob("*.wav"))
    vecs  = []
    for f in files:
        try:
            v = get_mel_vector(f)
            vecs.append((f, v))
        except Exception as e:
            print(f"  [!] Skipping {f.name}: {e}")
    class_vecs[cls]  = vecs
    class_means[cls] = np.mean([v for _, v in vecs], axis=0)
    print(f"  {cls}: {len(vecs)} files loaded")


# ── Step 2: Decide fate of every file ────────────────────────────────
print("\n" + "=" * 60)
print("  PHASE 2 — Auditing files")
print("=" * 60)

audit_rows = []   # for CSV log
decisions  = {}   # filepath -> 'KEEP' | 'QUARANTINE' | 'REMOVE'

for cls in CLASSES:
    vecs      = class_vecs[cls]
    own_mean  = class_means[cls]
    other_cls = [c for c in CLASSES if c != cls]

    for fpath, v in vecs:
        own_sim    = cosine_sim(v, own_mean)
        other_sims = {c: cosine_sim(v, class_means[c]) for c in other_cls}
        max_other  = max(other_sims.values())
        closest    = max(other_sims, key=other_sims.get)

        # Decision logic
        if own_sim < REMOVE_OWN_SIM or max_other > own_sim + REMOVE_OTHER_MARGIN:
            decision = 'REMOVE'
        elif own_sim < REVIEW_OWN_SIM or max_other > own_sim + REVIEW_OTHER_EQUAL:
            decision = 'QUARANTINE'
        else:
            decision = 'KEEP'

        decisions[fpath] = decision
        audit_rows.append({
            'class'       : cls,
            'file'        : fpath.name,
            'own_sim'     : round(own_sim, 4),
            'max_other_sim': round(max_other, 4),
            'closest_class': closest,
            'decision'    : decision,
        })


# ── Step 3: Create output folders & act on decisions ─────────────────
print("\n" + "=" * 60)
print("  PHASE 3 — Copying / moving files")
print("=" * 60)

clean_dir = Path(OUT_DIR) / 'clean'
quar_dir  = Path(OUT_DIR) / 'quarantine'
removed_dir = Path(OUT_DIR) / 'removed'

for cls in CLASSES:
    (clean_dir / cls).mkdir(parents=True, exist_ok=True)
    (quar_dir  / cls).mkdir(parents=True, exist_ok=True)
    (removed_dir / cls).mkdir(parents=True, exist_ok=True)

counts = {cls: {'KEEP': 0, 'QUARANTINE': 0, 'REMOVE': 0} for cls in CLASSES}

for fpath, decision in decisions.items():
    cls = fpath.parent.name
    counts[cls][decision] += 1

    if decision == 'KEEP':
        shutil.copy2(fpath, clean_dir / cls / fpath.name)
    elif decision == 'QUARANTINE':
        shutil.copy2(fpath, quar_dir / cls / fpath.name)
    else:  # REMOVE
        shutil.copy2(fpath, removed_dir / cls / fpath.name)
        # Original file is NOT deleted — only copied to 'removed' folder.
        # If you're confident, uncomment the line below to delete originals:
        # fpath.unlink()


# ── Step 4: Save CSV audit log ────────────────────────────────────────
log_path = Path(OUT_DIR) / 'audit_log.csv'
with open(log_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=audit_rows[0].keys())
    writer.writeheader()
    writer.writerows(audit_rows)


# ── Step 5: Print summary ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  CLEANING SUMMARY")
print("=" * 60)
print(f"  {'Class':<35} {'KEEP':>6} {'QUARANTINE':>12} {'REMOVE':>8}")
print(f"  {'-'*35}  {'-'*6}  {'-'*10}  {'-'*8}")

total_keep = total_quar = total_rem = 0
for cls in CLASSES:
    k = counts[cls]['KEEP']
    q = counts[cls]['QUARANTINE']
    r = counts[cls]['REMOVE']
    total_keep += k; total_quar += q; total_rem += r
    short = cls.split('_')[0][:6] + '_' + cls.split('_')[1][:4]
    print(f"  {cls:<35} {k:>6} {q:>12} {r:>8}")

print(f"  {'TOTAL':<35} {total_keep:>6} {total_quar:>12} {total_rem:>8}")
print(f"\n  Audit log saved → {log_path}")
print(f"  Clean files     → {clean_dir}")
print(f"  Quarantine      → {quar_dir}")
print(f"  Removed (copy)  → {removed_dir}")
print("=" * 60)
print("\n  ✅ Done! Use the 'clean/' folder for CNN training.")
print("  📋 Check audit_log.csv to review every decision.")
print("  ⚠️  Original files are untouched. Only copies were moved.")