import librosa
from pathlib import Path
from collections import Counter

DATASET = r'C:\Users\HP\Downloads\Foss_hack\Data-Pool-Bio-Acoustics-\exps\dataset_ready'

CLASSES = [
    'Duttaphrynus_melanostictus',
    'Euphlyctis_cyanophlyctis',
    'Hoplobatrachus_tigerinus',
    'Microhyla_ornata',
    'Polypedates_maculatus',
]

for cls in CLASSES:
    folder = Path(DATASET) / cls
    files  = sorted(folder.glob("*.wav"))

    durations = []
    print(f"\n{'='*55}")
    print(f"  {cls}  ({len(files)} files)")
    print(f"{'='*55}")
    print(f"  {'Filename':<45} {'Duration':>8}")
    print(f"  {'-'*45}  {'-'*8}")

    for f in files:
        try:
            dur = librosa.get_duration(path=str(f))
            durations.append(round(dur, 2))
            flag = "  [!] NOT 3s" if abs(dur - 3.0) > 0.05 else ""
            print(f"  {f.name:<45} {dur:>7.2f}s{flag}")
        except Exception as e:
            print(f"  {f.name:<45} ERROR: {e}")

    if durations:
        counts = Counter(round(d, 1) for d in durations)
        not_3s = [d for d in durations if abs(d - 3.0) > 0.05]
        print(f"\n  Min : {min(durations):.2f}s")
        print(f"  Max : {max(durations):.2f}s")
        print(f"  Exactly 3.0s : {sum(1 for d in durations if abs(d-3.0) <= 0.05)}/{len(durations)}")
        if not_3s:
            print(f"  NOT 3s       : {len(not_3s)} files → {sorted(set(not_3s))}")
        else:
            print(f"  All files are exactly 3s — no issues")