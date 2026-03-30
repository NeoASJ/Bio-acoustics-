"""
Outlier Remover + Re-balancer
==============================
1. Deletes confirmed outlier files from dataset_final/
2. Removes augmented copies of bad source files
3. Re-augments to bring each class back to TARGET_SAMPLES

Usage:
    python remove_outliers.py --input dataset_final/ --output dataset_ready/
"""

import os
import shutil
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
from tqdm import tqdm

try:
    from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift
    augment = Compose([
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
        TimeStretch(min_rate=0.85, max_rate=1.15, p=0.5),
        PitchShift(min_semitones=-2, max_semitones=2, p=0.5),
        Shift(min_shift=-0.2, max_shift=0.2, p=0.3),
    ])
    def augment_audio(y): return augment(samples=y.astype(np.float32), sample_rate=SR)
except:
    def augment_audio(y):
        choice = np.random.randint(3)
        if choice == 0: return (y + np.random.randn(len(y)) * 0.005).astype(np.float32)
        elif choice == 1: return np.roll(y, np.random.randint(-SR, SR)).astype(np.float32)
        else: return (y * np.random.uniform(0.8, 1.2)).astype(np.float32)

SR             = 22050
DURATION       = 5
TARGET_SAMPLES = 100

# ── Files to remove per class ──────────────────────────────────────────────────
# Source: outlier_summary.txt — only removing high z-score (>2.5) real files
# and their augmented copies. Keeping borderline ones (z 1.5-2.5) since
# some variation is natural in frog calls.

REMOVE = {
    "Duttaphrynus_melanostictus": [
        # z=3.13 — same cross-contaminated recordings found in Euphlyctis
        "11_925665.wav",
        "duttaphrynus_melanostictus_925665.wav",
        "aug_0000_11_925665.wav",
        # z=3.04
        "7_928658.wav",
        "aug_0004_7_928658.wav",
        # z=2.76
        "6_928661.wav",
        "aug_0003_6_928661.wav",
    ],
    "Euphlyctis_cyanophlyctis": [
        # z=3.05 — augmented from a bad source
        "aug_0009_Euphlyctis_cyanophlyctis_7.wav",
        # z=3.02 — bad real file
        "Euphlyctis_cyanophlyctis_3.wav",
        # z=2.91 — bad augmented
        "aug_0012_gbif_3090825973.wav",
    ],
    "Hoplobatrachus_tigerinus": [
        # z=3.87 — worst outlier in the whole dataset
        "Hoplobatrachus_tigerinus_19.wav",
        # z=2.93, 2.70 — augmented copies of bad source
        "aug_0008_gbif_5995311309.wav",
        "aug_0053_gbif_5995311309.wav",
        # z=2.68, 2.59 — bad source + augmented
        "aug_0041_XC1040393 - Indus Valley Bullfrog - Hoplobatrachus tigerinus.wav",
        "gbif_5995311309.wav",
        "XC1040393 - Indus Valley Bullfrog - Hoplobatrachus tigerinus.wav",
    ],
    "Microhyla_ornata": [
        # z=3.11 — worst, augmented from bad source Microhyla_ornata_5
        "aug_0013_Microhyla_ornata_5.wav",
        # z=2.31
        "aug_0026_6_973811.wav",
        # z=2.29
        "aug_0037_Microhyla_ornata_5.wav",
    ],
    "Polypedates_maculatus": [
        # z=3.65-3.68 — bad source + all its augmented copies
        "Polypedates_maculatus_0.wav",
        "aug_0043_Polypedates_maculatus_0.wav",
        "aug_0023_Polypedates_maculatus_0.wav",
        "aug_0003_Polypedates_maculatus_0.wav",
        "aug_0063_Polypedates_maculatus_0.wav",
        # z=2.84
        "aug_0051_Polypedates_maculatus_3.wav",
    ],
}


def load_audio(path):
    try:
        y, _ = librosa.load(path, sr=SR, mono=True, duration=DURATION)
        target = SR * DURATION
        if len(y) < target:
            y = np.pad(y, (0, target - len(y)))
        return y[:target].astype(np.float32)
    except:
        return None


def run(input_dir, output_dir):
    input_dir  = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    classes = sorted([d for d in input_dir.iterdir()
                      if d.is_dir() and not d.name.startswith("_")])

    print(f"\nProcessing {len(classes)} classes...\n")

    for cls in classes:
        cname   = cls.name
        out_cls = output_dir / cname
        out_cls.mkdir(exist_ok=True)

        to_remove = set(REMOVE.get(cname, []))
        all_wavs  = sorted(cls.glob("*.wav"))

        kept    = []
        removed = []

        print(f"{'='*55}")
        print(f"  {cname}")

        for f in all_wavs:
            if f.name in to_remove:
                removed.append(f.name)
                print(f"  [REMOVED] {f.name}")
            else:
                dst = out_cls / f.name
                shutil.copy2(f, dst)
                kept.append(dst)

        print(f"  Removed: {len(removed)}  |  Kept: {len(kept)}")

        # re-augment to reach TARGET_SAMPLES
        n_needed = TARGET_SAMPLES - len(kept)
        if n_needed > 0 and len(kept) > 0:
            print(f"  Re-augmenting +{n_needed} files...")
            # prefer real files (no aug_ prefix) as sources
            real = [f for f in kept if not f.name.startswith("aug_")]
            sources = real if real else kept

            for i in tqdm(range(n_needed), ncols=55, leave=False):
                src = sources[i % len(sources)]
                y   = load_audio(src)
                if y is None:
                    continue
                y_aug    = augment_audio(y)
                aug_name = f"aug_new_{i:04d}_{src.stem}.wav"
                sf.write(out_cls / aug_name, y_aug, SR)

        final = len(list(out_cls.glob("*.wav")))
        print(f"  Final count: {final}")

    print(f"\n{'='*55}")
    print("  DONE")
    print(f"{'='*55}")
    print(f"  Clean dataset ready at: {output_dir}/")
    print(f"\n  Next step:")
    print(f"    python train.py --data {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="dataset_final",   help="dataset_final/ folder")
    parser.add_argument("--output", default="dataset_ready",   help="output folder")
    args = parser.parse_args()
    run(args.input, args.output)