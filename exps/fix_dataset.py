"""
Dataset Fixer + Balancer
========================
1. Delete silent/near-silent files (black spectrograms)
2. Flag cross-contaminated files (wrong species name in wrong folder)
3. Augment classes with < N_SAMPLES to balance the dataset
4. Re-export clean balanced WAVs ready for training

Requirements:
    pip install librosa soundfile numpy scipy pydub tqdm audiomentations

Usage:
    python fix_dataset.py --input dataset_clean/wavs --output dataset_final/
"""

import os
import shutil
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from tqdm import tqdm

# audiomentations for augmentation
try:
    from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift
    HAS_AUG = True
except ImportError:
    print("[WARN] audiomentations not found. pip install audiomentations")
    print("       Augmentation will use basic numpy methods instead.\n")
    HAS_AUG = False


# ── Config ─────────────────────────────────────────────────────────────────────

SAMPLE_RATE     = 22050
DURATION        = 5
TARGET_SAMPLES  = 100       # balance every class to this many samples
SILENCE_THRESH  = 0.001     # RMS below this = silent file (delete it)
MIN_ACTIVE_RATIO = 0.1      # at least 10% of frames must be non-silent

# ── Helpers ────────────────────────────────────────────────────────────────────

def load_audio(path):
    try:
        y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True, duration=DURATION)
        target = SAMPLE_RATE * DURATION
        if len(y) < target:
            y = np.pad(y, (0, target - len(y)))
        return y[:target]
    except:
        return None


def is_silent(y):
    """True if file is basically silent (black spectrogram)."""
    rms = np.sqrt(np.mean(y ** 2))
    if rms < SILENCE_THRESH:
        return True
    # check what fraction of short frames have signal
    frame_rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    active = np.sum(frame_rms > SILENCE_THRESH) / len(frame_rms)
    return active < MIN_ACTIVE_RATIO


def is_cross_contaminated(filepath, class_name):
    """
    Flag if filename contains a DIFFERENT species name.
    e.g. 'duttaphrynus_melanostictus_925665.wav' inside Euphlyctis folder.
    """
    fname = filepath.stem.lower()
    cls   = class_name.lower()

    # known species keywords
    species = [
        "duttaphrynus", "euphlyctis", "hoplobatrachus",
        "microhyla", "polypedates", "nyctibatrachus",
        "indirana", "fejervarya", "sphaerotheca"
    ]

    for sp in species:
        if sp in fname and sp not in cls:
            return True, sp   # found a different species name in filename
    return False, None


# ── Augmentation ───────────────────────────────────────────────────────────────

if HAS_AUG:
    augment = Compose([
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
        TimeStretch(min_rate=0.85, max_rate=1.15, p=0.5),
        PitchShift(min_semitones=-2, max_semitones=2, p=0.5),
        Shift(min_shift=-0.2, max_shift=0.2, p=0.3),
    ])

    def augment_audio(y):
        return augment(samples=y.astype(np.float32), sample_rate=SAMPLE_RATE)

else:
    def augment_audio(y):
        """Basic augmentation without audiomentations."""
        choice = np.random.randint(3)
        if choice == 0:
            # add noise
            noise = np.random.randn(len(y)) * 0.005
            return (y + noise).astype(np.float32)
        elif choice == 1:
            # time shift
            shift = np.random.randint(-SAMPLE_RATE, SAMPLE_RATE)
            return np.roll(y, shift).astype(np.float32)
        else:
            # amplitude scale
            scale = np.random.uniform(0.8, 1.2)
            return (y * scale).astype(np.float32)


# ── Main pipeline ──────────────────────────────────────────────────────────────

def run(input_dir, output_dir):
    input_dir  = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    classes = sorted([d for d in input_dir.iterdir() if d.is_dir()])
    if not classes:
        print(f"No class folders found in {input_dir}")
        return

    print(f"\nFound {len(classes)} classes\n")

    report_lines = ["DATASET FIX REPORT\n" + "="*60 + "\n"]
    class_counts = {}

    for cls in classes:
        cname = cls.name
        out_cls = output_dir / cname
        out_cls.mkdir(exist_ok=True)

        wav_files = list(cls.glob("*.wav"))
        print(f"\n{'='*60}")
        print(f"  {cname}  ({len(wav_files)} files)")
        print(f"{'='*60}")

        report_lines.append(f"\n{cname}:\n")

        kept    = []
        deleted = []
        flagged = []

        for f in tqdm(wav_files, ncols=60, leave=False):
            y = load_audio(f)
            if y is None:
                deleted.append((f.name, "unreadable"))
                continue

            # check silence
            if is_silent(y):
                deleted.append((f.name, "silent"))
                continue

            # check cross-contamination
            contaminated, found_sp = is_cross_contaminated(f, cname)
            if contaminated:
                flagged.append((f.name, found_sp))
                # still keep it but warn — user decides
                # copy to a quarantine folder so user can check
                q_dir = output_dir / "_quarantine" / cname
                q_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, q_dir / f.name)
                report_lines.append(
                    f"  [CROSS-CONTAMINATION] {f.name}  "
                    f"(contains '{found_sp}' but is in {cname})\n"
                )
                continue

            # good file — copy to output
            dst = out_cls / f.name
            shutil.copy2(f, dst)
            kept.append(dst)

        # report deleted
        for fname, reason in deleted:
            report_lines.append(f"  [DELETED-{reason.upper()}] {fname}\n")

        print(f"  Kept      : {len(kept)}")
        print(f"  Silent    : {len(deleted)}")
        print(f"  Flagged   : {len(flagged)}")

        # ── Augment to reach TARGET_SAMPLES ───────────────────────
        n_needed = TARGET_SAMPLES - len(kept)

        if n_needed <= 0:
            print(f"  Augment   : not needed ({len(kept)} >= {TARGET_SAMPLES})")
            # if over 100, trim to exactly 100
            if len(kept) > TARGET_SAMPLES:
                for extra in kept[TARGET_SAMPLES:]:
                    extra.unlink()
                kept = kept[:TARGET_SAMPLES]
                print(f"  Trimmed to {TARGET_SAMPLES}")
        elif len(kept) == 0:
            print(f"  [WARN] No valid files left for {cname} — skipping augmentation")
            report_lines.append(f"  [WARN] No valid files — class may be unusable\n")
        else:
            print(f"  Augmenting to reach {TARGET_SAMPLES} (+{n_needed} synthetic)...")
            aug_count = 0
            src_cycle = kept.copy()

            for i in tqdm(range(n_needed), ncols=60, leave=False):
                src = src_cycle[i % len(src_cycle)]
                y   = load_audio(src)
                if y is None:
                    continue
                y_aug = augment_audio(y)
                aug_name = f"aug_{i:04d}_{src.stem}.wav"
                sf.write(out_cls / aug_name, y_aug, SAMPLE_RATE)
                aug_count += 1

            report_lines.append(f"  Augmented: +{aug_count} synthetic files\n")
            print(f"  Done — {len(kept) + aug_count} total files")

        final_count = len(list(out_cls.glob("*.wav")))
        class_counts[cname] = final_count

    # ── Final summary ──────────────────────────────────────────────────────────
    print(f"\n\n{'='*60}")
    print("  FINAL CLASS COUNTS")
    print(f"{'='*60}")
    for cname, count in class_counts.items():
        bar = "#" * (count // 5)
        print(f"  {cname:<40} {count:>4}  {bar}")

    quarantine = output_dir / "_quarantine"
    if quarantine.exists():
        q_files = list(quarantine.rglob("*.wav"))
        print(f"\n  Quarantine folder: {quarantine}")
        print(f"  {len(q_files)} files flagged for manual review")

    # save report
    report_path = output_dir / "fix_report.txt"
    with open(report_path, "w") as fp:
        fp.writelines(report_lines)

    print(f"\n  Report saved: {report_path}")
    print(f"  Clean data  : {output_dir}/")
    print(f"\nDone! Your dataset is balanced and clean.\n")
    print("Next step:")
    print("  python train.py --data dataset_final/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="dataset_clean/wavs",
                        help="Path to the wavs/ folder from prepare_dataset.py")
    parser.add_argument("--output", default="dataset_final",
                        help="Output folder for clean balanced dataset")
    args = parser.parse_args()
    run(args.input, args.output)