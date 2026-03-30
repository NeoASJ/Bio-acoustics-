"""
Frog Dataset Preparation Pipeline
===================================
1. Convert all mp3/m4a/m4p/mp4a -> wav
2. Sample up to 100 per class
3. Generate spectrograms per class
4. Detect & report outliers (files that look different from their class)

Requirements:
    pip install librosa soundfile numpy matplotlib scipy scikit-learn pydub tqdm
    Also install ffmpeg: https://ffmpeg.org/download.html  (needed by pydub)

Usage:
    python prepare_dataset.py --input audio/ --output dataset_clean/
"""

import os
import sys
import shutil
import random
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import librosa
import librosa.display
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler


# ── Config ────────────────────────────────────────────────────────────────────

SAMPLE_RATE    = 22050
DURATION       = 5        # seconds to use per file
N_MELS         = 128
HOP_LENGTH     = 512
N_SAMPLES      = 100      # max samples per class
OUTLIER_THRESH = 2.0      # z-score threshold to flag outlier
RANDOM_SEED    = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ── Audio conversion ──────────────────────────────────────────────────────────

def convert_to_wav(src_path, dst_path):
    """Convert any audio file to wav using pydub (requires ffmpeg)."""
    from pydub import AudioSegment
    ext = src_path.suffix.lower()
    try:
        if ext == ".mp3":
            audio = AudioSegment.from_mp3(src_path)
        elif ext in [".m4a", ".m4p", ".mp4a", ".aac"]:
            audio = AudioSegment.from_file(src_path, format="m4a")
        elif ext == ".ogg":
            audio = AudioSegment.from_ogg(src_path)
        elif ext == ".flac":
            audio = AudioSegment.from_file(src_path, format="flac")
        elif ext == ".wav":
            shutil.copy2(src_path, dst_path)
            return True
        else:
            audio = AudioSegment.from_file(src_path)
        audio = audio.set_frame_rate(SAMPLE_RATE).set_channels(1)
        audio.export(dst_path, format="wav")
        return True
    except Exception as e:
        print(f"    [SKIP] {src_path.name}: {e}")
        return False


# ── Feature extraction ────────────────────────────────────────────────────────

def load_audio(path):
    """Load audio, pad/trim to DURATION seconds."""
    try:
        y, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True,
                             duration=DURATION)
        target_len = SAMPLE_RATE * DURATION
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)))
        else:
            y = y[:target_len]
        return y
    except Exception as e:
        return None


def extract_features(y):
    """Extract mel spectrogram + summary feature vector."""
    mel = librosa.feature.melspectrogram(
        y=y, sr=SAMPLE_RATE, n_mels=N_MELS, hop_length=HOP_LENGTH)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # flat feature vector: mean + std of each mel band
    feat = np.concatenate([mel_db.mean(axis=1), mel_db.std(axis=1)])
    return mel_db, feat


# ── Outlier detection ─────────────────────────────────────────────────────────

def find_outliers(features, filenames, threshold=OUTLIER_THRESH):
    """
    Flag files whose feature vector is far from the class centroid.
    Returns list of (filename, z_score) for outliers, sorted worst-first.
    """
    if len(features) < 3:
        return []

    scaler = StandardScaler()
    X = scaler.fit_transform(np.array(features))
    centroid = X.mean(axis=0, keepdims=True)
    dists = cdist(X, centroid, metric="euclidean").flatten()

    mean_d = dists.mean()
    std_d  = dists.std() + 1e-9
    z_scores = (dists - mean_d) / std_d

    outliers = []
    for fname, z in zip(filenames, z_scores):
        if z > threshold:
            outliers.append((fname, round(float(z), 2)))

    return sorted(outliers, key=lambda x: -x[1])


# ── Spectrogram grid plot ─────────────────────────────────────────────────────

def plot_class_spectrograms(class_name, spectrograms, filenames,
                             outlier_names, out_path, max_show=16):
    """Save a grid of spectrograms for a class, highlighting outliers."""
    n = min(len(spectrograms), max_show)
    cols = 4
    rows = (n + cols - 1) // cols

    fig = plt.figure(figsize=(cols * 3.5, rows * 2.8 + 0.6))
    fig.suptitle(f"{class_name}  ({len(spectrograms)} samples)",
                 fontsize=13, fontweight="bold", y=1.01)

    for i in range(n):
        ax = fig.add_subplot(rows, cols, i + 1)
        librosa.display.specshow(spectrograms[i], sr=SAMPLE_RATE,
                                 hop_length=HOP_LENGTH, x_axis=None,
                                 y_axis=None, ax=ax, cmap="magma")
        fname = Path(filenames[i]).name
        is_out = fname in outlier_names
        color  = "#e74c3c" if is_out else "white"
        label  = f"[OUTLIER] {fname}" if is_out else fname
        ax.set_title(label, fontsize=6, color=color,
                     bbox=dict(fc="black", pad=1, alpha=0.5) if is_out else {})
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight",
                facecolor="#1a1a2e")
    plt.close()


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run(input_dir, output_dir):
    input_dir  = Path(input_dir)
    output_dir = Path(output_dir)
    wav_dir    = output_dir / "wavs"
    spec_dir   = output_dir / "spectrograms"
    output_dir.mkdir(parents=True, exist_ok=True)
    wav_dir.mkdir(exist_ok=True)
    spec_dir.mkdir(exist_ok=True)

    classes = sorted([d for d in input_dir.iterdir() if d.is_dir()])
    if not classes:
        print(f"No subfolders found in {input_dir}. Check your path.")
        sys.exit(1)

    print(f"\nFound {len(classes)} classes: {[c.name for c in classes]}\n")

    AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".m4p", ".mp4a", ".aac",
                  ".ogg", ".flac"}

    all_outliers = {}   # class -> list of (fname, z_score)
    summary      = []

    for cls in classes:
        cname = cls.name
        print(f"\n{'='*55}")
        print(f"  Class: {cname}")
        print(f"{'='*55}")

        # ── Step 1: find all audio files ──────────────────────────
        audio_files = [f for f in cls.iterdir()
                       if f.suffix.lower() in AUDIO_EXTS]
        print(f"  Found {len(audio_files)} audio files")

        if not audio_files:
            print("  [WARN] No audio files — skipping.")
            continue

        # ── Step 2: sample up to N_SAMPLES ────────────────────────
        if len(audio_files) > N_SAMPLES:
            audio_files = random.sample(audio_files, N_SAMPLES)
            print(f"  Sampled {N_SAMPLES} files")
        else:
            print(f"  Using all {len(audio_files)} files (less than {N_SAMPLES})")

        # ── Step 3: convert to wav ─────────────────────────────────
        cls_wav_dir = wav_dir / cname
        cls_wav_dir.mkdir(exist_ok=True)

        converted = []
        print(f"  Converting to WAV...")
        for f in tqdm(audio_files, ncols=60, leave=False):
            dst = cls_wav_dir / (f.stem + ".wav")
            if dst.exists() or convert_to_wav(f, dst):
                converted.append(dst)

        print(f"  Converted: {len(converted)} files")

        # ── Step 4: extract features & spectrograms ────────────────
        spectrograms = []
        features     = []
        valid_files  = []

        print(f"  Extracting features...")
        for wav in tqdm(converted, ncols=60, leave=False):
            y = load_audio(wav)
            if y is None:
                continue
            mel_db, feat = extract_features(y)
            spectrograms.append(mel_db)
            features.append(feat)
            valid_files.append(str(wav))

        print(f"  Valid files: {len(valid_files)}")

        # ── Step 5: detect outliers ────────────────────────────────
        outliers = find_outliers(features, valid_files)
        outlier_names = {Path(f).name for f, _ in outliers}
        all_outliers[cname] = outliers

        if outliers:
            print(f"\n  OUTLIERS ({len(outliers)} flagged):")
            for fname, z in outliers:
                print(f"    z={z:5.2f}  {Path(fname).name}")
        else:
            print(f"  No outliers detected.")

        # ── Step 6: save spectrogram grid ─────────────────────────
        spec_path = spec_dir / f"{cname}.png"
        plot_class_spectrograms(cname, spectrograms, valid_files,
                                outlier_names, spec_path)
        print(f"  Spectrogram grid saved -> {spec_path}")

        summary.append({
            "class": cname,
            "original": len(audio_files),
            "converted": len(converted),
            "valid": len(valid_files),
            "outliers": len(outliers),
        })

    # ── Final report ───────────────────────────────────────────────────────────
    print(f"\n\n{'='*55}")
    print("  FINAL SUMMARY")
    print(f"{'='*55}")
    print(f"  {'Class':<35} {'Files':>6} {'Valid':>6} {'Outliers':>9}")
    print(f"  {'-'*35} {'------':>6} {'-----':>6} {'--------':>9}")
    for s in summary:
        print(f"  {s['class']:<35} {s['converted']:>6} {s['valid']:>6} {s['outliers']:>9}")

    print(f"\n  Spectrograms saved to : {spec_dir}/")
    print(f"  Clean WAVs saved to   : {wav_dir}/")

    # Save outlier report
    report_path = output_dir / "outlier_report.txt"
    with open(report_path, "w") as fp:
        fp.write("OUTLIER REPORT\n" + "="*55 + "\n\n")
        for cname, outliers in all_outliers.items():
            fp.write(f"\n{cname}:\n")
            if outliers:
                for fname, z in outliers:
                    fp.write(f"  [z={z:.2f}]  {Path(fname).name}\n")
            else:
                fp.write("  (none)\n")
    print(f"  Outlier report saved  : {report_path}")
    print(f"\nDone!\n")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Frog dataset prep pipeline")
    parser.add_argument("--input",  default="audio",        help="Path to audio/ folder")
    parser.add_argument("--output", default="dataset_clean", help="Output folder")
    args = parser.parse_args()
    run(args.input, args.output)