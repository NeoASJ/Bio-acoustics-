"""
Spectrogram Similarity Checker
================================
For each class:
- Computes average (mean) spectrogram of the class
- Measures how different each file is from that mean
- Ranks files from most-different to least-different
- Saves a report image showing the mean + top outliers side by side

Requirements:
    pip install librosa soundfile numpy matplotlib scipy scikit-learn tqdm

Usage:
    python check_spectrograms.py --input dataset_final/ --output similarity_report/
"""

import os
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import librosa
import librosa.display
from pathlib import Path
from tqdm import tqdm
from scipy.stats import zscore
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# ── Config ─────────────────────────────────────────────────────────────────────

SAMPLE_RATE  = 22050
DURATION     = 5
N_MELS       = 128
HOP_LENGTH   = 512
TOP_OUTLIERS = 6       # how many worst files to highlight per class


# ── Audio + feature utils ──────────────────────────────────────────────────────

def load_mel(path):
    try:
        y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True, duration=DURATION)
        target = SAMPLE_RATE * DURATION
        if len(y) < target:
            y = np.pad(y, (0, target - len(y)))
        y = y[:target]
        mel = librosa.feature.melspectrogram(
            y=y, sr=SAMPLE_RATE, n_mels=N_MELS, hop_length=HOP_LENGTH)
        return librosa.power_to_db(mel, ref=np.max)
    except:
        return None


def mel_to_vec(mel_db):
    """Flatten mel into a compact feature vector."""
    return np.concatenate([mel_db.mean(axis=1), mel_db.std(axis=1)])


# ── Similarity report per class ────────────────────────────────────────────────

def analyze_class(cname, wav_files, out_dir):
    print(f"\n  Loading {len(wav_files)} files...")
    
    mels     = []
    vecs     = []
    names    = []

    for f in tqdm(wav_files, ncols=55, leave=False):
        mel = load_mel(f)
        if mel is None:
            continue
        mels.append(mel)
        vecs.append(mel_to_vec(mel))
        names.append(f.name)

    if len(mels) < 3:
        print(f"  [SKIP] Not enough valid files.")
        return []

    # ── Compute similarity scores ──────────────────────────────────────────────
    X       = StandardScaler().fit_transform(np.array(vecs))
    centroid = X.mean(axis=0)
    dists   = np.linalg.norm(X - centroid, axis=1)
    mean_d  = dists.mean()
    std_d   = dists.std() + 1e-9
    z_scores = (dists - mean_d) / std_d

    # rank: highest z = most different
    ranked  = sorted(zip(names, mels, z_scores), key=lambda x: -x[2])
    outliers = [(n, m, z) for n, m, z in ranked if z > 1.5]

    # ── Plot 1: mean spectrogram + distance distribution ──────────────────────
    mean_mel = np.mean(mels, axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor("#1a1a2e")

    ax = axes[0]
    librosa.display.specshow(mean_mel, sr=SAMPLE_RATE, hop_length=HOP_LENGTH,
                             x_axis=None, y_axis=None, ax=ax, cmap="magma")
    ax.set_title(f"Class mean spectrogram\n{cname}", color="white",
                 fontsize=11, fontweight="bold")
    ax.axis("off")

    ax = axes[1]
    ax.set_facecolor("#1a1a2e")
    colors = ["#e74c3c" if z > 1.5 else "#2ecc71" for _, _, z in ranked]
    ax.barh(range(len(ranked)), [z for _, _, z in ranked],
            color=colors, height=0.7)
    ax.axvline(x=1.5, color="orange", linestyle="--", linewidth=1.2,
               label="outlier threshold")
    ax.set_yticks(range(len(ranked)))
    ax.set_yticklabels([n[:28] for n, _, _ in ranked],
                       fontsize=5, color="white")
    ax.set_xlabel("Z-score (distance from class mean)", color="white",
                  fontsize=9)
    ax.set_title("Similarity ranking\nred = outlier", color="white",
                 fontsize=11)
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#444")
    ax.legend(fontsize=8, labelcolor="white", facecolor="#1a1a2e")

    plt.tight_layout()
    plt.savefig(out_dir / f"{cname}_similarity.png", dpi=100,
                bbox_inches="tight", facecolor="#1a1a2e")
    plt.close()

    # ── Plot 2: top outliers vs mean ───────────────────────────────────────────
    if outliers:
        n_show  = min(len(outliers), TOP_OUTLIERS)
        ncols   = n_show + 1   # +1 for the mean
        fig, axes = plt.subplots(1, ncols, figsize=(ncols * 3.2, 3.5))
        fig.patch.set_facecolor("#1a1a2e")
        fig.suptitle(f"{cname} — outliers vs class mean",
                     color="white", fontsize=12, fontweight="bold")

        if ncols == 1:
            axes = [axes]

        # first panel: mean
        librosa.display.specshow(mean_mel, sr=SAMPLE_RATE,
                                 hop_length=HOP_LENGTH,
                                 x_axis=None, y_axis=None,
                                 ax=axes[0], cmap="magma")
        axes[0].set_title("CLASS MEAN", color="#2ecc71",
                          fontsize=8, fontweight="bold")
        axes[0].axis("off")

        # outlier panels
        for i, (fname, mel, z) in enumerate(outliers[:n_show]):
            ax = axes[i + 1]
            librosa.display.specshow(mel, sr=SAMPLE_RATE,
                                     hop_length=HOP_LENGTH,
                                     x_axis=None, y_axis=None,
                                     ax=ax, cmap="magma")
            ax.set_title(f"z={z:.2f}\n{fname[:22]}",
                         color="#e74c3c", fontsize=7)
            ax.axis("off")

        plt.tight_layout()
        plt.savefig(out_dir / f"{cname}_outliers.png", dpi=100,
                    bbox_inches="tight", facecolor="#1a1a2e")
        plt.close()

    # ── Print ranked list ──────────────────────────────────────────────────────
    print(f"\n  Ranked by distance from class mean (worst first):")
    print(f"  {'File':<45} {'Z-score':>8}  Status")
    print(f"  {'-'*45} {'-'*8}  ------")
    for fname, _, z in ranked:
        status = "*** OUTLIER ***" if z > 1.5 else "ok"
        color_start = "\033[91m" if z > 1.5 else "\033[92m"
        color_end   = "\033[0m"
        print(f"  {color_start}{fname:<45} {z:>8.2f}  {status}{color_end}")

    return [(cname, fname, round(float(z), 2))
            for fname, _, z in ranked if z > 1.5]


# ── Main ───────────────────────────────────────────────────────────────────────

def run(input_dir, output_dir):
    input_dir  = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    classes = sorted([d for d in input_dir.iterdir()
                      if d.is_dir() and not d.name.startswith("_")])
    if not classes:
        print(f"No class folders in {input_dir}")
        return

    print(f"\nChecking {len(classes)} classes...\n")

    all_outliers = []

    for cls in classes:
        cname    = cls.name
        wav_files = sorted(cls.glob("*.wav"))
        print(f"{'='*60}")
        print(f"  {cname}  ({len(wav_files)} files)")
        print(f"{'='*60}")

        cls_outliers = analyze_class(cname, wav_files, output_dir)
        all_outliers.extend(cls_outliers)

    # ── Master summary ─────────────────────────────────────────────────────────
    print(f"\n\n{'='*60}")
    print("  MASTER OUTLIER SUMMARY")
    print(f"{'='*60}")

    if not all_outliers:
        print("  No outliers found — dataset looks consistent!")
    else:
        print(f"  {len(all_outliers)} files flagged across all classes:\n")
        by_class = {}
        for cname, fname, z in all_outliers:
            by_class.setdefault(cname, []).append((fname, z))

        for cname, items in by_class.items():
            print(f"  {cname}  ({len(items)} outliers):")
            for fname, z in sorted(items, key=lambda x: -x[1]):
                print(f"    z={z:5.2f}  {fname}")

    # save text report
    report = output_dir / "outlier_summary.txt"
    with open(report, "w") as fp:
        fp.write("SPECTROGRAM SIMILARITY REPORT\n" + "="*60 + "\n\n")
        if not all_outliers:
            fp.write("No outliers detected.\n")
        else:
            by_class = {}
            for cname, fname, z in all_outliers:
                by_class.setdefault(cname, []).append((fname, z))
            for cname, items in by_class.items():
                fp.write(f"\n{cname}:\n")
                for fname, z in sorted(items, key=lambda x: -x[1]):
                    fp.write(f"  z={z:.2f}  {fname}\n")

    print(f"\n  Images saved to : {output_dir}/")
    print(f"  Report saved to : {report}")
    print(f"\n  For each class you get:")
    print(f"    <class>_similarity.png  — ranked bar chart + mean spectrogram")
    print(f"    <class>_outliers.png    — outliers vs mean side by side")
    print(f"\nDone!\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="dataset_final",
                        help="Path to dataset_final/ folder")
    parser.add_argument("--output", default="similarity_report",
                        help="Where to save report images")
    args = parser.parse_args()
    run(args.input, args.output)