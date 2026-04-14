"""
Class Separability Visualizer
================================
1. Mean spectrogram per class (how each class "looks")
2. Class confusion heatmap (how similar classes are to each other)
3. PCA 2D scatter (how well separated classes are in feature space)
4. Per-class spread (how tight/consistent each class is internally)

Requirements:
    pip install librosa numpy matplotlib scipy scikit-learn tqdm

Usage:
    python visualize_classes.py --input dataset_ready/ --output class_viz/
"""

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
from pathlib import Path
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist


SR         = 22050
DURATION   = 5
N_MELS     = 128
HOP_LENGTH = 512
MAX_FILES  = 50     # use up to 50 files per class for speed

COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]
BG     = "#1a1a2e"


def load_mel(path):
    try:
        y, _ = librosa.load(path, sr=SR, mono=True, duration=DURATION)
        target = SR * DURATION
        if len(y) < target:
            y = np.pad(y, (0, target - len(y)))
        y = y[:target]
        mel = librosa.feature.melspectrogram(
            y=y, sr=SR, n_mels=N_MELS, hop_length=HOP_LENGTH)
        return librosa.power_to_db(mel, ref=np.max)
    except:
        return None


def mel_to_vec(mel_db):
    return np.concatenate([mel_db.mean(axis=1), mel_db.std(axis=1)])


def load_class(cls_dir, max_files=MAX_FILES):
    wavs = sorted(cls_dir.glob("*.wav"))[:max_files]
    mels, vecs = [], []
    for f in tqdm(wavs, ncols=50, leave=False):
        m = load_mel(f)
        if m is not None:
            mels.append(m)
            vecs.append(mel_to_vec(m))
    return mels, vecs


def run(input_dir, output_dir):
    input_dir  = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    classes = sorted([d for d in input_dir.iterdir()
                      if d.is_dir() and not d.name.startswith("_")])
    n = len(classes)
    names  = [c.name.replace("_", " ") for c in classes]
    short  = [c.name.split("_")[0][:8] for c in classes]

    print(f"\nLoading {n} classes...\n")

    all_mels = []
    all_vecs = []

    for i, cls in enumerate(classes):
        print(f"  [{i+1}/{n}] {cls.name}")
        mels, vecs = load_class(cls)
        all_mels.append(mels)
        all_vecs.append(vecs)
        print(f"        loaded {len(mels)} files")

    # ── Plot 1: Mean spectrogram per class ────────────────────────────────────
    print("\nGenerating mean spectrograms...")
    fig, axes = plt.subplots(1, n, figsize=(n * 3.5, 4))
    fig.patch.set_facecolor(BG)
    fig.suptitle("Mean spectrogram per class", color="white",
                 fontsize=13, fontweight="bold", y=1.02)

    mean_vecs = []
    for i, (mels, ax) in enumerate(zip(all_mels, axes)):
        mean_mel = np.mean(mels, axis=0)
        mean_vecs.append(mel_to_vec(mean_mel))
        librosa.display.specshow(mean_mel, sr=SR, hop_length=HOP_LENGTH,
                                 x_axis=None, y_axis=None, ax=ax, cmap="magma")
        ax.set_title(names[i], color=COLORS[i], fontsize=9, fontweight="bold",
                     wrap=True)
        ax.axis("off")
        # draw colored border
        for spine in ax.spines.values():
            spine.set_edgecolor(COLORS[i])
            spine.set_linewidth(2)
            spine.set_visible(True)

    plt.tight_layout()
    out = output_dir / "1_mean_spectrograms.png"
    plt.savefig(out, dpi=120, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  Saved: {out}")

    # ── Plot 2: Inter-class similarity heatmap ────────────────────────────────
    print("Generating similarity heatmap...")
    mean_mat = np.array(mean_vecs)
    scaler   = StandardScaler()
    mean_mat = scaler.fit_transform(mean_mat)
    sim_mat  = 1 - cdist(mean_mat, mean_mat, metric="cosine")

    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    im = ax.imshow(sim_mat, cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(short, rotation=35, ha="right",
                       color="white", fontsize=9)
    ax.set_yticklabels(short, color="white", fontsize=9)

    for i in range(n):
        for j in range(n):
            val = sim_mat[i, j]
            color = "black" if val > 0.6 else "white"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=10, color=color, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    ax.set_title("Inter-class similarity\n(1.0 = identical, 0.0 = completely different)",
                 color="white", fontsize=11, pad=12)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#444")

    plt.tight_layout()
    out = output_dir / "2_class_similarity_heatmap.png"
    plt.savefig(out, dpi=120, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  Saved: {out}")

    # ── Plot 3: PCA scatter — how separated are classes ───────────────────────
    print("Generating PCA scatter plot...")

    all_X      = []
    all_labels = []
    for i, vecs in enumerate(all_vecs):
        all_X.extend(vecs)
        all_labels.extend([i] * len(vecs))

    X      = StandardScaler().fit_transform(np.array(all_X))
    pca    = PCA(n_components=2, random_state=42)
    X_2d   = pca.fit_transform(X)
    labels = np.array(all_labels)
    var    = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    for i in range(n):
        mask = labels == i
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   c=COLORS[i], label=names[i],
                   alpha=0.6, s=30, edgecolors="none")

        # draw centroid marker
        cx, cy = X_2d[mask, 0].mean(), X_2d[mask, 1].mean()
        ax.scatter(cx, cy, c=COLORS[i], s=200, marker="*",
                   edgecolors="white", linewidths=0.8, zorder=5)

    ax.set_xlabel(f"PC1 ({var[0]*100:.1f}% variance)", color="white", fontsize=10)
    ax.set_ylabel(f"PC2 ({var[1]*100:.1f}% variance)", color="white", fontsize=10)
    ax.set_title("PCA — class separation in feature space\n"
                 "(well-separated clusters = model will learn easily)",
                 color="white", fontsize=11, pad=12)
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#444")

    legend = ax.legend(fontsize=9, framealpha=0.3,
                       facecolor="#2a2a4e", labelcolor="white",
                       loc="best")
    plt.tight_layout()
    out = output_dir / "3_pca_scatter.png"
    plt.savefig(out, dpi=120, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  Saved: {out}")

    # ── Plot 4: Intra-class spread (how consistent each class is) ─────────────
    print("Generating intra-class spread chart...")

    spreads = []
    for i, vecs in enumerate(all_vecs):
        X_cls    = StandardScaler().fit_transform(np.array(vecs))
        centroid = X_cls.mean(axis=0, keepdims=True)
        dists    = np.linalg.norm(X_cls - centroid, axis=1)
        spreads.append(dists)

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    parts = ax.violinplot(spreads, positions=range(n), showmedians=True,
                          showextrema=True)

    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(COLORS[i])
        pc.set_alpha(0.7)
    parts["cmedians"].set_color("white")
    parts["cmaxes"].set_color("#888")
    parts["cmins"].set_color("#888")
    parts["cbars"].set_color("#888")

    ax.set_xticks(range(n))
    ax.set_xticklabels(names, rotation=20, ha="right",
                       color="white", fontsize=9)
    ax.set_ylabel("Distance from class centroid", color="white", fontsize=10)
    ax.set_title("Intra-class spread\n"
                 "(narrow violin = consistent class, wide = high variation)",
                 color="white", fontsize=11, pad=12)
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#444")

    # add mean labels
    for i, d in enumerate(spreads):
        ax.text(i, max(d) + 0.1, f"σ={np.std(d):.2f}",
                ha="center", color=COLORS[i], fontsize=9, fontweight="bold")

    plt.tight_layout()
    out = output_dir / "4_intra_class_spread.png"
    plt.savefig(out, dpi=120, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  Saved: {out}")

    # ── Terminal summary ───────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("  CLASS SEPARABILITY SUMMARY")
    print(f"{'='*55}")
    print(f"\n  Inter-class similarity (lower = more distinct = better):")
    for i in range(n):
        for j in range(i+1, n):
            sim = sim_mat[i, j]
            flag = "*** TOO SIMILAR ***" if sim > 0.85 else \
                   "  similar" if sim > 0.70 else "  distinct"
            print(f"  {short[i]:<10} vs {short[j]:<10}  {sim:.2f}  {flag}")

    print(f"\n  Intra-class consistency (lower std = more consistent):")
    for i, d in enumerate(spreads):
        flag = "consistent" if np.std(d) < 1.5 else "variable"
        print(f"  {names[i]:<35}  σ={np.std(d):.2f}  {flag}")

    print(f"\n  All plots saved to: {output_dir}/")
    print(f"\n  Files generated:")
    print(f"    1_mean_spectrograms.png     — visual fingerprint of each class")
    print(f"    2_class_similarity_heatmap.png — how confused the model might get")
    print(f"    3_pca_scatter.png           — cluster separation in feature space")
    print(f"    4_intra_class_spread.png    — consistency within each class")
    print(f"\nDone!\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="dataset_ready", help="dataset_ready/ folder")
    parser.add_argument("--output", default="class_viz",     help="output folder")
    args = parser.parse_args()
    run(args.input, args.output)