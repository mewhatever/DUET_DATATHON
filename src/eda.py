"""
Exploratory Data Analysis: a text report + plots that justify the modelling choices.

    python -m src.eda --data-root data --out outputs/eda

Produces in <out>/:
    stats.json                  machine-readable summary
    class_distribution.png      the ~30x imbalance (drives copy-paste + per-class thresholds)
    box_area_hist.png           the tiny-object problem (drives 1280px + P2 head)
    objects_per_image.png       scene density (drives mosaic + max_det)
    images_per_camera.png       only 4 scenes (drives the frame-phase validation split)
    annotated_samples.png       sanity-check boxes; includes rare-class crops
"""

from __future__ import annotations

import argparse
from pathlib import Path

from . import NUM_CLASSES
from .dataset import dataset_stats, load_train_df
from .utils import save_json


def _savefig(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)


def plot_class_distribution(df, out: Path) -> None:
    import matplotlib.pyplot as plt

    counts = df["class_id"].value_counts().reindex(range(NUM_CLASSES), fill_value=0).sort_index()
    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(counts.index.astype(str), counts.values, color="#3b7dd8")
    # highlight rare classes (< 500 instances)
    for b, v in zip(bars, counts.values):
        if v < 500:
            b.set_color("#d8603b")
        ax.text(b.get_x() + b.get_width() / 2, v + 10, str(int(v)), ha="center", va="bottom", fontsize=8)
    ax.set_title("Class distribution (orange = rare, <500 instances)")
    ax.set_xlabel("class_id")
    ax.set_ylabel("# instances")
    _savefig(fig, out / "class_distribution.png")


def plot_box_area(df, out: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    area = (df["width"] * df["height"]).values
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(np.clip(area, 1e-5, 0.2), bins=80, color="#3b7dd8")
    ax.axvline(0.0025, color="#d8603b", ls="--", label="very small (0.25% area)")
    ax.axvline(0.01, color="#7a3bd8", ls="--", label="small (1% area)")
    ax.set_title("Bounding-box area (normalized)  —  74.8%% are < 1%% of the image")
    ax.set_xlabel("box area (w*h, fraction of image)")
    ax.set_ylabel("# boxes")
    ax.legend()
    _savefig(fig, out / "box_area_hist.png")


def plot_objects_per_image(df, out: Path) -> None:
    import matplotlib.pyplot as plt

    per_img = df.groupby("stem").size()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(per_img.values, bins=range(0, per_img.max() + 2), color="#3b7dd8", align="left")
    ax.axvline(per_img.mean(), color="#d8603b", ls="--", label=f"mean={per_img.mean():.1f}")
    ax.set_title("Objects per image (dense scenes -> mosaic helps)")
    ax.set_xlabel("# objects in image")
    ax.set_ylabel("# images")
    ax.legend()
    _savefig(fig, out / "objects_per_image.png")


def plot_camera_counts(df, out: Path) -> None:
    import matplotlib.pyplot as plt

    cam_imgs = df.groupby("camera")["stem"].nunique().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(cam_imgs.index, cam_imgs.values, color="#3b7dd8")
    for i, v in enumerate(cam_imgs.values):
        ax.text(i, v + 1, str(int(v)), ha="center", va="bottom")
    ax.set_title("Images per camera (only 4 scenes -> frame-phase split)")
    ax.set_ylabel("# train images")
    _savefig(fig, out / "images_per_camera.png")


def plot_annotated_samples(df, images_dir: Path, out: Path, n: int = 8) -> None:
    """Grid of images with GT boxes drawn; biased toward frames containing rare classes."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import matplotlib.image as mpimg

    # Prefer images that contain a rare class so the crop shows hard cases.
    rare = set(c for c in range(NUM_CLASSES) if (df["class_id"] == c).sum() < 500)
    rare_stems = df[df["class_id"].isin(rare)]["stem"].unique().tolist()
    other_stems = df["stem"].unique().tolist()
    chosen, seen = [], set()
    for s in rare_stems + other_stems:
        if s not in seen:
            chosen.append(s)
            seen.add(s)
        if len(chosen) >= n:
            break

    cols = 4
    rows = (len(chosen) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 2.4))
    axes = axes.ravel()
    for ax, stem in zip(axes, chosen):
        img_path = images_dir / f"{stem}.jpg"
        if not img_path.exists():
            ax.axis("off")
            continue
        im = mpimg.imread(img_path)
        h, w = im.shape[:2]
        ax.imshow(im)
        for _, r in df[df["stem"] == stem].iterrows():
            x = (r["x_center"] - r["width"] / 2) * w
            y = (r["y_center"] - r["height"] / 2) * h
            col = "#d8603b" if int(r["class_id"]) in rare else "#27e36b"
            ax.add_patch(patches.Rectangle((x, y), r["width"] * w, r["height"] * h,
                                           fill=False, edgecolor=col, linewidth=1.0))
            ax.text(x, max(y - 2, 0), str(int(r["class_id"])), color=col, fontsize=6)
        ax.set_title(stem.split("^")[0] + " …" + stem[-4:], fontsize=7)
        ax.axis("off")
    for ax in axes[len(chosen):]:
        ax.axis("off")
    fig.suptitle("GT samples (orange = rare class). Note many tiny, occluded boxes.", fontsize=11)
    _savefig(fig, out / "annotated_samples.png")


def run_eda(data_root: str | Path = "data", out: str | Path = "outputs/eda") -> dict:
    data_root = Path(data_root)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    df = load_train_df(data_root / "train" / "train.csv")
    images_dir = data_root / "train" / "images"

    stats = dataset_stats(df)
    save_json(stats, out / "stats.json")

    # Text report
    print("\n================ EDA REPORT ================")
    print(f"images={stats['n_images']}  objects={stats['n_objects']}  "
          f"obj/img mean={stats['objects_per_image']['mean']:.1f} max={stats['objects_per_image']['max']}")
    print(f"imbalance(max/min)={stats['imbalance_ratio']:.1f}  "
          f"rarest=class{stats['rarest_class']}  commonest=class{stats['commonest_class']}")
    print(f"box area p50={stats['box_area']['p50']:.5f}  "
          f"tiny(<0.25%)={stats['box_area']['frac_tiny_lt_0p0025']*100:.1f}%  "
          f"small(<1%)={stats['box_area']['frac_small_lt_0p01']*100:.1f}%")
    print(f"images per camera: {stats['images_per_camera']}")
    print("class counts:", stats["class_counts"])
    print("============================================\n")

    try:
        plot_class_distribution(df, out)
        plot_box_area(df, out)
        plot_objects_per_image(df, out)
        plot_camera_counts(df, out)
        plot_annotated_samples(df, images_dir, out)
        print(f"[eda] plots written to {out}")
    except Exception as e:  # plotting is best-effort (e.g. matplotlib missing)
        print(f"[eda] WARNING: plotting skipped ({type(e).__name__}: {e})")
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description="EDA report + plots")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--out", default="outputs/eda")
    args = ap.parse_args()
    run_eda(args.data_root, args.out)


if __name__ == "__main__":
    main()
