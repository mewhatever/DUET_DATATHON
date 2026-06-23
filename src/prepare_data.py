"""
Convert the competition CSV into an Ultralytics-ready dataset.

Steps
-----
1. Parse ``data/train/train.csv`` -> one YOLO label file ``labels/<stem>.txt`` per
   image (lines: ``class cx cy w h``; boxes are already normalized -> copied as-is).
2. Link the ``.jpg`` images into ``<out>/images/`` (so Ultralytics' images<->labels
   path derivation works).  The ``.txt`` image_id in the CSV maps to ``<stem>.jpg``.
3. Build the **camera-stratified frame-phase folds** (see src/dataset.py) and write
   one image-list per fold:  ``train_fold{k}.txt`` / ``val_fold{k}.txt``.
4. Write ``configs/data.yaml`` pointing at the chosen validation fold.

Run again on every new environment (local vs Kaggle) so absolute paths are correct.

    python -m src.prepare_data --data-root data --out data/yolo --folds 5 --val-fold 0
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from . import NUM_CLASSES
from .dataset import assign_folds, load_train_df
from .utils import PROJECT_ROOT, save_yaml, stem_of


def _link(src: Path, dst: Path, mode: str) -> None:
    """Create dst pointing at src using symlink (default) or copy."""
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "copy":
        import shutil

        shutil.copy2(src, dst)
        return
    try:
        os.symlink(src.resolve(), dst)
    except OSError:  # filesystems without symlink support -> copy
        import shutil

        shutil.copy2(src, dst)


def prepare(
    data_root: str | Path = "data",
    out: str | Path = "data/yolo",
    folds: int = 5,
    val_fold: int = 0,
    link_mode: str = "symlink",
    config_out: str | Path = "configs/data.yaml",
) -> dict:
    data_root = Path(data_root)
    out = Path(out)
    train_csv = data_root / "train" / "train.csv"
    src_images = data_root / "train" / "images"
    assert train_csv.exists(), f"missing {train_csv}"
    assert src_images.exists(), f"missing {src_images}"

    img_dir = out / "images"
    lbl_dir = out / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    df = load_train_df(train_csv)

    # ---- 1+2: write labels and link images, grouped by image ----
    n_box_clipped = 0
    stems = []
    for stem, g in df.groupby("stem"):
        stems.append(stem)
        # label file
        lines = []
        for _, r in g.iterrows():
            cls = int(r["class_id"])
            cx, cy, w, h = float(r["x_center"]), float(r["y_center"]), float(r["width"]), float(r["height"])
            # YOLO requires center in [0,1]; clip defensively and count.
            cc = (min(max(cx, 0.0), 1.0), min(max(cy, 0.0), 1.0), min(max(w, 0.0), 1.0), min(max(h, 0.0), 1.0))
            if (cc != (cx, cy, w, h)):
                n_box_clipped += 1
            lines.append(f"{cls} {cc[0]:.6f} {cc[1]:.6f} {cc[2]:.6f} {cc[3]:.6f}")
        (lbl_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n")

        # image link
        src_img = src_images / f"{stem}.jpg"
        if not src_img.exists():
            raise FileNotFoundError(f"image referenced by train.csv not found: {src_img}")
        _link(src_img, img_dir / f"{stem}.jpg", link_mode)

    # ---- 3: folds -> image-list files (absolute paths) ----
    fold_map = assign_folds(stems, k=folds)
    # IMPORTANT: do NOT .resolve() — that would follow the symlink back into the
    # (read-only) source images dir, and Ultralytics would then derive the label
    # path next to the *source* images instead of <out>/labels. Use the writable
    # <out>/images/ path so images<->labels derivation lands in <out>/labels.
    abs_img = lambda s: str((img_dir / f"{s}.jpg").absolute())
    for k in range(folds):
        tr = sorted(abs_img(s) for s in stems if fold_map[s] != k)
        va = sorted(abs_img(s) for s in stems if fold_map[s] == k)
        (out / f"train_fold{k}.txt").write_text("\n".join(tr) + "\n")
        (out / f"val_fold{k}.txt").write_text("\n".join(va) + "\n")

    # ---- 4: data.yaml for the chosen val fold ----
    names = {i: f"class_{i}" for i in range(NUM_CLASSES)}
    data_yaml = {
        "path": str(out.resolve()),
        "train": f"train_fold{val_fold}.txt",
        "val": f"val_fold{val_fold}.txt",
        "nc": NUM_CLASSES,
        "names": names,
    }
    config_out = Path(config_out)
    if not config_out.is_absolute():
        config_out = PROJECT_ROOT / config_out
    save_yaml(data_yaml, config_out)

    # ---- validation / summary ----
    n_labels = len(list(lbl_dir.glob("*.txt")))
    n_val = sum(1 for s in stems if fold_map[s] == val_fold)
    summary = {
        "n_images": len(stems),
        "n_label_files": n_labels,
        "boxes_clipped": n_box_clipped,
        "folds": folds,
        "val_fold": val_fold,
        "n_train": len(stems) - n_val,
        "n_val": n_val,
        "data_yaml": str(config_out),
        "yolo_root": str(out.resolve()),
    }
    assert n_labels == len(stems), f"label files ({n_labels}) != images ({len(stems)})"
    print("[prepare_data] " + ", ".join(f"{k}={v}" for k, v in summary.items()))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the Ultralytics dataset from train.csv")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--out", default="data/yolo")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--val-fold", type=int, default=0)
    ap.add_argument("--link-mode", choices=["symlink", "copy"], default="symlink")
    ap.add_argument("--config-out", default="configs/data.yaml")
    args = ap.parse_args()
    prepare(
        data_root=args.data_root,
        out=args.out,
        folds=args.folds,
        val_fold=args.val_fold,
        link_mode=args.link_mode,
        config_out=args.config_out,
    )


if __name__ == "__main__":
    main()
