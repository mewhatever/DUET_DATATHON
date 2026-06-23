"""
Dataset parsing, statistics, and the **validation split**.

The single most important design choice in this competition lives here.

Why a frame-phase split (and not random / not group-by-camera)
--------------------------------------------------------------
EDA shows the data is only **4 cameras** (== 4 time-windows / scenes), and the
**test set is interleaved frames of those same 4 cameras** (test frame 061 sits
between train frames 060 and 062; no frame overlap). Therefore:

* random split  -> leaks near-duplicate adjacent frames *and* does not preserve
  the per-camera composition the leaderboard is graded on.
* group-by-camera (hold out a whole camera) -> measures *cross-camera* transfer,
  which is NOT what the test set asks (it contains the same 4 cameras). With only
  4 groups it is also high-variance.
* **camera-stratified frame-phase** -> within each camera, sort frames and assign
  `fold = rank % k`. The held-out fold is then temporally interleaved among the
  training frames *exactly like the real test set*, and all 4 scenes appear in both
  splits in proportion. This makes local val mAP track the leaderboard.

Caveat (documented, not a bug): adjacent near-duplicate frames make absolute val
mAP mildly optimistic — but the test set has the identical structure, so the
optimism is *calibrated to the leaderboard*, not spurious.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from . import CAMERAS, NUM_CLASSES
from .utils import stem_of

# CCTV01^20260318-110000-20260318-120000_000062(.txt|.jpg)
_ID_RE = re.compile(r"^(?P<cam>[^\^]+)\^(?P<window>.+)_(?P<frame>\d+)$")


@dataclass(frozen=True)
class ImageMeta:
    stem: str          # full id without extension
    camera: str        # e.g. "CCTV01"
    window: str        # e.g. "20260318-110000-20260318-120000"
    frame: int         # frame number within the window


def parse_image_id(image_id: str) -> ImageMeta:
    """Decompose an image_id into (camera, time-window, frame number)."""
    s = stem_of(image_id)
    m = _ID_RE.match(s)
    if not m:
        # Defensive fallback: unknown layout -> treat whole thing as one camera/window.
        return ImageMeta(stem=s, camera=s.split("^")[0] if "^" in s else s, window="NA", frame=0)
    return ImageMeta(stem=s, camera=m.group("cam"), window=m.group("window"), frame=int(m.group("frame")))


def camera_of(image_id: str) -> str:
    return parse_image_id(image_id).camera


# --------------------------------------------------------------------------- #
# Folds
# --------------------------------------------------------------------------- #
def assign_folds(stems: List[str], k: int = 5) -> Dict[str, int]:
    """
    Camera-stratified frame-phase fold assignment.

    Within each camera, frames are sorted by frame number and assigned
    ``fold = rank % k``. Returns {stem: fold_index in [0, k)}.
    """
    by_cam: Dict[str, List[ImageMeta]] = {}
    for s in stems:
        m = parse_image_id(s)
        by_cam.setdefault(m.camera, []).append(m)

    folds: Dict[str, int] = {}
    for cam, metas in by_cam.items():
        metas.sort(key=lambda m: m.frame)            # temporal order -> phase = rank % k
        for rank, m in enumerate(metas):
            folds[m.stem] = rank % k
    return folds


def split_train_val(stems: List[str], val_fold: int = 0, k: int = 5) -> Tuple[List[str], List[str]]:
    """Return (train_stems, val_stems) for a given held-out fold of the phase split."""
    folds = assign_folds(stems, k=k)
    train = sorted(s for s in stems if folds[s] != val_fold)
    val = sorted(s for s in stems if folds[s] == val_fold)
    return train, val


# --------------------------------------------------------------------------- #
# Statistics (used by eda.py and prepare_data.py sanity checks)
# --------------------------------------------------------------------------- #
def load_train_df(csv_path: str | Path):
    """Load train.csv into a DataFrame with an added 'camera' column."""
    import pandas as pd

    df = pd.read_csv(csv_path)
    df["camera"] = df["image_id"].map(camera_of)
    df["stem"] = df["image_id"].map(stem_of)
    df["frame"] = df["image_id"].map(lambda x: parse_image_id(x).frame)
    return df


def dataset_stats(df) -> dict:
    """Summarize the training annotations (counts, imbalance, box sizes, density)."""
    import numpy as np

    n_imgs = df["stem"].nunique()
    n_obj = len(df)
    per_img = df.groupby("stem").size()
    cls_counts = df["class_id"].value_counts().sort_index().to_dict()
    cam_counts = df.groupby("camera")["stem"].nunique().to_dict()
    area = (df["width"] * df["height"]).values
    counts = np.array([cls_counts.get(c, 0) for c in range(NUM_CLASSES)])
    nonzero = counts[counts > 0]

    return {
        "n_images": int(n_imgs),
        "n_objects": int(n_obj),
        "objects_per_image": {
            "min": int(per_img.min()),
            "max": int(per_img.max()),
            "mean": float(per_img.mean()),
            "median": float(per_img.median()),
        },
        "class_counts": {int(c): int(cls_counts.get(c, 0)) for c in range(NUM_CLASSES)},
        "imbalance_ratio": float(nonzero.max() / nonzero.min()) if nonzero.size else 0.0,
        "rarest_class": int(counts.argmin()),
        "commonest_class": int(counts.argmax()),
        "images_per_camera": {str(k): int(v) for k, v in cam_counts.items()},
        "box_area": {
            "p05": float(np.percentile(area, 5)),
            "p50": float(np.percentile(area, 50)),
            "p95": float(np.percentile(area, 95)),
            "frac_tiny_lt_0p0025": float((area < 0.0025).mean()),
            "frac_small_lt_0p01": float((area < 0.01).mean()),
        },
    }


def list_train_stems(images_dir: str | Path) -> List[str]:
    """All training image stems present on disk."""
    return sorted(stem_of(p.name) for p in Path(images_dir).glob("*.jpg"))
