"""
Foundational helpers shared across the pipeline.

Deliberately light on heavy imports: torch / cv2 are imported lazily *inside* the
functions that need them, so data-prep, EDA and the metric unit-test all run on a
plain CPU machine (e.g. a Mac without CUDA) where torch may be absent.

Contents
--------
* reproducibility ........ set_seed
* id helpers ............. stem_of / to_txt_id / to_jpg_name  (the .txt vs .jpg gotcha)
* (de)serialization ...... load_yaml / save_yaml / load_json / save_json
* box geometry ........... yolo_to_xyxy / xyxy_to_yolo / box_iou_xyxy
* metric ................. compute_map50  (VOC all-points AP @ IoU 0.50)
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

# Project root = parent of this src/ directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def set_seed(seed: int = 42) -> None:
    """Seed python / numpy / torch (if installed) for deterministic-ish runs."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:  # torch is optional on the data-prep / EDA machine
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Determinism without crashing on ops that lack a deterministic kernel.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# image_id helpers  (train.csv & sample_submission use ".txt"; images are ".jpg")
# --------------------------------------------------------------------------- #
def stem_of(image_id: str) -> str:
    """Strip a trailing .txt/.jpg/.jpeg/.png so ids compare regardless of extension."""
    name = os.path.basename(str(image_id))
    for ext in (".txt", ".jpg", ".jpeg", ".png", ".JPG", ".PNG"):
        if name.endswith(ext):
            return name[: -len(ext)]
    return name


def to_txt_id(image_id: str) -> str:
    """Canonical submission / label id  ->  '<stem>.txt' (what the grader expects)."""
    return stem_of(image_id) + ".txt"


def to_jpg_name(image_id: str) -> str:
    """Canonical image filename  ->  '<stem>.jpg' (what is on disk)."""
    return stem_of(image_id) + ".jpg"


# --------------------------------------------------------------------------- #
# (de)serialization
# --------------------------------------------------------------------------- #
def load_yaml(path: str | Path) -> dict:
    import yaml

    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_yaml(obj: dict, path: str | Path) -> None:
    import yaml

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(obj, f, sort_keys=False, default_flow_style=False)


def load_json(path: str | Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def save_json(obj, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


# --------------------------------------------------------------------------- #
# Box geometry.  YOLO boxes are (cx, cy, w, h) normalized to [0, 1].
# --------------------------------------------------------------------------- #
def yolo_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """(cx,cy,w,h) -> (x1,y1,x2,y2), same units. Accepts (N,4)."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    out = np.empty_like(boxes)
    out[:, 0] = cx - w / 2.0
    out[:, 1] = cy - h / 2.0
    out[:, 2] = cx + w / 2.0
    out[:, 3] = cy + h / 2.0
    return out


def xyxy_to_yolo(boxes: np.ndarray) -> np.ndarray:
    """(x1,y1,x2,y2) -> (cx,cy,w,h)."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    out = np.empty_like(boxes)
    out[:, 0] = (x1 + x2) / 2.0
    out[:, 1] = (y1 + y2) / 2.0
    out[:, 2] = (x2 - x1)
    out[:, 3] = (y2 - y1)
    return out


def box_iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between boxes a (N,4) and b (M,4) in xyxy. Returns (N,M)."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float64)
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])      # (N,M,2)
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])      # (N,M,2)
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


# --------------------------------------------------------------------------- #
# Metric: mean Average Precision @ IoU 0.50  (VOC all-points / COCO-style area).
# --------------------------------------------------------------------------- #
def _voc_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """Continuous area under the (monotonic-enveloped) precision-recall curve."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    for i in range(mpre.size - 1, 0, -1):           # precision envelope
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]         # recall changes
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def _to_array(rows: Sequence, ncol: int) -> np.ndarray:
    arr = np.asarray(list(rows), dtype=np.float64) if len(rows) else np.zeros((0, ncol))
    return arr.reshape(-1, ncol)


def compute_map50(
    predictions: Dict[str, Sequence],
    ground_truths: Dict[str, Sequence],
    num_classes: int = 13,
    iou_thr: float = 0.5,
) -> Tuple[float, Dict[int, float]]:
    """
    Standard detection mAP@0.5 — the SAME definition we optimize for the leaderboard.

    Parameters
    ----------
    predictions   : {image_id: array-like of [class, conf, cx, cy, w, h]}  (normalized boxes)
    ground_truths : {image_id: array-like of [class, cx, cy, w, h]}        (normalized boxes)
                    image_id keys are matched by stem, so .txt vs .jpg does not matter.
    num_classes   : number of classes (13 here).

    Returns
    -------
    (mean_ap, {class_id: ap})  — classes with no GT are excluded from the mean,
    matching how VOC/COCO report mAP.
    """
    # Re-key everything by stem so prediction/GT extensions need not match.
    preds = {stem_of(k): _to_array(v, 6) for k, v in predictions.items()}
    gts = {stem_of(k): _to_array(v, 5) for k, v in ground_truths.items()}

    per_class_ap: Dict[int, float] = {}
    aps: List[float] = []

    for cls in range(num_classes):
        # ---- collect this class's GT boxes per image + total count ----
        gt_by_img: Dict[str, np.ndarray] = {}
        n_gt = 0
        for img, g in gts.items():
            if g.shape[0] == 0:
                continue
            m = g[g[:, 0] == cls]
            if m.shape[0]:
                gt_by_img[img] = yolo_to_xyxy(m[:, 1:5])
                n_gt += m.shape[0]
        if n_gt == 0:
            continue  # no GT for this class -> not counted in mAP

        # ---- collect this class's predictions across all images ----
        entries = []  # (conf, img, xyxy)
        for img, p in preds.items():
            if p.shape[0] == 0:
                continue
            m = p[p[:, 0] == cls]
            if m.shape[0] == 0:
                continue
            xyxy = yolo_to_xyxy(m[:, 2:6])
            for conf, bb in zip(m[:, 1], xyxy):
                entries.append((float(conf), img, bb))

        if not entries:
            per_class_ap[cls] = 0.0
            aps.append(0.0)
            continue

        entries.sort(key=lambda e: e[0], reverse=True)  # descending confidence
        matched = {img: np.zeros(len(b), dtype=bool) for img, b in gt_by_img.items()}
        tp = np.zeros(len(entries))
        fp = np.zeros(len(entries))

        for i, (_, img, bb) in enumerate(entries):
            gboxes = gt_by_img.get(img)
            if gboxes is None or gboxes.shape[0] == 0:
                fp[i] = 1
                continue
            ious = box_iou_xyxy(bb[None, :], gboxes)[0]
            j = int(np.argmax(ious))
            if ious[j] >= iou_thr and not matched[img][j]:
                tp[i] = 1
                matched[img][j] = True
            else:
                fp[i] = 1

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        recall = tp_cum / (n_gt + 1e-12)
        precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
        ap = _voc_ap(recall, precision)
        per_class_ap[cls] = ap
        aps.append(ap)

    mean_ap = float(np.mean(aps)) if aps else 0.0
    return mean_ap, per_class_ap


def read_sample_ids(sample_submission_csv: str | Path) -> List[str]:
    """Return the ordered list of test image_ids (as given, '.txt') from sample_submission."""
    import csv

    ids: List[str] = []
    with open(sample_submission_csv, "r", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)  # header
        for row in reader:
            if row:
                ids.append(row[0].strip())
    return ids
