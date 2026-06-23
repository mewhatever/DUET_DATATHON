"""
Combine multiple prediction caches with Weighted Box Fusion (WBF).

WBF averages the coordinates of agreeing boxes (rather than discarding them like
NMS), which is why it beats NMS for fusing diverse members — different folds,
input scales, or YOLO + RT-DETR. Tune ``iou_thr`` / ``skip_box_thr`` (and member
``weights``) on the val split via the same metric in src/validate.py.

    python -m src.ensemble --preds outputs/predictions/fold0.pkl fold1.pkl ... \
        --out outputs/predictions/wbf.pkl --iou 0.55 --skip 0.001

Output is a prediction cache with the same schema as src/inference.py, so it feeds
straight into src/submission.py.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .utils import yolo_to_xyxy, xyxy_to_yolo


def _load_cache(path: str) -> Dict[str, np.ndarray]:
    with open(path, "rb") as f:
        payload = pickle.load(f)
    return payload["preds"] if isinstance(payload, dict) and "preds" in payload else payload


def fuse_caches(cache_paths: List[str], weights: Optional[List[float]] = None,
                iou_thr: float = 0.55, skip_box_thr: float = 0.001,
                conf_type: str = "avg") -> Dict[str, np.ndarray]:
    """WBF-fuse N caches -> {stem: ndarray [class, conf, cx, cy, w, h]}."""
    from ensemble_boxes import weighted_boxes_fusion

    caches = [_load_cache(p) for p in cache_paths]
    weights = weights or [1.0] * len(caches)
    stems = sorted(set().union(*[set(c.keys()) for c in caches]))

    fused: Dict[str, np.ndarray] = {}
    for stem in stems:
        boxes_l, scores_l, labels_l, w_l = [], [], [], []
        for c, wt in zip(caches, weights):
            arr = c.get(stem, np.zeros((0, 6)))
            if arr.shape[0] == 0:
                continue
            xyxy = np.clip(yolo_to_xyxy(arr[:, 2:6]), 0.0, 1.0)  # WBF wants xyxy in [0,1]
            boxes_l.append(xyxy.tolist())
            scores_l.append(arr[:, 1].tolist())
            labels_l.append(arr[:, 0].astype(int).tolist())
            w_l.append(wt)

        if not boxes_l:
            fused[stem] = np.zeros((0, 6))
            continue

        boxes, scores, labels = weighted_boxes_fusion(
            boxes_l, scores_l, labels_l, weights=w_l,
            iou_thr=iou_thr, skip_box_thr=skip_box_thr, conf_type=conf_type,
        )
        if len(boxes) == 0:
            fused[stem] = np.zeros((0, 6))
            continue
        yolo = xyxy_to_yolo(np.asarray(boxes))
        fused[stem] = np.concatenate(
            [np.asarray(labels).reshape(-1, 1), np.asarray(scores).reshape(-1, 1), yolo], axis=1
        )
    return fused


def main() -> None:
    ap = argparse.ArgumentParser(description="WBF-fuse prediction caches")
    ap.add_argument("--preds", nargs="+", required=True, help="two or more .pkl caches")
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", nargs="+", type=float, default=None, help="per-member weights")
    ap.add_argument("--iou", type=float, default=0.55)
    ap.add_argument("--skip", type=float, default=0.001)
    ap.add_argument("--conf-type", default="avg", choices=["avg", "max", "box_and_model_avg", "absent_model_aware_avg"])
    args = ap.parse_args()

    fused = fuse_caches(args.preds, args.weights, args.iou, args.skip, args.conf_type)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump({"preds": fused, "meta": {"members": args.preds, "iou": args.iou, "skip": args.skip}}, f)
    n_boxes = sum(len(v) for v in fused.values())
    print(f"[ensemble] fused {len(args.preds)} caches -> {len(fused)} images, {n_boxes} boxes -> {out}")


if __name__ == "__main__":
    main()
