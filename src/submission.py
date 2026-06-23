"""
Write the competition submission CSV from a prediction cache.

Format (matches data/test/sample_submission.csv EXACTLY):
    image_id,PredictionString
    <stem>.txt,<class conf cx cy w h  class conf cx cy w h ...>
    <stem>.txt,                       <-- images with no detections still appear

Gotchas handled here (each a silent score-killer otherwise)
* image_id uses **.txt** (sample_submission uses .txt; images are .jpg).
* confidence is emitted on the sample's **0-100** scale (configurable via conf_scale;
  mAP is rank-invariant so the scale does not change the score, but we mirror the sample).
* **every** test image appears (empty PredictionString if no detections). The id list
  comes from the test-images directory (327 imgs) because sample_submission.csv here is
  only a truncated 10-row example -- see collect_test_ids().
* boxes/conf are clipped to valid ranges and class ids cast to int.

    python -m src.submission --preds outputs/predictions/e00.pkl \
        --infer-config configs/inference.yaml \
        --sample data/test/sample_submission.csv \
        --out outputs/submissions/submission_e00.csv
"""

from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path
from typing import Dict

import numpy as np

from .utils import load_yaml, read_sample_ids, stem_of, to_txt_id


def _load_preds(path: str) -> Dict[str, np.ndarray]:
    with open(path, "rb") as f:
        payload = pickle.load(f)
    return payload["preds"] if isinstance(payload, dict) and "preds" in payload else payload


def _format_image(boxes: np.ndarray, class_conf: Dict[int, float],
                  conf_scale: float, conf_dec: int, box_dec: int) -> str:
    """boxes: ndarray [class, conf, cx, cy, w, h] -> 'PredictionString' for one image."""
    if boxes is None or len(boxes) == 0:
        return ""
    order = np.argsort(-boxes[:, 1])  # highest confidence first
    parts = []
    for row in boxes[order]:
        cls = int(round(row[0]))
        conf = float(row[1])
        if conf < class_conf.get(cls, 0.0):
            continue
        conf = min(max(conf, 0.0), 1.0) * conf_scale
        cx, cy, w, h = (min(max(float(v), 0.0), 1.0) for v in row[2:6])
        parts.append(
            f"{cls} {conf:.{conf_dec}f} {cx:.{box_dec}f} {cy:.{box_dec}f} {w:.{box_dec}f} {h:.{box_dec}f}"
        )
    return " ".join(parts)


def collect_test_ids(test_images_dir: str | None = None, sample_csv: str | None = None,
                     preds: Dict[str, np.ndarray] | None = None) -> list:
    """
    Authoritative, COMPLETE list of test image stems to emit.

    The provided sample_submission.csv is only a *truncated* example (10 rows) while
    the real test set has 327 images, so the test-images directory is the source of
    truth. We union it with any sample ids and the prediction-cache keys so no
    required id is ever dropped.
    """
    stems = set()
    if test_images_dir and Path(test_images_dir).is_dir():
        stems |= {stem_of(p.name) for p in Path(test_images_dir).glob("*.jpg")}
    if sample_csv and Path(sample_csv).exists():
        stems |= {stem_of(i) for i in read_sample_ids(sample_csv)}
    if preds:
        stems |= set(preds.keys())
    if not stems:
        raise ValueError("no test ids found (provide --test-images, --sample, or a non-empty cache)")
    return sorted(stems)


def write_submission(preds: Dict[str, np.ndarray], out_csv: str,
                     test_images_dir: str | None = None, sample_csv: str | None = None,
                     infer_cfg: dict | None = None) -> dict:
    infer_cfg = infer_cfg or {}
    conf_scale = float(infer_cfg.get("conf_scale", 100.0))
    conf_dec = int(infer_cfg.get("conf_decimals", 2))
    box_dec = int(infer_cfg.get("box_decimals", 6))
    raw_cc = infer_cfg.get("class_conf", {}) or {}
    class_conf = {int(k): float(v) for k, v in raw_cc.items()}

    ids = collect_test_ids(test_images_dir, sample_csv, preds)
    out = Path(out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)

    n_with, n_boxes, n_empty = 0, 0, 0
    with open(out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "PredictionString"])
        for stem in ids:
            boxes = preds.get(stem, np.zeros((0, 6)))
            pred_str = _format_image(boxes, class_conf, conf_scale, conf_dec, box_dec)
            if pred_str:
                n_with += 1
                n_boxes += len(pred_str.split()) // 6
            else:
                n_empty += 1
            writer.writerow([to_txt_id(stem), pred_str])

    summary = {"rows": len(ids), "with_detections": n_with,
               "empty": n_empty, "total_boxes": n_boxes, "path": str(out)}
    assert summary["rows"] == len(ids)
    print("[submission] " + ", ".join(f"{k}={v}" for k, v in summary.items()))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Write submission CSV from a prediction cache")
    ap.add_argument("--preds", required=True, help="prediction cache .pkl (from inference.py / ensemble.py)")
    ap.add_argument("--test-images", default="data/test/images",
                    help="authoritative source of the COMPLETE test id list (327 images)")
    ap.add_argument("--sample", default="data/test/sample_submission.csv",
                    help="optional; only unioned in (it is a truncated 10-row example)")
    ap.add_argument("--infer-config", default="configs/inference.yaml")
    ap.add_argument("--out", default="outputs/submissions/submission.csv")
    args = ap.parse_args()

    preds = _load_preds(args.preds)
    infer_cfg = load_yaml(args.infer_config) if Path(args.infer_config).exists() else {}
    write_submission(preds, args.out, args.test_images, args.sample, infer_cfg)


if __name__ == "__main__":
    main()
