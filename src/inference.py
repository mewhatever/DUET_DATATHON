"""
Run a trained model over the 327 test images and cache raw detections.

The cache (a pickle) is the hand-off point to ``src/ensemble.py`` (WBF across
caches) and ``src/submission.py`` (CSV writer). Boxes are stored normalized as
``[class, conf, cx, cy, w, h]`` per image stem — confidences are kept on the 0-1
model scale here; the 0-100 submission scaling happens only in submission.py.

    python -m src.inference --weights checkpoints/e00.../weights/best.pt \
        --data-root data --infer-config configs/inference.yaml \
        --out outputs/predictions/e00.pkl --tta

For a 5-fold ensemble, run this once per fold to N caches, then src/ensemble.py.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np

from .model import load_model
from .utils import load_yaml, stem_of
from .validate import predict_raw


def load_test_images(data_root: str | Path) -> List[str]:
    d = Path(data_root) / "test" / "images"
    imgs = sorted(str(p) for p in d.glob("*.jpg"))
    if not imgs:
        raise FileNotFoundError(f"no test images under {d}")
    return imgs


def infer_to_cache(weights: str, data_root: str, infer_config: str, out_path: str,
                   overrides: dict | None = None) -> Dict[str, np.ndarray]:
    cfg = load_yaml(infer_config)
    if overrides:
        cfg.update({k: v for k, v in overrides.items() if v is not None})

    model = load_model(weights)
    images = load_test_images(data_root)
    print(f"[inference] {len(images)} test images | imgsz={cfg.get('imgsz')} "
          f"conf={cfg.get('conf')} iou={cfg.get('iou')} tta={cfg.get('tta')}")

    preds = predict_raw(
        model, images,
        imgsz=int(cfg.get("imgsz", 1280)),
        conf=float(cfg.get("conf", 0.001)),
        iou=float(cfg.get("iou", 0.6)),
        max_det=int(cfg.get("max_det", 300)),
        tta=bool(cfg.get("tta", True)),
    )

    # Guarantee every test image is present (empty array if the model returned none).
    for p in images:
        preds.setdefault(stem_of(Path(p).name), np.zeros((0, 6)))

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "preds": preds,
        "meta": {"weights": str(weights), "imgsz": cfg.get("imgsz"),
                 "iou": cfg.get("iou"), "tta": cfg.get("tta"), "n_images": len(images)},
    }
    with open(out, "wb") as f:
        pickle.dump(payload, f)
    n_boxes = sum(len(v) for v in preds.values())
    print(f"[inference] cached {len(preds)} images, {n_boxes} boxes -> {out}")
    return preds


def main() -> None:
    ap = argparse.ArgumentParser(description="Predict test images -> prediction cache")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--infer-config", default="configs/inference.yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--conf", type=float, default=None)
    ap.add_argument("--iou", type=float, default=None)
    ap.add_argument("--tta", dest="tta", action="store_true", default=None)
    ap.add_argument("--no-tta", dest="tta", action="store_false")
    args = ap.parse_args()

    overrides = {"imgsz": args.imgsz, "conf": args.conf, "iou": args.iou, "tta": args.tta}
    infer_to_cache(args.weights, args.data_root, args.infer_config, args.out, overrides)


if __name__ == "__main__":
    main()
