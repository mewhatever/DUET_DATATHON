"""
Validation + threshold tuning on the test-mimicking frame-phase split.

Reports the metric we are actually graded on — mAP@0.5 — broken down **globally,
per camera, and per class**, using our own VOC-style implementation
(``src.utils.compute_map50``) so it matches the leaderboard definition exactly.

    # plain evaluation
    python -m src.validate --weights checkpoints/e00.../weights/best.pt --data configs/data.yaml

    # also tune the NMS IoU and persist inference settings
    python -m src.validate --weights ... --data configs/data.yaml --tune --save configs/inference.yaml

A note on thresholds and mAP
----------------------------
mAP@0.5 integrates the full precision-recall curve, so it is *invariant* to a
confidence cutoff applied afterward — the AP-optimal confidence threshold is ~0
(emit everything, ranked). We therefore keep ``conf`` very low and tune the levers
that genuinely move AP: **NMS IoU**, inference **resolution**, and **TTA** (and,
for ensembles, the WBF settings in src/ensemble.py). Per-class ``class_conf`` stays
low by default but is reported per class so you can see weak classes at a glance.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from . import CAMERAS, NUM_CLASSES
from .dataset import camera_of
from .model import load_model
from .utils import compute_map50, load_yaml, save_yaml, stem_of


def _read_val_list(data_yaml: str) -> Tuple[Path, List[str]]:
    cfg = load_yaml(data_yaml)
    root = Path(cfg["path"])
    val = cfg["val"]
    val_path = root / val if not Path(val).is_absolute() else Path(val)
    if val_path.is_dir():
        images = sorted(str(p) for p in val_path.glob("*.jpg"))
    else:
        images = [ln.strip() for ln in val_path.read_text().splitlines() if ln.strip()]
    return root, images


def _gt_from_labels(yolo_root: Path, stems: List[str]) -> Dict[str, np.ndarray]:
    gts: Dict[str, np.ndarray] = {}
    lbl_dir = yolo_root / "labels"
    for s in stems:
        f = lbl_dir / f"{s}.txt"
        rows = []
        if f.exists():
            for ln in f.read_text().splitlines():
                p = ln.split()
                if len(p) == 5:
                    rows.append([float(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4])])
        gts[s] = np.array(rows, dtype=np.float64).reshape(-1, 5)
    return gts


def predict_raw(model, image_paths: List[str], imgsz: int, conf: float, iou: float,
                max_det: int, tta: bool) -> Dict[str, np.ndarray]:
    """Run inference -> {stem: ndarray of [class, conf, cx, cy, w, h] (normalized)}."""
    preds: Dict[str, np.ndarray] = {}
    results = model.predict(source=image_paths, imgsz=imgsz, conf=conf, iou=iou,
                            max_det=max_det, augment=tta, verbose=False, stream=True)
    for r in results:
        stem = stem_of(Path(r.path).name)
        b = r.boxes
        if b is None or b.shape[0] == 0:
            preds[stem] = np.zeros((0, 6))
            continue
        xywhn = b.xywhn.cpu().numpy()
        cls = b.cls.cpu().numpy().reshape(-1, 1)
        cf = b.conf.cpu().numpy().reshape(-1, 1)
        preds[stem] = np.concatenate([cls, cf, xywhn], axis=1)
    return preds


def evaluate(preds: Dict[str, np.ndarray], gts: Dict[str, np.ndarray]):
    """Return (global_map, per_class_ap, per_camera_map)."""
    gmap, per_class = compute_map50(preds, gts, num_classes=NUM_CLASSES)

    per_cam = {}
    for cam in CAMERAS:
        sub_p = {k: v for k, v in preds.items() if camera_of(k) == cam}
        sub_g = {k: v for k, v in gts.items() if camera_of(k) == cam}
        if sub_g:
            per_cam[cam], _ = compute_map50(sub_p, sub_g, num_classes=NUM_CLASSES)
    return gmap, per_class, per_cam


def print_report(gmap, per_class, per_cam, title="VALIDATION") -> None:
    print(f"\n================ {title} : mAP@0.5 ================")
    print(f"  GLOBAL mAP@0.5 = {gmap:.4f}")
    print("  per-camera:  " + "  ".join(f"{c}={per_cam.get(c, float('nan')):.4f}" for c in CAMERAS))
    print("  per-class AP:")
    for c in range(NUM_CLASSES):
        if c in per_class:
            print(f"     class {c:2d}: AP={per_class[c]:.4f}")
    print("=" * (34 + len(title)) + "\n")


def tune(weights: str, data_yaml: str, base_infer: dict,
         iou_grid=(0.5, 0.55, 0.6, 0.65, 0.7)) -> dict:
    """Sweep NMS IoU on val, keep the best; return an updated inference config dict."""
    model = load_model(weights)
    root, image_paths = _read_val_list(data_yaml)
    stems = [stem_of(Path(p).name) for p in image_paths]
    gts = _gt_from_labels(root, stems)

    imgsz = int(base_infer.get("imgsz", 1280))
    conf = float(base_infer.get("conf", 0.001))
    max_det = int(base_infer.get("max_det", 300))
    tta = bool(base_infer.get("tta", True))

    best = {"iou": None, "map": -1.0, "per_class": {}, "per_cam": {}}
    for iou in iou_grid:
        preds = predict_raw(model, image_paths, imgsz, conf, iou, max_det, tta)
        gmap, per_class, per_cam = evaluate(preds, gts)
        print(f"[tune] iou={iou:.2f} -> mAP@0.5={gmap:.4f}")
        if gmap > best["map"]:
            best = {"iou": iou, "map": gmap, "per_class": per_class, "per_cam": per_cam}

    print_report(best["map"], best["per_class"], best["per_cam"], title=f"BEST (iou={best['iou']})")

    out = dict(base_infer)
    out["iou"] = float(best["iou"])
    out["imgsz"] = imgsz
    out["conf"] = conf
    out["tta"] = tta
    # keep class_conf low (AP-optimal); record measured per-class AP for inspection
    out["_val_global_map"] = round(float(best["map"]), 4)
    out["_val_per_camera"] = {k: round(float(v), 4) for k, v in best["per_cam"].items()}
    out["_val_per_class_ap"] = {int(k): round(float(v), 4) for k, v in best["per_class"].items()}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate + tune on the frame-phase val split")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", default="configs/data.yaml")
    ap.add_argument("--infer-config", default="configs/inference.yaml")
    ap.add_argument("--tune", action="store_true", help="sweep NMS IoU and save inference config")
    ap.add_argument("--save", default=None, help="where to write the tuned inference config")
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--no-tta", action="store_true")
    args = ap.parse_args()

    base_infer = load_yaml(args.infer_config)
    if args.imgsz:
        base_infer["imgsz"] = args.imgsz
    if args.no_tta:
        base_infer["tta"] = False

    if args.tune:
        tuned = tune(args.weights, args.data, base_infer)
        save_to = args.save or args.infer_config
        save_yaml(tuned, save_to)
        print(f"[validate] tuned inference config saved -> {save_to}")
    else:
        model = load_model(args.weights)
        root, image_paths = _read_val_list(args.data)
        stems = [stem_of(Path(p).name) for p in image_paths]
        gts = _gt_from_labels(root, stems)
        preds = predict_raw(model, image_paths,
                            imgsz=int(base_infer.get("imgsz", 1280)),
                            conf=float(base_infer.get("conf", 0.001)),
                            iou=float(base_infer.get("iou", 0.6)),
                            max_det=int(base_infer.get("max_det", 300)),
                            tta=bool(base_infer.get("tta", True)))
        gmap, per_class, per_cam = evaluate(preds, gts)
        print_report(gmap, per_class, per_cam)


if __name__ == "__main__":
    main()
