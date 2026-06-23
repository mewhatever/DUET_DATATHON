"""
OPTIONAL semi-supervised pseudo-labeling.  *** DISABLED BY DEFAULT ***

!!! RULES WARNING !!!
The competition forbids "using test labels" and "any form of test-set leakage".
Auto-labeling the *test* images and training on them is plausibly test-set leakage
and could disqualify a submission. No separate unlabeled pool was provided, so this
module is OFF by default. Only enable it on an **external** Bangladesh-traffic image
pool that the rules explicitly allow.

Workflow (standard high-confidence pseudo-labeling)
1. Train an initial detector on the 810 labeled images.
2. Predict on the (allowed) unlabeled pool; keep only high-confidence boxes.
3. Write YOLO labels + a combined image-list (labeled + pseudo).
4. Retrain; keep only if val mAP@0.5 improves.

    python -m src.pseudo_label --weights checkpoints/e00.../weights/best.pt \
        --images <ALLOWED_UNLABELED_DIR> --out data/yolo_pseudo \
        --conf 0.5 --i-confirm-rules-compliant
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .model import load_model
from .utils import stem_of


def generate_pseudo_labels(weights: str, images_dir: str, out_dir: str,
                           conf: float = 0.5, imgsz: int = 1280, tta: bool = True,
                           per_class_conf: Optional[Dict[int, float]] = None) -> int:
    """Predict on an unlabeled pool and write high-confidence YOLO labels. Returns #boxes kept."""
    images_dir = Path(images_dir)
    out = Path(out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)

    model = load_model(weights)
    images = sorted(str(p) for p in images_dir.glob("*.jpg"))
    if not images:
        raise FileNotFoundError(f"no images under {images_dir}")

    kept = 0
    results = model.predict(source=images, imgsz=imgsz, conf=conf, iou=0.6,
                            max_det=300, augment=tta, verbose=False, stream=True)
    for r in results:
        stem = stem_of(Path(r.path).name)
        lines = []
        b = r.boxes
        if b is not None and b.shape[0] > 0:
            xywhn = b.xywhn.cpu().numpy()
            cls = b.cls.cpu().numpy().astype(int)
            cf = b.conf.cpu().numpy()
            for c, s, box in zip(cls, cf, xywhn):
                thr = (per_class_conf or {}).get(int(c), conf)
                if s < thr:
                    continue
                lines.append(f"{int(c)} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}")
                kept += 1
        (out / "labels" / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        dst = out / "images" / f"{stem}.jpg"
        if not dst.exists():
            try:
                dst.symlink_to(Path(r.path).resolve())
            except OSError:
                import shutil

                shutil.copy2(r.path, dst)
    print(f"[pseudo_label] wrote pseudo-labels for {len(images)} images, kept {kept} boxes -> {out}")
    return kept


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate pseudo-labels (off by default; rules-risky)")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--images", required=True, help="ALLOWED unlabeled image dir (NOT the test set)")
    ap.add_argument("--out", default="data/yolo_pseudo")
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--i-confirm-rules-compliant", action="store_true",
                    help="required acknowledgement that this image pool is rules-allowed")
    args = ap.parse_args()

    if not args.i_confirm_rules_compliant:
        raise SystemExit(
            "Refusing to run: pseudo-labeling can be test-set leakage under the rules.\n"
            "Pass --i-confirm-rules-compliant ONLY if --images is an allowed external pool."
        )
    generate_pseudo_labels(args.weights, args.images, args.out, args.conf, args.imgsz, not args.no_tta)


if __name__ == "__main__":
    main()
