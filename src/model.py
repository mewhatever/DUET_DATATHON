"""
Model factory.

Wraps Ultralytics so train / validate / inference all build models the same way:

    build_model("yolo11m", "yolo11m.pt")          # baseline
    build_model("yolo11m", "yolo11m.pt", p2=True) # + stride-4 P2 small-object head
    build_model("rtdetr-l", "rtdetr-l.pt")        # transformer member for the ensemble

P2 head
-------
The P2 variant adds a detection layer at stride 4 (vs the default min stride 8),
which is the single biggest lever for the 74.8%% of boxes that are tiny. We load the
``*-p2`` architecture yaml and transfer compatible COCO-pretrained weights with
``.load()``. Ultralytics parses the scale letter (the ``m`` in ``yolo11m-p2.yaml``)
from the filename. If a model family ships no p2 cfg, we fall back to the YOLOv8 p2
cfg at the same scale.
"""

from __future__ import annotations

import re
from typing import Optional


def is_rtdetr(arch: str) -> bool:
    return "rtdetr" in arch.lower() or "r-detr" in arch.lower()


def _scale_letter(arch: str) -> str:
    """Extract the size letter (n/s/m/l/x) from e.g. 'yolo11m' or 'yolov8l'."""
    m = re.search(r"(?:yolov?\d+)([nsmlx])$", arch.lower())
    return m.group(1) if m else "m"


def build_model(arch: str = "yolo11m", weights: Optional[str] = None, p2: bool = False):
    """
    Construct an (untrained-head) Ultralytics model ready for ``.train(...)``.

    arch    : 'yolo11m' | 'yolov8m' | 'yolo11l' | 'rtdetr-l' | 'rtdetr-x' | ...
    weights : COCO-pretrained checkpoint to start from (auto-downloaded by Ultralytics).
    p2      : attach the stride-4 P2 head (YOLO only).
    """
    if is_rtdetr(arch):
        from ultralytics import RTDETR

        return RTDETR(weights or f"{arch}.pt")

    from ultralytics import YOLO

    if not p2:
        # Standard path: load the pretrained .pt directly (carries architecture + weights).
        return YOLO(weights or f"{arch}.pt")

    # --- P2 path: load the architecture cfg, then transfer pretrained weights ---
    scale = _scale_letter(arch)
    candidates = [f"{arch}-p2.yaml", f"yolov8{scale}-p2.yaml", "yolov8-p2.yaml"]
    model = None
    for cfg in candidates:
        try:
            model = YOLO(cfg)
            print(f"[model] built P2 architecture from '{cfg}'")
            break
        except Exception as e:
            print(f"[model] P2 cfg '{cfg}' unavailable ({e})")
    if model is None:
        raise RuntimeError("No P2 architecture cfg could be loaded; train without --p2.")

    if weights:
        try:
            model.load(weights)  # transfer matching COCO-pretrained layers
            print(f"[model] transferred pretrained weights from '{weights}' into the P2 model")
        except Exception as e:
            print(f"[model] WARNING: could not transfer weights ({e}); training P2 head from scratch")
    return model


def load_model(weights: str):
    """
    Load a *trained* checkpoint for validation / inference, auto-detecting YOLO vs RT-DETR.
    """
    from ultralytics import YOLO

    try:
        return YOLO(weights)
    except Exception:
        from ultralytics import RTDETR

        return RTDETR(weights)
