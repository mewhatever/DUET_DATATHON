"""
DUET CSE Carnival 2026 — Bangladesh Highway Vehicle Detection.

A modular object-detection pipeline (data prep -> EDA -> train -> validate ->
inference -> ensemble -> submission) built around the measured properties of the
provided CCTV dataset. See the README and the plan for the full strategy.
"""

__version__ = "1.0.0"

# Stable, dependency-light constants other modules import.
NUM_CLASSES = 13
IMG_W = 1280
IMG_H = 720
CAMERAS = ("CCTV01", "CCTV02", "CCTV07", "CCTV10")
SEED = 42
