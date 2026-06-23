"""
Extra Albumentations pipeline (weather / blur / sensor noise) hooked into Ultralytics.

Why these, for THIS dataset
---------------------------
The 4 cameras span day (CCTV01/02), dawn (CCTV07, 06:00) and dusk (CCTV10, 17:00),
and the brief calls out rain / fog / low-light / low-resolution CCTV. These photometric
and degradation transforms make the detector robust to those conditions. They are
*photometric only* -> bounding boxes pass through unchanged.

Usage
-----
Set ``augment.use_albumentations: true`` in the train config; ``src/train.py`` then
calls :func:`apply_albumentations_patch` before training. We override the ``transform``
attribute of Ultralytics' built-in ``Albumentations`` block (keeping its ``__call__`` and
its expected ``format="yolo", label_fields=["class_labels"]`` contract), so it works
across Ultralytics versions without reimplementing the call path.

We probe multiple constructor signatures per transform because Albumentations renamed
several arguments around v1.4 (e.g. ``ImageCompression``, ``RandomFog``).
"""

from __future__ import annotations

from typing import List


def _first_ok(*candidates):
    """Return the first transform that constructs without error (handles A version drift)."""
    last = None
    for factory in candidates:
        try:
            return factory()
        except Exception as e:  # signature changed between versions
            last = e
    print(f"[augmentations] skipped a transform (no compatible signature): {last}")
    return None


def build_weather_transforms(p_scale: float = 1.0) -> List:
    """Build the list of Albumentations transforms (mild probabilities by default)."""
    import albumentations as A

    p = lambda x: min(max(x * p_scale, 0.0), 1.0)
    T = [
        # --- lighting: day/dawn/dusk + night robustness ---
        _first_ok(lambda: A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=p(0.25))),
        _first_ok(lambda: A.RandomGamma(gamma_limit=(70, 140), p=p(0.15))),
        _first_ok(lambda: A.CLAHE(clip_limit=2.0, p=p(0.10))),
        # --- weather ---
        _first_ok(
            lambda: A.RandomFog(fog_coef_range=(0.1, 0.3), alpha_coef=0.08, p=p(0.10)),  # A>=1.4
            lambda: A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, alpha_coef=0.08, p=p(0.10)),
        ),
        _first_ok(
            lambda: A.RandomRain(brightness_coefficient=0.9, drop_width=1, blur_value=3, p=p(0.08)),
        ),
        _first_ok(lambda: A.RandomShadow(p=p(0.10))),
        # --- motion / focus blur (moving vehicles, soft CCTV optics) ---
        _first_ok(lambda: A.MotionBlur(blur_limit=5, p=p(0.10))),
        _first_ok(lambda: A.MedianBlur(blur_limit=3, p=p(0.05))),
        # --- sensor noise + compression (cheap CCTV) ---
        _first_ok(
            lambda: A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=p(0.10)),
        ),
        _first_ok(
            lambda: A.GaussNoise(std_range=(0.02, 0.08), p=p(0.10)),                      # A>=1.4
            lambda: A.GaussNoise(var_limit=(10.0, 50.0), p=p(0.10)),
        ),
        _first_ok(
            lambda: A.ImageCompression(quality_range=(40, 80), p=p(0.15)),                # A>=1.4
            lambda: A.ImageCompression(quality_lower=40, quality_upper=80, p=p(0.15)),
        ),
    ]
    return [t for t in T if t is not None]


def apply_albumentations_patch(p: float = 1.0, p_scale: float = 1.0) -> bool:
    """
    Monkeypatch Ultralytics' Albumentations block to use our weather pipeline.

    Returns True on success. Safe to call once before ``model.train(...)``.
    """
    try:
        import albumentations as A
        import ultralytics.data.augment as aug
    except Exception as e:
        print(f"[augmentations] Albumentations/Ultralytics unavailable -> patch skipped ({e})")
        return False

    transforms = build_weather_transforms(p_scale=p_scale)
    if not transforms:
        print("[augmentations] no compatible transforms built -> patch skipped")
        return False

    orig_init = aug.Albumentations.__init__

    def patched_init(self, p_inner: float = 1.0):
        # Let Ultralytics set up its own attributes (contains_spatial, etc.) first ...
        try:
            orig_init(self, p_inner)
        except Exception:
            self.p = p_inner
            self.transform = None
            self.contains_spatial = False
        # ... then override the transform with ours.
        self.transform = A.Compose(
            transforms,
            bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
        )
        self.p = p
        print(f"[augmentations] injected {len(transforms)} weather/blur/noise transforms (p={p})")

    aug.Albumentations.__init__ = patched_init
    return True
