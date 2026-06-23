# DUET CSE Carnival 2026 — Bangladesh Highway Vehicle Detection

End-to-end object-detection pipeline for detecting **13 vehicle classes** in Bangladeshi highway
CCTV imagery, scored on **mAP@0.5**. Built around the *measured* properties of the provided dataset
(not generic assumptions) — see [`/Users/mohaiminul/.claude/plans/master-prompt-typed-peach.md`](.) for the full strategy write-up.

## TL;DR strategy
- **Model:** Ultralytics **YOLO11-m @ 1280px** (native CCTV resolution). Best mAP@0.5 per GPU-hour on
  only ~810 images; COCO-pretrained, strong small-object support via mosaic/copy-paste and an optional **P2 head**.
- **Validation:** *camera-stratified frame-phase split*. The test set is **interleaved frames of the same
  4 cameras** as train, so this split mimics the leaderboard far better than random or group-by-camera. Always
  report **per-camera** mAP (CCTV10 is over-weighted in the test set).
- **Inference:** low conf threshold (mAP rewards ranked recall) + NMS/per-class threshold tuning on the
  val split + optional TTA and **WBF** ensembling.
- **Format gotchas handled:** submission `image_id` uses **`.txt`** (images are `.jpg`); confidence mirrors the
  sample's **0–100** scale; **all 327** test images are always emitted (empty string if no detections).

## Dataset facts (from `src/eda.py`)
| | |
|---|---|
| Train images / objects | 810 / 10,475 (mean 12.9 boxes/img, max 38) |
| Test images | 327 |
| Image size | 1280×720 (train == test) |
| Cameras (scenes) | 4 — CCTV01 (74% of objects), CCTV02, CCTV07, CCTV10 |
| Small objects | 74.8% of boxes < 1% of image area |
| Class imbalance | ~30× (cls 3 = 2480 instances, cls 8 = 83) |

## Project layout
```
configs/        data.yaml (generated), train_baseline.yaml, train_p2.yaml, inference.yaml
src/            utils, dataset, prepare_data, eda, augmentations, model,
                train, validate, inference, ensemble, pseudo_label, submission
notebooks/      kaggle_runner.ipynb  (end-to-end on Kaggle GPU)
checkpoints/    trained weights (gitignored)
outputs/        eda plots, prediction caches, submissions (gitignored)
data/           train/ (images + train.csv), test/ (images + sample_submission.csv)
```

## Quickstart

### 0) Install
```bash
pip install -r requirements.txt          # On Kaggle, ultralytics/torch are preinstalled
```

### 1) Prepare data (CSV → YOLO labels + folds + data.yaml)
```bash
python -m src.prepare_data --data-root data --out data/yolo --folds 5 --val-fold 0
# writes data/yolo/{images,labels}/, fold image-lists, and configs/data.yaml
```

### 2) (optional) EDA report + plots
```bash
python -m src.eda --data-root data --out outputs/eda
```

### 3) Train the baseline (Kaggle GPU)
```bash
python -m src.train --config configs/train_baseline.yaml --data configs/data.yaml \
                    --name e00_yolo11m_1280 --device 0
# Multi-GPU (Kaggle T4 x2):  --device 0,1
# Train all 5 folds:         --folds 5
```

### 4) Validate + tune thresholds on the test-mimicking val split
```bash
python -m src.validate --weights checkpoints/e00_yolo11m_1280/weights/best.pt \
                       --data configs/data.yaml --tune --save configs/inference.yaml
```

### 5) Inference on the 327 test images
```bash
python -m src.inference --weights checkpoints/e00_yolo11m_1280/weights/best.pt \
                        --data-root data --infer-config configs/inference.yaml \
                        --out outputs/predictions/e00.pkl --tta
```

### 6) Build the submission CSV (exact competition format)
```bash
python -m src.submission --preds outputs/predictions/e00.pkl \
                         --infer-config configs/inference.yaml \
                         --test-images data/test/images \
                         --out outputs/submissions/submission_e00.csv
```
> The complete test id list (327) comes from `--test-images`, because
> `sample_submission.csv` is only a **truncated 10-row example**. Every test image is
> emitted (empty `PredictionString` if it has no detections).

### (scale-up) 5-fold + WBF ensemble
```bash
python -m src.ensemble --preds outputs/predictions/fold0.pkl outputs/predictions/fold1.pkl ... \
                       --out outputs/predictions/wbf.pkl --iou 0.55 --skip 0.001
python -m src.submission --preds outputs/predictions/wbf.pkl ...
```

## Reproducibility
- Global seed = 42 (`src/utils.set_seed`), deterministic Ultralytics flags.
- Every result goes in [`experiments.md`](experiments.md) with global + per-camera mAP@0.5.

## Class names
The exact id→name mapping (Bus, Truck, Motorcycle, CNG, Rickshaw, Easy-bike, Leguna, …) was **not provided**
and does not affect mAP. Placeholders `class_0…class_12` are used in `configs/data.yaml`; edit the `names:`
list there if you have the official mapping.

## Competition-rules compliance
No manual labeling of test images, no test labels, no hand-correction. `ground_truth_sample.csv` (1 image) is
used **only** to unit-test the local metric, never for training. Pseudo-labeling (`src/pseudo_label.py`) is
disabled by default and flagged as rules-risky (pseudo-labeling the test set may count as test-set leakage).
