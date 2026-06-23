# Experiment Tracking — Bangladesh Highway Vehicle Detection

> Every decision must be tied to a **measured** delta on the test-mimicking validation split
> (camera-stratified frame-phase, see `src/dataset.py`). Always log **global + per-camera** mAP@0.5.
> Seed is fixed (42) for reproducibility. LB = public leaderboard score after submission.

## How to read this table
- **Val mAP@0.5** is computed with our own VOC-style metric in `src/validate.py` (unit-tested against
  `data/test/ground_truth_sample.csv`) so it matches the grader's definition.
- **Per-cam** = `CCTV01 / CCTV02 / CCTV07 / CCTV10` mAP@0.5. CCTV10 is over-weighted in the test set,
  so weight your decisions toward the under-trained cameras.
- Bump one variable at a time. Cheapest high-expected-gain experiments first (rows are ordered by priority).

| Exp | Model | imgsz | Key augs | Epochs | Infer (conf/iou/TTA) | Val mAP@0.5 | Per-cam (01/02/07/10) | LB | Notes |
|-----|-------|-------|----------|--------|----------------------|-------------|-----------------------|----|-------|
| E00 | yolo11m | 1280 | mosaic1.0 cp0.3 mixup0.1 erasing0.4 | 120 | 0.001 / 0.6 / off | _TBD_ | _/_/_/_  | _ | Baseline (default config) |
| E01 | yolo11m | 1280 | + tuned per-class conf | 120 | tuned / tuned / off | _TBD_ | _/_/_/_  | _ | Threshold tuning only (no retrain) |
| E02 | yolo11m | 1280 | E01 | 120 | tuned / tuned / **on** | _TBD_ | _/_/_/_  | _ | + TTA at inference |
| E03 | yolo11m-**p2** | 1280 | E00 augs | 120 | tuned / tuned / on | _TBD_ | _/_/_/_  | _ | P2 head (small-object lever) |
| E04 | yolo11m | **1536** | E00 augs | 120 | tuned / tuned / on | _TBD_ | _/_/_/_  | _ | Higher res |
| E05 | yolo11**l** | 1280 | E00 augs | 120 | tuned / tuned / on | _TBD_ | _/_/_/_  | _ | Bigger backbone (watch overfit) |
| E06 | yolo11m | 1280 | cp **0.5** | 120 | tuned / tuned / on | _TBD_ | _/_/_/_  | _ | Stronger copy-paste for rare classes |
| E07 | 5×yolo11m (folds) | 1280 | E00 augs | 120 | WBF | _TBD_ | _/_/_/_  | _ | 5-fold + WBF ensemble |
| E08 | + rtdetr-l | 1280 | — | 150 | WBF | _TBD_ | _/_/_/_  | _ | Add transformer member to WBF |

## Decisions log
- _yyyy-mm-dd_ — Baseline E00 trained; record val/LB here and pick next experiment by largest expected gain/hour.

## Rare-class watch (instances in train)
`cls8=83, cls7=127, cls6=164, cls11=239, cls9=265, cls4=270, cls12=372, cls5=455`.
Track these AP columns separately when copy-paste / per-class thresholds change — they are noisy, do not over-read single swings.
