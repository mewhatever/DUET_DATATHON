"""
Training entry point.

    # baseline single fold (default config -> YOLO11-m @1280)
    python -m src.train --config configs/train_baseline.yaml --data configs/data.yaml \
                        --name e00_yolo11m_1280 --device 0

    # multi-GPU (Kaggle T4 x2)
    python -m src.train ... --device 0,1

    # train all 5 phase-folds for the WBF ensemble
    python -m src.train ... --folds 5 --name e07_yolo11m

    # quick CPU/MPS smoke test of the whole chain
    python -m src.train ... --device cpu --imgsz 320 --epochs 1 --batch 2 --name smoke

Weights land in  checkpoints/<name>[/_fold{k}]/weights/best.pt .
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional

from .model import build_model
from .utils import PROJECT_ROOT, load_yaml, save_yaml, set_seed

# Keys we never forward to Ultralytics' model.train() (handled by us instead).
_NON_ULTRA_KEYS = {"arch", "weights", "p2"}


def _fold_data_yaml(base_data_yaml: str, fold: int, out_dir: Path) -> str:
    """Derive a per-fold data.yaml (train_fold{k}.txt / val_fold{k}.txt) from the base one."""
    base = load_yaml(base_data_yaml)
    cfg = dict(base)
    cfg["train"] = f"train_fold{fold}.txt"
    cfg["val"] = f"val_fold{fold}.txt"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"data_fold{fold}.yaml"
    save_yaml(cfg, path)
    return str(path)


def _train_fold_subprocess(args, data_k: str, name_k: str) -> str:
    """
    Train ONE fold in a fresh `python -m src.train --folds 1` subprocess.

    Why a subprocess and not just calling train_one() in a loop: with multi-GPU
    (`--device 0,1`) Ultralytics launches a torch.distributed (DDP) process group
    for each `model.train()` call. Doing that 5 times inside the SAME parent
    process leaves stale process groups / NCCL state and the 2nd+ fold hangs or
    crashes. A fresh process per fold sets DDP up and tears it down cleanly, so
    `--folds 5 --device 0,1` is reliable. (Single-GPU also benefits: no CUDA
    fragmentation carried across folds.)
    """
    cmd = [sys.executable, "-m", "src.train",
           "--config", args.config, "--data", data_k,
           "--name", name_k, "--project", args.project, "--folds", "1"]
    if args.device is not None:
        cmd += ["--device", args.device]
    for flag, val in (("--epochs", args.epochs), ("--imgsz", args.imgsz), ("--batch", args.batch)):
        if val is not None:
            cmd += [flag, str(val)]
    print(f"[train] fold {name_k} -> subprocess: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))

    project = str(PROJECT_ROOT / args.project) if not Path(args.project).is_absolute() else args.project
    best = Path(project) / name_k / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError(f"fold {name_k} finished but {best} is missing")
    return str(best)


def train_one(
    config: str,
    data: str,
    name: str,
    device: Optional[str] = None,
    project: str = "checkpoints",
    overrides: Optional[dict] = None,
):
    """Train a single model and return the path to best.pt."""
    cfg = load_yaml(config)
    mcfg = cfg.get("model", {})
    tcfg = dict(cfg.get("train", {}))
    acfg = cfg.get("augment", {})
    if overrides:
        tcfg.update({k: v for k, v in overrides.items() if v is not None})

    set_seed(int(tcfg.get("seed", 42)))

    # Optional weather/blur Albumentations pipeline.
    if acfg.get("use_albumentations", False):
        from .augmentations import apply_albumentations_patch

        apply_albumentations_patch(p=1.0, p_scale=float(acfg.get("p_scale", 1.0)))

    model = build_model(arch=mcfg.get("arch", "yolo11m"),
                        weights=mcfg.get("weights"),
                        p2=bool(mcfg.get("p2", False)))

    train_kwargs = {k: v for k, v in tcfg.items() if k not in _NON_ULTRA_KEYS}
    # Force an ABSOLUTE project dir. Some Ultralytics versions nest a *relative*
    # project under runs/detect/<project>, which hides best.pt from the paths the
    # pipeline expects (checkpoints/<name>/weights/best.pt).
    project = str(PROJECT_ROOT / project) if not Path(project).is_absolute() else project
    train_kwargs.update(data=data, project=project, name=name, exist_ok=True)
    if device is not None:
        train_kwargs["device"] = device

    print(f"[train] model={mcfg.get('arch')} p2={mcfg.get('p2', False)} "
          f"imgsz={train_kwargs.get('imgsz')} epochs={train_kwargs.get('epochs')} "
          f"batch={train_kwargs.get('batch')} device={device} name={name}")
    results = model.train(**train_kwargs)

    best = Path(getattr(results, "save_dir", Path(project) / name)) / "weights" / "best.pt"
    print(f"[train] DONE -> {best}  (metrics dir: {getattr(results, 'save_dir', '?')})")
    return str(best)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the detector (single or all folds)")
    ap.add_argument("--config", default="configs/train_baseline.yaml")
    ap.add_argument("--data", default="configs/data.yaml")
    ap.add_argument("--name", default="e00_yolo11m_1280")
    ap.add_argument("--project", default="checkpoints")
    ap.add_argument("--device", default=None, help="'0' | '0,1' | 'cpu' | 'mps' (default: auto)")
    ap.add_argument("--folds", type=int, default=1, help="train K phase-folds (default 1 = use --data as is)")
    # common quick overrides
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    args = ap.parse_args()

    overrides = {"epochs": args.epochs, "imgsz": args.imgsz, "batch": args.batch}

    if args.folds <= 1:
        train_one(args.config, args.data, args.name, args.device, args.project, overrides)
        return

    # Multi-fold: train each fold in its OWN process so multi-GPU DDP is set up and
    # torn down cleanly per fold (see _train_fold_subprocess for the why).
    fold_dir = PROJECT_ROOT / "configs" / "_folds"
    best_paths = []
    for k in range(args.folds):
        data_k = _fold_data_yaml(args.data, k, fold_dir)
        best = _train_fold_subprocess(args, data_k, f"{args.name}_fold{k}")
        best_paths.append(best)
    print("[train] all folds done:")
    for p in best_paths:
        print("   ", p)


if __name__ == "__main__":
    main()
