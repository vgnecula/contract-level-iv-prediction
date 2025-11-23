#!/usr/bin/env python
"""
Generic training script for cross-sectional options data.

Supports:
  - Binary tasks (e.g. TCTA + BinaryHead)
  - Regression tasks (e.g. TCTA + RegressionHead)

The behavior is controlled entirely via the YAML config, e.g.:

task:
  type: binary            # or "regression"
  model_builder: tcta_binary   # or "tcta_regression"
  target_key: targets_dir      # or another npz key

Usage:
    python scripts/train_generic.py --config configs/tcta_binary.yaml
"""

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import torch

from lib import (
    BinaryTrainer,
    RegressionTrainer,
    build_tcta_binary_model,
    build_tcta_regression_model,
    build_tta_binary_model,
    build_tta_regression_model,
    build_lstm_binary_model,
    build_lstm_regression_model,
    build_fulltemp_tcta_binary_model,
    build_fulltemp_tcta_regression_model,
)
from lib.utils import load_config, set_seed, resolve_path


# --------------------------------------------------------------------- #
# REGISTRIES
# --------------------------------------------------------------------- #

# map string from YAML -> model builder functions
MODEL_BUILDERS = {
    "tcta_binary": build_tcta_binary_model,
    "tcta_regression": build_tcta_regression_model,
    "tta_binary": build_tta_binary_model,
    "tta_regression": build_tta_regression_model,
    "lstm_binary": build_lstm_binary_model,
    "lstm_regression": build_lstm_regression_model,
    "fulltemp_tcta_binary": build_fulltemp_tcta_binary_model,
    "fulltemp_tcta_regression": build_fulltemp_tcta_regression_model,
}

# map task type -> trainer class
TRAINERS = {
    "binary": BinaryTrainer,
    "regression": RegressionTrainer,
}


# --------------------------------------------------------------------- #
# MAIN
# --------------------------------------------------------------------- #
def main(config_path_str: Optional[str] = None) -> None:
    
    project_root = Path(__file__).resolve().parents[1]
    config_path = resolve_path(project_root, config_path_str)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    cfg = load_config(config_path)

    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    # ---------------------------- TASK -------------------------------- #
    task_cfg: Dict[str, Any] = cfg.get("task", {})
    task_type: str = task_cfg.get("type", "binary")  # default to binary
    model_builder_key: str = task_cfg.get("model_builder", "tcta_binary")
    target_key: str = task_cfg.get("target_key", "targets_dir")

    if task_type not in TRAINERS:
        raise ValueError(
            f"Unknown task.type='{task_type}'. "
            f"Expected one of: {list(TRAINERS.keys())}"
        )

    if model_builder_key not in MODEL_BUILDERS:
        raise ValueError(
            f"Unknown task.model_builder='{model_builder_key}'. "
            f"Expected one of: {list(MODEL_BUILDERS.keys())}"
        )

    TrainerCls = TRAINERS[task_type]
    model_builder_fn = MODEL_BUILDERS[model_builder_key]

    # ---------------------------- PATHS ------------------------------- #
    data_cfg: Dict[str, Any] = cfg["data"]
    output_cfg: Dict[str, Any] = cfg["output"]

    dataset_entries_dir = resolve_path(project_root, data_cfg["dataset_entries_dir"])
    output_root = resolve_path(project_root, output_cfg["root_dir"])
    output_root.mkdir(parents=True, exist_ok=True)

    # ------------------------- SPLIT CONFIGS --------------------------- #
    splits_cfg: List[Dict[str, Any]] = cfg["splits"]
    split_configs: List[Tuple[float, float]] = [
        (float(s["train_ratio"]), float(s["val_ratio"])) for s in splits_cfg
    ]

    # ---------------------- MODEL / TRAIN PARAMS ----------------------- #
    model_cfg: Dict[str, Any] = cfg["model"]
    train_cfg: Dict[str, Any] = cfg["training"]

    def build_model_fn(input_size: int) -> torch.nn.Module:
        return model_builder_fn(input_size=input_size, **model_cfg) 

    print(f"Using config:          {config_path}")
    print(f"Task type:             {task_type}")
    print(f"Model builder key:     {model_builder_key}")
    print(f"Target key:            {target_key}")
    print(f"Dataset entries dir:   {dataset_entries_dir}")
    print(f"Output root:           {output_root}")
    print(f"Splits:                {split_configs}")

    all_training_summaries: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ #
    # CROSS-VALIDATION LOOP (TRAINING ONLY)
    # ------------------------------------------------------------------ #
    for split_idx, (train_ratio, val_ratio) in enumerate(split_configs):
        split_output_dir = output_root / f"split_{split_idx}"
        split_output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'=' * 60}")
        print(f"[{task_type.upper()}] Training Split Configuration {split_idx}")
        print(
            f"Train: {train_ratio:.0%}, Val: {val_ratio:.0%}, "
            f"Test: {1 - train_ratio - val_ratio:.0%}"
        )
        print(f"Split output dir: {split_output_dir}")
        print(f"{'=' * 60}\n")

        # ------------------------- TRAINER ---------------------------- #
        trainer = TrainerCls(
            dataset_entries_dir=str(dataset_entries_dir),
            output_dir=str(split_output_dir),
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            target_key=target_key,
            build_model_fn=build_model_fn,
            model_kwargs={},  # model_cfg captured via closure
        )

        # --------------------------- TRAIN ---------------------------- #
        split_training_summary = trainer.train(**train_cfg)
        trainer.save_training_summary(split_idx, split_training_summary, str(split_output_dir))
        all_training_summaries.append(split_training_summary)

    # ------------------------------------------------------------------ #
    # GLOBAL SUMMARY OVER SPLITS
    # ------------------------------------------------------------------ #
    summary_path = output_root / "training_cross_validation_summary.txt"
    with summary_path.open("w") as f:
        f.write(f"{task_type.capitalize()} Training Cross Validation Summary\n")
        f.write("=" * 50 + "\n\n")
        for split_idx, training_summary in enumerate(all_training_summaries):
            f.write(f"\nSplit {split_idx}:\n")
            f.write("-" * 30 + "\n")
            for key, value in training_summary.items():
                f.write(f"  {key}: {value}\n")

    print(f"\n[{task_type.upper()}] Done. Global training summary written to: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generic trainer for TCTA-style models.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Path to YAML config (relative to project root or absolute). "
            "Default: configs/tcta_binary.yaml"
        ),
    )
    args = parser.parse_args()
    main(args.config)
