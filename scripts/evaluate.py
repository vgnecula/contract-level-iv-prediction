# scripts/evaluate_generic.py

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from lib import (
    CrossSectionalDataset,
    BinaryClassificationEvaluator,
    RegressionEvaluator,
    build_tcta_binary_model,
    build_tcta_regression_model,
    build_tta_binary_model,
    build_tta_regression_model,
    build_lstm_binary_model,
    build_lstm_regression_model,
    build_fulltemp_tcta_binary_model,
    build_fulltemp_tcta_regression_model,
)
from lib.utils import load_config, set_seed, resolve_path, get_device


# ------------------------------------------------------------
# REGISTRIES
# ------------------------------------------------------------
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

EVALUATORS = {
    "binary": BinaryClassificationEvaluator,
    "regression": RegressionEvaluator,
}


def main(config_path_str: Optional[str] = None) -> None:
    project_root = Path(__file__).resolve().parents[1]

    # ---------------- CONFIG ----------------
    if config_path_str is None:
        # fallback default
        config_path = project_root / "configs" / "tcta_binary.yaml"
    else:
        config_path = resolve_path(project_root, config_path_str)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    cfg = load_config(config_path)

    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    data_cfg: Dict[str, Any] = cfg["data"]
    output_cfg: Dict[str, Any] = cfg["output"]
    task_cfg: Dict[str, Any] = cfg.get("task", {})

    dataset_entries_dir = resolve_path(project_root, data_cfg["dataset_entries_dir"])
    output_root = resolve_path(project_root, output_cfg["root_dir"])
    output_root.mkdir(parents=True, exist_ok=True)

    task_type = task_cfg["type"]               # "binary" or "regression"
    model_builder_key = task_cfg["model_builder"]  # e.g. "tcta_binary"
    target_key = task_cfg.get("target_key", "targets_dir")

    splits_cfg: List[Dict[str, Any]] = cfg["splits"]
    split_configs: List[Tuple[float, float]] = [
        (float(s["train_ratio"]), float(s["val_ratio"])) for s in splits_cfg
    ]

    model_cfg: Dict[str, Any] = cfg["model"]
    train_cfg: Dict[str, Any] = cfg["training"]
    batch_size: int = int(train_cfg["batch_size"])

    model_builder_fn = MODEL_BUILDERS[model_builder_key]
    evaluator_cls = EVALUATORS[task_type]

    def build_model_fn(input_size: int) -> torch.nn.Module:
        return model_builder_fn(input_size=input_size, **model_cfg) 

    device = get_device()

    print(f"Using config:        {config_path}")
    print(f"Dataset entries dir: {dataset_entries_dir}")
    print(f"Output root:         {output_root}")
    print(f"Task type:           {task_type}")
    print(f"Model builder:       {model_builder_key}")
    print(f"Target key:          {target_key}")
    print(f"Splits:              {split_configs}")

    # ---------------- LOOP OVER SPLITS ----------------
    for split_idx, (train_ratio, val_ratio) in enumerate(split_configs):
        split_output_dir = output_root / f"split_{split_idx}"
        best_model_path = split_output_dir / "best_model.pt"

        print(f"\n{'=' * 50}")
        print(f"[{task_type.upper()}] Evaluating Split {split_idx}")
        print(
            f"Train: {train_ratio:.0%}, Val: {val_ratio:.0%}, "
            f"Test: {1 - train_ratio - val_ratio:.0%}"
        )
        print(f"Split output dir: {split_output_dir}")
        print(f"{'=' * 50}\n")

        if not best_model_path.exists():
            print(f"[WARN] best_model.pt not found for split {split_idx} at {best_model_path}")
            print("       Skipping this split.")
            continue

        # datasets
        val_dataset = CrossSectionalDataset(
            dataset_entries_dir=str(dataset_entries_dir),
            split_type="val",
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            target_key=target_key,
        )
        test_dataset = CrossSectionalDataset(
            dataset_entries_dir=str(dataset_entries_dir),
            split_type="test",
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            target_key=target_key,
        )

        if len(val_dataset) == 0:
            raise RuntimeError(f"Validation dataset is empty for split {split_idx}.")

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        sample_sequences, _, _ = next(iter(val_loader))
        input_size = sample_sequences.shape[-1]

        model = build_model_fn(input_size=input_size).to(device)
        state_dict = torch.load(best_model_path, map_location=device)
        model.load_state_dict(state_dict)

        evaluator = evaluator_cls(model, device)

        print("\nEvaluating validation set...")
        if task_type == "binary":
            evaluator.evaluate_and_save(
                base_output_dir=str(split_output_dir),
                data_loader=val_loader,
                split_type="validation",
                threshold=0.5,
            )
        else:
            evaluator.evaluate_and_save(
                base_output_dir=str(split_output_dir),
                data_loader=val_loader,
                split_type="validation",
            )

        print("\nEvaluating test set...")
        if task_type == "binary":
            evaluator.evaluate_and_save(
                base_output_dir=str(split_output_dir),
                data_loader=test_loader,
                split_type="test",
                threshold=0.5,
            )
        else:
            evaluator.evaluate_and_save(
                base_output_dir=str(split_output_dir),
                data_loader=test_loader,
                split_type="test",
            )

        print(f"\n[{task_type.upper()}] Finished evaluation for split {split_idx}.")

    print("\nAll done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generic evaluator for binary/regression models.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config (relative to project root or absolute).",
    )
    args = parser.parse_args()
    main(args.config)
