# scripts/evaluate_baselines_regression.py

import os
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from lib import (
    CrossSectionalDataset,
    LastDayAbsChangePred,
    MeanAbsChangePred,
    ZeroAbsChangePred,
    RandomAbsChangePred,
)


def _compute_logm_bucket_metrics(
    preds: np.ndarray,
    targets: np.ndarray,
    logm: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """
    Per-bucket regression metrics by log-moneyness.
    """
    bins = [-np.inf, -0.1, 0.1, np.inf]
    labels = ["ITM", "ATM", "OTM"]

    bucket_stats: Dict[str, Dict[str, float]] = {}

    for i in range(len(labels)):
        left, right = bins[i], bins[i + 1]
        mask = (logm >= left) & (logm < right)

        if mask.sum() == 0:
            continue

        b_preds = preds[mask]
        b_targets = targets[mask]
        residuals = b_preds - b_targets

        mse = float(np.mean((b_preds - b_targets) ** 2))
        mae = float(np.mean(np.abs(b_preds - b_targets)))
        rmse = float(np.sqrt(mse))

        bucket_stats[labels[i]] = {
            "count": int(mask.sum()),
            "mse": mse,
            "mae": mae,
            "rmse": rmse,
            "mean_pred": float(b_preds.mean()),
            "std_pred": float(b_preds.std()),
            "mean_target": float(b_targets.mean()),
            "std_target": float(b_targets.std()),
            "mean_resid": float(residuals.mean()),
            "std_resid": float(residuals.std()),
        }

    return bucket_stats


def _save_preds_vs_targets(
    preds: np.ndarray,
    targets: np.ndarray,
    output_path: str,
) -> None:
    plt.figure(figsize=(8, 8))
    plt.scatter(targets, preds, alpha=0.4, s=10)
    min_v = min(targets.min(), preds.min())
    max_v = max(targets.max(), preds.max())

    plt.plot([min_v, max_v], [min_v, max_v], "r--", linewidth=2, label="y = x")
    plt.xlabel("True Values (|ΔIV|)")
    plt.ylabel("Predicted Values (|ΔIV|)")
    plt.title("Predicted vs True Values")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def _save_residuals_histogram(
    preds: np.ndarray,
    targets: np.ndarray,
    output_path: str,
) -> None:
    residuals = preds - targets
    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=50, alpha=0.7)
    plt.title("Residuals Histogram (pred - true)")
    plt.xlabel("Residual (|ΔIV|)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def _save_prediction_distribution(
    preds: np.ndarray,
    targets: np.ndarray,
    output_path: str,
) -> None:
    plt.figure(figsize=(10, 6))
    plt.hist(targets, bins=50, alpha=0.5, label="True |ΔIV|", density=True)
    plt.hist(preds, bins=50, alpha=0.5, label="Predicted |ΔIV|", density=True)
    plt.title("Prediction vs True Distribution (|ΔIV|)")
    plt.xlabel("|ΔIV|")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def _save_detailed_metrics(
    metrics: Dict[str, Any],
    baseline_name: str,
    split_type: str,
    output_dir: str,
) -> None:
    preds = metrics["preds"]
    targets = metrics["targets"]
    residuals = preds - targets
    logm = metrics["log_moneyness"]

    filepath = os.path.join(output_dir, "detailed_metrics.txt")
    with open(filepath, "w") as f:
        f.write(f"{split_type.capitalize()} Regression Evaluation Metrics - {baseline_name}\n")
        f.write("=" * 50 + "\n\n")

        # Basic metrics
        f.write("Basic Metrics:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Baseline:                     {baseline_name}\n")
        f.write(f"MSE:       {metrics['mse']:.6f}\n")
        f.write(f"RMSE:      {metrics['rmse']:.6f}\n")
        f.write(f"MAE:       {metrics['mae']:.6f}\n\n")

        # Prediction statistics (eval space)
        f.write("Prediction Statistics:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Mean prediction:  {preds.mean():.6f}\n")
        f.write(f"Std prediction:   {preds.std():.6f}\n")
        f.write(f"Min prediction:   {preds.min():.6f}\n")
        f.write(f"Max prediction:   {preds.max():.6f}\n\n")

        # Target statistics (eval space)
        f.write("Target Statistics:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Mean target:      {targets.mean():.6f}\n")
        f.write(f"Std target:       {targets.std():.6f}\n")
        f.write(f"Min target:       {targets.min():.6f}\n")
        f.write(f"Max target:       {targets.max():.6f}\n\n")

        # Residual statistics
        f.write("Residual Statistics (pred - true):\n")
        f.write("-" * 30 + "\n")
        f.write(f"Mean residual:    {residuals.mean():.6f}\n")
        f.write(f"Std residual:     {residuals.std():.6f}\n")
        f.write(f"Min residual:     {residuals.min():.6f}\n")
        f.write(f"Max residual:     {residuals.max():.6f}\n")

        # Per-Log-Moneyness Bucket Metrics
        f.write("\nPer-Log-Moneyness Bucket Metrics:\n")
        f.write("-" * 30 + "\n")
        bucket_stats = _compute_logm_bucket_metrics(
            preds=preds,
            targets=targets,
            logm=logm,
        )
        for bucket_name, stats in bucket_stats.items():
            f.write(f"\nBucket: {bucket_name}\n")
            f.write(f"  Count:       {stats['count']}\n")
            f.write(f"  MSE:         {stats['mse']:.6f}\n")
            f.write(f"  RMSE:        {stats['rmse']:.6f}\n")
            f.write(f"  MAE:         {stats['mae']:.6f}\n")
            f.write(f"  Mean pred:   {stats['mean_pred']:.6f}\n")
            f.write(f"  Std pred:    {stats['std_pred']:.6f}\n")
            f.write(f"  Mean target: {stats['mean_target']:.6f}\n")
            f.write(f"  Std target:  {stats['std_target']:.6f}\n")
            f.write(f"  Mean resid:  {stats['mean_resid']:.6f}\n")
            f.write(f"  Std resid:   {stats['std_resid']:.6f}\n")


def evaluate_baseline(
    baseline,
    sequences: np.ndarray,
    targets: np.ndarray,
    logm: np.ndarray,
    output_dir: str,
    split_type: str,
) -> Dict[str, Any]:
    """
    Evaluate a single regression baseline and save plots/metrics.

    Args:
        baseline: object with `.name` and `.predict(sequence) -> float`
        sequences: array of shape (N, T, F)
        targets: array of shape (N,)  (|ΔIV|)
        logm: array of shape (N,)
        output_dir: directory where plots + metrics are saved
        split_type: "test" or "validation"
    """
    os.makedirs(output_dir, exist_ok=True)

    # ---------- PREDICTIONS ----------
    preds_list: List[float] = []
    for seq in sequences:
        preds_list.append(float(baseline.predict(seq)))
    preds = np.array(preds_list, dtype=np.float64)

    metrics: Dict[str, Any] = {
        "log_moneyness": logm,
    }

    # ---------- MAIN METRICS (EVAL SPACE: |ΔIV|) ----------
    preds_eval = preds
    targets_eval = targets

    mse = float(np.mean((preds_eval - targets_eval) ** 2))
    mae = float(np.mean(np.abs(preds_eval - targets_eval)))
    rmse = float(np.sqrt(mse))

    metrics["mse"] = mse
    metrics["mae"] = mae
    metrics["rmse"] = rmse
    metrics["preds"] = preds_eval
    metrics["targets"] = targets_eval

    print(f"\n{baseline.name} - {split_type.capitalize()} Regression Results:")
    print(f"MSE:  {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE:  {mae:.6f}")

    _save_preds_vs_targets(
        preds=metrics["preds"],
        targets=metrics["targets"],
        output_path=os.path.join(output_dir, "preds_vs_targets.png"),
    )
    _save_residuals_histogram(
        preds=metrics["preds"],
        targets=metrics["targets"],
        output_path=os.path.join(output_dir, "residuals_histogram.png"),
    )
    _save_prediction_distribution(
        preds=metrics["preds"],
        targets=metrics["targets"],
        output_path=os.path.join(output_dir, "prediction_distribution.png"),
    )

    # ---------- SAVE TEXT METRICS ----------
    _save_detailed_metrics(
        metrics=metrics,
        baseline_name=baseline.name,
        split_type=split_type,
        output_dir=output_dir,
    )

    return metrics


def main():
    data_dir = "data/dataset_entries/"
    output_root = "models_output/baselines_reg"

    target_key = "targets_mag"

    splits: List[Tuple[float, float]] = [
        (0.6, 0.2),
        (0.7, 0.15),
        (0.8, 0.1),
    ]

    # load feature columns for init of baseline predictors
    example_batch = np.load(os.path.join(data_dir, "batch_0000.npz"), allow_pickle=True)
    feature_columns = example_batch["feature_columns"]
    print("Loaded feature columns:", feature_columns)

    # initialize baselines
    baselines = [
        LastDayAbsChangePred(feature_columns=feature_columns),
        MeanAbsChangePred(feature_columns=feature_columns),
        ZeroAbsChangePred(),
        RandomAbsChangePred(low=0.0, high=0.1),
    ]

    # eval over each split
    for split_idx, (train_ratio, val_ratio) in enumerate(splits):
        print("\n" + "=" * 60)
        print(
            f"Split {split_idx}: "
            f"{train_ratio:.0%}/{val_ratio:.0%}/{1 - train_ratio - val_ratio:.0%}"
        )
        print("=" * 60)

        test_dataset = CrossSectionalDataset(
            dataset_entries_dir=data_dir,
            split_type="test",
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            target_key=target_key,
        )

        all_sequences: List[np.ndarray] = []
        all_targets: List[float] = []
        all_logm: List[float] = []

        for i in range(len(test_dataset)):
            # CrossSectionalDataset should return (sequences, targets, logm)
            sequences, targets, logm = test_dataset[i]
            # sequences: (C, T, F), targets/logm: (C,)
            for j in range(sequences.shape[0]):
                all_sequences.append(sequences[j].numpy())
                all_targets.append(float(targets[j].item()))
                all_logm.append(float(logm[j].item()))

        sequences_arr = np.array(all_sequences)
        targets_arr = np.array(all_targets)
        logm_arr = np.array(all_logm)

        print(f"Test set size: {len(targets_arr)} contracts")

        split_dir = Path(output_root) / f"split_{split_idx}"

        for baseline in baselines:
            baseline_dir = split_dir / baseline.name / "test"
            baseline_dir.mkdir(parents=True, exist_ok=True)

            _ = evaluate_baseline(
                baseline=baseline,
                sequences=sequences_arr,
                targets=targets_arr,
                logm=logm_arr,
                output_dir=str(baseline_dir),
                split_type="test",
            )

        val_dataset = CrossSectionalDataset(
            dataset_entries_dir=data_dir,
            split_type="val",
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            target_key=target_key,
        )

        val_sequences_list: List[np.ndarray] = []
        val_targets_list: List[float] = []
        val_logm_list: List[float] = []

        for i in range(len(val_dataset)):
            sequences, targets, logm = val_dataset[i]
            for j in range(sequences.shape[0]):
                val_sequences_list.append(sequences[j].numpy())
                val_targets_list.append(float(targets[j].item()))
                val_logm_list.append(float(logm[j].item()))

        val_sequences_arr = np.array(val_sequences_list)
        val_targets_arr = np.array(val_targets_list)
        val_logm_arr = np.array(val_logm_list)

        print(f"\nValidation set size: {len(val_targets_arr)} contracts")

        for baseline in baselines:
            baseline_dir = split_dir / baseline.name / "validation"
            baseline_dir.mkdir(parents=True, exist_ok=True)

            _ = evaluate_baseline(
                baseline=baseline,
                sequences=val_sequences_arr,
                targets=val_targets_arr,
                logm=val_logm_arr,
                output_dir=str(baseline_dir),
                split_type="validation",
            )

    print("\n" + "=" * 60)
    print("All regression baseline evaluations complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
