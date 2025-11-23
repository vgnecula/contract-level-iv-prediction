# scripts/evaluate_baselines_binary.py

import os
import sys
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from lib import (
    CrossSectionalDataset,
    MajorityTrendPred,
    LastDayPred,
    BiasedRandomPred,
    RandomPred,
)


# --------------------------------------------------------------------- #
# PER-BUCKET METRICS (by log-moneyness)
# --------------------------------------------------------------------- #
def _compute_logm_bucket_metrics(
    predictions: np.ndarray,
    true_labels: np.ndarray,
    logm: np.ndarray,
):
    """
    Compute per-bucket metrics by log-moneyness using HARD LABELS only.
    """
    bins = [-np.inf, -0.1, 0.1, np.inf]
    labels = ["ITM", "ATM", "OTM"]

    bucket_stats = {}

    for i in range(len(labels)):
        left, right = bins[i], bins[i + 1]
        mask = (logm >= left) & (logm < right)

        if mask.sum() == 0:
            continue

        bucket_targets = true_labels[mask]
        bucket_preds = predictions[mask]

        n_total = int(mask.sum())
        n_pos = int((bucket_targets == 1).sum())
        n_neg = int((bucket_targets == 0).sum())
        pos_rate = n_pos / n_total if n_total > 0 else 0.0

        # Mean/std of predictions (0/1) -> predicted positive rate in that bucket
        mean_pred = float(bucket_preds.mean())
        std_pred = float(bucket_preds.std())

        acc = accuracy_score(bucket_targets, bucket_preds)
        report = classification_report(
            bucket_targets,
            bucket_preds,
            output_dict=True,
            zero_division=0,
        )
        macro_f1 = report["macro avg"]["f1-score"]

        bucket_stats[labels[i]] = {
            "count": n_total,
            "n_pos": n_pos,
            "n_neg": n_neg,
            "pos_rate": pos_rate,
            "mean_pred": mean_pred,
            "std_pred": std_pred,
            "accuracy": float(acc),
            "macro_f1": float(macro_f1),
        }

    return bucket_stats


# --------------------------------------------------------------------- #
# BASELINE EVALUATION
# --------------------------------------------------------------------- #
def evaluate_baseline(baseline, sequences, targets, logm, output_dir):
    """
    Evaluate a single baseline and save results (with logm buckets).

    We treat the baseline as returning:
        pred: hard 0/1 label
    All metrics are computed from the hard labels.
    """

    # Get predictions
    predictions = []

    for seq in sequences:
        pred = baseline.predict(seq)
        predictions.append(pred)

    predictions = np.array(predictions)

    # Calculate core metrics from HARD LABELS
    accuracy = accuracy_score(targets, predictions)
    cm = confusion_matrix(targets, predictions)
    class_report = classification_report(
        targets,
        predictions,
        output_dict=True,
        zero_division=0,
    )

    # Print results
    print(f"\n{baseline.name} Results:")
    print(f"Accuracy: {accuracy:.4f}")

    # Save confusion matrix plot
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title(f"Confusion Matrix - {baseline.name}")
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["0", "1"])
    plt.yticks(tick_marks, ["0", "1"])

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(cm[i, j], "d"),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()


    # Save detailed metrics (matching evaluator style, plus buckets)
    with open(os.path.join(output_dir, "detailed_metrics.txt"), "w") as f:
        f.write(f"Test Evaluation Metrics - {baseline.name}\n")
        f.write("=" * 50 + "\n\n")

        # Basic metrics
        f.write("Basic Metrics:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Accuracy: {accuracy:.4f}\n\n")

        # Class distribution
        f.write("Class Distribution:\n")
        f.write("-" * 30 + "\n")
        f.write("True distribution:\n")
        for label in [0, 1]:
            count = (targets == label).sum()
            f.write(f"  Class {label}: {count} ({count / len(targets):.2%})\n")

        f.write("\nPredicted distribution:\n")
        for label in [0, 1]:
            count = (predictions == label).sum()
            f.write(f"  Class {label}: {count} ({count / len(predictions):.2%})\n")

        # Global classification report (hard labels)
        f.write("\nDetailed Classification Report (GLOBAL):\n")
        f.write("-" * 30 + "\n")
        f.write(
            classification_report(
                targets,
                predictions,
                digits=4,
                zero_division=0,
            )
        )

        f.write("\nPer-Log-Moneyness Bucket Metrics (scalar):\n")
        f.write("-" * 30 + "\n")
        bucket_stats = _compute_logm_bucket_metrics(
            predictions=predictions,
            true_labels=targets,
            logm=logm,
        )

        for bucket_name, stats in bucket_stats.items():
            f.write(f"\nBucket: {bucket_name}\n")
            f.write(f"  Count:       {stats['count']}\n")
            f.write(f"  n_pos:       {stats['n_pos']}\n")
            f.write(f"  n_neg:       {stats['n_neg']}\n")
            f.write(f"  pos_rate:    {stats['pos_rate']:.4f}\n")
            f.write(f"  mean_pred:   {stats['mean_pred']:.4f}\n")
            f.write(f"  std_pred:    {stats['std_pred']:.4f}\n")
            f.write(f"  Accuracy:    {stats['accuracy']:.4f}\n")
            f.write(f"  Macro F1:    {stats['macro_f1']:.4f}\n")

        # classification reports per bucket
        f.write("\nPer-Log-Moneyness Bucket Classification Reports:\n")
        f.write("-" * 30 + "\n")

        bins = [-np.inf, -0.1, 0.1, np.inf]
        labels = ["ITM", "ATM", "OTM"]

        for i, bucket_name in enumerate(labels):
            left, right = bins[i], bins[i + 1]
            mask = (logm >= left) & (logm < right)

            if mask.sum() == 0:
                continue

            bucket_targets = targets[mask]
            bucket_pred_labels = predictions[mask]

            f.write(f"\nBucket: {bucket_name} (count = {mask.sum()})\n")
            f.write(
                classification_report(
                    bucket_targets,
                    bucket_pred_labels,
                    digits=4,
                    zero_division=0,
                )
            )

    return {
        "accuracy": accuracy,
        "confusion_matrix": cm,
    }


def main():

    data_dir = "data/dataset_entries/"
    output_root = "models_output/baselines_binary"

    splits = [
        (0.6, 0.2),
        (0.7, 0.15),
        (0.8, 0.1),
    ]

    example_batch = np.load(os.path.join(data_dir, "batch_0000.npz"), allow_pickle=True)
    feature_columns = example_batch["feature_columns"]

    # Initialize baselines
    baselines = [
        MajorityTrendPred(feature_columns=feature_columns),
        LastDayPred(feature_columns=feature_columns),
        BiasedRandomPred(class1_prob=0.7),
        RandomPred(),
    ]

    # Evaluate for each split
    for split_idx, (train_ratio, val_ratio) in enumerate(splits):
        print(f"\n{'=' * 60}")
        print(
            f"Split {split_idx}: "
            f"{train_ratio:.0%}/{val_ratio:.0%}/"
            f"{1 - train_ratio - val_ratio:.0%}"
        )
        print("=" * 60)

        test_dataset = CrossSectionalDataset(
            dataset_entries_dir=data_dir,
            split_type="test",
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            target_key="targets_dir",
        )

        # Collect all sequences, targets, logm (per contract)
        all_sequences = []
        all_targets = []
        all_logm = []

        for i in range(len(test_dataset)):
            sequences, targets, logm = test_dataset[i]
            # sequences: (C, T, F), targets/logm: (C,)
            for j in range(sequences.shape[0]):
                all_sequences.append(sequences[j].numpy())
                all_targets.append(targets[j].item())
                all_logm.append(logm[j].item())

        sequences = np.array(all_sequences)
        targets = np.array(all_targets)
        logm = np.array(all_logm)

        print(f"Test set size: {len(targets)} contracts")

        # Evaluate each baseline
        split_dir = Path(output_root) / f"split_{split_idx}"

        for baseline in baselines:
            baseline_dir = split_dir / baseline.name / "test"
            baseline_dir.mkdir(parents=True, exist_ok=True)

            _ = evaluate_baseline(
                baseline,
                sequences,
                targets,
                logm,
                baseline_dir,
            )

        val_dataset = CrossSectionalDataset(
            dataset_entries_dir=data_dir,
            split_type="val",
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            target_key="targets_dir",
        )

        val_sequences = []
        val_targets = []
        val_logm = []

        for i in range(len(val_dataset)):
            sequences, targets, logm = val_dataset[i]
            for j in range(sequences.shape[0]):
                val_sequences.append(sequences[j].numpy())
                val_targets.append(targets[j].item())
                val_logm.append(logm[j].item())

        val_sequences = np.array(val_sequences)
        val_targets = np.array(val_targets)
        val_logm = np.array(val_logm)

        print(f"\nValidation set size: {len(val_targets)} contracts")

        for baseline in baselines:
            baseline_dir = split_dir / baseline.name / "validation"
            baseline_dir.mkdir(parents=True, exist_ok=True)

            _ = evaluate_baseline(
                baseline,
                val_sequences,
                val_targets,
                val_logm,
                baseline_dir,
            )

    print("\n" + "=" * 60)
    print("All baseline evaluations complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
