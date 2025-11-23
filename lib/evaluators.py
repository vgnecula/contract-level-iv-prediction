# lib/evaluators.py

import os
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    auc,
)


class BinaryClassificationEvaluator:
    """
    Binary Classification model evaluator (for binary heads).

    Assumes:
    - model(x) -> logits of shape (B, C, 1)
    - targets are 0/1 labels of shape (B, C)
    """

    def __init__(self, model: torch.nn.Module, device: str) -> None:
        self.model = model
        self.criterion = nn.BCEWithLogitsLoss(reduction="mean")
        self.device = device

    # ------------------------------------------------------------------ #
    # CORE EVALUATION
    # ------------------------------------------------------------------ #
    def evaluate(
        self,
        data_loader,
        split_type: str,
        threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Returns:
            {
                'loss': float,
                'auc': float,
                'accuracy': float,
                'class_report': dict,
                'confusion_matrix': np.ndarray,
                'pred_prob': np.ndarray,
                'pred_labels': np.ndarray,
                'targets': np.ndarray,
            }
        """
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        all_logm = []

        with torch.no_grad():
            for batch_entry_data, batch_entry_targets, batch_logm in data_loader:
                batch_entry_data = batch_entry_data.to(self.device)   # (B, C, T, F)
                batch_entry_targets = batch_entry_targets.to(self.device)  # (B, C)

                outputs = self.model(batch_entry_data)  # (B, C, 1)

                loss = self.criterion(outputs, batch_entry_targets.unsqueeze(-1))
                total_loss += loss.item()

                # Model outputs raw logits and then, BCEWithLogitsLoss takes logits directly and applies sigmoid internally for the loss.
                # -> however, for metrics, we need probabilities in [0,1], so we need a sigmoid here
                batch_probs = torch.sigmoid(outputs).squeeze(-1).reshape(-1).cpu().numpy()
                batch_targets_np = batch_entry_targets.reshape(-1).cpu().numpy()
                batch_logm_np = batch_logm.reshape(-1).cpu().numpy()

                all_preds.extend(batch_probs)
                all_targets.extend(batch_targets_np)
                all_logm.extend(batch_logm_np)

        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        all_logm = np.array(all_logm)

        pred_labels = (all_preds >= threshold).astype(int)

        metrics: Dict[str, Any] = {
            "loss": total_loss / len(data_loader),
            "auc": roc_auc_score(all_targets, all_preds),
            "accuracy": accuracy_score(all_targets, pred_labels),
            "class_report": classification_report(
                all_targets, pred_labels, output_dict=True
            ),
            "confusion_matrix": confusion_matrix(all_targets, pred_labels),
            "pred_prob": all_preds,
            "pred_labels": pred_labels,
            "targets": all_targets,
            "log_moneyness": all_logm, 
        }

        self._print_evaluation_summary(metrics, split_type, threshold)
        return metrics

    def evaluate_and_save(
        self,
        base_output_dir: str,
        data_loader,
        split_type: str,
        threshold: float = 0.5,
    ) -> None:
        """
        Run evaluation and save:
        - ROC curve
        - confusion matrix
        - prediction histogram
        - detailed_metrics.txt
        """
        metrics = self.evaluate(data_loader, split_type, threshold)
        split_dir = os.path.join(base_output_dir, split_type)
        os.makedirs(split_dir, exist_ok=True)
        self._save_evaluation_results(metrics, split_type, split_dir, threshold)

    def _print_evaluation_summary(
        self,
        metrics: Dict[str, Any],
        split_type: str,
        threshold: float,
    ) -> None:
        print(f"\n{split_type.capitalize()} Results (threshold={threshold}):")
        print(f"Loss: {metrics['loss']:.4f}")
        print(f"AUC: {metrics['auc']:.4f}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print("\nClassification Report:")
        print(
            classification_report(
                metrics["targets"],
                metrics["pred_labels"],
                digits=4,
            )
        )

    def _save_evaluation_results(
        self,
        metrics: Dict[str, Any],
        split_type: str,
        output_dir: str,
        threshold: float,
    ) -> None:
        self._save_roc_curve(metrics, output_dir)
        self._save_confusion_matrix(metrics, output_dir, threshold)
        self._save_prediction_distribution(metrics, output_dir, threshold)
        self._save_detailed_metrics(metrics, output_dir, threshold, split_type)

    def _save_roc_curve(self, metrics: Dict[str, Any], output_dir: str) -> None:
        fpr, tpr, _ = roc_curve(metrics["targets"], metrics["pred_prob"])
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 8))
        plt.plot(fpr, tpr, linewidth=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic (ROC) Curve")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "roc_curve.png"))
        plt.close()

    def _save_confusion_matrix(
        self,
        metrics: Dict[str, Any],
        output_dir: str,
        threshold: float,
    ) -> None:
        cm = metrics["confusion_matrix"]
        plt.figure(figsize=(6, 6))
        plt.imshow(cm, interpolation="nearest", cmap="Blues")
        plt.title(f"Confusion Matrix (threshold={threshold})")
        plt.colorbar()
        tick_marks = np.arange(2)
        classes = ["0", "1"]
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)

        # annotate counts
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

    def _save_prediction_distribution(
        self,
        metrics: Dict[str, Any],
        output_dir: str,
        threshold: float,
    ) -> None:
        preds = metrics["pred_prob"]
        plt.figure(figsize=(10, 6))
        plt.hist(preds, bins=50, alpha=0.7)
        plt.axvline(x=threshold, color="r", linestyle="--", label=f"Threshold = {threshold}")
        plt.title("Prediction Probability Distribution")
        plt.xlabel("Predicted Probability (class 1)")
        plt.ylabel("Count")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "prediction_distribution.png"))
        plt.close()

    def _compute_logm_bucket_metrics(
        self,
        preds: np.ndarray,
        true_labels: np.ndarray,
        logm: np.ndarray,
        threshold: float,
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute per-bucket metrics by log-moneyness.

        Buckets:
            [-inf, -0.1)  -> "ITM"
            [-0.1, 0.1)   -> "ATM"
            [0.1, inf)    -> "OTM"
        """
        bins = [-np.inf, -0.1, 0.1, np.inf]
        labels = ["ITM", "ATM", "OTM"]

        bucket_stats: Dict[str, Dict[str, float]] = {}

        for i in range(len(labels)):
            left, right = bins[i], bins[i + 1]
            mask = (logm >= left) & (logm < right)

            if mask.sum() == 0:
                continue

            bucket_preds = preds[mask]
            bucket_targets = true_labels[mask]
            bucket_pred_labels = (bucket_preds >= threshold).astype(int)

            n_total = int(mask.sum())
            n_pos = int((bucket_targets == 1).sum())
            n_neg = int((bucket_targets == 0).sum())
            pos_rate = n_pos / n_total
            mean_pred = float(bucket_preds.mean())
            std_pred = float(bucket_preds.std())

            # safe AUC: only if both classes present
            try:
                auc_val = roc_auc_score(bucket_targets, bucket_preds)
            except ValueError:
                auc_val = float("nan")

            acc = accuracy_score(bucket_targets, bucket_pred_labels)
            report = classification_report(
                bucket_targets, bucket_pred_labels, output_dict=True
            )
            macro_f1 = report["macro avg"]["f1-score"]

            bucket_stats[labels[i]] = {
                "count": n_total,
                "n_pos": n_pos,
                "n_neg": n_neg,
                "pos_rate": pos_rate,
                "mean_pred": mean_pred,
                "std_pred": std_pred,
                "auc": float(auc_val),
                "accuracy": float(acc),
                "macro_f1": float(macro_f1),
            }

        return bucket_stats
 


    def _save_detailed_metrics(
        self,
        metrics: Dict[str, Any],
        output_dir: str,
        threshold: float,
        split_type: str,
    ) -> None:
        preds = metrics["pred_prob"]
        pred_labels = metrics["pred_labels"]
        true_labels = metrics["targets"]
        logm = metrics["log_moneyness"]

        filepath = os.path.join(output_dir, "detailed_metrics.txt")
        with open(filepath, "w") as f:
            f.write(f"{split_type.capitalize()} Evaluation Metrics\n")
            f.write("=" * 50 + "\n\n")

            # --------- GLOBAL METRICS --------- #
            f.write("Basic Metrics:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Loss: {metrics['loss']:.4f}\n")
            f.write(f"AUC: {metrics['auc']:.4f}\n")
            f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
            f.write(f"Threshold: {threshold:.4f}\n\n")

            f.write("Prediction Statistics:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Mean prediction: {preds.mean():.4f}\n")
            f.write(f"Std prediction: {preds.std():.4f}\n")
            f.write(f"Min prediction: {preds.min():.4f}\n")
            f.write(f"Max prediction: {preds.max():.4f}\n\n")

            f.write("Class Distribution:\n")
            f.write("-" * 30 + "\n")
            f.write("True distribution:\n")
            for label in [0, 1]:
                count = (true_labels == label).sum()
                f.write(f"  Class {label}: {count} ({count / len(true_labels):.2%})\n")

            f.write("\nPredicted distribution:\n")
            for label in [0, 1]:
                count = (pred_labels == label).sum()
                f.write(f"  Class {label}: {count} ({count / len(pred_labels):.2%})\n")

            f.write("\nDetailed Classification Report (GLOBAL):\n")
            f.write("-" * 30 + "\n")
            f.write(
                classification_report(
                    true_labels,
                    pred_labels,
                    digits=4,
                )
            )

            # --------- SCALAR BUCKET METRICS --------- #
            f.write("\nPer-Log-Moneyness Bucket Metrics (scalar):\n")
            f.write("-" * 30 + "\n")
            bucket_stats = self._compute_logm_bucket_metrics(
                preds=preds,
                true_labels=true_labels,
                logm=logm,
                threshold=threshold,
            )

            for bucket_name, stats in bucket_stats.items():
                f.write(f"\nBucket: {bucket_name}\n")
                f.write(f"  Count:       {stats['count']}\n")
                f.write(f"  n_pos:       {stats['n_pos']}\n")
                f.write(f"  n_neg:       {stats['n_neg']}\n")
                f.write(f"  pos_rate:    {stats['pos_rate']:.4f}\n")
                f.write(f"  mean_pred:   {stats['mean_pred']:.4f}\n")
                f.write(f"  std_pred:    {stats['std_pred']:.4f}\n")
                f.write(f"  AUC:         {stats['auc']:.4f}\n")
                f.write(f"  Accuracy:    {stats['accuracy']:.4f}\n")
                f.write(f"  Macro F1:    {stats['macro_f1']:.4f}\n")

            # --------- FULL CLASSIFICATION REPORT PER BUCKET --------- #
            f.write("\nPer-Log-Moneyness Bucket Classification Reports:\n")
            f.write("-" * 30 + "\n")

            bins = [-np.inf, -0.1, 0.1, np.inf]
            labels = ["ITM", "ATM", "OTM"]

            for i, bucket_name in enumerate(labels):
                left, right = bins[i], bins[i + 1]
                mask = (logm >= left) & (logm < right)

                if mask.sum() == 0:
                    continue

                bucket_targets = true_labels[mask]
                bucket_pred_labels = pred_labels[mask]

                f.write(f"\nBucket: {bucket_name} (count = {mask.sum()})\n")
                f.write(
                    classification_report(
                        bucket_targets,
                        bucket_pred_labels,
                        digits=4,
                    )
                )

class RegressionEvaluator:
    """
    Regression model evaluator (for reg heads).

    Assumes:
    - model(x) -> predictions of shape (B, C, 1)
    - targets are real-valued |ΔIV| of shape (B, C)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: str,
    ) -> None:
        self.model = model
        self.criterion = nn.MSELoss(reduction="mean")
        self.device = device

    def evaluate(
        self,
        data_loader,
        split_type: str,
    ) -> Dict[str, Any]:
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        all_logm = []

        with torch.no_grad():
            for batch_entry_data, batch_entry_targets, batch_logm in data_loader:
                batch_entry_data = batch_entry_data.to(self.device)       # (B, C, T, F)
                batch_entry_targets = batch_entry_targets.to(self.device) # (B, C)

                outputs = self.model(batch_entry_data)  # (B, C, 1)

                # loss is in the same space as training: |ΔIV|
                loss = self.criterion(outputs, batch_entry_targets.unsqueeze(-1))
                total_loss += loss.item()

                batch_preds = outputs.squeeze(-1).reshape(-1).cpu().numpy()

                batch_targets_np = batch_entry_targets.reshape(-1).cpu().numpy()

                batch_logm_np = batch_logm.reshape(-1).cpu().numpy()

                all_preds.extend(batch_preds)
                all_targets.extend(batch_targets_np)
                all_logm.extend(batch_logm_np)

        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        all_logm = np.array(all_logm)

        # In mag mode, eval space == training space == |ΔIV|
        preds_eval = all_preds
        targets_eval = all_targets

        mse = float(np.mean((preds_eval - targets_eval) ** 2))
        mae = float(np.mean(np.abs(preds_eval - targets_eval)))
        rmse = float(np.sqrt(mse))

        metrics: Dict[str, Any] = {
            "loss": total_loss / len(data_loader),   # criterion loss, |ΔIV|
            "mse": mse,
            "mae": mae,
            "rmse": rmse,
            "targets": targets_eval,
            "preds": preds_eval,
            "log_moneyness": all_logm,
        }

        self._print_evaluation_summary(metrics, split_type)
        return metrics

    def evaluate_and_save(
        self,
        base_output_dir: str,
        data_loader,
        split_type: str,
    ) -> None:
        """
        Run evaluation and save:
        - preds_vs_targets scatter
        - residuals histogram
        - prediction distribution
        - detailed_metrics.txt
        """
        metrics = self.evaluate(data_loader, split_type)
        split_dir = os.path.join(base_output_dir, split_type)
        os.makedirs(split_dir, exist_ok=True)
        self._save_evaluation_results(metrics, split_type, split_dir)

    def _print_evaluation_summary(
        self,
        metrics: Dict[str, Any],
        split_type: str,
    ) -> None:
        print(f"\n{split_type.capitalize()} Regression Results:")
        print(f"Loss (criterion, |ΔIV| space): {metrics['loss']:.6f}")
        print(f"MSE (|ΔIV|):                   {metrics['mse']:.6f}")
        print(f"RMSE (|ΔIV|):                  {metrics['rmse']:.6f}")
        print(f"MAE (|ΔIV|):                   {metrics['mae']:.6f}")


    def _save_evaluation_results(
        self,
        metrics: Dict[str, Any],
        split_type: str,
        output_dir: str,
    ) -> None:
        self._save_preds_vs_targets(metrics, output_dir)
        self._save_residuals_histogram(metrics, output_dir)
        self._save_prediction_distribution(metrics, output_dir)
        self._save_detailed_metrics(metrics, output_dir, split_type)

    def _save_preds_vs_targets(
        self,
        metrics: Dict[str, Any],
        output_dir: str,
    ) -> None:
        preds = metrics["preds"]
        targets = metrics["targets"]

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
        plt.savefig(os.path.join(output_dir, "preds_vs_targets.png"))
        plt.close()

    def _save_residuals_histogram(
        self,
        metrics: Dict[str, Any],
        output_dir: str,
    ) -> None:
        preds = metrics["preds"]
        targets = metrics["targets"]
        residuals = preds - targets

        plt.figure(figsize=(10, 6))
        plt.hist(residuals, bins=50, alpha=0.7)
        plt.title("Residuals Histogram (pred - true)")
        plt.xlabel("Residual (|ΔIV|)")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "residuals_histogram.png"))
        plt.close()

    def _save_prediction_distribution(
        self,
        metrics: Dict[str, Any],
        output_dir: str,
    ) -> None:
        preds = metrics["preds"]
        targets = metrics["targets"]

        plt.figure(figsize=(10, 6))
        plt.hist(targets, bins=50, alpha=0.5, label="True |ΔIV|", density=True)
        plt.hist(preds, bins=50, alpha=0.5, label="Predicted |ΔIV|", density=True)
        plt.title("Prediction vs True Distribution (|ΔIV|)")
        plt.xlabel("|ΔIV|")
        plt.ylabel("Density")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "prediction_distribution.png"))
        plt.close()

    def _compute_logm_bucket_metrics(
        self,
        preds: np.ndarray,
        targets: np.ndarray,
        logm: np.ndarray,
    ) -> Dict[str, Dict[str, float]]:
        
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

    def _save_detailed_metrics(
        self,
        metrics: Dict[str, Any],
        output_dir: str,
        split_type: str,
    ) -> None:
        preds = metrics["preds"]
        targets = metrics["targets"]
        residuals = preds - targets
        logm = metrics["log_moneyness"]

        filepath = os.path.join(output_dir, "detailed_metrics.txt")
        with open(filepath, "w") as f:
            f.write(f"{split_type.capitalize()} Regression Evaluation Metrics\n")
            f.write("=" * 50 + "\n\n")

            # Basic metrics
            f.write("Basic Metrics:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Loss (criterion, training):    {metrics['loss']:.6f}\n")
            f.write(f"MSE (eval |ΔIV| space):       {metrics['mse']:.6f}\n")
            f.write(f"RMSE (eval |ΔIV| space):      {metrics['rmse']:.6f}\n")
            f.write(f"MAE (eval |ΔIV| space):       {metrics['mae']:.6f}\n\n")

            # Prediction statistics (eval space)
            f.write("Prediction Statistics (eval |ΔIV| space):\n")
            f.write("-" * 30 + "\n")
            f.write(f"Mean prediction:  {preds.mean():.6f}\n")
            f.write(f"Std prediction:   {preds.std():.6f}\n")
            f.write(f"Min prediction:   {preds.min():.6f}\n")
            f.write(f"Max prediction:   {preds.max():.6f}\n\n")

            # Target statistics (eval space)
            f.write("Target Statistics (eval |ΔIV| space):\n")
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

            # --------- Per-Log-Moneyness Bucket Metrics --------- #
            f.write("\nPer-Log-Moneyness Bucket Metrics (eval |ΔIV| space):\n")
            f.write("-" * 30 + "\n")
            bucket_stats = self._compute_logm_bucket_metrics(
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

