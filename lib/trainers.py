# lib/trainers.py
import os
from typing import Callable, Dict, Any, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, classification_report

from .cross_sectional_dataset import CrossSectionalDataset
from .evaluators import BinaryClassificationEvaluator, RegressionEvaluator
from .utils import get_device


class BinaryTrainer:
    """
    Trainer for per-contract binary classification head based models / built wrappers.
    """

    def __init__(
        self,
        dataset_entries_dir: str,
        output_dir: str,
        train_ratio: float,
        val_ratio: float,
        target_key: str,
        build_model_fn: Callable[..., nn.Module],
        model_kwargs: Dict[str, Any],
    ) -> None:
        self.dataset_entries_dir = dataset_entries_dir
        self.output_dir = output_dir
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.target_key = target_key

        self.build_model_fn = build_model_fn
        self.model_kwargs = model_kwargs

        self.device = get_device()
        self.model: nn.Module = None  # will be set in train()
        self.input_size: int = -1     # inferred from data in train()

    def train(
        self,
        batch_size: int,
        effective_batch_size: int,
        epochs: int,
        min_epochs: int,
        patience: int,
        learning_rate: float,
        weight_decay: float = 0.001,
        beta1: float = 0.95,
        beta2: float = 0.999,
        use_scheduler: bool = False,          
        scheduler_type: str = "cosine_restart",  # either we have cosine scheduler, or not at all
    ) -> Dict[str, Any]:

        epoch_metrics_dict: Dict[str, List[float]] = {
            "learning_rates": [],
            "train_losses": [],
            "val_losses": [],
            "train_aucs": [],
            "val_aucs": [],
            "train_f1": [],
            "val_f1": [],
        }

        # -------------------------- DATASETS & LOADERS --------------------- #
        train_dataset = CrossSectionalDataset(
            dataset_entries_dir=self.dataset_entries_dir,
            split_type="train",
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
            target_key=self.target_key,
        )
        val_dataset = CrossSectionalDataset(
            dataset_entries_dir=self.dataset_entries_dir,
            split_type="val",
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
            target_key=self.target_key,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        # Infer input size from one batch
        sample_sequences, _, _ = next(iter(train_loader))
        # sample_sequences: (B, C, T, F)
        self.input_size = sample_sequences.shape[-1]

        # --------------------------- MODEL & OPTIM ------------------------- #
        self.model = self.build_model_fn(
            input_size=self.input_size,
            **self.model_kwargs,
        ).to(self.device)

        criterion = nn.BCEWithLogitsLoss(reduction="mean")

        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(beta1, beta2),
        )
        optimizer.zero_grad()

        # LR scheduler - set up, if we have it
        num_cycles = 3
        cycle_length = max(1, epochs // num_cycles)

        scheduler = None
        if use_scheduler:
            if scheduler_type == "cosine_restart":
                scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer,
                    T_0=cycle_length,
                    T_mult=1,
                    eta_min=learning_rate / 100,
                )
            else:
                raise ValueError(f"Unknown scheduler_type: {scheduler_type}")


        evaluator = BinaryClassificationEvaluator(self.model, self.device)

        best_val_auc = 0.0
        best_epoch = 0
        patience_counter = 0
        early_stopped = False

        # epoch loop 
        for epoch in range(epochs):
            all_train_preds: List[float] = []
            all_train_targets: List[float] = []

            self.model.train()
            accumulated_entries = 0
            train_loss = 0.0

            current_lr = optimizer.param_groups[0]["lr"]
            print(f"\nEpoch {epoch+1}/{epochs} - current LR: {current_lr:.6f}")

            # --- BATCH LOOP --- 
            for batch_idx, (batch_entry_data, batch_entry_targets, _) in enumerate(
                tqdm(train_loader, desc=f"Training epoch {epoch+1}")
            ):
                batch_entry_data = batch_entry_data.to(self.device)   # (B, C, T, F)
                batch_entry_targets = batch_entry_targets.to(self.device)  # (B, C)

                outputs = self.model(batch_entry_data)  # (B, C, 1)
                loss = criterion(outputs, batch_entry_targets.unsqueeze(-1))
                loss = loss / effective_batch_size
                loss.backward()

                # Store preds/targets for epoch-level metrics
                batch_preds = torch.sigmoid(outputs).squeeze(-1).reshape(-1).detach().cpu().numpy()
                
                batch_targets = batch_entry_targets.reshape(-1).cpu().numpy()

                all_train_preds.extend(batch_preds)
                all_train_targets.extend(batch_targets)

                accumulated_entries += 1
                train_loss += loss.item() * effective_batch_size

                if accumulated_entries >= effective_batch_size:
                    optimizer.step()
                    optimizer.zero_grad()
                    accumulated_entries = 0

            # Handle leftover gradients (if any)
            if accumulated_entries > 0:
                scale_factor = effective_batch_size / accumulated_entries
                for param in self.model.parameters():
                    if param.grad is not None:
                        param.grad.data.mul_(scale_factor)
                optimizer.step()
                optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()
 

            # ---------------------- TRAIN METRICS (EPOCH) ------------------- #
            all_train_preds_np = np.array(all_train_preds)
            all_train_targets_np = np.array(all_train_targets)
            train_pred_labels = (all_train_preds_np >= 0.5).astype(int)

            train_auc = roc_auc_score(all_train_targets_np, all_train_preds_np)
            train_report = classification_report(
                all_train_targets_np, train_pred_labels, output_dict=True
            )
            train_f1 = train_report["macro avg"]["f1-score"]

            # validation evaluation -> with the best model saved
            val_metrics = evaluator.evaluate(
                data_loader=val_loader,
                split_type="validation",
                threshold=0.5,
            )


            epoch_metrics_dict["learning_rates"].append(current_lr)
            epoch_metrics_dict["train_losses"].append(train_loss / len(train_loader))
            epoch_metrics_dict["val_losses"].append(val_metrics["loss"])
            epoch_metrics_dict["train_aucs"].append(train_auc)
            epoch_metrics_dict["val_aucs"].append(val_metrics["auc"])
            epoch_metrics_dict["train_f1"].append(train_f1)
            epoch_metrics_dict["val_f1"].append(
                val_metrics["class_report"]["macro avg"]["f1-score"]
            )

            # early stopping
            if epoch > min_epochs:
                if val_metrics["auc"] > best_val_auc:
                    best_val_auc = val_metrics["auc"]
                    best_epoch = epoch
                    patience_counter = 0

                    best_path = os.path.join(self.output_dir, "best_model.pt")
                    torch.save(self.model.state_dict(), best_path)
                    print(f"Saved new best model with Val AUC: {best_val_auc:.4f}")
                else:
                    patience_counter += 1
                    print(f"Patience counter: {patience_counter}/{patience}")

                if patience_counter >= patience and epoch >= (num_cycles - 1) * cycle_length: 
                    early_stopped = True
                    print(f"\nEarly stopping triggered at epoch {epoch}")
                    break
            else:
                print(f"Warmup phase: {epoch}/{min_epochs}")

        # If we never saved a best model, save final
        best_model_path = os.path.join(self.output_dir, "best_model.pt")
        if not os.path.exists(best_model_path):
            torch.save(self.model.state_dict(), best_model_path)

        return {
            "best_model_path": best_model_path,
            "best_epoch": best_epoch,
            "best_val_auc": best_val_auc,
            "early_stopped": early_stopped,
            "final_learning_rate": current_lr,
            "epoch_metrics_dict": epoch_metrics_dict,
        }

    # --------------------------------------------------------------------- #
    # PLOTTING & SUMMARY
    # --------------------------------------------------------------------- #
    def _plot_training_metrics(self, epoch_metrics_dict: Dict[str, List[float]], output_dir: str) -> None:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Training Metrics Over Time", fontsize=16)

        # LR
        ax1.plot(epoch_metrics_dict["learning_rates"], "b-", label="Learning Rate")
        ax1.set_title("Learning Rate vs Epochs")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("LR")
        ax1.grid(True)

        # Loss
        ax2.plot(epoch_metrics_dict["train_losses"], "b-", label="Train Loss")
        ax2.plot(epoch_metrics_dict["val_losses"], "r-", label="Val Loss")
        ax2.set_title("Loss vs Epochs")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")
        ax2.legend()
        ax2.grid(True)

        # AUC
        ax3.plot(epoch_metrics_dict["train_aucs"], "b-", label="Train AUC")
        ax3.plot(epoch_metrics_dict["val_aucs"], "r-", label="Val AUC")
        ax3.set_title("AUC vs Epochs")
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("AUC")
        ax3.legend()
        ax3.grid(True)

        # Macro F1
        ax4.plot(epoch_metrics_dict["train_f1"], "b-", label="Train Macro F1")
        ax4.plot(epoch_metrics_dict["val_f1"], "r-", label="Val Macro F1")
        ax4.set_title("Macro F1 vs Epochs")
        ax4.set_xlabel("Epoch")
        ax4.set_ylabel("Macro F1")
        ax4.legend()
        ax4.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "training_metrics.png"))
        plt.close()

    def save_training_summary(
        self,
        split_idx: int,
        split_training_summary: Dict[str, Any],
        output_dir: str,
    ) -> None:
        # training curves
        if "epoch_metrics_dict" in split_training_summary:
            self._plot_training_metrics(split_training_summary["epoch_metrics_dict"], output_dir)

        train_ratio = self.train_ratio
        val_ratio = self.val_ratio
        test_ratio = 1 - train_ratio - val_ratio

        with open(os.path.join(output_dir, f"split{split_idx}_summary.txt"), "w") as f:
            f.write(f"Split Configuration {split_idx}\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"Train Ratio: {train_ratio:.2%}\n")
            f.write(f"Validation Ratio: {val_ratio:.2%}\n")
            f.write(f"Test Ratio: {test_ratio:.2%}\n\n")

            f.write(f"Split Configuration {split_idx} Training Summary:\n")
            f.write("-" * 30 + "\n")

            for key, value in split_training_summary.items():
                if key != "epoch_metrics_dict":
                    f.write(f"{key}: {value}\n")


class RegressionTrainer:
    """
    Trainer for regression head based models / built wrappers.
    """

    def __init__(
        self,
        dataset_entries_dir: str,
        output_dir: str,
        train_ratio: float,
        val_ratio: float,
        target_key: str,
        build_model_fn: Callable[..., nn.Module],
        model_kwargs: Dict[str, Any],
    ) -> None:
        self.dataset_entries_dir = dataset_entries_dir
        self.output_dir = output_dir
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.target_key = target_key

        self.build_model_fn = build_model_fn
        self.model_kwargs = model_kwargs

        self.device = get_device()
        self.model: nn.Module = None  # will be set in train()
        self.input_size: int = -1     # inferred from data in train()

    def train(
        self,
        batch_size: int,
        effective_batch_size: int,
        epochs: int,
        min_epochs: int,
        patience: int,
        learning_rate: float,
        weight_decay: float = 0.001,
        beta1: float = 0.95,
        beta2: float = 0.999,
        use_scheduler: bool = False, 
        scheduler_type: str = "cosine_restart", # either we have cosine scheduler, or not at all 
    ) -> Dict[str, Any]:

        epoch_metrics_dict: Dict[str, List[float]] = {
            "learning_rates": [],
            "train_losses": [],
            "val_losses": [],
            "train_mse": [],
            "val_mse": [],
            "train_mae": [],
            "val_mae": [],
        }

        # -------------------------- DATASETS & LOADERS --------------------- #
        train_dataset = CrossSectionalDataset(
            dataset_entries_dir=self.dataset_entries_dir,
            split_type="train",
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
            target_key=self.target_key,
        )
        val_dataset = CrossSectionalDataset(
            dataset_entries_dir=self.dataset_entries_dir,
            split_type="val",
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
            target_key=self.target_key,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        # Infer input size from one batch
        sample_sequences, _, _ = next(iter(train_loader))
        # sample_sequences: (B, C, T, F)
        self.input_size = sample_sequences.shape[-1]

        # --------------------------- MODEL & OPTIM ------------------------- #
        self.model = self.build_model_fn(
            input_size=self.input_size,
            **self.model_kwargs,
        ).to(self.device)

        criterion = nn.MSELoss(reduction="mean")

        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(beta1, beta2),
        )
        optimizer.zero_grad()

        # Scheduler set up, if we have it
        num_cycles = 3
        cycle_length = max(1, epochs // num_cycles)

        scheduler = None
        if use_scheduler:
            if scheduler_type == "cosine_restart":
                scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer,
                    T_0=cycle_length,
                    T_mult=1,
                    eta_min=learning_rate / 100,
                )
            else:
                raise ValueError(f"Unknown scheduler_type: {scheduler_type}")

        evaluator = RegressionEvaluator(
            self.model,
            self.device
        )

        best_val_mse = float("inf")
        best_epoch = 0
        patience_counter = 0
        early_stopped = False

        # EPOCH LOOP
        for epoch in range(epochs):
            all_train_preds: List[float] = []
            all_train_targets: List[float] = []

            self.model.train()
            accumulated_entries = 0
            train_loss = 0.0

            current_lr = optimizer.param_groups[0]["lr"]
            print(f"\n Epoch {epoch+1}/{epochs} - current LR: {current_lr:.6f}")
            
            # ---------------------------------------------------------------------- 
            #  BATCH LOOP
            # ----------------------------------------------------------------------
            for batch_idx, (batch_entry_data, batch_entry_targets, _) in enumerate(
                tqdm(train_loader, desc=f"Reg Training epoch {epoch+1}")
            ):
                batch_entry_data = batch_entry_data.to(self.device)   # (B, C, T, F)
                batch_entry_targets = batch_entry_targets.to(self.device)  # (B, C)

                outputs = self.model(batch_entry_data)  # (B, C, 1)
                loss = criterion(outputs, batch_entry_targets.unsqueeze(-1))
                loss = loss / effective_batch_size
                loss.backward()

                # Store preds/targets for epoch-level metrics
                batch_preds = outputs.squeeze(-1).reshape(-1).detach().cpu().numpy()
                
                batch_targets = batch_entry_targets.reshape(-1).cpu().numpy()
                

                all_train_preds.extend(batch_preds)
                all_train_targets.extend(batch_targets)

                accumulated_entries += 1
                train_loss += loss.item() * effective_batch_size

                if accumulated_entries >= effective_batch_size:
                    optimizer.step()
                    optimizer.zero_grad()
                    accumulated_entries = 0

            # handle leftover gradients
            if accumulated_entries > 0:
                scale_factor = effective_batch_size / accumulated_entries
                for param in self.model.parameters():
                    if param.grad is not None:
                        param.grad.data.mul_(scale_factor)
                optimizer.step()
                optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()


            # ---------------------- TRAIN METRICS (EPOCH) ------------------- #
            all_train_preds_np = np.array(all_train_preds)
            all_train_targets_np = np.array(all_train_targets)

            train_mse = float(np.mean((all_train_preds_np - all_train_targets_np) ** 2))
            train_mae = float(np.mean(np.abs(all_train_preds_np - all_train_targets_np)))

            # val evaluation -> with the best model saved
            val_metrics = evaluator.evaluate(
                data_loader=val_loader,
                split_type="validation",
            )

            val_loss = val_metrics["loss"]
            val_mse = val_metrics["mse"]
            val_mae = val_metrics["mae"]

            epoch_metrics_dict["learning_rates"].append(current_lr)
            epoch_metrics_dict["train_losses"].append(train_loss / len(train_loader))
            epoch_metrics_dict["val_losses"].append(val_loss)
            epoch_metrics_dict["train_mse"].append(train_mse)
            epoch_metrics_dict["val_mse"].append(val_mse)
            epoch_metrics_dict["train_mae"].append(train_mae)
            epoch_metrics_dict["val_mae"].append(val_mae)

            print(
                f"Epoch {epoch+1}: "
                f"train_loss={epoch_metrics_dict['train_losses'][-1]:.6f}, "
                f"val_loss={val_loss:.6f}, "
                f"train_mse={train_mse:.6f}, val_mse={val_mse:.6f}, "
                f"train_mae={train_mae:.6f}, val_mae={val_mae:.6f}"
            )

            # early stopping
            if epoch > min_epochs:
                if val_mse < best_val_mse:
                    best_val_mse = val_mse
                    best_epoch = epoch
                    patience_counter = 0

                    best_path = os.path.join(self.output_dir, "best_model.pt")
                    torch.save(self.model.state_dict(), best_path)
                    print(f"Saved new best model with Val MSE: {best_val_mse:.6f}")
                else:
                    patience_counter += 1
                    print(f"Patience counter: {patience_counter}/{patience}")

                if patience_counter >= patience and epoch >= (num_cycles - 1) * cycle_length:  
                    early_stopped = True
                    print(f"\nEarly stopping triggered at epoch {epoch}")
                    break
            else:
                print(f" Warmup phase: {epoch}/{min_epochs}")

        # if we never saved a best model, save final
        best_model_path = os.path.join(self.output_dir, "best_model.pt")
        if not os.path.exists(best_model_path):
            torch.save(self.model.state_dict(), best_model_path)

        return {
            "best_model_path": best_model_path,
            "best_epoch": best_epoch,
            "best_val_mse": best_val_mse,
            "early_stopped": early_stopped,
            "final_learning_rate": current_lr,
            "epoch_metrics_dict": epoch_metrics_dict,
        }

    # --------------------------------------------------------------------- #
    # PLOTTING & SUMMARY
    # --------------------------------------------------------------------- #
    def _plot_training_metrics(self, epoch_metrics_dict: Dict[str, List[float]], output_dir: str) -> None:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Regression Training Metrics Over Time", fontsize=16)

        # LR
        ax1.plot(epoch_metrics_dict["learning_rates"], "b-", label="Learning Rate")
        ax1.set_title("Learning Rate vs Epochs")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("LR")
        ax1.grid(True)

        # Loss
        ax2.plot(epoch_metrics_dict["train_losses"], "b-", label="Train Loss")
        ax2.plot(epoch_metrics_dict["val_losses"], "r-", label="Val Loss")
        ax2.set_title("Loss vs Epochs")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")
        ax2.legend()
        ax2.grid(True)

        # MSE
        ax3.plot(epoch_metrics_dict["train_mse"], "b-", label="Train MSE")
        ax3.plot(epoch_metrics_dict["val_mse"], "r-", label="Val MSE")
        ax3.set_title("MSE vs Epochs")
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("MSE")
        ax3.legend()
        ax3.grid(True)

        # MAE
        ax4.plot(epoch_metrics_dict["train_mae"], "b-", label="Train MAE")
        ax4.plot(epoch_metrics_dict["val_mae"], "r-", label="Val MAE")
        ax4.set_title("MAE vs Epochs")
        ax4.set_xlabel("Epoch")
        ax4.set_ylabel("MAE")
        ax4.legend()
        ax4.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "training_metrics.png"))
        plt.close()

    def save_training_summary(
        self,
        split_idx: int,
        split_training_summary: Dict[str, Any],
        output_dir: str,
    ) -> None:
        # training curves
        if "epoch_metrics_dict" in split_training_summary:
            self._plot_training_metrics(split_training_summary["epoch_metrics_dict"], output_dir)

        train_ratio = self.train_ratio
        val_ratio = self.val_ratio
        test_ratio = 1 - train_ratio - val_ratio

        with open(os.path.join(output_dir, f"Regression_split{split_idx}_summary.txt"), "w") as f:
            f.write(f"Regression Split Configuration {split_idx}\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"Train Ratio: {train_ratio:.2%}\n")
            f.write(f"Validation Ratio: {val_ratio:.2%}\n")
            f.write(f"Test Ratio: {test_ratio:.2%}\n\n")

            f.write(f"Split Configuration {split_idx} Training Summary:\n")
            f.write("-" * 30 + "\n")

            for key, value in split_training_summary.items():
                if key != "epoch_metrics_dict":
                    f.write(f"{key}: {value}\n")
