# lib/cross_sectional_dataset.py

import os
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class CrossSectionalDataset(Dataset):

    def __init__(
        self,
        dataset_entries_dir: str,
        split_type: str,
        train_ratio: float,
        val_ratio: float,
        target_key: str,
    ) -> None:
        
        assert split_type in {"train", "val", "test"}, "split_type must be 'train', 'val', or 'test'"

        self.dataset_entries_dir = dataset_entries_dir
        self.split = split_type
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.target_key = target_key

        self.all_entries = sorted(
            os.path.join(dataset_entries_dir, f)
            for f in os.listdir(dataset_entries_dir)
            if f.endswith(".npz")
        )

        self.val_start_idx, self.test_start_idx = self._get_split_indices( self.train_ratio, self.val_ratio)

        if split_type == "train":
            self.split_entries = self.all_entries[: self.val_start_idx]
        elif split_type == "val":
            self.split_entries = self.all_entries[self.val_start_idx : self.test_start_idx]
        else:  # "test"
            self.split_entries = self.all_entries[self.test_start_idx :]

    def _get_split_indices(self, train_ratio: float, val_ratio: float) -> Tuple[int, int]:
        """
        Choose split boundaries so that approx. train_ratio / val_ratio of
        *contracts* (not files) go into each split.
        Returns:
            val_start_idx: first index of val files
            test_start_idx: first index of test files
        """
        # First pass – total number of contracts
        total_contracts = 0
        for entry_file in self.all_entries:
            data = np.load(entry_file, allow_pickle=True)
            total_contracts += len(data["sequences"])

        target_train_contracts = int(total_contracts * train_ratio)
        target_val_contracts = int(total_contracts * val_ratio)

        # Find where train ends
        contracts_so_far = 0
        val_start_idx = 0
        for i, entry_file in enumerate(self.all_entries):
            data = np.load(entry_file, allow_pickle=True)
            contracts_so_far += len(data["sequences"])
            if contracts_so_far >= target_train_contracts:
                val_start_idx = i
                break

        # Find where val ends
        contracts_so_far = 0
        test_start_idx = val_start_idx
        for i, entry_file in enumerate(self.all_entries[val_start_idx:], start=val_start_idx):
            data = np.load(entry_file, allow_pickle=True)
            contracts_so_far += len(data["sequences"])
            if contracts_so_far >= target_val_contracts:
                test_start_idx = i
                break

        return val_start_idx, test_start_idx

    def __len__(self) -> int:
        return len(self.split_entries)

    def __getitem__(self, idx: int):
        entry_file = self.split_entries[idx]
        data = np.load(entry_file, allow_pickle=True)

        sequences = data["sequences"] # shape: (num_contracts, seq_len, num_features)
        targets = data[self.target_key] # shape: (num_contracts,)
        log_m = data["log_moneyness_ref"] # shape: (num_contracts,) -> for buckets

        return torch.FloatTensor(sequences), torch.FloatTensor(targets), torch.FloatTensor(log_m)