# models/baselines_reg.py

import numpy as np
from typing import Sequence


class LastDayAbsChangePred:
    """
    Predicts next-day |del_IV| using the last observed absolute IV change
    in the window.

    Assumes:
        - sequence: np.ndarray of shape (T, F)
        - feature_columns contains "ImpliedVolatility"
    """

    def __init__(self, feature_columns: Sequence[str]) -> None:
        self.name = "LastDayAbsChange"
        feature_columns = list(feature_columns)
        if "ImpliedVolatility" not in feature_columns:
            raise ValueError("ImpliedVolatility must be in feature_columns")
        self.iv_idx = feature_columns.index("ImpliedVolatility")

    def predict(self, sequence: np.ndarray) -> float:
        """
        Single sequence prediction.

        Args:
            sequence: (T, F) array of features for a single contract

        Returns:
            Scalar prediction for |del_IV| (non-negative float).
        """
        iv_values = sequence[:, self.iv_idx]
        if iv_values.shape[0] < 2:
            # Not enough history, default to 0
            return 0.0

        last_change = iv_values[-1] - iv_values[-2]
        return float(abs(last_change))


class MeanAbsChangePred:
    """
    Predicts next-day |del_IV| as the mean absolute IV change
    over the whole window.
    """

    def __init__(self, feature_columns: Sequence[str]) -> None:
        self.name = "MeanAbsChange"
        feature_columns = list(feature_columns)
        if "ImpliedVolatility" not in feature_columns:
            raise ValueError("ImpliedVolatility must be in feature_columns")
        self.iv_idx = feature_columns.index("ImpliedVolatility")

    def predict(self, sequence: np.ndarray) -> float:
        """
        Single sequence prediction.

        Returns:
            Scalar prediction for |Δdel_IV|
        """
        iv_values = sequence[:, self.iv_idx]
        if iv_values.shape[0] < 2:
            return 0.0

        changes = np.diff(iv_values)
        abs_changes = np.abs(changes)
        return float(abs_changes.mean())


class ZeroAbsChangePred:
    """
    always predicts 0.
    """

    def __init__(self) -> None:
        self.name = "ZeroAbsChange"

    def predict(self, sequence: np.ndarray) -> float:
        return 0.0


class RandomAbsChangePred:
    """
    Random baseline for |ΔIV| predictions.

    Draws a value from Uniform[low, high] independently for each sequence.
    """

    def __init__(self, low: float = 0.0, high: float = 0.1, seed: int = 42) -> None:
        if low < 0:
            raise ValueError("low must be >= 0")
        if high <= low:
            raise ValueError("high must be > low")
        self.name = f"RandomAbsChange_[{low:.3f},{high:.3f}]"
        self.low = low
        self.high = high
        self._rng = np.random.default_rng(seed)

    def predict(self, sequence: np.ndarray) -> float:
        return float(self._rng.uniform(self.low, self.high))
