# lib/models/heads.py

from typing import Optional
import torch
import torch.nn as nn


# ---------------------------------------------------------------------
# Shared per-contract MLP head (used for both binary + regression)
# ---------------------------------------------------------------------
class _MLPHead(nn.Module):
    """
    Generic per-contract MLP head.
    Input:  (B, C, D)
    Output: (B, C, 1)
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        hidden_dim = hidden_dim or in_dim // 2

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BinaryHead(_MLPHead):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.25,
    ) -> None:
        super().__init__(in_dim=in_dim, hidden_dim=hidden_dim, dropout=dropout)


class RegressionHead(_MLPHead):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.25,
    ) -> None:
        super().__init__(in_dim=in_dim, hidden_dim=hidden_dim, dropout=dropout)


# ---------------------------------------------------------------------
# FULL-TEMP heads: shared large MLP + thin task-specific wrappers
# ---------------------------------------------------------------------
class _FullTempMLPHead(nn.Module):
    """
    Generic head for the full-temporal TCTA variant.

    Linear((T+1)*d_model -> 4*d_model) -> GELU -> Dropout ->
    Linear(4*d_model -> d_model) -> LayerNorm(d_model) -> GELU -> Dropout ->
    Linear(d_model -> 1)
    """

    def __init__(
        self,
        in_dim: int,   # typically (seq_length * d_model) + d_model
        d_model: int,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FullTempBinaryHead(_FullTempMLPHead):
    def __init__(
        self,
        in_dim: int,
        d_model: int,
        dropout: float = 0.25,
    ) -> None:
        super().__init__(in_dim=in_dim, d_model=d_model, dropout=dropout)


class FullTempRegressionHead(_FullTempMLPHead):
    def __init__(
        self,
        in_dim: int,
        d_model: int,
        dropout: float = 0.25,
    ) -> None:
        super().__init__(in_dim=in_dim, d_model=d_model, dropout=dropout)
