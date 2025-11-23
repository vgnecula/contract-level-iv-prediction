# lib/models/wrappers.py

import torch
import torch.nn as nn

from .backbones import TCTABackbone, FullTempTCTABackbone, TTABackbone, LSTMBackbone
from .heads import BinaryHead, RegressionHead, FullTempBinaryHead, FullTempRegressionHead 


class ModelWithHead(nn.Module):
    """
    Generic: backbone(x) -> embeddings, head(emb) -> output.
    """

    def __init__(self, backbone: nn.Module, head: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.backbone(x)
        return self.head(emb)


def build_tcta_binary_model(
    input_size: int,
    d_model: int,
    nhead: int,
    num_layers: int,
    dropout: float,
) -> ModelWithHead:
    
    backbone = TCTABackbone(
        input_size=input_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dropout=dropout,
    )

    head = BinaryHead(in_dim=backbone.out_dim, dropout=dropout)
    return ModelWithHead(backbone, head)


def build_tcta_regression_model(
    input_size: int,
    d_model: int,
    nhead: int,
    num_layers: int,
    dropout: float,
) -> ModelWithHead:
    
    backbone = TCTABackbone(
        input_size=input_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dropout=dropout,
    )
    head = RegressionHead(in_dim=backbone.out_dim, dropout=dropout)
    return ModelWithHead(backbone, head)

def build_tta_binary_model(
    input_size: int,
    d_model: int,
    nhead: int,
    num_layers: int,
    dropout: float,
) -> ModelWithHead:
    backbone = TTABackbone(
        input_size=input_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dropout=dropout,
    )

    # we want TTA to be dmodel -> dmodel
    head = BinaryHead(
        in_dim=backbone.out_dim,
        hidden_dim=backbone.out_dim,  # <- KEY LINE (we want no reduction here)
        dropout=dropout,
    )

    return ModelWithHead(backbone, head)


def build_tta_regression_model(
    input_size: int,
    d_model: int,
    nhead: int,
    num_layers: int,
    dropout: float,
) -> ModelWithHead:
    backbone = TTABackbone(
        input_size=input_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dropout=dropout,
    )

    # we want TTA to be dmodel -> dmodel
    head = RegressionHead(
        in_dim=backbone.out_dim,
        hidden_dim=backbone.out_dim,  # <- KEY LINE (we want no reduction here)
        dropout=dropout,
    ) 
    return ModelWithHead(backbone, head)

def build_lstm_binary_model(
    input_size: int,
    d_model: int,
    num_layers: int,
    dropout: float,
    bidirectional: bool = True,
) -> ModelWithHead:
    backbone = LSTMBackbone(
        input_size=input_size,
        d_model=d_model,
        num_layers=num_layers,
        dropout=dropout,
        bidirectional=bidirectional,
    )
    head = BinaryHead(in_dim=backbone.out_dim, dropout=dropout)
    return ModelWithHead(backbone, head)


def build_lstm_regression_model(
    input_size: int,
    d_model: int,
    num_layers: int,
    dropout: float,
    bidirectional: bool = True,
) -> ModelWithHead:
    backbone = LSTMBackbone(
        input_size=input_size,
        d_model=d_model,
        num_layers=num_layers,
        dropout=dropout,
        bidirectional=bidirectional,
    )
    head = RegressionHead(in_dim=backbone.out_dim, dropout=dropout)
    return ModelWithHead(backbone, head)

def build_fulltemp_tcta_binary_model(
    input_size: int,
    d_model: int,
    nhead: int,
    num_layers: int,
    dropout: float,
    seq_length: int,
) -> ModelWithHead:
    """
    Full-temporal TCTA (9-day concat) with classifier matching the original
    9-day concat TCTA_Model.
    """
    backbone = FullTempTCTABackbone(
        input_size=input_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dropout=dropout,
        seq_length=seq_length,
    )

    head = FullTempBinaryHead(
        in_dim=backbone.out_dim,
        d_model=d_model,
        dropout=dropout,
    )

    return ModelWithHead(backbone, head)


def build_fulltemp_tcta_regression_model(
    input_size: int,
    d_model: int,
    nhead: int,
    num_layers: int,
    dropout: float,
    seq_length: int,
) -> ModelWithHead:
    backbone = FullTempTCTABackbone(
        input_size=input_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dropout=dropout,
        seq_length=seq_length,
    )

    head = FullTempRegressionHead(
        in_dim=backbone.out_dim,
        d_model=d_model,
        dropout=dropout,
    )

    return ModelWithHead(backbone, head)
