# lib/models/backbones.py

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_len).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        x = x + self.pe[: x.size(1)]
        return self.dropout(x)


class TCTABackbone(nn.Module):
    """
    Input:  x (B, C, T, F)
    Output: embeddings (B, C, 2 * d_model)
    """

    def __init__(
        self,
        input_size: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dropout: float,
    ) -> None:
    
        super().__init__()
        self.d_model = d_model
        self.out_dim = 2 * d_model  # temporal + cross-sectional

        # feature projection
        self.feature_projection = nn.Sequential(
            nn.Linear(input_size, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Temporal encoder
        # batch_first = True -> means that we expect a 3D tensor (batch_size, sequence_length, embedding_space)
        # Now this is what the mapping means for us: (Remember: feed 1 contract at a time, and we will be computing attention within the sequence)
            # batch_size = Number of contracts, 1 batch = 1 contract;
            # sequence_length = our temporal sequence (basically those are the tokens we compute attention on)
            # embedding_space = our projected features
        self.temporal_pos_encoder = PositionalEncoding(d_model, dropout)
        self.temporal_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=d_model * 4,
                    dropout=dropout,
                    batch_first=True,
                )
                for _ in range(num_layers)
            ]
        )

        # Cross-sectional encoder
        # batch_first = True -> means that we expect a 3D tensor (batch_size, sequence_length, embedding_space)
            # batch_size = number of dataset entries, for us, we will feed 1 file at a time
            # sequence_length = number of contracts
            # embedding_space = the enhanced temporal vector from the last timestamp of each contract
        self.cross_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=d_model * 4,
                    dropout=dropout,
                    batch_first=True,
                )
                for _ in range(num_layers)
            ]
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # x = (batch_size, num_contracts, seq_length, features)
        batch_size, num_contracts, seq_len, num_features = x.shape

        # Transforms to: x = (batch_size * num_contracts, seq_length, features)
        # creates a sequence of contracts (however, our batch is 1, so that has basically no effects - we want to keep entries separate)
        # -1 is a place holder for a dimension that is inferred by (Total_Number_of_Elements / Present_dimensions)
        # e.g. -1 dim = batch_size * num_contracts * seq_length * features / (seq_length * features) = batch_size * num_contracts
        x = x.view(-1, seq_len, num_features)

        x = self.feature_projection(x)
        x = self.temporal_pos_encoder(x)

        for layer in self.temporal_layers:
            x = layer(x)

        # back to (B, C, T, d_model)
        x = x.view(batch_size, num_contracts, seq_len, self.d_model)

        # temporal context: last time step per contract
        temporal = x[:, :, -1, :]            # (B, C, d_model)

        # cross-sectional over contracts
        cross = temporal.clone()             # (B, C, d_model)
        for layer in self.cross_layers:
            cross = layer(cross)

        # concat temporal + cross
        embeddings = torch.cat([temporal, cross], dim=-1)  # (B, C, 2*d_model)
        return embeddings

class TTABackbone(nn.Module):
    """
    Temporal-only transformer backbone (no cross-sectional attention).

    Input:  x (B, C, T, F)
    Output: embeddings (B, C, d_model)
    """

    def __init__(
        self,
        input_size: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.out_dim = d_model

        self.feature_projection = nn.Sequential(
            nn.Linear(input_size, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.temporal_pos_encoder = PositionalEncoding(d_model, dropout)
        self.temporal_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=d_model * 4,
                    dropout=dropout,
                    batch_first=True,
                )
                for _ in range(num_layers)
            ]
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        batch_size, num_contracts, seq_len, num_features = x.shape

        x = x.view(-1, seq_len, num_features)

        x = self.feature_projection(x)
        x = self.temporal_pos_encoder(x)

        for layer in self.temporal_layers:
            x = layer(x)

        # (B, C, T, d_model)
        x = x.view(batch_size, num_contracts, seq_len, self.d_model)

        # last time step per contract: (B, C, d_model)
        temporal_context = x[:, :, -1, :]

        return temporal_context
    
class LSTMBackbone(nn.Module):
    """
    Temporal-only LSTM backbone.

    Input:  x (B, C, T, F)
    Output: embeddings (B, C, D) with D = 2 * d_model (bidirectional)
    """

    def __init__(
        self,
        input_size: int,
        d_model: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool = True,
    ) -> None:
        
        super().__init__()
        self.d_model = d_model
        self.bidirectional = bidirectional
        self.out_dim = d_model * (2 if bidirectional else 1)

        # same feature projection style as transformers
        self.feature_projection = nn.Sequential(
            nn.Linear(input_size, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True,
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if "weight" in name:
                    nn.init.xavier_uniform_(param)
                elif "bias" in name:
                    nn.init.zeros_(param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        B, C, T, F = x.shape

        # (B*C, T, F)
        x = x.view(-1, T, F)

        # (B*C, T, d_model)
        x = self.feature_projection(x)

        # lstm_out: (B*C, T, out_dim)
        lstm_out, _ = self.lstm(x)

        # take last time step: (B*C, out_dim)
        last_t = lstm_out[:, -1, :]

        # reshape back: (B, C, out_dim)
        emb = last_t.view(B, C, self.out_dim)
        return emb
    

class FullTempTCTABackbone(nn.Module):
    """
    TCTA variant that uses the full temporal sequence (all T days) as temporal
    context (flattened), plus cross-sectional attention on the last timestamp.

    Input:  x (B, C, T, F)
    Output: embeddings (B, C, D) with D = d_model * (T + 1)
            (T * d_model from temporal context + d_model from cross-sectional)
    """

    def __init__(
        self,
        input_size: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dropout: float,
        seq_length: int,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_length = seq_length
        self.out_dim = d_model * (seq_length + 1)

        # Feature projection
        self.feature_projection = nn.Sequential(
            nn.Linear(input_size, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Temporal encoder
        self.temporal_pos_encoder = PositionalEncoding(d_model, dropout)
        self.temporal_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=d_model * 4,
                    dropout=dropout,
                    batch_first=True,
                )
                for _ in range(num_layers)
            ]
        )

        # Cross-sectional encoder
        self.cross_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=d_model * 4,
                    dropout=dropout,
                    batch_first=True,
                )
                for _ in range(num_layers)
            ]
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, F)
        B, C, T, F = x.shape
        assert T == self.seq_length, f"Expected seq_len={self.seq_length}, got {T}"

        # (B*C, T, F)
        x = x.view(-1, T, F)

        # (B*C, T, d_model)
        x = self.feature_projection(x)
        x = self.temporal_pos_encoder(x)

        for layer in self.temporal_layers:
            x = layer(x)

        # (B, C, T, d_model)
        x = x.view(B, C, T, self.d_model)

        # full temporal context flattened: (B, C, T * d_model)
        temporal_context = x.reshape(B, C, -1)

        # last timestamp for cross-sectional attention: (B, C, d_model)
        last_timestamp = x[:, :, -1, :]
        cross = last_timestamp.clone()

        for layer in self.cross_layers:
            cross = layer(cross)

        # (B, C, T*d_model + d_model) = (B, C, out_dim)
        combined = torch.cat([temporal_context, cross], dim=-1)
        return combined
