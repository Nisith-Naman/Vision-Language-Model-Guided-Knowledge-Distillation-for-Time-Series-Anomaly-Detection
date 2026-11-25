"""Recurrent student for time-series anomaly detection and step localization.

Architecture (all optional parts default to the original GRU student so older
checkpoints still load):

    Input            (B, L, F)
    [Conv front-end] two 1-D convolutions; the second may downsample by ``downsample``
    GRU backbone     (B, L', H·dirs)
    Window head      pooled sequence representation -> 2 logits      (B, 2)
    Step head        per-timestep representation    -> 1 logit/step  (B, L)
    [Embedding head] pooled representation -> teacher embedding space (B, E)

Which signal trains which head:
    * window head  <- ground-truth window label (cross-entropy)
                      + teacher soft label        (temperature-scaled KL)
    * step head    <- ground-truth timestep labels (weighted BCE)
    * embed head   <- frozen CLIP image embedding (cosine feature distillation)
    The embedding head is used only during training; inference needs the GRU only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    """Configuration dataclass for GRUStudent model hyperparameters."""

    input_size: int
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.3
    seq_len: int = 128
    bidirectional: bool = False
    conv_channels: int = 0
    conv_kernel: int = 5
    downsample: int = 1
    pooling: str = "mean"
    embed_dim: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        """Deserialize configuration from dictionary."""
        return cls(
            input_size=int(data["input_size"]),
            hidden_size=int(data.get("hidden_size", 64)),
            num_layers=int(data.get("num_layers", 2)),
            dropout=float(data.get("dropout", 0.3)),
            seq_len=int(data.get("seq_len", 128)),
            bidirectional=bool(data.get("bidirectional", False)),
            conv_channels=int(data.get("conv_channels", 0)),
            conv_kernel=int(data.get("conv_kernel", 5)),
            downsample=int(data.get("downsample", 1)),
            pooling=str(data.get("pooling", "mean")),
            embed_dim=int(data.get("embed_dim", 0)),
        )


class GRUStudent(nn.Module):
    """GRU student network for window classification and step localization."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        seq_len: int = 128,
        bidirectional: bool = False,
        conv_channels: int = 0,
        conv_kernel: int = 5,
        downsample: int = 1,
        pooling: str = "mean",
        embed_dim: int = 0,
    ) -> None:
        super().__init__()
        if pooling not in ("mean", "meanmax"):
            raise ValueError(f"pooling must be 'mean' or 'meanmax', got {pooling!r}")
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.dropout_rate = float(dropout)
        self.seq_len = int(seq_len)
        self.bidirectional = bool(bidirectional)
        self.conv_channels = int(conv_channels)
        self.conv_kernel = int(conv_kernel)
        self.downsample = max(1, int(downsample))
        self.pooling = pooling
        self.embed_dim = int(embed_dim)

        # Optional convolutional front-end: learns local shape/frequency filters
        # before the recurrent layer and can shorten long windows for speed.
        if self.conv_channels > 0:
            pad = self.conv_kernel // 2
            self.frontend = nn.Sequential(
                nn.Conv1d(self.input_size, self.conv_channels, self.conv_kernel, padding=pad),
                nn.BatchNorm1d(self.conv_channels),
                nn.GELU(),
                nn.Conv1d(
                    self.conv_channels, self.conv_channels, self.conv_kernel,
                    stride=self.downsample, padding=pad,
                ),
                nn.BatchNorm1d(self.conv_channels),
                nn.GELU(),
            )
            gru_input = self.conv_channels
        else:
            if self.downsample != 1:
                raise ValueError("downsample > 1 requires conv_channels > 0")
            self.frontend = None
            gru_input = self.input_size

        gru_dropout = dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=gru_input,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=gru_dropout,
            bidirectional=bidirectional,
        )

        feature_dim = hidden_size * (2 if bidirectional else 1)
        pooled_dim = feature_dim * (2 if pooling == "meanmax" else 1)
        self.drop = nn.Dropout(p=dropout)

        self.window_head = nn.Sequential(
            nn.Linear(pooled_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, 2),
        )
        self.step_head = nn.Sequential(
            nn.Linear(feature_dim, max(16, hidden_size // 2)),
            nn.ReLU(),
            nn.Linear(max(16, hidden_size // 2), 1),
        )
        self.embed_head = (
            nn.Sequential(nn.Linear(pooled_dim, hidden_size * 2), nn.GELU(), nn.Linear(hidden_size * 2, self.embed_dim))
            if self.embed_dim > 0
            else None
        )

    def forward(
        self, x: torch.Tensor, return_embedding: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Return ``(window_logits (B,2), step_logits (B,L))`` and optionally the
        projected embedding ``(B,E)`` used for feature distillation."""
        batch_size, seq_len, feat_dim = x.shape
        if feat_dim != self.input_size:
            raise ValueError(
                f"Expected input_size={self.input_size}, got tensor with {feat_dim} features"
            )

        h = x
        if self.frontend is not None:
            h = self.frontend(x.transpose(1, 2)).transpose(1, 2)  # (B, L', C)

        out, _ = self.gru(h)  # (B, L', feature_dim)

        if self.pooling == "meanmax":
            pooled = torch.cat([out.mean(dim=1), out.amax(dim=1)], dim=-1)
        else:
            pooled = out.mean(dim=1)
        window_logits = self.window_head(self.drop(pooled))

        step_logits = self.step_head(self.drop(out)).squeeze(-1)  # (B, L')
        if step_logits.shape[1] != seq_len:
            step_logits = F.interpolate(
                step_logits.unsqueeze(1), size=seq_len, mode="linear", align_corners=False
            ).squeeze(1)

        if return_embedding:
            if self.embed_head is None:
                raise RuntimeError("Model was built with embed_dim=0; no embedding head.")
            return window_logits, step_logits, self.embed_head(pooled)
        return window_logits, step_logits

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """Return window softmax probabilities (B,2) and step sigmoid scores (B,L)."""
        was_training = self.training
        self.eval()
        device = next(self.parameters()).device
        window_logits, step_logits = self.forward(x.to(device))
        window_proba = F.softmax(window_logits, dim=-1).cpu().numpy()
        step_scores = torch.sigmoid(step_logits).cpu().numpy()
        if was_training:
            self.train()
        return window_proba, step_scores

    def parameter_count(self) -> int:
        """Return the number of trainable parameters used at inference
        (the training-only embedding head is excluded)."""
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if self.embed_head is not None:
            total -= sum(p.numel() for p in self.embed_head.parameters() if p.requires_grad)
        return total

    def get_config(self) -> ModelConfig:
        """Retrieve model configuration."""
        return ModelConfig(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout_rate,
            seq_len=self.seq_len,
            bidirectional=self.bidirectional,
            conv_channels=self.conv_channels,
            conv_kernel=self.conv_kernel,
            downsample=self.downsample,
            pooling=self.pooling,
            embed_dim=self.embed_dim,
        )
