from __future__ import annotations

import torch
from torch import nn


class PersonalizationAdapter(nn.Module):
    """Small adapter for user-specific fine-tuning."""

    def __init__(self, dim: int, bottleneck: int = 16):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(dim, bottleneck),
            nn.ReLU(),
            nn.Linear(bottleneck, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.adapter(x)


class BiLSTMMultiHead(nn.Module):
    """
    Task-5/6 model:
    - exercise classification head
    - quality regression head
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 1,
        num_classes: int = 1,
        adapter_bottleneck: int = 16,
    ):
        super().__init__()
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        enc_dim = hidden_dim * 2
        self.adapter = PersonalizationAdapter(enc_dim, bottleneck=adapter_bottleneck)
        self.class_head = nn.Linear(enc_dim, num_classes)
        self.quality_head = nn.Sequential(nn.Linear(enc_dim, enc_dim // 2), nn.ReLU(), nn.Linear(enc_dim // 2, 1))

    def forward(self, x: torch.Tensor, use_adapter: bool = False):
        out, _ = self.encoder(x)  # (B,T,2H)
        pooled = out.mean(dim=1)
        if use_adapter:
            pooled = self.adapter(pooled)
        logits = self.class_head(pooled)
        quality = self.quality_head(pooled).squeeze(-1)
        return logits, quality


def freeze_base_model_for_personalization(model: BiLSTMMultiHead):
    for p in model.encoder.parameters():
        p.requires_grad = False
    for p in model.class_head.parameters():
        p.requires_grad = False
    for p in model.quality_head.parameters():
        p.requires_grad = False
    for p in model.adapter.parameters():
        p.requires_grad = True

