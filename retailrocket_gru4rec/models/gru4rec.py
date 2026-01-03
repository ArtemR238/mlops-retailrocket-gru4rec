from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn


@dataclass(frozen=True)
class GRU4RecConfig:
    vocab_size: int
    pad_id: int
    embedding_dim: int
    hidden_dim: int
    num_layers: int
    dropout: float


class GRU4Rec(nn.Module):
    """A minimal GRU-based next-item model.

    Input:
      - sequence: [B, L] (padded)
      - lengths:  [B]
    Output:
      - logits:   [B, vocab_size]
    """

    def __init__(self, cfg: GRU4RecConfig):
        super().__init__()
        self.cfg = cfg

        self.item_embedding = nn.Embedding(
            num_embeddings=cfg.vocab_size,
            embedding_dim=cfg.embedding_dim,
            padding_idx=cfg.pad_id,
        )
        self.gru = nn.GRU(
            input_size=cfg.embedding_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(cfg.dropout)
        self.output = nn.Linear(cfg.hidden_dim, cfg.vocab_size)

    def forward(self, sequence: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        emb = self.item_embedding(sequence)  # [B, L, D]
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h_last = self.gru(packed)  # [num_layers, B, H]
        h = h_last[-1]  # [B, H]
        h = self.dropout(h)
        logits = self.output(h)  # [B, V]
        return logits
