from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pytorch_lightning as pl
import torch
import torch.nn as nn

from retailrocket_gru4rec.models.gru4rec import GRU4Rec, GRU4RecConfig
from retailrocket_gru4rec.training.metrics import mrr_at_k, ndcg_at_k, recall_at_k


@dataclass(frozen=True)
class OptimConfig:
    learning_rate: float
    weight_decay: float


class GRU4RecLightning(pl.LightningModule):
    def __init__(
        self,
        model_cfg: GRU4RecConfig,
        optim_cfg: OptimConfig,
        k_metrics: List[int],
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["k_metrics"])
        self.model = GRU4Rec(model_cfg)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=model_cfg.pad_id)
        self.k_metrics = k_metrics

    def forward(self, sequence: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        return self.model(sequence=sequence, lengths=lengths)

    def training_step(self, batch, batch_idx):
        logits = self(sequence=batch["sequence"], lengths=batch["lengths"])
        loss = self.loss_fn(logits, batch["target_item"])
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(sequence=batch["sequence"], lengths=batch["lengths"])
        loss = self.loss_fn(logits, batch["target_item"])
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

        for k in self.k_metrics:
            self.log(f"val_recall@{k}", recall_at_k(logits, batch["target_item"], k), on_epoch=True)
            self.log(f"val_mrr@{k}", mrr_at_k(logits, batch["target_item"], k), on_epoch=True)
            self.log(f"val_ndcg@{k}", ndcg_at_k(logits, batch["target_item"], k), on_epoch=True)

    def test_step(self, batch, batch_idx):
        logits = self(sequence=batch["sequence"], lengths=batch["lengths"])
        loss = self.loss_fn(logits, batch["target_item"])
        self.log("test_loss", loss, on_epoch=True)

        for k in self.k_metrics:
            self.log(
                f"test_recall@{k}", recall_at_k(logits, batch["target_item"], k), on_epoch=True
            )
            self.log(f"test_mrr@{k}", mrr_at_k(logits, batch["target_item"], k), on_epoch=True)
            self.log(f"test_ndcg@{k}", ndcg_at_k(logits, batch["target_item"], k), on_epoch=True)

    def configure_optimizers(self):
        optim_cfg = self.hparams["optim_cfg"]
        return torch.optim.AdamW(
            self.parameters(),
            lr=float(optim_cfg["learning_rate"]),
            weight_decay=float(optim_cfg["weight_decay"]),
        )
