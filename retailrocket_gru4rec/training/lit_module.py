from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
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


def _register_torch_safe_globals() -> None:
    """
    torch>=2.6 safe-unpickler (weights_only=True) блокирует кастомные классы.
    Разрешаем наши dataclass-ы, чтобы чекпойнты Lightning грузились без ошибок.
    """
    try:
        add_safe_globals = torch.serialization.add_safe_globals
    except AttributeError:
        # torch < 2.6
        return

    try:
        add_safe_globals([GRU4RecConfig, OptimConfig])
    except Exception:
        # Не ломаем импорт из-за allowlist
        return


_register_torch_safe_globals()


class GRU4RecLightning(pl.LightningModule):
    def __init__(
        self,
        model_cfg: GRU4RecConfig,
        optim_cfg: OptimConfig,
        k_metrics: List[int],
    ):
        super().__init__()

        self.model_cfg = model_cfg
        self.optim_cfg = optim_cfg
        self.k_metrics = list(k_metrics)

        # В checkpoint кладём только JSON-сериализуемые структуры,
        # чтобы weights_only unpickler не падал.
        model_cfg_hp = asdict(model_cfg) if is_dataclass(model_cfg) else dict(model_cfg)
        optim_cfg_hp = asdict(optim_cfg) if is_dataclass(optim_cfg) else dict(optim_cfg)

        self.save_hyperparameters(
            {
                "model_cfg": model_cfg_hp,
                "optim_cfg": optim_cfg_hp,
                "k_metrics": self.k_metrics,
            }
        )

        self.model = GRU4Rec(model_cfg)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=model_cfg.pad_id)

    def forward(self, sequence: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        return self.model(sequence=sequence, lengths=lengths)

    def training_step(self, batch, batch_idx):
        logits = self(sequence=batch["sequence"], lengths=batch["lengths"])
        loss = self.loss_fn(logits, batch["target_item"])
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(sequence=batch["sequence"], lengths=batch["lengths"])
        loss = self.loss_fn(logits, batch["target_item"])
        self.log("val/loss", loss, on_epoch=True, prog_bar=True)

        for k in self.k_metrics:
            self.log(
                f"val/recall_at_{k}",
                recall_at_k(logits, batch["target_item"], k),
                on_epoch=True,
            )
            self.log(
                f"val/mrr_at_{k}",
                mrr_at_k(logits, batch["target_item"], k),
                on_epoch=True,
            )
            self.log(
                f"val/ndcg_at_{k}",
                ndcg_at_k(logits, batch["target_item"], k),
                on_epoch=True,
            )

    def test_step(self, batch, batch_idx):
        logits = self(sequence=batch["sequence"], lengths=batch["lengths"])
        loss = self.loss_fn(logits, batch["target_item"])
        self.log("test/loss", loss, on_epoch=True)

        for k in self.k_metrics:
            self.log(
                f"test/recall_at_{k}",
                recall_at_k(logits, batch["target_item"], k),
                on_epoch=True,
            )
            self.log(
                f"test/mrr_at_{k}",
                mrr_at_k(logits, batch["target_item"], k),
                on_epoch=True,
            )
            self.log(
                f"test/ndcg_at_{k}",
                ndcg_at_k(logits, batch["target_item"], k),
                on_epoch=True,
            )

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=float(self.optim_cfg.learning_rate),
            weight_decay=float(self.optim_cfg.weight_decay),
        )
