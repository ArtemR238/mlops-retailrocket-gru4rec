from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pytorch_lightning as pl


class SaveCurvesCallback(pl.Callback):
    """Save training curves (loss + metrics) into plots_dir as PNGs."""

    def __init__(self, plots_dir: Path):
        super().__init__()
        self.plots_dir = plots_dir
        self.history = {}

    def on_validation_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        metrics = trainer.callback_metrics
        # collect scalar metrics
        for k, v in metrics.items():
            if hasattr(v, "item"):
                self.history.setdefault(k, []).append(float(v.item()))

    def on_fit_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        def plot_series(key: str, fname: str) -> None:
            ys = self.history.get(key)
            if not ys:
                return
            plt.figure()
            plt.plot(range(1, len(ys) + 1), ys)
            plt.xlabel("epoch")
            plt.ylabel(key)
            plt.title(key)
            plt.tight_layout()
            plt.savefig(self.plots_dir / fname)
            plt.close()

        # At least 3 graphs required
        plot_series("train_loss", "train_loss.png")
        plot_series("val_loss", "val_loss.png")
        plot_series("val_recall@20", "val_recall@20.png")
        plot_series("val_mrr@20", "val_mrr@20.png")
        plot_series("val_ndcg@20", "val_ndcg@20.png")
