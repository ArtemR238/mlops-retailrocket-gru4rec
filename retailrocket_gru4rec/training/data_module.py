from __future__ import annotations

from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from retailrocket_gru4rec.data.dataset import SessionNextItemDataset, collate_pad


class RetailRocketDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_path: Path,
        val_path: Path,
        test_path: Path,
        max_session_len: int,
        batch_size: int,
        num_workers: int,
        seed: int,
        pad_id: int = 0,
    ) -> None:
        super().__init__()
        self.train_path = Path(train_path)
        self.val_path = Path(val_path)
        self.test_path = Path(test_path)

        self.max_session_len = int(max_session_len)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.seed = int(seed)
        self.pad_id = int(pad_id)

        self._train_ds: Optional[SessionNextItemDataset] = None
        self._val_ds: Optional[SessionNextItemDataset] = None
        self._test_ds: Optional[SessionNextItemDataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit"):
            self._train_ds = SessionNextItemDataset(self.train_path)
            self._val_ds = SessionNextItemDataset(self.val_path)

        if stage in (None, "test", "predict"):
            self._test_ds = SessionNextItemDataset(self.test_path)

    def train_dataloader(self) -> DataLoader:
        assert self._train_ds is not None
        return DataLoader(
            self._train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=lambda b: collate_pad(b, pad_id=self.pad_id),
        )

    def val_dataloader(self) -> DataLoader:
        assert self._val_ds is not None
        return DataLoader(
            self._val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=lambda b: collate_pad(b, pad_id=self.pad_id),
        )

    def test_dataloader(self) -> DataLoader:
        assert self._test_ds is not None
        return DataLoader(
            self._test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=lambda b: collate_pad(b, pad_id=self.pad_id),
        )
