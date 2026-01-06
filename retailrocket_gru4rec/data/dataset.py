from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class VocabSpec:
    pad_id: int
    oov_id: int
    vocab_size: int


def load_vocab(vocab_path: Path) -> VocabSpec:

    payload = json.loads(vocab_path.read_text(encoding="utf-8"))
    pad_id = int(payload["pad_id"])
    oov_id = int(payload["oov_id"])
    vocab_size = max(payload["item_to_id"].values(), default=1) + 1
    return VocabSpec(pad_id=pad_id, oov_id=oov_id, vocab_size=vocab_size)


class SessionNextItemDataset(Dataset):
    """For each session, use prefix as input and last item as target next-item.

    items: list[int] (already encoded)
    Returns:
      - sequence: LongTensor [L-1]
      - target_item: LongTensor []
    """

    def __init__(self, parquet_path: Path):
        self.df = pd.read_parquet(parquet_path)
        if "items" not in self.df.columns:
            raise ValueError("Expected 'items' column in parquet")
        self.items = self.df["items"].tolist()

        self.items = [seq for seq in self.items if len(seq) >= 2]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        seq = self.items[idx]
        sequence = torch.tensor(seq[:-1], dtype=torch.long)
        target_item = torch.tensor(seq[-1], dtype=torch.long)
        return {"sequence": sequence, "target_item": target_item}


def collate_pad(batch: List[Dict[str, Any]], pad_id: int) -> Dict[str, torch.Tensor]:
    sequences = [x["sequence"] for x in batch]
    targets = torch.stack([x["target_item"] for x in batch], dim=0)

    lengths = torch.tensor([len(s) for s in sequences], dtype=torch.long)
    max_len = int(lengths.max().item())

    padded = torch.full((len(sequences), max_len), pad_id, dtype=torch.long)
    for i, seq in enumerate(sequences):
        padded[i, : len(seq)] = seq

    return {"sequence": padded, "lengths": lengths, "target_item": targets}
