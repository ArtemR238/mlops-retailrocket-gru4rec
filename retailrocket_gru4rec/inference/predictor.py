from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import torch

from retailrocket_gru4rec.data.dataset import collate_pad, load_vocab
from retailrocket_gru4rec.models.gru4rec import GRU4Rec, GRU4RecConfig


@dataclass(frozen=True)
class InferenceResult:
    topk_items: List[List[int]]
    topk_scores: List[List[float]]


def load_model_from_ckpt(
    checkpoint_path: Path,
    vocab_size: int,
    pad_id: int,
    embedding_dim: int,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
    device: str,
) -> GRU4Rec:
    # For simplicity, load state_dict saved by Lightning checkpoints
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"]
    # strip 'model.' prefix used in LightningModule
    model_state = {
        k.replace("model.", ""): v for k, v in state_dict.items() if k.startswith("model.")
    }

    model = GRU4Rec(
        GRU4RecConfig(
            vocab_size=vocab_size,
            pad_id=pad_id,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
    )
    model.load_state_dict(model_state, strict=True)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict_topk(
    model: GRU4Rec,
    sequences: List[List[int]],
    pad_id: int,
    k: int,
    device: str,
) -> InferenceResult:
    batch = [
        {"sequence": torch.tensor(seq, dtype=torch.long), "target_item": torch.tensor(0)}
        for seq in sequences
    ]
    collated = collate_pad(batch, pad_id=pad_id)
    seq = collated["sequence"].to(device)
    lengths = collated["lengths"].to(device)
    logits = model(seq, lengths)  # [B, V]
    topk_scores, topk_indices = torch.topk(logits, k=k, dim=1)
    return InferenceResult(
        topk_items=topk_indices.cpu().tolist(),
        topk_scores=topk_scores.cpu().tolist(),
    )


def run_offline_inference(
    vocab_path: Path,
    checkpoint_path: Path,
    request_path: Path,
    output_path: Path,
    model_params: dict,
    device: str,
) -> None:
    spec = load_vocab(vocab_path)
    payload = json.loads(request_path.read_text(encoding="utf-8"))
    sequences = payload["sequences"]
    k = int(payload.get("k", 20))

    model = load_model_from_ckpt(
        checkpoint_path=checkpoint_path,
        vocab_size=spec.vocab_size,
        pad_id=spec.pad_id,
        embedding_dim=int(model_params["embedding_dim"]),
        hidden_dim=int(model_params["hidden_dim"]),
        num_layers=int(model_params["num_layers"]),
        dropout=float(model_params["dropout"]),
        device=device,
    )

    result = predict_topk(model=model, sequences=sequences, pad_id=spec.pad_id, k=k, device=device)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {"topk_items": result.topk_items, "topk_scores": result.topk_scores}, ensure_ascii=False
        ),
        encoding="utf-8",
    )
