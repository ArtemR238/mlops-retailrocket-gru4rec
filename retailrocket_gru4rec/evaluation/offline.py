from __future__ import annotations

"""Offline evaluation utilities.

This module provides:
  - a simple TopPopular baseline,
  - offline evaluation for GRU4Rec checkpoints,
  - unified metrics (Recall@K, MRR@K, NDCG@K).

All inputs are assumed to be the *processed* parquet files produced by
`retailrocket_gru4rec.data.preprocess`.
"""

from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
import torch

from retailrocket_gru4rec.data.dataset import load_vocab
from retailrocket_gru4rec.inference.predictor import load_model_from_ckpt, predict_topk


def _unwrap_topk(x: Any) -> np.ndarray:
    """
    predict_topk() может вернуть:
      - np.ndarray (B, K)
      - torch.Tensor (B, K)
      - InferenceResult(...) с полем topk_items / items / item_ids / recs / predictions

    Приводим к np.ndarray[int64] (B, K).
    """
    # unwrap common container attrs
    for attr in ("topk_items", "items", "item_ids", "recs", "predictions", "topk"):
        if hasattr(x, attr):
            x = getattr(x, attr)
            break

    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()

    x = np.asarray(x, dtype=np.int64)
    if x.ndim != 2:
        raise ValueError(f"Expected topk shape (B, K), got shape={x.shape}")
    return x


def _iter_items(sessions: Iterable[List[int]], pad_id: int) -> Iterable[int]:
    for seq in sessions:
        for x in seq:
            if x != pad_id:
                yield int(x)


def build_top_popular_items(train_parquet: Path, vocab_path: Path, topn: int = 1000) -> List[int]:
    """Compute global item popularity list from train parquet."""
    spec = load_vocab(vocab_path)
    df = pd.read_parquet(train_parquet, columns=["items"])
    counter = Counter(_iter_items(df["items"].tolist(), pad_id=spec.pad_id))
    return [item for item, _ in counter.most_common(topn)]


def _metrics_from_recs(recs: np.ndarray, targets: np.ndarray, k: int) -> Dict[str, float]:
    """Compute Recall@k, MRR@k, NDCG@k for single-target next-item prediction.

    recs: (N, k) integer item ids
    targets: (N,) integer item id
    """
    assert recs.ndim == 2 and recs.shape[1] >= k
    recs_k = recs[:, :k]

    # hit positions: -1 if not found
    hits = recs_k == targets[:, None]
    hit_any = hits.any(axis=1)

    # rank (1-based) where hit occurs
    ranks = np.argmax(hits, axis=1) + 1
    ranks = np.where(hit_any, ranks, 0)

    recall = float(hit_any.mean())

    mask = ranks > 0
    if mask.any():
        mrr = float((1.0 / ranks[mask]).mean())
        ndcg = float((1.0 / np.log2(ranks[mask] + 1)).mean())
    else:
        mrr = 0.0
        ndcg = 0.0

    return {"recall": recall, "mrr": mrr, "ndcg": ndcg}


def evaluate_top_popular(
    train_parquet: Path,
    test_parquet: Path,
    vocab_path: Path,
    k_list: List[int],
    exclude_seen: bool = True,
) -> Dict[str, Dict[str, float]]:
    """Evaluate TopPopular baseline.

    We take the last item in a session as target and use the prefix as context.
    Recommendations are the global popularity list (optionally excluding items
    already present in the prefix).
    """
    spec = load_vocab(vocab_path)
    max_k = max(k_list)

    popular = build_top_popular_items(
        train_parquet=train_parquet, vocab_path=vocab_path, topn=max_k * 50
    )

    df_test = pd.read_parquet(test_parquet, columns=["items"])
    sessions = [seq for seq in df_test["items"].tolist() if len(seq) >= 2]

    targets = np.asarray([int(seq[-1]) for seq in sessions], dtype=np.int64)

    recs = np.zeros((len(sessions), max_k), dtype=np.int64)

    for i, seq in enumerate(sessions):
        if exclude_seen:
            seen = set(int(x) for x in seq[:-1] if int(x) != spec.pad_id)
            filtered = [x for x in popular if x not in seen]
            recs[i, :] = np.asarray(filtered[:max_k], dtype=np.int64)
        else:
            recs[i, :] = np.asarray(popular[:max_k], dtype=np.int64)

    report: Dict[str, Dict[str, float]] = {}
    for k in k_list:
        report[str(k)] = _metrics_from_recs(recs=recs, targets=targets, k=int(k))
    return report


def evaluate_gru4rec_from_checkpoint(
    checkpoint_path: Path,
    vocab_path: Path,
    model_params: Dict[str, Any],
    test_parquet: Path,
    k_list: List[int],
    batch_size: int = 512,
    device: str = "cpu",
) -> Dict[str, Dict[str, float]]:
    """Evaluate GRU4Rec checkpoint on processed test parquet.

    `model_params` must contain:
      - embedding_dim, hidden_dim, num_layers, dropout
    """
    spec = load_vocab(vocab_path)
    max_k = max(k_list)

    df_test = pd.read_parquet(test_parquet, columns=["items"])
    sessions = [seq for seq in df_test["items"].tolist() if len(seq) >= 2]

    # input is prefix, target is last item
    prefixes = [seq[:-1] for seq in sessions]
    targets = np.asarray([int(seq[-1]) for seq in sessions], dtype=np.int64)

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

    recs = np.zeros((len(prefixes), max_k), dtype=np.int64)

    # batching
    max_session_len = max(len(p) for p in prefixes)
    max_session_len = max(max_session_len, 1)

    def pad_batch(batch_prefixes: List[List[int]]) -> torch.Tensor:
        arr = np.full(
            (len(batch_prefixes), max_session_len), fill_value=spec.pad_id, dtype=np.int64
        )
        for i, seq in enumerate(batch_prefixes):
            trunc = seq[-max_session_len:]
            arr[i, -len(trunc) :] = np.asarray(trunc, dtype=np.int64)
        return torch.from_numpy(arr).long()

    start = 0
    while start < len(prefixes):
        end = min(start + batch_size, len(prefixes))
        batch = prefixes[start:end]
        seqs = pad_batch(batch).to(device)
        topk_items = predict_topk(
            model=model,
            sequences=seqs,
            k=max_k,
            pad_id=spec.pad_id,
            device=device,
        )

        topk_items_np = _unwrap_topk(topk_items)
        recs[start:end, :] = topk_items_np
        start = end

    report: Dict[str, Dict[str, float]] = {}
    for k in k_list:
        report[str(k)] = _metrics_from_recs(recs=recs, targets=targets, k=int(k))
    return report
