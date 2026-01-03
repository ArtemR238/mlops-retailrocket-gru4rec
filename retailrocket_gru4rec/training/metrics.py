from __future__ import annotations

from typing import Dict, List

import torch


def recall_at_k(logits: torch.Tensor, targets: torch.Tensor, k: int) -> torch.Tensor:
    # logits: [B, V], targets: [B]
    topk = torch.topk(logits, k=k, dim=1).indices  # [B, k]
    hit = (topk == targets.unsqueeze(1)).any(dim=1).float()
    return hit.mean()


def mrr_at_k(logits: torch.Tensor, targets: torch.Tensor, k: int) -> torch.Tensor:
    topk = torch.topk(logits, k=k, dim=1).indices  # [B, k]
    # ranks: 1..k if found else 0
    matches = topk == targets.unsqueeze(1)
    # first match index
    ranks = torch.where(
        matches.any(dim=1), matches.float().argmax(dim=1) + 1, torch.zeros_like(targets)
    )
    ranks = ranks.float()
    mrr = torch.where(ranks > 0, 1.0 / ranks, torch.zeros_like(ranks))
    return mrr.mean()


def ndcg_at_k(logits: torch.Tensor, targets: torch.Tensor, k: int) -> torch.Tensor:
    topk = torch.topk(logits, k=k, dim=1).indices
    matches = topk == targets.unsqueeze(1)
    ranks = torch.where(
        matches.any(dim=1), matches.float().argmax(dim=1) + 1, torch.zeros_like(targets)
    )
    ranks = ranks.float()
    dcg = torch.where(ranks > 0, 1.0 / torch.log2(ranks + 1.0), torch.zeros_like(ranks))
    # ideal DCG is always 1 (relevant item at rank 1)
    return dcg.mean()
