from __future__ import annotations

import json
import random
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch
import torch.nn as nn
from dvc.repo import Repo
from hydra import compose, initialize_config_dir
from mlflow.tracking import MlflowClient
from omegaconf import DictConfig
from pytorch_lightning.loggers import CSVLogger, MLFlowLogger


def find_repo_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    while True:
        if (current / "pyproject.toml").exists():
            return current
        if current.parent == current:
            raise RuntimeError("Could not find repository root (pyproject.toml not found).")
        current = current.parent


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_git_commit_id(repo_root: Path) -> Optional[str]:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(repo_root))
        return out.decode("utf-8").strip()
    except Exception:
        return None


def _normalize_overrides(overrides) -> List[str]:
    if overrides is None:
        return []
    if isinstance(overrides, tuple):
        return [x for x in overrides if x]
    return [x for x in overrides if x]


def _build_logger(cfg: DictConfig, repo_root: Path):
    tracking_uri = str(cfg.logging.tracking_uri)
    experiment_name = str(cfg.logging.experiment_name)
    run_name = (
        str(cfg.logging.run_name) if "run_name" in cfg.logging and cfg.logging.run_name else None
    )

    try:
        client = MlflowClient(tracking_uri=tracking_uri)
        client.get_experiment_by_name(experiment_name)

        return MLFlowLogger(
            experiment_name=experiment_name,
            tracking_uri=tracking_uri,
            run_name=run_name,
        )
    except Exception as e:
        fail = (
            bool(cfg.logging.fail_if_unavailable) if "fail_if_unavailable" in cfg.logging else True
        )
        if fail:
            raise RuntimeError(
                "MLflow недоступен по tracking_uri="
                f"{tracking_uri}. Запустите `mlflow server` на этой машине "
                "или пробросьте порт 8080, либо поставьте logging.fail_if_unavailable=false."
            ) from e

        logs_dir = repo_root / "artifacts" / "csv_logs"
        return CSVLogger(save_dir=str(logs_dir), name="pl")


def _flatten_for_mlflow(d: Dict[str, Any], prefix: str = "") -> Dict[str, str]:
    out: Dict[str, str] = {}
    for k, v in d.items():
        key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
        if isinstance(v, dict):
            out.update(_flatten_for_mlflow(v, key))
        elif isinstance(v, (list, tuple)):
            out[key] = json.dumps(v, ensure_ascii=False)
        else:
            out[key] = str(v)
    return out


def _compose_config(overrides: List[str]) -> DictConfig:
    repo_root = find_repo_root()
    config_dir = repo_root / "configs"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name="config", overrides=overrides)
    return cfg


def _dvc_pull(repo_root: Path, targets: Iterable[Path]) -> None:
    rel_targets: List[str] = []
    for t in targets:
        try:
            rel_targets.append(str(t.relative_to(repo_root)))
        except Exception:
            rel_targets.append(str(t))

    if not rel_targets:
        return

    try:

        with Repo(str(repo_root)) as repo:
            for t in rel_targets:
                try:
                    repo.pull(targets=[t])
                except Exception:
                    pass
        return
    except Exception:
        pass

    try:
        subprocess.run(["dvc", "pull", *rel_targets], check=False)
    except FileNotFoundError:
        pass


class GRU4RecOnnxWrapper(nn.Module):
    """
    ONNX-friendly wrapper: single input `sequence` -> `logits`.
    ВАЖНО: sequence должен быть RIGHT-padded (PAD в конце).
    """

    def __init__(self, model: nn.Module, pad_id: int):
        super().__init__()
        self.item_embedding = model.item_embedding
        self.gru = model.gru
        self.dropout = model.dropout
        self.output = model.output

        self.register_buffer(
            "pad_id_t", torch.tensor(int(pad_id), dtype=torch.long), persistent=False
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        mask = sequence.ne(self.pad_id_t)
        lengths = mask.sum(dim=1).clamp(min=1)

        emb = self.item_embedding(sequence)

        out, _ = self.gru(emb)

        B, T, H = out.shape
        idx = (lengths - 1).view(B, 1, 1).expand(B, 1, H)
        h_last = out.gather(1, idx).squeeze(1)

        h_last = self.dropout(h_last)
        logits = self.output(h_last)
        return logits
