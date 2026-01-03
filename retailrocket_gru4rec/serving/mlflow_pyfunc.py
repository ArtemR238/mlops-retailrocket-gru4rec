from __future__ import annotations

"""MLflow pyfunc wrapper for GRU4Rec.

This implements a self-contained MLflow model suitable for `mlflow models serve`.

We pass required artifacts (checkpoint + vocab) via MLflow's `artifacts=`
mechanism and keep model hyperparameters inside the PythonModel instance.

Expected input format for inference:
  - pandas.DataFrame with a column "session" containing a list[int] of item ids
    (already encoded with the project vocab).
Example:
  pd.DataFrame({"session": [[12, 53, 9], [10, 1]]})

Output format:
  - pandas.DataFrame with a column "recommendations" containing list[int] of top-k items.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlflow.pyfunc
import numpy as np
import pandas as pd
import torch

from retailrocket_gru4rec.data.dataset import VocabSpec, load_vocab
from retailrocket_gru4rec.inference.predictor import load_model_from_ckpt, predict_topk


@dataclass(frozen=True)
class PyfuncParams:
    """Parameters needed to reconstruct the model at serving time."""

    embedding_dim: int
    hidden_dim: int
    num_layers: int
    dropout: float
    topk: int
    device: str = "cpu"


class GRU4RecPyFunc(mlflow.pyfunc.PythonModel):
    def __init__(self, params: PyfuncParams):
        self.params = params
        self._model: Optional[torch.nn.Module] = None
        self._spec: Optional[VocabSpec] = None

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        checkpoint_path = Path(context.artifacts["checkpoint"])
        vocab_path = Path(context.artifacts["vocab"])

        spec = load_vocab(vocab_path)
        self._spec = spec

        self._model = load_model_from_ckpt(
            checkpoint_path=checkpoint_path,
            vocab_size=spec.vocab_size,
            pad_id=spec.pad_id,
            embedding_dim=self.params.embedding_dim,
            hidden_dim=self.params.hidden_dim,
            num_layers=self.params.num_layers,
            dropout=self.params.dropout,
            device=self.params.device,
        )

    def predict(self, context: mlflow.pyfunc.PythonModelContext, model_input: Any) -> Any:
        if self._model is None or self._spec is None:
            raise RuntimeError("Model is not loaded. load_context() was not called.")

        if not isinstance(model_input, pd.DataFrame):
            raise TypeError("Input must be a pandas.DataFrame")

        if "session" not in model_input.columns:
            raise ValueError('Input DataFrame must contain a "session" column')

        sessions: List[List[int]] = model_input["session"].tolist()

        max_len = max((len(s) for s in sessions), default=1)
        max_len = max(max_len, 1)

        arr = np.full((len(sessions), max_len), fill_value=self._spec.pad_id, dtype=np.int64)
        for i, s in enumerate(sessions):
            trunc = s[-max_len:]
            arr[i, -len(trunc) :] = np.asarray(trunc, dtype=np.int64)

        sequences = torch.from_numpy(arr).long().to(self.params.device)

        topk_items = predict_topk(self._model, sequences=sequences, topk=int(self.params.topk))
        return pd.DataFrame({"recommendations": topk_items.tolist()})


def save_pyfunc_model(
    model_dir: Path,
    checkpoint_path: Path,
    vocab_path: Path,
    model_params: Dict[str, Any],
    topk: int = 20,
    device: str = "cpu",
    code_paths: Optional[List[str]] = None,
) -> None:
    """Save MLflow pyfunc model to a local directory.

    Args:
        model_dir: output dir (will be overwritten if exists)
        checkpoint_path: Lightning checkpoint path
        vocab_path: vocab.json path
        model_params: dict with embedding_dim, hidden_dim, num_layers, dropout
        topk: number of recommendations to return
        device: cpu/cuda
        code_paths: directories to ship with the MLflow model (so it can import this package)
    """
    params = PyfuncParams(
        embedding_dim=int(model_params["embedding_dim"]),
        hidden_dim=int(model_params["hidden_dim"]),
        num_layers=int(model_params["num_layers"]),
        dropout=float(model_params["dropout"]),
        topk=int(topk),
        device=str(device),
    )

    artifacts = {"checkpoint": str(checkpoint_path), "vocab": str(vocab_path)}

    mlflow.pyfunc.save_model(
        path=str(model_dir),
        python_model=GRU4RecPyFunc(params=params),
        artifacts=artifacts,
        code_path=code_paths or [],
    )
