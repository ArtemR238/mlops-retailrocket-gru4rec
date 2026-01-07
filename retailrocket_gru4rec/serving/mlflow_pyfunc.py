from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

import mlflow.pyfunc
import numpy as np
import pandas as pd

from retailrocket_gru4rec.data.dataset import VocabSpec, load_vocab


@dataclass(frozen=True)
class PyfuncParams:
    topk: int = 20


class GRU4RecOnnxPyFunc(mlflow.pyfunc.PythonModel):
    def __init__(self, params: PyfuncParams):
        self.params = params
        self._spec: Optional[VocabSpec] = None

        self._sess: Any = None
        self._input_name: Optional[str] = None
        self._output_name: Optional[str] = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_sess"] = None
        state["_input_name"] = None
        state["_output_name"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def load_context(self, context) -> None:
        try:
            import onnxruntime as ort  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "onnxruntime is not installed. Add it to dependencies: `poetry add onnxruntime`."
            ) from e

        onnx_path = Path(context.artifacts["onnx_model"])
        vocab_path = Path(context.artifacts["vocab"])

        self._spec = load_vocab(vocab_path)

        self._sess = ort.InferenceSession(
            str(onnx_path),
            providers=["CPUExecutionProvider"],
        )
        self._input_name = self._sess.get_inputs()[0].name
        self._output_name = self._sess.get_outputs()[0].name

    def predict(self, context, model_input):
        if (
            self._sess is None
            or self._spec is None
            or self._input_name is None
            or self._output_name is None
        ):
            raise RuntimeError("Model is not loaded. load_context() was not called.")

        if not isinstance(model_input, pd.DataFrame):
            raise TypeError("Input must be a pandas.DataFrame")

        if "session" not in model_input.columns:
            raise ValueError('Input DataFrame must contain a "session" column')

        sessions: List[List[int]] = model_input["session"].tolist()

        norm_sessions: List[List[int]] = []
        for s in sessions:
            if s is None or (isinstance(s, float) and np.isnan(s)):
                norm_sessions.append([])
            else:
                norm_sessions.append([int(x) for x in s])

        max_len = max((len(s) for s in norm_sessions), default=1)
        max_len = max(max_len, 1)

        arr = np.full((len(norm_sessions), max_len), fill_value=self._spec.pad_id, dtype=np.int64)
        for i, s in enumerate(norm_sessions):
            trunc = s[-max_len:]
            if trunc:
                arr[i, : len(trunc)] = np.asarray(trunc, dtype=np.int64)

        logits = self._sess.run([self._output_name], {self._input_name: arr})[0]  # (B, V)
        logits = np.asarray(logits)

        if 0 <= self._spec.pad_id < logits.shape[1]:
            logits[:, self._spec.pad_id] = -1e9
        if 0 <= self._spec.oov_id < logits.shape[1]:
            logits[:, self._spec.oov_id] = -1e9

        k = int(self.params.topk)
        k = max(1, min(k, logits.shape[1]))

        idx_part = np.argpartition(-logits, kth=k - 1, axis=1)[:, :k]
        scores_part = np.take_along_axis(logits, idx_part, axis=1)
        order = np.argsort(-scores_part, axis=1)
        topk_idx = np.take_along_axis(idx_part, order, axis=1)
        topk_scores = np.take_along_axis(scores_part, order, axis=1)

        return pd.DataFrame({"recommendations": topk_idx.tolist(), "scores": topk_scores.tolist()})


def save_pyfunc_model(
    model_dir: Path,
    onnx_path: Path,
    vocab_path: Path,
    topk: int = 20,
    code_paths: Optional[List[str]] = None,
) -> None:
    if model_dir.exists():
        shutil.rmtree(model_dir)

    params = PyfuncParams(topk=int(topk))

    artifacts = {
        "onnx_model": str(onnx_path),
        "vocab": str(vocab_path),
    }

    input_example = pd.DataFrame({"session": [[1, 2, 3], [10, 11]]})

    mlflow.pyfunc.save_model(
        path=str(model_dir),
        python_model=GRU4RecOnnxPyFunc(params=params),
        artifacts=artifacts,
        code_paths=code_paths or [],
        input_example=input_example,
    )
