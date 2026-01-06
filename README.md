## 1. Project overview (semantic content)

The goal of this project is to build a sequential recommender system that predicts which products a user is most likely to interact with next in an online store, based purely on item ID interaction histories (views, add-to-cart events, purchases).

Concretely, given a sequence of a user’s recent interactions with product IDs, the model outputs a ranked list of candidate items (**top-K next-item recommendation**). This corresponds to real-world use cases such as “You might also like” or personalized carousels on product and home pages.

To keep the project feasible and lightweight (compared to transformer-based models), the implemented core model is **GRU4Rec**, designed for session-based recommendation. For production-style inference, the trained model is **exported to ONNX** and served via **MLflow pyfunc**, using **onnxruntime** for execution.

---

## 2. Key ideas / design choices

- **Training framework**: PyTorch Lightning (reproducible training loop, metrics logging).
- **Model**: GRU4Rec next-item prediction (session-based sequential recommendation).
- **Production preparation**: mandatory **ONNX export**.
- **Inference**: only via **MLflow Serving**, loading a pyfunc model that runs inference in **onnxruntime**.
- **Reproducibility & portability**: data and artifacts are managed via **DVC**, so the reviewer can run inference without Kaggle.

---

## 3. Repository structure

- `retailrocket_gru4rec/`
  - `commands.py` — CLI entry points:
    - `preprocess`, `train`, `evaluate`, `export_mlflow`, `serve_mlflow`
  - `data/` — download + preprocessing logic
  - `models/` — GRU4Rec implementation
  - `training/` — Lightning module, datamodule, callbacks, metrics
  - `evaluation/` — offline evaluation utilities
  - `serving/mlflow_pyfunc.py` — MLflow pyfunc wrapper using ONNX + onnxruntime
  - `utils.py` — helpers (repo root, config composition, DVC pull helper, ONNX wrapper, etc.)
- `configs/` — Hydra configs (`data/`, `model/`, `train/`, `eval/`, `export/`, `serve/`, `logging/`)
- `dvc.yaml` — DVC pipeline definition

### DVC-managed outputs

- `data/processed/` — processed parquet + vocab
- `artifacts/checkpoints/` — Lightning checkpoints
- `artifacts/onnx/` — ONNX export
- `artifacts/mlflow_model/` — MLflow pyfunc model directory (ONNX + vocab inside)
- `plots/` — evaluation reports / curves (if enabled)

---

## 4. Requirements

This repository uses **Poetry** for dependency management. All commands below assume you run tools through Poetry.

Key runtime tools:

- `poetry`
- `dvc` (used as `poetry run dvc ...`)
- `mlflow` (used indirectly by `serve_mlflow`)
- `onnxruntime` (required for inference inside MLflow pyfunc)

---

## 5. Setup

### 5.1 Clone and install dependencies

```bash
git clone https://github.com/ArtemR238/mlops-retailrocket-gru4rec.git
cd mlops-retailrocket-gru4rec

poetry install
```

### 5.2 Configure DVC remote credentials (required to pull data/artifacts)

This project expects the reviewer to pull everything needed from a DVC remote (S3-compatible storage). Set credentials via environment variables (**do not commit secrets to git**):

```bash
export AWS_ACCESS_KEY_ID="***"
export AWS_SECRET_ACCESS_KEY="***"
export AWS_DEFAULT_REGION="ru-central1"
```

If your environment uses a custom S3 endpoint (typical for S3-compatible storage), configure it as well (only if applicable in your setup):

```bash
export AWS_ENDPOINT_URL="https://<your-s3-endpoint>"
```

---

## 6. Train

Training is orchestrated via **DVC stages** (defined in `dvc.yaml`).

### 6.1 Preprocess (build `data/processed/`)

```bash
poetry run dvc repro preprocess
```

Outputs:

- `data/processed/train.parquet`
- `data/processed/val.parquet`
- `data/processed/test.parquet`
- `data/processed/vocab.json`

### 6.2 Train (produces checkpoints + ONNX)

```bash
poetry run dvc repro train
```

Outputs:

- `artifacts/checkpoints/` (best + last checkpoints)
- `artifacts/onnx/` (ONNX model export)

**Important:** ONNX export is a mandatory part of training and happens at the end of the `train` command.

### 6.3 Optional: Evaluate

```bash
poetry run dvc repro evaluate
```

Outputs:

- `plots/` (curves + JSON report, depending on configuration)

---

## 7. Production preparation

### 7.1 ONNX export (mandatory)

ONNX export is performed automatically as the final part of `train`.

Expected location:

```text
artifacts/onnx/
```

### 7.2 Export MLflow model directory (pyfunc + ONNX + vocab)

This creates a local MLflow model directory that packages:

- ONNX model
- vocabulary/spec (`vocab.json`)
- pyfunc wrapper code

```bash
poetry run dvc repro export_mlflow
```

Output:

```text
artifacts/mlflow_model/
```

---

## 8. Infer (MLflow Serving)

This project performs inference **only via MLflow Serving** (no offline inference CLI).

### 8.1 Fast path (recommended for reviewers): inference without training

Pull the ready-to-serve model from DVC (requires credentials):

```bash
poetry run dvc pull artifacts/mlflow_model
```

Start the server (choose a free port, e.g. `5001`):

```bash
poetry run python -m retailrocket_gru4rec.commands serve_mlflow serve.host=127.0.0.1 serve.port=5001
```

The service exposes the standard MLflow scoring endpoint:

- `POST /invocations`

### 8.2 Full path: preprocess → train → export → serve

```bash
poetry run dvc repro preprocess
poetry run dvc repro train
poetry run dvc repro export_mlflow

poetry run python -m retailrocket_gru4rec.commands serve_mlflow serve.host=127.0.0.1 serve.port=5001
```

---

## 9. Request / response format

### 9.1 Request format

The serving model expects a pandas-like dataframe with a single required column:

- `session`: a list of item IDs (`int`)

Example request (two sessions) using MLflow JSON format:

```bash
curl -s -X POST "http://127.0.0.1:5001/invocations"   -H "Content-Type: application/json"   -d '{
    "dataframe_records": [
      {"session": [1, 2, 3, 4]},
      {"session": [10, 11]}
    ]
  }'
```

### 9.2 Response format

The response is a dataframe-like JSON with two columns:

- `recommendations`: top-K item IDs per input row
- `scores`: corresponding model scores per item

Example (shape and values will vary):

```json
[
  {
    "recommendations": [123, 42, 555, 9, 77],
    "scores": [12.3, 11.1, 10.8, 10.2, 9.9]
  },
  {
    "recommendations": [11, 10, 99, 5, 1],
    "scores": [8.4, 8.1, 7.9, 7.2, 6.8]
  }
]
```

---

## 10. DVC pipeline stages (reference)

Current `dvc.yaml` stages:

- `preprocess` → outputs `data/processed`
- `train` → outputs `artifacts/checkpoints`, `artifacts/onnx`
- `evaluate` → outputs `plots`
- `export_mlflow` → outputs `artifacts/mlflow_model`

You can run a specific stage by name:

```bash
poetry run dvc repro export_mlflow
```

---
