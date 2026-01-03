# RetailRocket GRU4Rec (Next-Item Recommendation)

This repository implements a lightweight **sequential recommender system** on the **RetailRocket** clickstream dataset.
The model predicts the **next item** in a user's session given a sequence of past item interactions and returns **top-K**
recommendations.

## Setup

### Prerequisites

- Python 3.10+ recommended
- [Poetry](https://python-poetry.org/) for dependency management
- (Optional) DVC for data versioning: `pip install dvc`

### Install

```bash
poetry install
poetry run pre-commit install
poetry run pre-commit run -a
```

## Train

Training will (1) ensure data is available (via DVC pull if configured, otherwise download),
(2) preprocess into sessions + temporal split, and (3) train a GRU-based next-item model with PyTorch Lightning.

```bash
poetry run rrgr train
```

Override config values via Hydra-style overrides:

```bash
poetry run rrgr train model.hidden_dim=64 train.max_epochs=3 data.min_session_len=3
```

Artifacts:

- checkpoints are stored under `artifacts/`
- plots and training curves are stored under `plots/`

## Production preparation

### Export to ONNX

```bash
poetry run rrgr export_onnx artifacts.checkpoint_path=artifacts/checkpoints/last.ckpt
```

### TensorRT conversion (optional)

A conversion helper script is provided:

```bash
bash scripts/convert_tensorrt.sh artifacts/model.onnx artifacts/model.trt
```

## Infer

Offline inference (batch) reads a JSON with a list of sequences and returns top-K item IDs:

```bash
poetry run rrgr infer infer.input_path=examples/infer_request.json infer.output_path=artifacts/infer_response.json
```

Example request format:

```json
{
  "sequences": [
    [12, 53, 91],
    [77, 77, 13, 5]
  ],
  "k": 20
}
```

## Data source

RetailRocket dataset (Kaggle): https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset

If Kaggle API credentials are configured, the project can download data automatically.
Otherwise, download manually and place the ZIP under `data/raw/ecommerce-dataset.zip`.
