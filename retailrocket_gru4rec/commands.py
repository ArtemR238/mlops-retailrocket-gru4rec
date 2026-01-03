# retailrocket_gru4rec/commands.py
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import fire
import pytorch_lightning as pl
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger

from retailrocket_gru4rec.data.dataset import load_vocab
from retailrocket_gru4rec.data.download import ensure_raw_data
from retailrocket_gru4rec.data.preprocess import preprocess_retailrocket, unzip_raw
from retailrocket_gru4rec.evaluation.offline import (
    evaluate_gru4rec_from_checkpoint,
    evaluate_top_popular,
)
from retailrocket_gru4rec.inference.predictor import load_model_from_ckpt, run_offline_inference
from retailrocket_gru4rec.training.callbacks import SaveCurvesCallback
from retailrocket_gru4rec.training.data_module import RetailRocketDataModule
from retailrocket_gru4rec.training.lit_module import GRU4RecLightning
from retailrocket_gru4rec.utils import find_repo_root, get_git_commit_id


def _normalize_overrides(overrides: Tuple[str, ...] | List[str] | None) -> List[str]:
    if overrides is None:
        return []
    if isinstance(overrides, tuple):
        return [x for x in overrides if x]
    return [x for x in overrides if x]


def _flatten_for_mlflow(d: Dict[str, Any], prefix: str = "") -> Dict[str, str]:
    """Flatten nested dict into MLflow-friendly (key -> str(value)) map."""
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
    """Hydra compose API (CLI-friendly): config is loaded from repo_root/configs."""
    repo_root = find_repo_root()
    config_dir = repo_root / "configs"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        # convention: root config file is configs/config.yaml
        cfg = compose(config_name="config", overrides=overrides)
    return cfg


def _dvc_pull(repo_root: Path, targets: Iterable[Path]) -> None:
    """Try to fetch targets via DVC Python API; fallback to CLI; do not hard-fail."""
    rel_targets: List[str] = []
    for t in targets:
        try:
            rel_targets.append(str(t.relative_to(repo_root)))
        except Exception:
            rel_targets.append(str(t))

    if not rel_targets:
        return

    # 1) Python API (preferred by the requirements)
    try:
        from dvc.repo import Repo  # type: ignore

        with Repo(str(repo_root)) as repo:
            # Pull each target best-effort to avoid “not in dvc” killing the run
            for t in rel_targets:
                try:
                    repo.pull(targets=[t])
                except Exception:
                    pass
        return
    except Exception:
        pass

    # 2) CLI fallback
    try:
        subprocess.run(["dvc", "pull", *rel_targets], check=False)
    except FileNotFoundError:
        # DVC not installed in PATH — fine, we can still download raw and preprocess.
        pass


def _ensure_processed_data(cfg: DictConfig) -> Path:
    """Ensure data/processed contains train/val/test parquet + vocab; otherwise build it."""
    repo_root = find_repo_root()

    raw_dir = repo_root / Path(cfg.data.raw_dir)
    processed_dir = repo_root / Path(cfg.data.processed_dir)
    raw_zip = raw_dir / str(cfg.data.raw_zip_name)

    # DVC integration: try to pull both processed artifacts and raw zip if they are tracked.
    if bool(cfg.data.use_dvc_pull):
        _dvc_pull(repo_root, targets=[processed_dir, raw_zip])

    vocab_path = processed_dir / "vocab.json"
    train_path = processed_dir / "train.parquet"
    val_path = processed_dir / "val.parquet"
    test_path = processed_dir / "test.parquet"

    if vocab_path.exists() and train_path.exists() and val_path.exists() and test_path.exists():
        return processed_dir

    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # If raw zip is missing, download (Kaggle credentials must be configured in env/home).
    if not raw_zip.exists():
        ensure_raw_data(
            raw_dir=raw_dir,
            dataset=str(cfg.data.kaggle_dataset),
            zip_name=str(cfg.data.raw_zip_name),
        )

    # Unzip once into raw_dir (events.csv, etc.)
    unzip_dir = raw_dir / "unzipped"
    if not unzip_dir.exists():
        unzip_dir.mkdir(parents=True, exist_ok=True)
        unzip_raw(zip_path=raw_zip, output_dir=unzip_dir)

    processed_dir.mkdir(parents=True, exist_ok=True)

    test_size = cfg.data.get("test_size", None)
    test_size = None if test_size is None else float(test_size)

    # Build processed parquet + vocab
    preprocess_retailrocket(
        input_dir=unzip_dir,
        output_dir=processed_dir,
        min_session_len=int(cfg.data.min_session_len),
        max_session_len=int(cfg.data.max_session_len),
        test_size=test_size,
        session_gap_minutes=int(cfg.data.session_gap_minutes),
        min_item_interactions=int(cfg.data.min_item_interactions),
        max_users=None if cfg.data.max_users is None else int(cfg.data.max_users),
        train_frac=float(cfg.data.train_frac),
        val_frac=float(cfg.data.val_frac),
        test_frac=float(cfg.data.test_frac),
    )

    return processed_dir


# -------------------------
# CLI commands
# -------------------------
def preprocess(*overrides: str) -> None:
    """Prepare processed parquet files (recommended for `dvc repro`)."""
    cfg = _compose_config(_normalize_overrides(overrides))
    processed_dir = _ensure_processed_data(cfg)
    print(f"Processed data is ready: {processed_dir}")


def train(*overrides: str) -> None:
    """Train GRU4Rec with PyTorch Lightning + MLflow logging."""
    cfg = _compose_config(_normalize_overrides(overrides))
    repo_root = find_repo_root()

    seed = int(cfg.seed) if "seed" in cfg else int(cfg.data.seed)
    pl.seed_everything(seed, workers=True)

    processed_dir = _ensure_processed_data(cfg)
    spec = load_vocab(processed_dir / "vocab.json")

    dm = RetailRocketDataModule(
        train_path=processed_dir / "train.parquet",
        val_path=processed_dir / "val.parquet",
        test_path=processed_dir / "test.parquet",
        max_session_len=int(cfg.data.max_session_len),
        batch_size=int(cfg.train.batch_size),
        num_workers=int(cfg.train.num_workers),
        seed=seed,
    )

    model = GRU4RecLightning(
        vocab_size=spec.vocab_size,
        pad_id=spec.pad_id,
        embedding_dim=int(cfg.model.embedding_dim),
        hidden_dim=int(cfg.model.hidden_dim),
        num_layers=int(cfg.model.num_layers),
        dropout=float(cfg.model.dropout),
        lr=float(cfg.train.lr),
        weight_decay=float(cfg.train.weight_decay),
        k_metrics=list(cfg.train.k_metrics),
    )

    mlf_logger = MLFlowLogger(
        experiment_name=str(cfg.logging.experiment_name),
        tracking_uri=str(cfg.logging.tracking_uri),
        run_name=str(cfg.logging.run_name) if "run_name" in cfg.logging else None,
    )

    # Log params in a safe, flat form
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)  # type: ignore[assignment]
    flat_params = _flatten_for_mlflow(cfg_dict if isinstance(cfg_dict, dict) else {})
    flat_params["git_commit_id"] = get_git_commit_id(repo_root) or "unknown"
    flat_params["seed"] = str(seed)
    mlf_logger.log_hyperparams(flat_params)

    artifacts_dir = repo_root / "artifacts"
    plots_dir = repo_root / "plots"
    checkpoints_dir = artifacts_dir / "checkpoints"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    ckpt_cb = ModelCheckpoint(
        dirpath=str(checkpoints_dir),
        filename="epoch={epoch}-val_loss={val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    lr_cb = LearningRateMonitor(logging_interval="step")
    curves_cb = SaveCurvesCallback(plots_dir=plots_dir)

    trainer = pl.Trainer(
        max_epochs=int(cfg.train.max_epochs),
        accelerator=str(cfg.train.accelerator),
        devices=int(cfg.train.devices),
        deterministic=True,
        logger=mlf_logger,
        callbacks=[ckpt_cb, lr_cb, curves_cb],
        log_every_n_steps=int(cfg.train.log_every_n_steps),
        default_root_dir=str(artifacts_dir),
        enable_checkpointing=True,
    )

    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm, ckpt_path="best")


def evaluate(*overrides: str) -> None:
    """Offline evaluation: TopPopular baseline vs GRU4Rec. Writes JSON report to plots/."""
    cfg = _compose_config(_normalize_overrides(overrides))
    repo_root = find_repo_root()

    processed_dir = _ensure_processed_data(cfg)
    train_path = processed_dir / "train.parquet"
    test_path = processed_dir / "test.parquet"
    vocab_path = processed_dir / "vocab.json"

    k_list = list(cfg.train.k_metrics)
    checkpoint_path = repo_root / str(cfg.eval.checkpoint_path)

    top_pop = evaluate_top_popular(
        train_parquet=train_path,
        test_parquet=test_path,
        vocab_path=vocab_path,
        k_list=k_list,
        exclude_seen=True,
    )
    gru = evaluate_gru4rec_from_checkpoint(
        checkpoint_path=checkpoint_path,
        vocab_path=vocab_path,
        model_params=OmegaConf.to_container(cfg.model, resolve=True),  # type: ignore[arg-type]
        test_parquet=test_path,
        k_list=k_list,
        batch_size=int(cfg.eval.batch_size),
        device=str(cfg.eval.device),
    )

    report = {"top_popular": top_pop, "gru4rec": gru}

    out_path = repo_root / str(cfg.eval.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Offline evaluation written to: {out_path}")

    if bool(cfg.eval.log_to_mlflow):
        import mlflow

        mlflow.set_tracking_uri(str(cfg.logging.tracking_uri))
        mlflow.set_experiment(str(cfg.logging.experiment_name))
        with mlflow.start_run(run_name="offline_eval"):
            mlflow.log_param("git_commit_id", get_git_commit_id(repo_root) or "unknown")
            cfg_dict = OmegaConf.to_container(cfg, resolve=True)  # type: ignore[assignment]
            for k, v in _flatten_for_mlflow(cfg_dict if isinstance(cfg_dict, dict) else {}).items():
                mlflow.log_param(k, v)

            for model_name, metrics_by_k in report.items():
                for k, metric_map in metrics_by_k.items():
                    for metric_name, val in metric_map.items():
                        mlflow.log_metric(f"{model_name}.{metric_name}@{k}", float(val))

            mlflow.log_artifact(str(out_path))


def infer(*overrides: str) -> None:
    """Offline inference from JSON requests -> JSON recommendations."""
    cfg = _compose_config(_normalize_overrides(overrides))
    repo_root = find_repo_root()

    processed_dir = _ensure_processed_data(cfg)
    spec = load_vocab(processed_dir / "vocab.json")

    checkpoint_path = repo_root / str(cfg.infer.checkpoint_path)
    requests_path = repo_root / str(cfg.infer.requests_path)
    output_path = repo_root / str(cfg.infer.output_path)

    run_offline_inference(
        checkpoint_path=checkpoint_path,
        vocab_path=processed_dir / "vocab.json",
        vocab_size=spec.vocab_size,
        pad_id=spec.pad_id,
        embedding_dim=int(cfg.model.embedding_dim),
        hidden_dim=int(cfg.model.hidden_dim),
        num_layers=int(cfg.model.num_layers),
        dropout=float(cfg.model.dropout),
        device=str(cfg.infer.device),
        requests_path=requests_path,
        output_path=output_path,
        topk=int(cfg.infer.topk),
        max_session_len=int(cfg.data.max_session_len),
    )
    print(f"Inference saved to: {output_path}")


def export_onnx(*overrides: str) -> None:
    """Export GRU4Rec checkpoint to ONNX."""
    cfg = _compose_config(_normalize_overrides(overrides))
    repo_root = find_repo_root()

    processed_dir = _ensure_processed_data(cfg)
    spec = load_vocab(processed_dir / "vocab.json")

    checkpoint_path = repo_root / str(cfg.export.checkpoint_path)
    onnx_path = repo_root / str(cfg.export.onnx_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    model = load_model_from_ckpt(
        checkpoint_path=checkpoint_path,
        vocab_size=spec.vocab_size,
        pad_id=spec.pad_id,
        embedding_dim=int(cfg.model.embedding_dim),
        hidden_dim=int(cfg.model.hidden_dim),
        num_layers=int(cfg.model.num_layers),
        dropout=float(cfg.model.dropout),
        device="cpu",
    )

    dummy_seq = torch.ones((1, int(cfg.data.max_session_len)), dtype=torch.long)
    dummy_lengths = torch.tensor([int(cfg.data.max_session_len)], dtype=torch.long)

    torch.onnx.export(
        model,
        (dummy_seq, dummy_lengths),
        str(onnx_path),
        input_names=["sequence", "lengths"],
        output_names=["logits"],
        dynamic_axes={
            "sequence": {0: "batch", 1: "seq_len"},
            "lengths": {0: "batch"},
            "logits": {0: "batch"},
        },
        opset_version=17,
    )
    print(f"Exported ONNX to: {onnx_path}")


def export_mlflow(*overrides: str) -> None:
    from retailrocket_gru4rec.serving.mlflow_pyfunc import save_pyfunc_model

    """Export MLflow pyfunc model directory for `mlflow models serve`."""
    cfg = _compose_config(_normalize_overrides(overrides))
    repo_root = find_repo_root()

    processed_dir = _ensure_processed_data(cfg)
    vocab_path = processed_dir / "vocab.json"
    checkpoint_path = repo_root / str(cfg.serve.checkpoint_path)
    model_dir = repo_root / str(cfg.serve.model_dir)

    model_dir.parent.mkdir(parents=True, exist_ok=True)

    save_pyfunc_model(
        model_dir=model_dir,
        checkpoint_path=checkpoint_path,
        vocab_path=vocab_path,
        model_params=OmegaConf.to_container(cfg.model, resolve=True),
        topk=int(cfg.serve.topk),
        device=str(cfg.serve.device),
        code_paths=["retailrocket_gru4rec"],
    )
    print(f"Exported MLflow model to: {model_dir}")


def main() -> None:
    fire.Fire(
        {
            "preprocess": preprocess,
            "train": train,
            "evaluate": evaluate,
            "infer": infer,
            "export_onnx": export_onnx,
            "export_mlflow": export_mlflow,
        }
    )


if __name__ == "__main__":
    main()
