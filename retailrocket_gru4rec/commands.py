from __future__ import annotations

import json
import subprocess
from pathlib import Path

import fire
import mlflow
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from retailrocket_gru4rec.data.dataset import load_vocab
from retailrocket_gru4rec.data.download import ensure_raw_data
from retailrocket_gru4rec.data.preprocess import preprocess_retailrocket, unzip_raw
from retailrocket_gru4rec.evaluation.offline import (
    evaluate_gru4rec_from_checkpoint,
    evaluate_top_popular,
)
from retailrocket_gru4rec.inference.predictor import load_model_from_ckpt
from retailrocket_gru4rec.models.gru4rec import GRU4RecConfig
from retailrocket_gru4rec.serving.mlflow_pyfunc import save_pyfunc_model
from retailrocket_gru4rec.training.callbacks import SaveCurvesCallback
from retailrocket_gru4rec.training.data_module import RetailRocketDataModule
from retailrocket_gru4rec.training.lit_module import GRU4RecLightning, OptimConfig
from retailrocket_gru4rec.utils import (
    GRU4RecOnnxWrapper,
    _build_logger,
    _compose_config,
    _dvc_pull,
    _flatten_for_mlflow,
    _normalize_overrides,
    find_repo_root,
    get_git_commit_id,
)


def _ensure_processed_data(cfg: DictConfig) -> Path:
    """Ensure data/processed contains train/val/test parquet + vocab; otherwise build it."""
    repo_root = find_repo_root()

    raw_dir = repo_root / Path(cfg.data.raw_dir)
    processed_dir = repo_root / Path(cfg.data.processed_dir)
    raw_zip = raw_dir / str(cfg.data.raw_zip_name)

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

    if not raw_zip.exists():
        ensure_raw_data(
            raw_dir=raw_dir,
            dataset=str(cfg.data.kaggle_dataset),
            zip_name=str(cfg.data.raw_zip_name),
        )

    unzip_dir = raw_dir / "unzipped"
    if not unzip_dir.exists():
        unzip_dir.mkdir(parents=True, exist_ok=True)
        unzip_raw(zip_path=raw_zip, output_dir=unzip_dir)

    processed_dir.mkdir(parents=True, exist_ok=True)

    test_size = cfg.data.get("test_size", None)
    test_size = None if test_size is None else float(test_size)

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


def preprocess(*overrides: str) -> None:
    """Prepare processed parquet files (recommended for `dvc repro`)."""
    cfg = _compose_config(_normalize_overrides(overrides))
    processed_dir = _ensure_processed_data(cfg)
    print(f"Processed data is ready: {processed_dir}")


def _export_onnx_from_cfg(cfg: DictConfig) -> Path:
    repo_root = find_repo_root()
    processed_dir = _ensure_processed_data(cfg)
    spec = load_vocab(processed_dir / "vocab.json")

    checkpoint_path = repo_root / str(cfg.export.checkpoint_path)
    onnx_path = repo_root / str(cfg.export.onnx_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    if onnx_path.exists():
        return onnx_path

    base_model = load_model_from_ckpt(
        checkpoint_path=checkpoint_path,
        vocab_size=spec.vocab_size,
        pad_id=spec.pad_id,
        embedding_dim=int(cfg.model.embedding_dim),
        hidden_dim=int(cfg.model.hidden_dim),
        num_layers=int(cfg.model.num_layers),
        dropout=float(cfg.model.dropout),
        device="cpu",
    )
    base_model.eval()

    wrapper = GRU4RecOnnxWrapper(base_model, spec.pad_id).eval()

    dummy_batch = int(cfg.export.get("dummy_batch", 1))
    dummy_seq_len = int(cfg.export.get("dummy_seq_len", int(cfg.data.max_session_len)))
    fill_id = 1 if spec.pad_id != 1 else 2
    dummy_seq = torch.full((dummy_batch, dummy_seq_len), fill_value=fill_id, dtype=torch.long)

    opset = int(cfg.export.get("opset", 18))

    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            (dummy_seq,),
            str(onnx_path),
            input_names=["sequence"],
            output_names=["logits"],
            dynamic_axes={"sequence": {0: "batch", 1: "seq_len"}, "logits": {0: "batch"}},
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,
        )

    return onnx_path


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

    model_cfg = GRU4RecConfig(
        vocab_size=spec.vocab_size,
        pad_id=spec.pad_id,
        embedding_dim=int(cfg.model.embedding_dim),
        hidden_dim=int(cfg.model.hidden_dim),
        num_layers=int(cfg.model.num_layers),
        dropout=float(cfg.model.dropout),
    )

    optim_cfg = OptimConfig(
        learning_rate=float(cfg.train.learning_rate),
        weight_decay=float(cfg.train.weight_decay),
    )

    model = GRU4RecLightning(
        model_cfg=model_cfg,
        optim_cfg=optim_cfg,
        k_metrics=list(cfg.train.k_metrics),
    )

    mlf_logger = _build_logger(cfg, repo_root)

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
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
        filename="epoch={epoch}-step={step}",
        monitor="val/loss",
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
        gradient_clip_val=float(cfg.train.gradient_clip_val),
    )

    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm, ckpt_path="best")

    if bool(cfg.export.get("required", True)):
        onnx_path = _export_onnx_from_cfg(cfg)
        print(f"ONNX exported (mandatory): {onnx_path}")


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


def export_mlflow(*overrides: str) -> None:
    cfg = _compose_config(_normalize_overrides(overrides))
    repo_root = find_repo_root()

    processed_dir = _ensure_processed_data(cfg)
    vocab_path = processed_dir / "vocab.json"

    onnx_path = _export_onnx_from_cfg(cfg)

    model_dir = repo_root / str(cfg.serve.model_dir)
    model_dir.parent.mkdir(parents=True, exist_ok=True)

    save_pyfunc_model(
        model_dir=model_dir,
        onnx_path=onnx_path,
        vocab_path=vocab_path,
        topk=int(cfg.serve.topk),
        code_paths=[str(repo_root / "retailrocket_gru4rec")],
    )
    print(f"Exported MLflow(ONNX) model to: {model_dir}")


def serve_mlflow(*overrides: str) -> None:
    cfg = _compose_config(_normalize_overrides(overrides))
    repo_root = find_repo_root()

    model_dir = repo_root / str(cfg.serve.model_dir)
    host = str(getattr(cfg.serve, "host", "127.0.0.1"))
    port = int(getattr(cfg.serve, "port", 5000))

    if not model_dir.exists():
        export_mlflow(*overrides)

    cmd = [
        "mlflow",
        "models",
        "serve",
        "-m",
        str(model_dir),
        "-h",
        host,
        "-p",
        str(port),
        "--no-conda",
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    fire.Fire(
        {
            "preprocess": preprocess,
            "train": train,
            "evaluate": evaluate,
            "export_mlflow": export_mlflow,
            "serve_mlflow": serve_mlflow,
        }
    )


if __name__ == "__main__":
    main()
