from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


def _has_kaggle_credentials() -> bool:
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    return kaggle_json.exists() or ("KAGGLE_USERNAME" in os.environ and "KAGGLE_KEY" in os.environ)


def download_retailrocket_kaggle(raw_dir: Path, dataset: str, zip_name: str) -> Path:
    """Download Kaggle dataset zip into raw_dir using kaggle CLI.

    Requires `kaggle` command available and credentials configured.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / zip_name
    if zip_path.exists():
        return zip_path

    # The kaggle command comes from the `kaggle` Python package.
    # We call it via subprocess to avoid extra runtime dependencies.
    cmd = ["kaggle", "datasets", "download", "-d", dataset, "-p", str(raw_dir), "--force"]
    subprocess.check_call(cmd)
    # Kaggle usually names the archive after the dataset slug (ecommerce-dataset.zip),
    # but we normalize to zip_name.
    downloaded = next(raw_dir.glob("*.zip"))
    if downloaded.name != zip_name:
        shutil.move(str(downloaded), str(zip_path))
    return zip_path


def ensure_raw_data(raw_dir: Path, dataset: str, zip_name: str) -> Path:
    """Ensure raw ZIP is present. If missing, try to download via Kaggle CLI."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / zip_name
    if zip_path.exists():
        return zip_path

    # Try Kaggle download
    try:
        return download_retailrocket_kaggle(raw_dir=raw_dir, dataset=dataset, zip_name=zip_name)
    except Exception as exc:
        raise RuntimeError(
            f"Raw dataset ZIP not found at {zip_path}.\n"
            "Option A (recommended): configure Kaggle API credentials and install kaggle CLI:\n"
            "  pip install kaggle\n"
            "  export KAGGLE_USERNAME=...; export KAGGLE_KEY=...\n"
            "  (or put ~/.kaggle/kaggle.json)\n"
            "Option B: download manually from Kaggle and place it as:\n"
            f"  {zip_path}\n"
        ) from exc
