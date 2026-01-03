from __future__ import annotations

import os
import random
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np
import torch


def find_repo_root(start: Path | None = None) -> Path:
    """Locate repo root by walking up until pyproject.toml is found."""
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
