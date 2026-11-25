"""Shared pytest hooks for CPU and CUDA runs."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_NVCC_CANDIDATES = (
    "/usr/local/cuda/bin/nvcc",
    "/usr/local/cuda-13.0/bin/nvcc",
)


def cuda_available() -> bool:
    """True when nvcc and a visible NVIDIA GPU are both present."""
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        for candidate in _NVCC_CANDIDATES:
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                nvcc = candidate
                break
    if not nvcc:
        return False
    smi = shutil.which("nvidia-smi")
    if smi is None:
        return False
    try:
        result = subprocess.run(
            [smi, "-L"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0 and "GPU" in result.stdout


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "gpu: requires nvcc and a visible NVIDIA GPU; skipped when CUDA is absent",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if cuda_available():
        return
    skip = pytest.mark.skip(reason="CUDA is not available")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)
