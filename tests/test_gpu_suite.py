"""GPU-only engine checks.

Collected on every runner and skipped when nvcc or a GPU is absent.
On a CUDA machine they build with the native architecture and run the
existing engine tests. They do not change search behaviour.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "cuda" / "engine"
CUDA = ROOT / "cuda"

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def _cuda_env() -> dict[str, str]:
    env = os.environ.copy()
    # "native" asks nvcc for the GPU in this machine. The default Makefile
    # target is sm_121 (DGX Spark) and will not load on other boards.
    env.setdefault("CUDA_SINGLE_GPU_TARGET", "native")
    return env


@pytest.mark.gpu
def test_transposition_table_and_zobrist() -> None:
    subprocess.run(
        ["make", "-C", str(ENGINE), "engine_test"],
        check=True,
        env=_cuda_env(),
        timeout=600,
    )


@pytest.mark.gpu
def test_uci_smoke() -> None:
    subprocess.run(
        ["make", "-C", str(ENGINE), "uci_test"],
        check=True,
        env=_cuda_env(),
        timeout=600,
    )


@pytest.mark.gpu
def test_chess_mcts_smoke() -> None:
    subprocess.run(
        ["make", "-C", str(CUDA), "chess_mcts"],
        check=True,
        env=_cuda_env(),
        timeout=600,
    )
    binary = CUDA / "chess_mcts"
    result = subprocess.run(
        [str(binary), "--simulations", "32", "--json"],
        input=START_FEN + "\n",
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr
    assert "bestmove" in result.stdout
