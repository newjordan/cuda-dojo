"""Host build of the engine move generator. No CUDA required."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tests.conftest import cuda_available

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "cuda" / "engine"


@pytest.fixture(scope="module")
def symmetry_binary() -> Path:
    subprocess.run(["make", "-C", str(ENGINE), "host"], check=True)
    binary = ENGINE / "tests" / "test_makeunmake_symmetry_host"
    assert binary.is_file()
    return binary


def test_make_unmake_symmetry(symmetry_binary: Path) -> None:
    result = subprocess.run(
        [str(symmetry_binary)],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "failures = 0" in result.stdout
    assert "total_trials = 0" not in result.stdout


def test_run_tests_script_skips_gpu_binaries() -> None:
    subprocess.run(["make", "-C", str(ENGINE), "host"], check=True)
    result = subprocess.run(
        ["bash", str(ENGINE / "run_tests.sh")],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    if cuda_available():
        return
    combined = result.stdout + result.stderr
    assert "SKIP tests/test_tt (CUDA is not available)" in combined
    assert "SKIP tests/test_zobrist (CUDA is not available)" in combined
