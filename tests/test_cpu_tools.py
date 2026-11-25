"""Import and shape checks for the CPU-side research tools."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "cuda"))

from compression.board_fold import (  # noqa: E402
    active_fold_positions,
    board_from_fen,
    fold_horizontal,
    fold_relationship,
    fold_vertical,
)
from compression.geometry_encoders import composite_encode  # noqa: E402
from crawler_control.build_seed_manifest import (  # noqa: E402
    build_candidates,
    load_profiles,
)
from nn_compare import cpu_forward, fen_to_features_and_side  # noqa: E402

import render_pipeline  # noqa: E402

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def test_starting_position_fold_is_centered() -> None:
    board = board_from_fen(START_FEN)
    assert len(board) == 64
    assert sum(1 for piece in board if piece) == 32
    _canon, _mask, _warmth, (rank, file) = active_fold_positions(board)
    # White and black mirror each other by rank, so the centroid rank stays
    # on the midline. The queen (weight 9) is on the d-file and the king
    # (weight 4) is on the e-file, so the centroid file sits queenside of center.
    assert math.isclose(rank, 3.5, abs_tol=1e-9)
    assert 3.0 < file < 3.5


def test_fold_variants_return_pairs() -> None:
    board = board_from_fen(START_FEN)
    for folder in (fold_vertical, fold_horizontal, fold_relationship):
        compressed, outliers = folder(board)
        assert compressed is not None
        assert isinstance(outliers, list)


def test_geometry_vector_is_55_finite_dims() -> None:
    board = board_from_fen(START_FEN)
    vector = composite_encode(board)
    assert len(vector) == 55
    assert all(math.isfinite(value) for value in vector)


def test_render_pipeline_imports_active_fold() -> None:
    board = board_from_fen(START_FEN)
    encoding = render_pipeline.active_fold_encode(board)
    assert math.isfinite(encoding.centroid_r)
    assert math.isfinite(encoding.centroid_f)
    assert isinstance(encoding.fold_signature, int)


def test_crawler_manifest_candidates_are_well_formed() -> None:
    profiles = {profile["name"]: profile for profile in load_profiles()["profiles"]}
    candidates = build_candidates(profiles)
    assert candidates
    for candidate in candidates:
        assert candidate["name"]
        assert candidate["role"]
        assert candidate["runner_mode"] in {"script", "env_bundle"}
        assert candidate["runner_path"]


def test_cpu_forward_empty_board_is_bias_chain() -> None:
    """Zero weights and a zero feature vector follow the documented GEMM path."""
    import numpy as np

    w1 = np.zeros((768, 256), dtype=np.float32)
    b1 = np.zeros(256, dtype=np.float32)
    w2 = np.zeros((256, 64), dtype=np.float32)
    b2 = np.full(64, 0.25, dtype=np.float32)
    w3 = np.zeros((64, 1), dtype=np.float32)
    b3 = np.array([1.5], dtype=np.float32)
    features, side = fen_to_features_and_side("8/8/8/8/8/8/8/8 w - - 0 1")
    assert side == "w"
    assert float(features.sum()) == 0.0
    # h2 = relu(0.25) = 0.25, then y = 0 + b3.
    assert cpu_forward(w1, b1, w2, b2, w3, b3, features, "w") == pytest.approx(1.5)
    assert cpu_forward(w1, b1, w2, b2, w3, b3, features, "b") == pytest.approx(-1.5)


def test_start_fen_features_have_32_ones() -> None:
    features, side = fen_to_features_and_side(START_FEN)
    assert side == "w"
    assert int(features.sum()) == 32
