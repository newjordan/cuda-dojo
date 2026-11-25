"""Perft on the standard chessprogramming.org positions.

python-chess checks the published node counts at depths that finish quickly.
The host build of cuda/engine (same move generator as the CUDA engine) checks
every canonical position at depth 4, including the counts that are too large
for a pure-Python walk in CI.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import chess
import pytest

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "cuda" / "engine"

# Published perft counts. https://www.chessprogramming.org/Perft_Results
KNOWN: dict[tuple[str, int], int] = {
    ("startpos", 1): 20,
    ("startpos", 2): 400,
    ("startpos", 3): 8902,
    ("startpos", 4): 197281,
    ("startpos", 5): 4865609,
    ("kiwipete", 1): 48,
    ("kiwipete", 2): 2039,
    ("kiwipete", 3): 97862,
    ("kiwipete", 4): 4085603,
    ("position3", 1): 14,
    ("position3", 2): 191,
    ("position3", 3): 2812,
    ("position3", 4): 43238,
    ("position3", 5): 674624,
    ("position4", 1): 6,
    ("position4", 2): 264,
    ("position4", 3): 9467,
    ("position4", 4): 422333,
    ("position5", 1): 44,
    ("position5", 2): 1486,
    ("position5", 3): 62379,
    ("position5", 4): 2103487,
    ("position6", 1): 46,
    ("position6", 2): 2079,
    ("position6", 3): 89890,
    ("position6", 4): 3894594,
}

FENS = {
    "startpos": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "kiwipete": "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "position3": "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "position4": "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    "position5": "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    "position6": "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
}

# Depth-4 of the busy positions is millions of nodes. The C++ host perft
# covers those. python-chess covers every published count at or below this cap.
_MAX_PYTHON_NODES = 500_000

PYTHON_CASES = [
    (name, FENS[name], depth, nodes)
    for (name, depth), nodes in KNOWN.items()
    if nodes <= _MAX_PYTHON_NODES
]

_PERFT_LINE = re.compile(
    r"\[(PASS|FAIL)\].*depth=(\d+)\s+got=(-?\d+)\s+expected=(-?\d+)"
)


def perft(board: chess.Board, depth: int) -> int:
    if depth == 0:
        return 1
    if depth == 1:
        return board.legal_moves.count()
    nodes = 0
    for move in board.legal_moves:
        board.push(move)
        nodes += perft(board, depth - 1)
        board.pop()
    return nodes


@pytest.fixture(scope="module")
def host_perft_binary() -> Path:
    subprocess.run(["make", "-C", str(ENGINE), "host"], check=True)
    binary = ENGINE / "engine_test_perft_host"
    assert binary.is_file(), f"missing host perft binary: {binary}"
    return binary


@pytest.mark.parametrize(
    ("name", "fen", "depth", "expected"),
    PYTHON_CASES,
    ids=[f"{name}-d{depth}" for name, _, depth, _ in PYTHON_CASES],
)
def test_python_chess_perft(name: str, fen: str, depth: int, expected: int) -> None:
    board = chess.Board(fen)
    assert perft(board, depth) == expected, name


def test_host_engine_perft_depth4(host_perft_binary: Path) -> None:
    """The engine move generator matches the six depth-4 perft results."""
    result = subprocess.run(
        [str(host_perft_binary)],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    found: dict[int, int] = {}
    for line in result.stdout.splitlines():
        match = _PERFT_LINE.search(line)
        if not match:
            continue
        status, depth_s, got_s, expected_s = match.groups()
        assert status == "PASS", line
        assert int(depth_s) == 4, line
        got = int(got_s)
        expected = int(expected_s)
        assert got == expected, line
        found[expected] = got

    depth4 = {nodes for (name, depth), nodes in KNOWN.items() if depth == 4}
    assert set(found) == depth4
    assert "failures=0/6" in result.stdout
