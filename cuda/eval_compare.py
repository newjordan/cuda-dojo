#!/usr/bin/env python3
"""
eval_compare.py — verify GPU static eval (batch_eval / gpu_fighter d_evaluate)
against an EXACT integer-math Python port of the same PeSTO tapered eval.

This is the allowed CPU exception: CPU is used ONLY as a reference oracle to
verify that the GPU eval implementation is mathematically correct. If they
disagree on any FEN, the GPU is computing the wrong chess.

GPU side: invokes ./batch_eval (in this directory). batch_eval.cu and
gpu_fighter.cu d_evaluate are byte-identical PeSTO implementations sharing
the same __constant__ tables, integer arithmetic, side-to-move convention,
and (mg*phase + eg*(24-phase))/24 reduction. Verified by reading both files.

Output:
  cuda/eval_report.json — per-FEN cpu_score, gpu_score, diff, plus diff
  histogram and the top 20 worst disagreements.

Usage:
  python3 cuda/eval_compare.py [--n 1000] [--seed 0] [--book PATH] \
      [--binary PATH] [--out PATH] [--sanity-only]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
GPU_SPINE_DIR = REPO_ROOT / "gpu_spine"
CUDA_DIR = REPO_ROOT / "cuda"

# =============================================================================
# PeSTO PSTs — copied verbatim from cuda/gpu_fighter.cu (== batch_eval.cu).
# Index 0 = a8 (top-left when iterating sq=0..63 with FEN), 63 = h1.
# =============================================================================

MG_PAWN = [
     0,  0,  0,  0,  0,  0,  0,  0,
    98,134, 61, 95, 68,126, 34,-11,
    -6,  7, 26, 31, 65, 56, 25,-20,
   -14, 13,  6, 21, 23, 12, 17,-23,
   -27, -2, -5, 12, 17,  6, 10,-25,
   -26, -4, -4,-10,  3,  3, 33,-12,
   -35, -1,-20,-23,-15, 24, 38,-22,
     0,  0,  0,  0,  0,  0,  0,  0,
]
MG_KNIGHT = [
   -167,-89,-34,-49, 61,-97, 50,-73,
    -73,-41, 72, 36, 23, 62,  7,-17,
    -47, 60, 37, 65, 84,129, 73, 44,
     -9, 17, 19, 53, 37, 69, 18, 22,
    -13,  4, 16, 13, 28, 19, 21, -8,
    -23, -9, 12, 10, 19, 17, 25,-16,
    -29,-53,-12, -3, -1, 18,-14,-19,
   -105,-21,-58,-33,-17,-28,-19,-23,
]
MG_BISHOP = [
    -29,  4,-82,-37,-25,-42,  7, -8,
    -26, 16,-18,-13, 30, 59, 18,-47,
    -16, 37, 43, 40, 35, 50, 37, -2,
     -4,  5, 19, 50, 37, 37,  7, -2,
     -6, 13, 13, 26, 34, 12, 10,  4,
      0, 15, 15, 15, 14, 27, 18, 10,
      4, 15, 16,  0,  7, 21, 33,  1,
    -33, -3,-14,-21,-13,-12,-39,-21,
]
MG_ROOK = [
     32, 42, 32, 51, 63,  9, 31, 43,
     27, 32, 58, 62, 80, 67, 26, 44,
     -5, 19, 26, 36, 17, 45, 61, 16,
    -24,-11,  7, 26, 24, 35, -8,-20,
    -36,-26,-12, -1,  9, -7,  6,-23,
    -45,-25,-16,-17,  3,  0, -5,-33,
    -44,-16,-20, -9, -1, 11, -6,-71,
    -19,-13,  1, 17, 16,  7,-37,-26,
]
MG_QUEEN = [
    -28,  0, 29, 12, 59, 44, 43, 45,
    -24,-39, -5,  1,-16, 57, 28, 54,
    -13,-17,  7,  8, 29, 56, 47, 57,
    -27,-27,-16,-16, -1, 17, -2,  1,
     -9,-26, -9,-10, -2, -4,  3, -3,
    -14, -2,-11, -2, -5,  2, 14,  5,
    -35, -8, 11,  2,  8, 15, -3,  1,
     -1,-18, -9, 10,-15,-25,-31,-50,
]
MG_KING = [
    -65, 23, 16,-15,-56,-34,  2, 13,
     29, -1,-20, -7, -8, -4,-38,-29,
     -9, 24,  2,-16,-20,  6, 22,-22,
    -17,-20,-12,-27,-30,-25,-14,-36,
    -49, -1,-27,-39,-46,-44,-33,-51,
    -14,-14,-22,-46,-44,-30,-15,-27,
      1,  7, -8,-64,-43,-16,  9,  8,
    -15, 36, 12,-54,  8,-28, 24, 14,
]
EG_PAWN = [
      0,  0,  0,  0,  0,  0,  0,  0,
    178,173,158,134,147,132,165,187,
     94,100, 85, 67, 56, 53, 82, 84,
     32, 24, 13,  5, -2,  4, 17, 17,
     13,  9, -3, -7, -7, -8,  3, -1,
      4,  7, -6,  1,  0, -5, -1, -8,
     13,  8,  8, 10, 13,  0,  2, -7,
      0,  0,  0,  0,  0,  0,  0,  0,
]
EG_KNIGHT = [
    -58,-38,-13,-28,-31,-27,-63,-99,
    -25, -8,-25, -2, -9,-25,-24,-52,
    -24,-20, 10,  9, -1, -9,-19,-41,
    -17,  3, 22, 22, 22, 11,  8,-18,
    -18, -6, 16, 25, 16, 17,  4,-18,
    -23, -3, -1, 15, 10, -3,-20,-22,
    -42,-20,-10, -5, -2,-20,-23,-44,
    -29,-51,-23,-15,-22,-18,-50,-64,
]
EG_BISHOP = [
    -14,-21,-11, -8, -7, -9,-17,-24,
     -8, -4,  7,-12, -3,-13, -4,-14,
      2, -8,  0, -1, -2,  6,  0,  4,
     -3,  9, 12,  9, 14, 10,  3,  2,
     -6,  3, 13, 19,  7, 10, -3, -9,
    -12, -3,  8, 10, 13,  3, -7,-15,
    -14,-18, -7, -1,  4, -9,-15,-27,
    -23, -9,-23, -5, -9,-16, -5,-17,
]
EG_ROOK = [
     13, 10, 18, 15, 12, 12,  8,  5,
     11, 13, 13, 11, -3,  7,  7,  8,
      7,  7,  7,  5,  4, -3, -5,  3,
      4,  3, 13,  1,  2,  1, -1,  2,
      3,  5,  8,  4, -5, -6, -8,-11,
     -4,  0, -5, -1, -7,-12, -8,-16,
     -6, -6,  0,  2, -9, -9,-11, -3,
     -9,  2,  3, -1, -5,-13,  4,-20,
]
EG_QUEEN = [
     -9, 22, 22, 27, 27, 19, 10, 20,
    -17, 20, 32, 41, 58, 25, 30,  0,
    -20,  6,  9, 49, 47, 35, 19,  9,
      3, 22, 24, 45, 57, 40, 57, 36,
    -18, 28, 19, 47, 31, 34, 39, 23,
    -16,-27, 15,  6,  9, 17, 10,  5,
    -22,-23,-30,-16,-16,-23,-36,-32,
    -33,-28,-22,-43, -5,-32,-20,-41,
]
EG_KING = [
    -74,-35,-18,-18,-11, 15,  4,-17,
    -12, 17, 14, 17, 17, 38, 23, 11,
     10, 17, 23, 15, 20, 45, 44, 13,
     -8, 22, 24, 27, 26, 33, 26,  3,
    -18, -4, 21, 24, 27, 23,  9,-11,
    -19, -3, 11, 21, 23, 16,  7, -9,
    -27,-11,  4, 13, 14,  4, -5,-17,
    -53,-34,-21,-11,-28,-14,-24,-43,
]

# Material values (centipawns). Index 0 unused, 1..6 = P,N,B,R,Q,K.
MG_VALS = [0, 82, 337, 365, 477, 1025, 0]
EG_VALS = [0, 94, 281, 297, 512, 936, 0]
# Tapered phase weights. P=0, N=1, B=1, R=2, Q=4, K=0. Max sum at game start = 24.
PHASE_W = [0, 0, 1, 1, 2, 4, 0]

MG_TABLES = [None, MG_PAWN, MG_KNIGHT, MG_BISHOP, MG_ROOK, MG_QUEEN, MG_KING]
EG_TABLES = [None, EG_PAWN, EG_KNIGHT, EG_BISHOP, EG_ROOK, EG_QUEEN, EG_KING]

# Piece codes (mirrors gpu_fighter.cu).
EMPTY, WP, WN, WB, WR, WQ, WK = 0, 1, 2, 3, 4, 5, 6
BP, BN, BB, BR, BQ, BK = 7, 8, 9, 10, 11, 12
WHITE_SIDE, BLACK_SIDE = 0, 1


def trunc_div(num: int, den: int) -> int:
    """C-style integer division: truncates toward zero (Python // floors)."""
    q = num // den
    if (num % den) != 0 and ((num < 0) ^ (den < 0)):
        q += 1
    return q


def mirror_sq(sq: int) -> int:
    """Vertical mirror (rank flip) — matches gpu_fighter.cu's mirror_sq."""
    return ((7 - (sq >> 3)) << 3) | (sq & 7)


def parse_fen(fen: str):
    """Returns (board[64], side) using gpu_fighter piece encoding.

    sq=0 corresponds to FEN's rank 8 file a (the first rank parsed),
    sq=63 to rank 1 file h. White pieces use codes 1..6 (PAWN..KING),
    black 7..12. side: 0=white, 1=black.
    """
    board = [EMPTY] * 64
    sq = 0
    i = 0
    n = len(fen)
    while i < n and fen[i] != ' ':
        c = fen[i]
        i += 1
        if c == '/':
            continue
        if '1' <= c <= '8':
            sq += int(c)
            continue
        piece = {
            'P': WP, 'N': WN, 'B': WB, 'R': WR, 'Q': WQ, 'K': WK,
            'p': BP, 'n': BN, 'b': BB, 'r': BR, 'q': BQ, 'k': BK,
        }.get(c, EMPTY)
        if sq < 64:
            board[sq] = piece
        sq += 1
    while i < n and fen[i] == ' ':
        i += 1
    side = BLACK_SIDE if (i < n and fen[i] == 'b') else WHITE_SIDE
    return board, side


def cpu_evaluate(fen: str) -> int:
    """EXACT Python port of gpu_fighter.cu d_evaluate (and batch_eval kernel).

    All math integer. Same PSTs, same material, same phase weights, same
    tapered formula, same C-style integer division (truncates toward zero),
    same side-to-move flip at the end.
    """
    board, side = parse_fen(fen)
    mg = 0
    eg = 0
    phase = 0
    for sq in range(64):
        p = board[sq]
        if p == EMPTY:
            continue
        is_white = (WP <= p <= WK)
        pt = p if is_white else p - 6  # 1..6
        psq = sq if is_white else mirror_sq(sq)
        mv = MG_VALS[pt]
        ev = EG_VALS[pt]
        mpst = MG_TABLES[pt][psq]
        epst = EG_TABLES[pt][psq]
        sign = 1 if is_white else -1
        mg += sign * (mv + mpst)
        eg += sign * (ev + epst)
        phase += PHASE_W[pt]
    if phase > 24:
        phase = 24
    score = trunc_div(mg * phase + eg * (24 - phase), 24)
    return score if side == WHITE_SIDE else -score


def gpu_evaluate_batch(fens, binary: str) -> list:
    """Invoke ./batch_eval --plain with all FENs on stdin; return list of ints.

    batch_eval.cu's eval kernel is BYTE-IDENTICAL to gpu_fighter.cu's
    d_evaluate: same __constant__ PSTs, same MG_VALS/EG_VALS/PHASE_W, same
    integer math, same side-to-move flip. Verified by reading both files.
    """
    if not fens:
        return []
    proc = subprocess.run(
        [binary, "--plain"],
        input="\n".join(fens) + "\n",
        capture_output=True, text=True, timeout=120,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"batch_eval failed (rc={proc.returncode}):\n{proc.stderr}"
        )
    out_lines = [ln for ln in proc.stdout.strip().split("\n") if ln.strip() != ""]
    if len(out_lines) != len(fens):
        raise RuntimeError(
            f"batch_eval returned {len(out_lines)} scores, expected {len(fens)}.\n"
            f"stderr: {proc.stderr}"
        )
    return [int(s) for s in out_lines]


SANITY_FENS = [
    # (label, fen)
    ("startpos",       "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
    ("kiwipete",       "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"),
    ("KK_endgame",     "8/2k5/8/8/8/8/2K5/8 w - - 0 1"),
    ("italian_mid",    "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"),
    ("pawn_promote",   "8/4k3/8/8/3p4/3P4/4K3/8 w - - 0 1"),
]


def run_sanity(binary: str) -> bool:
    fens = [f for _, f in SANITY_FENS]
    cpu = [cpu_evaluate(f) for f in fens]
    gpu = gpu_evaluate_batch(fens, binary)
    print("=== Sanity check (5 hand-picked FENs) ===")
    print(f"{'label':<14} {'cpu':>8} {'gpu':>8} {'diff':>6}  fen")
    ok = True
    for (label, fen), c, g in zip(SANITY_FENS, cpu, gpu):
        d = g - c
        flag = "" if d == 0 else "  <-- MISMATCH"
        if d != 0:
            ok = False
        print(f"{label:<14} {c:>8} {g:>8} {d:>6}  {fen}{flag}")
    print()
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--book", default=str(GPU_SPINE_DIR / "book.jsonl"),
                    help="JSONL file with one {fen: ...} per line")
    ap.add_argument("--binary", default=str(CUDA_DIR / "batch_eval"),
                    help="Path to batch_eval (or any binary that takes FENs on stdin and emits one score per line in --plain mode)")
    ap.add_argument("--n", type=int, default=1000, help="Sample size from book")
    ap.add_argument("--seed", type=int, default=0, help="Random seed for sampling")
    ap.add_argument("--out", default=str(CUDA_DIR / "eval_report.json"))
    ap.add_argument("--sanity-only", action="store_true",
                    help="Run only the 5 hand-picked sanity positions")
    args = ap.parse_args()

    binary = args.binary
    if not os.path.isfile(binary) or not os.access(binary, os.X_OK):
        print(f"ERROR: GPU eval binary not found / not executable: {binary}", file=sys.stderr)
        return 2

    sanity_ok = run_sanity(binary)
    if args.sanity_only:
        return 0 if sanity_ok else 1
    if not sanity_ok:
        print("Sanity check failed — proceeding to full sweep anyway to characterise the bug.\n")

    # Sample N FENs from the book.
    book_path = args.book
    if not os.path.isfile(book_path):
        print(f"ERROR: book not found: {book_path}", file=sys.stderr)
        return 2

    fens = []
    with open(book_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            fen = obj.get("fen")
            if isinstance(fen, str) and fen:
                fens.append(fen)

    if len(fens) == 0:
        print("ERROR: book has no FENs", file=sys.stderr)
        return 2

    rng = random.Random(args.seed)
    sample = rng.sample(fens, k=min(args.n, len(fens)))
    print(f"Loaded {len(fens)} FENs from book; sampling {len(sample)} (seed={args.seed}).")

    cpu_scores = [cpu_evaluate(f) for f in sample]
    gpu_scores = gpu_evaluate_batch(sample, binary)

    diffs = [g - c for c, g in zip(cpu_scores, gpu_scores)]
    abs_diffs = [abs(d) for d in diffs]
    n = len(sample)
    n_match = sum(1 for d in diffs if d == 0)
    rate = n_match / n if n else 0.0

    # Histogram of |gpu - cpu|.
    buckets = [
        ("==0",      lambda d: d == 0),
        ("1-5",      lambda d: 1   <= d <= 5),
        ("6-25",     lambda d: 6   <= d <= 25),
        ("26-100",   lambda d: 26  <= d <= 100),
        ("101-500",  lambda d: 101 <= d <= 500),
        (">500",     lambda d: d   >  500),
    ]
    histogram = {label: sum(1 for d in abs_diffs if pred(d)) for label, pred in buckets}

    # Top 20 worst disagreements (by |diff|, then by raw diff for stability).
    ranked = sorted(
        range(n),
        key=lambda i: (-abs_diffs[i], cpu_scores[i], gpu_scores[i]),
    )
    worst = []
    for i in ranked[:20]:
        worst.append({
            "fen": sample[i],
            "cpu_score": cpu_scores[i],
            "gpu_score": gpu_scores[i],
            "diff": diffs[i],
        })

    # Per-position records (compact: all rows, with diff for inspection).
    records = [
        {"fen": sample[i], "cpu_score": cpu_scores[i], "gpu_score": gpu_scores[i], "diff": diffs[i]}
        for i in range(n)
    ]

    report = {
        "binary": binary,
        "book": book_path,
        "n_tested": n,
        "n_exact_match": n_match,
        "agreement_rate": rate,
        "max_abs_diff": max(abs_diffs) if abs_diffs else 0,
        "mean_diff": (sum(diffs) / n) if n else 0.0,
        "diff_histogram": histogram,
        "diff_value_counts": dict(Counter(diffs).most_common(20)),
        "worst_20": worst,
        "records": records,
    }

    out_path = args.out
    Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print()
    print(f"=== Eval comparison: {n} positions ===")
    print(f"Exact matches:   {n_match}/{n}  ({rate*100:.2f}%)")
    print(f"Max |diff|:      {report['max_abs_diff']}")
    print(f"Mean (gpu-cpu):  {report['mean_diff']:+.3f}")
    print("Histogram of |gpu - cpu|:")
    for label, _ in buckets:
        print(f"  {label:<10} {histogram[label]}")
    if n_match < n:
        print()
        print("Top disagreements:")
        for w in worst[:5]:
            print(f"  diff={w['diff']:+6d}  cpu={w['cpu_score']:+6d}  gpu={w['gpu_score']:+6d}  fen={w['fen']}")
    print()
    print(f"Wrote: {out_path}")

    return 0 if n_match == n else 1


if __name__ == "__main__":
    sys.exit(main())
