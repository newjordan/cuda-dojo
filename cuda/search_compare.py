#!/usr/bin/env python3
"""
search_compare.py — verify GPU alpha-beta search vs a CPU Python negamax
using a faithful CPU port of the GPU's PeSTO PST tapered eval.

Goal: isolate "search bugs" from "eval bugs". The CPU eval is hand-ported
from gpu_fighter.cu's d_evaluate(); the CPU negamax mirrors d_negamax():
plain alpha-beta with NO move ordering, no qsearch, no killers, no TT.
python-chess is used ONLY for board state / movegen / make / unmake.

Usage:
  python3 cuda/search_compare.py                     # full run, 50 positions
  python3 cuda/search_compare.py --n 10               # quick run
  python3 cuda/search_compare.py --hand                # 3 hand-check positions
  python3 cuda/search_compare.py --eval-only          # 5-pos eval sanity only

Outputs cuda/search_report.json on full run.
"""

import argparse
import json
import os
import random
import subprocess
import sys
import time

import chess

# -----------------------------------------------------------------------------
# GPU square layout
#
# In gpu_fighter.cu, parse_fen walks the FEN left-to-right top-to-bottom and
# fills board[sq] starting at sq=0 for a8.  So:
#     gpu_sq(file f, rank r in chess sense [0=rank1,...,7=rank8]) = (7-r)*8 + f
#
# python-chess square: rank*8 + file (0=a1, 63=h8).
# Conversion: gpu_sq = chess_sq XOR 56  (vertical flip).
# Equivalently: gpu_sq = (7 - chess.square_rank(s)) * 8 + chess.square_file(s).
# -----------------------------------------------------------------------------

def chess_to_gpu(sq: int) -> int:
    return sq ^ 56  # flip rank

# mirror_sq from gpu_fighter.cu: ((7 - (sq >> 3)) << 3) | (sq & 7)
def gpu_mirror(sq: int) -> int:
    return ((7 - (sq >> 3)) << 3) | (sq & 7)

# -----------------------------------------------------------------------------
# PeSTO PSTs — copied verbatim from gpu_fighter.cu (d_MG_*, d_EG_*).
# Indexed by GPU square (a8 = 0).
# -----------------------------------------------------------------------------
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

# Material values (d_MG_VALS / d_EG_VALS), index 1..6 = pawn,knight,bishop,rook,queen,king
MG_VALS = [0, 82, 337, 365, 477, 1025, 0]
EG_VALS = [0, 94, 281, 297, 512, 936, 0]
PHASE_W = [0, 0, 1, 1, 2, 4, 0]

MG_PSTS = [None, MG_PAWN, MG_KNIGHT, MG_BISHOP, MG_ROOK, MG_QUEEN, MG_KING]
EG_PSTS = [None, EG_PAWN, EG_KNIGHT, EG_BISHOP, EG_ROOK, EG_QUEEN, EG_KING]

INF_SCORE = 200000
MATE_SCORE = 100000


def evaluate_pesto(board: chess.Board) -> int:
    """Pure CPU port of d_evaluate() in gpu_fighter.cu.

    Tapered PeSTO eval: side-to-move POV centipawns. Identical formula
    and integer arithmetic as the GPU. Score = (mg*phase + eg*(24-phase)) // 24.
    Phase clamped at 24.
    """
    mg = 0
    eg = 0
    phase = 0
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is None:
            continue
        gsq = chess_to_gpu(sq)
        is_white = piece.color == chess.WHITE
        pt = piece.piece_type  # 1..6 == pawn..king (matches our tables)
        psq = gsq if is_white else gpu_mirror(gsq)
        mv = MG_VALS[pt]
        ev = EG_VALS[pt]
        mpst = MG_PSTS[pt][psq]
        epst = EG_PSTS[pt][psq]
        sign = 1 if is_white else -1
        mg += sign * (mv + mpst)
        eg += sign * (ev + epst)
        phase += PHASE_W[pt]
    if phase > 24:
        phase = 24
    # NOTE: GPU uses C integer division. In Python, mg*phase + eg*(24-phase)
    # may be negative; C truncates toward zero, Python // floors. Use a manual
    # truncate to match exactly.
    num = mg * phase + eg * (24 - phase)
    score = int(num / 24) if num < 0 else num // 24  # truncate toward zero
    # Actually simpler: replicate trunc-toward-zero division.
    score = _trunc_div(mg * phase + eg * (24 - phase), 24)
    side_white = (board.turn == chess.WHITE)
    return score if side_white else -score


def _trunc_div(a: int, b: int) -> int:
    # C-style truncate-toward-zero integer division.
    q = abs(a) // abs(b)
    if (a < 0) ^ (b < 0):
        q = -q
    return q


# -----------------------------------------------------------------------------
# CPU negamax — mirror of d_negamax in gpu_fighter.cu
# Plain alpha-beta, NO move ordering (GPU has none either at root negamax),
# NO qsearch, NO TT, NO killers. Returns side-to-move POV.
# Iterative or recursive — recursive is simpler and equivalent here.
# -----------------------------------------------------------------------------

def negamax(board: chess.Board, depth: int, alpha: int, beta: int) -> int:
    if depth == 0:
        return evaluate_pesto(board)

    best = -INF_SCORE
    legal_count = 0
    # python-chess generates pseudo-legal moves in an order that does NOT
    # match the GPU's by-square scan, but for plain alpha-beta with no
    # ordering both implementations should reach the same minimax score
    # at the same depth (cutoffs may differ but the final score is the
    # exact minimax value when no heuristic pruning is involved).
    for move in board.legal_moves:
        legal_count += 1
        board.push(move)
        score = -negamax(board, depth - 1, -beta, -alpha)
        board.pop()
        if score > best:
            best = score
        if best > alpha:
            alpha = best
        if alpha >= beta:
            return best
    if legal_count == 0:
        if board.is_check():
            # mate: GPU does -MATE_SCORE + (100 - depth_at_terminal).
            # In d_negamax the frame is at the terminal node where no legal
            # move exists. The "depth" stored in the frame is the remaining
            # depth at that node. For us, depth is also the remaining depth
            # (we just entered with depth>0 and found no legal moves).
            return -MATE_SCORE + (100 - depth)
        return 0
    return best


def root_search(board: chess.Board, depth: int):
    """Mirror of root_search_kernel: enumerate legal root moves, search each
    child with d_negamax(child, depth-1) and negate. Return (best_uci, best_score, all_scores).

    --depth N (gpu_fighter CLI):
        search_depth = depth - 1  (clamped to >=1)
        each root thread does d_negamax(child, search_depth) so
        the total tree depth is 1 (root move) + search_depth (children).
    """
    search_depth = depth - 1
    if search_depth < 1:
        search_depth = 1

    legal = list(board.legal_moves)
    if not legal:
        return ("0000", 0, [])

    scores = []
    for mv in legal:
        board.push(mv)
        opp = negamax(board, search_depth, -INF_SCORE, INF_SCORE)
        board.pop()
        scores.append((mv.uci(), -opp))

    # Pick best, ties broken by FIRST occurrence (matches GPU's
    # `if (h_scores[i] > h_scores[best])` strict-greater rule).
    best_idx = 0
    for i in range(1, len(scores)):
        if scores[i][1] > scores[best_idx][1]:
            best_idx = i
    return (scores[best_idx][0], scores[best_idx][1], scores)


# -----------------------------------------------------------------------------
# GPU runner: spawn ./cuda/gpu_fighter --depth N, feed the FEN on stdin.
# -----------------------------------------------------------------------------

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GPU_BIN = os.path.join(REPO, "cuda", "gpu_fighter")


def run_gpu(fen: str, depth: int, timeout: float = 30.0):
    proc = subprocess.run(
        [GPU_BIN, "--depth", str(depth)],
        input=(fen + "\n").encode(),
        capture_output=True,
        timeout=timeout,
        cwd=REPO,
    )
    out = proc.stdout.decode("utf-8", errors="replace").strip()
    if not out:
        raise RuntimeError(f"gpu_fighter empty output. stderr={proc.stderr.decode()!r}")
    # Last line should be the JSON result (daemon prints one per FEN).
    last = out.splitlines()[-1].strip()
    j = json.loads(last)
    return j


# -----------------------------------------------------------------------------
# Eval-only sanity: GPU at depth=1 with --qsearch off does d_negamax(child,1)
# which returns d_evaluate(child) negated then re-negated (because we negate
# returnedScore in root). So a depth=1 root search picks the move with the
# highest "negated child eval" == "score from our side after the move".
# That's not raw eval of the position; that's eval-after-best-1ply.
#
# To compare raw eval directly, we use a separate trick: d_evaluate(s) is the
# eval at the ROOT. We can't easily extract that from gpu_fighter without
# changing the binary.  But d_negamax at depth=0 returns d_evaluate(root).
# Unfortunately the kernel forces search_depth >= 1.  So the cleanest CPU/GPU
# eval check at the position level is:
#
#   For each root move m, GPU score of m == -evaluate(child_after_m)  (depth-1 search).
#
# That is: gpu_root_search(depth=1) returns, for each legal root m,
#   -d_negamax(child, 1) which at depth=1 enumerates child's legal moves and
#   returns the best -evaluate(grandchild). So depth=1 isn't a raw-eval check.
#
# We instead expose eval indirectly: enumerate captures-only positions where
# search makes the eval easy to spot, and do per-position move-by-move
# spot checks at depth=2 vs depth=3, plus for the eval sanity we evaluate
# the 5 starting-style positions on CPU, compare to GPU at depth=2/3 and
# look for systematic offset (eval mismatch causes constant offset over many
# positions).
# -----------------------------------------------------------------------------

EVAL_SANITY_FENS = [
    # Starting position
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    # Kiwipete
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    # Simple endgame K+P vs K
    "8/8/8/4k3/8/4K3/4P3/8 w - - 0 1",
    # Material lopsided (black has extra queen)
    "4k3/8/8/8/8/8/8/4K2Q w - - 0 1",
    # Black to move, even material
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1",
]


HAND_FENS = [
    ("startpos",
     "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
    ("kiwipete",
     "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"),
    ("kp_v_k",
     "8/8/8/4k3/8/4K3/4P3/8 w - - 0 1"),
]


def cmd_eval_sanity():
    """Eval sanity: pure CPU eval on 5 positions, displayed alongside what
    the GPU returns at depth=2 (the lowest meaningful search depth). We
    cannot pull raw GPU eval without modifying the binary, so this gives
    us a sanity check on rough magnitudes and signs.

    More importantly: at depth=2, gpu_score for the position equals
        -evaluate(grandchild_after_best_pair)
    and on CPU we run our own depth-2 search using OUR ported eval. If both
    agree, the eval port is consistent. If GPU consistently differs by a
    constant, eval magnitudes likely differ. If GPU has wildly different
    moves and scores, search semantics differ.
    """
    print("=" * 72)
    print("EVAL SANITY (5 positions)")
    print("CPU eval = direct ported PeSTO. GPU score = depth-2 root search.")
    print("=" * 72)
    rows = []
    for fen in EVAL_SANITY_FENS:
        b = chess.Board(fen)
        cpu_eval_raw = evaluate_pesto(b)
        try:
            gpu = run_gpu(fen, 2)
            gpu_best = gpu.get("bestmove", "?")
            gpu_score = gpu.get("score", "?")
        except Exception as e:
            gpu_best = f"ERR:{e}"
            gpu_score = "?"
        cpu_best, cpu_score, _ = root_search(b, 2)
        rows.append({
            "fen": fen,
            "cpu_raw_eval": cpu_eval_raw,
            "cpu_d2_best": cpu_best,
            "cpu_d2_score": cpu_score,
            "gpu_d2_best": gpu_best,
            "gpu_d2_score": gpu_score,
        })
        print(f"\nFEN {fen}")
        print(f"  CPU raw eval (PeSTO, side POV): {cpu_eval_raw}")
        print(f"  CPU d=2 search: best={cpu_best:7s}  score={cpu_score:+5d}")
        try:
            gpu_sc_str = f"{int(gpu_score):+5d}"
        except Exception:
            gpu_sc_str = str(gpu_score)
        print(f"  GPU d=2 search: best={str(gpu_best):7s}  score={gpu_sc_str}")
    return rows


def cmd_hand_check():
    """Print CPU vs GPU side by side for 3 hand-check positions at d=2 and d=3."""
    print("=" * 72)
    print("HAND CHECK (3 positions, d=2 and d=3)")
    print("=" * 72)
    results = []
    for label, fen in HAND_FENS:
        for d in (2, 3):
            b = chess.Board(fen)
            t0 = time.time()
            cpu_best, cpu_score, _ = root_search(b, d)
            cpu_ms = 1000 * (time.time() - t0)
            try:
                gpu = run_gpu(fen, d)
                gpu_best = gpu.get("bestmove", "?")
                gpu_score = gpu.get("score", "?")
                gpu_ms = gpu.get("kernel_ms", -1)
            except Exception as e:
                gpu_best = f"ERR:{e}"
                gpu_score = "?"
                gpu_ms = -1
            same_move = (cpu_best == gpu_best)
            try:
                same_score = abs(int(gpu_score) - int(cpu_score)) <= 1
            except Exception:
                same_score = False
            try:
                gpu_sc_str = f"{int(gpu_score):+6d}"
            except Exception:
                gpu_sc_str = str(gpu_score)
            print(f"\n[{label}] depth={d}")
            print(f"  FEN {fen}")
            print(f"  CPU:  best={cpu_best:7s}  score={cpu_score:+6d}  ({cpu_ms:.0f} ms)")
            print(f"  GPU:  best={str(gpu_best):7s}  score={gpu_sc_str}  ({gpu_ms} ms kernel)")
            print(f"  match: move={'Y' if same_move else 'N'}  score={'Y' if same_score else 'N'}")
            results.append({
                "label": label,
                "fen": fen,
                "depth": d,
                "cpu_best": cpu_best, "cpu_score": cpu_score,
                "gpu_best": gpu_best, "gpu_score": gpu_score,
                "same_move": same_move, "same_score": same_score,
            })
    return results


def load_book(path: str, n: int, seed: int = 1234):
    fens = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                j = json.loads(line)
            except Exception:
                continue
            fen = j.get("fen")
            if fen:
                fens.append(fen)
    rng = random.Random(seed)
    rng.shuffle(fens)
    out = []
    for fen in fens:
        # Filter out terminal/near-terminal positions: we want positions with
        # at least 2 legal moves so the comparison is meaningful.
        try:
            b = chess.Board(fen)
            if b.is_game_over():
                continue
            if b.legal_moves.count() < 2:
                continue
        except Exception:
            continue
        out.append(fen)
        if len(out) >= n:
            break
    return out


def gpu_score_for_specific_move(fen: str, move_uci: str, depth: int) -> int:
    """When CPU and GPU pick different moves, we still want to confirm they
    agree on the score for a SPECIFIC move (the GPU's choice). The GPU's
    JSON output already includes per-root-move scores; we pull them from
    a single GPU call and look up the specific move."""
    j = run_gpu(fen, depth)
    for entry in j.get("moves", []):
        if entry.get("move") == move_uci:
            return int(entry.get("score"))
    raise KeyError(f"GPU did not score move {move_uci} for {fen}")


def cpu_score_for_specific_move(fen: str, move_uci: str, depth: int) -> int:
    """Run a CPU search where we force the root move and report the score."""
    b = chess.Board(fen)
    found = None
    for mv in b.legal_moves:
        if mv.uci() == move_uci:
            found = mv
            break
    if found is None:
        raise KeyError(f"CPU does not consider {move_uci} legal in {fen}")
    search_depth = depth - 1
    if search_depth < 1:
        search_depth = 1
    b.push(found)
    opp = negamax(b, search_depth, -INF_SCORE, INF_SCORE)
    b.pop()
    return -opp


def run_full(n: int, seed: int):
    book_path = os.path.join(REPO, "gpu_spine", "book.jsonl")
    fens = load_book(book_path, n, seed)
    print(f"loaded {len(fens)} positions from book")

    report = []
    summary = {
        "d2_move_match": 0, "d2_score_match": 0, "d2_total": 0,
        "d3_move_match": 0, "d3_score_match": 0, "d3_total": 0,
        "d2_score_match_with_swap": 0, "d3_score_match_with_swap": 0,
    }
    worst = []  # list of dicts for largest disagreements

    t_start = time.time()
    for idx, fen in enumerate(fens):
        for d in (2, 3):
            b = chess.Board(fen)
            try:
                t0 = time.time()
                cpu_best, cpu_score, _ = root_search(b, d)
                cpu_ms = 1000 * (time.time() - t0)
            except Exception as e:
                report.append({"fen": fen, "depth": d, "error": f"cpu:{e}"})
                continue
            try:
                gpu = run_gpu(fen, d, timeout=60.0)
                gpu_best = gpu.get("bestmove")
                gpu_score = int(gpu.get("score"))
                gpu_moves = {m["move"]: int(m["score"]) for m in gpu.get("moves", [])}
            except Exception as e:
                report.append({"fen": fen, "depth": d, "error": f"gpu:{e}"})
                continue

            move_match = (cpu_best == gpu_best)
            score_match = abs(cpu_score - gpu_score) <= 1

            # If moves differ but scores match (multi-best-move case), record.
            # Also: cross-check by asking each side to score the OTHER's pick.
            cross_score_match = False
            cpu_score_for_gpu_move = None
            if not move_match:
                try:
                    cpu_score_for_gpu_move = cpu_score_for_specific_move(fen, gpu_best, d)
                    if abs(cpu_score_for_gpu_move - gpu_score) <= 1:
                        # GPU's bestmove gets the same score on CPU as GPU
                        # reports. That's the multi-best-move case.
                        cross_score_match = True
                except Exception:
                    pass

            entry = {
                "fen": fen,
                "depth": d,
                "cpu_best": cpu_best,
                "gpu_best": gpu_best,
                "cpu_score": cpu_score,
                "gpu_score": gpu_score,
                "move_match": move_match,
                "score_match": score_match,
                "cross_score_match": cross_score_match,
                "cpu_ms": round(cpu_ms, 1),
            }
            if not move_match and gpu_moves and cpu_best in gpu_moves:
                entry["gpu_score_for_cpu_move"] = gpu_moves[cpu_best]
            if cpu_score_for_gpu_move is not None:
                entry["cpu_score_for_gpu_move"] = cpu_score_for_gpu_move
            report.append(entry)

            key = f"d{d}"
            summary[f"{key}_total"] += 1
            if move_match:
                summary[f"{key}_move_match"] += 1
            if score_match:
                summary[f"{key}_score_match"] += 1
            if score_match or cross_score_match:
                summary[f"{key}_score_match_with_swap"] += 1

            # Track worst
            try:
                diff = abs(cpu_score - gpu_score)
            except Exception:
                diff = 0
            if (not move_match) or diff > 5:
                worst.append({
                    "fen": fen, "depth": d,
                    "cpu_best": cpu_best, "gpu_best": gpu_best,
                    "cpu_score": cpu_score, "gpu_score": gpu_score,
                    "diff": diff,
                    "cross_score_match": cross_score_match,
                })

        if (idx + 1) % 5 == 0 or idx + 1 == len(fens):
            elapsed = time.time() - t_start
            print(f"  [{idx+1}/{len(fens)}] elapsed {elapsed:.1f}s   "
                  f"d2 mv={summary['d2_move_match']}/{summary['d2_total']} "
                  f"sc={summary['d2_score_match']}/{summary['d2_total']}   "
                  f"d3 mv={summary['d3_move_match']}/{summary['d3_total']} "
                  f"sc={summary['d3_score_match']}/{summary['d3_total']}")

    out_path = os.path.join(REPO, "cuda", "search_report.json")
    with open(out_path, "w") as f:
        json.dump({
            "summary": summary,
            "n_positions": len(fens),
            "seed": seed,
            "worst_disagreements": sorted(worst, key=lambda r: -r["diff"])[:20],
            "per_position": report,
        }, f, indent=2)
    print(f"\nwrote {out_path}")
    print(f"summary: {json.dumps(summary, indent=2)}")
    return summary, report, worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--hand", action="store_true",
                    help="run only the 3 hand-check positions")
    ap.add_argument("--eval-only", action="store_true",
                    help="run only the 5-pos eval sanity check")
    args = ap.parse_args()

    if not os.path.exists(GPU_BIN):
        print(f"ERROR: gpu_fighter binary not found at {GPU_BIN}")
        print("Build with: cd cuda && make  (or ./build_fighter.sh in gpu_spine)")
        sys.exit(2)

    if args.eval_only:
        cmd_eval_sanity()
        return

    if args.hand:
        cmd_hand_check()
        return

    # Default: full pipeline.
    cmd_eval_sanity()
    print()
    cmd_hand_check()
    print()
    print("=" * 72)
    print(f"FULL RUN ({args.n} positions, d=2 and d=3)")
    print("=" * 72)
    run_full(args.n, args.seed)


if __name__ == "__main__":
    main()
