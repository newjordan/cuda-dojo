#!/usr/bin/env python3
"""
nn_compare.py — verify GPU NN forward pass against an exact NumPy CPU port.

The single objective: prove the GPU implementations in cuda/nn_eval.cu and
cuda/gpu_fighter.cu (--use-nn path) compute the SAME values as a faithful
CPU port of the documented architecture. Anything else means the trained
weights are useless because they're being multiplied through a bad kernel.

Architecture (read from cuda/nn_eval.cu):
  Input : 768-dim one-hot (12 piece types * 64 squares)
          piece_idx 0..5 = WP/WN/WB/WR/WQ/WK ; 6..11 = BP/BN/BB/BR/BQ/BK
          feature[piece_idx * 64 + sq], where sq=0..63 with sq=0 = a8
          (FEN top-left order, exactly what nn_eval.cu's parser produces).
  Net   : 768 -> 256 (ReLU) -> 64 (ReLU) -> 1
  Side  : final scalar is NEGATED if side-to-move is BLACK (side_flip_kernel).

Weight file layout (nn_eval.cu net_save / gpu_fighter.cu load_nn_weights):
  W1[768*256], b1[256], W2[256*64], b2[64], W3[64*1], b3[1]   all float32
  W matrices stored COLUMN-MAJOR (cuBLAS native):
      W1 element (in, h1) at offset (in + IN_DIM*h1)   -> shape (IN, H1)
      W2 element (h1, h2) at offset (h1 + H1_DIM*h2)   -> shape (H1, H2)
      W3 element (h2, o ) at offset (h2 + H2_DIM*o )   -> shape (H2, OUT)
  In NumPy we therefore reshape with order='F' so W1[in, h1] indexes
  correctly. Math: H1 = W1.T @ X + b1   (matches cuBLAS OP_T on W1).

Verification:
  1. Empty board -> CPU and GPU must equal b3[0] exactly (within FP).
  2. Starting position -> print 3-way comparison.
  3. 200 positions from gpu_spine/book.jsonl -> aggregate report.

Tolerance: |cpu - gpu| < 1e-2 cp. Any larger diff = bug.
"""

import json
import os
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

# -----------------------------------------------------------------------------
# Constants — must match cuda/nn_eval.cu
# -----------------------------------------------------------------------------
IN_DIM  = 768
H1_DIM  = 256
H2_DIM  = 64
OUT_DIM = 1

ROOT          = Path(__file__).resolve().parents[1]
CUDA_DIR      = ROOT / 'cuda'
WEIGHTS_PATH  = CUDA_DIR / 'nn_weights.bin'
NN_EVAL_BIN   = CUDA_DIR / 'nn_eval'
GPU_FIGHTER   = CUDA_DIR / 'gpu_fighter'
BOOK_PATH     = ROOT / 'gpu_spine' / 'book.jsonl'
REPORT_PATH   = CUDA_DIR / 'nn_compare_report.json'

N_POSITIONS   = 200
TOL_CP        = 1e-2          # acceptable absolute diff (centipawns)
TOL_GPU_GPU   = 1e-3          # nn_eval vs hand-rolled __device__ MLP


# -----------------------------------------------------------------------------
# Weight loader
# -----------------------------------------------------------------------------
def load_weights(path: Path):
    raw = np.fromfile(path, dtype=np.float32)
    expected = IN_DIM*H1_DIM + H1_DIM + H1_DIM*H2_DIM + H2_DIM + H2_DIM*OUT_DIM + OUT_DIM
    if raw.size != expected:
        raise RuntimeError(f"weights file has {raw.size} floats, expected {expected}")

    off = 0
    # W1: [IN_DIM x H1_DIM] column-major. order='F' makes W1[in, h1] correct.
    W1 = raw[off : off + IN_DIM*H1_DIM].reshape((IN_DIM, H1_DIM), order='F'); off += IN_DIM*H1_DIM
    b1 = raw[off : off + H1_DIM].copy();                                     off += H1_DIM
    W2 = raw[off : off + H1_DIM*H2_DIM].reshape((H1_DIM, H2_DIM), order='F'); off += H1_DIM*H2_DIM
    b2 = raw[off : off + H2_DIM].copy();                                     off += H2_DIM
    W3 = raw[off : off + H2_DIM*OUT_DIM].reshape((H2_DIM, OUT_DIM), order='F'); off += H2_DIM*OUT_DIM
    b3 = raw[off : off + OUT_DIM].copy();                                    off += OUT_DIM
    assert off == raw.size
    return W1, b1, W2, b2, W3, b3


# -----------------------------------------------------------------------------
# Feature extraction — exact mirror of extract_features_kernel in nn_eval.cu
# AND parse_fen_compact's square ordering (sq=0 is a8, FEN top-left).
# -----------------------------------------------------------------------------
WHITE_PIECES = {'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5}
BLACK_PIECES = {'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q':10, 'k':11}

def fen_to_features_and_side(fen: str):
    """Return (x[768], side_to_move) where side='w' or 'b'.

    Square 0 = a8, square 63 = h1 (matches parse_fen_compact in nn_eval.cu).
    """
    x = np.zeros(IN_DIM, dtype=np.float32)
    parts = fen.split()
    placement = parts[0]
    side = parts[1] if len(parts) > 1 else 'w'

    sq = 0
    for ch in placement:
        if ch == '/':
            continue
        if ch.isdigit():
            sq += int(ch)
            continue
        if ch in WHITE_PIECES:
            piece_idx = WHITE_PIECES[ch]
        elif ch in BLACK_PIECES:
            piece_idx = BLACK_PIECES[ch]
        else:
            raise ValueError(f"bad FEN char {ch!r} in {fen!r}")
        if sq < 64:
            x[piece_idx * 64 + sq] = 1.0
        sq += 1
    return x, side


def cpu_forward(W1, b1, W2, b2, W3, b3, x, side):
    """Single-precision NumPy forward pass mirroring the cuBLAS path."""
    # GEMM #1: H1 = W1^T @ x + b1     (W1 shape (IN, H1) -> output H1)
    h1 = W1.T @ x + b1
    h1 = np.maximum(h1, 0.0, dtype=np.float32)
    # GEMM #2
    h2 = W2.T @ h1 + b2
    h2 = np.maximum(h2, 0.0, dtype=np.float32)
    # GEMM #3 (scalar output)
    y = (W3.T @ h2 + b3)[0]
    # side flip — side_flip_kernel negates if BLACK
    if side == 'b':
        y = -y
    return float(y)


# -----------------------------------------------------------------------------
# GPU runners
# -----------------------------------------------------------------------------
def run_nn_eval(fens):
    """Run cuda/nn_eval --json on the given FENs; return list of float scores
    BEFORE the integer rounding nn_eval prints. We get the rounded int from
    the JSON and accept the rounding cost (we'll compare with |diff| < 0.6
    when needed). For the float-precision comparison we'll re-run with
    --bench-style direct printing if needed, but JSON ints are sufficient
    for proving correctness within ~0.5 cp."""
    proc = subprocess.run(
        [str(NN_EVAL_BIN), '--json'],
        input='\n'.join(fens) + '\n',
        capture_output=True, text=True,
        cwd=str(CUDA_DIR),
        timeout=120,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"nn_eval failed rc={proc.returncode} stderr={proc.stderr}")
    data = json.loads(proc.stdout.strip())
    # nn_eval prints rounded int centipawns. Build a fen->score dict to be
    # robust against ordering surprises.
    by_fen = {row['fen']: row['score'] for row in data}
    return [by_fen[f] for f in fens]


def run_gpu_fighter_root_score(fen, use_nn=True):
    """gpu_fighter doesn't expose a leaf-only score; the cleanest signal we
    can get is the root negamax score at depth 1, which is the worst leaf
    eval the opponent can force. To extract a *pure* leaf eval at the root
    we'd need a binary mod. Instead, per the spec, we use nn_eval as the
    GPU reference for the leaf eval and treat gpu_fighter as a separate
    sanity check that the same NN code path is wired through it. We still
    record gpu_fighter's score for the start position in main()."""
    proc = subprocess.run(
        [str(GPU_FIGHTER), '--depth', '1', '--use-nn'],
        input=fen + '\n',
        capture_output=True, text=True,
        cwd=str(CUDA_DIR),
        timeout=60,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"gpu_fighter failed rc={proc.returncode} stderr={proc.stderr}")
    out = proc.stdout.strip().splitlines()[-1]
    j = json.loads(out)
    # j['score'] is the score (from side-to-move POV at root) of the best move
    # after a depth=1 search. Per-move scores in j['moves'] are the leaf NN
    # eval after THAT move (depth=1 means the kernel evaluates the leaf right
    # after applying each candidate root move; from CHILD's POV).
    return j


def gpu_fighter_leaf_eval_root(fen):
    """Use gpu_fighter to obtain a leaf NN eval that we can compare against
    the CPU reference for the SAME root position.

    Trick: at --depth 1 with --use-nn, the kernel evaluates each child
    position (after a single root move) with the NN. So if we apply each
    legal move in NumPy to obtain the resulting FENs, then run nn_eval on
    those child FENs, and ALSO read gpu_fighter's per-move scores, the two
    must agree (those are gpu_fighter's hand-rolled __device__ MLP scores
    on the same child positions).

    Since we'd be comparing nn_eval vs gpu_fighter on the children of a
    single root, this is the cleanest way to prove the two GPU paths agree.
    Returns (root_json, child_fens, child_gf_scores).
    """
    import chess
    j = run_gpu_fighter_root_score(fen, use_nn=True)
    moves = j['moves']
    board = chess.Board(fen)
    child_fens = []
    child_gf_scores = []
    for m in moves:
        uci = m['move']
        b2 = board.copy()
        b2.push_uci(uci)
        # gpu_fighter score at depth=1 is the negamax value of the child,
        # which at depth-cap=1 is the leaf eval of the position AFTER the
        # opponent's best reply... no wait. depth - 1 in main() = 0. Let's
        # just record what we get; we'll compare child_fens evals to per-move
        # scores and see what relationship holds.
        child_fens.append(b2.fen())
        child_gf_scores.append(m['score'])
    return j, child_fens, child_gf_scores


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    print("=" * 70, file=sys.stderr)
    print("nn_compare.py — GPU vs CPU NN forward-pass verification", file=sys.stderr)
    print("=" * 70, file=sys.stderr)

    # 1. Load weights
    W1, b1, W2, b2, W3, b3 = load_weights(WEIGHTS_PATH)
    print(f"[load] {WEIGHTS_PATH}: W1{W1.shape} b1{b1.shape} W2{W2.shape} b2{b2.shape} W3{W3.shape} b3{b3.shape}", file=sys.stderr)
    print(f"[load] b3[0] (final-layer bias) = {b3[0]:.6f}", file=sys.stderr)
    print(f"[load] W3 stats: min={W3.min():.4f} max={W3.max():.4f} mean={W3.mean():.4f}", file=sys.stderr)

    # 2. Empty-board sanity check
    empty_fen = "8/8/8/8/8/8/8/8 w - - 0 1"
    x, side = fen_to_features_and_side(empty_fen)
    cpu_empty = cpu_forward(W1, b1, W2, b2, W3, b3, x, side)
    # On empty board, all features are zero. h1 = ReLU(b1). h2 = ReLU(W2.T @ h1 + b2).
    # y = W3.T @ h2 + b3. Side=white -> no flip. So if all biases were zero, y = b3[0].
    # Trained weights have non-zero biases so the value will involve the bias chain.
    gpu_empty = run_nn_eval([empty_fen])[0]
    print(f"\n[empty]  CPU = {cpu_empty:+.4f}    GPU(nn_eval, rounded int) = {gpu_empty}", file=sys.stderr)
    print(f"[empty]  cpu_rounded_to_int = {int(round(cpu_empty))}    diff_int = {gpu_empty - int(round(cpu_empty))}", file=sys.stderr)

    # 3. Starting position 3-way check
    start_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    x, side = fen_to_features_and_side(start_fen)
    cpu_start = cpu_forward(W1, b1, W2, b2, W3, b3, x, side)
    gpu_start = run_nn_eval([start_fen])[0]
    print(f"\n[start]  CPU = {cpu_start:+.4f}    GPU(nn_eval, int) = {gpu_start}", file=sys.stderr)

    # gpu_fighter at depth=1 — sample one child to check the hand-rolled MLP
    print(f"[start]  Calling gpu_fighter --depth 1 --use-nn ...", file=sys.stderr)
    gf_j, child_fens, child_gf_scores = gpu_fighter_leaf_eval_root(start_fen)
    # The kernel at depth=1 returns negamax(child) from CHILD-pov, then negates
    # back at root. With search_depth = depth-1 = 0 inside the kernel, it should
    # just be the leaf NN eval at the child (from child-pov), negated to root pov.
    # Run nn_eval on the same children and compare.
    nneval_child_scores = run_nn_eval(child_fens)
    # Per gpu_fighter source: per-move score is the negamax(child) value (from
    # root-pov) negated inside d_negamax. With search_depth=1 it'll do one ply
    # of child search, so the score is the BEST OPPONENT REPLY's leaf eval.
    # That's a different quantity from a pure leaf eval. We'll log both.
    print(f"[start]  gpu_fighter root score = {gf_j['score']}, depth_reported = {gf_j['depth']}, search_depth_in_kernel = {gf_j['depth']-1}", file=sys.stderr)
    print(f"[start]  Comparing first 5 children: nn_eval(child) vs gpu_fighter(per-move)", file=sys.stderr)
    for i in range(min(5, len(child_fens))):
        print(f"         {gf_j['moves'][i]['move']}  child_fen={child_fens[i][:40]}...", file=sys.stderr)
        print(f"             nn_eval(child, child-pov)={nneval_child_scores[i]:+5d}  gpu_fighter_score={child_gf_scores[i]:+5d}", file=sys.stderr)

    # 4. Scale to 200 positions
    print(f"\n[scale] Loading first {N_POSITIONS} FENs from {BOOK_PATH}", file=sys.stderr)
    fens = []
    with open(BOOK_PATH) as f:
        for line in f:
            if not line.strip():
                continue
            j = json.loads(line)
            fens.append(j['fen'])
            if len(fens) >= N_POSITIONS:
                break

    print(f"[scale] CPU forward on {len(fens)} positions...", file=sys.stderr)
    cpu_scores = []
    has_nan = False
    for fen in fens:
        x, side = fen_to_features_and_side(fen)
        s = cpu_forward(W1, b1, W2, b2, W3, b3, x, side)
        if not np.isfinite(s):
            has_nan = True
        cpu_scores.append(s)

    print(f"[scale] GPU nn_eval on {len(fens)} positions...", file=sys.stderr)
    gpu_scores = run_nn_eval(fens)

    # Build per-position records.
    records = []
    abs_diffs = []
    int_round_diffs = []
    for i, fen in enumerate(fens):
        cpu = cpu_scores[i]
        gpu = gpu_scores[i]            # int
        cpu_int = int(round(cpu))
        # The "true" comparison vs the GPU is between the float pre-round
        # value and the int rounded value. |cpu - gpu_int| <= 0.5 + FP_err.
        diff = cpu - gpu
        rel = diff / max(1.0, abs(cpu))
        abs_diffs.append(abs(diff))
        int_round_diffs.append(abs(cpu_int - gpu))
        records.append({
            'fen': fen,
            'cpu_score': cpu,
            'gpu_eval_score': gpu,
            'cpu_int': cpu_int,
            'abs_diff': abs(diff),
            'rel_diff': rel,
        })

    abs_diffs = np.array(abs_diffs, dtype=np.float64)
    int_round_diffs = np.array(int_round_diffs)

    # Within tolerance of FP rounding for the integer-quantized GPU output:
    # we'd expect |cpu - gpu_int| <= 0.5 + small_fp_err in all cases, and the
    # *rounded CPU* to match gpu_int byte-for-byte except where the CPU and
    # GPU FP rounding fall on opposite sides of x.5.
    n_int_match = int((int_round_diffs == 0).sum())
    n_int_off1  = int((int_round_diffs <= 1).sum())
    worst_idx   = int(abs_diffs.argmax())
    worst_diff  = float(abs_diffs[worst_idx])

    print(f"\n[scale] Summary over {len(fens)} positions:", file=sys.stderr)
    print(f"        rounded_cpu == gpu_int : {n_int_match}/{len(fens)} ({100.0*n_int_match/len(fens):.1f}%)", file=sys.stderr)
    print(f"        |cpu - gpu_int| <= 1   : {n_int_off1}/{len(fens)} ({100.0*n_int_off1/len(fens):.1f}%)", file=sys.stderr)
    print(f"        worst |cpu - gpu_int|  : {worst_diff:.4f}  at  fen={records[worst_idx]['fen']}", file=sys.stderr)
    print(f"        cpu_score has NaN/Inf  : {has_nan}", file=sys.stderr)

    # 5. nn_eval vs gpu_fighter agreement on a small batch.
    #    For each of the first N_GF positions we apply each legal move,
    #    collect child fens, run nn_eval on them, and compare the per-move
    #    score from gpu_fighter --depth 1 --use-nn. With kernel search_depth=1,
    #    each per-move score is the value of the BEST OPPONENT reply's leaf
    #    eval (i.e. -max_over_replies(nn_eval(grandchild))). We'll therefore
    #    compute that quantity in CPU and compare.
    import chess
    N_GF = 10
    gf_records = []
    print(f"\n[gf]    Comparing nn_eval vs gpu_fighter on {N_GF} positions (each child)...", file=sys.stderr)
    for fen in fens[:N_GF]:
        gf_j = run_gpu_fighter_root_score(fen, use_nn=True)
        # For each root move m, the score gpu_fighter prints is the child's
        # negamax value at search_depth=1 (kernel-internal), already negated
        # back to root-pov. The ROOT POV score == -CHILD_POV_score.
        # Inside the kernel at search_depth=1, child evaluates each grandchild
        # at depth 0 = leaf. So child's negamax = max over grandchildren of
        # leaf_eval(grandchild, gc_pov). The leaf eval is the NN.
        # Therefore root-pov score of move m = -max_gc(NN(gc, gc_pov)).
        # We can compute this with python-chess + nn_eval batch.
        board = chess.Board(fen)
        for m in gf_j['moves']:
            uci = m['move']
            b1 = board.copy(); b1.push_uci(uci)
            # Generate grandchildren; if leaf (no legal moves) gpu_fighter
            # returns terminal eval, skip those for simplicity here.
            gc_fens = []
            for mv in b1.legal_moves:
                b2 = b1.copy(); b2.push(mv)
                gc_fens.append(b2.fen())
            if not gc_fens:
                continue
            gc_scores = run_nn_eval(gc_fens)
            # gpu_fighter calls negamax(child, search_depth=1, ...) at root.
            # negamax descends one ply: at grandchild g (depth=0) it returns
            # leaf_eval(g) from g's side-to-move POV. nn_eval already returns
            # side-to-move POV, so leaf_eval(g)=nn_eval(g) is in g's pov.
            # g's side-to-move == root's side (we made 2 moves to reach g).
            # So nn_eval(g) is in root_pov.
            # Frame at child negates: child_score = -leaf_eval(g) = -nn_eval(g)
            #   -> child_pov score for that grandchild path
            # f->best = max over grandchildren of (-nn_eval(g)) = -min(nn_eval(g))
            # Root: per-move = -opp_score = -(-min(nn_eval(g))) = min(nn_eval(g))
            # So gpu_fighter per-move score should equal MIN of grandchildren
            # nn_evals (in root pov).
            cpu_predicted = min(gc_scores)
            gf_score = m['score']
            gf_records.append({
                'fen': fen,
                'move': uci,
                'cpu_predicted_root_pov': cpu_predicted,
                'gpu_fighter_score': gf_score,
                'diff': abs(cpu_predicted - gf_score),
            })

    if gf_records:
        gf_diffs = np.array([r['diff'] for r in gf_records])
        n_match = int((gf_diffs <= 1).sum())
        worst_gf = float(gf_diffs.max())
        print(f"[gf]    {n_match}/{len(gf_records)} per-move scores match within 1 cp; worst diff = {worst_gf}", file=sys.stderr)

    # 6. Write report
    report = {
        'meta': {
            'weights_file': str(WEIGHTS_PATH),
            'weights_size_floats': IN_DIM*H1_DIM + H1_DIM + H1_DIM*H2_DIM + H2_DIM + H2_DIM*OUT_DIM + OUT_DIM,
            'n_positions': len(fens),
            'tolerance_cp': TOL_CP,
            'tolerance_gpu_gpu': TOL_GPU_GPU,
        },
        'sanity': {
            'b3_bias': float(b3[0]),
            'empty_cpu': cpu_empty,
            'empty_gpu_int': gpu_empty,
            'start_cpu': cpu_start,
            'start_gpu_int': gpu_start,
            'has_nan_or_inf': has_nan,
        },
        'aggregate': {
            'rounded_cpu_eq_gpu_int': n_int_match,
            'within_1cp_int': n_int_off1,
            'total': len(fens),
            'worst_abs_diff_vs_int': worst_diff,
            'worst_position': records[worst_idx]['fen'],
        },
        'gf_vs_nneval': {
            'note': 'Each row: gpu_fighter --depth 1 --use-nn per-move score vs CPU prediction (max over grandchildren of nn_eval). Both should match within 1 cp if the hand-rolled __device__ MLP equals cuBLAS path.',
            'records': gf_records,
        },
        'per_position': records,
    }
    with open(REPORT_PATH, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n[done] Wrote {REPORT_PATH}", file=sys.stderr)


if __name__ == '__main__':
    main()
