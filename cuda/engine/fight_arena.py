#!/usr/bin/env python3
"""Two-engine UCI match arena: GPU engine FIGHTS Stockfish.

Each match starts from a different opening position sampled from the
elite-master corpus (depth-1 to depth-6 random opening prefix), so no
two matches are identical. Both engines play via UCI; the harness
relays moves and tracks the outcome.

Outputs: per-match result (1-0 / 0-1 / 1/2-1/2 / max-plies-cap),
plus aggregate W/L/D from the GPU engine's POV.
"""
from __future__ import annotations

import argparse
import random
import re
import subprocess
import sys
import time
from pathlib import Path

import chess  # CPU oracle for position->FEN, terminal detection (verification only)


def open_uci(path: str, hash_mb: int = 64) -> subprocess.Popen:
    proc = subprocess.Popen(
        [path],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL, text=True, bufsize=1,
    )
    proc.stdin.write("uci\n"); proc.stdin.flush()
    while True:
        line = proc.stdout.readline()
        if not line: raise RuntimeError(f"engine died: {path}")
        if line.strip() == "uciok": break
    proc.stdin.write(f"setoption name Hash value {hash_mb}\n")
    proc.stdin.write("isready\n"); proc.stdin.flush()
    while True:
        if proc.stdout.readline().strip() == "readyok": break
    return proc


def go(proc: subprocess.Popen, fen: str, moves: list[str], spec: str) -> str:
    """Send position + go, return bestmove UCI string."""
    pos = f"position fen {fen}"
    if moves:
        pos += " moves " + " ".join(moves)
    proc.stdin.write(pos + "\n")
    proc.stdin.write(f"go {spec}\n")
    proc.stdin.flush()
    while True:
        line = proc.stdout.readline()
        if not line: raise RuntimeError("engine died at go")
        line = line.strip()
        if line.startswith("bestmove"):
            parts = line.split()
            return parts[1] if len(parts) >= 2 else "0000"


def play_match(white: subprocess.Popen, black: subprocess.Popen,
                opening_fen: str, max_plies: int, spec: str) -> dict:
    """Play one match; return {result: str, plies: int, moves: [...]}."""
    board = chess.Board(opening_fen)
    moves: list[str] = []
    plies = 0
    while plies < max_plies:
        if board.is_checkmate():
            return {"result": "0-1" if board.turn == chess.WHITE else "1-0",
                    "plies": plies, "moves": moves, "termination": "checkmate"}
        if board.is_stalemate():
            return {"result": "1/2-1/2", "plies": plies, "moves": moves,
                    "termination": "stalemate"}
        if board.is_insufficient_material():
            return {"result": "1/2-1/2", "plies": plies, "moves": moves,
                    "termination": "insufficient_material"}
        if board.can_claim_threefold_repetition():
            return {"result": "1/2-1/2", "plies": plies, "moves": moves,
                    "termination": "threefold"}
        if board.can_claim_fifty_moves():
            return {"result": "1/2-1/2", "plies": plies, "moves": moves,
                    "termination": "fifty_move"}
        engine = white if board.turn == chess.WHITE else black
        try:
            uci = go(engine, opening_fen, moves, spec)
        except Exception as e:
            return {"result": "*", "plies": plies, "moves": moves,
                    "termination": f"engine_error: {e}"}
        if uci == "0000" or uci == "(none)" or not uci:
            return {"result": "*", "plies": plies, "moves": moves,
                    "termination": "engine_returned_null"}
        try:
            mv = chess.Move.from_uci(uci)
        except Exception:
            return {"result": "*", "plies": plies, "moves": moves,
                    "termination": f"invalid_uci: {uci!r}"}
        if mv not in board.legal_moves:
            return {"result": "*", "plies": plies, "moves": moves,
                    "termination": f"illegal_move: {uci}"}
        board.push(mv); moves.append(uci); plies += 1
    return {"result": "*", "plies": plies, "moves": moves,
            "termination": "max_plies"}


def sample_openings(pgn_path: Path, n: int, seed: int,
                     min_ply: int = 4, max_ply: int = 8) -> list[str]:
    """Sample n opening FENs from a PGN. Pure host PGN parse — no chess
    rules executed; we use python-chess only to convert SAN→FEN, which
    is verification (the GPU engine doesn't see this code path)."""
    from parity_test import stream_pgn, reservoir_sample
    rng = random.Random(seed)
    games, _ = reservoir_sample(stream_pgn(pgn_path), k=n*2, seed=seed,
                                 max_scan=20000)
    out = []
    for g in games:
        if len(out) >= n: break
        if not g["moves"]: continue
        ply = rng.randint(min_ply, min(max_ply, len(g["moves"]) - 1))
        board = chess.Board()
        try:
            for san in g["moves"][:ply]:
                board.push_san(san)
        except Exception:
            continue
        out.append(board.fen())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu",
        default="/home/frosty40/VC_V1/cuda-dojo-public/cuda/engine/engine")
    ap.add_argument("--stockfish",
        default="/home/frosty40/VC_V1/TheForge-Battle-Academy/trainers/stockfish/stockfish_bin")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--max-plies", type=int, default=200)
    ap.add_argument("--gpu-spec", default="depth 5",
                    help="UCI go spec for the GPU engine")
    ap.add_argument("--sf-spec", default="depth 3",
                    help="UCI go spec for Stockfish")
    ap.add_argument("--openings",
        default="/srv/models-hdd/chess-games/elo_stacks/double_stack/stack_elite_2400_3299/twic_archive.pgn")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"=== fight_arena: GPU engine vs Stockfish ===")
    print(f"GPU:        {args.gpu}    spec={args.gpu_spec!r}")
    print(f"Stockfish:  {args.stockfish}    spec={args.sf_spec!r}")
    print(f"openings:   {args.openings}")
    print(f"n_matches:  {args.n}    max_plies={args.max_plies}    seed={args.seed}")
    print()

    print(f"sampling {args.n} opening positions...")
    openings = sample_openings(Path(args.openings), args.n, args.seed)
    print(f"got {len(openings)} openings\n")

    gpu_w = 0; gpu_l = 0; draws = 0; errors = 0
    gpu_as_white_records = []
    gpu_as_black_records = []
    t0 = time.perf_counter()

    for i, opening_fen in enumerate(openings):
        gpu_white = (i % 2 == 0)  # alternate colors
        # Reuse engine processes across matches to save startup overhead.
        gpu = open_uci(args.gpu)
        sf = open_uci(args.stockfish)
        try:
            if gpu_white:
                w, b = gpu, sf
                spec_w, spec_b = args.gpu_spec, args.sf_spec
            else:
                w, b = sf, gpu
                spec_w, spec_b = args.sf_spec, args.gpu_spec
            # Send appropriate spec per-side.
            t_match = time.perf_counter()
            # play_match uses one spec per side via its `spec` arg, but we
            # need different specs per engine — patch by using engine-aware go.
            board = chess.Board(opening_fen)
            moves: list[str] = []
            result_str = "*"; termination = "max_plies"
            for ply in range(args.max_plies):
                if board.is_checkmate():
                    result_str = "0-1" if board.turn == chess.WHITE else "1-0"
                    termination = "checkmate"; break
                if board.is_stalemate():
                    result_str = "1/2-1/2"; termination = "stalemate"; break
                if board.is_insufficient_material():
                    result_str = "1/2-1/2"; termination = "insufficient"; break
                if board.can_claim_threefold_repetition():
                    result_str = "1/2-1/2"; termination = "threefold"; break
                if board.can_claim_fifty_moves():
                    result_str = "1/2-1/2"; termination = "fifty_move"; break
                if board.turn == chess.WHITE:
                    eng = w; spec = spec_w
                else:
                    eng = b; spec = spec_b
                try:
                    uci = go(eng, opening_fen, moves, spec)
                except Exception as e:
                    result_str = "*"; termination = f"err: {e}"; break
                if uci in ("0000", "(none)", ""):
                    termination = "null_move"; break
                try:
                    mv = chess.Move.from_uci(uci)
                except Exception:
                    termination = f"bad_uci:{uci}"; break
                if mv not in board.legal_moves:
                    termination = f"illegal:{uci}"; break
                board.push(mv); moves.append(uci)
            else:
                pass  # max_plies hit
            elapsed_ms = (time.perf_counter() - t_match) * 1000

            # Score from GPU's POV.
            if result_str == "1/2-1/2":
                gpu_outcome = "D"; draws += 1
            elif result_str == "1-0":
                gpu_outcome = "W" if gpu_white else "L"
                if gpu_white: gpu_w += 1
                else: gpu_l += 1
            elif result_str == "0-1":
                gpu_outcome = "L" if gpu_white else "W"
                if gpu_white: gpu_l += 1
                else: gpu_w += 1
            else:
                gpu_outcome = "?"; errors += 1
            rec = {
                "i": i, "gpu_color": "W" if gpu_white else "B",
                "result": result_str, "termination": termination,
                "plies": len(moves), "ms": elapsed_ms,
                "fen": opening_fen, "gpu_outcome": gpu_outcome,
            }
            (gpu_as_white_records if gpu_white else gpu_as_black_records).append(rec)
            print(f"  [{i+1:3d}/{len(openings)}] gpu={rec['gpu_color']}  "
                  f"{result_str:<8} {termination:<22} plies={rec['plies']:3d}  "
                  f"{elapsed_ms:6.0f}ms  -> GPU={gpu_outcome}")
        finally:
            for proc in (gpu, sf):
                try:
                    proc.stdin.write("quit\n"); proc.stdin.flush()
                    proc.wait(timeout=2)
                except Exception:
                    proc.terminate()

    total_s = time.perf_counter() - t0
    n = len(openings)

    # Score = W + 0.5*D
    gpu_score = gpu_w + 0.5 * draws
    sf_score = gpu_l + 0.5 * draws
    pct = gpu_score / n if n else 0
    elo_diff = -400 * (1 / max(pct, 1e-6) - 1)  # logistic Elo from score%
    if pct > 0 and pct < 1:
        import math
        elo_diff = -400 * math.log10(1 / pct - 1)

    print()
    print(f"=== Tournament report ===")
    print(f"matches:           {n}")
    print(f"GPU wins:          {gpu_w}")
    print(f"GPU losses:        {gpu_l}")
    print(f"draws:             {draws}")
    print(f"errors:            {errors}")
    print(f"GPU score:         {gpu_score}/{n} = {pct:.4f}")
    print(f"~Elo diff (GPU - Stockfish): {elo_diff:+.0f}")
    print(f"  (negative = GPU weaker; positive = GPU stronger)")
    print(f"total wall:        {total_s:.1f}s  ({total_s/n*1000:.0f}ms/match)")


if __name__ == "__main__":
    sys.exit(main())
