#!/usr/bin/env python3
"""CPU vs GPU referee outcome comparison.

For each PGN game, replay it through:
  CPU referee: python-chess (verification oracle, gold standard)
  GPU referee: dojo_ref (cuda-dojo via librefcuda.so)

Per game: did each referee accept every recorded move? Compute:
  - CPU solved set
  - GPU solved set
  - intersection (both solved), CPU-only, GPU-only, neither
  - agreement rate + sigma

Any game where the two referees disagree on a move's legality is a
parity bug to investigate.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import chess  # CPU verification oracle
import chess.pgn

import dojo_ref  # GPU referee

from parity_test import (
    initial_piece_map, sq_to_idx, update_piece_map,
    san_to_uci, stream_pgn, reservoir_sample,
)


# ---------------------------------------------------------------------------
# CPU replay via python-chess.
# ---------------------------------------------------------------------------

def cpu_replay(game: dict) -> tuple[bool, int, str]:
    """Replay through python-chess. Return (ok, plies_replayed, error_msg)."""
    board = chess.Board()
    plies = 0
    for san in game["moves"]:
        try:
            move = board.parse_san(san)
        except Exception as e:
            return False, plies, f"parse_san({san!r}): {e}"
        if move not in board.legal_moves:
            return False, plies, f"move {san} not in legal_moves"
        board.push(move)
        plies += 1
    return True, plies, ""


# ---------------------------------------------------------------------------
# GPU replay via dojo_ref (mirror of parity_test's logic).
# ---------------------------------------------------------------------------

def gpu_replay(game: dict) -> tuple[bool, int, str]:
    pos = dojo_ref.Position.startpos()
    pmap = initial_piece_map()
    mover_white = True
    plies = 0
    for san in game["moves"]:
        legal = pos.legal_moves()
        if not legal:
            return False, plies, "GPU reports no legal moves"
        uci = san_to_uci(san, pmap, legal, mover_white)
        if uci is None:
            return False, plies, f"SAN→UCI match failed for {san}"
        if uci not in legal:
            return False, plies, f"matched UCI {uci} not in legal set"
        try:
            pos = pos.make_move(uci)
        except Exception as e:
            return False, plies, f"make_move({uci}) raised: {e}"
        update_piece_map(pmap, uci, mover_white)
        mover_white = not mover_white
        plies += 1
    return True, plies, ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pgn",
        default="/srv/models-hdd/chess-games/elo_stacks/double_stack/stack_elite_2400_3299/twic_archive.pgn")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-scan", type=int, default=20000)
    args = ap.parse_args()

    print(f"Sampling {args.n} games from {args.pgn} (max_scan={args.max_scan}, seed={args.seed})")
    games, total_seen = reservoir_sample(
        stream_pgn(Path(args.pgn)), k=args.n,
        seed=args.seed, max_scan=args.max_scan,
    )
    print(f"Scanned {total_seen}, sampled {len(games)} for replay")
    print()

    cpu_pass: list[bool] = []
    gpu_pass: list[bool] = []
    cpu_plies: list[int] = []
    gpu_plies: list[int] = []
    divergences: list[dict] = []

    t0_total = time.perf_counter()
    t_cpu = 0.0
    t_gpu = 0.0
    for i, game in enumerate(games):
        t0 = time.perf_counter()
        c_ok, c_plies, c_err = cpu_replay(game)
        t_cpu += time.perf_counter() - t0

        t0 = time.perf_counter()
        g_ok, g_plies, g_err = gpu_replay(game)
        t_gpu += time.perf_counter() - t0

        cpu_pass.append(c_ok)
        gpu_pass.append(g_ok)
        cpu_plies.append(c_plies)
        gpu_plies.append(g_plies)

        marker = "==" if c_ok == g_ok else "!="
        if c_ok != g_ok:
            divergences.append({
                "idx": i, "game": game, "cpu": (c_ok, c_plies, c_err),
                "gpu": (g_ok, g_plies, g_err),
            })
        if (i + 1) % 25 == 0 or i + 1 == len(games):
            print(f"  [{i+1:3d}/{len(games)}] cpu={int(c_ok)} gpu={int(g_ok)} {marker}  "
                  f"({c_plies}/{g_plies} plies)")

    total_s = time.perf_counter() - t0_total
    n = len(games)

    cpu_solved = sum(cpu_pass)
    gpu_solved = sum(gpu_pass)
    both = sum(1 for a, b in zip(cpu_pass, gpu_pass) if a and b)
    cpu_only = sum(1 for a, b in zip(cpu_pass, gpu_pass) if a and not b)
    gpu_only = sum(1 for a, b in zip(cpu_pass, gpu_pass) if b and not a)
    neither = sum(1 for a, b in zip(cpu_pass, gpu_pass) if not a and not b)
    agreement = (both + neither) / n
    sigma_cpu = math.sqrt(cpu_solved/n * (1 - cpu_solved/n) / n)
    sigma_gpu = math.sqrt(gpu_solved/n * (1 - gpu_solved/n) / n)
    sigma_agree = math.sqrt(agreement * (1 - agreement) / n)

    print()
    print(f"=== CPU vs GPU referee comparison ===")
    print(f"games:                {n}")
    print(f"CPU solved:           {cpu_solved}/{n} = {cpu_solved/n:.4f} (σ={sigma_cpu:.4f})")
    print(f"GPU solved:           {gpu_solved}/{n} = {gpu_solved/n:.4f} (σ={sigma_gpu:.4f})")
    print(f"both solved:          {both}")
    print(f"CPU only:             {cpu_only}")
    print(f"GPU only:             {gpu_only}")
    print(f"neither:              {neither}")
    print(f"agreement:            {agreement:.4f} (σ={sigma_agree:.4f})")
    print(f"divergences:          {len(divergences)}")
    print()
    print(f"timing:")
    print(f"  CPU replay total:   {t_cpu:.2f}s  ({t_cpu*1000/n:.1f}ms/game)")
    print(f"  GPU replay total:   {t_gpu:.2f}s  ({t_gpu*1000/n:.1f}ms/game)")
    print(f"  total wall:         {total_s:.2f}s")
    if cpu_solved > 0 and gpu_solved > 0:
        ratio = t_gpu / t_cpu
        print(f"  GPU/CPU time ratio: {ratio:.2f}x  "
              f"(GPU is {ratio:.1f}x slower per game due to FFI overhead)")

    if divergences:
        print()
        print(f"=== {min(5, len(divergences))} divergence(s) ===")
        for d in divergences[:5]:
            hdrs = d["game"]["headers"]
            print(f"  game {d['idx']}: {hdrs.get('White','?')} vs {hdrs.get('Black','?')}")
            print(f"    cpu: ok={d['cpu'][0]} plies={d['cpu'][1]} err={d['cpu'][2]!r}")
            print(f"    gpu: ok={d['gpu'][0]} plies={d['gpu'][1]} err={d['gpu'][2]!r}")


if __name__ == "__main__":
    sys.exit(main())
