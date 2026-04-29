#!/usr/bin/env python3
"""Minimal GPU vs GPU UCI arena. Zero chess on the host.

Spawns two ./engine processes, drives them via UCI 'position startpos moves
...' + 'go movetime/depth ...'. The host maintains the UCI move list as a
string of UCI tokens — no legality, no FEN parsing, no python-chess. All
chess (movegen, legality, search, eval) executes in the GPU engine.

End conditions:
  - bestmove "0000" or empty  → engine reports no legal moves (terminal)
  - --max-plies cap            → safety
  - per-move time budget cap   → also safety

Outputs:
  - per-move ms (engine think time)
  - per-side total move count + total think-time
  - final move list
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


def open_engine(path: str) -> subprocess.Popen:
    proc = subprocess.Popen(
        [path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
    )
    # Handshake.
    proc.stdin.write("uci\n")
    proc.stdin.flush()
    while True:
        line = proc.stdout.readline()
        if not line:
            raise RuntimeError("engine died during uci handshake")
        if line.strip() == "uciok":
            break
    proc.stdin.write("isready\n")
    proc.stdin.flush()
    while True:
        line = proc.stdout.readline()
        if line.strip() == "readyok":
            break
    proc.stdin.write("ucinewgame\n")
    proc.stdin.flush()
    return proc


def go_move(proc: subprocess.Popen, moves: list[str], spec: str) -> tuple[str, float]:
    """Send 'position startpos moves ...' + 'go <spec>'; return (bestmove, ms)."""
    pos_cmd = "position startpos"
    if moves:
        pos_cmd += " moves " + " ".join(moves)
    proc.stdin.write(pos_cmd + "\n")
    proc.stdin.write(f"go {spec}\n")
    proc.stdin.flush()

    t0 = time.perf_counter()
    bestmove = ""
    while True:
        line = proc.stdout.readline()
        if not line:
            raise RuntimeError("engine died during go")
        line = line.strip()
        if line.startswith("bestmove"):
            parts = line.split()
            if len(parts) >= 2:
                bestmove = parts[1]
            break
    elapsed = (time.perf_counter() - t0) * 1000.0
    return bestmove, elapsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="./engine")
    ap.add_argument("--white", default=None, help="override white-side engine path")
    ap.add_argument("--black", default=None, help="override black-side engine path")
    ap.add_argument("--spec", default="movetime 200",
                    help="UCI go spec (e.g. 'depth 5', 'movetime 200')")
    ap.add_argument("--max-plies", type=int, default=200)
    args = ap.parse_args()

    white_path = Path(args.white or args.engine).resolve()
    black_path = Path(args.black or args.engine).resolve()
    for label, p in (("white", white_path), ("black", black_path)):
        if not p.is_file():
            print(f"{label} engine not found: {p}", file=sys.stderr)
            sys.exit(2)

    print(f"WHITE: {white_path.name}")
    print(f"BLACK: {black_path.name}")
    white = open_engine(str(white_path))
    black = open_engine(str(black_path))

    moves: list[str] = []
    per_side = [[], []]  # white-times, black-times
    side = 0  # 0=white to move
    t_game = time.perf_counter()
    end_reason = "max_plies"

    for ply in range(args.max_plies):
        proc = white if side == 0 else black
        bestmove, ms = go_move(proc, moves, args.spec)
        per_side[side].append(ms)
        print(f"ply {ply:3d}  {'W' if side==0 else 'B'}  {bestmove:<6}  {ms:7.1f} ms")
        if not bestmove or bestmove == "0000":
            end_reason = "no_legal_move (terminal)"
            break
        moves.append(bestmove)
        side ^= 1
    else:
        # exited via max_plies — already set
        pass

    total_s = time.perf_counter() - t_game

    def stats(times):
        if not times:
            return (0, 0.0, 0.0, 0.0)
        return (len(times), sum(times), sum(times) / len(times), max(times))

    w_n, w_sum, w_avg, w_max = stats(per_side[0])
    b_n, b_sum, b_avg, b_max = stats(per_side[1])

    print()
    print(f"=== Game over: {end_reason} ===")
    print(f"plies played:        {len(moves)}")
    print(f"total wall time:     {total_s:.2f}s")
    print(f"white moves:         n={w_n} sum={w_sum:.0f}ms avg={w_avg:.1f}ms max={w_max:.1f}ms")
    print(f"black moves:         n={b_n} sum={b_sum:.0f}ms avg={b_avg:.1f}ms max={b_max:.1f}ms")
    print(f"final move list:     {' '.join(moves)}")

    for proc in (white, black):
        try:
            proc.stdin.write("quit\n")
            proc.stdin.flush()
        except Exception:
            pass
        proc.wait(timeout=2)


if __name__ == "__main__":
    main()
