#!/usr/bin/env python3
"""GPU engine trust test against iter21 master-game positions.

For each (fen, played_uci) row in
runtime/datasets/omnifold_diagnostic_iter21.jsonl, send the FEN to the
GPU engine via UCI, ask for its bestmove + eval at fixed depth, and
check whether the engine's choice matches the master's actual move.

Random chance ≈ 1 / mean_legal_moves ≈ 3-4%. A trustworthy engine
should hit ~30-50% on master games at meaningful depth.

The host script does NOT touch chess. It just relays UCI strings.
All chess (movegen, search, eval) executes inside the GPU engine.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path


def open_engine(path: str) -> subprocess.Popen:
    proc = subprocess.Popen(
        [path],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL, text=True, bufsize=1,
    )
    proc.stdin.write("uci\n"); proc.stdin.flush()
    while True:
        line = proc.stdout.readline()
        if not line: raise RuntimeError("engine died at uci")
        if line.strip() == "uciok": break
    proc.stdin.write("isready\n"); proc.stdin.flush()
    while True:
        line = proc.stdout.readline()
        if line.strip() == "readyok": break
    return proc


SCORE_RE = re.compile(r"score (cp|mate) (-?\d+)")


def eval_position(proc: subprocess.Popen, fen: str, spec: str) -> tuple[str, int | None, str]:
    """Send `position fen <fen>` + `go <spec>`. Return (bestmove, score_cp, kind)."""
    proc.stdin.write(f"position fen {fen}\n")
    proc.stdin.write(f"go {spec}\n")
    proc.stdin.flush()
    last_score: int | None = None
    last_kind = "cp"
    while True:
        line = proc.stdout.readline()
        if not line: raise RuntimeError("engine died at go")
        line = line.strip()
        if line.startswith("info"):
            m = SCORE_RE.search(line)
            if m:
                last_kind = m.group(1)
                last_score = int(m.group(2))
        elif line.startswith("bestmove"):
            parts = line.split()
            best = parts[1] if len(parts) >= 2 else ""
            return best, last_score, last_kind


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="./engine")
    ap.add_argument("--diagnostic",
                    default="/home/frosty40/chess_train/runtime/datasets/omnifold_diagnostic_iter21.jsonl")
    ap.add_argument("--n", type=int, default=200, help="positions to evaluate")
    ap.add_argument("--spec", default="depth 5",
                    help="UCI go spec (e.g. 'depth 5', 'movetime 200')")
    args = ap.parse_args()

    engine_path = Path(args.engine).resolve()
    if not engine_path.is_file():
        print(f"engine not found: {engine_path}", file=sys.stderr); sys.exit(2)

    rows = []
    with open(args.diagnostic) as fh:
        for i, line in enumerate(fh):
            if i >= args.n: break
            rows.append(json.loads(line))

    print(f"engine: {engine_path.name}")
    print(f"positions: {len(rows)}  spec: {args.spec}")
    print(f"diagnostic: {args.diagnostic}")
    proc = open_engine(str(engine_path))

    hits = 0
    by_phase = {"opening": [0, 0], "middlegame": [0, 0], "endgame": [0, 0]}
    times = []
    evals_master_won = []  # eval score (white-POV cp) when master picked
    evals_engine = []      # all eval scores for context
    t0 = time.perf_counter()
    for i, rec in enumerate(rows):
        fen = rec["fen"]
        played = rec["played_uci"]
        phase = rec["phase"]
        t1 = time.perf_counter()
        best, score, kind = eval_position(proc, fen, args.spec)
        dt = (time.perf_counter() - t1) * 1000.0
        times.append(dt)
        # White-POV eval — UCI score is STM-POV; flip if black to move.
        mover_white = rec["mover_is_white"]
        if score is not None and kind == "cp":
            score_white_pov = score if mover_white else -score
            evals_engine.append(score_white_pov)
        match = (best == played)
        if match:
            hits += 1
        by_phase[phase][1] += 1
        if match: by_phase[phase][0] += 1
        if (i + 1) % 25 == 0:
            elapsed = time.perf_counter() - t0
            rate = (i + 1) / elapsed
            eta = (len(rows) - i - 1) / rate
            print(f"  [{i+1}/{len(rows)}]  hits={hits}  "
                  f"hit-rate={hits/(i+1):.3f}  "
                  f"({rate:.1f} pos/s, eta {eta:.0f}s)")

    total_s = time.perf_counter() - t0
    n = len(rows)

    proc.stdin.write("quit\n"); proc.stdin.flush()
    try: proc.wait(timeout=2)
    except: proc.terminate()

    print()
    print(f"=== Trust report ({engine_path.name}, spec={args.spec}) ===")
    print(f"positions:    {n}")
    print(f"hits:         {hits}  ({hits/n:.4f})")
    print(f"random ≈      0.033  (1 / typical-legal-moves)")
    print(f"per-position: median={statistics.median(times):.0f}ms  "
          f"mean={statistics.mean(times):.0f}ms  "
          f"max={max(times):.0f}ms")
    print(f"wall time:    {total_s:.1f}s")
    for phase in ("opening", "middlegame", "endgame"):
        h, t = by_phase[phase]
        if t:
            print(f"  {phase:<12} {h}/{t}  ({h/t:.4f})")
    if evals_engine:
        ev = evals_engine
        print(f"engine evals (cp, white-POV): "
              f"median={statistics.median(ev):+.0f}  "
              f"|<0|={sum(1 for x in ev if x < 0)}  "
              f"|>0|={sum(1 for x in ev if x > 0)}  "
              f"|≈0|={sum(1 for x in ev if abs(x) < 30)}")


if __name__ == "__main__":
    sys.exit(main())
