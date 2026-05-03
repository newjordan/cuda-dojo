#!/usr/bin/env python3
"""Run N GPU-vs-GPU games via gpu_arena_min, tally raw verdict counts.

No CPU chess. Each game is two ./engine processes; the host only
shuffles UCI tokens. Verdicts:
  - terminal_no_legal_move : engine returned bestmove "0000"
                             (natural mate or stalemate per UCI engine logic)
  - max_plies              : safety cap reached
  - engine_crash           : either side raised RuntimeError mid-game

We DO NOT parse the position to distinguish mate vs stalemate vs
threefold/fifty/insufficient — the engine binary is the source of truth
for "no legal move." If you want richer terminal taxonomy, run dojo_ref
side-by-side and compare.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run_one_game(engine: str, white_spec: str, black_spec: str,
                 max_plies: int, timeout_s: float) -> dict:
    cmd = [
        sys.executable, str(HERE / "gpu_arena_min.py"),
        "--engine", engine,
        "--max-plies", str(max_plies),
        "--spec", white_spec,
    ]
    # gpu_arena_min uses one --spec for both sides; we want different specs
    # so we pass --white/--black with the same engine path but emulate split
    # specs by running TWO instances of arena_min isn't straightforward.
    # Instead, use --spec with both sides equal and vary across games. To
    # vary depths between sides we shell out a small inline driver below.
    raise NotImplementedError("use _drive_split below")


def _drive_split(engine: str, white_spec: str, black_spec: str,
                 max_plies: int, timeout_s: float) -> dict:
    """Direct driver that supports per-side spec — copy of gpu_arena_min
    main loop, simplified to capture verdict + counters."""
    import gpu_arena_min as gam
    t0 = time.perf_counter()
    try:
        white = gam.open_engine(engine)
        black = gam.open_engine(engine)
    except Exception as e:
        return {"verdict": "engine_crash", "stage": "open",
                "error": str(e), "plies": 0, "wall_s": 0.0,
                "white_spec": white_spec, "black_spec": black_spec}

    moves: list[str] = []
    side = 0
    verdict = "max_plies"
    err = None
    deadline = t0 + timeout_s
    try:
        for _ in range(max_plies):
            if time.perf_counter() > deadline:
                verdict = "host_timeout"
                break
            spec = white_spec if side == 0 else black_spec
            try:
                bestmove, _ = gam.go_move(
                    white if side == 0 else black, moves, spec
                )
            except Exception as e:
                verdict = "engine_crash"
                err = f"side={'W' if side==0 else 'B'} {e}"
                break
            if not bestmove or bestmove == "0000":
                verdict = "terminal_no_legal_move"
                break
            moves.append(bestmove)
            side ^= 1
    finally:
        for proc in (white, black):
            try:
                proc.stdin.write("quit\n"); proc.stdin.flush()
            except Exception:
                pass
            try:
                proc.wait(timeout=2)
            except Exception:
                proc.kill()
    return {
        "verdict": verdict,
        "error": err,
        "plies": len(moves),
        "wall_s": round(time.perf_counter() - t0, 3),
        "white_spec": white_spec,
        "black_spec": black_spec,
        "moves": moves,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default=str(HERE / "engine"))
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--max-plies", type=int, default=300)
    ap.add_argument("--timeout-s", type=float, default=300.0,
                    help="per-game wall timeout (host-side safety)")
    ap.add_argument("--seed", type=int, default=0,
                    help="shuffle/select seed for the depth-pair sweep")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    sys.path.insert(0, str(HERE))  # for gpu_arena_min import in _drive_split

    # Pool of distinct depth/movetime pairs the engine has handled.
    # Shuffled deterministically by --seed and then trimmed to N.
    pool = [
        ("depth 2", "depth 2"),
        ("depth 2", "depth 3"),
        ("depth 3", "depth 2"),
        ("depth 3", "depth 3"),
        ("depth 3", "depth 4"),
        ("depth 4", "depth 3"),
        ("depth 4", "depth 4"),
        ("depth 4", "depth 5"),
        ("depth 5", "depth 4"),
        ("depth 5", "depth 3"),
        ("depth 3", "depth 5"),
        ("depth 5", "depth 5"),
        ("depth 2", "depth 4"),
        ("depth 4", "depth 2"),
        ("movetime 50",  "movetime 50"),
        ("movetime 100", "movetime 100"),
        ("movetime 200", "movetime 200"),
        ("movetime 300", "movetime 300"),
        ("movetime 400", "movetime 400"),
        ("movetime 100", "movetime 200"),
        ("movetime 200", "movetime 100"),
        ("movetime 100", "movetime 300"),
        ("movetime 300", "movetime 100"),
        ("movetime 50",  "movetime 200"),
        ("movetime 200", "movetime 50"),
        ("depth 2", "movetime 200"),
        ("movetime 200", "depth 2"),
        ("depth 3", "movetime 100"),
        ("movetime 100", "depth 3"),
        ("depth 4", "movetime 100"),
        ("movetime 100", "depth 4"),
        ("depth 3", "movetime 50"),
        ("movetime 50", "depth 3"),
        ("depth 5", "movetime 100"),
        ("movetime 100", "depth 5"),
        ("depth 4", "movetime 200"),
        ("movetime 200", "depth 4"),
        ("depth 2", "movetime 50"),
        ("movetime 50", "depth 2"),
        ("depth 5", "movetime 50"),
    ]
    import random
    rng = random.Random(args.seed)
    if args.seed == 0:
        # seed=0 → deterministic original order (iter-0 baseline reproducible).
        pairs = (pool * ((args.n // len(pool)) + 1))[:args.n]
    else:
        shuffled = list(pool)
        rng.shuffle(shuffled)
        pairs = (shuffled * ((args.n // len(shuffled)) + 1))[:args.n]

    games: list[dict] = []
    print(f"=== gpu_arena_loop: {args.n} games, max_plies={args.max_plies} ===")
    for i, (ws, bs) in enumerate(pairs):
        t0 = time.perf_counter()
        res = _drive_split(
            args.engine, ws, bs, args.max_plies, args.timeout_s,
        )
        dt = time.perf_counter() - t0
        # Drop heavy moves field from stdout summary line.
        flag = res["verdict"]
        print(
            f"  [{i:02d}]  W={ws:<14} B={bs:<14}  plies={res['plies']:3d}  "
            f"verdict={flag:<25}  wall={dt:.1f}s"
            + (f"  err={res.get('error')}" if res.get("error") else "")
        )
        games.append(res)

    counts: dict[str, int] = {}
    for g in games:
        counts[g["verdict"]] = counts.get(g["verdict"], 0) + 1

    print("\n=== TALLY ===")
    for v, c in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {v:<28} {c}/{args.n}")

    out_path = (Path(args.out) if args.out else
                HERE / f"arena_loop_{time.strftime('%Y%m%d-%H%M%S')}.json")
    out_path.write_text(json.dumps({
        "timestamp": time.time(),
        "engine": args.engine,
        "n": args.n,
        "max_plies": args.max_plies,
        "tally": counts,
        "games": games,
    }, indent=2))
    print(f"\nreceipt: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
