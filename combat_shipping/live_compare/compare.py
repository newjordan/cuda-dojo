#!/usr/bin/env python3
"""compare.py — 1:1 head-to-head harness for AgentChess (CPU referee)
vs cuda combat (GPU referee, dojo_ref).

For each matchup (a pair of fighter .js files), run the SAME pair
through:
  1) the production arbiter (Node, calls match-processor's playGame)
  2) the cuda arbiter (Python, calls our live_match.play_game)

Capture per-game: result, termination reason, ply count, move list,
wall-clock seconds. Compare game-level outcomes between the two
arbiters.

Output:
  - results/run_<timestamp>/cuda/*.json
  - results/run_<timestamp>/arbiter/*.json
  - results/run_<timestamp>/summary.json
  - stdout report
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LIVE_ARBITER = REPO_ROOT / "combat_shipping" / "live_arbiter"

DEFAULT_FIGHTERS_DIR = Path(
    "/home/frosty40/AgentChess_Server/match-processor/data"
)


def list_fighters(fighters_dir: Path) -> list[Path]:
    return sorted(p for p in fighters_dir.glob("match-*.js"))


def make_pairings(fighters: list[Path], n: int, seed: int) -> list[tuple[Path, Path]]:
    """Sample N matchups with fixed seed. White != Black."""
    rng = random.Random(seed)
    pairings = []
    for _ in range(n):
        a, b = rng.sample(fighters, 2)
        pairings.append((a, b))
    return pairings


def run_arbiter(white: Path, black: Path, match_id: str,
                max_plies: int, move_timeout_ms: int,
                out_dir: Path) -> dict:
    out_json = out_dir / f"{match_id}.json"
    cmd = [
        "node",
        str(LIVE_ARBITER / "run_arbiter_native.mjs"),
        "--white", str(white),
        "--black", str(black),
        "--match-id", match_id,
        "--max-plies", str(max_plies),
        "--move-timeout-ms", str(move_timeout_ms),
        "--out-json", str(out_json),
    ]
    t0 = time.monotonic()
    p = subprocess.run(cmd, capture_output=True,
                       timeout=(max_plies * move_timeout_ms) / 1000.0 + 60)
    dt = time.monotonic() - t0
    if p.returncode != 0 or not out_json.exists():
        return {
            "match_id": match_id, "error": "arbiter run failed",
            "stderr": p.stderr.decode("utf-8", "replace")[:500],
            "wall_seconds": round(dt, 3),
            "referee": "arbiter",
        }
    return json.loads(out_json.read_text())


def run_cuda(white: Path, black: Path, match_id: str,
             max_plies: int, move_timeout_ms: int,
             out_dir: Path) -> dict:
    out_json = out_dir / f"{match_id}.json"
    cmd = [
        sys.executable,
        str(LIVE_ARBITER / "live_match.py"),
        "--white", str(white),
        "--black", str(black),
        "--match-id", match_id,
        "--max-plies", str(max_plies),
        "--move-timeout-ms", str(move_timeout_ms),
        "--out-json", str(out_json),
    ]
    t0 = time.monotonic()
    p = subprocess.run(cmd, capture_output=True,
                       timeout=(max_plies * move_timeout_ms) / 1000.0 + 60,
                       cwd=str(LIVE_ARBITER))
    dt = time.monotonic() - t0
    if p.returncode != 0 or not out_json.exists():
        return {
            "match_id": match_id, "error": "cuda run failed",
            "stderr": p.stderr.decode("utf-8", "replace")[:500],
            "wall_seconds": round(dt, 3),
            "referee": "cuda",
        }
    return json.loads(out_json.read_text())


def compare_one(arb: dict, cud: dict) -> dict:
    """Compare two single-game outcomes."""
    if "error" in arb or "error" in cud:
        return {
            "result_match": False, "reason_match": False,
            "plies_arbiter": arb.get("plies"), "plies_cuda": cud.get("plies"),
            "error_arbiter": arb.get("error"), "error_cuda": cud.get("error"),
            "moves_match_prefix": 0,
        }
    result_match = arb.get("result") == cud.get("result")
    reason_match = arb.get("reason") == cud.get("reason")
    moves_a = arb.get("moves", [])
    moves_c = cud.get("moves", [])
    prefix = 0
    for x, y in zip(moves_a, moves_c):
        if x == y:
            prefix += 1
        else:
            break
    return {
        "result_match": result_match,
        "reason_match": reason_match,
        "plies_arbiter": arb.get("plies"),
        "plies_cuda": cud.get("plies"),
        "result_arbiter": arb.get("result"),
        "result_cuda": cud.get("result"),
        "reason_arbiter": arb.get("reason"),
        "reason_cuda": cud.get("reason"),
        "moves_match_prefix": prefix,
        "moves_arbiter_len": len(moves_a),
        "moves_cuda_len": len(moves_c),
        "wall_arbiter": arb.get("wall_seconds"),
        "wall_cuda": cud.get("wall_seconds"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10,
                    help="number of matchups to run")
    ap.add_argument("--seed", type=int, default=20260429)
    ap.add_argument("--max-plies", type=int, default=120)
    ap.add_argument("--move-timeout-ms", type=int, default=5500)
    ap.add_argument("--fighters-dir", default=str(DEFAULT_FIGHTERS_DIR))
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--concurrency", type=int, default=1,
                    help="number of matchups to run in parallel "
                         "(both sides for the same matchup are sequential)")
    args = ap.parse_args()

    fighters = list_fighters(Path(args.fighters_dir))
    if len(fighters) < 2:
        sys.exit(f"need >=2 fighters in {args.fighters_dir}")
    pairings = make_pairings(fighters, args.n, args.seed)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    if args.out_root:
        out_root = Path(args.out_root)
    else:
        out_root = REPO_ROOT / "combat_shipping" / "results" / f"compare_{stamp}"
    cuda_dir = out_root / "cuda"
    arb_dir = out_root / "arbiter"
    cuda_dir.mkdir(parents=True, exist_ok=True)
    arb_dir.mkdir(parents=True, exist_ok=True)

    print(f"[compare] N={args.n} seed={args.seed} "
          f"max_plies={args.max_plies} timeout={args.move_timeout_ms}ms")
    print(f"[compare] out: {out_root}")

    games = []

    def run_one(idx: int, white: Path, black: Path):
        # Use distinct match IDs for the two arbiters to avoid Docker
        # name collisions; both run the SAME fighter pair from a
        # fresh state.
        match_id_a = f"a{idx:04d}"
        match_id_c = f"c{idx:04d}"
        # Run the arbiter first, then cuda — sequentially within one
        # matchup. (Could run in parallel but keeps GPU load tame.)
        arb = run_arbiter(white, black, match_id_a,
                          args.max_plies, args.move_timeout_ms, arb_dir)
        cud = run_cuda(white, black, match_id_c,
                       args.max_plies, args.move_timeout_ms, cuda_dir)
        cmp_ = compare_one(arb, cud)
        cmp_.update({
            "idx": idx,
            "white": white.name, "black": black.name,
        })
        return cmp_

    if args.concurrency > 1:
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futs = {
                ex.submit(run_one, i, w, b): i
                for i, (w, b) in enumerate(pairings)
            }
            for f in as_completed(futs):
                g = f.result()
                games.append(g)
                _print_progress(g)
    else:
        for i, (w, b) in enumerate(pairings):
            g = run_one(i, w, b)
            games.append(g)
            _print_progress(g)

    games.sort(key=lambda g: g["idx"])

    summary = _summarize(games, args)
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_root / "games.json").write_text(json.dumps(games, indent=2))
    _print_summary(summary, out_root)


def _print_progress(g):
    if "error_arbiter" in g and g.get("error_arbiter"):
        print(f"  [{g['idx']:03d}] ARBITER ERROR: {g['error_arbiter']}")
        return
    if "error_cuda" in g and g.get("error_cuda"):
        print(f"  [{g['idx']:03d}] CUDA ERROR: {g['error_cuda']}")
        return
    rok = "OK" if g["result_match"] else "DIFF"
    tok = "OK" if g["reason_match"] else "DIFF"
    print(
        f"  [{g['idx']:03d}] result={rok} reason={tok} "
        f"plies={g['plies_arbiter']}/{g['plies_cuda']} "
        f"wall_arb={g['wall_arbiter']}s wall_cud={g['wall_cuda']}s "
        f"moves_match_prefix={g['moves_match_prefix']}"
    )


def _summarize(games, args):
    n = len(games)
    n_arb_err = sum(1 for g in games if g.get("error_arbiter"))
    n_cud_err = sum(1 for g in games if g.get("error_cuda"))
    valid = [g for g in games
             if not g.get("error_arbiter") and not g.get("error_cuda")]
    n_result_match = sum(1 for g in valid if g["result_match"])
    n_reason_match = sum(1 for g in valid if g["reason_match"])
    n_full_match = sum(
        1 for g in valid
        if g["result_match"] and g["reason_match"]
        and g["moves_match_prefix"] == g["moves_arbiter_len"] == g["moves_cuda_len"]
    )

    def _mean(xs):
        return round(sum(xs) / len(xs), 3) if xs else None

    wall_arb = [g["wall_arbiter"] for g in valid if g.get("wall_arbiter") is not None]
    wall_cud = [g["wall_cuda"] for g in valid if g.get("wall_cuda") is not None]
    plies_arb = [g["plies_arbiter"] for g in valid if g.get("plies_arbiter") is not None]
    plies_cud = [g["plies_cuda"] for g in valid if g.get("plies_cuda") is not None]

    speed_ratio = (round(_mean(wall_arb) / _mean(wall_cud), 3)
                   if wall_arb and wall_cud and _mean(wall_cud) else None)

    divergences = [
        {k: g[k] for k in (
            "idx", "white", "black",
            "result_arbiter", "result_cuda",
            "reason_arbiter", "reason_cuda",
            "plies_arbiter", "plies_cuda",
            "moves_match_prefix",
        ) if k in g}
        for g in valid if not (g["result_match"] and g["reason_match"])
    ]
    return {
        "n_total": n,
        "n_arbiter_errors": n_arb_err,
        "n_cuda_errors": n_cud_err,
        "n_valid_pairs": len(valid),
        "n_result_match": n_result_match,
        "n_reason_match": n_reason_match,
        "n_full_match_including_moves": n_full_match,
        "result_match_rate": (round(n_result_match / len(valid), 4)
                              if valid else None),
        "reason_match_rate": (round(n_reason_match / len(valid), 4)
                              if valid else None),
        "wall_arbiter_mean": _mean(wall_arb),
        "wall_cuda_mean": _mean(wall_cud),
        "speed_ratio_arb_over_cuda": speed_ratio,
        "plies_arbiter_mean": _mean(plies_arb),
        "plies_cuda_mean": _mean(plies_cud),
        "divergences": divergences,
        "args": {
            "n": args.n, "seed": args.seed,
            "max_plies": args.max_plies,
            "move_timeout_ms": args.move_timeout_ms,
            "fighters_dir": args.fighters_dir,
        },
    }


def _print_summary(s, out_root):
    print()
    print("=" * 60)
    print(f"SUMMARY  ({out_root})")
    print("=" * 60)
    print(f"N total                    : {s['n_total']}")
    print(f"N valid pairs              : {s['n_valid_pairs']}")
    print(f"  arbiter errors           : {s['n_arbiter_errors']}")
    print(f"  cuda errors              : {s['n_cuda_errors']}")
    print(f"result match (winner)      : {s['n_result_match']}/{s['n_valid_pairs']}"
          f"  ({s['result_match_rate']})")
    print(f"reason match (termination) : {s['n_reason_match']}/{s['n_valid_pairs']}"
          f"  ({s['reason_match_rate']})")
    print(f"full-match incl moves      : {s['n_full_match_including_moves']}/{s['n_valid_pairs']}")
    print(f"wall mean arbiter          : {s['wall_arbiter_mean']}s")
    print(f"wall mean cuda             : {s['wall_cuda_mean']}s")
    print(f"speed ratio (arb/cuda)     : {s['speed_ratio_arb_over_cuda']}")
    print(f"plies mean arb / cuda      : {s['plies_arbiter_mean']} / {s['plies_cuda_mean']}")
    if s["divergences"]:
        print()
        print(f"-- {len(s['divergences'])} divergence(s) --")
        for d in s["divergences"][:20]:
            print(f"  [{d['idx']}] {d.get('white')} vs {d.get('black')}: "
                  f"arb={d.get('result_arbiter')}/{d.get('reason_arbiter')}/{d.get('plies_arbiter')}p  "
                  f"cud={d.get('result_cuda')}/{d.get('reason_cuda')}/{d.get('plies_cuda')}p  "
                  f"prefix={d.get('moves_match_prefix')}")


if __name__ == "__main__":
    main()
