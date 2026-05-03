#!/usr/bin/env python3
"""run_validation.py — single-command harness over targets.json.

Tiers:
  --tier 1   rule correctness (G1 replay parity + G3 bench coalescer).
             ~30 seconds at smoke profile, ~5 minutes at full profile.
  --tier 2   tier 1 + daemon protocol (G4 smoke + G5 concurrent).
             ~5 minutes at smoke, ~10 minutes at full.
  --tier 3   tier 2 + live wall-clock parity (G6).
             ~20 minutes at smoke, ~60 minutes at full.

Profiles:
  --profile smoke   small N, fast iteration (default).
  --profile full    full sample sizes from targets.json.

Exits 0 if all selected gates pass, 1 otherwise. Writes a JSON receipt
to results/validation_<ts>.json next to the existing comparison runs.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
sys.path.insert(0, str(HERE))

from gates import (  # noqa: E402
    GateResult,
    gate_bench_coalescer,
    gate_daemon_concurrent,
    gate_daemon_smoke,
    gate_live_wall_ratio,
    gate_replay_parity,
    result_to_dict,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", type=int, choices=[1, 2, 3], default=2)
    ap.add_argument("--profile", choices=["smoke", "full"], default="smoke")
    ap.add_argument("--targets", default=str(HERE / "targets.json"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    targets = json.load(open(args.targets))

    print(f"=== validation harness  tier={args.tier}  profile={args.profile} ===\n")
    gates: list[GateResult] = []

    # G1
    if args.tier >= 1:
        cfg = targets["G1_replay_parity"]
        n = cfg[f"{args.profile}_n"]
        gates.append(gate_replay_parity(
            n=n, pgn=cfg["pgn_corpus"],
            min_parity=cfg["min_referee_parity_rate"],
            min_subset=cfg["min_replayable_subset_rate"],
        ))
        _print_one(gates[-1])

    # G3
    if args.tier >= 1:
        cfg = targets["G3_bench_coalescer"]
        gates.append(gate_bench_coalescer(min_speedup=cfg["min_speedup"]))
        _print_one(gates[-1])

    # G4
    if args.tier >= 2:
        cfg = targets["G4_daemon_smoke"]
        gates.append(gate_daemon_smoke(max_plies=cfg["max_plies"]))
        _print_one(gates[-1])

    # G5
    if args.tier >= 2:
        cfg = targets["G5_daemon_concurrent"]
        n = cfg[f"{args.profile}_n"]
        gates.append(gate_daemon_concurrent(
            n=n, max_plies=cfg["max_plies"]))
        _print_one(gates[-1])

    # G6
    if args.tier >= 3:
        cfg = targets["G6_live_wall_ratio"]
        n = cfg[f"{args.profile}_n"]
        gates.append(gate_live_wall_ratio(
            n=n, agent_cpus=cfg["agent_cpus"],
            agent_memory=cfg["agent_memory"],
            max_plies=cfg["max_plies"],
            min_ratio=cfg["min_ratio"], max_ratio=cfg["max_ratio"],
        ))
        _print_one(gates[-1])

    # ------- Summary
    n_pass = sum(1 for g in gates if g.passed)
    print(f"\n=== {n_pass}/{len(gates)} GATES PASS ===")

    if args.out:
        out = Path(args.out)
    else:
        out = (PKG / "results" /
               f"validation_{time.strftime('%Y%m%d-%H%M%S')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "timestamp": time.time(),
        "tier": args.tier,
        "profile": args.profile,
        "n_pass": n_pass,
        "n_total": len(gates),
        "gates": [result_to_dict(g) for g in gates],
    }, indent=2))
    print(f"receipt: {out}")
    return 0 if n_pass == len(gates) else 1


def _print_one(g: GateResult) -> None:
    flag = "PASS" if g.passed else "FAIL"
    print(f"  [{flag}]  {g.name}  ({g.duration_s:.1f}s)")
    print(f"          target:   {g.target}")
    print(f"          observed: {g.observed}")


if __name__ == "__main__":
    sys.exit(main())
