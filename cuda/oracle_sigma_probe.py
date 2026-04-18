#!/usr/bin/env python3
"""
oracle_sigma_probe.py

Take oracle-enriched rows, pick the worst disagreement positions, run chess_mcts
on them, and report whether the live GPU search is moving toward or away from
the Stockfish oracle.
"""

import argparse
import json
import os
import subprocess


CUDA_DIR = os.path.dirname(os.path.abspath(__file__))
MCTS_BIN = os.path.join(CUDA_DIR, "chess_mcts")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--top-k", type=int, default=16)
    ap.add_argument("--simulations", type=int, default=2000)
    return ap.parse_args()


def load_rows(path):
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def run_mcts(fens, simulations):
    proc = subprocess.run(
        [MCTS_BIN, "--simulations", str(simulations), "--json"],
        input="\n".join(fens) + "\n",
        text=True,
        capture_output=True,
        cwd=CUDA_DIR,
        check=True,
    )
    results = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line:
            results.append(json.loads(line))
    return results


def result_bestmove(result):
    bestmove = result.get("bestmove")
    if bestmove:
        return bestmove
    moves = result.get("moves") or []
    if moves:
        return moves[0].get("move", "0000")
    return "0000"


def rate(n, d):
    return n / d if d else None


def main():
    args = parse_args()
    rows = load_rows(args.input)
    ranked = sorted(
        rows,
        key=lambda row: (
            float(row.get("sigma_score", 0.0)),
            float(row.get("sigma_ref_vs_sf_abs_score_delta_cp", 0.0)),
        ),
        reverse=True,
    )[: args.top_k]
    fens = [row["fen"] for row in ranked]
    results = run_mcts(fens, args.simulations)

    paired = []
    mcts_vs_sf = 0
    mcts_vs_ref = 0
    mcts_vs_lozza = 0
    for row, result in zip(ranked, results):
        mcts_move = result_bestmove(result)
        sf_move = row.get("oracle_sf_bestmove")
        ref_move = row.get("move") or row.get("sf_bestmove") or row.get("bestmove")
        lozza_move = row.get("oracle_lozza_bestmove")
        paired_row = {
            "fen": row["fen"],
            "sigma_score": row.get("sigma_score"),
            "sigma_bucket": row.get("sigma_bucket"),
            "reference_move": ref_move,
            "oracle_sf_bestmove": sf_move,
            "oracle_lozza_bestmove": lozza_move,
            "mcts_bestmove": mcts_move,
            "sigma_ref_vs_sf_abs_score_delta_cp": row.get("sigma_ref_vs_sf_abs_score_delta_cp"),
            "mcts_vs_sf_move_match": mcts_move == sf_move if mcts_move and sf_move else None,
            "mcts_vs_reference_move_match": mcts_move == ref_move if mcts_move and ref_move else None,
            "mcts_vs_lozza_move_match": mcts_move == lozza_move if mcts_move and lozza_move else None,
        }
        if paired_row["mcts_vs_sf_move_match"]:
            mcts_vs_sf += 1
        if paired_row["mcts_vs_reference_move_match"]:
            mcts_vs_ref += 1
        if paired_row["mcts_vs_lozza_move_match"]:
            mcts_vs_lozza += 1
        paired.append(paired_row)

    report = {
        "input": args.input,
        "rows_considered": len(rows),
        "top_k": len(paired),
        "simulations": args.simulations,
        "mcts_vs_sf_move_match_rate": rate(mcts_vs_sf, len(paired)),
        "mcts_vs_reference_move_match_rate": rate(mcts_vs_ref, len(paired)),
        "mcts_vs_lozza_move_match_rate": rate(mcts_vs_lozza, len(paired)),
        "positions": paired,
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
