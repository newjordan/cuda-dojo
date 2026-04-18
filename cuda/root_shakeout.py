#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import time
from pathlib import Path


CUDA_DIR = Path(__file__).resolve().parent
MCTS_BIN = CUDA_DIR / "chess_mcts"
REPORT_PATH = CUDA_DIR / "root_shakeout_report.json"

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
MATE_FEN = "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"
QUEEN_FEN = "rnbqkbnr/ppp2ppp/8/4p3/3Q4/8/PPP1PPPP/RNB1KBNR b KQkq - 0 3"
STRONG_OPENINGS = {"e2e4", "d2d4", "g1f3", "c2c4", "b1c3"}

BATCH_CHOICES = [0, 4, 8, 12, 16, 24, 32, 48, 64]
SELECTION_CHOICES = [0.0, 0.00025, 0.0005, 0.00075, 0.001]


def run_mcts(fen: str, simulations: int, args: list[str]) -> dict:
    proc = subprocess.run(
        [str(MCTS_BIN), "--simulations", str(simulations), "--json", *args],
        input=fen + "\n",
        capture_output=True,
        text=True,
        cwd=CUDA_DIR,
        check=True,
    )
    line = proc.stdout.strip().splitlines()[-1]
    return json.loads(line)


def posterior_visits(move: dict) -> int:
    return int(move.get(
        "posterior_visits",
        move.get("effective_visits", move.get("visits", 0) + move.get("prior_visits", 0)),
    ))


def posterior_distribution(result: dict) -> dict[str, float]:
    weights = {}
    total = 0
    for move in result.get("moves", []):
        weight = posterior_visits(move)
        if weight <= 0:
            continue
        uci = move.get("move")
        if not uci:
            continue
        weights[uci] = float(weight)
        total += weight
    if total <= 0:
        bestmove = result.get("bestmove")
        return {bestmove: 1.0} if bestmove else {}
    return {uci: weight / float(total) for uci, weight in weights.items()}


def js_divergence_bits(p: dict[str, float], q: dict[str, float]) -> float:
    keys = set(p) | set(q)
    if not keys:
        return 0.0

    def _kl(a: dict[str, float], b: dict[str, float]) -> float:
        total = 0.0
        for key in keys:
            av = a.get(key, 0.0)
            if av <= 0.0:
                continue
            bv = max(b.get(key, 0.0), 1e-12)
            total += av * math.log2(av / bv)
        return total

    mean = {key: 0.5 * (p.get(key, 0.0) + q.get(key, 0.0)) for key in keys}
    return 0.5 * _kl(p, mean) + 0.5 * _kl(q, mean)


def ordered_distribution_items(result: dict) -> list[tuple[str, float]]:
    probs = posterior_distribution(result)
    return sorted(probs.items(), key=lambda item: item[1], reverse=True)


def top_k_move_set(result: dict, k: int) -> set[str]:
    return {uci for uci, _ in ordered_distribution_items(result)[:k]}


def top_k_jaccard(results: list[dict], k: int) -> float:
    sets = [top_k_move_set(result, k) for result in results]
    if len(sets) < 2:
        return 1.0 if sets else 0.0
    total = 0.0
    pairs = 0
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            union = sets[i] | sets[j]
            total += len(sets[i] & sets[j]) / float(len(union)) if union else 1.0
            pairs += 1
    return total / float(pairs) if pairs else 0.0


def mean_pairwise_js_bits(results: list[dict]) -> float:
    dists = [posterior_distribution(result) for result in results]
    if len(dists) < 2:
        return 0.0
    total = 0.0
    pairs = 0
    for i in range(len(dists)):
        for j in range(i + 1, len(dists)):
            total += js_divergence_bits(dists[i], dists[j])
            pairs += 1
    return total / float(pairs) if pairs else 0.0


def target_rank(result: dict, target_move: str) -> int:
    for rank, (uci, _) in enumerate(ordered_distribution_items(result), start=1):
        if uci == target_move:
            return rank
    return 0


def entropy_metrics(result: dict) -> dict:
    probs = posterior_distribution(result)
    n = len(probs)
    if n == 0:
        return {
            "entropy_bits": 0.0,
            "normalized_entropy": 0.0,
            "effective_branching_factor": 0.0,
            "top1_share": 0.0,
            "top2_share": 0.0,
            "top1_top2_gap": 0.0,
            "mass_outside_top2": 0.0,
            "mass_outside_top4": 0.0,
        }
    ordered = sorted(probs.values(), reverse=True)
    entropy_bits = -sum(p * math.log2(p) for p in ordered if p > 0.0)
    normalized_entropy = entropy_bits / math.log2(n) if n > 1 else 0.0
    top1 = ordered[0]
    top2 = ordered[1] if len(ordered) > 1 else 0.0
    return {
        "entropy_bits": entropy_bits,
        "normalized_entropy": normalized_entropy,
        "effective_branching_factor": 2.0 ** entropy_bits,
        "top1_share": top1,
        "top2_share": top2,
        "top1_top2_gap": top1 - top2,
        "mass_outside_top2": max(0.0, 1.0 - sum(ordered[:2])),
        "mass_outside_top4": max(0.0, 1.0 - sum(ordered[:4])),
    }


def distribution_summary(results: list[dict], target_move: str | None = None) -> dict:
    metrics = [entropy_metrics(result) for result in results]
    moves = [result.get("bestmove", "0000") for result in results]
    modal_move = max(set(moves), key=moves.count) if moves else None
    posterior_leader_hits = 0
    target_ranks = []
    for result in results:
        ordered = ordered_distribution_items(result)
        if ordered and result.get("bestmove") == ordered[0][0]:
            posterior_leader_hits += 1
        if target_move is not None:
            target_ranks.append(target_rank(result, target_move))
    summary = {
        "mean_normalized_entropy": sum(item["normalized_entropy"] for item in metrics) / len(metrics),
        "mean_effective_branching_factor": sum(item["effective_branching_factor"] for item in metrics) / len(metrics),
        "mean_top1_share": sum(item["top1_share"] for item in metrics) / len(metrics),
        "mean_top2_share": sum(item["top2_share"] for item in metrics) / len(metrics),
        "mean_top1_top2_gap": sum(item["top1_top2_gap"] for item in metrics) / len(metrics),
        "mean_mass_outside_top2": sum(item["mass_outside_top2"] for item in metrics) / len(metrics),
        "mean_mass_outside_top4": sum(item["mass_outside_top4"] for item in metrics) / len(metrics),
        "mean_pairwise_js_bits": mean_pairwise_js_bits(results),
        "top2_set_jaccard": top_k_jaccard(results, 2),
        "agreement_rate": moves.count(modal_move) / float(len(moves)) if modal_move else 0.0,
        "modal_move": modal_move,
        "unique_moves": len(set(moves)),
        "bestmove_matches_posterior_leader_rate": posterior_leader_hits / float(len(results)),
    }
    if target_move is not None:
        summary["mean_target_rank"] = (
            sum(target_ranks) / float(len(target_ranks)) if target_ranks else 0.0
        )
    return summary


def sample_config(rng: random.Random, idx: int) -> dict:
    batch = rng.choice(BATCH_CHOICES)
    selection = rng.choice(SELECTION_CHOICES)
    prior = round(rng.uniform(0.0, 1.0), 4)
    return {
        "name": f"shakeout_{idx:03d}",
        "opening_prior_scale": prior,
        "opening_selection_scale": selection,
        "root_batch_size": batch,
        "args": [
            "--seed-mode", "time",
            "--opening-prior-scale", f"{prior:.4f}",
            "--opening-selection-scale", f"{selection:.6f}",
            "--root-batch-size", str(batch),
        ],
    }


def baseline_configs() -> list[dict]:
    return [
        {
            "name": "promoted_default_tuned_legacy",
            "opening_prior_scale": 0.7532,
            "opening_selection_scale": 0.00075,
            "root_batch_size": 0,
            "args": [
                "--seed-mode", "time",
                "--opening-prior-scale", "0.7532",
                "--opening-selection-scale", "0.00075",
                "--root-batch-size", "0",
            ],
        },
        {
            "name": "promoted_default_tuned_batched_8",
            "opening_prior_scale": 0.7532,
            "opening_selection_scale": 0.00075,
            "root_batch_size": 8,
            "args": [
                "--seed-mode", "time",
                "--opening-prior-scale", "0.7532",
                "--opening-selection-scale", "0.00075",
                "--root-batch-size", "8",
            ],
        },
        {
            "name": "promoted_default_tuned_batched_16",
            "opening_prior_scale": 0.7532,
            "opening_selection_scale": 0.00075,
            "root_batch_size": 16,
            "args": [
                "--seed-mode", "time",
                "--opening-prior-scale", "0.7532",
                "--opening-selection-scale", "0.00075",
                "--root-batch-size", "16",
            ],
        },
        {
            "name": "promoted_default_tuned_batched_32",
            "opening_prior_scale": 0.7532,
            "opening_selection_scale": 0.00075,
            "root_batch_size": 32,
            "args": [
                "--seed-mode", "time",
                "--opening-prior-scale", "0.7532",
                "--opening-selection-scale", "0.00075",
                "--root-batch-size", "32",
            ],
        },
        {
            "name": "region_band_batched_8",
            "opening_prior_scale": 0.8032,
            "opening_selection_scale": 0.001,
            "root_batch_size": 8,
            "args": [
                "--seed-mode", "time",
                "--opening-prior-scale", "0.8032",
                "--opening-selection-scale", "0.001",
                "--root-batch-size", "8",
            ],
        },
        {
            "name": "region_band_batched_16",
            "opening_prior_scale": 0.8032,
            "opening_selection_scale": 0.001,
            "root_batch_size": 16,
            "args": [
                "--seed-mode", "time",
                "--opening-prior-scale", "0.8032",
                "--opening-selection-scale", "0.001",
                "--root-batch-size", "16",
            ],
        },
        {
            "name": "region_band_batched_32",
            "opening_prior_scale": 0.8032,
            "opening_selection_scale": 0.001,
            "root_batch_size": 32,
            "args": [
                "--seed-mode", "time",
                "--opening-prior-scale", "0.8032",
                "--opening-selection-scale", "0.001",
                "--root-batch-size", "32",
            ],
        },
        {
            "name": "tuned_legacy_hi_prior",
            "opening_prior_scale": 0.9326,
            "opening_selection_scale": 0.00025,
            "root_batch_size": 0,
            "args": [
                "--seed-mode", "time",
                "--opening-prior-scale", "0.9326",
                "--opening-selection-scale", "0.00025",
                "--root-batch-size", "0",
            ],
        },
        {
            "name": "tuned_legacy_mid_prior",
            "opening_prior_scale": 0.7326,
            "opening_selection_scale": 0.0005,
            "root_batch_size": 0,
            "args": [
                "--seed-mode", "time",
                "--opening-prior-scale", "0.7326",
                "--opening-selection-scale", "0.0005",
                "--root-batch-size", "0",
            ],
        },
        {
            "name": "baseline_legacy_stochastic",
            "opening_prior_scale": 0.5,
            "opening_selection_scale": 0.0005,
            "root_batch_size": 0,
            "args": [
                "--seed-mode", "time",
                "--opening-prior-scale", "0.5",
                "--opening-selection-scale", "0.0005",
                "--root-batch-size", "0",
            ],
        },
        {
            "name": "baseline_batched_16",
            "opening_prior_scale": 0.5,
            "opening_selection_scale": 0.0005,
            "root_batch_size": 16,
            "args": [
                "--seed-mode", "time",
                "--opening-prior-scale", "0.5",
                "--opening-selection-scale", "0.0005",
                "--root-batch-size", "16",
            ],
        },
        {
            "name": "baseline_batched_8",
            "opening_prior_scale": 0.5,
            "opening_selection_scale": 0.0005,
            "root_batch_size": 8,
            "args": [
                "--seed-mode", "time",
                "--opening-prior-scale", "0.5",
                "--opening-selection-scale", "0.0005",
                "--root-batch-size", "8",
            ],
        },
        {
            "name": "baseline_batched_32",
            "opening_prior_scale": 0.5,
            "opening_selection_scale": 0.0005,
            "root_batch_size": 32,
            "args": [
                "--seed-mode", "time",
                "--opening-prior-scale", "0.5",
                "--opening-selection-scale", "0.0005",
                "--root-batch-size", "32",
            ],
        },
        {
            "name": "baseline_batched_64",
            "opening_prior_scale": 0.5,
            "opening_selection_scale": 0.0005,
            "root_batch_size": 64,
            "args": [
                "--seed-mode", "time",
                "--opening-prior-scale", "0.5",
                "--opening-selection-scale", "0.0005",
                "--root-batch-size", "64",
            ],
        },
    ]


def evaluate_config(config: dict, simulations: int, opening_runs: int, tactical_runs: int) -> dict:
    opening_results = [run_mcts(START_FEN, simulations, config["args"]) for _ in range(opening_runs)]
    opening_moves = [result["bestmove"] for result in opening_results]
    strong_opening_fraction = (
        sum(1 for move in opening_moves if move in STRONG_OPENINGS) / float(len(opening_moves))
    )
    opening_distribution = distribution_summary(opening_results)

    mate_results = [run_mcts(MATE_FEN, simulations, config["args"]) for _ in range(tactical_runs)]
    queen_results = [run_mcts(QUEEN_FEN, simulations, config["args"]) for _ in range(tactical_runs)]
    mate_hit_rate = sum(1 for result in mate_results if result["bestmove"] == "h5f7") / float(len(mate_results))
    queen_hit_rate = sum(1 for result in queen_results if result["bestmove"] == "e5d4") / float(len(queen_results))
    mate_distribution = distribution_summary(mate_results, target_move="h5f7")
    queen_distribution = distribution_summary(queen_results, target_move="e5d4")

    # Tactical correctness dominates. Opening sanity is second. Entropy/gap are tie-breakers.
    score = (
        4.0 * mate_hit_rate +
        4.0 * queen_hit_rate +
        2.0 * strong_opening_fraction +
        1.0 * queen_distribution["mean_top1_top2_gap"] +
        0.5 * opening_distribution["mean_top1_top2_gap"] -
        0.5 * queen_distribution["mean_normalized_entropy"]
    )

    return {
        **config,
        "score": score,
        "opening_moves": opening_moves,
        "strong_opening_fraction": strong_opening_fraction,
        "opening_distribution": opening_distribution,
        "mate_results": [result["bestmove"] for result in mate_results],
        "mate_distribution": mate_distribution,
        "queen_results": [result["bestmove"] for result in queen_results],
        "mate_hit_rate": mate_hit_rate,
        "queen_hit_rate": queen_hit_rate,
        "queen_distribution": queen_distribution,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Randomized stochastic root-knob shakeout.")
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--simulations", type=int, default=1000)
    parser.add_argument("--opening-runs", type=int, default=6)
    parser.add_argument("--tactical-runs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260417)
    args = parser.parse_args()

    if not MCTS_BIN.exists():
        raise SystemExit(f"missing binary: {MCTS_BIN}")

    rng = random.Random(args.seed)
    configs = baseline_configs()
    for idx in range(args.samples):
        configs.append(sample_config(rng, idx))

    started = time.time()
    results = [
        evaluate_config(config, args.simulations, args.opening_runs, args.tactical_runs)
        for config in configs
    ]
    results.sort(key=lambda item: item["score"], reverse=True)

    payload = {
        "kind": "root_shakeout",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_sec": round(time.time() - started, 2),
        "simulations": args.simulations,
        "opening_runs": args.opening_runs,
        "tactical_runs": args.tactical_runs,
        "results": results,
        "winner": results[0]["name"] if results else None,
    }
    REPORT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {REPORT_PATH}")
    if results:
        top = results[0]
        print(
            f"winner={top['name']} score={top['score']:.3f} "
            f"strong_opening={top['strong_opening_fraction']:.2f} "
            f"mate={top['mate_hit_rate']:.2f} queen={top['queen_hit_rate']:.2f} "
            f"batch={top['root_batch_size']} prior={top['opening_prior_scale']:.4f} "
            f"sel={top['opening_selection_scale']:.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
