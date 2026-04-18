#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import subprocess
import time
from collections import Counter
from pathlib import Path


CUDA_DIR = Path(__file__).resolve().parent
MCTS_BIN = CUDA_DIR / "chess_mcts"
REPORT_PATH = CUDA_DIR / "mcts_ablation_report.json"

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
STRONG_OPENINGS = {"e2e4", "d2d4", "g1f3", "c2c4", "b1c3"}
MATE_FEN = "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"
QUEEN_FEN = "rnbqkbnr/ppp2ppp/8/4p3/3Q4/8/PPP1PPPP/RNB1KBNR b KQkq - 0 3"
QUEEN_WIN_MOVES = {"e5d4"}


CONFIGS = [
    {
        "name": "deterministic_posterior",
        "args": ["--seed-mode", "fen", "--opening-selection-scale", "0.0", "--opening-prior-scale", "0.0"],
    },
    {
        "name": "deterministic_opening_light",
        "args": ["--seed-mode", "fen", "--opening-selection-scale", "0.0005", "--opening-prior-scale", "0.5"],
    },
    {
        "name": "deterministic_opening_default",
        "args": ["--seed-mode", "fen", "--opening-selection-scale", "0.001", "--opening-prior-scale", "1.0"],
    },
]


def run_mcts(fen: str, simulations: int, extra_args: list[str]) -> dict:
    proc = subprocess.run(
        [str(MCTS_BIN), "--simulations", str(simulations), "--json", *extra_args],
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
    moves = result.get("moves", [])
    weights = {}
    total = 0
    for move in moves:
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
        }
    ordered = sorted(probs.values(), reverse=True)
    entropy_bits = -sum(p * math.log2(p) for p in ordered if p > 0.0)
    normalized_entropy = entropy_bits / math.log2(n) if n > 1 else 0.0
    top1_share = ordered[0]
    top2_share = ordered[1] if len(ordered) > 1 else 0.0
    return {
        "entropy_bits": entropy_bits,
        "normalized_entropy": normalized_entropy,
        "effective_branching_factor": 2.0 ** entropy_bits,
        "top1_share": top1_share,
        "top2_share": top2_share,
        "top1_top2_gap": top1_share - top2_share,
    }


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
            bv = b.get(key, 0.0)
            total += av * math.log2(av / bv)
        return total

    m = {key: 0.5 * (p.get(key, 0.0) + q.get(key, 0.0)) for key in keys}
    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def mean_pairwise_js_divergence(results: list[dict]) -> float:
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


def repeatability(config: dict) -> dict:
    results = []
    for _ in range(2):
        results.append(run_mcts(START_FEN, 1000, config["args"]))
    moves = [result["bestmove"] for result in results]
    ctr = Counter(moves)
    modal_move, modal_count = ctr.most_common(1)[0]
    entropy_runs = [entropy_metrics(result) for result in results]
    return {
        "runs": moves,
        "agreement_rate": modal_count / len(moves),
        "unique_moves": len(ctr),
        "modal_move": modal_move,
        "average_entropy_bits": sum(run["entropy_bits"] for run in entropy_runs) / len(entropy_runs),
        "average_normalized_entropy": (
            sum(run["normalized_entropy"] for run in entropy_runs) / len(entropy_runs)
        ),
        "average_effective_branching_factor": (
            sum(run["effective_branching_factor"] for run in entropy_runs) / len(entropy_runs)
        ),
        "average_top1_top2_gap": (
            sum(run["top1_top2_gap"] for run in entropy_runs) / len(entropy_runs)
        ),
        "mean_pairwise_js_divergence_bits": mean_pairwise_js_divergence(results),
    }


def opening_concentration(config: dict) -> dict:
    results = []
    for _ in range(4):
        results.append(run_mcts(START_FEN, 1000, config["args"]))
    moves = [result["bestmove"] for result in results]
    ctr = Counter(moves)
    strong = sum(ctr[m] for m in STRONG_OPENINGS)
    modal_move, modal_count = ctr.most_common(1)[0]
    entropy_runs = [entropy_metrics(result) for result in results]
    return {
        "runs": moves,
        "unique_moves": len(ctr),
        "modal_move": modal_move,
        "modal_fraction": modal_count / len(moves),
        "strong_opening_fraction": strong / len(moves),
        "average_entropy_bits": sum(run["entropy_bits"] for run in entropy_runs) / len(entropy_runs),
        "average_normalized_entropy": (
            sum(run["normalized_entropy"] for run in entropy_runs) / len(entropy_runs)
        ),
        "average_effective_branching_factor": (
            sum(run["effective_branching_factor"] for run in entropy_runs) / len(entropy_runs)
        ),
        "average_top1_top2_gap": (
            sum(run["top1_top2_gap"] for run in entropy_runs) / len(entropy_runs)
        ),
        "mean_pairwise_js_divergence_bits": mean_pairwise_js_divergence(results),
    }


def tactical_probe(config: dict, fen: str, expected: set[str]) -> dict:
    by_sims = {}
    matches = 0
    for sims in (1000,):
        result = run_mcts(fen, sims, config["args"])
        bestmove = result["bestmove"]
        ok = bestmove in expected
        if ok:
            matches += 1
        by_sims[str(sims)] = {
            "bestmove": bestmove,
            "selection_metric": result.get("selection_metric"),
            "top3": [move["move"] for move in result.get("moves", [])[:3]],
            "matches_expected": ok,
            "distribution": entropy_metrics(result),
        }
    return {
        "matches": matches,
        "by_sim_count": by_sims,
    }


def score_config(result: dict) -> float:
    return (
        2.0 * result["repeatability"]["agreement_rate"] +
        1.5 * result["opening"]["strong_opening_fraction"] +
        2.0 * float(result["mate"]["matches"]) +
        2.5 * float(result["queen"]["matches"])
    )


def main() -> int:
    if not MCTS_BIN.exists():
        raise SystemExit(f"missing binary: {MCTS_BIN}")

    started = time.time()
    results = []
    for config in CONFIGS:
        entry = {
            "name": config["name"],
            "args": config["args"],
        }
        entry["repeatability"] = repeatability(config)
        entry["opening"] = opening_concentration(config)
        entry["mate"] = tactical_probe(config, MATE_FEN, {"h5f7"})
        entry["queen"] = tactical_probe(config, QUEEN_FEN, QUEEN_WIN_MOVES)
        entry["score"] = score_config(entry)
        results.append(entry)

    results.sort(key=lambda item: item["score"], reverse=True)
    payload = {
        "kind": "mcts_ablation",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_sec": round(time.time() - started, 2),
        "configs": results,
        "winner": results[0]["name"] if results else None,
    }
    REPORT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {REPORT_PATH}")
    if results:
        print(f"winner={results[0]['name']} score={results[0]['score']:.3f}")
        for item in results:
            mate_total = len(item["mate"]["by_sim_count"])
            queen_total = len(item["queen"]["by_sim_count"])
            print(
                f"{item['name']}: score={item['score']:.3f} "
                f"repeat={item['repeatability']['agreement_rate']:.2f} "
                f"opening={item['opening']['strong_opening_fraction']:.2f} "
                f"mate={item['mate']['matches']}/{mate_total} "
                f"queen={item['queen']['matches']}/{queen_total}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
