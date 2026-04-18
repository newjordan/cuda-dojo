#!/usr/bin/env python3
"""
Stochastic correctness audit for cuda/chess_mcts.

Six checks (each writes a section to mcts_audit_report.json):
  1. Legal-move sanity (vs python-chess) on 50 random book positions
  2. Repeatability with same seed (chess_mcts is time-seeded -> not reproducible)
  3. Sim-count scaling on tactical puzzles (1k -> 100k)
  4. Move-distribution diversity on starting position (50 runs)
  5. Root-UCT regression probe: coverage, tactical visit concentration, anti-flatness
  6. Comparison vs a tiny CPU Python MCTS (random rollouts)

This is the allowed CPU exception: CPU is the reference, GPU is under audit.

Usage:
    cd cuda
    python3 mcts_audit.py

Outputs:
    cuda/mcts_audit_report.json
"""

import json
import math
import os
import random
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import chess

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CUDA_DIR = PROJECT_ROOT / "cuda"
BOOK_PATH = PROJECT_ROOT / "gpu_spine" / "book.jsonl"
MCTS_BIN = CUDA_DIR / "chess_mcts"
REPORT_PATH = CUDA_DIR / "mcts_audit_report.json"

# Tactical puzzles
PUZZLES = {
    "mate_in_1_qxf7": {
        # Scholar's mate after 1.e4 e5 2.Bc4 Nc6 3.Qh5 Nf6.
        # The actual mate is Qxf7# from h5, not from f3.
        "fen": "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
        "expected": ["h5f7"],
        "note": "Qxf7# (Scholar's mate)",
    },
    "mate_in_2_bodens_setup": {
        "fen": "2kr1b1r/pp1n1ppp/2p1b3/4P3/2B5/2N2N2/PPP2PPP/R1B1K2R w KQ - 0 9",
        "expected": [],  # no single forced answer asserted; we observe convergence
        "note": "Boden's mate setup; check convergence behavior",
    },
    "free_queen_capture": {
        # White just played Qd1-d4 hanging the queen. Black to move, ...exd4 is
        # the strongest clean conversion and remains the strict target.
        "fen": "rnbqkbnr/ppp2ppp/8/4p3/3Q4/8/PPP1PPPP/RNB1KBNR b KQkq - 0 3",
        "expected": ["e5d4"],
        "note": "Black should grab the hanging queen with exd4",
    },
}

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
STRONG_OPENINGS = {"e2e4", "d2d4", "g1f3", "c2c4", "b1c3"}

# ---------------------------------------------------------------------------
# chess_mcts driver
# ---------------------------------------------------------------------------

def run_mcts(fens, simulations, timeout_s=600):
    """Run chess_mcts --json on a list of FENs (one per line). Returns list of dicts."""
    if not MCTS_BIN.exists():
        raise FileNotFoundError(f"chess_mcts binary not found at {MCTS_BIN}")
    inp = "\n".join(fens) + "\n"
    proc = subprocess.run(
        [str(MCTS_BIN), "--simulations", str(simulations), "--json"],
        input=inp,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    out = proc.stdout.strip()
    results = []
    for line in out.split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            results.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"  [warn] bad JSON line: {line[:120]} ({e})", file=sys.stderr)
    return results


def run_mcts_single(fen, simulations, timeout_s=600):
    res = run_mcts([fen], simulations, timeout_s=timeout_s)
    return res[0] if res else None


def posterior_visits(move):
    return move.get(
        "posterior_visits",
        move.get("effective_visits", move.get("visits", 0) + move.get("prior_visits", 0)),
    )


def posterior_distribution_from_moves(moves):
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
        return {}
    return {uci: weight / float(total) for uci, weight in weights.items()}


def distribution_metrics_from_moves(moves):
    probs = posterior_distribution_from_moves(moves)
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


def distribution_metrics_from_result(result):
    return distribution_metrics_from_moves(result.get("moves", []))


def js_divergence_bits(p, q):
    keys = set(p) | set(q)
    if not keys:
        return 0.0

    def _kl(a, b):
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


def mean_pairwise_js_divergence(results):
    dists = [posterior_distribution_from_moves(result.get("moves", [])) for result in results]
    if len(dists) < 2:
        return 0.0
    total = 0.0
    pairs = 0
    for i in range(len(dists)):
        for j in range(i + 1, len(dists)):
            total += js_divergence_bits(dists[i], dists[j])
            pairs += 1
    return total / float(pairs) if pairs else 0.0


# ---------------------------------------------------------------------------
# Tiny CPU MCTS (random rollouts, UCT selection at root, single-level)
# This is now a legacy baseline, not an exact algorithm mirror, because
# chess_mcts may evolve its root allocation policy independently.
# ---------------------------------------------------------------------------

def cpu_random_playout(board, max_depth=200):
    """One random playout from `board`. Returns winner from White's perspective:
    +1 white win, 0 draw, -1 black win."""
    b = board.copy(stack=False)
    depth = 0
    while not b.is_game_over(claim_draw=False) and depth < max_depth:
        moves = list(b.legal_moves)
        if not moves:
            break
        b.push(random.choice(moves))
        depth += 1
    if depth >= max_depth and not b.is_game_over():
        # Material eval fallback (matches GPU behavior)
        vals = {chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
                chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0}
        score = 0
        for sq, p in b.piece_map().items():
            v = vals[p.piece_type]
            score += v if p.color == chess.WHITE else -v
        if score > 100:
            return 1
        if score < -100:
            return -1
        return 0
    outcome = b.outcome(claim_draw=False)
    if outcome is None:
        return 0
    if outcome.winner is None:
        return 0
    return 1 if outcome.winner == chess.WHITE else -1


def cpu_mcts(fen, simulations):
    """CPU MCTS mirroring chess_mcts: even-split sims per root move, random rollouts.
    Returns (bestmove_uci, winrate, per_move_dict)."""
    board = chess.Board(fen)
    legal = list(board.legal_moves)
    if not legal:
        return ("0000", 0.0, {})
    if len(legal) == 1:
        return (legal[0].uci(), 0.5, {legal[0].uci(): {"winrate": 0.5, "visits": 1}})

    sims_per_move = max(10, simulations // len(legal))
    root_side = board.turn  # True=white
    per_move = {}
    for mv in legal:
        wins = 0.0
        visits = 0
        for _ in range(sims_per_move):
            b = board.copy(stack=False)
            b.push(mv)
            r = cpu_random_playout(b)
            # Convert to root-side perspective
            if root_side == chess.WHITE:
                score = 1.0 if r == 1 else (0.5 if r == 0 else 0.0)
            else:
                score = 1.0 if r == -1 else (0.5 if r == 0 else 0.0)
            wins += score
            visits += 1
        wr = wins / visits if visits else 0.0
        per_move[mv.uci()] = {"winrate": wr, "visits": visits}

    best = max(per_move.items(), key=lambda kv: (kv[1]["winrate"], kv[1]["visits"]))
    return (best[0], best[1]["winrate"], per_move)


# ---------------------------------------------------------------------------
# Check 1: Legal-move sanity
# ---------------------------------------------------------------------------

def check_legal_moves(n_positions=50, sims=1000, seed=42):
    print(f"[check 1] Legal-move sanity on {n_positions} random book positions (sims={sims})")
    if not BOOK_PATH.exists():
        return {"error": f"book not found at {BOOK_PATH}"}
    rng = random.Random(seed)
    fens = []
    with open(BOOK_PATH) as f:
        for line in f:
            try:
                rec = json.loads(line)
                if "fen" in rec:
                    fens.append(rec["fen"])
            except json.JSONDecodeError:
                continue
    if len(fens) < n_positions:
        n_positions = len(fens)
    sample = rng.sample(fens, n_positions)

    results = run_mcts(sample, sims)
    by_fen = {r["fen"]: r for r in results}

    illegal = []
    missing = []
    bad_uci = []
    legal_count = 0
    for fen in sample:
        res = by_fen.get(fen)
        if res is None:
            missing.append(fen)
            continue
        bm = res.get("bestmove", "")
        try:
            board = chess.Board(fen)
            ref_legal = list(board.legal_moves)
            if not ref_legal:
                # Terminal position (mate/stalemate) — engine should report 0000
                if bm == "0000":
                    legal_count += 1
                else:
                    illegal.append({"fen": fen, "bestmove": bm,
                                    "legal_moves": [], "note": "terminal but engine returned non-null move"})
                continue
            mv = chess.Move.from_uci(bm)
            if mv in ref_legal:
                legal_count += 1
            else:
                illegal.append({"fen": fen, "bestmove": bm,
                                "legal_moves": [m.uci() for m in ref_legal]})
        except Exception as e:
            bad_uci.append({"fen": fen, "bestmove": bm, "error": str(e)})

    rate = legal_count / max(1, len(sample))
    return {
        "n_positions": len(sample),
        "legal_count": legal_count,
        "legal_rate": rate,
        "illegal_examples": illegal[:5],
        "n_illegal": len(illegal),
        "bad_uci_examples": bad_uci[:5],
        "n_bad_uci": len(bad_uci),
        "missing_responses": len(missing),
        "pass": rate == 1.0 and not bad_uci,
    }


# ---------------------------------------------------------------------------
# Check 2: Repeatability with same seed
# ---------------------------------------------------------------------------

def check_repeatability(n_runs=5, n_fens=5, sims=5000, seed=7):
    print(f"[check 2] Repeatability: {n_runs} runs x {n_fens} FENs (sims={sims})")
    rng = random.Random(seed)
    fens = []
    with open(BOOK_PATH) as f:
        for line in f:
            try:
                rec = json.loads(line)
                if "fen" in rec:
                    fens.append(rec["fen"])
            except json.JSONDecodeError:
                continue
    sample = rng.sample(fens, n_fens)

    per_fen = {}
    for fen in sample:
        moves = []
        winrates = []
        results = []
        for _ in range(n_runs):
            r = run_mcts_single(fen, sims)
            if r:
                results.append(r)
                moves.append(r.get("bestmove", ""))
                winrates.append(r.get("winrate", 0.0))
        ctr = Counter(moves)
        most_common, count = ctr.most_common(1)[0] if ctr else ("", 0)
        agreement = count / max(1, len(moves))
        run_metrics = [distribution_metrics_from_result(result) for result in results]
        per_fen[fen] = {
            "runs": moves,
            "winrates": winrates,
            "modal_move": most_common,
            "agreement_rate": agreement,
            "unique_moves": len(ctr),
            "average_entropy_bits": (
                sum(item["entropy_bits"] for item in run_metrics) / max(1, len(run_metrics))
            ),
            "average_normalized_entropy": (
                sum(item["normalized_entropy"] for item in run_metrics) / max(1, len(run_metrics))
            ),
            "average_effective_branching_factor": (
                sum(item["effective_branching_factor"] for item in run_metrics)
                / max(1, len(run_metrics))
            ),
            "average_top1_top2_gap": (
                sum(item["top1_top2_gap"] for item in run_metrics) / max(1, len(run_metrics))
            ),
            "mean_pairwise_js_divergence_bits": mean_pairwise_js_divergence(results),
        }

    avg_agreement = sum(d["agreement_rate"] for d in per_fen.values()) / max(1, len(per_fen))
    avg_entropy = sum(d["average_entropy_bits"] for d in per_fen.values()) / max(1, len(per_fen))
    avg_js = (
        sum(d["mean_pairwise_js_divergence_bits"] for d in per_fen.values())
        / max(1, len(per_fen))
    )
    return {
        "n_runs_per_fen": n_runs,
        "n_fens": len(sample),
        "average_agreement": avg_agreement,
        "average_entropy_bits": avg_entropy,
        "average_pairwise_js_divergence_bits": avg_js,
        "per_fen": per_fen,
        "note": "Default chess_mcts path is deterministic under seed_mode=fen; low JS divergence is expected.",
    }


# ---------------------------------------------------------------------------
# Check 3: Sim-count scaling
# ---------------------------------------------------------------------------

def check_sim_scaling(sim_counts=(1000, 5000, 25000, 100000)):
    print(f"[check 3] Sim-count scaling on tactical puzzles, sims={sim_counts}")
    out = {}
    for name, info in PUZZLES.items():
        fen = info["fen"]
        expected = info["expected"]
        per_count = {}
        for sc in sim_counts:
            r = run_mcts_single(fen, sc, timeout_s=900)
            if r is None:
                per_count[str(sc)] = {"error": "no response"}
                continue
            bm = r.get("bestmove", "")
            wr = r.get("winrate", 0.0)
            moves = r.get("moves", [])
            by_winrate = _sort_moves_by_winrate(moves)
            by_posterior = _sort_moves_by_posterior(moves)
            by_root_score = _sort_moves_by_root_score(moves)
            winrate_ranks = {}
            posterior_ranks = {}
            root_score_ranks = {}
            for em in expected:
                winrate_ranks[em] = _move_rank(by_winrate, em)
                posterior_ranks[em] = _move_rank(by_posterior, em)
                root_score_ranks[em] = _move_rank(by_root_score, em)
            per_count[str(sc)] = {
                "bestmove": bm,
                "winrate": wr,
                "posterior_winrate": r.get("posterior_winrate", wr),
                "root_score": r.get("root_score"),
                "selection_metric": r.get("selection_metric"),
                "bestmove_alignment": _bestmove_alignment(r),
                "distribution": distribution_metrics_from_result(r),
                "matches_expected": bm in expected if expected else None,
                "expected_rank": winrate_ranks,
                "expected_posterior_rank": posterior_ranks,
                "expected_root_score_rank": root_score_ranks,
                "top3": [m.get("move") for m in by_winrate[:3]],
                "top3_by_posterior": [m.get("move") for m in by_posterior[:3]],
                "top3_by_root_score": [m.get("move") for m in by_root_score[:3]],
            }
        # Did 100K converge?
        last = per_count.get(str(sim_counts[-1]), {})
        converged = last.get("matches_expected") if expected else None
        out[name] = {
            "fen": fen,
            "expected": expected,
            "note": info["note"],
            "by_sim_count": per_count,
            "converged_at_max": converged,
        }
    return out


# ---------------------------------------------------------------------------
# Check 4: Move distribution diversity
# ---------------------------------------------------------------------------

def check_diversity(n_runs=50, sims=10000):
    print(f"[check 4] Move-distribution diversity on start position ({n_runs} runs, sims={sims})")
    moves = []
    winrates = []
    results = []
    for i in range(n_runs):
        r = run_mcts_single(START_FEN, sims)
        if r:
            results.append(r)
            moves.append(r.get("bestmove", ""))
            winrates.append(r.get("winrate", 0.0))
        time.sleep(0.02)
    ctr = Counter(moves)
    n = sum(ctr.values())
    histogram = {mv: {"count": c, "frac": c / n} for mv, c in ctr.most_common()}
    strong_count = sum(c for mv, c in ctr.items() if mv in STRONG_OPENINGS)
    strong_frac = strong_count / max(1, n)
    unique = len(ctr)
    most_common_mv, most_common_count = ctr.most_common(1)[0] if ctr else ("", 0)
    most_common_frac = most_common_count / max(1, n)
    diagnosis = "unknown"
    if unique == 1:
        diagnosis = "BROKEN: always picks the same move (no exploration)"
    elif most_common_frac > 0.95:
        diagnosis = "SUSPICIOUS: one move dominates >95% (low diversity)"
    elif strong_frac >= 0.5:
        diagnosis = "REASONABLE: strong openings dominate"
    elif unique >= 15:
        diagnosis = "SUSPICIOUS: nearly-uniform over all 20 moves (UCT/eval may be flat)"
    else:
        diagnosis = "MIXED: not dominated by strong openings but not uniform either"
    run_metrics = [distribution_metrics_from_result(result) for result in results]
    return {
        "n_runs": len(moves),
        "sims_per_run": sims,
        "histogram": histogram,
        "unique_moves": unique,
        "strong_opening_fraction": strong_frac,
        "modal_move": most_common_mv,
        "modal_fraction": most_common_frac,
        "mean_winrate": sum(winrates) / max(1, len(winrates)),
        "average_entropy_bits": (
            sum(item["entropy_bits"] for item in run_metrics) / max(1, len(run_metrics))
        ),
        "average_normalized_entropy": (
            sum(item["normalized_entropy"] for item in run_metrics) / max(1, len(run_metrics))
        ),
        "average_effective_branching_factor": (
            sum(item["effective_branching_factor"] for item in run_metrics)
            / max(1, len(run_metrics))
        ),
        "average_top1_top2_gap": (
            sum(item["top1_top2_gap"] for item in run_metrics) / max(1, len(run_metrics))
        ),
        "mean_pairwise_js_divergence_bits": mean_pairwise_js_divergence(results),
        "diagnosis": diagnosis,
    }


# ---------------------------------------------------------------------------
# Check 5: Root-UCT regression probe
# ---------------------------------------------------------------------------

def _posterior_visits(move):
    return posterior_visits(move)


def _posterior_winrate(move):
    return move.get("posterior_winrate", move.get("winrate", 0.0))


def _root_score(move):
    return move.get("root_score", _posterior_winrate(move))


def _sort_moves_by_visits(moves):
    return sorted(
        moves,
        key=lambda m: (m.get("visits", 0), m.get("winrate", 0.0)),
        reverse=True,
    )


def _sort_moves_by_effective_visits(moves):
    return sorted(
        moves,
        key=lambda m: (
            _posterior_visits(m),
            _posterior_winrate(m),
        ),
        reverse=True,
    )


def _sort_moves_by_winrate(moves):
    return sorted(
        moves,
        key=lambda m: (m.get("winrate", 0.0), m.get("visits", 0)),
        reverse=True,
    )


def _sort_moves_by_posterior(moves):
    return sorted(
        moves,
        key=lambda m: (
            _posterior_winrate(m),
            _posterior_visits(m),
        ),
        reverse=True,
    )


def _sort_moves_by_root_score(moves):
    return sorted(
        moves,
        key=lambda m: (
            _root_score(m),
            _posterior_visits(m),
            _posterior_winrate(m),
        ),
        reverse=True,
    )


def _move_rank(moves_sorted, target):
    for i, move in enumerate(moves_sorted):
        if move.get("move") == target:
            return i + 1
    return None


def _bestmove_alignment(result):
    moves = result.get("moves", [])
    bestmove = result.get("bestmove", "")
    if not moves:
        return {
            "selection_metric": result.get("selection_metric"),
            "selection_shortlist_visits": result.get("selection_shortlist_visits"),
            "leaders": {},
            "mismatches": {},
        }

    leaders = {
        "visits": _sort_moves_by_visits(moves)[0].get("move"),
        "posterior_visits": _sort_moves_by_effective_visits(moves)[0].get("move"),
        "posterior_winrate": _sort_moves_by_posterior(moves)[0].get("move"),
        "root_score": _sort_moves_by_root_score(moves)[0].get("move"),
    }
    mismatches = {
        metric: (bestmove != leader if leader else None)
        for metric, leader in leaders.items()
    }
    return {
        "selection_metric": result.get("selection_metric"),
        "selection_shortlist_visits": result.get("selection_shortlist_visits"),
        "leaders": leaders,
        "mismatches": mismatches,
    }


def check_root_uct_regression(low_sims=64, high_sims=5000):
    print(f"[check 5] Root-UCT regression probe (low={low_sims}, high={high_sims})")

    tactical = {}
    for name in ("mate_in_1_qxf7", "free_queen_capture"):
        info = PUZZLES[name]
        tactical[name] = {"fen": info["fen"], "expected": info["expected"], "by_sim_count": {}}
        for sims in (low_sims, high_sims):
            result = run_mcts_single(info["fen"], sims, timeout_s=900)
            if result is None:
                tactical[name]["by_sim_count"][str(sims)] = {"error": "no response"}
                continue
            moves = result.get("moves", [])
            total_visits = sum(m.get("visits", 0) for m in moves)
            visited_moves = sum(1 for m in moves if m.get("visits", 0) > 0)
            total_effective_visits = sum(_posterior_visits(m) for m in moves)
            by_visits = _sort_moves_by_visits(moves)
            by_effective = _sort_moves_by_effective_visits(moves)
            by_winrate = _sort_moves_by_winrate(moves)
            by_posterior = _sort_moves_by_posterior(moves)
            by_root_score = _sort_moves_by_root_score(moves)
            expected = info["expected"][0] if info["expected"] else None
            expected_row = next((m for m in moves if m.get("move") == expected), None) if expected else None
            bestmove = result.get("bestmove", "")
            tactical[name]["by_sim_count"][str(sims)] = {
                "bestmove": bestmove,
                "winrate": result.get("winrate", 0.0),
                "posterior_winrate": result.get("posterior_winrate", result.get("winrate", 0.0)),
                "root_score": result.get("root_score"),
                "selection_metric": result.get("selection_metric"),
                "selection_shortlist_visits": result.get("selection_shortlist_visits"),
                "bestmove_alignment": _bestmove_alignment(result),
                "distribution": distribution_metrics_from_result(result),
                "total_visits": total_visits,
                "total_effective_visits": total_effective_visits,
                "visited_moves": visited_moves,
                "root_coverage_fraction": visited_moves / max(1, len(moves)),
                "top3_by_visits": [m.get("move") for m in by_visits[:3]],
                "top3_by_effective_visits": [m.get("move") for m in by_effective[:3]],
                "top3_by_winrate": [m.get("move") for m in by_winrate[:3]],
                "top3_by_posterior": [m.get("move") for m in by_posterior[:3]],
                "top3_by_root_score": [m.get("move") for m in by_root_score[:3]],
                "expected_visit_rank": _move_rank(by_visits, expected) if expected else None,
                "expected_effective_visit_rank": _move_rank(by_effective, expected) if expected else None,
                "expected_winrate_rank": _move_rank(by_winrate, expected) if expected else None,
                "expected_posterior_rank": _move_rank(by_posterior, expected) if expected else None,
                "expected_root_score_rank": _move_rank(by_root_score, expected) if expected else None,
                "expected_visit_share": (
                    expected_row.get("visits", 0) / max(1, total_visits)
                    if expected_row is not None else None
                ),
                "expected_effective_visit_share": (
                    _posterior_visits(expected_row)
                    / max(1, total_effective_visits)
                    if expected_row is not None else None
                ),
                "bestmove_matches_posterior_top1": (
                    by_posterior[0].get("move") == bestmove if by_posterior else None
                ),
                "bestmove_matches_effective_top1": (
                    by_effective[0].get("move") == bestmove if by_effective else None
                ),
            }

    start = run_mcts_single(START_FEN, high_sims, timeout_s=900)
    start_summary = {"error": "no response"}
    if start is not None:
        moves = start.get("moves", [])
        total_visits = sum(m.get("visits", 0) for m in moves)
        total_effective_visits = sum(_posterior_visits(m) for m in moves)
        visited_moves = sum(1 for m in moves if m.get("visits", 0) > 0)
        by_visits = _sort_moves_by_visits(moves)
        by_effective = _sort_moves_by_effective_visits(moves)
        by_posterior = _sort_moves_by_posterior(moves)
        by_root_score = _sort_moves_by_root_score(moves)
        strong_visits = sum(
            m.get("visits", 0) for m in moves if m.get("move") in STRONG_OPENINGS
        )
        strong_effective_visits = sum(
            _posterior_visits(m)
            for m in moves if m.get("move") in STRONG_OPENINGS
        )
        modal = by_effective[0] if by_effective else {"move": "", "effective_visits": 0}
        start_summary = {
            "bestmove": start.get("bestmove", ""),
            "winrate": start.get("winrate", 0.0),
            "posterior_winrate": start.get("posterior_winrate", start.get("winrate", 0.0)),
            "root_score": start.get("root_score"),
            "selection_metric": start.get("selection_metric"),
            "selection_shortlist_visits": start.get("selection_shortlist_visits"),
            "bestmove_alignment": _bestmove_alignment(start),
            "distribution": distribution_metrics_from_result(start),
            "total_visits": total_visits,
            "total_effective_visits": total_effective_visits,
            "visited_moves": visited_moves,
            "root_coverage_fraction": visited_moves / max(1, len(moves)),
            "modal_by_visits": modal.get("move", ""),
            "modal_effective_visit_share": (
                _posterior_visits(modal)
                / max(1, total_effective_visits)
            ),
            "top5_by_visits": [m.get("move") for m in by_visits[:5]],
            "top5_by_effective_visits": [m.get("move") for m in by_effective[:5]],
            "top5_by_posterior": [m.get("move") for m in by_posterior[:5]],
            "top5_by_root_score": [m.get("move") for m in by_root_score[:5]],
            "strong_opening_visit_share": strong_visits / max(1, total_visits),
            "strong_opening_effective_visit_share": (
                strong_effective_visits / max(1, total_effective_visits)
            ),
            "bestmove_matches_posterior_top1": (
                by_posterior[0].get("move") == start.get("bestmove", "")
                if by_posterior else None
            ),
        }

    return {
        "low_sims": low_sims,
        "high_sims": high_sims,
        "tactical": tactical,
        "start_position": start_summary,
    }


# ---------------------------------------------------------------------------
# Check 6: GPU vs CPU MCTS
# ---------------------------------------------------------------------------

def check_vs_cpu(sims=10000, n_positions=5, seed=11):
    print(f"[check 6] GPU vs CPU MCTS on {n_positions} positions (sims={sims})")
    rng = random.Random(seed)
    fens = []
    with open(BOOK_PATH) as f:
        for line in f:
            try:
                rec = json.loads(line)
                if "fen" in rec:
                    fens.append(rec["fen"])
            except json.JSONDecodeError:
                continue
    sample = rng.sample(fens, n_positions)

    rows = []
    agree_top3 = 0
    winrate_close = 0
    for fen in sample:
        # GPU
        gpu = run_mcts_single(fen, sims)
        if gpu is None:
            continue
        gpu_bm = gpu.get("bestmove", "")
        gpu_wr = gpu.get("posterior_winrate", gpu.get("winrate", 0.0))
        gpu_moves_sorted = _sort_moves_by_posterior(gpu.get("moves", []))
        gpu_top3 = [m.get("move") for m in gpu_moves_sorted[:3]]

        # CPU
        random.seed(0xC0FFEE ^ hash(fen) & 0xFFFFFFFF)
        cpu_bm, cpu_wr, cpu_per_move = cpu_mcts(fen, sims)
        cpu_top3 = [k for k, _ in sorted(
            cpu_per_move.items(),
            key=lambda kv: (kv[1]["winrate"], kv[1]["visits"]),
            reverse=True,
        )[:3]]

        in_top3 = gpu_bm in cpu_top3 or cpu_bm in gpu_top3
        wr_diff = abs(gpu_wr - cpu_wr)
        close = wr_diff <= 0.15  # slightly looser than 0.1 because rollouts are noisy
        if in_top3:
            agree_top3 += 1
        if close:
            winrate_close += 1
        rows.append({
            "fen": fen,
            "gpu_bestmove": gpu_bm,
            "gpu_winrate": gpu_wr,
            "gpu_top3": gpu_top3,
            "cpu_bestmove": cpu_bm,
            "cpu_winrate": cpu_wr,
            "cpu_top3": cpu_top3,
            "in_top3_match": in_top3,
            "winrate_diff": wr_diff,
            "winrate_close_15": close,
        })

    return {
        "n_positions": len(rows),
        "sims": sims,
        "top3_agreement_rate": agree_top3 / max(1, len(rows)),
        "winrate_close_rate": winrate_close / max(1, len(rows)),
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    if not MCTS_BIN.exists():
        print(f"ERROR: chess_mcts binary not found at {MCTS_BIN}", file=sys.stderr)
        sys.exit(1)
    if not BOOK_PATH.exists():
        print(f"ERROR: book not found at {BOOK_PATH}", file=sys.stderr)
        sys.exit(1)

    report = {
        "binary": str(MCTS_BIN),
        "binary_mtime": os.path.getmtime(MCTS_BIN),
        "book": str(BOOK_PATH),
        "python_chess_version": chess.__version__,
        "started_at": time.time(),
    }

    print("=" * 70)
    report["check_1_legal_moves"] = check_legal_moves()
    print("=" * 70)
    report["check_2_repeatability"] = check_repeatability()
    print("=" * 70)
    report["check_3_sim_scaling"] = check_sim_scaling()
    print("=" * 70)
    report["check_4_diversity"] = check_diversity()
    print("=" * 70)
    report["check_5_root_uct"] = check_root_uct_regression()
    print("=" * 70)
    report["check_6_vs_cpu"] = check_vs_cpu()
    print("=" * 70)

    report["finished_at"] = time.time()
    report["elapsed_s"] = report["finished_at"] - report["started_at"]

    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote {REPORT_PATH}  (elapsed {report['elapsed_s']:.1f}s)")


if __name__ == "__main__":
    main()
