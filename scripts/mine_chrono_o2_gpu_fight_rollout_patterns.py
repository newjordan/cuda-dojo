#!/usr/bin/env python3
"""Mine chrono/O2 motifs against GPU-only forced-candidate fight rollouts.

The host code selects recorded frontier rows, matches the row move through
refcuda GPU legal moves, forces that move with refcuda GPU make_move, then lets
both sides continue via refcuda GPU search. Terminal checks also call refcuda.
No host chess library is imported and no CPU legal-move/referee path is used.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from mine_chrono_o2_gpu_search_patterns import (
    DEFAULT_CONDITION,
    DEFAULT_LIB,
    RefCuda,
    as_number,
    chrono_rows_by_hash,
    combinations,
    complete_fen,
    frontier_of,
    motif_id,
    read_json,
    root_records,
    round6,
    row_tags,
    sha256_file,
    write_json,
)


SCHEMA_VERSION = "dojo.chrono_o2_gpu_fight_rollout_pattern_mining.v1"
SIDE_NAMES = {"w": "white", "b": "black"}


class FightRefCuda(RefCuda):
    def __init__(self, lib_path: Path):
        super().__init__(lib_path)
        self.lib.refc_legal_moves.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_int]
        self.lib.refc_legal_moves.restype = ctypes.c_int
        self.lib.refc_make_move.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_void_p]
        self.lib.refc_make_move.restype = ctypes.c_int
        self.lib.refc_is_checkmate.argtypes = [ctypes.c_void_p]
        self.lib.refc_is_checkmate.restype = ctypes.c_int
        self.lib.refc_is_stalemate.argtypes = [ctypes.c_void_p]
        self.lib.refc_is_stalemate.restype = ctypes.c_int
        self.lib.refc_search_new_game.argtypes = []
        self.lib.refc_search_new_game.restype = None

    def legal_moves(self, pos: ctypes.Array[ctypes.c_char]) -> list[dict[str, Any]]:
        buf = (ctypes.c_int32 * 256)()
        n = int(self.lib.refc_legal_moves(pos, buf, 256))
        if n < 0:
            raise RuntimeError(f"refc_legal_moves failed rc={n}")
        return [{"encoded": int(buf[i]), "uci": self.move_to_uci(int(buf[i]))} for i in range(n)]

    def move_from_uci(self, pos: ctypes.Array[ctypes.c_char], uci: str) -> int | None:
        for move in self.legal_moves(pos):
            if move["uci"] == uci:
                return int(move["encoded"])
        return None

    def make_move(self, pos: ctypes.Array[ctypes.c_char], move: int) -> ctypes.Array[ctypes.c_char]:
        out = ctypes.create_string_buffer(self.position_size)
        rc = int(self.lib.refc_make_move(pos, ctypes.c_int32(move), out))
        if rc != 0:
            raise RuntimeError(f"refc_make_move failed rc={rc}")
        return out

    def is_checkmate(self, pos: ctypes.Array[ctypes.c_char]) -> bool:
        return bool(int(self.lib.refc_is_checkmate(pos)))

    def is_stalemate(self, pos: ctypes.Array[ctypes.c_char]) -> bool:
        return bool(int(self.lib.refc_is_stalemate(pos)))

    def search_best_pos(self, pos: ctypes.Array[ctypes.c_char], depth: int, movetime_ms: int) -> dict[str, Any]:
        move_out = ctypes.c_int32(0)
        score_out = ctypes.c_int(0)
        rc = int(self.lib.refc_search_best_move(
            pos,
            ctypes.c_int(depth),
            ctypes.c_int(movetime_ms),
            ctypes.byref(move_out),
            ctypes.byref(score_out),
        ))
        if rc < 0:
            raise RuntimeError(f"refc_search_best_move failed rc={rc}")
        move = int(move_out.value)
        return {
            "returnCode": rc,
            "bestMoveEncoded": move,
            "bestMove": self.move_to_uci(move),
            "scoreCp": int(score_out.value),
        }


def side_from_fen(fen: str) -> str:
    parts = complete_fen(fen).split()
    return parts[1] if len(parts) > 1 and parts[1] in SIDE_NAMES else "w"


def other_side(side: str) -> str:
    return "b" if side == "w" else "w"


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def accepted(bridge_row: dict[str, Any], chrono_row: dict[str, Any]) -> bool:
    frontier = frontier_of(bridge_row)
    injection = (bridge_row.get("pzrgCandidate") or {}).get("injection_relevance") or {}
    return bool(
        (frontier.get("gate") or {}).get("acceptedUsefulInjection")
        or injection.get("accepted_useful_injection")
        or injection.get("promotion_gate_approved")
        or (chrono_row.get("runtimeChoiceSignal") or {}).get("acceptedUsefulInjection")
    )


def selected(bridge_row: dict[str, Any], chrono_row: dict[str, Any]) -> bool:
    frontier = frontier_of(bridge_row)
    return bool((frontier.get("gate") or {}).get("selectedMoveInFrontier") or (chrono_row.get("runtimeChoiceSignal") or {}).get("selectedMoveInFrontier"))


def candidate_rows(
    bridge_rows: list[dict[str, Any]],
    chrono_by_hash: dict[str, dict[str, Any]],
    max_rank: int,
    allowed_root_ids: set[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for idx, bridge_row in enumerate(bridge_rows):
        frontier = frontier_of(bridge_row)
        rank = max(1, int(as_number(frontier.get("rank") or bridge_row.get("rank"), 999)))
        if rank > max_rank:
            continue
        row_hash = bridge_row.get("logicRayFrontierHash")
        chrono_row = chrono_by_hash.get(row_hash)
        if not chrono_row:
            missing.append({"rowIndex": idx, "hash": row_hash})
            continue
        root_id = str(frontier.get("rootId") or bridge_row.get("rootId") or chrono_row.get("rootId") or "root")
        if allowed_root_ids is not None and root_id not in allowed_root_ids:
            continue
        root_fen = str(frontier.get("rootFen") or bridge_row.get("rootFen") or chrono_row.get("rootFen") or "")
        move = str(frontier.get("move") or bridge_row.get("move") or chrono_row.get("move") or "")
        rows.append({
            "bridgeRow": bridge_row,
            "chronoRow": chrono_row,
            "hash": row_hash,
            "bridgeId": bridge_row.get("bridgeId"),
            "rootId": root_id,
            "rootFen": complete_fen(root_fen),
            "move": move,
            "originalRank": rank,
            "accepted": accepted(bridge_row, chrono_row),
            "selected": selected(bridge_row, chrono_row),
            "tags": row_tags(bridge_row, chrono_row),
        })
    return rows, missing


def rollout_candidate(refc: FightRefCuda, item: dict[str, Any], depth: int, movetime_ms: int, max_plies: int) -> dict[str, Any]:
    candidate_side = side_from_fen(item["rootFen"])
    current_side = other_side(candidate_side)
    refc.lib.refc_search_new_game()
    try:
        root_pos = refc.position(item["rootFen"])
        forced_move = refc.move_from_uci(root_pos, item["move"])
        if forced_move is None:
            return {
                "status": "illegal_candidate",
                "candidateSide": SIDE_NAMES[candidate_side],
                "fightScore": None,
                "reason": "candidate move not found in refcuda GPU legal moves",
            }
        pos = refc.make_move(root_pos, forced_move)
        played = [item["move"]]
        last_score = None
        last_score_side = None
        for ply in range(max_plies + 1):
            if refc.is_checkmate(pos):
                winner = other_side(current_side)
                return {
                    "status": "checkmate",
                    "candidateSide": SIDE_NAMES[candidate_side],
                    "winner": SIDE_NAMES[winner],
                    "pliesAfterForcedMove": ply,
                    "moves": played,
                    "fightScore": 1.0 if winner == candidate_side else -1.0,
                }
            if refc.is_stalemate(pos):
                return {
                    "status": "stalemate",
                    "candidateSide": SIDE_NAMES[candidate_side],
                    "winner": None,
                    "pliesAfterForcedMove": ply,
                    "moves": played,
                    "fightScore": 0.0,
                }
            if ply >= max_plies:
                signed_score = None
                if last_score is not None and last_score_side is not None:
                    signed_score = last_score if last_score_side == candidate_side else -last_score
                capped = clamp((signed_score or 0) / 1000.0, -1.0, 1.0)
                return {
                    "status": "max_plies",
                    "candidateSide": SIDE_NAMES[candidate_side],
                    "winner": None,
                    "pliesAfterForcedMove": ply,
                    "moves": played,
                    "lastScoreCpMoverPov": signed_score,
                    "fightScore": round6(capped),
                }
            search = refc.search_best_pos(pos, depth, movetime_ms)
            last_score = int(search["scoreCp"])
            last_score_side = current_side
            move = int(search["bestMoveEncoded"])
            if not search["bestMove"] or search["bestMove"] == "0000":
                return {
                    "status": "null_bestmove",
                    "candidateSide": SIDE_NAMES[candidate_side],
                    "winner": None,
                    "pliesAfterForcedMove": ply,
                    "moves": played,
                    "fightScore": None,
                }
            played.append(str(search["bestMove"]))
            pos = refc.make_move(pos, move)
            current_side = other_side(current_side)
    except Exception as exc:  # noqa: BLE001 - persisted for lab receipt
        return {
            "status": "error",
            "candidateSide": SIDE_NAMES.get(candidate_side, candidate_side),
            "fightScore": None,
            "error": str(exc),
        }


def attach_rollouts(refc: FightRefCuda, rows: list[dict[str, Any]], depth: int, movetime_ms: int, max_plies: int) -> list[dict[str, Any]]:
    out = []
    for idx, item in enumerate(rows):
        rollout = rollout_candidate(refc, item, depth, movetime_ms, max_plies)
        score = rollout.get("fightScore")
        out.append({
            **item,
            "rolloutIndex": idx,
            "rollout": rollout,
            "fightScore": float(score) if isinstance(score, (int, float)) and math.isfinite(float(score)) else None,
            "tagSet": set(item["tags"]),
        })
    return out


def group_by_root(items: list[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        if item["fightScore"] is not None:
            groups[item["rootId"]].append(item)
    return sorted((root_id, sorted(rows, key=lambda row: row["originalRank"])) for root_id, rows in groups.items())


def matches_motif(item: dict[str, Any], motif: dict[str, Any]) -> bool:
    return all(tag in item["tagSet"] for tag in motif["tags"])


def public_choice(item: dict[str, Any], score: float) -> dict[str, Any]:
    return {
        "bridgeId": item.get("bridgeId"),
        "hash": item.get("hash"),
        "move": item.get("move"),
        "originalRank": item.get("originalRank"),
        "rankingScore": round6(score),
        "fightScore": round6(item.get("fightScore")),
        "rolloutStatus": (item.get("rollout") or {}).get("status"),
        "acceptedUsefulInjection": item.get("accepted"),
        "selectedMoveInFrontier": item.get("selected"),
    }


def evaluate_ranked(groups: list[tuple[str, list[dict[str, Any]]]], score_fn, name: str) -> dict[str, Any]:
    roots = []
    score_sum = 0.0
    positive = neutral = negative = accepted_top1 = selected_top1 = 0
    for root_id, rows in groups:
        ranked = sorted(
            [{"item": item, "score": score_fn(item)} for item in rows],
            key=lambda entry: (-entry["score"], entry["item"]["originalRank"]),
        )
        if not ranked:
            continue
        top = ranked[0]["item"]
        fight_score = float(top["fightScore"])
        score_sum += fight_score
        if fight_score > 0:
            positive += 1
        elif fight_score < 0:
            negative += 1
        else:
            neutral += 1
        if top["accepted"]:
            accepted_top1 += 1
        if top["selected"]:
            selected_top1 += 1
        roots.append({
            "rootId": root_id,
            "rowCount": len(rows),
            "top1": public_choice(top, ranked[0]["score"]),
        })
    root_count = len(roots)
    return {
        "name": name,
        "rootCount": root_count,
        "scoreSum": round6(score_sum),
        "meanFightScore": round6(score_sum / root_count) if root_count else None,
        "positiveTop1": positive,
        "neutralTop1": neutral,
        "negativeTop1": negative,
        "acceptedUsefulTop1": accepted_top1,
        "selectedMoveTop1": selected_top1,
        "roots": roots,
    }


def evaluate_frontier(groups: list[tuple[str, list[dict[str, Any]]]]) -> dict[str, Any]:
    return evaluate_ranked(groups, lambda item: -item["originalRank"], "frontier_rank")


def evaluate_motif(motif: dict[str, Any], groups: list[tuple[str, list[dict[str, Any]]]]) -> dict[str, Any]:
    return evaluate_ranked(groups, lambda item: 1 if matches_motif(item, motif) else 0, motif["id"])


def metric_delta(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    mean_delta = None
    if candidate.get("meanFightScore") is not None and baseline.get("meanFightScore") is not None:
        mean_delta = round6(candidate["meanFightScore"] - baseline["meanFightScore"])
    return {
        "meanFightScore": mean_delta,
        "positiveTop1": candidate["positiveTop1"] - baseline["positiveTop1"],
        "acceptedUsefulTop1": candidate["acceptedUsefulTop1"] - baseline["acceptedUsefulTop1"],
    }


def metrics_public(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "rootCount": metrics["rootCount"],
        "meanFightScore": metrics["meanFightScore"],
        "positiveTop1": metrics["positiveTop1"],
        "neutralTop1": metrics["neutralTop1"],
        "negativeTop1": metrics["negativeTop1"],
        "acceptedUsefulTop1": metrics["acceptedUsefulTop1"],
        "selectedMoveTop1": metrics["selectedMoveTop1"],
    }


def generate_motifs(items: list[dict[str, Any]], max_size: int, min_roots: int, min_rows: int) -> list[dict[str, Any]]:
    stats: dict[tuple[str, ...], dict[str, Any]] = {}
    for item in items:
        if item["fightScore"] is None:
            continue
        for combo in combinations(item["tags"], max_size):
            entry = stats.setdefault(combo, {
                "id": motif_id(combo),
                "tags": combo,
                "rows": 0,
                "rootIds": set(),
                "scoreSum": 0.0,
                "positiveRows": 0,
                "acceptedRows": 0,
                "selectedRows": 0,
                "examples": [],
            })
            entry["rows"] += 1
            entry["rootIds"].add(item["rootId"])
            entry["scoreSum"] += float(item["fightScore"])
            if item["fightScore"] > 0:
                entry["positiveRows"] += 1
            if item["accepted"]:
                entry["acceptedRows"] += 1
            if item["selected"]:
                entry["selectedRows"] += 1
            if len(entry["examples"]) < 8:
                entry["examples"].append({
                    "rootId": item["rootId"],
                    "move": item["move"],
                    "originalRank": item["originalRank"],
                    "fightScore": round6(item["fightScore"]),
                    "rolloutStatus": (item.get("rollout") or {}).get("status"),
                    "acceptedUsefulInjection": item["accepted"],
                    "selectedMoveInFrontier": item["selected"],
                    "hash": item["hash"],
                })
    return [{
        **entry,
        "rootCount": len(entry["rootIds"]),
        "meanFightScore": round6(entry["scoreSum"] / entry["rows"]) if entry["rows"] else None,
    } for entry in stats.values() if entry["rows"] >= min_rows and len(entry["rootIds"]) >= min_roots]


def motif_public(motif: dict[str, Any], corpus_rows: int) -> dict[str, Any]:
    return {
        "id": motif["id"],
        "tags": list(motif["tags"]),
        "size": len(motif["tags"]),
        "rowCount": motif["rows"],
        "rootCount": motif["rootCount"],
        "meanFightScore": motif["meanFightScore"],
        "positiveRows": motif["positiveRows"],
        "corpusRowShare": round6(motif["rows"] / corpus_rows) if corpus_rows else 0,
        "acceptedRows": motif["acceptedRows"],
        "selectedRows": motif["selectedRows"],
        "examples": motif["examples"],
    }


def objective(metrics: dict[str, Any]) -> float:
    return as_number(metrics.get("meanFightScore"), -999) * 10000 + metrics["positiveTop1"] * 100 + metrics["acceptedUsefulTop1"]


def rank_motifs(motifs: list[dict[str, Any]], groups: list[tuple[str, list[dict[str, Any]]]], baseline: dict[str, Any], top_k: int, corpus_rows: int) -> list[dict[str, Any]]:
    ranked = []
    for motif in motifs:
        metrics = evaluate_motif(motif, groups)
        ranked.append({
            "motif": motif,
            "metrics": metrics,
            "delta": metric_delta(metrics, baseline),
            "score": objective(metrics) + as_number(motif.get("meanFightScore"), -999) * 100 + motif["rootCount"],
        })
    ranked.sort(key=lambda item: (-item["score"], item["motif"]["id"]))
    return [{
        **motif_public(entry["motif"], corpus_rows),
        "metrics": metrics_public(entry["metrics"]),
        "deltaVsFrontierRank": entry["delta"],
    } for entry in ranked[:top_k]]


def root_folds(root_ids: list[str], folds: int) -> list[list[str]]:
    return [[root_id for idx, root_id in enumerate(root_ids) if idx % folds == fold] for fold in range(folds)]


def filter_groups(groups: list[tuple[str, list[dict[str, Any]]]], root_ids: set[str]) -> list[tuple[str, list[dict[str, Any]]]]:
    return [(root_id, rows) for root_id, rows in groups if root_id in root_ids]


def aggregate(name: str, metrics_list: list[dict[str, Any]]) -> dict[str, Any]:
    root_count = sum(item["rootCount"] for item in metrics_list)
    score_sum = sum(as_number(item.get("scoreSum"), 0) for item in metrics_list)
    positive = sum(item["positiveTop1"] for item in metrics_list)
    neutral = sum(item["neutralTop1"] for item in metrics_list)
    negative = sum(item["negativeTop1"] for item in metrics_list)
    accepted_top1 = sum(item["acceptedUsefulTop1"] for item in metrics_list)
    selected_top1 = sum(item["selectedMoveTop1"] for item in metrics_list)
    return {
        "name": name,
        "rootCount": root_count,
        "scoreSum": round6(score_sum),
        "meanFightScore": round6(score_sum / root_count) if root_count else None,
        "positiveTop1": positive,
        "neutralTop1": neutral,
        "negativeTop1": negative,
        "acceptedUsefulTop1": accepted_top1,
        "selectedMoveTop1": selected_top1,
        "roots": [root for metrics in metrics_list for root in metrics["roots"]],
    }


def select_best_train(motifs: list[dict[str, Any]], train_groups: list[tuple[str, list[dict[str, Any]]]]) -> dict[str, Any] | None:
    scored = []
    for motif in motifs:
        metrics = evaluate_motif(motif, train_groups)
        scored.append({"motif": motif, "metrics": metrics, "score": objective(metrics) + motif["rootCount"]})
    scored.sort(key=lambda item: (-item["score"], item["motif"]["id"]))
    return scored[0] if scored else None


def cross_validate(items: list[dict[str, Any]], groups: list[tuple[str, list[dict[str, Any]]]], args: argparse.Namespace) -> dict[str, Any]:
    root_ids = [root_id for root_id, _ in groups]
    folds = root_folds(root_ids, min(args.folds, len(root_ids)))
    baseline_evals = []
    candidate_evals = []
    reports = []
    for fold_idx, eval_roots in enumerate(folds):
        eval_set = set(eval_roots)
        train_set = set(root_ids) - eval_set
        train_groups = filter_groups(groups, train_set)
        eval_groups = filter_groups(groups, eval_set)
        train_items = [item for item in items if item["rootId"] in train_set and item["fightScore"] is not None]
        train_motifs = generate_motifs(train_items, args.max_size, args.min_roots, args.min_rows)
        train_baseline = evaluate_frontier(train_groups)
        eval_baseline = evaluate_frontier(eval_groups)
        selected = select_best_train(train_motifs, train_groups)
        train_candidate = evaluate_motif(selected["motif"], train_groups) if selected else train_baseline
        eval_candidate = evaluate_motif(selected["motif"], eval_groups) if selected else eval_baseline
        baseline_evals.append(eval_baseline)
        candidate_evals.append(eval_candidate)
        reports.append({
            "fold": fold_idx,
            "trainRootCount": len(train_set),
            "evalRootCount": len(eval_set),
            "trainMotifCount": len(train_motifs),
            "selectedMotif": {
                **motif_public(selected["motif"], len(train_items)),
                "trainMetrics": metrics_public(selected["metrics"]),
                "trainDeltaVsFrontierRank": metric_delta(train_candidate, train_baseline),
            } if selected else None,
            "evalBaseline": metrics_public(eval_baseline),
            "evalCandidate": metrics_public(eval_candidate),
            "evalDeltaVsBaseline": metric_delta(eval_candidate, eval_baseline),
        })
    baseline = aggregate("frontier_rank_cross_validation_baseline", baseline_evals)
    candidate = aggregate("gpu_fight_rollout_pattern_cross_validation", candidate_evals)
    delta = metric_delta(candidate, baseline)
    return {
        "baseline": baseline,
        "candidate": candidate,
        "deltaVsFrontierRank": delta,
        "observedLift": (
            delta["meanFightScore"] is not None and delta["meanFightScore"] > 0
        ) or delta["positiveTop1"] > 0,
        "folds": reports,
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bridge", required=True)
    parser.add_argument("--chrono", required=True)
    parser.add_argument("--omnifold", required=True)
    parser.add_argument("--out")
    parser.add_argument("--lib", default=str(DEFAULT_LIB))
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--movetime-ms", type=int, default=0)
    parser.add_argument("--max-plies", type=int, default=4)
    parser.add_argument("--max-rank", type=int, default=2)
    parser.add_argument(
        "--root-ids",
        default="",
        help="Optional comma-separated root ids for a focused GPU fight replay.",
    )
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--max-size", type=int, default=3)
    parser.add_argument("--min-roots", type=int, default=3)
    parser.add_argument("--min-rows", type=int, default=6)
    parser.add_argument("--top-k", type=int, default=24)
    parser.add_argument("--condition-source", default=str(DEFAULT_CONDITION))
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    bridge_path = Path(args.bridge).resolve()
    chrono_path = Path(args.chrono).resolve()
    omnifold_path = Path(args.omnifold).resolve()
    lib_path = Path(args.lib).resolve()
    bridge = read_json(bridge_path)
    chrono = read_json(chrono_path)
    omnifold = read_json(omnifold_path)
    rows = bridge.get("rows") or []
    roots = root_records(rows)
    allowed_root_ids = {
        root_id.strip()
        for root_id in str(args.root_ids or "").split(",")
        if root_id.strip()
    } or None
    candidates, missing = candidate_rows(rows, chrono_rows_by_hash(chrono), args.max_rank, allowed_root_ids)
    refc = FightRefCuda(lib_path)
    refc.lib.refc_search_init()
    try:
        items = attach_rollouts(refc, candidates, args.depth, args.movetime_ms, args.max_plies)
    finally:
        refc.lib.refc_search_shutdown()
    valid_items = [item for item in items if item["fightScore"] is not None]
    groups = group_by_root(valid_items)
    baseline = evaluate_frontier(groups)
    motifs = generate_motifs(valid_items, args.max_size, args.min_roots, args.min_rows)
    top_patterns = rank_motifs(motifs, groups, baseline, args.top_k, len(valid_items))
    cv = cross_validate(valid_items, groups, args) if groups else None
    status_counts: dict[str, int] = {}
    for item in items:
        status = str((item.get("rollout") or {}).get("status") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
    active_families = [
        family for family in (omnifold.get("foldFamilies") or [])
        if family.get("status") == "active_frontier_attachable"
    ]
    output = {
        "schemaVersion": SCHEMA_VERSION,
        "generatedAt": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "condition": {
            "source": str(Path(args.condition_source).resolve()),
            "runLabel": "scout_new_experiment_chrono_o2_gpu_forced_candidate_fight_rollout_pattern_mining",
            "changedFields": "post-hoc forced-candidate GPU self-play rollout labels plus motif mining only; no runtime behavior changed"
            + ("; root_id_subset_filter_applied" if allowed_root_ids is not None else ""),
            "labCondition": "scout/new_experiment/source_temporal_proxy_top_rank_frontier_rows",
            "metric": "root-fold mean forced-candidate GPU self-play fight score",
        },
        "sources": {
            "bridgePath": str(bridge_path),
            "bridgeSha256": sha256_file(bridge_path),
            "bridgeRows": len(rows),
            "chronoPath": str(chrono_path),
            "chronoSha256": sha256_file(chrono_path),
            "chronoRows": len(chrono.get("rows") or []),
            "omnifoldPath": str(omnifold_path),
            "omnifoldSha256": sha256_file(omnifold_path),
            "libRefCudaPath": str(lib_path),
            "libRefCudaSha256": sha256_file(lib_path),
            "rootCount": len(roots),
            "candidateRows": len(candidates),
            "validRolloutRows": len(valid_items),
            "missingChronoRows": len(missing),
            "rootIdSubset": sorted(allowed_root_ids) if allowed_root_ids is not None else None,
        },
        "gpuFightRolloutLabel": {
            "schemaVersion": "dojo.refcuda_gpu_forced_candidate_fight_rollout_label.v1",
            "depth": args.depth,
            "movetimeMs": args.movetime_ms,
            "maxPliesAfterForcedMove": args.max_plies,
            "maxRank": args.max_rank,
            "hostRole": "json_orchestration_and_uci_string_matching_only",
            "gpuWork": [
                "refc_legal_moves",
                "refc_make_move",
                "refc_is_checkmate",
                "refc_is_stalemate",
                "refc_search_best_move",
            ],
            "statusCounts": dict(sorted(status_counts.items())),
        },
        "omnifoldAttribution": {
            "activeFoldFamilyCount": len(active_families),
            "activeFoldFamilies": [
                {
                    "id": family.get("id"),
                    "foldFamily": family.get("foldFamily"),
                    "orderSet": family.get("orderSet"),
                    "activeVariantCount": family.get("activeVariantCount"),
                }
                for family in active_families
            ],
            "invarianceStatus": "frontier_attachable_only_untrained_omnifold_delta_not_claimed",
            "offManifoldAuditRows": (omnifold.get("aggregate") or {}).get("offManifoldAuditRows"),
        },
        "corpus": {
            "rowCount": len(items),
            "validRolloutRows": len(valid_items),
            "rootCount": len(groups),
            "acceptedRows": sum(1 for item in valid_items if item["accepted"]),
            "selectedRows": sum(1 for item in valid_items if item["selected"]),
            "positiveRows": sum(1 for item in valid_items if item["fightScore"] and item["fightScore"] > 0),
            "neutralRows": sum(1 for item in valid_items if item["fightScore"] == 0),
            "negativeRows": sum(1 for item in valid_items if item["fightScore"] and item["fightScore"] < 0),
        },
        "miningPolicy": {
            "algorithm": "enumerate_tag_conjunctions_then_rank_by_forced_candidate_gpu_fight_rollout_score",
            "maxMotifSize": args.max_size,
            "minRoots": args.min_roots,
            "minRows": args.min_rows,
            "candidateMotifs": len(motifs),
            "topK": args.top_k,
            "noRuntimePromotion": True,
        },
        "baseline": {"frontierRank": metrics_public(baseline)},
        "topGlobalPatterns": top_patterns,
        "crossValidation": {
            "baseline": metrics_public(cv["baseline"]) if cv else None,
            "candidate": metrics_public(cv["candidate"]) if cv else None,
            "deltaVsFrontierRank": cv["deltaVsFrontierRank"] if cv else None,
            "observedLift": cv["observedLift"] if cv else False,
            "folds": cv["folds"] if cv else [],
        },
        "rollouts": [{
            "rootId": item["rootId"],
            "bridgeId": item.get("bridgeId"),
            "hash": item.get("hash"),
            "move": item["move"],
            "rank": item["originalRank"],
            "acceptedUsefulInjection": item["accepted"],
            "selectedMoveInFrontier": item["selected"],
            "fightScore": round6(item["fightScore"]),
            "rollout": item["rollout"],
            "tags": item["tags"],
        } for item in items],
        "promotionPolicy": {
            "status": "not_promoted",
            "reason": "forced-candidate GPU fight rollouts are a micro scout over top-ranked source-temporal rows; promotion requires a frozen heldout gate/fight condition with accepted-injection lift",
            "blockers": [
                "micro_rollout_depth_and_ply_cap_only",
                "source_temporal_proxy_sidecar_used",
                "no_frozen_heldout_gate_or_external_fight_result",
            ],
        },
    }
    out_path = Path(args.out).resolve() if args.out else chrono_path.with_name(chrono_path.name.replace(".json", ".gpu_fight_rollout_patterns.json"))
    write_json(out_path, output)
    print(json.dumps({
        "ok": True,
        "output": str(out_path),
        "depth": args.depth,
        "maxPliesAfterForcedMove": args.max_plies,
        "maxRank": args.max_rank,
        "rootCount": len(groups),
        "candidateRows": len(candidates),
        "validRolloutRows": len(valid_items),
        "statusCounts": dict(sorted(status_counts.items())),
        "candidateMotifs": len(motifs),
        "baselineMeanFightScore": baseline["meanFightScore"],
        "crossValidatedMeanFightScore": cv["candidate"]["meanFightScore"] if cv else None,
        "crossValidatedDelta": cv["deltaVsFrontierRank"] if cv else None,
        "observedLift": cv["observedLift"] if cv else False,
        "promote": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
