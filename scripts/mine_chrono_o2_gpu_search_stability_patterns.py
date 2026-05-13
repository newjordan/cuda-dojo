#!/usr/bin/env python3
"""Mine chrono/O2 motifs against multi-depth GPU-search stability labels."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from mine_chrono_o2_gpu_search_patterns import (
    DEFAULT_CONDITION,
    DEFAULT_LIB,
    RefCuda,
    chrono_rows_by_hash,
    complete_fen,
    combinations,
    frontier_of,
    motif_id,
    public_metrics,
    read_json,
    root_records,
    round6,
    row_tags,
    run_gpu_search_labels,
    sha256_file,
    write_json,
)


SCHEMA_VERSION = "dojo.chrono_o2_gpu_search_stability_pattern_mining.v1"


def parse_depths(text: str) -> list[int]:
    depths = [int(part.strip()) for part in text.split(",") if part.strip()]
    if len(depths) < 2 or any(depth < 1 for depth in depths):
        raise ValueError("--depths must contain at least two positive integers")
    return depths


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


def root_fen_map(bridge_rows: list[dict[str, Any]]) -> dict[str, str]:
    return {row["rootId"]: row["rootFen"] for row in root_records(bridge_rows)}


def run_depth_labels(refc: RefCuda, roots: dict[str, str], depths: list[int], movetime_ms: int) -> dict[int, dict[str, dict[str, Any]]]:
    labels: dict[int, dict[str, dict[str, Any]]] = {}
    root_records_for_search = [{"rootId": root_id, "rootFen": fen} for root_id, fen in sorted(roots.items())]
    for depth in depths:
        labels[depth] = run_gpu_search_labels(refc, root_records_for_search, depth, movetime_ms)
    return labels


def stable_root_labels(depth_labels: dict[int, dict[str, dict[str, Any]]], depths: list[int]) -> dict[str, dict[str, Any]]:
    root_ids = sorted(set.intersection(*(set(depth_labels[depth].keys()) for depth in depths)))
    stable: dict[str, dict[str, Any]] = {}
    for root_id in root_ids:
        per_depth = {str(depth): depth_labels[depth][root_id] for depth in depths}
        ok = all(label.get("status") == "ok" for label in per_depth.values())
        best_moves = [label.get("bestMove") for label in per_depth.values()]
        stable_best = ok and len(set(best_moves)) == 1
        stable[root_id] = {
            "rootId": root_id,
            "status": "stable" if stable_best else "unstable",
            "searchOk": ok,
            "stableBestMove": best_moves[0] if stable_best else None,
            "bestMovesByDepth": {depth: per_depth[str(depth)].get("bestMove") for depth in map(str, depths)},
            "scoresByDepth": {depth: per_depth[str(depth)].get("scoreCp") for depth in map(str, depths)},
            "labelsByDepth": per_depth,
        }
    return stable


def join_items(bridge_rows: list[dict[str, Any]], chrono_by_hash: dict[str, dict[str, Any]], stable_labels: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    items: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for index, bridge_row in enumerate(bridge_rows):
        row_hash = bridge_row.get("logicRayFrontierHash")
        chrono_row = chrono_by_hash.get(row_hash)
        frontier = frontier_of(bridge_row)
        if not chrono_row:
            missing.append({"rowIndex": index, "hash": row_hash})
            continue
        root_id = str(frontier.get("rootId") or bridge_row.get("rootId") or chrono_row.get("rootId") or "root")
        label = stable_labels.get(root_id, {})
        move = str(frontier.get("move") or bridge_row.get("move") or chrono_row.get("move"))
        tags = row_tags(bridge_row, chrono_row)
        items.append({
            "bridgeRow": bridge_row,
            "chronoRow": chrono_row,
            "hash": row_hash,
            "bridgeId": bridge_row.get("bridgeId"),
            "rootId": root_id,
            "move": move,
            "originalRank": max(1, int(float(frontier.get("rank") or bridge_row.get("rank") or chrono_row.get("rank") or 1))),
            "accepted": accepted(bridge_row, chrono_row),
            "selected": selected(bridge_row, chrono_row),
            "stableRoot": label.get("status") == "stable",
            "stableBestMove": label.get("stableBestMove"),
            "stableBestMoveMatch": label.get("status") == "stable" and move == label.get("stableBestMove"),
            "tags": tags,
            "tagSet": set(tags),
        })
    return items, missing


def generate_motifs(items: list[dict[str, Any]], max_size: int, min_roots: int, min_rows: int) -> list[dict[str, Any]]:
    stats: dict[tuple[str, ...], dict[str, Any]] = {}
    for item in items:
        for combo in combinations(item["tags"], max_size):
            entry = stats.setdefault(combo, {
                "id": motif_id(combo),
                "tags": combo,
                "rows": 0,
                "stableBestRows": 0,
                "acceptedRows": 0,
                "selectedRows": 0,
                "rootIds": set(),
                "stableBestRootIds": set(),
                "examples": [],
            })
            entry["rows"] += 1
            entry["rootIds"].add(item["rootId"])
            if item["stableBestMoveMatch"]:
                entry["stableBestRows"] += 1
                entry["stableBestRootIds"].add(item["rootId"])
            if item["accepted"]:
                entry["acceptedRows"] += 1
            if item["selected"]:
                entry["selectedRows"] += 1
            if len(entry["examples"]) < 8:
                entry["examples"].append({
                    "rootId": item["rootId"],
                    "move": item["move"],
                    "originalRank": item["originalRank"],
                    "stableBestMove": item["stableBestMove"],
                    "stableBestMoveMatch": item["stableBestMoveMatch"],
                    "acceptedUsefulInjection": item["accepted"],
                    "selectedMoveInFrontier": item["selected"],
                    "hash": item["hash"],
                })
    result = []
    for entry in stats.values():
        if entry["rows"] >= min_rows and len(entry["rootIds"]) >= min_roots:
            result.append({
                **entry,
                "rootCount": len(entry["rootIds"]),
                "stableBestRootCount": len(entry["stableBestRootIds"]),
            })
    return result


def group_by_root(items: list[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        groups[item["rootId"]].append(item)
    return sorted(groups.items())


def filter_groups(groups: list[tuple[str, list[dict[str, Any]]]], root_ids: set[str]) -> list[tuple[str, list[dict[str, Any]]]]:
    return [(root_id, rows) for root_id, rows in groups if root_id in root_ids]


def matches_motif(item: dict[str, Any], motif: dict[str, Any]) -> bool:
    return all(tag in item["tagSet"] for tag in motif["tags"])


def public_ranked(entry: dict[str, Any]) -> dict[str, Any]:
    item = entry["item"]
    return {
        "bridgeId": item.get("bridgeId"),
        "hash": item.get("hash"),
        "move": item.get("move"),
        "originalRank": item.get("originalRank"),
        "comparisonRank": entry.get("comparisonRank"),
        "score": round6(entry.get("score")),
        "stableBestMove": item.get("stableBestMove"),
        "stableBestMoveMatch": item.get("stableBestMoveMatch"),
        "acceptedUsefulInjection": item.get("accepted"),
        "selectedMoveInFrontier": item.get("selected"),
    }


def evaluate_ranked(groups: list[tuple[str, list[dict[str, Any]]]], score_fn, name: str) -> dict[str, Any]:
    roots = []
    top1 = top3 = accepted_top1 = selected_top1 = 0
    rank_sum = rank_count = 0
    for root_id, rows in groups:
        ranked = sorted(
            [{"item": item, "score": score_fn(item)} for item in rows],
            key=lambda entry: (-entry["score"], entry["item"]["originalRank"]),
        )
        for index, entry in enumerate(ranked, start=1):
            entry["comparisonRank"] = index
        first_stable = next((entry for entry in ranked if entry["item"]["stableBestMoveMatch"]), None)
        if ranked and ranked[0]["item"]["stableBestMoveMatch"]:
            top1 += 1
        if any(entry["item"]["stableBestMoveMatch"] for entry in ranked[:3]):
            top3 += 1
        if ranked and ranked[0]["item"]["accepted"]:
            accepted_top1 += 1
        if ranked and ranked[0]["item"]["selected"]:
            selected_top1 += 1
        if first_stable:
            rank_sum += first_stable["comparisonRank"]
            rank_count += 1
        roots.append({
            "rootId": root_id,
            "rowCount": len(rows),
            "stableBestMove": rows[0].get("stableBestMove"),
            "top1": public_ranked(ranked[0]) if ranked else None,
            "stableBestMoveBestRank": first_stable["comparisonRank"] if first_stable else None,
        })
    return {
        "name": name,
        "rootCount": len(groups),
        "top1StableBestMoveMatches": top1,
        "top3StableBestMoveMatches": top3,
        "acceptedUsefulTop1": accepted_top1,
        "selectedMoveTop1": selected_top1,
        "stableBestRankSum": rank_sum,
        "stableBestRankCount": rank_count,
        "meanStableBestMoveRank": round6(rank_sum / rank_count) if rank_count else None,
        "roots": roots,
    }


def evaluate_frontier(groups: list[tuple[str, list[dict[str, Any]]]]) -> dict[str, Any]:
    return evaluate_ranked(groups, lambda item: -item["originalRank"], "frontier_rank")


def evaluate_motif(motif: dict[str, Any], groups: list[tuple[str, list[dict[str, Any]]]]) -> dict[str, Any]:
    return evaluate_ranked(groups, lambda item: 1 if matches_motif(item, motif) else 0, motif["id"])


def metric_delta(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    mean_delta = None
    if candidate.get("meanStableBestMoveRank") is not None and baseline.get("meanStableBestMoveRank") is not None:
        mean_delta = round6(candidate["meanStableBestMoveRank"] - baseline["meanStableBestMoveRank"])
    return {
        "top1": candidate["top1StableBestMoveMatches"] - baseline["top1StableBestMoveMatches"],
        "top3": candidate["top3StableBestMoveMatches"] - baseline["top3StableBestMoveMatches"],
        "meanStableBestMoveRank": mean_delta,
    }


def metrics_public(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "rootCount": metrics["rootCount"],
        "top1StableBestMoveMatches": metrics["top1StableBestMoveMatches"],
        "top3StableBestMoveMatches": metrics["top3StableBestMoveMatches"],
        "acceptedUsefulTop1": metrics["acceptedUsefulTop1"],
        "selectedMoveTop1": metrics["selectedMoveTop1"],
        "meanStableBestMoveRank": metrics["meanStableBestMoveRank"],
    }


def objective(metrics: dict[str, Any]) -> float:
    return (
        metrics["top1StableBestMoveMatches"] * 10000
        + metrics["top3StableBestMoveMatches"] * 1000
        - (metrics["meanStableBestMoveRank"] if metrics["meanStableBestMoveRank"] is not None else 999) * 10
    )


def motif_public(motif: dict[str, Any], corpus_rows: int) -> dict[str, Any]:
    return {
        "id": motif["id"],
        "tags": list(motif["tags"]),
        "size": len(motif["tags"]),
        "rowCount": motif["rows"],
        "rootCount": motif["rootCount"],
        "stableBestRows": motif["stableBestRows"],
        "stableBestRootCount": motif["stableBestRootCount"],
        "stableBestRate": round6(motif["stableBestRows"] / motif["rows"]) if motif["rows"] else 0,
        "corpusRowShare": round6(motif["rows"] / corpus_rows) if corpus_rows else 0,
        "acceptedRows": motif["acceptedRows"],
        "selectedRows": motif["selectedRows"],
        "examples": motif["examples"],
    }


def rank_motifs(motifs: list[dict[str, Any]], groups: list[tuple[str, list[dict[str, Any]]]], baseline: dict[str, Any], top_k: int, corpus_rows: int) -> list[dict[str, Any]]:
    scored = []
    for motif in motifs:
        metrics = evaluate_motif(motif, groups)
        scored.append({
            "motif": motif,
            "metrics": metrics,
            "delta": metric_delta(metrics, baseline),
            "score": objective(metrics) + (motif["stableBestRows"] / motif["rows"] if motif["rows"] else 0) * 100 + motif["rootCount"],
        })
    scored.sort(key=lambda item: (-item["score"], item["motif"]["id"]))
    return [{
        **motif_public(item["motif"], corpus_rows),
        "metrics": metrics_public(item["metrics"]),
        "deltaVsFrontierRank": item["delta"],
    } for item in scored[:top_k]]


def root_folds(root_ids: list[str], folds: int) -> list[list[str]]:
    return [[root_id for idx, root_id in enumerate(root_ids) if idx % folds == fold] for fold in range(folds)]


def aggregate(name: str, metrics_list: list[dict[str, Any]]) -> dict[str, Any]:
    root_count = sum(item["rootCount"] for item in metrics_list)
    top1 = sum(item["top1StableBestMoveMatches"] for item in metrics_list)
    top3 = sum(item["top3StableBestMoveMatches"] for item in metrics_list)
    accepted = sum(item["acceptedUsefulTop1"] for item in metrics_list)
    selected = sum(item["selectedMoveTop1"] for item in metrics_list)
    rank_sum = sum(item["stableBestRankSum"] for item in metrics_list)
    rank_count = sum(item["stableBestRankCount"] for item in metrics_list)
    return {
        "name": name,
        "rootCount": root_count,
        "top1StableBestMoveMatches": top1,
        "top3StableBestMoveMatches": top3,
        "acceptedUsefulTop1": accepted,
        "selectedMoveTop1": selected,
        "stableBestRankSum": rank_sum,
        "stableBestRankCount": rank_count,
        "meanStableBestMoveRank": round6(rank_sum / rank_count) if rank_count else None,
        "roots": [root for metrics in metrics_list for root in metrics["roots"]],
    }


def select_best_train(motifs: list[dict[str, Any]], train_groups: list[tuple[str, list[dict[str, Any]]]]) -> dict[str, Any] | None:
    scored = []
    for motif in motifs:
        metrics = evaluate_motif(motif, train_groups)
        scored.append({
            "motif": motif,
            "metrics": metrics,
            "score": objective(metrics) + motif["stableBestRows"] * 10 + motif["rootCount"],
        })
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
        train_items = [item for item in items if item["rootId"] in train_set]
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
    candidate = aggregate("gpu_search_stability_pattern_cross_validation", candidate_evals)
    delta = metric_delta(candidate, baseline)
    return {
        "baseline": baseline,
        "candidate": candidate,
        "deltaVsFrontierRank": delta,
        "observedLift": delta["top1"] > 0 or delta["top3"] > 0 or (delta["meanStableBestMoveRank"] is not None and delta["meanStableBestMoveRank"] < 0),
        "folds": reports,
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bridge", required=True)
    parser.add_argument("--chrono", required=True)
    parser.add_argument("--omnifold", required=True)
    parser.add_argument("--out")
    parser.add_argument("--lib", default=str(DEFAULT_LIB))
    parser.add_argument("--depths", default="4,5")
    parser.add_argument("--movetime-ms", type=int, default=0)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--max-size", type=int, default=3)
    parser.add_argument("--min-roots", type=int, default=3)
    parser.add_argument("--min-rows", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=24)
    parser.add_argument("--condition-source", default=str(DEFAULT_CONDITION))
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    depths = parse_depths(args.depths)
    bridge_path = Path(args.bridge).resolve()
    chrono_path = Path(args.chrono).resolve()
    omnifold_path = Path(args.omnifold).resolve()
    lib_path = Path(args.lib).resolve()
    bridge = read_json(bridge_path)
    chrono = read_json(chrono_path)
    omnifold = read_json(omnifold_path)
    rows = bridge.get("rows") or []
    roots = root_fen_map(rows)
    depth_labels = run_depth_labels(RefCuda(lib_path), roots, depths, args.movetime_ms)
    stable_labels = stable_root_labels(depth_labels, depths)
    items, missing = join_items(rows, chrono_rows_by_hash(chrono), stable_labels)
    stable_root_count = sum(1 for label in stable_labels.values() if label["status"] == "stable")
    groups = group_by_root([item for item in items if item["stableRoot"]])
    baseline = evaluate_frontier(groups)
    motifs = generate_motifs([item for item in items if item["stableRoot"]], args.max_size, args.min_roots, args.min_rows)
    top_patterns = rank_motifs(motifs, groups, baseline, args.top_k, len(items))
    cv = cross_validate([item for item in items if item["stableRoot"]], groups, args) if groups else None
    active_families = [
        family for family in (omnifold.get("foldFamilies") or [])
        if family.get("status") == "active_frontier_attachable"
    ]
    output = {
        "schemaVersion": SCHEMA_VERSION,
        "generatedAt": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "condition": {
            "source": str(Path(args.condition_source).resolve()),
            "runLabel": "scout_new_experiment_chrono_o2_gpu_search_stability_pattern_mining",
            "changedFields": "post-hoc multi-depth GPU search stability labels plus motif mining only; no runtime behavior changed",
            "labCondition": "scout/new_experiment/source_temporal_proxy_when_input_chrono_sidecar_is_source_order_proxy",
            "metric": "root-fold top-k agreement with refcuda GPU bestmove stable across depths",
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
            "joinedRows": len(items),
            "missingChronoRows": len(missing),
        },
        "gpuSearchStabilityLabel": {
            "schemaVersion": "dojo.refcuda_gpu_search_stability_label.v1",
            "depths": depths,
            "movetimeMs": args.movetime_ms,
            "rootCount": len(roots),
            "stableRootCount": stable_root_count,
            "unstableRootCount": len(roots) - stable_root_count,
            "rootLabels": list(stable_labels.values()),
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
            "rootCount": len(roots),
            "stableRootCount": stable_root_count,
            "stableRows": sum(1 for item in items if item["stableRoot"]),
            "stableBestMoveMatchRows": sum(1 for item in items if item["stableBestMoveMatch"]),
            "acceptedRows": sum(1 for item in items if item["accepted"]),
            "selectedRows": sum(1 for item in items if item["selected"]),
        },
        "miningPolicy": {
            "algorithm": "enumerate_tag_conjunctions_then_rank_by_multi_depth_gpu_search_stable_bestmove",
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
        "promotionPolicy": {
            "status": "not_promoted",
            "reason": "multi-depth GPU-search stability motifs are post-hoc source-temporal evidence; promotion requires heldout accepted-injection or fight evidence",
            "blockers": [
                "posthoc_gpu_search_stability_pattern_mining_not_runtime_evidence",
                "source_temporal_proxy_sidecar_used",
                "heldout_gpu_gate_not_rerun_with_frozen_stability_motif",
            ],
        },
    }
    out_path = Path(args.out).resolve() if args.out else chrono_path.with_name(chrono_path.name.replace(".json", ".gpu_search_stability_patterns.json"))
    write_json(out_path, output)
    print(__import__("json").dumps({
        "ok": True,
        "output": str(out_path),
        "depths": depths,
        "rootCount": len(roots),
        "stableRootCount": stable_root_count,
        "candidateMotifs": len(motifs),
        "baselineTop1Stable": baseline["top1StableBestMoveMatches"],
        "crossValidatedTop1Stable": cv["candidate"]["top1StableBestMoveMatches"] if cv else None,
        "crossValidatedDelta": cv["deltaVsFrontierRank"] if cv else None,
        "observedLift": cv["observedLift"] if cv else False,
        "promote": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
