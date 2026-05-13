#!/usr/bin/env python3
"""Evaluate a frozen chrono/O2 motif on heldout roots with GPU search labels."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from mine_chrono_o2_gpu_search_patterns import (
    DEFAULT_CONDITION,
    DEFAULT_LIB,
    RefCuda,
    chrono_rows_by_hash,
    complete_fen,
    evaluate_frontier,
    evaluate_motif,
    filter_groups,
    group_by_root,
    joined_items,
    metric_delta,
    public_metrics,
    read_json,
    root_records,
    round6,
    run_gpu_search_labels,
    sha256_file,
    write_json,
)


SCHEMA_VERSION = "dojo.frozen_chrono_o2_gpu_search_motif_heldout_eval.v1"


def root_id_set(bridge_rows: list[dict[str, Any]]) -> set[str]:
    return {str(row["rootId"]) for row in root_records(bridge_rows)}


def evaluate_accepted(groups: list[tuple[str, list[dict[str, Any]]]], score_fn, name: str) -> dict[str, Any]:
    roots: list[dict[str, Any]] = []
    top1 = top3 = selected_top1 = gpu_best_top1 = 0
    accepted_rank_sum = accepted_rank_count = 0
    for root_id, rows in groups:
        ranked = sorted(
            [{"item": item, "score": score_fn(item)} for item in rows],
            key=lambda entry: (-entry["score"], entry["item"]["originalRank"]),
        )
        for index, entry in enumerate(ranked, start=1):
            entry["comparisonRank"] = index
        first_accepted = next((entry for entry in ranked if entry["item"]["accepted"]), None)
        if ranked and ranked[0]["item"]["accepted"]:
            top1 += 1
        if any(entry["item"]["accepted"] for entry in ranked[:3]):
            top3 += 1
        if ranked and ranked[0]["item"]["selected"]:
            selected_top1 += 1
        if ranked and ranked[0]["item"]["gpuBestMoveMatch"]:
            gpu_best_top1 += 1
        if first_accepted:
            accepted_rank_sum += first_accepted["comparisonRank"]
            accepted_rank_count += 1
        roots.append({
            "rootId": root_id,
            "rowCount": len(rows),
            "top1": public_ranked(ranked[0]) if ranked else None,
            "acceptedBestRank": first_accepted["comparisonRank"] if first_accepted else None,
        })
    return {
        "name": name,
        "rootCount": len(groups),
        "top1AcceptedUsefulInjections": top1,
        "top3AcceptedUsefulInjections": top3,
        "selectedMoveTop1": selected_top1,
        "gpuBestMoveTop1": gpu_best_top1,
        "acceptedRankSum": accepted_rank_sum,
        "acceptedRankCount": accepted_rank_count,
        "meanAcceptedCandidateRank": round6(accepted_rank_sum / accepted_rank_count) if accepted_rank_count else None,
        "roots": roots,
    }


def public_ranked(entry: dict[str, Any]) -> dict[str, Any]:
    item = entry["item"]
    return {
        "bridgeId": item.get("bridgeId"),
        "hash": item.get("hash"),
        "move": item.get("move"),
        "originalRank": item.get("originalRank"),
        "comparisonRank": entry.get("comparisonRank"),
        "score": round6(entry.get("score")),
        "gpuBestMove": item.get("gpuBestMove"),
        "gpuBestMoveMatch": item.get("gpuBestMoveMatch"),
        "acceptedUsefulInjection": item.get("accepted"),
        "selectedMoveInFrontier": item.get("selected"),
    }


def accepted_delta(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    mean_delta = None
    if candidate.get("meanAcceptedCandidateRank") is not None and baseline.get("meanAcceptedCandidateRank") is not None:
        mean_delta = round6(candidate["meanAcceptedCandidateRank"] - baseline["meanAcceptedCandidateRank"])
    return {
        "top1": candidate["top1AcceptedUsefulInjections"] - baseline["top1AcceptedUsefulInjections"],
        "top3": candidate["top3AcceptedUsefulInjections"] - baseline["top3AcceptedUsefulInjections"],
        "meanAcceptedCandidateRank": mean_delta,
    }


def public_accepted(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "rootCount": metrics["rootCount"],
        "top1AcceptedUsefulInjections": metrics["top1AcceptedUsefulInjections"],
        "top3AcceptedUsefulInjections": metrics["top3AcceptedUsefulInjections"],
        "selectedMoveTop1": metrics["selectedMoveTop1"],
        "gpuBestMoveTop1": metrics["gpuBestMoveTop1"],
        "meanAcceptedCandidateRank": metrics["meanAcceptedCandidateRank"],
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bridge", required=True)
    parser.add_argument("--chrono", required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--out")
    parser.add_argument("--lib", default=str(DEFAULT_LIB))
    parser.add_argument("--condition-source", default=str(DEFAULT_CONDITION))
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    bridge_path = Path(args.bridge).resolve()
    chrono_path = Path(args.chrono).resolve()
    condition_path = Path(args.condition).resolve()
    lib_path = Path(args.lib).resolve()
    condition = read_json(condition_path)
    bridge = read_json(bridge_path)
    chrono = read_json(chrono_path)
    train_bridge_path = Path(condition["sourceCondition"]["sourceBridgePath"]).resolve()
    train_bridge = read_json(train_bridge_path)
    full_rows = bridge.get("rows") or []
    train_roots = root_id_set(train_bridge.get("rows") or [])
    full_roots = root_id_set(full_rows)
    heldout_roots = sorted(full_roots - train_roots)
    overlap_roots = sorted(full_roots & train_roots)
    heldout_rows = [
        row for row in full_rows
        if str((row.get("logicRayFrontier") or (row.get("pzrgCandidate") or {}).get("logicRayFrontier") or {}).get("rootId") or row.get("rootId")) in heldout_roots
    ]
    depth = int(condition["sourceCondition"].get("gpuSearchDepth") or 4)
    movetime_ms = int(condition["sourceCondition"].get("gpuSearchMovetimeMs") or 0)
    labels = run_gpu_search_labels(RefCuda(lib_path), [{"rootId": root_id, "rootFen": fen} for root_id, fen in sorted({
        str((row.get("logicRayFrontier") or (row.get("pzrgCandidate") or {}).get("logicRayFrontier") or {}).get("rootId") or row.get("rootId")):
        complete_fen(str((row.get("logicRayFrontier") or (row.get("pzrgCandidate") or {}).get("logicRayFrontier") or {}).get("rootFen") or row.get("rootFen")))
        for row in heldout_rows
    }.items())], depth, movetime_ms)
    items, missing = joined_items(heldout_rows, chrono_rows_by_hash(chrono), labels)
    groups = group_by_root(items)
    motif = {
        "id": condition["frozenMotif"]["id"],
        "tags": tuple(condition["frozenMotif"]["tags"]),
    }
    baseline_gpu = evaluate_frontier(groups)
    motif_gpu = evaluate_motif(motif, groups)
    baseline_acc = evaluate_accepted(groups, lambda item: -item["originalRank"], "frontier_rank_accepted")
    motif_acc = evaluate_accepted(groups, lambda item: 1 if all(tag in item["tagSet"] for tag in motif["tags"]) else 0, "frozen_motif_accepted")
    gpu_delta = metric_delta(motif_gpu, baseline_gpu)
    acc_delta = accepted_delta(motif_acc, baseline_acc)
    search_ok = sum(1 for label in labels.values() if label.get("status") == "ok")
    output = {
        "schemaVersion": SCHEMA_VERSION,
        "generatedAt": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "condition": {
            "source": str(Path(args.condition_source).resolve()),
            "runLabel": "heldout_gpu_search_probe_for_frozen_chrono_o2_motif",
            "changedFields": "heldout evaluation of frozen motif condition only; no runtime behavior changed",
            "labCondition": "heldout_probe_over_roots_excluded_from_frozen_condition_source",
            "metric": "heldout top-k agreement with fixed-depth refcuda GPU search bestmove plus accepted useful injection rank",
        },
        "sources": {
            "bridgePath": str(bridge_path),
            "bridgeSha256": sha256_file(bridge_path),
            "chronoPath": str(chrono_path),
            "chronoSha256": sha256_file(chrono_path),
            "conditionPath": str(condition_path),
            "conditionSha256": sha256_file(condition_path),
            "libRefCudaPath": str(lib_path),
            "libRefCudaSha256": sha256_file(lib_path),
            "trainBridgePath": str(train_bridge_path),
            "trainBridgeSha256": sha256_file(train_bridge_path),
        },
        "split": {
            "fullRootCount": len(full_roots),
            "trainRootCount": len(train_roots),
            "heldoutRootCount": len(heldout_roots),
            "overlapRootCount": len(overlap_roots),
            "heldoutRoots": heldout_roots,
            "trainRootsPresentInFull": overlap_roots,
            "heldoutRows": len(heldout_rows),
            "joinedRows": len(items),
            "missingChronoRows": len(missing),
        },
        "frozenCondition": {
            "conditionId": condition["conditionId"],
            "conditionHash": condition["conditionHash"],
            "motifId": motif["id"],
            "tags": list(motif["tags"]),
            "sourceObservedEvidence": condition.get("observedEvidence"),
        },
        "gpuSearchLabel": {
            "schemaVersion": "dojo.refcuda_gpu_search_bestmove_label.v1",
            "depth": depth,
            "movetimeMs": movetime_ms,
            "rootCount": len(heldout_roots),
            "searchOkRoots": search_ok,
            "searchErrorRoots": len(heldout_roots) - search_ok,
            "rootLabels": list(labels.values()),
        },
        "heldoutGpuBestMetrics": {
            "baseline": public_metrics(baseline_gpu),
            "candidate": public_metrics(motif_gpu),
            "deltaVsFrontierRank": gpu_delta,
            "observedLift": (
                gpu_delta["top1"] > 0
                or gpu_delta["top3"] > 0
                or (gpu_delta["meanGpuBestMoveRank"] is not None and gpu_delta["meanGpuBestMoveRank"] < 0)
            ),
        },
        "heldoutAcceptedInjectionMetrics": {
            "baseline": public_accepted(baseline_acc),
            "candidate": public_accepted(motif_acc),
            "deltaVsFrontierRank": acc_delta,
            "observedLift": (
                acc_delta["top1"] > 0
                or acc_delta["top3"] > 0
                or (acc_delta["meanAcceptedCandidateRank"] is not None and acc_delta["meanAcceptedCandidateRank"] < 0)
            ),
        },
        "promotionPolicy": {
            "status": "not_promoted",
            "reason": "heldout probe does not by itself promote runtime behavior; accepted useful injection lift and fight/gate stability must both be judged before promotion",
            "blockers": [
                "runtime_not_changed",
                "fight_loop_not_run",
                "trained_omnifold_delta_not_measured",
            ],
        },
    }
    out_path = Path(args.out).resolve() if args.out else condition_path.with_name(condition_path.name.replace(".json", ".heldout_eval.json"))
    write_json(out_path, output)
    print(json.dumps({
        "ok": True,
        "output": str(out_path),
        "conditionHash": condition["conditionHash"],
        "heldoutRootCount": len(heldout_roots),
        "searchOkRoots": search_ok,
        "gpuBestDelta": gpu_delta,
        "acceptedDelta": acc_delta,
        "gpuBestObservedLift": output["heldoutGpuBestMetrics"]["observedLift"],
        "acceptedObservedLift": output["heldoutAcceptedInjectionMetrics"]["observedLift"],
        "promote": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
