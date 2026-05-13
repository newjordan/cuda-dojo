#!/usr/bin/env python3
"""Mine chrono/O2 tactical motifs against fixed-depth GPU search labels.

The host code loads JSON, launches refcuda GPU search once per root FEN, and
mines tag conjunctions over recorded sidecar rows. It does not generate legal
moves, run a CPU chess library, or promote runtime behavior.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import itertools
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "dojo.chrono_o2_gpu_search_pattern_mining.v1"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LIB = REPO_ROOT / "cuda" / "engine" / "librefcuda.so"
DEFAULT_CONDITION = Path("/home/frosty40/cuda_dojo/LOGIC_RAY_FRONTIER_RUNTIME_RECEIPT_2026-05-06.md")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def complete_fen(fen: str) -> str:
    parts = str(fen or "").strip().split()
    if len(parts) == 2:
        return " ".join(parts + ["-", "-", "0", "1"])
    if len(parts) == 3:
        return " ".join(parts + ["-", "0", "1"])
    if len(parts) == 4:
        return " ".join(parts + ["0", "1"])
    if len(parts) == 5:
        return " ".join(parts + ["1"])
    return " ".join(parts)


def round6(value: float | int | None) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return round(number, 6)


def as_number(value: Any, fallback: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    return number if math.isfinite(number) else fallback


def default_out_path(chrono_path: Path, depth: int) -> Path:
    return chrono_path.with_name(chrono_path.name.replace(".json", f".gpu_search_depth{depth}_patterns.json"))


class RefCuda:
    def __init__(self, lib_path: Path):
        self.lib_path = lib_path
        self.lib = ctypes.CDLL(str(lib_path))
        self.lib.refc_position_size.restype = ctypes.c_int
        self.lib.refc_parse_fen.argtypes = [ctypes.c_char_p, ctypes.c_void_p]
        self.lib.refc_parse_fen.restype = ctypes.c_int
        self.lib.refc_move_to_uci.argtypes = [ctypes.c_int32, ctypes.c_char_p]
        self.lib.refc_move_to_uci.restype = None
        self.lib.refc_search_init.argtypes = []
        self.lib.refc_search_init.restype = None
        self.lib.refc_search_shutdown.argtypes = []
        self.lib.refc_search_shutdown.restype = None
        self.lib.refc_search_best_move.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int),
        ]
        self.lib.refc_search_best_move.restype = ctypes.c_int
        self.position_size = int(self.lib.refc_position_size())
        if self.position_size <= 0:
            raise RuntimeError(f"invalid refcuda Position size: {self.position_size}")

    def position(self, fen: str) -> ctypes.Array[ctypes.c_char]:
        buf = ctypes.create_string_buffer(self.position_size)
        rc = int(self.lib.refc_parse_fen(complete_fen(fen).encode("utf-8"), buf))
        if rc != 0:
            raise ValueError(f"refc_parse_fen failed rc={rc}")
        return buf

    def move_to_uci(self, move: int) -> str:
        out = ctypes.create_string_buffer(8)
        self.lib.refc_move_to_uci(ctypes.c_int32(move), out)
        return out.value.decode("ascii")

    def search_best(self, fen: str, depth: int, movetime_ms: int) -> dict[str, Any]:
        pos = self.position(fen)
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
        return {
            "ok": True,
            "returnCode": rc,
            "bestMoveEncoded": int(move_out.value),
            "bestMove": self.move_to_uci(int(move_out.value)),
            "scoreCp": int(score_out.value),
        }


def frontier_of(row: dict[str, Any]) -> dict[str, Any]:
    return row.get("logicRayFrontier") or (row.get("pzrgCandidate") or {}).get("logicRayFrontier") or {}


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


def add_tag(tags: set[str], tag: str, enabled: bool = True) -> None:
    if enabled:
        tags.add(tag)


def tag_safe(value: Any) -> str:
    text = str(value or "unknown")
    return "".join(ch if ch.isalnum() or ch in "_.:-" else "_" for ch in text)


def bucket_replies(value: Any) -> str:
    replies = as_number(value)
    if replies <= 20:
        return "reply_low"
    if replies <= 36:
        return "reply_mid"
    return "reply_high"


def bucket_balance(value: Any) -> str:
    balance = as_number(value)
    if balance > 0:
        return "contact_positive"
    if balance < 0:
        return "contact_negative"
    return "contact_neutral"


def row_tags(bridge_row: dict[str, Any], chrono_row: dict[str, Any]) -> list[str]:
    frontier = frontier_of(bridge_row)
    action = chrono_row.get("actionFeatures") or {}
    action_flags = action.get("flags") or {}
    action_scalars = action.get("scalars") or {}
    tactical = chrono_row.get("tacticalContact") or {}
    tactical_flags = tactical.get("flags") or {}
    counts = tactical.get("counts") or {}
    diagnostics = chrono_row.get("diagnostics") or {}
    tags: set[str] = set()

    add_tag(tags, f"piece:{tag_safe(action.get('piece'))}", bool(action.get("piece")))
    add_tag(tags, f"family:{tag_safe(action.get('family'))}", bool(action.get("family")))
    captured = action.get("capturedPiece")
    add_tag(tags, f"captured:{tag_safe(captured)}", bool(captured and captured != "empty"))
    add_tag(tags, "action:capture", bool(action_flags.get("capture")))
    add_tag(tags, "action:promotion", bool(action_flags.get("promotion")))
    add_tag(tags, "action:diagonal", bool(action_flags.get("diagonal")))
    add_tag(tags, "action:center16", bool(action_flags.get("center16")))
    add_tag(tags, "action:forward", bool(action_flags.get("forwardPositive")))
    add_tag(tags, "action:backward_or_level", bool(action_flags.get("backwardOrLevel")))
    add_tag(tags, "action:home_departure", bool(action_flags.get("homeDeparture")))
    add_tag(tags, "action:same_file", bool(action_flags.get("sameFile")))
    add_tag(tags, "action:same_rank", bool(action_flags.get("sameRank")))
    add_tag(tags, "action:long_move", as_number(action_scalars.get("distance")) >= 2.5)

    add_tag(tags, "tactical:gpu_verified", tactical.get("gpuVerified") is True)
    add_tag(tags, "tactical:defended_before", bool(tactical_flags.get("destinationDefendedBefore")))
    add_tag(tags, "tactical:attacked_before", bool(tactical_flags.get("destinationAttackedBefore")))
    add_tag(tags, "tactical:defended_after", bool(tactical_flags.get("destinationDefendedAfter")))
    add_tag(tags, "tactical:attacked_after", bool(tactical_flags.get("destinationAttackedAfter")))
    add_tag(tags, "tactical:safe_after", bool(tactical_flags.get("destinationDefendedAfter") and not tactical_flags.get("destinationAttackedAfter")))
    add_tag(tags, "tactical:loose_after", bool(tactical_flags.get("destinationAttackedAfter") and not tactical_flags.get("destinationDefendedAfter")))
    add_tag(tags, "tactical:gives_check", bool(tactical_flags.get("givesCheckAfter")))
    add_tag(tags, "tactical:capture_like", bool(tactical_flags.get("captureLike")))
    add_tag(tags, "tactical:defended_capture", bool(tactical_flags.get("captureLike") and tactical_flags.get("destinationDefendedAfter")))
    add_tag(tags, "tactical:attacked_capture", bool(tactical_flags.get("captureLike") and tactical_flags.get("destinationAttackedAfter")))
    add_tag(tags, "tactical:safe_capture", bool(tactical_flags.get("captureLike") and tactical_flags.get("destinationDefendedAfter") and not tactical_flags.get("destinationAttackedAfter")))
    add_tag(tags, "tactical:mover_in_check", bool(tactical_flags.get("moverInCheckBefore")))
    add_tag(tags, f"tactical:{bucket_replies(counts.get('opponentLegalRepliesAfter'))}")
    add_tag(tags, f"tactical:{bucket_balance(counts.get('contactBalanceAfter'))}")

    add_tag(tags, f"chrono:phase:{tag_safe((chrono_row.get('timePhase') or {}).get('phase'))}", bool((chrono_row.get("timePhase") or {}).get("phase")))
    add_tag(tags, f"chrono:pressure:{tag_safe((chrono_row.get('pressureDrift') or {}).get('bucket'))}", bool((chrono_row.get("pressureDrift") or {}).get("bucket")))
    add_tag(tags, f"chrono:relation:{tag_safe((chrono_row.get('relationDrift') or {}).get('bucket'))}", bool((chrono_row.get("relationDrift") or {}).get("bucket")))
    add_tag(tags, f"chrono:contortion:{tag_safe((chrono_row.get('pathContortion') or {}).get('bucket'))}", bool((chrono_row.get("pathContortion") or {}).get("bucket")))
    add_tag(tags, f"chrono:uncertainty:{tag_safe((chrono_row.get('uncertainty') or {}).get('bucket'))}", bool((chrono_row.get("uncertainty") or {}).get("bucket")))
    add_tag(tags, "chrono:stable_score", as_number(diagnostics.get("stabilityScore")) >= 0.7)
    add_tag(tags, "chrono:unstable_score", as_number(diagnostics.get("stabilityScore")) < 0.4)

    pzrg = frontier.get("pzrg4d") or {}
    add_tag(tags, f"pzrg:pressure:{tag_safe(pzrg.get('pressure'))}", bool(pzrg.get("pressure")))
    add_tag(tags, f"pzrg:expression:{tag_safe(pzrg.get('chessExpression'))}", bool(pzrg.get("chessExpression")))
    return sorted(tags)


def chrono_rows_by_hash(chrono: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["logicRayFrontierHash"]: row for row in chrono.get("rows", []) if row.get("logicRayFrontierHash")}


def root_records(bridge_rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    seen: dict[str, str] = {}
    for row in bridge_rows:
        frontier = frontier_of(row)
        root_id = frontier.get("rootId") or row.get("rootId") or "root"
        root_fen = frontier.get("rootFen") or row.get("rootFen")
        if root_fen and root_id not in seen:
            seen[str(root_id)] = complete_fen(str(root_fen))
    return [{"rootId": root_id, "rootFen": fen} for root_id, fen in sorted(seen.items())]


def run_gpu_search_labels(refc: RefCuda, roots: list[dict[str, str]], depth: int, movetime_ms: int) -> dict[str, dict[str, Any]]:
    refc.lib.refc_search_init()
    labels: dict[str, dict[str, Any]] = {}
    try:
        for root in roots:
            root_id = root["rootId"]
            try:
                labels[root_id] = {
                    "rootId": root_id,
                    "rootFen": root["rootFen"],
                    "status": "ok",
                    **refc.search_best(root["rootFen"], depth, movetime_ms),
                }
            except Exception as exc:  # noqa: BLE001 - saved in artifact
                labels[root_id] = {
                    "rootId": root_id,
                    "rootFen": root["rootFen"],
                    "status": "error",
                    "error": str(exc),
                }
    finally:
        refc.lib.refc_search_shutdown()
    return labels


def joined_items(bridge_rows: list[dict[str, Any]], chrono_by_hash: dict[str, dict[str, Any]], labels: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
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
        move = str(frontier.get("move") or bridge_row.get("move") or chrono_row.get("move"))
        label = labels.get(root_id, {})
        tags = row_tags(bridge_row, chrono_row)
        items.append({
            "bridgeRow": bridge_row,
            "chronoRow": chrono_row,
            "hash": row_hash,
            "bridgeId": bridge_row.get("bridgeId"),
            "rootId": root_id,
            "move": move,
            "originalRank": max(1, int(as_number(frontier.get("rank") or bridge_row.get("rank") or chrono_row.get("rank"), 1))),
            "accepted": accepted(bridge_row, chrono_row),
            "selected": selected(bridge_row, chrono_row),
            "gpuSearchOk": label.get("status") == "ok",
            "gpuBestMove": label.get("bestMove"),
            "gpuBestMoveMatch": label.get("status") == "ok" and move == label.get("bestMove"),
            "tags": tags,
            "tagSet": set(tags),
        })
    return items, missing


def combinations(tags: list[str], max_size: int) -> list[tuple[str, ...]]:
    result: list[tuple[str, ...]] = []
    for size in range(1, max_size + 1):
        result.extend(itertools.combinations(tags, size))
    return result


def motif_id(tags: tuple[str, ...]) -> str:
    return "motif_" + sha256_text("|".join(tags))[:16]


def generate_motifs(items: list[dict[str, Any]], max_size: int, min_roots: int, min_rows: int) -> list[dict[str, Any]]:
    stats: dict[tuple[str, ...], dict[str, Any]] = {}
    for item in items:
        for combo in combinations(item["tags"], max_size):
            entry = stats.setdefault(combo, {
                "id": motif_id(combo),
                "tags": combo,
                "rows": 0,
                "gpuBestRows": 0,
                "acceptedRows": 0,
                "selectedRows": 0,
                "rootIds": set(),
                "gpuBestRootIds": set(),
                "examples": [],
            })
            entry["rows"] += 1
            entry["rootIds"].add(item["rootId"])
            if item["gpuBestMoveMatch"]:
                entry["gpuBestRows"] += 1
                entry["gpuBestRootIds"].add(item["rootId"])
            if item["accepted"]:
                entry["acceptedRows"] += 1
            if item["selected"]:
                entry["selectedRows"] += 1
            if len(entry["examples"]) < 8:
                entry["examples"].append({
                    "rootId": item["rootId"],
                    "move": item["move"],
                    "originalRank": item["originalRank"],
                    "gpuBestMove": item.get("gpuBestMove"),
                    "gpuBestMoveMatch": item["gpuBestMoveMatch"],
                    "acceptedUsefulInjection": item["accepted"],
                    "selectedMoveInFrontier": item["selected"],
                    "hash": item["hash"],
                })
    motifs = []
    for entry in stats.values():
        if entry["rows"] >= min_rows and len(entry["rootIds"]) >= min_roots:
            motifs.append({
                **entry,
                "rootCount": len(entry["rootIds"]),
                "gpuBestRootCount": len(entry["gpuBestRootIds"]),
            })
    return motifs


def group_by_root(items: list[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        groups[item["rootId"]].append(item)
    return sorted(groups.items())


def filter_groups(groups: list[tuple[str, list[dict[str, Any]]]], root_ids: set[str]) -> list[tuple[str, list[dict[str, Any]]]]:
    return [(root_id, rows) for root_id, rows in groups if root_id in root_ids]


def matches_motif(item: dict[str, Any], motif: dict[str, Any]) -> bool:
    return all(tag in item["tagSet"] for tag in motif["tags"])


def evaluate_ranked_groups(groups: list[tuple[str, list[dict[str, Any]]]], score_fn, name: str) -> dict[str, Any]:
    roots = []
    top1 = top3 = accepted_top1 = selected_top1 = 0
    best_rank_sum = best_rank_count = 0
    for root_id, rows in groups:
        ranked = sorted(
            [{"item": item, "score": score_fn(item)} for item in rows],
            key=lambda entry: (-entry["score"], entry["item"]["originalRank"]),
        )
        for index, entry in enumerate(ranked, start=1):
            entry["comparisonRank"] = index
        first_best = next((entry for entry in ranked if entry["item"]["gpuBestMoveMatch"]), None)
        if ranked and ranked[0]["item"]["gpuBestMoveMatch"]:
            top1 += 1
        if any(entry["item"]["gpuBestMoveMatch"] for entry in ranked[:3]):
            top3 += 1
        if ranked and ranked[0]["item"]["accepted"]:
            accepted_top1 += 1
        if ranked and ranked[0]["item"]["selected"]:
            selected_top1 += 1
        if first_best:
            best_rank_sum += first_best["comparisonRank"]
            best_rank_count += 1
        roots.append({
            "rootId": root_id,
            "rowCount": len(rows),
            "gpuBestMove": rows[0].get("gpuBestMove"),
            "top1": public_ranked(first_best if False else ranked[0]) if ranked else None,
            "gpuBestMoveBestRank": first_best["comparisonRank"] if first_best else None,
        })
    return {
        "name": name,
        "rootCount": len(groups),
        "top1GpuBestMoveMatches": top1,
        "top3GpuBestMoveMatches": top3,
        "acceptedUsefulTop1": accepted_top1,
        "selectedMoveTop1": selected_top1,
        "gpuBestRankSum": best_rank_sum,
        "gpuBestRankCount": best_rank_count,
        "meanGpuBestMoveRank": round6(best_rank_sum / best_rank_count) if best_rank_count else None,
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


def evaluate_frontier(groups: list[tuple[str, list[dict[str, Any]]]]) -> dict[str, Any]:
    return evaluate_ranked_groups(groups, lambda item: -item["originalRank"], "frontier_rank")


def evaluate_motif(motif: dict[str, Any], groups: list[tuple[str, list[dict[str, Any]]]]) -> dict[str, Any]:
    return evaluate_ranked_groups(groups, lambda item: 1 if matches_motif(item, motif) else 0, motif["id"])


def metric_delta(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    mean_delta = None
    if candidate.get("meanGpuBestMoveRank") is not None and baseline.get("meanGpuBestMoveRank") is not None:
        mean_delta = round6(candidate["meanGpuBestMoveRank"] - baseline["meanGpuBestMoveRank"])
    return {
        "top1": candidate["top1GpuBestMoveMatches"] - baseline["top1GpuBestMoveMatches"],
        "top3": candidate["top3GpuBestMoveMatches"] - baseline["top3GpuBestMoveMatches"],
        "meanGpuBestMoveRank": mean_delta,
    }


def objective(metrics: dict[str, Any]) -> float:
    mean_rank = metrics.get("meanGpuBestMoveRank")
    return (
        metrics["top1GpuBestMoveMatches"] * 10000
        + metrics["top3GpuBestMoveMatches"] * 1000
        - as_number(mean_rank, 999) * 10
    )


def public_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "rootCount": metrics["rootCount"],
        "top1GpuBestMoveMatches": metrics["top1GpuBestMoveMatches"],
        "top3GpuBestMoveMatches": metrics["top3GpuBestMoveMatches"],
        "acceptedUsefulTop1": metrics["acceptedUsefulTop1"],
        "selectedMoveTop1": metrics["selectedMoveTop1"],
        "meanGpuBestMoveRank": metrics["meanGpuBestMoveRank"],
    }


def motif_public(motif: dict[str, Any], corpus_rows: int) -> dict[str, Any]:
    return {
        "id": motif["id"],
        "tags": list(motif["tags"]),
        "size": len(motif["tags"]),
        "rowCount": motif["rows"],
        "rootCount": motif["rootCount"],
        "gpuBestRows": motif["gpuBestRows"],
        "gpuBestRootCount": motif["gpuBestRootCount"],
        "gpuBestRate": round6(motif["gpuBestRows"] / motif["rows"]) if motif["rows"] else 0,
        "corpusRowShare": round6(motif["rows"] / corpus_rows) if corpus_rows else 0,
        "acceptedRows": motif["acceptedRows"],
        "selectedRows": motif["selectedRows"],
        "examples": motif["examples"],
    }


def rank_motifs(motifs: list[dict[str, Any]], groups: list[tuple[str, list[dict[str, Any]]]], baseline: dict[str, Any], limit: int, corpus_rows: int) -> list[dict[str, Any]]:
    ranked = []
    for motif in motifs:
        metrics = evaluate_motif(motif, groups)
        ranked.append({
            "motif": motif,
            "metrics": metrics,
            "delta": metric_delta(metrics, baseline),
            "sortScore": objective(metrics) + (motif["gpuBestRows"] / motif["rows"] if motif["rows"] else 0) * 100 + motif["rootCount"],
        })
    ranked.sort(key=lambda item: (-item["sortScore"], item["motif"]["id"]))
    return [{
        **motif_public(entry["motif"], corpus_rows),
        "metrics": public_metrics(entry["metrics"]),
        "deltaVsFrontierRank": entry["delta"],
    } for entry in ranked[:limit]]


def root_folds(root_ids: list[str], folds: int) -> list[list[str]]:
    return [[root_id for idx, root_id in enumerate(root_ids) if idx % folds == fold] for fold in range(folds)]


def aggregate(name: str, metrics_list: list[dict[str, Any]]) -> dict[str, Any]:
    root_count = sum(item["rootCount"] for item in metrics_list)
    top1 = sum(item["top1GpuBestMoveMatches"] for item in metrics_list)
    top3 = sum(item["top3GpuBestMoveMatches"] for item in metrics_list)
    accepted_top1 = sum(item["acceptedUsefulTop1"] for item in metrics_list)
    selected_top1 = sum(item["selectedMoveTop1"] for item in metrics_list)
    rank_sum = sum(item["gpuBestRankSum"] for item in metrics_list)
    rank_count = sum(item["gpuBestRankCount"] for item in metrics_list)
    return {
        "name": name,
        "rootCount": root_count,
        "top1GpuBestMoveMatches": top1,
        "top3GpuBestMoveMatches": top3,
        "acceptedUsefulTop1": accepted_top1,
        "selectedMoveTop1": selected_top1,
        "gpuBestRankSum": rank_sum,
        "gpuBestRankCount": rank_count,
        "meanGpuBestMoveRank": round6(rank_sum / rank_count) if rank_count else None,
        "roots": [root for item in metrics_list for root in item["roots"]],
    }


def select_best_train_motif(motifs: list[dict[str, Any]], train_groups: list[tuple[str, list[dict[str, Any]]]], baseline: dict[str, Any]) -> dict[str, Any] | None:
    scored = []
    for motif in motifs:
        metrics = evaluate_motif(motif, train_groups)
        scored.append({
            "motif": motif,
            "metrics": metrics,
            "delta": metric_delta(metrics, baseline),
            "score": objective(metrics) + motif["gpuBestRows"] * 10 + motif["rootCount"],
        })
    scored.sort(key=lambda item: (-item["score"], item["motif"]["id"]))
    return scored[0] if scored else None


def cross_validate(items: list[dict[str, Any]], all_groups: list[tuple[str, list[dict[str, Any]]]], args: argparse.Namespace) -> dict[str, Any]:
    root_ids = [root_id for root_id, _ in all_groups]
    folds = root_folds(root_ids, min(args.folds, len(root_ids)))
    fold_reports = []
    candidate_evals = []
    baseline_evals = []
    for fold_idx, eval_roots in enumerate(folds):
        eval_set = set(eval_roots)
        train_set = set(root_ids) - eval_set
        train_groups = filter_groups(all_groups, train_set)
        eval_groups = filter_groups(all_groups, eval_set)
        train_items = [item for item in items if item["rootId"] in train_set]
        train_motifs = generate_motifs(train_items, args.max_size, args.min_roots, args.min_rows)
        train_baseline = evaluate_frontier(train_groups)
        eval_baseline = evaluate_frontier(eval_groups)
        selected = select_best_train_motif(train_motifs, train_groups, train_baseline)
        train_candidate = evaluate_motif(selected["motif"], train_groups) if selected else train_baseline
        eval_candidate = evaluate_motif(selected["motif"], eval_groups) if selected else eval_baseline
        candidate_evals.append(eval_candidate)
        baseline_evals.append(eval_baseline)
        fold_reports.append({
            "fold": fold_idx,
            "trainRootCount": len(train_set),
            "evalRootCount": len(eval_set),
            "trainMotifCount": len(train_motifs),
            "selectedMotif": {
                **motif_public(selected["motif"], len(train_items)),
                "trainMetrics": public_metrics(selected["metrics"]),
                "trainDeltaVsFrontierRank": selected["delta"],
            } if selected else None,
            "evalBaseline": public_metrics(eval_baseline),
            "evalCandidate": public_metrics(eval_candidate),
            "evalDeltaVsBaseline": metric_delta(eval_candidate, eval_baseline),
        })
    baseline = aggregate("frontier_rank_cross_validation_baseline", baseline_evals)
    candidate = aggregate("gpu_search_pattern_cross_validation", candidate_evals)
    delta = metric_delta(candidate, baseline)
    observed_lift = (
        delta["top1"] > 0
        or delta["top3"] > 0
        or (delta["meanGpuBestMoveRank"] is not None and delta["meanGpuBestMoveRank"] < 0)
    )
    return {
        "baseline": baseline,
        "candidate": candidate,
        "deltaVsFrontierRank": delta,
        "observedLift": observed_lift,
        "folds": fold_reports,
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bridge", required=True)
    parser.add_argument("--chrono", required=True)
    parser.add_argument("--omnifold", required=True)
    parser.add_argument("--out")
    parser.add_argument("--lib", default=str(DEFAULT_LIB))
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--movetime-ms", type=int, default=0)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--max-size", type=int, default=3)
    parser.add_argument("--min-roots", type=int, default=3)
    parser.add_argument("--min-rows", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=24)
    parser.add_argument("--condition-source", default=str(DEFAULT_CONDITION))
    parser.add_argument("--run-label", default="posthoc_chrono_o2_gpu_search_pattern_mining")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    bridge_path = Path(args.bridge).resolve()
    chrono_path = Path(args.chrono).resolve()
    omnifold_path = Path(args.omnifold).resolve()
    lib_path = Path(args.lib).resolve()
    out_path = Path(args.out).resolve() if args.out else default_out_path(chrono_path, args.depth)
    if args.depth < 1:
        print("--depth must be >= 1", file=sys.stderr)
        return 2
    bridge = read_json(bridge_path)
    chrono = read_json(chrono_path)
    omnifold = read_json(omnifold_path)
    bridge_rows = bridge.get("rows") or []
    refc = RefCuda(lib_path)
    roots = root_records(bridge_rows)
    labels = run_gpu_search_labels(refc, roots, args.depth, args.movetime_ms)
    items, missing = joined_items(bridge_rows, chrono_rows_by_hash(chrono), labels)
    groups = group_by_root(items)
    baseline = evaluate_frontier(groups)
    motifs = generate_motifs(items, args.max_size, args.min_roots, args.min_rows)
    top_patterns = rank_motifs(motifs, groups, baseline, args.top_k, len(items))
    cv = cross_validate(items, groups, args)
    search_ok = sum(1 for label in labels.values() if label.get("status") == "ok")
    gpu_best_rows = sum(1 for item in items if item["gpuBestMoveMatch"])
    active_families = [
        family for family in (omnifold.get("foldFamilies") or [])
        if family.get("status") == "active_frontier_attachable"
    ]
    output = {
        "schemaVersion": SCHEMA_VERSION,
        "generatedAt": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "condition": {
            "source": str(Path(args.condition_source).resolve()),
            "runLabel": args.run_label,
            "changedFields": "post-hoc GPU search labels plus motif mining only; no runtime behavior changed",
            "labCondition": "posthoc/subset when input bridge/chrono is a subset",
            "metric": "root-fold top-k agreement with fixed-depth refcuda GPU search bestmove",
        },
        "sources": {
            "bridgePath": str(bridge_path),
            "bridgeSha256": sha256_file(bridge_path),
            "bridgeRows": len(bridge_rows),
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
        "gpuSearchLabel": {
            "schemaVersion": "dojo.refcuda_gpu_search_bestmove_label.v1",
            "depth": args.depth,
            "movetimeMs": args.movetime_ms,
            "rootCount": len(roots),
            "searchOkRoots": search_ok,
            "searchErrorRoots": len(roots) - search_ok,
            "hostRole": "json_orchestration_and_cuda_ffi_launch_only",
            "gpuWork": "refc_search_best_move depth-limited search_root through librefcuda",
            "rootLabels": list(labels.values()),
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
            "rootCount": len(groups),
            "gpuBestMoveMatchRows": gpu_best_rows,
            "gpuBestMoveMatchRate": round6(gpu_best_rows / len(items)) if items else 0,
            "acceptedRows": sum(1 for item in items if item["accepted"]),
            "selectedRows": sum(1 for item in items if item["selected"]),
        },
        "miningPolicy": {
            "algorithm": "enumerate_tag_conjunctions_then_rank_by_root_group_gpu_search_bestmove_agreement",
            "maxMotifSize": args.max_size,
            "minRoots": args.min_roots,
            "minRows": args.min_rows,
            "candidateMotifs": len(motifs),
            "topK": args.top_k,
            "noRuntimePromotion": True,
            "excludedRuntimeBehavior": [
                "no CPU chess library",
                "no legal move generation on host",
                "no trained OmniFold delta claim",
            ],
        },
        "baseline": {
            "frontierRank": public_metrics(baseline),
        },
        "topGlobalPatterns": top_patterns,
        "crossValidation": {
            "baseline": public_metrics(cv["baseline"]),
            "candidate": public_metrics(cv["candidate"]),
            "deltaVsFrontierRank": cv["deltaVsFrontierRank"],
            "observedLift": cv["observedLift"],
            "folds": cv["folds"],
        },
        "promotionPolicy": {
            "status": "not_promoted",
            "reason": "post-hoc GPU search motif labels require frozen heldout gate evidence before runtime promotion",
            "blockers": [
                "posthoc_gpu_search_pattern_mining_not_runtime_evidence",
                "trained_omnifold_delta_not_measured",
                "heldout_gpu_gate_not_rerun_with_frozen_gpu_search_motif",
            ],
            "requiredNextEvidence": [
                "freeze any lifted motif condition without looking at new gate labels",
                "rerun heldout GPU gate or fight loop with the frozen motif condition",
                "measure accepted useful injection lift and GPU search/fight stability",
            ],
        },
    }
    write_json(out_path, output)
    print(json.dumps({
        "ok": True,
        "output": str(out_path),
        "depth": args.depth,
        "rootCount": len(roots),
        "searchOkRoots": search_ok,
        "joinedRows": len(items),
        "candidateMotifs": len(motifs),
        "baselineTop1GpuBest": baseline["top1GpuBestMoveMatches"],
        "crossValidatedTop1GpuBest": cv["candidate"]["top1GpuBestMoveMatches"],
        "crossValidatedDelta": cv["deltaVsFrontierRank"],
        "observedLift": cv["observedLift"],
        "promote": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
