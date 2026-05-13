#!/usr/bin/env python3
"""GPU-backed source temporal transition verifier.

This script is host-side orchestration only. It uses cuda/engine/librefcuda.so
for legal move enumeration and make_move on GPU, then compares the resulting
Position to the next recorded FEN through the refcuda semantic comparator.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "dojo.source_temporal_transition_gpu_verify.v1"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LIB = REPO_ROOT / "cuda" / "engine" / "librefcuda.so"
DEFAULT_SOURCE = REPO_ROOT / "gpu_spine" / "book.jsonl"
DEFAULT_CONDITION = Path("/home/frosty40/cuda_dojo/LOGIC_RAY_FRONTIER_RUNTIME_RECEIPT_2026-05-06.md")
MAX_MOVES = 256


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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


def fen_hash(fen: str) -> str:
    return hashlib.sha256(complete_fen(fen).encode("utf-8")).hexdigest()


class RefCuda:
    def __init__(self, lib_path: Path):
        self.lib_path = lib_path
        self.lib = ctypes.CDLL(str(lib_path))
        self.lib.refc_position_size.restype = ctypes.c_int
        self.lib.refc_parse_fen.argtypes = [ctypes.c_char_p, ctypes.c_void_p]
        self.lib.refc_parse_fen.restype = ctypes.c_int
        self.lib.refc_legal_moves.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_int]
        self.lib.refc_legal_moves.restype = ctypes.c_int
        self.lib.refc_make_move.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_void_p]
        self.lib.refc_make_move.restype = ctypes.c_int
        self.lib.refc_move_to_uci.argtypes = [ctypes.c_int32, ctypes.c_char_p]
        self.lib.refc_move_to_uci.restype = None
        self.lib.refc_position_equal_semantic.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
        self.lib.refc_position_equal_semantic.restype = ctypes.c_int
        self.position_size = int(self.lib.refc_position_size())

    def position(self, fen: str) -> ctypes.Array[ctypes.c_char]:
        buf = ctypes.create_string_buffer(self.position_size)
        rc = self.lib.refc_parse_fen(complete_fen(fen).encode("utf-8"), buf)
        if rc != 0:
            raise ValueError(f"refc_parse_fen failed rc={rc}")
        return buf

    def legal_moves(self, pos: ctypes.Array[ctypes.c_char]) -> list[tuple[int, str]]:
        moves = (ctypes.c_int32 * MAX_MOVES)()
        n = int(self.lib.refc_legal_moves(pos, moves, MAX_MOVES))
        if n < 0:
            raise RuntimeError(f"refc_legal_moves failed rc={n}")
        result: list[tuple[int, str]] = []
        for idx in range(n):
            out = ctypes.create_string_buffer(8)
            self.lib.refc_move_to_uci(ctypes.c_int32(moves[idx]), out)
            result.append((int(moves[idx]), out.value.decode("ascii")))
        return result

    def make_move(self, pos: ctypes.Array[ctypes.c_char], move: int) -> ctypes.Array[ctypes.c_char]:
        out = ctypes.create_string_buffer(self.position_size)
        rc = int(self.lib.refc_make_move(pos, ctypes.c_int32(move), out))
        if rc != 0:
            raise RuntimeError(f"refc_make_move failed rc={rc}")
        return out

    def equal_semantic(self, a: ctypes.Array[ctypes.c_char], b: ctypes.Array[ctypes.c_char], include_clocks: bool = False) -> bool:
        rc = int(self.lib.refc_position_equal_semantic(a, b, 1 if include_clocks else 0))
        if rc < 0:
            raise RuntimeError(f"refc_position_equal_semantic failed rc={rc}")
        return rc == 1


def read_source_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        fen = row.get("fen")
        move = row.get("sf_bestmove")
        if not fen or not move or move == "(none)":
            continue
        rows.append({
            "lineNumber": line_number,
            "sourceIndex": len(rows),
            "fen": complete_fen(fen),
            "move": str(move),
            "scoreCp": row.get("sf_score_cp"),
            "depth": row.get("sf_depth"),
        })
    return rows


def root_temporal_rows(bridge_path: Path | None) -> list[dict[str, Any]]:
    if bridge_path is None:
        return []
    bridge = read_json(bridge_path)
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for row in bridge.get("rows", []):
        frontier = row.get("logicRayFrontier") or (row.get("pzrgCandidate") or {}).get("logicRayFrontier") or {}
        source_temporal = frontier.get("sourceTemporal") or {}
        root_id = frontier.get("rootId") or row.get("rootId")
        source_index = source_temporal.get("sourceIndex")
        if root_id is None or source_index is None:
            continue
        key = (root_id, int(source_index))
        if key in seen:
            continue
        seen.add(key)
        result.append({
            "rootId": root_id,
            "rootFen": frontier.get("rootFen") or row.get("rootFen"),
            "sourceIndex": int(source_index),
            "sourcePly": source_temporal.get("sourcePly"),
            "sourceSequenceId": source_temporal.get("sourceSequenceId"),
            "previousFenHash": source_temporal.get("previousFenHash"),
            "nextFenHash": source_temporal.get("nextFenHash"),
        })
    return sorted(result, key=lambda item: (item["sourceIndex"], item["rootId"]))


def verify_edges(refc: RefCuda, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    for idx in range(len(rows) - 1):
        current = rows[idx]
        nxt = rows[idx + 1]
        edge = {
            "fromSourceIndex": current["sourceIndex"],
            "toSourceIndex": nxt["sourceIndex"],
            "fromLineNumber": current["lineNumber"],
            "toLineNumber": nxt["lineNumber"],
            "fromFenHash": fen_hash(current["fen"]),
            "toFenHash": fen_hash(nxt["fen"]),
            "sourceMove": current["move"],
            "sourceDepth": current.get("depth"),
            "sourceScoreCp": current.get("scoreCp"),
            "gpuLegalMove": False,
            "gpuMoveApplied": False,
            "semanticMatchNoClocks": False,
            "semanticMatchWithClocks": False,
            "status": "unchecked",
            "legalCount": 0,
        }
        try:
            pos = refc.position(current["fen"])
            next_pos = refc.position(nxt["fen"])
            legal = refc.legal_moves(pos)
            edge["legalCount"] = len(legal)
            move = next((encoded for encoded, uci in legal if uci == current["move"]), None)
            if move is None:
                edge["status"] = "source_move_not_gpu_legal"
            else:
                edge["gpuLegalMove"] = True
                applied = refc.make_move(pos, move)
                edge["gpuMoveApplied"] = True
                edge["semanticMatchNoClocks"] = refc.equal_semantic(applied, next_pos, include_clocks=False)
                edge["semanticMatchWithClocks"] = refc.equal_semantic(applied, next_pos, include_clocks=True)
                edge["status"] = "verified_transition" if edge["semanticMatchNoClocks"] else "gpu_applied_move_does_not_match_next_fen"
        except Exception as exc:  # noqa: BLE001 - include error in artifact
            edge["status"] = "verification_error"
            edge["error"] = str(exc)
        edges.append(edge)
    return edges


def edge_by_pair(edges: list[dict[str, Any]]) -> dict[tuple[int, int], dict[str, Any]]:
    return {(int(edge["fromSourceIndex"]), int(edge["toSourceIndex"])): edge for edge in edges}


def build_root_report(roots: list[dict[str, Any]], edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_pair = edge_by_pair(edges)
    result: list[dict[str, Any]] = []
    for root in roots:
        source_index = int(root["sourceIndex"])
        previous_edge = by_pair.get((source_index - 1, source_index))
        next_edge = by_pair.get((source_index, source_index + 1))
        previous_ok = bool(previous_edge and previous_edge.get("semanticMatchNoClocks"))
        next_ok = bool(next_edge and next_edge.get("semanticMatchNoClocks"))
        result.append({
            **root,
            "gpuVerifiedPreviousTransition": previous_ok,
            "gpuVerifiedNextTransition": next_ok,
            "gpuVerifiedAnyNeighbor": previous_ok or next_ok,
            "gpuVerifiedBothNeighbors": previous_ok and next_ok,
            "previousEdgeStatus": previous_edge.get("status") if previous_edge else "missing_source_edge",
            "nextEdgeStatus": next_edge.get("status") if next_edge else "missing_source_edge",
        })
    return result


def count_by(items: list[Any], key_fn) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        key = str(key_fn(item))
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--bridge", default=None)
    parser.add_argument("--lib", default=str(DEFAULT_LIB))
    parser.add_argument("--out", required=True)
    parser.add_argument("--condition-source", default=str(DEFAULT_CONDITION))
    args = parser.parse_args()

    source_path = Path(args.source).resolve()
    bridge_path = Path(args.bridge).resolve() if args.bridge else None
    lib_path = Path(args.lib).resolve()
    out_path = Path(args.out).resolve()
    if not lib_path.exists():
        raise SystemExit(f"missing librefcuda.so at {lib_path}; run make -C cuda/engine librefcuda")

    source_rows = read_source_rows(source_path)
    refc = RefCuda(lib_path)
    edges = verify_edges(refc, source_rows)
    root_rows = root_temporal_rows(bridge_path)
    root_report = build_root_report(root_rows, edges)
    verified_edges = [edge for edge in edges if edge.get("semanticMatchNoClocks")]
    verified_roots_any = [row for row in root_report if row["gpuVerifiedAnyNeighbor"]]
    verified_roots_both = [row for row in root_report if row["gpuVerifiedBothNeighbors"]]
    output = {
        "schemaVersion": SCHEMA_VERSION,
        "generatedAt": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "condition": {
            "source": str(Path(args.condition_source).resolve()),
            "runLabel": "gpu_refcuda_source_transition_verify",
            "changedFields": "none; GPU-backed verifier over recorded source rows",
            "hostRole": "orchestration_only_no_cpu_chess_rules",
        },
        "sources": {
            "sourcePath": str(source_path),
            "sourceSha256": sha256_file(source_path),
            "bridgePath": str(bridge_path) if bridge_path else None,
            "bridgeSha256": sha256_file(bridge_path) if bridge_path else None,
            "refcudaLibPath": str(lib_path),
            "refcudaLibSha256": sha256_file(lib_path),
            "positionSize": refc.position_size,
        },
        "summary": {
            "sourceRows": len(source_rows),
            "transitionEdges": len(edges),
            "gpuLegalEdges": sum(1 for edge in edges if edge.get("gpuLegalMove")),
            "gpuVerifiedEdgesNoClocks": len(verified_edges),
            "gpuVerifiedEdgesWithClocks": sum(1 for edge in edges if edge.get("semanticMatchWithClocks")),
            "edgeStatusCounts": count_by(edges, lambda edge: edge.get("status")),
            "bridgeRootCount": len(root_report),
            "bridgeRootsWithGpuVerifiedAnyNeighbor": len(verified_roots_any),
            "bridgeRootsWithGpuVerifiedBothNeighbors": len(verified_roots_both),
        },
        "promotionPolicy": {
            "status": "not_promoted",
            "reason": "transition verification evidence only; source rows are upgraded only when GPU-applied source moves match adjacent FENs",
        },
        "rootTransitionVerification": root_report,
        "transitionEdges": edges,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "ok": True,
        "output": str(out_path),
        **output["summary"],
        "promote": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
