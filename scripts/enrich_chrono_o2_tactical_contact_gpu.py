#!/usr/bin/env python3
"""GPU-backed tactical-contact enrichment for PZRG_CHRONO_O2 sidecars.

Host code here only loads JSON, matches UCI strings returned by refcuda, and
copies compact CUDA results into the sidecar artifact. Legal moves, attack
maps, checks, and reply counts all come from cuda/engine/librefcuda.so.
"""
from __future__ import annotations

import argparse
import ctypes
import datetime
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "dojo.chrono_o2_gpu_tactical_contact_enrichment.v1"
CONTACT_SCHEMA_VERSION = "dojo.chrono_o2_gpu_tactical_contact.v1"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LIB = REPO_ROOT / "cuda" / "engine" / "librefcuda.so"
DEFAULT_CONDITION = Path("/home/frosty40/cuda_dojo/LOGIC_RAY_FRONTIER_RUNTIME_RECEIPT_2026-05-06.md")
MAX_MOVES = 256
TACTICAL_CONTACT_N = 16


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def default_out_path(chrono_path: Path) -> Path:
    return chrono_path.with_name(chrono_path.name.replace(".json", ".tactical_contact.json"))


def clamp01(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, value))


class RefCuda:
    def __init__(self, lib_path: Path):
        self.lib_path = lib_path
        self.lib = ctypes.CDLL(str(lib_path))
        self.lib.refc_position_size.restype = ctypes.c_int
        self.lib.refc_parse_fen.argtypes = [ctypes.c_char_p, ctypes.c_void_p]
        self.lib.refc_parse_fen.restype = ctypes.c_int
        self.lib.refc_legal_moves.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_int]
        self.lib.refc_legal_moves.restype = ctypes.c_int
        self.lib.refc_move_to_uci.argtypes = [ctypes.c_int32, ctypes.c_char_p]
        self.lib.refc_move_to_uci.restype = None
        self.lib.refc_tactical_contact.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
        ]
        self.lib.refc_tactical_contact.restype = ctypes.c_int
        self.position_size = int(self.lib.refc_position_size())
        if self.position_size <= 0:
            raise RuntimeError(f"invalid refcuda Position size: {self.position_size}")

    def position(self, fen: str) -> ctypes.Array[ctypes.c_char]:
        buf = ctypes.create_string_buffer(self.position_size)
        rc = int(self.lib.refc_parse_fen(complete_fen(fen).encode("utf-8"), buf))
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

    def tactical_contact(self, pos: ctypes.Array[ctypes.c_char], move: int) -> list[int]:
        out = (ctypes.c_int * TACTICAL_CONTACT_N)()
        rc = int(self.lib.refc_tactical_contact(pos, ctypes.c_int32(move), out, TACTICAL_CONTACT_N))
        if rc != 0:
            raise RuntimeError(f"refc_tactical_contact failed rc={rc}")
        return [int(out[idx]) for idx in range(TACTICAL_CONTACT_N)]


class GpuRootCache:
    def __init__(self, refc: RefCuda):
        self.refc = refc
        self._cache: dict[str, tuple[ctypes.Array[ctypes.c_char], dict[str, int], int]] = {}

    def get(self, fen: str) -> tuple[ctypes.Array[ctypes.c_char], dict[str, int], int]:
        key = complete_fen(fen)
        if key not in self._cache:
            pos = self.refc.position(key)
            legal = self.refc.legal_moves(pos)
            by_uci = {uci: encoded for encoded, uci in legal}
            self._cache[key] = (pos, by_uci, len(legal))
        return self._cache[key]


def bool_at(raw: list[int], index: int) -> bool:
    return bool(raw[index])


def contact_object(raw: list[int]) -> dict[str, Any]:
    flags = {
        "destinationDefendedBefore": bool_at(raw, 6),
        "destinationAttackedBefore": bool_at(raw, 7),
        "destinationDefendedAfter": bool_at(raw, 8),
        "destinationAttackedAfter": bool_at(raw, 9),
        "moverInCheckBefore": bool_at(raw, 10),
        "givesCheckAfter": bool_at(raw, 11),
        "captureLike": bool_at(raw, 14),
        "promotion": bool_at(raw, 15),
    }
    replies_after = max(0, int(raw[12]))
    contact_balance = int(raw[13])
    return {
        "schemaVersion": CONTACT_SCHEMA_VERSION,
        "source": "refcuda_gpu_tactical_contact_v1",
        "gpuVerified": True,
        "flags": flags,
        "counts": {
            "opponentLegalRepliesAfter": replies_after,
            "contactBalanceAfter": contact_balance,
        },
        "squares": {
            "from": int(raw[0]),
            "to": int(raw[1]),
        },
        "pieces": {
            "movingPiece": int(raw[2]),
            "capturedPiece": int(raw[3]),
        },
        "contactVector4": [
            1 if flags["destinationDefendedAfter"] else 0,
            1 if flags["destinationAttackedAfter"] else 0,
            1 if flags["givesCheckAfter"] else 0,
            round(clamp01(replies_after / 64.0), 6),
        ],
    }


def enrich_rows(refc: RefCuda, rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cache = GpuRootCache(refc)
    enriched: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    enriched_count = 0
    missing_legal = 0
    parse_errors = 0
    contact_errors = 0
    legal_counts: list[int] = []
    reply_counts: list[int] = []
    balance_counts: list[int] = []
    defended_after = 0
    attacked_after = 0
    gives_check = 0
    capture_like = 0

    for row_index, row in enumerate(rows):
        next_row = dict(row)
        root_fen = str(row.get("rootFen") or "")
        move_uci = str(row.get("move") or "")
        try:
            pos, legal_by_uci, legal_count = cache.get(root_fen)
            legal_counts.append(legal_count)
            encoded = legal_by_uci.get(move_uci)
            if encoded is None:
                missing_legal += 1
                failures.append({
                    "rowIndex": row_index,
                    "rootId": row.get("rootId"),
                    "move": move_uci,
                    "status": "frontier_move_not_returned_by_refcuda_gpu_legal_moves",
                    "legalCount": legal_count,
                })
            else:
                contact = contact_object(refc.tactical_contact(pos, encoded))
                next_row["tacticalContact"] = contact
                enriched_count += 1
                reply_counts.append(contact["counts"]["opponentLegalRepliesAfter"])
                balance_counts.append(contact["counts"]["contactBalanceAfter"])
                defended_after += 1 if contact["flags"]["destinationDefendedAfter"] else 0
                attacked_after += 1 if contact["flags"]["destinationAttackedAfter"] else 0
                gives_check += 1 if contact["flags"]["givesCheckAfter"] else 0
                capture_like += 1 if contact["flags"]["captureLike"] else 0
        except ValueError as exc:
            parse_errors += 1
            failures.append({
                "rowIndex": row_index,
                "rootId": row.get("rootId"),
                "move": move_uci,
                "status": "refcuda_parse_fen_error",
                "error": str(exc),
            })
        except Exception as exc:  # noqa: BLE001 - save exact artifact failure
            contact_errors += 1
            failures.append({
                "rowIndex": row_index,
                "rootId": row.get("rootId"),
                "move": move_uci,
                "status": "refcuda_tactical_contact_error",
                "error": str(exc),
            })
        enriched.append(next_row)

    stats = {
        "schemaVersion": SCHEMA_VERSION,
        "rows": len(rows),
        "enrichedRows": enriched_count,
        "gpuVerifiedRows": enriched_count,
        "missingLegalMoveRows": missing_legal,
        "parseErrorRows": parse_errors,
        "contactErrorRows": contact_errors,
        "uniqueRootFens": len(cache._cache),
        "meanGpuLegalMovesPerRoot": round(sum(legal_counts) / len(legal_counts), 6) if legal_counts else None,
        "meanOpponentLegalRepliesAfter": round(sum(reply_counts) / len(reply_counts), 6) if reply_counts else None,
        "meanContactBalanceAfter": round(sum(balance_counts) / len(balance_counts), 6) if balance_counts else None,
        "destinationDefendedAfterRows": defended_after,
        "destinationAttackedAfterRows": attacked_after,
        "givesCheckAfterRows": gives_check,
        "captureLikeRows": capture_like,
        "failures": failures[:25],
        "failureCount": len(failures),
    }
    return enriched, stats


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chrono", required=True, help="Input PZRG_CHRONO_O2 sidecar bundle")
    parser.add_argument("--out", help="Output enriched sidecar bundle")
    parser.add_argument("--lib", default=str(DEFAULT_LIB), help="Path to librefcuda.so")
    parser.add_argument("--condition-source", default=str(DEFAULT_CONDITION))
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    chrono_path = Path(args.chrono).resolve()
    out_path = Path(args.out).resolve() if args.out else default_out_path(chrono_path)
    lib_path = Path(args.lib).resolve()
    condition_source = Path(args.condition_source).resolve()

    if not chrono_path.exists():
        print(f"missing chrono sidecar: {chrono_path}", file=sys.stderr)
        return 2
    if not lib_path.exists():
        print(f"missing refcuda library: {lib_path}", file=sys.stderr)
        return 2

    payload = read_json(chrono_path)
    rows = payload.get("rows") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        print("chrono sidecar must be a JSON object with rows[]", file=sys.stderr)
        return 2

    refc = RefCuda(lib_path)
    enriched_rows, stats = enrich_rows(refc, rows)
    output = dict(payload)
    output["generatedAt"] = payload.get("generatedAt")
    output["rowCount"] = len(enriched_rows)
    output["rows"] = enriched_rows
    output["tacticalContactEnrichment"] = {
        **stats,
        "generatedAt": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "condition": {
            "source": str(condition_source),
            "runLabel": "posthoc_gpu_tactical_contact_enrichment_for_chrono_o2_sidecar",
            "changedFields": "sidecar rows gain tacticalContact from refcuda GPU legal-move/contact ABI; no runtime behavior changed",
            "labCondition": "posthoc/subset when input sidecar is a subset",
            "metric": "GPU legal move match count and CUDA tactical contact coverage",
        },
        "sources": {
            "chronoPath": str(chrono_path),
            "chronoSha256": sha256_file(chrono_path),
            "libRefCudaPath": str(lib_path),
            "libRefCudaSha256": sha256_file(lib_path),
        },
        "hostRole": "json_orchestration_and_uci_match_only",
        "gpuWork": [
            "refc_parse_fen layout parse into engine Position",
            "refc_legal_moves CUDA legal move enumeration",
            "refc_tactical_contact CUDA attack/check/reply/contact kernel",
        ],
    }
    output["stats"] = {
        **(payload.get("stats") if isinstance(payload.get("stats"), dict) else {}),
        "tacticalContact": stats,
    }
    write_json(out_path, output)
    print(json.dumps({
        "ok": stats["failureCount"] == 0 and stats["enrichedRows"] == len(rows),
        "out": str(out_path),
        "stats": stats,
    }, indent=2))
    return 0 if stats["failureCount"] == 0 and stats["enrichedRows"] == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
