#!/usr/bin/env python3
"""Validate CUDA Dojo logicRayFrontier rows against the canonical schema.

This is a host-side contract check only. It does not parse chess positions,
generate legal moves, or repair runtime evidence.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCHEMA = REPO_ROOT / "schemas" / "logic_ray_frontier.schema.json"


def load_json(path: str | None) -> Any:
    if path in (None, "-"):
        return json.loads(sys.stdin.read())
    return json.loads(Path(path).read_text(encoding="utf-8"))


def iter_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict) and isinstance(payload.get("rows"), list):
        return [row for row in payload["rows"] if isinstance(row, dict)]
    if isinstance(payload, dict):
        return [payload]
    return []


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path", nargs="?", default="-")
    parser.add_argument("--schema", default=str(DEFAULT_SCHEMA))
    args = parser.parse_args()

    schema_path = Path(args.schema)
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    payload = load_json(args.json_path)
    rows = iter_rows(payload)
    validator = Draft202012Validator(schema, format_checker=FormatChecker())

    failures: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
      for error in sorted(validator.iter_errors(row), key=lambda item: list(item.path)):
          failures.append({
              "rowIndex": idx,
              "path": "/".join(str(part) for part in error.path),
              "message": error.message,
          })

    summary = {
        "ok": not failures and bool(rows),
        "schema": str(schema_path),
        "rowCount": len(rows),
        "failures": failures,
    }
    print(json.dumps(summary, indent=2))
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
