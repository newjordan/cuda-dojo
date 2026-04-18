#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = ROOT / "crawler_seed_manifest.json"
REPORT_PATH = ROOT / "crawler_launch_probe_report.json"


def load_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def run_probe(candidate: dict) -> dict:
    if candidate["runner_mode"] != "script":
        return {
            "name": candidate["name"],
            "runner_mode": candidate["runner_mode"],
            "skipped": True,
            "reason": "env_bundle_not_launchable_by_probe",
        }

    cmd = [sys.executable, candidate["runner_path"], *candidate.get("args", []), "--dry-run"]
    started = time.time()
    completed = subprocess.run(
        cmd,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    stdout = completed.stdout.strip().splitlines()
    stderr = completed.stderr.strip().splitlines()
    return {
        "name": candidate["name"],
        "runner_mode": candidate["runner_mode"],
        "role": candidate["role"],
        "battery_profile": candidate["battery_profile"],
        "returncode": completed.returncode,
        "elapsed_sec": round(time.time() - started, 2),
        "ok": completed.returncode == 0,
        "command": cmd,
        "stdout_tail": stdout[-12:],
        "stderr_tail": stderr[-12:],
    }


def main() -> int:
    manifest = load_manifest()
    results = [run_probe(candidate) for candidate in manifest.get("candidates", [])]
    payload = {
        "kind": "crawler_launch_probe",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "manifest": str(MANIFEST_PATH),
        "results": results,
        "ok_count": sum(1 for item in results if item.get("ok")),
        "skipped_count": sum(1 for item in results if item.get("skipped")),
        "failed_count": sum(1 for item in results if (not item.get("ok")) and (not item.get("skipped"))),
    }
    REPORT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {REPORT_PATH}")
    print(
        f"ok={payload['ok_count']} skipped={payload['skipped_count']} failed={payload['failed_count']}"
    )
    return 0 if payload["failed_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
