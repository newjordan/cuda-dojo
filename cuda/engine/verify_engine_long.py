#!/usr/bin/env python3
"""
verify_engine_long.py

Longer-run engine verification focused on the UCI engine binary itself.

Checks:
  1. Eval-service runtime probe routed through the engine binary.
  2. Scheduler runtime probe routed through the engine binary.
  3. Search runtime probe routed through the live engine search path.
  4. UCI position sweep across canonical FENs at multiple depths/movetimes.
  5. Paired self-play arena run at fixed depth.
  6. Paired self-play arena run at fixed movetime.
  7. Paired arena run against Stockfish as an external legality / stability
     benchmark.

Artifacts are written under cuda/engine/artifacts/<run_name>/:
  - summary.json
  - scheduler_runtime_probe/report.json
  - position_sweep/position_sweep.json
  - <scenario>/report.json
  - <scenario>/games.jsonl
  - <scenario>/games.pgn
  - <scenario>/stdout.log
  - <scenario>/stderr.log
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import chess


ENGINE_DIR = Path(__file__).resolve().parent
CUDA_DIR = ENGINE_DIR.parent
PROJECT_ROOT = CUDA_DIR.parent
ARENA_PY = CUDA_DIR / "gpu_arena.py"
ENGINE_BIN = ENGINE_DIR / "engine"
DEFAULT_ARTIFACT_ROOT = ENGINE_DIR / "artifacts"
DEFAULT_STOCKFISH = PROJECT_ROOT / "trainers" / "stockfish" / "stockfish_bin"

POSITION_CASES = [
    {
        "name": "startpos",
        "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    },
    {
        "name": "kiwipete",
        "fen": "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    },
    {
        "name": "position3",
        "fen": "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    },
    {
        "name": "position4",
        "fen": "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    },
    {
        "name": "position5",
        "fen": "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    },
    {
        "name": "position6",
        "fen": "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
    },
]


def _mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


class UciSession:
    def __init__(self, engine_bin: Path, stderr_log: Path, hash_mb: int):
        self.stderr_handle = open(stderr_log, "w", encoding="utf-8")
        self.proc = subprocess.Popen(
            [str(engine_bin)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.stderr_handle,
            text=True,
            bufsize=1,
        )
        self._send("uci")
        self._wait_for("uciok", timeout_s=10.0)
        self._send(f"setoption name Hash value {hash_mb}")
        self._send("isready")
        self._wait_for("readyok", timeout_s=10.0)
        self._send("ucinewgame")
        self._send("isready")
        self._wait_for("readyok", timeout_s=10.0)

    def _send(self, cmd: str) -> None:
        if self.proc.stdin is None:
            raise RuntimeError("engine stdin is closed")
        self.proc.stdin.write(cmd + "\n")
        self.proc.stdin.flush()

    def _readline(self, timeout_s: float) -> str:
        if self.proc.stdout is None:
            raise RuntimeError("engine stdout is closed")
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError(f"engine closed stdout (rc={self.proc.poll()})")
        return line.rstrip("\n")

    def _wait_for(self, prefix: str, timeout_s: float) -> str:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            line = self._readline(max(0.1, deadline - time.time()))
            if line.startswith(prefix):
                return line
        raise TimeoutError(f"timed out waiting for {prefix}")

    def search(self, fen: str, *, depth: int | None = None, movetime: int | None = None) -> dict:
        if depth is None and movetime is None:
            raise ValueError("depth or movetime must be set")

        board = chess.Board(fen)
        self._send(f"position fen {fen}")
        if depth is not None:
            self._send(f"go depth {depth}")
            timeout_s = max(10.0, 5.0 * depth)
            mode = {"type": "depth", "value": depth}
        else:
            self._send(f"go movetime {movetime}")
            timeout_s = max(10.0, movetime / 1000.0 + 5.0)
            mode = {"type": "movetime", "value": movetime}

        meta: dict[str, object] = {
            "mode": mode,
            "info_lines": 0,
        }
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            line = self._readline(max(0.1, deadline - time.time()))
            if line.startswith("info "):
                meta["info_lines"] = int(meta["info_lines"]) + 1
                parts = {
                    "depth": r"depth (\d+)",
                    "nodes": r"nodes (\d+)",
                    "nps": r"nps (\d+)",
                    "time_ms": r"time (\d+)",
                    "score_cp": r"score cp (-?\d+)",
                    "score_mate": r"score mate (-?\d+)",
                }
                for key, pattern in parts.items():
                    import re
                    m = re.search(pattern, line)
                    if m:
                        meta[key] = int(m.group(1))
            if line.startswith("bestmove "):
                bestmove = line.split()[1]
                move = chess.Move.from_uci(bestmove)
                legal = move in board.legal_moves
                meta["bestmove"] = bestmove
                meta["legal"] = legal
                return meta
        raise TimeoutError(f"timed out waiting for bestmove for mode={mode}")

    def close(self) -> None:
        try:
            self._send("quit")
            self.proc.wait(timeout=3)
        except Exception:
            self.proc.kill()
        finally:
            self.stderr_handle.close()


def run_position_sweep(out_dir: Path, engine_bin: Path, hash_mb: int) -> dict:
    _mkdir(out_dir)
    stderr_log = out_dir / "position_sweep.stderr.log"
    out_json = out_dir / "position_sweep.json"

    session = UciSession(engine_bin, stderr_log, hash_mb=hash_mb)
    try:
        results = []
        failures = []
        for case in POSITION_CASES:
            for depth in (1, 2, 3):
                result = session.search(case["fen"], depth=depth)
                record = {
                    "case": case["name"],
                    "fen": case["fen"],
                    **result,
                }
                results.append(record)
                if not record["legal"] or int(record["info_lines"]) == 0:
                    failures.append(record)
            for movetime in (50, 100):
                result = session.search(case["fen"], movetime=movetime)
                record = {
                    "case": case["name"],
                    "fen": case["fen"],
                    **result,
                }
                results.append(record)
                if not record["legal"] or int(record["info_lines"]) == 0:
                    failures.append(record)

        payload = {
            "kind": "engine_position_sweep",
            "engine": str(engine_bin),
            "hash_mb": hash_mb,
            "positions": [case["name"] for case in POSITION_CASES],
            "total_queries": len(results),
            "failures": len(failures),
            "passed": len(failures) == 0,
            "results": results,
        }
        out_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return {
            "name": "position_sweep",
            "status": "passed" if payload["passed"] else "failed",
            "report_json": str(out_json),
            "stderr_log": str(stderr_log),
            "total_queries": payload["total_queries"],
            "failures": payload["failures"],
        }
    finally:
        session.close()


def run_scheduler_runtime_probe(out_dir: Path, engine_bin: Path) -> dict:
    _mkdir(out_dir)
    report_json = out_dir / "report.json"
    stdout_log = out_dir / "stdout.log"
    stderr_log = out_dir / "stderr.log"

    cmd = [str(engine_bin), "--scheduler-probe"]
    proc = subprocess.run(
        cmd,
        cwd=ENGINE_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    stdout_log.write_text(proc.stdout, encoding="utf-8")
    stderr_log.write_text(proc.stderr, encoding="utf-8")

    payload: dict[str, object]
    parse_error = None
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        parse_error = str(exc)
        payload = {
            "kind": "scheduler_runtime_probe",
            "status": "failed",
            "passed": False,
            "parse_error": parse_error,
            "stdout_log": str(stdout_log),
            "stderr_log": str(stderr_log),
        }

    payload["returncode"] = proc.returncode
    report_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    passed = proc.returncode == 0 and bool(payload.get("passed"))
    return {
        "name": "scheduler_runtime_probe",
        "status": "passed" if passed else "failed",
        "returncode": proc.returncode,
        "report_json": str(report_json),
        "stdout_log": str(stdout_log),
        "stderr_log": str(stderr_log),
        "parse_error": parse_error,
    }


def run_eval_service_probe(out_dir: Path, engine_bin: Path) -> dict:
    _mkdir(out_dir)
    report_json = out_dir / "report.json"
    stdout_log = out_dir / "stdout.log"
    stderr_log = out_dir / "stderr.log"

    cmd = [str(engine_bin), "--eval-service-probe"]
    proc = subprocess.run(
        cmd,
        cwd=ENGINE_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    stdout_log.write_text(proc.stdout, encoding="utf-8")
    stderr_log.write_text(proc.stderr, encoding="utf-8")

    payload: dict[str, object]
    parse_error = None
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        parse_error = str(exc)
        payload = {
            "kind": "eval_service_probe",
            "status": "failed",
            "passed": False,
            "parse_error": parse_error,
            "stdout_log": str(stdout_log),
            "stderr_log": str(stderr_log),
        }

    payload["returncode"] = proc.returncode
    report_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    passed = proc.returncode == 0 and bool(payload.get("passed"))
    return {
        "name": "eval_service_probe",
        "status": "passed" if passed else "failed",
        "returncode": proc.returncode,
        "report_json": str(report_json),
        "stdout_log": str(stdout_log),
        "stderr_log": str(stderr_log),
        "parse_error": parse_error,
    }


def run_search_runtime_probe(out_dir: Path, engine_bin: Path) -> dict:
    _mkdir(out_dir)
    report_json = out_dir / "report.json"
    stdout_log = out_dir / "stdout.log"
    stderr_log = out_dir / "stderr.log"

    cmd = [str(engine_bin), "--search-runtime-probe"]
    proc = subprocess.run(
        cmd,
        cwd=ENGINE_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    stdout_log.write_text(proc.stdout, encoding="utf-8")
    stderr_log.write_text(proc.stderr, encoding="utf-8")

    payload: dict[str, object]
    parse_error = None
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        parse_error = str(exc)
        payload = {
            "kind": "search_runtime_probe",
            "status": "failed",
            "passed": False,
            "parse_error": parse_error,
            "stdout_log": str(stdout_log),
            "stderr_log": str(stderr_log),
        }

    payload["returncode"] = proc.returncode
    report_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    passed = proc.returncode == 0 and bool(payload.get("passed"))
    return {
        "name": "search_runtime_probe",
        "status": "passed" if passed else "failed",
        "returncode": proc.returncode,
        "report_json": str(report_json),
        "stdout_log": str(stdout_log),
        "stderr_log": str(stderr_log),
        "parse_error": parse_error,
    }


def run_arena_scenario(name: str, out_dir: Path, python_bin: str, base_args: list[str]) -> dict:
    _mkdir(out_dir)
    report_json = out_dir / "report.json"
    games_jsonl = out_dir / "games.jsonl"
    games_pgn = out_dir / "games.pgn"
    stdout_log = out_dir / "stdout.log"
    stderr_log = out_dir / "stderr.log"

    cmd = [
        python_bin,
        str(ARENA_PY),
        "--report-json",
        str(report_json),
        "--games-jsonl",
        str(games_jsonl),
        "--pgn-out",
        str(games_pgn),
        "--event",
        name,
        *base_args,
    ]

    with open(stdout_log, "w", encoding="utf-8") as out, open(stderr_log, "w", encoding="utf-8") as err:
        rc = subprocess.run(cmd, cwd=ENGINE_DIR, stdout=out, stderr=err, text=True).returncode

    parsed = {}
    if report_json.exists():
        parsed = json.loads(report_json.read_text(encoding="utf-8"))

    return {
        "name": name,
        "status": "passed" if rc == 0 else "failed",
        "returncode": rc,
        "command": cmd,
        "report_json": str(report_json),
        "games_jsonl": str(games_jsonl),
        "games_pgn": str(games_pgn),
        "stdout_log": str(stdout_log),
        "stderr_log": str(stderr_log),
        "summary": parsed.get("summary", {}),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--python-bin", default=sys.executable)
    ap.add_argument("--engine-bin", default=str(ENGINE_BIN))
    ap.add_argument("--stockfish-bin", default=str(DEFAULT_STOCKFISH))
    ap.add_argument("--out-dir", help="artifact directory; default is a timestamped directory under cuda/engine/artifacts")
    ap.add_argument("--hash-mb", type=int, default=64)
    ap.add_argument("--selfplay-games", type=int, default=8)
    ap.add_argument("--stockfish-games", type=int, default=6)
    ap.add_argument("--movetime-games", type=int, default=6)
    ap.add_argument("--skip-stockfish", action="store_true")
    args = ap.parse_args()

    engine_bin = Path(args.engine_bin)
    stockfish_bin = Path(args.stockfish_bin)
    if not engine_bin.exists():
        raise SystemExit(f"engine binary not found: {engine_bin}")
    if not ARENA_PY.exists():
        raise SystemExit(f"arena harness not found: {ARENA_PY}")

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        stamp = time.strftime("verify_long_%Y%m%d_%H%M%S")
        out_dir = DEFAULT_ARTIFACT_ROOT / stamp
    _mkdir(out_dir)

    checks = []
    checks.append(run_eval_service_probe(out_dir / "eval_service_probe", engine_bin))
    checks.append(run_scheduler_runtime_probe(out_dir / "scheduler_runtime_probe", engine_bin))
    checks.append(run_search_runtime_probe(out_dir / "search_runtime_probe", engine_bin))
    checks.append(run_position_sweep(out_dir / "position_sweep", engine_bin, hash_mb=args.hash_mb))

    checks.append(
        run_arena_scenario(
            "selfplay_depth2_paired",
            out_dir / "selfplay_depth2_paired",
            args.python_bin,
            [
                "--games", str(args.selfplay_games),
                "--paired",
                "--opening-plies", "8",
                "--max-plies", "100",
                "--quiet",
                f"cuda_engine,depth=2,hash_mb={args.hash_mb}",
                f"cuda_engine,depth=2,hash_mb={args.hash_mb}",
            ],
        )
    )

    checks.append(
        run_arena_scenario(
            "selfplay_movetime75_paired",
            out_dir / "selfplay_movetime75_paired",
            args.python_bin,
            [
                "--games", str(args.movetime_games),
                "--paired",
                "--opening-plies", "10",
                "--max-plies", "100",
                "--quiet",
                f"cuda_engine,movetime=75,hash_mb={args.hash_mb}",
                f"cuda_engine,movetime=75,hash_mb={args.hash_mb}",
            ],
        )
    )

    if args.skip_stockfish or not stockfish_bin.exists():
        checks.append(
            {
                "name": "vs_stockfish_depth1_paired",
                "status": "skipped",
                "reason": "stockfish unavailable or explicitly skipped",
            }
        )
    else:
        checks.append(
            run_arena_scenario(
                "vs_stockfish_depth1_paired",
                out_dir / "vs_stockfish_depth1_paired",
                args.python_bin,
                [
                    "--games", str(args.stockfish_games),
                    "--paired",
                    "--opening-plies", "8",
                    "--max-plies", "100",
                    "--quiet",
                    f"cuda_engine,depth=2,hash_mb={args.hash_mb}",
                    f"stockfish,depth=1,threads=1,hash_mb=16",
                ],
            )
        )

    summary = {
        "kind": "engine_verify_long",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "artifact_dir": str(out_dir),
        "engine_bin": str(engine_bin),
        "checks": checks,
        "passed": all(check["status"] in ("passed", "skipped") for check in checks),
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(f"artifact_dir={out_dir}")
    for check in checks:
        line = f"{check['status']:7s} {check['name']}"
        if "report_json" in check:
            line += f" -> {check['report_json']}"
        print(line)
    print(f"summary={summary_path}")
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
