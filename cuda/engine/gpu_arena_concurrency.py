#!/usr/bin/env python3
"""Measure real all-GPU chess-game concurrency.

This is deliberately NOT the AgentChess broker path. It runs no Docker
fighters and executes no JS/Python player programs. Each game is two UCI
instances of the CUDA engine; the host only relays UCI tokens, tracks a move
list, and records timing.

Run label: new_experiment. The script prints and saves the condition so a
future run can be compared without guessing.
"""
from __future__ import annotations

import argparse
import concurrent.futures as futures
import hashlib
import json
import queue
import random
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import TextIO


HERE = Path(__file__).resolve().parent

SPEC_POOL: list[tuple[str, str]] = [
    ("depth 2", "depth 2"),
    ("depth 2", "depth 3"),
    ("depth 3", "depth 2"),
    ("depth 3", "depth 3"),
    ("depth 3", "depth 4"),
    ("depth 4", "depth 3"),
    ("depth 4", "depth 4"),
    ("depth 4", "depth 5"),
    ("depth 5", "depth 4"),
    ("depth 5", "depth 5"),
    ("movetime 50", "movetime 50"),
    ("movetime 100", "movetime 100"),
    ("movetime 200", "movetime 200"),
    ("movetime 100", "movetime 200"),
    ("movetime 200", "movetime 100"),
    ("depth 3", "movetime 100"),
    ("movetime 100", "depth 3"),
    ("depth 4", "movetime 100"),
    ("movetime 100", "depth 4"),
]


@dataclass
class GameResult:
    game_id: int
    white_spec: str
    black_spec: str
    verdict: str
    plies: int
    wall_s: float
    error: str | None = None
    moves: list[str] | None = None


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_value(args: list[str]) -> str | None:
    try:
        p = subprocess.run(
            ["git", *args],
            cwd=HERE,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return None
    if p.returncode != 0:
        return None
    return p.stdout.strip() or None


class LineReader:
    """Timeout-capable readline without select/TextIO buffering traps."""

    def __init__(self, stream: TextIO):
        self._q: queue.Queue[str] = queue.Queue()
        self._thread = threading.Thread(target=self._read_loop, args=(stream,), daemon=True)
        self._thread.start()

    def _read_loop(self, stream: TextIO) -> None:
        try:
            for line in stream:
                self._q.put(line)
        finally:
            self._q.put("")

    def readline(self, timeout_s: float) -> str | None:
        try:
            return self._q.get(timeout=timeout_s)
        except queue.Empty:
            return None


def proc_reader(proc: subprocess.Popen) -> LineReader:
    reader = getattr(proc, "_uci_line_reader", None)
    if reader is None:
        raise RuntimeError("engine line reader missing")
    return reader


def send(proc: subprocess.Popen, line: str) -> None:
    if proc.stdin is None:
        raise RuntimeError("engine stdin closed")
    proc.stdin.write(line + "\n")
    proc.stdin.flush()


def open_engine(path: Path, handshake_timeout_s: float) -> subprocess.Popen:
    proc = subprocess.Popen(
        [str(path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    proc._uci_line_reader = LineReader(proc.stdout)  # type: ignore[attr-defined]
    send(proc, "uci")
    while True:
        line = proc_reader(proc).readline(handshake_timeout_s)
        if line is None:
            raise RuntimeError("engine handshake timeout at uci")
        if line == "":
            raise RuntimeError("engine died during uci handshake")
        if line.strip() == "uciok":
            break
    send(proc, "isready")
    while True:
        line = proc_reader(proc).readline(handshake_timeout_s)
        if line is None:
            raise RuntimeError("engine handshake timeout at isready")
        if line == "":
            raise RuntimeError("engine died during isready")
        if line.strip() == "readyok":
            break
    send(proc, "ucinewgame")
    return proc


def close_engine(proc: subprocess.Popen) -> None:
    try:
        if proc.poll() is None:
            send(proc, "quit")
            proc.wait(timeout=2)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def go_move(
    proc: subprocess.Popen,
    moves: list[str],
    spec: str,
    per_move_timeout_s: float,
) -> str:
    if proc.stdout is None:
        raise RuntimeError("engine stdout closed")
    pos_cmd = "position startpos"
    if moves:
        pos_cmd += " moves " + " ".join(moves)
    send(proc, pos_cmd)
    send(proc, f"go {spec}")
    while True:
        line = proc_reader(proc).readline(per_move_timeout_s)
        if line is None:
            raise TimeoutError(f"go timeout after {per_move_timeout_s}s for {spec}")
        if line == "":
            raise RuntimeError("engine died during go")
        stripped = line.strip()
        if stripped.startswith("bestmove"):
            parts = stripped.split()
            return parts[1] if len(parts) >= 2 else "0000"


def run_game(
    game_id: int,
    engine_path: Path,
    white_spec: str,
    black_spec: str,
    max_plies: int,
    game_timeout_s: float,
    per_move_timeout_s: float,
    handshake_timeout_s: float,
    keep_moves: bool,
) -> GameResult:
    t0 = time.perf_counter()
    moves: list[str] = []
    white: subprocess.Popen | None = None
    black: subprocess.Popen | None = None
    verdict = "max_plies"
    error = None
    try:
        white = open_engine(engine_path, handshake_timeout_s)
        black = open_engine(engine_path, handshake_timeout_s)
        side = 0
        deadline = t0 + game_timeout_s
        for _ply in range(max_plies):
            if time.perf_counter() > deadline:
                verdict = "host_timeout"
                break
            proc = white if side == 0 else black
            spec = white_spec if side == 0 else black_spec
            bestmove = go_move(proc, moves, spec, per_move_timeout_s)
            if not bestmove or bestmove == "0000":
                verdict = "terminal_no_legal_move"
                break
            moves.append(bestmove)
            side ^= 1
    except TimeoutError as e:
        verdict = "host_timeout"
        error = str(e)
    except Exception as e:
        verdict = "engine_error"
        error = str(e)
    finally:
        for proc in (white, black):
            if proc is not None:
                close_engine(proc)
    return GameResult(
        game_id=game_id,
        white_spec=white_spec,
        black_spec=black_spec,
        verdict=verdict,
        plies=len(moves),
        wall_s=round(time.perf_counter() - t0, 3),
        error=error,
        moves=moves if keep_moves else None,
    )


def nvidia_smi_sample() -> dict | None:
    try:
        p = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=3,
        )
    except Exception:
        return None
    if p.returncode != 0:
        return None
    rows = []
    for line in p.stdout.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            rows.append({"mem_mb": int(parts[0]), "util_pct": int(parts[1])})
        except ValueError:
            continue
    if not rows:
        return None
    return {
        "gpus": rows,
        "mem_mb_total": sum(row["mem_mb"] for row in rows),
        "util_pct_max": max(row["util_pct"] for row in rows),
    }


class GpuSampler:
    def __init__(self, interval_s: float):
        self.interval_s = interval_s
        self.samples: list[dict] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "GpuSampler":
        if self.interval_s <= 0:
            return self

        def loop() -> None:
            while not self._stop.is_set():
                sample = nvidia_smi_sample()
                if sample is not None:
                    sample["t"] = time.time()
                    self.samples.append(sample)
                self._stop.wait(self.interval_s)

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_args) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def summary(self) -> dict:
        if not self.samples:
            return {}
        return {
            "sample_count": len(self.samples),
            "peak_mem_mb_total": max(s["mem_mb_total"] for s in self.samples),
            "peak_util_pct": max(s["util_pct_max"] for s in self.samples),
        }


def choose_specs(n: int, seed: int) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    pool = list(SPEC_POOL)
    if seed != 0:
        rng.shuffle(pool)
    return (pool * ((n // len(pool)) + 1))[:n]


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    vals = sorted(values)
    idx = (len(vals) - 1) * pct
    lo = int(idx)
    hi = min(lo + 1, len(vals) - 1)
    frac = idx - lo
    return vals[lo] * (1 - frac) + vals[hi] * frac


def run_level(args: argparse.Namespace, level: int, specs: list[tuple[str, str]]) -> dict:
    t0 = time.perf_counter()
    games: list[GameResult] = []
    with GpuSampler(args.gpu_sample_s) as sampler:
        with futures.ThreadPoolExecutor(max_workers=level) as ex:
            futs = [
                ex.submit(
                    run_game,
                    i,
                    args.engine,
                    ws,
                    bs,
                    args.max_plies,
                    args.game_timeout_s,
                    args.per_move_timeout_s,
                    args.handshake_timeout_s,
                    args.keep_moves,
                )
                for i, (ws, bs) in enumerate(specs)
            ]
            for fut in futures.as_completed(futs):
                games.append(fut.result())
    wall_s = time.perf_counter() - t0
    games.sort(key=lambda g: g.game_id)
    counts: dict[str, int] = {}
    for g in games:
        counts[g.verdict] = counts.get(g.verdict, 0) + 1
    game_walls = [g.wall_s for g in games]
    plies = [g.plies for g in games]
    ok_games = sum(1 for g in games if g.verdict != "engine_error")
    return {
        "concurrency": level,
        "n": len(games),
        "wall_s": round(wall_s, 3),
        "games_per_hour": round(len(games) * 3600.0 / wall_s, 3) if wall_s > 0 else None,
        "ok_games_per_hour": round(ok_games * 3600.0 / wall_s, 3) if wall_s > 0 else None,
        "plies_per_hour": round(sum(plies) * 3600.0 / wall_s, 3) if wall_s > 0 else None,
        "verdicts": counts,
        "avg_plies": round(statistics.mean(plies), 3) if plies else 0,
        "game_wall_s": {
            "p50": round(percentile(game_walls, 0.50), 3) if game_walls else None,
            "p90": round(percentile(game_walls, 0.90), 3) if game_walls else None,
            "max": round(max(game_walls), 3) if game_walls else None,
        },
        "gpu": sampler.summary(),
        "games": [asdict(g) for g in games],
    }


def parse_concurrency(raw: str) -> list[int]:
    vals = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        val = int(part)
        if val <= 0:
            raise ValueError("concurrency levels must be positive")
        vals.append(val)
    if not vals:
        raise ValueError("no concurrency levels")
    return vals


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", type=Path, default=HERE / "engine")
    ap.add_argument("--concurrency", default="1,2,4,8")
    ap.add_argument("--n-per-level", type=int, default=8)
    ap.add_argument("--max-plies", type=int, default=120)
    ap.add_argument("--game-timeout-s", type=float, default=300.0)
    ap.add_argument("--per-move-timeout-s", type=float, default=30.0)
    ap.add_argument("--handshake-timeout-s", type=float, default=10.0)
    ap.add_argument("--gpu-sample-s", type=float, default=0.0,
                    help="sample nvidia-smi every N seconds; 0 disables")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--keep-moves", action="store_true")
    args = ap.parse_args()

    args.engine = args.engine.resolve()
    if not args.engine.is_file():
        print(f"engine not found: {args.engine}", file=sys.stderr)
        return 2
    if args.n_per_level <= 0:
        print("--n-per-level must be positive", file=sys.stderr)
        return 2
    levels = parse_concurrency(args.concurrency)
    specs = choose_specs(args.n_per_level, args.seed)

    condition = {
        "run_label": "new_experiment",
        "host_role": "UCI relay only; no Docker, no JS/Python fighter programs, no host chess rules",
        "engine": str(args.engine),
        "engine_sha256": sha256_file(args.engine),
        "git_branch": git_value(["branch", "--show-current"]),
        "git_commit": git_value(["rev-parse", "HEAD"]),
        "git_dirty": bool(git_value(["status", "--porcelain"])),
        "cwd": str(HERE),
        "seed": args.seed,
        "n_per_level": args.n_per_level,
        "concurrency_levels": levels,
        "max_plies": args.max_plies,
        "game_timeout_s": args.game_timeout_s,
        "per_move_timeout_s": args.per_move_timeout_s,
        "gpu_sample_s": args.gpu_sample_s,
        "specs": [{"white": ws, "black": bs} for ws, bs in specs],
        "argv": sys.argv,
    }

    print("=== gpu_arena_concurrency condition ===")
    print(json.dumps(condition, indent=2))
    print()

    results = []
    for level in levels:
        print(f"=== concurrency {level}: {args.n_per_level} all-GPU games ===")
        level_result = run_level(args, level, specs)
        results.append(level_result)
        print(
            "  wall={wall_s:.1f}s games/hr={games_per_hour:.1f} "
            "plies/hr={plies_per_hour:.1f} verdicts={verdicts}".format(**level_result)
        )
        gpu = level_result.get("gpu") or {}
        if gpu:
            print(
                f"  gpu_peak_mem_mb={gpu.get('peak_mem_mb_total')} "
                f"gpu_peak_util_pct={gpu.get('peak_util_pct')}"
            )
        print()

    out_path = args.out or HERE / f"arena_concurrency_{time.strftime('%Y%m%d-%H%M%S')}.json"
    payload = {
        "timestamp": time.time(),
        "condition": condition,
        "levels": results,
    }
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"receipt: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
