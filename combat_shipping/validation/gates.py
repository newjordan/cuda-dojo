from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
LIVE_ARBITER = PKG / "live_arbiter"
BROKER_INT = PKG / "broker_integration"
HARNESS = PKG / "harness"
LIVE_COMPARE = PKG / "live_compare"
DEFAULT_FIGHTERS_DIR = Path(
    "/home/frosty40/AgentChess_Server/match-processor/data"
)


@dataclass
class GateResult:
    name: str
    target: str
    observed: str
    passed: bool
    duration_s: float = 0.0
    details: dict = field(default_factory=dict)


def _list_fighters() -> list[Path]:
    return sorted(DEFAULT_FIGHTERS_DIR.glob("match-*.js"))


# ---------------------------------------------------------------------- G1
def gate_replay_parity(n: int, pgn: str,
                       min_parity: float, min_subset: float) -> GateResult:
    t0 = time.time()
    out = Path(f"/tmp/validation_replay_{int(t0)}.json")
    cmd = [
        sys.executable, str(HARNESS / "replay_shadow.py"),
        "--n", str(n),
        "--max-scan", str(n * 2),
        "--pgn", pgn,
        "--out", str(out),
    ]
    try:
        p = subprocess.run(cmd, capture_output=True,
                           timeout=60 + n * 0.2)
    except subprocess.TimeoutExpired as e:
        return GateResult(
            "G1 replay_parity", f"≥{min_parity:.4f}", "TIMEOUT",
            False, time.time() - t0,
            {"stderr": (e.stderr.decode("utf-8", "replace")[-500:]
                        if e.stderr else "")},
        )
    dt = time.time() - t0

    if p.returncode != 0 or not out.exists():
        return GateResult(
            "G1 replay_parity", f"≥{min_parity:.4f}", "ERROR",
            False, dt,
            {"returncode": p.returncode,
             "stderr_tail": p.stderr.decode("utf-8", "replace")[-500:]},
        )

    data = json.loads(out.read_text())
    s = data["summary"]
    parity = s["referee_parity_count"] / s["n"]
    subset_n = s.get("replayable_subset_total", 0)
    subset_match = s.get("replayable_subset_match", 0)
    subset_rate = subset_match / max(1, subset_n)
    passed = parity >= min_parity and subset_rate >= min_subset
    return GateResult(
        "G1 replay_parity",
        f"parity ≥{min_parity:.4f}; replayable_subset = {min_subset:.4f}",
        f"parity={parity:.4f} ({s['referee_parity_count']}/{s['n']}); "
        f"subset={subset_rate:.4f} ({subset_match}/{subset_n})",
        passed, dt,
        {"n": s["n"], "parity": parity,
         "subset_n": subset_n, "subset_rate": subset_rate,
         "out_path": str(out)},
    )


# ---------------------------------------------------------------------- G3
def gate_bench_coalescer(min_speedup: float) -> GateResult:
    t0 = time.time()
    p = subprocess.run(
        [sys.executable, str(BROKER_INT / "_bench_coalescer.py")],
        capture_output=True, timeout=120,
    )
    dt = time.time() - t0
    out = p.stdout.decode("utf-8", "replace")

    parity_ok = "[gate 1] PASS" in out
    stress_ok = "gate 3 stress:   PASS" in out

    speedup = None
    for line in out.splitlines():
        if line.startswith("  speedup:") and "target" in line:
            try:
                speedup = float(line.split()[1].rstrip("×××x"))
            except Exception:
                pass
            break

    speedup_ok = speedup is not None and speedup >= min_speedup
    passed = parity_ok and stress_ok and speedup_ok
    speedup_str = f"{speedup:.2f}×" if speedup is not None else "?"
    return GateResult(
        "G3 bench_coalescer",
        f"parity PASS, speedup ≥{min_speedup:.2f}×, stress PASS",
        (f"parity={'PASS' if parity_ok else 'FAIL'}, "
         f"speedup={speedup_str}, "
         f"stress={'PASS' if stress_ok else 'FAIL'}"),
        passed, dt,
        {"parity": parity_ok, "speedup": speedup, "stress": stress_ok,
         "stdout_tail": out[-1500:]},
    )


# --- daemon spawn / shutdown helpers ---------------------------------------
class _Daemon:
    def __init__(self, socket_path: str):
        self.socket_path = socket_path
        self.proc: subprocess.Popen | None = None
        self.log = Path(f"/tmp/validation_daemon_{int(time.time())}.log")

    def __enter__(self):
        try:
            os.unlink(self.socket_path)
        except FileNotFoundError:
            pass
        self.proc = subprocess.Popen(
            [sys.executable, str(BROKER_INT / "cuda_arbiter_daemon_batched.py"),
             "--socket", self.socket_path],
            stdout=open(self.log, "w"), stderr=subprocess.STDOUT,
            cwd=str(BROKER_INT),
            preexec_fn=os.setsid,
        )
        # Wait for socket.
        for _ in range(60):
            if Path(self.socket_path).exists():
                # Give the dojo_ref preload a beat.
                time.sleep(1.0)
                return self
            time.sleep(0.5)
        raise RuntimeError(f"daemon never opened {self.socket_path}; "
                           f"see {self.log}")

    def __exit__(self, *_):
        if self.proc and self.proc.poll() is None:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
                self.proc.wait(timeout=5)
            except Exception:
                try:
                    os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
                except Exception:
                    pass


def _daemon_request(socket_path: str, req: dict, timeout_s: float) -> dict:
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout_s)
    s.connect(socket_path)
    s.sendall((json.dumps(req) + "\n").encode("utf-8"))
    buf = b""
    deadline = time.monotonic() + timeout_s
    while b"\n" not in buf:
        if time.monotonic() > deadline:
            raise TimeoutError("daemon response timeout")
        chunk = s.recv(65536)
        if not chunk:
            break
        buf += chunk
    s.close()
    return json.loads(buf.split(b"\n", 1)[0])


# ---------------------------------------------------------------------- G4
def gate_daemon_smoke(max_plies: int) -> GateResult:
    fighters = _list_fighters()
    if len(fighters) < 2:
        return GateResult(
            "G4 daemon_smoke", "1 game OK", "no fighters",
            False, 0.0, {})
    t0 = time.time()
    sock = "/tmp/cuda_arbiter_validation.sock"
    try:
        with _Daemon(sock):
            req = {
                "id": 1, "match_id": "valG4",
                "white": str(fighters[0]), "black": str(fighters[1]),
                "white_lang": "js", "black_lang": "js",
                "white_name": "G4-w", "black_name": "G4-b",
                "max_plies": max_plies, "move_timeout_ms": 5500,
            }
            resp = _daemon_request(sock, req,
                                   timeout_s=max_plies * 7.0 + 30)
        dt = time.time() - t0
    except Exception as e:
        return GateResult(
            "G4 daemon_smoke", "1 game OK", f"ERROR: {e}",
            False, time.time() - t0, {})

    has_error = bool(resp.get("error"))
    crash_reasons = {"crash"}
    is_crash = resp.get("reason") in crash_reasons
    passed = not has_error and not is_crash and resp.get("plies", 0) > 0
    return GateResult(
        "G4 daemon_smoke",
        "1 real fighter game; plies>0; no daemon error",
        (f"plies={resp.get('plies')} result={resp.get('result')} "
         f"reason={resp.get('reason')}"),
        passed, dt,
        {"resp": {k: v for k, v in resp.items()
                  if k not in ("moves",)},
         "moves_len": len(resp.get("moves", []))},
    )


# ---------------------------------------------------------------------- G5
def gate_daemon_concurrent(n: int, max_plies: int) -> GateResult:
    fighters = _list_fighters()
    if len(fighters) < 2:
        return GateResult(
            "G5 daemon_concurrent", f"{n} games OK", "no fighters",
            False, 0.0, {})
    t0 = time.time()
    sock = "/tmp/cuda_arbiter_validation.sock"
    pairs = [(fighters[i % len(fighters)],
              fighters[(i + 1) % len(fighters)]) for i in range(n)]
    results: list[dict | Exception] = [None] * n  # type: ignore

    def run(i: int) -> None:
        req = {
            "id": i, "match_id": f"valG5_{i:02d}",
            "white": str(pairs[i][0]), "black": str(pairs[i][1]),
            "white_lang": "js", "black_lang": "js",
            "white_name": f"w{i}", "black_name": f"b{i}",
            "max_plies": max_plies, "move_timeout_ms": 5500,
        }
        try:
            results[i] = _daemon_request(sock, req,
                                         timeout_s=max_plies * 7.0 + 30)
        except Exception as e:
            results[i] = e

    try:
        with _Daemon(sock):
            with ThreadPoolExecutor(max_workers=n) as ex:
                list(ex.map(run, range(n)))
        dt = time.time() - t0
    except Exception as e:
        return GateResult(
            "G5 daemon_concurrent", f"{n} games OK", f"DAEMON-ERROR: {e}",
            False, time.time() - t0, {})

    n_err = sum(1 for r in results if isinstance(r, Exception))
    n_payload_err = sum(1 for r in results
                        if isinstance(r, dict) and r.get("error"))
    n_crash = sum(1 for r in results
                  if isinstance(r, dict) and r.get("reason") == "crash")
    n_ok = n - n_err - n_payload_err - n_crash
    passed = n_err == 0 and n_payload_err == 0 and n_crash == 0
    return GateResult(
        "G5 daemon_concurrent",
        f"{n} games complete; 0 socket/daemon errors",
        f"ok={n_ok}/{n} sock_err={n_err} payload_err={n_payload_err} crash={n_crash}",
        passed, dt,
        {"reasons": [r.get("reason") if isinstance(r, dict) else "exc"
                     for r in results],
         "wall_max_s": max(
             (r.get("wall_seconds", 0.0) if isinstance(r, dict) else 0.0)
             for r in results)},
    )


# ---------------------------------------------------------------------- G6
def gate_live_wall_ratio(n: int, agent_cpus: str, agent_memory: str,
                         max_plies: int,
                         min_ratio: float, max_ratio: float) -> GateResult:
    t0 = time.time()
    out_root = (PKG / "results" /
                f"validation_live_{time.strftime('%Y%m%d-%H%M%S')}")
    env = os.environ.copy()
    env["AGENT_CPUS"] = agent_cpus
    env["AGENT_MEMORY"] = agent_memory
    cmd = [
        sys.executable, str(LIVE_COMPARE / "compare.py"),
        "--n", str(n), "--max-plies", str(max_plies),
        "--move-timeout-ms", "5500",
        "--concurrency", "1",
        "--out-root", str(out_root),
    ]
    try:
        subprocess.run(cmd, capture_output=True,
                       timeout=n * 240, env=env, check=False)
    except subprocess.TimeoutExpired:
        return GateResult(
            "G6 live_wall_ratio",
            f"ratio ∈ [{min_ratio}, {max_ratio}]", "TIMEOUT",
            False, time.time() - t0, {})
    dt = time.time() - t0

    summary_path = out_root / "summary.json"
    if not summary_path.exists():
        return GateResult(
            "G6 live_wall_ratio",
            f"ratio ∈ [{min_ratio}, {max_ratio}]", "NO SUMMARY",
            False, dt, {})

    s = json.loads(summary_path.read_text())
    ratio = s.get("speed_ratio_arb_over_cuda", 0.0)
    passed = (s.get("n_arbiter_errors", 0) + s.get("n_cuda_errors", 0)) == 0 \
        and min_ratio <= ratio <= max_ratio
    return GateResult(
        "G6 live_wall_ratio",
        f"arb/cuda ratio ∈ [{min_ratio}, {max_ratio}], no errors",
        f"ratio={ratio:.3f} arb={s.get('wall_arbiter_mean'):.1f}s "
        f"cuda={s.get('wall_cuda_mean'):.1f}s "
        f"errors={s.get('n_arbiter_errors',0)+s.get('n_cuda_errors',0)}",
        passed, dt,
        {"summary_path": str(summary_path),
         "ratio": ratio,
         "wall_arbiter_mean": s.get("wall_arbiter_mean"),
         "wall_cuda_mean": s.get("wall_cuda_mean"),
         "result_match_rate": s.get("result_match_rate")},
    )


def result_to_dict(g: GateResult) -> dict:
    return asdict(g)
