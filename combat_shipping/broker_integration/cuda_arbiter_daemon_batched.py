#!/usr/bin/env python3
"""cuda_arbiter_daemon_batched.py — persistent CUDA arbiter service
with cross-thread request coalescing on the dojo_ref FFI.

Same socket protocol as cuda_arbiter_daemon.py, but each per-game
thread submits its `legal_moves` / `is_check` / `make_move` requests
to a shared `BatchCoalescer`. A single dispatcher thread drains
pending requests on a short deadline (~5 ms) or batch-size threshold,
groups them by op kind, and issues ONE batched FFI call per group:

    dojo_ref.legal_moves_batched(positions)
    dojo_ref.is_check_batched(positions)
    dojo_ref.make_move_batched(positions, ucis)

Each waiting thread is unblocked with its own slice of the result.

Correctness invariant: the batched FFI is byte-identical to N serial
single-position calls (already verified at the FFI layer). The
dispatcher is a pure transport; no fallback to single-position calls
is permitted on the dispatcher path — that would defeat the goal.

Dev socket: /tmp/cuda_arbiter_batched.sock (NOT /tmp/cuda_arbiter.sock).
The production daemon is untouched.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import re
import socket
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import Future
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Make live_arbiter importable so we reuse gpu_state.HostState verbatim
# and the same Docker fighter helpers as the production daemon.
_HERE = Path(__file__).resolve().parent
_LIVE_ARBITER = _HERE.parent / "live_arbiter"
if str(_LIVE_ARBITER) not in sys.path:
    sys.path.insert(0, str(_LIVE_ARBITER))

import dojo_ref  # noqa: E402
from gpu_state import HostState  # noqa: E402
import live_match  # noqa: E402  (Docker helpers + _result + build_pgn_uci)


DEFAULT_SOCKET = os.environ.get(
    "CUDA_ARBITER_BATCHED_SOCKET", "/tmp/cuda_arbiter_batched.sock"
)
UCI_RE = re.compile(r"^[a-h][1-8][a-h][1-8][qrbn]?$")


def _log(msg: str) -> None:
    sys.stderr.write(f"[batched-daemon {time.strftime('%H:%M:%S')}] {msg}\n")
    sys.stderr.flush()


# ----------------------------------------------------------------------
# Coalescing dispatcher.
# ----------------------------------------------------------------------

OP_LEGAL = "legal_moves"
OP_IS_CHECK = "is_check"
OP_MAKE_MOVE = "make_move"


@dataclass
class _Req:
    op: str
    position: Any           # dojo_ref.Position
    uci: str | None         # only for make_move
    future: Future          # set_result with the answer; set_exception on error


class BatchCoalescer:
    """Cross-thread coalescer for dojo_ref FFI calls.

    Worker threads call .legal_moves(pos) / .is_check(pos) /
    .make_move(pos, uci) — they enqueue a request and block on a
    Future. A single dispatcher thread runs the drain loop, groups by
    op kind, issues one batched FFI call per group, and resolves the
    futures.

    The dispatcher is the SOLE caller of dojo_ref's batched ABI in this
    daemon. There is no per-position-call fallback in this code path.
    """

    def __init__(
        self,
        max_batch: int = 32,
        deadline_ms: float = 0.5,
        idle_poll_ms: float = 0.1,
    ) -> None:
        self._q: "queue.Queue[_Req]" = queue.Queue()
        self._max_batch = max_batch
        self._deadline_s = deadline_ms / 1000.0
        self._idle_poll_s = idle_poll_ms / 1000.0
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="batch-coalescer", daemon=True,
        )
        # Stats — useful for tuning and reporting.
        self._stat_lock = threading.Lock()
        self.batches_dispatched = 0
        self.requests_served = 0
        self.batch_size_histogram: dict[int, int] = {}

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    # ---- Public per-thread API ----

    def legal_moves(self, pos) -> list[str]:
        f: Future = Future()
        self._q.put(_Req(OP_LEGAL, pos, None, f))
        return f.result()

    def is_check(self, pos) -> bool:
        f: Future = Future()
        self._q.put(_Req(OP_IS_CHECK, pos, None, f))
        return f.result()

    def make_move(self, pos, uci: str):
        f: Future = Future()
        self._q.put(_Req(OP_MAKE_MOVE, pos, uci, f))
        return f.result()

    # ---- Dispatcher loop ----

    def _drain(self) -> list[_Req]:
        """Block until at least one request arrives, then NON-BLOCKING
        drain everything else currently queued (up to `max_batch`).
        Returns the accumulated list (or empty on idle-tick wakeup).

        Steady-state behaviour with N concurrent worker threads: while
        one batch is in flight on the GPU, the workers (each blocked
        on its previous future) wake up, do a tiny bit of host work,
        and queue their next request. By the time the dispatcher
        returns from the FFI, all N workers have queued — drain picks
        them all up in one shot, dispatches one batched FFI for N.
        Tail thread doesn't wait for an artificial deadline.

        For the rare case where the queue is empty after the first
        item (e.g. a single-threaded client), we briefly poll up to
        `deadline_ms` to give a sibling worker a chance to land. This
        mostly helps at low concurrency; at N≥4 the non-blocking
        drain already catches everything.
        """
        try:
            first = self._q.get(timeout=0.25)
        except queue.Empty:
            return []
        batch: list[_Req] = [first]
        # Fast path: drain everything currently queued, no waiting.
        while len(batch) < self._max_batch:
            try:
                batch.append(self._q.get_nowait())
            except queue.Empty:
                break
        # Tiny-deadline path for low-concurrency workloads where
        # threads might still be in their host-work phase. Capped at
        # deadline_ms total to avoid hurting the steady-state path.
        if len(batch) < self._max_batch and self._deadline_s > 0:
            deadline = time.monotonic() + self._deadline_s
            while len(batch) < self._max_batch:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    batch.append(self._q.get(timeout=min(remaining, self._idle_poll_s)))
                except queue.Empty:
                    # No more arrivals in this slice — fire what we have.
                    break
        return batch

    def _run(self) -> None:
        while not self._stop.is_set():
            batch = self._drain()
            if not batch:
                continue
            # Group by op kind, preserving order within each group so
            # results align positionally with requests.
            groups: dict[str, list[_Req]] = {OP_LEGAL: [], OP_IS_CHECK: [], OP_MAKE_MOVE: []}
            for r in batch:
                groups[r.op].append(r)

            with self._stat_lock:
                self.batches_dispatched += 1
                self.requests_served += len(batch)
                self.batch_size_histogram[len(batch)] = (
                    self.batch_size_histogram.get(len(batch), 0) + 1
                )

            for op, reqs in groups.items():
                if not reqs:
                    continue
                positions = [r.position for r in reqs]
                try:
                    if op == OP_LEGAL:
                        results = dojo_ref.legal_moves_batched(positions)
                    elif op == OP_IS_CHECK:
                        results = dojo_ref.is_check_batched(positions)
                    elif op == OP_MAKE_MOVE:
                        ucis = [r.uci for r in reqs]
                        results = dojo_ref.make_move_batched(positions, ucis)
                    else:
                        raise RuntimeError(f"unknown op kind: {op}")
                except Exception as exc:
                    # Whole-group failure: surface to every waiter so
                    # they can fail their game cleanly. Do NOT retry
                    # per-position — the FFI is the source of truth and
                    # if it threw, we want that signal raw.
                    for r in reqs:
                        r.future.set_exception(exc)
                    continue
                if len(results) != len(reqs):
                    err = RuntimeError(
                        f"batched {op} returned {len(results)} results for "
                        f"{len(reqs)} requests"
                    )
                    for r in reqs:
                        r.future.set_exception(err)
                    continue
                for r, res in zip(reqs, results):
                    r.future.set_result(res)

    def stats_snapshot(self) -> dict:
        with self._stat_lock:
            return {
                "batches_dispatched": self.batches_dispatched,
                "requests_served": self.requests_served,
                "avg_batch_size": (
                    self.requests_served / max(1, self.batches_dispatched)
                ),
                "histogram": dict(self.batch_size_histogram),
            }


# ----------------------------------------------------------------------
# Coalesced game loop. Drop-in replacement for live_match.play_game,
# routing dojo_ref calls through the coalescer instead of calling the
# per-position FFI directly. Docker fighter sandbox + host bookkeeping
# logic is identical to the production play_game (imported helpers).
# ----------------------------------------------------------------------

def play_game_coalesced(
    coalescer: BatchCoalescer,
    white_path: str,
    black_path: str,
    match_id: str | None = None,
    max_plies: int = 500,
    move_timeout_ms: int = 5500,
    white_name: str | None = None,
    black_name: str | None = None,
) -> dict:
    if match_id is None:
        match_id = uuid.uuid4().hex[:8]
    white_name = white_name or Path(white_path).stem
    black_name = black_name or Path(black_path).stem

    white_code = Path(white_path).read_text()
    black_code = Path(black_path).read_text()

    pos = dojo_ref.Position.startpos()
    host = HostState()
    move_log: list[str] = []
    t_start = time.monotonic()

    white = live_match.start_container(match_id, "white", white_code)
    black = live_match.start_container(match_id, "black", black_code)

    try:
        for ply in range(max_plies):
            is_white_turn = host.side == "w"
            agent = white if is_white_turn else black

            if host.halfmove >= 100:
                return live_match._result(
                    "draw", "50-move", ply, move_log,
                    white_name, black_name, t_start, match_id, max_plies,
                )
            if host.insufficient_material():
                return live_match._result(
                    "draw", "insufficient", ply, move_log,
                    white_name, black_name, t_start, match_id, max_plies,
                )
            count = host.record_position_for_threefold()
            if count >= 3:
                return live_match._result(
                    "draw", "threefold", ply, move_log,
                    white_name, black_name, t_start, match_id, max_plies,
                )

            # GPU via coalescer.
            legal = coalescer.legal_moves(pos)
            if not legal:
                if coalescer.is_check(pos):
                    winner = "black" if is_white_turn else "white"
                    return live_match._result(
                        winner, "checkmate", ply, move_log,
                        white_name, black_name, t_start, match_id, max_plies,
                    )
                return live_match._result(
                    "draw", "stalemate", ply, move_log,
                    white_name, black_name, t_start, match_id, max_plies,
                )

            fen = host.fen()
            uci = live_match.get_agent_move(
                agent["name"], agent["ext"], fen, move_timeout_ms,
            )

            if uci in ("__CRASH__", "__OOM__"):
                code = white_code if is_white_turn else black_code
                live_match.stop_container(agent["name"])
                fresh = live_match.start_container(
                    match_id + "r", "white" if is_white_turn else "black", code,
                )
                if is_white_turn:
                    white = fresh; agent = white
                else:
                    black = fresh; agent = black
                uci = live_match.get_agent_move(
                    agent["name"], agent["ext"], fen, move_timeout_ms,
                )

            if uci == "__TIMEOUT__":
                winner = "black" if is_white_turn else "white"
                return live_match._result(
                    winner, "timeout", ply, move_log,
                    white_name, black_name, t_start, match_id, max_plies,
                )
            if uci in ("__CRASH__", "__OOM__"):
                winner = "black" if is_white_turn else "white"
                reason = "oom" if uci == "__OOM__" else "crash"
                return live_match._result(
                    winner, reason, ply, move_log,
                    white_name, black_name, t_start, match_id, max_plies,
                )
            if not UCI_RE.match(uci):
                winner = "black" if is_white_turn else "white"
                return live_match._result(
                    winner, "invalid_format", ply, move_log,
                    white_name, black_name, t_start, match_id, max_plies,
                )
            if uci not in legal:
                winner = "black" if is_white_turn else "white"
                return live_match._result(
                    winner, "illegal", ply, move_log,
                    white_name, black_name, t_start, match_id, max_plies,
                )

            move_log.append(uci)
            pos = coalescer.make_move(pos, uci)
            host.apply_uci(uci)

        return live_match._result(
            "draw", "max_plies", max_plies, move_log,
            white_name, black_name, t_start, match_id, max_plies,
        )
    finally:
        live_match.stop_container(white["name"])
        live_match.stop_container(black["name"])


# ----------------------------------------------------------------------
# Socket connection handler.
# ----------------------------------------------------------------------

def handle_client(coalescer: BatchCoalescer, conn: socket.socket, peer: str) -> None:
    rfile = conn.makefile("rb", buffering=0)
    wfile = conn.makefile("wb", buffering=0)
    try:
        for raw in rfile:
            line = raw.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
            except Exception as e:
                wfile.write((json.dumps({"error": f"bad json: {e}"}) + "\n").encode("utf-8"))
                continue

            req_id = req.get("id")
            match_id = req.get("match_id")
            try:
                out = play_game_coalesced(
                    coalescer,
                    white_path=req["white"],
                    black_path=req["black"],
                    match_id=match_id,
                    max_plies=int(req.get("max_plies", 500)),
                    move_timeout_ms=int(req.get("move_timeout_ms", 5500)),
                    white_name=req.get("white_name"),
                    black_name=req.get("black_name"),
                )
                payload = {k: v for k, v in out.items() if k != "pgn"}
                payload["id"] = req_id
            except Exception as e:
                payload = {
                    "id": req_id,
                    "match_id": match_id,
                    "result": "draw",
                    "reason": "crash",
                    "plies": 0,
                    "moves": [],
                    "pgn_result": "1/2-1/2",
                    "error": f"daemon play_game error: {e}",
                }
            wfile.write((json.dumps(payload) + "\n").encode("utf-8"))
    except (BrokenPipeError, ConnectionResetError):
        pass
    except Exception as e:
        _log(f"client {peer} error: {e}")
    finally:
        try: rfile.close()
        except Exception: pass
        try: wfile.close()
        except Exception: pass
        try: conn.close()
        except Exception: pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--socket", default=DEFAULT_SOCKET)
    ap.add_argument("--backlog", type=int, default=64)
    ap.add_argument("--max-batch", type=int, default=32,
                    help="dispatcher max coalesced batch size")
    ap.add_argument("--deadline-ms", type=float, default=0.5,
                    help="dispatcher drain deadline once first request arrives")
    args = ap.parse_args()

    _log("preloading dojo_ref (one CUDA context, shared across threads)")
    _ = dojo_ref.Position.startpos().legal_moves()
    _log("dojo_ref ready")

    coalescer = BatchCoalescer(
        max_batch=args.max_batch, deadline_ms=args.deadline_ms,
    )
    coalescer.start()
    _log(
        f"coalescer running (max_batch={args.max_batch}, "
        f"deadline_ms={args.deadline_ms})"
    )

    sock_path = Path(args.socket)
    if sock_path.exists():
        sock_path.unlink()
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.bind(str(sock_path))
    os.chmod(str(sock_path), 0o666)
    sock.listen(args.backlog)
    _log(f"listening on {sock_path}")

    threads: list[threading.Thread] = []
    try:
        while True:
            conn, _ = sock.accept()
            peer = f"fd{conn.fileno()}"
            t = threading.Thread(
                target=handle_client, args=(coalescer, conn, peer), daemon=True,
            )
            t.start()
            threads.append(t)
            if len(threads) > 64:
                threads = [t for t in threads if t.is_alive()]
    except KeyboardInterrupt:
        _log("SIGINT — shutdown")
        _log(f"coalescer stats: {coalescer.stats_snapshot()}")
    finally:
        coalescer.stop()
        try: sock.close()
        except Exception: pass
        try: sock_path.unlink()
        except Exception: pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
