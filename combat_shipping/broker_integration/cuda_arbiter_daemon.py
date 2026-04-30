#!/usr/bin/env python3
"""cuda_arbiter_daemon.py — persistent CUDA arbiter service.

Long-lived process that loads dojo_ref ONCE at startup, then accepts
game requests over a unix socket. Each request runs the same logic as
live_match.play_game but inside the persistent process — no per-game
python startup, no per-game CUDA context init, no per-game library
import. The CUDA context is shared across all in-flight matches.

Protocol (line-delimited JSON over the unix socket):
  in:  {"id": int, "match_id": str, "white": "/path/white.js",
         "black": "/path/black.js", "white_lang": "js"|"py",
         "black_lang": "js"|"py", "white_name": str, "black_name": str,
         "max_plies": int, "move_timeout_ms": int}
  out: {"id": int, "match_id": str, "result": "white"|"black"|"draw",
         "pgn_result": "1-0"|"0-1"|"1/2-1/2",
         "reason": str, "plies": int, "moves": [...],
         "wall_seconds": float, "max_plies": int, "referee": "cuda"}

Concurrency: each connection runs in its own thread. play_game spends
most of its wall time in subprocess.run waiting on docker — releases
the GIL, so threading is sufficient for this workload. dojo_ref FFI
calls are short and serialize harmlessly under the CUDA context.

Container-name collisions are avoided because live_match.play_game
names containers `cuda-{match_id}-{color}` and match_ids come from
match-processor's broker (UUIDs).
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import threading
import time
from pathlib import Path

# Make live_arbiter importable so we reuse play_game verbatim.
_HERE = Path(__file__).resolve().parent
_LIVE_ARBITER = _HERE.parent / "live_arbiter"
if str(_LIVE_ARBITER) not in sys.path:
    sys.path.insert(0, str(_LIVE_ARBITER))

import dojo_ref  # noqa: E402
from live_match import play_game  # noqa: E402


DEFAULT_SOCKET = os.environ.get("CUDA_ARBITER_SOCKET", "/tmp/cuda_arbiter.sock")


def _log(msg: str) -> None:
    sys.stderr.write(f"[daemon {time.strftime('%H:%M:%S')}] {msg}\n")
    sys.stderr.flush()


def handle_client(conn: socket.socket, peer: str) -> None:
    """One TCP-equivalent connection. Caller may pipeline multiple game
    requests on the same connection; each is one JSON line in, one out."""
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
                out = play_game(
                    white_path=req["white"],
                    black_path=req["black"],
                    match_id=match_id,
                    max_plies=int(req.get("max_plies", 500)),
                    move_timeout_ms=int(req.get("move_timeout_ms", 5500)),
                    white_name=req.get("white_name"),
                    black_name=req.get("black_name"),
                )
                # Drop pgn — JS side rebuilds via prod pgn-builder for SAN.
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
    ap.add_argument("--socket", default=DEFAULT_SOCKET,
                    help=f"Unix socket path (default {DEFAULT_SOCKET})")
    ap.add_argument("--backlog", type=int, default=64,
                    help="listen backlog")
    args = ap.parse_args()

    _log("preloading dojo_ref (one CUDA context, shared across threads)")
    _ = dojo_ref.Position.startpos().legal_moves()
    _log("dojo_ref ready")

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
            t = threading.Thread(target=handle_client, args=(conn, peer), daemon=True)
            t.start()
            threads.append(t)
            if len(threads) > 64:
                threads = [t for t in threads if t.is_alive()]
    except KeyboardInterrupt:
        _log("SIGINT — shutdown")
    finally:
        try: sock.close()
        except Exception: pass
        try: sock_path.unlink()
        except Exception: pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
