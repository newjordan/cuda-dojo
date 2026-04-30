#!/usr/bin/env python3
"""live_match.py — drive ONE live chess game between two JS fighters
where the referee runs on the GPU (dojo_ref / librefcuda.so).

This is a drop-in replacement for AgentChess match-processor's CPU
referee path. The fighter sandboxing (Docker, --network none,
--read-only, --memory limit, --pids-limit) mirrors the arbiter's
sandboxed-referee.js. The CHESS COMPUTE — legal-move generation,
make-move, check/checkmate/stalemate detection — runs on the GPU
through dojo_ref. Host-side bookkeeping (FEN formatting, castling
right tracking, threefold/fifty/insufficient draws) is state-format
only; no chess decisions on host.

Usage (CLI, JSON-out):
    python3 live_match.py \
        --white /path/to/whiteFighter.js \
        --black /path/to/blackFighter.js \
        --match-id myMatch01 \
        --max-plies 500 \
        --move-timeout-ms 5500

Or as a library:
    from live_match import play_game
    result = play_game(white_path=..., black_path=...)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import re
import uuid
from pathlib import Path

# dojo_ref is the GPU referee (built into librefcuda.so via maturin).
import dojo_ref

from gpu_state import HostState

UCI_RE = re.compile(r"^[a-h][1-8][a-h][1-8][qrbn]?$")

SANDBOX_IMAGE = os.environ.get("CUDA_SANDBOX_IMAGE", "agentchess-sandbox:latest")
# AGENT_* env vars are shared with the lifted-caps arbiter bridge so
# both sides of the head-to-head run under identical docker resources.
# Defaults match prod (cpus=0.5, mem=256m).
AGENT_CPUS = os.environ.get("AGENT_CPUS", "0.5")
AGENT_MEMORY = os.environ.get("AGENT_MEMORY", os.environ.get("CUDA_AGENT_MEMORY", "256m"))
AGENT_PIDS_LIMIT = os.environ.get("AGENT_PIDS_LIMIT", "32")
AGENT_TMPFS_SIZE = os.environ.get("AGENT_TMPFS_SIZE", "10m")
DEFAULT_MEMORY = AGENT_MEMORY  # back-compat alias


# ----------------------------------------------------------------------
# Docker fighter sandbox (mirrors sandboxed-referee.js startContainer
# /getAgentMove / stopContainer). KEEP IDENTICAL knobs so fighter
# behavior is unchanged from production.
# ----------------------------------------------------------------------

def _detect_ext(code: str) -> str:
    if "require(" in code and "import " not in code:
        return ".js"
    return ".mjs"


def start_container(match_id: str, color: str, agent_code: str) -> dict:
    name = f"cuda-{match_id}-{color}"
    ext = _detect_ext(agent_code)

    # Spawn the sandbox container, sleep infinity so we can exec into it
    # repeatedly. Same docker flags as production arbiter.
    subprocess.run(
        [
            "docker", "run", "-d",
            "--name", name,
            "--network", "none",
            "--read-only",
            "--memory", AGENT_MEMORY,
            "--cpus", AGENT_CPUS,
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges",
            "--pids-limit", AGENT_PIDS_LIMIT,
            "--tmpfs", f"/tmp:size={AGENT_TMPFS_SIZE},nodev,nosuid",
            SANDBOX_IMAGE,
            "sleep", "infinity",
        ],
        check=True, capture_output=True, timeout=15,
    )

    # Pipe agent code into /tmp/agent.<ext> via docker exec stdin.
    p = subprocess.run(
        ["docker", "exec", "-i", name, "sh", "-c", f"cat > /tmp/agent{ext}"],
        input=agent_code.encode("utf-8"),
        check=True, capture_output=True, timeout=10,
    )
    return {"name": name, "ext": ext}


def get_agent_move(name: str, ext: str, fen: str, timeout_ms: int) -> str:
    """Send FEN to fighter on stdin, capture UCI on stdout. Same
    contract as arbiter's getAgentMove."""
    timeout_sec = max(1, (timeout_ms + 999) // 1000)
    try:
        p = subprocess.run(
            ["docker", "exec", "-i", name,
             "timeout", str(timeout_sec),
             "node", f"/tmp/agent{ext}"],
            input=(fen + "\n").encode("utf-8"),
            capture_output=True,
            timeout=(timeout_ms + 2000) / 1000.0,
        )
    except subprocess.TimeoutExpired:
        return "__TIMEOUT__"

    if p.returncode == 124:  # GNU timeout fired
        return "__TIMEOUT__"
    if p.returncode == 137:  # OOM kill
        return "__OOM__"
    if p.returncode != 0:
        # Crash: log to stderr, return sentinel.
        sys.stderr.write(
            f"[live_match] CRASH {name} exit={p.returncode} "
            f"stderr={p.stderr.decode('utf-8', 'replace')[:200]}\n"
        )
        return "__CRASH__"
    return p.stdout.decode("utf-8", "replace").strip()


def stop_container(name: str) -> None:
    subprocess.run(
        ["docker", "rm", "-f", name],
        capture_output=True, timeout=15,
    )


# ----------------------------------------------------------------------
# Game loop. Referee = dojo_ref (GPU). Fighter sandbox = Docker.
# ----------------------------------------------------------------------

def play_game(
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

    # Spin GPU position + host bookkeeping side-channel.
    pos = dojo_ref.Position.startpos()
    host = HostState()
    move_log: list[str] = []
    t_start = time.monotonic()

    # Spawn Docker sandboxes (mirrors arbiter exactly).
    white = start_container(match_id, "white", white_code)
    black = start_container(match_id, "black", black_code)

    try:
        for ply in range(max_plies):
            is_white_turn = host.side == "w"
            agent = white if is_white_turn else black

            # ------------------------------------------------------
            # Draw checks BEFORE asking fighter (arbiter ordering).
            # ------------------------------------------------------
            if host.halfmove >= 100:
                return _result(
                    "draw", "50-move", ply, move_log,
                    white_name, black_name, t_start, match_id, max_plies,
                )
            if host.insufficient_material():
                return _result(
                    "draw", "insufficient", ply, move_log,
                    white_name, black_name, t_start, match_id, max_plies,
                )
            count = host.record_position_for_threefold()
            if count >= 3:
                return _result(
                    "draw", "threefold", ply, move_log,
                    white_name, black_name, t_start, match_id, max_plies,
                )

            # ------------------------------------------------------
            # GPU: legal move list. If empty -> mate or stalemate.
            # ------------------------------------------------------
            legal = pos.legal_moves()
            if not legal:
                if pos.is_check():
                    winner = "black" if is_white_turn else "white"
                    return _result(
                        winner, "checkmate", ply, move_log,
                        white_name, black_name, t_start, match_id, max_plies,
                    )
                return _result(
                    "draw", "stalemate", ply, move_log,
                    white_name, black_name, t_start, match_id, max_plies,
                )

            # ------------------------------------------------------
            # Ask the fighter.
            # ------------------------------------------------------
            fen = host.fen()
            uci = get_agent_move(agent["name"], agent["ext"], fen, move_timeout_ms)

            if uci in ("__CRASH__", "__OOM__"):
                # One retry with a fresh container, mirroring arbiter.
                code = white_code if is_white_turn else black_code
                stop_container(agent["name"])
                fresh = start_container(
                    match_id + "r", "white" if is_white_turn else "black", code
                )
                if is_white_turn:
                    white = fresh; agent = white
                else:
                    black = fresh; agent = black
                uci = get_agent_move(
                    agent["name"], agent["ext"], fen, move_timeout_ms
                )

            if uci == "__TIMEOUT__":
                winner = "black" if is_white_turn else "white"
                return _result(
                    winner, "timeout", ply, move_log,
                    white_name, black_name, t_start, match_id, max_plies,
                )
            if uci in ("__CRASH__", "__OOM__"):
                winner = "black" if is_white_turn else "white"
                reason = "oom" if uci == "__OOM__" else "crash"
                return _result(
                    winner, reason, ply, move_log,
                    white_name, black_name, t_start, match_id, max_plies,
                )

            if not UCI_RE.match(uci):
                winner = "black" if is_white_turn else "white"
                return _result(
                    winner, "invalid_format", ply, move_log,
                    white_name, black_name, t_start, match_id, max_plies,
                )

            # ------------------------------------------------------
            # GPU: legality check.
            # ------------------------------------------------------
            if uci not in legal:
                winner = "black" if is_white_turn else "white"
                return _result(
                    winner, "illegal", ply, move_log,
                    white_name, black_name, t_start, match_id, max_plies,
                )

            # ------------------------------------------------------
            # GPU: apply move; host: mirror state.
            # ------------------------------------------------------
            move_log.append(uci)
            pos = pos.make_move(uci)
            host.apply_uci(uci)

        # Max plies reached.
        return _result(
            "draw", "max_plies", max_plies, move_log,
            white_name, black_name, t_start, match_id, max_plies,
        )

    finally:
        stop_container(white["name"])
        stop_container(black["name"])


def _result(result, reason, plies, moves, white, black,
            t_start, match_id, max_plies):
    if result == "white":
        pgn_result = "1-0"
    elif result == "black":
        pgn_result = "0-1"
    else:
        pgn_result = "1/2-1/2"

    pgn = build_pgn_uci(white, black, moves, pgn_result, reason, match_id)
    return {
        "match_id": match_id,
        "result": result,
        "pgn_result": pgn_result,
        "reason": reason,
        "plies": plies,
        "moves": moves,
        "wall_seconds": round(time.monotonic() - t_start, 3),
        "max_plies": max_plies,
        "referee": "cuda",
        "pgn": pgn,
    }


def build_pgn_uci(white_name, black_name, moves, result, reason, match_id):
    """Minimal PGN with UCI moves. We deliberately don't compute SAN
    here because that would need a CPU chess library on our side. The
    comparison harness compares result/reason/plies/UCI move list,
    which is the substantive game equivalence."""
    headers = [
        f'[Event "ChessAgents Arena (CUDA referee)"]',
        f'[Site "GPU Match Engine"]',
        f'[Date "{time.strftime("%Y.%m.%d")}"]',
        f'[White "{white_name}"]',
        f'[Black "{black_name}"]',
        f'[Result "{result}"]',
        f'[Termination "{reason}"]',
        f'[MoveFormat "uci"]',
        f'[MatchId "{match_id}"]',
    ]
    move_text = ""
    for i, m in enumerate(moves):
        if i % 2 == 0:
            move_text += f"{i // 2 + 1}. "
        move_text += m + " "
    move_text += result
    return "\n".join(headers) + "\n\n" + move_text.strip() + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--white", required=True, help="path to white fighter .js/.mjs")
    ap.add_argument("--black", required=True, help="path to black fighter .js/.mjs")
    ap.add_argument("--match-id", default=None)
    ap.add_argument("--max-plies", type=int, default=500)
    ap.add_argument("--move-timeout-ms", type=int, default=5500)
    ap.add_argument("--white-name", default=None)
    ap.add_argument("--black-name", default=None)
    ap.add_argument("--out-json", default=None,
                    help="write JSON result here (default: stdout)")
    ap.add_argument("--out-pgn", default=None,
                    help="write PGN here (default: skip)")
    args = ap.parse_args()

    out = play_game(
        white_path=args.white,
        black_path=args.black,
        match_id=args.match_id,
        max_plies=args.max_plies,
        move_timeout_ms=args.move_timeout_ms,
        white_name=args.white_name,
        black_name=args.black_name,
    )
    payload = {k: v for k, v in out.items() if k != "pgn"}
    if args.out_pgn:
        Path(args.out_pgn).write_text(out["pgn"])
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(payload, indent=2))
    else:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
