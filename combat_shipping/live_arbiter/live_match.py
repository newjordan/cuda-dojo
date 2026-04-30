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

def _detect_ext(code: str, language: str = "js") -> str:
    """Pick the file extension + interpreter dispatch based on the
    fighter language declared by the broker. CRITICAL: prod fighters
    are a mix of JS and Python; if we get this wrong, a Python fighter
    is run by node (or vice versa) and crashes immediately on syntax.

    `language` comes from the broker as "js" or "py" — same as
    sandboxed-referee.js's `language` parameter. Default 'js' is for
    callers that don't yet pass it (e.g., legacy direct CLI uses)."""
    if language == "py":
        return ".py"
    # JS: distinguish CommonJS vs ESM by source content.
    if "require(" in code and "import " not in code:
        return ".js"
    return ".mjs"


def start_container(match_id: str, color: str, agent_code: str,
                    language: str = "js") -> dict:
    """Start a Docker sandbox for a fighter. Defensive against the
    failure modes we see at production concurrency:
      - stale container with the same name from a prior run that didn't
        clean up (docker daemon retains it briefly after rm)
      - docker daemon load-induced delays on `docker run`
      - transient `docker exec` failures right after container creation

    Raises RuntimeError with a useful message on terminal failure;
    callers (play_game) translate that to a 'crash' result. Does NOT
    use check=True on the run subprocess so we can read returncode +
    stderr to decide retry vs surrender."""
    name = f"cuda-{match_id}-{color}"
    ext = _detect_ext(agent_code, language)

    # Pre-clean any stale container with this name. Idempotent — silently
    # succeeds if no such container exists.
    subprocess.run(
        ["docker", "rm", "-f", name],
        capture_output=True, timeout=10,
    )

    last_err = None
    for attempt in range(3):
        try:
            r = subprocess.run(
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
                capture_output=True, timeout=30,
            )
        except subprocess.TimeoutExpired as e:
            last_err = f"docker run timed out (attempt {attempt+1}/3)"
            time.sleep(0.5)
            continue
        if r.returncode == 0:
            break
        # Conflict on container name → retry after rm.
        stderr = r.stderr.decode("utf-8", "replace")
        last_err = f"docker run rc={r.returncode} stderr={stderr[:300]}"
        if "already in use" in stderr or "Conflict" in stderr:
            subprocess.run(["docker", "rm", "-f", name],
                           capture_output=True, timeout=10)
        time.sleep(0.5)
    else:
        raise RuntimeError(f"start_container failed after 3 attempts: {last_err}")

    # Pipe agent code into /tmp/agent.<ext>. Retry on transient failures
    # (rare race where `docker exec` lands before the runtime is ready).
    last_err = None
    for attempt in range(3):
        try:
            p = subprocess.run(
                ["docker", "exec", "-i", name, "sh", "-c", f"cat > /tmp/agent{ext}"],
                input=agent_code.encode("utf-8"),
                capture_output=True, timeout=20,
            )
        except subprocess.TimeoutExpired:
            last_err = f"docker exec timed out (attempt {attempt+1}/3)"
            time.sleep(0.5)
            continue
        if p.returncode == 0:
            break
        last_err = f"docker exec rc={p.returncode} stderr={p.stderr.decode('utf-8','replace')[:300]}"
        time.sleep(0.5)
    else:
        # Container is up but we can't write the fighter — kill it.
        subprocess.run(["docker", "rm", "-f", name],
                       capture_output=True, timeout=10)
        raise RuntimeError(f"docker exec (write fighter) failed after 3 attempts: {last_err}")

    return {"name": name, "ext": ext}


def get_agent_move(name: str, ext: str, fen: str, timeout_ms: int,
                   diag: dict | None = None) -> str:
    """Send FEN to fighter on stdin, capture UCI on stdout. Same
    contract as arbiter's getAgentMove.

    Interpreter is chosen from the file extension: .py -> python3,
    .js / .mjs -> node. Mirrors sandboxed-referee.js. Getting this
    wrong is the difference between a working fighter and a SyntaxError
    on first move.

    If `diag` is provided, populates it on every call with the actual
    docker exec returncode + stderr/stdout tails so callers can carry
    the failure detail into the result JSON. This is the diagnostic
    bus for crash / timeout / oom triage."""
    timeout_sec = max(1, (timeout_ms + 999) // 1000)
    runtime = "python3" if ext == ".py" else "node"
    try:
        p = subprocess.run(
            ["docker", "exec", "-i", name,
             "timeout", str(timeout_sec),
             runtime, f"/tmp/agent{ext}"],
            input=(fen + "\n").encode("utf-8"),
            capture_output=True,
            timeout=(timeout_ms + 2000) / 1000.0,
        )
    except subprocess.TimeoutExpired as e:
        if diag is not None:
            diag.update({
                "outcome": "host_timeout",
                "returncode": None,
                "stderr_tail": "",
                "stdout_tail": "",
                "container": name,
            })
        return "__TIMEOUT__"

    stderr_tail = p.stderr.decode("utf-8", "replace")[-1500:]
    stdout_tail = p.stdout.decode("utf-8", "replace")[-200:]
    if diag is not None:
        diag.update({
            "returncode": p.returncode,
            "stderr_tail": stderr_tail,
            "stdout_tail": stdout_tail,
            "container": name,
        })

    # If the fighter wrote a valid UCI move to stdout *before* the
    # timeout / non-zero exit, accept it. Python fighters notoriously
    # don't auto-exit after print() if their script has trailing code
    # or non-daemon threads — `timeout` then kills them at the budget
    # boundary and we see rc=124 even though the move is in stdout.
    # Mirror what the production arbiter does in spirit: parse the move
    # if it's there.
    full_stdout = p.stdout.decode("utf-8", "replace")
    last_line = (full_stdout.strip().splitlines() or [""])[-1].strip()
    UCI_LINE = __import__("re").compile(r"^[a-h][1-8][a-h][1-8][qrbn]?$")
    move_in_stdout = bool(UCI_LINE.match(last_line))

    if p.returncode == 124:
        if move_in_stdout:
            if diag is not None:
                diag["outcome"] = "ok_late"
                diag["note"] = "rc=124 but valid move in stdout (fighter slow to exit)"
            return last_line
        if diag is not None: diag["outcome"] = "guest_timeout"
        return "__TIMEOUT__"
    if p.returncode == 137:
        if move_in_stdout:
            if diag is not None:
                diag["outcome"] = "ok_oom"
                diag["note"] = "rc=137 but valid move in stdout (oom-killed after responding)"
            return last_line
        if diag is not None: diag["outcome"] = "oom"
        return "__OOM__"
    if p.returncode != 0:
        if move_in_stdout:
            if diag is not None:
                diag["outcome"] = "ok_crashed"
                diag["note"] = f"rc={p.returncode} but valid move in stdout (fighter crashed after responding)"
            return last_line
        if diag is not None: diag["outcome"] = "crash"
        # Keep the legacy stderr line for `[live_match] CRASH` log scrapers.
        sys.stderr.write(
            f"[live_match] CRASH {name} exit={p.returncode} "
            f"stderr={stderr_tail[:200]}\n"
        )
        return "__CRASH__"
    if diag is not None: diag["outcome"] = "ok"
    return last_line if move_in_stdout else (stdout_tail.strip() if stdout_tail else full_stdout.strip())


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
    white_lang: str = "js",
    black_lang: str = "js",
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

    # Spawn Docker sandboxes (mirrors arbiter exactly). Pass language
    # so .py fighters get python3 and .mjs/.js fighters get node.
    white = start_container(match_id, "white", white_code, white_lang)
    black = start_container(match_id, "black", black_code, black_lang)

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
            # Ask the fighter. `diag` carries the docker exec result
            # detail so we can attach the actual fighter exit signal
            # to the result on crash/timeout/oom.
            # ------------------------------------------------------
            fen = host.fen()
            diag: dict = {}
            uci = get_agent_move(agent["name"], agent["ext"], fen, move_timeout_ms, diag)
            last_diag = dict(diag)
            last_diag["side"] = "white" if is_white_turn else "black"

            if uci in ("__CRASH__", "__OOM__"):
                # One retry with a fresh container, mirroring arbiter.
                code = white_code if is_white_turn else black_code
                lang = white_lang if is_white_turn else black_lang
                stop_container(agent["name"])
                fresh = start_container(
                    match_id + "r", "white" if is_white_turn else "black", code, lang
                )
                if is_white_turn:
                    white = fresh; agent = white
                else:
                    black = fresh; agent = black
                diag = {}
                uci = get_agent_move(
                    agent["name"], agent["ext"], fen, move_timeout_ms, diag
                )
                last_diag = dict(diag)
                last_diag["side"] = "white" if is_white_turn else "black"
                last_diag["after_retry"] = True

            if uci == "__TIMEOUT__":
                winner = "black" if is_white_turn else "white"
                return _result(
                    winner, "timeout", ply, move_log,
                    white_name, black_name, t_start, match_id, max_plies,
                    fighter_diag=last_diag,
                )
            if uci in ("__CRASH__", "__OOM__"):
                winner = "black" if is_white_turn else "white"
                reason = "oom" if uci == "__OOM__" else "crash"
                return _result(
                    winner, reason, ply, move_log,
                    white_name, black_name, t_start, match_id, max_plies,
                    fighter_diag=last_diag,
                )

            if not UCI_RE.match(uci):
                winner = "black" if is_white_turn else "white"
                return _result(
                    winner, "invalid_format", ply, move_log,
                    white_name, black_name, t_start, match_id, max_plies,
                    fighter_diag={"raw_stdout": uci[:200], "side": "white" if is_white_turn else "black"},
                )

            # ------------------------------------------------------
            # GPU: legality check.
            # ------------------------------------------------------
            if uci not in legal:
                winner = "black" if is_white_turn else "white"
                # Diagnostic: capture what the fighter actually returned
                # (possibly truncated by stdout_tail[-200:]) and a sample
                # of legal moves it could have picked. This lets us
                # distinguish "fighter has bug" from "we mis-parsed
                # stdout / clipped a multi-line output."
                illegal_diag = dict(last_diag)
                illegal_diag["uci_returned"] = uci
                illegal_diag["legal_sample"] = legal[:8]
                illegal_diag["legal_count"] = len(legal)
                return _result(
                    winner, "illegal", ply, move_log,
                    white_name, black_name, t_start, match_id, max_plies,
                    fighter_diag=illegal_diag,
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
            t_start, match_id, max_plies, fighter_diag=None):
    if result == "white":
        pgn_result = "1-0"
    elif result == "black":
        pgn_result = "0-1"
    else:
        pgn_result = "1/2-1/2"

    pgn = build_pgn_uci(white, black, moves, pgn_result, reason, match_id)
    out = {
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
    if fighter_diag is not None:
        out["fighter_diag"] = fighter_diag
    return out


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
    ap.add_argument("--white", required=True, help="path to white fighter .js/.mjs/.py")
    ap.add_argument("--black", required=True, help="path to black fighter .js/.mjs/.py")
    ap.add_argument("--match-id", default=None)
    ap.add_argument("--max-plies", type=int, default=500)
    ap.add_argument("--move-timeout-ms", type=int, default=5500)
    ap.add_argument("--white-name", default=None)
    ap.add_argument("--black-name", default=None)
    ap.add_argument("--white-lang", choices=["js", "py"], default="js",
                    help="white fighter language (js|py); MUST match what the broker sent")
    ap.add_argument("--black-lang", choices=["js", "py"], default="js",
                    help="black fighter language (js|py); MUST match what the broker sent")
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
        white_lang=args.white_lang,
        black_lang=args.black_lang,
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
