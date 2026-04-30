#!/usr/bin/env python3
"""cuda_referee.py — GPU referee adapter using dojo_ref (combat branch).

Same I/O contract as bridges/arbiter_referee.mjs. All chess work
executes on the GPU via librefcuda.so; this script is pure
orchestration.

I/O: line-delimited JSON over stdin/stdout.
  in:  {"id": int, "moves": ["e2e4", "e7e5", ...], "max_plies": 500}
  out: {"id": int, "ok": bool, "plies_replayed": int,
        "terminal": "checkmate"|"stalemate"|"threefold"|"fifty"|
                    "insufficient"|"max_plies"|"undecided"|null,
        "winner_after_terminal": "white"|"black"|"draw"|null,
        "error": str|None}

The GPU referee (dojo_ref) detects checkmate / stalemate natively.
Threefold / fifty-move / insufficient-material are NOT in the dojo_ref
ABI today, so we delegate the host-side bookkeeping to HostState
(live_arbiter/gpu_state.py) — the same incremental tracker used by
the live game-serving path. Castling rights and the en-passant target
are mutated as moves are applied, exactly as arbiter chess-engine.js
does, so the threefold key matches arbiter getBoardKey() byte-for-byte.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Single source of truth for arbiter-equivalent bookkeeping (castling,
# ep, halfmove, insufficient, board_key) — imported from the live path.
_LIVE_ARBITER = Path(__file__).resolve().parent.parent / "live_arbiter"
if str(_LIVE_ARBITER) not in sys.path:
    sys.path.insert(0, str(_LIVE_ARBITER))
from gpu_state import HostState  # noqa: E402

import dojo_ref  # noqa: E402


# --------------------------------------------------------------------------
# Replay loop. dojo_ref does all chess COMPUTE; HostState does the
# bookkeeping the dojo_ref ABI does not expose.
# --------------------------------------------------------------------------

def replay_game(uci_moves: list[str], max_plies: int = 500) -> dict:
    pos = dojo_ref.Position.startpos()
    hs = HostState()
    plies = 0

    for uci in uci_moves:
        if plies >= max_plies:
            return {"ok": True, "plies_replayed": plies, "terminal": "max_plies",
                    "winner_after_terminal": "draw", "error": None}

        # Termination order matches arbiter: 50-move → insufficient →
        # threefold → no-legal-moves.
        if hs.halfmove >= 100:
            return {"ok": True, "plies_replayed": plies, "terminal": "fifty",
                    "winner_after_terminal": "draw", "error": None}
        if hs.insufficient_material():
            return {"ok": True, "plies_replayed": plies, "terminal": "insufficient",
                    "winner_after_terminal": "draw", "error": None}
        if hs.record_position_for_threefold() >= 3:
            return {"ok": True, "plies_replayed": plies, "terminal": "threefold",
                    "winner_after_terminal": "draw", "error": None}

        legal = pos.legal_moves()
        if not legal:
            in_check = pos.is_check()
            winner = ("black" if hs.side == "w" else "white") if in_check else "draw"
            return {"ok": True, "plies_replayed": plies,
                    "terminal": "checkmate" if in_check else "stalemate",
                    "winner_after_terminal": winner, "error": None}

        if uci not in legal:
            return {"ok": False, "plies_replayed": plies, "terminal": None,
                    "winner_after_terminal": None,
                    "error": f"illegal move {uci} at ply {plies} (side={hs.side})"}

        try:
            pos = pos.make_move(uci)
        except Exception as e:
            return {"ok": False, "plies_replayed": plies, "terminal": None,
                    "winner_after_terminal": None,
                    "error": f"make_move({uci}) raised: {e}"}
        hs.apply_uci(uci)
        plies += 1

    # End of move list — same checks the arbiter would run at the start
    # of the next ply if the game continued. Threefold uses peek (no
    # mutation) since recording was done at the top of each iteration.
    if hs.halfmove >= 100:
        return {"ok": True, "plies_replayed": plies, "terminal": "fifty",
                "winner_after_terminal": "draw", "error": None}
    if hs.insufficient_material():
        return {"ok": True, "plies_replayed": plies, "terminal": "insufficient",
                "winner_after_terminal": "draw", "error": None}
    if hs.history[hs.board_key()] + 1 >= 3:
        return {"ok": True, "plies_replayed": plies, "terminal": "threefold",
                "winner_after_terminal": "draw", "error": None}
    if pos.is_checkmate():
        winner = "black" if hs.side == "w" else "white"
        return {"ok": True, "plies_replayed": plies, "terminal": "checkmate",
                "winner_after_terminal": winner, "error": None}
    if pos.is_stalemate():
        return {"ok": True, "plies_replayed": plies, "terminal": "stalemate",
                "winner_after_terminal": "draw", "error": None}
    return {"ok": True, "plies_replayed": plies, "terminal": "undecided",
            "winner_after_terminal": None, "error": None}


def main():
    sys.stderr.write("[cuda_referee] dojo_ref loaded\n")
    sys.stderr.write("[cuda_referee] ready\n")
    sys.stderr.flush()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception as e:
            print(json.dumps({"error": f"bad json: {e}"}), flush=True)
            continue
        out = replay_game(req.get("moves", []),
                          req.get("max_plies", 500))
        out["id"] = req.get("id")
        print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
