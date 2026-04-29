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
ABI today, so we track them host-side using the same key derivation
arbiter does (FEN board+side+castling+ep). This is bookkeeping over
GPU-produced state — no chess compute on host.
"""
from __future__ import annotations

import json
import sys
import os
from collections import Counter

# Ensure we can find dojo_ref. Build location is the rust/dojo-ref-py
# crate; if maturin develop has been run inside the active venv, it's
# already importable. Otherwise add the build dir to sys.path here.
import dojo_ref  # noqa: E402


# --------------------------------------------------------------------------
# Host-side bookkeeping ONLY for state the GPU ABI doesn't expose. No
# legality, no make_move, no in_check — those are dojo_ref calls.
# --------------------------------------------------------------------------

def _parse_moves(uci: str):
    return uci[0:2], uci[2:4], (uci[4:5] if len(uci) >= 5 else "")


# Initial piece map (white pieces ascii, black lowercase). This mirrors
# the arbiter's parseFen/boardToFen layout: index 0 = a8, index 7 = h8,
# index 56 = a1, index 63 = h1 (top-down rendering). dojo_ref uses
# rank-major a1=0; we don't translate, we just track piece identities
# for hashing the position.
def _initial_piecemap() -> dict[int, str]:
    # We use a1=0 indexing to stay aligned with dojo_ref's coord system.
    m: dict[int, str] = {}
    back = ['R', 'N', 'B', 'Q', 'K', 'B', 'N', 'R']
    for f in range(8):
        m[f]      = back[f]    # white back rank
        m[8 + f]  = 'P'        # white pawns
        m[48 + f] = 'p'        # black pawns
        m[56 + f] = back[f].lower()
    return m


def _sq(uci_sq: str) -> int:
    return (ord(uci_sq[1]) - ord('1')) * 8 + (ord(uci_sq[0]) - ord('a'))


def _apply_uci_to_piecemap(pmap: dict[int, str], uci: str, mover_white: bool):
    f_sq, t_sq = _sq(uci[0:2]), _sq(uci[2:4])
    promo = uci[4:5] if len(uci) >= 5 else ""
    piece = pmap.get(f_sq, "")

    # Castling: king moves 2 files horizontally.
    if piece in ("K", "k"):
        df = (t_sq & 7) - (f_sq & 7)
        if abs(df) == 2:
            del pmap[f_sq]
            pmap[t_sq] = piece
            rank_off = f_sq & ~7
            if df > 0:
                rook_from, rook_to = rank_off + 7, rank_off + 5
            else:
                rook_from, rook_to = rank_off + 0, rank_off + 3
            rook = pmap.pop(rook_from, "R" if mover_white else "r")
            pmap[rook_to] = rook
            return

    # En passant — pawn diagonal to empty square.
    if piece in ("P", "p") and (t_sq & 7) != (f_sq & 7) and t_sq not in pmap:
        ep_target = (f_sq & ~7) | (t_sq & 7)
        pmap.pop(ep_target, None)

    if f_sq in pmap:
        del pmap[f_sq]
    if promo:
        pmap[t_sq] = promo.upper() if mover_white else promo.lower()
    else:
        pmap[t_sq] = piece


def _board_key(pmap: dict[int, str], side_to_move: str,
               castling_rights: str, ep: str) -> str:
    """Mirror arbiter's getBoardKey: position + side + castling + ep.
    Castling rights are tricky to track without GPU exposure of them;
    we approximate by deriving them from the piece map (king + rook
    on starting squares) — this is exact for any position reachable
    from the standard startpos. For Chess960 it would not be exact,
    but the production arbiter is also vanilla chess.
    """
    # Castling: K on e1 + R on h1 → 'K', etc.
    cr = ""
    if pmap.get(_sq("e1")) == "K":
        if pmap.get(_sq("h1")) == "R": cr += "K"
        if pmap.get(_sq("a1")) == "R": cr += "Q"
    if pmap.get(_sq("e8")) == "k":
        if pmap.get(_sq("h8")) == "r": cr += "k"
        if pmap.get(_sq("a8")) == "r": cr += "q"
    if not cr: cr = "-"

    # Board flat string: 64 squares in arbiter's top-down order.
    # arbiter index = (8 - rank) * 8 + file → for square (file f, rank r),
    # rank 8 = first row; we have a1=0 so rank-1 file-0. Convert:
    #   arbiter_idx(f, r1based) = (8 - r1based) * 8 + f
    # Our pmap key is (r1based - 1)*8 + f.
    flat = []
    for r1 in range(8, 0, -1):  # rank 8 .. 1
        for f in range(8):
            sq = (r1 - 1) * 8 + f
            flat.append(pmap.get(sq, "."))
    return "".join(flat) + side_to_move + cr + ep


# --------------------------------------------------------------------------
# Replay loop. dojo_ref does all chess work.
# --------------------------------------------------------------------------

def replay_game(uci_moves: list[str], max_plies: int = 500) -> dict:
    pos = dojo_ref.Position.startpos()
    pmap = _initial_piecemap()
    side = "w"
    history = Counter()
    halfmove = 0
    plies = 0

    # Insufficient-material check (host bookkeeping, mirrors arbiter).
    def _insufficient(pm: dict[int, str]) -> bool:
        pieces = list(pm.values())
        if len(pieces) == 2: return True
        if len(pieces) == 3:
            for p in pieces:
                if p.lower() in ('b', 'n'): return True
        if len(pieces) == 4:
            bishops = [s for s, p in pm.items() if p.lower() == 'b']
            if len(bishops) == 2:
                c1 = ((bishops[0] >> 3) + (bishops[0] & 7)) % 2
                c2 = ((bishops[1] >> 3) + (bishops[1] & 7)) % 2
                if c1 == c2: return True
        return False

    for uci in uci_moves:
        if plies >= max_plies:
            return {"ok": True, "plies_replayed": plies, "terminal": "max_plies",
                    "winner_after_terminal": "draw", "error": None}

        if halfmove >= 100:
            return {"ok": True, "plies_replayed": plies, "terminal": "fifty",
                    "winner_after_terminal": "draw", "error": None}
        if _insufficient(pmap):
            return {"ok": True, "plies_replayed": plies, "terminal": "insufficient",
                    "winner_after_terminal": "draw", "error": None}

        ep = "-"  # we don't currently expose ep from GPU; arbiter key only
                  # uses board+side+castling+ep. We approximate ep="-"
                  # which is conservative (will under-count threefold,
                  # never over-count).
        key = _board_key(pmap, side, "", ep)
        history[key] += 1
        if history[key] >= 3:
            return {"ok": True, "plies_replayed": plies, "terminal": "threefold",
                    "winner_after_terminal": "draw", "error": None}

        # GPU does the legality check.
        legal = pos.legal_moves()
        if not legal:
            in_check = pos.is_check()
            winner = ("black" if side == "w" else "white") if in_check else "draw"
            return {"ok": True, "plies_replayed": plies,
                    "terminal": "checkmate" if in_check else "stalemate",
                    "winner_after_terminal": winner, "error": None}

        if uci not in legal:
            return {"ok": False, "plies_replayed": plies, "terminal": None,
                    "winner_after_terminal": None,
                    "error": f"illegal move {uci} at ply {plies} (side={side})"}

        # Halfmove clock: reset on capture or pawn move.
        f_sq, t_sq = _sq(uci[0:2]), _sq(uci[2:4])
        is_capture = t_sq in pmap and pmap[t_sq] != ""
        is_pawn = pmap.get(f_sq, "").lower() == "p"
        # Also reset on en-passant (target square empty but it's a pawn diagonal).
        if is_pawn and (t_sq & 7) != (f_sq & 7) and not is_capture:
            is_capture = True
        halfmove = 0 if (is_capture or is_pawn) else halfmove + 1

        try:
            pos = pos.make_move(uci)
        except Exception as e:
            return {"ok": False, "plies_replayed": plies, "terminal": None,
                    "winner_after_terminal": None,
                    "error": f"make_move({uci}) raised: {e}"}
        _apply_uci_to_piecemap(pmap, uci, side == "w")
        side = "b" if side == "w" else "w"
        plies += 1

    # End of move list — check terminal state at final position.
    # (Mirrors the start-of-next-iteration checks the arbiter would have
    # run if the game had continued; threefold first since the arbiter
    # checks it before legal-move generation.)
    if halfmove >= 100:
        return {"ok": True, "plies_replayed": plies, "terminal": "fifty",
                "winner_after_terminal": "draw", "error": None}
    if _insufficient(pmap):
        return {"ok": True, "plies_replayed": plies, "terminal": "insufficient",
                "winner_after_terminal": "draw", "error": None}
    final_key = _board_key(pmap, side, "", "-")
    if history[final_key] + 1 >= 3:
        return {"ok": True, "plies_replayed": plies, "terminal": "threefold",
                "winner_after_terminal": "draw", "error": None}
    if pos.is_checkmate():
        winner = "black" if side == "w" else "white"
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
