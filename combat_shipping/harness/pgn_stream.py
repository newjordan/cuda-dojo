"""pgn_stream.py — minimal PGN streaming + SAN→UCI matching.

Pure text processing. Imports dojo_ref only to ask the GPU for legal
moves at each position (used to disambiguate SAN). No host-side chess
move generation.

This is intentionally a shrunken copy of cuda/engine/parity_test.py —
combat_shipping should run from a clean clone without any cuda/engine/
dependencies bleeding in.
"""
from __future__ import annotations

import re
import random
from pathlib import Path

import dojo_ref


# --------------------------------------------------------------------------
# PGN parsing — text only.
# --------------------------------------------------------------------------

def stream_pgn(path: Path):
    """Yield {'headers': dict, 'moves': [san...]} per game. Streams line
    by line so multi-GB PGNs don't blow memory."""
    headers: dict[str, str] = {}
    in_moves = False
    move_buf: list[str] = []
    header_re = re.compile(r'\[(\w+)\s+"(.*)"\]')
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                if in_moves and move_buf:
                    yield _flush(headers, move_buf)
                    headers, move_buf, in_moves = {}, [], False
                continue
            if stripped.startswith("["):
                if in_moves:
                    yield _flush(headers, move_buf)
                    headers, move_buf, in_moves = {}, [], False
                m = header_re.match(stripped)
                if m:
                    headers[m.group(1)] = m.group(2)
            else:
                in_moves = True
                move_buf.append(stripped)
        if in_moves and move_buf:
            yield _flush(headers, move_buf)


def _flush(headers, move_buf):
    mt = " ".join(move_buf)
    mt = re.sub(r"\{[^}]*\}", " ", mt)
    mt = re.sub(r"\$\d+", " ", mt)
    mt = re.sub(r"\d+\.(\.\.)?", " ", mt)
    mt = re.sub(r"(1-0|0-1|1/2-1/2|\*)\s*$", " ", mt)
    tokens = [t for t in mt.split() if t]
    return {"headers": dict(headers), "moves": tokens}


def reservoir_sample(stream, k: int, seed: int = 42, max_scan: int | None = None):
    rng = random.Random(seed)
    res, n = [], 0
    for item in stream:
        n += 1
        if len(res) < k:
            res.append(item)
        else:
            j = rng.randint(0, n - 1)
            if j < k:
                res[j] = item
        if max_scan is not None and n >= max_scan:
            break
    return res, n


# --------------------------------------------------------------------------
# Host-side piece-map for SAN disambiguation. Pure bookkeeping over
# state we already needed for the cuda_referee bridge.
# --------------------------------------------------------------------------

def _initial_piecemap() -> dict[int, str]:
    m: dict[int, str] = {}
    back = ['R', 'N', 'B', 'Q', 'K', 'B', 'N', 'R']
    for f in range(8):
        m[f] = back[f]
        m[8 + f] = 'P'
        m[48 + f] = 'p'
        m[56 + f] = back[f].lower()
    return m


def _sq(sq: str) -> int:
    return (ord(sq[1]) - ord('1')) * 8 + (ord(sq[0]) - ord('a'))


def _update_piecemap(pmap: dict[int, str], uci: str, mover_white: bool):
    f_sq, t_sq = _sq(uci[0:2]), _sq(uci[2:4])
    promo = uci[4:5] if len(uci) >= 5 else ""
    piece = pmap.get(f_sq, "")
    if piece in ("K", "k") and abs((t_sq & 7) - (f_sq & 7)) == 2:
        del pmap[f_sq]
        pmap[t_sq] = piece
        rank_off = f_sq & ~7
        if (t_sq & 7) > (f_sq & 7):
            rook_from, rook_to = rank_off + 7, rank_off + 5
        else:
            rook_from, rook_to = rank_off + 0, rank_off + 3
        rook = pmap.pop(rook_from, "R" if mover_white else "r")
        pmap[rook_to] = rook
        return
    if piece in ("P", "p") and (t_sq & 7) != (f_sq & 7) and t_sq not in pmap:
        ep_target = (f_sq & ~7) | (t_sq & 7)
        pmap.pop(ep_target, None)
    if f_sq in pmap:
        del pmap[f_sq]
    if promo:
        pmap[t_sq] = promo.upper() if mover_white else promo.lower()
    else:
        pmap[t_sq] = piece


SAN_RE = re.compile(
    r"^"
    r"(?P<piece>[NBRQK])?"
    r"(?P<from_file>[a-h])?"
    r"(?P<from_rank>[1-8])?"
    r"(?P<capture>x)?"
    r"(?P<to>[a-h][1-8])"
    r"(?:=(?P<promo>[NBRQ]))?"
    r"(?P<check>[+#])?"
    r"$"
)


def _san_to_uci(san: str, pmap: dict[int, str], legal_ucis: list[str],
                mover_white: bool) -> str | None:
    s = san.strip()
    while s and s[-1] in "+#":
        s = s[:-1]
    if s in ("O-O", "0-0"):
        for u in legal_ucis:
            if mover_white and u in ("e1g1",): return u
            if not mover_white and u in ("e8g8",): return u
        return None
    if s in ("O-O-O", "0-0-0"):
        for u in legal_ucis:
            if mover_white and u in ("e1c1",): return u
            if not mover_white and u in ("e8c8",): return u
        return None
    m = SAN_RE.match(s)
    if not m:
        return None
    piece = m.group("piece") or "P"
    from_file = m.group("from_file")
    from_rank = m.group("from_rank")
    to_sq = m.group("to")
    promo = m.group("promo")
    if not m.group("piece") and m.group("from_file") and m.group("capture"):
        piece = "P"
    target_piece = piece if mover_white else piece.lower()
    candidates = []
    for u in legal_ucis:
        u_from, u_to = u[0:2], u[2:4]
        u_promo = u[4:5] if len(u) >= 5 else ""
        if u_to != to_sq: continue
        if promo and u_promo.upper() != promo: continue
        if not promo and u_promo: continue
        f_idx = _sq(u_from)
        if pmap.get(f_idx) != target_piece: continue
        if from_file and u_from[0] != from_file: continue
        if from_rank and u_from[1] != from_rank: continue
        candidates.append(u)
    if len(candidates) == 1:
        return candidates[0]
    return None


def game_san_to_uci(san_moves: list[str]) -> tuple[list[str], int, str]:
    """Convert a SAN move list to UCI by querying the GPU for legal
    moves at each ply. Returns (uci_list, ply_failed_at_or_-1, err_msg).
    """
    pos = dojo_ref.Position.startpos()
    pmap = _initial_piecemap()
    mover_white = True
    out: list[str] = []
    for i, san in enumerate(san_moves):
        legal = pos.legal_moves()
        if not legal:
            return out, i, "GPU reports no legal moves but PGN has more"
        uci = _san_to_uci(san, pmap, legal, mover_white)
        if uci is None:
            return out, i, f"SAN→UCI failed for {san!r}"
        if uci not in legal:
            return out, i, f"matched UCI {uci} not in GPU legal set"
        try:
            pos = pos.make_move(uci)
        except Exception as e:
            return out, i, f"make_move({uci}) raised: {e}"
        _update_piecemap(pmap, uci, mover_white)
        mover_white = not mover_white
        out.append(uci)
    return out, -1, ""
