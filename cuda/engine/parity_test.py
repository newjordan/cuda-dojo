#!/usr/bin/env python3
"""Parity test: replay PGN games through dojo_ref's GPU referee and check
each move's legality + final-state agreement.

For every game in the input PGN:
  1. Initialize a dojo_ref.Position to startpos (GPU).
  2. Track a host-side piece map (just for SAN→UCI disambiguation;
     bookkeeping, not chess work).
  3. For each SAN move in the game's mainline:
     a. Ask the GPU referee for legal UCI moves at the current position.
     b. Parse the SAN: extract piece, source disambiguator, destination,
        promotion, castling.
     c. Match against the legal moves to find the unique UCI.
     d. Verify GPU agrees the move is legal; apply it via Position.make_move.
     e. Update host piece map.
  4. After all moves, check that the recorded Result matches the GPU's
     terminal detection (is_checkmate / is_stalemate / undecided).

Any failure (illegal move per GPU, no SAN match, terminal mismatch) is
reported per game. Pure bookkeeping on host; all chess on GPU.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import dojo_ref


# ---------------------------------------------------------------------------
# PGN parsing — pure text. No chess work.
# ---------------------------------------------------------------------------

def parse_pgn(path: Path) -> list[dict]:
    """Load all games from a small PGN. For huge files use stream_pgn()."""
    return list(stream_pgn(path))


def stream_pgn(path: Path):
    """Yield {'headers': {...}, 'moves': [san...]} dicts for each game.
    Streams line-by-line so multi-GB PGNs don't blow memory.
    """
    headers: dict[str, str] = {}
    in_moves = False
    move_buf: list[str] = []
    header_re = re.compile(r'\[(\w+)\s+"(.*)"\]')
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                # blank line — separates header block / moves block / games
                if in_moves and move_buf:
                    mt = " ".join(move_buf)
                    mt = re.sub(r"\{[^}]*\}", " ", mt)
                    mt = re.sub(r"\$\d+", " ", mt)
                    mt = re.sub(r"\d+\.(\.\.)?", " ", mt)
                    mt = re.sub(r"(1-0|0-1|1/2-1/2|\*)\s*$", " ", mt)
                    tokens = [t for t in mt.split() if t]
                    yield {"headers": headers, "moves": tokens}
                    headers = {}
                    move_buf = []
                    in_moves = False
                continue
            if stripped.startswith("["):
                if in_moves:
                    # Previous game ended without trailing blank line.
                    mt = " ".join(move_buf)
                    mt = re.sub(r"\{[^}]*\}", " ", mt)
                    mt = re.sub(r"\$\d+", " ", mt)
                    mt = re.sub(r"\d+\.(\.\.)?", " ", mt)
                    mt = re.sub(r"(1-0|0-1|1/2-1/2|\*)\s*$", " ", mt)
                    tokens = [t for t in mt.split() if t]
                    yield {"headers": headers, "moves": tokens}
                    headers = {}
                    move_buf = []
                    in_moves = False
                m = header_re.match(stripped)
                if m: headers[m.group(1)] = m.group(2)
            else:
                in_moves = True
                move_buf.append(stripped)
        # tail flush
        if in_moves and move_buf:
            mt = " ".join(move_buf)
            mt = re.sub(r"\{[^}]*\}", " ", mt)
            mt = re.sub(r"\$\d+", " ", mt)
            mt = re.sub(r"\d+\.(\.\.)?", " ", mt)
            mt = re.sub(r"(1-0|0-1|1/2-1/2|\*)\s*$", " ", mt)
            tokens = [t for t in mt.split() if t]
            yield {"headers": headers, "moves": tokens}


def reservoir_sample(stream, k: int, seed: int = 42, max_scan: int | None = None):
    """Reservoir-sample k items from the stream.
    O(k) memory regardless of stream length.
    """
    import random
    rng = random.Random(seed)
    reservoir: list = []
    n = 0
    for item in stream:
        n += 1
        if len(reservoir) < k:
            reservoir.append(item)
        else:
            j = rng.randint(0, n - 1)
            if j < k:
                reservoir[j] = item
        if max_scan is not None and n >= max_scan:
            break
    return reservoir, n


# ---------------------------------------------------------------------------
# Host-side piece map for SAN disambiguation. Pure bookkeeping.
# ---------------------------------------------------------------------------

def initial_piece_map() -> dict[int, str]:
    """64-square map. squares 0..63 with a1=0, h8=63."""
    m: dict[int, str] = {}
    back = ['R', 'N', 'B', 'Q', 'K', 'B', 'N', 'R']
    # White rank 1 (squares 0..7)
    for f in range(8):
        m[f] = back[f]
    # White rank 2 (squares 8..15) — pawns
    for f in range(8):
        m[8 + f] = 'P'
    # Black rank 7 (squares 48..55) — pawns
    for f in range(8):
        m[48 + f] = 'p'
    # Black rank 8 (squares 56..63)
    for f in range(8):
        m[56 + f] = back[f].lower()
    return m


def sq_to_idx(sq: str) -> int:
    return (ord(sq[1]) - ord('1')) * 8 + (ord(sq[0]) - ord('a'))


def idx_to_sq(i: int) -> str:
    return chr(ord('a') + (i & 7)) + chr(ord('1') + (i >> 3))


def update_piece_map(pmap: dict[int, str], uci: str, mover_white: bool) -> None:
    """Apply a UCI move to the piece map. Handles regular moves, captures,
    castling, en passant, promotions. Pure bookkeeping.
    """
    f_sq = sq_to_idx(uci[0:2])
    t_sq = sq_to_idx(uci[2:4])
    promo = uci[4:5] if len(uci) >= 5 else ""

    piece = pmap.get(f_sq, "")

    # Castling: king moves 2 squares horizontally.
    if piece in ("K", "k"):
        if abs((t_sq & 7) - (f_sq & 7)) == 2:
            # Move king
            del pmap[f_sq]
            pmap[t_sq] = piece
            # Move rook
            rank_offset = f_sq & ~7
            if (t_sq & 7) > (f_sq & 7):  # kingside
                rook_from = rank_offset + 7
                rook_to   = rank_offset + 5
            else:                        # queenside
                rook_from = rank_offset + 0
                rook_to   = rank_offset + 3
            rook = pmap.pop(rook_from, "R" if mover_white else "r")
            pmap[rook_to] = rook
            return

    # En passant: pawn diagonal to empty square.
    if piece in ("P", "p") and (t_sq & 7) != (f_sq & 7) and t_sq not in pmap:
        # Captured pawn is on the from-rank, to-file
        ep_target = (f_sq & ~7) | (t_sq & 7)
        pmap.pop(ep_target, None)

    # Regular: clear from, set to (with promotion if any).
    if f_sq in pmap:
        del pmap[f_sq]
    if promo:
        new_piece = promo.upper() if mover_white else promo.lower()
        pmap[t_sq] = new_piece
    else:
        pmap[t_sq] = piece


# ---------------------------------------------------------------------------
# SAN → UCI matching.
# ---------------------------------------------------------------------------

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


def san_to_uci(san: str, pmap: dict[int, str], legal_ucis: list[str],
                mover_white: bool) -> str | None:
    """Return the UCI string in `legal_ucis` that the SAN refers to, or None."""
    s = san.strip()
    # Castling.
    if s in ("O-O", "0-0"):
        # Kingside: king from e to g, same rank.
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
    piece = m.group("piece") or "P"  # pawn default
    from_file = m.group("from_file")
    from_rank = m.group("from_rank")
    to_sq = m.group("to")
    promo = m.group("promo")

    # If there's a from_file but no piece, it's a pawn capture: "exd5"
    if not m.group("piece") and m.group("from_file") and m.group("capture"):
        piece = "P"

    target_piece = piece if mover_white else piece.lower()

    candidates = []
    for u in legal_ucis:
        u_from = u[0:2]
        u_to = u[2:4]
        u_promo = u[4:5] if len(u) >= 5 else ""
        if u_to != to_sq:
            continue
        if promo and u_promo.upper() != promo:
            continue
        if not promo and u_promo:
            continue
        # Piece on from-square must match target_piece.
        f_idx = sq_to_idx(u_from)
        if pmap.get(f_idx) != target_piece:
            continue
        # Disambiguator constraints.
        if from_file and u_from[0] != from_file:
            continue
        if from_rank and u_from[1] != from_rank:
            continue
        candidates.append(u)

    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) == 0:
        return None
    # >1: SAN insufficiently specified or our piece-map is wrong. Bail.
    return None


# ---------------------------------------------------------------------------
# Per-game replay.
# ---------------------------------------------------------------------------

def replay_game(game: dict) -> dict:
    """Replay one game through dojo_ref. Returns a result record."""
    headers = game["headers"]
    moves = game["moves"]
    result_str = headers.get("Result", "*")

    pos = dojo_ref.Position.startpos()
    pmap = initial_piece_map()
    mover_white = True

    for ply_idx, san in enumerate(moves):
        legal_ucis = pos.legal_moves()
        if not legal_ucis:
            return {"ok": False, "ply": ply_idx, "san": san,
                    "msg": "GPU reports no legal moves but PGN has more"}
        uci = san_to_uci(san, pmap, legal_ucis, mover_white)
        if uci is None:
            return {"ok": False, "ply": ply_idx, "san": san,
                    "msg": f"SAN→UCI match failed (legal: {legal_ucis[:8]}...)"}
        if uci not in legal_ucis:
            return {"ok": False, "ply": ply_idx, "san": san,
                    "msg": f"matched UCI {uci} not in GPU legal set"}
        try:
            pos = pos.make_move(uci)
        except Exception as e:
            return {"ok": False, "ply": ply_idx, "san": san,
                    "msg": f"make_move({uci}) raised: {e}"}
        update_piece_map(pmap, uci, mover_white)
        mover_white = not mover_white

    # Terminal-state agreement check.
    gpu_term = "checkmate" if pos.is_checkmate() \
        else "stalemate" if pos.is_stalemate() \
        else "undecided"
    pgn_term = headers.get("Termination", "")
    pgn_result = result_str

    return {
        "ok": True,
        "plies_replayed": len(moves),
        "gpu_terminal": gpu_term,
        "pgn_termination": pgn_term,
        "pgn_result": pgn_result,
        "white": headers.get("White", ""),
        "black": headers.get("Black", ""),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pgn",
        default="/home/frosty40/AgentChess_Server/match-processor/games.pgn")
    ap.add_argument("--n", type=int, default=100,
                    help="Max games to test (clamped to available)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-scan", type=int, default=None,
                    help="Cap how many games we read from the stream "
                         "before stopping (reservoir sampling). Default: "
                         "read whole file.")
    args = ap.parse_args()

    print(f"Streaming {args.pgn} (reservoir-sample n={args.n})")
    games, total_seen = reservoir_sample(
        stream_pgn(Path(args.pgn)),
        k=args.n,
        seed=args.seed,
        max_scan=args.max_scan,
    )
    print(f"Scanned {total_seen} games; sampled {len(games)} for replay")
    if not games:
        print("No games found.", file=sys.stderr); sys.exit(2)
    print(f"Replaying {len(games)} games via dojo_ref GPU referee")

    pass_count = 0
    fail_count = 0
    total_plies = 0
    failures = []

    for i, game in enumerate(games):
        result = replay_game(game)
        if result.get("ok"):
            pass_count += 1
            total_plies += result["plies_replayed"]
            white = result["white"][:14]; black = result["black"][:14]
            print(f"  [{i+1:3d}/{len(games)}] PASS  {result['plies_replayed']:3d} plies  "
                  f"{white:<14} vs {black:<14}  "
                  f"GPU:{result['gpu_terminal']:<10}  PGN:{result['pgn_result']}")
        else:
            fail_count += 1
            failures.append((i, result, game["headers"]))
            print(f"  [{i+1:3d}/{len(games)}] FAIL  ply {result['ply']:3d}  "
                  f"san={result['san']}  msg={result['msg']}")

    print()
    print(f"=== Parity report ===")
    print(f"games tested:         {len(games)}")
    print(f"pass (every move legal per GPU + replay completes): {pass_count}")
    print(f"fail (illegal move / SAN match fail / make_move):   {fail_count}")
    print(f"total plies verified: {total_plies}")

    if failures:
        print()
        print("First 5 failures:")
        for idx, result, hdrs in failures[:5]:
            print(f"  game #{idx}: {hdrs.get('White','?')} vs {hdrs.get('Black','?')}")
            print(f"    ply {result['ply']} ({result['san']}): {result['msg']}")


if __name__ == "__main__":
    sys.exit(main())
