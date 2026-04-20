"""
board_fold.py — Self-contained board compression / fold system

Extracted from:
  tools/active_fold.py          — warmth/centroid/fold core
  tools/measure_folding_compression.py — three fold variants

NO external dependencies. Pure board[64] → folded representation.

board[64] integer encoding:
  0=empty  1=wP 2=wN 3=wB 4=wR 5=wQ 6=wK  7=bP 8=bN 9=bB 10=bR 11=bQ 12=bK
"""

import math

# ─────────────────────────────────────────────────────────────────────────────
# PIECE WEIGHTS & CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

# Strategic weights (how much each piece pulls the warmth centroid)
PIECE_WEIGHT = {
    1: 1.0,  # pawn
    2: 3.0,  # knight
    3: 3.0,  # bishop
    4: 5.0,  # rook
    5: 9.0,  # queen
    6: 4.0,  # king (lower — king hides, doesn't define strategic center)
    7: 1.0, 8: 3.0, 9: 3.0, 10: 5.0, 11: 9.0, 12: 4.0,
}

# Warmth spread: neighboring squares of a piece get partial warmth
SPREAD_WEIGHT = 0.25

# Strategic zone radius from fold center — squares within this distance
# keep their full absolute encoding; outside squares are compressed
OUTLIER_DIST_THRESHOLD = 2.5


# ─────────────────────────────────────────────────────────────────────────────
# CORE: WARMTH / CENTROID / ACTIVE FOLD
# (from tools/active_fold.py)
# ─────────────────────────────────────────────────────────────────────────────

def compute_warmth(board):
    """
    Per-square strategic weight. A square is "warm" if it has
    a valuable piece on it or is near valuable pieces.

    Returns warmth[64] as floats.
    """
    # Base warmth from pieces
    warmth = [PIECE_WEIGHT.get(board[sq], 0.0) for sq in range(64)]

    # Spread warmth to adjacent squares (a piece heats its neighborhood)
    spread = [0.0] * 64
    for sq in range(64):
        if warmth[sq] == 0.0:
            continue
        r, f = sq // 8, sq % 8
        for dr in (-1, 0, 1):
            for df in (-1, 0, 1):
                if dr == 0 and df == 0:
                    continue
                nr, nf = r + dr, f + df
                if 0 <= nr < 8 and 0 <= nf < 8:
                    spread[nr * 8 + nf] += warmth[sq] * SPREAD_WEIGHT

    return [warmth[sq] + spread[sq] for sq in range(64)]


def find_fold_center(warmth):
    """
    Weighted centroid of the warmth map.
    This is the center of strategic gravity for the current position —
    the raytrace origin for all geometry encoders.

    Returns (fold_rank: float, fold_file: float) in board coordinates [0,7].
    """
    total = sum(warmth) or 1.0
    fold_r = sum(warmth[sq] * (sq // 8) for sq in range(64)) / total
    fold_f = sum(warmth[sq] * (sq % 8)  for sq in range(64)) / total
    return fold_r, fold_f


def active_fold_positions(board):
    """
    Core active folding algorithm.

    For each square, assigns a canonical position index for the
    positional encoding. Strategic squares keep their absolute
    position (unique encoding). Quiet squares are reflected across
    the fold center and share encoding with their mirror.

    Returns:
        canon_pos[64]    — positional index for each square (0-63)
        outlier_mask[64] — 1 if square is in the strategic zone
        warmth[64]       — per-square warmth
        center           — (fold_rank, fold_file) centroid
    """
    warmth = compute_warmth(board)
    fold_r, fold_f = find_fold_center(warmth)

    canon_pos = []
    outlier_mask = []

    for sq in range(64):
        r, f = sq // 8, sq % 8
        dist = math.sqrt((r - fold_r) ** 2 + (f - fold_f) ** 2)

        if dist < OUTLIER_DIST_THRESHOLD:
            # Strategic zone: keep absolute position for unique encoding
            canon_pos.append(sq)
            outlier_mask.append(1)
        else:
            # Quiet zone: reflect across fold center, share with mirror
            mirror_r = int(round(2 * fold_r - r))
            mirror_f = int(round(2 * fold_f - f))
            mirror_r = max(0, min(7, mirror_r))
            mirror_f = max(0, min(7, mirror_f))
            mirror_sq = mirror_r * 8 + mirror_f
            # Canonical = lower index of (sq, mirror) so both sides agree
            canon_pos.append(min(sq, mirror_sq))
            outlier_mask.append(0)

    return canon_pos, outlier_mask, warmth, (fold_r, fold_f)


def fold_board(board):
    """
    Full active fold: returns compressed representation + outlier list.

    compressed: dict mapping canon_pos → piece (quiet zone, one piece per slot)
    outliers:   list of (sq, piece, warmth) for the strategic zone
    center:     (fold_rank, fold_file) centroid
    """
    canon_pos, outlier_mask, warmth, center = active_fold_positions(board)

    compressed = {}
    outliers = []

    for sq in range(64):
        piece = board[sq]
        if piece == 0:
            continue
        if outlier_mask[sq]:
            outliers.append((sq, piece, round(warmth[sq], 3)))
        else:
            cp = canon_pos[sq]
            if cp in compressed and compressed[cp] != piece:
                # Collision: two different pieces mapped to same canonical slot
                # The one with higher warmth wins; the other becomes an outlier
                existing_sq = next(s for s in range(64)
                                   if canon_pos[s] == cp and not outlier_mask[s]
                                   and board[s] == compressed[cp])
                if warmth[sq] > warmth[existing_sq]:
                    outliers.append((existing_sq, compressed[cp],
                                     round(warmth[existing_sq], 3)))
                    compressed[cp] = piece
                else:
                    outliers.append((sq, piece, round(warmth[sq], 3)))
            else:
                compressed[cp] = piece

    return compressed, outliers, warmth, center


# ─────────────────────────────────────────────────────────────────────────────
# THREE FOLD VARIANTS
# (from tools/measure_folding_compression.py)
# ─────────────────────────────────────────────────────────────────────────────

def fold_vertical(board):
    """
    Variant A — Fold along d/e file boundary (vertical axis).
    Each rank: files a-d mirror files h-e.
    Symmetric pairs compress cleanly; asymmetric pairs signal file imbalance.

    Returns (compressed[32], outliers: list of (sq, piece))
    """
    compressed = [0] * 32   # 8 ranks x 4 folded files
    outliers = []
    for rank in range(8):
        for file in range(4):
            left_sq  = rank * 8 + file          # files a,b,c,d
            right_sq = rank * 8 + (7 - file)    # files h,g,f,e
            lp = board[left_sq]
            rp = board[right_sq]
            fold_sq = rank * 4 + file
            if lp == 0 and rp == 0:
                compressed[fold_sq] = 0
            elif lp == 0:
                compressed[fold_sq] = rp
                outliers.append((right_sq, rp))  # piece with no mirror
            elif rp == 0:
                compressed[fold_sq] = lp
                outliers.append((left_sq, lp))
            elif lp == rp:
                compressed[fold_sq] = lp          # symmetric — compresses cleanly
            else:
                # Both sides have different pieces — broken symmetry
                compressed[fold_sq] = lp          # keep left, flag right as outlier
                outliers.append((right_sq, rp))
    return compressed, outliers


def fold_horizontal(board):
    """
    Variant B — Fold along 4th/5th rank boundary (horizontal axis).
    White pieces in ranks 1-4 (sq 32-63) mirror black pieces in ranks 5-8 (sq 0-31).
    Outlier = a piece that has crossed the center (advanced pawn, invasion).
    Captures invasion depth and territorial imbalance.

    Returns (compressed[32], outliers: list of (sq, piece))
    """
    compressed = [0] * 32
    outliers = []
    WHITE = set(range(1, 7))
    BLACK = set(range(7, 13))
    WHITE_TO_BLACK = {1:7, 2:8, 3:9, 4:10, 5:11, 6:12}

    for rank in range(4):  # ranks 0-3 = ranks 8-5 (black territory)
        for file in range(8):
            top_sq    = rank * 8 + file          # black's half (ranks 8..5)
            bottom_sq = (7 - rank) * 8 + file   # white's half (ranks 1..4)
            tp = board[top_sq]
            bp = board[bottom_sq]
            fold_sq = rank * 8 + file

            if tp == 0 and bp == 0:
                compressed[fold_sq] = 0
            elif tp == 0 and bp != 0:
                # White piece in own half only — normal
                compressed[fold_sq] = bp
            elif tp != 0 and bp == 0:
                # Black piece in own half only — normal
                compressed[fold_sq] = tp
            elif tp in BLACK and bp in WHITE and WHITE_TO_BLACK[bp] == tp:
                # Perfect mirror — symmetric position
                compressed[fold_sq] = bp
            else:
                # Broken symmetry — one side has advanced or retreated unusually
                compressed[fold_sq] = bp if bp else tp
                # Flag whichever piece is "out of place"
                if tp in WHITE:  # white piece in black's territory — invasion
                    outliers.append((top_sq, tp))
                if bp in BLACK:  # black piece in white's territory — invasion
                    outliers.append((bottom_sq, bp))
                if tp in BLACK and bp in BLACK:
                    outliers.append((top_sq, tp))
                if tp in WHITE and bp in WHITE:
                    outliers.append((bottom_sq, bp))

    return compressed, outliers


def fold_relationship(board):
    """
    Variant C — 4x4 zone co-occupancy.
    Cluster squares into 16 symmetric quadrant zones. For each zone, record
    which white and black pieces co-occupy it. Contested zones (both colors
    present) are outliers — they signal where strategic contact is happening.
    Captures the geometric distribution of inter-side contact.

    Returns (compressed[32], outliers: list of (sq, piece))
    """
    ZONES = 16
    zone_white = [0] * ZONES
    zone_black = [0] * ZONES
    outliers = []

    WHITE_TYPES = {1, 2, 3, 4, 5, 6}
    BLACK_TYPES = {7, 8, 9, 10, 11, 12}

    for sq in range(64):
        rank = sq // 8
        file = sq % 8
        zone = (min(rank, 7 - rank) // 2) * 4 + (min(file, 7 - file) // 2)
        p = board[sq]
        if p in WHITE_TYPES:
            if zone_white[zone] and zone_white[zone] != p:
                outliers.append((sq, p))  # multiple white piece types in same zone
            zone_white[zone] = p
        elif p in BLACK_TYPES:
            if zone_black[zone] and zone_black[zone] != p:
                outliers.append((sq, p))
            zone_black[zone] = p

    # Compressed: 16 zones x 2 (white, black) = 32 values
    compressed = []
    for z in range(ZONES):
        compressed.append(zone_white[z])
        compressed.append(zone_black[z])

    return compressed, outliers


# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

PIECE_MAP = {
    'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
    'p': 7, 'n': 8, 'b': 9, 'r': 10, 'q': 11, 'k': 12,
}

PIECE_NAMES = {
    0: '.', 1: 'P', 2: 'N', 3: 'B', 4: 'R', 5: 'Q', 6: 'K',
    7: 'p', 8: 'n', 9: 'b', 10: 'r', 11: 'q', 12: 'k',
}


def board_from_fen(fen):
    """Parse FEN string into board[64] int array."""
    board = [0] * 64
    sq = 0
    for ch in fen.split()[0]:
        if ch == '/':
            continue
        if ch.isdigit():
            sq += int(ch)
        else:
            board[sq] = PIECE_MAP.get(ch, 0)
            sq += 1
    return board


def print_board(board, outlier_mask=None, center=None):
    """Pretty-print a board with optional strategic zone highlighting."""
    files = 'abcdefgh'
    if center:
        print(f"  Fold center: rank={center[0]:.2f} file={center[1]:.2f} "
              f"({files[int(round(center[1]))].upper()}{8 - int(round(center[0]))})")
    for rank in range(8):
        row = []
        for file in range(8):
            sq = rank * 8 + file
            p = PIECE_NAMES[board[sq]]
            if outlier_mask and outlier_mask[sq]:
                p = f'[{p}]'
            else:
                p = f' {p} '
            row.append(p)
        print(f"  {8-rank} {''.join(row)}")
    print(f"    a  b  c  d  e  f  g  h")


# ─────────────────────────────────────────────────────────────────────────────
# DEMO
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    STARTING_FEN = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'

    test_positions = [
        ('Starting position',
         'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'),
        ('Ruy Lopez (e4 e5 Nf3 Nc6 Bb5)',
         'r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 2 4'),
        ('Queenless endgame (passed pawn)',
         '8/8/3k4/8/3P4/3K4/8/8 w - - 0 1'),
    ]

    for name, fen in test_positions:
        board = board_from_fen(fen)
        canon_pos, outlier_mask, warmth, center = active_fold_positions(board)

        print(f"\n{'='*62}")
        print(f"  {name}")
        print(f"{'='*62}")
        print_board(board, outlier_mask, center)

        n_strategic = sum(outlier_mask)
        n_quiet = 64 - n_strategic
        print(f"\n  Strategic zone : {n_strategic} squares  (brackets = full resolution)")
        print(f"  Quiet zone     : {n_quiet} squares  (compressed toward centroid)")

        print(f"\n  --- Variant A: fold_vertical ---")
        vc, vo = fold_vertical(board)
        print(f"  Compressed: {len(vc)} slots  |  Outliers: {len(vo)}")

        print(f"\n  --- Variant B: fold_horizontal ---")
        hc, ho = fold_horizontal(board)
        print(f"  Compressed: {len(hc)} slots  |  Outliers: {len(ho)}")

        print(f"\n  --- Variant C: fold_relationship ---")
        rc, ro = fold_relationship(board)
        print(f"  Compressed: {len(rc)} slots  |  Outliers: {len(ro)}")
