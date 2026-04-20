"""
geometry_encoders.py — Self-contained FrostMatrix geometry encoder set

Extracted from tools/render_pipeline.py (G1-G17) and tools/active_fold.py.

NO render pipeline dependencies — no PGN reading, no clustering, no numpy.
Pure board[64] int → feature vector (55 floats).

board[64] integer encoding:
  0=empty  1=wP 2=wN 3=wB 4=wR 5=wQ 6=wK  7=bP 8=bN 9=bB 10=bR 11=bQ 12=bK

Full feature vector layout (55 dimensions per board position):
  [0:4]   4 base dims from active_fold_encode
              centroid_r, centroid_f, compression, outlier_frac
  [4:7]   G1  tension_gradient       3 dims
  [7:10]  G2  pawn_corridor          3 dims
  [10:13] G3  king_shell             3 dims
  [13:16] G4  mobility_field         3 dims
  [16:18] G5  color_complex          2 dims
  [18:21] G6  outpost_gravity        3 dims
  [21:24] G7  phase_gradient         3 dims
  [24:27] G8  diagonal_axis          3 dims
  [27:30] G9  territorial_frontier   3 dims
  [30:33] G10 resonance_cluster      3 dims
  [33:36] G11 vertical_fold          3 dims
  [36:39] G12 horizontal_fold        3 dims
  [39:43] G13 relationship_fold      4 dims
  [43:46] G14 pawn_chain             3 dims
  [46:49] G15 pin_xray               3 dims
  [49:52] G16 open_file              3 dims
  [52:55] G17 knight_outpost         3 dims
"""

import math
from typing import List, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# ACTIVE FOLD BASE (inline — no import needed)
# ─────────────────────────────────────────────────────────────────────────────

_PIECE_WEIGHT = {
    1: 1.0, 2: 3.0, 3: 3.0, 4: 5.0, 5: 9.0, 6: 4.0,
    7: 1.0, 8: 3.0, 9: 3.0, 10: 5.0, 11: 9.0, 12: 4.0,
}
_SPREAD_WEIGHT = 0.25
_OUTLIER_DIST  = 2.5


def _compute_warmth(board: List[int]) -> List[float]:
    warmth = [_PIECE_WEIGHT.get(board[sq], 0.0) for sq in range(64)]
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
                    spread[nr * 8 + nf] += warmth[sq] * _SPREAD_WEIGHT
    return [warmth[sq] + spread[sq] for sq in range(64)]


def _find_fold_center(warmth: List[float]) -> Tuple[float, float]:
    total = sum(warmth) or 1.0
    fold_r = sum(warmth[sq] * (sq // 8) for sq in range(64)) / total
    fold_f = sum(warmth[sq] * (sq % 8)  for sq in range(64)) / total
    return fold_r, fold_f


def _active_fold_positions(board: List[int]):
    warmth = _compute_warmth(board)
    fold_r, fold_f = _find_fold_center(warmth)
    canon_pos, outlier_mask = [], []
    for sq in range(64):
        r, f = sq // 8, sq % 8
        dist = math.sqrt((r - fold_r) ** 2 + (f - fold_f) ** 2)
        if dist < _OUTLIER_DIST:
            canon_pos.append(sq)
            outlier_mask.append(1)
        else:
            mirror_r = max(0, min(7, int(round(2 * fold_r - r))))
            mirror_f = max(0, min(7, int(round(2 * fold_f - f))))
            canon_pos.append(min(sq, mirror_r * 8 + mirror_f))
            outlier_mask.append(0)
    return canon_pos, outlier_mask, warmth, (fold_r, fold_f)


def active_fold_encode(board: List[int]) -> List[float]:
    """
    Base 4-dim encoding from the active fold system.
    Returns [centroid_r, centroid_f, compression, outlier_frac].
    """
    warmth = _compute_warmth(board)
    fold_r, fold_f = _find_fold_center(warmth)
    _, outlier_mask, _, _ = _active_fold_positions(board)

    quiet_count  = sum(1 for i in range(64) if not outlier_mask[i])
    compression  = quiet_count / 64.0

    placed       = [i for i in range(64) if board[i] != 0]
    outlier_frac = (sum(outlier_mask[i] for i in placed) / len(placed)) if placed else 0.0

    return [fold_r / 7.0, fold_f / 7.0, compression, outlier_frac]


# ─────────────────────────────────────────────────────────────────────────────
# SHARED HELPERS
# ─────────────────────────────────────────────────────────────────────────────

_PH_W = [0, 0, 1, 1, 2, 4, 0]   # phase weights indexed by piece_type 1-6

def _pt(p: int) -> int:  return p if p <= 6 else p - 6
def _isw(p: int) -> bool: return 1 <= p <= 6
def _isb(p: int) -> bool: return 7 <= p <= 12
def _wpw(p: int) -> float: return [0, 1, 3, 3, 5, 9, 4][_pt(p)] if p else 0.0

_RAY_DIRS = {
    2: [(-2,-1),(-2,1),(-1,-2),(-1,2),(1,-2),(1,2),(2,-1),(2,1)],  # knight
    3: [(-1,-1),(-1,1),(1,-1),(1,1)],   # bishop
    4: [(-1,0),(1,0),(0,-1),(0,1)],     # rook
    5: [(-1,-1),(-1,1),(1,-1),(1,1),(-1,0),(1,0),(0,-1),(0,1)],  # queen
    6: [(-1,-1),(-1,1),(1,-1),(1,1),(-1,0),(1,0),(0,-1),(0,1)],  # king
    1: [],   # pawn handled per-encoder
}


# ─────────────────────────────────────────────────────────────────────────────
# G1: TENSION GRADIENT EIGENVECTOR
# ─────────────────────────────────────────────────────────────────────────────

def tension_gradient_encode(b: List[int]) -> List[float]:
    """
    Every piece radiates force along its attack rays. Sum gives a 2D vector
    field. The dominant eigenvector is the main axis of conflict.
    Output: [axis_r, axis_f, axis_strength]  (3 floats)
    """
    dr_sum, df_sum = 0.0, 0.0
    total_w = 0.0
    for sq in range(64):
        p = b[sq]
        if not p: continue
        r, f = sq >> 3, sq & 7
        pt = _pt(p)
        sign = 1.0 if _isw(p) else -1.0
        w = _wpw(p)
        dirs = _RAY_DIRS.get(pt, [])
        if pt == 1:
            dirs = [(-1,-1),(-1,1)] if _isw(p) else [(1,-1),(1,1)]
        for dr, df in dirs:
            nr, nf = r + dr, f + df
            if 0 <= nr < 8 and 0 <= nf < 8:
                dr_sum += sign * w * dr
                df_sum += sign * w * df
        total_w += w
    if total_w == 0: return [0.0, 0.0, 0.0]
    axis_r = dr_sum / total_w
    axis_f = df_sum / total_w
    strength = math.sqrt(axis_r**2 + axis_f**2)
    return [axis_r, axis_f, strength]


# ─────────────────────────────────────────────────────────────────────────────
# G2: PAWN CORRIDOR TOPOLOGY
# ─────────────────────────────────────────────────────────────────────────────

def pawn_corridor_encode(b: List[int]) -> List[float]:
    """
    Pawns as walls dividing the board into corridors. Encodes the open/closed
    topology of each file and the structural asymmetry between flanks.
    Output: [open_file_centroid_f, semi_open_imbalance, closure_index]  (3 floats)
    """
    open_files, semi_w, semi_b = [], [], []
    for f in range(8):
        has_wp = any(b[r*8+f] == 1 for r in range(8))
        has_bp = any(b[r*8+f] == 7 for r in range(8))
        if not has_wp and not has_bp: open_files.append(f)
        elif not has_wp and has_bp:   semi_w.append(f)
        elif has_wp and not has_bp:   semi_b.append(f)
    open_centroid = sum(open_files) / len(open_files) / 7.0 if open_files else 0.5
    semi_imbalance = (len(semi_w) - len(semi_b)) / 8.0
    closure = 1.0 - (len(open_files) + 0.5*(len(semi_w)+len(semi_b))) / 8.0
    return [open_centroid, semi_imbalance, closure]


# ─────────────────────────────────────────────────────────────────────────────
# G3: KING SHELL RADIANCE
# ─────────────────────────────────────────────────────────────────────────────

def king_shell_encode(b: List[int]) -> List[float]:
    """
    Concentric Chebyshev rings around each king. The overlap zone (contested
    safety space) and shield asymmetry are key structural signals.
    Output: [king_distance, overlap_norm, shield_asymmetry]  (3 floats)
    """
    wk = next((sq for sq in range(64) if b[sq] == 6), None)
    bk = next((sq for sq in range(64) if b[sq] == 12), None)
    if wk is None or bk is None: return [0.0, 0.0, 0.0]
    wkr, wkf = wk >> 3, wk & 7
    bkr, bkf = bk >> 3, bk & 7
    king_dist = max(abs(wkr-bkr), abs(wkf-bkf)) / 7.0

    w_shield, b_shield = 0, 0
    for sq in range(64):
        p = b[sq]
        if not p: continue
        r, f = sq >> 3, sq & 7
        if _isw(p) and max(abs(r-wkr), abs(f-wkf)) <= 2: w_shield += 1
        if _isb(p) and max(abs(r-bkr), abs(f-bkf)) <= 2: b_shield += 1

    overlap = sum(1 for sq in range(64)
                  if max(abs((sq>>3)-wkr), abs((sq&7)-wkf)) <= 2
                  and max(abs((sq>>3)-bkr), abs((sq&7)-bkf)) <= 2) / 25.0

    shield_asym = (w_shield - b_shield) / 9.0
    return [king_dist, overlap, shield_asym]


# ─────────────────────────────────────────────────────────────────────────────
# G4: MOBILITY ACTIVITY FIELD
# ─────────────────────────────────────────────────────────────────────────────

_MOB_BASE = {1: 2, 2: 4, 3: 7, 4: 7, 5: 14, 6: 3}


def _estimate_mobility(b: List[int], sq: int) -> float:
    p = b[sq]
    if not p: return 0.0
    pt = _pt(p)
    dirs = _RAY_DIRS.get(pt, [])
    if pt == 1:
        dirs = [(-1,0),(-1,-1),(-1,1)] if _isw(p) else [(1,0),(1,-1),(1,1)]
    if not dirs: return _MOB_BASE.get(pt, 1) / 14.0
    r, f = sq >> 3, sq & 7
    free = sum(1 for dr, df in dirs
               if 0 <= r+dr < 8 and 0 <= f+df < 8
               and not (_isw(p) and _isw(b[(r+dr)*8+(f+df)]))
               and not (_isb(p) and _isb(b[(r+dr)*8+(f+df)])))
    return free / max(len(dirs), 1)


def mobility_field_encode(b: List[int]) -> List[float]:
    """
    Mobility-weighted centroid: WHERE pieces are effectively active, not just
    where they sit. Uses estimated mobility (not full legal move generation).
    Output: [mobility_centroid_r, mobility_centroid_f, mobility_imbalance]  (3 floats)
    """
    wr, wf, wm = 0.0, 0.0, 0.0
    br, bf, bm = 0.0, 0.0, 0.0
    for sq in range(64):
        p = b[sq]
        if not p: continue
        mob = _estimate_mobility(b, sq)
        r, f = sq >> 3, sq & 7
        if _isw(p): wr += r*mob; wf += f*mob; wm += mob
        else:       br += r*mob; bf += f*mob; bm += mob
    if wm == 0 and bm == 0: return [3.5/7, 3.5/7, 0.0]
    total = wm + bm
    cr = (wr + br) / total / 7.0
    cf = (wf + bf) / total / 7.0
    imbalance = (wm - bm) / max(wm + bm, 1e-6)
    return [cr, cf, imbalance]


# ─────────────────────────────────────────────────────────────────────────────
# G5: COLOR COMPLEX IMBALANCE
# ─────────────────────────────────────────────────────────────────────────────

def color_complex_encode(b: List[int]) -> List[float]:
    """
    Dark/light square piece density differential per side.
    Reveals bad bishops, color weaknesses, and bishop-of-opposite-color structures.
    Output: [light_sq_advantage, dark_sq_advantage]  (2 floats)
    """
    wl, wd, bl, bd = 0.0, 0.0, 0.0, 0.0
    for sq in range(64):
        p = b[sq]
        if not p: continue
        w = _wpw(p)
        light = (sq ^ (sq >> 3)) & 1
        if _isw(p):
            if light: wl += w
            else:     wd += w
        else:
            if light: bl += w
            else:     bd += w
    total = max(wl+wd+bl+bd, 1.0)
    return [(wl - bl) / total, (wd - bd) / total]


# ─────────────────────────────────────────────────────────────────────────────
# G6: OUTPOST GRAVITY WELLS
# ─────────────────────────────────────────────────────────────────────────────

def outpost_gravity_encode(b: List[int]) -> List[float]:
    """
    Squares that cannot be attacked by enemy pawns and are defended by friendly
    pawns act as gravitational anchors for the game's structural balance.
    Output: [outpost_centroid_r, outpost_centroid_f, outpost_density]  (3 floats)
    """
    white_outposts, black_outposts = [], []
    for sq in range(64):
        r, f = sq >> 3, sq & 7
        if r < 4:
            bp_attack = any(b[nr*8+nf] == 7
                            for nr, nf in [(r+1, f-1),(r+1, f+1)]
                            if 0 <= nr < 8 and 0 <= nf < 8)
            wp_defend = any(b[nr*8+nf] == 1
                            for nr, nf in [(r+1, f-1),(r+1, f+1)]
                            if 0 <= nr < 8 and 0 <= nf < 8)
            if not bp_attack and wp_defend: white_outposts.append(sq)
        if r > 3:
            wp_attack = any(b[nr*8+nf] == 1
                            for nr, nf in [(r-1, f-1),(r-1, f+1)]
                            if 0 <= nr < 8 and 0 <= nf < 8)
            bp_defend = any(b[nr*8+nf] == 7
                            for nr, nf in [(r-1, f-1),(r-1, f+1)]
                            if 0 <= nr < 8 and 0 <= nf < 8)
            if not wp_attack and bp_defend: black_outposts.append(sq)

    all_outposts = white_outposts + black_outposts
    if not all_outposts: return [3.5/7, 3.5/7, 0.0]
    cr = sum(sq >> 3 for sq in all_outposts) / len(all_outposts) / 7.0
    cf = sum(sq & 7  for sq in all_outposts) / len(all_outposts) / 7.0
    density = len(all_outposts) / 16.0
    return [cr, cf, density]


# ─────────────────────────────────────────────────────────────────────────────
# G7: PHASE TOPOLOGY GRADIENT
# ─────────────────────────────────────────────────────────────────────────────

def phase_gradient_encode(b: List[int]) -> List[float]:
    """
    Per-quadrant material phase. The gradient vector points from the most
    endgame-like quadrant toward the most middlegame-like.
    Output: [gradient_r, gradient_f, phase_range]  (3 floats)
    """
    phase_q = [0.0, 0.0, 0.0, 0.0]
    for sq in range(64):
        p = b[sq]
        if not p: continue
        pt = _pt(p)
        r, f = sq >> 3, sq & 7
        q = (1 if f >= 4 else 0) + (2 if r >= 4 else 0)
        phase_q[q] += _PH_W[pt] if pt < len(_PH_W) else 0

    max_q = max(phase_q) or 1.0
    top = (phase_q[0] + phase_q[1]) / (2 * max_q)
    bot = (phase_q[2] + phase_q[3]) / (2 * max_q)
    lft = (phase_q[0] + phase_q[2]) / (2 * max_q)
    rgt = (phase_q[1] + phase_q[3]) / (2 * max_q)
    grad_r = (top - bot)
    grad_f = (rgt - lft)
    phase_range = (max(phase_q) - min(phase_q)) / max_q
    return [grad_r, grad_f, phase_range]


# ─────────────────────────────────────────────────────────────────────────────
# G8: DIAGONAL LOAD AXIS
# ─────────────────────────────────────────────────────────────────────────────

def diagonal_axis_encode(b: List[int]) -> List[float]:
    """
    The most piece-loaded diagonal in each direction defines an oblique fold
    axis. Separates diagonal-pressure openings (Sicilian, Nimzo) from
    orthogonal ones.
    Output: [ne_load, nw_load, diagonal_dominance]  (3 floats)
    """
    ne_diag = [0.0] * 15
    nw_diag = [0.0] * 15
    total_w = 0.0
    for sq in range(64):
        p = b[sq]
        if not p: continue
        w = _wpw(p)
        r, f = sq >> 3, sq & 7
        ne_diag[(r - f) + 7] += w
        nw_diag[r + f]        += w
        total_w += w
    if total_w == 0: return [0.0, 0.0, 0.0]
    ne_max = max(ne_diag) / total_w
    nw_max = max(nw_diag) / total_w
    dominance = abs(ne_max - nw_max)
    return [ne_max, nw_max, dominance]


# ─────────────────────────────────────────────────────────────────────────────
# G9: TERRITORIAL FRONTIER
# ─────────────────────────────────────────────────────────────────────────────

def territorial_frontier_encode(b: List[int]) -> List[float]:
    """
    Control differential per square → boundary between white/black territory.
    Frontier geometry (centroid, fragmentation) characterizes strategic contact.
    Output: [frontier_centroid_r, frontier_centroid_f, fragmentation]  (3 floats)
    """
    control = [0] * 64
    for sq in range(64):
        p = b[sq]
        if not p: continue
        pt = _pt(p)
        sign = 1 if _isw(p) else -1
        dirs = _RAY_DIRS.get(pt, [])
        if pt == 1:
            dirs = [(-1,-1),(-1,1)] if _isw(p) else [(1,-1),(1,1)]
        r, f = sq >> 3, sq & 7
        for dr, df in dirs:
            nr, nf = r+dr, f+df
            if 0 <= nr < 8 and 0 <= nf < 8:
                control[nr*8+nf] += sign

    frontier = [sq for sq in range(64)
                if control[sq] == 0
                or any(control[sq] * control[(sq>>3)*8 + (sq&7) + d] < 0
                       for d in [-8, 8, -1, 1]
                       if 0 <= (sq>>3)*8+(sq&7)+d < 64)]

    if not frontier: return [3.5/7, 3.5/7, 0.0]
    cr = sum(sq >> 3 for sq in frontier) / len(frontier) / 7.0
    cf = sum(sq & 7  for sq in frontier) / len(frontier) / 7.0
    var = sum(((sq>>3)/7.0 - cr)**2 + ((sq&7)/7.0 - cf)**2 for sq in frontier)
    fragmentation = min(var / len(frontier), 1.0)
    return [cr, cf, fragmentation]


# ─────────────────────────────────────────────────────────────────────────────
# G10: RESONANCE CLUSTER SIGNATURE
# ─────────────────────────────────────────────────────────────────────────────

def resonance_cluster_encode(b: List[int]) -> List[float]:
    """
    Coordinated piece pairs (batteries, outpost+defender, rook+queen lift).
    WHERE pieces resonate together and HOW strongly defines the attack signature.
    Output: [resonance_centroid_r, resonance_centroid_f, resonance_strength]  (3 floats)
    """
    rr, rf, strength = 0.0, 0.0, 0.0
    for sq1 in range(64):
        p1 = b[sq1]
        if not p1: continue
        r1, f1 = sq1 >> 3, sq1 & 7
        for sq2 in range(sq1+1, 64):
            p2 = b[sq2]
            if not p2: continue
            if _isw(p1) != _isw(p2): continue
            r2, f2 = sq2 >> 3, sq2 & 7
            on_rank = r1 == r2
            on_file = f1 == f2
            on_diag = abs(r1-r2) == abs(f1-f2)
            if not (on_rank or on_file or on_diag): continue
            w = (_wpw(p1) * _wpw(p2)) ** 0.5
            cr = (r1 + r2) / 2.0
            cf = (f1 + f2) / 2.0
            rr += cr * w; rf += cf * w; strength += w

    if strength == 0: return [3.5/7, 3.5/7, 0.0]
    return [rr / strength / 7.0, rf / strength / 7.0, min(strength / 50.0, 1.0)]


# ─────────────────────────────────────────────────────────────────────────────
# G11: VERTICAL FOLD COMPRESSION SIGNAL
# ─────────────────────────────────────────────────────────────────────────────

def vertical_fold_encode(b: List[int]) -> List[float]:
    """
    From fold_vertical(): mirrors files a-d with h-e per rank.
    Compression = symmetric pairs; asymmetry = broken file symmetry.
    Captures file imbalance (which flank is "unfolded" / occupied differently).
    Output: [compression_ratio, asymmetry_score, outlier_centroid_f]  (3 floats)
    """
    symmetric = 0; asymmetric = 0; empty_pairs = 0; outlier_files = []
    for rank in range(8):
        for file in range(4):
            lp = b[rank*8 + file]
            rp = b[rank*8 + (7 - file)]
            if lp == 0 and rp == 0:
                empty_pairs += 1
            elif lp == 0 or rp == 0:
                asymmetric += 1
                outlier_files.append(file if rp == 0 else (7 - file))
            elif lp == rp:
                symmetric += 1
            else:
                asymmetric += 1
                outlier_files.append(7 - file)
    denom = max(1, 32 - empty_pairs)
    compression = symmetric / denom
    asym_score   = asymmetric / denom
    outlier_f    = (sum(outlier_files) / max(1, len(outlier_files)) / 7.0
                    if outlier_files else 0.5)
    return [compression, asym_score, outlier_f]


# ─────────────────────────────────────────────────────────────────────────────
# G12: HORIZONTAL FOLD INVASION SIGNAL
# ─────────────────────────────────────────────────────────────────────────────

def horizontal_fold_encode(b: List[int]) -> List[float]:
    """
    From fold_horizontal(): mirrors white/black halves (ranks 1-4 vs 5-8).
    Outlier = a piece that has crossed the center.
    Captures invasion depth and territorial imbalance.
    Output: [compression_ratio, invasion_density, rank_boundary_pressure]  (3 floats)
    """
    WHITE = set(range(1, 7)); BLACK = set(range(7, 13))
    W2B = {1:7, 2:8, 3:9, 4:10, 5:11, 6:12}
    symmetric = 0; total = 0; invasion_ranks = []
    for rank in range(4):
        for file in range(8):
            top_sq = rank * 8 + file
            bot_sq = (7 - rank) * 8 + file
            tp = b[top_sq]; bp = b[bot_sq]
            if tp == 0 and bp == 0: continue
            total += 1
            if tp in BLACK and bp in WHITE and W2B.get(bp, 0) == tp:
                symmetric += 1
            else:
                if tp in WHITE:
                    invasion_ranks.append(rank)
                elif bp in BLACK:
                    invasion_ranks.append(7 - rank)
    compression      = symmetric / max(1, total)
    invasion_density = len(invasion_ranks) / 64.0
    rank_pressure = ((4 - sum(invasion_ranks) / max(1, len(invasion_ranks))) / 4.0
                     if invasion_ranks else 0.0)
    return [compression, invasion_density, rank_pressure]


# ─────────────────────────────────────────────────────────────────────────────
# G13: RELATIONSHIP FOLD ZONE CO-OCCUPANCY
# ─────────────────────────────────────────────────────────────────────────────

def relationship_fold_encode(b: List[int]) -> List[float]:
    """
    From fold_relationship(): 16 symmetric quadrant zones.
    White + black pieces in same zone = contested.
    Captures the geometric distribution of inter-side contact.
    Output: [conflict_density, max_conflict_norm, zone_centroid_r, zone_centroid_f]  (4 floats)
    """
    WHITE = set(range(1, 7)); BLACK = set(range(7, 13))
    zone_w = [0] * 16; zone_b = [0] * 16; zone_conflict = [0] * 16
    for sq in range(64):
        rank = sq // 8; file = sq % 8
        zone = (min(rank, 7 - rank) // 2) * 4 + (min(file, 7 - file) // 2)
        p = b[sq]
        if p in WHITE:
            if zone_w[zone] and zone_w[zone] != p: zone_conflict[zone] += 1
            zone_w[zone] = p
        elif p in BLACK:
            if zone_b[zone] and zone_b[zone] != p: zone_conflict[zone] += 1
            zone_b[zone] = p
    contested_rs = []; contested_fs = []
    for z in range(16):
        if zone_w[z] and zone_b[z]:
            contested_rs.append((z // 4) * 2 + 1)
            contested_fs.append((z  % 4) * 2 + 1)
    conflict_density = len(contested_rs) / 16.0
    max_conflict     = min(max(zone_conflict) / 4.0, 1.0)
    centroid_r = (sum(contested_rs) / max(1, len(contested_rs)) / 7.0
                  if contested_rs else 0.5)
    centroid_f = (sum(contested_fs) / max(1, len(contested_fs)) / 7.0
                  if contested_fs else 0.5)
    return [conflict_density, max_conflict, centroid_r, centroid_f]


# ─────────────────────────────────────────────────────────────────────────────
# G14: PAWN CHAIN TOPOLOGY
# ─────────────────────────────────────────────────────────────────────────────

def pawn_chain_encode(b: List[int]) -> List[float]:
    """
    Pawn chains: pawns diagonally defending same-color pawns.
    Chain density and max length capture structural pawn geometry
    (hedgehog, Maroczy bind, isolated queen pawn, etc.)
    Output: [chain_density, max_chain_length_norm, chain_centroid_r]  (3 floats)
    """
    WP = 1; BP = 7

    def _chain_stats(ptype: int, fwd: int) -> Tuple[int, int]:
        sqs = [(sq >> 3, sq & 7) for sq in range(64) if b[sq] == ptype]
        sq_set = set(sqs)
        chained = sum(1 for (r, f) in sqs
                      if (r - fwd, f-1) in sq_set or (r - fwd, f+1) in sq_set)
        max_len = 0
        for (r, f) in sqs:
            length = 1; cr, cf = r, f
            while True:
                nxt = [(cr + fwd, cf-1), (cr + fwd, cf+1)]
                found = next((n for n in nxt if n in sq_set), None)
                if found: cr, cf = found; length += 1
                else: break
            max_len = max(max_len, length)
        return chained, max_len

    w_chained, w_max = _chain_stats(WP, -1)
    b_chained, b_max = _chain_stats(BP,  1)
    all_pawns = [sq for sq in range(64) if b[sq] in {WP, BP}]
    total_pawns = len(all_pawns)
    chain_density = (w_chained + b_chained) / max(1, total_pawns)
    max_len_norm  = max(w_max, b_max) / 8.0
    centroid_r    = (sum(sq >> 3 for sq in all_pawns) / max(1, total_pawns) / 7.0
                     if all_pawns else 0.5)
    return [chain_density, max_len_norm, centroid_r]


# ─────────────────────────────────────────────────────────────────────────────
# G15: PIN AND X-RAY AXIS
# ─────────────────────────────────────────────────────────────────────────────

def pin_xray_encode(b: List[int]) -> List[float]:
    """
    Sliders (R/B/Q) aligned with enemy king on rank, file, or diagonal.
    Captures potential pin/x-ray pressure without requiring move generation.
    Output: [axis_r, axis_f, pin_density]  (3 floats)
    """
    W_SLIDERS = {3, 4, 5}; B_SLIDERS = {9, 10, 11}

    def _find_king(color_base: int) -> int:
        for sq in range(64):
            if b[sq] == color_base + 5: return sq
        return -1

    def _axis_score(slider_type_set, king_sq):
        if king_sq < 0: return 0.0, 0.0, 0.0
        kr = king_sq >> 3; kf = king_sq & 7
        aligned = []
        for sq in range(64):
            p = b[sq]
            if p not in slider_type_set: continue
            sr = sq >> 3; sf = sq & 7
            pt = p if p <= 6 else p - 6
            on_rank = sr == kr; on_file = sf == kf
            on_diag = abs(sr-kr) == abs(sf-kf) and sr != kr
            if pt in {4, 5} and (on_rank or on_file): aligned.append(sq)
            elif pt in {3, 5} and on_diag:            aligned.append(sq)
        if not aligned: return 0.0, 0.0, 0.0
        axis_r = sum(s >> 3 for s in aligned) / len(aligned) / 7.0
        axis_f = sum(s  & 7 for s in aligned) / len(aligned) / 7.0
        return axis_r, axis_f, min(len(aligned) / 4.0, 1.0)

    wk = _find_king(0); bk = _find_king(6)
    wr, wf, wd = _axis_score(W_SLIDERS, bk)
    br, bf, bd = _axis_score(B_SLIDERS, wk)
    density = (wd + bd) / 2.0
    axis_r  = (wr * wd + br * bd) / max(1e-6, wd + bd) if density > 0 else 0.5
    axis_f  = (wf * wd + bf * bd) / max(1e-6, wd + bd) if density > 0 else 0.5
    return [axis_r, axis_f, density]


# ─────────────────────────────────────────────────────────────────────────────
# G16: OPEN FILE DISTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────

def open_file_encode(b: List[int]) -> List[float]:
    """
    Open/semi-open files determine rook and queen activity lanes.
    File centroid and semi-open imbalance encode the board's "lane geometry."
    Output: [open_file_centroid_f, semi_imbalance, file_span]  (3 floats)
    """
    WP = 1; BP = 7
    w_files = {sq & 7 for sq in range(64) if b[sq] == WP}
    b_files = {sq & 7 for sq in range(64) if b[sq] == BP}
    open_files = [f for f in range(8) if f not in w_files and f not in b_files]
    w_semi = [f for f in range(8) if f not in w_files and f in b_files]
    b_semi = [f for f in range(8) if f in w_files and f not in b_files]
    open_centroid_f = (sum(open_files) / max(1, len(open_files)) / 7.0
                       if open_files else 0.5)
    semi_imbalance  = (len(w_semi) - len(b_semi)) / 8.0
    all_pawn_files  = list(w_files | b_files)
    file_span = ((max(all_pawn_files) - min(all_pawn_files)) / 7.0
                 if len(all_pawn_files) >= 2 else 0.0)
    return [open_centroid_f, semi_imbalance, file_span]


# ─────────────────────────────────────────────────────────────────────────────
# G17: KNIGHT OUTPOST DENSITY
# ─────────────────────────────────────────────────────────────────────────────

def knight_outpost_encode(b: List[int]) -> List[float]:
    """
    Outposts: squares that can't be attacked by enemy pawns (classic definition).
    Occupied outposts (knight on outpost square) are the strongest signal.
    Output: [w_outpost_norm, b_outpost_norm, outpost_centroid_r]  (3 floats)
    """
    WP = 1; BP = 7; WN = 2; BN = 8

    def _pawn_attacks(ptype: int, sq: int) -> bool:
        rank = sq >> 3; file = sq & 7
        dr = 1 if ptype == WP else -1
        atk_rank = rank + dr
        if not (0 <= atk_rank <= 7): return False
        return ((file > 0 and b[atk_rank*8 + file-1] == ptype) or
                (file < 7 and b[atk_rank*8 + file+1] == ptype))

    w_occ = 0; b_occ = 0; occ_sqs = []
    for sq in range(64):
        rank = sq >> 3
        if rank <= 3 and not _pawn_attacks(BP, sq):
            if b[sq] == WN: w_occ += 1; occ_sqs.append(sq)
        if rank >= 4 and not _pawn_attacks(WP, sq):
            if b[sq] == BN: b_occ += 1; occ_sqs.append(sq)

    centroid_r = (sum(sq >> 3 for sq in occ_sqs) / max(1, len(occ_sqs)) / 7.0
                  if occ_sqs else 0.5)
    return [min(w_occ, 4) / 4.0, min(b_occ, 4) / 4.0, centroid_r]


# ─────────────────────────────────────────────────────────────────────────────
# COMPOSITE ENCODER — all 17 geometries + active fold base
# Full FrostMatrix signal: 4 base + 51 extended = 55 dims per ply
# ─────────────────────────────────────────────────────────────────────────────

GEOMETRY_ENCODERS = [
    ('G1  tension_gradient',     tension_gradient_encode),
    ('G2  pawn_corridor',        pawn_corridor_encode),
    ('G3  king_shell',           king_shell_encode),
    ('G4  mobility_field',       mobility_field_encode),
    ('G5  color_complex',        color_complex_encode),
    ('G6  outpost_gravity',      outpost_gravity_encode),
    ('G7  phase_gradient',       phase_gradient_encode),
    ('G8  diagonal_axis',        diagonal_axis_encode),
    ('G9  territorial_frontier', territorial_frontier_encode),
    ('G10 resonance_cluster',    resonance_cluster_encode),
    ('G11 vertical_fold',        vertical_fold_encode),
    ('G12 horizontal_fold',      horizontal_fold_encode),
    ('G13 relationship_fold',    relationship_fold_encode),
    ('G14 pawn_chain',           pawn_chain_encode),
    ('G15 pin_xray',             pin_xray_encode),
    ('G16 open_file',            open_file_encode),
    ('G17 knight_outpost',       knight_outpost_encode),
]


def composite_encode(board: List[int]) -> List[float]:
    """
    Full FrostMatrix encoder.
    Returns a 55-dimensional feature vector:
      [0:4]   active_fold base (centroid_r, centroid_f, compression, outlier_frac)
      [4:55]  17 geometry encoders (G1-G17), concatenated

    This vector for each ply of a game forms the input to the FrostMatrix
    trajectory clustering space (L4-L5 of the render pipeline).
    """
    vec = active_fold_encode(board)
    for _, enc_fn in GEOMETRY_ENCODERS:
        vec.extend(enc_fn(board))
    return vec


# ─────────────────────────────────────────────────────────────────────────────
# DEMO
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    PIECE_MAP = {
        'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
        'p': 7, 'n': 8, 'b': 9, 'r': 10, 'q': 11, 'k': 12,
    }

    def board_from_fen(fen):
        board = [0] * 64
        sq = 0
        for ch in fen.split()[0]:
            if ch == '/': continue
            if ch.isdigit(): sq += int(ch)
            else: board[sq] = PIECE_MAP.get(ch, 0); sq += 1
        return board

    STARTING_FEN = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
    board = board_from_fen(STARTING_FEN)

    print("FrostMatrix geometry encoding — starting position")
    print(f"{'='*60}")
    vec = composite_encode(board)
    print(f"Feature vector: {len(vec)} dimensions")
    print()

    labels = ['centroid_r', 'centroid_f', 'compression', 'outlier_frac']
    print("BASE (active_fold, dims 0-3):")
    for i, lbl in enumerate(labels):
        print(f"  [{i:2d}] {lbl:<22} = {vec[i]:.4f}")
    print()

    offset = 4
    dims_per_encoder = [3,3,3,3,2,3,3,3,3,3,3,3,4,3,3,3,3]
    print("EXTENDED GEOMETRY ENCODERS (dims 4-54):")
    for (name, _), n_dims in zip(GEOMETRY_ENCODERS, dims_per_encoder):
        vals = vec[offset:offset+n_dims]
        vals_str = ', '.join(f'{v:.4f}' for v in vals)
        print(f"  [{offset:2d}:{offset+n_dims:2d}] {name:<26} = [{vals_str}]")
        offset += n_dims

    print()
    print(f"Total: {len(vec)} dims  (4 base + {len(vec)-4} extended from {len(GEOMETRY_ENCODERS)} encoders)")
