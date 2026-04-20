#!/usr/bin/env python3
"""
render_pipeline.py — FrostMatrix Rendering Pipeline

The FrostMatrix is the dojo's geometric model of how a chess game unfolds.
It is the equivalent of depth layers in a neural network — each stage lifts
raw positional data to a higher geometric abstraction. The more robust and
interlinked the FrostMatrix, the smarter the fighters it trains.

Transforms a raw game corpus into a versioned FrostMatrix package by
composing the existing fold / spiderweb / sankey systems as discrete
depth layers.

DEPTH LAYERS
  L1 | CorpusReader      raw games  →  (board[64][], result)
  L2 | PositionEncoder   board[64]  →  PositionEncoding (warmth, centroid, compression, fold_signature)
  L3 | TrajectoryBuilder game       →  GameTrajectory (centroid path + fold shapes per ply)
  L4 | PrefixAggregator  paths      →  per-prefix mean trajectory + variance + count
  L5 | FamilyClusterer   mean paths →  N geometric clusters → family centroids
  L6 | SankeyRelabeler   topology   →  existing sankey nodes relabeled with geometric family IDs
  L7 | TransitionBuilder game paths →  P[depth][family_i][family_j] transition surfaces
  L8 | GMatrixBuilder    nodes×fam  →  affinity[opening_node × N_families] (replaces name-based G)
  L9 | PackageBuilder               →  versioned, dojo-loadable opening_package_vN.json + .bin

AGNOSTIC DESIGN — the pipeline core knows nothing about chess.
  corpus_reader_fn(path) → Iterator[GameRecord]
      Yields raw game data. Default: PGN reader. Swap for JSONL, database, etc.
  encode_fn(board_ints) → PositionEncoding
      Maps board[64] ints to geometric encoding. Default: active_fold.
      Swap for any board encoder — this is the only chess-aware seam.
  distance_fn(vec_a, vec_b) → float
      Trajectory distance. Default: normalized Euclidean (same as spiderweb).
  n_families, max_depth, min_games — all parameters, never hardcoded.

EXISTING SYSTEMS CONSUMED (not modified):
  tools/active_fold.py              — compute_warmth, find_fold_center, active_fold_positions
  opening_prefix_analysis.json      — prefix tree topology (L4 seed)
  sankey_data.json                  — tree topology skeleton (L6 relabeling target)
  wildcard_opening_seeds.jsonl      — injected into package after L5

Usage:
  python3 render_pipeline.py [--config config.json]
  python3 render_pipeline.py --pgn /path/to/games.pgn --n-families 33 --max-depth 10
"""

import sys, os, json, math, hashlib, time, struct, argparse
from pathlib import Path
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from typing import Callable, Iterator, List, Tuple, Dict, Optional, Any

import numpy as np
import chess
import chess.pgn

# ── Import existing systems (never modify, only import) ──────────────────────
_here = Path(__file__).parent
sys.path.insert(0, str(_here))
from active_fold import compute_warmth, find_fold_center, active_fold_positions
# ──────────────────────���──────────────────────────────��───────────────────────

FROSTMATRIX_VERSION = '1.0'
PIPELINE_VERSION = FROSTMATRIX_VERSION  # alias


# ═════════��════════════════════════���════════════════════════════════════════════
# DATA STRUCTURES
# ═════════════════════════════���═════════════════════════════════════════════════

@dataclass
class GameRecord:
    """Agnostic game representation — no chess semantics here."""
    boards: List[List[int]]   # board[64] int at each ply (0=empty,1-6 white,7-12 black)
    result: str               # "1-0", "0-1", "1/2-1/2", "*"
    prefix: str               # space-joined UCI moves up to max_depth
    metadata: dict = field(default_factory=dict)

@dataclass
class PositionEncoding:
    """
    L2 output — geometric signature of a single board state.
    This is the measurement instrument: tells you WHERE the game is happening
    and HOW symmetric/contested the board is, without needing to know the rules.

    extended: additional geometry channels from the 10 FrostMatrix encoders.
    When present, feature_vec() returns a 33-dimensional vector (4 base + 29 extended).
    """
    warmth:         List[float]  # [64] per-square strategic weight
    centroid_r:     float        # rank of strategic center of gravity [0,7]
    centroid_f:     float        # file of strategic center of gravity [0,7]
    compression:    float        # fraction of squares in quiet zone (0=none, 1=all)
    outlier_frac:   float        # fraction of placed pieces in strategic zone
    fold_signature: int          # 64-bit mask: bit=1 → square in strategic zone
    extended:       Optional[List[float]] = None  # 29 additional geometry channels

    def feature_vec(self) -> List[float]:
        """Base 4 + optional 29 extended = up to 33-dimensional geometric vector."""
        base = [self.centroid_r, self.centroid_f, self.compression, self.outlier_frac]
        return base + self.extended if self.extended else base

@dataclass
class GameTrajectory:
    """L3 output — a game expressed as a path through geometric space."""
    prefix:      str
    result:      str
    encodings:   List[PositionEncoding]  # one per ply, length ≤ max_depth
    family_path: List[int] = field(default_factory=list)  # filled by L7

    def feature_matrix(self) -> np.ndarray:
        """Shape (max_depth, 4) — the game's geometric fingerprint."""
        return np.array([e.feature_vec() for e in self.encodings], dtype=np.float32)

    def fold_shape(self) -> List[int]:
        """Sequence of fold_signature bitmasks — the raytraced shape of this game."""
        return [e.fold_signature for e in self.encodings]

@dataclass
class FamilyCluster:
    """L5 output — one geometric monster family."""
    id:                    int
    centroid_feature:      np.ndarray    # (max_depth × fdims,) mean feature (all channels)
    variance_feature:      np.ndarray    # (max_depth × fdims,) per-dim variance
    centroid_trajectory:   List[List[float]]  # [(r,f) × max_depth] raw centroid path (ch 0,1)
    mean_fold_shape:       List[float]   # [max_depth] mean fold_signature (as float, avg of bitmasks)
    compression_profile:   List[float]   # [max_depth] mean compression per ply
    outlier_profile:       List[float]   # [max_depth] mean outlier_frac per ply
    game_count:            int
    example_prefixes:      List[str]
    # Disentangled signal: deviation of (r,f) from ply-mean across all families.
    # Raw centroid_trajectory is degenerate (~board center) because all openings start central.
    # delta shows what makes THIS family different from the average at each ply.
    delta_centroid_trajectory: List[List[float]] = field(default_factory=list)
    discriminant_channels:     List[int]          = field(default_factory=list)


# ═════���════════════════════════���══════════════════════════════════���═════════════
# PIPELINE CONFIG
# ══════════════���════════════════════════════════════════════════════════════════

DEFAULT_CONFIG: Dict[str, Any] = {
    # Input paths
    'pgn_path':        '/srv/models-hdd/chess-games/lichess_openings_2024_01.pgn',
    'prefix_analysis': '/srv/models-hdd/chess-games/opening_prefix_analysis.json',
    'sankey_path':     '/srv/models-hdd/chess-games/sankey_data.json',
    'wildcard_path':   '/srv/models-hdd/chess-games/wildcard_opening_seeds.jsonl',
    'output_dir':      '/srv/models-hdd/chess-games/',

    # Geometry parameters
    'max_depth':    10,    # plies to trace per game
    'n_families':   33,    # number of geometric clusters
    'min_games':    20,    # minimum games for a prefix to enter L5
    'kmeans_iters': 150,   # k-means iterations
    'kmeans_runs':  5,     # independent restarts (best inertia wins)

    # Pluggable interfaces (None = use defaults defined below)
    'encode_fn':         None,  # board[64] → PositionEncoding
    'distance_fn':       None,  # (vec_a, vec_b) → float
    'corpus_reader_fn':  None,  # path → Iterator[GameRecord]

    # Output
    'package_prefix': 'opening_package',
    'log_every':      25_000,
}


# ═════════════���════════════════════��══════════════════════���═════════════════════
# L1 | CORPUS READER — default: PGN
# Agnostic contract: yields GameRecord(boards, result, prefix, metadata)
# ═════════��════════════════════════════════════════════════════════════���════════

_PIECE_MAP = {'P':1,'N':2,'B':3,'R':4,'Q':5,'K':6,
              'p':7,'n':8,'b':9,'r':10,'q':11,'k':12}

def _board_to_ints(board: chess.Board) -> List[int]:
    """python-chess Board → plain int[64] (0=empty, 1-6 white, 7-12 black)."""
    out = [0] * 64
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p:
            base = p.piece_type  # 1-6
            out[sq] = base if p.color == chess.WHITE else base + 6
    return out

def pgn_corpus_reader(pgn_path: str, max_depth: int) -> Iterator[GameRecord]:
    """
    Default corpus reader: streams PGN, yields GameRecord per game.
    Applies moves up to max_depth, capturing board state at each ply.
    All chess-specific logic is isolated here.
    """
    with open(pgn_path, encoding='utf-8', errors='replace') as fh:
        while True:
            game = chess.pgn.read_game(fh)
            if game is None:
                break

            result = game.headers.get('Result', '*')
            board  = game.board()
            boards = [_board_to_ints(board)]  # ply 0 = starting position
            moves_uci = []

            for move in game.mainline_moves():
                if len(moves_uci) >= max_depth:
                    break
                board.push(move)
                boards.append(_board_to_ints(board))
                moves_uci.append(move.uci())

            if len(moves_uci) < 2:  # too short to be meaningful
                continue

            yield GameRecord(
                boards  = boards,
                result  = result,
                prefix  = ' '.join(moves_uci),
                metadata= {
                    'white_elo': game.headers.get('WhiteElo'),
                    'black_elo': game.headers.get('BlackElo'),
                },
            )


# ════════════════════════���══════════════════════════════════════════════════════
# L2 | POSITION ENCODER — default: active_fold
# Maps board[64] ints → PositionEncoding
# Agnostic: swap this function to change the geometric basis of the pipeline.
# ════════════════════════════════════════════════════════════════════��══════════

def active_fold_encode(board_ints: List[int]) -> PositionEncoding:
    """
    Default encoder. Delegates entirely to existing active_fold.py — no
    re-implementation. Extracts 4 geometric features + fold_signature.
    """
    warmth                         = compute_warmth(board_ints)
    fold_r, fold_f                 = find_fold_center(warmth)
    canon_pos, outlier_mask, _, _  = active_fold_positions(board_ints)

    # Compression: fraction of squares mapped to a shared (quiet-zone) canonical slot
    quiet_count = sum(1 for i in range(64) if not outlier_mask[i])
    compression = quiet_count / 64.0

    # Outlier fraction: among placed pieces, how many are in strategic zone
    placed = [i for i in range(64) if board_ints[i] != 0]
    outlier_frac = (sum(outlier_mask[i] for i in placed) / len(placed)) if placed else 0.0

    # Fold signature: 64-bit int where bit i = 1 iff square i is in strategic zone
    fold_sig = 0
    for i in range(64):
        if outlier_mask[i]:
            fold_sig |= (1 << i)

    return PositionEncoding(
        warmth         = warmth,
        centroid_r     = fold_r,
        centroid_f     = fold_f,
        compression    = compression,
        outlier_frac   = outlier_frac,
        fold_signature = fold_sig,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# FROSTMATRIX GEOMETRY ENCODERS — 10 novel board geometry signals
#
# Each encoder: board_ints[64] → List[float]
# All are chess-specific but fully agnostic to the pipeline — they are
# swappable defaults, not hardwired logic. Each captures a fundamentally
# different geometric property of the position.
#
# Piece type conventions (board_ints):
#   0=empty  1=wP 2=wN 3=wB 4=wR 5=wQ 6=wK  7=bP 8=bN 9=bB 10=bR 11=bQ 12=bK
# ═══════════════════════════════════════════════════════════════════════════════

# Phase weights (same as tapered eval)
_PH_W = [0, 0, 1, 1, 2, 4, 0]   # indexed by piece_type (1-6)

def _pt(p):   return p if p <= 6 else p - 6          # piece type 1-6
def _isw(p):  return 1 <= p <= 6                      # is white
def _isb(p):  return 7 <= p <= 12                     # is black
def _wpw(p):  return [0,1,3,3,5,9,4][_pt(p)] if p else 0  # strategic weight


# ── G1: Tension Gradient Eigenvector ─────────────────────────────────────────
# Every piece radiates force along its attack rays. Sum gives a 2D vector field.
# The dominant eigenvector is the main axis of conflict.
# Output: [axis_r, axis_f, axis_strength]  (3 floats)

_RAY_DIRS = {
    2: [(-2,-1),(-2,1),(-1,-2),(-1,2),(1,-2),(1,2),(2,-1),(2,1)],  # knight
    3: [(-1,-1),(-1,1),(1,-1),(1,1)],   # bishop
    4: [(-1,0),(1,0),(0,-1),(0,1)],     # rook
    5: [(-1,-1),(-1,1),(1,-1),(1,1),(-1,0),(1,0),(0,-1),(0,1)],  # queen
    6: [(-1,-1),(-1,1),(1,-1),(1,1),(-1,0),(1,0),(0,-1),(0,1)],  # king
    1: [],   # pawn handled separately
}

def tension_gradient_encode(b: List[int]) -> List[float]:
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
        if pt == 1:  # pawn attacks diagonally forward
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


# ── G2: Pawn Corridor Topology ────────────────────────────────────────────────
# Pawns as walls dividing the board into corridors. Encodes the open/closed
# topology of each file and the structural asymmetry between flanks.
# Output: [open_file_centroid_f, semi_open_imbalance, closure_index]  (3 floats)

def pawn_corridor_encode(b: List[int]) -> List[float]:
    open_files, semi_w, semi_b = [], [], []
    for f in range(8):
        has_wp = any(b[r*8+f] == 1 for r in range(8))
        has_bp = any(b[r*8+f] == 7 for r in range(8))
        if not has_wp and not has_bp: open_files.append(f)
        elif not has_wp and has_bp:   semi_w.append(f)  # open for white advance
        elif has_wp and not has_bp:   semi_b.append(f)  # open for black advance
    open_centroid = sum(open_files) / len(open_files) / 7.0 if open_files else 0.5
    semi_imbalance = (len(semi_w) - len(semi_b)) / 8.0   # [-1, 1]
    closure = 1.0 - (len(open_files) + 0.5*(len(semi_w)+len(semi_b))) / 8.0
    return [open_centroid, semi_imbalance, closure]


# ── G3: King Shell Radiance ───────────────────────────────────────────────────
# Concentric Chebyshev rings around each king. The overlap zone (contested
# safety space) and shield asymmetry are key structural signals.
# Output: [king_distance, overlap_norm, shield_asymmetry]  (3 floats)

def king_shell_encode(b: List[int]) -> List[float]:
    wk = next((sq for sq in range(64) if b[sq] == 6), None)
    bk = next((sq for sq in range(64) if b[sq] == 12), None)
    if wk is None or bk is None: return [0.0, 0.0, 0.0]
    wkr, wkf = wk >> 3, wk & 7
    bkr, bkf = bk >> 3, bk & 7
    king_dist = max(abs(wkr-bkr), abs(wkf-bkf)) / 7.0   # Chebyshev, normalized

    # Shell sizes: count friendly pieces within radius 2 of each king
    w_shield, b_shield = 0, 0
    for sq in range(64):
        p = b[sq]
        if not p: continue
        r, f = sq >> 3, sq & 7
        if _isw(p) and max(abs(r-wkr), abs(f-wkf)) <= 2: w_shield += 1
        if _isb(p) and max(abs(r-bkr), abs(f-bkf)) <= 2: b_shield += 1

    # Overlap: squares within radius 2 of BOTH kings
    overlap = sum(1 for sq in range(64)
                  if max(abs((sq>>3)-wkr), abs((sq&7)-wkf)) <= 2
                  and max(abs((sq>>3)-bkr), abs((sq&7)-bkf)) <= 2) / 25.0

    shield_asym = (w_shield - b_shield) / 9.0   # normalized by max
    return [king_dist, overlap, shield_asym]


# ── G4: Mobility Activity Field ───────────────────────────────────────────────
# Mobility-weighted centroid: WHERE pieces are effectively active, not just
# where they sit. Uses estimated mobility (not full legal move generation).
# Output: [mobility_centroid_r, mobility_centroid_f, mobility_imbalance]  (3 floats)

_MOB_BASE = {1:2, 2:4, 3:7, 4:7, 5:14, 6:3}   # rough max mobility per piece type

def _estimate_mobility(b: List[int], sq: int) -> float:
    """Estimate mobility as proportion of available directions that aren't blocked by friendlies."""
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


# ── G5: Color Complex Imbalance ───────────────────────────────────────────────
# Dark/light square piece density differential per side.
# Reveals bad bishops, color weaknesses, and bishop-of-opposite-color structures.
# Output: [light_sq_advantage, dark_sq_advantage]  (2 floats)

def color_complex_encode(b: List[int]) -> List[float]:
    wl, wd, bl, bd = 0.0, 0.0, 0.0, 0.0
    for sq in range(64):
        p = b[sq]
        if not p: continue
        w = _wpw(p)
        light = (sq ^ (sq >> 3)) & 1   # 1 if light square
        if _isw(p):
            if light: wl += w
            else:     wd += w
        else:
            if light: bl += w
            else:     bd += w
    total = max(wl+wd+bl+bd, 1.0)
    return [(wl - bl) / total, (wd - bd) / total]


# ── G6: Outpost Gravity Wells ─────────────────────────────────────────────────
# Squares that cannot be attacked by enemy pawns and are defended by friendly
# pawns act as gravitational anchors for the game's structural balance.
# Output: [outpost_centroid_r, outpost_centroid_f, outpost_density]  (3 floats)

def outpost_gravity_encode(b: List[int]) -> List[float]:
    # White outpost: sq where no bP can attack and some wP defends
    white_outposts, black_outposts = [], []
    for sq in range(64):
        r, f = sq >> 3, sq & 7
        # White outpost (in enemy half: r < 4)
        if r < 4:
            bp_attack = any(b[nr*8+nf] == 7
                           for nr, nf in [(r+1, f-1),(r+1, f+1)]
                           if 0 <= nr < 8 and 0 <= nf < 8)
            wp_defend = any(b[nr*8+nf] == 1
                           for nr, nf in [(r+1, f-1),(r+1, f+1)]
                           if 0 <= nr < 8 and 0 <= nf < 8)
            if not bp_attack and wp_defend: white_outposts.append(sq)
        # Black outpost (in enemy half: r > 3)
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
    density = len(all_outposts) / 16.0   # normalized by theoretical max
    return [cr, cf, density]


# ── G7: Phase Topology Gradient ───────────────────────────────────────────────
# Per-quadrant material phase. The gradient vector points from the most
# endgame-like quadrant toward the most middlegame-like.
# Output: [gradient_r, gradient_f, phase_range]  (3 floats)

def phase_gradient_encode(b: List[int]) -> List[float]:
    phase_q = [0.0, 0.0, 0.0, 0.0]   # Q0=top-left Q1=top-right Q2=bot-left Q3=bot-right
    for sq in range(64):
        p = b[sq]
        if not p: continue
        pt = _pt(p)
        r, f = sq >> 3, sq & 7
        q = (1 if f >= 4 else 0) + (2 if r >= 4 else 0)
        phase_q[q] += _PH_W[pt] if pt < len(_PH_W) else 0

    max_q = max(phase_q) or 1.0
    # Gradient: top half - bottom half for r, right half - left half for f
    top = (phase_q[0] + phase_q[1]) / (2 * max_q)
    bot = (phase_q[2] + phase_q[3]) / (2 * max_q)
    lft = (phase_q[0] + phase_q[2]) / (2 * max_q)
    rgt = (phase_q[1] + phase_q[3]) / (2 * max_q)
    grad_r = (top - bot)
    grad_f = (rgt - lft)
    phase_range = (max(phase_q) - min(phase_q)) / max_q
    return [grad_r, grad_f, phase_range]


# ── G8: Diagonal Load Axis ────────────────────────────────────────────────────
# The most piece-loaded diagonal in each direction defines an oblique fold axis.
# Separates diagonal-pressure openings (Sicilian, Nimzo) from orthogonal ones.
# Output: [ne_load, nw_load, diagonal_dominance]  (3 floats)
# ne_load: normalized weight on the heaviest NE diagonal (a1-h8 family)
# nw_load: normalized weight on the heaviest NW diagonal (a8-h1 family)

def diagonal_axis_encode(b: List[int]) -> List[float]:
    # NE diagonals: sq where (r - f) is constant, range -7..7
    # NW diagonals: sq where (r + f) is constant, range 0..14
    ne_diag = [0.0] * 15   # index = (r-f)+7
    nw_diag = [0.0] * 15   # index = r+f
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
    dominance = abs(ne_max - nw_max)   # how much one axis dominates
    return [ne_max, nw_max, dominance]


# ── G9: Territorial Frontier ─────────────────────────────────────────────────
# Control differential per square → boundary between white/black territory.
# Frontier geometry (centroid, fragmentation) characterizes strategic contact.
# Output: [frontier_centroid_r, frontier_centroid_f, fragmentation]  (3 floats)

def territorial_frontier_encode(b: List[int]) -> List[float]:
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

    # Frontier: squares where control crosses zero (contested boundary)
    frontier = [sq for sq in range(64)
                if control[sq] == 0
                or any(control[sq] * control[(sq>>3)*8 + (sq&7) + d] < 0
                       for d in [-8, 8, -1, 1]
                       if 0 <= (sq>>3)*8+(sq&7)+d < 64)]

    if not frontier: return [3.5/7, 3.5/7, 0.0]
    cr = sum(sq >> 3 for sq in frontier) / len(frontier) / 7.0
    cf = sum(sq & 7  for sq in frontier) / len(frontier) / 7.0
    # Fragmentation: variance of frontier squares (high = scattered islands)
    var = sum(((sq>>3)/7.0 - cr)**2 + ((sq&7)/7.0 - cf)**2 for sq in frontier)
    fragmentation = min(var / len(frontier), 1.0)
    return [cr, cf, fragmentation]


# ── G10: Resonance Cluster Signature ─────────────────────────────────────────
# Coordinated piece pairs (batteries, outpost+defender, rook+queen lift).
# WHERE pieces resonate together and HOW strongly defines the attack signature.
# Output: [resonance_centroid_r, resonance_centroid_f, resonance_strength]  (3 floats)

def resonance_cluster_encode(b: List[int]) -> List[float]:
    rr, rf, strength = 0.0, 0.0, 0.0
    for sq1 in range(64):
        p1 = b[sq1]
        if not p1: continue
        r1, f1 = sq1 >> 3, sq1 & 7
        for sq2 in range(sq1+1, 64):
            p2 = b[sq2]
            if not p2: continue
            if _isw(p1) != _isw(p2): continue   # same side only
            r2, f2 = sq2 >> 3, sq2 & 7
            pt1, pt2 = _pt(p1), _pt(p2)
            # Battery: same rank, file, or diagonal
            on_rank = r1 == r2
            on_file = f1 == f2
            on_diag = abs(r1-r2) == abs(f1-f2)
            if not (on_rank or on_file or on_diag): continue
            # Resonance weight: product of piece weights
            w = (_wpw(p1) * _wpw(p2)) ** 0.5
            cr = (r1 + r2) / 2.0
            cf = (f1 + f2) / 2.0
            rr += cr * w; rf += cf * w; strength += w

    if strength == 0: return [3.5/7, 3.5/7, 0.0]
    return [rr / strength / 7.0, rf / strength / 7.0, min(strength / 50.0, 1.0)]


# ── G11: Vertical Fold Compression Signal ────────────────────────────────────
# From measure_folding_compression.py fold_vertical():
# Mirrors files a-d ↔ h-e per rank. Compression = symmetric pairs; asymmetry =
# broken symmetry (one side empty or different piece). Captures file imbalance.
# Output: [compression_ratio, asymmetry_score, outlier_centroid_f]  (3 floats)

def vertical_fold_encode(b: List[int]) -> List[float]:
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


# ── G12: Horizontal Fold Invasion Signal ──────────────────────────────────────
# From measure_folding_compression.py fold_horizontal():
# Mirrors white/black halves (ranks 1-4 ↔ 5-8). Outlier = a piece that has
# crossed the center. Captures invasion depth and territorial imbalance.
# Output: [compression_ratio, invasion_density, rank_boundary_pressure]  (3 floats)

def horizontal_fold_encode(b: List[int]) -> List[float]:
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
                if tp in WHITE:          # white piece deep in black's territory
                    invasion_ranks.append(rank)
                elif bp in BLACK:        # black piece deep in white's territory
                    invasion_ranks.append(7 - rank)
    compression      = symmetric / max(1, total)
    invasion_density = len(invasion_ranks) / 64.0
    # rank_pressure: closer to 0 = deeper invasion (more dangerous)
    rank_pressure = ((4 - sum(invasion_ranks) / max(1, len(invasion_ranks))) / 4.0
                     if invasion_ranks else 0.0)
    return [compression, invasion_density, rank_pressure]


# ── G13: Relationship Fold Zone Co-occupancy ─────────────────────────────────
# From measure_folding_compression.py fold_relationship():
# 16 symmetric quadrant zones. White + black pieces in same zone = contested.
# Captures the geometric distribution of inter-side contact.
# Output: [conflict_density, max_conflict_norm, zone_centroid_r, zone_centroid_f]  (4 floats)

def relationship_fold_encode(b: List[int]) -> List[float]:
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


# ── G14: Pawn Chain Topology ──────────────────────────────────────────────────
# Pawn chains: pawns diagonally defending same-color pawns. Chain density and
# max length capture structural pawn geometry (hedgehog, maroczy, IQP etc.)
# Output: [chain_density, max_chain_length_norm, chain_centroid_r]  (3 floats)

def pawn_chain_encode(b: List[int]) -> List[float]:
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

    w_chained, w_max = _chain_stats(WP, -1)   # white advances toward rank 0
    b_chained, b_max = _chain_stats(BP,  1)   # black advances toward rank 7
    all_pawns = [sq for sq in range(64) if b[sq] in {WP, BP}]
    total_pawns = len(all_pawns)
    chain_density = (w_chained + b_chained) / max(1, total_pawns)
    max_len_norm  = max(w_max, b_max) / 8.0
    centroid_r    = (sum(sq >> 3 for sq in all_pawns) / max(1, total_pawns) / 7.0
                     if all_pawns else 0.5)
    return [chain_density, max_len_norm, centroid_r]


# ── G15: Pin and X-Ray Axis ───────────────────────────────────────────────────
# Sliders (R/B/Q) aligned with enemy king on rank, file, or diagonal.
# Captures potential pin/x-ray pressure without requiring move generation.
# Output: [axis_r, axis_f, pin_density]  (3 floats)

def pin_xray_encode(b: List[int]) -> List[float]:
    W_SLIDERS = {3, 4, 5}; B_SLIDERS = {9, 10, 11}

    def _find_king(color_base: int) -> int:
        for sq in range(64):
            if b[sq] == color_base + 5: return sq   # K=6 for white, k=12 for black
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


# ── G16: Open File Distribution ───────────────────────────────────────────────
# Open/semi-open files determine rook and queen activity lanes. File centroid and
# semi-open imbalance encode the board's "lane geometry."
# Output: [open_file_centroid_f, semi_imbalance, file_span]  (3 floats)

def open_file_encode(b: List[int]) -> List[float]:
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


# ── G17: Knight Outpost Density ───────────────────────────────────────────────
# Outposts: squares that can't be attacked by enemy pawns (classic outpost def).
# Occupied outposts (knight on outpost square) are the strongest signal.
# Output: [w_outpost_norm, b_outpost_norm, outpost_centroid_r]  (3 floats)

def knight_outpost_encode(b: List[int]) -> List[float]:
    WP = 1; BP = 7; WN = 2; BN = 8

    def _pawn_attacks(ptype: int, sq: int) -> bool:
        rank = sq >> 3; file = sq & 7
        dr = 1 if ptype == WP else -1   # white pawns attack upward (rank-1), black downward
        atk_rank = rank + dr
        if not (0 <= atk_rank <= 7): return False
        return ((file > 0 and b[atk_rank*8 + file-1] == ptype) or
                (file < 7 and b[atk_rank*8 + file+1] == ptype))

    w_occ = 0; b_occ = 0; occ_sqs = []
    for sq in range(64):
        rank = sq >> 3
        if rank <= 3 and not _pawn_attacks(BP, sq):  # can't be attacked by black pawn
            if b[sq] == WN: w_occ += 1; occ_sqs.append(sq)
        if rank >= 4 and not _pawn_attacks(WP, sq):  # can't be attacked by white pawn
            if b[sq] == BN: b_occ += 1; occ_sqs.append(sq)

    centroid_r = (sum(sq >> 3 for sq in occ_sqs) / max(1, len(occ_sqs)) / 7.0
                  if occ_sqs else 0.5)
    return [min(w_occ, 4) / 4.0, min(b_occ, 4) / 4.0, centroid_r]


# ── COMPOSITE ENCODER — all 17 geometries + active fold ──────────────────────
# Full FrostMatrix signal: 4 (active fold) + 51 (17 encoders) = 55 dims/ply.
# Feature vector: 55 × (max_depth+1) = 605 dimensions for clustering.
#
# Encoder breakdown (51 extended channels):
#   G1  tension_gradient     3   G10 resonance_cluster   3
#   G2  pawn_corridor        3   G11 vertical_fold       3
#   G3  king_shell           3   G12 horizontal_fold     3
#   G4  mobility_field       3   G13 relationship_fold   4
#   G5  color_complex        2   G14 pawn_chain          3
#   G6  outpost_gravity      3   G15 pin_xray            3
#   G7  phase_gradient       3   G16 open_file           3
#   G8  diagonal_axis        3   G17 knight_outpost      3
#   G9  territorial_frontier 3

GEOMETRY_ENCODERS = [
    ('tension_gradient',     tension_gradient_encode),
    ('pawn_corridor',        pawn_corridor_encode),
    ('king_shell',           king_shell_encode),
    ('mobility_field',       mobility_field_encode),
    ('color_complex',        color_complex_encode),
    ('outpost_gravity',      outpost_gravity_encode),
    ('phase_gradient',       phase_gradient_encode),
    ('diagonal_axis',        diagonal_axis_encode),
    ('territorial_frontier', territorial_frontier_encode),
    ('resonance_cluster',    resonance_cluster_encode),
    ('vertical_fold',        vertical_fold_encode),
    ('horizontal_fold',      horizontal_fold_encode),
    ('relationship_fold',    relationship_fold_encode),
    ('pawn_chain',           pawn_chain_encode),
    ('pin_xray',             pin_xray_encode),
    ('open_file',            open_file_encode),
    ('knight_outpost',       knight_outpost_encode),
]

def composite_encode(board_ints: List[int]) -> PositionEncoding:
    """
    Full FrostMatrix encoder: active_fold base (4) + all 10 geometry signals (29) = 33 dims.
    Each geometry encoder is independent and orthogonal — they measure different
    structural properties of the board and will separate family clusters that
    the base active_fold centroid alone cannot distinguish.
    """
    base = active_fold_encode(board_ints)
    extended = []
    for _, enc_fn in GEOMETRY_ENCODERS:
        extended.extend(enc_fn(board_ints))
    base.extended = extended
    return base


# ═══════════════════════════════════════════════════════════════════════════════
# L3 | TRAJECTORY BUILDER
# Game (list of boards) → GameTrajectory (centroid path + fold shapes)
# ═════════════════════��════════════════════════════════════════════════════════��

def build_trajectory(record: GameRecord, encode_fn: Callable, max_depth: int) -> GameTrajectory:
    encodings = []
    for board_ints in record.boards[:max_depth + 1]:
        encodings.append(encode_fn(board_ints))
    return GameTrajectory(
        prefix    = record.prefix,
        result    = record.result,
        encodings = encodings,
    )


# ═══════════════════════════════════���══════════════════════════════���════════════
# L4 | PREFIX AGGREGATOR
# Many game trajectories → per-prefix: mean feature matrix, variance, count
# Uses the existing prefix_analysis.json topology as a validation scaffold.
# ════════════════════════════════════════════════════════════════��══════════════

class PrefixAggregator:
    """
    For every sub-prefix of every game, accumulates the FULL trajectory of that game.
    This means prefix "e2e4" knows not just what ply-1 looks like, but what the entire
    game trajectory looked like for all games that opened with 1.e4.
    That full trajectory is the geometric signature used for family clustering.
    feature_dims must match the output of the encode_fn (e.g. 55 for composite_encode).
    """
    def __init__(self, max_depth: int, feature_dims: int = 55):
        self.max_depth    = max_depth
        self.fdims        = feature_dims
        D                 = max_depth + 1
        self._sums        = defaultdict(lambda: np.zeros((D, feature_dims), np.float64))
        self._sq_sums     = defaultdict(lambda: np.zeros((D, feature_dims), np.float64))
        self._fold_sums   = defaultdict(lambda: np.zeros(D, np.float64))
        self._depth_hits  = defaultdict(lambda: np.zeros(D, np.int32))  # per-ply game count
        self._game_counts = defaultdict(int)

    def add(self, traj: GameTrajectory):
        moves = traj.prefix.split()
        n_enc = len(traj.encodings)
        # For every sub-prefix this game passed through, accumulate its full trajectory.
        # Prefix "" gets the full trajectory of every game.
        # Prefix "e2e4 c7c5" gets full trajectories of all Sicilian games.
        for prefix_depth in range(min(len(moves) + 1, self.max_depth + 1)):
            sub_prefix = ' '.join(moves[:prefix_depth])
            self._game_counts[sub_prefix] += 1
            for enc_depth in range(n_enc):
                enc = traj.encodings[enc_depth]
                vec = np.array(enc.feature_vec(), dtype=np.float64)
                self._sums[sub_prefix][enc_depth]      += vec
                self._sq_sums[sub_prefix][enc_depth]   += vec ** 2
                self._fold_sums[sub_prefix][enc_depth] += enc.fold_signature
                self._depth_hits[sub_prefix][enc_depth] += 1

    def get(self, min_games: int) -> Dict[str, dict]:
        """Returns prefix → {mean, variance, fold_mean, count} for prefixes ≥ min_games."""
        result = {}
        for prefix, count in self._game_counts.items():
            if count < min_games:
                continue
            hits  = self._depth_hits[prefix].copy()
            hits[hits == 0] = 1  # avoid div-by-zero for depths no game reached
            mean  = (self._sums[prefix] / hits[:, None]).astype(np.float32)
            var   = ((self._sq_sums[prefix] / hits[:, None]) - mean.astype(np.float64) ** 2).astype(np.float32)
            fold  = (self._fold_sums[prefix] / hits).astype(np.float32)
            result[prefix] = {
                'mean':      mean,   # (max_depth+1, 4) — full trajectory
                'variance':  var,
                'fold_mean': fold,   # (max_depth+1,) — mean strategic zone bitmap
                'count':     count,
            }
        return result


# ═════════════════���═══════════════════════════���═════════════════════════════���═══
# L5 | FAMILY CLUSTERER — k-means on trajectory feature space
# Normalizes per-dimension (same as spiderweb distance metric), then clusters.
# ════════════════════════════���═══════════════════════════════��══════════════════

def _normalize_matrix(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-column normalization: (X - mean) / std. Returns (X_norm, means, stds)."""
    means = X.mean(axis=0)
    stds  = X.std(axis=0)
    stds[stds < 1e-8] = 1.0   # avoid division by zero on constant features
    return (X - means) / stds, means, stds

def _kmeans(X: np.ndarray, k: int, max_iter: int = 150, n_runs: int = 5, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    K-means with multiple restarts. Returns (labels, centers) for best inertia run.
    X: (N, D) float32. Uses normalized Euclidean distance (same as spiderweb metric).
    """
    best_labels, best_centers, best_inertia = None, None, np.inf
    rng = np.random.RandomState(seed)

    for run in range(n_runs):
        # k-means++ init
        idx = [rng.randint(0, len(X))]
        for _ in range(k - 1):
            chosen = X[idx]                              # (len(idx), D)
            diff   = X[:, None, :] - chosen[None, :, :] # (N, len(idx), D)
            d2     = (diff ** 2).sum(axis=2).min(axis=1) # (N,) min sq dist to any chosen
            d2    /= d2.sum()
            idx.append(rng.choice(len(X), p=d2))
        centers = X[idx].copy()

        labels = np.zeros(len(X), dtype=np.int32)
        for it in range(max_iter):
            # Assign: (N, k) distance matrix
            diffs   = X[:, None, :] - centers[None, :, :]   # (N, k, D)
            dists   = np.sqrt((diffs ** 2).sum(axis=2))       # (N, k)
            new_labels = dists.argmin(axis=1)
            if np.all(new_labels == labels) and it > 0:
                break
            labels = new_labels
            # Update centers
            for j in range(k):
                mask = labels == j
                if mask.any():
                    centers[j] = X[mask].mean(axis=0)

        inertia = sum(np.linalg.norm(X[i] - centers[labels[i]]) ** 2 for i in range(len(X)))
        if inertia < best_inertia:
            best_inertia, best_labels, best_centers = inertia, labels.copy(), centers.copy()
        print(f'    [clusterer] run {run+1}/{n_runs}  inertia={inertia:.2f}', flush=True)

    return best_labels, best_centers

def cluster_families(agg: Dict[str, dict], n_families: int,
                     max_depth: int, kmeans_iters: int, kmeans_runs: int) -> Tuple[List[FamilyCluster], Dict[str, int]]:
    """
    L5: clusters aggregated prefix trajectories into n_families geometric families.
    Returns (families, prefix_to_family_id).
    """
    prefixes  = list(agg.keys())
    # Feature vector per prefix: flatten mean matrix — dim = (max_depth+1) × fdims
    # fdims is auto-detected from the aggregated data (supports any encoder)
    sample_mean = agg[prefixes[0]]['mean']
    feat_dim  = sample_mean.size   # (max_depth+1) * n_geo_channels
    X_raw     = np.zeros((len(prefixes), feat_dim), dtype=np.float32)
    for i, p in enumerate(prefixes):
        X_raw[i] = agg[p]['mean'].flatten()

    print(f'  [L5] clustering {len(prefixes)} prefixes into {n_families} families '
          f'(feature dim={feat_dim})...', flush=True)

    X_norm, feat_means, feat_stds = _normalize_matrix(X_raw)
    labels, centers_norm = _kmeans(X_norm, k=n_families,
                                   max_iter=kmeans_iters, n_runs=kmeans_runs)
    # Denormalize centers back to raw feature space
    centers_raw = centers_norm * feat_stds + feat_means

    # Build FamilyCluster objects
    prefix_to_fid: Dict[str, int] = {}
    families: List[Optional[FamilyCluster]] = [None] * n_families

    for fid in range(n_families):
        mask = labels == fid
        member_prefixes = [prefixes[i] for i in range(len(prefixes)) if mask[i]]
        counts = [agg[p]['count'] for p in member_prefixes]
        total  = sum(counts)

        # Centroid trajectory: mean centroid path in (r, f) space
        # channels 0,1 are always centroid_r, centroid_f regardless of total fdims
        fdims    = feat_dim // (max_depth + 1)
        mean_mat = centers_raw[fid].reshape(max_depth + 1, fdims)
        centroid_traj = [[float(mean_mat[d, 0]), float(mean_mat[d, 1])]
                         for d in range(max_depth + 1)]

        # Mean fold shape: weighted average fold_signature across members
        fold_shape = np.zeros(max_depth + 1, dtype=np.float64)
        comp_profile = np.zeros(max_depth + 1, dtype=np.float64)
        out_profile  = np.zeros(max_depth + 1, dtype=np.float64)
        for p, c in zip(member_prefixes, counts):
            w = c / total if total > 0 else 1.0 / len(member_prefixes)
            fold_shape   += agg[p]['fold_mean']  * w
            comp_profile += agg[p]['mean'][:, 2] * w  # compression channel (idx 2)
            out_profile  += agg[p]['mean'][:, 3] * w  # outlier_frac channel (idx 3)

        # Sort examples by count descending
        example_prefixes = sorted(member_prefixes, key=lambda p: -agg[p]['count'])[:5]

        # Variance: mean across members of their per-prefix variance
        var_mat = np.zeros_like(agg[prefixes[0]]['variance'])
        for p, c in zip(member_prefixes, counts):
            w = c / total if total > 0 else 1.0 / len(member_prefixes)
            var_mat += agg[p]['variance'] * w

        families[fid] = FamilyCluster(
            id                  = fid,
            centroid_feature    = centers_raw[fid],
            variance_feature    = var_mat.flatten(),
            centroid_trajectory = centroid_traj,
            mean_fold_shape     = fold_shape.tolist(),
            compression_profile = comp_profile.tolist(),
            outlier_profile     = out_profile.tolist(),
            game_count          = total,
            example_prefixes    = example_prefixes,
        )
        prefix_to_fid.update({p: fid for p in member_prefixes})

    # ── Disentangle: compute delta trajectories ───────────────────────────────
    # Raw centroid_r/f (channels 0,1) are near-identical across families because
    # all openings start near the board center. Subtract ply-mean to expose what
    # makes each family geometrically distinct.
    fdims_outer = feat_dim // (max_depth + 1)
    all_centers_mat = np.stack([
        centers_raw[fid].reshape(max_depth + 1, fdims_outer)
        for fid in range(n_families)
    ])  # (n_families, max_depth+1, fdims)

    # Inter-family variance: which channels actually separate families?
    inter_fam_var    = all_centers_mat.var(axis=0).mean(axis=0)   # (fdims,)
    top_disc_channels = np.argsort(inter_fam_var)[::-1][:10].tolist()

    ply_mean_mat = all_centers_mat.mean(axis=0)   # (max_depth+1, fdims)

    for fid in range(n_families):
        fam_mat = centers_raw[fid].reshape(max_depth + 1, fdims_outer)
        delta   = fam_mat - ply_mean_mat
        families[fid].delta_centroid_trajectory = [
            [float(delta[d, 0]), float(delta[d, 1])]
            for d in range(max_depth + 1)
        ]
        families[fid].discriminant_channels = top_disc_channels

    top5_info = [f'ch{c}={inter_fam_var[c]:.4f}' for c in top_disc_channels[:5]]
    print(f'  [L5] top discriminant channels: {top5_info}', flush=True)
    print(f'  [L5] centroid_r/f inter-family var: '
          f'ch0={inter_fam_var[0]:.5f}  ch1={inter_fam_var[1]:.5f}  '
          f'(delta range: ±{float(np.abs(ply_mean_mat[:,0]).max()):.2f})', flush=True)

    return families, prefix_to_fid


# ═════════════��═══════════════════════════���═════════════════════════════════════
# L6 | SANKEY RELABELER
# Reads existing sankey_data.json (correct topology) and replaces name-based
# family labels with geometric family IDs from L5. Topology unchanged.
# ════════════════════════════════��══════════════════════════════════════════════

def relabel_sankey(sankey_path: str, prefix_to_fid: Dict[str, int],
                   families: List[FamilyCluster]) -> dict:
    """
    Relabels sankey endpoint nodes with geometric family IDs.
    All counts, links, and tree topology preserved exactly.
    Only the 'monster' and 'geometric_family_id' fields are updated.
    """
    with open(sankey_path) as f:
        sankey = json.load(f)

    remapped = 0
    for node in sankey.get('nodes', []):
        prefix = node.get('prefix', '')
        if prefix and not prefix.startswith('MONSTER:'):
            if prefix in prefix_to_fid:
                node['geometric_family_id'] = prefix_to_fid[prefix]
                remapped += 1
            else:
                # Use nearest ancestor's family
                moves = prefix.split()
                for depth in range(len(moves) - 1, 0, -1):
                    anc = ' '.join(moves[:depth])
                    if anc in prefix_to_fid:
                        node['geometric_family_id'] = prefix_to_fid[anc]
                        node['geometric_family_inherited'] = True
                        remapped += 1
                        break

    print(f'  [L6] relabeled {remapped}/{len(sankey.get("nodes", []))} sankey nodes', flush=True)
    return sankey


# ═════════════════════════════════════════════════════════���═════════════════════
# L7 | TRANSITION BUILDER
# Game trajectories with family labels → P[depth][from_fam][to_fam] matrices
# ══════════════════════════════════════════════════════════════════════���════════

def build_transitions(trajectories: List[GameTrajectory], n_families: int,
                      max_depth: int, prefix_to_fid: Dict[str, int]) -> np.ndarray:
    """
    Returns transitions[max_depth][n_families][n_families] as float32 probability tensor.
    transitions[d][i][j] = P(game transitions to family j at depth d+1 | it was in family i at depth d)
    """
    counts = np.zeros((max_depth, n_families, n_families), dtype=np.float64)
    moves_looked = 0

    for traj in trajectories:
        moves = traj.prefix.split()
        for depth in range(min(len(moves) - 1, max_depth - 1)):
            p_from = ' '.join(moves[:depth])
            p_to   = ' '.join(moves[:depth + 1])
            fid_from = prefix_to_fid.get(p_from)
            fid_to   = prefix_to_fid.get(p_to)
            if fid_from is not None and fid_to is not None:
                counts[depth, fid_from, fid_to] += 1
                moves_looked += 1

    # Normalize rows to probabilities
    row_sums = counts.sum(axis=2, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    transitions = (counts / row_sums).astype(np.float32)
    print(f'  [L7] built {max_depth}×{n_families}×{n_families} transition tensor '
          f'from {moves_looked:,} transitions', flush=True)
    return transitions


# ═══════════════════════════════════════════════════════════════════════════════
# L8 | G MATRIX BUILDER
# opening_nodes × n_families affinity matrix (replaces name-based G)
# Affinity = cosine similarity between node's trajectory feature and family centroid.
# ════════════════════════════════════════════════════════════════════��══════════

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def build_g_matrix(sankey: dict, agg: Dict[str, dict],
                   families: List[FamilyCluster],
                   prefix_to_fid: Dict[str, int]) -> Tuple[np.ndarray, List[str]]:
    """
    Builds G matrix [n_opening_nodes × n_families] float32.
    Each row is a soft distribution over families derived from geometric affinity.
    Returns (G, node_prefix_list) — node_prefix_list[i] = which prefix is row i.
    """
    nodes = [n for n in sankey.get('nodes', [])
             if not n.get('prefix', '').startswith('MONSTER:')]
    node_prefixes = [n.get('prefix', '') for n in nodes]
    n_nodes    = len(node_prefixes)
    n_families = len(families)
    G = np.zeros((n_nodes, n_families), dtype=np.float32)

    fam_centroids = np.stack([f.centroid_feature for f in families])  # (n_fam, feat_dim)

    for i, prefix in enumerate(node_prefixes):
        if prefix in agg:
            node_feat = agg[prefix]['mean'].flatten()
        else:
            # Nearest ancestor fallback (fixes wildcard escape gradient bug)
            moves = prefix.split()
            node_feat = None
            for depth in range(len(moves) - 1, -1, -1):
                anc = ' '.join(moves[:depth])
                if anc in agg:
                    node_feat = agg[anc]['mean'].flatten()
                    break
            if node_feat is None:
                node_feat = fam_centroids.mean(axis=0)  # uniform fallback

        sims = np.array([_cosine_sim(node_feat, fam_centroids[j]) for j in range(n_families)])
        # Softmax to get soft distribution
        sims = sims - sims.max()
        exp_s = np.exp(sims)
        G[i] = (exp_s / exp_s.sum()).astype(np.float32)

    print(f'  [L8] built G matrix {n_nodes}×{n_families} '
          f'(wildcard fallback fixed via ancestry)', flush=True)
    return G, node_prefixes


# ═════════════════════════════════════════════════���═════════════════════════════
# L9 | PACKAGE BUILDER
# Assembles versioned dojo-loadable artifact.
# JSON: human-readable config + families + transitions + prefix_to_family
# BIN: G matrix as raw float32 (fast CUDA load) + transition tensor
# ═══════════════════════════════════════════════════════════════════════��═══════

def build_package(families: List[FamilyCluster], transitions: np.ndarray,
                  G: np.ndarray, node_prefixes: List[str],
                  relabeled_sankey: dict, prefix_to_fid: Dict[str, int],
                  wildcards_path: str, config: dict, input_hash: str,
                  output_dir: str, version: int) -> str:
    """Writes opening_package_vN.json and opening_package_vN.bin. Returns output path stem."""
    out_stem = Path(output_dir) / f"{config['package_prefix']}_v{version}"
    n_fam    = config['n_families']
    max_d    = config['max_depth']

    # Wildcard seeds: inject with geometric family assignment (not name-based)
    wildcards = []
    if Path(wildcards_path).exists():
        with open(wildcards_path) as f:
            for line in f:
                wc = json.loads(line.strip())
                prefix = wc.get('prefix', '')
                wc['geometric_family_id'] = prefix_to_fid.get(prefix)
                if wc['geometric_family_id'] is None:
                    # Ancestry fallback — fixes the wildcard escape gradient bug
                    moves = prefix.split()
                    for depth in range(len(moves) - 1, -1, -1):
                        anc = ' '.join(moves[:depth])
                        if anc in prefix_to_fid:
                            wc['geometric_family_id'] = prefix_to_fid[anc]
                            wc['geometric_family_inherited_from'] = anc
                            break
                wildcards.append(wc)

    pkg = {
        'version':        version,
        'frostmatrix_version': FROSTMATRIX_VERSION,
        'rendered_at':    time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'input_hash':     input_hash,
        'params': {
            'n_families': n_fam,
            'max_depth':  max_d,
            'min_games':  config['min_games'],
            'encode_fn':  config.get('_encode_fn_name', 'active_fold'),
        },
        'families': [
            {
                'id':                  f.id,
                'game_count':          f.game_count,
                'example_prefixes':    f.example_prefixes,
                'centroid_trajectory':       f.centroid_trajectory,
                'delta_centroid_trajectory': f.delta_centroid_trajectory,
                'discriminant_channels':     f.discriminant_channels,
                'compression_profile':       f.compression_profile,
                'outlier_profile':           f.outlier_profile,
                'mean_fold_shape':           f.mean_fold_shape,
                'variance_feature':          f.variance_feature.tolist(),
            }
            for f in families
        ],
        'prefix_to_family': prefix_to_fid,
        'wildcards':        wildcards,
        'g_matrix_meta': {
            'shape':         list(G.shape),
            'dtype':         'float32',
            'node_prefixes': node_prefixes,
            'bin_offset':    0,  # G is first tensor in .bin file
        },
        'transitions_meta': {
            'shape':      list(transitions.shape),  # [max_depth, n_fam, n_fam]
            'dtype':      'float32',
            'bin_offset': G.nbytes,
        },
        'sankey':           relabeled_sankey,
    }

    # Write JSON
    json_path = str(out_stem) + '.json'
    with open(json_path, 'w') as f:
        json.dump(pkg, f, indent=2)
    print(f'  [L9] wrote {json_path}  ({Path(json_path).stat().st_size // 1024}KB)', flush=True)

    # Write binary pack: G (float32) + transitions (float32)
    bin_path = str(out_stem) + '.bin'
    with open(bin_path, 'wb') as f:
        f.write(G.astype(np.float32).tobytes())
        f.write(transitions.astype(np.float32).tobytes())
    print(f'  [L9] wrote {bin_path}  ({Path(bin_path).stat().st_size // 1024}KB)', flush=True)

    return str(out_stem)


# ══════════��══════════════════════════════════════════════════════════════════��═
# PIPELINE ORCHESTRATOR
# ═════════════���══════════════════════════════════════��══════════════════════════

def run(config: Dict[str, Any]):
    t_start = time.time()
    cfg = {**DEFAULT_CONFIG, **config}

    # Resolve pluggable interfaces to defaults if not provided
    encode_fn  = cfg['encode_fn']  or composite_encode
    reader_fn  = cfg['corpus_reader_fn'] or (lambda path: pgn_corpus_reader(path, cfg['max_depth']))
    cfg['_encode_fn_name'] = encode_fn.__name__

    print(f'\n{"═"*70}')
    print(f'  FROSTMATRIX v{FROSTMATRIX_VERSION}')
    print(f'  n_families={cfg["n_families"]}  max_depth={cfg["max_depth"]}  '
          f'min_games={cfg["min_games"]}')
    print(f'  encoder: {cfg["_encode_fn_name"]}')
    print(f'{"═"*70}\n', flush=True)

    # ── L1-L3: Stream corpus, build trajectories, aggregate ──────────────────
    print('L1→L3  Streaming corpus + encoding positions + building trajectories...', flush=True)
    # Detect feature dimension from a test encode so aggregator allocates correctly
    _test_enc = encode_fn([0]*64)
    _fdims    = len(_test_enc.feature_vec())
    print(f'  feature_dims={_fdims} ({_fdims - 4} extended channels from {len(GEOMETRY_ENCODERS)} encoders)',
          flush=True)
    aggregator   = PrefixAggregator(cfg['max_depth'], feature_dims=_fdims)
    trajectories = []   # kept for L7 transition building
    game_count   = 0
    t_l3 = time.time()

    for record in reader_fn(cfg['pgn_path']):
        traj = build_trajectory(record, encode_fn, cfg['max_depth'])
        aggregator.add(traj)
        trajectories.append(traj)
        game_count += 1
        if game_count % cfg['log_every'] == 0:
            elapsed = time.time() - t_l3
            rate = game_count / elapsed
            print(f'  {game_count:>7,} games  {rate:,.0f} games/s  '
                  f'prefixes_seen={len(aggregator._game_counts):,}', flush=True)

    print(f'  → {game_count:,} games processed in {time.time()-t_l3:.0f}s', flush=True)

    # ── L4: Aggregate per prefix ─────────────────────��────────────────────────
    print('\nL4  Aggregating trajectories per prefix...', flush=True)
    agg = aggregator.get(min_games=cfg['min_games'])
    print(f'  → {len(agg):,} prefixes with ≥{cfg["min_games"]} games', flush=True)

    # ── L5: Cluster into geometric families ──────────────────────────────────
    print('\nL5  Clustering into geometric families...', flush=True)
    families, prefix_to_fid = cluster_families(
        agg, cfg['n_families'], cfg['max_depth'],
        cfg['kmeans_iters'], cfg['kmeans_runs'],
    )
    for f in sorted(families, key=lambda x: -x.game_count):
        # Show delta (deviation from ply-mean) — raw trajectory is degenerate ~(3.5,3.4) for all families
        traj_str = ' → '.join(f'({c[0]:+.2f},{c[1]:+.2f})' for c in f.delta_centroid_trajectory[:4])
        disc_ch = f.discriminant_channels[:3] if f.discriminant_channels else []
        print(f'  family {f.id:>2}  games={f.game_count:>7,}  Δ(r,f): {traj_str}  top_ch={disc_ch}', flush=True)

    # ── L6: Relabel sankey ────────────────────────────���───────────────────────
    print('\nL6  Relabeling sankey topology...', flush=True)
    relabeled_sankey = relabel_sankey(cfg['sankey_path'], prefix_to_fid, families)

    # ── L7: Transition surfaces ───────────────────────────────────────────────
    print('\nL7  Building transition surfaces...', flush=True)
    transitions = build_transitions(trajectories, cfg['n_families'],
                                    cfg['max_depth'], prefix_to_fid)

    # ── L8: G matrix ─────────────────────────���────────────────────────────────
    print('\nL8  Building G matrix (geometric affinity)...', flush=True)
    G, node_prefixes = build_g_matrix(relabeled_sankey, agg, families, prefix_to_fid)

    # ── L9: Package ────────────────────────────────────────────────────��──────
    print('\nL9  Building dojo package...', flush=True)
    input_hash = hashlib.sha256(
        (cfg['pgn_path'] + str(cfg['n_families']) + str(cfg['max_depth'])
         + cfg['_encode_fn_name']).encode()
    ).hexdigest()[:16]

    # Auto-increment version based on existing packages
    out_dir = cfg['output_dir']
    version = 1
    while (Path(out_dir) / f"{cfg['package_prefix']}_v{version}.json").exists():
        version += 1

    out_stem = build_package(
        families, transitions, G, node_prefixes, relabeled_sankey,
        prefix_to_fid, cfg['wildcard_path'], cfg, input_hash, out_dir, version,
    )

    elapsed = time.time() - t_start
    print(f'\n{"═"*70}')
    print(f'  FROSTMATRIX RENDER COMPLETE  {elapsed/60:.1f} min')
    print(f'  Package: {out_stem}.json')
    print(f'  Families: {len(families)}  Prefixes: {len(prefix_to_fid):,}  '
          f'G shape: {G.shape}')
    print(f'{"═"*70}\n', flush=True)
    return out_stem


# ════════════════���════════════════════════���═════════════════════════════════════
# CLI
# ═══════════════════════════════════════��═══════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='FrostMatrix — render dojo opening geometry package')
    parser.add_argument('--config',      help='JSON config file (overrides all other flags)')
    parser.add_argument('--pgn',         help='PGN corpus path')
    parser.add_argument('--n-families',  type=int, help='Number of geometric families')
    parser.add_argument('--max-depth',   type=int, help='Plies to trace per game')
    parser.add_argument('--min-games',   type=int, help='Min games per prefix for L5')
    parser.add_argument('--output-dir',  help='Output directory')
    parser.add_argument('--kmeans-runs', type=int, help='K-means restarts')
    args = parser.parse_args()

    cfg = {}
    if args.config:
        with open(args.config) as f:
            cfg = json.load(f)
    if args.pgn:         cfg['pgn_path']     = args.pgn
    if args.n_families:  cfg['n_families']   = args.n_families
    if args.max_depth:   cfg['max_depth']    = args.max_depth
    if args.min_games:   cfg['min_games']    = args.min_games
    if args.output_dir:  cfg['output_dir']   = args.output_dir
    if args.kmeans_runs: cfg['kmeans_runs']  = args.kmeans_runs

    run(cfg)

if __name__ == '__main__':
    main()
