"""
train_v3_frostmatrix.py — PyTorch training wrapper for FrostMatrix Graph Transformer v3

Mirrors the architecture in transformer_v3_frostmatrix.cu exactly:
  - SEQ_LEN_V3 = 129 (65 family axis nodes + 64 board squares)
  - D_MODEL = 128, N_HEADS = 4, N_LAYERS = 4, D_FFN = 341
  - N_GEO_CHANNELS = 55 (4 base + 51 from 17 geometry encoders)
  - N_FAMILIES = 33, N_UNKNOWNS = 32, N_FAM_NODES = 65
  - Highway attention (3 directions), board→family cross-attention, SwiGLU FFN
  - Policy head: linear(D_MODEL, VOCAB_SIZE=513)
  - Value head: linear(D_MODEL, 1) + tanh → [-1, 1]

Training data schema (training_v2.jsonl):
  {"fen": "...", "family_id": 0, "move": "g1f3", "outcome": 1, "tokenizer": "v2"}

Move vocabulary: UCI strings are mapped to indices 0..511 by a vocabulary built
from the training corpus (first pass). Index 512 is reserved for unknown moves.
VOCAB_SIZE = 513 = 512 known moves + 1 unknown.

Weight export: raw float32 binary files named to match TransformerWeights struct
fields in the .cu file, plus a JSON manifest.

Usage:
  python train_v3_frostmatrix.py [--data PATH] [--epochs N] [--checkpoint PATH]
                                  [--export-only]
"""

import argparse
import json
import math
import os
import struct
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, IterableDataset

# ─── Architecture constants (must match transformer_v3_frostmatrix.cu) ────────

N_FAMILIES     = 33
N_UNKNOWNS     = 32       # one between each consecutive family pair
N_FAM_NODES    = 65       # 33 + 32 — the full Y-axis
N_BOARD_SQ     = 64
SEQ_LEN_V3     = 129      # N_FAM_NODES + N_BOARD_SQ

D_MODEL        = 128
D_HEAD         = 32       # D_MODEL / N_HEADS
N_HEADS        = 4
N_LAYERS       = 4
D_FFN          = 341      # SwiGLU reduced FFN
VOCAB_SIZE     = 4096     # from_sq*64+to_sq — deterministic, covers all legal moves

N_HW_DIRS      = 3        # lateral, vertical, diagonal
D_HW_HEAD      = 32       # highway head dim per direction (3×32=96, projected→128)
HW_SMOOTH_SIGMA = 2.0

N_GEO_CHANNELS = 55       # 4 base + 51 extended (17 geometry encoders G1-G17)

BATCH_SIZE     = 256

# ─── Default paths ────────────────────────────────────────────────────────────

DEFAULT_DATA_PATH = "/srv/models-hdd/chess-games/training_v2.jsonl"
DEFAULT_CKPT_DIR  = "/srv/models-hdd/chess-games/training_runs/v3_frostmatrix"

# ─── Move encoding — deterministic from_sq*64+to_sq ──────────────────────────
# Matches POLICY_SIZE=4096 in transformer_v3_frostmatrix.cu.
# No vocab file, no corpus scan. Every UCI move → unique index in [0,4095].

_UCI_FILE = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7}

def uci_to_policy_idx(uci: str) -> int:
    """UCI move → policy index in [0, 4095] = from_sq*64 + to_sq.
    Board layout matches fen_to_board: index 0 = a8, index 63 = h1."""
    if len(uci) < 4:
        return 0
    try:
        from_sq = (8 - int(uci[1])) * 8 + _UCI_FILE[uci[0]]
        to_sq   = (8 - int(uci[3])) * 8 + _UCI_FILE[uci[2]]
        return from_sq * 64 + to_sq
    except (KeyError, ValueError, IndexError):
        return 0


# ─── FEN → board tensor ───────────────────────────────────────────────────────

# Piece type map: lowercase FEN char → integer 1-6 (white), 7-12 (black)
# 0 = empty square
_FEN_PIECE = {
    'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
    'p': 7, 'n': 8, 'b': 9, 'r': 10, 'q': 11, 'k': 12,
}

def fen_to_board(fen: str) -> List[int]:
    """
    Parse the piece placement part of a FEN string into a flat list of 64 ints.
    Index 0 = a8 (rank 8, file a), index 63 = h1.
    Values: 0=empty, 1-6=white PNBRQK, 7-12=black pnbrqk.
    """
    placement = fen.split()[0]
    board = []
    for char in placement:
        if char == '/':
            continue
        elif char.isdigit():
            board.extend([0] * int(char))
        else:
            board.append(_FEN_PIECE.get(char, 0))
    # Pad to 64 if malformed
    board = board[:64]
    while len(board) < 64:
        board.append(0)
    return board


def fen_to_side_to_move(fen: str) -> int:
    """Return 1 if white to move, -1 if black."""
    parts = fen.split()
    if len(parts) >= 2:
        return 1 if parts[1] == 'w' else -1
    return 1


def fen_to_castling(fen: str) -> List[float]:
    """Return 4-bit castling rights as floats: [KQ kq] white/black king/queen."""
    parts = fen.split()
    rights = parts[2] if len(parts) >= 3 else "-"
    return [
        1.0 if 'K' in rights else 0.0,
        1.0 if 'Q' in rights else 0.0,
        1.0 if 'k' in rights else 0.0,
        1.0 if 'q' in rights else 0.0,
    ]


def build_geo_vector(board: List[int], side: int, castling: List[float]) -> List[float]:
    """
    Build the N_GEO_CHANNELS=55 geometry vector for a position.

    The CUDA comment specifies:
      4 base + 51 extended (17 geometry encoders G1-G17):
        G1-G4:  piece presence per type (4 channels = 4 encoders)
        G5:     side to move
        G6-G10: material count per piece type
        G11-G13: fold variants (3 channels)
        G14:    pawn chain connectivity
        G15:    pin/xray potential
        G16:    open files
        G17:    knight outposts
      Plus 4 castling channels = 55 total

    This is a board-level feature vector (not per-square).
    In the CUDA kernel geo_project_kernel it is applied per-square via the same
    projection — so we replicate the board-level geo vector to all 64 squares.
    The geo_vecs input is [batch × N_BOARD_SQ × N_GEO_CHANNELS].

    We compute a position-level 55-dim feature and broadcast to all squares.
    Per-square specialization would require a more detailed encoder; this matches
    what the CUDA forward pass actually does (single geo_vecs tensor, no per-square
    distinction in the projection kernel).
    """
    feats: List[float] = []

    # ── G1-G4: white piece type presences (4 base channels) ─────────────────
    # G1: any white piece, G2: any black piece, G3: pawn count ratio, G4: piece balance
    white_pieces = [p for p in board if 1 <= p <= 6]
    black_pieces = [p for p in board if 7 <= p <= 12]
    feats.append(min(len(white_pieces) / 16.0, 1.0))    # G1
    feats.append(min(len(black_pieces) / 16.0, 1.0))    # G2
    w_pawns = board.count(1)
    b_pawns = board.count(7)
    feats.append(w_pawns / 8.0)                          # G3
    feats.append(b_pawns / 8.0)                          # G4

    # ── G5: side to move ─────────────────────────────────────────────────────
    feats.append(1.0 if side == 1 else 0.0)

    # ── G6-G10: material count per piece type (white − black, normalized) ───
    for pt in range(1, 6):  # P N B R Q
        wc = board.count(pt)
        bc = board.count(pt + 6)
        feats.append((wc - bc + 8) / 16.0)   # normalized to [0,1]

    # ── G11-G13: fold variants (3 channels) ──────────────────────────────────
    # Fold: reflect board horizontally, vertically, diagonal (symmetry detectors)
    # G11: horizontal symmetry score
    h_sym = sum(1 for i in range(32) if board[i] != 0 and board[i] == board[i + 32]) / 32.0
    feats.append(h_sym)

    # G12: vertical (file) symmetry
    v_sym = sum(
        1 for rank in range(8) for f in range(4)
        if board[rank*8+f] != 0 and board[rank*8+f] == board[rank*8+(7-f)]
    ) / 32.0
    feats.append(v_sym)

    # G13: diagonal symmetry (transpose)
    d_sym = sum(
        1 for r in range(8) for f in range(8)
        if board[r*8+f] != 0 and board[r*8+f] == board[f*8+r]
    ) / 64.0
    feats.append(d_sym)

    # ── G14: pawn chain connectivity ─────────────────────────────────────────
    # Count white pawns that protect another white pawn diagonally
    pawn_chain = 0
    for sq in range(64):
        if board[sq] == 1:  # white pawn
            r, f = divmod(sq, 8)
            # Pawn on r protects r-1 rank diagonally (attacks forward)
            if r > 0:
                if f > 0 and board[(r-1)*8+(f-1)] == 1:
                    pawn_chain += 1
                if f < 7 and board[(r-1)*8+(f+1)] == 1:
                    pawn_chain += 1
    feats.append(min(pawn_chain / 8.0, 1.0))

    # ── G15: pin/xray potential (sliding piece vs king alignment) ─────────────
    # Approximate: count sliding pieces (B,R,Q) on same rank/file/diagonal as kings
    w_king_sq = next((i for i, p in enumerate(board) if p == 6), -1)
    b_king_sq = next((i for i, p in enumerate(board) if p == 12), -1)
    pin_score = 0.0
    for sq, piece in enumerate(board):
        if piece in (4, 5) or piece in (10, 11):  # R or Q (either side)
            if w_king_sq >= 0:
                kr, kf = divmod(w_king_sq, 8)
                sr, sf = divmod(sq, 8)
                if kr == sr or kf == sf:
                    pin_score += 0.1
            if b_king_sq >= 0:
                kr, kf = divmod(b_king_sq, 8)
                sr, sf = divmod(sq, 8)
                if kr == sr or kf == sf:
                    pin_score += 0.1
        if piece in (3, 5) or piece in (9, 11):  # B or Q (either side)
            if w_king_sq >= 0:
                kr, kf = divmod(w_king_sq, 8)
                sr, sf = divmod(sq, 8)
                if abs(kr - sr) == abs(kf - sf):
                    pin_score += 0.1
            if b_king_sq >= 0:
                kr, kf = divmod(b_king_sq, 8)
                sr, sf = divmod(sq, 8)
                if abs(kr - sr) == abs(kf - sf):
                    pin_score += 0.1
    feats.append(min(pin_score, 1.0))

    # ── G16: open files (no pawns of either color) ───────────────────────────
    open_files = 0
    for f in range(8):
        col_pieces = [board[r*8+f] for r in range(8)]
        if 1 not in col_pieces and 7 not in col_pieces:
            open_files += 1
    feats.append(open_files / 8.0)

    # ── G17: knight outposts (knight in opponent's half, no pawn attacks) ────
    outpost_score = 0.0
    for sq, piece in enumerate(board):
        r, f = divmod(sq, 8)
        if piece == 2 and r <= 3:  # white knight in opponent's half (rows 0-3 = ranks 5-8)
            # Check if black pawns can attack this square
            attacked = False
            if r < 7 and f > 0 and board[(r+1)*8+(f-1)] == 7:
                attacked = True
            if r < 7 and f < 7 and board[(r+1)*8+(f+1)] == 7:
                attacked = True
            if not attacked:
                outpost_score += 0.25
        if piece == 8 and r >= 4:  # black knight in opponent's half
            attacked = False
            if r > 0 and f > 0 and board[(r-1)*8+(f-1)] == 1:
                attacked = True
            if r > 0 and f < 7 and board[(r-1)*8+(f+1)] == 1:
                attacked = True
            if not attacked:
                outpost_score += 0.25
    feats.append(min(outpost_score, 1.0))

    # ── Castling: 4 channels ──────────────────────────────────────────────────
    feats.extend(castling)

    # Pad to exactly N_GEO_CHANNELS=55
    # Current count: 4 + 1 + 5 + 3 + 1 + 1 + 1 + 1 + 4 = 21
    # We need 55 total, so add 34 more feature channels using extended piece stats
    current = len(feats)
    needed = N_GEO_CHANNELS - current

    # Extended features: per-square piece-type one-hot stats
    piece_counts = [0] * 13
    for p in board:
        piece_counts[p] += 1

    # Normalize and pack more piece-level stats to fill to 55
    extended = []
    # Piece presence normalized (13 types × various stats)
    for pt in range(1, 13):
        extended.append(piece_counts[pt] / 16.0)
    # Center control: pieces near center (d4,d5,e4,e5 = squares 27,28,35,36)
    center_sqs = [27, 28, 35, 36]
    extended_center = [board[sq] for sq in center_sqs]
    for pt in range(1, 7):  # white pieces
        extended.append(sum(1 for p in extended_center if p == pt) / 4.0)
    # Mobility proxy: number of legal-looking destinations per sliding piece
    ext_mobility = 0.0
    for sq, piece in enumerate(board):
        if piece in (4, 5, 10, 11):  # rooks and queens
            r, f = divmod(sq, 8)
            # Count open squares in cardinal directions until blocked
            for dr, df in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nf = r+dr, f+df
                while 0 <= nr < 8 and 0 <= nf < 8:
                    if board[nr*8+nf] != 0:
                        break
                    ext_mobility += 0.01
                    nr += dr; nf += df
    extended.append(min(ext_mobility, 1.0))

    extended = extended[:needed]
    while len(extended) < needed:
        extended.append(0.0)

    feats.extend(extended)

    assert len(feats) == N_GEO_CHANNELS, f"geo vector length {len(feats)} != {N_GEO_CHANNELS}"
    return feats


# ─── Dataset ─────────────────────────────────────────────────────────────────

class FrostMatrixDataset(IterableDataset):
    """
    Streaming dataset over training_v2.jsonl.

    Each line:
      {"fen": str, "family_id": int, "move": str, "outcome": int, "tokenizer": str}

    Yields tuples of tensors:
      board_tokens: LongTensor [64]       — piece type at each square (0-12)
      family_id:    LongTensor []          — opening family (0..N_FAMILIES-1)
      geo_vecs:     FloatTensor [64, 55]  — geometry features per square
      move_idx:     LongTensor []          — policy target in [0, VOCAB_SIZE-1]
      outcome:      FloatTensor []         — value target in {-1.0, 0.0, 1.0}
    """

    def __init__(self, data_path: str,
                 start_byte: int = 0, max_samples: Optional[int] = None):
        self.data_path = data_path
        self.start_byte = start_byte
        self.max_samples = max_samples

    def __iter__(self):
        n = 0
        with open(self.data_path, "r") as f:
            if self.start_byte > 0:
                f.seek(self.start_byte)
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                fen       = obj.get("fen", "")
                family_id = int(obj.get("family_id", 0))
                move_uci  = obj.get("move", "")
                outcome   = float(obj.get("outcome", 0))

                # Clamp family_id to valid range
                family_id = max(0, min(family_id, N_FAMILIES - 1))

                # Board parsing
                board = fen_to_board(fen)
                side  = fen_to_side_to_move(fen)
                castling = fen_to_castling(fen)

                # Geo vector (replicated to all 64 squares)
                geo_base = build_geo_vector(board, side, castling)
                geo_vecs = [geo_base] * N_BOARD_SQ   # [64 × 55]

                # Move → deterministic policy index (from_sq*64+to_sq)
                move_idx = uci_to_policy_idx(move_uci)

                yield (
                    torch.tensor(board, dtype=torch.long),               # [64]
                    torch.tensor(family_id, dtype=torch.long),           # scalar
                    torch.tensor(geo_vecs, dtype=torch.float32),         # [64, 55]
                    torch.tensor(move_idx, dtype=torch.long),            # scalar
                    torch.tensor(outcome, dtype=torch.float32),          # scalar
                )

                n += 1
                if self.max_samples is not None and n >= self.max_samples:
                    return


# ─── Model — FrostMatrix Graph Transformer v3 ────────────────────────────────

class SwiGLUFFN(nn.Module):
    """
    SwiGLU FFN with smeargate approximation.
    up and gate project D_MODEL → D_FFN; down projects D_FFN → D_MODEL.
    Smeargate: geometric mean with neighbor in D_FFN dim (approximated as
    element-wise SiLU during training — exact warp-shuffle smearing is a CUDA
    inference detail not reproducible in PyTorch without custom kernels).
    """

    def __init__(self):
        super().__init__()
        self.up   = nn.Linear(D_MODEL, D_FFN, bias=False)
        self.gate = nn.Linear(D_MODEL, D_FFN, bias=False)
        self.down = nn.Linear(D_FFN, D_MODEL, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., D_MODEL]
        h    = self.up(x)
        g    = self.gate(x)
        silu = g * torch.sigmoid(g)   # SiLU activation on gate
        h    = h * silu               # gated linear unit
        # Smeargate approximation: geometric mean with shifted neighbor
        # Use roll-by-1 along last dim as proxy for warp-neighbor
        h_nb = torch.roll(h, shifts=-1, dims=-1)
        h_nb[..., -1] = h[..., -1]   # boundary: self
        sign = torch.sign(h)
        sign[sign == 0] = 1.0
        h = torch.where(
            h * h_nb > 0,
            torch.sqrt(torch.abs(h * h_nb) + 1e-8) * sign,
            torch.zeros_like(h)
        )
        return self.down(h)


class HighwayAttention(nn.Module):
    """
    Tri-directional highway attention over the family axis (N_FAM_NODES=65 nodes).

    Three orthogonal directions (lateral, vertical, diagonal) each use independent
    Q, K, V projections of dimension D_HW_HEAD=32. Results are concatenated
    (3×32=96) and projected back to D_MODEL=128 via hw_Wo.

    The smooth_weights (Gaussian kernel over family distances) bias the lateral
    (dir=0) attention scores via additive log-space bias, matching the CUDA kernel.
    """

    def __init__(self):
        super().__init__()
        self.hw_Wq = nn.ModuleList([nn.Linear(D_MODEL, D_HW_HEAD, bias=False) for _ in range(N_HW_DIRS)])
        self.hw_Wk = nn.ModuleList([nn.Linear(D_MODEL, D_HW_HEAD, bias=False) for _ in range(N_HW_DIRS)])
        self.hw_Wv = nn.ModuleList([nn.Linear(D_MODEL, D_HW_HEAD, bias=False) for _ in range(N_HW_DIRS)])
        self.hw_Wo = nn.Linear(N_HW_DIRS * D_HW_HEAD, D_MODEL, bias=False)

        # Gaussian smoothing weights for lateral highway: [N_FAMILIES × N_FAMILIES]
        # Initialized as uniform (Gaussian fallback from the CUDA code)
        smooth = self._build_smooth_weights()
        self.register_buffer("smooth_weights", smooth)

        # Pre-compute attention masks for each direction
        # [N_FAM_NODES × N_FAM_NODES] boolean allowed masks
        self.register_buffer("mask_lateral",  self._build_mask(0))
        self.register_buffer("mask_vertical", self._build_mask(1))
        self.register_buffer("mask_diagonal", self._build_mask(2))

    def _build_smooth_weights(self) -> torch.Tensor:
        """Build Gaussian smoothing weights matching CUDA build_smooth_weights (fallback)."""
        sigma2 = HW_SMOOTH_SIGMA ** 2
        w = torch.zeros(N_FAMILIES, N_FAMILIES)
        for i in range(N_FAMILIES):
            for j in range(N_FAMILIES):
                d = abs(i - j)
                wij = 1.0 if d == 0 else (math.exp(-d*d / (2*sigma2)) if d <= 3 else 0.0)
                w[i, j] = wij
            w[i] /= w[i].sum().clamp(min=1e-8)
        return w

    def _build_mask(self, direction: int) -> torch.Tensor:
        """Build direction-specific attention mask [N_FAM_NODES × N_FAM_NODES]."""
        mask = torch.zeros(N_FAM_NODES, N_FAM_NODES, dtype=torch.bool)
        for qi in range(N_FAM_NODES):
            qi_fam = (qi % 2 == 0)
            qi_id  = qi // 2
            for ki in range(N_FAM_NODES):
                ki_fam = (ki % 2 == 0)
                ki_id  = ki // 2
                allowed = False
                if direction == 0:
                    # LATERAL: family → family only
                    allowed = qi_fam and ki_fam
                elif direction == 1:
                    # VERTICAL: family ↔ adjacent unknowns
                    if qi_fam and not ki_fam:
                        allowed = (ki_id == qi_id - 1 or ki_id == qi_id) and 0 <= ki_id < N_UNKNOWNS
                    elif not qi_fam and ki_fam:
                        allowed = (ki_id == qi_id or ki_id == qi_id + 1) and ki_id < N_FAMILIES
                elif direction == 2:
                    # DIAGONAL: family → unknowns ±2 or ±3 families away
                    if qi_fam and not ki_fam:
                        skip = abs(ki_id - qi_id)
                        allowed = (skip == 2 or skip == 3) and 0 <= ki_id < N_UNKNOWNS
                mask[qi, ki] = allowed
        return mask

    def forward(self, fam_x: torch.Tensor) -> torch.Tensor:
        """
        fam_x: [B, N_FAM_NODES, D_MODEL]
        Returns: [B, N_FAM_NODES, D_MODEL] — highway-attended family axis
        """
        B = fam_x.size(0)
        scale = 1.0 / math.sqrt(D_HW_HEAD)

        masks = [self.mask_lateral, self.mask_vertical, self.mask_diagonal]
        dir_outputs = []

        for d in range(N_HW_DIRS):
            q = self.hw_Wq[d](fam_x)   # [B, N_FAM_NODES, D_HW_HEAD]
            k = self.hw_Wk[d](fam_x)
            v = self.hw_Wv[d](fam_x)

            # Scaled dot-product attention
            scores = torch.bmm(q, k.transpose(-2, -1)) * scale   # [B, F, F]

            # Apply direction mask: set disallowed positions to -inf
            mask = masks[d]   # [F, F]
            scores = scores.masked_fill(~mask.unsqueeze(0), float('-inf'))

            # Add lateral smoothing bias for direction 0 (family→family only)
            if d == 0:
                # smooth_weights: [N_FAMILIES, N_FAMILIES]
                # Expand to [N_FAM_NODES, N_FAM_NODES] — only family nodes (even indices)
                fam_indices = torch.arange(0, N_FAM_NODES, 2, device=fam_x.device)
                # Build a full [F, F] bias tensor initialized to 0
                smooth_bias = torch.zeros(N_FAM_NODES, N_FAM_NODES, device=fam_x.device)
                # Insert log(smooth_weights) at family-family positions
                log_sw = torch.log(self.smooth_weights + 1e-8)   # [N_FAM, N_FAM]
                # Scatter into the full matrix
                rows = fam_indices.unsqueeze(1).expand(N_FAMILIES, N_FAMILIES)
                cols = fam_indices.unsqueeze(0).expand(N_FAMILIES, N_FAMILIES)
                smooth_bias[rows, cols] = log_sw
                scores = scores + smooth_bias.unsqueeze(0)

            # Softmax (scores with -inf will produce 0 weight after softmax)
            # Handle rows that are all -inf (no valid attention targets)
            valid_mask = (scores != float('-inf')).any(dim=-1, keepdim=True)
            scores = torch.where(valid_mask, scores, torch.zeros_like(scores))
            attn = torch.softmax(scores, dim=-1)
            attn = attn * valid_mask.float()

            out = torch.bmm(attn, v)    # [B, F, D_HW_HEAD]
            dir_outputs.append(out)

        # Concatenate 3 directions: [B, F, 3*D_HW_HEAD] → project to [B, F, D_MODEL]
        combined = torch.cat(dir_outputs, dim=-1)   # [B, F, 96]
        return self.hw_Wo(combined)                  # [B, F, D_MODEL]


class BoardCrossAttention(nn.Module):
    """
    Board → family cross-attention.
    Board tokens (queries) attend over the full family axis (keys/values).
    This is how each board position "finds its family" dynamically.
    """

    def __init__(self):
        super().__init__()
        self.ca_Wq = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.ca_Wk = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.ca_Wv = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.ca_Wo = nn.Linear(D_MODEL, D_MODEL, bias=False)

    def forward(self, board_x: torch.Tensor, fam_x: torch.Tensor) -> torch.Tensor:
        """
        board_x: [B, N_BOARD_SQ, D_MODEL]
        fam_x:   [B, N_FAM_NODES, D_MODEL]
        Returns: [B, N_BOARD_SQ, D_MODEL]
        """
        scale = 1.0 / math.sqrt(D_MODEL)
        q = self.ca_Wq(board_x)    # [B, 64, D_MODEL]
        k = self.ca_Wk(fam_x)      # [B, 65, D_MODEL]
        v = self.ca_Wv(fam_x)      # [B, 65, D_MODEL]

        scores = torch.bmm(q, k.transpose(-2, -1)) * scale   # [B, 64, 65]
        attn   = torch.softmax(scores, dim=-1)                # [B, 64, 65]
        out    = torch.bmm(attn, v)                           # [B, 64, D_MODEL]
        return board_x + self.ca_Wo(out)


class TransformerLayer(nn.Module):
    """
    Standard pre-norm transformer layer applied over the full SEQ_LEN_V3=129 sequence.
    LayerNorm → Multi-head self-attention → residual →
    LayerNorm → SwiGLU+smeargate FFN → residual
    """

    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(D_MODEL)
        self.ln2 = nn.LayerNorm(D_MODEL)

        # Multi-head self-attention projections
        self.Wq = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.Wk = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.Wv = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.Wo = nn.Linear(D_MODEL, D_MODEL, bias=False)

        self.ffn = SwiGLUFFN()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, SEQ_LEN_V3, D_MODEL]"""
        B, S, D = x.shape

        # ── Self-attention ────────────────────────────────────────────────────
        residual = x
        x = self.ln1(x)

        # Project Q, K, V
        q = self.Wq(x)   # [B, S, D]
        k = self.Wk(x)
        v = self.Wv(x)

        # Reshape to multi-head: [B, H, S, D_HEAD]
        q = q.view(B, S, N_HEADS, D_HEAD).transpose(1, 2)
        k = k.view(B, S, N_HEADS, D_HEAD).transpose(1, 2)
        v = v.view(B, S, N_HEADS, D_HEAD).transpose(1, 2)

        scale  = 1.0 / math.sqrt(D_HEAD)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale   # [B, H, S, S]
        attn   = torch.softmax(scores, dim=-1)                   # [B, H, S, S]
        out    = torch.matmul(attn, v)                           # [B, H, S, D_HEAD]

        # Merge heads
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        x = residual + self.Wo(out)

        # ── FFN ───────────────────────────────────────────────────────────────
        residual = x
        x = self.ln2(x)
        x = residual + self.ffn(x)

        return x


class FrostMatrixV3(nn.Module):
    """
    FrostMatrix Graph Transformer v3 — PyTorch training mirror.

    Matches the weight structure in transformer_v3_frostmatrix.cu exactly.
    """

    def __init__(self):
        super().__init__()

        # ── Token embedding (board piece types + opening prefix IDs) ─────────
        self.token_embed = nn.Embedding(VOCAB_SIZE, D_MODEL)

        # ── Family axis ───────────────────────────────────────────────────────
        # family_embed: [N_FAMILIES, D_MODEL] — from FrostMatrix centroids
        # unknown_embed: [N_UNKNOWNS, D_MODEL] — learnable latent nodes
        self.family_embed  = nn.Embedding(N_FAMILIES, D_MODEL)
        self.unknown_embed = nn.Embedding(N_UNKNOWNS, D_MODEL)
        nn.init.normal_(self.unknown_embed.weight, std=0.1)   # small init as in CUDA

        # ── Geometry projection: geo channels → D_MODEL ───────────────────────
        # geo_proj_W: [N_GEO_CHANNELS, D_MODEL], geo_proj_b: [D_MODEL]
        self.geo_proj = nn.Linear(N_GEO_CHANNELS, D_MODEL, bias=True)

        # ── Highway attention (tri-directional over family axis) ───────────────
        self.highway = HighwayAttention()

        # ── Board → Family cross-attention ────────────────────────────────────
        self.cross_attn = BoardCrossAttention()

        # ── Standard transformer layers (N_LAYERS) ────────────────────────────
        self.layers = nn.ModuleList([TransformerLayer() for _ in range(N_LAYERS)])

        # ── Output heads ──────────────────────────────────────────────────────
        self.policy_W = nn.Linear(D_MODEL, VOCAB_SIZE, bias=False)
        self.value_W  = nn.Linear(D_MODEL, 1, bias=True)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights matching CUDA alloc_weights scales."""
        s_e  = math.sqrt(2.0 / D_MODEL)
        s_a  = math.sqrt(2.0 / (D_MODEL + D_HEAD))
        s_hw = math.sqrt(2.0 / (D_MODEL + D_HW_HEAD))
        s_f  = math.sqrt(2.0 / (D_MODEL + D_FFN))

        # token / family embeddings
        nn.init.normal_(self.token_embed.weight, std=s_e)
        nn.init.normal_(self.family_embed.weight, std=s_e)
        nn.init.normal_(self.geo_proj.weight, std=s_e)
        nn.init.zeros_(self.geo_proj.bias)

        # Highway projections
        for d in range(N_HW_DIRS):
            nn.init.normal_(self.highway.hw_Wq[d].weight, std=s_hw)
            nn.init.normal_(self.highway.hw_Wk[d].weight, std=s_hw)
            nn.init.normal_(self.highway.hw_Wv[d].weight, std=s_hw)
        nn.init.normal_(self.highway.hw_Wo.weight, std=s_hw)

        # Cross-attention
        for w in [self.cross_attn.ca_Wq, self.cross_attn.ca_Wk,
                  self.cross_attn.ca_Wv, self.cross_attn.ca_Wo]:
            nn.init.normal_(w.weight, std=s_a)

        # Transformer layers
        for layer in self.layers:
            for w in [layer.Wq, layer.Wk, layer.Wv, layer.Wo]:
                nn.init.normal_(w.weight, std=s_a)
            nn.init.normal_(layer.ffn.up.weight, std=s_f)
            nn.init.normal_(layer.ffn.gate.weight, std=s_f)
            nn.init.normal_(layer.ffn.down.weight, std=s_f)
            nn.init.ones_(layer.ln1.weight)
            nn.init.zeros_(layer.ln1.bias)
            nn.init.ones_(layer.ln2.weight)
            nn.init.zeros_(layer.ln2.bias)

        # Output heads
        nn.init.normal_(self.policy_W.weight, std=s_e)
        nn.init.normal_(self.value_W.weight, std=s_e)
        nn.init.zeros_(self.value_W.bias)

    def build_family_axis(self, B: int, device: torch.device) -> torch.Tensor:
        """
        Build the Y-axis sequence of N_FAM_NODES=65 tokens.
        Layout: [fam_0, unk_01, fam_1, unk_12, ..., unk_31-32, fam_32]
        Matches build_family_axis_kernel in the CUDA file.
        Returns: [B, N_FAM_NODES, D_MODEL]
        """
        fam_ids  = torch.arange(N_FAMILIES, device=device)   # [33]
        unk_ids  = torch.arange(N_UNKNOWNS, device=device)   # [32]
        fam_emb  = self.family_embed(fam_ids)                 # [33, D_MODEL]
        unk_emb  = self.unknown_embed(unk_ids)                # [32, D_MODEL]

        # Interleave: fam[0], unk[0], fam[1], unk[1], ..., unk[31], fam[32]
        tokens = []
        for i in range(N_FAMILIES):
            tokens.append(fam_emb[i])
            if i < N_UNKNOWNS:
                tokens.append(unk_emb[i])

        axis = torch.stack(tokens, dim=0)   # [65, D_MODEL]
        return axis.unsqueeze(0).expand(B, -1, -1)   # [B, 65, D_MODEL]

    def forward(
        self,
        board_tokens: torch.Tensor,   # [B, 64] — piece type indices
        family_id:    torch.Tensor,   # [B] — opening family index (not used as direct input)
        geo_vecs:     torch.Tensor,   # [B, 64, N_GEO_CHANNELS]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          policy_logits: [B, VOCAB_SIZE]
          value:         [B] — tanh output in [-1, 1]
        """
        B      = board_tokens.size(0)
        device = board_tokens.device

        # ── Step 1: Build family axis ─────────────────────────────────────────
        fam_x = self.build_family_axis(B, device)   # [B, 65, D]

        # ── Step 2: Embed board tokens from piece type IDs ───────────────────
        # board_tokens: [B, 64] piece indices in [0,12]
        # Map through token_embed (first 13 slots = piece types)
        board_x = self.token_embed(board_tokens)    # [B, 64, D]

        # ── Step 3: Project FrostMatrix geometry into board tokens ─────────────
        # geo_proj: [B, 64, 55] → [B, 64, D]; add tanh-gated projection
        geo_proj = torch.tanh(self.geo_proj(geo_vecs))   # [B, 64, D]
        board_x = board_x + geo_proj

        # ── Step 4: Highway attention over family axis (3 directions) ─────────
        hw_delta = self.highway(fam_x)              # [B, 65, D]
        fam_x = fam_x + hw_delta                    # residual add

        # ── Step 5: Board → Family cross-attention ────────────────────────────
        board_x = self.cross_attn(board_x, fam_x)  # [B, 64, D]

        # ── Step 6: Concatenate family axis + board into full sequence ─────────
        x = torch.cat([fam_x, board_x], dim=1)     # [B, 129, D]

        # ── Step 7: N_LAYERS standard transformer layers ──────────────────────
        for layer in self.layers:
            x = layer(x)

        # ── Step 8: Policy + value heads ──────────────────────────────────────
        # Pool first board token (position N_FAM_NODES in full sequence)
        board_repr = x[:, N_FAM_NODES, :]           # [B, D] — first board token

        policy_logits = self.policy_W(board_repr)    # [B, VOCAB_SIZE]
        value         = torch.tanh(self.value_W(board_repr).squeeze(-1))   # [B]

        return policy_logits, value


# ─── Loss function ────────────────────────────────────────────────────────────

def compute_loss(
    policy_logits: torch.Tensor,   # [B, VOCAB_SIZE]
    value:         torch.Tensor,   # [B]
    move_idx:      torch.Tensor,   # [B] long
    outcome:       torch.Tensor,   # [B] float in {-1, 0, 1}
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns (total_loss, policy_loss, value_loss).
    Total = policy_loss + 0.5 * value_loss
    """
    policy_loss = F.cross_entropy(policy_logits, move_idx)
    value_loss  = F.mse_loss(value, outcome)
    total_loss  = policy_loss + 0.5 * value_loss
    return total_loss, policy_loss, value_loss


# ─── Weight export ────────────────────────────────────────────────────────────

def export_weights(model: FrostMatrixV3, export_dir: str) -> None:
    """
    Export model weights as float32 raw binary files matching the
    TransformerWeights struct layout in transformer_v3_frostmatrix.cu.

    Also writes a JSON manifest listing each weight tensor: name, shape, dtype,
    filename, byte_offset=0 (each is its own file).

    Binary format: raw row-major float32, matching how alloc_weights populates
    CUDA device memory (cudaMemcpy from host float*).
    """
    os.makedirs(export_dir, exist_ok=True)
    manifest = []

    def save_tensor(name: str, tensor: torch.Tensor) -> None:
        """Save a single tensor as float32 binary and record in manifest."""
        t = tensor.detach().cpu().to(torch.float32).contiguous()
        fname = name.replace("[", "_").replace("]", "").replace("/", "_") + ".bin"
        fpath = os.path.join(export_dir, fname)
        with open(fpath, "wb") as f:
            f.write(t.numpy().tobytes())
        manifest.append({
            "name":      name,
            "shape":     list(t.shape),
            "dtype":     "float32",
            "numel":     t.numel(),
            "bytes":     t.numel() * 4,
            "file":      fname,
        })
        print(f"  exported {name}: {list(t.shape)} → {fname}")

    print(f"\n[export] writing weights to {export_dir}")

    # ── token_embed: [VOCAB_SIZE × D_MODEL] ───────────────────────────────────
    save_tensor("token_embed", model.token_embed.weight)

    # ── Per-layer weights ─────────────────────────────────────────────────────
    for l, layer in enumerate(model.layers):
        save_tensor(f"Wq[{l}]",        layer.Wq.weight)      # [D_MODEL × D_MODEL]
        save_tensor(f"Wk[{l}]",        layer.Wk.weight)
        save_tensor(f"Wv[{l}]",        layer.Wv.weight)
        save_tensor(f"Wo[{l}]",        layer.Wo.weight)
        save_tensor(f"ffn_up[{l}]",    layer.ffn.up.weight)  # [D_FFN × D_MODEL] (Linear stores weight transposed)
        save_tensor(f"ffn_gate[{l}]",  layer.ffn.gate.weight)
        save_tensor(f"ffn_down[{l}]",  layer.ffn.down.weight)
        save_tensor(f"ln1_w[{l}]",     layer.ln1.weight)     # [D_MODEL]
        save_tensor(f"ln1_b[{l}]",     layer.ln1.bias)
        save_tensor(f"ln2_w[{l}]",     layer.ln2.weight)
        save_tensor(f"ln2_b[{l}]",     layer.ln2.bias)

    # ── Output heads ──────────────────────────────────────────────────────────
    save_tensor("policy_W", model.policy_W.weight)   # [VOCAB_SIZE × D_MODEL]
    save_tensor("value_W",  model.value_W.weight)    # [1 × D_MODEL]
    save_tensor("value_b",  model.value_W.bias)      # [1]

    # ── Family axis weights ────────────────────────────────────────────────────
    save_tensor("fax/family_embed",   model.family_embed.weight)   # [N_FAMILIES × D_MODEL]
    save_tensor("fax/unknown_embed",  model.unknown_embed.weight)  # [N_UNKNOWNS × D_MODEL]
    save_tensor("fax/smooth_weights", model.highway.smooth_weights) # [N_FAMILIES × N_FAMILIES]

    # Highway projections: hw_Wq, hw_Wk, hw_Wv per direction
    for d in range(N_HW_DIRS):
        save_tensor(f"fax/hw_Wq[{d}]", model.highway.hw_Wq[d].weight)  # [D_HW_HEAD × D_MODEL]
        save_tensor(f"fax/hw_Wk[{d}]", model.highway.hw_Wk[d].weight)
        save_tensor(f"fax/hw_Wv[{d}]", model.highway.hw_Wv[d].weight)

    # hw_Wo: [N_HW_DIRS × D_HW_HEAD × D_MODEL] — store as [D_MODEL × (3*D_HW_HEAD)]
    save_tensor("fax/hw_Wo", model.highway.hw_Wo.weight)

    # Cross-attention
    save_tensor("fax/ca_Wq", model.cross_attn.ca_Wq.weight)   # [D_MODEL × D_MODEL]
    save_tensor("fax/ca_Wk", model.cross_attn.ca_Wk.weight)
    save_tensor("fax/ca_Wv", model.cross_attn.ca_Wv.weight)
    save_tensor("fax/ca_Wo", model.cross_attn.ca_Wo.weight)

    # Geometry projection
    save_tensor("fax/geo_proj_W", model.geo_proj.weight)   # [D_MODEL × N_GEO_CHANNELS]
    save_tensor("fax/geo_proj_b", model.geo_proj.bias)     # [D_MODEL]

    # ── Write manifest ─────────────────────────────────────────────────────────
    manifest_path = os.path.join(export_dir, "weights_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump({
            "architecture": "FrostMatrix_v3",
            "constants": {
                "N_FAMILIES":    N_FAMILIES,
                "N_UNKNOWNS":    N_UNKNOWNS,
                "N_FAM_NODES":   N_FAM_NODES,
                "N_BOARD_SQ":    N_BOARD_SQ,
                "SEQ_LEN_V3":    SEQ_LEN_V3,
                "D_MODEL":       D_MODEL,
                "D_HEAD":        D_HEAD,
                "N_HEADS":       N_HEADS,
                "N_LAYERS":      N_LAYERS,
                "D_FFN":         D_FFN,
                "VOCAB_SIZE":    VOCAB_SIZE,
                "N_HW_DIRS":     N_HW_DIRS,
                "D_HW_HEAD":     D_HW_HEAD,
                "N_GEO_CHANNELS": N_GEO_CHANNELS,
            },
            "tensors": manifest,
        }, f, indent=2)
    print(f"[export] manifest written to {manifest_path}")
    print(f"[export] total tensors: {len(manifest)}")
    total_params = sum(m["numel"] for m in manifest)
    print(f"[export] total parameters: {total_params:,} ({total_params*4/1e6:.1f} MB)")


# ─── Checkpoint save/load ─────────────────────────────────────────────────────

def save_checkpoint(
    model:     FrostMatrixV3,
    optimizer: torch.optim.Optimizer,
    scaler:    GradScaler,
    step:      int,
    epoch:     int,
    loss:      float,
    ckpt_dir:  str,
) -> str:
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f"ckpt_epoch{epoch}_step{step}.pt")
    torch.save({
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state":    scaler.state_dict(),
        "step":            step,
        "epoch":           epoch,
        "loss":            loss,
        "arch": {
            "N_FAMILIES": N_FAMILIES, "N_UNKNOWNS": N_UNKNOWNS,
            "D_MODEL": D_MODEL, "N_HEADS": N_HEADS, "N_LAYERS": N_LAYERS,
            "D_FFN": D_FFN, "VOCAB_SIZE": VOCAB_SIZE,
            "N_GEO_CHANNELS": N_GEO_CHANNELS,
        },
    }, path)
    return path


def load_checkpoint(
    path:      str,
    model:     FrostMatrixV3,
    optimizer: Optional[torch.optim.Optimizer],
    scaler:    Optional[GradScaler],
) -> Tuple[int, int, float]:
    """Returns (step, epoch, loss)."""
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scaler is not None and "scaler_state" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state"])
    step  = ckpt.get("step", 0)
    epoch = ckpt.get("epoch", 0)
    loss  = ckpt.get("loss", float("nan"))
    return step, epoch, loss


def find_latest_checkpoint(ckpt_dir: str) -> Optional[str]:
    """Find the checkpoint with the highest step number in ckpt_dir."""
    if not os.path.isdir(ckpt_dir):
        return None
    ckpts = [f for f in os.listdir(ckpt_dir) if f.startswith("ckpt_") and f.endswith(".pt")]
    if not ckpts:
        return None
    # Parse step numbers
    def parse_step(name: str) -> int:
        try:
            return int(name.split("step")[1].split(".")[0])
        except (IndexError, ValueError):
            return -1
    ckpts.sort(key=parse_step, reverse=True)
    return os.path.join(ckpt_dir, ckpts[0])


# ─── Training loop ────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: CUDA not available — running on CPU (not recommended)")

    ckpt_dir = os.path.join(DEFAULT_CKPT_DIR)
    os.makedirs(ckpt_dir, exist_ok=True)

    print(f"[vocab] deterministic from_sq*64+to_sq encoding, POLICY_SIZE={VOCAB_SIZE}")

    # ── Model ──────────────────────────────────────────────────────────────────
    model = FrostMatrixV3().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] FrostMatrix v3: {n_params:,} trainable parameters")

    # ── Optimizer + AMP ────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-2,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    scaler = GradScaler(enabled=(device.type == "cuda"))

    # ── Resume from checkpoint ─────────────────────────────────────────────────
    start_step  = 0
    start_epoch = 0

    ckpt_path = args.checkpoint
    if ckpt_path is None:
        ckpt_path = find_latest_checkpoint(ckpt_dir)

    if ckpt_path is not None and os.path.exists(ckpt_path):
        print(f"[ckpt] resuming from {ckpt_path}")
        start_step, start_epoch, prev_loss = load_checkpoint(
            ckpt_path, model, optimizer, scaler
        )
        print(f"[ckpt] step={start_step} epoch={start_epoch} loss={prev_loss:.4f}")
    else:
        print("[ckpt] starting from scratch")

    # ── Dataset + DataLoader ───────────────────────────────────────────────────
    def make_dataset(epoch_num: int) -> FrostMatrixDataset:
        return FrostMatrixDataset(args.data)

    def collate_fn(batch):
        board_tokens = torch.stack([b[0] for b in batch])   # [B, 64]
        family_ids   = torch.stack([b[1] for b in batch])   # [B]
        geo_vecs     = torch.stack([b[2] for b in batch])   # [B, 64, 55]
        move_idxs    = torch.stack([b[3] for b in batch])   # [B]
        outcomes     = torch.stack([b[4] for b in batch])   # [B]
        return board_tokens, family_ids, geo_vecs, move_idxs, outcomes

    # ── Training loop ──────────────────────────────────────────────────────────
    global_step = start_step
    model.train()

    for epoch in range(start_epoch, args.epochs):
        dataset    = make_dataset(epoch)
        dataloader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=(device.type == "cuda"),
            prefetch_factor=2,
        )

        epoch_policy_loss = 0.0
        epoch_value_loss  = 0.0
        epoch_total_loss  = 0.0
        epoch_steps       = 0
        t_start           = time.time()

        for batch in dataloader:
            board_tokens, family_ids, geo_vecs, move_idxs, outcomes = batch

            board_tokens = board_tokens.to(device, non_blocking=True)
            family_ids   = family_ids.to(device, non_blocking=True)
            geo_vecs     = geo_vecs.to(device, non_blocking=True)
            move_idxs    = move_idxs.to(device, non_blocking=True)
            outcomes     = outcomes.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=(device.type == "cuda")):
                policy_logits, value = model(board_tokens, family_ids, geo_vecs)
                total_loss, policy_loss, value_loss = compute_loss(
                    policy_logits, value, move_idxs, outcomes
                )

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            global_step      += 1
            epoch_steps      += 1
            epoch_total_loss  += total_loss.item()
            epoch_policy_loss += policy_loss.item()
            epoch_value_loss  += value_loss.item()

            # ── Log every 100 steps ───────────────────────────────────────────
            if global_step % 100 == 0:
                avg_total  = epoch_total_loss  / epoch_steps
                avg_policy = epoch_policy_loss / epoch_steps
                avg_value  = epoch_value_loss  / epoch_steps
                elapsed    = time.time() - t_start
                steps_per_sec = epoch_steps / elapsed if elapsed > 0 else 0.0
                print(
                    f"[train] epoch={epoch+1}/{args.epochs} step={global_step:,} "
                    f"loss={avg_total:.4f} "
                    f"policy={avg_policy:.4f} "
                    f"value={avg_value:.4f} "
                    f"({steps_per_sec:.1f} steps/s)"
                )

            # ── Save checkpoint every 1000 steps ──────────────────────────────
            if global_step % 1000 == 0:
                ckpt_saved = save_checkpoint(
                    model, optimizer, scaler,
                    global_step, epoch,
                    epoch_total_loss / epoch_steps,
                    ckpt_dir,
                )
                print(f"[ckpt] saved → {ckpt_saved}")

        # ── End of epoch ───────────────────────────────────────────────────────
        avg_total  = epoch_total_loss  / max(epoch_steps, 1)
        avg_policy = epoch_policy_loss / max(epoch_steps, 1)
        avg_value  = epoch_value_loss  / max(epoch_steps, 1)
        elapsed    = time.time() - t_start
        print(
            f"\n[epoch] {epoch+1}/{args.epochs} complete — "
            f"steps={epoch_steps:,} loss={avg_total:.4f} "
            f"policy={avg_policy:.4f} value={avg_value:.4f} "
            f"time={elapsed:.1f}s\n"
        )

        # Save end-of-epoch checkpoint
        ckpt_saved = save_checkpoint(
            model, optimizer, scaler,
            global_step, epoch + 1,
            avg_total,
            ckpt_dir,
        )
        print(f"[ckpt] end-of-epoch checkpoint → {ckpt_saved}")

    # ── Export weights after training ──────────────────────────────────────────
    export_dir = os.path.join(ckpt_dir, "export")
    export_weights(model, export_dir)
    print(f"\n[done] training complete. weights exported to {export_dir}")


def export_only(args: argparse.Namespace) -> None:
    """Load the latest checkpoint and export weights without training."""
    ckpt_dir = DEFAULT_CKPT_DIR
    ckpt_path = args.checkpoint or find_latest_checkpoint(ckpt_dir)

    if ckpt_path is None:
        print(f"[error] no checkpoint found in {ckpt_dir}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = FrostMatrixV3().to(device)
    scaler = GradScaler(enabled=False)

    print(f"[export-only] loading {ckpt_path}")
    step, epoch, loss, _ = load_checkpoint(ckpt_path, model, None, scaler)
    print(f"[export-only] loaded step={step} epoch={epoch} loss={loss:.4f}")

    model.eval()
    export_dir = os.path.join(ckpt_dir, "export")
    export_weights(model, export_dir)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FrostMatrix v3 training wrapper — PyTorch mirror of transformer_v3_frostmatrix.cu"
    )
    parser.add_argument(
        "--data",
        default=DEFAULT_DATA_PATH,
        help=f"Path to training JSONL (default: {DEFAULT_DATA_PATH})",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to checkpoint to resume from (default: latest in ckpt_dir)",
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Skip training; just export weights from latest checkpoint",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.export_only:
        export_only(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
