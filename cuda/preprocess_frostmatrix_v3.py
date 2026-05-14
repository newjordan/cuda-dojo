#!/usr/bin/env python3
"""
preprocess_frostmatrix_v3.py

Parse training_v2.jsonl once → extract only the fields FrostMatrix v3 actually uses
→ save as compact PyTorch tensors in a single .pt file.

What FrostMatrix v3 uses:
  - board_tokens: [N, 64]  int8    piece type per square (0-12)
  - family_ids:   [N]      int16   opening family (0-65)
  - move_froms:   [N]      int8    from-square index (0-63)
  - move_tos:     [N]      int8    to-square index (0-63)
  - outcomes:     [N]      float16 value target (-1, 0, 1)

What it does NOT use (removed/deprecated):
  - geo vectors: disabled in model (line 800: "destroys 84%→27% accuracy")
  - fen strings: only parsed to extract board tokens
  - tokenizer field: unused

Output: single .pt file with a dict of tensors, loadable via torch.load().

Usage:
  python cuda/preprocess_frostmatrix_v3.py \
    --input training_v2.jsonl \
    --output training_v2_preprocessed.pt \
    [--max-rows N]
"""

import argparse
import json
import os
import sys
import time
import torch
from typing import Optional, Tuple

# ── Constants (must match train_v3_frostmatrix.py) ─────────────────────────

PIECE_MAP: dict = {
    'P': 1,  'N': 2,  'B': 3,  'R': 4,  'Q': 5,  'K': 6,
    'p': 7,  'n': 8,  'b': 9,  'r': 10, 'q': 11, 'k': 12,
}
N_SQUARES = 64
N_FAMILIES = 66  # 0-65 (65 family-axis nodes + 1 padding)

# ── FEN Parsing (minimal — only board layout) ──────────────────────────────

def fen_to_board(fen: str):
    """Parse FEN board section → list of 64 piece indices (0=empty)."""
    board = [0] * N_SQUARES
    parts = fen.strip().split()
    if not parts:
        return board
    rank_strs = parts[0].split('/')
    sq = 0
    for rank_str in rank_strs:
        for ch in rank_str:
            if ch.isdigit():
                sq += int(ch)
            else:
                if sq < N_SQUARES:
                    board[sq] = PIECE_MAP.get(ch, 0)
                sq += 1
        if sq >= N_SQUARES:
            break
    return board


def uci_to_squares(move_uci: str) -> Tuple[int, int]:
    """Convert UCI move (e.g. 'e2e4') → (from_sq, to_sq) indices."""
    if len(move_uci) < 4:
        return 0, 0
    try:
        f_col = ord(move_uci[0]) - ord('a')
        f_row = int(move_uci[1]) - 1
        t_col = ord(move_uci[2]) - ord('a')
        t_row = int(move_uci[3]) - 1
        from_sq = max(0, min(63, f_row * 8 + f_col))
        to_sq   = max(0, min(63, t_row * 8 + t_col))
        return from_sq, to_sq
    except (ValueError, IndexError):
        return 0, 0


# ── Preprocessing ──────────────────────────────────────────────────────────

def preprocess(
    input_path: str,
    output_path: str,
    max_rows: Optional[int] = None,
    report_interval: int = 500_000,
):
    print(f"[preprocess] input:  {input_path}")
    print(f"[preprocess] output: {output_path}")
    file_size = os.path.getsize(input_path)
    print(f"[preprocess] size:   {file_size / 1e9:.2f} GB")

    # Pre-allocate lists (we'll convert to tensors at the end)
    board_tokens_list = []
    family_ids_list  = []
    move_froms_list  = []
    move_tos_list    = []
    outcomes_list    = []

    t_start = time.time()
    bytes_read = 0
    line_count = 0
    parse_errors = 0
    skip_no_move = 0

    with open(input_path, 'r') as f:
        for line in f:
            bytes_read += len(line.encode('utf-8'))
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                parse_errors += 1
                continue

            # ── Extract fields ──────────────────────────────────────────
            fen = obj.get('fen', '')
            family_id = int(obj.get('family_id', 0))
            move_uci = obj.get('move', '') or obj.get('uci_move', '')
            outcome = float(obj.get('outcome', 0.5))

            if not fen or not move_uci:
                skip_no_move += 1
                continue

            # ── Parse FEN → board ───────────────────────────────────────
            board = fen_to_board(fen)

            # ── Parse move → from/to squares ────────────────────────────
            from_sq, to_sq = uci_to_squares(move_uci)

            # ── Remap outcome: {0→-1, 0.5→0, 1→1} ─────────────────────
            outcome_remapped = outcome * 2.0 - 1.0

            # ── Clamp family_id ─────────────────────────────────────────
            family_id = max(0, min(family_id, N_FAMILIES - 1))

            # ── Accumulate ──────────────────────────────────────────────
            board_tokens_list.append(board)
            family_ids_list.append(family_id)
            move_froms_list.append(from_sq)
            move_tos_list.append(to_sq)
            outcomes_list.append(outcome_remapped)

            line_count += 1

            # Progress report
            if line_count % report_interval == 0:
                elapsed = time.time() - t_start
                rate = line_count / max(elapsed, 1)
                pct = (bytes_read / file_size * 100) if file_size > 0 else 0
                mem_est = sum(len(lst) for lst in [
                    board_tokens_list, family_ids_list,
                    move_froms_list, move_tos_list, outcomes_list,
                ])
                print(f"[preprocess] {line_count/1e6:.1f}M rows | "
                      f"{rate:.0f} rows/s | {pct:.1f}% | "
                      f"mem_est: {mem_est/1e6:.1f}M elems")

            if max_rows is not None and line_count >= max_rows:
                break

    elapsed = time.time() - t_start
    print(f"\n[preprocess] parsed {line_count:,} rows in {elapsed:.1f}s "
          f"({line_count/elapsed:.0f} rows/s)")
    if parse_errors:
        print(f"[preprocess] JSON parse errors: {parse_errors}")
    if skip_no_move:
        print(f"[preprocess] skipped (no move):  {skip_no_move}")

    # ── Convert to tensors ──────────────────────────────────────────────
    print("[preprocess] converting to tensors...")
    t_convert = time.time()

    board_tokens = torch.tensor(board_tokens_list, dtype=torch.int8)    # [N, 64]
    family_ids   = torch.tensor(family_ids_list,  dtype=torch.int16)    # [N]
    move_froms   = torch.tensor(move_froms_list,  dtype=torch.int8)     # [N]
    move_tos     = torch.tensor(move_tos_list,    dtype=torch.int8)     # [N]
    outcomes     = torch.tensor(outcomes_list,    dtype=torch.float16)   # [N]

    del (board_tokens_list, family_ids_list, move_froms_list,
         move_tos_list, outcomes_list)

    print(f"[preprocess] tensors built in {time.time() - t_convert:.1f}s")
    print(f"  board_tokens: {board_tokens.shape} {board_tokens.dtype} "
          f"({board_tokens.nbytes / 1e9:.3f} GB)")
    print(f"  family_ids:   {family_ids.shape} {family_ids.dtype} "
          f"({family_ids.nbytes / 1e6:.1f} MB)")
    print(f"  move_froms:   {move_froms.shape} {move_froms.dtype} "
          f"({move_froms.nbytes / 1e6:.1f} MB)")
    print(f"  move_tos:     {move_tos.shape} {move_tos.dtype} "
          f"({move_tos.nbytes / 1e6:.1f} MB)")
    print(f"  outcomes:     {outcomes.shape} {outcomes.dtype} "
          f"({outcomes.nbytes / 1e6:.1f} MB)")
    total_gb = sum(t.nbytes for t in [board_tokens, family_ids,
                                       move_froms, move_tos, outcomes]) / 1e9
    print(f"  TOTAL: {total_gb:.3f} GB for {line_count:,} samples")

    # ── Save ────────────────────────────────────────────────────────────
    print(f"[preprocess] saving to {output_path}...")
    t_save = time.time()
    data = {
        'board_tokens': board_tokens,
        'family_ids':   family_ids,
        'move_froms':   move_froms,
        'move_tos':     move_tos,
        'outcomes':     outcomes,
        'n_samples':    line_count,
        'version':      'frostmatrix_v3_preprocessed_v1',
    }
    torch.save(data, output_path)
    print(f"[preprocess] saved in {time.time() - t_save:.1f}s")
    print(f"[preprocess] done. {line_count:,} samples → {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess FrostMatrix V3 training data')
    parser.add_argument('--input', required=True, help='Path to training_v2.jsonl')
    parser.add_argument('--output', required=True, help='Output .pt file path')
    parser.add_argument('--max-rows', type=int, default=None,
                        help='Maximum rows to process (for testing)')
    parser.add_argument('--report-interval', type=int, default=1_000_000,
                        help='Rows between progress reports')
    args = parser.parse_args()

    preprocess(
        input_path=args.input,
        output_path=args.output,
        max_rows=args.max_rows,
        report_interval=args.report_interval,
    )
