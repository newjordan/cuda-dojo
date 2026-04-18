#!/usr/bin/env python3
"""
perft_compare.py — accuracy ablation of GPU movegen vs python-chess gold standard.

Invocation:
    python3 cuda/perft_compare.py [--max-book N] [--canonical-depth D]
                                   [--book-depth D] [--report PATH]
                                   [--binary PATH]

What it does
------------
1. Runs perft on the 6 canonical positions (start, Kiwipete, P3, P4, P5, P6)
   at depths 1..CANONICAL_DEPTH on both:
     - python-chess (CPU reference, gold standard)
     - cuda/gpu_fighter --perft DEPTH (host mirror of GPU movegen)
2. Samples MAX_BOOK FENs from gpu_spine/book.jsonl and runs perft d=1..BOOK_DEPTH
   on both engines.
3. For each (fen, depth) records: cpu_count, gpu_count, match.
4. For each MISMATCH at depth==1, computes the SET-DIFFERENCE of legal moves
   (uci) so the user can see which moves the GPU added or missed.
5. Writes a JSON report and prints a summary.

The host-side h_generate_moves / h_make_move / h_in_check in gpu_fighter.cu are
bit-for-bit mirrors of the device-side d_generate_moves / d_make_move /
d_is_square_attacked. Verifying the host mirror IS verifying the GPU movegen.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from typing import Dict, List, Tuple

import chess  # python-chess 1.11.2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_BIN = os.path.join(ROOT, "cuda", "gpu_fighter")
DEFAULT_BOOK = os.path.join(ROOT, "gpu_spine", "book.jsonl")

CANONICAL = [
    ("startpos", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
    ("kiwipete", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"),
    ("position3", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"),
    ("position4", "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1"),
    ("position5", "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8"),
    ("position6", "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10"),
]

# Known gold-standard perft values (chessprogramming.org wiki).
# Used as a SECOND check that python-chess is correct (and that we haven't
# misconfigured anything).
KNOWN: Dict[Tuple[str, int], int] = {
    ("startpos", 1): 20,
    ("startpos", 2): 400,
    ("startpos", 3): 8902,
    ("startpos", 4): 197281,
    ("startpos", 5): 4865609,
    ("kiwipete", 1): 48,
    ("kiwipete", 2): 2039,
    ("kiwipete", 3): 97862,
    ("kiwipete", 4): 4085603,
    ("position3", 1): 14,
    ("position3", 2): 191,
    ("position3", 3): 2812,
    ("position3", 4): 43238,
    ("position3", 5): 674624,
    ("position4", 1): 6,
    ("position4", 2): 264,
    ("position4", 3): 9467,
    ("position4", 4): 422333,
    ("position5", 1): 44,
    ("position5", 2): 1486,
    ("position5", 3): 62379,
    ("position5", 4): 2103487,
    ("position6", 1): 46,
    ("position6", 2): 2079,
    ("position6", 3): 89890,
    ("position6", 4): 3894594,
}


# -------------------- python-chess perft --------------------
def cpu_perft(board: chess.Board, depth: int) -> int:
    if depth == 0:
        return 1
    if depth == 1:
        return board.legal_moves.count()
    n = 0
    for mv in board.legal_moves:
        board.push(mv)
        n += cpu_perft(board, depth - 1)
        board.pop()
    return n


def cpu_legal_uci(board: chess.Board) -> List[str]:
    return sorted(mv.uci() for mv in board.legal_moves)


# -------------------- GPU perft via gpu_fighter --binary --perft --------------------
def gpu_perft_one(binary: str, fen: str, depth: int, divide: bool = False
                  ) -> Tuple[int, List[Tuple[str, int]]]:
    """
    Returns (total, [(uci, subcount), ...])
    subcount list is empty unless divide=True.
    """
    args = [binary, "--perft", str(depth)]
    if divide:
        args.append("--perft-divide")
    res = subprocess.run(
        args,
        input=fen + "\n",
        capture_output=True,
        text=True,
        timeout=600,
    )
    if res.returncode != 0:
        raise RuntimeError(f"gpu_fighter exit {res.returncode}: {res.stderr.strip()}")
    total = -1
    children: List[Tuple[str, int]] = []
    for line in res.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("FEN ") or line == "END":
            continue
        if line.startswith("TOTAL "):
            total = int(line.split()[1])
        elif line.startswith("PERFT "):
            parts = line.split()
            total = int(parts[2])
        else:
            # "<uci> <count>"
            parts = line.split()
            if len(parts) == 2:
                try:
                    children.append((parts[0], int(parts[1])))
                except ValueError:
                    pass
    if divide and total < 0:
        total = sum(c for _, c in children)
    if total < 0:
        raise RuntimeError(f"gpu_fighter produced no PERFT/TOTAL line. stdout=\n{res.stdout}")
    return total, children


# -------------------- main flow --------------------
def run(binary: str, book_path: str, max_book: int,
        canonical_depth: int, book_depth: int, report_path: str,
        seed: int = 1337) -> int:
    if not os.path.isfile(binary):
        print(f"FATAL: GPU binary not found: {binary}", file=sys.stderr)
        return 2
    if not os.access(binary, os.X_OK):
        print(f"FATAL: GPU binary not executable: {binary}", file=sys.stderr)
        return 2

    # ---- assemble position list ----
    positions: List[Tuple[str, str, int]] = []  # (label, fen, max_depth)
    for label, fen in CANONICAL:
        positions.append((label, fen, canonical_depth))

    book_fens: List[str] = []
    if os.path.isfile(book_path):
        with open(book_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if "fen" in rec:
                        book_fens.append(rec["fen"])
                except json.JSONDecodeError:
                    continue
    rng = random.Random(seed)
    if max_book > 0 and book_fens:
        if len(book_fens) > max_book:
            book_fens = rng.sample(book_fens, max_book)
        for i, fen in enumerate(book_fens):
            positions.append((f"book{i:04d}", fen, book_depth))

    # ---- run perft on each ----
    results: List[Dict] = []
    mismatches: List[Dict] = []
    n_tested = 0
    n_matched = 0
    cpu_time = 0.0
    gpu_time = 0.0

    # Sanity: confirm python-chess matches KNOWN values where available.
    sanity_failures: List[str] = []
    for label, fen in CANONICAL:
        for d in range(1, canonical_depth + 1):
            if (label, d) not in KNOWN:
                continue
            board = chess.Board(fen)
            t0 = time.time()
            cnt = cpu_perft(board, d)
            cpu_time += time.time() - t0
            if cnt != KNOWN[(label, d)]:
                sanity_failures.append(
                    f"python-chess BROKEN: {label} d={d} cpu={cnt} expected={KNOWN[(label,d)]}"
                )
    if sanity_failures:
        for s in sanity_failures:
            print(s, file=sys.stderr)
        return 3

    # ---- the actual GPU vs CPU comparison ----
    for label, fen, max_d in positions:
        for d in range(1, max_d + 1):
            board = chess.Board(fen)
            t0 = time.time()
            cpu_count = cpu_perft(board, d)
            cpu_time += time.time() - t0

            t0 = time.time()
            try:
                gpu_count, _ = gpu_perft_one(binary, fen, d, divide=False)
            except Exception as e:
                gpu_count = -1
                gpu_err = repr(e)
            else:
                gpu_err = None
            gpu_time += time.time() - t0

            match = (gpu_count == cpu_count)
            n_tested += 1
            if match:
                n_matched += 1

            # Cross-check vs KNOWN if available
            known = KNOWN.get((label, d))
            known_ok = (known is None) or (cpu_count == known)

            rec = {
                "label": label,
                "fen": fen,
                "depth": d,
                "cpu_count": cpu_count,
                "gpu_count": gpu_count,
                "match": match,
                "known": known,
                "known_ok": known_ok,
            }
            if gpu_err:
                rec["gpu_error"] = gpu_err
            results.append(rec)

            if not match:
                # Compute set-diff at depth 1 for diagnostic.
                cpu_moves = set(cpu_legal_uci(chess.Board(fen)))
                gpu_moves: List[str] = []
                try:
                    _, divide = gpu_perft_one(binary, fen, max(d, 1), divide=True)
                    gpu_moves = [u for u, _ in divide]
                except Exception as e:
                    rec["divide_error"] = repr(e)
                gpu_set = set(gpu_moves)
                # NOTE: python-chess emits castling as e1g1, gpu emits e1g1 too
                # (since FLAG_CASTLE_K writes from=60 to=62 -> "e1g1"). Same scheme.
                gpu_only = sorted(gpu_set - cpu_moves)
                cpu_only = sorted(cpu_moves - gpu_set)
                mismatches.append({
                    "label": label,
                    "fen": fen,
                    "depth": d,
                    "cpu_count": cpu_count,
                    "gpu_count": gpu_count,
                    "delta": gpu_count - cpu_count,
                    "gpu_only_at_d1": gpu_only,
                    "cpu_only_at_d1": cpu_only,
                })

            tag = "OK" if match else "MISMATCH"
            kn_str = ""
            if known is not None:
                kn_str = f" (known={known}, cpu_known_ok={known_ok})"
            sys.stdout.write(
                f"  [{tag}] {label:>10s} d={d}  cpu={cpu_count:>10d}  gpu={gpu_count:>10d}{kn_str}\n"
            )
            sys.stdout.flush()

    # ---- summary ----
    summary = {
        "n_tested": n_tested,
        "n_matched": n_matched,
        "agreement_rate": (n_matched / n_tested) if n_tested else 0.0,
        "cpu_seconds": round(cpu_time, 3),
        "gpu_seconds": round(gpu_time, 3),
        "binary": binary,
        "book_path": book_path,
        "book_sample_size": max_book,
        "canonical_depth": canonical_depth,
        "book_depth": book_depth,
        "n_mismatches": len(mismatches),
        "mismatch_labels": sorted({m["label"] for m in mismatches}),
    }
    out = {"summary": summary, "results": results, "mismatches": mismatches}
    with open(report_path, "w") as f:
        json.dump(out, f, indent=2)

    print()
    print("================ PERFT COMPARE SUMMARY ================")
    print(f"  positions tested      : {n_tested}")
    print(f"  matched               : {n_matched}")
    print(f"  agreement rate        : {summary['agreement_rate']*100:.4f}%")
    print(f"  mismatches            : {len(mismatches)}")
    print(f"  CPU time (python-chess): {summary['cpu_seconds']}s")
    print(f"  GPU time (gpu_fighter) : {summary['gpu_seconds']}s")
    print(f"  report                : {report_path}")
    if mismatches:
        print()
        print("---- TOP MISMATCHES (up to 10) ----")
        for m in mismatches[:10]:
            print(f"  {m['label']} d={m['depth']}  cpu={m['cpu_count']} "
                  f"gpu={m['gpu_count']} delta={m['delta']}")
            if m["gpu_only_at_d1"]:
                print(f"     GPU-only @d1: {m['gpu_only_at_d1']}")
            if m["cpu_only_at_d1"]:
                print(f"     CPU-only @d1: {m['cpu_only_at_d1']}")
            print(f"     FEN: {m['fen']}")
    return 0 if n_matched == n_tested else 1


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default=DEFAULT_BIN, help="path to gpu_fighter binary")
    ap.add_argument("--book", default=DEFAULT_BOOK, help="path to book.jsonl")
    ap.add_argument("--max-book", type=int, default=100,
                    help="number of book positions to sample (default 100, 0 to disable)")
    ap.add_argument("--canonical-depth", type=int, default=4,
                    help="max depth for canonical positions (default 4)")
    ap.add_argument("--book-depth", type=int, default=2,
                    help="max depth for book positions (default 2)")
    ap.add_argument("--report", default=os.path.join(ROOT, "cuda", "perft_report.json"),
                    help="JSON report output path")
    ap.add_argument("--seed", type=int, default=1337, help="rng seed for sampling")
    args = ap.parse_args(argv)

    return run(
        binary=args.binary,
        book_path=args.book,
        max_book=args.max_book,
        canonical_depth=args.canonical_depth,
        book_depth=args.book_depth,
        report_path=args.report,
        seed=args.seed,
    )


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
