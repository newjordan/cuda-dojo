#!/usr/bin/env python3
"""Validation harness for cuda_arbiter_daemon_batched.py.

We exercise the BatchCoalescer in-process, with deterministic
in-Python "fighters" (no docker), so the measurement isolates the
dispatcher's effect on dojo_ref FFI throughput and is reproducible.

Validation gates (Tier 3d task spec):
  1. Parity: 4 concurrent games × 30 plies, deterministic fighters.
     Move list / outcome must be byte-identical between serial and
     coalesced paths.
  2. Perf: 8 concurrent games × 30 plies, serial vs coalesced wall
     clock. Target ≥ 1.5× speedup.
  3. Stress: 16 concurrent games × 60 plies, no crash / deadlock.

Both perf paths use the SAME deterministic fighter; the only
difference is whether dojo_ref calls are coalesced or serial.
"""
from __future__ import annotations

import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_LIVE_ARBITER = _HERE.parent / "live_arbiter"
if str(_LIVE_ARBITER) not in sys.path:
    sys.path.insert(0, str(_LIVE_ARBITER))
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import dojo_ref  # noqa: E402
from gpu_state import HostState  # noqa: E402

from cuda_arbiter_daemon_batched import BatchCoalescer  # noqa: E402


# ----- Deterministic in-process fighter ---------------------------------
# Picks the legal move at index (ply * seed_a + seed_b) mod len(legal).
# Pure function of (legal_moves_list, ply, seed_a, seed_b) — identical
# choice for identical input regardless of which dispatch path is used.

def deterministic_fighter(legal: list[str], ply: int, seed_a: int, seed_b: int) -> str:
    return legal[(ply * seed_a + seed_b) % len(legal)]


# ----- Two play_game_inproc variants. Identical control flow; differ ----
#       only in how legal_moves / is_check / make_move dispatch.

def play_game_serial(seed_a: int, seed_b: int, max_plies: int) -> dict:
    """Direct dojo_ref calls, no coalescer. Mirrors the production
    daemon's per-position dispatch."""
    pos = dojo_ref.Position.startpos()
    host = HostState()
    moves: list[str] = []
    for ply in range(max_plies):
        if host.halfmove >= 100:
            return {"result": "draw", "reason": "50-move", "moves": moves}
        if host.insufficient_material():
            return {"result": "draw", "reason": "insufficient", "moves": moves}
        count = host.record_position_for_threefold()
        if count >= 3:
            return {"result": "draw", "reason": "threefold", "moves": moves}

        legal = pos.legal_moves()
        if not legal:
            if pos.is_check():
                winner = "black" if host.side == "w" else "white"
                return {"result": winner, "reason": "checkmate", "moves": moves}
            return {"result": "draw", "reason": "stalemate", "moves": moves}

        uci = deterministic_fighter(legal, ply, seed_a, seed_b)
        moves.append(uci)
        pos = pos.make_move(uci)
        host.apply_uci(uci)
    return {"result": "draw", "reason": "max_plies", "moves": moves}


def play_game_coalesced(coalescer: BatchCoalescer,
                        seed_a: int, seed_b: int, max_plies: int) -> dict:
    """All dojo_ref calls go through the coalescer's dispatcher thread."""
    pos = dojo_ref.Position.startpos()
    host = HostState()
    moves: list[str] = []
    for ply in range(max_plies):
        if host.halfmove >= 100:
            return {"result": "draw", "reason": "50-move", "moves": moves}
        if host.insufficient_material():
            return {"result": "draw", "reason": "insufficient", "moves": moves}
        count = host.record_position_for_threefold()
        if count >= 3:
            return {"result": "draw", "reason": "threefold", "moves": moves}

        legal = coalescer.legal_moves(pos)
        if not legal:
            if coalescer.is_check(pos):
                winner = "black" if host.side == "w" else "white"
                return {"result": winner, "reason": "checkmate", "moves": moves}
            return {"result": "draw", "reason": "stalemate", "moves": moves}

        uci = deterministic_fighter(legal, ply, seed_a, seed_b)
        moves.append(uci)
        pos = coalescer.make_move(pos, uci)
        host.apply_uci(uci)
    return {"result": "draw", "reason": "max_plies", "moves": moves}


# ----- Gates ------------------------------------------------------------

def gate1_parity(n_games: int = 4, max_plies: int = 30) -> bool:
    print(f"\n[gate 1] PARITY: {n_games} concurrent games × {max_plies} plies")
    # (seed_a, seed_b) pairs -> distinct deterministic fighter behaviors.
    seeds = [(3, 7), (5, 11), (2, 13), (17, 4)][:n_games]

    serial_results = []
    for sa, sb in seeds:
        serial_results.append(play_game_serial(sa, sb, max_plies))

    coalescer = BatchCoalescer(max_batch=32, deadline_ms=0.5)
    coalescer.start()
    coalesced_results = [None] * n_games

    def worker(i: int, sa: int, sb: int) -> None:
        coalesced_results[i] = play_game_coalesced(coalescer, sa, sb, max_plies)

    threads = [
        threading.Thread(target=worker, args=(i, sa, sb))
        for i, (sa, sb) in enumerate(seeds)
    ]
    for t in threads: t.start()
    for t in threads: t.join()
    coalescer.stop()

    all_ok = True
    for i, (s, c) in enumerate(zip(serial_results, coalesced_results)):
        same_moves = s["moves"] == c["moves"]
        same_result = s["result"] == c["result"] and s["reason"] == c["reason"]
        ok = same_moves and same_result
        all_ok = all_ok and ok
        print(f"  game {i}: {'PASS' if ok else 'FAIL'} "
              f"(plies={len(s['moves'])} vs {len(c['moves'])}; "
              f"{s['result']}/{s['reason']} vs {c['result']}/{c['reason']})")
        if not same_moves:
            # Show first divergence.
            for j, (a, b) in enumerate(zip(s["moves"], c["moves"])):
                if a != b:
                    print(f"    first divergence at ply {j}: serial={a} coalesced={b}")
                    break
            if len(s["moves"]) != len(c["moves"]):
                print(f"    move-list lengths differ: {len(s['moves'])} vs {len(c['moves'])}")
    print(f"[gate 1] {'PASS' if all_ok else 'FAIL'}")
    return all_ok


def gate2_perf(n_games: int = 8, max_plies: int = 30) -> tuple[float, float, float]:
    print(f"\n[gate 2] PERF: {n_games} concurrent games × {max_plies} plies")
    seeds = [(3 + i, 7 + i * 2) for i in range(n_games)]

    # Serial: even though we run them concurrently across threads, each
    # thread issues per-position FFI calls. Mirrors the prod daemon.
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=n_games) as ex:
        ser_futs = [ex.submit(play_game_serial, sa, sb, max_plies) for sa, sb in seeds]
        ser_results = [f.result() for f in ser_futs]
    t_serial = time.monotonic() - t0

    # Coalesced.
    coalescer = BatchCoalescer(max_batch=32, deadline_ms=0.5)
    coalescer.start()
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=n_games) as ex:
        coa_futs = [
            ex.submit(play_game_coalesced, coalescer, sa, sb, max_plies)
            for sa, sb in seeds
        ]
        coa_results = [f.result() for f in coa_futs]
    t_coalesced = time.monotonic() - t0
    coalescer.stop()

    speedup = t_serial / t_coalesced if t_coalesced > 0 else float("inf")
    print(f"  serial    wall: {t_serial:.3f}s  (per-game {t_serial/n_games*1000:.1f} ms)")
    print(f"  coalesced wall: {t_coalesced:.3f}s  (per-game {t_coalesced/n_games*1000:.1f} ms)")
    print(f"  speedup: {speedup:.2f}×  (target ≥ 1.5×)")
    print(f"  coalescer stats: {coalescer.stats_snapshot()}")
    # Cross-check: results align between paths (sanity, not a separate gate).
    aligned = all(s["moves"] == c["moves"] for s, c in zip(ser_results, coa_results))
    print(f"  parity at perf scale: {'PASS' if aligned else 'FAIL'}")
    return t_serial, t_coalesced, speedup


def gate3_stress(n_games: int = 16, max_plies: int = 60,
                 timeout_s: float = 60.0) -> bool:
    print(f"\n[gate 3] STRESS: {n_games} concurrent games × {max_plies} plies "
          f"(timeout {timeout_s}s)")
    seeds = [(3 + i, 7 + i * 5) for i in range(n_games)]
    coalescer = BatchCoalescer(max_batch=32, deadline_ms=0.5)
    coalescer.start()

    results: list[dict] = []
    errors: list[BaseException] = []
    done = threading.Event()

    def runner():
        try:
            with ThreadPoolExecutor(max_workers=n_games) as ex:
                futs = [
                    ex.submit(play_game_coalesced, coalescer, sa, sb, max_plies)
                    for sa, sb in seeds
                ]
                for f in futs:
                    results.append(f.result())
        except BaseException as e:
            errors.append(e)
        finally:
            done.set()

    t0 = time.monotonic()
    th = threading.Thread(target=runner, daemon=True)
    th.start()
    finished = done.wait(timeout=timeout_s)
    wall = time.monotonic() - t0
    coalescer.stop()

    if not finished:
        print(f"  FAIL: did not complete within {timeout_s}s — possible deadlock")
        return False
    if errors:
        print(f"  FAIL: exceptions raised: {errors!r}")
        return False
    print(f"  completed {len(results)}/{n_games} games in {wall:.2f}s")
    print(f"  coalescer stats: {coalescer.stats_snapshot()}")
    return len(results) == n_games


def main() -> int:
    print("Tier 3d: BatchCoalescer validation harness")
    # Warm dojo_ref so the first batch isn't paying init cost on the dev path.
    _ = dojo_ref.Position.startpos().legal_moves()

    g1 = gate1_parity()
    t_s, t_c, speedup = gate2_perf()
    # Also measure at 16 (same concurrency the prod load is being sized for).
    print("\n[gate 2b] PERF AT N=16 (same workload as stress, but vs serial):")
    t_s16, t_c16, sp16 = gate2_perf(n_games=16, max_plies=30)
    g3 = gate3_stress()

    print("\n=== SUMMARY ===")
    print(f"  gate 1 parity:   {'PASS' if g1 else 'FAIL'}")
    print(f"  gate 2 perf:     {speedup:.2f}× (serial {t_s:.3f}s, "
          f"coalesced {t_c:.3f}s; target ≥ 1.5×) "
          f"{'PASS' if speedup >= 1.5 else 'FAIL'}")
    print(f"  gate 3 stress:   {'PASS' if g3 else 'FAIL'}")
    return 0 if (g1 and speedup >= 1.5 and g3) else 1


if __name__ == "__main__":
    sys.exit(main())
