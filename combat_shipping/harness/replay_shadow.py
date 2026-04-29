#!/usr/bin/env python3
"""replay_shadow.py — Mode A shadow comparison.

Reads N games from an arbiter-produced PGN, converts each SAN move
list to UCI (using GPU-derived legal moves at every ply for
disambiguation), then sends the UCI list to BOTH:

  - bridges/arbiter_referee.mjs  (CPU referee, ground truth — imports
                                 match-processor/src/chess-engine.js
                                 read-only)
  - bridges/cuda_referee.py      (GPU referee, dojo_ref via librefcuda.so)

Each replays the same UCI list and reports legality + terminal state.
We compare the two records: outcome agreement, header-result match,
per-game wall-clock.

Usage:
    python3 replay_shadow.py \\
        --pgn /srv/models-hdd/chess-games/games.pgn \\
        --n 500 --max-scan 5000 --seed 42 \\
        --out results/shadow_<ts>.json

NO CPU CHESS COMPUTE on the cuda side: dojo_ref is GPU-backed.
The arbiter side IS the CPU ground truth — we are comparing against
it, not depending on it for the cuda result.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from pgn_stream import stream_pgn, reservoir_sample, game_san_to_uci  # noqa: E402


# --------------------------------------------------------------------------
# Long-running subprocess workers, one per referee. Line-delimited JSON.
# --------------------------------------------------------------------------

class RefereeWorker:
    def __init__(self, name: str, cmd: list[str], cwd: Path | None = None,
                 env: dict | None = None):
        self.name = name
        self.proc = subprocess.Popen(
            cmd, cwd=str(cwd) if cwd else None, env=env or os.environ.copy(),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1,
        )
        # Wait for "ready" banner on stderr.
        for _ in range(50):
            line = self.proc.stderr.readline()
            if not line:
                break
            print(f"  [{name}] {line.rstrip()}", file=sys.stderr)
            if "ready" in line.lower():
                break

    def query(self, payload: dict) -> dict:
        self.proc.stdin.write(json.dumps(payload) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError(f"{self.name} closed stdout unexpectedly")
        return json.loads(line)

    def close(self):
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


# --------------------------------------------------------------------------
# Result-extraction helpers.
# --------------------------------------------------------------------------

PGN_RESULT_TO_WINNER = {
    "1-0": "white", "0-1": "black", "1/2-1/2": "draw", "*": None,
}


def _winner_to_result(w: str | None) -> str:
    if w == "white": return "1-0"
    if w == "black": return "0-1"
    if w == "draw":  return "1/2-1/2"
    return "*"


def _aggregate(report: list[dict]) -> dict:
    n = len(report)
    arb_ok = sum(1 for r in report if r["arbiter"]["ok"])
    cuda_ok = sum(1 for r in report if r["cuda"]["ok"])
    both_ok = sum(1 for r in report if r["arbiter"]["ok"] and r["cuda"]["ok"])
    neither = sum(1 for r in report if not r["arbiter"]["ok"] and not r["cuda"]["ok"])
    arb_only = arb_ok - both_ok
    cuda_only = cuda_ok - both_ok
    legality_agreement = (both_ok + neither) / n if n else 0.0

    # The headline "referee parity" metric: do both referees produce the
    # SAME (ok, terminal, winner_after_terminal) triple on every game?
    # Anything else is a referee-level divergence and a ship-blocker.
    referee_parity_n = sum(
        1 for r in report
        if r["arbiter"]["ok"] == r["cuda"]["ok"]
        and r["arbiter"]["terminal"] == r["cuda"]["terminal"]
        and r["arbiter"]["winner_after_terminal"]
            == r["cuda"]["winner_after_terminal"]
    )

    # Of games both referees accepted, do their reconstructed winners
    # agree, AND do they match the recorded PGN Result?
    # Note: the PGN Result can disagree with both referees when the
    # game ended for a non-replayable reason (fighter timeout, OOM,
    # or a draw rule that depends on en-passant which our replay
    # history hash approximates). Such games are NOT referee
    # divergences — both bridges agree the position is "undecided".
    term_match = 0
    header_match = 0
    same_terminal = 0
    arb_terms: Counter = Counter()
    cuda_terms: Counter = Counter()
    pgn_results: Counter = Counter()
    pgn_terms: Counter = Counter()
    nonreplayable_terms = {"timeout", "crash", "oom", "invalid_format",
                           "illegal", "max_plies", "error",
                           "threefold", "50-move", "insufficient"}
    replayable_match = 0
    replayable_total = 0
    for r in report:
        a, c = r["arbiter"], r["cuda"]
        if a["ok"]:
            arb_terms[a["terminal"] or "—"] += 1
        if c["ok"]:
            cuda_terms[c["terminal"] or "—"] += 1
        pgn_results[r["pgn_result"]] += 1
        pgn_terms[r["pgn_termination"] or "—"] += 1
        if a["ok"] and c["ok"]:
            same_t = (a["terminal"] == c["terminal"])
            same_w = (a["winner_after_terminal"] == c["winner_after_terminal"])
            if same_t and same_w:
                term_match += 1
            if same_t:
                same_terminal += 1
            arb_w = _winner_to_result(a["winner_after_terminal"])
            if arb_w == r["pgn_result"]:
                header_match += 1
            # Replayable subset: PGN's termination is one the referee
            # is supposed to detect from move list alone (mate / stalemate).
            pgn_t = (r["pgn_termination"] or "").strip()
            if pgn_t in ("checkmate", "stalemate"):
                replayable_total += 1
                if arb_w == r["pgn_result"] and a["terminal"] == pgn_t:
                    replayable_match += 1

    arb_replay_total = sum(r["timing_ms"]["arbiter"] for r in report)
    cuda_replay_total = sum(r["timing_ms"]["cuda"] for r in report)
    san_to_uci_total = sum(r["timing_ms"]["san_to_uci"] for r in report)

    def sigma(p, n):
        return math.sqrt(p * (1 - p) / n) if n else 0.0

    return {
        "n": n,
        "arbiter_accepted": arb_ok,
        "cuda_accepted": cuda_ok,
        "both_accepted": both_ok,
        "arbiter_only": arb_only,
        "cuda_only": cuda_only,
        "neither": neither,
        "legality_agreement": legality_agreement,
        "legality_agreement_sigma": sigma(legality_agreement, n),
        "arbiter_acceptance_rate": arb_ok / n if n else 0.0,
        "cuda_acceptance_rate": cuda_ok / n if n else 0.0,
        "referee_parity": referee_parity_n / n if n else 0.0,
        "referee_parity_sigma": sigma(referee_parity_n / n, n),
        "referee_parity_count": referee_parity_n,
        "outcome_match_rate":
            (term_match / both_ok) if both_ok else 0.0,
        "outcome_match_sigma":
            sigma(term_match / both_ok, both_ok) if both_ok else 0.0,
        "same_terminal_kind_rate":
            (same_terminal / both_ok) if both_ok else 0.0,
        "header_result_match_rate":
            (header_match / both_ok) if both_ok else 0.0,
        "replayable_subset_total": replayable_total,
        "replayable_subset_match": replayable_match,
        "replayable_subset_match_rate":
            (replayable_match / replayable_total) if replayable_total else 0.0,
        "arbiter_terminals": dict(arb_terms),
        "cuda_terminals": dict(cuda_terms),
        "pgn_results": dict(pgn_results),
        "pgn_terminations": dict(pgn_terms),
        "arbiter_replay_total_ms": arb_replay_total,
        "cuda_replay_total_ms": cuda_replay_total,
        "san_to_uci_total_ms": san_to_uci_total,
        "arbiter_ms_per_game": arb_replay_total / n if n else 0.0,
        "cuda_ms_per_game": cuda_replay_total / n if n else 0.0,
        "san_to_uci_ms_per_game": san_to_uci_total / n if n else 0.0,
        "ratio_cuda_over_arbiter":
            (cuda_replay_total / arb_replay_total) if arb_replay_total else 0.0,
        "arbiter_games_per_hour":
            (3600_000.0 / (arb_replay_total / n)) if arb_replay_total and n else 0.0,
        "cuda_games_per_hour":
            (3600_000.0 / (cuda_replay_total / n)) if cuda_replay_total and n else 0.0,
    }


# --------------------------------------------------------------------------
# Main.
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pgn",
        default="/srv/models-hdd/chess-games/games.pgn")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--max-scan", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None,
        help="JSON output path. Default: results/shadow_<ts>.json")
    ap.add_argument("--arbiter-src",
        default="/home/frosty40/AgentChess_Server/match-processor/src",
        help="Arbiter source dir (read-only import target)")
    ap.add_argument("--max-plies", type=int, default=500)
    ap.add_argument("--show-divergences", type=int, default=10)
    ap.add_argument("--quick-fail-skip", action="store_true",
        help="If SAN→UCI conversion fails (PGN parsing, not a referee "
             "issue), drop the game and continue rather than counting "
             "it as a referee divergence.")
    args = ap.parse_args()

    pgn_path = Path(args.pgn)
    if not pgn_path.exists():
        print(f"ERROR: PGN not found: {pgn_path}", file=sys.stderr)
        sys.exit(2)

    out_path = Path(args.out) if args.out else \
        ROOT / "results" / f"shadow_{int(time.time())}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"=== cuda combat shadow comparison (Mode A: replay) ===")
    print(f"PGN:           {pgn_path}")
    print(f"N target:      {args.n}  (reservoir, max_scan={args.max_scan})")
    print(f"Arbiter src:   {args.arbiter_src}")
    print(f"Output:        {out_path}")
    print()

    # 1) Sample N games.
    t0 = time.perf_counter()
    games, total_seen = reservoir_sample(
        stream_pgn(pgn_path), k=args.n, seed=args.seed,
        max_scan=args.max_scan,
    )
    print(f"[stream] scanned {total_seen} games, sampled {len(games)} "
          f"in {time.perf_counter()-t0:.2f}s")

    # 2) Convert SAN → UCI per game (uses GPU for legal-move enumeration).
    san_to_uci_games: list[tuple[dict, list[str], int, str]] = []
    skipped: list[tuple[int, dict, int, str]] = []
    t0 = time.perf_counter()
    for i, g in enumerate(games):
        ucis, fail_ply, err = game_san_to_uci(g["moves"])
        if fail_ply >= 0:
            skipped.append((i, g, fail_ply, err))
            if args.quick_fail_skip:
                continue
            # Still feed the partial list; both referees will reject the
            # bad ply consistently if the issue is in the PGN.
        san_to_uci_games.append((g, ucis, fail_ply, err))
    san_to_uci_ms_total = (time.perf_counter() - t0) * 1000.0
    print(f"[san2uci] {len(san_to_uci_games)} games converted "
          f"({len(skipped)} had partial failures) in {san_to_uci_ms_total:.0f}ms")
    if skipped:
        print(f"          first skip example: game #{skipped[0][0]} "
              f"ply {skipped[0][2]}: {skipped[0][3]}")

    # 3) Spin up the two referee workers.
    arb_env = os.environ.copy()
    arb_env["ARBITER_SRC"] = args.arbiter_src
    print()
    print("[bridge] starting arbiter_referee.mjs ...")
    arbiter = RefereeWorker(
        "arbiter",
        ["node", str(ROOT / "bridges" / "arbiter_referee.mjs")],
        env=arb_env,
    )
    print("[bridge] starting cuda_referee.py ...")
    cuda = RefereeWorker(
        "cuda",
        [sys.executable, str(ROOT / "bridges" / "cuda_referee.py")],
        env=os.environ.copy(),
    )
    print()

    # 4) Replay each game through both, time it.
    report: list[dict] = []
    for i, (g, ucis, fail_ply, err) in enumerate(san_to_uci_games):
        san_to_uci_ms = 0.0
        # arbiter
        t0 = time.perf_counter()
        a_resp = arbiter.query({"id": i, "moves": ucis,
                                "max_plies": args.max_plies})
        a_ms = (time.perf_counter() - t0) * 1000.0
        # cuda
        t0 = time.perf_counter()
        c_resp = cuda.query({"id": i, "moves": ucis,
                             "max_plies": args.max_plies})
        c_ms = (time.perf_counter() - t0) * 1000.0

        rec = {
            "idx": i,
            "white": g["headers"].get("White", "?"),
            "black": g["headers"].get("Black", "?"),
            "pgn_result": g["headers"].get("Result", "*"),
            "pgn_termination": g["headers"].get("Termination", ""),
            "san_plies": len(g["moves"]),
            "uci_plies": len(ucis),
            "san_to_uci_failed_at_ply": fail_ply,
            "san_to_uci_error": err,
            "arbiter": a_resp,
            "cuda":    c_resp,
            "timing_ms": {
                "arbiter": a_ms,
                "cuda": c_ms,
                "san_to_uci": san_to_uci_ms,
            },
        }
        report.append(rec)
        if (i + 1) % 50 == 0 or i + 1 == len(san_to_uci_games):
            ok_a = "Y" if a_resp["ok"] else "N"
            ok_c = "Y" if c_resp["ok"] else "N"
            same = "==" if a_resp.get("terminal") == c_resp.get("terminal") else "!="
            print(f"  [{i+1:4d}/{len(san_to_uci_games)}] "
                  f"a={ok_a}/{a_resp.get('terminal','—'):>11} "
                  f"c={ok_c}/{c_resp.get('terminal','—'):>11} {same} "
                  f"({a_ms:5.1f}ms/{c_ms:5.1f}ms)")

    # 5) Close workers, aggregate.
    arbiter.close()
    cuda.close()

    # Rebalance: san_to_uci_ms is per-batch, distribute uniformly.
    if san_to_uci_games:
        per = san_to_uci_ms_total / len(san_to_uci_games)
        for r in report:
            r["timing_ms"]["san_to_uci"] = per

    agg = _aggregate(report)
    agg["pgn"] = str(pgn_path)
    agg["seed"] = args.seed
    agg["max_scan"] = args.max_scan
    agg["total_games_in_pgn_sample_window"] = total_seen
    agg["skipped_san_failures"] = len(skipped)

    # 6) Print human summary.
    print()
    print("=" * 60)
    print("  SHADOW COMPARISON REPORT")
    print("=" * 60)
    print(f"  N games:            {agg['n']}")
    print()
    print(f"  *** REFEREE PARITY: {agg['referee_parity_count']}/{agg['n']} = "
          f"{agg['referee_parity']:.4f} (σ={agg['referee_parity_sigma']:.4f}) ***")
    print(f"      (both bridges produced identical (ok, terminal, winner)"
          f" triples)")
    print()
    print(f"  arbiter accepted:   {agg['arbiter_accepted']}/{agg['n']} "
          f"= {agg['arbiter_acceptance_rate']:.4f}")
    print(f"  cuda accepted:      {agg['cuda_accepted']}/{agg['n']} "
          f"= {agg['cuda_acceptance_rate']:.4f}")
    print(f"  both accepted:      {agg['both_accepted']}")
    print(f"  arbiter-only:       {agg['arbiter_only']}")
    print(f"  cuda-only:          {agg['cuda_only']}")
    print(f"  legality agreement: {agg['legality_agreement']:.4f} "
          f"(σ={agg['legality_agreement_sigma']:.4f})")
    print()
    print(f"  --- of {agg['both_accepted']} games both accepted ---")
    print(f"  same terminal kind:    {agg['same_terminal_kind_rate']:.4f}")
    print(f"  outcome match:         {agg['outcome_match_rate']:.4f} "
          f"(σ={agg['outcome_match_sigma']:.4f})")
    print(f"  header-result match:   {agg['header_result_match_rate']:.4f}")
    print()
    print(f"  --- replayable subset (PGN ended in mate/stalemate) ---")
    print(f"  total:    {agg['replayable_subset_total']}")
    print(f"  matched:  {agg['replayable_subset_match']} "
          f"({agg['replayable_subset_match_rate']:.4f})")
    print(f"      (excludes timeout/crash/threefold-draws — non-replayable")
    print(f"       game endings that depend on data outside the move list)")
    print()
    print("  --- terminal histogram ---")
    print(f"  arbiter: {agg['arbiter_terminals']}")
    print(f"  cuda:    {agg['cuda_terminals']}")
    print(f"  pgn Res: {agg['pgn_results']}")
    print()
    print("  --- timing (referee replay only) ---")
    print(f"  arbiter:  {agg['arbiter_replay_total_ms']:>10.1f} ms total "
          f"({agg['arbiter_ms_per_game']:.2f} ms/game, "
          f"{agg['arbiter_games_per_hour']:>9.0f} games/hr)")
    print(f"  cuda:     {agg['cuda_replay_total_ms']:>10.1f} ms total "
          f"({agg['cuda_ms_per_game']:.2f} ms/game, "
          f"{agg['cuda_games_per_hour']:>9.0f} games/hr)")
    print(f"  ratio:    cuda is {agg['ratio_cuda_over_arbiter']:.2f}x "
          f"the arbiter's per-game replay time")
    print(f"  san→uci:  {agg['san_to_uci_total_ms']:.1f} ms total "
          f"({agg['san_to_uci_ms_per_game']:.2f} ms/game)")

    # 7) Show divergences.
    div = [r for r in report
           if r["arbiter"]["ok"] != r["cuda"]["ok"]
           or (r["arbiter"]["ok"] and r["cuda"]["ok"]
               and r["arbiter"]["terminal"] != r["cuda"]["terminal"])]
    print()
    print(f"  --- {len(div)} divergent game(s) ---")
    for r in div[:args.show_divergences]:
        print(f"  game #{r['idx']:4d}: {r['white']:<18} vs {r['black']:<18} "
              f"plies={r['uci_plies']}")
        print(f"      arbiter ok={r['arbiter']['ok']} "
              f"term={r['arbiter']['terminal']!r} "
              f"winner={r['arbiter']['winner_after_terminal']!r} "
              f"err={r['arbiter']['error']!r}")
        print(f"      cuda    ok={r['cuda']['ok']} "
              f"term={r['cuda']['terminal']!r} "
              f"winner={r['cuda']['winner_after_terminal']!r} "
              f"err={r['cuda']['error']!r}")
        print(f"      pgn:    Result={r['pgn_result']} "
              f"Termination={r['pgn_termination']}")

    # 8) Persist full report.
    with open(out_path, "w") as f:
        json.dump({"summary": agg, "per_game": report}, f, indent=2)
    print()
    print(f"  full per-game report → {out_path}")


if __name__ == "__main__":
    main()
