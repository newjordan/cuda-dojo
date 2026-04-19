#!/usr/bin/env python3
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"


def run(cmd, cwd=ROOT, allowed_codes=None):
    if allowed_codes is None:
        allowed_codes = {0}
    proc = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    if proc.returncode not in allowed_codes:
        raise RuntimeError(
            f"command failed: {' '.join(cmd)}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    return proc.stdout


def load_json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def main():
    python_bin = sys.executable
    steps = [
        (["make", "-C", str(ROOT), "false_finish_regression", f"PYTHON_BIN={python_bin}"], {0}),
        ([python_bin, "king_pressure_regression.py"], {0}),
        ([python_bin, "queen_pressure_regression.py"], {0, 1}),
        (["make", "-C", str(ROOT), "family_persistence_matrix", f"PYTHON_BIN={python_bin}"], {0}),
        (["make", "-C", str(ROOT), "oracle_sigma_probe", f"PYTHON_BIN={python_bin}"], {0}),
    ]

    for step, allowed_codes in steps:
        print(f"[stability_gate] running: {' '.join(step)}", flush=True)
        run(step, allowed_codes=allowed_codes)

    false_finish = load_json(ARTIFACTS / "false_finish_regression" / "report.json")
    king = load_json(ARTIFACTS / "king_pressure_regression" / "report.json")
    queen = load_json(ARTIFACTS / "queen_pressure_regression" / "report.json")
    family = load_json(ARTIFACTS / "family_persistence_matrix" / "report.json")
    oracle = load_json(ARTIFACTS / "oracle_sigma_smoke" / "probe_report.json")

    failures = []

    if false_finish.get("matched") != false_finish.get("cases"):
        failures.append(
            f"false_finish_regression regressed: {false_finish.get('matched')}/{false_finish.get('cases')}"
        )

    if king.get("matched") != king.get("cases"):
        failures.append(
            f"king_pressure_regression regressed: {king.get('matched')}/{king.get('cases')}"
        )

    queen_expected_floor = 5
    queen_allowed_misses = {"queen_intrusion_01", "queen_intrusion_03", "queen_intrusion_04"}
    queen_misses = {row["id"] for row in queen.get("mismatches", [])}
    unexpected_queen_misses = sorted(queen_misses - queen_allowed_misses)
    if queen.get("matched", 0) < queen_expected_floor:
        failures.append(
            f"queen_pressure_regression below floor: {queen.get('matched')}/{queen.get('cases')}"
        )
    if unexpected_queen_misses:
        failures.append(
            "queen_pressure_regression has new unexpected misses: "
            + ", ".join(unexpected_queen_misses)
        )

    motif_1000 = family.get("motif_budget_summary", {}).get("1000", {}).get("queen_intrusion", {})
    if motif_1000.get("exact_hit_rate", 0.0) < 0.76:
        failures.append(
            f"family_persistence 1000 queen_intrusion exact_hit_rate regressed: {motif_1000.get('exact_hit_rate')}"
        )
    if motif_1000.get("guard_top3_rate", 0.0) < 0.84:
        failures.append(
            f"family_persistence 1000 queen_intrusion guard_top3_rate regressed: {motif_1000.get('guard_top3_rate')}"
        )

    symmetry_1000 = family.get("symmetry_budget_summary", {}).get("1000", {}).get("right_cold_focused", {})
    if symmetry_1000.get("exact_hit_rate", 0.0) < 0.66:
        failures.append(
            f"family_persistence 1000 right_cold_focused exact_hit_rate regressed: {symmetry_1000.get('exact_hit_rate')}"
        )
    if symmetry_1000.get("guard_top3_rate", 0.0) < 0.83:
        failures.append(
            f"family_persistence 1000 right_cold_focused guard_top3_rate regressed: {symmetry_1000.get('guard_top3_rate')}"
        )

    if oracle.get("mcts_vs_sf_move_match_rate", 0.0) < 0.375:
        failures.append(
            f"oracle_sigma_probe vs Stockfish regressed: {oracle.get('mcts_vs_sf_move_match_rate')}"
        )
    if oracle.get("mcts_vs_reference_move_match_rate", 0.0) < 0.3125:
        failures.append(
            f"oracle_sigma_probe vs reference regressed: {oracle.get('mcts_vs_reference_move_match_rate')}"
        )

    summary = {
        "false_finish_matched": false_finish.get("matched"),
        "false_finish_cases": false_finish.get("cases"),
        "king_matched": king.get("matched"),
        "king_cases": king.get("cases"),
        "queen_matched": queen.get("matched"),
        "queen_cases": queen.get("cases"),
        "queen_mismatches": sorted(queen_misses),
        "family_1000_queen_intrusion_exact_hit_rate": motif_1000.get("exact_hit_rate"),
        "family_1000_queen_intrusion_guard_top3_rate": motif_1000.get("guard_top3_rate"),
        "family_1000_right_cold_focused_exact_hit_rate": symmetry_1000.get("exact_hit_rate"),
        "family_1000_right_cold_focused_guard_top3_rate": symmetry_1000.get("guard_top3_rate"),
        "oracle_mcts_vs_sf_move_match_rate": oracle.get("mcts_vs_sf_move_match_rate"),
        "oracle_mcts_vs_reference_move_match_rate": oracle.get("mcts_vs_reference_move_match_rate"),
        "status": "ok" if not failures else "failed",
        "failures": failures,
    }

    out_dir = ARTIFACTS / "stability_gate"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "report.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
        fh.write("\n")

    print(json.dumps(summary, indent=2, sort_keys=True))
    if failures:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
