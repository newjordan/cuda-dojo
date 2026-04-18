#!/usr/bin/env python3
"""
kernel_pressure_ablate.py

Bench the custom recursive chess kernel under varying launch and stack settings.
Current scope: fused_qsearch recursion pressure.
"""

import argparse
import datetime as dt
import json
import os
import statistics
import subprocess
import sys


CUDA_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ARTIFACT_ROOT = os.path.join(CUDA_DIR, "artifacts")
FUSED_QSEARCH_BIN = os.path.join(CUDA_DIR, "fused_qsearch")

FEN_CASES = [
    {
        "label": "opening_quiet",
        "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    },
    {
        "label": "middlegame_mixed",
        "fen": "r1bq1rk1/pp1nbppp/2p1pn2/2Pp4/3P4/2N1PN2/PPQ1BPPP/R1B2RK1 w - - 0 10",
    },
    {
        "label": "castled_tension",
        "fen": "r3k2r/ppp2ppp/2n5/3qp3/3P4/2P2N2/PP2QPPP/R3K2R w KQkq - 0 1",
    },
    {
        "label": "free_queen_capture",
        "fen": "rnbqkbnr/ppp2ppp/8/4p3/3Q4/8/PPP1PPPP/RNB1KBNR b KQkq - 0 3",
    },
    {
        "label": "pressure_capture_dense_a",
        "fen": "3q1rk1/pp3p2/2n1p1pp/P1Qn2B1/3bp3/1bN4P/1Pr2PP1/1R2K1NR b K - 1 19",
    },
    {
        "label": "pressure_capture_dense_b",
        "fen": "rn1r4/pR3pkp/2p1bnp1/2PpN3/3P4/q1NBPQ1P/P1PB1PP1/5RK1 w - - 1 16",
    },
    {
        "label": "pressure_capture_dense_c",
        "fen": "N6r/p4k2/b7/3PP2p/PbP2P1q/1P2n1P1/2n4P/R1KQ1B1R b - - 4 21",
    },
]


def parse_csv_ints(text, default):
    if not text:
        return default
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--depths", default="4,6,8")
    ap.add_argument("--block-sizes", default="32,64,128")
    ap.add_argument("--stack-kb", default="16,24,32,48")
    ap.add_argument("--bench", type=int, default=4096)
    ap.add_argument("--bench-sizes", default="")
    ap.add_argument("--warmup-runs", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--artifact-dir", default="")
    return ap.parse_args()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def parse_score_vector(stdout):
    scores = []
    for line in (stdout or "").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            scores.append(int(line))
        except ValueError:
            pass
    return scores


def aggregate_numeric(values):
    if not values:
        return 0.0
    return statistics.median(values)


def run_case(depth, block_size, stack_kb, bench_n, warmup_runs, repeats):
    cmd = [
        FUSED_QSEARCH_BIN,
        "--depth", str(depth),
        "--bench", str(bench_n),
        "--block-size", str(block_size),
        "--stack-kb", str(stack_kb),
        "--warmup-runs", str(warmup_runs),
        "--metrics-json",
        "--plain",
    ]
    stdin_payload = "\n".join(case["fen"] for case in FEN_CASES) + "\n"
    result = {
        "depth": depth,
        "bench": bench_n,
        "block_size": block_size,
        "stack_kb": stack_kb,
        "warmup_runs": warmup_runs,
        "repeats": repeats,
    }

    runs = []
    for _ in range(repeats):
        proc = subprocess.run(
            cmd,
            cwd=CUDA_DIR,
            input=stdin_payload,
            text=True,
            capture_output=True,
        )
        run = {
            "returncode": proc.returncode,
            "stderr_tail": (proc.stderr or "").strip().splitlines()[-5:],
            "score_vector": parse_score_vector(proc.stdout),
        }
        if proc.returncode == 0:
            metrics = None
            for line in reversed((proc.stderr or "").splitlines()):
                line = line.strip()
                if line.startswith("{") and line.endswith("}"):
                    try:
                        metrics = json.loads(line)
                        break
                    except json.JSONDecodeError:
                        pass
            if metrics is None:
                run["status"] = "failed"
                run["error"] = "missing_metrics_json"
            else:
                run.update(metrics)
                run["status"] = "ok"
        else:
            run["status"] = "failed"
        runs.append(run)

    result["runs"] = runs
    ok_runs = [run for run in runs if run.get("status") == "ok"]
    if not ok_runs:
        result["status"] = "failed"
        return result

    numeric_fields = [
        "kernel_ms",
        "positions_per_sec",
        "nodes_per_sec",
        "total_nodes",
        "min_nodes_per_position",
        "mean_nodes_per_position",
        "p95_nodes_per_position",
        "max_nodes_per_position",
        "max_ply_reached",
        "mean_max_ply_per_position",
        "depth_cap_hits",
        "standpat_beta_cutoffs",
        "standpat_delta_prunes",
        "move_delta_prunes",
        "beta_cutoffs_after_search",
        "illegal_move_rejects",
        "total_capture_moves_generated",
        "mean_capture_moves_per_node",
        "max_capture_moves_in_node",
        "stack_bytes_requested",
        "stack_bytes_effective",
        "num_regs",
        "local_size_bytes",
        "shared_size_bytes",
        "max_threads_per_block",
        "occupancy_estimate",
    ]
    for field in numeric_fields:
        result[field] = aggregate_numeric([run[field] for run in ok_runs])

    score_vectors = [tuple(run.get("score_vector", [])) for run in ok_runs]
    stable_score_vector = score_vectors[0] if score_vectors else ()
    result["score_vector"] = list(stable_score_vector)
    result["repeat_score_vector_stable"] = len(set(score_vectors)) <= 1
    result["repeat_score_vector_variants"] = [list(vector) for vector in sorted(set(score_vectors))]
    result["returncode"] = 0
    result["stderr_tail"] = ok_runs[-1].get("stderr_tail", [])
    result["status"] = "ok"
    return result


def annotate_reference_deltas(grouped_cases):
    for group in grouped_cases.values():
        if not group:
            continue
        reference = sorted(
            group,
            key=lambda item: (-item["stack_kb"], item["block_size"], item["kernel_ms"]),
        )[0]
        ref_scores = reference.get("score_vector", [])
        for item in group:
            mismatch_count = 0
            max_abs_delta = 0
            for lhs, rhs in zip(item.get("score_vector", []), ref_scores):
                if lhs != rhs:
                    mismatch_count += 1
                max_abs_delta = max(max_abs_delta, abs(lhs - rhs))
            item["reference_case"] = {
                "bench": reference["bench"],
                "block_size": reference["block_size"],
                "stack_kb": reference["stack_kb"],
            }
            item["score_mismatch_count"] = mismatch_count
            item["max_abs_score_delta"] = max_abs_delta


def main():
    args = parse_args()
    depths = parse_csv_ints(args.depths, [4, 6, 8])
    blocks = parse_csv_ints(args.block_sizes, [32, 64, 128])
    stacks = parse_csv_ints(args.stack_kb, [16, 24, 32, 48])
    benches = parse_csv_ints(args.bench_sizes, [args.bench])

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    artifact_dir = args.artifact_dir or os.path.join(DEFAULT_ARTIFACT_ROOT, f"kernel_pressure_ablate_{stamp}")
    ensure_dir(artifact_dir)

    if not args.skip_build:
        subprocess.run(["make", "-B", "fused_qsearch"], cwd=CUDA_DIR, check=True)

    results = []
    by_depth = {}
    by_depth_bench = {}
    grouped_cases = {}
    for depth in depths:
        depth_results = []
        for bench_n in benches:
            bucket_key = f"{depth}@{bench_n}"
            bucket = []
            for block_size in blocks:
                for stack_kb in stacks:
                    case = run_case(
                        depth,
                        block_size,
                        stack_kb,
                        bench_n,
                        args.warmup_runs,
                        args.repeats,
                    )
                    results.append(case)
                    if case.get("status") == "ok":
                        depth_results.append(case)
                        bucket.append(case)
            grouped_cases[bucket_key] = bucket
            by_depth_bench[bucket_key] = {
                "ranked": sorted(bucket, key=lambda item: item["nodes_per_sec"], reverse=True),
                "winner": None,
            }
            if by_depth_bench[bucket_key]["ranked"]:
                by_depth_bench[bucket_key]["winner"] = by_depth_bench[bucket_key]["ranked"][0]
        by_depth[str(depth)] = {
            "ranked": sorted(depth_results, key=lambda item: item["nodes_per_sec"], reverse=True),
            "winner": None,
        }
        if by_depth[str(depth)]["ranked"]:
            by_depth[str(depth)]["winner"] = by_depth[str(depth)]["ranked"][0]

    annotate_reference_deltas(grouped_cases)

    ranked_all = sorted(
        (item for item in results if item.get("status") == "ok"),
        key=lambda item: (item["depth"], item["bench"], item["nodes_per_sec"]),
        reverse=True,
    )

    report = {
        "artifact_dir": artifact_dir,
        "bench_sizes": benches,
        "fen_cases": FEN_CASES,
        "results": results,
        "by_depth": by_depth,
        "by_depth_bench": by_depth_bench,
        "ranked_all": ranked_all,
    }
    out_path = os.path.join(artifact_dir, "kernel_pressure_report.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
    print(json.dumps({
        "artifact_dir": artifact_dir,
        "report_path": out_path,
        "depth_winners": {
            depth: info["winner"] for depth, info in by_depth_bench.items()
        },
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
