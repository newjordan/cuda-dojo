#!/usr/bin/env python3
"""
training_arch_ablate.py

Builds a unified dojo training corpus, runs a small set of training-architecture
ablations through nn_train, and writes a ranked JSON report.
"""

import argparse
import datetime as dt
import json
import os
import subprocess
import sys


CUDA_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ARTIFACT_ROOT = os.path.join(CUDA_DIR, "artifacts")


VARIANTS = [
    {
        "name": "recall_library_shuffle",
        "source_filter": {"book"},
        "target_key": "target_recall_cp",
        "weight_key": "sample_weight_uniform",
        "access_mode": "shuffle",
        "band_weights": "fallback=1.0,library=1.0,cool=1.0,warm=1.0,hot=1.0,fracture=1.0",
    },
    {
        "name": "instinct_selfplay_pressure",
        "source_filter": {"self_play"},
        "target_key": "target_instinct_cp",
        "weight_key": "sample_weight_pressure",
        "access_mode": "pressure_band",
        "band_weights": "fallback=1.0,library=0.0,cool=0.8,warm=1.2,hot=1.6,fracture=1.9",
    },
    {
        "name": "bridge_gradient_shuffle",
        "source_filter": None,
        "target_key": "target_gradient_cp",
        "weight_key": "sample_weight_pressure",
        "access_mode": "shuffle",
        "band_weights": "fallback=1.0,library=0.75,cool=0.95,warm=1.15,hot=1.45,fracture=1.75",
    },
    {
        "name": "bridge_gradient_banded_relation",
        "source_filter": None,
        "target_key": "target_gradient_cp",
        "weight_key": "sample_weight_pressure",
        "access_mode": "banded_relation_cluster",
        "band_weights": "fallback=1.0,library=0.55,cool=0.90,warm=1.25,hot=1.65,fracture=1.95",
    },
    {
        "name": "bridge_storage_relation",
        "source_filter": None,
        "target_key": "target_bridge_cp",
        "weight_key": "sample_weight_uniform",
        "access_mode": "relation_cluster",
        "band_weights": "fallback=1.0,library=1.0,cool=1.0,warm=1.0,hot=1.0,fracture=1.0",
    },
]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--max-book", type=int, default=0)
    ap.add_argument("--max-self-play", type=int, default=0)
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--artifact-dir", default="")
    ap.add_argument("--variants", default="", help="comma-separated subset of variant names to run")
    return ap.parse_args()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def run(cmd, cwd, stdout_path=None, stderr_path=None):
    out_fh = open(stdout_path, "w", encoding="utf-8") if stdout_path else open(os.devnull, "w", encoding="utf-8")
    err_fh = open(stderr_path, "w", encoding="utf-8") if stderr_path else open(os.devnull, "w", encoding="utf-8")
    try:
        return subprocess.run(cmd, cwd=cwd, stdout=out_fh, stderr=err_fh, text=True)
    finally:
        out_fh.close()
        err_fh.close()


def build_corpus(args, artifact_dir):
    corpus_path = os.path.join(artifact_dir, "training_corpus.jsonl")
    summary_path = os.path.join(artifact_dir, "training_corpus_summary.json")
    cmd = [
        sys.executable,
        os.path.join(CUDA_DIR, "build_training_corpus.py"),
        "--output", corpus_path,
    ]
    if args.max_book:
        cmd += ["--max-book", str(args.max_book)]
    if args.max_self_play:
        cmd += ["--max-self-play", str(args.max_self_play)]
    result = subprocess.run(cmd, cwd=CUDA_DIR, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"build_training_corpus failed: {result.stderr or result.stdout}")
    summary = json.loads(result.stdout)
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
    return corpus_path, summary


def filter_variant_rows(corpus_path, variant_path, source_filter):
    rows = 0
    with open(corpus_path, "r", encoding="utf-8") as src, open(variant_path, "w", encoding="utf-8") as dst:
        for line in src:
            row = json.loads(line)
            if source_filter and row.get("source") not in source_filter:
                continue
            if row.get("fen") is None:
                continue
            dst.write(json.dumps(row, sort_keys=True) + "\n")
            rows += 1
    return rows


def get_metric(report, group_name, name, fallback=0.0):
    group = report.get(group_name, {})
    entry = group.get(name)
    if not entry:
        return fallback
    return float(entry.get("mae_gain", fallback))


def score_report(report):
    total_gain = float(report["total"]["mae_gain"])
    book_gain = get_metric(report, "by_source", "book")
    self_play_gain = get_metric(report, "by_source", "self_play")
    library_gain = get_metric(report, "by_pressure_band", "library")
    hot_gain = get_metric(report, "by_pressure_band", "hot")
    fracture_gain = get_metric(report, "by_pressure_band", "fracture")
    return (
        total_gain
        + 0.45 * self_play_gain
        + 0.30 * hot_gain
        + 0.20 * fracture_gain
        + 0.15 * book_gain
        + 0.10 * library_gain
    )


def main():
    args = parse_args()
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    artifact_dir = args.artifact_dir or os.path.join(DEFAULT_ARTIFACT_ROOT, f"training_arch_ablate_{stamp}")
    ensure_dir(artifact_dir)

    if not args.skip_build:
        subprocess.run(["bash", os.path.join(CUDA_DIR, "build_nn_train.sh")], cwd=CUDA_DIR, check=True)

    corpus_path, corpus_summary = build_corpus(args, artifact_dir)
    variant_filter = {item.strip() for item in args.variants.split(",") if item.strip()}

    results = []
    for variant in VARIANTS:
        if variant_filter and variant["name"] not in variant_filter:
            continue
        variant_dir = os.path.join(artifact_dir, variant["name"])
        ensure_dir(variant_dir)
        variant_data = os.path.join(variant_dir, "data.jsonl")
        row_count = filter_variant_rows(corpus_path, variant_data, variant["source_filter"])
        if row_count < 100:
            results.append({
                "name": variant["name"],
                "status": "skipped",
                "rows": row_count,
                "reason": "too_few_rows",
            })
            continue

        report_path = os.path.join(variant_dir, "report.json")
        stdout_path = os.path.join(variant_dir, "stdout.log")
        stderr_path = os.path.join(variant_dir, "stderr.log")
        cmd = [
            os.path.join(CUDA_DIR, "nn_train"),
            "--data", variant_data,
            "--epochs", str(args.epochs),
            "--batch", str(args.batch),
            "--lr", str(args.lr),
            "--seed", str(args.seed),
            "--target-key", variant["target_key"],
            "--weight-key", variant["weight_key"],
            "--access-mode", variant["access_mode"],
            "--band-weights", variant["band_weights"],
            "--report-json", report_path,
        ]
        proc = run(cmd, cwd=variant_dir, stdout_path=stdout_path, stderr_path=stderr_path)
        if proc.returncode != 0 or not os.path.exists(report_path):
            results.append({
                "name": variant["name"],
                "status": "failed",
                "rows": row_count,
                "returncode": proc.returncode,
                "stdout_log": stdout_path,
                "stderr_log": stderr_path,
            })
            continue

        with open(report_path, "r", encoding="utf-8") as fh:
            report = json.load(fh)
        composite = score_report(report)
        results.append({
            "name": variant["name"],
            "status": "ok",
            "rows": row_count,
            "target_key": variant["target_key"],
            "weight_key": variant["weight_key"],
            "access_mode": variant["access_mode"],
            "band_weights": variant["band_weights"],
            "report_path": report_path,
            "stdout_log": stdout_path,
            "stderr_log": stderr_path,
            "composite_score": round(composite, 6),
            "total_mae_gain": report["total"]["mae_gain"],
            "book_mae_gain": get_metric(report, "by_source", "book"),
            "self_play_mae_gain": get_metric(report, "by_source", "self_play"),
            "library_mae_gain": get_metric(report, "by_pressure_band", "library"),
            "hot_mae_gain": get_metric(report, "by_pressure_band", "hot"),
            "fracture_mae_gain": get_metric(report, "by_pressure_band", "fracture"),
        })

    ranked = sorted((r for r in results if r.get("status") == "ok"),
                    key=lambda item: item["composite_score"], reverse=True)
    report = {
        "artifact_dir": artifact_dir,
        "corpus_path": corpus_path,
        "corpus_summary": corpus_summary,
        "epochs": args.epochs,
        "batch": args.batch,
        "lr": args.lr,
        "seed": args.seed,
        "results": results,
        "ranked": ranked,
        "winner": ranked[0] if ranked else None,
    }

    out_path = os.path.join(artifact_dir, "training_arch_ablation_report.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
    print(json.dumps({
        "report_path": out_path,
        "winner": ranked[0]["name"] if ranked else None,
        "artifact_dir": artifact_dir,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
