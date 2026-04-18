#!/usr/bin/env python3
"""
build_training_corpus.py

Builds a unified dojo training corpus with explicit target families and metadata:
- recall targets from book / engine scores
- instinct targets from self-play outcomes
- bridge / gradient targets that blend recall and instinct under pressure
- pressure bands, phase buckets, and relation/storage buckets for access-policy ablations
"""

import argparse
import json
import os
from collections import Counter


BOOK_DEFAULT = os.path.join(os.path.dirname(__file__), "..", "gpu_spine", "book.jsonl")
SELF_PLAY_DEFAULT = os.path.join(os.path.dirname(__file__), "self_play_data.jsonl")

PIECE_ORDER = "PNBRQKpnbrqk"


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--book", default=BOOK_DEFAULT)
    ap.add_argument("--self-play", dest="self_play", default=SELF_PLAY_DEFAULT)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-book", type=int, default=0)
    ap.add_argument("--max-self-play", type=int, default=0)
    ap.add_argument("--sigma-weighting", choices=["off", "soft", "hard"], default="off")
    ap.add_argument("--sigma-max-cp", type=float, default=400.0)
    ap.add_argument("--budget-opening-share", type=float, default=0.55)
    ap.add_argument("--budget-middlegame-share", type=float, default=0.30)
    ap.add_argument("--budget-endgame-share", type=float, default=0.15)
    return ap.parse_args()


def load_jsonl(path, limit=0):
    rows = []
    if not path or not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def piece_counts(fen):
    board = fen.split(" ", 1)[0]
    counts = Counter()
    for ch in board:
        if ch in PIECE_ORDER:
            counts[ch] += 1
    return counts


def parse_fullmove(fen):
    parts = fen.split()
    if len(parts) >= 6:
        try:
            return int(parts[5])
        except ValueError:
            return 1
    return 1


def phase_bucket(fen):
    counts = piece_counts(fen)
    total_pieces = sum(counts.values())
    queens = counts["Q"] + counts["q"]
    non_pawn_material = (
        9 * queens
        + 5 * (counts["R"] + counts["r"])
        + 3 * (counts["B"] + counts["b"] + counts["N"] + counts["n"])
    )
    fullmove = parse_fullmove(fen)
    if fullmove <= 10 and total_pieces >= 24:
        return "opening"
    if total_pieces <= 10 or (queens == 0 and non_pawn_material <= 18):
        return "endgame"
    return "middlegame"


def material_signature(fen):
    counts = piece_counts(fen)
    side = fen.split()[1] if len(fen.split()) > 1 else "w"
    return (
        f"stm:{side}:"
        f"P{counts['P']}N{counts['N']}B{counts['B']}R{counts['R']}Q{counts['Q']}K{counts['K']}:"
        f"p{counts['p']}n{counts['n']}b{counts['b']}r{counts['r']}q{counts['q']}k{counts['k']}"
    )


def pressure_features(recall_cp, instinct_cp):
    gap = abs(recall_cp - instinct_cp)
    sign_flip = 1.0 if (recall_cp > 0 > instinct_cp) or (recall_cp < 0 < instinct_cp) else 0.0
    magnitude = min(abs(recall_cp), 500) / 500.0
    score = min(1.0, gap / 650.0 + 0.35 * sign_flip + 0.15 * magnitude)
    if score < 0.20:
        band = "cool"
    elif score < 0.45:
        band = "warm"
    elif score < 0.75:
        band = "hot"
    else:
        band = "fracture"
    return score, band


def bridge_target(recall_cp, instinct_cp, band):
    alpha_by_band = {
        "cool": 0.80,
        "warm": 0.60,
        "hot": 0.35,
        "fracture": 0.15,
    }
    alpha = alpha_by_band.get(band, 0.60)
    return int(round(alpha * recall_cp + (1.0 - alpha) * instinct_cp))


def gradient_target(recall_cp, instinct_cp, pressure_score):
    alpha = max(0.15, min(0.85, 0.85 - 0.65 * pressure_score))
    return int(round(alpha * recall_cp + (1.0 - alpha) * instinct_cp))


def pressure_weight(band, pressure_score):
    base = {
        "library": 0.75,
        "cool": 0.95,
        "warm": 1.15,
        "hot": 1.45,
        "fracture": 1.75,
    }.get(band, 1.0)
    return round(base + 0.25 * pressure_score, 4)


def attach_optional_oracle_fields(source_row, built_row):
    for key, value in source_row.items():
        if key.startswith("oracle_") or key.startswith("sigma_"):
            built_row[key] = value
    return built_row


def sigma_weight(row, mode, sigma_max_cp):
    if mode == "off":
        return 1.0
    abs_delta = float(row.get("sigma_ref_vs_sf_abs_score_delta_cp") or 0.0)
    move_miss = 1.0 if row.get("sigma_ref_vs_sf_move_match") is False else 0.0
    sign_miss = 1.0 if row.get("sigma_ref_vs_sf_sign_match") is False else 0.0
    oracle_split = 1.0 if row.get("sigma_lozza_vs_sf_move_match") is False else 0.0
    severity = min(abs_delta / max(1.0, sigma_max_cp), 1.0)
    if mode == "soft":
        return round(1.0 + 0.55 * severity + 0.35 * move_miss + 0.20 * sign_miss + 0.15 * oracle_split, 4)
    return round(1.0 + 0.90 * severity + 0.55 * move_miss + 0.35 * sign_miss + 0.25 * oracle_split, 4)


def phase_budget_hint(phase, args):
    if phase == "opening":
        return "opening_midgame", round(args.budget_opening_share, 4)
    if phase == "middlegame":
        return "middlegame_core", round(args.budget_middlegame_share, 4)
    return "endgame_tail", round(args.budget_endgame_share, 4)


def finalize_row(source_row, built_row, args):
    built = attach_optional_oracle_fields(source_row, built_row)
    sigma = sigma_weight(source_row, args.sigma_weighting, args.sigma_max_cp)
    built["sample_weight_sigma"] = sigma
    built["sample_weight_sigma_pressure"] = round(built["sample_weight_pressure"] * sigma, 4)
    built["sample_weight_sigma_uniform"] = round(built["sample_weight_uniform"] * sigma, 4)
    region, share = phase_budget_hint(built["phase_bucket"], args)
    built["budget_region"] = region
    built["budget_share_hint"] = share
    return built


def make_book_row(row, args):
    fen = row["fen"]
    recall = int(row["sf_score_cp"])
    phase = phase_bucket(fen)
    material = material_signature(fen)
    built = {
        "fen": fen,
        "source": "book",
        "phase_bucket": phase,
        "pressure_score": 0.0,
        "pressure_band": "library",
        "relation_bucket": f"book:{phase}:{material}",
        "storage_bucket": f"{phase}:{material}",
        "target_recall_cp": recall,
        "target_bridge_cp": recall,
        "target_gradient_cp": recall,
        "sample_weight_uniform": 1.0,
        "sample_weight_pressure": pressure_weight("library", 0.0),
    }
    return finalize_row(row, built, args)


def make_self_play_row(row, args):
    fen = row["fen"]
    recall = int(row.get("engine_score_cp", 0))
    instinct = int(row["game_value_cp"])
    phase = phase_bucket(fen)
    pressure_score, band = pressure_features(recall, instinct)
    game_id = int(row.get("game_id", 0))
    ply = int(row.get("ply", 0))
    window = ply // 8
    built = {
        "fen": fen,
        "source": "self_play",
        "phase_bucket": phase,
        "pressure_score": round(pressure_score, 4),
        "pressure_band": band,
        "relation_bucket": f"self_play:g{game_id}:w{window}:{phase}",
        "storage_bucket": f"{phase}:{material_signature(fen)}",
        "target_recall_cp": recall,
        "target_instinct_cp": instinct,
        "target_bridge_cp": bridge_target(recall, instinct, band),
        "target_gradient_cp": gradient_target(recall, instinct, pressure_score),
        "sample_weight_uniform": 1.0,
        "sample_weight_pressure": pressure_weight(band, pressure_score),
    }
    return finalize_row(row, built, args)


def main():
    args = parse_args()
    book_rows = load_jsonl(args.book, args.max_book)
    self_rows = load_jsonl(args.self_play, args.max_self_play)

    out_rows = []
    source_counts = Counter()
    band_counts = Counter()
    phase_counts = Counter()

    for row in book_rows:
        built = make_book_row(row, args)
        out_rows.append(built)
        source_counts[built["source"]] += 1
        band_counts[built["pressure_band"]] += 1
        phase_counts[built["phase_bucket"]] += 1

    for row in self_rows:
        built = make_self_play_row(row, args)
        out_rows.append(built)
        source_counts[built["source"]] += 1
        band_counts[built["pressure_band"]] += 1
        phase_counts[built["phase_bucket"]] += 1

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        for row in out_rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")

    summary = {
        "output": args.output,
        "rows": len(out_rows),
        "source_counts": dict(source_counts),
        "pressure_band_counts": dict(band_counts),
        "phase_counts": dict(phase_counts),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
