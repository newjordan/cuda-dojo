#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import root_shakeout as rs


CUDA_DIR = Path(__file__).resolve().parent
DEFAULT_SEED_REPORT = CUDA_DIR / "root_shakeout_report.json"
REPORT_PATH = CUDA_DIR / "root_region_report.json"


def clip_prior(value: float) -> float:
    return max(0.0, min(1.0, round(value, 4)))


def clip_selection(value: float) -> float:
    return max(0.0, round(value, 6))


def normalize_batch(batch: int) -> int:
    if batch in rs.BATCH_CHOICES:
        return batch
    return min(rs.BATCH_CHOICES, key=lambda item: abs(item - batch))


def config_key(batch: int, prior: float, selection: float) -> tuple[int, float, float]:
    return (normalize_batch(batch), clip_prior(prior), clip_selection(selection))


def make_config(name: str, batch: int, prior: float, selection: float) -> dict:
    batch = normalize_batch(batch)
    prior = clip_prior(prior)
    selection = clip_selection(selection)
    return {
        "name": name,
        "opening_prior_scale": prior,
        "opening_selection_scale": selection,
        "root_batch_size": batch,
        "args": [
            "--seed-mode",
            "time",
            "--opening-prior-scale",
            f"{prior:.4f}",
            "--opening-selection-scale",
            f"{selection:.6f}",
            "--root-batch-size",
            str(batch),
        ],
    }


def load_seed_report(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def is_success(result: dict, min_opening: float) -> bool:
    return (
        result.get("mate_hit_rate", 0.0) >= 1.0
        and result.get("queen_hit_rate", 0.0) >= 1.0
        and result.get("strong_opening_fraction", 0.0) >= min_opening
    )


def choose_centers(report: dict, top_k: int, min_opening: float) -> list[dict]:
    ranked = sorted(report.get("results", []), key=lambda item: item["score"], reverse=True)
    preferred = [item for item in ranked if is_success(item, min_opening)]
    chosen = []
    seen = set()
    for item in preferred + ranked:
        key = config_key(
            item["root_batch_size"],
            item["opening_prior_scale"],
            item["opening_selection_scale"],
        )
        if key in seen:
            continue
        seen.add(key)
        chosen.append(
            make_config(
                item["name"],
                item["root_batch_size"],
                item["opening_prior_scale"],
                item["opening_selection_scale"],
            )
        )
        if len(chosen) >= top_k:
            break
    return chosen


def aggregate_fire_results(config: dict, fires: list[dict], min_opening: float) -> dict:
    scores = [fire["score"] for fire in fires]
    mean_score = sum(scores) / len(scores)
    variance = sum((score - mean_score) ** 2 for score in scores) / len(scores)
    score_std = variance ** 0.5
    success_failures = sum(0 if is_success(fire, min_opening) else 1 for fire in fires)
    failure_penalty = 0.75 * float(success_failures)
    risk_adjusted_score = mean_score - score_std - failure_penalty

    def average_distribution(field: str) -> dict:
        numeric_keys = [
            "mean_normalized_entropy",
            "mean_effective_branching_factor",
            "mean_top1_share",
            "mean_top2_share",
            "mean_top1_top2_gap",
            "mean_mass_outside_top2",
            "mean_mass_outside_top4",
            "mean_pairwise_js_bits",
            "top2_set_jaccard",
            "agreement_rate",
            "bestmove_matches_posterior_leader_rate",
            "mean_target_rank",
        ]
        out = {}
        for key in numeric_keys:
            values = [
                float(fire[field][key])
                for fire in fires
                if key in fire.get(field, {})
            ]
            if values:
                out[key] = sum(values) / float(len(values))
        modal_moves = [fire[field].get("modal_move") for fire in fires if fire.get(field, {}).get("modal_move")]
        if modal_moves:
            out["modal_moves"] = modal_moves
        unique_moves = [int(fire[field].get("unique_moves", 0)) for fire in fires if "unique_moves" in fire.get(field, {})]
        if unique_moves:
            out["max_unique_moves"] = max(unique_moves)
        return out

    opening_distribution = average_distribution("opening_distribution")
    queen_distribution = average_distribution("queen_distribution")
    mate_distribution = average_distribution("mate_distribution")

    opening_entropy = opening_distribution["mean_normalized_entropy"]
    opening_gap = opening_distribution["mean_top1_top2_gap"]
    queen_entropy = queen_distribution["mean_normalized_entropy"]
    queen_gap = queen_distribution["mean_top1_top2_gap"]

    return {
        **config,
        "score": risk_adjusted_score,
        "score_mean": mean_score,
        "score_std": score_std,
        "failure_penalty": failure_penalty,
        "risk_adjusted_score": risk_adjusted_score,
        "strong_opening_fraction": min(fire["strong_opening_fraction"] for fire in fires),
        "mate_hit_rate": min(fire["mate_hit_rate"] for fire in fires),
        "queen_hit_rate": min(fire["queen_hit_rate"] for fire in fires),
        "opening_distribution": opening_distribution,
        "mate_distribution": mate_distribution,
        "queen_distribution": queen_distribution,
        "opening_sequences": [fire["opening_moves"] for fire in fires],
        "queen_sequences": [fire["queen_results"] for fire in fires],
        "mate_sequences": [fire["mate_results"] for fire in fires],
        "fires": fires,
    }


def evaluate_config_with_fires(
    config: dict,
    simulations: int,
    opening_runs: int,
    tactical_runs: int,
    double_firings: int,
    min_opening: float,
) -> dict:
    fires = [
        rs.evaluate_config(config, simulations, opening_runs, tactical_runs)
        for _ in range(double_firings)
    ]
    return aggregate_fire_results(config, fires, min_opening)


def batch_neighbors(batch: int, radius: int) -> list[int]:
    idx = rs.BATCH_CHOICES.index(normalize_batch(batch))
    out = []
    for delta in range(-radius, radius + 1):
        if delta == 0:
            continue
        j = idx + delta
        if 0 <= j < len(rs.BATCH_CHOICES):
            out.append(rs.BATCH_CHOICES[j])
    return out


def neighborhood(center: dict, prior_step: float, selection_step: float, batch_radius: int) -> list[dict]:
    batch = center["root_batch_size"]
    prior = center["opening_prior_scale"]
    selection = center["opening_selection_scale"]
    configs: dict[tuple[int, float, float], dict] = {}

    def add(name: str, new_batch: int, new_prior: float, new_selection: float) -> None:
        cfg = make_config(name, new_batch, new_prior, new_selection)
        configs[config_key(cfg["root_batch_size"], cfg["opening_prior_scale"], cfg["opening_selection_scale"])] = cfg

    add(center["name"], batch, prior, selection)

    for delta in (-2.0 * prior_step, -prior_step, prior_step, 2.0 * prior_step):
        add(f"{center['name']}_prior_{delta:+.4f}", batch, prior + delta, selection)

    for delta in (-2.0 * selection_step, -selection_step, selection_step, 2.0 * selection_step):
        add(f"{center['name']}_sel_{delta:+.6f}", batch, prior, selection + delta)

    for new_batch in batch_neighbors(batch, batch_radius):
        add(f"{center['name']}_batch_{new_batch}", new_batch, prior, selection)

    for prior_delta in (-prior_step, prior_step):
        for selection_delta in (-selection_step, selection_step):
            add(
                f"{center['name']}_diag_{prior_delta:+.4f}_{selection_delta:+.6f}",
                batch,
                prior + prior_delta,
                selection + selection_delta,
            )

    return list(configs.values())


def finite_difference(
    result_map: dict[tuple[int, float, float], dict],
    center: dict,
    prior_step: float,
    selection_step: float,
) -> dict:
    batch = center["root_batch_size"]
    prior = clip_prior(center["opening_prior_scale"])
    selection = clip_selection(center["opening_selection_scale"])

    def fetch(new_batch: int, new_prior: float, new_selection: float) -> dict | None:
        return result_map.get(config_key(new_batch, new_prior, new_selection))

    def derivative(dim: str) -> dict:
        if dim == "prior":
            minus_cfg = fetch(batch, prior - prior_step, selection)
            plus_cfg = fetch(batch, prior + prior_step, selection)
            delta = prior_step
        else:
            minus_cfg = fetch(batch, prior, selection - selection_step)
            plus_cfg = fetch(batch, prior, selection + selection_step)
            delta = selection_step

        if minus_cfg and plus_cfg and delta > 0:
            slope = (score_of(plus_cfg) - score_of(minus_cfg)) / (2.0 * delta)
            mode = "central"
        elif plus_cfg and delta > 0:
            center_score = score_of(fetch(batch, prior, selection))
            slope = (score_of(plus_cfg) - center_score) / delta
            mode = "forward"
        elif minus_cfg and delta > 0:
            center_score = score_of(fetch(batch, prior, selection))
            slope = (center_score - score_of(minus_cfg)) / delta
            mode = "backward"
        else:
            slope = 0.0
            mode = "none"

        return {"slope": slope, "mode": mode}

    batch_idx = rs.BATCH_CHOICES.index(normalize_batch(batch))
    lower_batch = rs.BATCH_CHOICES[batch_idx - 1] if batch_idx > 0 else None
    upper_batch = rs.BATCH_CHOICES[batch_idx + 1] if batch_idx + 1 < len(rs.BATCH_CHOICES) else None
    center_cfg = fetch(batch, prior, selection)
    lower_cfg = fetch(lower_batch, prior, selection) if lower_batch is not None else None
    upper_cfg = fetch(upper_batch, prior, selection) if upper_batch is not None else None
    if lower_cfg and upper_cfg and upper_batch != lower_batch:
        batch_slope = (score_of(upper_cfg) - score_of(lower_cfg)) / float(upper_batch - lower_batch)
        batch_mode = "central"
    elif upper_cfg and center_cfg and upper_batch is not None and upper_batch != batch:
        batch_slope = (score_of(upper_cfg) - score_of(center_cfg)) / float(upper_batch - batch)
        batch_mode = "forward"
    elif lower_cfg and center_cfg and lower_batch is not None and batch != lower_batch:
        batch_slope = (score_of(center_cfg) - score_of(lower_cfg)) / float(batch - lower_batch)
        batch_mode = "backward"
    else:
        batch_slope = 0.0
        batch_mode = "none"

    return {
        "prior": derivative("prior"),
        "selection": derivative("selection"),
        "batch": {"slope": batch_slope, "mode": batch_mode},
    }


def summarize_success_region(results: list[dict], min_opening: float) -> dict:
    success = [item for item in results if is_success(item, min_opening)]
    if not success:
        return {
            "count": 0,
            "batch_values": [],
            "prior_range": None,
            "selection_range": None,
        }

    prior_values = [item["opening_prior_scale"] for item in success]
    selection_values = [item["opening_selection_scale"] for item in success]
    batch_values = sorted({item["root_batch_size"] for item in success})
    return {
        "count": len(success),
        "batch_values": batch_values,
        "prior_range": [min(prior_values), max(prior_values)],
        "selection_range": [min(selection_values), max(selection_values)],
    }


def score_of(result: dict) -> float:
    return float(result.get("risk_adjusted_score", result.get("score", 0.0)))


def build_beacon(result: dict, min_opening: float) -> dict:
    tactical_pass = (
        result.get("mate_hit_rate", 0.0) >= 1.0
        and result.get("queen_hit_rate", 0.0) >= 1.0
    )
    opening_pass = result.get("strong_opening_fraction", 0.0) >= min_opening
    score_std = float(result.get("score_std", 0.0))
    failure_penalty = float(result.get("failure_penalty", 0.0))
    opening_entropy = float(result["opening_distribution"]["mean_normalized_entropy"])
    queen_entropy = float(result["queen_distribution"]["mean_normalized_entropy"])
    opening_gap = float(result["opening_distribution"]["mean_top1_top2_gap"])
    queen_gap = float(result["queen_distribution"]["mean_top1_top2_gap"])
    risk_adjusted_score = score_of(result)

    # Sankey intuition: unstable or noisy branches wither; sharp, repeatable
    # tactical branches retain width.
    success_factor = (
        0.45 * float(result.get("queen_hit_rate", 0.0))
        + 0.35 * float(result.get("mate_hit_rate", 0.0))
        + 0.20 * float(result.get("strong_opening_fraction", 0.0))
    )
    sharpness_factor = max(0.05, 1.0 - 0.50 * (opening_entropy + queen_entropy))
    gap_factor = min(2.0, 1.0 + 2.50 * (0.70 * queen_gap + 0.30 * opening_gap))
    stability_factor = 1.0 / (1.0 + score_std + failure_penalty)
    tactical_factor = 1.0 if tactical_pass else 0.20
    opening_factor = 1.0 if opening_pass else 0.50

    branch_mass_raw = max(0.0, risk_adjusted_score) * success_factor
    branch_mass_raw *= sharpness_factor * gap_factor * stability_factor
    branch_mass_raw *= tactical_factor * opening_factor

    return {
        "tactical_pass": tactical_pass,
        "opening_pass": opening_pass,
        "success_pass": tactical_pass and opening_pass,
        "score_mean": float(result.get("score_mean", result.get("score", 0.0))),
        "score_std": score_std,
        "failure_penalty": failure_penalty,
        "risk_adjusted_score": risk_adjusted_score,
        "opening_entropy": opening_entropy,
        "opening_gap": opening_gap,
        "queen_entropy": queen_entropy,
        "queen_gap": queen_gap,
        "success_factor": success_factor,
        "sharpness_factor": sharpness_factor,
        "gap_factor": gap_factor,
        "stability_factor": stability_factor,
        "branch_mass_raw": branch_mass_raw,
        "branch_mass": branch_mass_raw,
    }


def annotate_branch_mass(results: list[dict], min_opening: float) -> list[dict]:
    total_mass = 0.0
    for result in results:
        result["beacon"] = build_beacon(result, min_opening)
        total_mass += result["beacon"]["branch_mass_raw"]

    if total_mass <= 0.0:
        total_mass = float(len(results)) if results else 1.0
        for result in results:
            result["beacon"]["branch_mass"] = 1.0 / total_mass
    else:
        for result in results:
            result["beacon"]["branch_mass"] = result["beacon"]["branch_mass_raw"] / total_mass
    return results


def rank_key(result: dict) -> tuple[float, float, float, float]:
    beacon = result["beacon"]
    return (
        float(beacon["branch_mass"]),
        float(beacon["risk_adjusted_score"]),
        float(result.get("queen_hit_rate", 0.0)),
        float(result.get("strong_opening_fraction", 0.0)),
    )


def recommended_step(center: dict, gradients: dict, prior_step: float, selection_step: float) -> dict:
    batch = center["root_batch_size"]
    prior = center["opening_prior_scale"]
    selection = center["opening_selection_scale"]

    if gradients["prior"]["slope"] > 1e-6:
        prior += prior_step
    elif gradients["prior"]["slope"] < -1e-6:
        prior -= prior_step

    if gradients["selection"]["slope"] > 1e-6:
        selection += selection_step
    elif gradients["selection"]["slope"] < -1e-6:
        selection -= selection_step

    batch_idx = rs.BATCH_CHOICES.index(normalize_batch(batch))
    if gradients["batch"]["slope"] > 1e-6 and batch_idx + 1 < len(rs.BATCH_CHOICES):
        batch = rs.BATCH_CHOICES[batch_idx + 1]
    elif gradients["batch"]["slope"] < -1e-6 and batch_idx > 0:
        batch = rs.BATCH_CHOICES[batch_idx - 1]

    return make_config(f"{center['name']}_gradient_step", batch, prior, selection)


def choose_promoted_centers(results: list[dict], top_k: int) -> list[dict]:
    ranked = sorted(results, key=rank_key, reverse=True)
    promoted = []
    seen = set()
    for item in ranked:
        key = config_key(
            item["root_batch_size"],
            item["opening_prior_scale"],
            item["opening_selection_scale"],
        )
        if key in seen:
            continue
        seen.add(key)
        promoted.append(
            make_config(
                item["name"],
                item["root_batch_size"],
                item["opening_prior_scale"],
                item["opening_selection_scale"],
            )
        )
        if len(promoted) >= top_k:
            break
    return promoted


def evaluate_layer(
    layer_idx: int,
    centers: list[dict],
    prior_step: float,
    selection_step: float,
    batch_radius: int,
    simulations: int,
    opening_runs: int,
    tactical_runs: int,
    double_firings: int,
    min_opening: float,
    top_k: int,
) -> dict:
    configs: dict[tuple[int, float, float], dict] = {}
    for center in centers:
        for cfg in neighborhood(center, prior_step, selection_step, batch_radius):
            key = config_key(
                cfg["root_batch_size"],
                cfg["opening_prior_scale"],
                cfg["opening_selection_scale"],
            )
            configs[key] = cfg

    started = time.time()
    results = [
        evaluate_config_with_fires(
            cfg,
            simulations,
            opening_runs,
            tactical_runs,
            double_firings,
            min_opening,
        )
        for cfg in configs.values()
    ]
    annotate_branch_mass(results, min_opening)
    results.sort(key=rank_key, reverse=True)
    result_map = {
        config_key(item["root_batch_size"], item["opening_prior_scale"], item["opening_selection_scale"]): item
        for item in results
    }

    center_reports = []
    gradient_promotions = []
    for center in centers:
        gradients = finite_difference(result_map, center, prior_step, selection_step)
        step_cfg = recommended_step(center, gradients, prior_step, selection_step)
        center_reports.append(
            {
                "center": center,
                "gradients": gradients,
                "recommended_step": step_cfg,
            }
        )
        gradient_promotions.append(step_cfg)

    promoted = choose_promoted_centers(results, top_k)
    max_promoted = top_k + len(gradient_promotions)
    promoted_keys = {
        config_key(item["root_batch_size"], item["opening_prior_scale"], item["opening_selection_scale"])
        for item in promoted
    }
    for cfg in gradient_promotions:
        key = config_key(cfg["root_batch_size"], cfg["opening_prior_scale"], cfg["opening_selection_scale"])
        if key in promoted_keys:
            continue
        promoted.append(cfg)
        promoted_keys.add(key)
        if len(promoted) >= max_promoted:
            break

    return {
        "layer_index": layer_idx,
        "elapsed_sec": round(time.time() - started, 2),
        "candidate_count": len(results),
        "centers": center_reports,
        "promoted_centers": promoted,
        "winner": results[0]["name"] if results else None,
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate around successful root-search regions.")
    parser.add_argument("--seed-report", type=Path, default=DEFAULT_SEED_REPORT)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--min-opening", type=float, default=1.0)
    parser.add_argument("--prior-step", type=float, default=0.05)
    parser.add_argument("--selection-step", type=float, default=0.00025)
    parser.add_argument("--batch-radius", type=int, default=2)
    parser.add_argument("--simulations", type=int, default=1000)
    parser.add_argument("--opening-runs", type=int, default=3)
    parser.add_argument("--tactical-runs", type=int, default=3)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--double-firings", type=int, default=2)
    args = parser.parse_args()

    seed_report = load_seed_report(args.seed_report)
    centers = choose_centers(seed_report, args.top_k, args.min_opening)
    started = time.time()
    layers = []
    all_results = []
    active_centers = centers
    for layer_idx in range(args.layers):
        if not active_centers:
            break
        layer = evaluate_layer(
            layer_idx,
            active_centers,
            args.prior_step,
            args.selection_step,
            args.batch_radius,
            args.simulations,
            args.opening_runs,
            args.tactical_runs,
            args.double_firings,
            args.min_opening,
            args.top_k,
        )
        layers.append(layer)
        all_results.extend(layer["results"])
        active_centers = layer["promoted_centers"]

    all_results.sort(key=rank_key, reverse=True)
    winner = all_results[0] if all_results else None

    payload = {
        "kind": "root_region_calibration",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_sec": round(time.time() - started, 2),
        "seed_report": str(args.seed_report),
        "simulations": args.simulations,
        "opening_runs": args.opening_runs,
        "tactical_runs": args.tactical_runs,
        "layers": args.layers,
        "double_firings": args.double_firings,
        "initial_centers": centers,
        "layer_reports": layers,
        "success_region": summarize_success_region(all_results, args.min_opening),
        "winner": winner["name"] if winner else None,
        "winner_beacon": winner["beacon"] if winner else None,
        "results": all_results,
    }
    REPORT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {REPORT_PATH}")
    if winner:
        top = winner
        print(
            f"winner={top['name']} score={score_of(top):.3f} "
            f"strong_opening={top['strong_opening_fraction']:.2f} "
            f"mate={top['mate_hit_rate']:.2f} queen={top['queen_hit_rate']:.2f} "
            f"batch={top['root_batch_size']} prior={top['opening_prior_scale']:.4f} "
            f"sel={top['opening_selection_scale']:.6f} "
            f"branch_mass={top['beacon']['branch_mass']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
