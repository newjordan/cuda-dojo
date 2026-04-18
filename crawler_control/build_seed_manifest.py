#!/usr/bin/env python3
from __future__ import annotations

import json
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROFILES_PATH = ROOT / "purpose_profiles.json"
MANIFEST_PATH = ROOT / "crawler_seed_manifest.json"


def load_profiles() -> dict:
    return json.loads(PROFILES_PATH.read_text(encoding="utf-8"))


def script_candidate(
    profile: dict,
    name: str,
    hypothesis: str,
    args: list[str],
    tags: list[str],
) -> dict:
    return {
        "name": name,
        "role": profile["role"],
        "runner_mode": profile["runner_mode"],
        "runner_path": profile["runner_path"],
        "battery_profile": profile["battery_profile"],
        "hypothesis": hypothesis,
        "tags": tags,
        "args": args,
        "evidence": profile["evidence"],
    }


def env_candidate(
    profile: dict,
    name: str,
    hypothesis: str,
    env: dict[str, str],
    tags: list[str],
) -> dict:
    return {
        "name": name,
        "role": profile["role"],
        "runner_mode": profile["runner_mode"],
        "runner_path": profile["runner_path"],
        "battery_profile": profile["battery_profile"],
        "hypothesis": hypothesis,
        "tags": tags,
        "env": env,
        "evidence": profile["evidence"],
    }


def build_candidates(profiles: dict[str, dict]) -> list[dict]:
    candidates: list[dict] = []

    beacon = profiles["beacon_anchor"]
    candidates.extend(
        [
            script_candidate(
                beacon,
                "beacon_litecrawler_s4",
                "Keep one stable local crawler anchor alive for drift detection and launch sanity.",
                ["--seed", "4", "--world-size", "1", "--max-wallclock-seconds", "600"],
                ["anchor", "litecrawler", "stable"],
            ),
            script_candidate(
                beacon,
                "beacon_litecrawler_s7",
                "Check that the anchor shape is not a single-seed artifact.",
                ["--seed", "7", "--world-size", "1", "--max-wallclock-seconds", "600"],
                ["anchor", "litecrawler", "repeatability"],
            ),
        ]
    )

    explorer = profiles["explorer_vortex"]
    candidates.extend(
        [
            script_candidate(
                explorer,
                "explorer_vortex_local_s4",
                "Probe whether the novelty line is even worth budget under a cheap local launch.",
                ["--seed", "4", "--world-size", "1", "--max-wallclock-seconds", "300"],
                ["novelty", "vortex", "local"],
            ),
            script_candidate(
                explorer,
                "explorer_vortex_full_s4",
                "Preserve an explicit full-scale launch template for the risk-on branch.",
                ["--seed", "4", "--world-size", "8", "--max-wallclock-seconds", "600"],
                ["novelty", "vortex", "fullscale"],
            ),
        ]
    )

    recur = profiles["recur_quality_line"]
    candidates.extend(
        [
            env_candidate(
                recur,
                "sp8192_recur_parresid_qk5",
                "Strong SP8192 recurrence baseline with parallel residuals and legal TTT style knobs.",
                {
                    "VOCAB_SIZE": "8192",
                    "NUM_FLAT_LAYERS": "4",
                    "NUM_CRAWLER_LAYERS": "2",
                    "CRAWLER_LOOPS": "2",
                    "PARALLEL_RESIDUALS": "1",
                    "LEGAL_SCORE_FIRST_TTT": "1",
                    "QK_GAIN_INIT": "5.0",
                },
                ["sp8192", "recurrence", "parallel_residuals", "quality"],
            ),
            env_candidate(
                recur,
                "sp8192_recur_parresid_qk525",
                "Highest-confidence quality seed from the late SP8192 record line.",
                {
                    "VOCAB_SIZE": "8192",
                    "NUM_FLAT_LAYERS": "4",
                    "NUM_CRAWLER_LAYERS": "2",
                    "CRAWLER_LOOPS": "2",
                    "PARALLEL_RESIDUALS": "1",
                    "LEGAL_SCORE_FIRST_TTT": "1",
                    "QK_GAIN_INIT": "5.25",
                },
                ["sp8192", "recurrence", "parallel_residuals", "qk525", "quality"],
            ),
            env_candidate(
                recur,
                "sp4096_recur_parresid",
                "Lower-vocab recurrence branch to test whether the quality line is tokenizer-bound.",
                {
                    "VOCAB_SIZE": "4096",
                    "NUM_FLAT_LAYERS": "4",
                    "NUM_CRAWLER_LAYERS": "2",
                    "CRAWLER_LOOPS": "2",
                    "PARALLEL_RESIDUALS": "1",
                    "LEGAL_SCORE_FIRST_TTT": "1",
                    "QK_GAIN_INIT": "5.0",
                },
                ["sp4096", "recurrence", "parallel_residuals", "quality"],
            ),
        ]
    )

    depth = profiles["depth_split_line"]
    candidates.extend(
        [
            env_candidate(
                depth,
                "depth_split_7f3c_loop3",
                "Use the stable 7F+3C x3 regime as a depth anchor.",
                {
                    "NUM_FLAT_LAYERS": "7",
                    "NUM_CRAWLER_LAYERS": "3",
                    "CRAWLER_LOOPS": "3",
                    "MLP_MULT": "4.0",
                },
                ["depth_split", "7f3c", "loop3"],
            ),
            env_candidate(
                depth,
                "depth_split_8f3c_loop3",
                "Probe the late-record quality regime with one more flat layer before compression.",
                {
                    "NUM_FLAT_LAYERS": "8",
                    "NUM_CRAWLER_LAYERS": "3",
                    "CRAWLER_LOOPS": "3",
                    "MLP_MULT": "4.0",
                },
                ["depth_split", "8f3c", "loop3"],
            ),
            env_candidate(
                depth,
                "depth_split_4f2c_loop2",
                "Cheap TON-E-like rhythm seed for faster architectural iteration.",
                {
                    "NUM_FLAT_LAYERS": "4",
                    "NUM_CRAWLER_LAYERS": "2",
                    "CRAWLER_LOOPS": "2",
                    "MLP_MULT": "4.0",
                },
                ["depth_split", "4f2c", "loop2", "cheap"],
            ),
        ]
    )

    compressor = profiles["compressor_line"]
    candidates.extend(
        [
            env_candidate(
                compressor,
                "compressor_int6_loopaware",
                "Treat loop-aware GPTQ + int6 as the first compression gate once a quality branch survives.",
                {
                    "EXPORT_QUANT": "int6",
                    "LOOP_AWARE_GPTQ": "1",
                    "SELECTIVE_PRUNE_ENABLE": "1",
                    "SELECTIVE_PRUNE_FACTOR": "8",
                },
                ["compression", "int6", "gptq", "prune"],
            ),
            env_candidate(
                compressor,
                "compressor_int8_flat",
                "Keep a flatter int8 export path alive for anchor/comparison purposes.",
                {
                    "EXPORT_QUANT": "int8_flat",
                    "LOOP_AWARE_GPTQ": "0",
                    "SELECTIVE_PRUNE_ENABLE": "0",
                },
                ["compression", "int8_flat", "anchor"],
            ),
        ]
    )

    verifier = profiles["verifier_line"]
    candidates.extend(
        [
            script_candidate(
                verifier,
                "verifier_litecrawler_pair",
                "Repeat-run verifier to confirm the local anchor survives fixed launch conditions.",
                ["--seed", "4", "--world-size", "1", "--max-wallclock-seconds", "600"],
                ["verify", "litecrawler", "repeatability"],
            ),
        ]
    )

    return candidates


def main() -> int:
    raw = load_profiles()
    profiles = {profile["name"]: profile for profile in raw["profiles"]}
    candidates = build_candidates(profiles)
    payload = {
        "kind": "crawler_seed_manifest",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source_profiles": str(PROFILES_PATH),
        "candidate_count": len(candidates),
        "candidates": candidates,
    }
    MANIFEST_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {MANIFEST_PATH}")
    print(f"candidates={len(candidates)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
