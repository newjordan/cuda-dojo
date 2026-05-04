#!/usr/bin/env python3
"""Train a tiny FFN root-policy residual from gpu_forge policy samples.

This is an explicit learned residual lane, not a strict JS/Python parity proof.
Input reports must be generated with `--gpu-emit-all` so every comparable root
position carries GPU root candidates plus the CPU fighter move label.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT_GLOB = str(REPO_ROOT / "runtime/reports/*_accuracy.cpu_gpu_accuracy.json")
DEFAULT_OUT = REPO_ROOT / "runtime/reports/ffn_policy_eval_latest.json"
DEFAULT_WEIGHTS = REPO_ROOT / "runtime/reports/ffn_policy_latest.pt"

PIECE_TO_FLOAT = {
    "P": 1.0, "N": 2.0, "B": 3.0, "R": 4.0, "Q": 5.0, "K": 6.0,
    "p": -1.0, "n": -2.0, "b": -3.0, "r": -4.0, "q": -5.0, "k": -6.0,
}
FIGHTER_IDS = {
    "razor_x": 0,
    "queensguard": 1,
    "firebird": 2,
    "fortress": 3,
    "razorblade_ii": 4,
}


def parse_square(text: str) -> int:
    if len(text) < 2:
        return -1
    file_idx = ord(text[0]) - ord("a")
    try:
        rank = int(text[1])
    except ValueError:
        return -1
    if file_idx < 0 or file_idx > 7 or rank < 1 or rank > 8:
        return -1
    return (8 - rank) * 8 + file_idx


def board_from_fen(fen: str) -> list[str]:
    board: list[str] = []
    for ch in fen.split()[0]:
        if ch == "/":
            continue
        if ch.isdigit():
            board.extend(["."] * int(ch))
        else:
            board.append(ch)
    board = board[:64]
    while len(board) < 64:
        board.append(".")
    return board


def fighter_id(report: dict[str, Any]) -> int:
    text = (report.get("fighter") or "").lower()
    for key, idx in FIGHTER_IDS.items():
        if key in text:
            return idx
    blob = (report.get("fighterBlob") or "").lower()
    for key, idx in FIGHTER_IDS.items():
        if key in blob:
            return idx
    return len(FIGHTER_IDS)


def move_features(fen: str, root: dict[str, Any], best_score: float, legal_count: int, fid: int) -> list[float]:
    move = root.get("move") or "a1a1"
    board = board_from_fen(fen)
    from_sq = parse_square(move[:2])
    to_sq = parse_square(move[2:4])
    moving = PIECE_TO_FLOAT.get(board[from_sq], 0.0) if 0 <= from_sq < 64 else 0.0
    captured = PIECE_TO_FLOAT.get(board[to_sq], 0.0) if 0 <= to_sq < 64 else 0.0
    score = float(root.get("score") or 0.0)
    rank = float(root.get("rank") or 0.0)
    order = float(root.get("order") or 0.0)
    from_rank = from_sq // 8 if from_sq >= 0 else 0
    from_file = from_sq % 8 if from_sq >= 0 else 0
    to_rank = to_sq // 8 if to_sq >= 0 else 0
    to_file = to_sq % 8 if to_sq >= 0 else 0
    side = 1.0 if " w " in f" {fen} " else -1.0
    promo = 1.0 if len(move) > 4 else 0.0
    family = [0.0] * (len(FIGHTER_IDS) + 1)
    family[min(fid, len(family) - 1)] = 1.0
    return [
        score / 20.0,
        (score - best_score) / 20.0,
        rank / 64.0,
        order / 64.0,
        legal_count / 128.0,
        moving / 6.0,
        captured / 6.0,
        from_rank / 7.0,
        from_file / 7.0,
        to_rank / 7.0,
        to_file / 7.0,
        side,
        promo,
        1.0 if root.get("is_best") else 0.0,
    ] + family


def load_groups(paths: list[str]) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for path in paths:
        report = json.loads(Path(path).read_text())
        samples = report.get("policySamples") or []
        fid = fighter_id(report)
        for sample in samples:
            roots = sample.get("gpu_root") or []
            engine = sample.get("engine")
            label = next((idx for idx, item in enumerate(roots) if item.get("move") == engine), -1)
            if label < 0 or not roots:
                continue
            best_score = max(float(item.get("score") or 0.0) for item in roots)
            legal_count = int(sample.get("legal_count") or len(roots))
            groups.append({
                "fen": sample.get("fen") or "",
                "engine": engine,
                "gpu": sample.get("mcts"),
                "features": [move_features(sample.get("fen") or "", item, best_score, legal_count, fid) for item in roots],
                "label": label,
            })
    return groups


class RootPolicyFFN(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 64),
            nn.SiLU(),
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def group_loss(model: RootPolicyFFN, group: dict[str, Any], device: torch.device) -> torch.Tensor:
    x = torch.tensor(group["features"], dtype=torch.float32, device=device)
    logits = model(x).unsqueeze(0)
    target = torch.tensor([group["label"]], dtype=torch.long, device=device)
    return F.cross_entropy(logits, target)


@torch.no_grad()
def evaluate(model: RootPolicyFFN, groups: list[dict[str, Any]], device: torch.device) -> dict[str, Any]:
    if not groups:
        return {"count": 0, "gpuTop1": 0.0, "ffnTop1": 0.0}
    gpu_hit = 0
    ffn_hit = 0
    for group in groups:
        if group["gpu"] == group["engine"]:
            gpu_hit += 1
        x = torch.tensor(group["features"], dtype=torch.float32, device=device)
        pred = int(torch.argmax(model(x)).item())
        if pred == group["label"]:
            ffn_hit += 1
    return {
        "count": len(groups),
        "gpuTop1": gpu_hit / len(groups),
        "ffnTop1": ffn_hit / len(groups),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports", nargs="*", default=None)
    parser.add_argument("--report-glob", default=DEFAULT_REPORT_GLOB)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--seed", type=int, default=20260504)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    paths = args.reports or sorted(glob.glob(args.report_glob))
    groups = load_groups(paths)
    if not groups:
        raise SystemExit("No policySamples found. Re-run the matrix with --gpu-emit-all first.")

    random.shuffle(groups)
    split = max(1, int(len(groups) * 0.8))
    train_groups = groups[:split]
    eval_groups = groups[split:] or groups[:]
    dim = len(groups[0]["features"][0])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RootPolicyFFN(dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)

    for _ in range(args.epochs):
        random.shuffle(train_groups)
        for group in train_groups:
            opt.zero_grad(set_to_none=True)
            loss = group_loss(model, group, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

    train_eval = evaluate(model, train_groups, device)
    holdout_eval = evaluate(model, eval_groups, device)
    full_eval = evaluate(model, groups, device)

    out = {
        "label": "learned_ffn_residual_not_strict_js_parity",
        "reports": paths,
        "totalGroups": len(groups),
        "train": train_eval,
        "holdout": holdout_eval,
        "full": full_eval,
        "featureDim": dim,
        "epochs": args.epochs,
        "seed": args.seed,
        "device": str(device),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    torch.save({"state_dict": model.state_dict(), "feature_dim": dim, "metadata": out}, args.weights)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
