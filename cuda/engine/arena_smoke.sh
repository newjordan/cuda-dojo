#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

ENGINE=./engine
ARENA=../gpu_arena.py
STOCKFISH=../../trainers/stockfish/stockfish_bin
PYTHON_BIN="${PYTHON:-$(command -v python3 || true)}"

if [[ ! -x "$ENGINE" ]]; then
    echo "FAIL: $ENGINE not found / not executable. Run 'make engine' first." >&2
    exit 2
fi

if [[ ! -f "$ARENA" ]]; then
    echo "FAIL: arena harness not found: $ARENA" >&2
    exit 2
fi

if [[ -z "$PYTHON_BIN" ]]; then
    echo "FAIL: python3 not found in PATH." >&2
    exit 2
fi

if [[ ! -f "$STOCKFISH" ]]; then
    echo "FAIL: stockfish binary not found: $STOCKFISH" >&2
    exit 2
fi

"$PYTHON_BIN" "$ARENA" \
    --games 1 \
    --max-plies 12 \
    --quiet \
    cuda_engine,depth=2 \
    stockfish,depth=1
