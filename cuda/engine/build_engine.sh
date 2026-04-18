#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

MAKE_BIN="${MAKE:-make}"

if ! command -v "$MAKE_BIN" >/dev/null 2>&1; then
    echo "FAIL: make not found: $MAKE_BIN" >&2
    exit 2
fi

echo "=== Building cuda/engine with current single-GPU target ==="
echo "CUDA_SINGLE_GPU_TARGET=${CUDA_SINGLE_GPU_TARGET:-dgx-spark}"

"$MAKE_BIN" \
    engine \
    engine_test_perft \
    tests/test_makeunmake_symmetry \
    tests/test_zobrist \
    tests/test_tt

echo "=== Build complete ==="
ls -la \
    ./engine \
    ./libengine.a \
    ./engine_test_perft \
    ./tests/test_makeunmake_symmetry \
    ./tests/test_zobrist \
    ./tests/test_tt
