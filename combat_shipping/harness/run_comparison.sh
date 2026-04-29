#!/usr/bin/env bash
# run_comparison.sh — one-shot orchestration for the cuda combat shadow
# comparison. Verifies the build, then runs replay_shadow.py with sane
# defaults.
#
# Usage:
#   bash harness/run_comparison.sh                       # N=500 default
#   bash harness/run_comparison.sh --n 1000              # bigger run
#   bash harness/run_comparison.sh --pgn /path/to.pgn    # custom corpus

set -euo pipefail

# Resolve package root (this script lives in harness/).
PKG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOJO_ROOT="$(cd "${PKG_ROOT}/.." && pwd)"

echo "=== cuda combat shadow comparison ==="
echo "package: ${PKG_ROOT}"
echo "dojo:    ${DOJO_ROOT}"

# 1) Sanity check: dojo_ref importable?
if ! python3 -c "import dojo_ref" 2>/dev/null; then
    echo "ERROR: cannot import dojo_ref. Build the combat branch first:"
    echo "  cd ${DOJO_ROOT}/cuda/engine && make engine librefcuda"
    echo "  cd ${DOJO_ROOT}/rust/dojo-ref-py && maturin develop --release"
    exit 2
fi
echo "  dojo_ref:        ok"

# 2) Sanity check: arbiter source readable?
ARBITER_SRC="${ARBITER_SRC:-/home/frosty40/AgentChess_Server/match-processor/src}"
if [[ ! -f "${ARBITER_SRC}/chess-engine.js" ]]; then
    echo "ERROR: arbiter chess-engine.js not found at ${ARBITER_SRC}"
    echo "Set ARBITER_SRC env var to the right path."
    exit 2
fi
echo "  arbiter src:     ${ARBITER_SRC}"

# 3) Sanity check: node available?
if ! command -v node >/dev/null 2>&1; then
    echo "ERROR: node not in PATH (required for arbiter_referee.mjs)"
    exit 2
fi
echo "  node:            $(node --version)"

# 4) Pass through args.
export ARBITER_SRC
exec python3 "${PKG_ROOT}/harness/replay_shadow.py" \
    --arbiter-src "${ARBITER_SRC}" \
    "$@"
