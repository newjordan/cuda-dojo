#!/usr/bin/env bash
# run.sh — one-shot setup + comparison run.
#
# Usage:
#   bash run.sh                       # N=10, max_plies=120
#   bash run.sh --n 50                # bigger run
#   bash run.sh --n 50 --max-plies 240
set -euo pipefail

PKG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOJO_ROOT="$(cd "${PKG_ROOT}/.." && pwd)"

echo "=== cuda combat live-arbiter comparison ==="
echo "package: ${PKG_ROOT}"
echo "dojo:    ${DOJO_ROOT}"

# 1) dojo_ref importable?
if ! python3 -c "import dojo_ref" 2>/dev/null; then
    echo "ERROR: cannot import dojo_ref. Build first:"
    echo "  cd ${DOJO_ROOT}/cuda/engine && make engine librefcuda"
    echo "  cd ${DOJO_ROOT}/rust/dojo-ref-py && maturin develop --release"
    exit 2
fi
echo "  dojo_ref:        ok"

# 2) arbiter source present?
ARBITER_SRC="${ARBITER_SRC:-/home/frosty40/AgentChess_Server/match-processor/src}"
if [[ ! -f "${ARBITER_SRC}/sandboxed-referee.js" ]]; then
    echo "ERROR: arbiter source not found at ${ARBITER_SRC}"
    exit 2
fi
echo "  arbiter src:     ${ARBITER_SRC}"

# 3) docker + sandbox image?
if ! command -v docker >/dev/null 2>&1; then
    echo "ERROR: docker not on PATH"
    exit 2
fi
if ! docker image inspect agentchess-sandbox:latest >/dev/null 2>&1; then
    echo "ERROR: docker image agentchess-sandbox:latest not built"
    exit 2
fi
echo "  docker sandbox:  ok"

# 4) node?
if ! command -v node >/dev/null 2>&1; then
    echo "ERROR: node not on PATH"; exit 2
fi
echo "  node:            $(node --version)"

# 5) Run.
export ARBITER_SRC
exec python3 "${PKG_ROOT}/live_compare/compare.py" "$@"
