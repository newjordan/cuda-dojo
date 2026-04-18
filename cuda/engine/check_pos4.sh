#!/bin/bash
# Cross-check gpu_fighter's perft against engine's perft for position 4.
set -euo pipefail
cd "$(dirname "$0")/.."
echo "=== gpu_fighter perft 4 on Position 4 ==="
echo 'r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2pP/R2Q1RK1 w kq - 0 1' | ./gpu_fighter --perft 4
