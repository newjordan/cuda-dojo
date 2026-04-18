#!/bin/bash
# install_stockfish.sh — World Class Instructor Setup
set -e

ARCH="armv8"
if [[ "$(uname -m)" == "x86_64" ]]; then
    ARCH="x86-64-avx2"
fi

echo "🚀 Installing Stockfish 18 ($ARCH)..."

TMP_DIR="/tmp/stockfish_build_$(date +%s)"
mkdir -p "$TMP_DIR"

git clone --depth 1 -b sf_18 https://github.com/official-stockfish/Stockfish.git "$TMP_DIR"
cd "$TMP_DIR/src"

make -j$(nproc) build ARCH=$ARCH
cp stockfish "$(dirname "$0")/trainers/stockfish/stockfish_bin"

echo "✅ Stockfish 18 installed successfully to trainers/stockfish/stockfish_bin"
rm -rf "$TMP_DIR"
