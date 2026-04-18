#!/bin/bash
set -e
cd "$(dirname "$0")"
echo "=== Compiling nn_train ==="
make -B nn_train 2>&1
echo "nn_train compiled OK"
ls -la nn_train
