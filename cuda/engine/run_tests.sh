#!/bin/bash
# Run move-generator tests. These binaries are host code: build them with
# `make host` on a CPU machine, or with nvcc via `make engine_test_perft`.
# Transposition-table and Zobrist tests launch CUDA kernels and are skipped
# when nvcc or a GPU is not available.
set +e
cd "$(dirname "$0")"

cuda_available() {
    local nvcc_bin=""
    if command -v nvcc >/dev/null 2>&1; then
        nvcc_bin="$(command -v nvcc)"
    elif [[ -x /usr/local/cuda/bin/nvcc ]]; then
        nvcc_bin=/usr/local/cuda/bin/nvcc
    elif [[ -x /usr/local/cuda-13.0/bin/nvcc ]]; then
        nvcc_bin=/usr/local/cuda-13.0/bin/nvcc
    fi
    [[ -n "$nvcc_bin" ]] || return 1
    command -v nvidia-smi >/dev/null 2>&1 || return 1
    nvidia-smi -L 2>/dev/null | grep -q GPU
}

PERFT=./engine_test_perft
SYM=./tests/test_makeunmake_symmetry
if [[ ! -x "$PERFT" && -x ./engine_test_perft_host ]]; then
    PERFT=./engine_test_perft_host
fi
if [[ ! -x "$SYM" && -x ./tests/test_makeunmake_symmetry_host ]]; then
    SYM=./tests/test_makeunmake_symmetry_host
fi

echo "=== engine perft ==="
if [[ -x "$PERFT" ]]; then
    "$PERFT"
    PERFT_RC=$?
else
    echo "FAIL: perft binary not built. Run: make -C cuda/engine host" >&2
    PERFT_RC=1
fi

echo
echo "=== make/unmake symmetry ==="
if [[ -x "$SYM" ]]; then
    "$SYM"
    SYM_RC=$?
else
    echo "FAIL: symmetry binary not built. Run: make -C cuda/engine host" >&2
    SYM_RC=1
fi

echo
TT_RC=0
ZOB_RC=0
if cuda_available; then
    echo "=== GPU tests ==="
    if [[ -x ./tests/test_tt ]]; then
        ./tests/test_tt
        TT_RC=$?
    else
        echo "SKIP tests/test_tt (CUDA is available, binary is not built)"
    fi
    if [[ -x ./tests/test_zobrist ]]; then
        ./tests/test_zobrist
        ZOB_RC=$?
    else
        echo "SKIP tests/test_zobrist (CUDA is available, binary is not built)"
    fi
else
    echo "SKIP tests/test_tt (CUDA is not available)"
    echo "SKIP tests/test_zobrist (CUDA is not available)"
fi

echo
echo "=== summary ==="
echo "perft        rc=$PERFT_RC ($PERFT)"
echo "symmetry     rc=$SYM_RC ($SYM)"
echo "test_tt      rc=$TT_RC"
echo "test_zobrist rc=$ZOB_RC"

if [[ "$PERFT_RC" -ne 0 || "$SYM_RC" -ne 0 || "$TT_RC" -ne 0 || "$ZOB_RC" -ne 0 ]]; then
    exit 1
fi
exit 0
