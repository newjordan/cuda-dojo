#!/bin/bash
# Run engine test binaries.
set +e
cd "$(dirname "$0")"

echo "=== engine_test_perft ==="
./engine_test_perft
PERFT_RC=$?

echo
echo "=== tests/test_makeunmake_symmetry ==="
./tests/test_makeunmake_symmetry
SYM_RC=$?

echo
echo "=== summary ==="
echo "engine_test_perft               rc=$PERFT_RC"
echo "tests/test_makeunmake_symmetry  rc=$SYM_RC"
