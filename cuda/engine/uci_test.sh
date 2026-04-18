#!/usr/bin/env bash
# =============================================================================
# uci_test.sh — real UCI smoke test against ./engine.
# =============================================================================
set -u

cd "$(dirname "$0")"

ENGINE=./engine
if [[ ! -x "$ENGINE" ]]; then
    echo "FAIL: $ENGINE not found / not executable. Run 'make engine' first." >&2
    exit 2
fi

PASS=0
FAIL=0
TOTAL=0

# Run a UCI script and capture stdout. Stderr is kept separately because the
# engine writes diagnostics there.
run_uci() {
    local script="$1"
    local timeout_s="$2"
    local out_file="$3"
    local err_file="$4"
    timeout "${timeout_s}" "$ENGINE" >"$out_file" 2>"$err_file" <<<"$script"
    return $?
}

check() {
    local name="$1"
    local ok="$2"
    local detail="${3:-}"
    TOTAL=$((TOTAL + 1))
    if [[ "$ok" == "1" ]]; then
        PASS=$((PASS + 1))
        echo "PASS  $name"
    else
        FAIL=$((FAIL + 1))
        echo "FAIL  $name  $detail"
    fi
}

# ----- T1: uci handshake --------------------------------------------------
out=$(mktemp); err=$(mktemp)
run_uci $'uci\nquit\n' 5 "$out" "$err"
ok=1
grep -q '^id name '   "$out" || ok=0
grep -q '^id author ' "$out" || ok=0
grep -q '^option name Hash '    "$out" || ok=0
grep -q '^option name Threads ' "$out" || ok=0
grep -q '^uciok$'     "$out" || ok=0
check "T1 uci handshake" "$ok" "(see $out)"

# ----- T2: isready --------------------------------------------------------
out=$(mktemp); err=$(mktemp)
run_uci $'isready\nquit\n' 5 "$out" "$err"
ok=1
grep -q '^readyok$' "$out" || ok=0
check "T2 isready -> readyok" "$ok" "(see $out)"

# ----- T3: startpos + go depth 3 ------------------------------------------
out=$(mktemp); err=$(mktemp)
(
    printf 'uci\n'
    printf 'ucinewgame\n'
    printf 'position startpos\n'
    printf 'go depth 3\n'
    sleep 0.5
    printf 'quit\n'
) | timeout 5 "$ENGINE" >"$out" 2>"$err"
ok=1
bm=$(grep -m1 '^bestmove ' "$out" | awk '{print $2}')
if [[ -z "$bm" ]]; then
    ok=0
elif ! [[ "$bm" =~ ^[a-h][1-8][a-h][1-8][qrbn]?$ ]]; then
    ok=0
elif ! [[ "$bm" =~ ^(e2e4|d2d4|g1f3|c2c4|b1c3)$ ]]; then
    ok=0
fi
check "T3 startpos+go depth 3 -> sensible bestmove ($bm)" "$ok" "(see $out)"

# Also check at least one info line was emitted.
ok=1
grep -q '^info depth ' "$out" || ok=0
check "T3b info line emitted during search" "$ok" "(see $out)"

# ----- T4: e4 e5 depth 3 --------------------------------------------------
out=$(mktemp); err=$(mktemp)
(
    printf 'position startpos moves e2e4 e7e5\n'
    printf 'go depth 3\n'
    sleep 0.5
    printf 'quit\n'
) | timeout 5 "$ENGINE" >"$out" 2>"$err"
ok=1
bm=$(grep -m1 '^bestmove ' "$out" | awk '{print $2}')
[[ -n "$bm" && "$bm" =~ ^[a-h][1-8][a-h][1-8][qrbn]?$ ]] || ok=0
[[ "$bm" != "a2a3" && "$bm" != "h2h4" ]] || ok=0
check "T4 e4 e5 + go depth 3 -> reasonable reply ($bm)" "$ok" "(see $out)"

# ----- T5: kiwipete fen + go movetime 200 ---------------------------------
KIWI='r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1'
out=$(mktemp); err=$(mktemp)
(
    printf 'position fen %s\n' "$KIWI"
    printf 'go movetime 200\n'
    sleep 0.5
    printf 'quit\n'
) | timeout 3 "$ENGINE" >"$out" 2>"$err"
ok=1
bm=$(grep -m1 '^bestmove ' "$out" | awk '{print $2}')
[[ -n "$bm" && "$bm" =~ ^[a-h][1-8][a-h][1-8][qrbn]?$ ]] || ok=0
grep -q '^info depth ' "$out" || ok=0
check "T5 kiwipete + go movetime 200 -> bestmove ($bm)" "$ok" "(see $out)"

# ----- T6: setoption Hash 128 ---------------------------------------------
out=$(mktemp); err=$(mktemp)
(
    printf 'uci\n'
    printf 'setoption name Hash value 128\n'
    printf 'isready\n'
    printf 'position startpos\n'
    printf 'go depth 1\n'
    sleep 0.2
    printf 'quit\n'
) | timeout 5 "$ENGINE" >"$out" 2>"$err"
ok=1
grep -q '^readyok$' "$out" || ok=0
grep -q '^bestmove ' "$out" || ok=0
check "T6 setoption Hash 128 -> search still works" "$ok" "(out=$out err=$err)"

# ----- T7: quit returns rc=0 ---------------------------------------------
out=$(mktemp); err=$(mktemp)
echo 'quit' | timeout 5 "$ENGINE" >"$out" 2>"$err"
rc=$?
ok=1
[[ $rc -eq 0 ]] || ok=0
check "T7 quit -> rc=0 (rc=$rc)" "$ok"

# ----- T8: unknown command does NOT crash --------------------------------
out=$(mktemp); err=$(mktemp)
run_uci $'foobarbaz xyz 123\nisready\nquit\n' 5 "$out" "$err"
rc=$?
ok=1
[[ $rc -eq 0 ]] || ok=0
grep -q '^readyok$' "$out" || ok=0
check "T8 unknown cmd survives, isready still works (rc=$rc)" "$ok" "(out=$out err=$err)"

# ----- T9: go infinite can be stopped ------------------------------------
out=$(mktemp); err=$(mktemp)
(
    printf 'position startpos\n'
    printf 'go infinite\n'
    sleep 0.2
    printf 'stop\n'
    sleep 0.2
    printf 'quit\n'
) | timeout 5 "$ENGINE" >"$out" 2>"$err"
rc=$?
ok=1
[[ $rc -eq 0 ]] || ok=0
grep -q '^bestmove ' "$out" || ok=0
check "T9 go infinite + stop returns bestmove (rc=$rc)" "$ok" "(out=$out err=$err)"

# ----- T10: isready during search responds without killing search ---------
out=$(mktemp); err=$(mktemp)
(
    printf 'position startpos\n'
    printf 'go infinite\n'
    sleep 0.2
    printf 'isready\n'
    sleep 0.2
    printf 'stop\n'
    sleep 0.2
    printf 'quit\n'
) | timeout 5 "$ENGINE" >"$out" 2>"$err"
rc=$?
ok=1
[[ $rc -eq 0 ]] || ok=0
grep -q '^readyok$' "$out" || ok=0
grep -q '^bestmove ' "$out" || ok=0
check "T10 isready during search still returns readyok (rc=$rc)" "$ok" "(out=$out err=$err)"

# ----- Summary ------------------------------------------------------------
echo
echo "==========================="
echo "  UCI tests: $PASS/$TOTAL passed"
echo "==========================="
[[ "$FAIL" -eq 0 ]] && exit 0 || exit 1
