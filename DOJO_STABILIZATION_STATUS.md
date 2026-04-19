# Dojo Stabilization Status

This file is the current engineering status summary for the public CUDA Dojo repo. It is intentionally narrower than the private academy it came from.

## Verified Baseline
- `cuda/engine/uci_test.sh`: `11/11` passing.
- `cuda/engine/tests/test_perft.cu`: canonical engine smoke path is passing.
- `cuda/engine/tests/test_makeunmake_symmetry.cu`: passing.
- `cuda/engine/tests/test_tt.cu`: passing after the TT publication/read hardening work.
- `cuda/chess_mcts.cu` regression gates:
  - false-finish: `5/6`
  - king pressure: `5/5`
  - queen pressure: `4/8`

## Search Quality Status
- The engine is no longer just legal and playable; it has repeatable regression gates around pressure and false-finish motifs.
- The current weakness is persistence, not basic move legality.
- At `1000` simulations, the family persistence matrix reports:
  - exact hits `9/14`
  - family top-3 hits `12/14`
- At `5000` simulations, the same matrix reports:
  - exact hits `7/14`
  - family top-3 hits `10/14`

Interpretation:
- the search often surfaces the correct family before it consistently holds the exact move
- budget scaling is still unstable on some motifs

## What Is Working
- `cuda/engine/` is a usable correctness-first baseline.
- `cuda/chess_mcts.cu` has live guard-family logic for difficult false-finish and regroup classes.
- Oracle comparison against Stockfish and Lozza exists and can be used to rank disagreement bands.
- Whitespace, no-man's-land, corridor, and family-persistence data can now be harvested instead of inferred by hand.

## What Still Needs Calibration
- Queen-pressure and reverse-trap motifs are still weak.
- Some positions improve at `1000` simulations and then regress again at larger budgets.
- Whitespace and tunnel fields are useful as telemetry and family features, but not yet sufficient as direct selectors.
- Scheduler and eval-service exist in the engine tree, but the public baseline is still not the final GPU-native runtime.

## Current High-Value Targets
- make guard-family signals survive the `250 -> 5000` budget ladder more consistently
- convert whitespace/tunnel shapes into bridge or guard features instead of raw chooser scores
- keep using disagreement-ranked oracle rows to choose regression cases
- improve queen-pressure behavior without giving back the current regroup and king-pressure gains

## Reproduce
```bash
make -C cuda/engine engine
bash cuda/engine/uci_test.sh
python3 cuda/false_finish_regression.py
python3 cuda/king_pressure_regression.py
python3 cuda/queen_pressure_regression.py
make -C cuda family_persistence_matrix
```
