# CUDA Dojo Action Plan

Last updated: 2026-04-18

## Scope

This is the current short-term implementation plan for the public repo.

Interpretation of "GPU-first":
- CPU is allowed for process startup, CUDA context setup, memory allocation, UCI/network I/O, and verification.
- CPU is not allowed to perform chess compute inside the engine.

Single-GPU target:
- Primary target: `CUDA_SINGLE_GPU_TARGET=dgx-spark` (`sm_121`)
- Scale-up path: larger single GPUs such as `h200`
- Multi-GPU is explicitly deferred

## What Exists Now

Implemented and verified:
- GPU alpha-beta baseline engine with working UCI
- Verified build/test surface in `cuda/engine/`
- Standalone GPU search path in `cuda/chess_mcts.cu`
- Regression gates for false-finish, king pressure, and queen pressure
- Oracle and sigma comparison scripts
- Family persistence matrix and whitespace/no-man's-land telemetry harvest

Not implemented yet:
- scheduler/eval-service as the primary engine runtime
- stable budget persistence on the hardest motif families
- real batched PUCT with policy/value evaluation
- stronger GPU eval backend beyond the baseline path
- complete single-GPU profiling and ablation coverage on the target hardware

## Workstreams

### 1. Search Runtime

Goal:
- push `cuda/engine/` from correctness-first baseline toward a more GPU-native runtime.

Deliverables:
- Integrate scheduler queues into the real engine path
- Route evaluation requests through eval-service instead of direct fallback-only search eval
- Preserve the current alpha-beta engine as a benchmark/baseline, not the end-state

Files:
- `cuda/engine/scheduler.cu`
- `cuda/engine/scheduler.cuh`
- `cuda/engine/queues.cuh`
- `cuda/engine/engine.cu`
- `cuda/engine/search.cu`

Exit criteria:
- Engine builds cleanly
- UCI tests still pass
- GPU-only smoke still passes
- runtime counters are meaningful enough to compare code paths

### 2. Search Quality

Goal:
- turn the current family, guard, and whitespace signals into more stable move quality.

Deliverables:
- keep exact regression gates green
- improve family persistence across the budget ladder
- convert whitespace/tunnel signals into bridge or guard features instead of direct chooser scores

Files:
- `cuda/chess_mcts.cu`
- `cuda/false_finish_regression.py`
- `cuda/king_pressure_regression.py`
- `cuda/queen_pressure_regression.py`
- `cuda/family_persistence_matrix.py`

Exit criteria:
- false-finish stays at or above `5/6`
- king pressure stays at `5/5`
- queen pressure improves over `4/8`
- family top-3 signal converts into more exact hits instead of only surviving as telemetry

### 3. Evaluator Service

Goal:
- turn eval-service from scaffolding into a real GPU eval pipeline.

Deliverables:
- keep bucketed request/result flow as the stable surface
- add at least one stronger backend path beyond pure fallback eval
- preserve a deterministic fallback backend for verification

Files:
- `cuda/engine/eval_service.cu`
- `cuda/engine/eval_service.cuh`
- `cuda/engine/eval.cu`
- `cuda/engine/Makefile`

Exit criteria:
- bucket dispatch works under engine-controlled runs
- fallback backend remains testable
- backend selection is explicit in runtime counters or artifacts

### 4. Oracle And Dataset Loop

Goal:
- make every run pay back into better regression rows or better training rows.

Deliverables:
- expand sigma-ranked disagreement coverage
- keep deriving reusable rows from persistence and whitespace runs
- add new dataset fields only when they separate exact hits from misses or guard hits from misses

Files:
- `cuda/oracle_enrich.mjs`
- `cuda/oracle_sigma_probe.py`
- `cuda/build_training_corpus.py`
- `cuda/family_persistence_matrix.py`

Exit criteria:
- new rows are preserved and documented
- worst disagreement bands can be rerun as direct regressions
- weak dataset ideas are rejected quickly

### 5. Verification And Ablations

Goal:
- make every architecture change pay rent in measured artifacts.

Deliverables:
- Keep `verify_engine_long.py` and `mcts_audit.py` as the source of truth
- Extend reporting where architecture changed
- Record single-GPU runs for DGX Spark and larger single-GPU targets

Files:
- `cuda/engine/verify_engine_long.py`
- `cuda/mcts_audit.py`
- `cuda/ENGINE_FACTORY_STATUS.md`

Exit criteria:
- Each workstream has a regression path
- artifacts live under `cuda/engine/artifacts/` or `cuda/artifacts/`
- Status doc matches the current reports

## Sequencing

1. Keep regression gates stable while improving queen-pressure and reverse-trap quality
2. Push whitespace/no-man's-land features upward into bridge and guard layers
3. Expand disagreement-ranked oracle rows and feed them back into regressions
4. Move scheduler/eval-service further into the live engine path
5. Re-run full build, audit, and verification sweep after each meaningful runtime change

## Rules

- No CPU chess compute in the engine
- No claims without local verification
- No multi-GPU work until single-GPU targets are pushed much harder
- Every meaningful change must end in build/test artifacts
