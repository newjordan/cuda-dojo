# CUDA Dojo Action Plan

Last updated: 2026-04-17

## Scope

This plan converts the DGX Spark / single-GPU research report into concrete repo work.

Interpretation of "fully GPU resident":
- CPU is allowed for process startup, CUDA context setup, memory allocation, UCI/network I/O, and verification.
- CPU is not allowed to perform chess compute inside the engine.

Single-GPU target:
- Primary shipping target: `CUDA_SINGLE_GPU_TARGET=dgx-spark` (`sm_121`)
- Scale-up path: larger single GPUs such as `h200`
- Multi-GPU is explicitly deferred

## What Exists Now

Implemented and verified:
- GPU alpha-beta baseline engine with working UCI
- Verified move generation, static eval, and shallow alpha-beta correctness
- GPU-only arena smoke for `cuda_engine`
- Queue/scheduler scaffolding in `cuda/engine/scheduler.*`
- Eval-service scaffolding with bucket sizes `1/8/32/64/128` in `cuda/engine/eval_service.*`
- Standalone GPU MCTS in `cuda/chess_mcts.cu`
- MCTS audit harness in `cuda/mcts_audit.py`
- Single-GPU target knobs in `cuda/Makefile` and `cuda/SINGLE_GPU_TARGETS.md`
- Live runtime seam: the real `search_root()` path now uses `scheduler.*` plus `eval_service.*` for the depth-1/root batch, with `search_runtime_probe` and long-verifier artifacts proving the integration
- TT publication/read path is hardened with a per-slot version guard; `tests/test_tt` is green on GB10 after the race fix
- MCTS opening policy is no longer flat-noise on `startpos`; the latest audit raises strong-opening fraction to `0.82` with only `4` modal choices across the diversity run

Not implemented yet:
- Scheduler/eval-service as the real engine runtime
- Real batched PUCT with policy/value evaluation
- cuBLASDx / CUTLASS / device-graph evaluator backend
- 128-bit TT line design described by the research report
- Full profiling and ablation program on the target GPUs

## Workstreams

### 1. Search Runtime

Goal:
- Replace "GPU alpha-beta baseline plus sidecar scaffolds" with a GPU-native search runtime.

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
- Scheduler counters become meaningful runtime artifacts

### 2. Evaluator Service

Goal:
- Turn eval-service from scaffolding into a real GPU eval pipeline.

Deliverables:
- Keep bucketed request/result flow as the stable surface
- Add at least one stronger backend path beyond pure PeSTO fallback
- Preserve fallback backend for deterministic verification

Files:
- `cuda/engine/eval_service.cu`
- `cuda/engine/eval_service.cuh`
- `cuda/engine/eval.cu`
- `cuda/engine/Makefile`

Exit criteria:
- Bucket dispatch works under engine-controlled runs
- Fallback backend remains testable
- Backend selection is explicit and surfaced in counters/artifacts

### 3. GPU MCTS / PUCT

Goal:
- Replace flat/random root behavior with a measurable path toward GPU-native PUCT.

Deliverables:
- Root allocation driven by search policy rather than equal split
- Tactical quality improvements that survive audit, not just probe FENs
- Audit exposes root coverage, concentration, and bestmove/report mismatches

Files:
- `cuda/chess_mcts.cu`
- `cuda/mcts_audit.py`

Immediate bounded upgrades:
- Fix final move / output / audit contract
- Improve root priors with reply-aware tactical features
- Add shallow tactical rollout bias

Exit criteria:
- `mcts_audit.py` remains fully green on legality
- Mate and hanging-queen probes improve in visit rank and/or chosen move
- Start-position distribution is no longer flat noise

### 4. Transposition Table

Goal:
- Close the gap between the current lockless TT and the report's stronger GPU TT design.

Deliverables:
- Audit current TT invariants and replacement policy
- Decide whether to keep the current design for baseline or introduce a 128-bit line path
- Add stress tests that target contention and replacement behavior

Files:
- `cuda/engine/tt.cu`
- `cuda/engine/tt.cuh`
- `cuda/engine/tests/test_tt.cu`

Exit criteria:
- TT tests pass
- Replacement policy is documented
- If 128-bit line support lands, it has explicit alignment/order guarantees

### 5. Verification And Ablations

Goal:
- Make every architecture change pay rent in measured engine artifacts.

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
- Artifacts live under `cuda/engine/artifacts/` or `cuda/`
- Status doc matches the current reports

## Sequencing

1. Stabilize MCTS output/audit contract
2. Improve MCTS tactical signal with bounded search-policy changes
3. Wire scheduler/eval-service into the engine runtime
4. Strengthen evaluator backend options
5. Revisit TT design once runtime/eval path is real
6. Run full build + audit + verification sweep on the active single-GPU target

## Rules

- No CPU chess compute in the engine
- No fake progress from agent self-tests without local verification
- No fighter-level detours
- No multi-GPU work until single-GPU targets are pushed much harder
- Every meaningful change must end in build/test artifacts
