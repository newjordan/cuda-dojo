# Dojo Architecture

This document describes the runtime and measurement surfaces that exist in the public repo today. It is not a future-state manifesto.

## Scope
- Single GPU is the primary target.
- CPU is allowed for startup, UCI, process control, and verification.
- CPU chess compute is not the intended engine architecture.

## Current Runtime Layers

### 1. Modular UCI Engine
Files:
- `cuda/engine/engine.cu`
- `cuda/engine/search.cu`
- `cuda/engine/uci.cu`
- `cuda/engine/movegen.cu`
- `cuda/engine/eval.cu`
- `cuda/engine/tt.cu`

Role:
- correctness-first baseline
- modular surface for move generation, eval, search, TT, and UCI
- reference runtime for build, perft, symmetry, and TT tests

Current state:
- buildable
- locally testable
- used as the main correctness anchor

### 2. Experimental GPU Search Path
Files:
- `cuda/chess_mcts.cu`
- `cuda/mcts_audit.py`
- `cuda/root_shakeout.py`
- `cuda/root_region_calibrate.py`

Role:
- fast search-policy experimentation
- root-allocation experiments
- family, alarm, whitespace, and tunnel feature experiments

Current state:
- active quality-improvement path
- stronger than earlier flat-root behavior
- still unstable across budget changes on some motif classes

### 3. Measurement And Regression Layer
Files:
- `cuda/false_finish_regression.py`
- `cuda/king_pressure_regression.py`
- `cuda/queen_pressure_regression.py`
- `cuda/family_persistence_matrix.py`
- `cuda/kernel_pressure_ablate.py`
- `cuda/oracle_enrich.mjs`
- `cuda/oracle_sigma_probe.py`

Role:
- decide whether an idea survives
- expose where the engine is still wrong
- separate telemetry from selector-worthy signals

Current state:
- this is the main decision surface for architecture changes
- persistence and disagreement datasets are already useful

## Live Search Signals
These are part of the current search/measurement stack.

### Root And Tactical Signals
- tactical score
- finish score
- reverse-trap risk
- sniper-lane score
- guard-family score

### Whitespace And Corridor Signals
- no-man's-land warm/cold/contested/void
- left/center/right no-man's-land sector ownership
- middle-ray warm/cold/balance/opportunity
- left/center/right tunnel hotspots
- whitespace sniper/defense gates

### Family And Persistence Signals
- grouped corridor bundles
- adversarial network mass
- family top-3 tracking across budgets
- selector-vs-guard gaps across budgets

## What Is Live Vs Telemetry Only

### Live In Search
- forced-mate selection
- root frontier logic
- guard-family selection for difficult false-finish classes
- baseline tactical and trap-aware scoring

### Telemetry First
- grouped ray/corridor bundles
- no-man's-land sector stacks
- whitespace tunnel hotspots
- adversarial linked-network mass
- family persistence rows

Rule:
- telemetry can be broad
- live selector logic has to survive regression gates

## Current Weak Spots
- queen-pressure classes are still under-resolved
- reverse-trap cases are still not consistently closed out
- some good moves appear in the right family at `1000` simulations and are lost again at `2000` or `5000`
- whitespace/tunnel features explain more than they decide

## Architecture Gaps
These components exist partially or as scaffolding, but are not yet the main runtime.

- scheduler/eval-service as the dominant engine runtime
- real batched PUCT with policy/value evaluation
- stronger GPU eval backend beyond the current baseline paths
- final persistence-aware selector that can keep family signals alive across budget changes

## Where New Work Should Go

### Search Quality
- `cuda/chess_mcts.cu`
- `cuda/false_finish_regression.py`
- `cuda/king_pressure_regression.py`
- `cuda/queen_pressure_regression.py`
- `cuda/family_persistence_matrix.py`

Use this path for:
- guard-family improvements
- whitespace-to-bridge feature work
- budget-persistence fixes

### Engine Runtime
- `cuda/engine/search.cu`
- `cuda/engine/scheduler.cu`
- `cuda/engine/eval_service.cu`
- `cuda/engine/engine.cu`

Use this path for:
- moving from scaffolding to runtime
- preserving the baseline while pushing toward a more GPU-native engine

### Oracles And Data
- `cuda/oracle_enrich.mjs`
- `cuda/oracle_sigma_probe.py`
- `cuda/build_training_corpus.py`

Use this path for:
- ranking weak positions
- mining disagreement bands
- generating better regression sets and training rows

## Design Rules
- do not promote a feature because it sounds right
- keep exact regression gates in front of broad selector changes
- treat whitespace and corridor signals as family inputs before using them as move choosers
- prefer persistent signals over one-budget wins
