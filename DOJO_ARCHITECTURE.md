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

---

## 4. FrostMatrix — Geometric Training Pipeline

The FrostMatrix is the geometric model of how chess games unfold. It replaces hand-coded evaluation signals with learned weights by building a structured data pipeline that feeds a transformer policy+value network.

### What It Is

Not a chess engine feature. A measurement instrument: `position → 55 geometric scalars`. The transformer is trained on these scalars. The live fighter uses the trained weights.

```
Raw PGN corpus (1.6M+ games)
    ↓ L1–L3: encode 55 geometric features per ply per game
    ↓ L4:    aggregate per opening prefix (mean trajectory + variance)
    ↓ L5:    cluster into 33 geometric families (k-means on 55-dim trajectories)
    ↓ L6:    relabel sankey topology with geometric family IDs
    ↓ L7:    build transition tensor P[ply][family_i][family_j]
    ↓ L8:    build G matrix — affinity[opening_node × 33_families]
    ↓ L9:    versioned opening_package_vN.json + .bin (raw float32 for CUDA)
```

### Geometry Encoders — 17 signals, 55 channels

Each board position is encoded as a 55-float vector composed of:

| Range | Encoder | Concept |
|-------|---------|---------|
| ch 0–3 | active_fold base | warmth centroid r/f, compression, outlier fraction |
| ch 4–6 | G1 tension_gradient | dominant conflict axis eigenvector |
| ch 7–9 | G2 pawn_corridor | file topology, semi-open imbalance, closure |
| ch 10–12 | G3 king_shell | king distance (Chebyshev), overlap zone, shield asymmetry |
| ch 13–15 | G4 mobility_field | mobility-weighted centroid, activity imbalance |
| ch 16–18 | G5 color_complex | bishop color concentration, color imbalance |
| ch 19–21 | G6 outpost_gravity | outpost control centroids |
| ch 22–24 | G7 phase_gradient | game phase weight, material balance |
| ch 25–27 | G8 diagonal_axis | diagonal dominance, bishop pair geometry |
| ch 28–30 | G9 territorial_frontier | territorial advance, rank push asymmetry |
| ch 31–33 | G10 resonance_cluster | piece clustering density, resonance score |
| ch 34–36 | G11 vertical_fold | vertical compression signal |
| ch 37–39 | G12 horizontal_fold | horizontal compression, invasion density |
| ch 40–43 | G13 relationship_fold | conflict density, zone centroids |
| ch 44–46 | G14 pawn_chain | chain density, max chain length |
| ch 47–49 | G15 pin_xray | pin/xray axis vectors |
| ch 50–52 | G16 open_file | open file centroid, semi-open imbalance |
| ch 53–54 | G17 knight_outpost | outpost density per side |

See `compression/ENCODERS.md` for full reference table with byte offsets.

### Transformer Architecture (cuda/transformer_v3_frostmatrix.cu)

- **SEQ_LEN=129**: 65 family-axis nodes (33 family centroids + 32 unknown-potential nodes interleaved on a Y-axis) + 64 board token positions
- **D_MODEL=128**, **N_HEADS=4**, **N_LAYERS=4**, **N_GEO_CHANNELS=55**
- **Highway attention**: LATERAL (family×family with Gaussian opening-proximity bias), VERTICAL (family↔adjacent unknown nodes), DIAGONAL (skip-3 Y-axis connections)
- **Cross-attention**: board tokens attend to family axis — the rendered package conditions the board representation
- **SwiGLU** feedforward in each layer
- **Policy head**: 4096-way softmax — deterministic `from_sq * 64 + to_sq` (no vocabulary file)
- **Value head**: scalar win probability via tanh

The family-axis Y-axis is the key structural idea: the FrostMatrix's understanding of the current opening family is embedded directly in the token sequence. Board representation is conditioned on geometric opening context.

### Training Loop (cuda/train_v3_frostmatrix.py)

```bash
python3 cuda/train_v3_frostmatrix.py --epochs 3
```

- Reads PGN corpus from `/srv/models-hdd/chess-games/games.pgn`
- Encodes board → 129 tokens (65 family-axis + 64 board)
- Policy target: deterministic `uci_to_policy_idx(best_move)` — `from_sq * 64 + to_sq`
- Value target: game outcome from PGN result header
- Loss: policy cross-entropy + value MSE
- Output: `models/transformer_v3_epoch_N.pt`

### Tiered Training Datasets (tools/build_tiered_training_datasets.mjs)

4-tier curriculum feeding the training loop:

- **Tier 1**: base positions — supervised on outcomes
- **Tier 2**: family-conditioned — positions tagged with geometric family ID
- **Tier 3**: spiderweb hard negatives — selected by symmetry hotzone signal, family overlap, queen pressure; teaches the network where it geometrically fails
- **Tier 4**: transition surface examples — family→family edge stubs from the transition tensor; teaches opening transition geometry

### Key Diagnostic: Degenerate Centroid Trajectories

Raw centroid_r/f (channels 0,1) are ~(3.5, 3.44) for all 33 families because all openings start from the same board. This looked like a clustering failure. It is not.

The inter-family standard deviation on centroid_r/f peaks at 0.07–0.09 by ply 10. The clustering is operating correctly on the other 53 channels. The fix is `delta_centroid_trajectory`: subtract the ply-wise mean across all families to expose family-specific signal.

```
fam 13 (Scandinavian)   Δr=-0.234  queen pull toward black's rank
fam 30 (e4 d5 thrust)   Δr=-0.245  central counter geometry
fam 18 (e6 Qe2)         Δf=+0.176  strong kingside axis
```

The `delta_centroid_trajectory` is stored in `opening_package_vN.json` alongside the raw trajectory. The family-axis tokens in the transformer use the delta representation so that each family token carries a unique geometric signature.

### Files

```
cuda/transformer_v3_frostmatrix.cu   — CUDA transformer kernel
cuda/train_v3_frostmatrix.py         — PyTorch training loop
tools/render_pipeline.py             — FrostMatrix 9-layer render pipeline
tools/vocab_bridge_v3.js             — 129-token encoding bridge
tools/build_tiered_training_datasets.mjs  — 4-tier curriculum builder
compression/geometry_encoders.py     — all 17 G1-G17 encoders (standalone)
compression/board_fold.py            — warmth centroid + fold variants (standalone)
compression/centroid_routing.js      — JS port of geometry system
compression/ENCODERS.md              — full encoder reference table
docs/VOCAB_SIZE_PATCH.md             — policy head token space documentation
```

### Current State

| Component | Status |
|-----------|--------|
| Render pipeline (55-dim) | Working — v1 rendered: 33 families, 54,795 prefixes |
| Geometry encoders G1–G17 | Complete |
| Delta centroid trajectories | Fixed — package patched, pipeline updated |
| Transformer v3 .cu | Built — POLICY_SIZE=4096, deterministic encoding |
| Training loop | Built — ready to run |
| Tiered dataset builder | Built — 4 tiers |
| First training run | Pending |

---

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

### FrostMatrix And Neural Training
- `tools/render_pipeline.py`
- `cuda/transformer_v3_frostmatrix.cu`
- `cuda/train_v3_frostmatrix.py`
- `tools/build_tiered_training_datasets.mjs`

Use this path for:
- rendering new opening packages from updated PGN corpus
- training the transformer policy+value network
- extending geometry encoders (add to G1-G17 set in tools/render_pipeline.py)
- building tiered training curricula

Rule: re-run the render pipeline after adding new encoders. The package version increments. The training loop reads the new package automatically.

## Design Rules
- do not promote a feature because it sounds right
- keep exact regression gates in front of broad selector changes
- treat whitespace and corridor signals as family inputs before using them as move choosers
- prefer persistent signals over one-budget wins
