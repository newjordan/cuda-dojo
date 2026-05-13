# CUDA Dojo Moonshot Scope

CUDA Dojo is the canonical home for the complex GPU chess system.

This includes:

- GPU MCTS and CUDA-native search.
- GPU referee/runtime integration.
- Omnifold, PZRG, FrostMatrix, spider-gradient, chronometric, and other
  high-complexity chess world-model ideas.
- Measurement gates that decide whether an idea survives into live GPU play.

## Boundary

Reference engines, PGN pulls, generic trainers, baseline labels, and generated
training corpora are inputs, not the core system. Keep those in the Reference
Chess Assets group unless a small fixture is required for a CUDA Dojo test.

Player pools, Vibe Cup submissions, arbiters, and packaged fighters are also
outside the moonshot core. They can consume CUDA Dojo outputs, but they should
not own Omnifold/PZRG/FrostMatrix research.

## Current Import Queue

The current outer workspace breakout is:

```text
/home/frosty40/VC_V1/_breakouts/00_cuda_dojo_moonshot
```

Academy migration candidates linked there:

```text
TheForge-Battle-Academy/cuda
TheForge-Battle-Academy/research/transformer_lab
TheForge-Battle-Academy/tools
TheForge-Battle-Academy/scripts
TheForge-Battle-Academy/gpu_spine
TheForge-Battle-Academy/schemas
TheForge-Battle-Academy/runtime/transformer
TheForge-Battle-Academy/runtime/reports
```

Do not bulk-import `runtime/transformer`. It is an artifact surface. Promote
only source files, schemas, launchers, small fixtures, and receipts.

