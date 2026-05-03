# GPU Fighter Accuracy

This directory is the tracked home for the CUDA dojo fighter-accuracy work.

The current source-of-truth implementation is still external:

- Lab: `/home/frosty40/VC_V1/.docker_labs/dojo_conversion_lab`
- Strict runner: `scripts/run_cpu_gpu_accuracy_matrix.mjs`
- Strict validator: `scripts/validate_cpu_gpu_accuracy.mjs`
- GPU comparator: `cuda/gpu_forge.cu`

That lab answers the question that matters for GPU-native matches:

> Given the same FEN, does the GPU fighter path choose the same move as the CPU
> fighter?

It is different from the conversion gate. The conversion gate proves source JS
to converted JS/IR/blob stability. The CPU-GPU accuracy gate proves move-choice
parity between the CPU fighter and the GPU runtime.

## Current Baseline

The preserved baseline from the strict lab is:

- [CPU-GPU accuracy matrix](baselines/external_dojo_conversion_lab_cpu_gpu_accuracy_matrix_2026-04-16.md): 0/5 pass
- [Conversion matrix](baselines/external_dojo_conversion_lab_conversion_matrix_2026-04-16.md): 5/5 pass

Interpretation:

- Conversion parity exists.
- GPU fighter move parity does not.
- Any "done" claim must pass the CPU-GPU accuracy matrix, not only the
  conversion matrix.

## Canonical Command

Baseline bridge, expecting the current failing condition:

```bash
npm run accuracy:gpu-fighter-baseline
```

Snapshot the existing external reports without launching the GPU comparator:

```bash
npm run accuracy:gpu-fighter-snapshot
```

Hard gate, failing nonzero until GPU fighter move parity is real:

```bash
npm run accuracy:gpu-fighter-gate
```

Both commands write a receipt under `fighter_accuracy/receipts/` with:

- exact command
- lab path
- lab git status
- source file hashes
- copied result reports
- stdout/stderr

The receipt is the condition artifact. If source files or the external lab
change, the hashes change.

The bridge has a default 300 second timeout. A timeout is a failing condition,
not a pass, and is recorded in the receipt.
