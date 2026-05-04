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
- [Tracked gate smoke matrix](baselines/tracked_gate_smoke_cpu_gpu_accuracy_matrix_2026-05-03.md): 0/5 pass
- [Razor X startpos trace](baselines/tracked_gate_razor_x_startpos_trace_2026-05-03.json): CPU `a2a3`, GPU `d2d4`, CPU move ranked `10/20` by GPU

Current local working condition after the CPU harness `currentFen` and
fresh-process fixes:

- `standalone_fresh_process+strict`: 0/5 pass on the 4-sample smoke.
- `standalone_fresh_process+strict_traincar_book`: 4/5 pass on the 4-sample
  smoke; RazorBlade II remains strict-failing.
- `--gpu-family-dispatch`: 5/5 pass on the 4-sample smoke; RazorBlade II is a
  proxy pass, not strict parity.
- `--gpu-family-dispatch --samples 8 --gpu-depth 3`: 3/5 pass; Razor X and
  Firebird fail on deeper Traincar search/eval mismatches.
- `--gpu-family-dispatch --samples 8 --gpu-depth 8`: 0/5 pass, so increasing
  GPU search depth alone is not a parity fix.
- Adding `--gpu-traincar-eval` to the 8-sample family-dispatch run stays 3/5.
- `--gpu-family-dispatch --samples 24 --gpu-depth 3`: 5/5 pass by threshold,
  with actual agreement `[73.7, 57.9, 57.9, 57.9, 94.7]%`; RazorBlade II is
  still proxy-labeled.
- Corrected N=24 ablations: `--gpu-root-order` drops average agreement to
  61.1%, `--gpu-serial-root` stays at 68.4%, depth 2 drops to 58.9%, and depth
  4 drops to 49.5%.
- Traincar opening-book entries are now embedded in the tracked CUDA fighter
  blobs and loaded from the blob before the source-file fallback.
- Opt-in `--gpu-cpu-shaped-search` was added as a CUDA search-shape ablation,
  but is not promoted: N=4 family-dispatch smoke peaked at 70.0% at depth 2 and
  regressed to 65.0% at depth 3.
- Opt-in `--gpu-traincar-root-tiebreak` was added as a Traincar root-order
  ablation, but is not promoted: N=4 family-dispatch depth 3 dropped to 55.0%.
- Current fixed N=4 Traincar-family disagreement classification: 0 book misses,
  2 tie/root-order cases, 1 deterministic CPU partial-timeout/rootBestMove case,
  and 1 eval/search case. The next real target is deterministic CPU clock /
  iterative-root emulation on GPU, not another fixed-depth search tweak.
- FFN is now a separate learned residual lane, not a strict parity claim.
  `--gpu-emit-all` records root-candidate policy samples for every comparable
  position, and `npm run policy:ffn-train` trains a tiny candidate scorer. On
  the N=24 family-dispatch emit-all set, GPU top-1 is 68.4%, the FFN residual is
  90.5% on the full labeled set and 78.9% on holdout.
- The learned residual now has an opt-in runtime path via
  `--gpu-ffn-policy fighter_accuracy/artifacts/ffn_policy_top32_n24_2026-05-04.json`. On the same N=24
  family-dispatch gate it raises runtime proxy agreement to 89.5% average
  across five fighters, with 0/5 exact and 5/5 proxy-labeled. This is a policy
  proxy condition, not strict JS/Python parity.
- Larger gates now pull additional legal FENs from `gpu_spine/book.jsonl`,
  followed by variant-book fallback FENs. On the mixed N=64 surface, strict
  family-dispatch is 87.2% and the tracked FFN proxy is 91.9%. On the heldout
  `--corpus-offset 24 --samples 64` surface, strict remains 87.2% and the FFN
  proxy drops to 83.1%, so the current FFN artifact is not promoted.
- Training a second FFN on the expanded N=64 emit-all set reaches 97.8% offline
  full-set top-1 and 90.6% offline holdout, but fails the offset-64 runtime
  gate: strict is 72.5%, FFN proxy is 64.4%, and Razor X still fails. Naive FFN
  reranking is therefore an ablation lane, not the current parity fix.

Interpretation:

- Conversion parity exists.
- GPU fighter move parity does not.
- Any "done" claim must pass the CPU-GPU accuracy matrix, not only the
  conversion matrix.
- Timeout is a failure condition, not a hidden skip.

## Canonical Commands

Tracked baseline/gate command:

```bash
npm run accuracy:gpu-fighter-baseline
```

This command fails nonzero until GPU fighter move parity is real. That is
intentional.

External bridge, expecting the current failing condition:

```bash
npm run accuracy:gpu-fighter-external-baseline
```

Snapshot the existing external reports without launching the GPU comparator:

```bash
npm run accuracy:gpu-fighter-external-snapshot
```

Trace the first disagreement for a single fighter/FEN:

```bash
npm run accuracy:trace-first
```

Train and test the learned FFN residual proxy:

```bash
npm run accuracy:gpu-fighter-baseline -- --samples 24 --configs 4 --sims 4 --timeout-ms 120000 --gpu-depth 3 --gpu-filter-legal --gpu-family-dispatch --gpu-emit-all
npm run policy:ffn-train -- --epochs 250
npm run accuracy:gpu-fighter-baseline -- --samples 24 --configs 4 --sims 4 --timeout-ms 120000 --gpu-depth 3 --gpu-filter-legal --gpu-family-dispatch --gpu-ffn-policy fighter_accuracy/artifacts/ffn_policy_top32_n24_2026-05-04.json
npm run accuracy:gpu-fighter-baseline -- --samples 64 --corpus-offset 24 --configs 4 --sims 4 --timeout-ms 120000 --gpu-depth 3 --gpu-filter-legal --gpu-family-dispatch --gpu-ffn-policy fighter_accuracy/artifacts/ffn_policy_top32_n24_2026-05-04.json
```

The external bridge commands write a receipt under `fighter_accuracy/receipts/`
with:

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

The tracked port provenance is in [PORT_MANIFEST.md](PORT_MANIFEST.md).
