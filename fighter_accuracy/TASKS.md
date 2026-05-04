# GPU Fighter Accuracy Task List

- [x] Create focused branch from `cuda_combat_arbiter_experimental`.
- [x] Preserve the strict external CPU-GPU accuracy baseline.
- [x] Preserve the external conversion matrix baseline for contrast.
- [x] Add a tracked bridge command that records source hashes and receipts.
- [x] Promote the minimum strict accuracy gate files from the external lab into
      this repo.
- [x] Reproduce a tracked 0/5 smoke matrix with explicit timeout conditions.
- [ ] Reproduce the full strict 24-sample 0/5 baseline from the tracked gate.
- [x] Add first mismatch trace for Razor X start position.
- [x] Add GPU root-candidate trace output for disagreements.
- [x] Rule out serial root search as a parity lift on the fixed 4-sample smoke.
- [x] Rule out root move ordering as a parity lift on the fixed 4-sample smoke.
- [x] Rule out first-legal policy as a general parity lift on the fixed 4-sample smoke.
- [x] Fix the CPU harness `currentFen` path so the oracle matches standalone
      fighter book/briefing behavior.
- [x] Fix the CPU harness batch loop so each FEN gets a fresh standalone
      fighter instance.
- [x] Add an opt-in Traincar book lookup ablation for the GPU comparator.
- [x] Validate mixed family dispatch: Traincar book plus RazorBlade II proxy.
- [x] Run N=24 family-dispatch gate; passes 5/5 by threshold but not 1:1.
- [x] Test deeper GPU search as a Traincar parity lift; depth 8 regressed.
- [x] Test the existing Traincar eval bridge under the corrected oracle; no lift.
- [x] Re-test root-order, serial-root, and nearby depth ablations under the
      corrected multi-fighter oracle.
- [x] Promote Traincar book data into the CUDA fighter blob/IR instead of
      loading it from repo source at runtime.
- [x] Add opt-in CPU-shaped CUDA search ablation; smoke regressed, not promoted.
- [x] Add opt-in Traincar root tie-break ablation; smoke regressed, not promoted.
- [x] Classify current N=4 Traincar disagreements by root cause.
- [x] Add emit-all GPU root policy sample capture for FFN training.
- [x] Add tiny learned FFN residual trainer and run N=24 policy ablation.
- [x] Add opt-in runtime FFN residual hook and validate N=24 proxy lift.
- [x] Expand larger validation gates with tracked `gpu_spine/book.jsonl` FENs.
- [x] Add `--corpus-offset` and run heldout N=64 FFN validation; current FFN
      regresses on heldout, so it is not promoted.
- [x] Train expanded N=64 FFN and test offset-64 heldout; still regresses, so
      naive FFN reranking remains blocked.
- [ ] Improve GPU fighter move parity against the CPU fighter.
- [ ] Implement source-family dispatch from CUDA fighter blob surfaces.
- [ ] Implement deterministic CPU timeout/rootBestMove behavior without using the
      CPU move as an oracle.
- [ ] Implement Traincar deterministic clock / iterative-root emulation on GPU.
- [ ] Promote learned FFN residual out of proxy status only after out-of-sample
      validation beats the fixed-depth GPU policy and the condition is labeled
      separately from strict JS/Python parity.
- [ ] Add throughput/concurrency curves only after the accuracy gate is honest.
