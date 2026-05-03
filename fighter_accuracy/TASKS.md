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
- [x] Add an opt-in Traincar book lookup ablation for the GPU comparator.
- [ ] Improve GPU fighter move parity against the CPU fighter.
- [ ] Promote Traincar book data into the CUDA fighter blob/IR instead of
      loading it from repo source at runtime.
- [ ] Implement source-family dispatch from CUDA fighter blob surfaces.
- [ ] Implement deterministic CPU timeout/rootBestMove behavior without using the
      CPU move as an oracle.
- [ ] Add throughput/concurrency curves only after the accuracy gate is honest.
