# Dojo Stabilization Status

## Stable Now
- The original academy repo is private; this public repo is the standalone collaboration surface.
- Active forge scripts now resolve from repo root instead of the old public nested path.
- Live forge output is isolated under `runtime/`.
- Snapshotting no longer auto-commits or auto-pushes.
- The clean forge preserves the prior legacy `variants/frostd4d_gpu_latest.js` as its seed when present.
- Legacy forge exhaust is archived under `archive/legacy_forge/` instead of sitting in the live source path.

## Live Paths
- `variants/`
  Architecture source and deliberate fighter branches.
- `runtime/current/`
  Current forge-working fighter copies.
- `runtime/generated_variants/`
  New generated fighters from active long runs.
- `runtime/logs/`
  Active forge and snapshot logs.
- `archive/legacy_forge/`
  Preserved old logs, snapshots, results, and generated fighters.

## Remaining Mess
- Some tracked files are still in a transitional git state because legacy files were moved into `archive/` and the move has not been committed.
- There are still active source edits outside the forge-path cleanup:
  - `codex_agent/variants/frostd4d_codex_deepseeker.js`
  - `cuda/gpu_forge.cu`
  - `cuda/gpu_forge`
  - `game_driver.mjs`
  - `variants/frostd4d_gpu_latest.js`
- Compiled CUDA binaries are still tracked in the repo and have not been reclassified yet.

## Good Enough To Discuss Fighter And Training?
Yes, almost.

The repo boundary is corrected and the forge runtime boundary is now corrected.
What remains is mostly git hygiene and architecture reconciliation, not repo-chaos.

That means fighter and training discussion can start once the remaining active source edits are intentionally classified:
- keep as canonical source
- archive as legacy
- or replace with a cleaner academy contract

## Next Recommended Step
Build a canonical academy contract around:
- trainer interface
- branch-evaluation interface
- fighter knob surface
- promotion / benchmark protocol

That discussion should happen against the cleaned live paths above, not the archived legacy exhaust.
