# Coach Weight Model

## Purpose
This note defines a practical coach-weight model for the dojo.

The target is not "one full weight table per coach." That duplicates signal and makes promotion noisy.
The target is a hierarchical model:

- one shared base weight tensor for signal that should generalize across coaches
- one residual delta tensor per coach for coach-specific corrections
- stochastic noise during training only, so the dojo explores a local neighborhood instead of hard-coding one brittle setting
- deterministic evaluation mode, so validation and promotion gates are reproducible

This fits the current dojo language in [DOJO.md](./DOJO.md), the existing delta-vector flow in `coached_ralph.js`, and the current phase buckets used in `controller_ablation_pipeline.js`.

## Current Weight Surface
Use the existing controller-style weight surface as the first implementation target:

- features: `pawnW`, `kingW`, `queenW`, `rookW`, `minorW`, `tempoW`
- primary phases: `opening`, `midQueens`, `midNoQueens`, `endgame`
- additive modifiers: `openPos`, `closedPos`

That gives a `6 x 6` tensor shape today.

The design should not assume this is the final feature set. It should assume:

- every coach uses the same feature list
- every coach uses the same phase list
- deltas are sparse residuals on top of the shared base

## Data Model
Persist weights as a base snapshot plus per-coach residuals.

```json
{
  "version": 1,
  "features": ["pawnW", "kingW", "queenW", "rookW", "minorW", "tempoW"],
  "phases": ["opening", "midQueens", "midNoQueens", "endgame", "openPos", "closedPos"],
  "limits": { "min": -4, "max": 16 },
  "base": {
    "opening":   { "pawnW": 6, "kingW": 12, "queenW": 10, "rookW": 4,  "minorW": 8, "tempoW": 10 },
    "midQueens": { "pawnW": 10, "kingW": 14, "queenW": 12, "rookW": 8,  "minorW": 8, "tempoW": 8 },
    "midNoQueens": { "pawnW": 10, "kingW": 6, "queenW": 0, "rookW": 10, "minorW": 8, "tempoW": 8 },
    "endgame":   { "pawnW": 16, "kingW": 4, "queenW": 0,  "rookW": 14, "minorW": 8, "tempoW": 12 },
    "openPos":   { "pawnW": -2, "kingW": 0, "queenW": 0,  "rookW": 3,  "minorW": 2, "tempoW": 0 },
    "closedPos": { "pawnW": 4,  "kingW": 0, "queenW": 0,  "rookW": -3, "minorW": 0, "tempoW": 0 }
  },
  "coaches": {
    "lozza": {
      "delta": {
        "opening":   { "pawnW": 0, "kingW": 1, "queenW": 0, "rookW": 0, "minorW": 0, "tempoW": 0 },
        "midQueens": { "pawnW": 0, "kingW": 1, "queenW": 0, "rookW": 0, "minorW": 0, "tempoW": 0 },
        "midNoQueens": { "pawnW": 1, "kingW": 0, "queenW": 0, "rookW": 0, "minorW": 0, "tempoW": 0 },
        "endgame":   { "pawnW": 1, "kingW": 0, "queenW": 0, "rookW": 1, "minorW": 0, "tempoW": 0 },
        "openPos":   { "pawnW": 0, "kingW": 0, "queenW": 0, "rookW": 0, "minorW": 0, "tempoW": 0 },
        "closedPos": { "pawnW": 0, "kingW": 0, "queenW": 0, "rookW": 0, "minorW": 0, "tempoW": 0 }
      },
      "noiseSigma": 0.5
    }
  }
}
```

Implementation rules:

- `base` owns the common signal.
- `coaches[coachId].delta` stores only the residual from the base.
- `noiseSigma` is a training-time parameter, not a promoted weight.
- every tensor must have the same keys in the same order.
- clamp only after all base, delta, and additive phase terms are summed.

## Resolution Rule
For one position and one coach, resolve the effective weight vector from the hierarchy instead of reading a flat table.

### Phase selection
Use one primary phase:

- `opening`
- `midQueens`
- `midNoQueens`
- `endgame`

Then apply zero or more additive modifiers:

- `openPos`
- `closedPos`

This matches the current pattern in `controller_ablation_pipeline.js`: one main phase plus structural adjustments.

### Effective weight equation
For feature `f` under coach `c`:

```text
primary = classify_primary_phase(ctx)
mods = classify_additive_modifiers(ctx)

w(f) =
  base[primary][f]
  + coachDelta[c][primary][f]
  + sum(base[m][f] + coachDelta[c][m][f] for m in mods)
  + noise(c, primary, f, mode)

effective(f) = clamp(round(w(f)), minWeight, maxWeight)
```

Where:

- `noise(...) = 0` in evaluation mode
- `noise(...)` is sampled from a zero-mean bounded distribution in training mode

### Reference pseudocode
```js
function resolveCoachWeights(snapshot, coachId, ctx, mode, rng) {
  const coach = snapshot.coaches[coachId];
  const primary = classifyPrimaryPhase(ctx);
  const mods = classifyAdditiveModifiers(ctx);
  const out = {};

  for (const feature of snapshot.features) {
    let v = snapshot.base[primary][feature] + coach.delta[primary][feature];

    for (const mod of mods) {
      v += snapshot.base[mod][feature];
      v += coach.delta[mod][feature];
    }

    if (mode === 'train') {
      v += sampleBoundedNoise(coach.noiseSigma, rng);
    }

    out[feature] = clamp(Math.round(v), snapshot.limits.min, snapshot.limits.max);
  }

  return out;
}
```

## Why The Hierarchy Matters
Use the shared base to absorb stable signal that multiple coaches keep rediscovering.
Use coach deltas for residual bias or specialty.

That gives three operational benefits:

- lower storage and smaller promotion diffs
- faster convergence because coaches start from a competent shared prior
- cleaner promotion logic because common improvements can be promoted to base while coach-specific differences stay local

Practical rule:

- if the same signed residual appears across multiple coaches and survives hold-out validation, promote it into `base`
- if the improvement only helps one coach or one narrow corpus, keep it in that coach's `delta`

## Training-Time Noise
Noise is for exploration, not for evaluation.

Requirements:

- zero mean over repeated samples
- bounded support so one sample cannot blow up the weight table
- applied after base + delta composition, before clamp
- configurable per coach
- fully disabled in evaluation mode

Recommended starting point:

- discrete Gaussian or triangular noise
- `sigma` in the `0.25` to `0.75` range in weight units
- hard cap at `+-2` weight units per feature

Do not:

- serialize sampled noise into promoted snapshots
- let noise change phase classification
- use noise during parity, gate, or benchmark runs

## Deterministic Evaluation Mode
Promotion and regression work only if evaluation is reproducible.

Evaluation mode should therefore:

- load one immutable snapshot
- disable all stochastic noise
- use a stable phase classifier
- resolve weights in a fixed feature order
- emit the resolved vector in debug logs when needed

For the same:

- snapshot id
- coach id
- FEN or evaluator context
- phase-classification code

the resolved effective weight vector must be identical on every run.

That determinism is required for:

- coach-agreement gates
- CPU/GPU parity checks
- A/B benchmark comparisons
- promotion decisions

## Phase-Conditioned Weights
Do not treat phase conditioning as optional decoration. It is part of the weight model.

### Primary phase
Use exactly one primary phase per position:

- `opening`
  Early development and dense material.
- `midQueens`
  Middlegame with queens still on.
- `midNoQueens`
  Middlegame without queens.
- `endgame`
  Reduced material and king activity dominates.

### Additive structural modifiers
These are not standalone phases. They are adjustments:

- `openPos`
  Increase or reduce weights when files/diagonals are open.
- `closedPos`
  Increase or reduce weights when pawn chains lock the board.

Operational rules:

- resolve primary phase first
- add structural modifiers second
- apply the same logic to `base` and `coach.delta`
- clamp once at the end

That avoids a common bug where the coach delta is applied only to the main phase but forgotten for open/closed board adjustments.

## Update Path
The dojo already speaks in delta vectors. Keep that contract.

Recommended write path:

1. Coach review produces category counts or direct feature deltas.
2. Convert that review into a residual update for `coach.delta`, not a full table overwrite.
3. Run training in `mode=train` with bounded noise.
4. Re-score on a hold-out set in `mode=eval`.
5. Promote or reject the residual.
6. Periodically fold stable, cross-coach residuals into `base`.

This preserves the existing dojo pattern:

- detect weakness
- propose delta vector
- validate
- promote or reject

## Acceptance Criteria
A coach-weight implementation is not done until it clears all of the following.

### Representation
- One shared `base` tensor exists and is used by every coach.
- Each coach stores only a same-shape `delta` tensor plus noise configuration.
- Zero delta for a coach produces the exact base behavior for that coach in eval mode.

### Resolution correctness
- For a fixed context, resolved weights equal `base + delta + additive modifiers`, clamped once at the end.
- `openPos` and `closedPos` adjustments apply to both `base` and `delta`.
- Missing features or phases fail validation instead of silently defaulting.

### Stochastic training behavior
- Training mode injects bounded, zero-mean noise.
- Eval mode injects no noise at all.
- Two eval runs with the same snapshot and context produce identical resolved weight vectors.

### Promotion gate
- Candidate clears the existing dojo gate in [DOJO.md](./DOJO.md): no illegal-move regressions, no time-cap regressions, no coach-agreement regression.
- Hold-out coach agreement improves relative to the parent snapshot.
- Deterministic eval logs show stable resolved weights across repeated runs.
- If a base promotion is proposed, the gain must survive across more than one coach or corpus slice.

### Observability
- Debug output can print `snapshot`, `coach`, `primaryPhase`, `modifiers`, and final resolved weight vector.
- A failed gate can show whether the miss came from base, coach delta, or training noise.

## Minimum Test Matrix
Before promotion, run at least this matrix:

1. Shape validation: snapshot keys and tensor dimensions.
2. Golden resolution fixtures: one case for each primary phase and each additive modifier.
3. Determinism check: repeated eval-mode resolution on the same inputs.
4. Noise statistics check: repeated train-mode resolution confirms near-zero mean and bounded spread.
5. Hold-out coach agreement check: candidate vs parent.
6. Existing dojo validation gate: legality, time caps, and agreement.

If any one of those fails, reject the snapshot. Do not "average it in later."

## Recommended First Cut
For the first implementation, keep scope narrow:

- keep the existing 6 features and 6 phase buckets
- add one shared base snapshot
- add one `delta` tensor per coach
- support one scalar `noiseSigma` per coach
- disable noise automatically in eval mode
- add one debug path that prints the resolved vector

That is enough to prove the hierarchy works before introducing per-feature sigma, low-rank deltas, or more complex promotion logic.
