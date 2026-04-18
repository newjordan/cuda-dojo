# CUDA Dojo Root-Search Distribution Metrics

This note defines the root-search distribution metrics that matter for the CUDA dojo and explains how to read them in the current `cuda/chess_mcts` / `mcts_audit.py` / `mcts_ablate.py` workflow.

The goal is to measure three different things without conflating them:

1. Search quality: does root mass concentrate on good chess moves, especially on forcing positions?
2. Determinism: does the same position produce the same root distribution across repeated runs?
3. GPU allocation quality: are simulations being spent on the right number of root candidates, or is the root either too flat or too collapsed?

## 1. Root objects emitted by `cuda/chess_mcts`

For one FEN, `cuda/chess_mcts --json` emits a root record with:

- `bestmove`
- `selection_metric`
- `selection_shortlist_visits`
- `simulations`
- `posterior_simulations`
- `moves[]`, where each move row includes:
  - `move`
  - `visits`
  - `prior_visits`
  - `posterior_visits`
  - `raw_winrate`
  - `prior_winrate`
  - `posterior_winrate`
  - `root_score`

For legal root moves `m_1, ..., m_n`, define:

```text
v_i = visits_i
pi_i = prior_visits_i
e_i = posterior_visits_i = v_i + pi_i
w_i = posterior_winrate_i
r_i = root_score_i
V = sum_i v_i = simulations
E = sum_i e_i = posterior_simulations
```

The most useful root distribution for dojo metrics is the posterior visit distribution

```text
p_i = e_i / E.
```

This is the default because:

- `bestmove` selection is based on posterior-style shortlist logic, not raw visits alone.
- priors are part of the engine's actual root policy, so they should be visible in the main diagnostic distribution.
- `posterior_visits` is already emitted directly by the engine.

The raw rollout distribution is still useful:

```text
p_i^raw = v_i / V.
```

Use `p^raw` when the question is "where did playout budget really go?" Use `p` when the question is "what root belief did the engine actually act on?"

If priors are being tuned, report both. A large gap between posterior and raw metrics means the priors are steering the root strongly.

## 2. The current engine selection rule

The engine does not pick `bestmove` by raw winrate alone.

At root, the code computes a posterior mean

```text
q_i = (wins_i + prior_wins_i) / (v_i + pi_i)
```

and a PUCT-style diagnostic score

```text
r_i = q_i + c_puct * mu_i * sqrt(V + 1) / (1 + v_i),
```

where `mu_i` is the prior mean for move `i`.

The final move is chosen from a shortlist:

```text
e_i >= max(16, floor(0.1 * max_j e_j)).
```

Inside that shortlist, selection is by one of:

```text
posterior_winrate
posterior_winrate + opening_selection_scale * opening_development_score
posterior_winrate + tactical_bias
```

depending on `selection_metric`.

That matters for interpretation:

- `root_score` is a diagnostic of exploration pressure, not a probability distribution.
- `bestmove` can differ from top-1 by `root_score`.
- `bestmove` can also differ from top-1 by posterior if opening or tactical bias is active.

So the distribution metrics below should be computed from visit mass, while `root_score` and `selection_metric` should be used to explain mismatches.

## 3. Shannon entropy

Given the posterior distribution `p = (p_1, ..., p_n)`, define Shannon entropy

```text
H(p) = -sum_i p_i log p_i,
```

with the convention `0 log 0 = 0`.

This satisfies

```text
0 <= H(p) <= log n.
```

Interpretation:

- low entropy: the root is concentrated on a small set of moves
- high entropy: the root is spread broadly across legal moves

For chess root search:

- on forcing tactical positions, good search should usually drive entropy downward as simulations increase
- on opening positions, entropy should not collapse to 0 immediately because multiple strong moves can be reasonable
- entropy alone is not "quality": a search can be confidently wrong

Use natural logs if you want `exp(H)` to be the effective branching factor below. If you use base-2 logs, the units are bits and `2^H` is the effective branching factor.

## 4. Normalized entropy

To compare positions with different legal move counts, normalize entropy by the maximum possible entropy:

```text
H_norm(p) = H(p) / log n,    for n >= 2,
H_norm(p) = 0,               for n <= 1.
```

Then

```text
0 <= H_norm(p) <= 1.
```

Interpretation:

- `H_norm` near 0: nearly all mass is on one move
- `H_norm` near 1: root mass is close to uniform over legal moves

This is the cleanest single number for "flat root vs concentrated root" across arbitrary positions.

For dojo use:

- on tactical puzzles, `H_norm` should fall with more simulations if the search is separating the winning move
- on the start position, a moderate `H_norm` is healthier than either extreme:
  - too low can mean premature collapse
  - too high can mean weak discrimination or near-random opening spread

## 5. Effective branching factor

Define the effective branching factor by

```text
B_eff(p) = exp(H(p)).
```

Properties:

```text
1 <= B_eff(p) <= n.
```

If the root distribution is exactly uniform on `k` moves and zero elsewhere, then

```text
B_eff(p) = k.
```

So `B_eff` is the number of equally likely root moves that would have the same entropy as the real distribution.

Interpretation:

- `B_eff ~= 1`: root is effectively single-threaded onto one move
- `B_eff ~= 2` to `4`: a small contender set remains
- `B_eff ~= n`: root is nearly flat

This is often more intuitive than entropy because it is expressed directly in "how many live root branches are we really funding?"

For GPU allocation, `B_eff` is the cleanest measure of how many root children are still competing for budget.

## 6. Top-1 / top-2 gap

Sort moves by posterior mass:

```text
p_(1) >= p_(2) >= ...
```

Define the main top-1 / top-2 gap by

```text
Delta_12 = p_(1) - p_(2).
```

This satisfies

```text
0 <= Delta_12 <= 1.
```

Interpretation:

- large `Delta_12`: the leader is clearly ahead in root mass
- small `Delta_12`: the top two moves are still unresolved

For root quality:

- on forcing positions, good search should usually increase `Delta_12` with more simulations
- on strategic openings, a small-to-moderate gap can be healthy if multiple strong moves remain plausible

There is also a useful secondary diagnostic gap on posterior value:

```text
Delta_12^Q = w_(1) - w_(2),
```

where `w_(1)` and `w_(2)` are the top two posterior winrates.

Read the two gaps together:

- large mass gap, tiny value gap: search is concentrating early despite weak value separation
- tiny mass gap, large value gap: search is under-allocating to a clearly better move

The primary dojo metric should be the mass gap `Delta_12`, because it measures actual root commitment.

## 7. Jensen-Shannon divergence across runs

For two runs of the same FEN and config, let `p` and `q` be the posterior root distributions on the same legal move support. Define

```text
m = (p + q) / 2
KL(p || m) = sum_i p_i log(p_i / m_i)
JS(p, q) = 0.5 * KL(p || m) + 0.5 * KL(q || m).
```

Properties:

- `JS(p, q) >= 0`
- `JS(p, q) = 0` iff the two distributions are identical
- with natural logs, `JS(p, q) <= log 2`
- with base-2 logs, `JS(p, q) <= 1`

Why JS instead of plain KL:

- JS is symmetric
- JS is always finite
- JS behaves well when some moves have zero mass in one run

In this workflow, no smoothing is needed if you compute JS over the full `moves[]` list from the same FEN, because `chess_mcts` emits every legal root move.

For more than two runs, use mean pairwise JS:

```text
mean_JS = average over all unordered run pairs JS(p^(a), p^(b)).
```

Interpretation:

- low JS: run-to-run root behavior is stable
- high JS: root mass moves around materially across runs

This is the best distribution-level determinism metric in the dojo.

## 8. How these metrics map to chess search quality

These metrics are useful only when read together.

### A. Good forcing-node behavior

Healthy pattern:

- expected tactical move reaches top-1 by posterior mass and posterior winrate
- `H_norm` decreases as simulations increase
- `B_eff` trends toward 1 or a small number
- `Delta_12` widens
- JS across repeated runs is low

This means the search is not merely exploring; it is converging.

### B. Flat or under-discriminating root

Warning pattern:

- `H_norm` near 1
- `B_eff` close to legal move count
- `Delta_12` tiny
- many different top-1 moves across runs
- start position has weak strong-opening concentration

This usually means the root policy is too flat: rollouts are noisy, priors are uninformative, or selection is not separating good moves.

### C. Premature collapse

Warning pattern:

- `H_norm` very low
- `Delta_12` large
- but the top move is tactically wrong or disagrees with posterior/value evidence

This means the search is confident, not correct. Low entropy is not automatically good.

### D. Stable but wrong

Another warning pattern:

- JS near 0 across runs
- low entropy
- large top-1 gap
- but the same bad move wins every time

Determinism is not quality. It only says the engine repeats itself.

## 9. How these metrics map to determinism

The audit already notes that default `chess_mcts` behavior is time-seeded. In `chess_mcts.cu`, `--seed-mode time` mixes time, clock, and legal-move count into the seed, while `--seed-mode fen` hashes the FEN and simulation count.

That gives a clean interpretation:

- with time seeding, nonzero JS across reruns is expected
- with FEN seeding, JS should drop substantially
- if JS remains materially nonzero under FEN seeding, the residual instability is coming from scheduling/order effects, race-sensitive tie breaking, or other non-seed nondeterminism

So:

- top-1 agreement is a coarse determinism measure
- mean JS is the finer one

Top-1 agreement can miss cases where the same move stays top-1 but the rest of the root distribution changes a lot. JS does not miss that.

## 10. How these metrics map to GPU batch allocation

At the system level, the dojo does not just need a best move; it needs to decide which positions deserve more simulations.

The right mental model is:

- high uncertainty positions should consume more marginal GPU budget
- already-resolved positions should be cut off earlier

The root metrics support that directly.

### A. Signals that a position deserves more simulations

- high `H_norm`
- large `B_eff`
- small `Delta_12`
- high mean JS across repeated probes

These indicate unresolved competition at the root or unstable ranking.

### B. Signals that a position can be deprioritized

- low `H_norm`
- small `B_eff`
- large `Delta_12`
- low JS across runs

These indicate the root is already concentrated and stable.

### C. A practical uncertainty score

One simple batch-allocation score is

```text
U = a * H_norm + b * (1 - Delta_12) + c * JS_norm,
```

where `JS_norm = JS / log 2` under natural logs, so all three terms lie in `[0, 1]`.

Allocate extra simulations in proportion to `U`, with floors and caps.

This is not the only policy, but it matches the root quantities the engine already exports.

### D. Tactical exception

On tactical positions, a high-entropy root is more suspicious than on quiet openings. So the batch scheduler should read these metrics in context:

- on forcing puzzles, unresolved entropy usually means "keep searching"
- on openings, moderate entropy is normal if the mass is concentrated on a sensible cluster

## 11. Mapping onto `mcts_audit.py`

`mcts_audit.py` already computes several precursors:

- check 2: repeatability
- check 3: simulation scaling on tactical puzzles
- check 4: diversity on the start position
- check 5: root UCT regression probe
- check 6: comparison vs tiny CPU MCTS

The new metrics fit the current checks naturally.

### Check 2: repeatability

Current outputs:

- modal move
- agreement rate
- unique move count

Add or interpret with:

- posterior distribution for each run: `p^(r)`
- mean pairwise JS across runs
- entropy variance across runs
- top-1 / top-2 gap variance across runs

This upgrades repeatability from "same move?" to "same root shape?"

### Check 3: simulation scaling

For each sim count already tested (`1000`, `5000`, `25000`, `100000`), compute:

- `H`
- `H_norm`
- `B_eff`
- `Delta_12`

Good tactical scaling means:

- expected move rank improves
- `H_norm` falls
- `B_eff` shrinks
- `Delta_12` increases

### Check 4: move-distribution diversity on the start position

The existing check uses:

- histogram of best moves across 50 runs
- strong opening fraction
- modal fraction
- diagnosis string

Those are useful but coarse. The cleaner root-level read is:

- entropy of each run's posterior root distribution
- average `H_norm`
- average `B_eff`
- average `Delta_12`
- mean pairwise JS across runs

This distinguishes:

- healthy opening diversity
- nearly uniform flatness
- pathological collapse to one move

### Check 5: root UCT regression probe

This is where the metrics belong most directly.

The check already tracks:

- `root_coverage_fraction`
- top moves by visits, posterior, and root score
- expected move ranks
- bestmove alignment

Entropy-style metrics complement coverage:

- coverage asks "how many children got any rollout?"
- entropy asks "how concentrated was the total root mass?"

Both are needed. A position can have full coverage and still be effectively single-branch if almost all mass lands on one move.

### Check 6: GPU vs CPU

If the CPU baseline provides per-move visit counts, build a CPU distribution and compare it to the GPU posterior distribution with JS as well as top-3 overlap.

That yields a stronger answer than top-3 agreement alone.

## 12. Mapping onto `mcts_ablate.py`

`mcts_ablate.py` currently scores configs with:

- repeatability agreement rate
- strong opening fraction
- tactical hit on mate test
- tactical hit on queen-capture test

That is directionally correct, but the distribution metrics make it sharper.

For each config, add:

- mean pairwise JS on repeated start-position runs
- `H_norm` and `B_eff` on the start position
- `Delta_12` on tactical probes
- optionally both posterior and raw versions

Then interpret configs as follows:

- good config:
  - low JS under `--seed-mode fen`
  - moderate opening entropy with strong openings concentrated
  - low tactical entropy and widening tactical top-1 gap
- bad config:
  - near-uniform opening root
  - high tactical entropy at high simulation count
  - tiny tactical `Delta_12`
  - large posterior/raw mismatch indicating over-strong priors

This is especially relevant because `mcts_ablate.py` varies:

- `--seed-mode`
- `--opening-selection-scale`
- `--opening-prior-scale`

Those knobs change both determinism and root-shape, not just bestmove frequency.

## 13. Recommended default reporting package

For each audited FEN, the dojo should prefer this bundle:

```text
bestmove
selection_metric
top-1 posterior move
top-1 root_score move
H
H_norm
B_eff
Delta_12
mean_JS_across_runs    (when repeated)
root_coverage_fraction
strong_opening_share   (opening positions)
expected_move_rank     (tactical probes)
```

This is enough to answer:

- is the root concentrated?
- is it concentrated on the right move?
- is it stable across runs?
- should this position receive more GPU budget?

## 14. Bottom line

For dojo practice, the safest summary is:

- entropy and `B_eff` measure how many root branches are still alive
- top-1 / top-2 gap measures how clearly the leader is ahead
- JS divergence measures run-to-run stability of the whole root distribution
- none of them alone measures chess quality
- chess quality appears when the correct move becomes top-1 while entropy falls, `B_eff` shrinks, the gap widens, and JS stays low

That is the root-search signature we want from `cuda/chess_mcts`.
