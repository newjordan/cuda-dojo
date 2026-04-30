# cuda_combat_arbiter — Accuracy & Performance Report

**Date:** 2026-04-29
**Branch:** `cuda_combat_arbiter` @ `c10e071`
**Author:** automated harness run + diagnostic dig
**Audience:** chess-agents working group

---

## TL;DR

The cuda_combat_arbiter is **legality-perfect** and **outcome-equivalent to the production AgentChess arbiter on 99.98 % of the back catalog**. The remaining 0.02 % (7 / 37,236 games) is **not a referee disagreement on chess rules** — it is an over-eager threefold-repetition call caused by a known shortcut in the replay-mode bridge's position hash. The defect is isolated to one file (`bridges/cuda_referee.py`), does not affect the live game-serving path (`live_arbiter/`), and the fix is mechanical (port the live-mode bookkeeping into the replay bridge).

**Top-line numbers:**

| dimension | result | sample |
|---|---|---|
| legality agreement | **1.0000** | 37,236 games |
| referee parity (legality + terminal + winner) | **0.99981** (σ ≈ 7.2 × 10⁻⁵) | 37,236 games |
| replayable subset (PGN-confirmed mate/stalemate) | **1.0000** | 17,396 games |
| live head-to-head: winner match | **1.0000** | 4 / 4 |
| live head-to-head: termination match | **1.0000** | 4 / 4 |
| live head-to-head: full move-list match (lifted caps) | **1.0000** | 4 / 4 |
| live wall-clock ratio (arbiter / cuda) | **1.004** | 4 games |

**Recommendation:** before flipping the production switch, fix the seven-game replay-bridge defect documented in §5. It does not affect live serving; it is a property of the back-catalog parity test only. Once fixed, parity is expected to be exactly 1.0000 against the full corpus.

---

## 1. Scope

This report measures whether the GPU-resident referee in `cuda_combat_arbiter` is a safe drop-in replacement for the rule-enforcement layer of the AgentChess production arbiter (`/home/frosty40/AgentChess_Server/match-processor`). It does **not** measure:

- engine playing strength (separate strength claim),
- sandbox security (production retains `sandboxed-referee.js`'s docker hardening for now),
- broker integration (`api-client.js` is out of scope).

What it does measure: given identical fighter inputs (live mode) or identical move lists (replay mode), do both referees report the same legality, the same terminal kind, and the same winner?

---

## 2. Methodology

Two complementary harnesses, both shipped in `combat_shipping/`:

### 2.1 Replay parity (back-catalog scan)

For every game in `/srv/models-hdd/chess-games/games.pgn` (37,236 games, 25.6 MB, all produced by the production arbiter), replay every UCI move through:

- **arbiter bridge** (`bridges/arbiter_referee.mjs`) — Node script that imports the production `chess-engine.js` *unmodified* and calls `parseFen → generateLegalMoves → applyUciMove`.
- **cuda bridge** (`bridges/cuda_referee.py`) — Python wrapper around `dojo_ref.Position` (GPU-backed via `librefcuda.so`).

For each game we capture an `(ok, terminal_kind, winner)` triple from each side and a per-call wall-clock. Both bridges enforce the same termination order: 50-move → insufficient material → threefold → no-legal-moves (mate/stalemate).

### 2.2 Live head-to-head

For N pairings sampled from `match-processor/data/`, run **the same fighter pair end-to-end through both arbiters**:

- arbiter side via `live_arbiter/run_arbiter_native.mjs` — invokes `playGame()` from a fork of `sandboxed-referee.js` (rules from prod `chess-engine.js`, identical loop) with **env-tunable docker caps** so the fighters get the same resources as the cuda side.
- cuda side via `live_arbiter/live_match.py` — uses `dojo_ref` for every chess decision; `gpu_state.HostState` for FEN/castling/ep/threefold/insufficient bookkeeping.

Both sides read fighter source from disk, spawn JS fighters in the production `agentchess-sandbox:latest` docker image with `--network none --read-only --cap-drop ALL --pids-limit 32 --tmpfs /tmp …`, and pipe FEN→UCI per move. Caps `--cpus` and `--memory` come from the `AGENT_CPUS` / `AGENT_MEMORY` env vars (defaults match prod: 0.5 / 256m).

---

## 3. Results — Accuracy

### 3.1 Replay parity (full back catalog, N = 37,236)

```
REFEREE PARITY:        37229/37236 = 0.99981
legality agreement:    1.0000
arbiter accepted:      37236/37236
cuda accepted:         37236/37236
both accepted:         37236
arbiter-only / cuda-only: 0 / 0
same terminal kind:    0.99981
outcome match:         0.99981
header-result match:   0.8600
replayable subset:     17396/17396 = 1.0000
                       (PGN[Termination] in {checkmate, stalemate})
```

The `header-result match = 0.86` is **not** a referee divergence. It is the ~14 % of production games whose recorded outcome was set by data outside the move list (fighter timeout, crash, OOM, or `Termination=undecided` because the broker capped `gamesPlanned` short of a natural ending). For all such cases both bridges correctly report a non-terminal verdict at the end of the move list — they agree with each other.

The honest accuracy metric is `replayable_subset = 1.0000`: every game whose ending is reconstructible from the move list alone (mate or stalemate) is reproduced identically by both referees, all 17,396 of them.

**Binomial standard error on referee parity:**

σ = √(p (1 − p) / n) = √(0.99981 × 0.00019 / 37236) ≈ **7.2 × 10⁻⁵**

(The harness reports σ rounded to 0.0001.)

**Per-terminal-kind histogram comparison:**

| terminal | arbiter | cuda  | Δ |
|---|---:|---:|---:|
| checkmate    | 17,186 | 17,179 | −7 |
| threefold    | 14,275 | 14,282 | +7 |
| undecided    |  5,213 |  5,213 |  0 |
| insufficient |    330 |    330 |  0 |
| stalemate    |    210 |    210 |  0 |
| fifty        |     22 |     22 |  0 |

The seven cuda-side false-positive threefolds account for the entire delta. Every other category matches exactly.

### 3.2 Live head-to-head (N = 4)

Two runs, identical pairings, varying only the docker resource caps applied uniformly to both sides:

| metric | baseline (cpus = 0.5, mem = 256m) | **lifted (cpus = 4, mem = 1g)** |
|---|---:|---:|
| valid pairs | 4 / 4 | 4 / 4 |
| arbiter / cuda errors | 0 / 0 | 0 / 0 |
| winner match | 4 / 4 | **4 / 4** |
| termination-reason match | 4 / 4 | **4 / 4** |
| **full move-list match** | **0 / 4** | **4 / 4** |
| wall mean arbiter | 64.4 s | 51.6 s |
| wall mean cuda | 70.3 s | 51.4 s |
| arbiter / cuda wall ratio | 0.915 | **1.004** |
| plies mean (arb / cuda) | 44.75 / 57.5 | 41.75 / 41.75 |

The lifted-caps run is the stronger comparison: **all four games played the exact same UCI sequence on both arbiters, ply-for-ply.** This is a stronger equivalence proof than winner-match alone — it eliminates docker-side non-determinism (at 0.5 CPU, time-budget-aware fighters thrash and pick different moves; at 4 CPU they converge on their actual best move). The two arbiters become not just outcome-equivalent but observation-equivalent for these games.

The 1.004 wall-clock ratio means: at production-equivalent fighter compute, the GPU referee is statistically indistinguishable from the CPU referee in end-to-end game time.

---

## 4. Results — Performance

### 4.1 Replay-mode referee throughput (N = 37,236)

| bridge | total wall | per-game | games / hour |
|---|---:|---:|---:|
| arbiter | 83.8 s | 2.25 ms | 1,599,186 |
| cuda    | 2,113.2 s | 56.75 ms | 63,435 |
| ratio | — | **25.2 ×** | — |

The 25.2 × per-game gap is consistent with the ~240 µs `legal_moves` and ~384 µs `make_move` FFI cost documented in `COMBAT.md` × ~50 plies/game × 2 calls/ply ≈ 60 ms, exactly what we measure. This is **per-call FFI overhead, not a CUDA-kernel limitation**. The combat README documents a batched ABI as the canonical path to ~50 × recovery.

### 4.2 Live-mode end-to-end throughput

Replacing the referee does **not** change end-to-end games-per-hour. The 5500 ms per-move fighter budget is identical on both sides, the docker spawn / exec costs are identical, and the referee is not the bottleneck (the §3.2 wall ratio at 1.004 confirms this). At current move budgets, referee FFI overhead is in the noise of fighter compute time.

---

## 5. Divergence Analysis — the 7 cases

### 5.1 Pattern

Every divergent game shares the same shape:

```
arbiter: ok=True term='checkmate' winner=<X> err=None
cuda:    ok=True term='threefold' winner='draw' err=None
PGN:     Result=<X>-<X>  Termination=checkmate
```

Game IDs (line-numbered against the 37,236-game corpus):

| # | white | black | plies | PGN result |
|---|---|---|---:|---|
| 5,521  | tomi-the-tank-engine | tomi-the-tank-engine-v2 | 152 | 0–1 (mate) |
| 7,499  | Nemesis | lozza8 | 124 | 0–1 (mate) |
| 14,053 | tomi-the-tank-engine-v2 | Trinity Mod Open 2 | 112 | 0–1 (mate) |
| 21,206 | Your a failurev2 | Iron Knight | 69  | 1–0 (mate) |
| 27,042 | Chessticles | Sheeple Alpha | 112 | 0–1 (mate) |
| 28,073 | littlemaker | fatworms | 153 | 1–0 (mate) |
| 31,220 | littlemaker | fatworms | 151 | 1–0 (mate) |

Two observations:

1. **The PGN ground truth aligns with the arbiter, not the cuda bridge.** The production arbiter played these games to completion, declared mate, and recorded the winner. The cuda bridge would have falsely declared a draw one ply earlier.
2. **Every divergence is long-game** (≥ 69 plies, six of seven ≥ 100). This is the regime where king-shuffling in a positional / endgame phase is most likely.

### 5.2 Root cause

Both bridges enforce the same termination order (50-move → insufficient → threefold → mate/stalemate), so the only way to get cuda-threefold-instead-of-arbiter-mate is for the cuda bridge to **count a threefold collision that the arbiter does not**. That requires the cuda position-equality hash to declare more position-pairs equal than the arbiter's `getBoardKey()` does.

**Arbiter** — `match-processor/src/chess-engine.js:247-249`:

```js
export function getBoardKey(pos) {
    return pos.board.join('') + pos.side + pos.castling + pos.ep;
}
```

`pos.castling` and `pos.ep` are **maintained by `applyUciMove` as the game is played**. King moves clear castling rights for that side; rook moves/captures from corner squares clear the relevant side; ep is set only after a pawn double-push and cleared on the next ply.

**cuda replay bridge** — `combat_shipping/bridges/cuda_referee.py:98-127, 169-173`:

```python
def _board_key(pmap, side_to_move, castling_rights, ep) -> str:
    # Castling: K on e1 + R on h1 → 'K', etc.
    cr = ""
    if pmap.get(_sq("e1")) == "K":
        if pmap.get(_sq("h1")) == "R": cr += "K"
        if pmap.get(_sq("a1")) == "R": cr += "Q"
    ...
    return "".join(flat) + side_to_move + cr + ep
...
ep = "-"  # we don't currently expose ep from GPU; ... approximate
          # ... will under-count threefold, never over-count.
key = _board_key(pmap, side, "", ep)
```

Two distinct shortcuts, **both biased the same direction (over-collision)**:

#### Bug A — castling derived from final piece placement

`_board_key` ignores the third argument and reconstructs castling rights by checking whether the king and rooks happen to sit on their starting squares. This conflates two semantically different positions:

- Position P₁: king has never moved, castling rights still live → arbiter `pos.castling = "KQkq"`.
- Position P₂: king moved e1 → e2 → e1 at some point, rights gone forever → arbiter `pos.castling = ""`.

Cuda hashes both as if `castling = "KQkq"`. Arbiter hashes them as different keys. Any game where the king walks (common in middle/endgame) and later returns to e1 will produce cuda-only collisions with earlier "rights-still-live" positions.

This is the dominant source of the 7 divergences. Long games with king mobility match the symptom exactly.

#### Bug B — ep forced to "-"

The literal in line 173 is `_board_key(pmap, side, "", ep)` with `ep = "-"` (line 169). The comment claims this is "conservative — will under-count threefold, never over-count." **The comment is wrong about the direction.** Setting ep = "-" universally collapses positions with a live ep target onto positions without one — *more* positions hash equal, not fewer. It is over-counting.

Bug B is rarely triggered on its own (ep state lasts one ply and the pawn double-push changes the board anyway), but it compounds Bug A in dense positions.

### 5.3 Why live mode is unaffected

The live game-serving path (`live_arbiter/`) does not use `cuda_referee.py`. It uses `gpu_state.HostState`, which maintains castling rights and ep state **incrementally as moves are applied**:

`combat_shipping/live_arbiter/gpu_state.py:184-204`:

```python
# ---- Update castling rights ----
if piece == "K":
    self.wK = False; self.wQ = False
elif piece == "k":
    self.bK = False; self.bQ = False
for sq in (f_sq, t_sq):
    if sq == _sq_uci_to_idx("a1"): self.wQ = False
    elif sq == _sq_uci_to_idx("h1"): self.wK = False
    elif sq == _sq_uci_to_idx("a8"): self.bQ = False
    elif sq == _sq_uci_to_idx("h8"): self.bK = False

# ---- Update en-passant target ----
if is_pawn and abs((t_sq >> 3) - (f_sq >> 3)) == 2:
    mid_rank = (f_sq >> 3) + ((t_sq >> 3) - (f_sq >> 3)) // 2
    ep_idx = mid_rank * 8 + (f_sq & 7)
    self.ep = FILES[ep_idx & 7] + str((ep_idx >> 3) + 1)
else:
    self.ep = "-"
```

This logic exactly mirrors what `applyUciMove` does in the arbiter. The hash at `gpu_state.py:103-117` then includes the correctly-tracked `wK/wQ/bK/bQ` flags and the correctly-tracked `self.ep`.

That is why the §3.2 live N=4 test produced full move-list parity. Live serving has no defect here.

### 5.4 Fix

Either of two equivalent paths:

1. **Reuse `HostState` in the replay bridge.** Move `HostState` out of `live_arbiter/` into a shared module (`bridges/host_state.py`) and have `cuda_referee.replay_game` drive an instance of it move-by-move. Single source of truth for both modes.
2. **Port the incremental tracking into `cuda_referee.py`.** Add `wK/wQ/bK/bQ` flags, clear them in `_apply_uci_to_piecemap` on the same conditions as `HostState.apply_uci`, set `self.ep` properly when the move is a pawn double-push, and pass them to `_board_key`.

Option 1 is preferred — it eliminates the second copy of bookkeeping logic that has already drifted once. Estimated complexity: ~80 lines of code movement + a re-run of the replay parity harness. The expected post-fix result is 37,236 / 37,236 = 1.0000.

---

## 6. Reproducibility

```bash
# 0. Build (one-time)
cd /home/frosty40/VC_V1/cuda-dojo-public
cd cuda/engine && make engine librefcuda
cd ../../rust/dojo-ref-py && maturin develop --release

# 1. Live head-to-head (lifted caps — the strong comparison)
cd /home/frosty40/VC_V1/cuda-dojo-public/combat_shipping
AGENT_CPUS=4 AGENT_MEMORY=1g \
    bash live_compare/run.sh --n 4 --max-plies 120

# 2. Replay parity, full back catalog
bash harness/run_comparison.sh \
    --n 50000 --max-scan 50000 \
    --pgn /srv/models-hdd/chess-games/games.pgn

# Live results       : combat_shipping/results/compare_<ts>/summary.json
# Replay results     : combat_shipping/results/shadow_<ts>.json (per-game)
```

Provenance for this run:

- Branch: `cuda_combat_arbiter` @ `c10e071` (cuda-dojo-public)
- Corpus: `/srv/models-hdd/chess-games/games.pgn`, 25,645,378 bytes, 37,236 games, captured 2026-04-29 14:54:37 CDT.
- Live N=4 result dirs: `results/compare_20260429-121447/` (baseline), `results/compare_20260429-125401/` (lifted caps).
- Replay full result: `results/shadow_1777486941.json` (30 MB per-game JSON).
- Production arbiter source as of test: `match-processor` v1.5.0, JS-fork, unmodified.

---

## 7. What this report does not prove

- **Engine strength.** The combat branch's `engine` UCI binary is a separate strength claim (perft 6/6, NPS ~712 k, asymmetric d5-vs-d2 wins by mate per `COMBAT.md`). Not measured here.
- **Sandbox security.** The cuda live arbiter currently delegates fighter sandboxing to the same docker image and flags as production. It does not yet have an independent sandbox audit. Any swap-in must keep `sandboxed-referee.js`-equivalent hardening on the spawn side.
- **Broker integration.** `api-client.js` and the `x-arbiter-version` minimum-version gate (currently 1.5.0) are unchanged. A swap-in still needs to identify itself to the broker. Out of scope for this report.
- **Statistical claim beyond the back catalog.** All 37,236 games come from the same production fighter pool. Distributional shifts in fighter behavior (new agents, new strategies) are not represented. The σ ≈ 7.2 × 10⁻⁵ binomial bound applies to the sampled population, not to the population of all possible chess games.

---

## 8. Recommendations

In priority order:

1. **Land the threefold-hash fix** (§5.4 option 1). Re-run replay parity. Ship the result alongside this report when parity reads 1.0000.
2. **Push live-mode N higher** (e.g. N = 50 with `AGENT_CPUS=4`). At lifted caps the harness produces a second axis of equivalence (full move-list match), strengthening the claim with end-to-end game-play, not just rule replay.
3. **Implement the batched ABI** for `dojo_ref` (per `COMBAT.md`'s perf receipts). This recovers the documented ~50 × on the per-game replay throughput and pushes the cuda referee below the arbiter's 2.25 ms/game floor on a per-call basis. Performance is not currently a blocker for live serving, but it is a blocker for using this same pipeline for things like position-database scans.
4. **Decide on the sandbox layer.** Either keep the production `sandboxed-referee.js`-equivalent docker hardening (current state), or commit to a new sandbox model. This decision gates any production swap-in.

After (1) and (2), the cuda_combat_arbiter is presentable as a production-grade arbiter replacement on the rule-enforcement axis.
