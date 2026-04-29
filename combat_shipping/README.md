# cuda-dojo combat_shipping — arbiter shadow comparison

Shippable, self-contained module that demonstrates `cuda-dojo` (combat
branch) acting as a **drop-in replacement** for the AgentChess
`match-processor` arbiter. The package is read-only with respect to
both upstream sources — it reads them to discover the I/O contract,
then runs a 1:1 shadow comparison.

This directory is the third-party-clonable artifact. It contains:

```
combat_shipping/
├── README.md                       (this file)
├── bridges/
│   ├── arbiter_referee.mjs         CPU ground-truth referee adapter
│   └── cuda_referee.py             GPU referee adapter (dojo_ref)
├── live_arbiter/                   *** LIVE GAME SERVING (the real ask) ***
│   ├── gpu_state.py                Host-side bookkeeping over GPU position
│   ├── live_match.py               Run ONE live game, GPU referee, Docker fighters
│   └── run_arbiter_native.mjs      Run ONE live game via production playGame()
├── live_compare/                   1:1 head-to-head harness
│   ├── compare.py                  Pair fighters, run BOTH arbiters, diff
│   └── run.sh                      One-command orchestration
├── harness/                        (older replay-mode harness; not live serving)
│   ├── pgn_stream.py
│   ├── replay_shadow.py
│   └── run_comparison.sh
└── results/                        Output dir for run reports
```

**The live arbiter (`live_arbiter/` + `live_compare/`) is what produces a
real apples-to-apples comparison: same fighters, fresh game, JS sandbox
unchanged, only the referee swapped between CPU (`chess-engine.js`) and
GPU (`dojo_ref`).**

---

## What the AgentChess arbiter actually does

(Discovered by reading `/home/frosty40/AgentChess_Server/match-processor/src/`.)

- Fighters are arbitrary **JS or Python source files** uploaded via the
  broker API (`api-client.js`). Each one is a single program that:
  1. reads a FEN string from stdin,
  2. writes a single UCI move to stdout,
  3. exits.
- `orchestrator.js` polls the broker for jobs, each job gives a pair of
  fighters and a `gamesPlanned` count.
- `sandboxed-referee.js` runs each fighter inside a docker container
  (`--network none`, `--read-only`, 0.5 CPU, 256m memory, 5500 ms
  per-move budget). The arbiter's host-side chess engine
  (`chess-engine.js`) generates legal moves, validates the fighter's
  UCI string, applies it, and checks for terminal states (mate,
  stalemate, threefold, 50-move, insufficient material, max plies =
  500).
- `pgn-builder.js` writes one PGN game per loop iteration to
  `/srv/models-hdd/chess-games/games.pgn`. Headers: Event, Site, Date,
  White, Black, Result, Termination. Movetext is SAN.

**The interface the cuda combat module has to mimic** is:
`(white_fighter_program, black_fighter_program) → PGN`.

For shadow comparison we cannot fetch the production fighter code blobs
(they are private uploads, not in the PGN), so we cannot re-run the
exact same matchups end-to-end against the GPU engine. What we CAN do
is verify the rule-enforcement layer at parity, which is the part the
combat branch is shipping. Two complementary modes:

### Mode A — Referee-parity replay (default)

Take arbiter-produced PGN games (36,926 in
`/srv/models-hdd/chess-games/games.pgn` as of writing) and replay every
move through:

1. **Arbiter referee** (`bridges/arbiter_referee.mjs`): Node script
   that imports `match-processor/src/chess-engine.js` *without
   modifying it* and replays the SAN/UCI move list through
   `parseFen → generateLegalMoves → applyUciMove`, plus terminal
   detection via `isInCheck / insufficientMaterial / 50-move /
   threefold`.
2. **CUDA referee** (`bridges/cuda_referee.py`): Python wrapper around
   `dojo_ref.Position` (GPU-backed via `librefcuda.so`). Same move
   list, same terminal checks via `is_checkmate / is_stalemate`.

Outcome: if every move is legal per both referees AND both reach the
same terminal verdict that matches the recorded `Result + Termination`
header, the cuda combat referee is at parity for that game.

### Mode B — GPU engine self-play (optional supplement)

Run the cuda `engine` UCI binary as both fighters at varied depths,
shadow-arbitrate each game with both referees. Same agreement metrics.
This is what the combat branch can actually deliver as a *replacement
arbiter* (engine + referee on GPU).

We cannot run Mode B *through the actual arbiter* because the arbiter
expects fighter source code blobs in a docker sandbox; the cuda engine
is a binary, not a docker-uploaded JS/Py program. To make engines
arbiter-compatible we would have to wrap each side in a small Python
fighter that shells to the engine binary — that's a future port step,
not part of this shadow comparison.

---

## Build chain (prerequisites)

This module assumes the cuda-dojo combat branch is already built:

```bash
cd /home/frosty40/VC_V1/cuda-dojo-public
# 1. UCI engine + GPU referee shared library
cd cuda/engine && make engine librefcuda
# 2. Python module
cd ../../rust/dojo-ref-py && maturin develop --release
# 3. Verify
python3 -c "import dojo_ref; print(dojo_ref.Position.startpos().legal_moves()[:3])"
```

Node.js (>= 18) for the arbiter-referee bridge:

```bash
node --version  # v18+ required for import.meta.url
```

---

## Run the LIVE comparison (the real ask)

The "shadow" mode below is REFEREE-REPLAY only — it does not run live
games, it only re-validates pre-played PGNs. For an actual 1:1
side-by-side on speed and accuracy, use the live arbiter:

```bash
cd /home/frosty40/VC_V1/cuda-dojo-public/combat_shipping
bash live_compare/run.sh --n 8                   # default ply cap 120
bash live_compare/run.sh --n 50 --max-plies 240  # bigger / longer games
```

This picks N pairings from `match-processor/data/`, plays each pair
through BOTH arbiters (production CPU referee, then GPU referee),
captures result/reason/plies/move-list/wall-clock, and prints a
divergence summary. Output: `results/compare_<timestamp>/`.

`live_arbiter/live_match.py` is the live runner. It reuses
`bridges/cuda_referee.py`'s GPU position state (dojo_ref) and adds a
`HostState` for the bookkeeping the dojo_ref ABI doesn't expose
(castling/ep/clocks for FEN; threefold history; insufficient material).
Docker-sandbox spawn matches `sandboxed-referee.js` byte-for-byte. NO
chess decisions are made on the host — every legality check, every
make-move, every check/mate/stalemate verdict is a `dojo_ref` call.

## Run the older shadow (replay) comparison

```bash
cd /home/frosty40/VC_V1/cuda-dojo-public/combat_shipping
bash harness/run_comparison.sh                  # default: replay mode, N=500
bash harness/run_comparison.sh --n 1000         # bigger sample
bash harness/run_comparison.sh --pgn /path.pgn  # custom corpus
```

Output is written to `results/shadow_<timestamp>.json` and a human
summary is printed to stdout. **Actual measured numbers** from
2026-04-29 against `/srv/models-hdd/chess-games/games.pgn`
(36,926 production-arbiter games):

```
N=500, seed=42:
  *** REFEREE PARITY: 500/500 = 1.0000 (σ=0.0000) ***
      (both bridges produced identical (ok, terminal, winner) triples)
  legality agreement:   500/500 = 1.0000
  same terminal kind:   500/500 = 1.0000
  outcome match:        500/500 = 1.0000
  replayable subset:    244/244 = 1.0000  (mate/stalemate-ending PGNs)
  arbiter terminals:    {checkmate:239, threefold:177, undecided:73,
                         stalemate:5, insufficient:5, fifty:1}
  cuda terminals:       {checkmate:239, threefold:177, undecided:73,
                         stalemate:5, insufficient:5, fifty:1}

  arbiter replay:    1.56s total ( 3.12 ms/game,  1,153,203 games/hr)
  cuda    replay:   30.32s total (60.63 ms/game,     59,375 games/hr)
  ratio:            cuda is 19.4x the arbiter's per-game replay time

N=1000, seed=7 (replication on a different sample):
  *** REFEREE PARITY: 1000/1000 = 1.0000 (σ=0.0000) ***
  legality agreement:   1000/1000 = 1.0000
  outcome match:        1000/1000 = 1.0000
  replayable subset:    500/500 = 1.0000
  arbiter:  2.89s (2.89 ms/game)  cuda: 59.05s (59.05 ms/game) — 20.4x
```

The `header_result_match_rate` is ~0.85, NOT a referee divergence:
the remaining ~15% are games that ended for non-replayable reasons
(fighter timeout, crash, OOM, or termination=`undecided` in the PGN
because gamesPlanned > plies-played). Both referees report the same
non-terminal verdict; the arbiter's runtime ended the game using
information outside the move list. The `replayable_subset` metric
isolates the rule-detectable endings (mate / stalemate) and shows
244/244 perfect agreement at N=500, 500/500 at N=1000.

The throughput numbers above are **referee-only replay throughput**
(ms/game to validate moves), not full-game-with-fighter throughput.
End-to-end games-per-hour is dominated by the 5500 ms per-move fighter
budget, which is the same on both sides. Replacing the referee does
not change end-to-end throughput unless the new referee is the
bottleneck (the current one is not).

---

## What "1:1 on speed and accuracy" means in this comparison

| dimension | what we measure | how | result (N=1000) |
|---|---|---|---|
| **rule correctness** | every move legal-per-arbiter ⇔ legal-per-cuda | both replay same UCI list | 1000/1000 |
| **terminal detection** | both flag the same game-end (mate/stale/draw) | check is_checkmate/is_stalemate + threefold/fifty/insufficient host hash | 1000/1000 |
| **header match** | recorded PGN Result equals what both referees compute | compare reconstructed result vs PGN Result tag | 877/1000 (others ended via timeout/crash, both bridges agree the position is non-terminal) |
| **replayable terminations** | when PGN says checkmate or stalemate, both compute same | filter PGN[Termination] in {checkmate,stalemate}, compare | 500/500 |
| **per-game replay latency** | total ms for one arbiter-produced game | wall-clock of replay loop | arbiter 2.89 ms, cuda 59.05 ms |
| **referee throughput** | games/hour each bridge can validate | 3.6e6 / ms_per_game | arbiter 1.24M/hr, cuda 60.9k/hr |

A "1:1" claim requires:
- referee parity == 1.0 over a meaningful N (we hit 1.0 at N=1000),
- replayable-subset match == 1.0 (we hit it at N=500 and N=1000),
- cuda referee latency within ~50× of arbiter referee on the
  current per-call FFI ABI (we measure 19–20×; the ceiling is the
  ~240µs/legal_moves + ~384µs/make_move FFI cost documented in
  `COMBAT.md`'s perf receipts).

**Speed caveat**: the cuda referee is *slower per replay* than the
arbiter referee. This is the FFI-per-call overhead, not a CUDA
limitation. The combat README documents a batched ABI as the obvious
~50× polish to recover. End-to-end games-per-hour is dominated by the
5500 ms per-move fighter budget (same on both sides), so referee
latency does not bottleneck production throughput at current move
budgets.

---

## What is NOT proven by this shadow comparison

- **Engine strength.** This compares referees, not players. The
  combat branch's `engine` binary is a separate strength claim
  (perft 6/6, NPS ~712k, asymmetric d5-vs-d2 wins by mate).
- **Sandboxing security.** The arbiter runs untrusted fighter code in
  docker. The cuda combat module does not yet have a sandbox layer;
  to ship as a full arbiter replacement we would still need
  `sandboxed-referee.js`'s docker hardening.
- **Broker integration.** `api-client.js` talks to chessagents.ai's
  job queue. Out of scope for a referee comparison.

What this DOES prove: if the arbiter were to delegate all
chess-rule + terminal-detection work to `dojo_ref`, every game in
production would resolve identically (within the agreement bound
above), and the per-move latency hit is the FFI overhead, not a
correctness regression.
