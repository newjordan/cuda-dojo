# cuda-dojo combat — GPU chess match resolver

Self-contained branch for using cuda-dojo as a **chess match resolver**:
load two engines (or two configurations of the same engine), pit them
against each other, run a parity check against any PGN corpus, and
get verifiable rule-correct outcomes — all chess work executing on
the GPU.

This branch is purposefully narrow. The chess-rules layer + UCI engine
binary + the FFI wrappers (Rust crate, Python module) are the whole
deliverable. No metric framework experiments, no training pipelines,
no research half-products. Just: **GPU rules, GPU search, GPU eval —
play matches, resolve outcomes.**

---

## What works today

| capability | proof |
|---|---|
| GPU chess rules | `make engine_test_perft` → 6/6 canonical perft positions to depth 4 (10.7M nodes verified) |
| GPU search | `engine` UCI binary, alpha-beta + TT + PeSTO, NPS ~712k |
| GPU game-play | `make dojo_ref_play` runs a Rust → GPU self-play game to terminal |
| Two-opponent match | `make dojo_ref_match` with `DOJO_W_DEPTH=5 DOJO_B_DEPTH=2`: 87 plies → checkmate (white wins) |
| Python access | `import dojo_ref; dojo_ref.Engine().search_depth(p, 5)` |
| Parity vs python-chess | `parity_compare.py --n 500` → 500/500 agreement, σ=0 |
| Multi-GB PGN handling | streaming + reservoir sampling supports the 12.6 GB elite-master corpus |

---

## Layered architecture

```
                                       Python users
                                          ↓
                                  (pyo3 / maturin)
                                          ↓
                                 rust/dojo-ref-py   →   `import dojo_ref`
                                          ↓
                                  rust/dojo-ref         (typed Rust API)
                                          ↓
                                  (cargo build + FFI)
                                          ↓
                                 librefcuda.so          (extern "C" ABI)
                                          ↓
                                cuda/engine/refcuda.cu  (kernel launchers)
                                          ↓
                                  CUDA kernels:
                                  movegen, make_unmake,
                                  attacks, search, eval,
                                  coord3d
                                          ↓
                                       GPU
```

**Rule of the road for this branch:** every chess decision (legal moves,
make_move, in_check, search, eval) executes inside a CUDA kernel. The
host code (C, Rust, Python) does kernel orchestration + type
conversion only. Replace any layer you don't want; the lower layers
keep working.

---

## Build chain

```bash
# 1. Engine + GPU referee shared library
cd cuda/engine
make engine               # UCI engine binary (search/eval/movegen)
make engine_test_perft    # rules-correctness test → must show 0 failures
make librefcuda           # librefcuda.so for FFI callers

# 2. Rust crate + Python module
make dojo_ref             # cargo build --release on the workspace
make dojo_ref_test        # smoke: Rust → GPU primitives
make dojo_ref_play        # Rust GPU self-play game to terminal

# 3. Python (pyo3) wheel
cd ../../rust/dojo-ref-py
maturin develop --release
python3 -c "import dojo_ref; print(dojo_ref.Engine().search_depth(dojo_ref.Position.startpos(), 5))"
```

---

## Use cases

### Run a real chess match between two engines

```bash
# As UCI engines (any UCI-speaking binary)
python3 cuda/engine/gpu_arena_min.py \
    --white  cuda/engine/engine \
    --black  /path/to/your/engine_binary \
    --spec   "depth 5" \
    --max-plies 200
```

### Verify a PGN corpus against the GPU rules (parity check)

```bash
# 100 random elite-master games from a PGN, replay every move through
# both python-chess and the GPU referee, sigma report.
python3 cuda/engine/parity_compare.py \
    --pgn /your/games.pgn \
    --n 500 --max-scan 50000
```

Receipt from this branch's tests on elite TWIC archive:
```
games:        500
CPU solved:   500/500 = 1.0000 (σ=0.0000)
GPU solved:   500/500 = 1.0000 (σ=0.0000)
agreement:    1.0000 (σ=0.0000)
divergences:  0
```

### Use the Rust API in your own engine harness

```rust
use dojo_ref::{Engine, Position};

let engine = Engine::new();
let pos = Position::startpos();
let (best_move, score_cp) = engine.search_depth(&pos, 6).unwrap();
println!("engine plays: {} (eval {} cp)", best_move.to_uci(), score_cp);
```

### Use the Python API

```python
import dojo_ref

p = dojo_ref.Position.startpos()
e = dojo_ref.Engine()
mv, sc = e.search_depth(p, 6)
print(f"engine plays: {mv} (eval {sc} cp)")

# Or run an asymmetric match (different depths per side):
#   DOJO_W_DEPTH=6 DOJO_B_DEPTH=3 cargo run --release \
#     --manifest-path rust/Cargo.toml --example match
```

---

## What's deliberately NOT in this branch

- **Training / supervised learning code.** Belongs upstream.
- **Omnifold metric experiments (CFX, coord3d ablations).** They live on `main`. The CFX d=1 score override (`bc37901`) is a known regression; this branch keeps it on `main` and may be reverted on a future `combat` iteration.
- **chess_train integration code.** That repo handles the data pipeline.

This branch is the kernel of the system: rules, search, eval, FFI. Everything you'd publish if someone asked "show me how cuda-dojo runs a chess match."

---

## Performance receipts

| metric | value |
|---|---|
| Perft 6/6 to depth 4 | 10,746,536 nodes verified |
| Search NPS at startpos | ~712k nodes/second |
| Self-play game depth 5, 60 plies | 27.34s wall (~450 ms/move) |
| Asymmetric match d5 vs d2, 87 plies → mate | 15.44s wall |
| Parity replay 500 elite-master games | 28.91s wall (~58 ms/game incl. FFI) |
| Per-FFI call overhead | ~240 µs legal_moves, ~384 µs make_move |

The per-call FFI overhead is the throughput ceiling for the current
unbatched ABI. A batched API (one kernel call per N positions) would
recover ~50× and is the obvious next polish if/when this branch sees
production use.

---

## Branch policy

`combat` is meant to track the *clean shippable subset* of cuda-dojo:
the rules + search + FFI you can hand a third party. New experimental
work (training, metric research, etc.) lives on `main` and other
branches. Merges back into `combat` only when the work is
shippable-clean — perft passes, FFI smoke passes, no regressions in
parity tests.
