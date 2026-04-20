# VOCAB_SIZE Patch — transformer_v3_frostmatrix.cu

**Date:** 2026-04-20  
**Investigated by:** vocab_bridge_v3.js  
**Status:** patch not yet applied — DO NOT modify the .cu until training code is updated in sync

---

## Problem

`cuda/transformer_v3_frostmatrix.cu` line 62 reads:

```c
#define VOCAB_SIZE      513   // 13 piece types + 500 opening family IDs
```

Both numbers in the comment are wrong, and the single constant conflates two
independent dimensions:

| What it's used for | Current value | Correct value |
|---|---|---|
| `token_embed` rows (piece-type input embedding) | 513 | **13** |
| `policy_W` columns (policy-head output logits) | 513 | **4096** |
| Opening family vocab size (informational) | claimed 500 | **8 463** |

---

## Root cause

The comment describes the V2 architecture where opening family IDs were injected
directly into the token sequence and looked up via `token_embed`.

In V3 the opening context is **structurally separated** into a 65-node family
axis that uses its own weight matrices (`fax.family_embed[33 × D_MODEL]` and
`fax.unknown_embed[32 × D_MODEL]`).  These are **never touched by `token_embed`**.

The V3 token sequence is:

```
positions [0 .. 64]   = 65 family-axis slots
                        built by build_family_axis_kernel from fax weights
                        (the integer token values at these positions are not
                         looked up via token_embed in the current implementation)

positions [65 .. 128] = 64 board-square tokens
                        piece-type IDs 0–12, looked up via token_embed
```

`token_embed` therefore only needs **13 rows** (one per piece type).

The policy head is a separate output projection and has nothing to do with the
input embedding size.  The natural target space for V3 policy prediction is
`from_square × 64 + to_square`, which has **4 096** entries.

---

## Actual vocab numbers (from opening_branch_vocab.json)

Verified by `tools/vocab_bridge_v3.js` against
`/srv/models-hdd/chess-games/opening_branch_vocab.json`:

```
generatedAt:        2026-04-20T03:24:59.718Z
entries.length:     8 463
maxPly:             12
minCount:           8
openingTokenId range: 0 – 8 462
```

Base tokenizer v1 vocab size: **40** (13 piece types + 27 aux tokens, IDs 0–39).  
V2 full vocab size: **8 504** (40 base + 1 pad + 8 463 opening families).

---

## Exact lines to change

### Remove

```c
#define VOCAB_SIZE      513   // 13 piece types + 500 opening family IDs
```

### Replace with

```c
// ── Vocabulary sizes ──────────────────────────────────────────────────────
// In V3 the opening context is carried by the 65-node family axis
// (fax.family_embed / fax.unknown_embed).  token_embed covers board squares only.
#define PIECE_VOCAB_SIZE    13    // piece-type IDs for board-square token_embed
                                  // 0=empty, 1-6=PNBRQK (white), 7-12=pnbrqk (black)
#define OPENING_VOCAB_SIZE  8463  // opening family entries in opening_branch_vocab.json
                                  // (reference constant — family axis sized by N_FAMILIES=33)
#define POLICY_OUT_SIZE     4096  // policy head output: from_sq*64 + to_sq
```

### Update all usages of VOCAB_SIZE

After the rename, update every use of `VOCAB_SIZE` in the file:

| Line | Old | New |
|---|---|---|
| `tw->token_embed = alloc_weights(VOCAB_SIZE * D_MODEL, s_e);` | `VOCAB_SIZE` | `PIECE_VOCAB_SIZE` |
| `tw->policy_W = alloc_weights(D_MODEL * VOCAB_SIZE, s_e);` | `VOCAB_SIZE` | `POLICY_OUT_SIZE` |
| `CUDA_CHECK(cudaMalloc(&buf->policy_logits, batch*VOCAB_SIZE*sizeof(float)));` | `VOCAB_SIZE` | `POLICY_OUT_SIZE` |
| `CUBLAS_CHECK(cublasSgemm(…VOCAB_SIZE, batch, D…))` (policy matmul) | `VOCAB_SIZE` | `POLICY_OUT_SIZE` |

There are **4 call sites** to update.  Use grep to verify before applying:

```bash
grep -n "VOCAB_SIZE" cuda/transformer_v3_frostmatrix.cu
```

Current output (6 hits — 1 define + 5 uses):

```
62:  #define VOCAB_SIZE      513
119: float *token_embed;    // [VOCAB_SIZE × D_MODEL]
126: float *policy_W;       // [D_MODEL × VOCAB_SIZE]
220: tw->token_embed = alloc_weights(VOCAB_SIZE * D_MODEL, s_e);
236: tw->policy_W    = alloc_weights(D_MODEL * VOCAB_SIZE, s_e);
282: cudaMalloc(&buf->policy_logits, batch*VOCAB_SIZE*sizeof(float));
696: cublasSgemm(…VOCAB_SIZE, batch…)   // policy logit matmul
```

Lines 119 and 126 are comments — update them too for clarity.

---

## Memory impact

| Buffer | Old size (batch=256) | New size |
|---|---|---|
| `token_embed` | 513 × 128 × 4 B = **262 KB** | 13 × 128 × 4 B = **6.6 KB** |
| `policy_W` | 128 × 513 × 4 B = **262 KB** | 128 × 4096 × 4 B = **2 MB** |
| `policy_logits` | 256 × 513 × 4 B = **525 KB** | 256 × 4096 × 4 B = **4 MB** |

Total GPU memory change: small net increase (~5.5 MB per forward buffer) —
acceptable.  The token_embed savings are negligible; the policy_logits growth is
the dominant term.

---

## Sequence layout reminder (for training code)

```
SEQ_LEN_V3 = 129
  [  0 ..  64 ] = 65 family-axis slots  (65 = N_FAM_NODES)
  [ 65 .. 128 ] = 64 board-square tokens (64 = N_BOARD_SQ)

Family-axis even slots (0, 2, 4, … 64): family nodes  (family_id = slot/2, range 0–32)
Family-axis odd  slots (1, 3, 5, … 63): unknown nodes (unknown_id = slot/2, range 0–31)

Board tokens: piece-type IDs in square order a1→h1→a2→h2→…→a8→h8
  0=empty, 1=P, 2=N, 3=B, 4=R, 5=Q, 6=K, 7=p, 8=n, 9=b, 10=r, 11=q, 12=k
```

Use `tools/vocab_bridge_v3.js` → `encode({ fen, moveHistory })` to produce the
correct 129-token array for any position.
