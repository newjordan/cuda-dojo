#!/usr/bin/env python3
"""validate_frostmatrix_parity.py — NumPy CPU forward pass vs CUDA kernel.

Loads trained .bin weights, runs both CPU (NumPy) and GPU (CUDA binary)
forward passes on test boards, and compares policy/value outputs.

Usage:
  python3 validate_frostmatrix_parity.py

Requires compiled: cuda/validate_frostmatrix
Weights from: /srv/models-hdd/chess-games/training_runs/v3_frostmatrix/export/
"""
import os
import struct
import subprocess
import sys
import numpy as np

# ── Constants (must match frostmatrix_eval.cuh and train_v3_frostmatrix.py) ──
D_MODEL        = 128
D_HEAD         = 32
N_HEADS        = 4
D_FFN          = 341
N_LAYERS       = 4
N_FAMILIES     = 33
N_UNKNOWNS     = 32
N_FAM_NODES    = 65
N_BOARD_SQ     = 64
SEQ_LEN        = 129
N_HW_DIRS      = 3
D_HW_HEAD      = 32
PIECE_VOCAB    = 13
POLICY_DIM     = 64

WEIGHTS_DIR  = "/srv/models-hdd/chess-games/training_runs/v3_frostmatrix/export"
CUDA_BIN     = os.path.join(os.path.dirname(__file__), "validate_frostmatrix")

TOLERANCE = 1e-4  # FP32 accumulated error

# ═══════════════════════════════════════════════════════════════════════════
# Weight loading
# ═══════════════════════════════════════════════════════════════════════════

def load_bin(name):
    path = os.path.join(WEIGHTS_DIR, name)
    with open(path, "rb") as f:
        return np.frombuffer(f.read(), dtype=np.float32).copy()

weights = {}

def load_weights():
    global weights
    w = weights

    w["token_embed"]     = load_bin("token_embed.bin").reshape(PIECE_VOCAB, D_MODEL)       # [13, 128]
    w["board_pos_embed"] = load_bin("board_pos_embed.bin").reshape(N_BOARD_SQ, D_MODEL)    # [64, 128]
    w["policy_from_W"]   = load_bin("policy_from_W.bin").reshape(POLICY_DIM, D_MODEL)      # [64, 128]
    w["policy_to_W"]     = load_bin("policy_to_W.bin").reshape(POLICY_DIM, D_MODEL)        # [64, 128]
    w["value_W"]         = load_bin("value_W.bin").reshape(1, D_MODEL)                     # [1, 128]
    w["value_b"]         = load_bin("value_b.bin")                                         # [1]

    for l in range(N_LAYERS):
        for name in ["Wq","Wk","Wv","Wo","ffn_up","ffn_gate","ffn_down","ln1_w","ln1_b","ln2_w","ln2_b"]:
            key = f"{name}_{l}"
            data = load_bin(f"{name}_{l}.bin")
            if name in ("Wq","Wk","Wv","Wo"):
                w[key] = data.reshape(D_MODEL, D_MODEL)        # [128, 128] — PyTorch Linear weight: (out, in)
            elif name in ("ffn_up","ffn_gate"):
                w[key] = data.reshape(D_FFN, D_MODEL)          # [341, 128]
            elif name == "ffn_down":
                w[key] = data.reshape(D_MODEL, D_FFN)          # [128, 341]
            elif name in ("ln1_w","ln2_w"):
                w[key] = data                                  # [128]
            elif name in ("ln1_b","ln2_b"):
                w[key] = data                                  # [128]

    # Family axis
    w["family_embed"]   = load_bin("fax_family_embed.bin").reshape(N_FAMILIES, D_MODEL)
    w["unknown_embed"]  = load_bin("fax_unknown_embed.bin").reshape(N_UNKNOWNS, D_MODEL)
    w["smooth_weights"] = load_bin("fax_smooth_weights.bin").reshape(N_FAMILIES, N_FAMILIES)

    # Highway per-direction
    for d in range(N_HW_DIRS):
        for proj in ["Wq","Wk","Wv"]:
            key = f"hw_{proj}_{d}"
            w[key] = load_bin(f"fax_hw_{proj}_{d}.bin").reshape(D_HW_HEAD, D_MODEL)   # [32, 128]
    w["hw_Wo"] = load_bin("fax_hw_Wo.bin").reshape(D_MODEL, N_HW_DIRS * D_HW_HEAD)  # [128, 96]

    # Cross-attention
    for proj in ["Wq","Wk","Wv","Wo"]:
        w[f"ca_{proj}"] = load_bin(f"fax_ca_{proj}.bin").reshape(D_MODEL, D_MODEL)   # [128, 128]

    print(f"[load] {len(w)} weight tensors from {WEIGHTS_DIR}")
    return w

# ═══════════════════════════════════════════════════════════════════════════
# NumPy CPU forward pass — exact mirror of frostmatrix_eval.cuh
# ═══════════════════════════════════════════════════════════════════════════

def silu(x):
    return x / (1.0 + np.exp(-x))

def layer_norm(x, w, b):
    """x: [N, D], w: [D], b: [D] — exact match to fm_layernorm_kernel"""
    mean = x.mean(axis=-1, keepdims=True)
    var  = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    inv  = 1.0 / np.sqrt(var + 1e-5)
    return (x - mean) * inv * w + b

def softmax(x, axis=-1):
    mx = x.max(axis=axis, keepdims=True)
    ex = np.exp(x - mx)
    return ex / (ex.sum(axis=axis, keepdims=True) + 1e-8)

def cpu_forward(tokens):
    """Tokens: [64] int, piece indices 0..12. Returns (from_probs [64], to_probs [64], value float)."""
    w = weights
    B = 1  # single board
    F = N_FAM_NODES

    # ── Step 1: Build family axis ────────────────────────────────────────
    fam = np.zeros((F, D_MODEL), dtype=np.float32)
    for i in range(N_FAMILIES):
        fam[2*i] = w["family_embed"][i]        # even = family
    for i in range(N_UNKNOWNS):
        fam[2*i + 1] = w["unknown_embed"][i]   # odd = unknown

    # ── Step 2: Embed board tokens + positional encoding ──────────────────
    board = w["token_embed"][tokens] + w["board_pos_embed"]   # [64, 128]

    # ── Step 3: Geo projection — DISABLED ──

    # ── Step 4: Highway attention over family axis ────────────────────────
    for dir_idx in range(N_HW_DIRS):
        # Q, K, V projections
        q = fam @ w[f"hw_Wq_{dir_idx}"].T   # [65, 32]
        k = fam @ w[f"hw_Wk_{dir_idx}"].T   # [65, 32]
        v = fam @ w[f"hw_Wv_{dir_idx}"].T   # [65, 32]

        # Scores with direction mask + smoothing bias
        scores = q @ k.T / np.sqrt(D_HW_HEAD)   # [65, 65]

        for qi in range(F):
            for ki in range(F):
                qi_fam = (qi % 2 == 0)
                ki_fam = (ki % 2 == 0)
                qi_id = qi // 2
                ki_id = ki // 2
                allowed = False
                if dir_idx == 0:
                    allowed = qi_fam and ki_fam
                elif dir_idx == 1:
                    if qi_fam and not ki_fam:
                        allowed = (ki_id == qi_id - 1 or ki_id == qi_id) and ki_id >= 0 and ki_id < N_UNKNOWNS
                    elif not qi_fam and ki_fam:
                        allowed = (ki_id == qi_id or ki_id == qi_id + 1) and ki_id < N_FAMILIES
                else:  # dir_idx == 2
                    if qi_fam and not ki_fam:
                        skip = abs(ki_id - qi_id)
                        allowed = (skip == 2 or skip == 3) and ki_id >= 0 and ki_id < N_UNKNOWNS
                if not allowed:
                    scores[qi, ki] = -1e9

        # Apply lateral smoothing bias for dir=0 family→family
        if dir_idx == 0:
            for qi in range(F):
                if qi % 2 == 0:   # family
                    for ki in range(F):
                        if ki % 2 == 0:  # family
                            scores[qi, ki] += np.log(w["smooth_weights"][qi//2, ki//2] + 1e-8)

        # Softmax + weighted value
        attn = softmax(scores, axis=-1)   # [65, 65]
        v_weighted = attn @ v             # [65, 32]
        # Store in place of q (like CUDA reuses hw_q buffer)
        if dir_idx == 0:
            hw_out0 = v_weighted
        elif dir_idx == 1:
            hw_out1 = v_weighted
        else:
            hw_out2 = v_weighted

    # Combine highway directions
    hw_combined = np.concatenate([hw_out0, hw_out1, hw_out2], axis=-1)  # [65, 96]
    hw_delta = hw_combined @ w["hw_Wo"].T                                # [65, 128]
    fam = fam + hw_delta

    # ── Step 5: Board → Family cross-attention ───────────────────────────
    q_ca = board @ w["ca_Wq"].T   # [64, 128]
    k_ca = fam @ w["ca_Wk"].T     # [65, 128]
    v_ca = fam @ w["ca_Wv"].T     # [65, 128]
    scores = q_ca @ k_ca.T / np.sqrt(D_MODEL)  # [64, 65]
    attn_ca = softmax(scores, axis=-1)          # [64, 65]
    ca_out = attn_ca @ v_ca                      # [64, 128]
    ca_proj = ca_out @ w["ca_Wo"].T              # [64, 128]
    board = board + ca_proj

    # ── Step 6: Concatenate into full sequence ───────────────────────────
    x = np.concatenate([fam, board], axis=0)   # [129, 128]

    # ── Step 7: Transformer layers ──────────────────────────────────────────
    for l in range(N_LAYERS):
        # LayerNorm 1 → Multi-head self-attention → residual
        x_norm = layer_norm(x, w[f"ln1_w_{l}"], w[f"ln1_b_{l}"])   # [129, 128]

        # QKV projections (using PyTorch weights: shape [128,128] = (out,in))
        Q = x_norm @ w[f"Wq_{l}"].T   # [129, 128]  (input @ W^T for Linear)
        K = x_norm @ w[f"Wk_{l}"].T
        V = x_norm @ w[f"Wv_{l}"].T

        # Reshape to multi-head: [129, 128] → [4, 129, 32]
        Q = Q.reshape(SEQ_LEN, N_HEADS, D_HEAD).transpose(1, 0, 2)  # [4, 129, 32]
        K = K.reshape(SEQ_LEN, N_HEADS, D_HEAD).transpose(1, 0, 2)
        V = V.reshape(SEQ_LEN, N_HEADS, D_HEAD).transpose(1, 0, 2)

        # Scaled dot-product attention
        attn_scores = Q @ K.transpose(0, 2, 1) / np.sqrt(D_HEAD)  # [4, 129, 129]
        attn_probs = softmax(attn_scores, axis=-1)
        attn_out = attn_probs @ V                                    # [4, 129, 32]

        # Merge heads → output projection → residual
        attn_out = attn_out.transpose(1, 0, 2).reshape(SEQ_LEN, D_MODEL)  # [129, 128]
        attn_out = attn_out @ w[f"Wo_{l}"].T                         # [129, 128]
        x = x + attn_out

        # LayerNorm 2 → SwiGLU FFN → residual
        x_norm = layer_norm(x, w[f"ln2_w_{l}"], w[f"ln2_b_{l}"])
        up   = x_norm @ w[f"ffn_up_{l}"].T    # [129, 341]
        gate = x_norm @ w[f"ffn_gate_{l}"].T  # [129, 341]
        h = up * silu(gate)
        # Smeargate (geometric mean with neighbor)
        h_shifted = np.roll(h, -1, axis=-1)
        h_shifted[:, -1] = h[:, -1]  # boundary: self
        smear = np.where(h * h_shifted > 0, np.sqrt(np.abs(h * h_shifted)) * np.sign(h), np.zeros_like(h))
        ffn_out = smear @ w[f"ffn_down_{l}"].T   # [129, 128]
        x = x + ffn_out

    # ── Step 8: Policy heads + value head ──────────────────────────────────
    pooled = x[N_FAM_NODES:, :].mean(axis=0)   # [128] — mean pool board tokens
    from_logits = pooled @ w["policy_from_W"].T   # [64] — raw logits
    to_logits   = pooled @ w["policy_to_W"].T     # [64]

    # Apply softmax (matches CUDA kernel which does in-place softmax)
    from_probs = softmax(from_logits.reshape(1, -1)).flatten()
    to_probs   = softmax(to_logits.reshape(1, -1)).flatten()

    value = np.tanh(np.dot(pooled, w["value_W"].flatten()) + w["value_b"][0])

    return from_probs.astype(np.float64), to_probs.astype(np.float64), float(value)

# ═══════════════════════════════════════════════════════════════════════════
# GPU runner
# ═══════════════════════════════════════════════════════════════════════════

def run_cuda(tokens):
    """Run CUDA validation binary, return (from_probs [64], to_probs [64], value float)."""
    input_str = " ".join(str(int(t)) for t in tokens) + "\n"
    proc = subprocess.run(
        [CUDA_BIN, WEIGHTS_DIR],
        input=input_str,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        timeout=30,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"CUDA binary failed: {proc.stderr.strip()}")
    lines = proc.stdout.strip().split("\n")
    from_probs = np.array([float(x) for x in lines[0].split()], dtype=np.float64)
    to_probs   = np.array([float(x) for x in lines[1].split()], dtype=np.float64)
    value      = float(lines[2].strip())
    return from_probs, to_probs, value

# ═══════════════════════════════════════════════════════════════════════════
# Test boards
# ═══════════════════════════════════════════════════════════════════════════

def starting_position():
    """Returns gpu_forge pieces [64] for starting position: a1..h8.
    gpu_forge: 0=empty, white: P=1,N=2,B=3,R=4,Q=5,K=6, black: p=8,n=7,b=10,r=9,q=11,k=13"""
    tokens = np.zeros(64, dtype=np.int32)
    # White back rank (a1..h1): R=4,N=2,B=3,Q=5,K=6,B=3,N=2,R=4
    back_white = [4, 2, 3, 5, 6, 3, 2, 4]
    # White pawns (a2..h2): P=1
    pawns_white = [1]*8
    # Black pawns (a7..h7): p=8
    pawns_black = [8]*8
    # Black back rank (a8..h8): r=9,n=7,b=10,q=11,k=13,b=10,n=7,r=9
    back_black = [9, 7, 10, 11, 13, 10, 7, 9]

    for i, p in enumerate(back_white):   tokens[i] = p
    for i, p in enumerate(pawns_white):  tokens[8+i] = p
    for i, p in enumerate(pawns_black):  tokens[48+i] = p
    for i, p in enumerate(back_black):   tokens[56+i] = p
    return tokens

def gpu_to_fm(pieces):
    """Convert gpu_forge piece encoding to FrostMatrix tokens.
    gpu_forge: 0=empty, 1-6=white(P,N,B,R,Q,K), black non-sequential (p=8,n=7,b=10,r=9,q=11,k=13)
    FrostMatrix: 0=empty, 1-6=white, 7-12=black(P,N,B,R,Q,K)"""
    gp_to_fm = {0:0,1:1,2:2,3:3,4:4,5:5,6:6,8:7,7:8,10:9,9:10,11:11,13:12}
    out = np.zeros_like(pieces)
    for i, p in enumerate(pieces):
        out[i] = gp_to_fm.get(int(p), 0)
    return out

# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    global weights
    print("=" * 70)
    print("FrostMatrix V3 CUDA vs CPU Parity Validation")
    print("=" * 70)

    # Load weights
    print("\n[1] Loading weights...")
    weights = load_weights()

    # Check CUDA binary
    if not os.path.exists(CUDA_BIN):
        print(f"\n[ERROR] CUDA binary not found: {CUDA_BIN}")
        print("  Build: nvcc -arch=native -O3 validate_frostmatrix.cu -o validate_frostmatrix -lcublas")
        sys.exit(1)

    # Test boards (gpu_forge encoding sent to CUDA binary; CPU uses FrostMatrix tokens)
    test_boards = [
        ("Starting position", starting_position()),
        ("Empty board",       np.zeros(64, dtype=np.int32)),
    ]

    all_pass = True
    for name, gpu_pieces in test_boards:
        print(f"\n[2] Testing: {name}")
        print(f"    Pieces: {list(gpu_pieces[:16])}...")

        # CPU forward (needs FrostMatrix tokens)
        fm_tokens = gpu_to_fm(gpu_pieces)
        cpu_from, cpu_to, cpu_value = cpu_forward(fm_tokens)
        print(f"    CPU:  value={cpu_value:.6f}  from_sq max={cpu_from.argmax()} ({cpu_from.max():.4f})  to_sq max={cpu_to.argmax()} ({cpu_to.max():.4f})")

        # GPU forward (takes gpu_forge pieces, converts internally)
        try:
            gpu_from, gpu_to, gpu_value = run_cuda(gpu_pieces)
            print(f"    GPU:  value={gpu_value:.6f}  from_sq max={gpu_from.argmax()} ({gpu_from.max():.4f})  to_sq max={gpu_to.argmax()} ({gpu_to.max():.4f})")
        except Exception as e:
            print(f"    GPU:  FAILED — {e}")
            all_pass = False
            continue

        # Compare
        from_diff = np.abs(cpu_from - gpu_from)
        to_diff   = np.abs(cpu_to - gpu_to)
        val_diff  = abs(cpu_value - gpu_value)

        from_max = from_diff.max()
        to_max   = to_diff.max()
        from_mean = from_diff.mean()
        to_mean   = to_diff.mean()

        from_pass = from_max < TOLERANCE
        to_pass   = to_max   < TOLERANCE
        val_pass  = val_diff  < TOLERANCE

        print(f"    from_sq: max_diff={from_max:.2e} mean_diff={from_mean:.2e} {'✓' if from_pass else '✗'}")
        print(f"    to_sq:   max_diff={to_max:.2e} mean_diff={to_mean:.2e} {'✓' if to_pass else '✗'}")
        print(f"    value:   diff={val_diff:.2e} {'✓' if val_pass else '✗'}")

        if not (from_pass and to_pass and val_pass):
            all_pass = False
            # Show worst mismatches
            if not from_pass:
                worst = from_diff.argmax()
                print(f"    WORST from_sq[{worst}]: cpu={cpu_from[worst]:.8f} gpu={gpu_from[worst]:.8f}")
            if not to_pass:
                worst = to_diff.argmax()
                print(f"    WORST to_sq[{worst}]: cpu={cpu_to[worst]:.8f} gpu={gpu_to[worst]:.8f}")

    print("\n" + "=" * 70)
    if all_pass:
        print("PASS — CUDA and CPU outputs match within tolerance")
    else:
        print("FAIL — differences exceed tolerance")
    print("=" * 70)
    return 0 if all_pass else 1

if __name__ == "__main__":
    sys.exit(main())
