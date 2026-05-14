/* ═══════════════════════════════════════════════════════════════════════════
 * VOCAB_SIZE PATCH — UPDATED for split policy heads (from_sq + to_sq)
 *
 * ARCHITECTURE:
 *   token_embed: [PIECE_VOCAB_SIZE=13 × D_MODEL=128]    — board piece type tokens
 *   policy_from_W: [D_MODEL=128 × 64]                    — from-square classifier
 *   policy_to_W:   [D_MODEL=128 × 64]                    — to-square classifier
 *   value_W:       [D_MODEL=128 × 1]                     — position value (-1..1)
 *
 * The old flat 4096-way policy head has been replaced with two independent
 * 64-way classifiers. Each square gets its own softmax, giving 64× denser
 * gradient per class (each square appears in ~1/64 of samples vs ~1/4096).
 *
 *   Effective move space: 64 × 64 = 4096 (unchanged)
 *   Policy output:        from_logits [B, 64] + to_logits [B, 64]
 *
 * This matches the Python model in train_v3_frostmatrix.py:
 *   policy_from_W.weight → [64, 128] → saved as policy_from_W.bin
 *   policy_to_W.weight   → [64, 128] → saved as policy_to_W.bin
 * ═══════════════════════════════════════════════════════════════════════════ */

/*
 * transformer_v3_frostmatrix.cu — FrostMatrix Graph Transformer
 *
 * Architecture extends v2 with a full Y-axis family structure:
 *
 *   Y-AXIS (Family Depth Layers):
 *     33 monster family nodes + 32 unknown potential nodes = 65 family axis nodes
 *     Family nodes: initialized from FrostMatrix package centroid features
 *     Unknown potentials: learnable — absorb hybrid/transitional positions that
 *       no known family can explain yet (the "dark matter" of the opening space)
 *
 *   HIGHWAY LINKS (tri-directional, orthogonal, smoothing):
 *     Direction 0 — LATERAL:   family[i] ↔ family[j], weighted by FrostMatrix
 *                               geometric distance (Gaussian kernel smoothing)
 *     Direction 1 — VERTICAL:  family[i] → unknown[i,i+1] → family[i+1]
 *                               (sequential evolution through opening space)
 *     Direction 2 — DIAGONAL:  family[i] → unknown nodes skipping ±3 families
 *                               (long-range structural resonance)
 *
 *   SEQUENCE LAYOUT:
 *     [fam_0 | unk_01 | fam_1 | ... | unk_31-32 | fam_32 | sq_0 | ... | sq_63]
 *        0       1        2              64          65-128
 *     SEQ_LEN_V3 = 65 (family axis) + 64 (board) = 129
 *
 *   BOARD → FAMILY CROSS-ATTENTION:
 *     After highway pass, each board token cross-attends to the full family axis.
 *     This is how the position "finds its family" — dynamically, not by lookup.
 *
 *   GEOMETRY INPUT (FrostMatrix 55-dim per ply):
 *     4 (active fold) + 51 (17 geometry encoders G1-G17) = 55 features
 *     Includes fold variants (G11-G13), pawn chains (G14), pin/xray (G15),
 *     open files (G16), knight outposts (G17).
 *     These are embedded into the board token representations.
 *
 * Compile:
 *   nvcc -arch=native -O3 --use_fast_math -DUSE_FROSTMATRIX_PKG \
 *        -o transformer_v3_frostmatrix transformer_v3_frostmatrix.cu -lcublas
 */

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

/* ═══════════════════════════════════════════════════════════════════════════
 * ARCHITECTURE CONSTANTS
 * ═══════════════════════════════════════════════════════════════════════════ */

#define N_FAMILIES      33
#define N_UNKNOWNS      32    // one between each consecutive family pair
#define N_FAM_NODES     65    // 33 + 32  — the full Y-axis
#define N_BOARD_SQ      64
#define SEQ_LEN_V3      (N_FAM_NODES + N_BOARD_SQ)  // = 129

#define D_MODEL         128
#define D_HEAD          32    // D_MODEL / N_HEADS = 128/4
#define N_HEADS         4
#define N_LAYERS        4
#define D_FFN           341   // SwiGLU reduced FFN (same as v2)
#define PIECE_VOCAB_SIZE  13      // token_embed rows — piece types 0..12
#define POLICY_OUT_DIM   64      // from_sq/to_sq classification (64 squares each)

// Highway-specific
#define N_HW_DIRS       3     // lateral, vertical, diagonal
#define D_HW_HEAD       32    // highway head dim per direction  (3×32=96, projected→128)
#define HW_SMOOTH_SIGMA 2.0f  // Gaussian smoothing bandwidth for lateral highway

// Geometry channels from FrostMatrix composite encoder
#define N_GEO_CHANNELS  55    // 4 base + 51 extended (17 geometry encoders: G1-G17)

#define BATCH_SIZE      256
#define MAX_GAME_PLY    11    // positions 0..10

/* ═══════════════════════════════════════════════════════════════════════════
 * CUDA HELPERS
 * ═══════════════════════════════════════════════════════════════════════════ */

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(_e), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t _s = (call); \
    if (_s != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error %d at %s:%d\n", (int)_s, __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

/* ═══════════════════════════════════════════════════════════════════════════
 * WEIGHT BUFFERS
 * ═══════════════════════════════════════════════════════════════════════════ */

struct FamilyAxisWeights {
    float *family_embed;    // [N_FAMILIES × D_MODEL]   — from FrostMatrix centroids
    float *unknown_embed;   // [N_UNKNOWNS × D_MODEL]   — learnable latent nodes
    float *smooth_weights;  // [N_FAMILIES × N_FAMILIES] — geometric Gaussian kernel
    // Highway projections: Q, K, V per direction
    float *hw_Wq[N_HW_DIRS]; // [D_MODEL × D_HW_HEAD] each
    float *hw_Wk[N_HW_DIRS];
    float *hw_Wv[N_HW_DIRS];
    float *hw_Wo;             // [N_HW_DIRS × D_HW_HEAD × D_MODEL] — output projection
    // Cross-attention: board queries into family axis keys/values
    float *ca_Wq;             // [D_MODEL × D_MODEL]
    float *ca_Wk;
    float *ca_Wv;
    float *ca_Wo;
    // Geometry embedding projection: geo channels → D_MODEL
    float *geo_proj_W;        // [N_GEO_CHANNELS × D_MODEL]
    float *geo_proj_b;        // [D_MODEL]
};

struct TransformerWeights {
    float *token_embed;   // [VOCAB_SIZE × D_MODEL]
    // Per-layer weights (reused from v2 structure)
    float *Wq[N_LAYERS], *Wk[N_LAYERS], *Wv[N_LAYERS], *Wo[N_LAYERS];
    float *ffn_up[N_LAYERS], *ffn_gate[N_LAYERS], *ffn_down[N_LAYERS];
    float *ln1_w[N_LAYERS], *ln1_b[N_LAYERS];
    float *ln2_w[N_LAYERS], *ln2_b[N_LAYERS];
    // Output heads — split policy (from_sq + to_sq, each 64-way)
    // Mirrors Python train_v3_frostmatrix.py: policy_from_W [64, D_MODEL], policy_to_W [64, D_MODEL]
    // Stored column-major in cuBLAS convention: [D_MODEL × 64]
    float *policy_from_W; // [D_MODEL × 64]
    float *policy_to_W;   // [D_MODEL × 64]
    float *value_W;       // [D_MODEL × 1]
    float *value_b;
    // Family axis weights
    FamilyAxisWeights fax;
};

struct ForwardBuffers {
    float *x;           // [BATCH × SEQ_LEN_V3 × D_MODEL]
    float *x_norm;
    float *q, *k, *v;
    float *attn_scores; // [BATCH × N_HEADS × SEQ_LEN_V3 × SEQ_LEN_V3]
    float *attn_out;
    float *ffn_h;
    float *ffn_gate_buf;
    float *hw_q[N_HW_DIRS], *hw_k[N_HW_DIRS], *hw_v[N_HW_DIRS];
    float *hw_attn;     // [BATCH × N_FAM_NODES × N_FAM_NODES]
    float *family_axis; // [BATCH × N_FAM_NODES × D_MODEL] — contiguous family axis scratch
    float *ca_q;        // cross-attention buffers
    float *ca_k, *ca_v, *ca_out;
    int   *canon_pos;   // [BATCH × SEQ_LEN_V3] — fold positions for board tokens
    float *geo_embed;   // [BATCH × N_BOARD_SQ × D_MODEL] — geometry projection
    float *pooled;           // [BATCH × D_MODEL] — pooled board representation for heads
    float *policy_from_logits; // [BATCH × 64] — from-square policy logits
    float *policy_to_logits;   // [BATCH × 64] — to-square policy logits
    float *value_out;
};

/* ═══════════════════════════════════════════════════════════════════════════
 * WEIGHT INITIALIZATION
 * ═══════════════════════════════════════════════════════════════════════════ */

static uint32_t _seed = 0xdeadbeef;
static float randn_next() {
    _seed ^= _seed << 13; _seed ^= _seed >> 17; _seed ^= _seed << 5;
    float u = (_seed >> 1) / (float)(1u << 31);
    float v = ((_seed * 6364136223846793005ULL + 1442695040888963407ULL) >> 1) / (float)(1u << 31);
    return sqrtf(-2.0f * logf(u + 1e-7f)) * cosf(2.0f * 3.14159265f * v);
}

static float* alloc_weights(int n, float scale) {
    float *h = (float*)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) h[i] = randn_next() * scale;
    float *d; CUDA_CHECK(cudaMalloc(&d, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d, h, n * sizeof(float), cudaMemcpyHostToDevice));
    free(h);
    return d;
}

/*
 * Build Gaussian smoothing weights for lateral highway from family centroid
 * distances. If FrostMatrix package is loaded, centroids come from there.
 * Otherwise falls back to uniform (identity-like) weights.
 */
static float* build_smooth_weights(float *family_centroids_cpu, int n_fam) {
    float *w = (float*)calloc(n_fam * n_fam, sizeof(float));
    float sigma2 = HW_SMOOTH_SIGMA * HW_SMOOTH_SIGMA;

    if (family_centroids_cpu) {
        // Gaussian kernel: w[i][j] = exp(-||c_i - c_j||² / (2σ²))
        // centroids are [n_fam × 2] (r, f) coordinates
        for (int i = 0; i < n_fam; i++) {
            float row_sum = 0.0f;
            for (int j = 0; j < n_fam; j++) {
                float dr = family_centroids_cpu[i*2]   - family_centroids_cpu[j*2];
                float df = family_centroids_cpu[i*2+1] - family_centroids_cpu[j*2+1];
                float d2 = dr*dr + df*df;
                w[i*n_fam+j] = expf(-d2 / (2.0f * sigma2));
                row_sum += w[i*n_fam+j];
            }
            for (int j = 0; j < n_fam; j++) w[i*n_fam+j] /= row_sum;  // normalize
        }
    } else {
        // Uniform fallback: each family attends equally to nearby ±3 families
        for (int i = 0; i < n_fam; i++) {
            float row_sum = 0.0f;
            for (int j = 0; j < n_fam; j++) {
                int d = abs(i - j);
                float wij = (d == 0) ? 1.0f : (d <= 3) ? expf(-d*d / (2.0f*sigma2)) : 0.0f;
                w[i*n_fam+j] = wij;
                row_sum += wij;
            }
            for (int j = 0; j < n_fam; j++) w[i*n_fam+j] /= row_sum;
        }
    }

    float *d_w; CUDA_CHECK(cudaMalloc(&d_w, n_fam*n_fam*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_w, w, n_fam*n_fam*sizeof(float), cudaMemcpyHostToDevice));
    free(w);
    return d_w;
}

void init_weights(TransformerWeights *tw, float *frostmatrix_centroids_cpu) {
    float s_e = sqrtf(2.0f / D_MODEL);
    float s_a = sqrtf(2.0f / (D_MODEL + D_HEAD));
    float s_f = sqrtf(2.0f / (D_MODEL + D_FFN));
    float s_hw = sqrtf(2.0f / (D_MODEL + D_HW_HEAD));

    tw->token_embed = alloc_weights(PIECE_VOCAB_SIZE * D_MODEL, s_e);

    for (int l = 0; l < N_LAYERS; l++) {
        tw->Wq[l]       = alloc_weights(D_MODEL * D_MODEL, s_a);
        tw->Wk[l]       = alloc_weights(D_MODEL * D_MODEL, s_a);
        tw->Wv[l]       = alloc_weights(D_MODEL * D_MODEL, s_a);
        tw->Wo[l]       = alloc_weights(D_MODEL * D_MODEL, s_a);
        tw->ffn_up[l]   = alloc_weights(D_MODEL * D_FFN, s_f);
        tw->ffn_gate[l] = alloc_weights(D_MODEL * D_FFN, s_f);
        tw->ffn_down[l] = alloc_weights(D_FFN * D_MODEL, s_f);
        tw->ln1_w[l]    = alloc_weights(D_MODEL, 1.0f);
        tw->ln1_b[l]    = alloc_weights(D_MODEL, 0.0f);
        tw->ln2_w[l]    = alloc_weights(D_MODEL, 1.0f);
        tw->ln2_b[l]    = alloc_weights(D_MODEL, 0.0f);
    }

    // Split policy heads — each 64-way classification
    tw->policy_from_W = alloc_weights(D_MODEL * POLICY_OUT_DIM, s_e);  // [128, 64]
    tw->policy_to_W   = alloc_weights(D_MODEL * POLICY_OUT_DIM, s_e);  // [128, 64]
    tw->value_W       = alloc_weights(D_MODEL, s_e);
    tw->value_b       = alloc_weights(1, 0.0f);

    // Family axis weights
    FamilyAxisWeights &fax = tw->fax;
    fax.family_embed  = alloc_weights(N_FAMILIES * D_MODEL, s_e);
    fax.unknown_embed = alloc_weights(N_UNKNOWNS * D_MODEL, s_e * 0.1f); // small init
    fax.smooth_weights = build_smooth_weights(frostmatrix_centroids_cpu, N_FAMILIES);

    for (int d = 0; d < N_HW_DIRS; d++) {
        fax.hw_Wq[d] = alloc_weights(D_MODEL * D_HW_HEAD, s_hw);
        fax.hw_Wk[d] = alloc_weights(D_MODEL * D_HW_HEAD, s_hw);
        fax.hw_Wv[d] = alloc_weights(D_MODEL * D_HW_HEAD, s_hw);
    }
    fax.hw_Wo   = alloc_weights(N_HW_DIRS * D_HW_HEAD * D_MODEL, s_hw);
    fax.ca_Wq   = alloc_weights(D_MODEL * D_MODEL, s_a);
    fax.ca_Wk   = alloc_weights(D_MODEL * D_MODEL, s_a);
    fax.ca_Wv   = alloc_weights(D_MODEL * D_MODEL, s_a);
    fax.ca_Wo   = alloc_weights(D_MODEL * D_MODEL, s_a);
    fax.geo_proj_W = alloc_weights(N_GEO_CHANNELS * D_MODEL, s_e);
    fax.geo_proj_b = alloc_weights(D_MODEL, 0.0f);
}

void alloc_buffers(ForwardBuffers *buf, int batch) {
    int S = SEQ_LEN_V3, D = D_MODEL;
    CUDA_CHECK(cudaMalloc(&buf->x,           batch*S*D*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buf->x_norm,      batch*S*D*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buf->q,           batch*S*D*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buf->k,           batch*S*D*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buf->v,           batch*S*D*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buf->attn_scores, batch*N_HEADS*S*S*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buf->attn_out,    batch*S*D*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buf->ffn_h,       batch*S*D_FFN*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buf->ffn_gate_buf,batch*S*D_FFN*sizeof(float)));
    for (int d = 0; d < N_HW_DIRS; d++) {
        CUDA_CHECK(cudaMalloc(&buf->hw_q[d], batch*N_FAM_NODES*D_HW_HEAD*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&buf->hw_k[d], batch*N_FAM_NODES*D_HW_HEAD*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&buf->hw_v[d], batch*N_FAM_NODES*D_HW_HEAD*sizeof(float)));
    }
    CUDA_CHECK(cudaMalloc(&buf->hw_attn,  batch*N_FAM_NODES*N_FAM_NODES*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buf->family_axis, batch*N_FAM_NODES*D_MODEL*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buf->ca_q,     batch*N_BOARD_SQ*D_MODEL*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buf->ca_k,     batch*N_FAM_NODES*D_MODEL*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buf->ca_v,     batch*N_FAM_NODES*D_MODEL*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buf->ca_out,   batch*N_BOARD_SQ*D_MODEL*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buf->canon_pos,batch*SEQ_LEN_V3*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&buf->geo_embed,batch*N_BOARD_SQ*D_MODEL*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buf->pooled, batch*D_MODEL*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buf->policy_from_logits, batch*POLICY_OUT_DIM*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buf->policy_to_logits,   batch*POLICY_OUT_DIM*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buf->value_out,     batch*sizeof(float)));
}

/* ═══════════════════════════════════════════════════════════════════════════
 * KERNELS — FAMILY AXIS INITIALIZATION
 * ═══════════════════════════════════════════════════════════════════════════ */

/*
 * Builds the Y-axis token sequence: interleave family and unknown nodes.
 * Layout: [fam_0, unk_01, fam_1, unk_12, ..., unk_31-32, fam_32]
 * Each family node = its learnable embedding.
 * Each unknown node = average of adjacent family embeddings (soft midpoint init at inference).
 */
__global__ void build_family_axis_kernel(
    float *family_axis,             // [batch × N_FAM_NODES × D_MODEL]
    const float *family_embed,      // [N_FAMILIES × D_MODEL]
    const float *unknown_embed,     // [N_UNKNOWNS × D_MODEL]
    int batch)
{
    // Each warp handles one D_MODEL vector; grid covers batch × N_FAM_NODES
    int token_idx = blockIdx.x;   // which Y-axis token (0..64)
    int b         = blockIdx.y;   // batch index
    int d         = threadIdx.x;  // D_MODEL thread

    if (b >= batch || token_idx >= N_FAM_NODES || d >= D_MODEL) return;

    int out_offset = (b * N_FAM_NODES + token_idx) * D_MODEL + d;
    bool is_family = (token_idx % 2 == 0);
    int  local_idx = token_idx / 2;  // family_id or unknown_id

    if (is_family) {
        family_axis[out_offset] = family_embed[local_idx * D_MODEL + d];
    } else {
        // Unknown potential: learned embedding
        family_axis[out_offset] = unknown_embed[local_idx * D_MODEL + d];
    }
}

__global__ void embed_board_tokens_v3_kernel(
    const int *tokens,          // [batch × SEQ_LEN_V3]
    const float *embed_table,   // [VOCAB_SIZE × D_MODEL]
    float *x,                   // [batch × SEQ_LEN_V3 × D_MODEL]
    int batch)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * N_BOARD_SQ * D_MODEL;
    if (idx >= total) return;

    int d  = idx % D_MODEL;
    int sq = (idx / D_MODEL) % N_BOARD_SQ;
    int b  = idx / (N_BOARD_SQ * D_MODEL);

    int tok = tokens[b * SEQ_LEN_V3 + N_FAM_NODES + sq];
    int out = ((b * SEQ_LEN_V3) + N_FAM_NODES + sq) * D_MODEL + d;
    x[out] = embed_table[tok * D_MODEL + d];
}

__global__ void board_pos_encode_v3_kernel(float *x, int batch) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * N_BOARD_SQ * D_MODEL;
    if (idx >= total) return;

    int d  = idx % D_MODEL;
    int sq = (idx / D_MODEL) % N_BOARD_SQ;
    int b  = idx / (N_BOARD_SQ * D_MODEL);

    int rank = sq / 8;
    int file = sq % 8;
    int folded_rank = min(rank, 7 - rank);
    int pos = folded_rank * 8 + file;
    float freq = powf(10000.f, -2.f * (d / 2) / (float)D_MODEL);
    float enc  = (d % 2 == 0) ? sinf((float)pos * freq) : cosf((float)pos * freq);
    int out = ((b * SEQ_LEN_V3) + N_FAM_NODES + sq) * D_MODEL + d;
    x[out] += enc;
}

__global__ void scatter_family_axis_kernel(
    float *x,                   // [batch × SEQ_LEN_V3 × D_MODEL]
    const float *family_axis,   // [batch × N_FAM_NODES × D_MODEL]
    int batch)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * N_FAM_NODES * D_MODEL;
    if (idx >= total) return;

    int d   = idx % D_MODEL;
    int tok = (idx / D_MODEL) % N_FAM_NODES;
    int b   = idx / (N_FAM_NODES * D_MODEL);
    x[((b * SEQ_LEN_V3) + tok) * D_MODEL + d] = family_axis[idx];
}

/* ═══════════════════════════════════════════════════════════════════════════
 * KERNELS — HIGHWAY ATTENTION
 * Three orthogonal attention passes over the family Y-axis.
 * Each direction uses independent Q,K,V projections (orthogonal constraint).
 * Results combined via output projection and added to family axis tokens.
 * ═══════════════════════════════════════════════════════════════════════════ */

/*
 * Compute highway attention scores for one direction.
 * Uses a sparse attention pattern per direction:
 *   LATERAL  (dir=0): all families × all families, weighted by smooth_weights
 *   VERTICAL (dir=1): each family node → its adjacent unknown nodes only
 *   DIAGONAL (dir=2): each family node → unknown nodes ±3 families away
 */
__global__ void highway_attn_scores_kernel(
    float *scores,              // [batch × N_FAM_NODES × N_FAM_NODES] — output
    const float *q,             // [batch × N_FAM_NODES × D_HW_HEAD]
    const float *k,             // [batch × N_FAM_NODES × D_HW_HEAD]
    const float *smooth_w,     // [N_FAMILIES × N_FAMILIES]
    int dir,                    // 0=lateral 1=vertical 2=diagonal
    int batch)
{
    int b  = blockIdx.z;
    int qi = blockIdx.y;   // query token in family axis (0..N_FAM_NODES-1)
    int ki = blockIdx.x;   // key token
    int d  = threadIdx.x;  // D_HW_HEAD

    if (b >= batch || qi >= N_FAM_NODES || ki >= N_FAM_NODES || d >= D_HW_HEAD) return;

    // Direction-based attention mask
    // token layout: even = family (family_id = token/2), odd = unknown (unknown_id = token/2)
    bool qi_fam = (qi % 2 == 0), ki_fam = (ki % 2 == 0);
    int qi_id = qi / 2, ki_id = ki / 2;
    bool allowed = false;

    if (dir == 0) {
        // LATERAL: family → family only, with geometric smoothing bias
        allowed = qi_fam && ki_fam;
    } else if (dir == 1) {
        // VERTICAL: family[i] ↔ unknown[i-1,i] and unknown[i,i+1]
        if (qi_fam && !ki_fam) {
            // family qi_id can attend to unknown ki_id if ki_id == qi_id-1 or ki_id == qi_id
            allowed = (ki_id == qi_id - 1 || ki_id == qi_id) && ki_id >= 0 && ki_id < N_UNKNOWNS;
        } else if (!qi_fam && ki_fam) {
            // unknown ki_id can attend to adjacent families
            allowed = (ki_id == qi_id || ki_id == qi_id + 1) && ki_id < N_FAMILIES;
        }
    } else {
        // DIAGONAL: family[i] ↔ unknowns bridging ±3 families away
        if (qi_fam && !ki_fam) {
            int skip = abs(ki_id - qi_id);
            allowed = (skip == 2 || skip == 3) && ki_id >= 0 && ki_id < N_UNKNOWNS;
        }
    }

    if (!allowed) {
        if (d == 0) scores[(b*N_FAM_NODES + qi)*N_FAM_NODES + ki] = -1e9f;
        return;
    }

    // Dot product q·k
    float dot = 0.0f;
    int qoff = (b*N_FAM_NODES + qi) * D_HW_HEAD;
    int koff = (b*N_FAM_NODES + ki) * D_HW_HEAD;
    // Warp reduce
    float val = q[qoff + d] * k[koff + d];
    for (int stride = 16; stride >= 1; stride >>= 1)
        val += __shfl_down_sync(0xffffffff, val, stride);

    if (d == 0) {
        float scale = 1.0f / sqrtf((float)D_HW_HEAD);
        float score = val * scale;
        // Apply lateral smoothing bias (only dir=0, family→family)
        if (dir == 0 && qi_fam && ki_fam) {
            float sw = smooth_w[qi_id * N_FAMILIES + ki_id];
            score += logf(sw + 1e-8f);  // additive log-space smoothing bias
        }
        scores[(b*N_FAM_NODES + qi)*N_FAM_NODES + ki] = score;
    }
}

/* Softmax + weighted value aggregation for highway (one direction) */
__global__ void highway_softmax_v_kernel(
    float *out,              // [batch × N_FAM_NODES × D_HW_HEAD]
    const float *scores,    // [batch × N_FAM_NODES × N_FAM_NODES]
    const float *v,         // [batch × N_FAM_NODES × D_HW_HEAD]
    int batch)
{
    int b  = blockIdx.y;
    int qi = blockIdx.x;
    int d  = threadIdx.x;
    if (b >= batch || qi >= N_FAM_NODES || d >= D_HW_HEAD) return;

    const float *row = scores + (b*N_FAM_NODES + qi)*N_FAM_NODES;
    // Numerically stable softmax
    float mx = -1e9f;
    for (int j = 0; j < N_FAM_NODES; j++) mx = fmaxf(mx, row[j]);
    float ex_sum = 0.0f;
    for (int j = 0; j < N_FAM_NODES; j++) ex_sum += expf(row[j] - mx);

    // Weighted sum of values
    float acc = 0.0f;
    for (int j = 0; j < N_FAM_NODES; j++) {
        float attn = expf(row[j] - mx) / (ex_sum + 1e-8f);
        acc += attn * v[(b*N_FAM_NODES + j)*D_HW_HEAD + d];
    }
    out[(b*N_FAM_NODES + qi)*D_HW_HEAD + d] = acc;
}

__global__ void combine_highway_kernel(
    float *family_axis,      // [batch × N_FAM_NODES × D_MODEL] — in-place residual add
    const float *dir0,       // [batch × N_FAM_NODES × D_HW_HEAD]
    const float *dir1,
    const float *dir2,
    const float *Wo,         // [(N_HW_DIRS × D_HW_HEAD) × D_MODEL]
    int batch)
{
    int token = blockIdx.x;
    int b = blockIdx.y;
    int d = threadIdx.x;
    if (b >= batch || token >= N_FAM_NODES || d >= D_MODEL) return;

    int base = (b * N_FAM_NODES + token) * D_HW_HEAD;
    float acc = 0.0f;
    for (int h = 0; h < D_HW_HEAD; h++) {
        acc += dir0[base + h] * Wo[((0 * D_HW_HEAD + h) * D_MODEL) + d];
        acc += dir1[base + h] * Wo[((1 * D_HW_HEAD + h) * D_MODEL) + d];
        acc += dir2[base + h] * Wo[((2 * D_HW_HEAD + h) * D_MODEL) + d];
    }
    family_axis[(b * N_FAM_NODES + token) * D_MODEL + d] += acc;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * KERNEL — GEOMETRY PROJECTION
 * Projects the 33-dimensional FrostMatrix geometry vector into D_MODEL space
 * and adds it to the corresponding board token embedding.
 * ═══════════════════════════════════════════════════════════════════════════ */

__global__ void geo_project_kernel(
    float *x,               // [batch × SEQ_LEN_V3 × D_MODEL] — board tokens start at N_FAM_NODES
    const float *geo_vecs,  // [batch × N_BOARD_SQ × N_GEO_CHANNELS]
    const float *proj_W,    // [N_GEO_CHANNELS × D_MODEL]
    const float *proj_b,    // [D_MODEL]
    int batch)
{
    int b  = blockIdx.z;
    int sq = blockIdx.y;
    int d  = threadIdx.x;  // D_MODEL
    if (b >= batch || sq >= N_BOARD_SQ || d >= D_MODEL) return;

    float acc = proj_b[d];
    const float *gv = geo_vecs + (b*N_BOARD_SQ + sq)*N_GEO_CHANNELS;
    for (int g = 0; g < N_GEO_CHANNELS; g++)
        acc += gv[g] * proj_W[g*D_MODEL + d];

    // Add to board token (board tokens start at position N_FAM_NODES in sequence)
    int tok = N_FAM_NODES + sq;
    x[(b*SEQ_LEN_V3 + tok)*D_MODEL + d] += tanhf(acc);  // tanh gate keeps magnitudes stable
}

/* ═══════════════════════════════════════════════════════════════════════════
 * KERNEL — BOARD → FAMILY CROSS-ATTENTION
 * Board tokens query the family axis. This is how each position "finds" its
 * monster family dynamically — not by lookup, but by attention over the Y-axis.
 * Unknown potential nodes participate as keys/values, so hybrid positions can
 * attend to the latent space between known families.
 * ═══════════════════════════════════════════════════════════════════════════ */

__global__ void board_cross_attn_kernel(
    float *x,                // [batch × SEQ_LEN_V3 × D_MODEL] — board section updated in place
    const float *family_axis,// [batch × N_FAM_NODES × D_MODEL]
    const float *ca_Wq,      // [D_MODEL × D_MODEL]
    const float *ca_Wk,
    const float *ca_Wv,
    const float *ca_Wo,
    int batch)
{
    int b  = blockIdx.y;
    int sq = blockIdx.x;
    int d  = threadIdx.x;
    if (b >= batch || sq >= N_BOARD_SQ || d >= D_MODEL) return;

    const int lane = d & 31;
    const int warp = d >> 5;
    __shared__ float q_vec[D_MODEL];
    __shared__ float attn_vec[D_MODEL];
    __shared__ float score_buf[N_FAM_NODES];
    __shared__ float warp_sum[4];

    float *board_tok = x + ((b * SEQ_LEN_V3) + N_FAM_NODES + sq) * D_MODEL;

    float q_d = 0.0f;
    for (int j = 0; j < D_MODEL; j++) q_d += board_tok[j] * ca_Wq[j * D_MODEL + d];
    q_vec[d] = q_d;
    __syncthreads();

    for (int fi = 0; fi < N_FAM_NODES; fi++) {
        const float *fx = family_axis + (b * N_FAM_NODES + fi) * D_MODEL;
        float k_d = 0.0f;
        for (int j = 0; j < D_MODEL; j++) k_d += fx[j] * ca_Wk[j * D_MODEL + d];
        float partial = q_vec[d] * k_d;
        for (int offset = 16; offset > 0; offset >>= 1)
            partial += __shfl_down_sync(0xffffffff, partial, offset);
        if (lane == 0) warp_sum[warp] = partial;
        __syncthreads();
        if (d == 0) {
            float total = 0.0f;
            for (int w = 0; w < 4; w++) total += warp_sum[w];
            score_buf[fi] = total / sqrtf((float)D_MODEL);
        }
        __syncthreads();
    }

    if (d == 0) {
        float mx = -1e30f;
        for (int fi = 0; fi < N_FAM_NODES; fi++) mx = fmaxf(mx, score_buf[fi]);
        float ex_sum = 0.0f;
        for (int fi = 0; fi < N_FAM_NODES; fi++) {
            score_buf[fi] = expf(score_buf[fi] - mx);
            ex_sum += score_buf[fi];
        }
        for (int fi = 0; fi < N_FAM_NODES; fi++) score_buf[fi] /= (ex_sum + 1e-8f);
    }
    __syncthreads();

    float attn_d = 0.0f;
    for (int fi = 0; fi < N_FAM_NODES; fi++) {
        const float *fx = family_axis + (b * N_FAM_NODES + fi) * D_MODEL;
        float v_d = 0.0f;
        for (int j = 0; j < D_MODEL; j++) v_d += fx[j] * ca_Wv[j * D_MODEL + d];
        attn_d += score_buf[fi] * v_d;
    }
    attn_vec[d] = attn_d;
    __syncthreads();

    float proj = 0.0f;
    for (int j = 0; j < D_MODEL; j++) proj += attn_vec[j] * ca_Wo[j * D_MODEL + d];
    board_tok[d] += proj;
}

__global__ void attn_scale_kernel(float *attn, int total, int d_head) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    attn[idx] /= sqrtf((float)d_head);
}

__global__ void softmax_kernel(float *x, int batch, int n) {
    int b = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    int lane = threadIdx.x & 31;
    if (b >= batch) return;

    float *row = x + b * n;
    float mx = -1e30f;
    for (int i = lane; i < n; i += 32) mx = fmaxf(mx, row[i]);
    for (int offset = 16; offset > 0; offset >>= 1)
        mx = fmaxf(mx, __shfl_down_sync(0xffffffff, mx, offset));
    mx = __shfl_sync(0xffffffff, mx, 0);

    float sum = 0.0f;
    for (int i = lane; i < n; i += 32) {
        float v = expf(row[i] - mx);
        row[i] = v;
        sum += v;
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    sum = __shfl_sync(0xffffffff, sum, 0);

    for (int i = lane; i < n; i += 32) row[i] /= (sum + 1e-8f);
}

__global__ void mean_pool_board_kernel(
    float *pooled,          // [batch × D_MODEL]
    const float *x,         // [batch × SEQ_LEN_V3 × D_MODEL]
    int batch)
{
    int b = blockIdx.x;
    int d = threadIdx.x;
    if (b >= batch || d >= D_MODEL) return;

    float acc = 0.0f;
    for (int sq = 0; sq < N_BOARD_SQ; sq++) {
        acc += x[((b * SEQ_LEN_V3) + N_FAM_NODES + sq) * D_MODEL + d];
    }
    pooled[b * D_MODEL + d] = acc / (float)N_BOARD_SQ;
}

__global__ void value_head_kernel(
    const float *pooled,    // [batch × D_MODEL]
    const float *w,         // [D_MODEL]
    const float *b,         // [1]
    float *out,
    int batch)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch) return;
    float v = b[0];
    for (int j = 0; j < D_MODEL; j++) v += pooled[i * D_MODEL + j] * w[j];
    out[i] = tanhf(v);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * KERNELS — STANDARD TRANSFORMER (reused from v2 structure)
 * LayerNorm, SwiGLU+smeargate FFN, multi-head self-attention
 * ═══════════════════════════════════════════════════════════════════════════ */

__global__ void layernorm_kernel(float *out, const float *x, const float *w,
                                  const float *b, int n_tokens, int d_model, int batch) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * n_tokens;
    if (idx >= total) return;
    const float *xi = x + idx * d_model;
    float *oi = out + idx * d_model;
    float mean = 0.0f, var = 0.0f;
    for (int d = 0; d < d_model; d++) mean += xi[d];
    mean /= d_model;
    for (int d = 0; d < d_model; d++) { float t = xi[d]-mean; var += t*t; }
    float inv_std = 1.0f / sqrtf(var/d_model + 1e-5f);
    for (int d = 0; d < d_model; d++) oi[d] = (xi[d]-mean)*inv_std*w[d] + b[d];
}

__global__ void swiglu_smeargate_kernel(
    float *out, const float *up_buf, const float *gate_buf,
    int n_tokens, int d_ffn, int batch)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * n_tokens * d_ffn;
    if (idx >= total) return;
    int fi = idx % d_ffn;
    int ti = idx / d_ffn;
    float up   = up_buf[ti*d_ffn + fi];
    float gate = gate_buf[ti*d_ffn + fi];
    float silu = gate / (1.0f + expf(-gate));
    float h = up * silu;
    // Smeargate: geometric mean with neighbor (warp-shuffle boundary)
    int lane = fi & 31;
    float nb = __shfl_down_sync(0xffffffff, h, 1);
    if (lane == 31) nb = h;   // boundary: self
    float smear = (h * nb > 0.0f) ? sqrtf(fabsf(h * nb)) * (h > 0.0f ? 1.0f : -1.0f) : 0.0f;
    out[ti*d_ffn + fi] = smear;
}

__global__ void residual_add_kernel(float *x, const float *delta, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += delta[i];
}

/* ═══════════════════════════════════════════════════════════════════════════
 * FORWARD PASS
 * ═══════════════════════════════════════════════════════════════════════════ */

void forward(
    const int *tokens,          // [batch × SEQ_LEN_V3] integer token IDs
    const float *geo_vecs,      // [batch × N_BOARD_SQ × N_GEO_CHANNELS] geometry
    TransformerWeights *tw,
    ForwardBuffers *buf,
    cublasHandle_t cublas,
    int batch)
{
    const float one=1.0f, zero=0.0f;
    int S = SEQ_LEN_V3, D = D_MODEL;
    int F = N_FAM_NODES;
    int B_S = batch * S;

    // ── Step 1: Build family axis tokens into x[0..N_FAM_NODES-1] ────────
    dim3 fax_grid(N_FAM_NODES, batch);
    build_family_axis_kernel<<<fax_grid, D>>>(
        buf->family_axis, tw->fax.family_embed, tw->fax.unknown_embed, batch);

    // ── Step 2: Embed board tokens into x[N_FAM_NODES..SEQ_LEN-1] ────────
    {
        int board_embed_total = batch * N_BOARD_SQ * D_MODEL;
        embed_board_tokens_v3_kernel<<<(board_embed_total + 255) / 256, 256>>>(
            tokens, tw->token_embed, buf->x, batch);
        board_pos_encode_v3_kernel<<<(board_embed_total + 255) / 256, 256>>>(buf->x, batch);
    }

    // ── Step 3: Project FrostMatrix geometry into board tokens ────────────
    {
        dim3 geo_grid(1, N_BOARD_SQ, batch);
        geo_project_kernel<<<geo_grid, D>>>(
            buf->x, geo_vecs, tw->fax.geo_proj_W, tw->fax.geo_proj_b, batch);
    }

    // ── Step 4: Highway attention — 3 orthogonal directions ──────────────
    // For each direction: project Q,K,V → compute scores → softmax → aggregate V
    // Then combine via output projection and add to family axis residual

    for (int dir = 0; dir < N_HW_DIRS; dir++) {
        // Project Q, K, V from family axis tokens
        CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
            D_HW_HEAD, batch*F, D, &one,
            tw->fax.hw_Wq[dir], D,
            buf->family_axis, D, &zero, buf->hw_q[dir], D_HW_HEAD));
        CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
            D_HW_HEAD, batch*F, D, &one,
            tw->fax.hw_Wk[dir], D,
            buf->family_axis, D, &zero, buf->hw_k[dir], D_HW_HEAD));
        CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
            D_HW_HEAD, batch*F, D, &one,
            tw->fax.hw_Wv[dir], D,
            buf->family_axis, D, &zero, buf->hw_v[dir], D_HW_HEAD));

        // Compute attention scores with direction-specific mask + smoothing
        dim3 score_grid(F, F, batch);
        highway_attn_scores_kernel<<<score_grid, D_HW_HEAD>>>(
            buf->hw_attn, buf->hw_q[dir], buf->hw_k[dir],
            tw->fax.smooth_weights, dir, batch);

        // Softmax + value aggregate
        dim3 sv_grid(F, batch);
        highway_softmax_v_kernel<<<sv_grid, D_HW_HEAD>>>(
            buf->hw_q[dir],   // reuse hw_q as output
            buf->hw_attn, buf->hw_v[dir], batch);
    }

    // Combine 3 highway directions via output projection → add to family axis
    {
        dim3 combine_grid(F, batch);
        combine_highway_kernel<<<combine_grid, D>>>(
            buf->family_axis,
            buf->hw_q[0], buf->hw_q[1], buf->hw_q[2],
            tw->fax.hw_Wo, batch);
    }

    // ── Step 5: Board → Family cross-attention ────────────────────────────
    // Board tokens query the family axis to find their geometric family
    {
        dim3 ca_grid(N_BOARD_SQ, batch);
        board_cross_attn_kernel<<<ca_grid, D>>>(
            buf->x,            // board section lives inside the full sequence tensor
            buf->family_axis,  // contiguous family axis as key/value source
            tw->fax.ca_Wq, tw->fax.ca_Wk, tw->fax.ca_Wv, tw->fax.ca_Wo,
            batch);
    }

    // Scatter updated family axis into the full sequence buffer before global transformer layers.
    {
        int family_total = batch * F * D;
        scatter_family_axis_kernel<<<(family_total + 255) / 256, 256>>>(
            buf->x, buf->family_axis, batch);
    }

    // ── Step 6: Standard transformer layers (N_LAYERS) ───────────────────
    // Each layer: LayerNorm → multi-head self-attention → residual →
    //             LayerNorm → SwiGLU+smeargate FFN → residual
    // Operates on the full SEQ_LEN_V3=129 sequence (family axis + board)
    for (int l = 0; l < N_LAYERS; l++) {
        int total_toks = batch * S;
        int ln_blocks = (total_toks + 255) / 256;

        layernorm_kernel<<<ln_blocks, 256>>>(
            buf->x_norm, buf->x, tw->ln1_w[l], tw->ln1_b[l], S, D, batch);

        // Self-attention: Q,K,V projections + scaled dot-product
        CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
            D, B_S, D, &one, tw->Wq[l], D, buf->x_norm, D, &zero, buf->q, D));
        CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
            D, B_S, D, &one, tw->Wk[l], D, buf->x_norm, D, &zero, buf->k, D));
        CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
            D, B_S, D, &one, tw->Wv[l], D, buf->x_norm, D, &zero, buf->v, D));

        CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
            S, S * N_HEADS * batch, D_HEAD,
            &one, buf->k, D_HEAD, buf->q, D_HEAD, &zero, buf->attn_scores, S));
        int attn_total = batch * N_HEADS * S * S;
        attn_scale_kernel<<<(attn_total + 255) / 256, 256>>>(buf->attn_scores, attn_total, D_HEAD);
        int attn_rows = batch * N_HEADS * S;
        softmax_kernel<<<(attn_rows + 7) / 8, 256>>>(buf->attn_scores, attn_rows, S);

        CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
            D_HEAD, S * N_HEADS * batch, S,
            &one, buf->v, D_HEAD, buf->attn_scores, S, &zero, buf->q, D_HEAD));
        CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
            D, B_S, D, &one, tw->Wo[l], D, buf->q, D, &zero, buf->attn_out, D));
        int n_res = batch*S*D;
        residual_add_kernel<<<(n_res+255)/256, 256>>>(buf->x, buf->attn_out, n_res);

        // FFN: SwiGLU + smeargate
        layernorm_kernel<<<ln_blocks, 256>>>(
            buf->x_norm, buf->x, tw->ln2_w[l], tw->ln2_b[l], S, D, batch);
        CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
            D_FFN, B_S, D, &one, tw->ffn_up[l],   D, buf->x_norm, D, &zero, buf->ffn_h, D_FFN));
        CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
            D_FFN, B_S, D, &one, tw->ffn_gate[l], D, buf->x_norm, D, &zero, buf->ffn_gate_buf, D_FFN));
        int n_ffn = batch*S*D_FFN;
        swiglu_smeargate_kernel<<<(n_ffn+255)/256, 256>>>(
            buf->ffn_h, buf->ffn_h, buf->ffn_gate_buf, S, D_FFN, batch);
        CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
            D, B_S, D_FFN, &one, tw->ffn_down[l], D_FFN, buf->ffn_h, D_FFN, &zero, buf->attn_out, D));
        residual_add_kernel<<<(n_res+255)/256, 256>>>(buf->x, buf->attn_out, n_res);
    }

    // ── Step 7: Split policy heads (from_sq + to_sq) + value head ────────────
    mean_pool_board_kernel<<<batch, D>>>(buf->pooled, buf->x, batch);
    // from-square policy: pool @ policy_from_W → [batch, 64]
    CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
        POLICY_OUT_DIM, batch, D, &one,
        tw->policy_from_W, D,
        buf->pooled, D,
        &zero, buf->policy_from_logits, POLICY_OUT_DIM));
    softmax_kernel<<<batch, 32>>>(buf->policy_from_logits, batch, POLICY_OUT_DIM);
    // to-square policy: pool @ policy_to_W → [batch, 64]
    CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
        POLICY_OUT_DIM, batch, D, &one,
        tw->policy_to_W, D,
        buf->pooled, D,
        &zero, buf->policy_to_logits, POLICY_OUT_DIM));
    softmax_kernel<<<batch, 32>>>(buf->policy_to_logits, batch, POLICY_OUT_DIM);
    value_head_kernel<<<(batch + 255) / 256, 256>>>(
        buf->pooled, tw->value_W, tw->value_b, buf->value_out, batch);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * MAIN — timing benchmark + FrostMatrix package loader
 * ═══════════════════════════════════════════════════════════════════════════ */

int main(int argc, char **argv) {
    printf("FrostMatrix Transformer v3\n");
    printf("  SEQ_LEN=%d  N_FAMILIES=%d  N_UNKNOWNS=%d  N_FAM_NODES=%d\n",
           SEQ_LEN_V3, N_FAMILIES, N_UNKNOWNS, N_FAM_NODES);
    printf("  Highway: %d directions × D_HW_HEAD=%d  |  D_MODEL=%d  N_LAYERS=%d\n",
           N_HW_DIRS, D_HW_HEAD, D_MODEL, N_LAYERS);
    printf("  Geometry channels: %d\n\n", N_GEO_CHANNELS);

    // Optionally load FrostMatrix centroids for smooth weight initialization
    // Usage: transformer_v3_frostmatrix --frostmatrix opening_package_v1.json
    float *frostmatrix_centroids = NULL;
    for (int i = 1; i < argc-1; i++) {
        if (strcmp(argv[i], "--frostmatrix") == 0) {
            printf("  [FrostMatrix] loading centroids from %s\n", argv[i+1]);
            // TODO: parse JSON package and extract family centroid_trajectory[0] r,f
            // For now: NULL → uniform smooth weights
        }
    }

    cublasHandle_t cublas;
    CUBLAS_CHECK(cublasCreate(&cublas));

    TransformerWeights tw = {};
    ForwardBuffers buf    = {};

    init_weights(&tw, frostmatrix_centroids);
    alloc_buffers(&buf, BATCH_SIZE);

    // Benchmark forward pass
    int *h_tokens = (int*)calloc(BATCH_SIZE * SEQ_LEN_V3, sizeof(int));
    float *h_geo  = (float*)calloc(BATCH_SIZE * N_BOARD_SQ * N_GEO_CHANNELS, sizeof(float));
    for (int b = 0; b < BATCH_SIZE; b++) {
        for (int sq = 0; sq < N_BOARD_SQ; sq++) {
            h_tokens[b * SEQ_LEN_V3 + N_FAM_NODES + sq] = rand() % 13;
        }
    }
    int *d_tokens; float *d_geo;
    CUDA_CHECK(cudaMalloc(&d_tokens, BATCH_SIZE*SEQ_LEN_V3*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_geo,    BATCH_SIZE*N_BOARD_SQ*N_GEO_CHANNELS*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_tokens, h_tokens, BATCH_SIZE*SEQ_LEN_V3*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_geo,    h_geo,    BATCH_SIZE*N_BOARD_SQ*N_GEO_CHANNELS*sizeof(float), cudaMemcpyHostToDevice));

    // Warmup
    for (int i = 0; i < 3; i++)
        forward(d_tokens, d_geo, &tw, &buf, cublas, BATCH_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed runs
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    int N_RUNS = 20;
    CUDA_CHECK(cudaEventRecord(t0));
    for (int i = 0; i < N_RUNS; i++)
        forward(d_tokens, d_geo, &tw, &buf, cublas, BATCH_SIZE);
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    float ms_per = ms / N_RUNS;
    float pos_per_sec = BATCH_SIZE / (ms_per / 1000.0f);

    printf("  %.2f ms/batch  |  %.0f positions/sec\n", ms_per, pos_per_sec);
    printf("  seq_len_v3=%d (%.1fx longer than v2 seq=65)\n",
           SEQ_LEN_V3, (float)SEQ_LEN_V3/65.0f);

    free(h_tokens); free(h_geo);
    cublasDestroy(cublas);
    return 0;
}
