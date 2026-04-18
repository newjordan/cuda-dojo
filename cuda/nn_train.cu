// =============================================================================
// CUDA DOJO — Neural Network Trainer (nn_train.cu)
// -----------------------------------------------------------------------------
// Real backprop training for the 768->256->64->1 chess eval MLP.
// All training compute is on GPU via cuBLAS GEMMs and custom element-wise
// kernels. CPU is used ONLY for: file I/O (book.jsonl), FEN parsing, and
// orchestration (epoch loop, batch index shuffling).
//
// Architecture mirrors nn_eval.cu exactly:
//   X   [IN_DIM x N]   column-major (pos=col, feature=row)
//   H1  [H1_DIM x N] = W1^T @ X    + b1   then ReLU
//   H2  [H2_DIM x N] = W2^T @ H1   + b2   then ReLU
//   Y   [1      x N] = W3^T @ H2   + b3
//
// Loss: MSE between Y and Stockfish target (clipped to +/-2000 cp).
// Note: training targets are stored from side-to-move POV (matches SF UCI
// `score cp`), so we DO NOT apply side_flip during training. Inference
// (nn_eval) keeps applying side_flip to convert to white-POV scores.
//
// Optimizer: Adam (per-parameter m,v moments).
//
// CLI:
//   ./nn_train --data <jsonl> --epochs N --batch B --lr L
// =============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#include <algorithm>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>

#include "include/dojo_types.h"

#define IN_DIM   768
#define H1_DIM   256
#define H2_DIM   64
#define OUT_DIM  1

#define WEIGHTS_FILE  "nn_weights.bin"
#define WEIGHTS_BACK  "nn_weights.bin.random"
#define LABEL_CLAMP   2000.0f

enum AccessMode {
    ACCESS_SHUFFLE = 0,
    ACCESS_PRESSURE_BAND = 1,
    ACCESS_RELATION_CLUSTER = 2,
    ACCESS_BANDED_RELATION_CLUSTER = 3,
};

struct LabelTable {
    std::vector<std::string> names;
};

struct BandWeights {
    float fallback;
    float library;
    float cool;
    float warm;
    float hot;
    float fracture;
};

struct SplitMetrics {
    int count;
    float before_mse;
    float before_mae;
    float after_mse;
    float after_mae;
};

#define CUDA_OK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while(0)
#define CUBLAS_OK(x) do { cublasStatus_t s = (x); if (s != CUBLAS_STATUS_SUCCESS) { \
    fprintf(stderr, "cuBLAS error %s:%d: %d\n", __FILE__, __LINE__, (int)s); exit(1); } } while(0)

static uint32_t fnv1a32(const char* s) {
    uint32_t h = 2166136261u;
    if (!s) return h;
    for (; *s; ++s) {
        h ^= (unsigned char)(*s);
        h *= 16777619u;
    }
    return h;
}

static int intern_label(LabelTable* table, const char* raw) {
    std::string label = (raw && raw[0]) ? std::string(raw) : std::string("unknown");
    for (int i = 0; i < (int)table->names.size(); ++i) {
        if (table->names[i] == label) return i;
    }
    table->names.push_back(label);
    return (int)table->names.size() - 1;
}

static const char* label_name(const LabelTable* table, int id) {
    if (!table || id < 0 || id >= (int)table->names.size()) return "unknown";
    return table->names[id].c_str();
}

static AccessMode parse_access_mode(const char* s) {
    if (!s || !strcmp(s, "shuffle")) return ACCESS_SHUFFLE;
    if (!strcmp(s, "pressure_band")) return ACCESS_PRESSURE_BAND;
    if (!strcmp(s, "relation_cluster")) return ACCESS_RELATION_CLUSTER;
    if (!strcmp(s, "banded_relation_cluster")) return ACCESS_BANDED_RELATION_CLUSTER;
    fprintf(stderr, "[nn_train] unknown access mode %s, using shuffle\n", s);
    return ACCESS_SHUFFLE;
}

static BandWeights default_band_weights() {
    BandWeights bw{};
    bw.fallback = 1.0f;
    bw.library = 1.0f;
    bw.cool = 1.0f;
    bw.warm = 1.0f;
    bw.hot = 1.0f;
    bw.fracture = 1.0f;
    return bw;
}

static void trim_ascii(char* s) {
    if (!s) return;
    int n = (int)strlen(s);
    int start = 0;
    while (start < n && (s[start] == ' ' || s[start] == '\t' || s[start] == '\n' || s[start] == '\r')) start++;
    int end = n;
    while (end > start && (s[end - 1] == ' ' || s[end - 1] == '\t' || s[end - 1] == '\n' || s[end - 1] == '\r')) end--;
    if (start > 0) memmove(s, s + start, (size_t)(end - start));
    s[end - start] = 0;
}

static void parse_band_weights(const char* spec, BandWeights* out) {
    *out = default_band_weights();
    if (!spec || !spec[0]) return;
    char buf[512];
    snprintf(buf, sizeof(buf), "%s", spec);
    char* tok = strtok(buf, ",");
    while (tok) {
        trim_ascii(tok);
        char* eq = strchr(tok, '=');
        if (eq) {
            *eq = 0;
            char* key = tok;
            char* value = eq + 1;
            trim_ascii(key);
            trim_ascii(value);
            float w = (float)atof(value);
            if (!strcmp(key, "library")) out->library = w;
            else if (!strcmp(key, "cool")) out->cool = w;
            else if (!strcmp(key, "warm")) out->warm = w;
            else if (!strcmp(key, "hot")) out->hot = w;
            else if (!strcmp(key, "fracture")) out->fracture = w;
            else if (!strcmp(key, "fallback")) out->fallback = w;
        }
        tok = strtok(NULL, ",");
    }
}

static float band_weight_for_name(const BandWeights* bw, const char* name) {
    if (!bw || !name) return 1.0f;
    if (!strcmp(name, "library")) return bw->library;
    if (!strcmp(name, "cool")) return bw->cool;
    if (!strcmp(name, "warm")) return bw->warm;
    if (!strcmp(name, "hot")) return bw->hot;
    if (!strcmp(name, "fracture")) return bw->fracture;
    return bw->fallback;
}

// -----------------------------------------------------------------------------
// Kernels
// -----------------------------------------------------------------------------

// Feature extraction (one block per position, IN_DIM threads available)
__global__ void extract_features_kernel(const CompactBoard* __restrict__ boards,
                                        float* __restrict__ feats, int n) {
    int pos = blockIdx.x;
    if (pos >= n) return;
    const CompactBoard* b = &boards[pos];
    float* out = feats + pos * IN_DIM;
    for (int idx = threadIdx.x; idx < IN_DIM; idx += blockDim.x) out[idx] = 0.0f;
    __syncthreads();
    for (int sq = threadIdx.x; sq < 64; sq += blockDim.x) {
        int p = b->board[sq];
        if (p == DOJO_EMPTY) continue;
        int piece_idx;
        if (p >= DOJO_WPAWN && p <= DOJO_WKING) piece_idx = p - 1;
        else                                     piece_idx = (p - DOJO_BPAWN) + 6;
        out[piece_idx * 64 + sq] = 1.0f;
    }
}

__global__ void relu_kernel(float* x, int total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) { float v = x[i]; x[i] = v > 0.0f ? v : 0.0f; }
}

__global__ void bias_add_kernel(float* out, const float* bias, int N, int D) {
    // out is [D x N] column-major: element (row, col) at row + col*D
    // bias[row] added to every column.
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = D * N;
    if (i < total) {
        int row = i % D;
        out[i] += bias[row];
    }
}

// dL/dPreActivation = dL/dActivation * (PreAct > 0)
// We store post-activation in H1/H2, but ReLU is idempotent on positives, so
// (post > 0) == (pre > 0). We mask the gradient by (h_post > 0).
__global__ void relu_backward_kernel(float* grad, const float* h_post, int total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) grad[i] = h_post[i] > 0.0f ? grad[i] : 0.0f;
}

// dL/dY = (Y - target) / N , clipped target to +/- LABEL_CLAMP
// Also computes squared error per element into 'sq' for loss reduction.
__global__ void mse_grad_kernel(const float* y, const float* target, const float* weight,
                                float* dy, float* sq, int N, float scale) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float t = target[i];
        if (t >  LABEL_CLAMP) t =  LABEL_CLAMP;
        if (t < -LABEL_CLAMP) t = -LABEL_CLAMP;
        float d = y[i] - t;
        float w = weight ? weight[i] : 1.0f;
        dy[i] = d * scale * w;     // weight is expected to have mean ~1 across the corpus
        sq[i] = d * d;
    }
}

// Just compute squared error (no gradient) for holdout eval.
__global__ void sqerr_kernel(const float* y, const float* target,
                             float* sq, float* abserr, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float t = target[i];
        if (t >  LABEL_CLAMP) t =  LABEL_CLAMP;
        if (t < -LABEL_CLAMP) t = -LABEL_CLAMP;
        float d = y[i] - t;
        sq[i]     = d * d;
        abserr[i] = fabsf(d);
    }
}

// Adam update: w -= lr * m_hat / (sqrt(v_hat) + eps)
__global__ void adam_kernel(float* w, float* m, float* v, const float* g,
                            float lr, float beta1, float beta2, float eps,
                            float bc1, float bc2, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float gi = g[i];
        float mi = beta1 * m[i] + (1.0f - beta1) * gi;
        float vi = beta2 * v[i] + (1.0f - beta2) * gi * gi;
        m[i] = mi;
        v[i] = vi;
        float m_hat = mi / bc1;
        float v_hat = vi / bc2;
        w[i] -= lr * m_hat / (sqrtf(v_hat) + eps);
    }
}

// Sum reduction (simple) — for small arrays (batch loss).
__global__ void sum_reduce_kernel(const float* x, float* out, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    sdata[tid] = (i < n) ? x[i] : 0.0f;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(out, sdata[0]);
}

// Reduce columns of a [D x N] column-major matrix into a length-D vector
// (sum along the N dimension). One block per row.
__global__ void col_sum_kernel(const float* X, float* out, int D, int N) {
    int row = blockIdx.x;
    if (row >= D) return;
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    float acc = 0.0f;
    for (int col = tid; col < N; col += blockDim.x) {
        acc += X[row + col * D];
    }
    sdata[tid] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) out[row] = sdata[0];
}

// -----------------------------------------------------------------------------
// Net + Optimizer state
// -----------------------------------------------------------------------------
struct Net {
    float *d_W1, *d_b1;
    float *d_W2, *d_b2;
    float *d_W3, *d_b3;
};

struct Grad {
    float *d_W1, *d_b1;
    float *d_W2, *d_b2;
    float *d_W3, *d_b3;
};

struct Adam {
    float *m_W1, *v_W1, *m_b1, *v_b1;
    float *m_W2, *v_W2, *m_b2, *v_b2;
    float *m_W3, *v_W3, *m_b3, *v_b3;
};

#define ALLOC_ZERO(p, n) do { CUDA_OK(cudaMalloc(&(p), sizeof(float)*(n))); CUDA_OK(cudaMemset((p), 0, sizeof(float)*(n))); } while(0)

static void net_alloc(Net* n) {
    CUDA_OK(cudaMalloc(&n->d_W1, sizeof(float)*IN_DIM*H1_DIM));
    CUDA_OK(cudaMalloc(&n->d_b1, sizeof(float)*H1_DIM));
    CUDA_OK(cudaMalloc(&n->d_W2, sizeof(float)*H1_DIM*H2_DIM));
    CUDA_OK(cudaMalloc(&n->d_b2, sizeof(float)*H2_DIM));
    CUDA_OK(cudaMalloc(&n->d_W3, sizeof(float)*H2_DIM*OUT_DIM));
    CUDA_OK(cudaMalloc(&n->d_b3, sizeof(float)*OUT_DIM));
}
static void net_free(Net* n) {
    cudaFree(n->d_W1); cudaFree(n->d_b1);
    cudaFree(n->d_W2); cudaFree(n->d_b2);
    cudaFree(n->d_W3); cudaFree(n->d_b3);
}
static void grad_alloc(Grad* g) {
    ALLOC_ZERO(g->d_W1, IN_DIM*H1_DIM);
    ALLOC_ZERO(g->d_b1, H1_DIM);
    ALLOC_ZERO(g->d_W2, H1_DIM*H2_DIM);
    ALLOC_ZERO(g->d_b2, H2_DIM);
    ALLOC_ZERO(g->d_W3, H2_DIM*OUT_DIM);
    ALLOC_ZERO(g->d_b3, OUT_DIM);
}
static void grad_free(Grad* g) {
    cudaFree(g->d_W1); cudaFree(g->d_b1);
    cudaFree(g->d_W2); cudaFree(g->d_b2);
    cudaFree(g->d_W3); cudaFree(g->d_b3);
}
static void adam_alloc(Adam* a) {
    ALLOC_ZERO(a->m_W1, IN_DIM*H1_DIM); ALLOC_ZERO(a->v_W1, IN_DIM*H1_DIM);
    ALLOC_ZERO(a->m_b1, H1_DIM);        ALLOC_ZERO(a->v_b1, H1_DIM);
    ALLOC_ZERO(a->m_W2, H1_DIM*H2_DIM); ALLOC_ZERO(a->v_W2, H1_DIM*H2_DIM);
    ALLOC_ZERO(a->m_b2, H2_DIM);        ALLOC_ZERO(a->v_b2, H2_DIM);
    ALLOC_ZERO(a->m_W3, H2_DIM*OUT_DIM); ALLOC_ZERO(a->v_W3, H2_DIM*OUT_DIM);
    ALLOC_ZERO(a->m_b3, OUT_DIM);        ALLOC_ZERO(a->v_b3, OUT_DIM);
}
static void adam_free(Adam* a) {
    cudaFree(a->m_W1); cudaFree(a->v_W1); cudaFree(a->m_b1); cudaFree(a->v_b1);
    cudaFree(a->m_W2); cudaFree(a->v_W2); cudaFree(a->m_b2); cudaFree(a->v_b2);
    cudaFree(a->m_W3); cudaFree(a->v_W3); cudaFree(a->m_b3); cudaFree(a->v_b3);
}

// He-init: stddev = sqrt(2/fan_in)
static void net_init(Net* n, unsigned long long seed) {
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, seed);
    float s1 = sqrtf(2.0f / IN_DIM);
    float s2 = sqrtf(2.0f / H1_DIM);
    float s3 = sqrtf(2.0f / H2_DIM);
    curandGenerateNormal(gen, n->d_W1, IN_DIM*H1_DIM, 0.0f, s1);
    curandGenerateNormal(gen, n->d_W2, H1_DIM*H2_DIM, 0.0f, s2);
    curandGenerateNormal(gen, n->d_W3, H2_DIM*OUT_DIM, 0.0f, s3);
    CUDA_OK(cudaMemset(n->d_b1, 0, sizeof(float)*H1_DIM));
    CUDA_OK(cudaMemset(n->d_b2, 0, sizeof(float)*H2_DIM));
    CUDA_OK(cudaMemset(n->d_b3, 0, sizeof(float)*OUT_DIM));
    curandDestroyGenerator(gen);
}

static int net_save(const Net* net, const char* path) {
    FILE* f = fopen(path, "wb");
    if (!f) return 0;
    size_t nW1 = (size_t)IN_DIM*H1_DIM, nW2 = (size_t)H1_DIM*H2_DIM, nW3 = (size_t)H2_DIM*OUT_DIM;
    float *buf = (float*)malloc(sizeof(float) * nW1);
    cudaMemcpy(buf, net->d_W1, sizeof(float)*nW1, cudaMemcpyDeviceToHost);
    fwrite(buf, sizeof(float), nW1, f);
    cudaMemcpy(buf, net->d_b1, sizeof(float)*H1_DIM, cudaMemcpyDeviceToHost);
    fwrite(buf, sizeof(float), H1_DIM, f);
    cudaMemcpy(buf, net->d_W2, sizeof(float)*nW2, cudaMemcpyDeviceToHost);
    fwrite(buf, sizeof(float), nW2, f);
    cudaMemcpy(buf, net->d_b2, sizeof(float)*H2_DIM, cudaMemcpyDeviceToHost);
    fwrite(buf, sizeof(float), H2_DIM, f);
    cudaMemcpy(buf, net->d_W3, sizeof(float)*nW3, cudaMemcpyDeviceToHost);
    fwrite(buf, sizeof(float), nW3, f);
    cudaMemcpy(buf, net->d_b3, sizeof(float)*OUT_DIM, cudaMemcpyDeviceToHost);
    fwrite(buf, sizeof(float), OUT_DIM, f);
    free(buf);
    fclose(f);
    return 1;
}

static int net_load(Net* net, const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return 0;
    size_t nW1 = (size_t)IN_DIM*H1_DIM, nW2 = (size_t)H1_DIM*H2_DIM, nW3 = (size_t)H2_DIM*OUT_DIM;
    float *buf = (float*)malloc(sizeof(float) * nW1);
    fread(buf, sizeof(float), nW1, f);
    cudaMemcpy(net->d_W1, buf, sizeof(float)*nW1, cudaMemcpyHostToDevice);
    fread(buf, sizeof(float), H1_DIM, f);
    cudaMemcpy(net->d_b1, buf, sizeof(float)*H1_DIM, cudaMemcpyHostToDevice);
    fread(buf, sizeof(float), nW2, f);
    cudaMemcpy(net->d_W2, buf, sizeof(float)*nW2, cudaMemcpyHostToDevice);
    fread(buf, sizeof(float), H2_DIM, f);
    cudaMemcpy(net->d_b2, buf, sizeof(float)*H2_DIM, cudaMemcpyHostToDevice);
    fread(buf, sizeof(float), nW3, f);
    cudaMemcpy(net->d_W3, buf, sizeof(float)*nW3, cudaMemcpyHostToDevice);
    fread(buf, sizeof(float), OUT_DIM, f);
    cudaMemcpy(net->d_b3, buf, sizeof(float)*OUT_DIM, cudaMemcpyHostToDevice);
    free(buf);
    fclose(f);
    return 1;
}

// -----------------------------------------------------------------------------
// FEN parsing (host)
// -----------------------------------------------------------------------------
static void parse_fen_compact(const char* fen, CompactBoard* cb) {
    memset(cb, 0, sizeof(CompactBoard));
    int sq = 0, i = 0;
    while (fen[i] && fen[i] != ' ') {
        char c = fen[i++];
        if (c == '/') continue;
        if (c >= '1' && c <= '8') { sq += (c - '0'); continue; }
        int piece = DOJO_EMPTY;
        switch (c) {
            case 'P': piece = DOJO_WPAWN; break;
            case 'N': piece = DOJO_WKNIGHT; break;
            case 'B': piece = DOJO_WBISHOP; break;
            case 'R': piece = DOJO_WROOK; break;
            case 'Q': piece = DOJO_WQUEEN; break;
            case 'K': piece = DOJO_WKING; break;
            case 'p': piece = DOJO_BPAWN; break;
            case 'n': piece = DOJO_BKNIGHT; break;
            case 'b': piece = DOJO_BBISHOP; break;
            case 'r': piece = DOJO_BROOK; break;
            case 'q': piece = DOJO_BQUEEN; break;
            case 'k': piece = DOJO_BKING; break;
        }
        if (sq < 64) cb->board[sq++] = (int8_t)piece;
    }
    while (fen[i] == ' ') i++;
    cb->side = (fen[i] == 'b') ? DOJO_BLACK : DOJO_WHITE;
}

// -----------------------------------------------------------------------------
// JSONL loader: extracts fen + sf_score_cp.
// -----------------------------------------------------------------------------
struct Sample {
    CompactBoard board;
    float        target;   // configurable target from side-to-move POV
    float        weight;
    uint32_t     relation_hash;
    uint8_t      source_id;
    uint8_t      band_id;
};

static int extract_str_field(const char* line, const char* key, char* out, int outsz) {
    char pat[64];
    snprintf(pat, sizeof(pat), "\"%s\"", key);
    const char* p = strstr(line, pat);
    if (!p) return 0;
    p = strchr(p, ':'); if (!p) return 0;
    p++; while (*p == ' ' || *p == '\t') p++;
    if (*p != '"') return 0;
    p++;
    int i = 0;
    while (*p && *p != '"' && i < outsz - 1) out[i++] = *p++;
    out[i] = 0;
    return i > 0;
}

static int extract_int_field(const char* line, const char* key, long* out) {
    char pat[64];
    snprintf(pat, sizeof(pat), "\"%s\"", key);
    const char* p = strstr(line, pat);
    if (!p) return 0;
    p = strchr(p, ':'); if (!p) return 0;
    p++; while (*p == ' ' || *p == '\t') p++;
    char* endp = NULL;
    long v = strtol(p, &endp, 10);
    if (endp == p) return 0;
    *out = v;
    return 1;
}

static int extract_float_field(const char* line, const char* key, float* out) {
    char pat[64];
    snprintf(pat, sizeof(pat), "\"%s\"", key);
    const char* p = strstr(line, pat);
    if (!p) return 0;
    p = strchr(p, ':'); if (!p) return 0;
    p++; while (*p == ' ' || *p == '\t') p++;
    char* endp = NULL;
    float v = strtof(p, &endp);
    if (endp == p) return 0;
    *out = v;
    return 1;
}

static int load_jsonl(const char* path,
                      const char* target_key,
                      const char* weight_key,
                      const char* source_key,
                      const char* band_key,
                      const char* relation_key,
                      Sample** out_samples, int* out_n,
                      LabelTable* sources, LabelTable* bands) {
    FILE* f = fopen(path, "r");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return 0; }
    int cap = 16384, n = 0;
    Sample* arr = (Sample*)malloc(sizeof(Sample) * cap);
    char line[2048];
    char fen[256];
    char source[64];
    char band[64];
    char relation[128];
    while (fgets(line, sizeof(line), f)) {
        float score = 0.0f;
        float weight = 1.0f;
        if (!extract_str_field(line, "fen", fen, sizeof(fen))) continue;
        if (!extract_float_field(line, target_key, &score)) continue;
        if (weight_key && weight_key[0]) {
            float parsed_weight = 1.0f;
            if (extract_float_field(line, weight_key, &parsed_weight) && parsed_weight > 0.0f) {
                weight = parsed_weight;
            }
        }
        if (!extract_str_field(line, source_key, source, sizeof(source))) snprintf(source, sizeof(source), "%s", "unknown");
        if (!extract_str_field(line, band_key, band, sizeof(band))) snprintf(band, sizeof(band), "%s", "unknown");
        if (!extract_str_field(line, relation_key, relation, sizeof(relation))) snprintf(relation, sizeof(relation), "%s", "unknown");
        if (n >= cap) {
            cap *= 2;
            arr = (Sample*)realloc(arr, sizeof(Sample) * cap);
        }
        parse_fen_compact(fen, &arr[n].board);
        arr[n].target = score;
        arr[n].weight = weight;
        arr[n].source_id = (uint8_t)intern_label(sources, source);
        arr[n].band_id = (uint8_t)intern_label(bands, band);
        arr[n].relation_hash = fnv1a32(relation);
        n++;
    }
    fclose(f);
    *out_samples = arr;
    *out_n = n;
    return 1;
}

// -----------------------------------------------------------------------------
// Forward + inference helpers
// -----------------------------------------------------------------------------
static void forward(cublasHandle_t cublas, const Net* net,
                    const float* d_X, float* d_H1, float* d_H2, float* d_Y, int N) {
    const float alpha = 1.0f, beta = 0.0f;
    int bs = 256;

    // H1 = W1^T @ X  : (H1 x N) = (H1 x IN) @ (IN x N)
    CUBLAS_OK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
        H1_DIM, N, IN_DIM,
        &alpha, net->d_W1, IN_DIM,
        d_X, IN_DIM,
        &beta, d_H1, H1_DIM));
    {
        int total = H1_DIM * N;
        bias_add_kernel<<<(total+bs-1)/bs, bs>>>(d_H1, net->d_b1, N, H1_DIM);
        relu_kernel<<<(total+bs-1)/bs, bs>>>(d_H1, total);
    }

    // H2 = W2^T @ H1
    CUBLAS_OK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
        H2_DIM, N, H1_DIM,
        &alpha, net->d_W2, H1_DIM,
        d_H1, H1_DIM,
        &beta, d_H2, H2_DIM));
    {
        int total = H2_DIM * N;
        bias_add_kernel<<<(total+bs-1)/bs, bs>>>(d_H2, net->d_b2, N, H2_DIM);
        relu_kernel<<<(total+bs-1)/bs, bs>>>(d_H2, total);
    }

    // Y = W3^T @ H2
    CUBLAS_OK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
        OUT_DIM, N, H2_DIM,
        &alpha, net->d_W3, H2_DIM,
        d_H2, H2_DIM,
        &beta, d_Y, OUT_DIM));
    {
        int total = OUT_DIM * N;
        bias_add_kernel<<<(total+bs-1)/bs, bs>>>(d_Y, net->d_b3, N, OUT_DIM);
    }
}

// -----------------------------------------------------------------------------
// Backward pass.
//   Inputs (all column-major):
//     X   [IN x N] (constant), H1 [H1 x N] (post-ReLU), H2 [H2 x N] (post-ReLU),
//     dY  [1 x N]
//   Computes grads w.r.t. W3, b3, W2, b2, W1, b1.
//
// Layer 3 (Y = W3^T @ H2 + b3):
//   dW3 [H2 x 1] = H2 @ dY^T          ->  cublasSgemm(N, T, H2, 1, N)
//   db3 [1]      = sum_cols(dY)
//   dH2 [H2 x N] = W3 @ dY            ->  cublasSgemm(N, N, H2, N, 1)
//   apply ReLU' (mask by H2 > 0)
//
// Layer 2 (H2 = W2^T @ H1 + b2):
//   dW2 [H1 x H2] = H1 @ dH2^T        ->  cublasSgemm(N, T, H1, H2, N)
//   db2 [H2]      = sum_cols(dH2)
//   dH1 [H1 x N]  = W2 @ dH2          ->  cublasSgemm(N, N, H1, N, H2)
//   apply ReLU' (mask by H1 > 0)
//
// Layer 1 (H1 = W1^T @ X + b1):
//   dW1 [IN x H1] = X @ dH1^T         ->  cublasSgemm(N, T, IN, H1, N)
//   db1 [H1]      = sum_cols(dH1)
// -----------------------------------------------------------------------------
static void backward(cublasHandle_t cublas, const Net* net, Grad* g,
                     const float* d_X,
                     float* d_H1, float* d_H2,    // post-ReLU activations
                     const float* d_dY,
                     float* d_dH1, float* d_dH2,
                     int N) {
    const float alpha = 1.0f, beta = 0.0f;
    int bs = 256;

    // ---- Layer 3 ----
    // dW3 [H2 x 1] = H2 [H2 x N] @ dY^T [N x 1]
    CUBLAS_OK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T,
        H2_DIM, OUT_DIM, N,
        &alpha, d_H2, H2_DIM,
        d_dY, OUT_DIM,
        &beta, g->d_W3, H2_DIM));

    // db3 = sum over columns of dY
    {
        int threads = 128;
        col_sum_kernel<<<OUT_DIM, threads, threads*sizeof(float)>>>(d_dY, g->d_b3, OUT_DIM, N);
    }

    // dH2 [H2 x N] = W3 [H2 x 1] @ dY [1 x N]
    CUBLAS_OK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
        H2_DIM, N, OUT_DIM,
        &alpha, net->d_W3, H2_DIM,
        d_dY, OUT_DIM,
        &beta, d_dH2, H2_DIM));

    // ReLU' mask using the post-activation H2 (h2>0)
    {
        int total = H2_DIM * N;
        relu_backward_kernel<<<(total+bs-1)/bs, bs>>>(d_dH2, d_H2, total);
    }

    // ---- Layer 2 ----
    // dW2 [H1 x H2] = H1 [H1 x N] @ dH2^T [N x H2]
    CUBLAS_OK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T,
        H1_DIM, H2_DIM, N,
        &alpha, d_H1, H1_DIM,
        d_dH2, H2_DIM,
        &beta, g->d_W2, H1_DIM));

    // db2 = sum cols of dH2
    {
        int threads = 128;
        col_sum_kernel<<<H2_DIM, threads, threads*sizeof(float)>>>(d_dH2, g->d_b2, H2_DIM, N);
    }

    // dH1 [H1 x N] = W2 [H1 x H2] @ dH2 [H2 x N]
    CUBLAS_OK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
        H1_DIM, N, H2_DIM,
        &alpha, net->d_W2, H1_DIM,
        d_dH2, H2_DIM,
        &beta, d_dH1, H1_DIM));

    {
        int total = H1_DIM * N;
        relu_backward_kernel<<<(total+bs-1)/bs, bs>>>(d_dH1, d_H1, total);
    }

    // ---- Layer 1 ----
    // dW1 [IN x H1] = X [IN x N] @ dH1^T [N x H1]
    CUBLAS_OK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T,
        IN_DIM, H1_DIM, N,
        &alpha, d_X, IN_DIM,
        d_dH1, H1_DIM,
        &beta, g->d_W1, IN_DIM));

    // db1 = sum cols of dH1
    {
        int threads = 128;
        col_sum_kernel<<<H1_DIM, threads, threads*sizeof(float)>>>(d_dH1, g->d_b1, H1_DIM, N);
    }
}

// -----------------------------------------------------------------------------
// Adam step over all parameter tensors.
// -----------------------------------------------------------------------------
static void adam_step(Net* net, Adam* a, Grad* g, int t,
                      float lr, float beta1, float beta2, float eps) {
    float bc1 = 1.0f - powf(beta1, (float)t);
    float bc2 = 1.0f - powf(beta2, (float)t);
    int bs = 256;

    auto step = [&](float* w, float* m, float* v, float* gr, int n) {
        adam_kernel<<<(n+bs-1)/bs, bs>>>(w, m, v, gr, lr, beta1, beta2, eps, bc1, bc2, n);
    };

    step(net->d_W1, a->m_W1, a->v_W1, g->d_W1, IN_DIM*H1_DIM);
    step(net->d_b1, a->m_b1, a->v_b1, g->d_b1, H1_DIM);
    step(net->d_W2, a->m_W2, a->v_W2, g->d_W2, H1_DIM*H2_DIM);
    step(net->d_b2, a->m_b2, a->v_b2, g->d_b2, H2_DIM);
    step(net->d_W3, a->m_W3, a->v_W3, g->d_W3, H2_DIM*OUT_DIM);
    step(net->d_b3, a->m_b3, a->v_b3, g->d_b3, OUT_DIM);
}

// -----------------------------------------------------------------------------
// Eval helpers: compute (mse, mae) on a slice of samples.
// We process in chunks of <= max_batch.
// -----------------------------------------------------------------------------
struct EvalBufs {
    CompactBoard* d_boards;
    float *d_X, *d_H1, *d_H2, *d_Y, *d_target, *d_sq, *d_abs;
    int max_batch;
};

static void eval_alloc(EvalBufs* b, int max_batch) {
    b->max_batch = max_batch;
    CUDA_OK(cudaMalloc(&b->d_boards, sizeof(CompactBoard)*max_batch));
    CUDA_OK(cudaMalloc(&b->d_X, sizeof(float)*IN_DIM*max_batch));
    CUDA_OK(cudaMalloc(&b->d_H1, sizeof(float)*H1_DIM*max_batch));
    CUDA_OK(cudaMalloc(&b->d_H2, sizeof(float)*H2_DIM*max_batch));
    CUDA_OK(cudaMalloc(&b->d_Y, sizeof(float)*OUT_DIM*max_batch));
    CUDA_OK(cudaMalloc(&b->d_target, sizeof(float)*max_batch));
    CUDA_OK(cudaMalloc(&b->d_sq, sizeof(float)*max_batch));
    CUDA_OK(cudaMalloc(&b->d_abs, sizeof(float)*max_batch));
}
static void eval_free(EvalBufs* b) {
    cudaFree(b->d_boards); cudaFree(b->d_X); cudaFree(b->d_H1); cudaFree(b->d_H2);
    cudaFree(b->d_Y); cudaFree(b->d_target); cudaFree(b->d_sq); cudaFree(b->d_abs);
}

// Sums a device array of length n into a host float (single-shot reduction).
static float device_sum(const float* d, int n, float* d_scratch) {
    CUDA_OK(cudaMemset(d_scratch, 0, sizeof(float)));
    int bs = 256;
    sum_reduce_kernel<<<(n+bs-1)/bs, bs, bs*sizeof(float)>>>(d, d_scratch, n);
    float h;
    CUDA_OK(cudaMemcpy(&h, d_scratch, sizeof(float), cudaMemcpyDeviceToHost));
    return h;
}

static void eval_holdout(cublasHandle_t cublas, const Net* net, EvalBufs* b,
                         const Sample* samples, const int* idx, int n,
                         float* out_mse, float* out_mae) {
    double total_sq = 0.0, total_abs = 0.0;
    int processed = 0;

    // device scratch for reduction
    float* d_scratch;
    CUDA_OK(cudaMalloc(&d_scratch, sizeof(float)));

    // staging on host
    CompactBoard* h_boards = (CompactBoard*)malloc(sizeof(CompactBoard) * b->max_batch);
    float* h_targets = (float*)malloc(sizeof(float) * b->max_batch);

    for (int off = 0; off < n; off += b->max_batch) {
        int B = (off + b->max_batch <= n) ? b->max_batch : (n - off);
        for (int i = 0; i < B; i++) {
            int s = idx[off + i];
            h_boards[i] = samples[s].board;
            h_targets[i] = samples[s].target;
        }
        CUDA_OK(cudaMemcpy(b->d_boards, h_boards, sizeof(CompactBoard)*B, cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(b->d_target, h_targets, sizeof(float)*B, cudaMemcpyHostToDevice));
        extract_features_kernel<<<B, 64>>>(b->d_boards, b->d_X, B);
        forward(cublas, net, b->d_X, b->d_H1, b->d_H2, b->d_Y, B);
        int bs = 256;
        sqerr_kernel<<<(B+bs-1)/bs, bs>>>(b->d_Y, b->d_target, b->d_sq, b->d_abs, B);

        total_sq  += device_sum(b->d_sq,  B, d_scratch);
        total_abs += device_sum(b->d_abs, B, d_scratch);
        processed += B;
    }

    cudaFree(d_scratch);
    free(h_boards);
    free(h_targets);

    *out_mse = (float)(total_sq / (double)processed);
    *out_mae = (float)(total_abs / (double)processed);
}

static void shuffle_ints(int* arr, int n) {
    for (int i = n - 1; i > 0; --i) {
        int j = rand() % (i + 1);
        int t = arr[i];
        arr[i] = arr[j];
        arr[j] = t;
    }
}

static void build_epoch_order_shuffle(const int* train_idx, int n_train, int* out) {
    for (int i = 0; i < n_train; ++i) out[i] = train_idx[i];
    shuffle_ints(out, n_train);
}

static void build_epoch_order_pressure_band(const Sample* samples, const int* train_idx, int n_train,
                                            const LabelTable* bands, const BandWeights* weights,
                                            int* out) {
    std::vector<std::vector<int>> by_band(bands->names.size());
    for (int i = 0; i < n_train; ++i) {
        const Sample& s = samples[train_idx[i]];
        if (s.band_id >= by_band.size()) continue;
        by_band[s.band_id].push_back(train_idx[i]);
    }
    for (auto& bucket : by_band) {
        if (!bucket.empty()) shuffle_ints(bucket.data(), (int)bucket.size());
    }
    std::vector<int> cursor(by_band.size(), 0);
    for (int out_i = 0; out_i < n_train; ++out_i) {
        float total_mass = 0.0f;
        for (int b = 0; b < (int)by_band.size(); ++b) {
            int remain = (int)by_band[b].size() - cursor[b];
            if (remain <= 0) continue;
            total_mass += remain * band_weight_for_name(weights, label_name(bands, b));
        }
        if (total_mass <= 0.0f) break;
        float pick = ((float)rand() / (float)RAND_MAX) * total_mass;
        int chosen = -1;
        for (int b = 0; b < (int)by_band.size(); ++b) {
            int remain = (int)by_band[b].size() - cursor[b];
            if (remain <= 0) continue;
            pick -= remain * band_weight_for_name(weights, label_name(bands, b));
            if (pick <= 0.0f) { chosen = b; break; }
        }
        if (chosen < 0) {
            for (int b = 0; b < (int)by_band.size(); ++b) {
                if (cursor[b] < (int)by_band[b].size()) { chosen = b; break; }
            }
        }
        out[out_i] = by_band[chosen][cursor[chosen]++];
    }
}

struct RelationEntry {
    uint32_t relation_hash;
    int index;
    int band_id;
};

static void build_epoch_order_relation_cluster(const Sample* samples, const int* train_idx, int n_train,
                                               int* out) {
    std::vector<RelationEntry> entries;
    entries.reserve(n_train);
    for (int i = 0; i < n_train; ++i) {
        const Sample& s = samples[train_idx[i]];
        entries.push_back({s.relation_hash, train_idx[i], (int)s.band_id});
    }
    std::sort(entries.begin(), entries.end(), [](const RelationEntry& a, const RelationEntry& b) {
        if (a.relation_hash != b.relation_hash) return a.relation_hash < b.relation_hash;
        return a.index < b.index;
    });

    std::vector<std::vector<int>> clusters;
    for (size_t i = 0; i < entries.size();) {
        size_t j = i + 1;
        while (j < entries.size() && entries[j].relation_hash == entries[i].relation_hash) ++j;
        std::vector<int> cluster;
        cluster.reserve(j - i);
        for (size_t k = i; k < j; ++k) cluster.push_back(entries[k].index);
        shuffle_ints(cluster.data(), (int)cluster.size());
        clusters.push_back(cluster);
        i = j;
    }
    if (!clusters.empty()) {
        for (int i = (int)clusters.size() - 1; i > 0; --i) {
            int j = rand() % (i + 1);
            std::swap(clusters[i], clusters[j]);
        }
    }
    int cursor = 0;
    for (const auto& cluster : clusters) {
        for (int idx : cluster) out[cursor++] = idx;
    }
}

struct BandCluster {
    int band_id;
    std::vector<int> indices;
};

static void build_epoch_order_banded_relation_cluster(const Sample* samples, const int* train_idx, int n_train,
                                                      const LabelTable* bands, const BandWeights* weights,
                                                      int* out) {
    std::vector<RelationEntry> entries;
    entries.reserve(n_train);
    for (int i = 0; i < n_train; ++i) {
        const Sample& s = samples[train_idx[i]];
        entries.push_back({s.relation_hash, train_idx[i], (int)s.band_id});
    }
    std::sort(entries.begin(), entries.end(), [](const RelationEntry& a, const RelationEntry& b) {
        if (a.band_id != b.band_id) return a.band_id < b.band_id;
        if (a.relation_hash != b.relation_hash) return a.relation_hash < b.relation_hash;
        return a.index < b.index;
    });

    std::vector<std::vector<BandCluster>> by_band(bands->names.size());
    for (size_t i = 0; i < entries.size();) {
        size_t j = i + 1;
        while (j < entries.size() &&
               entries[j].band_id == entries[i].band_id &&
               entries[j].relation_hash == entries[i].relation_hash) ++j;
        BandCluster cluster{};
        cluster.band_id = entries[i].band_id;
        cluster.indices.reserve(j - i);
        for (size_t k = i; k < j; ++k) cluster.indices.push_back(entries[k].index);
        shuffle_ints(cluster.indices.data(), (int)cluster.indices.size());
        if (cluster.band_id >= 0 && cluster.band_id < (int)by_band.size()) {
            by_band[cluster.band_id].push_back(cluster);
        }
        i = j;
    }

    for (auto& clusters : by_band) {
        if (clusters.empty()) continue;
        for (int i = (int)clusters.size() - 1; i > 0; --i) {
            int j = rand() % (i + 1);
            std::swap(clusters[i], clusters[j]);
        }
    }

    std::vector<int> cluster_cursor(by_band.size(), 0);
    int written = 0;
    while (written < n_train) {
        float total_mass = 0.0f;
        for (int b = 0; b < (int)by_band.size(); ++b) {
            int remain_clusters = (int)by_band[b].size() - cluster_cursor[b];
            if (remain_clusters <= 0) continue;
            int remain_items = 0;
            for (int c = cluster_cursor[b]; c < (int)by_band[b].size(); ++c) {
                remain_items += (int)by_band[b][c].indices.size();
            }
            total_mass += remain_items * band_weight_for_name(weights, label_name(bands, b));
        }
        if (total_mass <= 0.0f) break;
        float pick = ((float)rand() / (float)RAND_MAX) * total_mass;
        int chosen = -1;
        for (int b = 0; b < (int)by_band.size(); ++b) {
            if (cluster_cursor[b] >= (int)by_band[b].size()) continue;
            int remain_items = 0;
            for (int c = cluster_cursor[b]; c < (int)by_band[b].size(); ++c) {
                remain_items += (int)by_band[b][c].indices.size();
            }
            pick -= remain_items * band_weight_for_name(weights, label_name(bands, b));
            if (pick <= 0.0f) { chosen = b; break; }
        }
        if (chosen < 0) {
            for (int b = 0; b < (int)by_band.size(); ++b) {
                if (cluster_cursor[b] < (int)by_band[b].size()) { chosen = b; break; }
            }
        }
        const BandCluster& cluster = by_band[chosen][cluster_cursor[chosen]++];
        for (int idx : cluster.indices) out[written++] = idx;
    }
}

static void build_epoch_order(AccessMode mode, const Sample* samples, const int* train_idx, int n_train,
                              const LabelTable* bands, const BandWeights* weights, int* out) {
    switch (mode) {
        case ACCESS_PRESSURE_BAND:
            build_epoch_order_pressure_band(samples, train_idx, n_train, bands, weights, out);
            break;
        case ACCESS_RELATION_CLUSTER:
            build_epoch_order_relation_cluster(samples, train_idx, n_train, out);
            break;
        case ACCESS_BANDED_RELATION_CLUSTER:
            build_epoch_order_banded_relation_cluster(samples, train_idx, n_train, bands, weights, out);
            break;
        case ACCESS_SHUFFLE:
        default:
            build_epoch_order_shuffle(train_idx, n_train, out);
            break;
    }
}

static SplitMetrics collect_subset_metrics(cublasHandle_t cublas, const Net* before, const Net* after,
                                           EvalBufs* eb, const Sample* samples, const int* idx, int n,
                                           int source_id, int band_id) {
    std::vector<int> subset;
    subset.reserve(n);
    for (int i = 0; i < n; ++i) {
        const Sample& s = samples[idx[i]];
        if ((source_id >= 0 && (int)s.source_id != source_id) ||
            (band_id >= 0 && (int)s.band_id != band_id)) {
            continue;
        }
        subset.push_back(idx[i]);
    }
    SplitMetrics m{};
    m.count = (int)subset.size();
    if (subset.empty()) return m;
    eval_holdout(cublas, before, eb, samples, subset.data(), (int)subset.size(), &m.before_mse, &m.before_mae);
    eval_holdout(cublas, after, eb, samples, subset.data(), (int)subset.size(), &m.after_mse, &m.after_mae);
    return m;
}

static void write_metrics_json(FILE* fp, const char* name, const SplitMetrics& m, bool trailing) {
    fprintf(fp,
            "    \"%s\": {\"count\": %d, \"before_mse\": %.6f, \"before_mae\": %.6f, \"after_mse\": %.6f, \"after_mae\": %.6f, \"mae_gain\": %.6f}%s\n",
            name, m.count, m.before_mse, m.before_mae, m.after_mse, m.after_mae, m.before_mae - m.after_mae,
            trailing ? "," : "");
}

// -----------------------------------------------------------------------------
// MAIN
// -----------------------------------------------------------------------------
int main(int argc, char** argv) {
    const char* data_path = "../gpu_spine/book.jsonl";
    const char* target_key = "sf_score_cp";
    const char* weight_key = "sample_weight";
    const char* source_key = "source";
    const char* band_key = "pressure_band";
    const char* relation_key = "relation_bucket";
    const char* report_json_path = NULL;
    const char* access_mode_str = "shuffle";
    const char* band_weights_spec = "";
    int epochs = 50;
    int batch  = 256;
    float lr   = 1e-3f;
    int seed   = 1337;

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--data")  && i+1 < argc) data_path = argv[++i];
        else if (!strcmp(argv[i], "--target-key") && i+1 < argc) target_key = argv[++i];
        else if (!strcmp(argv[i], "--weight-key") && i+1 < argc) weight_key = argv[++i];
        else if (!strcmp(argv[i], "--source-key") && i+1 < argc) source_key = argv[++i];
        else if (!strcmp(argv[i], "--band-key") && i+1 < argc) band_key = argv[++i];
        else if (!strcmp(argv[i], "--relation-key") && i+1 < argc) relation_key = argv[++i];
        else if (!strcmp(argv[i], "--report-json") && i+1 < argc) report_json_path = argv[++i];
        else if (!strcmp(argv[i], "--access-mode") && i+1 < argc) access_mode_str = argv[++i];
        else if (!strcmp(argv[i], "--band-weights") && i+1 < argc) band_weights_spec = argv[++i];
        else if (!strcmp(argv[i], "--epochs")&& i+1 < argc) epochs    = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--batch") && i+1 < argc) batch     = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--lr")    && i+1 < argc) lr        = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--seed")  && i+1 < argc) seed      = atoi(argv[++i]);
    }

    AccessMode access_mode = parse_access_mode(access_mode_str);
    BandWeights band_weights{};
    parse_band_weights(band_weights_spec, &band_weights);

    fprintf(stderr, "[nn_train] data=%s epochs=%d batch=%d lr=%g seed=%d target=%s weight=%s access=%s\n",
            data_path, epochs, batch, lr, seed, target_key, weight_key, access_mode_str);

    Sample* samples = NULL;
    int total_n = 0;
    LabelTable source_labels{};
    LabelTable band_labels{};
    if (!load_jsonl(data_path, target_key, weight_key, source_key, band_key, relation_key,
                    &samples, &total_n, &source_labels, &band_labels) || total_n < 100) {
        fprintf(stderr, "Failed to load samples or too few (%d)\n", total_n);
        return 1;
    }
    fprintf(stderr, "[nn_train] loaded %d samples\n", total_n);

    // 80/20 split — use a deterministic shuffle of all indices first, then split.
    int* perm = (int*)malloc(sizeof(int) * total_n);
    for (int i = 0; i < total_n; i++) perm[i] = i;
    srand((unsigned)seed);
    for (int i = total_n - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int t = perm[i]; perm[i] = perm[j]; perm[j] = t;
    }
    int n_train = (int)(0.8 * total_n);
    int n_hold  = total_n - n_train;
    int* train_idx = perm;
    int* hold_idx  = perm + n_train;
    fprintf(stderr, "[nn_train] split train=%d holdout=%d\n", n_train, n_hold);

    // Backup existing weights file if present
    FILE* tf = fopen(WEIGHTS_FILE, "rb");
    if (tf) {
        fclose(tf);
        FILE* bak = fopen(WEIGHTS_BACK, "rb");
        if (!bak) {
            // copy
            FILE* in = fopen(WEIGHTS_FILE, "rb");
            FILE* out = fopen(WEIGHTS_BACK, "wb");
            if (in && out) {
                char buf[4096];
                size_t r;
                while ((r = fread(buf, 1, sizeof(buf), in)) > 0) fwrite(buf, 1, r, out);
                fclose(in); fclose(out);
                fprintf(stderr, "[nn_train] backed up existing weights -> %s\n", WEIGHTS_BACK);
            }
        } else {
            fclose(bak);
            fprintf(stderr, "[nn_train] backup %s already exists; not overwriting\n", WEIGHTS_BACK);
        }
    }

    // ---- Evaluate the existing (random) weights as our "before" baseline ----
    Net random_net{};
    net_alloc(&random_net);
    bool have_existing = false;
    {
        FILE* f = fopen(WEIGHTS_FILE, "rb");
        if (f) { fclose(f); have_existing = net_load(&random_net, WEIGHTS_FILE); }
    }
    if (!have_existing) {
        fprintf(stderr, "[nn_train] no existing weights file found; init random for 'before' baseline\n");
        net_init(&random_net, (unsigned long long)seed);
    }

    cublasHandle_t cublas;
    CUBLAS_OK(cublasCreate(&cublas));

    EvalBufs eb{};
    int eval_batch = 1024;
    eval_alloc(&eb, eval_batch);

    float before_mse = 0.0f, before_mae = 0.0f;
    eval_holdout(cublas, &random_net, &eb, samples, hold_idx, n_hold, &before_mse, &before_mae);
    fprintf(stderr, "[nn_train] BEFORE (random/existing) holdout MSE=%.2f  MAE=%.2f cp\n",
            before_mse, before_mae);

    // ---- Initialize a fresh net for training (He init) ----
    Net net{};
    net_alloc(&net);
    net_init(&net, (unsigned long long)seed);

    Grad grad{};
    grad_alloc(&grad);
    Adam adam{};
    adam_alloc(&adam);

    // Training device buffers (sized for max batch)
    CompactBoard* d_boards;
    float *d_X, *d_H1, *d_H2, *d_Y, *d_target, *d_weight, *d_dY, *d_dH1, *d_dH2, *d_sq;
    CUDA_OK(cudaMalloc(&d_boards, sizeof(CompactBoard)*batch));
    CUDA_OK(cudaMalloc(&d_X, sizeof(float)*IN_DIM*batch));
    CUDA_OK(cudaMalloc(&d_H1, sizeof(float)*H1_DIM*batch));
    CUDA_OK(cudaMalloc(&d_H2, sizeof(float)*H2_DIM*batch));
    CUDA_OK(cudaMalloc(&d_Y,  sizeof(float)*OUT_DIM*batch));
    CUDA_OK(cudaMalloc(&d_target, sizeof(float)*batch));
    CUDA_OK(cudaMalloc(&d_weight, sizeof(float)*batch));
    CUDA_OK(cudaMalloc(&d_dY, sizeof(float)*OUT_DIM*batch));
    CUDA_OK(cudaMalloc(&d_dH1, sizeof(float)*H1_DIM*batch));
    CUDA_OK(cudaMalloc(&d_dH2, sizeof(float)*H2_DIM*batch));
    CUDA_OK(cudaMalloc(&d_sq, sizeof(float)*batch));

    // host staging
    CompactBoard* h_boards = (CompactBoard*)malloc(sizeof(CompactBoard) * batch);
    float* h_targets = (float*)malloc(sizeof(float) * batch);
    float* h_weights = (float*)malloc(sizeof(float) * batch);

    // device scratch for batch loss reduction
    float* d_scratch;
    CUDA_OK(cudaMalloc(&d_scratch, sizeof(float)));

    int* shuf = (int*)malloc(sizeof(int) * n_train);
    for (int i = 0; i < n_train; i++) shuf[i] = train_idx[i];

    int t = 0;                    // global Adam step counter
    float prev_hold_mse = 1e30f;
    float best_hold_mse = 1e30f;
    int   best_epoch = 0;
    int   plateau = 0;
    const float beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f;

    // Snapshot weights for best-holdout checkpoint (kept on host).
    size_t nW1 = (size_t)IN_DIM*H1_DIM, nW2 = (size_t)H1_DIM*H2_DIM, nW3 = (size_t)H2_DIM*OUT_DIM;
    float* best_W1 = (float*)malloc(sizeof(float)*nW1);
    float* best_b1 = (float*)malloc(sizeof(float)*H1_DIM);
    float* best_W2 = (float*)malloc(sizeof(float)*nW2);
    float* best_b2 = (float*)malloc(sizeof(float)*H2_DIM);
    float* best_W3 = (float*)malloc(sizeof(float)*nW3);
    float* best_b3 = (float*)malloc(sizeof(float)*OUT_DIM);

    fprintf(stderr, "[nn_train] starting training (%d epochs, batch %d, lr %g)\n",
            epochs, batch, lr);

    for (int ep = 1; ep <= epochs; ep++) {
        build_epoch_order(access_mode, samples, train_idx, n_train, &band_labels, &band_weights, shuf);

        double train_sq_sum = 0.0;
        long   train_n_sum  = 0;

        for (int off = 0; off + batch <= n_train; off += batch) {
            for (int i = 0; i < batch; i++) {
                int s = shuf[off + i];
                h_boards[i]  = samples[s].board;
                h_targets[i] = samples[s].target;
                h_weights[i] = samples[s].weight;
            }
            CUDA_OK(cudaMemcpy(d_boards, h_boards, sizeof(CompactBoard)*batch, cudaMemcpyHostToDevice));
            CUDA_OK(cudaMemcpy(d_target, h_targets, sizeof(float)*batch, cudaMemcpyHostToDevice));
            CUDA_OK(cudaMemcpy(d_weight, h_weights, sizeof(float)*batch, cudaMemcpyHostToDevice));
            extract_features_kernel<<<batch, 64>>>(d_boards, d_X, batch);
            forward(cublas, &net, d_X, d_H1, d_H2, d_Y, batch);

            // dL/dY = (Y - target) / batch ; sq holds (y-t)^2
            int bs = 256;
            mse_grad_kernel<<<(batch+bs-1)/bs, bs>>>(d_Y, d_target, d_weight, d_dY, d_sq,
                                                    batch, 1.0f / (float)batch);

            // accumulate train MSE for reporting
            float bs_sum = device_sum(d_sq, batch, d_scratch);
            train_sq_sum += bs_sum;
            train_n_sum  += batch;

            // backward
            backward(cublas, &net, &grad, d_X, d_H1, d_H2, d_dY, d_dH1, d_dH2, batch);

            // adam step
            t++;
            adam_step(&net, &adam, &grad, t, lr, beta1, beta2, eps);
        }

        float train_mse = (float)(train_sq_sum / (double)train_n_sum);
        float hold_mse = 0.0f, hold_mae = 0.0f;
        eval_holdout(cublas, &net, &eb, samples, hold_idx, n_hold, &hold_mse, &hold_mae);

        // Snapshot best-holdout weights (host copy).
        bool improved = (hold_mse < best_hold_mse - 1.0f);
        if (improved) {
            best_hold_mse = hold_mse;
            best_epoch = ep;
            cudaMemcpy(best_W1, net.d_W1, sizeof(float)*nW1, cudaMemcpyDeviceToHost);
            cudaMemcpy(best_b1, net.d_b1, sizeof(float)*H1_DIM, cudaMemcpyDeviceToHost);
            cudaMemcpy(best_W2, net.d_W2, sizeof(float)*nW2, cudaMemcpyDeviceToHost);
            cudaMemcpy(best_b2, net.d_b2, sizeof(float)*H2_DIM, cudaMemcpyDeviceToHost);
            cudaMemcpy(best_W3, net.d_W3, sizeof(float)*nW3, cudaMemcpyDeviceToHost);
            cudaMemcpy(best_b3, net.d_b3, sizeof(float)*OUT_DIM, cudaMemcpyDeviceToHost);
        }

        fprintf(stderr, "[ep %2d] train_mse=%.2f  hold_mse=%.2f  hold_mae=%.2f cp%s\n",
                ep, train_mse, hold_mse, hold_mae, improved ? "  <-- best" : "");

        // checkpoint best-holdout to disk every 5 epochs
        if (ep % 5 == 0 || ep == epochs) {
            // restore best to device, save, then restore current
            cudaMemcpy(net.d_W1, best_W1, sizeof(float)*nW1, cudaMemcpyHostToDevice);
            cudaMemcpy(net.d_b1, best_b1, sizeof(float)*H1_DIM, cudaMemcpyHostToDevice);
            cudaMemcpy(net.d_W2, best_W2, sizeof(float)*nW2, cudaMemcpyHostToDevice);
            cudaMemcpy(net.d_b2, best_b2, sizeof(float)*H2_DIM, cudaMemcpyHostToDevice);
            cudaMemcpy(net.d_W3, best_W3, sizeof(float)*nW3, cudaMemcpyHostToDevice);
            cudaMemcpy(net.d_b3, best_b3, sizeof(float)*OUT_DIM, cudaMemcpyHostToDevice);
            net_save(&net, WEIGHTS_FILE);
            fprintf(stderr, "         saved best (epoch %d, hold_mse=%.2f) -> %s\n",
                    best_epoch, best_hold_mse, WEIGHTS_FILE);
            // (we don't need to restore "current" — Adam state is on m/v not on weights;
            //  but restoring lets training continue from whatever it was. To keep things
            //  reproducible, we leave the device with BEST weights so subsequent training
            //  continues from the best-known point. This acts as a soft restart.)
        }

        // plateau detection on holdout MSE (no improvement vs best)
        if (hold_mse + 1e-3f >= best_hold_mse) plateau++;
        else plateau = 0;
        prev_hold_mse = hold_mse;
        if (plateau >= 10 && ep >= 25) {
            fprintf(stderr, "[nn_train] holdout has not improved for %d epochs (best ep %d, mse %.2f); stopping at ep %d\n",
                    plateau, best_epoch, best_hold_mse, ep);
            break;
        }
    }

    // Restore best weights and save final
    cudaMemcpy(net.d_W1, best_W1, sizeof(float)*nW1, cudaMemcpyHostToDevice);
    cudaMemcpy(net.d_b1, best_b1, sizeof(float)*H1_DIM, cudaMemcpyHostToDevice);
    cudaMemcpy(net.d_W2, best_W2, sizeof(float)*nW2, cudaMemcpyHostToDevice);
    cudaMemcpy(net.d_b2, best_b2, sizeof(float)*H2_DIM, cudaMemcpyHostToDevice);
    cudaMemcpy(net.d_W3, best_W3, sizeof(float)*nW3, cudaMemcpyHostToDevice);
    cudaMemcpy(net.d_b3, best_b3, sizeof(float)*OUT_DIM, cudaMemcpyHostToDevice);
    net_save(&net, WEIGHTS_FILE);
    fprintf(stderr, "[nn_train] restored best-holdout weights (epoch %d) and saved -> %s\n",
            best_epoch, WEIGHTS_FILE);

    // ---- Final evaluation: trained vs random ----
    float after_mse = 0.0f, after_mae = 0.0f;
    eval_holdout(cublas, &net, &eb, samples, hold_idx, n_hold, &after_mse, &after_mae);
    SplitMetrics total_metrics{};
    total_metrics.count = n_hold;
    total_metrics.before_mse = before_mse;
    total_metrics.before_mae = before_mae;
    total_metrics.after_mse = after_mse;
    total_metrics.after_mae = after_mae;

    fprintf(stderr, "\n========== TRAINING SUMMARY ==========\n");
    fprintf(stderr, "Holdout positions: %d\n", n_hold);
    fprintf(stderr, "BEFORE (random):  MSE=%.2f  MAE=%.2f cp\n", before_mse, before_mae);
    fprintf(stderr, "AFTER  (trained): MSE=%.2f  MAE=%.2f cp\n", after_mse,  after_mae);
    fprintf(stderr, "MAE improvement:  %.2f cp (%.1f%%)\n",
            before_mae - after_mae,
            100.0f * (before_mae - after_mae) / (before_mae > 1e-6f ? before_mae : 1.0f));
    fprintf(stderr, "Weights saved to %s (random baseline preserved at %s)\n",
            WEIGHTS_FILE, WEIGHTS_BACK);
    fprintf(stderr, "======================================\n");

    std::vector<SplitMetrics> source_metrics(source_labels.names.size());
    for (int i = 0; i < (int)source_labels.names.size(); ++i) {
        source_metrics[i] = collect_subset_metrics(cublas, &random_net, &net, &eb, samples, hold_idx, n_hold, i, -1);
    }
    std::vector<SplitMetrics> band_metrics(band_labels.names.size());
    for (int i = 0; i < (int)band_labels.names.size(); ++i) {
        band_metrics[i] = collect_subset_metrics(cublas, &random_net, &net, &eb, samples, hold_idx, n_hold, -1, i);
    }

    if (report_json_path && report_json_path[0]) {
        FILE* report = fopen(report_json_path, "w");
        if (!report) {
            fprintf(stderr, "[nn_train] failed to open report path %s\n", report_json_path);
        } else {
            fprintf(report, "{\n");
            fprintf(report, "  \"data_path\": \"%s\",\n", data_path);
            fprintf(report, "  \"target_key\": \"%s\",\n", target_key);
            fprintf(report, "  \"weight_key\": \"%s\",\n", weight_key);
            fprintf(report, "  \"access_mode\": \"%s\",\n", access_mode_str);
            fprintf(report, "  \"epochs\": %d,\n", epochs);
            fprintf(report, "  \"batch\": %d,\n", batch);
            fprintf(report, "  \"lr\": %.8f,\n", lr);
            fprintf(report, "  \"seed\": %d,\n", seed);
            fprintf(report, "  \"band_weights\": {\n");
            fprintf(report, "    \"fallback\": %.4f,\n", band_weights.fallback);
            fprintf(report, "    \"library\": %.4f,\n", band_weights.library);
            fprintf(report, "    \"cool\": %.4f,\n", band_weights.cool);
            fprintf(report, "    \"warm\": %.4f,\n", band_weights.warm);
            fprintf(report, "    \"hot\": %.4f,\n", band_weights.hot);
            fprintf(report, "    \"fracture\": %.4f\n", band_weights.fracture);
            fprintf(report, "  },\n");
            fprintf(report, "  \"total\": {\n");
            fprintf(report, "    \"count\": %d,\n", total_metrics.count);
            fprintf(report, "    \"before_mse\": %.6f,\n", total_metrics.before_mse);
            fprintf(report, "    \"before_mae\": %.6f,\n", total_metrics.before_mae);
            fprintf(report, "    \"after_mse\": %.6f,\n", total_metrics.after_mse);
            fprintf(report, "    \"after_mae\": %.6f,\n", total_metrics.after_mae);
            fprintf(report, "    \"mae_gain\": %.6f\n", total_metrics.before_mae - total_metrics.after_mae);
            fprintf(report, "  },\n");
            fprintf(report, "  \"by_source\": {\n");
            for (int i = 0; i < (int)source_labels.names.size(); ++i) {
                write_metrics_json(report, label_name(&source_labels, i), source_metrics[i], i + 1 < (int)source_labels.names.size());
            }
            fprintf(report, "  },\n");
            fprintf(report, "  \"by_pressure_band\": {\n");
            for (int i = 0; i < (int)band_labels.names.size(); ++i) {
                write_metrics_json(report, label_name(&band_labels, i), band_metrics[i], i + 1 < (int)band_labels.names.size());
            }
            fprintf(report, "  }\n");
            fprintf(report, "}\n");
            fclose(report);
        }
    }

    // Cleanup
    free(best_W1); free(best_b1); free(best_W2); free(best_b2); free(best_W3); free(best_b3);
    free(shuf); free(h_boards); free(h_targets); free(h_weights); free(perm); free(samples);
    cudaFree(d_boards); cudaFree(d_X); cudaFree(d_H1); cudaFree(d_H2);
    cudaFree(d_Y); cudaFree(d_target); cudaFree(d_weight); cudaFree(d_dY); cudaFree(d_dH1);
    cudaFree(d_dH2); cudaFree(d_sq); cudaFree(d_scratch);
    eval_free(&eb);
    grad_free(&grad);
    adam_free(&adam);
    net_free(&net);
    net_free(&random_net);
    cublasDestroy(cublas);

    return 0;
}
