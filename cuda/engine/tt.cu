// =============================================================================
// tt.cu -- implementation of tt.cuh
//
// Single-slot transposition table with versioned publication.
// Physical entry layout (16 bytes, naturally aligned):
//
//     uint64_t guard ;  // = key XOR data
//     uint64_t data  ;  // packed (score|depth|bound|age|move)
//
// A separate 32-bit version word serializes writers per slot:
//   * odd  version -> write in progress
//   * even version -> stable payload
//
// Probes read a stable even version before and after sampling the payload.
// Stores acquire the slot with atomicCAS, update the payload, then publish the
// next even version. This keeps the search-facing 16-byte TT slot layout while
// eliminating mixed guard/data reads on current hardware.
// =============================================================================
#include "tt.cuh"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

namespace engine {

// ---------- physical 16-byte slot ----------
struct __align__(16) TTSlot {
    uint64_t guard;   // key XOR data
    uint64_t data;    // packed payload
};
static_assert(sizeof(TTSlot) == 16, "TTSlot must be exactly 16 bytes");
static_assert(alignof(TTSlot) == 16, "TTSlot must be 16-byte aligned");

// ---------- packing helpers ----------
//
//   bits  0..15  : score   (int16, biased into uint16 view)
//   bits 16..31  : depth   (int16, ditto)
//   bits 32..39  : bound   (uint8)
//   bits 40..47  : age     (uint8)
//   bits 48..63  : move    (uint16)
__host__ __device__ static inline uint64_t pack_data(int16_t score, int16_t depth,
                                                     uint8_t bound, uint8_t age,
                                                     uint16_t move) {
    uint64_t s = (uint16_t)score;
    uint64_t d = (uint16_t)depth;
    uint64_t b = bound;
    uint64_t a = age;
    uint64_t m = move;
    return (s & 0xFFFFULL)
         | ((d & 0xFFFFULL) << 16)
         | ((b & 0xFFULL)   << 32)
         | ((a & 0xFFULL)   << 40)
         | ((m & 0xFFFFULL) << 48);
}
__host__ __device__ static inline void unpack_data(uint64_t data,
                                                   int16_t& score, int16_t& depth,
                                                   uint8_t& bound, uint8_t& age,
                                                   uint16_t& move) {
    score = (int16_t)(uint16_t)( data        & 0xFFFFULL);
    depth = (int16_t)(uint16_t)((data >> 16) & 0xFFFFULL);
    bound = (uint8_t)          ((data >> 32) & 0xFFULL);
    age   = (uint8_t)          ((data >> 40) & 0xFFULL);
    move  = (uint16_t)         ((data >> 48) & 0xFFFFULL);
}

// ---------- device globals ----------
__device__ TTSlot*  d_tt_table       = nullptr;
__device__ uint32_t* d_tt_versions   = nullptr;
__device__ uint64_t d_tt_mask        = 0;
__device__ uint8_t  d_tt_current_age = 0;

// ---------- host mirrors ----------
static TTSlot*  g_h_tt_dev_ptr = nullptr;
static uint32_t* g_h_tt_versions = nullptr;
static uint64_t g_h_tt_n       = 0;
static uint64_t g_h_tt_mask    = 0;
static uint8_t  g_h_tt_age     = 0;

static uint64_t floor_pow2(uint64_t x) {
    if (x == 0) return 0;
    uint64_t p = 1;
    while ((p << 1) <= x) p <<= 1;
    return p;
}

static void push_device_globals() {
    cudaError_t err;
    err = cudaMemcpyToSymbol(d_tt_table,       &g_h_tt_dev_ptr, sizeof(TTSlot*));
    if (err != cudaSuccess) fprintf(stderr, "tt: push d_tt_table: %s\n", cudaGetErrorString(err));
    err = cudaMemcpyToSymbol(d_tt_versions,    &g_h_tt_versions, sizeof(uint32_t*));
    if (err != cudaSuccess) fprintf(stderr, "tt: push d_tt_versions: %s\n", cudaGetErrorString(err));
    err = cudaMemcpyToSymbol(d_tt_mask,        &g_h_tt_mask,    sizeof(uint64_t));
    if (err != cudaSuccess) fprintf(stderr, "tt: push d_tt_mask: %s\n", cudaGetErrorString(err));
    err = cudaMemcpyToSymbol(d_tt_current_age, &g_h_tt_age,     sizeof(uint8_t));
    if (err != cudaSuccess) fprintf(stderr, "tt: push d_tt_age: %s\n", cudaGetErrorString(err));
}

// ---------- host API ----------
uint64_t tt_init(size_t size_mb) {
    if (g_h_tt_dev_ptr) tt_free();
    size_t bytes = size_mb * 1024ULL * 1024ULL;
    uint64_t want = bytes / sizeof(TTSlot);
    uint64_t n    = floor_pow2(want);
    if (n < 1024) n = 1024;
    cudaError_t err = cudaMalloc(&g_h_tt_dev_ptr, n * sizeof(TTSlot));
    if (err != cudaSuccess) {
        fprintf(stderr, "tt_init: cudaMalloc(%lu = %.1f MB) failed: %s\n",
                (unsigned long)n, n * sizeof(TTSlot) / 1048576.0,
                cudaGetErrorString(err));
        g_h_tt_dev_ptr = nullptr;
        return 0;
    }
    err = cudaMalloc(&g_h_tt_versions, n * sizeof(uint32_t));
    if (err != cudaSuccess) {
        fprintf(stderr, "tt_init: cudaMalloc(versions) failed: %s\n",
                cudaGetErrorString(err));
        cudaFree(g_h_tt_dev_ptr);
        g_h_tt_dev_ptr = nullptr;
        g_h_tt_versions = nullptr;
        return 0;
    }
    g_h_tt_n    = n;
    g_h_tt_mask = n - 1;
    g_h_tt_age  = 0;
    push_device_globals();
    tt_clear();
    return n;
}

void tt_resize(int size_mb) {
    tt_init((size_t)size_mb);
}

void tt_free() {
    if (g_h_tt_dev_ptr) {
        cudaFree(g_h_tt_dev_ptr);
        g_h_tt_dev_ptr = nullptr;
    }
    if (g_h_tt_versions) {
        cudaFree(g_h_tt_versions);
        g_h_tt_versions = nullptr;
    }
    g_h_tt_n = 0; g_h_tt_mask = 0; g_h_tt_age = 0;
    push_device_globals();
}

void tt_clear() {
    if (!g_h_tt_dev_ptr) return;
    cudaMemset(g_h_tt_dev_ptr, 0, g_h_tt_n * sizeof(TTSlot));
    if (g_h_tt_versions) {
        cudaMemset(g_h_tt_versions, 0, g_h_tt_n * sizeof(uint32_t));
    }
}

void tt_age() {
    g_h_tt_age = (uint8_t)((g_h_tt_age + 1) & 0xFF);
    cudaError_t err = cudaMemcpyToSymbol(d_tt_current_age, &g_h_tt_age, sizeof(uint8_t));
    if (err != cudaSuccess) fprintf(stderr, "tt_age: %s\n", cudaGetErrorString(err));
}

uint64_t tt_num_entries() { return g_h_tt_n; }
uint8_t  tt_current_age() { return g_h_tt_age; }

double tt_fill_rate() {
    if (!g_h_tt_dev_ptr || g_h_tt_n == 0) return 0.0;
    uint64_t sample = g_h_tt_n < 65536 ? g_h_tt_n : 65536;
    TTSlot* host = (TTSlot*)malloc(sample * sizeof(TTSlot));
    if (!host) return 0.0;
    cudaMemcpy(host, g_h_tt_dev_ptr, sample * sizeof(TTSlot), cudaMemcpyDeviceToHost);
    uint64_t filled = 0;
    for (uint64_t i = 0; i < sample; i++) {
        // A slot is "filled" iff guard XOR data != 0 (i.e. key != 0).
        uint64_t key = host[i].guard ^ host[i].data;
        if (key != 0) filled++;
    }
    free(host);
    return (double)filled / (double)sample;
}

// ---------- device API ----------
__device__ TTEntry tt_probe(uint64_t hash) {
    TTEntry out{};
    if (d_tt_table == nullptr || d_tt_versions == nullptr) return out;
    uint64_t idx = hash & d_tt_mask;
    volatile TTSlot* slot = &d_tt_table[idx];
    volatile uint32_t* version = &d_tt_versions[idx];

    for (int attempt = 0; attempt < 4; ++attempt) {
        uint32_t v1 = atomicAdd((unsigned int*)version, 0u);
        if ((v1 & 1u) != 0u) continue;
        uint64_t guard = slot->guard;
        uint64_t data  = slot->data;
        __threadfence();
        uint32_t v2 = atomicAdd((unsigned int*)version, 0u);
        if (v1 != v2 || (v2 & 1u) != 0u) continue;
        uint64_t key = guard ^ data;
        out.key = key;
        unpack_data(data, out.score, out.depth, out.bound, out.age, out.move);
        return out;
    }

    uint64_t key = 0;
    out.key = key;
    return out;
}

__device__ void tt_store(uint64_t hash, int score, int depth,
                          uint8_t bound, uint16_t move) {
    if (d_tt_table == nullptr || d_tt_versions == nullptr) return;
    uint64_t idx = hash & d_tt_mask;
    volatile TTSlot* slot = &d_tt_table[idx];
    volatile uint32_t* version = &d_tt_versions[idx];

    // Read existing slot (lockless) to evaluate replace policy. We only need
    // depth and age out of cur_data, not the full unpack.
    TTSlot cur_slot{};
    uint32_t lock_version = 0;
    for (;;) {
        lock_version = atomicAdd((unsigned int*)version, 0u);
        if ((lock_version & 1u) != 0u) continue;
        if (atomicCAS((unsigned int*)version, lock_version, lock_version + 1u) == lock_version) {
            cur_slot.guard = slot->guard;
            cur_slot.data = slot->data;
            break;
        }
    }
    uint64_t cur_key = cur_slot.guard ^ cur_slot.data;

    int16_t  cur_depth = (int16_t)(uint16_t)((cur_slot.data >> 16) & 0xFFFFULL);
    uint8_t  cur_age   = (uint8_t)          ((cur_slot.data >> 40) & 0xFFULL);

    bool empty = (cur_key == 0 && cur_depth == 0);
    bool replace = empty
                 || (cur_age != d_tt_current_age)
                 || ((int)depth >= (int)cur_depth);
    if (!replace) {
        __threadfence();
        atomicExch((unsigned int*)version, lock_version + 2u);
        return;
    }

    int s = score; if (s >  32767) s =  32767; if (s < -32768) s = -32768;
    int d = depth; if (d >  32767) d =  32767; if (d < -32768) d = -32768;

    TTSlot new_slot{};
    new_slot.data = pack_data((int16_t)s, (int16_t)d, bound, d_tt_current_age, move);
    new_slot.guard = hash ^ new_slot.data;
    slot->guard = new_slot.guard;
    slot->data = new_slot.data;
    __threadfence();
    atomicExch((unsigned int*)version, lock_version + 2u);
}

} // namespace engine
