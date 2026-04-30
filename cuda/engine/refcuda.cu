// =============================================================================
// refcuda.cu — extern "C" GPU chess referee API for Rust/Python FFI.
//
// Exposes cuda-dojo's GPU chess primitives as a stable C ABI. Each entry
// point dispatches actual chess work to GPU kernels (movegen / make_unmake /
// in_check / coord3d run on the device). Host code is just orchestration:
// FEN parsing, host↔device copies, kernel launches.
//
// Built as `librefcuda.so` for FFI loading by Rust (cust / libloading) or
// Python (ctypes / pyo3 + cust). The chess rules are in CUDA C++ once and
// shared by all callers.
// =============================================================================
#include <cstring>
#include <cstdint>
#include <cstdio>
#include <vector>

#include <cuda_runtime.h>

#include "engine_types.h"
#include "movegen.cuh"
#include "make_unmake.cuh"
#include "attacks.cuh"
#include "coord3d.cuh"
#include "search.cuh"
#include "uci.h"
#include "tt.cuh"
#include "zobrist.cuh"

using namespace engine;

// Stub for the only uci.cu symbol search.cu uses. We never run a UCI
// command loop in this .so so external stop signals don't apply.
namespace uci {
    bool stop_requested() { return false; }
}

// =============================================================================
// GPU kernels — chess work runs HERE, not on host.
// =============================================================================
namespace {

__global__ void k_legal_moves(const Position* s, Move* out, int* out_count) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    Move pseudo[MAX_MOVES];
    int n = generate_moves(s, pseudo);
    int legal = 0;
    for (int i = 0; i < n; ++i) {
        Position c = *s;
        make_move(&c, pseudo[i]);
        if (in_check(&c, 1 - c.side)) continue;
        out[legal++] = pseudo[i];
    }
    *out_count = legal;
}

__global__ void k_make_move(const Position* s_in, Move mv, Position* s_out) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    *s_out = *s_in;
    make_move(s_out, mv);
}

__global__ void k_in_check(const Position* s, int* out) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    *out = in_check(s, s->side) ? 1 : 0;
}

__global__ void k_legal_count(const Position* s, int* out) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    Move buf[MAX_MOVES];
    int n = generate_moves(s, buf);
    int legal = 0;
    for (int i = 0; i < n; ++i) {
        Position c = *s;
        make_move(&c, buf[i]);
        if (!in_check(&c, 1 - c.side)) legal++;
    }
    *out = legal;
}

__global__ void k_coord3d(const Position* s, float* out_xyz, int* out_octant) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    Coord3D c = compute_coord3d(s);
    out_xyz[0] = c.x;
    out_xyz[1] = c.y;
    out_xyz[2] = c.z;
    *out_octant = c.octant_id;
}

// ----- Batched kernels -----------------------------------------------------
// Each block handles one position from the input array. Single thread per
// block (the inner work is serial today; SIMT-parallelizing per-position is
// a follow-up). The win vs single-position kernels is one launch +
// one synchronize for the whole batch instead of N each.

__global__ void k_legal_moves_batched(const Position* s, int n,
                                      Move* out, int* out_counts) {
    int idx = blockIdx.x;
    if (idx >= n || threadIdx.x != 0) return;
    Move pseudo[MAX_MOVES];
    int m = generate_moves(&s[idx], pseudo);
    Move* my_out = out + idx * MAX_MOVES;
    int legal = 0;
    for (int i = 0; i < m; ++i) {
        Position c = s[idx];
        make_move(&c, pseudo[i]);
        if (in_check(&c, 1 - c.side)) continue;
        my_out[legal++] = pseudo[i];
    }
    out_counts[idx] = legal;
}

__global__ void k_make_move_batched(const Position* s_in, int n,
                                    const Move* mvs, Position* s_out) {
    int idx = blockIdx.x;
    if (idx >= n || threadIdx.x != 0) return;
    s_out[idx] = s_in[idx];
    make_move(&s_out[idx], mvs[idx]);
}

__global__ void k_in_check_batched(const Position* s, int n, int* out) {
    int idx = blockIdx.x;
    if (idx >= n || threadIdx.x != 0) return;
    out[idx] = in_check(&s[idx], s[idx].side) ? 1 : 0;
}

} // anonymous namespace

// =============================================================================
// extern "C" API — stable ABI for FFI.
// =============================================================================
extern "C" {

/// Allocate a Position on the host. Caller must free with refc_position_free.
/// Returns NULL on alloc failure.
Position* refc_position_new(void) {
    Position* p = (Position*)std::calloc(1, sizeof(Position));
    return p;
}

void refc_position_free(Position* p) {
    if (p) std::free(p);
}

/// Parse a FEN string into a Position (host work; non-chess parsing).
/// Returns 0 on success, non-zero on parse failure.
int refc_parse_fen(const char* fen, Position* out) {
    if (!fen || !out) return -1;
    parse_fen(fen, out);
    return 0;
}

/// Compute legal moves for `pos` ON GPU. Writes up to `max_n` moves into
/// `moves_out` and returns the count actually written. The caller's
/// `moves_out` is host memory of length ≥ max_n.
int refc_legal_moves(const Position* pos, int32_t* moves_out, int max_n) {
    if (!pos || !moves_out || max_n <= 0) return -1;

    Position* d_pos = nullptr;
    Move* d_moves = nullptr;
    int* d_count = nullptr;
    cudaMalloc(&d_pos, sizeof(Position));
    cudaMalloc(&d_moves, sizeof(Move) * MAX_MOVES);
    cudaMalloc(&d_count, sizeof(int));

    cudaMemcpy(d_pos, pos, sizeof(Position), cudaMemcpyHostToDevice);
    k_legal_moves<<<1, 1>>>(d_pos, d_moves, d_count);
    cudaDeviceSynchronize();

    int count = 0;
    cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
    if (count > max_n) count = max_n;
    cudaMemcpy(moves_out, d_moves, sizeof(int32_t) * count,
               cudaMemcpyDeviceToHost);

    cudaFree(d_pos);
    cudaFree(d_moves);
    cudaFree(d_count);
    return count;
}

/// Apply move `mv` to `pos`, writing result into `out` (must be a different
/// Position). Move execution runs ON GPU.
int refc_make_move(const Position* pos, int32_t mv, Position* out) {
    if (!pos || !out) return -1;

    Position *d_in = nullptr, *d_out = nullptr;
    cudaMalloc(&d_in, sizeof(Position));
    cudaMalloc(&d_out, sizeof(Position));

    cudaMemcpy(d_in, pos, sizeof(Position), cudaMemcpyHostToDevice);
    k_make_move<<<1, 1>>>(d_in, mv, d_out);
    cudaDeviceSynchronize();
    cudaMemcpy(out, d_out, sizeof(Position), cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}

/// Side-to-move's king under attack (in_check on GPU).
int refc_is_check(const Position* pos) {
    if (!pos) return 0;
    Position* d_pos = nullptr;
    int* d_out = nullptr;
    cudaMalloc(&d_pos, sizeof(Position));
    cudaMalloc(&d_out, sizeof(int));

    cudaMemcpy(d_pos, pos, sizeof(Position), cudaMemcpyHostToDevice);
    k_in_check<<<1, 1>>>(d_pos, d_out);
    cudaDeviceSynchronize();
    int v = 0;
    cudaMemcpy(&v, d_out, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_pos);
    cudaFree(d_out);
    return v;
}

/// Number of legal moves at this position (computed on GPU).
int refc_legal_count(const Position* pos) {
    if (!pos) return 0;
    Position* d_pos = nullptr;
    int* d_out = nullptr;
    cudaMalloc(&d_pos, sizeof(Position));
    cudaMalloc(&d_out, sizeof(int));

    cudaMemcpy(d_pos, pos, sizeof(Position), cudaMemcpyHostToDevice);
    k_legal_count<<<1, 1>>>(d_pos, d_out);
    cudaDeviceSynchronize();
    int v = 0;
    cudaMemcpy(&v, d_out, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_pos);
    cudaFree(d_out);
    return v;
}

/// is_checkmate: in check AND zero legal moves.
int refc_is_checkmate(const Position* pos) {
    return refc_is_check(pos) && refc_legal_count(pos) == 0;
}

/// is_stalemate: NOT in check AND zero legal moves.
int refc_is_stalemate(const Position* pos) {
    return !refc_is_check(pos) && refc_legal_count(pos) == 0;
}

/// 3D coord (x,y,z) + octant id. Returns 0 on success.
int refc_coord3d(const Position* pos, float* xyz_out, int* octant_out) {
    if (!pos || !xyz_out || !octant_out) return -1;
    Position* d_pos = nullptr;
    float* d_xyz = nullptr;
    int* d_oct = nullptr;
    cudaMalloc(&d_pos, sizeof(Position));
    cudaMalloc(&d_xyz, sizeof(float) * 3);
    cudaMalloc(&d_oct, sizeof(int));

    cudaMemcpy(d_pos, pos, sizeof(Position), cudaMemcpyHostToDevice);
    k_coord3d<<<1, 1>>>(d_pos, d_xyz, d_oct);
    cudaDeviceSynchronize();
    cudaMemcpy(xyz_out, d_xyz, sizeof(float) * 3, cudaMemcpyDeviceToHost);
    cudaMemcpy(octant_out, d_oct, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_pos);
    cudaFree(d_xyz);
    cudaFree(d_oct);
    return 0;
}

/// UCI string for a move (host-only; format conversion is not chess work).
/// `uci_out` must hold at least 6 bytes (longest UCI = "e7e8q\0").
void refc_move_to_uci(int32_t mv, char* uci_out) {
    if (!uci_out) return;
    move_to_uci(mv, uci_out);
}

/// Sizeof exposed so FFI callers can allocate Position correctly.
int refc_position_size(void) {
    return (int)sizeof(Position);
}

// ============================================================================
// Batched FFI — process N positions in a single launch.
//
// Each batched call replaces N single-position calls and pays:
//   - 1 cudaMalloc per buffer (vs N)
//   - 1 H→D copy of size N×T (vs N copies of size T)
//   - 1 kernel launch (vs N)
//   - 1 cudaDeviceSynchronize (vs N)
//   - 1 D→H copy back (vs N)
//
// Output layout for legal_moves_batched:
//   moves_out[i * MAX_MOVES + 0 .. counts_out[i]-1] = legal moves for pos[i]
//   moves_out is host-side, length ≥ n * MAX_MOVES int32_t.
//   counts_out is host-side, length ≥ n.
// ============================================================================

/// Compute legal moves for N positions in one launch.
int refc_legal_moves_batched(const Position* positions, int n,
                             int32_t* moves_out, int* counts_out) {
    if (!positions || !moves_out || !counts_out || n <= 0) return -1;

    Position* d_pos = nullptr;
    Move* d_moves = nullptr;
    int* d_counts = nullptr;
    cudaMalloc(&d_pos, sizeof(Position) * n);
    cudaMalloc(&d_moves, sizeof(Move) * MAX_MOVES * n);
    cudaMalloc(&d_counts, sizeof(int) * n);

    cudaMemcpy(d_pos, positions, sizeof(Position) * n, cudaMemcpyHostToDevice);
    k_legal_moves_batched<<<n, 1>>>(d_pos, n, d_moves, d_counts);
    cudaDeviceSynchronize();

    cudaMemcpy(counts_out, d_counts, sizeof(int) * n, cudaMemcpyDeviceToHost);
    cudaMemcpy(moves_out, d_moves, sizeof(int32_t) * MAX_MOVES * n,
               cudaMemcpyDeviceToHost);

    cudaFree(d_pos);
    cudaFree(d_moves);
    cudaFree(d_counts);
    return 0;
}

/// Apply N moves to N positions, output N successor positions.
int refc_make_move_batched(const Position* positions, int n,
                           const int32_t* moves, Position* out) {
    if (!positions || !moves || !out || n <= 0) return -1;

    Position *d_in = nullptr, *d_out = nullptr;
    Move* d_mvs = nullptr;
    cudaMalloc(&d_in, sizeof(Position) * n);
    cudaMalloc(&d_out, sizeof(Position) * n);
    cudaMalloc(&d_mvs, sizeof(Move) * n);

    cudaMemcpy(d_in, positions, sizeof(Position) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mvs, moves, sizeof(Move) * n, cudaMemcpyHostToDevice);
    k_make_move_batched<<<n, 1>>>(d_in, n, d_mvs, d_out);
    cudaDeviceSynchronize();
    cudaMemcpy(out, d_out, sizeof(Position) * n, cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_mvs);
    return 0;
}

/// In-check predicate for N positions.
int refc_is_check_batched(const Position* positions, int n, int* out) {
    if (!positions || !out || n <= 0) return -1;

    Position* d_pos = nullptr;
    int* d_out = nullptr;
    cudaMalloc(&d_pos, sizeof(Position) * n);
    cudaMalloc(&d_out, sizeof(int) * n);

    cudaMemcpy(d_pos, positions, sizeof(Position) * n, cudaMemcpyHostToDevice);
    k_in_check_batched<<<n, 1>>>(d_pos, n, d_out);
    cudaDeviceSynchronize();
    cudaMemcpy(out, d_out, sizeof(int) * n, cudaMemcpyDeviceToHost);

    cudaFree(d_pos);
    cudaFree(d_out);
    return 0;
}

// =============================================================================
// Search — alpha-beta + PeSTO via cuda-dojo's existing GPU search kernels.
// Calls search_root which runs iterative deepening with TT/zobrist on GPU.
// =============================================================================

static bool g_search_initialised = false;

/// Initialize search subsystem (TT, zobrist, root buffers). Idempotent.
void refc_search_init(void) {
    if (!g_search_initialised) {
        if (!zobrist_initialized()) init_zobrist();
        tt_init();
        search_init();
        g_search_initialised = true;
    }
}

/// Reset between games (clears the host-side stop flag mirror).
void refc_search_new_game(void) {
    if (!g_search_initialised) refc_search_init();
    search_new_game();
}

/// Free search-side device memory.
void refc_search_shutdown(void) {
    if (g_search_initialised) {
        search_shutdown();
        g_search_initialised = false;
    }
}

/// Search `pos` to fixed depth (or movetime). Writes UCI bestmove (≤6 bytes
/// incl. NUL) into `uci_out`, score (cp from STM POV) into `score_out`.
/// Returns 0 on success, non-zero on error.
///
/// One of (depth, movetime_ms) should be > 0. depth takes precedence.
int refc_search(const Position* pos, int depth, int movetime_ms,
                char* uci_out, int* score_out)
{
    if (!pos || !uci_out) return -1;
    if (!g_search_initialised) refc_search_init();

    uci::SearchLimits limits;
    if (depth > 0) limits.depth = depth;
    else if (movetime_ms > 0) limits.movetime = movetime_ms;
    else limits.depth = 5;  // sensible default

    int last_score = 0;
    auto cb = [&](const uci::SearchInfo& info) {
        last_score = info.score_cp;
    };

    uci::SearchResult res = search_root(*pos, limits, cb);

    // Copy bestmove UCI (max 6 chars incl. NUL).
    std::strncpy(uci_out, res.bestmove.c_str(), 6);
    uci_out[5] = '\0';
    if (score_out) *score_out = last_score;
    return 0;
}

/// Convenience: search to depth, then return the encoded Move + score.
/// Maps the UCI string back to a Move by enumerating legal_moves and
/// matching by UCI text. Slightly slower than direct search but lets
/// callers stay in the typed Move/Position world.
int refc_search_best_move(const Position* pos, int depth, int movetime_ms,
                            int32_t* move_out, int* score_out)
{
    if (!pos || !move_out) return -1;

    char uci[6] = {0};
    int score = 0;
    if (refc_search(pos, depth, movetime_ms, uci, &score) != 0) return -1;
    if (score_out) *score_out = score;

    // Match UCI to a legal move.
    int32_t legal[MAX_MOVES];
    int n = refc_legal_moves(pos, legal, MAX_MOVES);
    for (int i = 0; i < n; ++i) {
        char buf[8] = {0};
        move_to_uci(legal[i], buf);
        if (std::strncmp(buf, uci, 6) == 0) {
            *move_out = legal[i];
            return 0;
        }
    }
    // Fell through — search returned a non-legal UCI? Use first legal.
    if (n > 0) {
        *move_out = legal[0];
        return 1;  // soft failure
    }
    return -1;
}

} // extern "C"
