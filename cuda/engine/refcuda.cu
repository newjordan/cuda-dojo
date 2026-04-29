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

using namespace engine;

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

} // extern "C"
