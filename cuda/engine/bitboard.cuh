// =============================================================================
// bitboard.cuh
//
// Tiny bit-twiddling helpers. The current verified movegen uses an 8x8
// mailbox (int8_t[64]), so bitboards are not yet used in the hot path; these
// utilities are provided so future search variants can switch over without
// touching the canonical types.
// =============================================================================
#ifndef ENGINE_BITBOARD_CUH
#define ENGINE_BITBOARD_CUH

#include <stdint.h>
#include <cuda_runtime.h>

namespace engine {

using Bitboard = uint64_t;

__host__ __device__ inline Bitboard bb_set(int sq)              { return Bitboard(1) << sq; }
__host__ __device__ inline bool     bb_test(Bitboard b, int sq) { return (b >> sq) & 1ULL; }
__host__ __device__ inline Bitboard bb_or(Bitboard a, Bitboard b)  { return a | b; }
__host__ __device__ inline Bitboard bb_and(Bitboard a, Bitboard b) { return a & b; }
__host__ __device__ inline Bitboard bb_clear(Bitboard b, int sq)   { return b & ~(Bitboard(1) << sq); }

__host__ __device__ inline int bb_popcount(Bitboard b) {
#ifdef __CUDA_ARCH__
    return __popcll(b);
#else
    return __builtin_popcountll(b);
#endif
}

__host__ __device__ inline int bb_lsb(Bitboard b) {
#ifdef __CUDA_ARCH__
    return __ffsll(b) - 1;
#else
    return __builtin_ctzll(b);
#endif
}

} // namespace engine

#endif // ENGINE_BITBOARD_CUH
