// =============================================================================
// zobrist.cuh
//
// Zobrist hashing for engine::Position. 781 random 64-bit keys:
//   12 piece-types * 64 squares = 768
// + 1 side-to-move
// + 4 castling-rights bits
// + 8 en-passant files
// = 781
//
// Keys live in __constant__ memory on the device (and a mirrored host array
// for __host__ callers). Initialized once via init_zobrist(seed). Seed is
// fixed across runs so logged hashes are reproducible.
//
// Two compute paths:
//   zobrist_full(pos)            -- compute from scratch (used for first
//                                   position and as a sanity oracle)
//   zobrist_update(h, pos, mv)   -- incremental update: given the hash of
//                                   pos BEFORE the move and the move itself,
//                                   return the hash AFTER the move. Reads
//                                   pos.board to know capture victim, etc.
//                                   Does NOT mutate pos.
//
// For 1000 random positions, walking from start with zobrist_full(start) +
// applying zobrist_update over each move must equal zobrist_full(end).
// test_zobrist verifies this.
// =============================================================================
#ifndef ENGINE_ZOBRIST_CUH
#define ENGINE_ZOBRIST_CUH

#include <stdint.h>
#include <cuda_runtime.h>
#include "engine_types.h"

namespace engine {

// Default seed -- chosen once, never change. Reproducible across runs.
constexpr uint64_t ZOBRIST_DEFAULT_SEED = 0xC0FFEE1234567890ULL;

// One-time host setup. Populates a __constant__ key table on the device
// AND a mirrored host array used by __host__ callers. Idempotent: calling
// it twice with the same seed is a no-op.
void init_zobrist(uint64_t seed = ZOBRIST_DEFAULT_SEED);

// True iff init_zobrist has been called this process.
bool zobrist_initialized();

// Compute hash from scratch. Both host and device callable.
__host__ __device__ uint64_t zobrist_full(const Position& pos);

// Incremental hash update. Takes the hash of `pos_before` and the move that
// will be applied to `pos_before`. Returns the hash that `zobrist_full`
// would produce on the resulting position (i.e. AFTER d_make_move).
//
// pos_before is read but NOT modified; the caller is responsible for
// applying the move separately.
__host__ __device__ uint64_t zobrist_update(uint64_t hash_before,
                                             const Position& pos_before,
                                             Move mv);

// Low-level accessors -- exposed for tests / debug. Index ranges:
//   piece in [1..12]   square in [0..63]
//   castle_bit in [0..3]  (CASTLE_WK=0, CASTLE_WQ=1, CASTLE_BK=2, CASTLE_BQ=3)
//   ep_file in [0..7]
__host__ __device__ uint64_t zobrist_piece_key(int piece, int sq);
__host__ __device__ uint64_t zobrist_side_key();
__host__ __device__ uint64_t zobrist_castle_key(int castle_bit);
__host__ __device__ uint64_t zobrist_ep_file_key(int file);

} // namespace engine

#endif // ENGINE_ZOBRIST_CUH
