// =============================================================================
// make_unmake.cuh
//
// Public API for making/unmaking moves on engine::Position.
//
//   make_move(pos, mv)            — extracted verbatim from gpu_fighter
//                                   (passes all perft tests in current build)
//   make_move(pos, mv, undo)      — same, but populates an Undo record so
//                                   the move can be reverted in place.
//   unmake_move(pos, mv, undo)    — true in-place reversal, paired 1:1 with
//                                   the recording overload above.
//
// The single-arg make_move() replicates the prior search behavior exactly
// (no Undo, callers either copy the Position before calling or accept the
// modification). The recording variant is a strict superset — it never
// changes the board mutation, only adds bookkeeping.
// =============================================================================
#ifndef ENGINE_MAKE_UNMAKE_CUH
#define ENGINE_MAKE_UNMAKE_CUH

#include "engine_types.h"

namespace engine {

// Per-move undo record. Captures everything that make_move can mutate
// beyond the from/to squares (which are recovered from the move itself).
struct Undo {
    int8_t  captured_piece;   // piece sitting on `to` before the move
    int8_t  ep_captured_sq;   // square of the ep-captured pawn, -1 if none
    int8_t  ep_before;        // s->ep before the move
    int8_t  castle_before;    // s->castle before the move
    int16_t halfmove_before;  // s->halfmove before the move
    int16_t fullmove_before;  // s->fullmove before the move
    int8_t  king_before[2];   // s->kingPos before the move
};

// Apply move to position in place. No undo record. Same body as
// gpu_fighter's d_make_move / h_make_move.
__host__ __device__ void make_move(Position* s, Move move);

// Apply move; populate `u` so a paired unmake_move call restores
// byte-identical state.
__host__ __device__ void make_move(Position* s, Move move, Undo* u);

// True in-place reversal of `make_move(s, move, u)`.
__host__ __device__ void unmake_move(Position* s, Move move, const Undo* u);

// Device aliases — match gpu_fighter naming so search code can be ported
// with a one-line rename.
__device__ inline void d_make_move(Position* s, Move move) { make_move(s, move); }

// Host alias.
inline void h_make_move(Position* s, Move move) { make_move(s, move); }

} // namespace engine

#endif // ENGINE_MAKE_UNMAKE_CUH
