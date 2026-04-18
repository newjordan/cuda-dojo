// =============================================================================
// movegen.cuh
//
// Pseudo-legal move generation (caller filters legality with in_check after
// make_move) extracted verbatim from gpu_fighter.cu.
//
// generate_moves passes perft on all 6 canonical chessprogramming.org
// positions at depth 4-5 (224 positions, no mismatches) — that contract is
// the verification target for engine_test_perft.
// =============================================================================
#ifndef ENGINE_MOVEGEN_CUH
#define ENGINE_MOVEGEN_CUH

#include "engine_types.h"
#include "attacks.cuh"
#include "make_unmake.cuh"

namespace engine {

// Generate all pseudo-legal moves for `s->side` into `moves`. Returns count.
// `moves` must hold at least MAX_MOVES entries.
__host__ __device__ int generate_moves(const Position* s, Move* moves);

// Device alias.
__device__ inline int d_generate_moves(const Position* s, Move* moves) {
    return generate_moves(s, moves);
}

// Host alias.
inline int h_generate_moves(const Position* s, Move* moves) {
    return generate_moves(s, moves);
}

// Host perft (full legality filter via in_check after make).
// Reproduces the contract from gpu_fighter.cu's h_perft.
long long perft(Position* s, int depth);

// Parse a FEN string into a Position. Same semantics as gpu_fighter parse_fen.
void parse_fen(const char* fen, Position* s);

// Encode a Move in long-algebraic UCI form into `uci` (caller supplies >=6 bytes).
void move_to_uci(Move move, char* uci);

} // namespace engine

#endif // ENGINE_MOVEGEN_CUH
