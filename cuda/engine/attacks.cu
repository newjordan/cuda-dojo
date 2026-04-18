// =============================================================================
// attacks.cu
//
// Implementation of square-attacked / in-check, extracted verbatim from
// gpu_fighter.cu's d_is_square_attacked and h_is_sq_attacked (which were
// already byte-identical except for compiler annotations). Single
// __host__ __device__ body.
// =============================================================================
#include "attacks.cuh"

namespace engine {

// Public constant-memory tables (other .cu files can reference these).
__constant__ int d_KNIGHT_OFFSETS[8] = {-17, -15, -10, -6, 6, 10, 15, 17};
__constant__ int d_KING_OFFSETS[8]   = { -9,  -8,  -7, -1, 1,  7,  8,  9};

// Private helper: absolute value usable on host and device.
__host__ __device__ static inline int abs_i(int x) { return x < 0 ? -x : x; }

__host__ __device__ bool is_square_attacked(const Position* s, int sq, int by_side) {
    // Local copies of offset tables — works on both host and device without
    // depending on __constant__ memory (which is device-only).
    const int KN[8] = {-17, -15, -10, -6, 6, 10, 15, 17};
    const int KG[8] = { -9,  -8,  -7, -1, 1,  7,  8,  9};

    int r = sq_rank(sq), f = sq_file(sq);

    // pawn attacks
    if (by_side == WHITE_SIDE) {
        if (r < 7) {
            if (f > 0 && s->board[sq + 7] == WPAWN) return true;
            if (f < 7 && s->board[sq + 9] == WPAWN) return true;
        }
    } else {
        if (r > 0) {
            if (f > 0 && s->board[sq - 9] == BPAWN) return true;
            if (f < 7 && s->board[sq - 7] == BPAWN) return true;
        }
    }

    // knight attacks
    int knightPiece = (by_side == WHITE_SIDE) ? WKNIGHT : BKNIGHT;
    for (int i = 0; i < 8; i++) {
        int t = sq + KN[i];
        if (t < 0 || t >= 64) continue;
        int dr = abs_i(sq_rank(t) - r), df = abs_i(sq_file(t) - f);
        if ((dr == 2 && df == 1) || (dr == 1 && df == 2))
            if (s->board[t] == knightPiece) return true;
    }

    // king attacks
    int kingPiece = (by_side == WHITE_SIDE) ? WKING : BKING;
    for (int i = 0; i < 8; i++) {
        int t = sq + KG[i];
        if (t < 0 || t >= 64) continue;
        int dr = abs_i(sq_rank(t) - r), df = abs_i(sq_file(t) - f);
        if (dr <= 1 && df <= 1)
            if (s->board[t] == kingPiece) return true;
    }

    // bishop / queen diagonals
    int bishop = (by_side == WHITE_SIDE) ? WBISHOP : BBISHOP;
    int queen  = (by_side == WHITE_SIDE) ? WQUEEN  : BQUEEN;
    int ddr[] = {-1,-1,1,1}, ddf[] = {-1,1,-1,1};
    for (int d = 0; d < 4; d++) {
        int cr = r + ddr[d], cf = f + ddf[d];
        while (on_board(cr, cf)) {
            int p = s->board[make_sq(cr, cf)];
            if (p != EMPTY) { if (p == bishop || p == queen) return true; break; }
            cr += ddr[d]; cf += ddf[d];
        }
    }

    // rook / queen orthogonals
    int rook = (by_side == WHITE_SIDE) ? WROOK : BROOK;
    int sdr[] = {-1,1,0,0}, sdf[] = {0,0,-1,1};
    for (int d = 0; d < 4; d++) {
        int cr = r + sdr[d], cf = f + sdf[d];
        while (on_board(cr, cf)) {
            int p = s->board[make_sq(cr, cf)];
            if (p != EMPTY) { if (p == rook || p == queen) return true; break; }
            cr += sdr[d]; cf += sdf[d];
        }
    }
    return false;
}

} // namespace engine
