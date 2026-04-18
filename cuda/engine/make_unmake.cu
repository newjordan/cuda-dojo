// =============================================================================
// make_unmake.cu
//
// Implementation of make_move (extracted verbatim from gpu_fighter.cu's
// d_make_move / h_make_move — those were already byte-identical) plus a
// new in-place unmake_move backed by an Undo record.
// =============================================================================
#include "make_unmake.cuh"

namespace engine {

// ---------- core mutation, shared by both make_move overloads ----------
__host__ __device__ static inline void apply_move_inplace(Position* s, Move move) {
    int from  = move_from(move);
    int to    = move_to(move);
    int flags = move_flags(move);
    int promo = move_promo(move);
    int piece = s->board[from];
    int side  = s->side;

    s->board[to]   = piece;
    s->board[from] = EMPTY;
    s->halfmove++;

    if (piece_type(piece) == 6) s->kingPos[side] = to;

    if (piece_type(piece) == 1) {
        s->halfmove = 0;
        if (flags == FLAG_DOUBLE) {
            s->ep = (from + to) / 2;
        } else if (flags == FLAG_EP) {
            int cs = (side == WHITE_SIDE) ? to + 8 : to - 8;
            s->board[cs] = EMPTY;
            s->ep = -1;
        } else if (flags == FLAG_PROMO) {
            s->board[to] = promo;
            s->ep = -1;
        } else {
            s->ep = -1;
        }
    } else {
        s->ep = -1;
    }

    if (flags == FLAG_CAPTURE) s->halfmove = 0;

    if (flags == FLAG_CASTLE_K) {
        if (side == WHITE_SIDE) { s->board[63] = EMPTY; s->board[61] = WROOK; }
        else                    { s->board[7]  = EMPTY; s->board[5]  = BROOK; }
    }
    if (flags == FLAG_CASTLE_Q) {
        if (side == WHITE_SIDE) { s->board[56] = EMPTY; s->board[59] = WROOK; }
        else                    { s->board[0]  = EMPTY; s->board[3]  = BROOK; }
    }

    if (piece == WKING) s->castle &= ~(CASTLE_WK | CASTLE_WQ);
    if (piece == BKING) s->castle &= ~(CASTLE_BK | CASTLE_BQ);
    if (from == 63 || to == 63) s->castle &= ~CASTLE_WK;
    if (from == 56 || to == 56) s->castle &= ~CASTLE_WQ;
    if (from == 7  || to == 7 ) s->castle &= ~CASTLE_BK;
    if (from == 0  || to == 0 ) s->castle &= ~CASTLE_BQ;

    s->side = 1 - side;
    if (side == BLACK_SIDE) s->fullmove++;
}

// ---------- public make_move (no undo) ----------
__host__ __device__ void make_move(Position* s, Move move) {
    apply_move_inplace(s, move);
}

// ---------- public make_move (with undo recording) ----------
__host__ __device__ void make_move(Position* s, Move move, Undo* u) {
    int to    = move_to(move);
    int flags = move_flags(move);
    int side  = s->side;

    // Snapshot pre-move state into the Undo record.
    u->ep_before        = s->ep;
    u->castle_before    = s->castle;
    u->halfmove_before  = s->halfmove;
    u->fullmove_before  = s->fullmove;
    u->king_before[0]   = s->kingPos[0];
    u->king_before[1]   = s->kingPos[1];

    // Captured piece bookkeeping (covers ordinary captures and en passant).
    if (flags == FLAG_EP) {
        int cs = (side == WHITE_SIDE) ? to + 8 : to - 8;
        u->captured_piece = EMPTY;       // square `to` itself was empty
        u->ep_captured_sq = (int8_t)cs;  // pawn that gets removed
    } else {
        u->captured_piece = s->board[to]; // EMPTY for quiet, piece for capture
        u->ep_captured_sq = -1;
    }

    apply_move_inplace(s, move);
}

// ---------- public unmake_move ----------
__host__ __device__ void unmake_move(Position* s, Move move, const Undo* u) {
    int from  = move_from(move);
    int to    = move_to(move);
    int flags = move_flags(move);

    // After make: it's opponent to move. Switch back so we know whose pieces
    // moved (and to restore fullmove if we incremented it).
    int mover_side = 1 - s->side;
    s->side = (int8_t)mover_side;

    // Recover moved piece. For a promotion, the square contains a promo piece;
    // the original was a pawn of the moving side.
    int piece_at_to = s->board[to];
    int moved_piece;
    if (flags == FLAG_PROMO) {
        moved_piece = (mover_side == WHITE_SIDE) ? WPAWN : BPAWN;
    } else {
        moved_piece = piece_at_to;
    }

    // Move the piece back to `from`.
    s->board[from] = (int8_t)moved_piece;
    s->board[to]   = EMPTY;

    // Restore captured piece, including ep.
    if (flags == FLAG_EP) {
        // Square `to` stays empty; the captured pawn lives at ep_captured_sq.
        int opp_pawn = (mover_side == WHITE_SIDE) ? BPAWN : WPAWN;
        s->board[u->ep_captured_sq] = (int8_t)opp_pawn;
    } else if (u->captured_piece != EMPTY) {
        s->board[to] = u->captured_piece;
    }

    // Undo castle rook hops.
    if (flags == FLAG_CASTLE_K) {
        if (mover_side == WHITE_SIDE) { s->board[61] = EMPTY; s->board[63] = WROOK; }
        else                          { s->board[5]  = EMPTY; s->board[7]  = BROOK; }
    }
    if (flags == FLAG_CASTLE_Q) {
        if (mover_side == WHITE_SIDE) { s->board[59] = EMPTY; s->board[56] = WROOK; }
        else                          { s->board[3]  = EMPTY; s->board[0]  = BROOK; }
    }

    // Restore meta state from the snapshot.
    s->ep         = u->ep_before;
    s->castle     = u->castle_before;
    s->halfmove   = u->halfmove_before;
    s->fullmove   = u->fullmove_before;
    s->kingPos[0] = u->king_before[0];
    s->kingPos[1] = u->king_before[1];
}

} // namespace engine
