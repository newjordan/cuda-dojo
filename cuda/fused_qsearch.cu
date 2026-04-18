// =============================================================================
// CUDA DOJO — fused_qsearch.cu
// Depth-N quiescence search over N positions in parallel on GPU.
//
// Approach: ONE THREAD PER POSITION. Each thread runs a sequential alpha-beta
// quiescence (captures + promotions only, stand-pat, delta pruning) on its
// assigned CompactBoard. Move/undo stacks live in per-thread local memory
// (cap MAX_QDEPTH). Leaf eval = material + PST (stub matching batch_eval.cu).
//
// Rationale: alpha-beta is branchy and warp-divergent; warp-per-position
// coordination is a heavier lift and was explicitly listed as optional.
// Sequential-per-thread is correct, simple, and meets the 10K@d4 < 5s target
// on GB10.
//
// Reuses the movegen / make_move logic shape from chess_mcts.cu but adapted
// to CompactBoard (no castle rights tracked on input) and to a qsearch stack
// that stores unmake info explicitly (chess_mcts uses copy-before-move).
//
// CLI: FENs on stdin -> JSON array of {fen,qscore,nodes} on stdout.
//   ./fused_qsearch --depth 4 --bench
// =============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <algorithm>
#include <limits.h>
#include <vector>
#include <cuda_runtime.h>

#include "include/dojo_types.h"

// -----------------------------------------------------------------------------
// Local aliases (short names for kernel code)
// -----------------------------------------------------------------------------
#define EMPTY   DOJO_EMPTY
#define WPAWN   DOJO_WPAWN
#define WKNIGHT DOJO_WKNIGHT
#define WBISHOP DOJO_WBISHOP
#define WROOK   DOJO_WROOK
#define WQUEEN  DOJO_WQUEEN
#define WKING   DOJO_WKING
#define BPAWN   DOJO_BPAWN
#define BKNIGHT DOJO_BKNIGHT
#define BBISHOP DOJO_BBISHOP
#define BROOK   DOJO_BROOK
#define BQUEEN  DOJO_BQUEEN
#define BKING   DOJO_BKING

#define WHITE_SIDE 0
#define BLACK_SIDE 1

#define IS_WHITE(p) ((p) >= WPAWN && (p) <= WKING)
#define IS_BLACK(p) ((p) >= BPAWN && (p) <= BKING)
#define PIECE_COLOR(p) (IS_WHITE(p) ? WHITE_SIDE : BLACK_SIDE)
#define PIECE_TYPE(p) (IS_WHITE(p) ? (p) : (p) - 6)

#define MAX_MOVES    40     // enough for captures/promotions only
#define MAX_QDEPTH   12     // hard cap on q-search ply
#define Q_INF        1000000
#define DELTA_STAND  1100   // big-stand-pat delta (matches JS quiescence)
#define DELTA_CAP    200    // per-capture delta margin

// Move encoding (same shape as chess_mcts.cu for familiarity)
#define MAKE_MOVE(from, to, promo, flags) \
    (((from) & 0x3F) | (((to) & 0x3F) << 6) | (((promo) & 0xF) << 12) | (((flags) & 0xF) << 16))
#define MOVE_FROM(m)  ((m) & 0x3F)
#define MOVE_TO(m)    (((m) >> 6) & 0x3F)
#define MOVE_PROMO(m) (((m) >> 12) & 0xF)
#define MOVE_FLAGS(m) (((m) >> 16) & 0xF)

#define FLAG_NONE    0
#define FLAG_CAPTURE 1
#define FLAG_EP      2
#define FLAG_PROMO   6

// -----------------------------------------------------------------------------
// Piece values (centipawns) — match batch_eval.cu for leaf consistency.
// -----------------------------------------------------------------------------
__constant__ int d_PVAL[13] = {
    0, 100, 320, 330, 500, 900, 20000,
       100, 320, 330, 500, 900, 20000
};

// MVV-LVA: [victim_type][attacker_type] -> score.  index by piece_type (1..6)
__constant__ int d_MVV_LVA[7][7] = {
    {0,0,0,0,0,0,0},           // 0 empty (unused)
    {0, 105,104,103,102,101,100}, // victim=PAWN
    {0, 205,204,203,202,201,200}, // victim=KNIGHT
    {0, 305,304,303,302,301,300}, // victim=BISHOP
    {0, 405,404,403,402,401,400}, // victim=ROOK
    {0, 505,504,503,502,501,500}, // victim=QUEEN
    {0, 605,604,603,602,601,600}  // victim=KING  (should never be captured but included)
};

// Compact middlegame PSTs (stubby — same spirit as batch_eval.cu but inlined small)
__constant__ int d_PST_PAWN[64] = {
     0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
     5,  5, 10, 25, 25, 10,  5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5, -5,-10,  0,  0,-10, -5,  5,
     5, 10, 10,-20,-20, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0
};
__constant__ int d_PST_KNIGHT[64] = {
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50
};
__constant__ int d_PST_BISHOP[64] = {
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10, 10,  5, 10, 10,  5, 10,-10,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -20,-10,-10,-10,-10,-10,-10,-20
};
__constant__ int d_PST_ROOK[64] = {
     0,  0,  0,  0,  0,  0,  0,  0,
     5, 10, 10, 10, 10, 10, 10,  5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     0,  0,  0,  5,  5,  0,  0,  0
};
__constant__ int d_PST_QUEEN[64] = {
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5,  5,  5,  5,  0,-10,
     -5,  0,  5,  5,  5,  5,  0, -5,
      0,  0,  5,  5,  5,  5,  0, -5,
    -10,  5,  5,  5,  5,  5,  0,-10,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20
};
__constant__ int d_PST_KING_MG[64] = {
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
     20, 20,  0,  0,  0,  0, 20, 20,
     20, 30, 10,  0,  0, 10, 30, 20
};

__constant__ int d_KNIGHT_OFFSETS[8] = {-17, -15, -10, -6, 6, 10, 15, 17};
__constant__ int d_KING_OFFSETS[8]   = {-9, -8, -7, -1, 1, 7, 8, 9};

// -----------------------------------------------------------------------------
// Device helpers
// -----------------------------------------------------------------------------
__host__ __device__ static inline int sq_rank(int sq) { return sq >> 3; }
__host__ __device__ static inline int sq_file(int sq) { return sq & 7; }
__host__ __device__ static inline int make_sq(int r, int f) { return (r << 3) | f; }
__host__ __device__ static inline int mirror_sq(int sq) {
    return ((7 - sq_rank(sq)) << 3) | sq_file(sq);
}
__host__ __device__ static inline bool on_board(int r, int f) {
    return r >= 0 && r < 8 && f >= 0 && f < 8;
}

// -----------------------------------------------------------------------------
// Per-thread working board. We don't need castling/ep history for qsearch
// (captures only, and ep captures are handled as single-use re-checks from
// the input CompactBoard which lacks ep state, so we conservatively skip EP).
// -----------------------------------------------------------------------------
struct QBoard {
    int8_t board[64];
    int8_t side;
    int8_t kingSq[2];   // [white, black]
};

__device__ static void qboard_from_compact(QBoard* qb, const CompactBoard* cb) {
    #pragma unroll
    for (int i = 0; i < 64; i++) qb->board[i] = cb->board[i];
    qb->side = cb->side;
    qb->kingSq[0] = -1;
    qb->kingSq[1] = -1;
    for (int i = 0; i < 64; i++) {
        int p = qb->board[i];
        if (p == WKING) qb->kingSq[0] = i;
        else if (p == BKING) qb->kingSq[1] = i;
    }
}

// -----------------------------------------------------------------------------
// Attack detection (by_side attacks sq?)
// -----------------------------------------------------------------------------
__device__ static bool is_square_attacked(const QBoard* s, int sq, int by_side) {
    int r = sq_rank(sq);
    int f = sq_file(sq);

    // Pawn attacks
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

    // Knight
    int knightPiece = (by_side == WHITE_SIDE) ? WKNIGHT : BKNIGHT;
    for (int i = 0; i < 8; i++) {
        int t = sq + d_KNIGHT_OFFSETS[i];
        if (t < 0 || t >= 64) continue;
        int dr = abs(sq_rank(t) - r);
        int df = abs(sq_file(t) - f);
        if (((dr == 2 && df == 1) || (dr == 1 && df == 2)) && s->board[t] == knightPiece)
            return true;
    }

    // King
    int kingPiece = (by_side == WHITE_SIDE) ? WKING : BKING;
    for (int i = 0; i < 8; i++) {
        int t = sq + d_KING_OFFSETS[i];
        if (t < 0 || t >= 64) continue;
        if (abs(sq_rank(t) - r) <= 1 && abs(sq_file(t) - f) <= 1 && s->board[t] == kingPiece)
            return true;
    }

    // Sliding — diagonals
    int bishopPiece = (by_side == WHITE_SIDE) ? WBISHOP : BBISHOP;
    int queenPiece  = (by_side == WHITE_SIDE) ? WQUEEN  : BQUEEN;
    const int ddr[4] = {-1,-1,1,1};
    const int ddf[4] = {-1,1,-1,1};
    for (int d = 0; d < 4; d++) {
        int cr = r + ddr[d], cf = f + ddf[d];
        while (on_board(cr, cf)) {
            int p = s->board[make_sq(cr, cf)];
            if (p != EMPTY) {
                if (p == bishopPiece || p == queenPiece) return true;
                break;
            }
            cr += ddr[d]; cf += ddf[d];
        }
    }

    // Sliding — ranks/files
    int rookPiece = (by_side == WHITE_SIDE) ? WROOK : BROOK;
    const int sdr[4] = {-1,1,0,0};
    const int sdf[4] = {0,0,-1,1};
    for (int d = 0; d < 4; d++) {
        int cr = r + sdr[d], cf = f + sdf[d];
        while (on_board(cr, cf)) {
            int p = s->board[make_sq(cr, cf)];
            if (p != EMPTY) {
                if (p == rookPiece || p == queenPiece) return true;
                break;
            }
            cr += sdr[d]; cf += sdf[d];
        }
    }

    return false;
}

__device__ static bool is_in_check(const QBoard* s, int side) {
    int ksq = s->kingSq[side];
    if (ksq < 0) return false;
    return is_square_attacked(s, ksq, 1 - side);
}

// -----------------------------------------------------------------------------
// Generate CAPTURES ONLY (plus queen promotions). This is the qsearch move set.
// -----------------------------------------------------------------------------
__device__ static int generate_captures(const QBoard* s, int* moves) {
    int count = 0;
    int side = s->side;
    int opp = 1 - side;

    for (int sq = 0; sq < 64; sq++) {
        int piece = s->board[sq];
        if (piece == EMPTY) continue;
        if (side == WHITE_SIDE && !IS_WHITE(piece)) continue;
        if (side == BLACK_SIDE && !IS_BLACK(piece)) continue;

        int r = sq_rank(sq);
        int f = sq_file(sq);
        int ptype = PIECE_TYPE(piece);

        if (ptype == 1) { // PAWN
            int dir = (side == WHITE_SIDE) ? -1 : 1;
            int promoRank = (side == WHITE_SIDE) ? 0 : 7;
            int nr = r + dir;

            // Quiet promotions (push to last rank) — still material-changing, include.
            if (on_board(nr, f) && s->board[make_sq(nr, f)] == EMPTY && nr == promoRank) {
                int pb = (side == WHITE_SIDE) ? WQUEEN : BQUEEN;
                moves[count++] = MAKE_MOVE(sq, make_sq(nr, f), pb, FLAG_PROMO);
            }
            // Diagonal captures
            for (int df = -1; df <= 1; df += 2) {
                int nf = f + df;
                if (!on_board(nr, nf)) continue;
                int t = make_sq(nr, nf);
                int tp = s->board[t];
                if (tp != EMPTY && PIECE_COLOR(tp) == opp) {
                    if (nr == promoRank) {
                        int pb = (side == WHITE_SIDE) ? WQUEEN : BQUEEN;
                        moves[count++] = MAKE_MOVE(sq, t, pb, FLAG_PROMO);
                    } else {
                        moves[count++] = MAKE_MOVE(sq, t, 0, FLAG_CAPTURE);
                    }
                }
                // EP: skipped (CompactBoard doesn't carry ep; acceptable noise for qsearch)
            }
        }
        else if (ptype == 2) { // KNIGHT
            for (int i = 0; i < 8; i++) {
                int t = sq + d_KNIGHT_OFFSETS[i];
                if (t < 0 || t >= 64) continue;
                int dr = abs(sq_rank(t) - r);
                int df2 = abs(sq_file(t) - f);
                if (!((dr == 2 && df2 == 1) || (dr == 1 && df2 == 2))) continue;
                int tp = s->board[t];
                if (tp != EMPTY && PIECE_COLOR(tp) == opp) {
                    moves[count++] = MAKE_MOVE(sq, t, 0, FLAG_CAPTURE);
                }
            }
        }
        else if (ptype == 3 || ptype == 5) { // BISHOP or QUEEN diagonals
            const int dr[4] = {-1,-1,1,1}, df[4] = {-1,1,-1,1};
            for (int d = 0; d < 4; d++) {
                int cr = r + dr[d], cf = f + df[d];
                while (on_board(cr, cf)) {
                    int t = make_sq(cr, cf);
                    int tp = s->board[t];
                    if (tp == EMPTY) { cr += dr[d]; cf += df[d]; continue; }
                    if (PIECE_COLOR(tp) == opp)
                        moves[count++] = MAKE_MOVE(sq, t, 0, FLAG_CAPTURE);
                    break;
                }
            }
            if (ptype == 3) continue;
        }
        if (ptype == 4 || ptype == 5) { // ROOK or QUEEN ranks/files
            const int dr[4] = {-1,1,0,0}, df[4] = {0,0,-1,1};
            for (int d = 0; d < 4; d++) {
                int cr = r + dr[d], cf = f + df[d];
                while (on_board(cr, cf)) {
                    int t = make_sq(cr, cf);
                    int tp = s->board[t];
                    if (tp == EMPTY) { cr += dr[d]; cf += df[d]; continue; }
                    if (PIECE_COLOR(tp) == opp)
                        moves[count++] = MAKE_MOVE(sq, t, 0, FLAG_CAPTURE);
                    break;
                }
            }
        }
        else if (ptype == 6) { // KING captures only
            for (int i = 0; i < 8; i++) {
                int t = sq + d_KING_OFFSETS[i];
                if (t < 0 || t >= 64) continue;
                if (abs(sq_rank(t) - r) > 1 || abs(sq_file(t) - f) > 1) continue;
                int tp = s->board[t];
                if (tp != EMPTY && PIECE_COLOR(tp) == opp) {
                    moves[count++] = MAKE_MOVE(sq, t, 0, FLAG_CAPTURE);
                }
            }
        }

        if (count >= MAX_MOVES - 4) break;
    }
    return count;
}

// -----------------------------------------------------------------------------
// make / unmake with explicit undo record (keeps qsearch copy-free).
// -----------------------------------------------------------------------------
struct QUndo {
    int move;
    int8_t captured;     // piece on 'to' before move (for captures)
    int8_t from_piece;   // piece on 'from' before (pre-promotion)
    int8_t prev_side;
    int8_t prev_wking;
    int8_t prev_bking;
};

struct QTrace {
    uint16_t max_ply_reached;
    uint16_t depth_cap_hits;
    uint16_t max_capture_moves_in_node;
    uint16_t reserved0;
    uint32_t standpat_beta_cutoffs;
    uint32_t standpat_delta_prunes;
    uint32_t move_delta_prunes;
    uint32_t beta_cutoffs_after_search;
    uint32_t illegal_move_rejects;
    uint32_t total_capture_moves_generated;
};

__device__ __forceinline__ static void qmake(QBoard* s, int move, QUndo* u) {
    int from = MOVE_FROM(move);
    int to   = MOVE_TO(move);
    int flags = MOVE_FLAGS(move);
    int promo = MOVE_PROMO(move);
    int piece = s->board[from];

    u->move = move;
    u->captured = s->board[to];
    u->from_piece = piece;
    u->prev_side = s->side;
    u->prev_wking = s->kingSq[0];
    u->prev_bking = s->kingSq[1];

    s->board[to] = (flags == FLAG_PROMO) ? (int8_t)promo : (int8_t)piece;
    s->board[from] = EMPTY;

    if (PIECE_TYPE(piece) == 6) s->kingSq[s->side] = (int8_t)to;
    s->side = 1 - s->side;
}

__device__ __forceinline__ static void qunmake(QBoard* s, const QUndo* u) {
    int from = MOVE_FROM(u->move);
    int to   = MOVE_TO(u->move);

    s->board[from] = u->from_piece;
    s->board[to]   = u->captured;
    s->side        = u->prev_side;
    s->kingSq[0]   = u->prev_wking;
    s->kingSq[1]   = u->prev_bking;
}

// -----------------------------------------------------------------------------
// Leaf eval: material + PST, returned from side-to-move POV.
// -----------------------------------------------------------------------------
__device__ __noinline__ static int evaluate_leaf(const QBoard* s) {
    int score = 0;
    for (int sq = 0; sq < 64; sq++) {
        int p = s->board[sq];
        if (p == EMPTY) continue;
        int val = d_PVAL[p];
        int ptype = PIECE_TYPE(p);
        int pstSq = IS_WHITE(p) ? sq : mirror_sq(sq);
        int pst = 0;
        switch (ptype) {
            case 1: pst = d_PST_PAWN[pstSq]; break;
            case 2: pst = d_PST_KNIGHT[pstSq]; break;
            case 3: pst = d_PST_BISHOP[pstSq]; break;
            case 4: pst = d_PST_ROOK[pstSq]; break;
            case 5: pst = d_PST_QUEEN[pstSq]; break;
            case 6: pst = d_PST_KING_MG[pstSq]; break;
        }
        if (IS_WHITE(p)) score += val + pst;
        else             score -= val + pst;
    }
    return (s->side == WHITE_SIDE) ? score : -score;
}

// -----------------------------------------------------------------------------
// Quiescence: iterative stackless isn't worth it here; use recursion depth cap.
// Alpha-beta with stand-pat + delta pruning + MVV-LVA ordering.
// Returns score from side-to-move POV.
// -----------------------------------------------------------------------------
__device__ static int qsearch(QBoard* s, int alpha, int beta, int ply, int max_depth,
                              unsigned int* nodes_out, QTrace* trace) {
    (*nodes_out)++;
    if (trace != nullptr && ply > trace->max_ply_reached) {
        trace->max_ply_reached = static_cast<uint16_t>(ply);
    }

    int standPat = evaluate_leaf(s);
    if (standPat >= beta) {
        if (trace != nullptr) trace->standpat_beta_cutoffs++;
        return beta;
    }
    if (standPat + DELTA_STAND < alpha) {
        if (trace != nullptr) trace->standpat_delta_prunes++;
        return alpha;
    }
    if (standPat > alpha) alpha = standPat;

    if (ply >= max_depth) {
        if (trace != nullptr) trace->depth_cap_hits++;
        return alpha;
    }

    int moves[MAX_MOVES];
    int16_t scores[MAX_MOVES];
    int n = generate_captures(s, moves);
    if (trace != nullptr) {
        trace->total_capture_moves_generated += static_cast<uint32_t>(n);
        if (n > static_cast<int>(trace->max_capture_moves_in_node)) {
            trace->max_capture_moves_in_node = static_cast<uint16_t>(n);
        }
    }
    // MVV-LVA ordering
    for (int i = 0; i < n; i++) {
        int m = moves[i];
        int to = MOVE_TO(m);
        int victim = s->board[to];
        int flags = MOVE_FLAGS(m);
        int attacker = PIECE_TYPE(s->board[MOVE_FROM(m)]);
        if (flags == FLAG_PROMO && victim == EMPTY) {
            scores[i] = 800;
        } else if (victim == EMPTY) {
            scores[i] = 0;
        } else {
            int vtype = PIECE_TYPE(victim);
            scores[i] = d_MVV_LVA[vtype][attacker];
            if (flags == FLAG_PROMO) scores[i] += 800;
        }
    }
    for (int i = 1; i < n; i++) {
        int km = moves[i];
        int16_t ks = scores[i];
        int j = i - 1;
        while (j >= 0 && scores[j] < ks) {
            moves[j + 1] = moves[j];
            scores[j + 1] = scores[j];
            j--;
        }
        moves[j + 1] = km;
        scores[j + 1] = ks;
    }

    for (int i = 0; i < n; i++) {
        int m = moves[i];
        int victim = s->board[MOVE_TO(m)];
        // King captures aren't legal in real chess — skip to avoid 20000cp
        // blowups from search paths that don't properly refute upstream moves.
        if (victim == WKING || victim == BKING) continue;
        int victimVal = (victim == EMPTY) ? 0 : d_PVAL[victim];
        // Per-move delta pruning
        if (standPat + victimVal + DELTA_CAP < alpha) {
            if (trace != nullptr) trace->move_delta_prunes++;
            continue;
        }

        QUndo u;
        qmake(s, m, &u);
        // Legality — side to move flipped; check if *mover* (prev side) left own king in check
        if (is_in_check(s, u.prev_side)) {
            if (trace != nullptr) trace->illegal_move_rejects++;
            qunmake(s, &u);
            continue;
        }
        int score = -qsearch(s, -beta, -alpha, ply + 1, max_depth, nodes_out, trace);
        qunmake(s, &u);

        if (score >= beta) {
            if (trace != nullptr) trace->beta_cutoffs_after_search++;
            return beta;
        }
        if (score > alpha) alpha = score;
    }
    return alpha;
}

// -----------------------------------------------------------------------------
// Kernel: one thread per position.
// -----------------------------------------------------------------------------
__global__ void fused_qsearch_kernel(const CompactBoard* boards,
                                     int* scores,
                                     unsigned int* nodes_counts,
                                     QTrace* traces,
                                     int n,
                                     int max_depth) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    QBoard qb;
    qboard_from_compact(&qb, &boards[tid]);

    unsigned int nodes = 0;
    QTrace trace{};
    int s = qsearch(&qb, -Q_INF, Q_INF, 0, max_depth, &nodes, &trace);
    scores[tid] = s;
    if (nodes_counts) nodes_counts[tid] = nodes;
    if (traces) traces[tid] = trace;
}

// =============================================================================
// HOST: FEN parsing into CompactBoard
// =============================================================================
static void parse_fen_compact(const char* fen, CompactBoard* cb) {
    for (int i = 0; i < 64; i++) cb->board[i] = EMPTY;
    cb->side = WHITE_SIDE;
    cb->padding[0] = cb->padding[1] = cb->padding[2] = 0;

    int sq = 0;
    int i = 0;
    while (fen[i] && fen[i] != ' ') {
        char c = fen[i++];
        if (c == '/') continue;
        if (c >= '1' && c <= '8') {
            sq += (c - '0');
        } else {
            int piece = EMPTY;
            switch (c) {
                case 'P': piece = WPAWN; break;
                case 'N': piece = WKNIGHT; break;
                case 'B': piece = WBISHOP; break;
                case 'R': piece = WROOK; break;
                case 'Q': piece = WQUEEN; break;
                case 'K': piece = WKING; break;
                case 'p': piece = BPAWN; break;
                case 'n': piece = BKNIGHT; break;
                case 'b': piece = BBISHOP; break;
                case 'r': piece = BROOK; break;
                case 'q': piece = BQUEEN; break;
                case 'k': piece = BKING; break;
            }
            if (sq < 64) cb->board[sq++] = piece;
        }
    }
    while (fen[i] == ' ') i++;
    cb->side = (fen[i] == 'b') ? BLACK_SIDE : WHITE_SIDE;
}

// =============================================================================
// HOST: main — reads FENs on stdin, emits JSON. Supports --bench for
// synthetic throughput test (replicates input up to N).
// =============================================================================
int main(int argc, char** argv) {
    int max_depth = 4;
    int bench_n = 0;      // if >0, replicate first FEN up to bench_n positions
    int json_out = 1;
    int verbose = 0;
    int block_size = 64;
    int stack_kb = 32;
    int metrics_json = 0;
    int warmup_runs = 1;

    for (int a = 1; a < argc; a++) {
        if (!strcmp(argv[a], "--depth") && a+1 < argc) max_depth = atoi(argv[++a]);
        else if (!strcmp(argv[a], "--bench") && a+1 < argc) bench_n = atoi(argv[++a]);
        else if (!strcmp(argv[a], "--plain")) json_out = 0;
        else if (!strcmp(argv[a], "--verbose")) verbose = 1;
        else if (!strcmp(argv[a], "--block-size") && a+1 < argc) block_size = atoi(argv[++a]);
        else if (!strcmp(argv[a], "--stack-kb") && a+1 < argc) stack_kb = atoi(argv[++a]);
        else if (!strcmp(argv[a], "--metrics-json")) metrics_json = 1;
        else if (!strcmp(argv[a], "--warmup-runs") && a+1 < argc) warmup_runs = atoi(argv[++a]);
    }
    if (max_depth < 1) max_depth = 1;
    if (max_depth > MAX_QDEPTH) max_depth = MAX_QDEPTH;
    if (block_size < 32) block_size = 32;
    if (block_size > 256) block_size = 256;
    if (stack_kb < 8) stack_kb = 8;
    if (stack_kb > 128) stack_kb = 128;
    if (warmup_runs < 0) warmup_runs = 0;

    // Slurp stdin FENs
    char line[512];
    CompactBoard* h_boards = (CompactBoard*)malloc(sizeof(CompactBoard) * DOJO_MAX_POSITIONS);
    char (*h_fens)[512] = (char (*)[512])malloc(512 * DOJO_MAX_POSITIONS);
    int n_in = 0;

    while (fgets(line, sizeof(line), stdin)) {
        int len = (int)strlen(line);
        while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r')) line[--len] = '\0';
        if (len < 10) continue;
        if (n_in >= DOJO_MAX_POSITIONS) break;
        parse_fen_compact(line, &h_boards[n_in]);
        strncpy(h_fens[n_in], line, 511);
        h_fens[n_in][511] = '\0';
        n_in++;
    }

    if (n_in == 0) {
        fprintf(stderr, "fused_qsearch: no FENs on stdin\n");
        free(h_boards); free(h_fens);
        return 1;
    }

    int n = n_in;
    if (bench_n > 0) {
        if (bench_n > DOJO_MAX_POSITIONS) bench_n = DOJO_MAX_POSITIONS;
        for (int i = n_in; i < bench_n; i++) {
            h_boards[i] = h_boards[i % n_in];
        }
        n = bench_n;
    }

    // Device alloc
    CompactBoard* d_boards;
    int* d_scores;
    unsigned int* d_nodes;
    QTrace* d_traces;
    cudaMalloc(&d_boards, sizeof(CompactBoard) * n);
    cudaMalloc(&d_scores, sizeof(int) * n);
    cudaMalloc(&d_nodes,  sizeof(unsigned int) * n);
    cudaMalloc(&d_traces, sizeof(QTrace) * n);
    cudaMemcpy(d_boards, h_boards, sizeof(CompactBoard) * n, cudaMemcpyHostToDevice);
    cudaMemset(d_nodes, 0, sizeof(unsigned int) * n);
    cudaMemset(d_traces, 0, sizeof(QTrace) * n);

    // Recursive qsearch needs a bigger per-thread stack than the default (1 KB).
    // Each frame: moves[MAX_MOVES]+scores[MAX_MOVES] = ~320B + board/locals; cap
    // at MAX_QDEPTH. Allocate generously.
    size_t stack_bytes = (size_t)stack_kb * 1024;
    cudaDeviceSetLimit(cudaLimitStackSize, stack_bytes);
    size_t effective_stack_bytes = 0;
    cudaDeviceGetLimit(&effective_stack_bytes, cudaLimitStackSize);

    int numBlocks = (n + block_size - 1) / block_size;
    for (int i = 0; i < warmup_runs; i++) {
        fused_qsearch_kernel<<<numBlocks, block_size>>>(d_boards, d_scores, d_nodes, d_traces, n, max_depth);
        cudaDeviceSynchronize();
    }

    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0); cudaEventCreate(&ev1);
    cudaEventRecord(ev0);
    fused_qsearch_kernel<<<numBlocks, block_size>>>(d_boards, d_scores, d_nodes, d_traces, n, max_depth);
    cudaEventRecord(ev1);
    cudaEventSynchronize(ev1);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, ev0, ev1);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        return 2;
    }

    int* h_scores = (int*)malloc(sizeof(int) * n);
    unsigned int* h_nodes = (unsigned int*)malloc(sizeof(unsigned int) * n);
    QTrace* h_traces = (QTrace*)malloc(sizeof(QTrace) * n);
    cudaMemcpy(h_scores, d_scores, sizeof(int) * n, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_nodes,  d_nodes,  sizeof(unsigned int) * n, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_traces, d_traces, sizeof(QTrace) * n, cudaMemcpyDeviceToHost);

    // Emit only the positions we actually had FENs for (n_in) unless bench
    int emit = (bench_n > 0) ? n_in : n_in;

    if (json_out) {
        printf("[");
        for (int i = 0; i < emit; i++) {
            printf("{\"fen\":\"%s\",\"qscore\":%d,\"nodes\":%u}",
                   h_fens[i], h_scores[i], h_nodes[i]);
            if (i < emit - 1) printf(",");
        }
        printf("]\n");
    } else {
        for (int i = 0; i < emit; i++) {
            printf("%d\n", h_scores[i]);
        }
    }

    if (verbose || bench_n > 0 || metrics_json) {
        unsigned long long total_nodes = 0;
        for (int i = 0; i < n; i++) total_nodes += h_nodes[i];
        unsigned int min_nodes = UINT_MAX;
        unsigned int max_nodes = 0;
        std::vector<unsigned int> sorted_nodes(h_nodes, h_nodes + n);
        std::sort(sorted_nodes.begin(), sorted_nodes.end());
        for (int i = 0; i < n; i++) {
            if (h_nodes[i] < min_nodes) min_nodes = h_nodes[i];
            if (h_nodes[i] > max_nodes) max_nodes = h_nodes[i];
        }
        double mean_nodes = n > 0 ? (double)total_nodes / (double)n : 0.0;
        unsigned int p95_nodes =
            sorted_nodes.empty() ? 0u : sorted_nodes[(sorted_nodes.size() * 95) / 100];
        unsigned long long total_depth_cap_hits = 0;
        unsigned long long total_standpat_beta_cutoffs = 0;
        unsigned long long total_standpat_delta_prunes = 0;
        unsigned long long total_move_delta_prunes = 0;
        unsigned long long total_beta_cutoffs_after_search = 0;
        unsigned long long total_illegal_move_rejects = 0;
        unsigned long long total_capture_moves_generated = 0;
        unsigned int max_ply_reached = 0;
        unsigned int max_capture_moves_in_node = 0;
        double sum_max_ply_per_position = 0.0;
        for (int i = 0; i < n; i++) {
            total_depth_cap_hits += h_traces[i].depth_cap_hits;
            total_standpat_beta_cutoffs += h_traces[i].standpat_beta_cutoffs;
            total_standpat_delta_prunes += h_traces[i].standpat_delta_prunes;
            total_move_delta_prunes += h_traces[i].move_delta_prunes;
            total_beta_cutoffs_after_search += h_traces[i].beta_cutoffs_after_search;
            total_illegal_move_rejects += h_traces[i].illegal_move_rejects;
            total_capture_moves_generated += h_traces[i].total_capture_moves_generated;
            sum_max_ply_per_position += (double)h_traces[i].max_ply_reached;
            if (h_traces[i].max_ply_reached > max_ply_reached) {
                max_ply_reached = h_traces[i].max_ply_reached;
            }
            if (h_traces[i].max_capture_moves_in_node > max_capture_moves_in_node) {
                max_capture_moves_in_node = h_traces[i].max_capture_moves_in_node;
            }
        }
        cudaFuncAttributes attr{};
        cudaFuncGetAttributes(&attr, fused_qsearch_kernel);
        int device = 0;
        cudaGetDevice(&device);
        cudaDeviceProp prop{};
        cudaGetDeviceProperties(&prop, device);
        int active_blocks = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &active_blocks,
            fused_qsearch_kernel,
            block_size,
            0);
        double occupancy_estimate = 0.0;
        if (prop.maxThreadsPerMultiProcessor > 0) {
            occupancy_estimate =
                (double)(active_blocks * block_size) / (double)prop.maxThreadsPerMultiProcessor;
        }
        fprintf(stderr,
                "fused_qsearch: depth=%d positions=%d block=%d stack_kb=%d kernel=%.2fms "
                "throughput=%.0f pos/sec total_nodes=%llu\n",
                max_depth, n, block_size, stack_kb, ms,
                (n * 1000.0) / (ms > 0 ? ms : 1.0), total_nodes);
        if (metrics_json) {
            fprintf(stderr,
                    "{\"depth\":%d,\"positions\":%d,\"block_size\":%d,\"stack_kb\":%d,"
                    "\"warmup_runs\":%d,\"stack_bytes_requested\":%zu,\"stack_bytes_effective\":%zu,"
                    "\"kernel_ms\":%.4f,\"positions_per_sec\":%.2f,\"nodes_per_sec\":%.2f,"
                    "\"total_nodes\":%llu,\"min_nodes_per_position\":%u,"
                    "\"mean_nodes_per_position\":%.4f,\"p95_nodes_per_position\":%u,"
                    "\"max_nodes_per_position\":%u,\"max_ply_reached\":%u,"
                    "\"mean_max_ply_per_position\":%.4f,\"depth_cap_hits\":%llu,"
                    "\"standpat_beta_cutoffs\":%llu,\"standpat_delta_prunes\":%llu,"
                    "\"move_delta_prunes\":%llu,\"beta_cutoffs_after_search\":%llu,"
                    "\"illegal_move_rejects\":%llu,\"total_capture_moves_generated\":%llu,"
                    "\"mean_capture_moves_per_node\":%.6f,\"max_capture_moves_in_node\":%u,"
                    "\"num_regs\":%d,\"local_size_bytes\":%zu,\"shared_size_bytes\":%zu,"
                    "\"max_threads_per_block\":%d,\"occupancy_estimate\":%.4f}\n",
                    max_depth, n, block_size, stack_kb,
                    warmup_runs, stack_bytes, effective_stack_bytes,
                    ms,
                    (n * 1000.0) / (ms > 0 ? ms : 1.0),
                    (total_nodes * 1000.0) / (ms > 0 ? ms : 1.0),
                    total_nodes, min_nodes == UINT_MAX ? 0u : min_nodes,
                    mean_nodes, p95_nodes, max_nodes, max_ply_reached,
                    n > 0 ? sum_max_ply_per_position / (double)n : 0.0,
                    total_depth_cap_hits, total_standpat_beta_cutoffs,
                    total_standpat_delta_prunes, total_move_delta_prunes,
                    total_beta_cutoffs_after_search, total_illegal_move_rejects,
                    total_capture_moves_generated,
                    total_nodes > 0 ? (double)total_capture_moves_generated / (double)total_nodes : 0.0,
                    max_capture_moves_in_node, attr.numRegs,
                    (size_t)attr.localSizeBytes, (size_t)attr.sharedSizeBytes,
                    attr.maxThreadsPerBlock, occupancy_estimate);
        }
    }

    free(h_scores); free(h_nodes); free(h_traces);
    cudaFree(d_boards); cudaFree(d_scores); cudaFree(d_nodes); cudaFree(d_traces);
    free(h_boards); free(h_fens);
    cudaEventDestroy(ev0); cudaEventDestroy(ev1);
    return 0;
}
