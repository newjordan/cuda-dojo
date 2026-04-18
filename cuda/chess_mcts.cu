#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

// =============================================================================
// CONSTANTS
// =============================================================================

#define MAX_MOVES 256
#define MAX_PLAYOUT_DEPTH 200
#define BOARD_SIZE 64
#define DEFAULT_SIMULATIONS 10000
#define BLOCK_SIZE 256
#define ROOT_CPUCT 1.35f
#define ROOT_BASE_PRIOR_VISITS 4
#define SHALLOW_TACTICAL_PLIES 6
#define TACTICAL_FORCE_THRESHOLD 800
#define TACTICAL_PREFER_THRESHOLD 150
#define TACTICAL_CHECK_BONUS 150
#define TACTICAL_MATE_BONUS 100000
#define TACTICAL_SELECTION_BIAS 0.00003f
#define TACTICAL_POSTERIOR_SLACK 0.12f
#define ROOT_ALARM_SELECTION_BIAS 0.00004f
#define OPENING_FULLMOVE_WINDOW 10
#define ROOT_HEAD_MAX 8
#define ROOT_POSTERIOR_GAP_SCALE 6.0f
#define ROOT_SCORE_GAP_SCALE 14.0f
#define ROOT_FRONTIER_MAX 16
#define ROOT_FRONTIER_MASS_MIN 0.55f
#define ROOT_FRONTIER_MASS_MAX 0.85f
#define ROOT_FRONTIER_TAU_MIN 0.03f
#define ROOT_FRONTIER_TAU_MAX 0.25f
#define ROOT_CPUCT_SCALE_MIN 0.70f
#define ROOT_CPUCT_SCALE_MAX 1.60f
#define ROOT_HALVING_FRONTIER_MIN 6
#define ROOT_HALVING_UNCERTAINTY_MIN 0.90f
#define ROOT_HALVING_POSTERIOR_GAP_MAX 0.05f
#define ROOT_HALVING_SCORE_GAP_MAX 0.020f
#define ROOT_COMPARISON_FRONTIER_MAX 3
#define ROOT_COMPARISON_UNCERTAINTY_MIN 0.55f
#define ROOT_COMPARISON_POSTERIOR_GAP_MAX 0.14f
#define ROOT_COMPARISON_SCORE_GAP_MAX 0.060f
#define ROOT_SCHEDULE_FRONTIER 0
#define ROOT_SCHEDULE_HALVING 1
#define ROOT_SCHEDULE_COMPARE 2
#define ALARM_BAND_LEFT 0
#define ALARM_BAND_CENTER 1
#define ALARM_BAND_RIGHT 2

struct RootAlarmFeatures {
    int friendly_alarm_delta;
    int enemy_alarm_delta;
    int band_alarm_delta;
    int sniper_lane_score;
    int reverse_trap_risk;
    int total;
};

// Piece definitions
#define EMPTY   0
#define WPAWN   1
#define WKNIGHT 2
#define WBISHOP 3
#define WROOK   4
#define WQUEEN  5
#define WKING   6
#define BPAWN   7
#define BKNIGHT 8
#define BBISHOP 9
#define BROOK   10
#define BQUEEN  11
#define BKING   12

#define WHITE_SIDE 0
#define BLACK_SIDE 1

#define IS_WHITE(p) ((p) >= WPAWN && (p) <= WKING)
#define IS_BLACK(p) ((p) >= BPAWN && (p) <= BKING)
#define PIECE_COLOR(p) (IS_WHITE(p) ? WHITE_SIDE : BLACK_SIDE)
#define PIECE_TYPE(p) (IS_WHITE(p) ? (p) : (p) - 6)

// Move encoding: from(6) | to(6) | promo(4) | flags(4) = 20 bits fits in int
#define MAKE_MOVE(from, to, promo, flags) (((from) & 0x3F) | (((to) & 0x3F) << 6) | (((promo) & 0xF) << 12) | (((flags) & 0xF) << 16))
#define MOVE_FROM(m) ((m) & 0x3F)
#define MOVE_TO(m) (((m) >> 6) & 0x3F)
#define MOVE_PROMO(m) (((m) >> 12) & 0xF)
#define MOVE_FLAGS(m) (((m) >> 16) & 0xF)

#define FLAG_NONE     0
#define FLAG_CAPTURE  1
#define FLAG_EP       2
#define FLAG_CASTLE_K 3
#define FLAG_CASTLE_Q 4
#define FLAG_DOUBLE   5
#define FLAG_PROMO    6

// Castle rights bitmask
#define CASTLE_WK 1
#define CASTLE_WQ 2
#define CASTLE_BK 4
#define CASTLE_BQ 8

// =============================================================================
// BOARD STATE (compact for GPU)
// =============================================================================

struct BoardState {
    int8_t board[64];     // piece on each square
    int8_t side;          // 0=white, 1=black
    int8_t castle;        // castle rights bitmask
    int8_t ep;            // en passant target square (-1 if none)
    int16_t halfmove;     // halfmove clock for 50-move rule
    int16_t fullmove;     // fullmove counter
    int8_t kingPos[2];    // king positions [white, black]
};

// =============================================================================
// PIECE-SQUARE TABLES (for material + positional evaluation)
// =============================================================================

__constant__ int PST_PAWN[64] = {
     0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
     5,  5, 10, 25, 25, 10,  5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5, -5,-10,  0,  0,-10, -5,  5,
     5, 10, 10,-20,-20, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0
};

__constant__ int PST_KNIGHT[64] = {
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50
};

__constant__ int PST_BISHOP[64] = {
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10, 10,  5, 10, 10,  5, 10,-10,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -20,-10,-10,-10,-10,-10,-10,-20
};

__constant__ int PST_ROOK[64] = {
     0,  0,  0,  0,  0,  0,  0,  0,
     5, 10, 10, 10, 10, 10, 10,  5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     0,  0,  0,  5,  5,  0,  0,  0
};

__constant__ int PST_QUEEN[64] = {
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5,  5,  5,  5,  0,-10,
     -5,  0,  5,  5,  5,  5,  0, -5,
      0,  0,  5,  5,  5,  5,  0, -5,
    -10,  5,  5,  5,  5,  5,  0,-10,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20
};

__constant__ int PST_KING_MG[64] = {
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
     20, 20,  0,  0,  0,  0, 20, 20,
     20, 30, 10,  0,  0, 10, 30, 20
};

__constant__ int PIECE_VALUES[13] = {
    0,    // EMPTY
    100,  // WPAWN
    320,  // WKNIGHT
    330,  // WBISHOP
    500,  // WROOK
    900,  // WQUEEN
    20000,// WKING
    100,  // BPAWN
    320,  // BKNIGHT
    330,  // BBISHOP
    500,  // BROOK
    900,  // BQUEEN
    20000 // BKING
};

// Knight move offsets
__constant__ int KNIGHT_OFFSETS[8] = {-17, -15, -10, -6, 6, 10, 15, 17};

// King move offsets
__constant__ int KING_OFFSETS[8] = {-9, -8, -7, -1, 1, 7, 8, 9};

// =============================================================================
// DEVICE HELPER FUNCTIONS
// =============================================================================

__host__ __device__ inline int sq_rank(int sq) { return sq >> 3; }
__host__ __device__ inline int sq_file(int sq) { return sq & 7; }
__host__ __device__ inline int make_sq(int rank, int file) { return (rank << 3) | file; }
__host__ __device__ inline int mirror_sq(int sq) { return ((7 - sq_rank(sq)) << 3) | sq_file(sq); }
__host__ __device__ inline int clamp_int(int value, int lo, int hi) {
    return value < lo ? lo : (value > hi ? hi : value);
}
__host__ __device__ inline float clamp_float(float value, float lo, float hi) {
    return value < lo ? lo : (value > hi ? hi : value);
}

__host__ __device__ inline bool on_board(int rank, int file) {
    return rank >= 0 && rank < 8 && file >= 0 && file < 8;
}

__host__ __device__ inline int piece_material_value(int piece) {
    switch (piece) {
        case WPAWN:
        case BPAWN:
            return 100;
        case WKNIGHT:
        case BKNIGHT:
            return 320;
        case WBISHOP:
        case BBISHOP:
            return 330;
        case WROOK:
        case BROOK:
            return 500;
        case WQUEEN:
        case BQUEEN:
            return 900;
        case WKING:
        case BKING:
            return 20000;
        default:
            return 0;
    }
}

__host__ __device__ inline int capture_piece_for_move(const BoardState* state, int move) {
    if (MOVE_FLAGS(move) == FLAG_EP) {
        return (state->side == WHITE_SIDE) ? BPAWN : WPAWN;
    }
    return state->board[MOVE_TO(move)];
}

__host__ __device__ inline int move_material_swing(const BoardState* state, int move) {
    int piece = state->board[MOVE_FROM(move)];
    int capture_value = piece_material_value(capture_piece_for_move(state, move));
    int promo_bonus = 0;
    if (MOVE_FLAGS(move) == FLAG_PROMO) {
        promo_bonus = piece_material_value(MOVE_PROMO(move)) - piece_material_value(piece);
    }
    return capture_value + promo_bonus;
}

__host__ __device__ inline float compute_root_puct_score(
    float playout_wins,
    int playout_visits,
    float prior_wins,
    int prior_visits,
    int total_playout_visits
) {
    int posterior_visits = playout_visits + prior_visits;
    float posterior_wins = playout_wins + prior_wins;
    float q = (posterior_visits > 0) ? posterior_wins / (float)posterior_visits : 0.5f;
    float prior_mean = (prior_visits > 0) ? prior_wins / (float)prior_visits : q;
    prior_mean = clamp_float(prior_mean, 0.0f, 1.0f);
    float u = ROOT_CPUCT * prior_mean * sqrtf((float)(total_playout_visits + 1)) / (1.0f + (float)playout_visits);
    return q + u;
}

__host__ __device__ inline float compute_root_uncertainty(
    float normalized_entropy,
    float top_mass,
    float second_mass,
    float score_gap
) {
    float posterior_gap = clamp_float(top_mass - second_mass, 0.0f, 1.0f);
    float uncertainty =
        0.50f * normalized_entropy +
        0.30f * (1.0f - posterior_gap) +
        0.20f * (1.0f - top_mass);
    if (score_gap > 0.0f) {
        uncertainty -= 0.10f * clamp_float(ROOT_SCORE_GAP_SCALE * score_gap, 0.0f, 1.0f);
    }
    return clamp_float(uncertainty, 0.0f, 1.0f);
}

enum SeedMode {
    SEED_MODE_FEN = 0,
    SEED_MODE_TIME = 1,
};

unsigned long long fnv1a64(const char* text) {
    unsigned long long hash = 1469598103934665603ULL;
    while (*text) {
        hash ^= (unsigned long long)(unsigned char)(*text++);
        hash *= 1099511628211ULL;
    }
    return hash;
}

unsigned long long make_seed(const char* fen,
                             int simulations,
                             int nlegal,
                             int seed_mode,
                             unsigned long long seed_base) {
    if (seed_mode == SEED_MODE_TIME) {
        return seed_base ^
               (unsigned long long)time(NULL) ^
               (unsigned long long)clock() ^
               (unsigned long long)nlegal;
    }
    unsigned long long hash = fnv1a64(fen);
    hash ^= (unsigned long long)simulations * 0x9E3779B185EBCA87ULL;
    hash ^= (unsigned long long)nlegal * 0xC2B2AE3D27D4EB4FULL;
    return seed_base ^ hash;
}

// =============================================================================
// ATTACK DETECTION (device)
// =============================================================================

__device__ bool is_square_attacked(const BoardState* state, int sq, int by_side) {
    int r = sq_rank(sq);
    int f = sq_file(sq);

    // Pawn attacks
    if (by_side == WHITE_SIDE) {
        if (r < 7) {
            if (f > 0 && state->board[sq + 7] == WPAWN) return true;
            if (f < 7 && state->board[sq + 9] == WPAWN) return true;
        }
    } else {
        if (r > 0) {
            if (f > 0 && state->board[sq - 9] == BPAWN) return true;
            if (f < 7 && state->board[sq - 7] == BPAWN) return true;
        }
    }

    // Knight attacks
    int knightPiece = (by_side == WHITE_SIDE) ? WKNIGHT : BKNIGHT;
    for (int i = 0; i < 8; i++) {
        int target = sq + KNIGHT_OFFSETS[i];
        if (target >= 0 && target < 64) {
            int tr = sq_rank(target);
            int tf = sq_file(target);
            int dr = abs(tr - r);
            int df = abs(tf - f);
            if ((dr == 2 && df == 1) || (dr == 1 && df == 2)) {
                if (state->board[target] == knightPiece) return true;
            }
        }
    }

    // King attacks
    int kingPiece = (by_side == WHITE_SIDE) ? WKING : BKING;
    for (int i = 0; i < 8; i++) {
        int target = sq + KING_OFFSETS[i];
        if (target >= 0 && target < 64) {
            int tr = sq_rank(target);
            int tf = sq_file(target);
            if (abs(tr - r) <= 1 && abs(tf - f) <= 1) {
                if (state->board[target] == kingPiece) return true;
            }
        }
    }

    // Sliding pieces: bishop/queen (diagonals)
    int bishopPiece = (by_side == WHITE_SIDE) ? WBISHOP : BBISHOP;
    int queenPiece = (by_side == WHITE_SIDE) ? WQUEEN : BQUEEN;
    int diag_dr[] = {-1, -1, 1, 1};
    int diag_df[] = {-1, 1, -1, 1};
    for (int d = 0; d < 4; d++) {
        int cr = r + diag_dr[d];
        int cf = f + diag_df[d];
        while (on_board(cr, cf)) {
            int piece = state->board[make_sq(cr, cf)];
            if (piece != EMPTY) {
                if (piece == bishopPiece || piece == queenPiece) return true;
                break;
            }
            cr += diag_dr[d];
            cf += diag_df[d];
        }
    }

    // Sliding pieces: rook/queen (ranks/files)
    int rookPiece = (by_side == WHITE_SIDE) ? WROOK : BROOK;
    int straight_dr[] = {-1, 1, 0, 0};
    int straight_df[] = {0, 0, -1, 1};
    for (int d = 0; d < 4; d++) {
        int cr = r + straight_dr[d];
        int cf = f + straight_df[d];
        while (on_board(cr, cf)) {
            int piece = state->board[make_sq(cr, cf)];
            if (piece != EMPTY) {
                if (piece == rookPiece || piece == queenPiece) return true;
                break;
            }
            cr += straight_dr[d];
            cf += straight_df[d];
        }
    }

    return false;
}

__device__ bool is_in_check(const BoardState* state, int side) {
    int kingSq = state->kingPos[side];
    return is_square_attacked(state, kingSq, 1 - side);
}

// =============================================================================
// MOVE GENERATION (device - pseudo-legal, validated after)
// =============================================================================

__device__ int generate_moves(const BoardState* state, int* moves) {
    int count = 0;
    int side = state->side;
    int opp = 1 - side;

    for (int sq = 0; sq < 64; sq++) {
        int piece = state->board[sq];
        if (piece == EMPTY) continue;
        if (side == WHITE_SIDE && !IS_WHITE(piece)) continue;
        if (side == BLACK_SIDE && !IS_BLACK(piece)) continue;

        int r = sq_rank(sq);
        int f = sq_file(sq);
        int ptype = PIECE_TYPE(piece);

        if (ptype == 1) { // PAWN
            int dir = (side == WHITE_SIDE) ? -1 : 1;
            int startRank = (side == WHITE_SIDE) ? 6 : 1;
            int promoRank = (side == WHITE_SIDE) ? 0 : 7;

            // Forward one
            int nr = r + dir;
            if (on_board(nr, f) && state->board[make_sq(nr, f)] == EMPTY) {
                if (nr == promoRank) {
                    int promoBase = (side == WHITE_SIDE) ? WQUEEN : BQUEEN;
                    // Queen, Rook, Bishop, Knight promotions
                    moves[count++] = MAKE_MOVE(sq, make_sq(nr, f), promoBase, FLAG_PROMO);
                    moves[count++] = MAKE_MOVE(sq, make_sq(nr, f), promoBase - 1, FLAG_PROMO);
                    moves[count++] = MAKE_MOVE(sq, make_sq(nr, f), promoBase - 2, FLAG_PROMO);
                    moves[count++] = MAKE_MOVE(sq, make_sq(nr, f), promoBase - 3, FLAG_PROMO);
                } else {
                    moves[count++] = MAKE_MOVE(sq, make_sq(nr, f), 0, FLAG_NONE);
                }
                // Forward two from starting rank
                if (r == startRank) {
                    int nr2 = r + 2 * dir;
                    if (state->board[make_sq(nr2, f)] == EMPTY) {
                        moves[count++] = MAKE_MOVE(sq, make_sq(nr2, f), 0, FLAG_DOUBLE);
                    }
                }
            }
            // Captures
            for (int df = -1; df <= 1; df += 2) {
                int nf = f + df;
                if (!on_board(nr, nf)) continue;
                int target = make_sq(nr, nf);
                int tp = state->board[target];
                if (tp != EMPTY && PIECE_COLOR(tp) == opp) {
                    if (nr == promoRank) {
                        int promoBase = (side == WHITE_SIDE) ? WQUEEN : BQUEEN;
                        moves[count++] = MAKE_MOVE(sq, target, promoBase, FLAG_PROMO);
                        moves[count++] = MAKE_MOVE(sq, target, promoBase - 1, FLAG_PROMO);
                        moves[count++] = MAKE_MOVE(sq, target, promoBase - 2, FLAG_PROMO);
                        moves[count++] = MAKE_MOVE(sq, target, promoBase - 3, FLAG_PROMO);
                    } else {
                        moves[count++] = MAKE_MOVE(sq, target, 0, FLAG_CAPTURE);
                    }
                }
                // En passant
                if (target == state->ep && state->ep >= 0) {
                    moves[count++] = MAKE_MOVE(sq, target, 0, FLAG_EP);
                }
            }
        }
        else if (ptype == 2) { // KNIGHT
            for (int i = 0; i < 8; i++) {
                int target = sq + KNIGHT_OFFSETS[i];
                if (target < 0 || target >= 64) continue;
                int tr = sq_rank(target);
                int tf = sq_file(target);
                int dr = abs(tr - r);
                int df2 = abs(tf - f);
                if (!((dr == 2 && df2 == 1) || (dr == 1 && df2 == 2))) continue;
                int tp = state->board[target];
                if (tp == EMPTY) {
                    moves[count++] = MAKE_MOVE(sq, target, 0, FLAG_NONE);
                } else if (PIECE_COLOR(tp) == opp) {
                    moves[count++] = MAKE_MOVE(sq, target, 0, FLAG_CAPTURE);
                }
            }
        }
        else if (ptype == 3 || ptype == 5) { // BISHOP or QUEEN (diagonals)
            int dirs[4][2] = {{-1,-1},{-1,1},{1,-1},{1,1}};
            for (int d = 0; d < 4; d++) {
                int cr = r + dirs[d][0];
                int cf = f + dirs[d][1];
                while (on_board(cr, cf)) {
                    int target = make_sq(cr, cf);
                    int tp = state->board[target];
                    if (tp == EMPTY) {
                        moves[count++] = MAKE_MOVE(sq, target, 0, FLAG_NONE);
                    } else {
                        if (PIECE_COLOR(tp) == opp) {
                            moves[count++] = MAKE_MOVE(sq, target, 0, FLAG_CAPTURE);
                        }
                        break;
                    }
                    cr += dirs[d][0];
                    cf += dirs[d][1];
                }
            }
            if (ptype == 3) continue; // bishop done, queen falls through to rook moves
        }
        if (ptype == 4 || ptype == 5) { // ROOK or QUEEN (straight)
            int dirs[4][2] = {{-1,0},{1,0},{0,-1},{0,1}};
            for (int d = 0; d < 4; d++) {
                int cr = r + dirs[d][0];
                int cf = f + dirs[d][1];
                while (on_board(cr, cf)) {
                    int target = make_sq(cr, cf);
                    int tp = state->board[target];
                    if (tp == EMPTY) {
                        moves[count++] = MAKE_MOVE(sq, target, 0, FLAG_NONE);
                    } else {
                        if (PIECE_COLOR(tp) == opp) {
                            moves[count++] = MAKE_MOVE(sq, target, 0, FLAG_CAPTURE);
                        }
                        break;
                    }
                    cr += dirs[d][0];
                    cf += dirs[d][1];
                }
            }
        }
        else if (ptype == 6) { // KING
            for (int i = 0; i < 8; i++) {
                int target = sq + KING_OFFSETS[i];
                if (target < 0 || target >= 64) continue;
                int tr = sq_rank(target);
                int tf = sq_file(target);
                if (abs(tr - r) > 1 || abs(tf - f) > 1) continue;
                int tp = state->board[target];
                if (tp == EMPTY) {
                    moves[count++] = MAKE_MOVE(sq, target, 0, FLAG_NONE);
                } else if (PIECE_COLOR(tp) == opp) {
                    moves[count++] = MAKE_MOVE(sq, target, 0, FLAG_CAPTURE);
                }
            }
            // Castling
            if (side == WHITE_SIDE && r == 7 && f == 4) {
                if ((state->castle & CASTLE_WK) && state->board[61] == EMPTY && state->board[62] == EMPTY
                    && state->board[63] == WROOK
                    && !is_square_attacked(state, 60, BLACK_SIDE)
                    && !is_square_attacked(state, 61, BLACK_SIDE)
                    && !is_square_attacked(state, 62, BLACK_SIDE)) {
                    moves[count++] = MAKE_MOVE(60, 62, 0, FLAG_CASTLE_K);
                }
                if ((state->castle & CASTLE_WQ) && state->board[59] == EMPTY && state->board[58] == EMPTY
                    && state->board[57] == EMPTY && state->board[56] == WROOK
                    && !is_square_attacked(state, 60, BLACK_SIDE)
                    && !is_square_attacked(state, 59, BLACK_SIDE)
                    && !is_square_attacked(state, 58, BLACK_SIDE)) {
                    moves[count++] = MAKE_MOVE(60, 58, 0, FLAG_CASTLE_Q);
                }
            }
            if (side == BLACK_SIDE && r == 0 && f == 4) {
                if ((state->castle & CASTLE_BK) && state->board[5] == EMPTY && state->board[6] == EMPTY
                    && state->board[7] == BROOK
                    && !is_square_attacked(state, 4, WHITE_SIDE)
                    && !is_square_attacked(state, 5, WHITE_SIDE)
                    && !is_square_attacked(state, 6, WHITE_SIDE)) {
                    moves[count++] = MAKE_MOVE(4, 6, 0, FLAG_CASTLE_K);
                }
                if ((state->castle & CASTLE_BQ) && state->board[3] == EMPTY && state->board[2] == EMPTY
                    && state->board[1] == EMPTY && state->board[0] == BROOK
                    && !is_square_attacked(state, 4, WHITE_SIDE)
                    && !is_square_attacked(state, 3, WHITE_SIDE)
                    && !is_square_attacked(state, 2, WHITE_SIDE)) {
                    moves[count++] = MAKE_MOVE(4, 2, 0, FLAG_CASTLE_Q);
                }
            }
        }

        if (count >= MAX_MOVES - 8) break; // safety
    }
    return count;
}

// =============================================================================
// MAKE MOVE (device)
// =============================================================================

__device__ void make_move(BoardState* state, int move) {
    int from = MOVE_FROM(move);
    int to = MOVE_TO(move);
    int flags = MOVE_FLAGS(move);
    int promo = MOVE_PROMO(move);
    int piece = state->board[from];
    int side = state->side;

    state->board[to] = piece;
    state->board[from] = EMPTY;
    state->halfmove++;

    // Update king position
    if (PIECE_TYPE(piece) == 6) {
        state->kingPos[side] = to;
    }

    // Pawn specifics
    if (PIECE_TYPE(piece) == 1) {
        state->halfmove = 0;
        if (flags == FLAG_DOUBLE) {
            state->ep = (from + to) / 2;
        } else if (flags == FLAG_EP) {
            int capturedSq = (side == WHITE_SIDE) ? to + 8 : to - 8;
            state->board[capturedSq] = EMPTY;
            state->ep = -1;
        } else if (flags == FLAG_PROMO) {
            state->board[to] = promo;
            state->ep = -1;
        } else {
            state->ep = -1;
        }
    } else {
        state->ep = -1;
    }

    // Captures reset halfmove
    if (flags == FLAG_CAPTURE) {
        state->halfmove = 0;
    }

    // Castling
    if (flags == FLAG_CASTLE_K) {
        if (side == WHITE_SIDE) {
            state->board[63] = EMPTY;
            state->board[61] = WROOK;
        } else {
            state->board[7] = EMPTY;
            state->board[5] = BROOK;
        }
    }
    if (flags == FLAG_CASTLE_Q) {
        if (side == WHITE_SIDE) {
            state->board[56] = EMPTY;
            state->board[59] = WROOK;
        } else {
            state->board[0] = EMPTY;
            state->board[3] = BROOK;
        }
    }

    // Update castling rights
    if (piece == WKING) state->castle &= ~(CASTLE_WK | CASTLE_WQ);
    if (piece == BKING) state->castle &= ~(CASTLE_BK | CASTLE_BQ);
    if (from == 63 || to == 63) state->castle &= ~CASTLE_WK;
    if (from == 56 || to == 56) state->castle &= ~CASTLE_WQ;
    if (from == 7 || to == 7) state->castle &= ~CASTLE_BK;
    if (from == 0 || to == 0) state->castle &= ~CASTLE_BQ;

    state->side = 1 - side;
    if (side == BLACK_SIDE) state->fullmove++;
}

// =============================================================================
// MATERIAL EVALUATION (device)
// =============================================================================

__device__ int evaluate_material(const BoardState* state) {
    int score = 0;
    for (int sq = 0; sq < 64; sq++) {
        int piece = state->board[sq];
        if (piece == EMPTY) continue;

        int val = PIECE_VALUES[piece];
        int ptype = PIECE_TYPE(piece);

        // Add PST bonus
        int pstSq = IS_WHITE(piece) ? sq : mirror_sq(sq);
        int pstBonus = 0;
        switch (ptype) {
            case 1: pstBonus = PST_PAWN[pstSq]; break;
            case 2: pstBonus = PST_KNIGHT[pstSq]; break;
            case 3: pstBonus = PST_BISHOP[pstSq]; break;
            case 4: pstBonus = PST_ROOK[pstSq]; break;
            case 5: pstBonus = PST_QUEEN[pstSq]; break;
            case 6: pstBonus = PST_KING_MG[pstSq]; break;
        }

        if (IS_WHITE(piece)) {
            score += val + pstBonus;
        } else {
            score -= val + pstBonus;
        }
    }
    // Return from side-to-move perspective
    return (state->side == WHITE_SIDE) ? score : -score;
}

// =============================================================================
// MCTS PLAYOUT KERNEL
// =============================================================================

__host__ __device__ int tactical_move_score(const BoardState* state, int move) {
    int score = move_material_swing(state, move);
    int moving = state->board[MOVE_FROM(move)];
    if (capture_piece_for_move(state, move) != EMPTY) {
        score -= piece_material_value(moving) / 4;
    }
    if (MOVE_FLAGS(move) == FLAG_CASTLE_K || MOVE_FLAGS(move) == FLAG_CASTLE_Q) {
        score += 80;
    }
    return score;
}

__host__ __device__ int opening_development_score(const BoardState* state, int move) {
    if (state->fullmove > OPENING_FULLMOVE_WINDOW || state->halfmove > 12) {
        return 0;
    }

    const int from = MOVE_FROM(move);
    const int to = MOVE_TO(move);
    const int piece = state->board[from];
    const int ptype = PIECE_TYPE(piece);
    const int from_rank = sq_rank(from);
    const int from_file = sq_file(from);
    const int to_rank = sq_rank(to);
    const int to_file = sq_file(to);
    int score = 0;

    if (MOVE_FLAGS(move) == FLAG_CASTLE_K || MOVE_FLAGS(move) == FLAG_CASTLE_Q) {
        score += 180;
    }

    if (ptype == 1) {
        const bool is_center_pawn = (from_file == 3 || from_file == 4);
        const bool is_semi_center_pawn = (from_file == 2 || from_file == 5);
        const bool advanced_two = abs(to_rank - from_rank) == 2;
        if (is_center_pawn) score += advanced_two ? 220 : 150;
        else if (is_semi_center_pawn) score += advanced_two ? 110 : 70;
        else if (from_file == 0 || from_file == 7) score -= 180;
        else score -= 40;
    } else if (ptype == 2) {
        if ((state->side == WHITE_SIDE && ((from == 62 && to == 45) || (from == 57 && to == 42))) ||
            (state->side == BLACK_SIDE && ((from == 6 && to == 21) || (from == 1 && to == 18)))) {
            score += 210;
        } else if (to_file >= 2 && to_file <= 5 && to_rank >= 2 && to_rank <= 5) {
            score += 110;
        } else {
            score -= 50;
        }
    } else if (ptype == 3) {
        if (to_file >= 2 && to_file <= 5 && to_rank >= 2 && to_rank <= 5) {
            score += 90;
        }
    } else if (ptype == 5) {
        score -= 180;
    } else if (ptype == 6 &&
               MOVE_FLAGS(move) != FLAG_CASTLE_K &&
               MOVE_FLAGS(move) != FLAG_CASTLE_Q) {
        score -= 220;
    }

    int center_distance = abs(3 - to_file) + abs(3 - to_rank);
    score += 18 - 6 * center_distance;
    return score;
}

__device__ int shallow_tactical_playout_score(const BoardState* state, int move) {
    int score = tactical_move_score(state, move);
    BoardState next = *state;
    make_move(&next, move);

    if (is_in_check(&next, next.side)) {
        score += TACTICAL_CHECK_BONUS;

        int reply_moves[MAX_MOVES];
        int nreplies = generate_moves(&next, reply_moves);
        int legal_replies = 0;
        for (int i = 0; i < nreplies; i++) {
            BoardState reply = next;
            make_move(&reply, reply_moves[i]);
            if (!is_in_check(&reply, 1 - reply.side)) {
                legal_replies++;
                break;
            }
        }
        if (legal_replies == 0) {
            score += TACTICAL_MATE_BONUS;
        }
    }

    int moved_piece = next.board[MOVE_TO(move)];
    if (moved_piece != EMPTY && is_square_attacked(&next, MOVE_TO(move), next.side)) {
        score -= piece_material_value(moved_piece) / 6;
    }

    return score;
}

__device__ int select_playout_move_index(
    const BoardState* state,
    const int* legal_moves,
    int nlegal,
    int depth,
    curandState* rng
) {
    int idx = curand(rng) % nlegal;
    if (depth >= SHALLOW_TACTICAL_PLIES) {
        return idx;
    }

    int best_idx = 0;
    int best_score = shallow_tactical_playout_score(state, legal_moves[0]);
    int best_tiebreak = tactical_move_score(state, legal_moves[0]);
    for (int i = 1; i < nlegal; i++) {
        int score = shallow_tactical_playout_score(state, legal_moves[i]);
        int tiebreak = tactical_move_score(state, legal_moves[i]);
        if (score > best_score ||
            (score == best_score && tiebreak > best_tiebreak) ||
            (score == best_score && tiebreak == best_tiebreak &&
             legal_moves[i] < legal_moves[best_idx])) {
            best_score = score;
            best_idx = i;
            best_tiebreak = tiebreak;
        }
    }

    if (best_score >= TACTICAL_FORCE_THRESHOLD) {
        return best_idx;
    }
    if (best_score >= TACTICAL_PREFER_THRESHOLD && curand_uniform(rng) < 0.80f) {
        return best_idx;
    }
    return idx;
}

__device__ float run_root_playout_score(
    const BoardState* initial_state,
    int root_move,
    curandState* rng
) {
    BoardState state = *initial_state;
    make_move(&state, root_move);

    // Root moves are host-filtered legal, but guard the kernel path.
    if (is_in_check(&state, 1 - state.side)) {
        return -1.0f;
    }

    int depth = 0;
    int result = 0; // 0 = ongoing
    int moves_buf[MAX_MOVES];

    while (depth < MAX_PLAYOUT_DEPTH) {
        if (state.halfmove >= 100) {
            result = 0;
            break;
        }

        int nmoves = generate_moves(&state, moves_buf);
        int legal_moves[MAX_MOVES];
        int nlegal = 0;

        for (int i = 0; i < nmoves && nlegal < MAX_MOVES; i++) {
            BoardState test = state;
            make_move(&test, moves_buf[i]);
            if (!is_in_check(&test, 1 - test.side)) {
                legal_moves[nlegal++] = moves_buf[i];
            }
        }

        if (nlegal == 0) {
            if (is_in_check(&state, state.side)) {
                result = (state.side == WHITE_SIDE) ? -1 : 1;
            } else {
                result = 0;
            }
            break;
        }

        int idx = select_playout_move_index(&state, legal_moves, nlegal, depth, rng);
        make_move(&state, legal_moves[idx]);
        depth++;
    }

    if (depth >= MAX_PLAYOUT_DEPTH) {
        int eval = evaluate_material(&state);
        int whiteEval = (state.side == WHITE_SIDE) ? eval : -eval;
        if (whiteEval > 100) result = 1;
        else if (whiteEval < -100) result = -1;
        else result = 0;
    }

    int rootSide = initial_state->side;
    if (rootSide == WHITE_SIDE) {
        return (result == 1) ? 1.0f : (result == 0) ? 0.5f : 0.0f;
    }
    return (result == -1) ? 1.0f : (result == 0) ? 0.5f : 0.0f;
}

__device__ int select_root_move_uct(
    const float* win_counts,
    const int* visit_counts,
    const int* inflight_visits,
    const float* prior_wins,
    const int* prior_visits,
    int num_root_moves
) {
    int total_visits = 0;
    int total_posterior_visits = 0;
    int top_posterior_visits = 0;
    int second_posterior_visits = 0;
    float best_base_score = -1e30f;
    float second_base_score = -1e30f;
    for (int i = 0; i < num_root_moves; i++) {
        total_visits += visit_counts[i] + inflight_visits[i];
        int posterior_visits = visit_counts[i] + prior_visits[i];
        total_posterior_visits += posterior_visits;
        if (posterior_visits > top_posterior_visits) {
            second_posterior_visits = top_posterior_visits;
            top_posterior_visits = posterior_visits;
        } else if (posterior_visits > second_posterior_visits) {
            second_posterior_visits = posterior_visits;
        }
    }

    float normalized_entropy = 0.0f;
    if (total_posterior_visits > 0 && num_root_moves > 1) {
        float entropy_bits = 0.0f;
        for (int i = 0; i < num_root_moves; i++) {
            int posterior_visits = visit_counts[i] + prior_visits[i];
            if (posterior_visits <= 0) continue;
            float mass = (float)posterior_visits / (float)total_posterior_visits;
            entropy_bits -= mass * (logf(mass) / logf(2.0f));
        }
        normalized_entropy = entropy_bits /
            (logf((float)num_root_moves) / logf(2.0f));
        normalized_entropy = clamp_float(normalized_entropy, 0.0f, 1.0f);
    }

    for (int i = 0; i < num_root_moves; i++) {
        int inflight = inflight_visits[i];
        int completed_visits = visit_counts[i];
        int posterior_visits = completed_visits + prior_visits[i];
        float posterior_wins = win_counts[i] + prior_wins[i];
        float q = (posterior_visits > 0)
            ? posterior_wins / (float)posterior_visits
            : 0.5f;
        float prior_mean = (prior_visits[i] > 0)
            ? prior_wins[i] / (float)prior_visits[i]
            : q;
        prior_mean = clamp_float(prior_mean, 0.0f, 1.0f);
        float base_u = ROOT_CPUCT * prior_mean *
            sqrtf((float)(total_visits + 1)) /
            (1.0f + (float)(completed_visits + inflight));
        float base_score = q + base_u;
        if (base_score > best_base_score) {
            second_base_score = best_base_score;
            best_base_score = base_score;
        } else if (base_score > second_base_score) {
            second_base_score = base_score;
        }
    }

    float top_mass = (total_posterior_visits > 0)
        ? (float)top_posterior_visits / (float)total_posterior_visits
        : 1.0f;
    float second_mass = (total_posterior_visits > 0)
        ? (float)second_posterior_visits / (float)total_posterior_visits
        : 0.0f;
    float score_gap = (second_base_score > -1e20f)
        ? (best_base_score - second_base_score)
        : 1.0f;
    float uncertainty = compute_root_uncertainty(
        normalized_entropy, top_mass, second_mass, score_gap
    );
    float cpuct_scale = clamp_float(
        ROOT_CPUCT_SCALE_MIN + 0.90f * uncertainty,
        ROOT_CPUCT_SCALE_MIN, ROOT_CPUCT_SCALE_MAX
    );
    float frontier_slack = 0.02f + 0.08f * uncertainty;

    int best_idx = 0;
    float best_score = -1e30f;

    for (int i = 0; i < num_root_moves; i++) {
        int inflight = inflight_visits[i];
        int completed_visits = visit_counts[i];
        int posterior_visits = completed_visits + prior_visits[i];
        float posterior_wins = win_counts[i] + prior_wins[i];
        float q = (posterior_visits > 0)
            ? posterior_wins / (float)posterior_visits
            : 0.5f;
        float prior_mean = (prior_visits[i] > 0)
            ? prior_wins[i] / (float)prior_visits[i]
            : q;
        prior_mean = clamp_float(prior_mean, 0.0f, 1.0f);
        float base_u = ROOT_CPUCT * prior_mean *
            sqrtf((float)(total_visits + 1)) /
            (1.0f + (float)(completed_visits + inflight));
        float base_score = q + base_u;
        float u = (ROOT_CPUCT * cpuct_scale) * prior_mean *
            sqrtf((float)(total_visits + 1)) /
            (1.0f + (float)(completed_visits + inflight));
        float score = q + u;
        if (base_score + frontier_slack < best_base_score) {
            continue;
        }
        if (score > best_score) {
            best_score = score;
            best_idx = i;
        }
    }

    return best_idx;
}

__global__ void init_root_priors_kernel(
    const BoardState* initial_state,
    const int* root_moves,
    int num_root_moves,
    float* prior_wins,
    int* prior_visits,
    float opening_prior_scale
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_root_moves) return;

    BoardState state = *initial_state;
    make_move(&state, root_moves[idx]);

    if (is_in_check(&state, 1 - state.side)) {
        prior_visits[idx] = 1;
        prior_wins[idx] = 0.0f;
        return;
    }

    int moves_buf[MAX_MOVES];
    int reply_buf[MAX_MOVES];
    int nmoves = generate_moves(&state, moves_buf);
    int nlegal = 0;
    for (int i = 0; i < nmoves && nlegal < MAX_MOVES; i++) {
        BoardState test = state;
        make_move(&test, moves_buf[i]);
        if (!is_in_check(&test, 1 - test.side)) {
            reply_buf[nlegal++] = moves_buf[i];
        }
    }

    if (nlegal == 0) {
        if (is_in_check(&state, state.side)) {
            prior_visits[idx] = ROOT_BASE_PRIOR_VISITS + 8;
            prior_wins[idx] = (float)prior_visits[idx];
        } else {
            prior_visits[idx] = ROOT_BASE_PRIOR_VISITS + 2;
            prior_wins[idx] = 0.5f * (float)prior_visits[idx];
        }
        return;
    }

    int eval = evaluate_material(&state);
    int root_eval = -eval;
    int worst_reply_eval = root_eval;
    int reply_pressure = 0;
    for (int i = 0; i < nlegal; i++) {
        BoardState reply = state;
        make_move(&reply, reply_buf[i]);
        int reply_eval = evaluate_material(&reply);
        if (reply_eval < worst_reply_eval) {
            worst_reply_eval = reply_eval;
        }
        if (is_in_check(&reply, reply.side)) {
            reply_pressure++;
        }
    }

    int tactical_bonus = tactical_move_score(initial_state, root_moves[idx]);
    float scaled_opening_bonus =
        opening_prior_scale * (float)opening_development_score(initial_state, root_moves[idx]);
    if (is_in_check(&state, state.side)) {
        tactical_bonus += 200;
    }

    int reply_aware_eval = (root_eval + 2 * worst_reply_eval) / 3;
    float eval_term = clamp_float((float)reply_aware_eval / 1800.0f, -0.32f, 0.35f);
    float tactical_term = clamp_float((float)tactical_bonus / 3000.0f, -0.08f, 0.22f);
    float opening_term = clamp_float(scaled_opening_bonus / 1800.0f, -0.18f, 0.24f);
    float pressure_term = -0.02f * (float)clamp_int(reply_pressure, 0, 4);
    float prior_mean = clamp_float(0.5f + eval_term + tactical_term +
                                   opening_term + pressure_term,
                                   0.02f, 0.98f);

    int base_prior_visits = ROOT_BASE_PRIOR_VISITS;
    if (tactical_bonus >= 300) base_prior_visits += 2;
    if (tactical_bonus >= TACTICAL_FORCE_THRESHOLD) base_prior_visits += 1;
    if (reply_aware_eval >= 500) base_prior_visits += 2;
    if (scaled_opening_bonus >= 180.0f) base_prior_visits += 2;
    else if (scaled_opening_bonus >= 90.0f) base_prior_visits += 1;
    if (is_in_check(&state, state.side)) base_prior_visits += 1;
    if (reply_aware_eval <= -500) base_prior_visits -= 1;
    if (scaled_opening_bonus <= -150.0f) base_prior_visits -= 1;
    base_prior_visits = clamp_int(base_prior_visits, 2, 12);

    prior_visits[idx] = base_prior_visits;
    prior_wins[idx] = prior_mean * (float)base_prior_visits;
}

__global__ void mcts_playout_kernel(
    const BoardState* initial_state,
    const int* root_moves,
    int num_root_moves,
    int total_simulations,
    float* win_counts,        // [num_root_moves]
    int* visit_counts,        // [num_root_moves]
    int* inflight_visits,     // [num_root_moves]
    const float* prior_wins,  // [num_root_moves]
    const int* prior_visits,  // [num_root_moves]
    unsigned long long seed
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;

    curandState rng;
    curand_init(seed, tid, 0, &rng);

    for (int sim_idx = tid; sim_idx < total_simulations; sim_idx += total_threads) {
        int move_idx = (sim_idx < num_root_moves)
            ? sim_idx
            : select_root_move_uct(
                win_counts, visit_counts, inflight_visits,
                prior_wins, prior_visits, num_root_moves
            );

        if (move_idx < 0 || move_idx >= num_root_moves) continue;

        atomicAdd(&inflight_visits[move_idx], 1);
        float score = run_root_playout_score(initial_state, root_moves[move_idx], &rng);
        atomicAdd(&inflight_visits[move_idx], -1);
        if (score < 0.0f) {
            continue;
        }

        atomicAdd(&visit_counts[move_idx], 1);
        atomicAdd(&win_counts[move_idx], score);
    }
}

__global__ void build_root_schedule_kernel(
    const BoardState* initial_state,
    const int* root_moves,
    const float* win_counts,
    const int* visit_counts,
    const float* prior_wins,
    const int* prior_visits,
    int num_root_moves,
    int completed_simulations,
    int batch_size,
    int tiered_allocation,
    int schedule_mode,
    int active_frontier_limit,
    int* out_schedule
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    int scheduled_visits[MAX_MOVES];
    float scheduled_wins[MAX_MOVES];
    int ranked_indices[MAX_MOVES];
    float ranked_scores[MAX_MOVES];
    int tactical_scores[MAX_MOVES];
    float frontier_weights[MAX_MOVES];
    int total_visits = 0;
    bool forcing_tactics = false;
    for (int i = 0; i < num_root_moves; i++) {
        scheduled_visits[i] = visit_counts[i];
        scheduled_wins[i] = win_counts[i];
        ranked_indices[i] = i;
        ranked_scores[i] = -1e30f;
        tactical_scores[i] = 0;
        if (tiered_allocation) {
            tactical_scores[i] = shallow_tactical_playout_score(initial_state, root_moves[i]);
            if (tactical_scores[i] >= TACTICAL_FORCE_THRESHOLD ||
                move_material_swing(initial_state, root_moves[i]) >= 700) {
                forcing_tactics = true;
            }
        }
        total_visits += visit_counts[i];
    }

    for (int sim = 0; sim < batch_size; sim++) {
        int global_sim = completed_simulations + sim;
        int move_idx;
        if (!tiered_allocation && global_sim < num_root_moves) {
            move_idx = global_sim;
        } else {
            int total_with_scheduled = total_visits + sim;
            float posterior_mass_total = 0.0f;
            for (int i = 0; i < num_root_moves; i++) {
                ranked_indices[i] = i;
                ranked_scores[i] = compute_root_puct_score(
                    scheduled_wins[i],
                    scheduled_visits[i],
                    prior_wins[i],
                    prior_visits[i],
                    total_with_scheduled
                );
                posterior_mass_total += (float)(scheduled_visits[i] + prior_visits[i]);
            }

            for (int i = 1; i < num_root_moves; i++) {
                int idx = ranked_indices[i];
                float score = ranked_scores[idx];
                int j = i - 1;
                while (j >= 0 && ranked_scores[ranked_indices[j]] < score) {
                    ranked_indices[j + 1] = ranked_indices[j];
                    j--;
                }
                ranked_indices[j + 1] = idx;
            }

            if (schedule_mode == ROOT_SCHEDULE_HALVING) {
                int active_limit = active_frontier_limit;
                if (active_limit <= 0 || active_limit > num_root_moves) {
                    active_limit = num_root_moves;
                }
                if (forcing_tactics || active_limit <= 1) {
                    move_idx = ranked_indices[0];
                } else {
                    move_idx = ranked_indices[sim % active_limit];
                }
            } else if (schedule_mode == ROOT_SCHEDULE_COMPARE) {
                int active_limit = active_frontier_limit;
                if (active_limit <= 0 || active_limit > num_root_moves) {
                    active_limit = num_root_moves;
                }
                if (forcing_tactics || active_limit <= 1) {
                    move_idx = ranked_indices[0];
                } else {
                    int best_slot = 0;
                    int best_compare_visits = 1 << 30;
                    float best_compare_score = -1e30f;
                    for (int i = 0; i < active_limit; i++) {
                        int idx = ranked_indices[i];
                        int compare_visits = scheduled_visits[idx];
                        float score = ranked_scores[idx];
                        if (compare_visits < best_compare_visits ||
                            (compare_visits == best_compare_visits && score > best_compare_score)) {
                            best_compare_visits = compare_visits;
                            best_compare_score = score;
                            best_slot = i;
                        }
                    }
                    move_idx = ranked_indices[best_slot];
                }
            } else if (!tiered_allocation || forcing_tactics || num_root_moves <= ROOT_HEAD_MAX) {
                move_idx = ranked_indices[0];
            } else {
                if (posterior_mass_total <= 0.0f) {
                    posterior_mass_total = (float)num_root_moves;
                }

                float posterior_entropy_bits = 0.0f;
                for (int i = 0; i < num_root_moves; i++) {
                    int idx = ranked_indices[i];
                    float mass =
                        (float)(scheduled_visits[idx] + prior_visits[idx]) / posterior_mass_total;
                    if (mass > 0.0f) {
                        posterior_entropy_bits -= mass * (logf(mass) / logf(2.0f));
                    }
                }
                float normalized_entropy = 0.0f;
                if (num_root_moves > 1) {
                    normalized_entropy = posterior_entropy_bits /
                        (logf((float)num_root_moves) / logf(2.0f));
                    normalized_entropy = clamp_float(normalized_entropy, 0.0f, 1.0f);
                }

                int top_idx = ranked_indices[0];
                float top_mass =
                    (float)(scheduled_visits[top_idx] + prior_visits[top_idx]) / posterior_mass_total;
                float second_mass = 0.0f;
                float score_gap = 1.0f;
                if (num_root_moves > 1) {
                    int second_idx = ranked_indices[1];
                    second_mass =
                        (float)(scheduled_visits[second_idx] + prior_visits[second_idx]) /
                        posterior_mass_total;
                    score_gap = ranked_scores[top_idx] - ranked_scores[second_idx];
                }
                float posterior_gap = clamp_float(top_mass - second_mass, 0.0f, 1.0f);
                float uncertainty = compute_root_uncertainty(
                    normalized_entropy, top_mass, second_mass, score_gap
                );

                float frontier_mass_target = clamp_float(
                    ROOT_FRONTIER_MASS_MIN +
                        (ROOT_FRONTIER_MASS_MAX - ROOT_FRONTIER_MASS_MIN) * uncertainty,
                    ROOT_FRONTIER_MASS_MIN,
                    ROOT_FRONTIER_MASS_MAX
                );
                float frontier_score_slack =
                    ROOT_FRONTIER_TAU_MIN +
                    (ROOT_FRONTIER_TAU_MAX - ROOT_FRONTIER_TAU_MIN) * uncertainty;
                int frontier_size = 0;
                float frontier_mass = 0.0f;
                while (frontier_size < num_root_moves && frontier_size < ROOT_FRONTIER_MAX) {
                    int idx = ranked_indices[frontier_size];
                    frontier_mass +=
                        (float)(scheduled_visits[idx] + prior_visits[idx]) / posterior_mass_total;
                    frontier_size++;
                    if (frontier_size >= 2) {
                        bool mass_ready = frontier_mass >= frontier_mass_target;
                        bool score_ready =
                            frontier_size < num_root_moves &&
                            ranked_scores[ranked_indices[frontier_size]] + frontier_score_slack <
                                ranked_scores[top_idx];
                        if (mass_ready || score_ready) {
                            break;
                        }
                    }
                }
                if (frontier_size < 2 && num_root_moves > 1) frontier_size = 2;
                if (frontier_size > num_root_moves) frontier_size = num_root_moves;

                int remaining_budget = batch_size - sim;
                bool leader_locked = false;
                if (frontier_size > 1) {
                    int leader_visits = scheduled_visits[top_idx] + prior_visits[top_idx];
                    int runner_visits =
                        scheduled_visits[ranked_indices[1]] + prior_visits[ranked_indices[1]];
                    leader_locked = leader_visits > runner_visits + remaining_budget;
                }

                if (leader_locked) {
                    move_idx = (remaining_budget > 1 && frontier_size > 1)
                        ? ranked_indices[1]
                        : ranked_indices[0];
                } else {
                    float tau =
                        ROOT_FRONTIER_TAU_MIN +
                        (ROOT_FRONTIER_TAU_MAX - ROOT_FRONTIER_TAU_MIN) * uncertainty;
                    float weight_sum = 0.0f;
                    for (int i = 0; i < frontier_size; i++) {
                        int idx = ranked_indices[i];
                        float score_delta = (ranked_scores[idx] - ranked_scores[top_idx]) / tau;
                        if (score_delta < -12.0f) score_delta = -12.0f;
                        frontier_weights[i] = expf(score_delta);
                        weight_sum += frontier_weights[i];
                    }
                    if (weight_sum <= 0.0f) {
                        move_idx = ranked_indices[0];
                    } else {
                        for (int i = 0; i < frontier_size; i++) {
                            frontier_weights[i] /= weight_sum;
                        }

                        if (frontier_size > 1) {
                            float leader_cap = 0.90f - 0.35f * uncertainty;
                            if (frontier_weights[0] > leader_cap) {
                                float excess = frontier_weights[0] - leader_cap;
                                frontier_weights[0] = leader_cap;
                                float other_sum = 0.0f;
                                for (int i = 1; i < frontier_size; i++) {
                                    other_sum += frontier_weights[i];
                                }
                                if (other_sum <= 0.0f) {
                                    float share = excess / (float)(frontier_size - 1);
                                    for (int i = 1; i < frontier_size; i++) {
                                        frontier_weights[i] += share;
                                    }
                                } else {
                                    for (int i = 1; i < frontier_size; i++) {
                                        frontier_weights[i] += excess * (frontier_weights[i] / other_sum);
                                    }
                                }
                            }
                        }

                        int best_frontier_slot = 0;
                        float best_deficit = -1e30f;
                        for (int i = 0; i < frontier_size; i++) {
                            int idx = ranked_indices[i];
                            float scheduled_share =
                                (float)(scheduled_visits[idx] + prior_visits[idx]) / posterior_mass_total;
                            float deficit = frontier_weights[i] - scheduled_share;
                            if (deficit > best_deficit) {
                                best_deficit = deficit;
                                best_frontier_slot = i;
                            }
                        }
                        move_idx = ranked_indices[best_frontier_slot];
                    }
                }
            }
        }
        out_schedule[sim] = move_idx;
        if (tiered_allocation) {
            int posterior_visits = scheduled_visits[move_idx] + prior_visits[move_idx];
            float posterior_mean = (posterior_visits > 0)
                ? (scheduled_wins[move_idx] + prior_wins[move_idx]) / (float)posterior_visits
                : 0.5f;
            scheduled_wins[move_idx] += posterior_mean;
        }
        scheduled_visits[move_idx] += 1;
    }
}

__global__ void scheduled_mcts_playout_kernel(
    const BoardState* initial_state,
    const int* root_moves,
    const int* schedule,
    int batch_size,
    int simulation_offset,
    float* win_counts,
    int* visit_counts,
    unsigned long long seed
) {
    int sim = blockIdx.x * blockDim.x + threadIdx.x;
    if (sim >= batch_size) return;

    int move_idx = schedule[sim];
    if (move_idx < 0) return;

    curandState rng;
    curand_init(seed, (unsigned long long)(simulation_offset + sim), 0, &rng);

    float score = run_root_playout_score(initial_state, root_moves[move_idx], &rng);
    if (score < 0.0f) return;

    atomicAdd(&visit_counts[move_idx], 1);
    atomicAdd(&win_counts[move_idx], score);
}

// =============================================================================
// HOST: FEN PARSER
// =============================================================================

void parse_fen(const char* fen, BoardState* state) {
    memset(state, 0, sizeof(BoardState));
    for (int i = 0; i < 64; i++) state->board[i] = EMPTY;
    state->ep = -1;
    state->kingPos[0] = -1;
    state->kingPos[1] = -1;

    int sq = 0;
    int i = 0;

    // Piece placement
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
                case 'K': piece = WKING; state->kingPos[WHITE_SIDE] = sq; break;
                case 'p': piece = BPAWN; break;
                case 'n': piece = BKNIGHT; break;
                case 'b': piece = BBISHOP; break;
                case 'r': piece = BROOK; break;
                case 'q': piece = BQUEEN; break;
                case 'k': piece = BKING; state->kingPos[BLACK_SIDE] = sq; break;
            }
            state->board[sq++] = piece;
        }
    }

    // Skip space
    while (fen[i] == ' ') i++;

    // Side to move
    state->side = (fen[i] == 'b') ? BLACK_SIDE : WHITE_SIDE;
    i++;
    while (fen[i] == ' ') i++;

    // Castling
    state->castle = 0;
    if (fen[i] == '-') {
        i++;
    } else {
        while (fen[i] && fen[i] != ' ') {
            switch (fen[i]) {
                case 'K': state->castle |= CASTLE_WK; break;
                case 'Q': state->castle |= CASTLE_WQ; break;
                case 'k': state->castle |= CASTLE_BK; break;
                case 'q': state->castle |= CASTLE_BQ; break;
            }
            i++;
        }
    }
    while (fen[i] == ' ') i++;

    // En passant
    if (fen[i] == '-') {
        state->ep = -1;
        i++;
    } else if (fen[i] >= 'a' && fen[i] <= 'h') {
        int file = fen[i] - 'a';
        i++;
        int rank = 8 - (fen[i] - '0');
        i++;
        state->ep = make_sq(rank, file);
    }
    while (fen[i] == ' ') i++;

    // Halfmove clock
    state->halfmove = 0;
    while (fen[i] >= '0' && fen[i] <= '9') {
        state->halfmove = state->halfmove * 10 + (fen[i] - '0');
        i++;
    }
    while (fen[i] == ' ') i++;

    // Fullmove number
    state->fullmove = 0;
    while (fen[i] >= '0' && fen[i] <= '9') {
        state->fullmove = state->fullmove * 10 + (fen[i] - '0');
        i++;
    }
}

// Host-side make_sq
int host_make_sq(int rank, int file) { return (rank << 3) | file; }

// =============================================================================
// HOST: MOVE GENERATION (mirrors device for root moves)
// =============================================================================

bool host_on_board(int rank, int file) {
    return rank >= 0 && rank < 8 && file >= 0 && file < 8;
}

int host_sq_rank(int sq) { return sq >> 3; }
int host_sq_file(int sq) { return sq & 7; }

static const int H_KNIGHT_OFFSETS[8] = {-17, -15, -10, -6, 6, 10, 15, 17};
static const int H_KING_OFFSETS[8] = {-9, -8, -7, -1, 1, 7, 8, 9};

bool host_is_square_attacked(const BoardState* state, int sq, int by_side) {
    int r = host_sq_rank(sq);
    int f = host_sq_file(sq);

    // Pawn attacks
    if (by_side == WHITE_SIDE) {
        if (r < 7) {
            if (f > 0 && state->board[sq + 7] == WPAWN) return true;
            if (f < 7 && state->board[sq + 9] == WPAWN) return true;
        }
    } else {
        if (r > 0) {
            if (f > 0 && state->board[sq - 9] == BPAWN) return true;
            if (f < 7 && state->board[sq - 7] == BPAWN) return true;
        }
    }

    // Knight
    int knightPiece = (by_side == WHITE_SIDE) ? WKNIGHT : BKNIGHT;
    for (int i = 0; i < 8; i++) {
        int target = sq + H_KNIGHT_OFFSETS[i];
        if (target >= 0 && target < 64) {
            int tr = host_sq_rank(target);
            int tf = host_sq_file(target);
            int dr = abs(tr - r);
            int df = abs(tf - f);
            if ((dr == 2 && df == 1) || (dr == 1 && df == 2)) {
                if (state->board[target] == knightPiece) return true;
            }
        }
    }

    // King
    int kingPiece = (by_side == WHITE_SIDE) ? WKING : BKING;
    for (int i = 0; i < 8; i++) {
        int target = sq + H_KING_OFFSETS[i];
        if (target >= 0 && target < 64) {
            int tr = host_sq_rank(target);
            int tf = host_sq_file(target);
            if (abs(tr - r) <= 1 && abs(tf - f) <= 1) {
                if (state->board[target] == kingPiece) return true;
            }
        }
    }

    // Bishop/Queen diagonals
    int bishopPiece = (by_side == WHITE_SIDE) ? WBISHOP : BBISHOP;
    int queenPiece = (by_side == WHITE_SIDE) ? WQUEEN : BQUEEN;
    int diag_dr[] = {-1, -1, 1, 1};
    int diag_df[] = {-1, 1, -1, 1};
    for (int d = 0; d < 4; d++) {
        int cr = r + diag_dr[d];
        int cf = f + diag_df[d];
        while (host_on_board(cr, cf)) {
            int piece = state->board[host_make_sq(cr, cf)];
            if (piece != EMPTY) {
                if (piece == bishopPiece || piece == queenPiece) return true;
                break;
            }
            cr += diag_dr[d];
            cf += diag_df[d];
        }
    }

    // Rook/Queen straight
    int rookPiece = (by_side == WHITE_SIDE) ? WROOK : BROOK;
    int straight_dr[] = {-1, 1, 0, 0};
    int straight_df[] = {0, 0, -1, 1};
    for (int d = 0; d < 4; d++) {
        int cr = r + straight_dr[d];
        int cf = f + straight_df[d];
        while (host_on_board(cr, cf)) {
            int piece = state->board[host_make_sq(cr, cf)];
            if (piece != EMPTY) {
                if (piece == rookPiece || piece == queenPiece) return true;
                break;
            }
            cr += straight_dr[d];
            cf += straight_df[d];
        }
    }

    return false;
}

bool host_is_in_check(const BoardState* state, int side) {
    return host_is_square_attacked(state, state->kingPos[side], 1 - side);
}

void host_make_move(BoardState* state, int move) {
    int from = MOVE_FROM(move);
    int to = MOVE_TO(move);
    int flags = MOVE_FLAGS(move);
    int promo = MOVE_PROMO(move);
    int piece = state->board[from];
    int side = state->side;

    state->board[to] = piece;
    state->board[from] = EMPTY;
    state->halfmove++;

    if (PIECE_TYPE(piece) == 6) {
        state->kingPos[side] = to;
    }

    if (PIECE_TYPE(piece) == 1) {
        state->halfmove = 0;
        if (flags == FLAG_DOUBLE) {
            state->ep = (from + to) / 2;
        } else if (flags == FLAG_EP) {
            int capturedSq = (side == WHITE_SIDE) ? to + 8 : to - 8;
            state->board[capturedSq] = EMPTY;
            state->ep = -1;
        } else if (flags == FLAG_PROMO) {
            state->board[to] = promo;
            state->ep = -1;
        } else {
            state->ep = -1;
        }
    } else {
        state->ep = -1;
    }

    if (flags == FLAG_CAPTURE) state->halfmove = 0;

    if (flags == FLAG_CASTLE_K) {
        if (side == WHITE_SIDE) { state->board[63] = EMPTY; state->board[61] = WROOK; }
        else { state->board[7] = EMPTY; state->board[5] = BROOK; }
    }
    if (flags == FLAG_CASTLE_Q) {
        if (side == WHITE_SIDE) { state->board[56] = EMPTY; state->board[59] = WROOK; }
        else { state->board[0] = EMPTY; state->board[3] = BROOK; }
    }

    if (piece == WKING) state->castle &= ~(CASTLE_WK | CASTLE_WQ);
    if (piece == BKING) state->castle &= ~(CASTLE_BK | CASTLE_BQ);
    if (from == 63 || to == 63) state->castle &= ~CASTLE_WK;
    if (from == 56 || to == 56) state->castle &= ~CASTLE_WQ;
    if (from == 7 || to == 7) state->castle &= ~CASTLE_BK;
    if (from == 0 || to == 0) state->castle &= ~CASTLE_BQ;

    state->side = 1 - side;
    if (side == BLACK_SIDE) state->fullmove++;
}

int host_generate_moves(const BoardState* state, int* moves) {
    int count = 0;
    int side = state->side;
    int opp = 1 - side;

    for (int sq = 0; sq < 64; sq++) {
        int piece = state->board[sq];
        if (piece == EMPTY) continue;
        if (side == WHITE_SIDE && !IS_WHITE(piece)) continue;
        if (side == BLACK_SIDE && !IS_BLACK(piece)) continue;

        int r = host_sq_rank(sq);
        int f = host_sq_file(sq);
        int ptype = PIECE_TYPE(piece);

        if (ptype == 1) { // PAWN
            int dir = (side == WHITE_SIDE) ? -1 : 1;
            int startRank = (side == WHITE_SIDE) ? 6 : 1;
            int promoRank = (side == WHITE_SIDE) ? 0 : 7;

            int nr = r + dir;
            if (host_on_board(nr, f) && state->board[host_make_sq(nr, f)] == EMPTY) {
                if (nr == promoRank) {
                    int promoBase = (side == WHITE_SIDE) ? WQUEEN : BQUEEN;
                    moves[count++] = MAKE_MOVE(sq, host_make_sq(nr, f), promoBase, FLAG_PROMO);
                    moves[count++] = MAKE_MOVE(sq, host_make_sq(nr, f), promoBase - 1, FLAG_PROMO);
                    moves[count++] = MAKE_MOVE(sq, host_make_sq(nr, f), promoBase - 2, FLAG_PROMO);
                    moves[count++] = MAKE_MOVE(sq, host_make_sq(nr, f), promoBase - 3, FLAG_PROMO);
                } else {
                    moves[count++] = MAKE_MOVE(sq, host_make_sq(nr, f), 0, FLAG_NONE);
                }
                if (r == startRank) {
                    int nr2 = r + 2 * dir;
                    if (state->board[host_make_sq(nr2, f)] == EMPTY) {
                        moves[count++] = MAKE_MOVE(sq, host_make_sq(nr2, f), 0, FLAG_DOUBLE);
                    }
                }
            }
            for (int df = -1; df <= 1; df += 2) {
                int nf = f + df;
                if (!host_on_board(nr, nf)) continue;
                int target = host_make_sq(nr, nf);
                int tp = state->board[target];
                if (tp != EMPTY && PIECE_COLOR(tp) == opp) {
                    if (nr == promoRank) {
                        int promoBase = (side == WHITE_SIDE) ? WQUEEN : BQUEEN;
                        moves[count++] = MAKE_MOVE(sq, target, promoBase, FLAG_PROMO);
                        moves[count++] = MAKE_MOVE(sq, target, promoBase - 1, FLAG_PROMO);
                        moves[count++] = MAKE_MOVE(sq, target, promoBase - 2, FLAG_PROMO);
                        moves[count++] = MAKE_MOVE(sq, target, promoBase - 3, FLAG_PROMO);
                    } else {
                        moves[count++] = MAKE_MOVE(sq, target, 0, FLAG_CAPTURE);
                    }
                }
                if (target == state->ep && state->ep >= 0) {
                    moves[count++] = MAKE_MOVE(sq, target, 0, FLAG_EP);
                }
            }
        }
        else if (ptype == 2) { // KNIGHT
            for (int i = 0; i < 8; i++) {
                int target = sq + H_KNIGHT_OFFSETS[i];
                if (target < 0 || target >= 64) continue;
                int tr = host_sq_rank(target);
                int tf = host_sq_file(target);
                int dr = abs(tr - r);
                int df2 = abs(tf - f);
                if (!((dr == 2 && df2 == 1) || (dr == 1 && df2 == 2))) continue;
                int tp = state->board[target];
                if (tp == EMPTY) {
                    moves[count++] = MAKE_MOVE(sq, target, 0, FLAG_NONE);
                } else if (PIECE_COLOR(tp) == opp) {
                    moves[count++] = MAKE_MOVE(sq, target, 0, FLAG_CAPTURE);
                }
            }
        }
        else if (ptype == 3 || ptype == 5) { // BISHOP or QUEEN
            int dirs[4][2] = {{-1,-1},{-1,1},{1,-1},{1,1}};
            for (int d = 0; d < 4; d++) {
                int cr = r + dirs[d][0];
                int cf = f + dirs[d][1];
                while (host_on_board(cr, cf)) {
                    int target = host_make_sq(cr, cf);
                    int tp = state->board[target];
                    if (tp == EMPTY) {
                        moves[count++] = MAKE_MOVE(sq, target, 0, FLAG_NONE);
                    } else {
                        if (PIECE_COLOR(tp) == opp)
                            moves[count++] = MAKE_MOVE(sq, target, 0, FLAG_CAPTURE);
                        break;
                    }
                    cr += dirs[d][0];
                    cf += dirs[d][1];
                }
            }
            if (ptype == 3) continue;
        }
        if (ptype == 4 || ptype == 5) { // ROOK or QUEEN
            int dirs[4][2] = {{-1,0},{1,0},{0,-1},{0,1}};
            for (int d = 0; d < 4; d++) {
                int cr = r + dirs[d][0];
                int cf = f + dirs[d][1];
                while (host_on_board(cr, cf)) {
                    int target = host_make_sq(cr, cf);
                    int tp = state->board[target];
                    if (tp == EMPTY) {
                        moves[count++] = MAKE_MOVE(sq, target, 0, FLAG_NONE);
                    } else {
                        if (PIECE_COLOR(tp) == opp)
                            moves[count++] = MAKE_MOVE(sq, target, 0, FLAG_CAPTURE);
                        break;
                    }
                    cr += dirs[d][0];
                    cf += dirs[d][1];
                }
            }
        }
        else if (ptype == 6) { // KING
            for (int i = 0; i < 8; i++) {
                int target = sq + H_KING_OFFSETS[i];
                if (target < 0 || target >= 64) continue;
                int tr = host_sq_rank(target);
                int tf = host_sq_file(target);
                if (abs(tr - r) > 1 || abs(tf - f) > 1) continue;
                int tp = state->board[target];
                if (tp == EMPTY) {
                    moves[count++] = MAKE_MOVE(sq, target, 0, FLAG_NONE);
                } else if (PIECE_COLOR(tp) == opp) {
                    moves[count++] = MAKE_MOVE(sq, target, 0, FLAG_CAPTURE);
                }
            }
            // Castling
            if (side == WHITE_SIDE && r == 7 && f == 4) {
                if ((state->castle & CASTLE_WK) && state->board[61] == EMPTY && state->board[62] == EMPTY
                    && state->board[63] == WROOK
                    && !host_is_square_attacked(state, 60, BLACK_SIDE)
                    && !host_is_square_attacked(state, 61, BLACK_SIDE)
                    && !host_is_square_attacked(state, 62, BLACK_SIDE)) {
                    moves[count++] = MAKE_MOVE(60, 62, 0, FLAG_CASTLE_K);
                }
                if ((state->castle & CASTLE_WQ) && state->board[59] == EMPTY && state->board[58] == EMPTY
                    && state->board[57] == EMPTY && state->board[56] == WROOK
                    && !host_is_square_attacked(state, 60, BLACK_SIDE)
                    && !host_is_square_attacked(state, 59, BLACK_SIDE)
                    && !host_is_square_attacked(state, 58, BLACK_SIDE)) {
                    moves[count++] = MAKE_MOVE(60, 58, 0, FLAG_CASTLE_Q);
                }
            }
            if (side == BLACK_SIDE && r == 0 && f == 4) {
                if ((state->castle & CASTLE_BK) && state->board[5] == EMPTY && state->board[6] == EMPTY
                    && state->board[7] == BROOK
                    && !host_is_square_attacked(state, 4, WHITE_SIDE)
                    && !host_is_square_attacked(state, 5, WHITE_SIDE)
                    && !host_is_square_attacked(state, 6, WHITE_SIDE)) {
                    moves[count++] = MAKE_MOVE(4, 6, 0, FLAG_CASTLE_K);
                }
                if ((state->castle & CASTLE_BQ) && state->board[3] == EMPTY && state->board[2] == EMPTY
                    && state->board[1] == EMPTY && state->board[0] == BROOK
                    && !host_is_square_attacked(state, 4, WHITE_SIDE)
                    && !host_is_square_attacked(state, 3, WHITE_SIDE)
                    && !host_is_square_attacked(state, 2, WHITE_SIDE)) {
                    moves[count++] = MAKE_MOVE(4, 2, 0, FLAG_CASTLE_Q);
                }
            }
        }

        if (count >= MAX_MOVES - 8) break;
    }
    return count;
}

// =============================================================================
// HOST: Move to UCI string
// =============================================================================

void move_to_uci(int move, char* uci) {
    int from = MOVE_FROM(move);
    int to = MOVE_TO(move);
    int promo = MOVE_PROMO(move);
    int flags = MOVE_FLAGS(move);

    uci[0] = 'a' + host_sq_file(from);
    uci[1] = '0' + (8 - host_sq_rank(from));
    uci[2] = 'a' + host_sq_file(to);
    uci[3] = '0' + (8 - host_sq_rank(to));
    uci[4] = '\0';

    if (flags == FLAG_PROMO) {
        int ptype = PIECE_TYPE(promo);
        switch (ptype) {
            case 5: uci[4] = 'q'; break;
            case 4: uci[4] = 'r'; break;
            case 3: uci[4] = 'b'; break;
            case 2: uci[4] = 'n'; break;
        }
        uci[5] = '\0';
    }
}

// =============================================================================
// HOST: JSON OUTPUT MODE
// =============================================================================

void output_json_with_fen(const char* fen,
                         int* legal_moves,
                         int nlegal,
                         float* h_wins,
                         int* h_visits,
                         float* h_prior_wins,
                         int* h_prior_visits,
                         int best_idx,
                         int total_sims,
                         int shortlist_threshold,
                         const char* selection_metric,
                         const char* best_uci) {
    int posterior_total_sims = total_sims;
    for (int i = 0; i < nlegal; i++) {
        posterior_total_sims += h_prior_visits[i];
    }

    printf("{\"fen\":\"%s\",\"bestmove\":\"%s\",", fen, best_uci);
    int best_posterior_visits = h_visits[best_idx] + h_prior_visits[best_idx];
    float best_raw = (h_visits[best_idx] > 0) ? h_wins[best_idx] / (float)h_visits[best_idx] : 0.5f;
    float best_prior = (h_prior_visits[best_idx] > 0) ? h_prior_wins[best_idx] / (float)h_prior_visits[best_idx] : 0.5f;
    float best_posterior = (best_posterior_visits > 0)
        ? (h_wins[best_idx] + h_prior_wins[best_idx]) / (float)best_posterior_visits
        : 0.5f;
    float best_root_score = compute_root_puct_score(
        h_wins[best_idx],
        h_visits[best_idx],
        h_prior_wins[best_idx],
        h_prior_visits[best_idx],
        total_sims
    );
    printf("\"winrate\":%.4f,", best_posterior);
    printf("\"raw_winrate\":%.4f,", best_raw);
    printf("\"prior_winrate\":%.4f,", best_prior);
    printf("\"posterior_winrate\":%.4f,", best_posterior);
    printf("\"root_score\":%.4f,", best_root_score);
    printf("\"visits\":%d,", h_visits[best_idx]);
    printf("\"prior_visits\":%d,", h_prior_visits[best_idx]);
    printf("\"posterior_visits\":%d,", best_posterior_visits);
    printf("\"effective_visits\":%d,", best_posterior_visits);
    printf("\"selection_metric\":\"%s\",",
           (selection_metric != NULL) ? selection_metric : "posterior_winrate_shortlist");
    printf("\"selection_shortlist_visits\":%d,", shortlist_threshold);
    printf("\"simulations\":%d,", total_sims);
    printf("\"posterior_simulations\":%d,", posterior_total_sims);
    printf("\"moves\":[");
    for (int i = 0; i < nlegal; i++) {
        char uci[6];
        move_to_uci(legal_moves[i], uci);
        int posterior_visits = h_visits[i] + h_prior_visits[i];
        float raw_wr = (h_visits[i] > 0) ? h_wins[i] / h_visits[i] : 0.5f;
        float prior_wr = (h_prior_visits[i] > 0) ? h_prior_wins[i] / (float)h_prior_visits[i] : 0.5f;
        float posterior = (posterior_visits > 0)
            ? (h_wins[i] + h_prior_wins[i]) / (float)posterior_visits
            : 0.5f;
        float root_score = compute_root_puct_score(
            h_wins[i], h_visits[i], h_prior_wins[i], h_prior_visits[i], total_sims
        );
        printf("{\"move\":\"%s\",\"winrate\":%.4f,\"raw_winrate\":%.4f,\"prior_winrate\":%.4f,\"posterior_winrate\":%.4f,\"root_score\":%.4f,\"visits\":%d,\"prior_visits\":%d,\"posterior_visits\":%d,\"effective_visits\":%d}",
               uci, raw_wr, raw_wr, prior_wr, posterior, root_score, h_visits[i], h_prior_visits[i], posterior_visits, posterior_visits);
        if (i < nlegal - 1) printf(",");
    }
    printf("]}\n");
}

// =============================================================================
// MAIN
// =============================================================================

int main(int argc, char** argv) {
    // Parse arguments
    int simulations = DEFAULT_SIMULATIONS;
    int json_output = 0;
    int verbose = 0;
    int seed_mode = SEED_MODE_FEN;
    unsigned long long seed_base = 0xC0DA2026ULL;
    float opening_prior_scale = 0.7532f;
    float opening_selection_scale = 0.00075f;
    int root_batch_size = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--simulations") == 0 && i + 1 < argc) {
            simulations = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--json") == 0) {
            json_output = 1;
        } else if (strcmp(argv[i], "--verbose") == 0) {
            verbose = 1;
        } else if (strcmp(argv[i], "--seed-mode") == 0 && i + 1 < argc) {
            const char* mode = argv[++i];
            if (strcmp(mode, "time") == 0) {
                seed_mode = SEED_MODE_TIME;
            } else {
                seed_mode = SEED_MODE_FEN;
            }
        } else if (strcmp(argv[i], "--seed-base") == 0 && i + 1 < argc) {
            seed_base = strtoull(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--opening-prior-scale") == 0 && i + 1 < argc) {
            opening_prior_scale = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--opening-selection-scale") == 0 && i + 1 < argc) {
            opening_selection_scale = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--root-batch-size") == 0 && i + 1 < argc) {
            root_batch_size = atoi(argv[++i]);
        }
    }

    // Allocate device memory once
    BoardState* d_state;
    int* d_moves;
    float* d_wins;
    int* d_visits;
    int* d_inflight_visits;
    float* d_prior_wins;
    int* d_prior_visits;
    int* d_schedule;

    cudaMalloc(&d_state, sizeof(BoardState));
    cudaMalloc(&d_moves, MAX_MOVES * sizeof(int));
    cudaMalloc(&d_wins, MAX_MOVES * sizeof(float));
    cudaMalloc(&d_visits, MAX_MOVES * sizeof(int));
    cudaMalloc(&d_inflight_visits, MAX_MOVES * sizeof(int));
    cudaMalloc(&d_prior_wins, MAX_MOVES * sizeof(float));
    cudaMalloc(&d_prior_visits, MAX_MOVES * sizeof(int));
    cudaMalloc(&d_schedule, MAX_MOVES * sizeof(int));

    // Read FEN from stdin (one per line)
    char fen[512];
    while (fgets(fen, sizeof(fen), stdin)) {
        // Trim newline
        int len = strlen(fen);
        while (len > 0 && (fen[len-1] == '\n' || fen[len-1] == '\r')) fen[--len] = '\0';
        if (len < 10) continue;

        if (verbose) fprintf(stderr, "FEN: %s\n", fen);

        // Parse FEN
        BoardState host_state;
        parse_fen(fen, &host_state);

        // Generate root moves on host
        int all_moves[MAX_MOVES];
        int nmoves = host_generate_moves(&host_state, all_moves);

        // Filter to legal moves
        int legal_moves[MAX_MOVES];
        int nlegal = 0;

        for (int i = 0; i < nmoves; i++) {
            BoardState test = host_state;
            host_make_move(&test, all_moves[i]);
            if (!host_is_in_check(&test, 1 - test.side)) {
                legal_moves[nlegal++] = all_moves[i];
            }
        }

        if (nlegal == 0) {
            if (json_output) {
                printf("{\"fen\":\"%s\",\"bestmove\":\"0000\",\"winrate\":0.0,\"raw_winrate\":0.0,\"prior_winrate\":0.0,\"posterior_winrate\":0.0,\"root_score\":0.0,\"visits\":0,\"prior_visits\":0,\"posterior_visits\":0,\"effective_visits\":0,\"selection_metric\":\"terminal\",\"selection_shortlist_visits\":0,\"simulations\":0,\"posterior_simulations\":0,\"moves\":[]}\n", fen);
            } else {
                printf("bestmove 0000 winrate 0.00 simulations 0\n");
            }
            fflush(stdout);
            continue;
        }

        if (nlegal == 1) {
            char uci[6];
            move_to_uci(legal_moves[0], uci);
            if (json_output) {
                printf("{\"fen\":\"%s\",\"bestmove\":\"%s\",\"winrate\":0.50,\"raw_winrate\":0.50,\"prior_winrate\":0.50,\"posterior_winrate\":0.50,\"root_score\":0.50,\"visits\":0,\"prior_visits\":0,\"posterior_visits\":0,\"effective_visits\":0,\"selection_metric\":\"forced\",\"selection_shortlist_visits\":0,\"simulations\":0,\"posterior_simulations\":0,\"moves\":[{\"move\":\"%s\",\"winrate\":0.50,\"raw_winrate\":0.50,\"prior_winrate\":0.50,\"posterior_winrate\":0.50,\"root_score\":0.50,\"visits\":0,\"prior_visits\":0,\"posterior_visits\":0,\"effective_visits\":0}]}\n", fen, uci, uci);
            } else {
                printf("bestmove %s winrate 0.50 simulations 0\n", uci);
            }
            fflush(stdout);
            continue;
        }

        // Reset/Update device memory
        cudaMemcpy(d_state, &host_state, sizeof(BoardState), cudaMemcpyHostToDevice);
        cudaMemcpy(d_moves, legal_moves, nlegal * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemset(d_wins, 0, nlegal * sizeof(float));
        cudaMemset(d_visits, 0, nlegal * sizeof(int));
        cudaMemset(d_inflight_visits, 0, nlegal * sizeof(int));
        cudaMemset(d_prior_wins, 0, nlegal * sizeof(float));
        cudaMemset(d_prior_visits, 0, nlegal * sizeof(int));

        // Spend the requested budget globally, while guaranteeing root coverage.
        int total_simulations = simulations;
        if (total_simulations < nlegal) total_simulations = nlegal;
        int playout_threads = BLOCK_SIZE;
        int numBlocks = (total_simulations + playout_threads - 1) / playout_threads;
        if (numBlocks > 1024) numBlocks = 1024;

        unsigned long long seed = make_seed(
            fen, total_simulations, nlegal, seed_mode, seed_base
        );

        int priorBlocks = (nlegal + BLOCK_SIZE - 1) / BLOCK_SIZE;
        init_root_priors_kernel<<<priorBlocks, BLOCK_SIZE>>>(
            d_state, d_moves, nlegal, d_prior_wins, d_prior_visits,
            opening_prior_scale
        );
        cudaDeviceSynchronize();

        int active_root_batch_size = root_batch_size;
        if (seed_mode == SEED_MODE_FEN) {
            if (active_root_batch_size <= 0) active_root_batch_size = BLOCK_SIZE;
        }
        if (active_root_batch_size > MAX_MOVES) active_root_batch_size = MAX_MOVES;

        float* h_wins = (float*)malloc(nlegal * sizeof(float));
        int* h_visits = (int*)malloc(nlegal * sizeof(int));
        float* h_prior_wins = (float*)malloc(nlegal * sizeof(float));
        int* h_prior_visits = (int*)malloc(nlegal * sizeof(int));

        bool has_forcing_tactic_root = false;
        for (int i = 0; i < nlegal; i++) {
            if (move_material_swing(&host_state, legal_moves[i]) >= 700 ||
                tactical_move_score(&host_state, legal_moves[i]) >= TACTICAL_FORCE_THRESHOLD) {
                has_forcing_tactic_root = true;
                break;
            }
        }

        if (seed_mode == SEED_MODE_FEN || active_root_batch_size > 0) {
            int batch_size = total_simulations;
            if (batch_size > active_root_batch_size) batch_size = active_root_batch_size;
            int completed_simulations = 0;
            bool halving_active = false;
            int halving_frontier_limit = 0;
            bool comparison_active = false;
            int comparison_frontier_limit = 0;
            while (completed_simulations < total_simulations) {
                int current_batch = total_simulations - completed_simulations;
                int schedule_mode = ROOT_SCHEDULE_FRONTIER;
                int active_frontier_limit = 0;
                if (halving_active && halving_frontier_limit >= 2) {
                    if (current_batch > halving_frontier_limit) {
                        current_batch = halving_frontier_limit;
                    }
                    schedule_mode = ROOT_SCHEDULE_HALVING;
                    active_frontier_limit = halving_frontier_limit;
                } else if (comparison_active && comparison_frontier_limit >= 2) {
                    if (current_batch > batch_size) current_batch = batch_size;
                    schedule_mode = ROOT_SCHEDULE_COMPARE;
                    active_frontier_limit = comparison_frontier_limit;
                } else {
                    if (current_batch > batch_size) current_batch = batch_size;
                }
                build_root_schedule_kernel<<<1, 1>>>(
                    d_state, d_moves,
                    d_wins, d_visits, d_prior_wins, d_prior_visits,
                    nlegal, completed_simulations, current_batch,
                    1,
                    schedule_mode,
                    active_frontier_limit,
                    d_schedule
                );
                int batch_blocks = (current_batch + BLOCK_SIZE - 1) / BLOCK_SIZE;
                scheduled_mcts_playout_kernel<<<batch_blocks, BLOCK_SIZE>>>(
                    d_state, d_moves, d_schedule, current_batch, completed_simulations,
                    d_wins, d_visits, seed
                );
                completed_simulations += current_batch;

                if (completed_simulations >= total_simulations ||
                    seed_mode != SEED_MODE_TIME ||
                    has_forcing_tactic_root ||
                    active_root_batch_size <= 0) {
                    halving_active = false;
                    continue;
                }

                cudaMemcpy(h_wins, d_wins, nlegal * sizeof(float), cudaMemcpyDeviceToHost);
                cudaMemcpy(h_visits, d_visits, nlegal * sizeof(int), cudaMemcpyDeviceToHost);
                cudaMemcpy(h_prior_wins, d_prior_wins, nlegal * sizeof(float), cudaMemcpyDeviceToHost);
                cudaMemcpy(h_prior_visits, d_prior_visits, nlegal * sizeof(int), cudaMemcpyDeviceToHost);

                int ranked_by_score[MAX_MOVES];
                float ranked_scores[MAX_MOVES];
                int total_playout_visits = 0;
                int total_eff_visits = 0;
                for (int i = 0; i < nlegal; i++) {
                    ranked_by_score[i] = i;
                    total_playout_visits += h_visits[i];
                    total_eff_visits += h_visits[i] + h_prior_visits[i];
                }
                for (int i = 0; i < nlegal; i++) {
                    ranked_scores[i] = compute_root_puct_score(
                        h_wins[i], h_visits[i], h_prior_wins[i], h_prior_visits[i], total_playout_visits
                    );
                }
                for (int i = 1; i < nlegal; i++) {
                    int idx = ranked_by_score[i];
                    float score = ranked_scores[idx];
                    int j = i - 1;
                    while (j >= 0 && ranked_scores[ranked_by_score[j]] < score) {
                        ranked_by_score[j + 1] = ranked_by_score[j];
                        j--;
                    }
                    ranked_by_score[j + 1] = idx;
                }

                float normalized_entropy = 0.0f;
                if (total_eff_visits > 0 && nlegal > 1) {
                    float entropy_bits = 0.0f;
                    for (int i = 0; i < nlegal; i++) {
                        int idx = ranked_by_score[i];
                        int eff_visits = h_visits[idx] + h_prior_visits[idx];
                        if (eff_visits <= 0) continue;
                        float mass = (float)eff_visits / (float)total_eff_visits;
                        entropy_bits -= mass * (logf(mass) / logf(2.0f));
                    }
                    normalized_entropy = entropy_bits /
                        (logf((float)nlegal) / logf(2.0f));
                    normalized_entropy = clamp_float(normalized_entropy, 0.0f, 1.0f);
                }

                int top_idx = ranked_by_score[0];
                float top_mass = (total_eff_visits > 0)
                    ? (float)(h_visits[top_idx] + h_prior_visits[top_idx]) / (float)total_eff_visits
                    : 1.0f;
                float second_mass = 0.0f;
                float score_gap = 1.0f;
                if (nlegal > 1) {
                    int second_idx = ranked_by_score[1];
                    second_mass = (total_eff_visits > 0)
                        ? (float)(h_visits[second_idx] + h_prior_visits[second_idx]) /
                            (float)total_eff_visits
                        : 0.0f;
                    score_gap = ranked_scores[top_idx] - ranked_scores[second_idx];
                }
                float posterior_gap = clamp_float(top_mass - second_mass, 0.0f, 1.0f);
                float uncertainty = compute_root_uncertainty(
                    normalized_entropy, top_mass, second_mass, score_gap
                );
                float frontier_mass_target = clamp_float(
                    ROOT_FRONTIER_MASS_MIN +
                        (ROOT_FRONTIER_MASS_MAX - ROOT_FRONTIER_MASS_MIN) * uncertainty,
                    ROOT_FRONTIER_MASS_MIN,
                    ROOT_FRONTIER_MASS_MAX
                );
                float frontier_score_slack =
                    ROOT_FRONTIER_TAU_MIN +
                    (ROOT_FRONTIER_TAU_MAX - ROOT_FRONTIER_TAU_MIN) * uncertainty;
                int frontier_size = 0;
                float frontier_mass = 0.0f;
                while (frontier_size < nlegal && frontier_size < ROOT_FRONTIER_MAX) {
                    int idx = ranked_by_score[frontier_size];
                    int eff_visits = h_visits[idx] + h_prior_visits[idx];
                    frontier_mass += (total_eff_visits > 0)
                        ? (float)eff_visits / (float)total_eff_visits
                        : 0.0f;
                    frontier_size++;
                    if (frontier_size >= 2) {
                        bool mass_ready = frontier_mass >= frontier_mass_target;
                        bool score_ready =
                            frontier_size < nlegal &&
                            ranked_scores[ranked_by_score[frontier_size]] + frontier_score_slack <
                                ranked_scores[top_idx];
                        if (mass_ready || score_ready) {
                            break;
                        }
                    }
                }
                if (frontier_size < 2 && nlegal > 1) frontier_size = 2;
                if (frontier_size > nlegal) frontier_size = nlegal;

                int remaining_after_batch = total_simulations - completed_simulations;
                bool leader_locked = false;
                if (frontier_size > 1) {
                    int leader_visits = h_visits[top_idx] + h_prior_visits[top_idx];
                    int runner_visits =
                        h_visits[ranked_by_score[1]] + h_prior_visits[ranked_by_score[1]];
                    leader_locked = leader_visits > runner_visits + remaining_after_batch;
                }
                bool flat_root =
                    frontier_size >= ROOT_HALVING_FRONTIER_MIN &&
                    uncertainty >= ROOT_HALVING_UNCERTAINTY_MIN &&
                    posterior_gap <= ROOT_HALVING_POSTERIOR_GAP_MAX &&
                    score_gap <= ROOT_HALVING_SCORE_GAP_MAX &&
                    remaining_after_batch >= frontier_size;
                bool narrow_comparison_root =
                    !flat_root &&
                    frontier_size >= 2 &&
                    frontier_size <= ROOT_COMPARISON_FRONTIER_MAX &&
                    uncertainty >= ROOT_COMPARISON_UNCERTAINTY_MIN &&
                    posterior_gap <= ROOT_COMPARISON_POSTERIOR_GAP_MAX &&
                    score_gap <= ROOT_COMPARISON_SCORE_GAP_MAX &&
                    remaining_after_batch >= 2;
                int comparison_limit = frontier_size;
                if (comparison_limit > ROOT_COMPARISON_FRONTIER_MAX) {
                    comparison_limit = ROOT_COMPARISON_FRONTIER_MAX;
                }
                if (leader_locked && comparison_limit > 2) {
                    comparison_limit = 2;
                }
                if (comparison_limit < 2 && frontier_size > 1) {
                    comparison_limit = 2;
                }

                if (!halving_active) {
                    if (flat_root) {
                        halving_active = true;
                        halving_frontier_limit = frontier_size;
                        comparison_active = false;
                        comparison_frontier_limit = 0;
                    }
                } else {
                    if (!flat_root || remaining_after_batch < 2) {
                        halving_active = false;
                        halving_frontier_limit = 0;
                    } else {
                        halving_frontier_limit = (halving_frontier_limit + 1) / 2;
                        if (halving_frontier_limit > frontier_size) {
                            halving_frontier_limit = frontier_size;
                        }
                        if (halving_frontier_limit < 2 || remaining_after_batch < halving_frontier_limit) {
                            halving_active = false;
                            halving_frontier_limit = 0;
                        }
                    }
                }

                if (!halving_active) {
                    if (!comparison_active) {
                        if (narrow_comparison_root) {
                            comparison_active = true;
                            comparison_frontier_limit = comparison_limit;
                        }
                    } else {
                        if (!narrow_comparison_root || remaining_after_batch < 2) {
                            comparison_active = false;
                            comparison_frontier_limit = 0;
                        } else {
                            comparison_frontier_limit = comparison_limit;
                        }
                    }
                } else {
                    comparison_active = false;
                    comparison_frontier_limit = 0;
                }
            }
        } else {
            // Launch kernel
            mcts_playout_kernel<<<numBlocks, playout_threads>>>(
                d_state, d_moves, nlegal, total_simulations,
                d_wins, d_visits, d_inflight_visits,
                d_prior_wins, d_prior_visits, seed
            );
        }

        cudaDeviceSynchronize();

        // Copy results back
        cudaMemcpy(h_wins, d_wins, nlegal * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_visits, d_visits, nlegal * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_prior_wins, d_prior_wins, nlegal * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_prior_visits, d_prior_visits, nlegal * sizeof(int), cudaMemcpyDeviceToHost);

        // Find best move
        int total_visits = 0;
        int max_eff_visits = 0;
        int ranked_by_visits[MAX_MOVES];

        for (int i = 0; i < nlegal; i++) {
            ranked_by_visits[i] = i;
            total_visits += h_visits[i];
            int eff_visits = h_visits[i] + h_prior_visits[i];
            if (eff_visits > max_eff_visits) {
                max_eff_visits = eff_visits;
            }
        }

        for (int i = 1; i < nlegal; i++) {
            int idx = ranked_by_visits[i];
            int eff_visits = h_visits[idx] + h_prior_visits[idx];
            int j = i - 1;
            while (j >= 0) {
                int prev_idx = ranked_by_visits[j];
                int prev_eff_visits = h_visits[prev_idx] + h_prior_visits[prev_idx];
                if (prev_eff_visits >= eff_visits) {
                    break;
                }
                ranked_by_visits[j + 1] = prev_idx;
                j--;
            }
            ranked_by_visits[j + 1] = idx;
        }

        int total_eff_visits = 0;
        for (int i = 0; i < nlegal; i++) {
            total_eff_visits += h_visits[i] + h_prior_visits[i];
        }
        float normalized_entropy = 0.0f;
        if (total_eff_visits > 0 && nlegal > 1) {
            float entropy_bits = 0.0f;
            for (int i = 0; i < nlegal; i++) {
                int idx = ranked_by_visits[i];
                int eff_visits = h_visits[idx] + h_prior_visits[idx];
                if (eff_visits <= 0) continue;
                float mass = (float)eff_visits / (float)total_eff_visits;
                entropy_bits -= mass * (logf(mass) / logf(2.0f));
            }
            normalized_entropy = entropy_bits /
                (logf((float)nlegal) / logf(2.0f));
            normalized_entropy = clamp_float(normalized_entropy, 0.0f, 1.0f);
        }
        float top_mass = (total_eff_visits > 0)
            ? (float)(h_visits[ranked_by_visits[0]] + h_prior_visits[ranked_by_visits[0]]) /
                (float)total_eff_visits
            : 1.0f;
        float second_mass = (total_eff_visits > 0 && nlegal > 1)
            ? (float)(h_visits[ranked_by_visits[1]] + h_prior_visits[ranked_by_visits[1]]) /
                (float)total_eff_visits
            : 0.0f;
        float score_gap = 1.0f;
        if (nlegal > 1) {
            int top_idx = ranked_by_visits[0];
            int second_idx = ranked_by_visits[1];
            float top_root_score = compute_root_puct_score(
                h_wins[top_idx], h_visits[top_idx], h_prior_wins[top_idx], h_prior_visits[top_idx], total_visits
            );
            float second_root_score = compute_root_puct_score(
                h_wins[second_idx], h_visits[second_idx], h_prior_wins[second_idx], h_prior_visits[second_idx], total_visits
            );
            score_gap = top_root_score - second_root_score;
        }
        float uncertainty = compute_root_uncertainty(
            normalized_entropy, top_mass, second_mass, score_gap
        );
        float frontier_mass_target = clamp_float(
            ROOT_FRONTIER_MASS_MIN + 0.30f * uncertainty,
            ROOT_FRONTIER_MASS_MIN, ROOT_FRONTIER_MASS_MAX
        );
        int frontier_size = 0;
        float frontier_mass = 0.0f;
        while (frontier_size < nlegal && frontier_size < ROOT_FRONTIER_MAX) {
            int idx = ranked_by_visits[frontier_size];
            int eff_visits = h_visits[idx] + h_prior_visits[idx];
            frontier_mass += (total_eff_visits > 0)
                ? (float)eff_visits / (float)total_eff_visits
                : 0.0f;
            frontier_size++;
            if (frontier_size >= 2 && frontier_mass >= frontier_mass_target) {
                break;
            }
        }
        if (frontier_size < 2 && nlegal > 1) frontier_size = 2;
        if (frontier_size > nlegal) frontier_size = nlegal;
        int shortlist_threshold = h_visits[ranked_by_visits[frontier_size - 1]] +
            h_prior_visits[ranked_by_visits[frontier_size - 1]];

        int best_idx = 0;
        float best_posterior = -1.0f;
        float best_selection = -1e30f;
        float best_root_score = -1e30f;
        int best_eff_visits = -1;
        bool has_forcing_tactic = false;
        for (int i = 0; i < nlegal; i++) {
            if (move_material_swing(&host_state, legal_moves[i]) >= 700) {
                has_forcing_tactic = true;
                break;
            }
        }
        const bool opening_bias_selection =
            !has_forcing_tactic &&
            opening_selection_scale > 0.0f &&
            host_state.fullmove <= OPENING_FULLMOVE_WINDOW &&
            host_state.halfmove <= 12;
        const char* selection_metric = opening_bias_selection
            ? "posterior_plus_opening_bias_frontier"
            : (has_forcing_tactic
                ? "posterior_plus_tactical_bias_frontier"
                : "posterior_winrate_frontier");

        for (int i = 0; i < nlegal; i++) {
            int eff_visits = h_visits[i] + h_prior_visits[i];
            float posterior = (eff_visits > 0)
                ? (h_wins[i] + h_prior_wins[i]) / (float)eff_visits
                : 0.5f;
            float selection_score = posterior;
            if (opening_bias_selection) {
                selection_score += opening_selection_scale *
                    (float)opening_development_score(&host_state, legal_moves[i]);
            } else if (has_forcing_tactic) {
                selection_score += TACTICAL_SELECTION_BIAS *
                    (float)tactical_move_score(&host_state, legal_moves[i]);
            }
            float root_score = compute_root_puct_score(
                h_wins[i], h_visits[i], h_prior_wins[i], h_prior_visits[i], total_visits
            );
            if (eff_visits < shortlist_threshold) {
                continue;
            }
            if (selection_score > best_selection ||
                (selection_score == best_selection && posterior > best_posterior) ||
                (selection_score == best_selection && posterior == best_posterior && eff_visits > best_eff_visits) ||
                (selection_score == best_selection && posterior == best_posterior &&
                 eff_visits == best_eff_visits && root_score > best_root_score)) {
                best_selection = selection_score;
                best_posterior = posterior;
                best_root_score = root_score;
                best_eff_visits = eff_visits;
                best_idx = i;
            }
        }

        if (best_eff_visits < 0) {
            for (int i = 0; i < nlegal; i++) {
                int eff_visits = h_visits[i] + h_prior_visits[i];
                float posterior = (eff_visits > 0)
                    ? (h_wins[i] + h_prior_wins[i]) / (float)eff_visits
                    : 0.5f;
                float selection_score = posterior;
                if (opening_bias_selection) {
                    selection_score += opening_selection_scale *
                        (float)opening_development_score(&host_state, legal_moves[i]);
                } else if (has_forcing_tactic) {
                    selection_score += TACTICAL_SELECTION_BIAS *
                        (float)tactical_move_score(&host_state, legal_moves[i]);
                }
                float root_score = compute_root_puct_score(
                    h_wins[i], h_visits[i], h_prior_wins[i], h_prior_visits[i], total_visits
                );
                if (selection_score > best_selection ||
                    (selection_score == best_selection && eff_visits > best_eff_visits) ||
                    (selection_score == best_selection && eff_visits == best_eff_visits && posterior > best_posterior) ||
                    (selection_score == best_selection && eff_visits == best_eff_visits &&
                     posterior == best_posterior && root_score > best_root_score)) {
                    best_selection = selection_score;
                    best_posterior = posterior;
                    best_root_score = root_score;
                    best_eff_visits = eff_visits;
                    best_idx = i;
                }
            }
        }

        if (has_forcing_tactic && best_eff_visits >= 0) {
            int current_tactical_score = tactical_move_score(&host_state, legal_moves[best_idx]);
            int tactical_best_idx = -1;
            int tactical_best_score = -1000000000;
            float tactical_best_posterior = -1.0f;
            float tactical_best_root_score = -1e30f;
            int tactical_best_eff_visits = -1;
            for (int i = 0; i < nlegal; i++) {
                int eff_visits = h_visits[i] + h_prior_visits[i];
                if (eff_visits < shortlist_threshold) {
                    continue;
                }
                float posterior = (eff_visits > 0)
                    ? (h_wins[i] + h_prior_wins[i]) / (float)eff_visits
                    : 0.5f;
                if (posterior + TACTICAL_POSTERIOR_SLACK < best_posterior) {
                    continue;
                }
                int tactical_score = tactical_move_score(&host_state, legal_moves[i]);
                if (tactical_score < TACTICAL_PREFER_THRESHOLD) {
                    continue;
                }
                float root_score = compute_root_puct_score(
                    h_wins[i], h_visits[i], h_prior_wins[i], h_prior_visits[i], total_visits
                );
                if (tactical_score > tactical_best_score ||
                    (tactical_score == tactical_best_score && posterior > tactical_best_posterior) ||
                    (tactical_score == tactical_best_score && posterior == tactical_best_posterior &&
                     eff_visits > tactical_best_eff_visits) ||
                    (tactical_score == tactical_best_score && posterior == tactical_best_posterior &&
                     eff_visits == tactical_best_eff_visits && root_score > tactical_best_root_score)) {
                    tactical_best_idx = i;
                    tactical_best_score = tactical_score;
                    tactical_best_posterior = posterior;
                    tactical_best_eff_visits = eff_visits;
                    tactical_best_root_score = root_score;
                }
            }
            if (tactical_best_idx >= 0 &&
                tactical_best_idx != best_idx &&
                tactical_best_score >= current_tactical_score + 100) {
                best_idx = tactical_best_idx;
                best_posterior = tactical_best_posterior;
                best_eff_visits = tactical_best_eff_visits;
                best_root_score = tactical_best_root_score;
                selection_metric = "posterior_tactical_shortlist";
            }
        }

        char best_uci[6];
        move_to_uci(legal_moves[best_idx], best_uci);

        // Output result
        if (json_output) {
            output_json_with_fen(
                fen, legal_moves, nlegal, h_wins, h_visits,
                h_prior_wins, h_prior_visits, best_idx, total_visits,
                shortlist_threshold, selection_metric, best_uci
            );
        } else {
            printf("bestmove %s winrate %.2f simulations %d\n", best_uci, best_posterior, total_visits);
        }
        fflush(stdout);

        free(h_wins);
        free(h_visits);
        free(h_prior_wins);
        free(h_prior_visits);
    }

    // Cleanup
    cudaFree(d_state);
    cudaFree(d_moves);
    cudaFree(d_wins);
    cudaFree(d_visits);
    cudaFree(d_inflight_visits);
    cudaFree(d_prior_wins);
    cudaFree(d_prior_visits);
    cudaFree(d_schedule);

    return 0;
}
