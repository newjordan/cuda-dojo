// =============================================================================
// eval.cu
//
// PeSTO tapered eval. Tables are extracted verbatim from gpu_fighter.cu
// (lines 83-205). When gpu_fighter and batch_eval differed, gpu_fighter
// wins per the extraction plan.
//
// Two storage backings: __constant__ for device, regular const arrays for
// host. The numbers are identical, so host eval and device eval return the
// exact same score for any position.
// =============================================================================
#include "eval.cuh"
#include "movegen.cuh"
#include "make_unmake.cuh"
#include "attacks.cuh"

namespace engine {

// CFX eval contribution: mobility-differential. Per the audit's
// Req 5 partner-counterfactual, "how restricted is the opponent"
// is a positional principle PeSTO doesn't directly capture.
//
// Returns legal-move count for `color` from `s`. Uses a null-move
// to swap turn when `color` != s->side; falls back to a piece-count
// estimate if the null-move would leave the king in check (illegal).
__host__ __device__ inline int legal_moves_for_color(const Position* s, int color) {
    Move buf[MAX_MOVES];
    Position np = *s;
    if (s->side != color) {
        np.side = (int8_t)color;
        np.ep   = -1;
        if (in_check(&np, np.side)) {
            // Null-move illegal — STM has check on `color`'s king. Cheap proxy:
            // count `color`'s movable pieces × 4. Doesn't perturb material-
            // dominated positions much; only kicks in for in-check leaves.
            int pieces = 0;
            for (int sq = 0; sq < 64; ++sq) {
                int p = s->board[sq];
                if (p != EMPTY && piece_color(p) == color) pieces++;
            }
            return pieces * 4;
        }
    }
    int n_pseudo = generate_moves(&np, buf);
    int legal = 0;
    for (int i = 0; i < n_pseudo; ++i) {
        Position c = np;
        make_move(&c, buf[i]);
        if (!in_check(&c, 1 - c.side)) legal++;
    }
    return legal;
}

// White-POV mobility differential. 2 centipawns per move-of-edge.
// Bounded magnitude: with typical mobility around 30, the term is
// O(60cp) — same order as a positional piece-square value, won't
// swamp material.
constexpr int MOBILITY_WEIGHT_CP = 2;

// =================== device-side __constant__ tables ===================
__constant__ int d_MG_PAWN[64] = {
     0,  0,  0,  0,  0,  0,  0,  0,
    98,134, 61, 95, 68,126, 34,-11,
    -6,  7, 26, 31, 65, 56, 25,-20,
   -14, 13,  6, 21, 23, 12, 17,-23,
   -27, -2, -5, 12, 17,  6, 10,-25,
   -26, -4, -4,-10,  3,  3, 33,-12,
   -35, -1,-20,-23,-15, 24, 38,-22,
     0,  0,  0,  0,  0,  0,  0,  0
};
__constant__ int d_MG_KNIGHT[64] = {
   -167,-89,-34,-49, 61,-97, 50,-73,
    -73,-41, 72, 36, 23, 62,  7,-17,
    -47, 60, 37, 65, 84,129, 73, 44,
     -9, 17, 19, 53, 37, 69, 18, 22,
    -13,  4, 16, 13, 28, 19, 21, -8,
    -23, -9, 12, 10, 19, 17, 25,-16,
    -29,-53,-12, -3, -1, 18,-14,-19,
   -105,-21,-58,-33,-17,-28,-19,-23
};
__constant__ int d_MG_BISHOP[64] = {
    -29,  4,-82,-37,-25,-42,  7, -8,
    -26, 16,-18,-13, 30, 59, 18,-47,
    -16, 37, 43, 40, 35, 50, 37, -2,
     -4,  5, 19, 50, 37, 37,  7, -2,
     -6, 13, 13, 26, 34, 12, 10,  4,
      0, 15, 15, 15, 14, 27, 18, 10,
      4, 15, 16,  0,  7, 21, 33,  1,
    -33, -3,-14,-21,-13,-12,-39,-21
};
__constant__ int d_MG_ROOK[64] = {
     32, 42, 32, 51, 63,  9, 31, 43,
     27, 32, 58, 62, 80, 67, 26, 44,
     -5, 19, 26, 36, 17, 45, 61, 16,
    -24,-11,  7, 26, 24, 35, -8,-20,
    -36,-26,-12, -1,  9, -7,  6,-23,
    -45,-25,-16,-17,  3,  0, -5,-33,
    -44,-16,-20, -9, -1, 11, -6,-71,
    -19,-13,  1, 17, 16,  7,-37,-26
};
__constant__ int d_MG_QUEEN[64] = {
    -28,  0, 29, 12, 59, 44, 43, 45,
    -24,-39, -5,  1,-16, 57, 28, 54,
    -13,-17,  7,  8, 29, 56, 47, 57,
    -27,-27,-16,-16, -1, 17, -2,  1,
     -9,-26, -9,-10, -2, -4,  3, -3,
    -14, -2,-11, -2, -5,  2, 14,  5,
    -35, -8, 11,  2,  8, 15, -3,  1,
     -1,-18, -9, 10,-15,-25,-31,-50
};
__constant__ int d_MG_KING[64] = {
    -65, 23, 16,-15,-56,-34,  2, 13,
     29, -1,-20, -7, -8, -4,-38,-29,
     -9, 24,  2,-16,-20,  6, 22,-22,
    -17,-20,-12,-27,-30,-25,-14,-36,
    -49, -1,-27,-39,-46,-44,-33,-51,
    -14,-14,-22,-46,-44,-30,-15,-27,
      1,  7, -8,-64,-43,-16,  9,  8,
    -15, 36, 12,-54,  8,-28, 24, 14
};
__constant__ int d_EG_PAWN[64] = {
      0,  0,  0,  0,  0,  0,  0,  0,
    178,173,158,134,147,132,165,187,
     94,100, 85, 67, 56, 53, 82, 84,
     32, 24, 13,  5, -2,  4, 17, 17,
     13,  9, -3, -7, -7, -8,  3, -1,
      4,  7, -6,  1,  0, -5, -1, -8,
     13,  8,  8, 10, 13,  0,  2, -7,
      0,  0,  0,  0,  0,  0,  0,  0
};
__constant__ int d_EG_KNIGHT[64] = {
    -58,-38,-13,-28,-31,-27,-63,-99,
    -25, -8,-25, -2, -9,-25,-24,-52,
    -24,-20, 10,  9, -1, -9,-19,-41,
    -17,  3, 22, 22, 22, 11,  8,-18,
    -18, -6, 16, 25, 16, 17,  4,-18,
    -23, -3, -1, 15, 10, -3,-20,-22,
    -42,-20,-10, -5, -2,-20,-23,-44,
    -29,-51,-23,-15,-22,-18,-50,-64
};
__constant__ int d_EG_BISHOP[64] = {
    -14,-21,-11, -8, -7, -9,-17,-24,
     -8, -4,  7,-12, -3,-13, -4,-14,
      2, -8,  0, -1, -2,  6,  0,  4,
     -3,  9, 12,  9, 14, 10,  3,  2,
     -6,  3, 13, 19,  7, 10, -3, -9,
    -12, -3,  8, 10, 13,  3, -7,-15,
    -14,-18, -7, -1,  4, -9,-15,-27,
    -23, -9,-23, -5, -9,-16, -5,-17
};
__constant__ int d_EG_ROOK[64] = {
     13, 10, 18, 15, 12, 12,  8,  5,
     11, 13, 13, 11, -3,  7,  7,  8,
      7,  7,  7,  5,  4, -3, -5,  3,
      4,  3, 13,  1,  2,  1, -1,  2,
      3,  5,  8,  4, -5, -6, -8,-11,
     -4,  0, -5, -1, -7,-12, -8,-16,
     -6, -6,  0,  2, -9, -9,-11, -3,
     -9,  2,  3, -1, -5,-13,  4,-20
};
__constant__ int d_EG_QUEEN[64] = {
     -9, 22, 22, 27, 27, 19, 10, 20,
    -17, 20, 32, 41, 58, 25, 30,  0,
    -20,  6,  9, 49, 47, 35, 19,  9,
      3, 22, 24, 45, 57, 40, 57, 36,
    -18, 28, 19, 47, 31, 34, 39, 23,
    -16,-27, 15,  6,  9, 17, 10,  5,
    -22,-23,-30,-16,-16,-23,-36,-32,
    -33,-28,-22,-43, -5,-32,-20,-41
};
__constant__ int d_EG_KING[64] = {
    -74,-35,-18,-18,-11, 15,  4,-17,
    -12, 17, 14, 17, 17, 38, 23, 11,
     10, 17, 23, 15, 20, 45, 44, 13,
     -8, 22, 24, 27, 26, 33, 26,  3,
    -18, -4, 21, 24, 27, 23,  9,-11,
    -19, -3, 11, 21, 23, 16,  7, -9,
    -27,-11,  4, 13, 14,  4, -5,-17,
    -53,-34,-21,-11,-28,-14,-24,-43
};
__constant__ int d_MG_VALS[7] = {0, 82, 337, 365, 477, 1025, 0};
__constant__ int d_EG_VALS[7] = {0, 94, 281, 297, 512, 936, 0};
__constant__ int d_PHASE_W[7] = {0, 0, 1, 1, 2, 4, 0};

// =================== host-side mirror tables (identical numbers) ===================
static const int h_MG_PAWN[64] = {
     0,  0,  0,  0,  0,  0,  0,  0,
    98,134, 61, 95, 68,126, 34,-11,
    -6,  7, 26, 31, 65, 56, 25,-20,
   -14, 13,  6, 21, 23, 12, 17,-23,
   -27, -2, -5, 12, 17,  6, 10,-25,
   -26, -4, -4,-10,  3,  3, 33,-12,
   -35, -1,-20,-23,-15, 24, 38,-22,
     0,  0,  0,  0,  0,  0,  0,  0
};
static const int h_MG_KNIGHT[64] = {
   -167,-89,-34,-49, 61,-97, 50,-73,
    -73,-41, 72, 36, 23, 62,  7,-17,
    -47, 60, 37, 65, 84,129, 73, 44,
     -9, 17, 19, 53, 37, 69, 18, 22,
    -13,  4, 16, 13, 28, 19, 21, -8,
    -23, -9, 12, 10, 19, 17, 25,-16,
    -29,-53,-12, -3, -1, 18,-14,-19,
   -105,-21,-58,-33,-17,-28,-19,-23
};
static const int h_MG_BISHOP[64] = {
    -29,  4,-82,-37,-25,-42,  7, -8,
    -26, 16,-18,-13, 30, 59, 18,-47,
    -16, 37, 43, 40, 35, 50, 37, -2,
     -4,  5, 19, 50, 37, 37,  7, -2,
     -6, 13, 13, 26, 34, 12, 10,  4,
      0, 15, 15, 15, 14, 27, 18, 10,
      4, 15, 16,  0,  7, 21, 33,  1,
    -33, -3,-14,-21,-13,-12,-39,-21
};
static const int h_MG_ROOK[64] = {
     32, 42, 32, 51, 63,  9, 31, 43,
     27, 32, 58, 62, 80, 67, 26, 44,
     -5, 19, 26, 36, 17, 45, 61, 16,
    -24,-11,  7, 26, 24, 35, -8,-20,
    -36,-26,-12, -1,  9, -7,  6,-23,
    -45,-25,-16,-17,  3,  0, -5,-33,
    -44,-16,-20, -9, -1, 11, -6,-71,
    -19,-13,  1, 17, 16,  7,-37,-26
};
static const int h_MG_QUEEN[64] = {
    -28,  0, 29, 12, 59, 44, 43, 45,
    -24,-39, -5,  1,-16, 57, 28, 54,
    -13,-17,  7,  8, 29, 56, 47, 57,
    -27,-27,-16,-16, -1, 17, -2,  1,
     -9,-26, -9,-10, -2, -4,  3, -3,
    -14, -2,-11, -2, -5,  2, 14,  5,
    -35, -8, 11,  2,  8, 15, -3,  1,
     -1,-18, -9, 10,-15,-25,-31,-50
};
static const int h_MG_KING[64] = {
    -65, 23, 16,-15,-56,-34,  2, 13,
     29, -1,-20, -7, -8, -4,-38,-29,
     -9, 24,  2,-16,-20,  6, 22,-22,
    -17,-20,-12,-27,-30,-25,-14,-36,
    -49, -1,-27,-39,-46,-44,-33,-51,
    -14,-14,-22,-46,-44,-30,-15,-27,
      1,  7, -8,-64,-43,-16,  9,  8,
    -15, 36, 12,-54,  8,-28, 24, 14
};
static const int h_EG_PAWN[64] = {
      0,  0,  0,  0,  0,  0,  0,  0,
    178,173,158,134,147,132,165,187,
     94,100, 85, 67, 56, 53, 82, 84,
     32, 24, 13,  5, -2,  4, 17, 17,
     13,  9, -3, -7, -7, -8,  3, -1,
      4,  7, -6,  1,  0, -5, -1, -8,
     13,  8,  8, 10, 13,  0,  2, -7,
      0,  0,  0,  0,  0,  0,  0,  0
};
static const int h_EG_KNIGHT[64] = {
    -58,-38,-13,-28,-31,-27,-63,-99,
    -25, -8,-25, -2, -9,-25,-24,-52,
    -24,-20, 10,  9, -1, -9,-19,-41,
    -17,  3, 22, 22, 22, 11,  8,-18,
    -18, -6, 16, 25, 16, 17,  4,-18,
    -23, -3, -1, 15, 10, -3,-20,-22,
    -42,-20,-10, -5, -2,-20,-23,-44,
    -29,-51,-23,-15,-22,-18,-50,-64
};
static const int h_EG_BISHOP[64] = {
    -14,-21,-11, -8, -7, -9,-17,-24,
     -8, -4,  7,-12, -3,-13, -4,-14,
      2, -8,  0, -1, -2,  6,  0,  4,
     -3,  9, 12,  9, 14, 10,  3,  2,
     -6,  3, 13, 19,  7, 10, -3, -9,
    -12, -3,  8, 10, 13,  3, -7,-15,
    -14,-18, -7, -1,  4, -9,-15,-27,
    -23, -9,-23, -5, -9,-16, -5,-17
};
static const int h_EG_ROOK[64] = {
     13, 10, 18, 15, 12, 12,  8,  5,
     11, 13, 13, 11, -3,  7,  7,  8,
      7,  7,  7,  5,  4, -3, -5,  3,
      4,  3, 13,  1,  2,  1, -1,  2,
      3,  5,  8,  4, -5, -6, -8,-11,
     -4,  0, -5, -1, -7,-12, -8,-16,
     -6, -6,  0,  2, -9, -9,-11, -3,
     -9,  2,  3, -1, -5,-13,  4,-20
};
static const int h_EG_QUEEN[64] = {
     -9, 22, 22, 27, 27, 19, 10, 20,
    -17, 20, 32, 41, 58, 25, 30,  0,
    -20,  6,  9, 49, 47, 35, 19,  9,
      3, 22, 24, 45, 57, 40, 57, 36,
    -18, 28, 19, 47, 31, 34, 39, 23,
    -16,-27, 15,  6,  9, 17, 10,  5,
    -22,-23,-30,-16,-16,-23,-36,-32,
    -33,-28,-22,-43, -5,-32,-20,-41
};
static const int h_EG_KING[64] = {
    -74,-35,-18,-18,-11, 15,  4,-17,
    -12, 17, 14, 17, 17, 38, 23, 11,
     10, 17, 23, 15, 20, 45, 44, 13,
     -8, 22, 24, 27, 26, 33, 26,  3,
    -18, -4, 21, 24, 27, 23,  9,-11,
    -19, -3, 11, 21, 23, 16,  7, -9,
    -27,-11,  4, 13, 14,  4, -5,-17,
    -53,-34,-21,-11,-28,-14,-24,-43
};
static const int h_MG_VALS[7] = {0, 82, 337, 365, 477, 1025, 0};
static const int h_EG_VALS[7] = {0, 94, 281, 297, 512, 936, 0};
static const int h_PHASE_W[7] = {0, 0, 1, 1, 2, 4, 0};

// ---------- device evaluator (extracted from gpu_fighter d_evaluate) ----------
__device__ Score d_evaluate(const Position* s) {
    int mg = 0, eg = 0, phase = 0;
    for (int sq = 0; sq < 64; sq++) {
        int p = s->board[sq];
        if (p == EMPTY) continue;
        int iw  = is_white(p);
        int pt  = iw ? p : p - 6;
        int psq = iw ? sq : mirror_sq(sq);
        int mv  = d_MG_VALS[pt], ev = d_EG_VALS[pt];
        int mpst = 0, epst = 0;
        switch (pt) {
            case 1: mpst = d_MG_PAWN[psq];   epst = d_EG_PAWN[psq];   break;
            case 2: mpst = d_MG_KNIGHT[psq]; epst = d_EG_KNIGHT[psq]; break;
            case 3: mpst = d_MG_BISHOP[psq]; epst = d_EG_BISHOP[psq]; break;
            case 4: mpst = d_MG_ROOK[psq];   epst = d_EG_ROOK[psq];   break;
            case 5: mpst = d_MG_QUEEN[psq];  epst = d_EG_QUEEN[psq];  break;
            case 6: mpst = d_MG_KING[psq];   epst = d_EG_KING[psq];   break;
        }
        int sign = iw ? 1 : -1;
        mg += sign * (mv + mpst);
        eg += sign * (ev + epst);
        phase += d_PHASE_W[pt];
    }
    if (phase > 24) phase = 24;
    int score = (mg * phase + eg * (24 - phase)) / 24;
    return (s->side == WHITE_SIDE) ? score : -score;
}

// ---------- host evaluator (numerically identical) ----------
Score h_evaluate(const Position* s) {
    int mg = 0, eg = 0, phase = 0;
    for (int sq = 0; sq < 64; sq++) {
        int p = s->board[sq];
        if (p == EMPTY) continue;
        int iw  = is_white(p);
        int pt  = iw ? p : p - 6;
        int psq = iw ? sq : mirror_sq(sq);
        int mv  = h_MG_VALS[pt], ev = h_EG_VALS[pt];
        int mpst = 0, epst = 0;
        switch (pt) {
            case 1: mpst = h_MG_PAWN[psq];   epst = h_EG_PAWN[psq];   break;
            case 2: mpst = h_MG_KNIGHT[psq]; epst = h_EG_KNIGHT[psq]; break;
            case 3: mpst = h_MG_BISHOP[psq]; epst = h_EG_BISHOP[psq]; break;
            case 4: mpst = h_MG_ROOK[psq];   epst = h_EG_ROOK[psq];   break;
            case 5: mpst = h_MG_QUEEN[psq];  epst = h_EG_QUEEN[psq];  break;
            case 6: mpst = h_MG_KING[psq];   epst = h_EG_KING[psq];   break;
        }
        int sign = iw ? 1 : -1;
        mg += sign * (mv + mpst);
        eg += sign * (ev + epst);
        phase += h_PHASE_W[pt];
    }
    if (phase > 24) phase = 24;
    int score = (mg * phase + eg * (24 - phase)) / 24;
    return (s->side == WHITE_SIDE) ? score : -score;
}

} // namespace engine
