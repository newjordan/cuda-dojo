// =============================================================================
// coord3d.cuh
//
// GPU port of relabeler/scripts/compute_3d_octant_coords.py. Computes the
// continuous (x, y, z) ∈ ℝ³ coordinates from a Position. White's POV.
//
//   y = 0.4·material/1000 + 0.3·(KS_w−KS_b) + 0.2·(ST_w−ST_b) + 0.1·mob_diff
//       (clipped to [-0.5, 0.5])
//   z = (non_pawn_material − 3500) / 3500          (signed; phase coordinate)
//   x = clustering(white) − clustering(black)      (signed; cluster diff)
//
// Same chess principle the Python version captures: outcome-gradient (y),
// game-phase (z), piece-clustering asymmetry (x). What a strong human player
// reads off the position at a glance, made into a 3-vector.
// =============================================================================
#ifndef ENGINE_COORD3D_CUH
#define ENGINE_COORD3D_CUH

#include "engine_types.h"
#include "attacks.cuh"
#include "movegen.cuh"
#include "make_unmake.cuh"

namespace engine {

struct Coord3D {
    float x;
    float y;
    float z;
    int   octant_id;  // (x<0)·1 + (y<0)·2 + (z<0)·4
};

// Centipawn piece values, mirrors PIECE_VALUES from the Python.
__host__ __device__ inline int piece_value_cp(int piece_type_1to6) {
    switch (piece_type_1to6) {
        case 1: return 100;  // pawn
        case 2: return 320;  // knight
        case 3: return 330;  // bishop
        case 4: return 500;  // rook
        case 5: return 900;  // queen
        default: return 0;   // king
    }
}

// White-POV centipawn material balance.
__host__ __device__ inline int material_balance_white_pov(const Position* s) {
    int total = 0;
    for (int sq = 0; sq < 64; ++sq) {
        int p = s->board[sq];
        if (p == EMPTY) continue;
        int t = piece_type(p);   // 1..6 regardless of color
        int v = piece_value_cp(t);
        if (is_white(p)) total += v;
        else total -= v;
    }
    return total;
}

// Sum of non-pawn material on the board (both sides). For phase z.
__host__ __device__ inline int non_pawn_material(const Position* s) {
    int total = 0;
    for (int sq = 0; sq < 64; ++sq) {
        int p = s->board[sq];
        if (p == EMPTY) continue;
        int t = piece_type(p);
        if (t == 1 || t == 6) continue;   // skip pawn + king
        total += piece_value_cp(t);
    }
    return total;
}

// Squares within Chebyshev distance ≤ 2 of king_sq. Returns count and
// fills `zone_sqs` (caller supplies array of size ≥ 25).
__host__ __device__ inline int king_zone(int king_sq, int* zone_sqs) {
    if (king_sq < 0 || king_sq > 63) return 0;
    int kf = king_sq & 7;
    int kr = king_sq >> 3;
    int n = 0;
    for (int dr = -2; dr <= 2; ++dr) {
        for (int df = -2; df <= 2; ++df) {
            int f = kf + df;
            int r = kr + dr;
            if (f < 0 || f > 7 || r < 0 || r > 7) continue;
            zone_sqs[n++] = r * 8 + f;
        }
    }
    return n;
}

// King safety [0, 1] for `color`. Mirrors the Python formula.
__host__ __device__ inline float king_safety_score(const Position* s, int color) {
    int king = s->kingPos[color];
    if (king < 0 || king > 63) return 0.0f;
    int zone[25];
    int n_zone = king_zone(king, zone);

    int own_def = 0;
    int opp_atk = 0;
    int escape = 0;
    int kf = king & 7;
    int kr = king >> 3;
    int opp = 1 - color;

    for (int i = 0; i < n_zone; ++i) {
        int sq = zone[i];
        int p = s->board[sq];
        // Own defenders (non-king pieces of own color).
        if (p != EMPTY && piece_color(p) == color && piece_type(p) != 6) {
            own_def++;
        }
        // Opp attackers covering this zone square.
        if (is_square_attacked(s, sq, opp)) {
            opp_atk++;
        }
        // Escape squares: adjacent to king, not own piece, not attacked.
        if (sq == king) continue;
        if (p != EMPTY && piece_color(p) == color) continue;
        int sf = sq & 7;
        int sr = sq >> 3;
        int df = sf - kf; if (df < 0) df = -df;
        int dr = sr - kr; if (dr < 0) dr = -dr;
        int cheby = df > dr ? df : dr;
        if (cheby == 1 && !is_square_attacked(s, sq, opp)) {
            escape++;
        }
    }

    float own_d = (own_def > 6 ? 6 : own_def) / 6.0f;
    int opa = opp_atk > 8 ? 8 : opp_atk;
    float opp_a = 1.0f - opa / 8.0f;
    int esc = escape > 4 ? 4 : escape;
    float esc_n = esc / 4.0f;
    float safety = 0.4f * own_d + 0.3f * opp_a + 0.3f * esc_n;
    if (safety < 0.0f) safety = 0.0f;
    if (safety > 1.0f) safety = 1.0f;
    return safety;
}

// Pawn structure score [0, 1] for `color`. Penalize doubled and isolated.
__host__ __device__ inline float structural_score(const Position* s, int color) {
    int pawn_code = (color == WHITE_SIDE) ? WPAWN : BPAWN;
    int files[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    int total = 0;
    for (int sq = 0; sq < 64; ++sq) {
        if (s->board[sq] == pawn_code) {
            files[sq & 7]++;
            total++;
        }
    }
    if (total == 0) return 0.5f;
    int doubled = 0, isolated = 0;
    for (int f = 0; f < 8; ++f) {
        if (files[f] >= 2) doubled++;
        if (files[f] > 0) {
            bool left_empty  = (f == 0) || (files[f - 1] == 0);
            bool right_empty = (f == 7) || (files[f + 1] == 0);
            if (left_empty && right_empty) isolated++;
        }
    }
    float weak_ratio = float(doubled + isolated) / float(total);
    float s_score = 1.0f - weak_ratio;
    if (s_score < 0.0f) s_score = 0.0f;
    if (s_score > 1.0f) s_score = 1.0f;
    return s_score;
}

// Legal-move count for `color`. If side-to-move != color, use null-move
// trick (flip side, clear ep) to invert.
__host__ __device__ inline int mobility_count(const Position* s, int color) {
    Move buf[MAX_MOVES];
    if (s->side == color) {
        Position child = *s;
        int n_pseudo = generate_moves(&child, buf);
        int legal = 0;
        for (int i = 0; i < n_pseudo; ++i) {
            Position c2 = child;
            make_move(&c2, buf[i]);
            if (!in_check(&c2, 1 - c2.side)) legal++;
        }
        return legal;
    }
    // Null-move: flip side and clear ep.
    Position null_pos = *s;
    null_pos.side = (int8_t)(1 - null_pos.side);
    null_pos.ep = -1;
    // If the original side-to-move's opponent is in check, null-move is
    // illegal; fall back to coarse estimate (piece count × 4).
    if (in_check(&null_pos, null_pos.side)) {
        int count = 0;
        for (int sq = 0; sq < 64; ++sq) {
            int p = s->board[sq];
            if (p != EMPTY && piece_color(p) == color) count++;
        }
        return count * 4;
    }
    int n_pseudo = generate_moves(&null_pos, buf);
    int legal = 0;
    for (int i = 0; i < n_pseudo; ++i) {
        Position c2 = null_pos;
        make_move(&c2, buf[i]);
        if (!in_check(&c2, 1 - c2.side)) legal++;
    }
    return legal;
}

// Clustering: mean inverse Chebyshev distance to own piece centroid.
__host__ __device__ inline float piece_clustering(const Position* s, int color) {
    int sqs[16];   // max 16 own pieces in standard chess
    int n = 0;
    for (int sq = 0; sq < 64; ++sq) {
        int p = s->board[sq];
        if (p != EMPTY && piece_color(p) == color) {
            if (n < 16) sqs[n++] = sq;
        }
    }
    if (n < 2) return 0.0f;
    float sf = 0.0f, sr = 0.0f;
    for (int i = 0; i < n; ++i) {
        sf += (sqs[i] & 7);
        sr += (sqs[i] >> 3);
    }
    sf /= n;
    sr /= n;
    float total_inv = 0.0f;
    for (int i = 0; i < n; ++i) {
        float f = float(sqs[i] & 7);
        float r = float(sqs[i] >> 3);
        float dfx = f - sf; if (dfx < 0) dfx = -dfx;
        float drx = r - sr; if (drx < 0) drx = -drx;
        float d = dfx > drx ? dfx : drx;
        total_inv += 1.0f / (d + 1.0f);
    }
    return total_inv / float(n);
}

// Full coord computation. Equivalent to compute_coordinates(fen) in
// compute_3d_octant_coords.py.
__host__ __device__ inline Coord3D compute_coord3d(const Position* s) {
    int material = material_balance_white_pov(s);
    float y_material = 0.4f * (material / 1000.0f);

    float ks_w = king_safety_score(s, WHITE_SIDE);
    float ks_b = king_safety_score(s, BLACK_SIDE);
    float y_king_safety = 0.3f * (ks_w - ks_b);

    float st_w = structural_score(s, WHITE_SIDE);
    float st_b = structural_score(s, BLACK_SIDE);
    float y_structure = 0.2f * (st_w - st_b);

    int mob_w = mobility_count(s, WHITE_SIDE);
    int mob_b = mobility_count(s, BLACK_SIDE);
    int mob_total = mob_w + mob_b;
    if (mob_total < 1) mob_total = 1;
    float y_mobility = 0.1f * (float(mob_w - mob_b) / float(mob_total));

    float y_raw = y_material + y_king_safety + y_structure + y_mobility;
    float y = y_raw;
    if (y < -0.5f) y = -0.5f;
    if (y >  0.5f) y =  0.5f;

    int npm = non_pawn_material(s);
    float z = float(npm - 3500) / 3500.0f;

    float own_clust = piece_clustering(s, WHITE_SIDE);
    float opp_clust = piece_clustering(s, BLACK_SIDE);
    float x = own_clust - opp_clust;

    int oct = (x < 0 ? 1 : 0) + (y < 0 ? 2 : 0) + (z < 0 ? 4 : 0);

    Coord3D c;
    c.x = x; c.y = y; c.z = z; c.octant_id = oct;
    return c;
}

} // namespace engine

#endif // ENGINE_COORD3D_CUH
