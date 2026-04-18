// =============================================================================
// test_zobrist.cu
//
// Verify that incremental Zobrist update equals full recomputation across a
// batch of random self-play game prefixes.
//
// For N=1000 random walks: from the standard start position, hash with
// zobrist_full(start). Then for each move, apply zobrist_update to fold
// the move into the running hash AND apply the move to the position via
// the existing gpu_fighter make_move logic. After every ply, check that
// zobrist_full(pos) == running_hash. Any mismatch is reported and counted
// as a failure.
//
// Move legality is taken from a host-side pseudo-legal generator that
// mirrors d_make_move (we don't need full check/legality pruning for the
// hash test; we only need to make sure the moves we generate hit every
// branch -- captures, double pawn pushes, EP, promo, castling).
// =============================================================================
#include "../zobrist.cuh"
#include "../engine_types.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>

using namespace engine;

// ---------- minimal host-side movegen + make_move ----------
//
// These mirror d_make_move from gpu_fighter.cu byte-for-byte. We need them
// to be host-callable so the test can run without a kernel launch.

static inline bool h_onb(int r, int f) { return r >= 0 && r < 8 && f >= 0 && f < 8; }

static const int H_KN[8] = {-17,-15,-10,-6, 6,10,15,17};
static const int H_KG[8] = { -9, -8, -7,-1, 1, 7, 8, 9};

static bool h_is_sq_attacked(const Position* s, int sq, int by) {
    int r = sq_rank(sq), f = sq_file(sq);
    if (by == WHITE_SIDE) {
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
    int np = (by == WHITE_SIDE) ? WKNIGHT : BKNIGHT;
    for (int i = 0; i < 8; i++) {
        int t = sq + H_KN[i];
        if (t < 0 || t >= 64) continue;
        int dr = abs(sq_rank(t) - r), df = abs(sq_file(t) - f);
        if ((dr == 2 && df == 1) || (dr == 1 && df == 2)) {
            if (s->board[t] == np) return true;
        }
    }
    int kp = (by == WHITE_SIDE) ? WKING : BKING;
    for (int i = 0; i < 8; i++) {
        int t = sq + H_KG[i];
        if (t < 0 || t >= 64) continue;
        int dr = abs(sq_rank(t) - r), df = abs(sq_file(t) - f);
        if (dr <= 1 && df <= 1) {
            if (s->board[t] == kp) return true;
        }
    }
    int bp = (by == WHITE_SIDE) ? WBISHOP : BBISHOP;
    int qp = (by == WHITE_SIDE) ? WQUEEN  : BQUEEN;
    int ddr[] = {-1,-1, 1, 1}, ddf[] = {-1, 1,-1, 1};
    for (int d = 0; d < 4; d++) {
        int cr = r + ddr[d], cf = f + ddf[d];
        while (h_onb(cr, cf)) {
            int p = s->board[make_sq(cr, cf)];
            if (p != EMPTY) { if (p == bp || p == qp) return true; break; }
            cr += ddr[d]; cf += ddf[d];
        }
    }
    int rp = (by == WHITE_SIDE) ? WROOK : BROOK;
    int rdr[] = {-1, 1, 0, 0}, rdf[] = { 0, 0,-1, 1};
    for (int d = 0; d < 4; d++) {
        int cr = r + rdr[d], cf = f + rdf[d];
        while (h_onb(cr, cf)) {
            int p = s->board[make_sq(cr, cf)];
            if (p != EMPTY) { if (p == rp || p == qp) return true; break; }
            cr += rdr[d]; cf += rdf[d];
        }
    }
    return false;
}

// Generate pseudo-legal moves for side-to-move. Returns count.
// (Pseudo-legal: no king-in-check filter. We discard any move that leaves
// our own king in check after make_move.)
static int h_gen_moves(const Position* s, Move* out) {
    int count = 0;
    int side = s->side;
    int opp  = 1 - side;
    for (int sq = 0; sq < 64; sq++) {
        int piece = s->board[sq];
        if (piece == EMPTY) continue;
        if (piece_color(piece) != side) continue;
        int pt = piece_type(piece);
        int r = sq_rank(sq), f = sq_file(sq);

        if (pt == 1) { // pawn
            int dir = (side == WHITE_SIDE) ? -1 : 1;
            int promoRank = (side == WHITE_SIDE) ? 0 : 7;
            int startRank = (side == WHITE_SIDE) ? 6 : 1;
            int nr = r + dir;
            if (h_onb(nr, f) && s->board[make_sq(nr, f)] == EMPTY) {
                if (nr == promoRank) {
                    int qb = (side == WHITE_SIDE) ? WQUEEN  : BQUEEN;
                    int rb = (side == WHITE_SIDE) ? WROOK   : BROOK;
                    int bb = (side == WHITE_SIDE) ? WBISHOP : BBISHOP;
                    int nb = (side == WHITE_SIDE) ? WKNIGHT : BKNIGHT;
                    out[count++] = make_move(sq, make_sq(nr, f), qb, FLAG_PROMO);
                    out[count++] = make_move(sq, make_sq(nr, f), rb, FLAG_PROMO);
                    out[count++] = make_move(sq, make_sq(nr, f), bb, FLAG_PROMO);
                    out[count++] = make_move(sq, make_sq(nr, f), nb, FLAG_PROMO);
                } else {
                    out[count++] = make_move(sq, make_sq(nr, f), 0, FLAG_NONE);
                    if (r == startRank) {
                        int nr2 = r + 2 * dir;
                        if (s->board[make_sq(nr2, f)] == EMPTY) {
                            out[count++] = make_move(sq, make_sq(nr2, f), 0, FLAG_DOUBLE);
                        }
                    }
                }
            }
            for (int df = -1; df <= 1; df += 2) {
                int nf = f + df;
                if (!h_onb(nr, nf)) continue;
                int t = make_sq(nr, nf);
                int tp = s->board[t];
                if (tp != EMPTY && piece_color(tp) == opp) {
                    if (nr == promoRank) {
                        int qb = (side == WHITE_SIDE) ? WQUEEN  : BQUEEN;
                        int rb = (side == WHITE_SIDE) ? WROOK   : BROOK;
                        int bb = (side == WHITE_SIDE) ? WBISHOP : BBISHOP;
                        int nb = (side == WHITE_SIDE) ? WKNIGHT : BKNIGHT;
                        out[count++] = make_move(sq, t, qb, FLAG_PROMO);
                        out[count++] = make_move(sq, t, rb, FLAG_PROMO);
                        out[count++] = make_move(sq, t, bb, FLAG_PROMO);
                        out[count++] = make_move(sq, t, nb, FLAG_PROMO);
                    } else {
                        out[count++] = make_move(sq, t, 0, FLAG_CAPTURE);
                    }
                }
                if (s->ep == t && tp == EMPTY) {
                    out[count++] = make_move(sq, t, 0, FLAG_EP);
                }
            }
        } else if (pt == 2) { // knight
            for (int i = 0; i < 8; i++) {
                int t = sq + H_KN[i];
                if (t < 0 || t >= 64) continue;
                int dr = abs(sq_rank(t) - r), df = abs(sq_file(t) - f);
                if (!((dr == 2 && df == 1) || (dr == 1 && df == 2))) continue;
                int tp = s->board[t];
                if (tp == EMPTY) out[count++] = make_move(sq, t, 0, FLAG_NONE);
                else if (piece_color(tp) == opp) out[count++] = make_move(sq, t, 0, FLAG_CAPTURE);
            }
        } else if (pt == 3 || pt == 5) { // bishop / queen-diagonals
            int ddr[] = {-1,-1,1,1}, ddf[] = {-1,1,-1,1};
            for (int d = 0; d < 4; d++) {
                int cr = r + ddr[d], cf = f + ddf[d];
                while (h_onb(cr, cf)) {
                    int t = make_sq(cr, cf); int tp = s->board[t];
                    if (tp == EMPTY) out[count++] = make_move(sq, t, 0, FLAG_NONE);
                    else { if (piece_color(tp) == opp) out[count++] = make_move(sq, t, 0, FLAG_CAPTURE); break; }
                    cr += ddr[d]; cf += ddf[d];
                }
            }
        }
        if (pt == 4 || pt == 5) { // rook / queen-orthogonal
            int ddr[] = {-1,1,0,0}, ddf[] = {0,0,-1,1};
            for (int d = 0; d < 4; d++) {
                int cr = r + ddr[d], cf = f + ddf[d];
                while (h_onb(cr, cf)) {
                    int t = make_sq(cr, cf); int tp = s->board[t];
                    if (tp == EMPTY) out[count++] = make_move(sq, t, 0, FLAG_NONE);
                    else { if (piece_color(tp) == opp) out[count++] = make_move(sq, t, 0, FLAG_CAPTURE); break; }
                    cr += ddr[d]; cf += ddf[d];
                }
            }
        }
        if (pt == 6) { // king
            for (int i = 0; i < 8; i++) {
                int t = sq + H_KG[i];
                if (t < 0 || t >= 64) continue;
                int dr = abs(sq_rank(t) - r), df = abs(sq_file(t) - f);
                if (dr > 1 || df > 1) continue;
                int tp = s->board[t];
                if (tp == EMPTY) out[count++] = make_move(sq, t, 0, FLAG_NONE);
                else if (piece_color(tp) == opp) out[count++] = make_move(sq, t, 0, FLAG_CAPTURE);
            }
            // Castling.
            if (side == WHITE_SIDE) {
                if ((s->castle & CASTLE_WK) && s->board[61]==EMPTY && s->board[62]==EMPTY && s->board[63]==WROOK
                    && !h_is_sq_attacked(s,60,opp) && !h_is_sq_attacked(s,61,opp) && !h_is_sq_attacked(s,62,opp))
                    out[count++] = make_move(60, 62, 0, FLAG_CASTLE_K);
                if ((s->castle & CASTLE_WQ) && s->board[59]==EMPTY && s->board[58]==EMPTY && s->board[57]==EMPTY && s->board[56]==WROOK
                    && !h_is_sq_attacked(s,60,opp) && !h_is_sq_attacked(s,59,opp) && !h_is_sq_attacked(s,58,opp))
                    out[count++] = make_move(60, 58, 0, FLAG_CASTLE_Q);
            } else {
                if ((s->castle & CASTLE_BK) && s->board[5]==EMPTY && s->board[6]==EMPTY && s->board[7]==BROOK
                    && !h_is_sq_attacked(s,4,opp) && !h_is_sq_attacked(s,5,opp) && !h_is_sq_attacked(s,6,opp))
                    out[count++] = make_move(4, 6, 0, FLAG_CASTLE_K);
                if ((s->castle & CASTLE_BQ) && s->board[3]==EMPTY && s->board[2]==EMPTY && s->board[1]==EMPTY && s->board[0]==BROOK
                    && !h_is_sq_attacked(s,4,opp) && !h_is_sq_attacked(s,3,opp) && !h_is_sq_attacked(s,2,opp))
                    out[count++] = make_move(4, 2, 0, FLAG_CASTLE_Q);
            }
        }
    }
    return count;
}

// Apply move to position. Mirrors d_make_move from gpu_fighter.cu.
static void h_make_move(Position* s, Move move) {
    int from = move_from(move), to = move_to(move);
    int flags = move_flags(move), promo = move_promo(move);
    int piece = s->board[from];
    int side  = s->side;
    s->board[to] = piece;
    s->board[from] = EMPTY;
    s->halfmove++;
    if (piece_type(piece) == 6) s->kingPos[side] = to;
    if (piece_type(piece) == 1) {
        s->halfmove = 0;
        if (flags == FLAG_DOUBLE) s->ep = (from + to) / 2;
        else if (flags == FLAG_EP) {
            int cs = (side == WHITE_SIDE) ? to + 8 : to - 8;
            s->board[cs] = EMPTY; s->ep = -1;
        } else if (flags == FLAG_PROMO) { s->board[to] = promo; s->ep = -1; }
        else s->ep = -1;
    } else s->ep = -1;
    if (flags == FLAG_CAPTURE) s->halfmove = 0;
    if (flags == FLAG_CASTLE_K) {
        if (side == WHITE_SIDE) { s->board[63]=EMPTY; s->board[61]=WROOK; }
        else                    { s->board[7]=EMPTY;  s->board[5]=BROOK; }
    }
    if (flags == FLAG_CASTLE_Q) {
        if (side == WHITE_SIDE) { s->board[56]=EMPTY; s->board[59]=WROOK; }
        else                    { s->board[0]=EMPTY;  s->board[3]=BROOK; }
    }
    if (piece == WKING) s->castle &= ~(CASTLE_WK|CASTLE_WQ);
    if (piece == BKING) s->castle &= ~(CASTLE_BK|CASTLE_BQ);
    if (from == 63 || to == 63) s->castle &= ~CASTLE_WK;
    if (from == 56 || to == 56) s->castle &= ~CASTLE_WQ;
    if (from == 7  || to == 7 ) s->castle &= ~CASTLE_BK;
    if (from == 0  || to == 0 ) s->castle &= ~CASTLE_BQ;
    s->side = 1 - side;
    if (side == BLACK_SIDE) s->fullmove++;
}

static void start_position(Position* s) {
    memset(s, 0, sizeof(Position));
    for (int i = 0; i < 64; i++) s->board[i] = EMPTY;
    s->ep = -1;
    // Black back rank (rank 0 = top).
    s->board[0]=BROOK; s->board[1]=BKNIGHT; s->board[2]=BBISHOP; s->board[3]=BQUEEN;
    s->board[4]=BKING; s->board[5]=BBISHOP; s->board[6]=BKNIGHT; s->board[7]=BROOK;
    for (int f = 0; f < 8; f++) s->board[8 + f] = BPAWN;
    for (int f = 0; f < 8; f++) s->board[48 + f] = WPAWN;
    s->board[56]=WROOK; s->board[57]=WKNIGHT; s->board[58]=WBISHOP; s->board[59]=WQUEEN;
    s->board[60]=WKING; s->board[61]=WBISHOP; s->board[62]=WKNIGHT; s->board[63]=WROOK;
    s->kingPos[WHITE_SIDE] = 60;
    s->kingPos[BLACK_SIDE] = 4;
    s->castle = CASTLE_WK | CASTLE_WQ | CASTLE_BK | CASTLE_BQ;
    s->side = WHITE_SIDE;
    s->halfmove = 0;
    s->fullmove = 1;
}

// xorshift64 -- tiny PRNG so the test is deterministic.
static uint64_t rngs = 0xDEADBEEFCAFEBABEULL;
static uint64_t xrng() {
    uint64_t x = rngs;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    rngs = x;
    return x;
}

// Pick a random pseudo-legal move whose result keeps our king safe.
// Returns false if no legal move exists (game over).
static bool pick_legal(const Position* s, Move* out) {
    Move buf[256];
    int n = h_gen_moves(s, buf);
    if (n == 0) return false;
    // Shuffle indices and try until one is legal.
    int order[256];
    for (int i = 0; i < n; i++) order[i] = i;
    for (int i = n - 1; i > 0; i--) {
        int j = (int)(xrng() % (uint64_t)(i + 1));
        int t = order[i]; order[i] = order[j]; order[j] = t;
    }
    int side = s->side;
    for (int i = 0; i < n; i++) {
        Position trial = *s;
        h_make_move(&trial, buf[order[i]]);
        int kp = trial.kingPos[side];
        if (kp < 0) continue;
        if (!h_is_sq_attacked(&trial, kp, 1 - side)) {
            *out = buf[order[i]];
            return true;
        }
    }
    return false;
}

int main(int argc, char** argv) {
    int n_games = (argc > 1) ? atoi(argv[1]) : 1000;
    int max_plies = (argc > 2) ? atoi(argv[2]) : 80;

    init_zobrist();
    if (!zobrist_initialized()) {
        fprintf(stderr, "zobrist init failed\n");
        return 2;
    }

    int total_checks = 0;
    int total_mismatches = 0;
    int games_played = 0;

    for (int g = 0; g < n_games; g++) {
        Position pos;
        start_position(&pos);
        uint64_t running = zobrist_full(pos);
        // Sanity: full hash on starting position is reproducible.
        if (g == 0) {
            uint64_t h2 = zobrist_full(pos);
            if (h2 != running) {
                fprintf(stderr, "FAIL: zobrist_full not deterministic at start.\n");
                return 1;
            }
        }
        for (int ply = 0; ply < max_plies; ply++) {
            Move mv;
            if (!pick_legal(&pos, &mv)) break;  // checkmate / stalemate
            uint64_t expect_after = zobrist_update(running, pos, mv);
            h_make_move(&pos, mv);
            uint64_t fresh_after = zobrist_full(pos);
            total_checks++;
            if (expect_after != fresh_after) {
                total_mismatches++;
                if (total_mismatches <= 5) {
                    fprintf(stderr,
                        "[game %d ply %d] MISMATCH: incremental=%016lx fresh=%016lx "
                        "move from=%d to=%d promo=%d flags=%d\n",
                        g, ply, (unsigned long)expect_after, (unsigned long)fresh_after,
                        move_from(mv), move_to(mv), move_promo(mv), move_flags(mv));
                }
            }
            running = fresh_after;  // re-sync from the oracle so a single
                                     // bad move doesn't cascade error reports
        }
        games_played++;
    }

    printf("test_zobrist: games=%d checks=%d mismatches=%d\n",
           games_played, total_checks, total_mismatches);
    if (total_mismatches == 0) {
        printf("test_zobrist: PASS\n");
        return 0;
    } else {
        printf("test_zobrist: FAIL\n");
        return 1;
    }
}
