// =============================================================================
// movegen.cu
//
// Pseudo-legal movegen extracted verbatim from gpu_fighter.cu's
// d_generate_moves / h_generate_moves (which were already byte-identical
// except for compiler annotations). Single __host__ __device__ body.
//
// Plus host helpers: parse_fen, move_to_uci, perft (matches the perft
// implementation gpu_fighter.cu was verified against).
// =============================================================================
#include "movegen.cuh"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

namespace engine {

__host__ __device__ static inline int abs_i(int x) { return x < 0 ? -x : x; }

// -------- pseudo-legal movegen, identical to verified gpu_fighter logic --------
__host__ __device__ int generate_moves(const Position* s, Move* moves) {
    const int KN[8] = {-17, -15, -10, -6, 6, 10, 15, 17};
    const int KG[8] = { -9,  -8,  -7, -1, 1,  7,  8,  9};

    int count = 0;
    int side = s->side, opp = 1 - side;

    for (int sq = 0; sq < 64; sq++) {
        int piece = s->board[sq];
        if (piece == EMPTY) continue;
        if (side == WHITE_SIDE && !is_white(piece)) continue;
        if (side == BLACK_SIDE && !is_black(piece)) continue;

        int r = sq_rank(sq), f = sq_file(sq);
        int ptype = piece_type(piece);

        if (ptype == 1) { // pawn
            int dir       = (side == WHITE_SIDE) ? -1 : 1;
            int startRank = (side == WHITE_SIDE) ?  6 : 1;
            int promoRank = (side == WHITE_SIDE) ?  0 : 7;
            int nr = r + dir;

            // single push
            if (on_board(nr, f) && s->board[make_sq(nr, f)] == EMPTY) {
                if (nr == promoRank) {
                    int pb = (side == WHITE_SIDE) ? WQUEEN : BQUEEN;
                    moves[count++] = make_move(sq, make_sq(nr,f), pb,   FLAG_PROMO);
                    moves[count++] = make_move(sq, make_sq(nr,f), pb-1, FLAG_PROMO);
                    moves[count++] = make_move(sq, make_sq(nr,f), pb-2, FLAG_PROMO);
                    moves[count++] = make_move(sq, make_sq(nr,f), pb-3, FLAG_PROMO);
                } else {
                    moves[count++] = make_move(sq, make_sq(nr,f), 0, FLAG_NONE);
                }
                if (r == startRank) {
                    int nr2 = r + 2*dir;
                    if (s->board[make_sq(nr2,f)] == EMPTY)
                        moves[count++] = make_move(sq, make_sq(nr2,f), 0, FLAG_DOUBLE);
                }
            }

            // pawn captures + ep
            for (int df = -1; df <= 1; df += 2) {
                int nf = f + df;
                if (!on_board(nr, nf)) continue;
                int t  = make_sq(nr, nf);
                int tp = s->board[t];
                if (tp != EMPTY && piece_color(tp) == opp) {
                    if (nr == promoRank) {
                        int pb = (side == WHITE_SIDE) ? WQUEEN : BQUEEN;
                        moves[count++] = make_move(sq, t, pb,   FLAG_PROMO);
                        moves[count++] = make_move(sq, t, pb-1, FLAG_PROMO);
                        moves[count++] = make_move(sq, t, pb-2, FLAG_PROMO);
                        moves[count++] = make_move(sq, t, pb-3, FLAG_PROMO);
                    } else {
                        moves[count++] = make_move(sq, t, 0, FLAG_CAPTURE);
                    }
                }
                if (t == s->ep && s->ep >= 0)
                    moves[count++] = make_move(sq, t, 0, FLAG_EP);
            }
        }
        else if (ptype == 2) { // knight
            for (int i = 0; i < 8; i++) {
                int t = sq + KN[i];
                if (t < 0 || t >= 64) continue;
                int dr = abs_i(sq_rank(t) - r), df2 = abs_i(sq_file(t) - f);
                if (!((dr==2 && df2==1) || (dr==1 && df2==2))) continue;
                int tp = s->board[t];
                if (tp == EMPTY) moves[count++] = make_move(sq, t, 0, FLAG_NONE);
                else if (piece_color(tp) == opp) moves[count++] = make_move(sq, t, 0, FLAG_CAPTURE);
            }
        }

        if (ptype == 3 || ptype == 5) { // bishop / queen diagonals
            int ddr[] = {-1,-1,1,1}, ddf[] = {-1,1,-1,1};
            for (int d = 0; d < 4; d++) {
                int cr = r + ddr[d], cf = f + ddf[d];
                while (on_board(cr, cf)) {
                    int t = make_sq(cr, cf);
                    int tp = s->board[t];
                    if (tp == EMPTY) moves[count++] = make_move(sq, t, 0, FLAG_NONE);
                    else { if (piece_color(tp)==opp) moves[count++] = make_move(sq, t, 0, FLAG_CAPTURE); break; }
                    cr += ddr[d]; cf += ddf[d];
                }
            }
        }

        if (ptype == 4 || ptype == 5) { // rook / queen orthogonals
            int sdr[] = {-1,1,0,0}, sdf[] = {0,0,-1,1};
            for (int d = 0; d < 4; d++) {
                int cr = r + sdr[d], cf = f + sdf[d];
                while (on_board(cr, cf)) {
                    int t = make_sq(cr, cf);
                    int tp = s->board[t];
                    if (tp == EMPTY) moves[count++] = make_move(sq, t, 0, FLAG_NONE);
                    else { if (piece_color(tp)==opp) moves[count++] = make_move(sq, t, 0, FLAG_CAPTURE); break; }
                    cr += sdr[d]; cf += sdf[d];
                }
            }
        }

        if (ptype == 6) { // king
            for (int i = 0; i < 8; i++) {
                int t = sq + KG[i];
                if (t < 0 || t >= 64) continue;
                int dr = abs_i(sq_rank(t)-r), df2 = abs_i(sq_file(t)-f);
                if (dr > 1 || df2 > 1) continue;
                int tp = s->board[t];
                if (tp == EMPTY) moves[count++] = make_move(sq, t, 0, FLAG_NONE);
                else if (piece_color(tp) == opp) moves[count++] = make_move(sq, t, 0, FLAG_CAPTURE);
            }
            // castling
            if (side == WHITE_SIDE && r == 7 && f == 4) {
                if ((s->castle & CASTLE_WK) && s->board[61]==EMPTY && s->board[62]==EMPTY && s->board[63]==WROOK
                    && !is_square_attacked(s,60,BLACK_SIDE) && !is_square_attacked(s,61,BLACK_SIDE) && !is_square_attacked(s,62,BLACK_SIDE))
                    moves[count++] = make_move(60,62,0,FLAG_CASTLE_K);
                if ((s->castle & CASTLE_WQ) && s->board[59]==EMPTY && s->board[58]==EMPTY && s->board[57]==EMPTY && s->board[56]==WROOK
                    && !is_square_attacked(s,60,BLACK_SIDE) && !is_square_attacked(s,59,BLACK_SIDE) && !is_square_attacked(s,58,BLACK_SIDE))
                    moves[count++] = make_move(60,58,0,FLAG_CASTLE_Q);
            }
            if (side == BLACK_SIDE && r == 0 && f == 4) {
                if ((s->castle & CASTLE_BK) && s->board[5]==EMPTY && s->board[6]==EMPTY && s->board[7]==BROOK
                    && !is_square_attacked(s,4,WHITE_SIDE) && !is_square_attacked(s,5,WHITE_SIDE) && !is_square_attacked(s,6,WHITE_SIDE))
                    moves[count++] = make_move(4,6,0,FLAG_CASTLE_K);
                if ((s->castle & CASTLE_BQ) && s->board[3]==EMPTY && s->board[2]==EMPTY && s->board[1]==EMPTY && s->board[0]==BROOK
                    && !is_square_attacked(s,4,WHITE_SIDE) && !is_square_attacked(s,3,WHITE_SIDE) && !is_square_attacked(s,2,WHITE_SIDE))
                    moves[count++] = make_move(4,2,0,FLAG_CASTLE_Q);
            }
        }

        if (count >= MAX_MOVES - 8) break;
    }
    return count;
}

// ---------- host: FEN parser (extracted verbatim from gpu_fighter.cu) ----------
void parse_fen(const char* fen, Position* s) {
    memset(s, 0, sizeof(Position));
    for (int i = 0; i < 64; i++) s->board[i] = EMPTY;
    s->ep = -1; s->kingPos[0] = -1; s->kingPos[1] = -1;
    int sq = 0, i = 0;
    while (fen[i] && fen[i] != ' ') {
        char c = fen[i++];
        if (c == '/') continue;
        if (c >= '1' && c <= '8') { sq += (c - '0'); continue; }
        int piece = EMPTY;
        switch (c) {
            case 'P': piece = WPAWN; break;
            case 'N': piece = WKNIGHT; break;
            case 'B': piece = WBISHOP; break;
            case 'R': piece = WROOK; break;
            case 'Q': piece = WQUEEN; break;
            case 'K': piece = WKING; s->kingPos[WHITE_SIDE] = sq; break;
            case 'p': piece = BPAWN; break;
            case 'n': piece = BKNIGHT; break;
            case 'b': piece = BBISHOP; break;
            case 'r': piece = BROOK; break;
            case 'q': piece = BQUEEN; break;
            case 'k': piece = BKING; s->kingPos[BLACK_SIDE] = sq; break;
        }
        s->board[sq++] = piece;
    }
    while (fen[i] == ' ') i++;
    s->side = (fen[i] == 'b') ? BLACK_SIDE : WHITE_SIDE;
    i++;
    while (fen[i] == ' ') i++;
    s->castle = 0;
    if (fen[i] == '-') i++;
    else while (fen[i] && fen[i] != ' ') {
        switch (fen[i]) {
            case 'K': s->castle |= CASTLE_WK; break;
            case 'Q': s->castle |= CASTLE_WQ; break;
            case 'k': s->castle |= CASTLE_BK; break;
            case 'q': s->castle |= CASTLE_BQ; break;
        }
        i++;
    }
    while (fen[i] == ' ') i++;
    if (fen[i] == '-') { s->ep = -1; i++; }
    else if (fen[i] >= 'a' && fen[i] <= 'h') {
        int file = fen[i] - 'a'; i++;
        int rank = 8 - (fen[i] - '0'); i++;
        s->ep = make_sq(rank, file);
    }
    while (fen[i] == ' ') i++;
    s->halfmove = 0;
    while (fen[i] >= '0' && fen[i] <= '9') { s->halfmove = s->halfmove*10 + (fen[i]-'0'); i++; }
    while (fen[i] == ' ') i++;
    s->fullmove = 1;
    if (fen[i] >= '0' && fen[i] <= '9') {
        s->fullmove = 0;
        while (fen[i] >= '0' && fen[i] <= '9') { s->fullmove = s->fullmove*10 + (fen[i]-'0'); i++; }
    }
}

// ---------- host: UCI encode (extracted verbatim from gpu_fighter.cu) ----------
void move_to_uci(Move move, char* uci) {
    int from = move_from(move), to = move_to(move);
    int flags = move_flags(move), promo = move_promo(move);
    uci[0] = 'a' + sq_file(from);
    uci[1] = '0' + (8 - sq_rank(from));
    uci[2] = 'a' + sq_file(to);
    uci[3] = '0' + (8 - sq_rank(to));
    uci[4] = '\0';
    if (flags == FLAG_PROMO) {
        int pt = piece_type(promo);
        switch (pt) {
            case 5: uci[4] = 'q'; break;
            case 4: uci[4] = 'r'; break;
            case 3: uci[4] = 'b'; break;
            case 2: uci[4] = 'n'; break;
        }
        uci[5] = '\0';
    }
}

// ---------- host perft (matches gpu_fighter.cu h_perft) ----------
long long perft(Position* s, int depth) {
    if (depth == 0) return 1LL;
    Move moves[MAX_MOVES];
    int n = generate_moves(s, moves);
    long long total = 0;
    for (int i = 0; i < n; i++) {
        Position child = *s;
        make_move(&child, moves[i]);
        // Legality: own king must not be in check after the move.
        if (in_check(&child, 1 - child.side)) continue;
        if (depth == 1) total += 1;
        else total += perft(&child, depth - 1);
    }
    return total;
}

} // namespace engine
