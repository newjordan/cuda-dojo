// =============================================================================
// engine_types.h
//
// Canonical types for the CUDA chess engine library. Every primitive
// (movegen, make/unmake, eval) operates on engine::Position.
//
// Layout note:
//   This struct is intentionally identical, byte-for-byte, to the
//   `BoardState` struct in cuda/gpu_fighter.cu — that is the layout the
//   verified search uses. We only rename it and move it under engine::.
//
// All public engine API uses fixed-size move encodings (32-bit int) so
// host and device callers see identical semantics.
// =============================================================================
#ifndef ENGINE_TYPES_H
#define ENGINE_TYPES_H

#include <stdint.h>
#include <cuda_runtime.h>

namespace engine {

// ---------- size constants ----------
constexpr int MAX_MOVES       = 256;
constexpr int MAX_ROOT_MOVES  = 256;
constexpr int MAX_PLY_MOVES   = 256;

// ---------- piece encoding ----------
// 0 = empty; 1..6 = white {P,N,B,R,Q,K}; 7..12 = black {P,N,B,R,Q,K}
constexpr int8_t EMPTY   = 0;
constexpr int8_t WPAWN   = 1;
constexpr int8_t WKNIGHT = 2;
constexpr int8_t WBISHOP = 3;
constexpr int8_t WROOK   = 4;
constexpr int8_t WQUEEN  = 5;
constexpr int8_t WKING   = 6;
constexpr int8_t BPAWN   = 7;
constexpr int8_t BKNIGHT = 8;
constexpr int8_t BBISHOP = 9;
constexpr int8_t BROOK   = 10;
constexpr int8_t BQUEEN  = 11;
constexpr int8_t BKING   = 12;

// ---------- side encoding ----------
constexpr int8_t WHITE_SIDE = 0;
constexpr int8_t BLACK_SIDE = 1;

// ---------- castle flag bits ----------
constexpr int8_t CASTLE_WK = 1;
constexpr int8_t CASTLE_WQ = 2;
constexpr int8_t CASTLE_BK = 4;
constexpr int8_t CASTLE_BQ = 8;

// ---------- move flag tags ----------
constexpr int FLAG_NONE     = 0;
constexpr int FLAG_CAPTURE  = 1;
constexpr int FLAG_EP       = 2;
constexpr int FLAG_CASTLE_K = 3;
constexpr int FLAG_CASTLE_Q = 4;
constexpr int FLAG_DOUBLE   = 5;
constexpr int FLAG_PROMO    = 6;

// ---------- score sentinels ----------
constexpr int MATE_SCORE = 100000;
constexpr int INF_SCORE  = 200000;

using Score = int;
using Move  = int;

// ---------- piece predicates / accessors ----------
__host__ __device__ inline bool is_white(int p) { return p >= WPAWN && p <= WKING; }
__host__ __device__ inline bool is_black(int p) { return p >= BPAWN && p <= BKING; }
__host__ __device__ inline int  piece_color(int p) { return is_white(p) ? WHITE_SIDE : BLACK_SIDE; }
__host__ __device__ inline int  piece_type(int p)  { return is_white(p) ? p : p - 6; }

// ---------- canonical Position struct ----------
// Byte-for-byte identical to gpu_fighter.cu's BoardState (verified-correct
// search uses this layout). 8 board bytes + 1+1+1 + 2+2 + 2 = 71 bytes
// before padding; with default alignment the total is ~72 bytes.
struct Position {
    int8_t  board[64];   // 0..63: A8 = 0, H1 = 63 (rank 0 = top)
    int8_t  side;        // WHITE_SIDE / BLACK_SIDE
    int8_t  castle;      // bitmask of CASTLE_*
    int8_t  ep;          // -1 if none, else target square
    int16_t halfmove;
    int16_t fullmove;
    int8_t  kingPos[2];  // [0] = white king sq, [1] = black king sq
};

// ---------- move encoding ----------
// 32-bit move:  bits  0..5  = from square
//               bits  6..11 = to square
//               bits 12..15 = promo piece (engine encoding)
//               bits 16..19 = flag tag
__host__ __device__ inline Move make_move(int from, int to, int promo, int flags) {
    return ((from) & 0x3F) | (((to) & 0x3F) << 6) | (((promo) & 0xF) << 12) | (((flags) & 0xF) << 16);
}
__host__ __device__ inline int move_from(Move m)  { return  m        & 0x3F; }
__host__ __device__ inline int move_to(Move m)    { return (m >>  6) & 0x3F; }
__host__ __device__ inline int move_promo(Move m) { return (m >> 12) & 0x0F; }
__host__ __device__ inline int move_flags(Move m) { return (m >> 16) & 0x0F; }

// ---------- square helpers ----------
__host__ __device__ inline int sq_rank(int sq)            { return sq >> 3; }
__host__ __device__ inline int sq_file(int sq)            { return sq & 7;  }
__host__ __device__ inline int make_sq(int r, int f)      { return (r << 3) | f; }
__host__ __device__ inline int mirror_sq(int sq)          { return ((7 - (sq >> 3)) << 3) | (sq & 7); }
__host__ __device__ inline bool on_board(int r, int f)    { return r >= 0 && r < 8 && f >= 0 && f < 8; }

} // namespace engine

#endif // ENGINE_TYPES_H
