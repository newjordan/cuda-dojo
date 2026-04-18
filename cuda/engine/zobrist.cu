// =============================================================================
// zobrist.cu -- implementation of zobrist.cuh
// =============================================================================
#include "zobrist.cuh"

#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>

namespace engine {

// ---------- key table layout ----------
// piece_keys[piece-1][sq]  for piece in 1..12, sq in 0..63    -> 12*64 = 768
// side_key                                                    -> 1
// castle_keys[bit]          for bit in 0..3                   -> 4
// ep_file_keys[file]        for file in 0..7                  -> 8
// total = 781
static constexpr int Z_NUM_PIECE_KEYS  = 12 * 64;
static constexpr int Z_NUM_CASTLE_KEYS = 4;
static constexpr int Z_NUM_EP_KEYS     = 8;
static constexpr int Z_TOTAL_KEYS      = Z_NUM_PIECE_KEYS + 1 + Z_NUM_CASTLE_KEYS + Z_NUM_EP_KEYS;
static_assert(Z_TOTAL_KEYS == 781, "zobrist key count must be 781");

// Device side: keys live in __constant__ memory for fast cached access.
__constant__ uint64_t d_zobrist_keys[Z_TOTAL_KEYS];

// Host side: mirror so __host__ callers (zobrist_full from host) work.
static uint64_t h_zobrist_keys[Z_TOTAL_KEYS];
static bool     g_initialized = false;
static uint64_t g_initialized_seed = 0;

// ---------- key index helpers ----------
__host__ __device__ static inline int idx_piece(int piece, int sq) {
    // piece in 1..12  ->  (piece-1)*64 + sq
    return (piece - 1) * 64 + sq;
}
__host__ __device__ static inline int idx_side()                  { return Z_NUM_PIECE_KEYS; }
__host__ __device__ static inline int idx_castle(int bit)         { return Z_NUM_PIECE_KEYS + 1 + bit; }
__host__ __device__ static inline int idx_ep(int file)            { return Z_NUM_PIECE_KEYS + 1 + Z_NUM_CASTLE_KEYS + file; }

// SplitMix64 -- deterministic, high-quality, used to seed Zobrist tables.
__host__ __device__ static inline uint64_t splitmix64(uint64_t& s) {
    uint64_t z = (s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

// ---------- accessor that selects host vs device table ----------
__host__ __device__ static inline uint64_t zk(int idx) {
#ifdef __CUDA_ARCH__
    return d_zobrist_keys[idx];
#else
    return h_zobrist_keys[idx];
#endif
}

// ---------- public low-level accessors ----------
__host__ __device__ uint64_t zobrist_piece_key(int piece, int sq) {
    return zk(idx_piece(piece, sq));
}
__host__ __device__ uint64_t zobrist_side_key() {
    return zk(idx_side());
}
__host__ __device__ uint64_t zobrist_castle_key(int castle_bit) {
    return zk(idx_castle(castle_bit));
}
__host__ __device__ uint64_t zobrist_ep_file_key(int file) {
    return zk(idx_ep(file));
}

// ---------- init ----------
bool zobrist_initialized() { return g_initialized; }

void init_zobrist(uint64_t seed) {
    if (g_initialized && g_initialized_seed == seed) return;
    uint64_t s = seed;
    for (int i = 0; i < Z_TOTAL_KEYS; i++) {
        h_zobrist_keys[i] = splitmix64(s);
    }
    cudaError_t err = cudaMemcpyToSymbol(d_zobrist_keys, h_zobrist_keys,
                                         sizeof(uint64_t) * Z_TOTAL_KEYS,
                                         0, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "init_zobrist: cudaMemcpyToSymbol failed: %s\n",
                cudaGetErrorString(err));
    }
    g_initialized = true;
    g_initialized_seed = seed;
}

// ---------- full hash ----------
__host__ __device__ uint64_t zobrist_full(const Position& pos) {
    uint64_t h = 0;
    for (int sq = 0; sq < 64; sq++) {
        int p = pos.board[sq];
        if (p != EMPTY) {
            h ^= zk(idx_piece(p, sq));
        }
    }
    if (pos.side == BLACK_SIDE) {
        h ^= zk(idx_side());
    }
    if (pos.castle & CASTLE_WK) h ^= zk(idx_castle(0));
    if (pos.castle & CASTLE_WQ) h ^= zk(idx_castle(1));
    if (pos.castle & CASTLE_BK) h ^= zk(idx_castle(2));
    if (pos.castle & CASTLE_BQ) h ^= zk(idx_castle(3));
    if (pos.ep >= 0) {
        h ^= zk(idx_ep(sq_file(pos.ep)));
    }
    return h;
}

// ---------- helper: castle-rights diff ----------
// Replicates the rule from gpu_fighter.cu::d_make_move: certain (from,to,piece)
// combos clear castle bits. We compute the new castle mask from the old mask
// and the move semantics, then XOR the differing bits in/out.
__host__ __device__ static inline int8_t compute_new_castle(int8_t old_castle,
                                                            int piece, int from, int to) {
    int8_t c = old_castle;
    if (piece == WKING) c &= ~(CASTLE_WK | CASTLE_WQ);
    if (piece == BKING) c &= ~(CASTLE_BK | CASTLE_BQ);
    if (from == 63 || to == 63) c &= ~CASTLE_WK;
    if (from == 56 || to == 56) c &= ~CASTLE_WQ;
    if (from == 7  || to == 7 ) c &= ~CASTLE_BK;
    if (from == 0  || to == 0 ) c &= ~CASTLE_BQ;
    return c;
}

// ---------- incremental update ----------
__host__ __device__ uint64_t zobrist_update(uint64_t hash_before,
                                             const Position& pos_before,
                                             Move mv) {
    uint64_t h = hash_before;
    int from  = move_from(mv);
    int to    = move_to(mv);
    int promo = move_promo(mv);
    int flags = move_flags(mv);

    int piece = pos_before.board[from];
    int side  = pos_before.side;

    // 1) Remove the moving piece from its source square.
    h ^= zk(idx_piece(piece, from));

    // 2) Remove a captured piece (if any). gpu_fighter encodes regular
    //    captures as FLAG_CAPTURE, en-passant as FLAG_EP, and PROMO captures
    //    just as FLAG_PROMO (the destination square gets overwritten by the
    //    promo piece, capturing implicitly). So we also need to check for an
    //    occupied destination on FLAG_PROMO.
    if (flags == FLAG_CAPTURE || flags == FLAG_PROMO) {
        int victim = pos_before.board[to];
        if (victim != EMPTY && piece_color(victim) == (1 - side)) {
            h ^= zk(idx_piece(victim, to));
        }
    } else if (flags == FLAG_EP) {
        // En-passant capture: victim sits on adjacent square, not on `to`.
        int cs = (side == WHITE_SIDE) ? to + 8 : to - 8;
        int victim = pos_before.board[cs];
        if (victim != EMPTY) {
            h ^= zk(idx_piece(victim, cs));
        }
    }

    // 3) Place the moving piece on the destination (with promotion if any).
    if (flags == FLAG_PROMO) {
        // Promo piece value comes pre-encoded as a full piece code (e.g. WQUEEN).
        h ^= zk(idx_piece(promo, to));
    } else {
        h ^= zk(idx_piece(piece, to));
    }

    // 4) Castling: also move the rook.
    if (flags == FLAG_CASTLE_K) {
        if (side == WHITE_SIDE) {
            h ^= zk(idx_piece(WROOK, 63));   // rook leaves H1
            h ^= zk(idx_piece(WROOK, 61));   // rook arrives F1
        } else {
            h ^= zk(idx_piece(BROOK, 7));    // H8
            h ^= zk(idx_piece(BROOK, 5));    // F8
        }
    } else if (flags == FLAG_CASTLE_Q) {
        if (side == WHITE_SIDE) {
            h ^= zk(idx_piece(WROOK, 56));   // A1
            h ^= zk(idx_piece(WROOK, 59));   // D1
        } else {
            h ^= zk(idx_piece(BROOK, 0));    // A8
            h ^= zk(idx_piece(BROOK, 3));    // D8
        }
    }

    // 5) Castling-rights diff.
    int8_t old_c = pos_before.castle;
    int8_t new_c = compute_new_castle(old_c, piece, from, to);
    int8_t diff  = old_c ^ new_c;
    if (diff & CASTLE_WK) h ^= zk(idx_castle(0));
    if (diff & CASTLE_WQ) h ^= zk(idx_castle(1));
    if (diff & CASTLE_BK) h ^= zk(idx_castle(2));
    if (diff & CASTLE_BQ) h ^= zk(idx_castle(3));

    // 6) En-passant square diff. Replicate the rule from d_make_move:
    //    new_ep = (from+to)/2 ONLY when piece is a pawn AND flags==FLAG_DOUBLE,
    //    otherwise new_ep = -1.
    int new_ep = -1;
    if (piece_type(piece) == 1 && flags == FLAG_DOUBLE) {
        new_ep = (from + to) / 2;
    }
    int old_ep = pos_before.ep;
    if (old_ep >= 0) h ^= zk(idx_ep(sq_file(old_ep)));
    if (new_ep >= 0) h ^= zk(idx_ep(sq_file(new_ep)));

    // 7) Side-to-move flips every ply.
    h ^= zk(idx_side());

    return h;
}

} // namespace engine
