// =============================================================================
// attacks.cuh
//
// Attack and check detection. Backed by a single __host__ __device__
// implementation extracted verbatim from the verified-correct movegen in
// gpu_fighter.cu (passes perft on all 6 canonical chessprogramming.org
// positions at depth 4-5, 224 positions, no mismatches).
// =============================================================================
#ifndef ENGINE_ATTACKS_CUH
#define ENGINE_ATTACKS_CUH

#include "engine_types.h"

namespace engine {

// Returns true if `sq` is attacked by side `by_side` in position `s`.
// __host__ __device__: same body for both worlds, no constant memory needed
// because we use a tiny on-stack copy of the offset tables.
__host__ __device__ bool is_square_attacked(const Position* s, int sq, int by_side);

// Convenience: is the king of `side` in check?
__host__ __device__ inline bool in_check(const Position* s, int side) {
    return is_square_attacked(s, s->kingPos[side], 1 - side);
}

// Device-only convenience aliases (match the original symbol names so other
// callers can swap in the engine namespace easily).
__device__ inline bool d_is_square_attacked(const Position* s, int sq, int by_side) {
    return is_square_attacked(s, sq, by_side);
}
__device__ inline bool d_is_in_check(const Position* s, int side) {
    return in_check(s, side);
}

// Host alias for symmetry with the gpu_fighter naming.
inline bool h_in_check(const Position* s, int side) {
    return in_check(s, side);
}

} // namespace engine

#endif // ENGINE_ATTACKS_CUH
