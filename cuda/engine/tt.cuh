// =============================================================================
// tt.cuh
//
// GPU-resident transposition table. Single-slot with versioned publication.
//
// Logical entry (the user-visible struct returned from tt_probe):
//   uint64_t key      ; full 64-bit Zobrist hash (verification on probe)
//   int16_t  score    ; cached subtree score
//   int16_t  depth    ; depth this score is valid for (>=0; <0 means qsearch)
//   uint8_t  bound    ; 0=EXACT, 1=LOWER, 2=UPPER  (TT_BOUND_*)
//   uint8_t  age      ; rolling search counter
//   uint16_t move     ; best move encoded in 16 bits (low 16 of engine::Move)
//   = 16 bytes, 16-byte aligned.
//
// Physical entry actually stored in device memory: a 16-byte struct of two
// 64-bit words:
//
//       data  = packed (score|depth|bound|age|move) as one 64-bit word
//       guard = key XOR data
//
// Publication is guarded by a per-slot version word:
//   * odd  version -> writer owns the slot
//   * even version -> slot is stable for readers
//
// Probes read the version before and after loading (guard, data) and treat
// any mismatch or odd version as a miss/retry. Stores acquire the version with
// atomicCAS, update the slot payload, then publish the next even version.
//
// The XOR guard is still used as the key check, but it is no longer relied on
// as the sole torn-write defense.
//
// Index = (hash & (n_entries - 1)).  n_entries is forced to be a power of 2.
//
// Replace policy (single-slot with depth + age preference):
//   Replace IFF
//     (1) slot is empty (key == 0 && depth == 0), OR
//     (2) age != current age (stale-from-previous-search), OR
//     (3) new depth >= existing depth.
//   Otherwise keep the existing entry. Standard depth/age scheme for a
//   1-slot table.
// =============================================================================
#ifndef ENGINE_TT_CUH
#define ENGINE_TT_CUH

#include <stdint.h>
#include <cuda_runtime.h>

namespace engine {

constexpr uint8_t TT_BOUND_EXACT = 0;
constexpr uint8_t TT_BOUND_LOWER = 1;
constexpr uint8_t TT_BOUND_UPPER = 2;

// User-visible (decoded) entry. Returned from tt_probe.
struct TTEntry {
    uint64_t key;
    int16_t  score;
    int16_t  depth;
    uint8_t  bound;
    uint8_t  age;
    uint16_t move;
};

// ---------- host API ----------
// Allocate the device-resident table. size_mb is rounded down to the nearest
// power-of-2 number of physical entries that fits. Default 64 MB.
// Returns the actual entry count.
uint64_t tt_init(size_t size_mb = 64);

// Reallocate the table to the requested size in megabytes and clear it.
void tt_resize(int size_mb);

// Free the table.
void tt_free();

// Zero out all entries (key=0 means empty).
void tt_clear();

// Bump the global age by 1 (mod 256). Call once per root search.
void tt_age();

// Stats accessors.
uint64_t tt_num_entries();
double   tt_fill_rate();   // sampled fraction of slots with non-zero key
uint8_t  tt_current_age();

// ---------- device API ----------
// Probe the slot at index = hash & mask. Returns a decoded TTEntry; the
// caller MUST check `entry.key == hash` for a true hit. A torn-but-detected
// write returns an entry whose key cannot match `hash` (it'll be the XOR
// of two unrelated halves), so the standard key-check naturally rejects it.
__device__ TTEntry tt_probe(uint64_t hash);

// Store with always-replace + depth-preference policy. `move` should be the
// low 16 bits of the engine::Move encoding (from+to+promo; flag bits are
// recoverable from the position so we omit them to fit 16 bits).
__device__ void tt_store(uint64_t hash, int score, int depth,
                          uint8_t bound, uint16_t move);

} // namespace engine

#endif // ENGINE_TT_CUH
