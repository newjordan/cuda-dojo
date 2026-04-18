// =============================================================================
// search.cuh
//
// Root-parallel alpha-beta GPU search backed by the engine library:
//   - movegen, attacks, make_move from cuda/engine/
//   - tapered PeSTO eval from engine/eval
//   - lockless TT from engine/tt
//   - Zobrist incremental hashing from engine/zobrist
//
// Architecture:
//   * Each *root move* is searched by exactly one CUDA thread.
//   * Each thread runs an iterative (explicit-stack) negamax with TT probe at
//     the top of every node and TT store after every node returns.
//   * Move ordering inside each node: TT move first, then captures by MVV-LVA,
//     then quiet killer moves, then other quiet moves.
//   * qsearch at depth==0 leaves: captures only with stand-pat.
//   * Host loops iterative deepening: depth 1..N. Between iterations, host
//     checks wall-clock time and (if budget exceeded) sets a __device__ stop
//     flag — kernels poll it every NODE_STOP_INTERVAL nodes and abort.
//   * After each iteration completes, the host extracts the principal variation
//     by walking the TT, then calls the UCI info callback. The host also keeps
//     the previous best root move at the head of the next iteration's root
//     list so equal-score ties stay stable.
//
// Score units: centipawns from side-to-move POV inside negamax. The host
// converts to UCI's "score cp X" (white POV) before emitting info lines.
// =============================================================================
#ifndef ENGINE_SEARCH_CUH
#define ENGINE_SEARCH_CUH

#include <cstdint>
#include <vector>
#include <string>

#include "engine_types.h"
#include "uci.h"

namespace engine {

// Maximum search depth supported by the explicit-stack negamax. Each frame
// holds a 256-entry move list, so per-thread stack grows ~1 KB per ply.
// 16 ply = ~16 KB which still fits comfortably with cudaDeviceSetLimit.
constexpr int MAX_SEARCH_DEPTH = 16;

// Maximum number of root moves we ever search in parallel. Equal to
// MAX_ROOT_MOVES from engine_types.h.
constexpr int SEARCH_MAX_ROOT_MOVES = 256;

struct SearchRuntimeReport {
    bool runtime_enabled = false;
    bool runtime_used = false;
    uint32_t root_move_count = 0;
    uint32_t reordered_root_moves = 0;
    uint32_t scheduler_frontier_seeded = 0;
    uint32_t scheduler_frontier_pops = 0;
    uint32_t scheduler_eval_requests = 0;
    uint32_t scheduler_eval_request_depth = 0;
    uint32_t eval_bucket_idx = 0;
    uint32_t eval_bucket_size = 0;
    uint32_t eval_request_count = 0;
    uint32_t eval_result_count = 0;
    uint32_t eval_dropped_requests = 0;
    uint32_t eval_backend = 0;
    uint64_t eval_total_dispatches = 0;
    uint64_t eval_fallback_dispatches = 0;
    uint64_t eval_failed_dispatches = 0;
    uint32_t top_order_moves[8]{};
    int32_t top_order_scores[8]{};
};

// One-shot init: reserves all device-side scratch memory used by the search
// kernels (root state buffer, root-move list, root-score list, stop flag).
// Call once after the device is selected and TT/Zobrist are initialised.
void search_init();

// Free all search-side device memory. Idempotent.
void search_shutdown();

// Reset per-game state (currently: clears the host-side stop flag mirror).
void search_new_game();

const SearchRuntimeReport& search_runtime_report();

// Top-level: run iterative deepening on `root` honoring the supplied UCI limits
// and emit info lines via `info_cb`. Returns the bestmove (UCI long-algebraic
// string) and optional ponder. `set_position` (uci.h) wires the engine's
// current Position into us via set_root_position(...) — this entry point only
// reads it.
uci::SearchResult search_root(const Position& root,
                              const uci::SearchLimits& limits,
                              uci::InfoCallback info_cb);

} // namespace engine

#endif // ENGINE_SEARCH_CUH
