// =============================================================================
// uci_stubs.cu — minimal placeholder implementations of engine::* so that
// uci.cu links and uci_test.sh can validate the protocol shell end-to-end.
//
// The real implementations come from sibling agents:
//   - engine::set_position  -> skeleton agent (make/unmake from FEN+moves)
//   - engine::search        -> search agent (alpha-beta or MCTS over the GPU)
//   - engine::tt_clear      -> TT agent (zero the transposition table)
//   - engine::tt_resize     -> TT agent (cudaMalloc a new sized TT)
//
// Stub behaviour:
//   - set_position: accepts everything, just records the FEN length.
//   - search: emits one info line per "depth" up to limits.depth (or 1 if
//     time-limited), returns "e2e4" as bestmove regardless of position.
//   - tt_clear / tt_resize: no-ops with a stderr log.
// =============================================================================
#include "uci.h"

#include <chrono>
#include <cstdio>
#include <thread>

namespace engine {

bool set_position(const std::string& fen,
                  const std::vector<std::string>& moves) {
    // Stub: pretend any non-empty FEN is valid.
    if (fen.empty()) return false;
    std::fprintf(stderr,
                 "[stub] set_position: fen_len=%zu moves=%zu\n",
                 fen.size(), moves.size());
    return true;
}

uci::SearchResult search(const uci::SearchLimits& limits,
                         uci::InfoCallback info_cb) {
    using clock = std::chrono::steady_clock;
    auto t0 = clock::now();

    int max_depth = (limits.depth > 0) ? limits.depth : 1;
    if (max_depth > 4) max_depth = 4; // stub cap so tests stay fast

    int budget_ms = 0;
    if (limits.movetime > 0) budget_ms = limits.movetime;
    else if (limits.wtime > 0 || limits.btime > 0) {
        budget_ms = limits.resolve_movetime(0); // assume white for stub
    }

    int64_t fake_nodes = 0;
    for (int d = 1; d <= max_depth; ++d) {
        if (uci::stop_requested()) break;
        // Pretend to "think" briefly so nps math doesn't divide by zero,
        // but stay well under typical UCI test timeouts.
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
        fake_nodes += 1234LL * d;

        auto now = clock::now();
        int64_t elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                 now - t0).count();
        if (elapsed_ms <= 0) elapsed_ms = 1;

        uci::SearchInfo info;
        info.depth    = d;
        info.seldepth = d;
        info.nodes    = fake_nodes;
        info.time_ms  = elapsed_ms;
        info.nps      = (fake_nodes * 1000) / elapsed_ms;
        info.score_cp = 25; // a polite "I'm slightly better"
        info.pv.push_back("e2e4");
        if (info_cb) info_cb(info);

        // Honour movetime budget if set.
        if (budget_ms > 0 && elapsed_ms >= budget_ms) break;
    }

    // Honour `go infinite` until stop.
    if (limits.infinite) {
        while (!uci::stop_requested()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }

    uci::SearchResult r;
    r.bestmove = "e2e4";
    return r;
}

void tt_clear() {
    std::fprintf(stderr, "[stub] tt_clear\n");
}

void tt_resize(int megabytes) {
    std::fprintf(stderr, "[stub] tt_resize %d MB\n", megabytes);
}

} // namespace engine
