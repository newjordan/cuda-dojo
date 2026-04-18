// =============================================================================
// time.cu — host-side time management.
// =============================================================================
#include "time.cuh"

#include <chrono>

namespace engine {

namespace {

using clock_t = std::chrono::steady_clock;

clock_t::time_point g_t0;
int64_t g_budget_ms = 0;
bool g_armed = false;

} // anonymous

int compute_movetime(int wtime, int btime, int winc, int binc, bool we_are_white) {
    int remaining = we_are_white ? wtime : btime;
    int inc       = we_are_white ? winc  : binc;
    if (remaining <= 0 && inc <= 0) return 0;

    // First-cut formula: budget = remaining/30 + inc/2, capped at remaining/4.
    int budget = remaining / 30 + inc / 2;
    int cap    = (remaining > 0) ? (remaining / 4) : budget;
    if (cap > 0 && budget > cap) budget = cap;
    if (budget < 1) budget = 1;
    return budget;
}

void start_timer(int ms) {
    g_t0 = clock_t::now();
    g_budget_ms = (ms > 0) ? ms : 0;
    g_armed = (ms > 0);
}

bool time_up() {
    if (!g_armed) return false;
    auto now = clock_t::now();
    int64_t e = std::chrono::duration_cast<std::chrono::milliseconds>(now - g_t0).count();
    return e >= g_budget_ms;
}

int64_t elapsed_ms() {
    auto now = clock_t::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(now - g_t0).count();
}

void clear_timer() {
    g_armed = false;
    g_budget_ms = 0;
}

} // namespace engine
