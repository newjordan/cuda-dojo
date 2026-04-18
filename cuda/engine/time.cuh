// =============================================================================
// time.cuh
//
// Tiny host-side time-management helpers for the search loop. The core search
// runs on the GPU; the host owns the wall-clock and signals "stop" via a
// device-resident flag (see search.cuh / search.cu).
// =============================================================================
#ifndef ENGINE_TIME_CUH
#define ENGINE_TIME_CUH

#include <cstdint>

namespace engine {

// Compute a per-move time budget in milliseconds from UCI-style time controls.
// Conservative formula: min(remaining/30 + inc/2, remaining/4).
//   wtime / btime : ms remaining on white / black clock
//   winc  / binc  : ms increment per move
//   we_are_white  : true iff side to move is white
// Returns >0 in ms, or 0 if neither time-control nor inc was specified
// (caller should fall back to a default).
int compute_movetime(int wtime, int btime, int winc, int binc, bool we_are_white);

// Start a wall-clock timer with the given budget in milliseconds.
// `ms <= 0` disables the timer; time_up() will always return false.
void start_timer(int ms);

// Returns true once the budget given to start_timer has elapsed.
bool time_up();

// Milliseconds elapsed since the last start_timer() call.
int64_t elapsed_ms();

// Reset the timer (so time_up always returns false until start_timer
// is called again).
void clear_timer();

} // namespace engine

#endif // ENGINE_TIME_CUH
