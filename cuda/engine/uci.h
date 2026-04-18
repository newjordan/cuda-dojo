// =============================================================================
// uci.h — UCI (Universal Chess Interface) protocol handler types & API
//
// This is the *protocol shell* of the GPU chess engine. It speaks UCI on
// stdin/stdout and dispatches into engine functions (set_position, search,
// tt_clear) provided by sibling agents (skeleton + TT + search).
//
// The handler is host-only — it's pure I/O. All heavy lifting happens behind
// the engine::* facade.
// =============================================================================
#ifndef CUDA_ENGINE_UCI_H
#define CUDA_ENGINE_UCI_H

#include <stdint.h>
#include <string>
#include <vector>
#include <functional>

namespace uci {

// -----------------------------------------------------------------------------
// Engine identity (advertised in `uci` reply).
// -----------------------------------------------------------------------------
constexpr const char* ENGINE_NAME   = "GPUChess 0.1";
constexpr const char* ENGINE_AUTHOR = "CUDA DOJO";

// -----------------------------------------------------------------------------
// Option — UCI `option name X type Y default D min M max N`.
// We support the spin (integer) type which covers Hash + Threads. Other UCI
// types (check/string/combo/button) can be added later; not in MVP scope.
// -----------------------------------------------------------------------------
struct Option {
    std::string name;
    std::string type;       // "spin" only for now
    int64_t default_value;
    int64_t min_value;
    int64_t max_value;
    int64_t value;          // current value (mutated by setoption)
};

// -----------------------------------------------------------------------------
// SearchLimits — populated from a `go` command. The search agent honours
// whichever of these is set; UCI provides exactly one of {depth, movetime,
// time-control, infinite}.
// -----------------------------------------------------------------------------
struct SearchLimits {
    int  depth     = 0;     // go depth N            (0 = not set)
    int  movetime  = 0;     // go movetime MS        (0 = not set, ms)
    int  wtime     = 0;     // go wtime W            (ms remaining, white)
    int  btime     = 0;     // go btime B            (ms remaining, black)
    int  winc      = 0;     // go winc Wi            (ms increment, white)
    int  binc      = 0;     // go binc Bi            (ms increment, black)
    int  movestogo = 0;     // go movestogo N        (0 = sudden death)
    bool infinite  = false; // go infinite

    // Resolve to a concrete movetime budget in ms, given side-to-move.
    // For tournament time-control the spec calls for min(remaining/30, 5000).
    int  resolve_movetime(int side_to_move /*0=white,1=black*/) const;
};

// -----------------------------------------------------------------------------
// SearchInfo — the search agent invokes the info_callback once per completed
// iterative-deepening iteration with one of these populated. The UCI handler
// formats it as `info depth ... pv ...` and emits to stdout.
// -----------------------------------------------------------------------------
struct SearchInfo {
    int depth        = 0;
    int seldepth     = 0;
    int64_t nodes    = 0;
    int64_t nps      = 0;
    int64_t time_ms  = 0;
    int score_cp     = 0;       // centipawn score from STM POV
    int mate_in      = 0;       // 0 = not a mate; positive = mate in N for STM
    std::vector<std::string> pv; // principal variation, UCI move strings
};

using InfoCallback = std::function<void(const SearchInfo&)>;

// -----------------------------------------------------------------------------
// SearchResult — what a search returns. bestmove is the UCI string ("e2e4",
// "e7e8q"); ponder may be empty.
// -----------------------------------------------------------------------------
struct SearchResult {
    std::string bestmove;
    std::string ponder;
};

// -----------------------------------------------------------------------------
// Token utilities — the parser splits whitespace and walks the token list.
// Forgiving: extra whitespace, missing args, unknown options never crash.
// -----------------------------------------------------------------------------
std::vector<std::string> tokenize(const std::string& line);

// True if `tok == needle` (case-sensitive — UCI is lowercase).
bool token_equals(const std::string& tok, const char* needle);

// -----------------------------------------------------------------------------
// Output helpers — line-buffered writes to stdout. Always end with '\n'.
// -----------------------------------------------------------------------------
void send_line(const std::string& line);
void send_id();
void send_options();
void send_uciok();
void send_readyok();
void send_info(const SearchInfo& info);
void send_bestmove(const SearchResult& result);

// -----------------------------------------------------------------------------
// Command handlers — each consumes the rest of a parsed line.
// -----------------------------------------------------------------------------
void handle_uci();
void handle_isready();
void handle_setoption(const std::vector<std::string>& tokens);
void handle_ucinewgame();
void handle_position(const std::vector<std::string>& tokens);
void handle_go(const std::vector<std::string>& tokens);
void handle_stop();
// quit returns true -> caller exits the read loop.
bool handle_quit();

// -----------------------------------------------------------------------------
// Top-level loop — read stdin line by line until quit / EOF.
// Returns 0 on clean exit.
// -----------------------------------------------------------------------------
int run();

// -----------------------------------------------------------------------------
// Option store — accessors for the engine to read the configured Hash size etc.
// -----------------------------------------------------------------------------
Option* find_option(const std::string& name);
void    set_option_value(const std::string& name, const std::string& value);
const std::vector<Option>& all_options();

// -----------------------------------------------------------------------------
// Position state — the parsed `position` command stashes the FEN + applied
// moves here for the engine to consume. Side-to-move is derived from FEN.
// -----------------------------------------------------------------------------
struct PositionState {
    std::string fen;                 // starting FEN (after "position startpos"
                                     // we expand to the standard initial FEN)
    std::vector<std::string> moves;  // UCI moves to apply on top of FEN
    int side_to_move = 0;            // 0=white, 1=black, derived from FEN field
};
const PositionState& current_position();

// -----------------------------------------------------------------------------
// Stop flag — set by the `stop` command, read by the search agent to abort
// iterative deepening early.
// -----------------------------------------------------------------------------
void  request_stop();
bool  stop_requested();
void  clear_stop();

} // namespace uci

// =============================================================================
// engine:: facade — implemented by sibling agents. UCI calls these via extern
// linkage. uci_stubs.cu provides minimal placeholders for stand-alone testing.
// =============================================================================
namespace engine {

// Set the board to `fen` and apply `moves` (UCI strings) on top.
// Returns false if FEN or any move is invalid (UCI handler logs to stderr).
bool set_position(const std::string& fen,
                  const std::vector<std::string>& moves);

// Search the current position with the given limits. The callback fires once
// per completed iterative-deepening depth. Honour uci::stop_requested() to
// abort early. Always return a valid bestmove (legal move from current pos).
uci::SearchResult search(const uci::SearchLimits& limits,
                         uci::InfoCallback info_cb);

// Clear the transposition table and any per-game state.
void tt_clear();

// Resize the transposition table (called when "Hash" option changes).
void tt_resize(int megabytes);

} // namespace engine

#endif // CUDA_ENGINE_UCI_H
