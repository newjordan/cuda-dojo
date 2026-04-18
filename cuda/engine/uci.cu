// =============================================================================
// uci.cu — UCI protocol handler implementation (host-only).
//
// This is the stdin/stdout shell of the GPU chess engine. It owns NO board
// state, NO search internals — it parses UCI lines, normalises them, and
// dispatches into engine::* (defined in sibling agent's files; or stubbed
// in uci_stubs.cu for stand-alone testing).
//
// Build:
//   nvcc -O3 -std=c++17 -o uci_test_engine uci.cu uci_stubs.cu
// =============================================================================
#include "uci.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace uci {

// -----------------------------------------------------------------------------
// Module-private state. UCI is single-threaded by spec on the I/O side so a
// plain global is the simplest correct choice.
// -----------------------------------------------------------------------------
namespace {

const std::string STARTPOS_FEN =
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

std::vector<Option> g_options;
PositionState       g_position;
std::atomic<bool>   g_stop_flag{false};
std::atomic<bool>   g_search_running{false};
bool                g_options_initialised = false;
std::thread         g_search_thread;
std::mutex          g_output_mutex;

void init_options_once() {
    if (g_options_initialised) return;
    g_options_initialised = true;
    // MVP: just Hash + Threads. Sized to match the GPU TT agent's defaults.
    g_options.push_back({"Hash",    "spin", 64, 1, 1024, 64});
    g_options.push_back({"Threads", "spin", 1,  1, 1,    1});

    // Default starting position so even a `go` before any `position` cmd works.
    g_position.fen = STARTPOS_FEN;
    g_position.moves.clear();
    g_position.side_to_move = 0;
}

// Side-to-move from FEN: 6 space-separated fields, second is "w" or "b".
int parse_stm_from_fen(const std::string& fen) {
    auto pos = fen.find(' ');
    if (pos == std::string::npos || pos + 1 >= fen.size()) return 0;
    return (fen[pos + 1] == 'b') ? 1 : 0;
}

void reap_finished_search() {
    if (g_search_thread.joinable() &&
        !g_search_running.load(std::memory_order_acquire)) {
        g_search_thread.join();
    }
}

void stop_and_wait_for_search() {
    request_stop();
    if (g_search_thread.joinable()) {
        g_search_thread.join();
    }
    g_search_running.store(false, std::memory_order_release);
}

} // namespace

// -----------------------------------------------------------------------------
// SearchLimits::resolve_movetime — pick a concrete budget in ms.
// -----------------------------------------------------------------------------
int SearchLimits::resolve_movetime(int side_to_move) const {
    if (movetime > 0) return movetime;
    if (depth > 0)    return 0;       // depth-limited; no time cap
    if (infinite)     return 0;       // no time cap

    int my_time = (side_to_move == 0) ? wtime : btime;
    int my_inc  = (side_to_move == 0) ? winc  : binc;
    if (my_time <= 0 && my_inc <= 0) return 1000; // sane default

    // Spec: first cut = min(remaining/30, 5000), plus a fraction of increment.
    int budget = my_time / 30 + my_inc / 2;
    if (budget > 5000) budget = 5000;
    if (budget < 10)   budget = 10;
    return budget;
}

// -----------------------------------------------------------------------------
// Token utilities.
// -----------------------------------------------------------------------------
std::vector<std::string> tokenize(const std::string& line) {
    std::vector<std::string> out;
    std::istringstream iss(line);
    std::string tok;
    while (iss >> tok) out.push_back(tok);
    return out;
}

bool token_equals(const std::string& tok, const char* needle) {
    return tok == needle;
}

// -----------------------------------------------------------------------------
// Output helpers — one line per call, flushed because stdout is line-buffered.
// -----------------------------------------------------------------------------
void send_line(const std::string& line) {
    std::lock_guard<std::mutex> lock(g_output_mutex);
    std::fputs(line.c_str(), stdout);
    std::fputc('\n', stdout);
    std::fflush(stdout);
}

void send_id() {
    send_line(std::string("id name ") + ENGINE_NAME);
    send_line(std::string("id author ") + ENGINE_AUTHOR);
}

void send_options() {
    init_options_once();
    for (const auto& o : g_options) {
        std::ostringstream os;
        os << "option name " << o.name
           << " type " << o.type
           << " default " << o.default_value
           << " min "     << o.min_value
           << " max "     << o.max_value;
        send_line(os.str());
    }
}

void send_uciok()   { send_line("uciok"); }
void send_readyok() { send_line("readyok"); }

void send_info(const SearchInfo& info) {
    std::ostringstream os;
    os << "info";
    if (info.depth > 0)    os << " depth "    << info.depth;
    if (info.seldepth > 0) os << " seldepth " << info.seldepth;
    if (info.nodes > 0)    os << " nodes "    << info.nodes;
    if (info.nps > 0)      os << " nps "      << info.nps;
    if (info.time_ms >= 0) os << " time "     << info.time_ms;
    if (info.mate_in != 0) os << " score mate " << info.mate_in;
    else                   os << " score cp "   << info.score_cp;
    if (!info.pv.empty()) {
        os << " pv";
        for (const auto& m : info.pv) os << " " << m;
    }
    send_line(os.str());
}

void send_bestmove(const SearchResult& result) {
    std::ostringstream os;
    os << "bestmove " << (result.bestmove.empty() ? "0000" : result.bestmove);
    if (!result.ponder.empty()) os << " ponder " << result.ponder;
    send_line(os.str());
}

// -----------------------------------------------------------------------------
// Option store accessors.
// -----------------------------------------------------------------------------
Option* find_option(const std::string& name) {
    init_options_once();
    for (auto& o : g_options) if (o.name == name) return &o;
    return nullptr;
}

void set_option_value(const std::string& name, const std::string& value) {
    Option* o = find_option(name);
    if (!o) {
        std::fprintf(stderr, "uci: unknown option '%s' ignored\n", name.c_str());
        return;
    }
    if (o->type == "spin") {
        try {
            int64_t v = std::stoll(value);
            if (v < o->min_value) v = o->min_value;
            if (v > o->max_value) v = o->max_value;
            o->value = v;
            // Side-effect for Hash: notify TT agent.
            if (o->name == "Hash") engine::tt_resize(static_cast<int>(v));
        } catch (...) {
            std::fprintf(stderr, "uci: bad spin value '%s' for '%s'\n",
                         value.c_str(), name.c_str());
        }
    } else {
        // Future: check/string/combo/button.
        std::fprintf(stderr, "uci: option type '%s' not supported\n",
                     o->type.c_str());
    }
}

const std::vector<Option>& all_options() {
    init_options_once();
    return g_options;
}

const PositionState& current_position() {
    init_options_once();
    return g_position;
}

void request_stop() { g_stop_flag.store(true,  std::memory_order_release); }
bool stop_requested() { return g_stop_flag.load(std::memory_order_acquire); }
void clear_stop()   { g_stop_flag.store(false, std::memory_order_release); }

// -----------------------------------------------------------------------------
// Command handlers.
// -----------------------------------------------------------------------------
void handle_uci() {
    init_options_once();
    send_id();
    send_options();
    send_uciok();
}

void handle_isready() {
    send_readyok();
}

// `setoption name <Name with possibly spaces> value <Value possibly multiword>`
// Accept both forms and tolerate "value" being absent (button-style options).
void handle_setoption(const std::vector<std::string>& tokens) {
    stop_and_wait_for_search();
    // tokens[0] == "setoption"
    size_t i = 1;
    if (i >= tokens.size() || tokens[i] != "name") {
        std::fprintf(stderr, "uci: setoption missing 'name'\n");
        return;
    }
    ++i;
    std::string name;
    while (i < tokens.size() && tokens[i] != "value") {
        if (!name.empty()) name.push_back(' ');
        name += tokens[i];
        ++i;
    }
    std::string value;
    if (i < tokens.size() && tokens[i] == "value") {
        ++i;
        while (i < tokens.size()) {
            if (!value.empty()) value.push_back(' ');
            value += tokens[i];
            ++i;
        }
    }
    if (name.empty()) {
        std::fprintf(stderr, "uci: setoption with empty name\n");
        return;
    }
    set_option_value(name, value);
}

void handle_ucinewgame() {
    stop_and_wait_for_search();
    init_options_once();
    engine::tt_clear();
    g_position.fen = STARTPOS_FEN;
    g_position.moves.clear();
    g_position.side_to_move = 0;
    clear_stop();
}

// `position [startpos | fen <6 fields>] [moves m1 m2 ...]`
void handle_position(const std::vector<std::string>& tokens) {
    stop_and_wait_for_search();
    init_options_once();
    // tokens[0] == "position"
    size_t i = 1;
    if (i >= tokens.size()) {
        std::fprintf(stderr, "uci: position missing args\n");
        return;
    }

    std::string fen;
    if (tokens[i] == "startpos") {
        fen = STARTPOS_FEN;
        ++i;
    } else if (tokens[i] == "fen") {
        ++i;
        // FEN is exactly 6 space-separated fields. Concatenate up to "moves".
        for (int field = 0; field < 6 && i < tokens.size(); ++field) {
            if (tokens[i] == "moves") break;
            if (!fen.empty()) fen.push_back(' ');
            fen += tokens[i];
            ++i;
        }
    } else {
        std::fprintf(stderr, "uci: position expected startpos|fen, got '%s'\n",
                     tokens[i].c_str());
        return;
    }

    std::vector<std::string> moves;
    if (i < tokens.size() && tokens[i] == "moves") {
        ++i;
        for (; i < tokens.size(); ++i) moves.push_back(tokens[i]);
    }

    g_position.fen = fen;
    g_position.moves = moves;
    g_position.side_to_move = parse_stm_from_fen(fen);
    // Side-to-move flips with each applied move (cheap, avoids re-parse).
    if (!moves.empty()) {
        g_position.side_to_move ^= (static_cast<int>(moves.size()) & 1);
    }

    if (!engine::set_position(fen, moves)) {
        std::fprintf(stderr, "uci: engine rejected position\n");
    }
}

// `go [depth N] [movetime MS] [wtime W btime B winc Wi binc Bi movestogo M]
//     [infinite]`
void handle_go(const std::vector<std::string>& tokens) {
    stop_and_wait_for_search();
    init_options_once();
    SearchLimits limits;
    // tokens[0] == "go"
    for (size_t i = 1; i < tokens.size(); ++i) {
        const std::string& t = tokens[i];
        auto next_int = [&]() -> int {
            if (i + 1 >= tokens.size()) return 0;
            try { return std::stoi(tokens[++i]); }
            catch (...) { return 0; }
        };
        if      (t == "depth")     limits.depth     = next_int();
        else if (t == "movetime")  limits.movetime  = next_int();
        else if (t == "wtime")     limits.wtime     = next_int();
        else if (t == "btime")     limits.btime     = next_int();
        else if (t == "winc")      limits.winc      = next_int();
        else if (t == "binc")      limits.binc      = next_int();
        else if (t == "movestogo") limits.movestogo = next_int();
        else if (t == "infinite")  limits.infinite  = true;
        // Silently ignore unknown go subcommands (ponder, mate, nodes, ...).
    }

    clear_stop();
    g_search_running.store(true, std::memory_order_release);
    g_search_thread = std::thread([limits]() {
        try {
            SearchResult result = engine::search(
                limits, [](const SearchInfo& info) { send_info(info); });
            send_bestmove(result);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "uci: search thread exception: %s\n", e.what());
            send_bestmove(SearchResult{"0000", ""});
        } catch (...) {
            std::fprintf(stderr, "uci: search thread unknown exception\n");
            send_bestmove(SearchResult{"0000", ""});
        }
        g_search_running.store(false, std::memory_order_release);
    });
}

void handle_stop() { request_stop(); }

bool handle_quit() {
    stop_and_wait_for_search();
    return true;
}

// -----------------------------------------------------------------------------
// run() — main loop. Read stdin, dispatch, never crash.
// -----------------------------------------------------------------------------
int run() {
    // Critical: UCI requires per-line replies. Default block buffering on a
    // pipe would batch everything until exit, which deadlocks GUIs.
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    std::setvbuf(stderr, nullptr, _IOLBF, 0);

    init_options_once();

    std::string line;
    while (std::getline(std::cin, line)) {
        reap_finished_search();
        // Strip trailing CR (Windows-style line endings) and surrounding ws.
        while (!line.empty() && (line.back() == '\r' || line.back() == ' '
                                 || line.back() == '\t')) {
            line.pop_back();
        }
        if (line.empty()) continue;

        std::vector<std::string> tokens = tokenize(line);
        if (tokens.empty()) continue;
        const std::string& cmd = tokens[0];

        try {
            if      (cmd == "uci")        handle_uci();
            else if (cmd == "isready")    handle_isready();
            else if (cmd == "setoption")  handle_setoption(tokens);
            else if (cmd == "ucinewgame") handle_ucinewgame();
            else if (cmd == "position")   handle_position(tokens);
            else if (cmd == "go")         handle_go(tokens);
            else if (cmd == "stop")       handle_stop();
            else if (cmd == "quit")       { handle_quit(); break; }
            else if (cmd == "ponderhit")  { /* MVP: ignore */ }
            else if (cmd == "debug")      { /* MVP: ignore */ }
            else if (cmd == "register")   { /* MVP: ignore */ }
            else {
                std::fprintf(stderr, "uci: unknown command '%s'\n", cmd.c_str());
            }
        } catch (const std::exception& e) {
            std::fprintf(stderr, "uci: exception handling '%s': %s\n",
                         cmd.c_str(), e.what());
        } catch (...) {
            std::fprintf(stderr, "uci: unknown exception handling '%s'\n",
                         cmd.c_str());
        }
    }
    stop_and_wait_for_search();
    return 0;
}

} // namespace uci

// -----------------------------------------------------------------------------
// main — exists in this TU so the stand-alone uci_test_engine binary works.
// When integrated into the real engine, the real `main` will call uci::run()
// and this one can be excluded with -DUCI_NO_MAIN.
// -----------------------------------------------------------------------------
#ifndef UCI_NO_MAIN
int main(int /*argc*/, char** /*argv*/) {
    return uci::run();
}
#endif
