// =============================================================================
// engine.cu
//
// Host-side bridge for the GPU chess engine:
//   - owns the current Position
//   - parses and applies UCI move lists on top of a FEN
//   - initialises CUDA, Zobrist, TT, and search
//   - runs the UCI loop
//
// All chess search/eval/movegen during `go` stays on the GPU. Host movegen is
// used only to validate and apply external UCI moves into the current position.
// =============================================================================
#include "uci.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "engine_types.h"
#include "eval.cuh"
#include "eval_service.cuh"
#include "movegen.cuh"
#include "make_unmake.cuh"
#include "scheduler.cuh"
#include "search.cuh"
#include "tt.cuh"
#include "zobrist.cuh"

namespace engine {
namespace {

constexpr const char* STARTPOS_FEN =
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
constexpr const char* EVAL_PROBE_FENS[] = {
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
};
constexpr int EVAL_PROBE_REQUESTS = 3;

Position g_current_position{};
bool g_have_position = false;

constexpr uint32_t SCHEDULER_PROBE_TASKS = 4u;

void print_scheduler_probe_json(const SchedulerSnapshot& snapshot, bool passed) {
    std::printf("{\n");
    std::printf("  \"kind\": \"scheduler_runtime_probe\",\n");
    std::printf("  \"status\": \"%s\",\n", passed ? "passed" : "failed");
    std::printf("  \"passed\": %s,\n", passed ? "true" : "false");
    std::printf("  \"config\": {\n");
    std::printf("    \"frontier_capacity\": 16,\n");
    std::printf("    \"eval_request_capacity\": 16,\n");
    std::printf("    \"eval_result_capacity\": 16,\n");
    std::printf("    \"launch_grid\": 1,\n");
    std::printf("    \"launch_block\": 8,\n");
    std::printf("    \"drain_on_frontier_empty\": true\n");
    std::printf("  },\n");
    std::printf("  \"expected\": {\n");
    std::printf("    \"seeded_tasks\": 4,\n");
    std::printf("    \"eval_requests\": 2,\n");
    std::printf("    \"eval_results\": 2,\n");
    std::printf("    \"terminal_results\": 1,\n");
    std::printf("    \"skipped_results\": 1\n");
    std::printf("  },\n");
    std::printf("  \"observed\": {\n");
    std::printf("    \"frontier_depth\": %u,\n", snapshot.frontier_depth);
    std::printf("    \"eval_request_depth\": %u,\n", snapshot.eval_request_depth);
    std::printf("    \"eval_result_depth\": %u,\n", snapshot.eval_result_depth);
    std::printf("    \"active_workers\": %u,\n", snapshot.active_workers);
    std::printf("    \"stop_flag\": %d,\n", snapshot.stop_flag);
    std::printf("    \"counters\": {\n");
    std::printf("      \"frontier_tasks_seeded\": %llu,\n",
                snapshot.counters.frontier_tasks_seeded);
    std::printf("      \"frontier_seed_drops\": %llu,\n",
                snapshot.counters.frontier_seed_drops);
    std::printf("      \"frontier_pops\": %llu,\n",
                snapshot.counters.frontier_pops);
    std::printf("      \"frontier_empty_spins\": %llu,\n",
                snapshot.counters.frontier_empty_spins);
    std::printf("      \"frontier_drain_exits\": %llu,\n",
                snapshot.counters.frontier_drain_exits);
    std::printf("      \"eval_bound_tasks\": %llu,\n",
                snapshot.counters.eval_bound_tasks);
    std::printf("      \"terminal_tasks\": %llu,\n",
                snapshot.counters.terminal_tasks);
    std::printf("      \"fast_path_tasks\": %llu,\n",
                snapshot.counters.fast_path_tasks);
    std::printf("      \"eval_requests_pushed\": %llu,\n",
                snapshot.counters.eval_requests_pushed);
    std::printf("      \"eval_request_drops\": %llu,\n",
                snapshot.counters.eval_request_drops);
    std::printf("      \"eval_results_pushed\": %llu,\n",
                snapshot.counters.eval_results_pushed);
    std::printf("      \"eval_result_drops\": %llu,\n",
                snapshot.counters.eval_result_drops);
    std::printf("      \"terminal_results\": %llu,\n",
                snapshot.counters.terminal_results);
    std::printf("      \"skipped_results\": %llu,\n",
                snapshot.counters.skipped_results);
    std::printf("      \"fast_path_results\": %llu,\n",
                snapshot.counters.fast_path_results);
    std::printf("      \"stop_checks\": %llu,\n",
                snapshot.counters.stop_checks);
    std::printf("      \"stop_breaks\": %llu,\n",
                snapshot.counters.stop_breaks);
    std::printf("      \"active_worker_peak\": %llu\n",
                snapshot.counters.active_worker_peak);
    std::printf("    }\n");
    std::printf("  }\n");
    std::printf("}\n");
}

int print_scheduler_probe_error(const char* stage, cudaError_t err) {
    std::printf("{\n");
    std::printf("  \"kind\": \"scheduler_runtime_probe\",\n");
    std::printf("  \"status\": \"failed\",\n");
    std::printf("  \"passed\": false,\n");
    std::printf("  \"stage\": \"%s\",\n", stage);
    std::printf("  \"cuda_error\": \"%s\"\n", cudaGetErrorString(err));
    std::printf("}\n");
    return 1;
}

int run_scheduler_probe() {
    SchedulerStorage storage{};
    SchedulerConfig config{};
    config.frontier_capacity = 16;
    config.eval_request_capacity = 16;
    config.eval_result_capacity = 16;
    config.drain_on_frontier_empty = 1;

    FrontierTask tasks[SCHEDULER_PROBE_TASKS]{};

    tasks[0].node_id = 1;
    tasks[0].state_slot = 11;
    tasks[0].parent_id = UINT32_MAX;
    tasks[0].root_index = 0;
    tasks[0].zobrist = 0x1111ULL;
    tasks[0].depth = 4;
    tasks[0].ply = 0;
    tasks[0].flags = FRONTIER_TASK_ROOT | FRONTIER_TASK_NEEDS_EVAL;

    tasks[1].node_id = 2;
    tasks[1].state_slot = 22;
    tasks[1].parent_id = 1;
    tasks[1].root_index = 0;
    tasks[1].zobrist = 0x2222ULL;
    tasks[1].depth = 3;
    tasks[1].ply = 1;
    tasks[1].flags = FRONTIER_TASK_NEEDS_EVAL;

    tasks[2].node_id = 3;
    tasks[2].state_slot = 33;
    tasks[2].parent_id = 1;
    tasks[2].root_index = 1;
    tasks[2].zobrist = 0x3333ULL;
    tasks[2].depth = 2;
    tasks[2].ply = 1;
    tasks[2].flags = FRONTIER_TASK_TERMINAL;

    tasks[3].node_id = 4;
    tasks[3].state_slot = 44;
    tasks[3].parent_id = 2;
    tasks[3].root_index = 1;
    tasks[3].zobrist = 0x4444ULL;
    tasks[3].depth = 1;
    tasks[3].ply = 2;
    tasks[3].flags = 0;

    cudaError_t err = scheduler_init(&storage, config);
    if (err != cudaSuccess) {
        return print_scheduler_probe_error("scheduler_init", err);
    }

    err = scheduler_seed_frontier(storage, tasks, SCHEDULER_PROBE_TASKS);
    if (err != cudaSuccess) {
        scheduler_shutdown(&storage);
        return print_scheduler_probe_error("scheduler_seed_frontier", err);
    }

    err = scheduler_launch(storage, dim3(1), dim3(8));
    if (err != cudaSuccess) {
        scheduler_shutdown(&storage);
        return print_scheduler_probe_error("scheduler_launch", err);
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        scheduler_shutdown(&storage);
        return print_scheduler_probe_error("cudaDeviceSynchronize", err);
    }

    SchedulerSnapshot snapshot{};
    err = scheduler_copy_snapshot(storage, &snapshot);
    scheduler_shutdown(&storage);
    if (err != cudaSuccess) {
        return print_scheduler_probe_error("scheduler_copy_snapshot", err);
    }

    const bool passed =
        snapshot.counters.frontier_tasks_seeded == SCHEDULER_PROBE_TASKS &&
        snapshot.counters.frontier_seed_drops == 0 &&
        snapshot.counters.frontier_pops == SCHEDULER_PROBE_TASKS &&
        snapshot.counters.eval_bound_tasks == 2 &&
        snapshot.counters.terminal_tasks == 1 &&
        snapshot.counters.fast_path_tasks == 2 &&
        snapshot.counters.eval_requests_pushed == 2 &&
        snapshot.counters.eval_request_drops == 0 &&
        snapshot.counters.eval_results_pushed == 2 &&
        snapshot.counters.eval_result_drops == 0 &&
        snapshot.counters.terminal_results == 1 &&
        snapshot.counters.skipped_results == 1 &&
        snapshot.counters.fast_path_results == 2 &&
        snapshot.counters.frontier_drain_exits >= 1 &&
        snapshot.counters.stop_breaks == 0 &&
        snapshot.counters.active_worker_peak >= 1 &&
        snapshot.frontier_depth == 0 &&
        snapshot.eval_request_depth == 2 &&
        snapshot.eval_result_depth == 2 &&
        snapshot.active_workers == 0 &&
        snapshot.stop_flag == 0;

    print_scheduler_probe_json(snapshot, passed);
    return passed ? 0 : 2;
}

void print_eval_service_probe_json(const EvalServiceReport& report,
                                   const EvalBucketCounters& counters,
                                   const EvalResult* results,
                                   const Score* expected,
                                   int bucket_idx,
                                   int request_count,
                                   bool passed) {
    const EvalBucketReport* bucket_report = eval_service_bucket_report(nullptr, bucket_idx);
    (void)bucket_report;
    std::printf("{\n");
    std::printf("  \"kind\": \"eval_service_probe\",\n");
    std::printf("  \"status\": \"%s\",\n", passed ? "passed" : "failed");
    std::printf("  \"passed\": %s,\n", passed ? "true" : "false");
    std::printf("  \"requested_backend\": \"%s\",\n",
                eval_service_backend_name(report.requested_backend));
    std::printf("  \"active_backend\": \"%s\",\n",
                eval_service_backend_name(report.active_backend));
    std::printf("  \"bucket_idx\": %d,\n", bucket_idx);
    std::printf("  \"request_count\": %d,\n", request_count);
    std::printf("  \"counters\": {\n");
    std::printf("    \"request_count\": %u,\n", counters.request_count);
    std::printf("    \"result_count\": %u,\n", counters.result_count);
    std::printf("    \"policy_count\": %u,\n", counters.policy_count);
    std::printf("    \"dropped_requests\": %u\n", counters.dropped_requests);
    std::printf("  },\n");
    std::printf("  \"results\": [\n");
    for (int i = 0; i < request_count; ++i) {
        std::printf(
            "    {\"request_id\": %u, \"value_cp\": %d, \"expected_value_cp\": %d, "
            "\"flags\": %u, \"policy_count\": %u}%s\n",
            results[i].request_id,
            static_cast<int>(results[i].value_cp),
            static_cast<int>(expected[i]),
            results[i].flags,
            results[i].policy_count,
            (i + 1 < request_count) ? "," : "");
    }
    std::printf("  ],\n");
    std::printf("  \"backend_report\": {\n");
    std::printf("    \"total_dispatches\": %llu,\n",
                static_cast<unsigned long long>(report.total_dispatches));
    std::printf("    \"total_fallback_dispatches\": %llu,\n",
                static_cast<unsigned long long>(report.total_fallback_dispatches));
    std::printf("    \"total_unavailable_dispatches\": %llu,\n",
                static_cast<unsigned long long>(report.total_unavailable_dispatches));
    std::printf("    \"total_failed_dispatches\": %llu\n",
                static_cast<unsigned long long>(report.total_failed_dispatches));
    std::printf("  }\n");
    std::printf("}\n");
}

int print_eval_service_probe_error(const char* stage, cudaError_t err) {
    std::printf("{\n");
    std::printf("  \"kind\": \"eval_service_probe\",\n");
    std::printf("  \"status\": \"failed\",\n");
    std::printf("  \"passed\": false,\n");
    std::printf("  \"stage\": \"%s\",\n", stage);
    std::printf("  \"cuda_error\": \"%s\"\n", cudaGetErrorString(err));
    std::printf("}\n");
    return 1;
}

int run_eval_service_probe() {
    if (!zobrist_initialized()) {
        init_zobrist();
    }

    EvalService service{};
    EvalServiceConfig config = eval_service_default_config();
    config.preferred_backend = EVAL_BACKEND_DEVICE_GRAPH;
    config.enable_graph_backend = true;

    cudaError_t err = eval_service_init(&service, &config);
    if (err != cudaSuccess) {
        return print_eval_service_probe_error("eval_service_init", err);
    }

    const int bucket_idx = eval_service_pick_bucket_for_count(EVAL_PROBE_REQUESTS);
    EvalRequest host_requests[eval_service_bucket_size(bucket_idx)]{};
    EvalResult host_results[eval_service_bucket_size(bucket_idx)]{};
    Score expected[EVAL_PROBE_REQUESTS]{};
    EvalBucketCounters counters{};

    for (int i = 0; i < EVAL_PROBE_REQUESTS; ++i) {
        Position pos{};
        parse_fen(EVAL_PROBE_FENS[i], &pos);
        host_requests[i].request_id = static_cast<uint32_t>(i + 1);
        host_requests[i].state_idx = static_cast<uint32_t>(100 + i);
        host_requests[i].zobrist = zobrist_full(pos);
        host_requests[i].flags = EVAL_REQUEST_NEEDS_VALUE |
                                 (i == 0 ? EVAL_REQUEST_NEEDS_POLICY : 0u);
        host_requests[i].position = pos;
        expected[i] = h_evaluate(&pos);
    }

    err = cudaMemcpy(eval_service_bucket_requests(&service, bucket_idx),
                     host_requests,
                     sizeof(host_requests),
                     cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        eval_service_shutdown(&service);
        return print_eval_service_probe_error("upload_requests", err);
    }

    counters.request_count = EVAL_PROBE_REQUESTS;
    counters.result_count = 0;
    counters.policy_count = 0;
    counters.dropped_requests = 0;
    err = cudaMemcpy(eval_service_bucket_counters(&service, bucket_idx),
                     &counters,
                     sizeof(counters),
                     cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        eval_service_shutdown(&service);
        return print_eval_service_probe_error("upload_counters", err);
    }

    err = eval_service_dispatch_bucket(&service, bucket_idx);
    if (err != cudaSuccess) {
        eval_service_shutdown(&service);
        return print_eval_service_probe_error("dispatch_bucket", err);
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        eval_service_shutdown(&service);
        return print_eval_service_probe_error("cudaDeviceSynchronize", err);
    }

    err = eval_service_snapshot_bucket(&service, bucket_idx, &counters);
    if (err != cudaSuccess) {
        eval_service_shutdown(&service);
        return print_eval_service_probe_error("snapshot_bucket", err);
    }

    err = cudaMemcpy(host_results,
                     eval_service_bucket_results(&service, bucket_idx),
                     sizeof(host_results),
                     cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        eval_service_shutdown(&service);
        return print_eval_service_probe_error("download_results", err);
    }

    const EvalServiceReport* report = eval_service_report(&service);
    bool passed =
        report != nullptr &&
        report->active_backend == EVAL_BACKEND_DEVICE_GRAPH &&
        counters.request_count == EVAL_PROBE_REQUESTS &&
        counters.result_count == EVAL_PROBE_REQUESTS &&
        counters.dropped_requests == 1 &&
        report->total_dispatches == 1 &&
        report->total_failed_dispatches == 0;

    for (int i = 0; i < EVAL_PROBE_REQUESTS; ++i) {
        const uint32_t flags = host_results[i].flags;
        passed = passed &&
            host_results[i].value_cp == expected[i] &&
            ((flags & EVAL_RESULT_VALUE_VALID) != 0u) &&
            ((flags & EVAL_RESULT_EXECUTED_VIA_GRAPH) != 0u);
        if (i == 0) {
            passed = passed && ((flags & EVAL_RESULT_POLICY_UNAVAILABLE) != 0u);
        }
    }

    print_eval_service_probe_json(*report, counters, host_results, expected,
                                  bucket_idx, EVAL_PROBE_REQUESTS, passed);
    eval_service_shutdown(&service);
    return passed ? 0 : 2;
}

void print_search_runtime_probe_json(const SearchRuntimeReport& report,
                                     const uci::SearchResult& result,
                                     bool passed) {
    std::printf("{\n");
    std::printf("  \"kind\": \"search_runtime_probe\",\n");
    std::printf("  \"status\": \"%s\",\n", passed ? "passed" : "failed");
    std::printf("  \"passed\": %s,\n", passed ? "true" : "false");
    std::printf("  \"bestmove\": \"%s\",\n", result.bestmove.c_str());
    std::printf("  \"runtime\": {\n");
    std::printf("    \"runtime_enabled\": %s,\n", report.runtime_enabled ? "true" : "false");
    std::printf("    \"runtime_used\": %s,\n", report.runtime_used ? "true" : "false");
    std::printf("    \"root_move_count\": %u,\n", report.root_move_count);
    std::printf("    \"reordered_root_moves\": %u,\n", report.reordered_root_moves);
    std::printf("    \"scheduler_frontier_seeded\": %u,\n", report.scheduler_frontier_seeded);
    std::printf("    \"scheduler_frontier_pops\": %u,\n", report.scheduler_frontier_pops);
    std::printf("    \"scheduler_eval_requests\": %u,\n", report.scheduler_eval_requests);
    std::printf("    \"scheduler_eval_request_depth\": %u,\n", report.scheduler_eval_request_depth);
    std::printf("    \"eval_bucket_idx\": %u,\n", report.eval_bucket_idx);
    std::printf("    \"eval_bucket_size\": %u,\n", report.eval_bucket_size);
    std::printf("    \"eval_request_count\": %u,\n", report.eval_request_count);
    std::printf("    \"eval_result_count\": %u,\n", report.eval_result_count);
    std::printf("    \"eval_dropped_requests\": %u,\n", report.eval_dropped_requests);
    std::printf("    \"eval_backend\": \"%s\",\n",
                eval_service_backend_name(static_cast<EvalBackendKind>(report.eval_backend)));
    std::printf("    \"eval_total_dispatches\": %llu,\n",
                static_cast<unsigned long long>(report.eval_total_dispatches));
    std::printf("    \"eval_fallback_dispatches\": %llu,\n",
                static_cast<unsigned long long>(report.eval_fallback_dispatches));
    std::printf("    \"eval_failed_dispatches\": %llu,\n",
                static_cast<unsigned long long>(report.eval_failed_dispatches));
    std::printf("    \"top_order\": [\n");
    for (int i = 0; i < 8; ++i) {
        if (report.top_order_moves[i] == 0u) break;
        char buf[8] = {0};
        move_to_uci(static_cast<Move>(report.top_order_moves[i]), buf);
        const bool last = (i == 7) || (report.top_order_moves[i + 1] == 0u);
        std::printf("      {\"move\": \"%s\", \"score\": %d}%s\n",
                    buf, report.top_order_scores[i], last ? "" : ",");
    }
    std::printf("    ]\n");
    std::printf("  }\n");
    std::printf("}\n");
}

int run_search_runtime_probe() {
    set_position(STARTPOS_FEN, {});
    uci::SearchLimits limits;
    limits.depth = 2;
    const uci::SearchResult result = search(limits, nullptr);
    const SearchRuntimeReport& report = search_runtime_report();
    const bool passed =
        report.runtime_enabled &&
        report.runtime_used &&
        report.root_move_count > 0 &&
        report.scheduler_frontier_seeded == report.root_move_count &&
        report.scheduler_frontier_pops == report.root_move_count &&
        report.scheduler_eval_requests == report.root_move_count &&
        report.eval_request_count == report.root_move_count &&
        report.eval_result_count == report.root_move_count &&
        report.eval_failed_dispatches == 0 &&
        !result.bestmove.empty() &&
        result.bestmove != "0000";
    print_search_runtime_probe_json(report, result, passed);
    return passed ? 0 : 2;
}

bool decode_uci_move(const Position& pos, const std::string& uci, Move* out_move) {
    Move moves[MAX_MOVES];
    int n = generate_moves(&pos, moves);
    for (int i = 0; i < n; ++i) {
        Position child = pos;
        h_make_move(&child, moves[i]);
        if (in_check(&child, 1 - child.side)) continue;

        char buf[8] = {0};
        move_to_uci(moves[i], buf);
        if (uci == buf) {
            *out_move = moves[i];
            return true;
        }
    }
    return false;
}

} // namespace

bool set_position(const std::string& fen,
                  const std::vector<std::string>& moves) {
    Position pos{};
    parse_fen(fen.c_str(), &pos);
    if (pos.kingPos[WHITE_SIDE] < 0 || pos.kingPos[BLACK_SIDE] < 0) {
        std::fprintf(stderr, "engine: invalid FEN (missing king): %s\n", fen.c_str());
        return false;
    }

    for (const std::string& uci_move : moves) {
        Move mv = 0;
        if (!decode_uci_move(pos, uci_move, &mv)) {
            std::fprintf(stderr, "engine: illegal move in position sequence: %s\n",
                         uci_move.c_str());
            return false;
        }
        h_make_move(&pos, mv);
    }

    g_current_position = pos;
    g_have_position = true;
    return true;
}

uci::SearchResult search(const uci::SearchLimits& limits,
                         uci::InfoCallback info_cb) {
    if (!g_have_position) {
        set_position(STARTPOS_FEN, {});
    }
    return search_root(g_current_position, limits, info_cb);
}

} // namespace engine

int main(int argc, char** argv) {
    cudaError_t err = cudaSetDevice(0);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "engine: cudaSetDevice failed: %s\n",
                     cudaGetErrorString(err));
        return 1;
    }

    engine::init_zobrist();

    if (argc > 1 && std::strcmp(argv[1], "--scheduler-probe") == 0) {
        const int rc = engine::run_scheduler_probe();
        cudaDeviceReset();
        return rc;
    }
    if (argc > 1 && std::strcmp(argv[1], "--eval-service-probe") == 0) {
        const int rc = engine::run_eval_service_probe();
        cudaDeviceReset();
        return rc;
    }

    if (engine::tt_init(64) == 0) {
        std::fprintf(stderr, "engine: tt_init failed\n");
        return 1;
    }

    engine::search_init();
    engine::set_position("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", {});

    if (argc > 1 && std::strcmp(argv[1], "--search-runtime-probe") == 0) {
        const int rc = engine::run_search_runtime_probe();
        engine::search_shutdown();
        engine::tt_free();
        cudaDeviceReset();
        return rc;
    }

    int rc = uci::run();

    engine::search_shutdown();
    engine::tt_free();
    cudaDeviceReset();
    return rc;
}
