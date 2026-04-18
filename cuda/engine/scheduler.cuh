#ifndef ENGINE_SCHEDULER_CUH
#define ENGINE_SCHEDULER_CUH

#include <stdint.h>
#include <cuda_runtime.h>

#include "engine_types.h"
#include "queues.cuh"

namespace engine {

// Frontier tasks are scheduler-owned search nodes. They intentionally point at
// future SoA state storage by index rather than embedding a Position here.
// That keeps the queue payload small and avoids baking an AoS model into the
// runtime scaffold.
enum FrontierTaskFlags : uint16_t {
    FRONTIER_TASK_ROOT         = 1u << 0,
    FRONTIER_TASK_NEEDS_EVAL   = 1u << 1,
    FRONTIER_TASK_TERMINAL     = 1u << 2,
    FRONTIER_TASK_RETRY_LATER  = 1u << 3,
};

enum SchedulerEvalRequestFlags : uint16_t {
    SCHED_EVAL_REQUEST_VALUE_ONLY   = 1u << 0,
    SCHED_EVAL_REQUEST_POLICY_VALUE = 1u << 1,
};

enum SchedulerEvalResultStatus : uint16_t {
    SCHED_EVAL_RESULT_PENDING  = 0,
    SCHED_EVAL_RESULT_READY    = 1,
    SCHED_EVAL_RESULT_TERMINAL = 2,
    SCHED_EVAL_RESULT_SKIPPED  = 3,
    SCHED_EVAL_RESULT_DROPPED  = 4,
};

enum SchedulerEvalResultFlags : uint16_t {
    SCHED_EVAL_RESULT_FROM_FAST_PATH = 1u << 0,
};

struct FrontierTask {
    uint32_t node_id = 0;          // Stable search-node handle.
    uint32_t state_slot = 0;       // Index into future SoA position storage.
    uint32_t parent_id = 0;        // UINT32_MAX is a natural root sentinel.
    uint32_t root_index = 0;       // Root-move shard for later reduction.
    uint64_t zobrist = 0;
    Move incoming_move = 0;
    Score alpha = -INF_SCORE;
    Score beta = INF_SCORE;
    uint16_t depth = 0;
    uint16_t ply = 0;
    uint16_t flags = 0;
    uint16_t reserved = 0;
};

struct SchedulerEvalRequest {
    uint32_t node_id = 0;
    uint32_t state_slot = 0;
    uint64_t zobrist = 0;
    uint16_t depth = 0;
    uint16_t ply = 0;
    uint16_t move_count_hint = 0;
    uint16_t flags = 0;
    uint16_t reserved = 0;
};

struct SchedulerEvalResult {
    uint32_t node_id = 0;
    uint32_t state_slot = 0;
    uint64_t zobrist = 0;
    Score value = 0;
    Move best_move = 0;
    uint16_t status = SCHED_EVAL_RESULT_PENDING;
    uint16_t flags = 0;
};

struct SchedulerCounters {
    unsigned long long frontier_tasks_seeded = 0;
    unsigned long long frontier_seed_drops = 0;
    unsigned long long frontier_pops = 0;
    unsigned long long frontier_empty_spins = 0;
    unsigned long long frontier_drain_exits = 0;
    unsigned long long eval_bound_tasks = 0;
    unsigned long long terminal_tasks = 0;
    unsigned long long fast_path_tasks = 0;
    unsigned long long eval_requests_pushed = 0;
    unsigned long long eval_request_drops = 0;
    unsigned long long eval_results_pushed = 0;
    unsigned long long eval_result_drops = 0;
    unsigned long long terminal_results = 0;
    unsigned long long skipped_results = 0;
    unsigned long long fast_path_results = 0;
    unsigned long long stop_checks = 0;
    unsigned long long stop_breaks = 0;
    unsigned long long active_worker_peak = 0;
};

struct SchedulerQueues {
    DeviceQueue<FrontierTask> frontier;
    DeviceQueue<SchedulerEvalRequest> eval_requests;
    DeviceQueue<SchedulerEvalResult> eval_results;
};

struct SchedulerRuntime {
    SchedulerQueues queues;
    int* stop_flag = nullptr;              // Host-owned stop signal.
    uint32_t* active_workers = nullptr;    // Threads currently inside kernel.
    SchedulerCounters* counters = nullptr; // Shared runtime diagnostics.
    uint32_t idle_backoff = 0;             // Reserved for later tuning.
    uint32_t drain_on_frontier_empty = 0;  // Probe mode: exit once work drains.
};

struct SchedulerConfig {
    uint32_t frontier_capacity = 4096;
    uint32_t eval_request_capacity = 2048;
    uint32_t eval_result_capacity = 2048;
    uint32_t idle_backoff = 0;
    uint32_t drain_on_frontier_empty = 0;
};

struct SchedulerSnapshot {
    SchedulerCounters counters{};
    uint32_t frontier_depth = 0;
    uint32_t eval_request_depth = 0;
    uint32_t eval_result_depth = 0;
    uint32_t active_workers = 0;
    int stop_flag = 0;
};

struct SchedulerStorage {
    QueueAllocation<FrontierTask> frontier;
    QueueAllocation<SchedulerEvalRequest> eval_requests;
    QueueAllocation<SchedulerEvalResult> eval_results;
    int* d_stop_flag = nullptr;
    uint32_t* d_active_workers = nullptr;
    SchedulerCounters* d_counters = nullptr;
    SchedulerConfig config{};

    bool initialized() const {
        return frontier.view.valid() &&
               eval_requests.view.valid() &&
               eval_results.view.valid() &&
               d_stop_flag != nullptr &&
               d_active_workers != nullptr &&
               d_counters != nullptr;
    }
};

// Allocate and initialize queue/control storage for the persistent scheduler.
cudaError_t scheduler_init(SchedulerStorage* storage,
                           const SchedulerConfig& config = SchedulerConfig(),
                           cudaStream_t stream = 0);

// Reset queues and runtime counters to an empty search session.
cudaError_t scheduler_reset(SchedulerStorage* storage,
                            cudaStream_t stream = 0);

// Flip the device-visible stop flag. The persistent kernel checks this on each
// frontier loop iteration.
cudaError_t scheduler_set_stop(const SchedulerStorage& storage,
                               int stop_value,
                               cudaStream_t stream = 0);

// Seed host-prepared frontier work into the device queue. Intended for runtime
// bring-up and verification until the full engine path produces frontier tasks.
cudaError_t scheduler_seed_frontier(const SchedulerStorage& storage,
                                    const FrontierTask* tasks,
                                    uint32_t count,
                                    cudaStream_t stream = 0);

// Copy counters back to the host for debugging / profiling.
cudaError_t scheduler_copy_counters(const SchedulerStorage& storage,
                                    SchedulerCounters* out,
                                    cudaStream_t stream = 0);

// Copy counters plus queue/worker state back to the host.
cudaError_t scheduler_copy_snapshot(const SchedulerStorage& storage,
                                    SchedulerSnapshot* out,
                                    cudaStream_t stream = 0);

// Tear down queue/control allocations. Safe to call on partially initialized
// storage.
void scheduler_shutdown(SchedulerStorage* storage);

// Materialize the device view consumed by scheduler_kernel().
SchedulerRuntime scheduler_runtime(const SchedulerStorage& storage);

// Launch the persistent scheduler kernel with the supplied launch shape.
cudaError_t scheduler_launch(const SchedulerStorage& storage,
                             dim3 grid,
                             dim3 block,
                             cudaStream_t stream = 0);

// Persistent-kernel placeholder:
// - drains frontier tasks
// - routes eval-bound nodes into eval_requests
// - emits terminal / fast-path placeholders into eval_results
//
// Future integration work will replace the fast path with TT probe,
// movegen/legality, expansion, and backup logic.
__global__ void scheduler_kernel(SchedulerRuntime runtime);

} // namespace engine

#endif // ENGINE_SCHEDULER_CUH
