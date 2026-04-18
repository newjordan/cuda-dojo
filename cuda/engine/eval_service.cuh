// =============================================================================
// eval_service.cuh
//
// Batched GPU-resident evaluator service scaffolding.
//
// Scope of this file:
//   * Defines fixed batch buckets: 1 / 8 / 32 / 64 / 128
//   * Defines request / result / policy scratch structs
//   * Defines a device-visible service view that search-side schedulers can use
//   * Exposes a small host API to allocate, reset, dispatch, and inspect the
//     service without wiring it into search yet
//
// Current execution model:
//   * One live batch buffer per bucket
//   * Request payloads currently carry a Position snapshot so the fallback
//     path is self-contained
//   * Future integration can replace snapshot usage with state_idx-driven SoA
//     loads without changing request/result IDs or bucket orchestration
//
// Extension points:
//   * CUTLASS / cuBLASDx-backed batched evaluator
//   * Host-instantiated / device-launched graph-backed evaluator buckets
//   * Policy head population through d_policy scratch
// =============================================================================
#ifndef ENGINE_EVAL_SERVICE_CUH
#define ENGINE_EVAL_SERVICE_CUH

#include <stdint.h>
#include <cuda_runtime.h>

#include "engine_types.h"

namespace engine {

constexpr int EVAL_SERVICE_BUCKET_COUNT = 5;
constexpr int EVAL_SERVICE_BACKEND_COUNT = 4;
constexpr int EVAL_SERVICE_MIN_BUCKET   = 1;
constexpr int EVAL_SERVICE_MAX_BUCKET   = 128;
constexpr int EVAL_SERVICE_DEFAULT_MAX_POLICY_MOVES = 64;

enum EvalBackendKind : uint32_t {
    EVAL_BACKEND_PESTO_FALLBACK = 0,
    EVAL_BACKEND_CUBLASDX       = 1,
    EVAL_BACKEND_CUTLASS        = 2,
    EVAL_BACKEND_DEVICE_GRAPH   = 3,
};

enum EvalRequestFlags : uint32_t {
    EVAL_REQUEST_NEEDS_VALUE   = 1u << 0,
    EVAL_REQUEST_NEEDS_POLICY  = 1u << 1,
    EVAL_REQUEST_HIGH_PRIORITY = 1u << 2,
};

enum EvalResultFlags : uint32_t {
    EVAL_RESULT_VALUE_VALID         = 1u << 0,
    EVAL_RESULT_POLICY_VALID        = 1u << 1,
    EVAL_RESULT_USED_FALLBACK       = 1u << 2,
    EVAL_RESULT_BACKEND_UNAVAILABLE = 1u << 3,
    EVAL_RESULT_POLICY_UNAVAILABLE  = 1u << 4,
    EVAL_RESULT_EXECUTED_VIA_GRAPH  = 1u << 5,
    EVAL_RESULT_ERROR               = 1u << 31,
};

// Standalone request payload. `position` keeps the scaffold usable before the
// scheduler-owned SoA state pool lands. `state_idx` is reserved for that later
// integration step and can already be populated by callers today.
struct EvalRequest {
    uint32_t request_id;
    uint32_t state_idx;
    uint64_t zobrist;
    uint32_t flags;
    int16_t  depth_remaining;
    int16_t  ply;
    int16_t  alpha;
    int16_t  beta;
    Position position;
};

// Result payload. Policy is represented indirectly through `policy_offset` and
// `policy_count` into the bucket's `EvalPolicyEntry` scratch.
struct EvalResult {
    uint32_t request_id;
    uint32_t state_idx;
    uint64_t zobrist;
    uint32_t flags;
    Score    value_cp;
    float    value_q;
    Move     suggested_move;
    float    suggested_move_logit;
    uint16_t policy_offset;
    uint16_t policy_count;
};

struct EvalPolicyEntry {
    uint32_t request_id;
    Move     move;
    float    logit;
};

struct EvalBucketCounters {
    uint32_t request_count;
    uint32_t result_count;
    uint32_t policy_count;
    uint32_t dropped_requests;
};

struct EvalBucketStorage {
    EvalRequest*        d_requests;
    EvalResult*         d_results;
    EvalPolicyEntry*    d_policy;
    EvalBucketCounters* d_counters;
    int                 batch_size;
    int                 max_policy_entries;
};

struct EvalServiceConfig {
    EvalBackendKind preferred_backend = EVAL_BACKEND_PESTO_FALLBACK;
    int             max_policy_moves_per_request = EVAL_SERVICE_DEFAULT_MAX_POLICY_MOVES;
    bool            enable_cublasdx_backend = false;
    bool            enable_cutlass_backend = false;
    bool            enable_graph_backend = false;
};

struct EvalBackendState {
    bool        enabled = false;
    bool        available = false;
    cudaError_t init_status = cudaSuccess;
};

struct EvalBucketReport {
    uint64_t        dispatches = 0;
    uint64_t        launches_by_backend[EVAL_SERVICE_BACKEND_COUNT]{};
    uint64_t        fallback_dispatches = 0;
    uint64_t        unavailable_dispatches = 0;
    uint64_t        failed_dispatches = 0;
    uint32_t        last_known_request_count = 0;
    uint32_t        last_known_result_count = 0;
    uint32_t        last_known_policy_count = 0;
    uint32_t        last_known_dropped_requests = 0;
    EvalBackendKind last_backend = EVAL_BACKEND_PESTO_FALLBACK;
    cudaError_t     last_backend_status = cudaSuccess;
    cudaError_t     last_return_status = cudaSuccess;
};

struct EvalServiceReport {
    EvalBackendKind  requested_backend = EVAL_BACKEND_PESTO_FALLBACK;
    EvalBackendKind  active_backend = EVAL_BACKEND_PESTO_FALLBACK;
    EvalBackendState backends[EVAL_SERVICE_BACKEND_COUNT]{};
    uint64_t         total_dispatches = 0;
    uint64_t         total_fallback_dispatches = 0;
    uint64_t         total_unavailable_dispatches = 0;
    uint64_t         total_failed_dispatches = 0;
    EvalBucketReport buckets[EVAL_SERVICE_BUCKET_COUNT]{};
};

struct EvalServiceDeviceView {
    EvalBucketStorage buckets[EVAL_SERVICE_BUCKET_COUNT];
    EvalBackendKind   active_backend;
};

struct EvalService {
    EvalServiceConfig      config{};
    EvalServiceReport      report{};
    EvalServiceDeviceView  host_view{};
    EvalServiceDeviceView* d_view = nullptr;
    cudaGraph_t            bucket_graphs[EVAL_SERVICE_BUCKET_COUNT]{};
    cudaGraphExec_t        bucket_graph_execs[EVAL_SERVICE_BUCKET_COUNT]{};
};

// Bucket helpers.
const int* eval_service_bucket_sizes();
int        eval_service_bucket_size(int bucket_idx);
int        eval_service_bucket_index_for_size(int batch_size);
int        eval_service_pick_bucket_for_count(int pending_requests);
const char* eval_service_backend_name(EvalBackendKind backend);

// Service lifecycle.
EvalServiceConfig eval_service_default_config();
cudaError_t       eval_service_init(EvalService* service,
                                    const EvalServiceConfig* config = nullptr);
void              eval_service_shutdown(EvalService* service);
cudaError_t       eval_service_reset(EvalService* service,
                                     cudaStream_t stream = nullptr);
cudaError_t       eval_service_upload_view(EvalService* service,
                                           cudaStream_t stream = nullptr);

// Bucket accessors for integration work.
EvalRequest*         eval_service_bucket_requests(EvalService* service, int bucket_idx);
EvalResult*          eval_service_bucket_results(EvalService* service, int bucket_idx);
EvalPolicyEntry*     eval_service_bucket_policy(EvalService* service, int bucket_idx);
EvalBucketCounters*  eval_service_bucket_counters(EvalService* service, int bucket_idx);
EvalServiceDeviceView* eval_service_device_view(EvalService* service);
bool                 eval_service_backend_enabled(const EvalService* service,
                                                  EvalBackendKind backend);
bool                 eval_service_backend_available(const EvalService* service,
                                                    EvalBackendKind backend);
EvalBackendKind      eval_service_active_backend(const EvalService* service);
const EvalServiceReport* eval_service_report(const EvalService* service);
const EvalBucketReport*  eval_service_bucket_report(const EvalService* service,
                                                    int bucket_idx);
cudaError_t          eval_service_snapshot_bucket(EvalService* service,
                                                  int bucket_idx,
                                                  EvalBucketCounters* out,
                                                  cudaStream_t stream = nullptr);

// Dispatch one bucket using whichever backend is active. Unsupported backends
// currently fall back to the built-in PeSTO device path and annotate results.
cudaError_t eval_service_dispatch_bucket(EvalService* service,
                                         int bucket_idx,
                                         cudaStream_t stream = nullptr);

// Lightweight device-side helpers for schedulers that want to populate the
// bucket buffers directly.
__host__ __device__ inline bool eval_service_valid_bucket(int bucket_idx) {
    return bucket_idx >= 0 && bucket_idx < EVAL_SERVICE_BUCKET_COUNT;
}

__device__ inline int eval_service_reserve_request(EvalBucketStorage bucket) {
    return static_cast<int>(atomicAdd(&bucket.d_counters->request_count, 1u));
}

__device__ inline int eval_service_reserve_policy(EvalBucketStorage bucket, int count) {
    return static_cast<int>(atomicAdd(&bucket.d_counters->policy_count,
                                      static_cast<uint32_t>(count)));
}

} // namespace engine

#endif // ENGINE_EVAL_SERVICE_CUH
