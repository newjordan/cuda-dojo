// =============================================================================
// eval_service.cu
//
// Batched evaluator service scaffolding.
//
// This file intentionally ships a working fallback backend today:
//   * request buffers are evaluated on-device with engine::d_evaluate
//   * bucket orchestration is explicit and fixed-size
//   * backend selection is resolved once at init time
//   * the device-graph backend now has an explicit per-bucket launch path
//
// CUTLASS / cuBLASDx / graph execution paths are left as isolated extension
// hooks so they can be wired in without redesigning the queue surface.
// =============================================================================
#include "eval_service.cuh"

#include <stddef.h>
#include <string.h>

#include "eval.cuh"

namespace engine {
namespace {

constexpr int kEvalBucketSizes[EVAL_SERVICE_BUCKET_COUNT] = {1, 8, 32, 64, 128};

__host__ __device__ inline bool valid_bucket_idx(int bucket_idx) {
    return bucket_idx >= 0 && bucket_idx < EVAL_SERVICE_BUCKET_COUNT;
}

inline int backend_index(EvalBackendKind backend) {
    switch (backend) {
        case EVAL_BACKEND_PESTO_FALLBACK: return 0;
        case EVAL_BACKEND_CUBLASDX: return 1;
        case EVAL_BACKEND_CUTLASS: return 2;
        case EVAL_BACKEND_DEVICE_GRAPH: return 3;
        default: return -1;
    }
}

inline bool backend_enabled(const EvalServiceConfig& config, EvalBackendKind backend) {
    switch (backend) {
        case EVAL_BACKEND_PESTO_FALLBACK:
            return true;
        case EVAL_BACKEND_CUBLASDX:
            return config.enable_cublasdx_backend;
        case EVAL_BACKEND_CUTLASS:
            return config.enable_cutlass_backend;
        case EVAL_BACKEND_DEVICE_GRAPH:
            return config.enable_graph_backend;
        default:
            return false;
    }
}

inline void zero_host_view(EvalServiceDeviceView* view) {
    if (view == nullptr) {
        return;
    }
    memset(view, 0, sizeof(EvalServiceDeviceView));
    view->active_backend = EVAL_BACKEND_PESTO_FALLBACK;
}

inline void zero_service_report(EvalServiceReport* report) {
    if (report == nullptr) {
        return;
    }
    memset(report, 0, sizeof(EvalServiceReport));
    report->requested_backend = EVAL_BACKEND_PESTO_FALLBACK;
    report->active_backend = EVAL_BACKEND_PESTO_FALLBACK;
}

inline void seed_backend_report(EvalService* service) {
    zero_service_report(&service->report);
    service->report.requested_backend = service->config.preferred_backend;

    for (int i = 0; i < EVAL_SERVICE_BACKEND_COUNT; ++i) {
        EvalBackendKind backend = static_cast<EvalBackendKind>(i);
        EvalBackendState* state = &service->report.backends[i];
        state->enabled = backend_enabled(service->config, backend);
        state->available = (backend == EVAL_BACKEND_PESTO_FALLBACK);
        state->init_status = (backend == EVAL_BACKEND_PESTO_FALLBACK)
                                 ? cudaSuccess
                                 : cudaErrorNotSupported;
    }
}

inline void clear_runtime_report(EvalService* service) {
    const EvalBackendKind requested_backend = service->report.requested_backend;
    const EvalBackendKind active_backend = service->report.active_backend;
    EvalBackendState backend_states[EVAL_SERVICE_BACKEND_COUNT]{};
    for (int i = 0; i < EVAL_SERVICE_BACKEND_COUNT; ++i) {
        backend_states[i] = service->report.backends[i];
    }

    zero_service_report(&service->report);
    service->report.requested_backend = requested_backend;
    service->report.active_backend = active_backend;
    for (int i = 0; i < EVAL_SERVICE_BACKEND_COUNT; ++i) {
        service->report.backends[i] = backend_states[i];
    }
}

inline EvalBackendKind resolve_backend(const EvalService& service) {
    const EvalBackendKind requested = service.config.preferred_backend;
    if (requested == EVAL_BACKEND_PESTO_FALLBACK) {
        return EVAL_BACKEND_PESTO_FALLBACK;
    }

    const EvalBackendKind candidates[EVAL_SERVICE_BACKEND_COUNT] = {
        requested,
        EVAL_BACKEND_DEVICE_GRAPH,
        EVAL_BACKEND_CUTLASS,
        EVAL_BACKEND_CUBLASDX,
    };
    bool visited[EVAL_SERVICE_BACKEND_COUNT]{};

    for (EvalBackendKind candidate : candidates) {
        const int idx = backend_index(candidate);
        if (idx < 0 || visited[idx]) {
            continue;
        }
        visited[idx] = true;

        const EvalBackendState& state = service.report.backends[idx];
        if (state.enabled && state.available) {
            return candidate;
        }
    }

    return EVAL_BACKEND_PESTO_FALLBACK;
}

inline void record_dispatch(EvalService* service,
                            int bucket_idx,
                            EvalBackendKind backend,
                            bool used_fallback,
                            bool backend_unavailable,
                            cudaError_t backend_status,
                            cudaError_t return_status) {
    if (service == nullptr || !valid_bucket_idx(bucket_idx)) {
        return;
    }

    EvalServiceReport* report = &service->report;
    EvalBucketReport* bucket_report = &report->buckets[bucket_idx];
    const int idx = backend_index(backend);

    ++report->total_dispatches;
    ++bucket_report->dispatches;
    bucket_report->last_backend = backend;
    bucket_report->last_backend_status = backend_status;
    bucket_report->last_return_status = return_status;

    if (idx >= 0) {
        ++bucket_report->launches_by_backend[idx];
    }
    if (used_fallback) {
        ++report->total_fallback_dispatches;
        ++bucket_report->fallback_dispatches;
    }
    if (backend_unavailable) {
        ++report->total_unavailable_dispatches;
        ++bucket_report->unavailable_dispatches;
    }
    if (backend_status != cudaSuccess || return_status != cudaSuccess) {
        ++report->total_failed_dispatches;
        ++bucket_report->failed_dispatches;
    }
}

inline void update_bucket_snapshot(EvalService* service,
                                   int bucket_idx,
                                   const EvalBucketCounters& counters) {
    if (service == nullptr || !valid_bucket_idx(bucket_idx)) {
        return;
    }

    EvalBucketReport* bucket_report = &service->report.buckets[bucket_idx];
    bucket_report->last_known_request_count = counters.request_count;
    bucket_report->last_known_result_count = counters.result_count;
    bucket_report->last_known_policy_count = counters.policy_count;
    bucket_report->last_known_dropped_requests = counters.dropped_requests;
}

__device__ inline float score_to_q(Score score) {
    float q = static_cast<float>(score) / 1000.0f;
    if (q > 1.0f) {
        return 1.0f;
    }
    if (q < -1.0f) {
        return -1.0f;
    }
    return q;
}

__global__ void reset_bucket_counters_kernel(EvalServiceDeviceView service) {
    const int bucket_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bucket_idx >= EVAL_SERVICE_BUCKET_COUNT) {
        return;
    }

    EvalBucketCounters* counters = service.buckets[bucket_idx].d_counters;
    counters->request_count = 0;
    counters->result_count = 0;
    counters->policy_count = 0;
    counters->dropped_requests = 0;
}

__global__ void fallback_eval_bucket_kernel(EvalServiceDeviceView service,
                                            int bucket_idx,
                                            uint32_t result_flags) {
    if (!valid_bucket_idx(bucket_idx)) {
        return;
    }

    EvalBucketStorage bucket = service.buckets[bucket_idx];
    const uint32_t request_count = bucket.d_counters->request_count;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= static_cast<int>(request_count) || tid >= bucket.batch_size) {
        return;
    }

    const EvalRequest& request = bucket.d_requests[tid];

    EvalResult result{};
    result.request_id = request.request_id;
    result.state_idx = request.state_idx;
    result.zobrist = request.zobrist;
    result.value_cp = d_evaluate(&request.position);
    result.value_q = score_to_q(result.value_cp);
    result.suggested_move = 0;
    result.suggested_move_logit = 0.0f;
    result.policy_offset = 0;
    result.policy_count = 0;
    result.flags = EVAL_RESULT_VALUE_VALID | result_flags;

    bucket.d_results[tid] = result;

    if ((request.flags & EVAL_REQUEST_NEEDS_POLICY) != 0u) {
        atomicAdd(&bucket.d_counters->dropped_requests, 1u);
        bucket.d_results[tid].flags |= EVAL_RESULT_POLICY_UNAVAILABLE;
    }

    if (threadIdx.x == 0) {
        bucket.d_counters->result_count = request_count;
    }
}

cudaError_t dispatch_cutlass_backend(EvalService* service,
                                     int bucket_idx,
                                     cudaStream_t stream) {
    (void)service;
    (void)bucket_idx;
    (void)stream;
    return cudaErrorNotSupported;
}

cudaError_t dispatch_cublasdx_backend(EvalService* service,
                                      int bucket_idx,
                                      cudaStream_t stream) {
    (void)service;
    (void)bucket_idx;
    (void)stream;
    return cudaErrorNotSupported;
}

cudaError_t dispatch_graph_backend(EvalService* service,
                                   int bucket_idx,
                                   cudaStream_t stream) {
    if (service == nullptr || !valid_bucket_idx(bucket_idx)) {
        return cudaErrorInvalidValue;
    }

    cudaGraphExec_t graph_exec = service->bucket_graph_execs[bucket_idx];
    if (graph_exec == nullptr) {
        return cudaErrorNotSupported;
    }

    return cudaGraphLaunch(graph_exec, stream);
}

cudaError_t dispatch_fallback_backend(EvalService* service,
                                      int bucket_idx,
                                      cudaStream_t stream,
                                      uint32_t result_flags) {
    if (service == nullptr || !valid_bucket_idx(bucket_idx)) {
        return cudaErrorInvalidValue;
    }

    const int threads = 128;
    const int batch_size = service->host_view.buckets[bucket_idx].batch_size;
    const int blocks = (batch_size + threads - 1) / threads;
    fallback_eval_bucket_kernel<<<blocks, threads, 0, stream>>>(service->host_view,
                                                                 bucket_idx,
                                                                 result_flags);
    return cudaGetLastError();
}

inline void zero_bucket_storage(EvalBucketStorage* bucket) {
    bucket->d_requests = nullptr;
    bucket->d_results = nullptr;
    bucket->d_policy = nullptr;
    bucket->d_counters = nullptr;
    bucket->batch_size = 0;
    bucket->max_policy_entries = 0;
}

void destroy_graph_backend(EvalService* service) {
    if (service == nullptr) {
        return;
    }

    for (int bucket_idx = 0; bucket_idx < EVAL_SERVICE_BUCKET_COUNT; ++bucket_idx) {
        if (service->bucket_graph_execs[bucket_idx] != nullptr) {
            cudaGraphExecDestroy(service->bucket_graph_execs[bucket_idx]);
            service->bucket_graph_execs[bucket_idx] = nullptr;
        }
        if (service->bucket_graphs[bucket_idx] != nullptr) {
            cudaGraphDestroy(service->bucket_graphs[bucket_idx]);
            service->bucket_graphs[bucket_idx] = nullptr;
        }
    }
}

cudaError_t init_graph_backend(EvalService* service) {
    if (service == nullptr) {
        return cudaErrorInvalidValue;
    }

    const int graph_backend_idx = backend_index(EVAL_BACKEND_DEVICE_GRAPH);
    EvalBackendState* graph_state = &service->report.backends[graph_backend_idx];
    graph_state->available = false;
    if (!graph_state->enabled) {
        graph_state->init_status = cudaSuccess;
        return cudaSuccess;
    }

    destroy_graph_backend(service);

    for (int bucket_idx = 0; bucket_idx < EVAL_SERVICE_BUCKET_COUNT; ++bucket_idx) {
        cudaError_t status = cudaGraphCreate(&service->bucket_graphs[bucket_idx], 0);
        if (status != cudaSuccess) {
            graph_state->init_status = status;
            destroy_graph_backend(service);
            return status;
        }

        const int threads = 128;
        const int batch_size = service->host_view.buckets[bucket_idx].batch_size;
        const int blocks = (batch_size + threads - 1) / threads;
        EvalServiceDeviceView graph_view = service->host_view;
        int graph_bucket_idx = bucket_idx;
        uint32_t graph_result_flags = EVAL_RESULT_EXECUTED_VIA_GRAPH;
        void* kernel_args[] = {
            &graph_view,
            &graph_bucket_idx,
            &graph_result_flags,
        };

        cudaKernelNodeParams kernel_params{};
        kernel_params.func = reinterpret_cast<void*>(fallback_eval_bucket_kernel);
        kernel_params.gridDim = dim3(blocks, 1, 1);
        kernel_params.blockDim = dim3(threads, 1, 1);
        kernel_params.sharedMemBytes = 0;
        kernel_params.kernelParams = kernel_args;

        cudaGraphNode_t kernel_node = nullptr;
        status = cudaGraphAddKernelNode(&kernel_node,
                                        service->bucket_graphs[bucket_idx],
                                        nullptr,
                                        0,
                                        &kernel_params);
        if (status != cudaSuccess) {
            graph_state->init_status = status;
            destroy_graph_backend(service);
            return status;
        }

        status = cudaGraphInstantiate(&service->bucket_graph_execs[bucket_idx],
                                      service->bucket_graphs[bucket_idx],
                                      nullptr,
                                      nullptr,
                                      0);
        if (status != cudaSuccess) {
            graph_state->init_status = status;
            destroy_graph_backend(service);
            return status;
        }
    }

    graph_state->available = true;
    graph_state->init_status = cudaSuccess;
    return cudaSuccess;
}

} // namespace

const int* eval_service_bucket_sizes() {
    return kEvalBucketSizes;
}

int eval_service_bucket_size(int bucket_idx) {
    if (!valid_bucket_idx(bucket_idx)) {
        return 0;
    }
    return kEvalBucketSizes[bucket_idx];
}

int eval_service_bucket_index_for_size(int batch_size) {
    for (int i = 0; i < EVAL_SERVICE_BUCKET_COUNT; ++i) {
        if (kEvalBucketSizes[i] == batch_size) {
            return i;
        }
    }
    return -1;
}

int eval_service_pick_bucket_for_count(int pending_requests) {
    if (pending_requests <= 0) {
        return 0;
    }
    for (int i = 0; i < EVAL_SERVICE_BUCKET_COUNT; ++i) {
        if (pending_requests <= kEvalBucketSizes[i]) {
            return i;
        }
    }
    return EVAL_SERVICE_BUCKET_COUNT - 1;
}

const char* eval_service_backend_name(EvalBackendKind backend) {
    switch (backend) {
        case EVAL_BACKEND_PESTO_FALLBACK:
            return "pesto-fallback";
        case EVAL_BACKEND_CUBLASDX:
            return "cublasdx";
        case EVAL_BACKEND_CUTLASS:
            return "cutlass";
        case EVAL_BACKEND_DEVICE_GRAPH:
            return "device-graph";
        default:
            return "unknown";
    }
}

EvalServiceConfig eval_service_default_config() {
    EvalServiceConfig config;
    config.preferred_backend = EVAL_BACKEND_PESTO_FALLBACK;
    config.max_policy_moves_per_request = EVAL_SERVICE_DEFAULT_MAX_POLICY_MOVES;
    config.enable_cublasdx_backend = false;
    config.enable_cutlass_backend = false;
    config.enable_graph_backend = false;
    return config;
}

cudaError_t eval_service_upload_view(EvalService* service, cudaStream_t stream) {
    if (service == nullptr || service->d_view == nullptr) {
        return cudaErrorInvalidValue;
    }
    return cudaMemcpyAsync(service->d_view,
                           &service->host_view,
                           sizeof(EvalServiceDeviceView),
                           cudaMemcpyHostToDevice,
                           stream);
}

cudaError_t eval_service_init(EvalService* service, const EvalServiceConfig* config) {
    if (service == nullptr) {
        return cudaErrorInvalidValue;
    }

    service->config = config != nullptr ? *config : eval_service_default_config();
    seed_backend_report(service);
    zero_host_view(&service->host_view);
    service->host_view.active_backend = EVAL_BACKEND_PESTO_FALLBACK;
    service->d_view = nullptr;

    cudaError_t status = cudaMalloc(&service->d_view, sizeof(EvalServiceDeviceView));
    if (status != cudaSuccess) {
        return status;
    }

    for (int bucket_idx = 0; bucket_idx < EVAL_SERVICE_BUCKET_COUNT; ++bucket_idx) {
        EvalBucketStorage* bucket = &service->host_view.buckets[bucket_idx];
        zero_bucket_storage(bucket);

        bucket->batch_size = kEvalBucketSizes[bucket_idx];
        bucket->max_policy_entries =
            bucket->batch_size * service->config.max_policy_moves_per_request;

        status = cudaMalloc(&bucket->d_requests,
                            sizeof(EvalRequest) * static_cast<size_t>(bucket->batch_size));
        if (status != cudaSuccess) {
            eval_service_shutdown(service);
            return status;
        }

        status = cudaMalloc(&bucket->d_results,
                            sizeof(EvalResult) * static_cast<size_t>(bucket->batch_size));
        if (status != cudaSuccess) {
            eval_service_shutdown(service);
            return status;
        }

        status = cudaMalloc(&bucket->d_policy,
                            sizeof(EvalPolicyEntry) *
                                static_cast<size_t>(bucket->max_policy_entries));
        if (status != cudaSuccess) {
            eval_service_shutdown(service);
            return status;
        }

        status = cudaMalloc(&bucket->d_counters, sizeof(EvalBucketCounters));
        if (status != cudaSuccess) {
            eval_service_shutdown(service);
            return status;
        }
    }

    status = init_graph_backend(service);
    if (status != cudaSuccess) {
        status = cudaSuccess;
    }

    service->host_view.active_backend = resolve_backend(*service);
    service->report.active_backend = service->host_view.active_backend;

    status = eval_service_reset(service);
    if (status != cudaSuccess) {
        eval_service_shutdown(service);
        return status;
    }

    status = eval_service_upload_view(service);
    if (status != cudaSuccess) {
        eval_service_shutdown(service);
        return status;
    }

    return cudaSuccess;
}

void eval_service_shutdown(EvalService* service) {
    if (service == nullptr) {
        return;
    }

    destroy_graph_backend(service);

    for (int bucket_idx = 0; bucket_idx < EVAL_SERVICE_BUCKET_COUNT; ++bucket_idx) {
        EvalBucketStorage* bucket = &service->host_view.buckets[bucket_idx];
        if (bucket->d_requests != nullptr) {
            cudaFree(bucket->d_requests);
        }
        if (bucket->d_results != nullptr) {
            cudaFree(bucket->d_results);
        }
        if (bucket->d_policy != nullptr) {
            cudaFree(bucket->d_policy);
        }
        if (bucket->d_counters != nullptr) {
            cudaFree(bucket->d_counters);
        }
        zero_bucket_storage(bucket);
    }

    if (service->d_view != nullptr) {
        cudaFree(service->d_view);
        service->d_view = nullptr;
    }

    zero_host_view(&service->host_view);
    service->config = EvalServiceConfig{};
    zero_service_report(&service->report);
}

cudaError_t eval_service_reset(EvalService* service, cudaStream_t stream) {
    if (service == nullptr) {
        return cudaErrorInvalidValue;
    }

    clear_runtime_report(service);

    const int threads = 32;
    const int blocks = (EVAL_SERVICE_BUCKET_COUNT + threads - 1) / threads;
    reset_bucket_counters_kernel<<<blocks, threads, 0, stream>>>(service->host_view);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        return status;
    }

    return eval_service_upload_view(service, stream);
}

EvalRequest* eval_service_bucket_requests(EvalService* service, int bucket_idx) {
    if (service == nullptr || !valid_bucket_idx(bucket_idx)) {
        return nullptr;
    }
    return service->host_view.buckets[bucket_idx].d_requests;
}

EvalResult* eval_service_bucket_results(EvalService* service, int bucket_idx) {
    if (service == nullptr || !valid_bucket_idx(bucket_idx)) {
        return nullptr;
    }
    return service->host_view.buckets[bucket_idx].d_results;
}

EvalPolicyEntry* eval_service_bucket_policy(EvalService* service, int bucket_idx) {
    if (service == nullptr || !valid_bucket_idx(bucket_idx)) {
        return nullptr;
    }
    return service->host_view.buckets[bucket_idx].d_policy;
}

EvalBucketCounters* eval_service_bucket_counters(EvalService* service, int bucket_idx) {
    if (service == nullptr || !valid_bucket_idx(bucket_idx)) {
        return nullptr;
    }
    return service->host_view.buckets[bucket_idx].d_counters;
}

EvalServiceDeviceView* eval_service_device_view(EvalService* service) {
    if (service == nullptr) {
        return nullptr;
    }
    return service->d_view;
}

bool eval_service_backend_enabled(const EvalService* service, EvalBackendKind backend) {
    if (service == nullptr) {
        return false;
    }
    const int idx = backend_index(backend);
    return idx >= 0 ? service->report.backends[idx].enabled : false;
}

bool eval_service_backend_available(const EvalService* service, EvalBackendKind backend) {
    if (service == nullptr) {
        return false;
    }
    const int idx = backend_index(backend);
    return idx >= 0 ? service->report.backends[idx].available : false;
}

EvalBackendKind eval_service_active_backend(const EvalService* service) {
    if (service == nullptr) {
        return EVAL_BACKEND_PESTO_FALLBACK;
    }
    return service->host_view.active_backend;
}

const EvalServiceReport* eval_service_report(const EvalService* service) {
    if (service == nullptr) {
        return nullptr;
    }
    return &service->report;
}

const EvalBucketReport* eval_service_bucket_report(const EvalService* service,
                                                   int bucket_idx) {
    if (service == nullptr || !valid_bucket_idx(bucket_idx)) {
        return nullptr;
    }
    return &service->report.buckets[bucket_idx];
}

cudaError_t eval_service_snapshot_bucket(EvalService* service,
                                         int bucket_idx,
                                         EvalBucketCounters* out,
                                         cudaStream_t stream) {
    if (service == nullptr || !valid_bucket_idx(bucket_idx)) {
        return cudaErrorInvalidValue;
    }

    EvalBucketCounters counters{};
    cudaError_t status = cudaMemcpyAsync(&counters,
                                         service->host_view.buckets[bucket_idx].d_counters,
                                         sizeof(EvalBucketCounters),
                                         cudaMemcpyDeviceToHost,
                                         stream);
    if (status != cudaSuccess) {
        return status;
    }

    status = cudaStreamSynchronize(stream);
    if (status != cudaSuccess) {
        return status;
    }

    update_bucket_snapshot(service, bucket_idx, counters);
    if (out != nullptr) {
        *out = counters;
    }
    return cudaSuccess;
}

cudaError_t eval_service_dispatch_bucket(EvalService* service,
                                         int bucket_idx,
                                         cudaStream_t stream) {
    if (service == nullptr || !valid_bucket_idx(bucket_idx)) {
        return cudaErrorInvalidValue;
    }

    const EvalBackendKind active_backend = service->host_view.active_backend;
    const EvalBackendKind requested_backend = service->report.requested_backend;
    bool used_fallback = false;
    bool backend_unavailable = false;
    EvalBackendKind executed_backend = active_backend;
    cudaError_t backend_status = cudaSuccess;
    cudaError_t return_status = cudaSuccess;

    switch (active_backend) {
        case EVAL_BACKEND_DEVICE_GRAPH:
            backend_status = dispatch_graph_backend(service, bucket_idx, stream);
            if (backend_status == cudaSuccess) {
                return_status = cudaSuccess;
                break;
            }
            used_fallback = true;
            executed_backend = EVAL_BACKEND_PESTO_FALLBACK;
            return_status = dispatch_fallback_backend(service,
                                                      bucket_idx,
                                                      stream,
                                                      EVAL_RESULT_USED_FALLBACK);
            break;
        case EVAL_BACKEND_CUBLASDX:
            backend_status = dispatch_cublasdx_backend(service, bucket_idx, stream);
            used_fallback = true;
            backend_unavailable = (backend_status == cudaErrorNotSupported);
            executed_backend = EVAL_BACKEND_PESTO_FALLBACK;
            return_status = dispatch_fallback_backend(
                service,
                bucket_idx,
                stream,
                EVAL_RESULT_USED_FALLBACK |
                    (backend_unavailable ? EVAL_RESULT_BACKEND_UNAVAILABLE : 0u));
            break;
        case EVAL_BACKEND_CUTLASS:
            backend_status = dispatch_cutlass_backend(service, bucket_idx, stream);
            used_fallback = true;
            backend_unavailable = (backend_status == cudaErrorNotSupported);
            executed_backend = EVAL_BACKEND_PESTO_FALLBACK;
            return_status = dispatch_fallback_backend(
                service,
                bucket_idx,
                stream,
                EVAL_RESULT_USED_FALLBACK |
                    (backend_unavailable ? EVAL_RESULT_BACKEND_UNAVAILABLE : 0u));
            break;
        case EVAL_BACKEND_PESTO_FALLBACK:
        default:
            backend_status = cudaSuccess;
            return_status = dispatch_fallback_backend(service, bucket_idx, stream, 0u);
            break;
    }

    if (active_backend != requested_backend &&
        requested_backend != EVAL_BACKEND_PESTO_FALLBACK) {
        backend_unavailable = true;
    }

    record_dispatch(service,
                    bucket_idx,
                    executed_backend,
                    used_fallback,
                    backend_unavailable,
                    backend_status,
                    return_status);
    return return_status;
}

} // namespace engine
