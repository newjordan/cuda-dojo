#include "scheduler.cuh"

namespace engine {
namespace {

template <typename T>
__global__ void init_queue_slots_kernel(QueueSlot<T>* slots, uint32_t capacity) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= capacity) return;
    slots[idx].sequence = idx;
}

__global__ void seed_frontier_tasks_kernel(DeviceQueue<FrontierTask> frontier,
                                           SchedulerCounters* counters,
                                           const FrontierTask* tasks,
                                           uint32_t count) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    if (frontier.try_push(tasks[idx])) {
        if (counters != nullptr) atomicAdd(&counters->frontier_tasks_seeded, 1ULL);
    } else {
        if (counters != nullptr) atomicAdd(&counters->frontier_seed_drops, 1ULL);
    }
}

template <typename T>
cudaError_t init_queue(QueueAllocation<T>* queue,
                       uint32_t capacity,
                       cudaStream_t stream) {
    if (queue == nullptr) return cudaErrorInvalidValue;
    if (!queue_capacity_is_valid(capacity)) return cudaErrorInvalidValue;

    cudaError_t err = cudaMalloc(&queue->d_slots, capacity * sizeof(QueueSlot<T>));
    if (err != cudaSuccess) return err;

    err = cudaMalloc(&queue->d_enqueue_pos, sizeof(uint32_t));
    if (err != cudaSuccess) return err;

    err = cudaMalloc(&queue->d_dequeue_pos, sizeof(uint32_t));
    if (err != cudaSuccess) return err;

    err = cudaMemsetAsync(queue->d_enqueue_pos, 0, sizeof(uint32_t), stream);
    if (err != cudaSuccess) return err;

    err = cudaMemsetAsync(queue->d_dequeue_pos, 0, sizeof(uint32_t), stream);
    if (err != cudaSuccess) return err;

    const dim3 block(256);
    const dim3 grid((capacity + block.x - 1u) / block.x);
    init_queue_slots_kernel<<<grid, block, 0, stream>>>(queue->d_slots, capacity);
    err = cudaGetLastError();
    if (err != cudaSuccess) return err;

    queue->view.slots = queue->d_slots;
    queue->view.enqueue_pos = queue->d_enqueue_pos;
    queue->view.dequeue_pos = queue->d_dequeue_pos;
    queue->view.capacity = capacity;
    queue->view.mask = capacity - 1u;
    return cudaSuccess;
}

template <typename T>
cudaError_t reset_queue(QueueAllocation<T>* queue, cudaStream_t stream) {
    if (queue == nullptr || !queue->view.valid()) return cudaErrorInvalidValue;

    cudaError_t err = cudaMemsetAsync(queue->d_enqueue_pos, 0, sizeof(uint32_t), stream);
    if (err != cudaSuccess) return err;

    err = cudaMemsetAsync(queue->d_dequeue_pos, 0, sizeof(uint32_t), stream);
    if (err != cudaSuccess) return err;

    const dim3 block(256);
    const dim3 grid((queue->view.capacity + block.x - 1u) / block.x);
    init_queue_slots_kernel<<<grid, block, 0, stream>>>(queue->d_slots,
                                                         queue->view.capacity);
    return cudaGetLastError();
}

template <typename T>
void free_queue(QueueAllocation<T>* queue) {
    if (queue == nullptr) return;

    if (queue->d_slots != nullptr) cudaFree(queue->d_slots);
    if (queue->d_enqueue_pos != nullptr) cudaFree(queue->d_enqueue_pos);
    if (queue->d_dequeue_pos != nullptr) cudaFree(queue->d_dequeue_pos);

    *queue = QueueAllocation<T>{};
}

__device__ __forceinline__ void counter_inc(unsigned long long* counter) {
    if (counter != nullptr) atomicAdd(counter, 1ULL);
}

__device__ __forceinline__ void counter_max(unsigned long long* counter,
                                            unsigned long long value) {
    if (counter != nullptr) atomicMax(counter, value);
}

template <typename T>
cudaError_t copy_device_scalar(const T* src, T* dst, cudaStream_t stream) {
    if (src == nullptr || dst == nullptr) return cudaErrorInvalidValue;
    return cudaMemcpyAsync(dst,
                           src,
                           sizeof(T),
                           cudaMemcpyDeviceToHost,
                           stream);
}

template <typename T>
cudaError_t copy_queue_depth(const QueueAllocation<T>& queue,
                             uint32_t* out,
                             cudaStream_t stream) {
    if (out == nullptr || !queue.view.valid()) return cudaErrorInvalidValue;

    uint32_t enqueue = 0;
    uint32_t dequeue = 0;
    cudaError_t err = copy_device_scalar(queue.d_enqueue_pos, &enqueue, stream);
    if (err != cudaSuccess) return err;
    err = copy_device_scalar(queue.d_dequeue_pos, &dequeue, stream);
    if (err != cudaSuccess) return err;
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) return err;

    *out = enqueue - dequeue;
    return cudaSuccess;
}

} // namespace

cudaError_t scheduler_init(SchedulerStorage* storage,
                           const SchedulerConfig& config,
                           cudaStream_t stream) {
    if (storage == nullptr) return cudaErrorInvalidValue;

    scheduler_shutdown(storage);
    storage->config = config;

    cudaError_t err = init_queue(&storage->frontier,
                                 config.frontier_capacity,
                                 stream);
    if (err != cudaSuccess) {
        scheduler_shutdown(storage);
        return err;
    }

    err = init_queue(&storage->eval_requests,
                     config.eval_request_capacity,
                     stream);
    if (err != cudaSuccess) {
        scheduler_shutdown(storage);
        return err;
    }

    err = init_queue(&storage->eval_results,
                     config.eval_result_capacity,
                     stream);
    if (err != cudaSuccess) {
        scheduler_shutdown(storage);
        return err;
    }

    err = cudaMalloc(&storage->d_stop_flag, sizeof(int));
    if (err != cudaSuccess) {
        scheduler_shutdown(storage);
        return err;
    }

    err = cudaMalloc(&storage->d_active_workers, sizeof(uint32_t));
    if (err != cudaSuccess) {
        scheduler_shutdown(storage);
        return err;
    }

    err = cudaMalloc(&storage->d_counters, sizeof(SchedulerCounters));
    if (err != cudaSuccess) {
        scheduler_shutdown(storage);
        return err;
    }

    err = scheduler_reset(storage, stream);
    if (err != cudaSuccess) {
        scheduler_shutdown(storage);
        return err;
    }

    return cudaStreamSynchronize(stream);
}

cudaError_t scheduler_reset(SchedulerStorage* storage, cudaStream_t stream) {
    if (storage == nullptr || !storage->initialized()) return cudaErrorInvalidValue;

    cudaError_t err = reset_queue(&storage->frontier, stream);
    if (err != cudaSuccess) return err;

    err = reset_queue(&storage->eval_requests, stream);
    if (err != cudaSuccess) return err;

    err = reset_queue(&storage->eval_results, stream);
    if (err != cudaSuccess) return err;

    err = cudaMemsetAsync(storage->d_stop_flag, 0, sizeof(int), stream);
    if (err != cudaSuccess) return err;

    err = cudaMemsetAsync(storage->d_active_workers, 0, sizeof(uint32_t), stream);
    if (err != cudaSuccess) return err;

    err = cudaMemsetAsync(storage->d_counters, 0, sizeof(SchedulerCounters), stream);
    if (err != cudaSuccess) return err;

    return cudaStreamSynchronize(stream);
}

cudaError_t scheduler_set_stop(const SchedulerStorage& storage,
                               int stop_value,
                               cudaStream_t stream) {
    if (!storage.initialized()) return cudaErrorInvalidValue;
    return cudaMemcpyAsync(storage.d_stop_flag,
                           &stop_value,
                           sizeof(int),
                           cudaMemcpyHostToDevice,
                           stream);
}

cudaError_t scheduler_seed_frontier(const SchedulerStorage& storage,
                                    const FrontierTask* tasks,
                                    uint32_t count,
                                    cudaStream_t stream) {
    if (!storage.initialized()) return cudaErrorInvalidValue;
    if (count == 0) return cudaSuccess;
    if (tasks == nullptr) return cudaErrorInvalidValue;

    FrontierTask* d_tasks = nullptr;
    cudaError_t err = cudaMalloc(&d_tasks, count * sizeof(FrontierTask));
    if (err != cudaSuccess) return err;

    err = cudaMemcpyAsync(d_tasks,
                          tasks,
                          count * sizeof(FrontierTask),
                          cudaMemcpyHostToDevice,
                          stream);
    if (err != cudaSuccess) {
        cudaFree(d_tasks);
        return err;
    }

    const dim3 block(256);
    const dim3 grid((count + block.x - 1u) / block.x);
    seed_frontier_tasks_kernel<<<grid, block, 0, stream>>>(storage.frontier.view,
                                                           storage.d_counters,
                                                           d_tasks,
                                                           count);
    err = cudaGetLastError();
    if (err == cudaSuccess) {
        err = cudaStreamSynchronize(stream);
    }
    cudaFree(d_tasks);
    return err;
}

cudaError_t scheduler_copy_counters(const SchedulerStorage& storage,
                                    SchedulerCounters* out,
                                    cudaStream_t stream) {
    if (!storage.initialized() || out == nullptr) return cudaErrorInvalidValue;
    return cudaMemcpyAsync(out,
                           storage.d_counters,
                           sizeof(SchedulerCounters),
                           cudaMemcpyDeviceToHost,
                           stream);
}

cudaError_t scheduler_copy_snapshot(const SchedulerStorage& storage,
                                    SchedulerSnapshot* out,
                                    cudaStream_t stream) {
    if (!storage.initialized() || out == nullptr) return cudaErrorInvalidValue;

    *out = SchedulerSnapshot{};

    cudaError_t err = scheduler_copy_counters(storage, &out->counters, stream);
    if (err != cudaSuccess) return err;

    err = copy_device_scalar(storage.d_active_workers, &out->active_workers, stream);
    if (err != cudaSuccess) return err;

    err = copy_device_scalar(storage.d_stop_flag, &out->stop_flag, stream);
    if (err != cudaSuccess) return err;

    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) return err;

    err = copy_queue_depth(storage.frontier, &out->frontier_depth, stream);
    if (err != cudaSuccess) return err;
    err = copy_queue_depth(storage.eval_requests, &out->eval_request_depth, stream);
    if (err != cudaSuccess) return err;
    err = copy_queue_depth(storage.eval_results, &out->eval_result_depth, stream);
    if (err != cudaSuccess) return err;

    return cudaSuccess;
}

void scheduler_shutdown(SchedulerStorage* storage) {
    if (storage == nullptr) return;

    free_queue(&storage->frontier);
    free_queue(&storage->eval_requests);
    free_queue(&storage->eval_results);

    if (storage->d_stop_flag != nullptr) cudaFree(storage->d_stop_flag);
    if (storage->d_active_workers != nullptr) cudaFree(storage->d_active_workers);
    if (storage->d_counters != nullptr) cudaFree(storage->d_counters);

    storage->d_stop_flag = nullptr;
    storage->d_active_workers = nullptr;
    storage->d_counters = nullptr;
}

SchedulerRuntime scheduler_runtime(const SchedulerStorage& storage) {
    SchedulerRuntime runtime{};
    runtime.queues.frontier = storage.frontier.view;
    runtime.queues.eval_requests = storage.eval_requests.view;
    runtime.queues.eval_results = storage.eval_results.view;
    runtime.stop_flag = storage.d_stop_flag;
    runtime.active_workers = storage.d_active_workers;
    runtime.counters = storage.d_counters;
    runtime.idle_backoff = storage.config.idle_backoff;
    runtime.drain_on_frontier_empty = storage.config.drain_on_frontier_empty;
    return runtime;
}

cudaError_t scheduler_launch(const SchedulerStorage& storage,
                             dim3 grid,
                             dim3 block,
                             cudaStream_t stream) {
    if (!storage.initialized()) return cudaErrorInvalidValue;

    scheduler_kernel<<<grid, block, 0, stream>>>(scheduler_runtime(storage));
    return cudaGetLastError();
}

__global__ void scheduler_kernel(SchedulerRuntime runtime) {
    if (runtime.active_workers != nullptr) {
        const uint32_t active = atomicAdd(runtime.active_workers, 1u) + 1u;
        counter_max(runtime.counters ? &runtime.counters->active_worker_peak : nullptr,
                    static_cast<unsigned long long>(active));
    }

    for (;;) {
        if (runtime.stop_flag != nullptr) {
            counter_inc(runtime.counters ? &runtime.counters->stop_checks : nullptr);
            if (*runtime.stop_flag != 0) {
                counter_inc(runtime.counters ? &runtime.counters->stop_breaks : nullptr);
                break;
            }
        }

        FrontierTask task{};
        if (!runtime.queues.frontier.try_pop(&task)) {
            counter_inc(runtime.counters ? &runtime.counters->frontier_empty_spins : nullptr);
            if (runtime.drain_on_frontier_empty != 0u) {
                counter_inc(runtime.counters ? &runtime.counters->frontier_drain_exits : nullptr);
                break;
            }
            continue;
        }

        counter_inc(runtime.counters ? &runtime.counters->frontier_pops : nullptr);

        if ((task.flags & FRONTIER_TASK_NEEDS_EVAL) != 0u) {
            counter_inc(runtime.counters ? &runtime.counters->eval_bound_tasks : nullptr);
            SchedulerEvalRequest request{};
            request.node_id = task.node_id;
            request.state_slot = task.state_slot;
            request.zobrist = task.zobrist;
            request.depth = task.depth;
            request.ply = task.ply;
            request.flags = SCHED_EVAL_REQUEST_POLICY_VALUE;

            if (runtime.queues.eval_requests.try_push(request)) {
                counter_inc(runtime.counters ? &runtime.counters->eval_requests_pushed : nullptr);
            } else {
                counter_inc(runtime.counters ? &runtime.counters->eval_request_drops : nullptr);
            }
            continue;
        }

        counter_inc(runtime.counters ? &runtime.counters->fast_path_tasks : nullptr);
        if ((task.flags & FRONTIER_TASK_TERMINAL) != 0u) {
            counter_inc(runtime.counters ? &runtime.counters->terminal_tasks : nullptr);
        }

        SchedulerEvalResult result{};
        result.node_id = task.node_id;
        result.state_slot = task.state_slot;
        result.zobrist = task.zobrist;
        result.value = 0;
        result.best_move = 0;
        result.status = ((task.flags & FRONTIER_TASK_TERMINAL) != 0u)
                            ? SCHED_EVAL_RESULT_TERMINAL
                            : SCHED_EVAL_RESULT_SKIPPED;
        result.flags = SCHED_EVAL_RESULT_FROM_FAST_PATH;

        if (runtime.queues.eval_results.try_push(result)) {
            counter_inc(runtime.counters ? &runtime.counters->eval_results_pushed : nullptr);
            counter_inc(runtime.counters ? &runtime.counters->fast_path_results : nullptr);
            if (result.status == SCHED_EVAL_RESULT_TERMINAL) {
                counter_inc(runtime.counters ? &runtime.counters->terminal_results : nullptr);
            } else if (result.status == SCHED_EVAL_RESULT_SKIPPED) {
                counter_inc(runtime.counters ? &runtime.counters->skipped_results : nullptr);
            }
        } else {
            counter_inc(runtime.counters ? &runtime.counters->eval_result_drops : nullptr);
        }
    }

    if (runtime.active_workers != nullptr) {
        atomicSub(runtime.active_workers, 1u);
    }
}

} // namespace engine
