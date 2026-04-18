#ifndef ENGINE_QUEUES_CUH
#define ENGINE_QUEUES_CUH

#include <stdint.h>
#include <cuda_runtime.h>

namespace engine {

// Bounded MPMC ring queue based on per-slot sequence numbers.
//
// Why this shape:
// - The scheduler, expansion workers, and evaluator service will all need
//   independent ownership over frontier/eval traffic.
// - Producers publish payloads without taking a global lock.
// - Consumers can drain the same queue concurrently.
//
// Constraints:
// - capacity must be a power of two.
// - payload types must be trivially copyable PODs.
// - host code owns allocation/reset; device code owns push/pop.

__host__ __device__ inline bool queue_capacity_is_valid(uint32_t capacity) {
    return capacity >= 2u && (capacity & (capacity - 1u)) == 0u;
}

template <typename T>
struct QueueSlot {
    uint32_t sequence;
    T payload;
};

template <typename T>
struct DeviceQueue {
    QueueSlot<T>* slots = nullptr;
    uint32_t* enqueue_pos = nullptr;
    uint32_t* dequeue_pos = nullptr;
    uint32_t capacity = 0;
    uint32_t mask = 0;

    __host__ __device__ bool valid() const {
        return slots != nullptr &&
               enqueue_pos != nullptr &&
               dequeue_pos != nullptr &&
               queue_capacity_is_valid(capacity) &&
               mask == capacity - 1u;
    }

    __device__ __forceinline__ bool try_push(const T& item) const {
        if (!valid()) return false;

        uint32_t pos = atomicAdd(enqueue_pos, 0u);
        for (;;) {
            QueueSlot<T>* slot = &slots[pos & mask];
            const uint32_t seq = atomicAdd(&slot->sequence, 0u);
            const int32_t diff =
                static_cast<int32_t>(seq) - static_cast<int32_t>(pos);

            if (diff == 0) {
                if (atomicCAS(enqueue_pos, pos, pos + 1u) == pos) {
                    slot->payload = item;
                    __threadfence();
                    atomicExch(&slot->sequence, pos + 1u);
                    return true;
                }
            } else if (diff < 0) {
                return false;
            } else {
                pos = atomicAdd(enqueue_pos, 0u);
            }
        }
    }

    __device__ __forceinline__ bool try_pop(T* out) const {
        if (!valid() || out == nullptr) return false;

        uint32_t pos = atomicAdd(dequeue_pos, 0u);
        for (;;) {
            QueueSlot<T>* slot = &slots[pos & mask];
            const uint32_t seq = atomicAdd(&slot->sequence, 0u);
            const int32_t diff =
                static_cast<int32_t>(seq) -
                static_cast<int32_t>(pos + 1u);

            if (diff == 0) {
                if (atomicCAS(dequeue_pos, pos, pos + 1u) == pos) {
                    *out = slot->payload;
                    __threadfence();
                    atomicExch(&slot->sequence, pos + capacity);
                    return true;
                }
            } else if (diff < 0) {
                return false;
            } else {
                pos = atomicAdd(dequeue_pos, 0u);
            }
        }
    }

    __device__ __forceinline__ uint32_t approx_size() const {
        const uint32_t tail = atomicAdd(enqueue_pos, 0u);
        const uint32_t head = atomicAdd(dequeue_pos, 0u);
        return tail - head;
    }
};

template <typename T>
struct QueueAllocation {
    DeviceQueue<T> view;
    QueueSlot<T>* d_slots = nullptr;
    uint32_t* d_enqueue_pos = nullptr;
    uint32_t* d_dequeue_pos = nullptr;
};

} // namespace engine

#endif // ENGINE_QUEUES_CUH
