// =============================================================================
// test_tt.cu
//
// Concurrent store/probe correctness for the lockless TT.
//
// Setup:
//   * tt_init(1 MB)  -> 64 Ki entries (16-byte slots).
//   * Generate N=10000 (key, score, depth, bound, move) tuples on the host.
//     Keys are random 64-bit; we set the low 1 bit to 1 so no key is zero
//     (zero-key entries collide with the "empty slot" sentinel).
//   * Launch a kernel with N threads. Each thread calls tt_store(...) once.
//
// Concurrent storms collide: many threads will hash to the same slot (by
// table index = key & mask). Some stores are dropped by the depth-preference
// rule; that is expected and not a torn write.
//
// Verification:
//   * Build a host-side index { table_idx -> list of stored tuples }.
//   * For each non-empty slot in the table, decode (key, score, depth,
//     bound, age, move). Find a tuple in the list for that slot whose
//     (key, score, depth, bound, move) matches exactly. If none match, it
//     is a TORN WRITE.
//   * Report:
//        torn_writes : N entries that don't match any input tuple
//        fill_rate   : # non-empty slots / total slots
//
// PASS criteria: torn_writes == 0.
// =============================================================================
#include "../tt.cuh"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <unordered_map>

using namespace engine;

struct InputTuple {
    uint64_t key;
    int16_t  score;
    int16_t  depth;
    uint8_t  bound;
    uint16_t move;
};

__global__ void k_store_many(const InputTuple* tuples, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    InputTuple t = tuples[i];
    tt_store(t.key, (int)t.score, (int)t.depth, t.bound, t.move);
}

__global__ void k_store_one(InputTuple tuple) {
    tt_store(tuple.key, (int)tuple.score, (int)tuple.depth, tuple.bound, tuple.move);
}

__global__ void k_probe_one(uint64_t key, TTEntry* out) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    *out = tt_probe(key);
}

// Read the entire raw table back to the host. We need the raw 16-byte slots
// (guard|data) so we can detect torn writes; the public tt_probe uses the
// device API only. Instead, we reconstruct via tt_probe inside a kernel.

__global__ void k_dump_table(uint64_t n_entries, TTEntry* out) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_entries) return;
    // tt_probe takes a hash; for full dump, we feed `i` as if it were the
    // hash mod-mask. The slot index is `i & mask` which equals `i` when
    // i < n_entries. But probe also unpacks data via guard^data which is
    // exactly what we want. The returned `key` is whatever was stored
    // (or guard^data if torn). bound/age/score/depth/move come from the
    // raw `data` word. We set out[i] to that struct.
    TTEntry e = tt_probe(i);
    out[i] = e;
}

int main(int argc, char** argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 10000;
    size_t mb = (argc > 2) ? (size_t)atoi(argv[2]) : 1;

    uint64_t n_entries = tt_init(mb);
    if (n_entries == 0) {
        fprintf(stderr, "tt_init failed\n");
        return 2;
    }
    printf("test_tt: tt has %lu entries (%.1f MB), N=%d concurrent stores\n",
           (unsigned long)n_entries,
           n_entries * 16.0 / 1048576.0, N);

    // Generate input tuples. Deterministic xorshift.
    std::vector<InputTuple> h_in(N);
    uint64_t r = 0x123456789ABCDEF0ULL;
    auto next = [&]() {
        r ^= r << 13; r ^= r >> 7; r ^= r << 17;
        return r;
    };
    for (int i = 0; i < N; i++) {
        uint64_t k = next();
        if (k == 0) k = 1;
        // Force low bit to 1 to ensure non-zero (and that no key collides
        // with the "empty" sentinel).
        k |= 1ULL;
        h_in[i].key   = k;
        h_in[i].score = (int16_t)((int32_t)(next() & 0xFFFF) - 32768);
        // Depth: spread across [1..63] so depth-preference replaces happen.
        h_in[i].depth = (int16_t)(1 + (next() % 63));
        h_in[i].bound = (uint8_t)(next() % 3);
        h_in[i].move  = (uint16_t)(next() & 0xFFFF);
    }

    InputTuple* d_in = nullptr;
    cudaMalloc(&d_in, N * sizeof(InputTuple));
    cudaMemcpy(d_in, h_in.data(), N * sizeof(InputTuple), cudaMemcpyHostToDevice);

    // Storm. 256 threads/block.
    int block = 256;
    int grid  = (N + block - 1) / block;
    k_store_many<<<grid, block>>>(d_in, N);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "k_store_many failed: %s\n", cudaGetErrorString(err));
        return 2;
    }

    // Dump every slot via tt_probe.
    TTEntry* d_dump = nullptr;
    cudaMalloc(&d_dump, n_entries * sizeof(TTEntry));
    int dgrid = (int)((n_entries + 255) / 256);
    k_dump_table<<<dgrid, 256>>>(n_entries, d_dump);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "k_dump_table failed: %s\n", cudaGetErrorString(err));
        return 2;
    }
    std::vector<TTEntry> h_dump(n_entries);
    cudaMemcpy(h_dump.data(), d_dump, n_entries * sizeof(TTEntry), cudaMemcpyDeviceToHost);

    // Build slot -> input-tuple-list index for fast verification.
    uint64_t mask = n_entries - 1;
    std::unordered_map<uint64_t, std::vector<int>> by_slot;
    by_slot.reserve(N * 2);
    for (int i = 0; i < N; i++) {
        uint64_t idx = h_in[i].key & mask;
        by_slot[idx].push_back(i);
    }

    // Walk every slot. Non-empty slots whose contents don't match any
    // candidate are torn writes.
    uint64_t filled = 0;
    uint64_t torn   = 0;
    uint64_t collisions_observed = 0;  // slots with >1 candidate
    for (uint64_t i = 0; i < n_entries; i++) {
        const TTEntry& e = h_dump[i];
        if (e.key == 0) continue;
        filled++;
        auto it = by_slot.find(i);
        if (it == by_slot.end()) {
            // Stored to a slot no input mapped to -> impossible unless torn.
            torn++;
            continue;
        }
        if (it->second.size() > 1) collisions_observed++;
        bool ok = false;
        for (int candidate : it->second) {
            const InputTuple& t = h_in[candidate];
            if (t.key == e.key &&
                t.score == e.score &&
                t.depth == e.depth &&
                t.bound == e.bound &&
                t.move  == e.move) {
                ok = true; break;
            }
        }
        if (!ok) {
            if (torn < 5) {
                fprintf(stderr,
                    "[slot %lu] torn? key=%016lx score=%d depth=%d bound=%u move=%u age=%u "
                    "(candidates=%zu)\n",
                    (unsigned long)i, (unsigned long)e.key,
                    (int)e.score, (int)e.depth, e.bound, e.move, e.age,
                    it->second.size());
            }
            torn++;
        }
    }

    double fill_rate = (double)filled / (double)n_entries;
    printf("test_tt: filled=%lu / %lu  (%.4f), torn_writes=%lu, "
           "slots_with_collisions=%lu\n",
           (unsigned long)filled, (unsigned long)n_entries, fill_rate,
           (unsigned long)torn, (unsigned long)collisions_observed);

    // Quick API sanity: tt_fill_rate (sampled) should be in the same ballpark.
    double sampled = tt_fill_rate();
    printf("test_tt: tt_fill_rate() (sampled) = %.4f\n", sampled);

    // Replacement-policy sanity:
    //   * same-age shallower entry must not overwrite deeper one
    //   * newer-age entry may replace an older-age entry
    tt_clear();
    TTEntry* d_probe = nullptr;
    cudaMalloc(&d_probe, sizeof(TTEntry));

    InputTuple deep{};
    deep.key = 0x123456789ABCDEF1ULL;
    deep.score = 111;
    deep.depth = 8;
    deep.bound = TT_BOUND_EXACT;
    deep.move = 0x1111;

    InputTuple shallow_same_age = deep;
    shallow_same_age.score = 77;
    shallow_same_age.depth = 4;
    shallow_same_age.move = 0x2222;

    InputTuple stale_replacement = deep;
    stale_replacement.score = -55;
    stale_replacement.depth = 2;
    stale_replacement.move = 0x3333;

    k_store_one<<<1, 1>>>(deep);
    cudaDeviceSynchronize();
    k_store_one<<<1, 1>>>(shallow_same_age);
    cudaDeviceSynchronize();
    k_probe_one<<<1, 1>>>(deep.key, d_probe);
    cudaDeviceSynchronize();
    TTEntry policy_entry{};
    cudaMemcpy(&policy_entry, d_probe, sizeof(TTEntry), cudaMemcpyDeviceToHost);
    bool policy_ok =
        policy_entry.key == deep.key &&
        policy_entry.score == deep.score &&
        policy_entry.depth == deep.depth &&
        policy_entry.move == deep.move;

    tt_age();
    k_store_one<<<1, 1>>>(stale_replacement);
    cudaDeviceSynchronize();
    k_probe_one<<<1, 1>>>(deep.key, d_probe);
    cudaDeviceSynchronize();
    cudaMemcpy(&policy_entry, d_probe, sizeof(TTEntry), cudaMemcpyDeviceToHost);
    policy_ok = policy_ok &&
        policy_entry.key == deep.key &&
        policy_entry.score == stale_replacement.score &&
        policy_entry.depth == stale_replacement.depth &&
        policy_entry.move == stale_replacement.move;
    cudaFree(d_probe);

    cudaFree(d_in);
    cudaFree(d_dump);
    tt_free();

    if (torn == 0 && policy_ok) {
        printf("test_tt: PASS\n");
        return 0;
    } else {
        printf("test_tt: FAIL (%lu torn writes, policy_ok=%s)\n",
               (unsigned long)torn, policy_ok ? "true" : "false");
        return 1;
    }
}
