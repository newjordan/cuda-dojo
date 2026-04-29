// =============================================================================
// cfx_batch.cu
//
// Batched GPU CFX kernel: reads FENs from stdin (one per line), launches ONE
// kernel that processes all positions in parallel, prints per-candidate
// (uci, n_resp) for every legal candidate at every position.
//
// Architecture:
//   - blockIdx.x       → which FEN in the batch
//   - threadIdx.x      → which candidate move at that FEN (after legality)
//   - block 0 handles FEN[0], etc.
//   - per-block shared array holds the legal candidates after filtering
//
// This kills the per-FEN subprocess overhead from cfx_smoke and is the form
// that drops into the Omnifold engine via a Python binding (next iter).
// =============================================================================
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime.h>

#include "movegen.cuh"
#include "make_unmake.cuh"
#include "attacks.cuh"
#include "coord3d.cuh"

using namespace engine;

#ifndef CFX_MAX_MOVES
#define CFX_MAX_MOVES 256
#endif
#ifndef CFX_MAX_FENS
#define CFX_MAX_FENS 4096
#endif

// One block per FEN, one thread per candidate.
// Each thread computes the full CFX feature set (n_resp, mean_dist,
// std_dist, max_dist) for its candidate using GPU compute_coord3d. All
// chess-rule and coord work executes on the device.
//
// Output layout: per (fen_idx, cand_idx):
//   n_resp_out[..]    — partner-legal-response count
//   mean_dist_out[..] — mean ‖coord(response) − coord(candidate)‖ in 3D
//   std_dist_out[..]  — stddev of those distances
//   max_dist_out[..]  — max of those distances
//   legal_out[..]     — the candidate move (UCI host-side)
__global__ void cfx_batch_kernel(
    const Position* parents,
    int* n_resp_out,
    float* mean_dist_out,
    float* std_dist_out,
    float* max_dist_out,
    Move* legal_out,
    int* n_legal_out
) {
    int fen_idx = blockIdx.x;
    Position parent = parents[fen_idx];

    __shared__ Move legal_buf[CFX_MAX_MOVES];
    __shared__ int n_legal;

    if (threadIdx.x == 0) {
        // Filter pseudo-legal candidates down to fully-legal moves.
        Move pseudo[CFX_MAX_MOVES];
        int n_pseudo = generate_moves(&parent, pseudo);
        int n = 0;
        for (int i = 0; i < n_pseudo; ++i) {
            Position child = parent;
            make_move(&child, pseudo[i]);
            if (!in_check(&child, 1 - child.side)) {
                legal_buf[n++] = pseudo[i];
            }
        }
        n_legal = n;
        n_legal_out[fen_idx] = n;
    }
    __syncthreads();

    int cand_idx = threadIdx.x;
    if (cand_idx >= n_legal) return;

    Move cand = legal_buf[cand_idx];
    legal_out[fen_idx * CFX_MAX_MOVES + cand_idx] = cand;

    // Apply candidate → child position.
    Position child = parent;
    Undo undo;
    make_move(&child, cand, &undo);

    // Compute the candidate's own coord (X_m).
    Coord3D cand_coord = compute_coord3d(&child);

    // Enumerate partner pseudo-legal responses, filter for legality, and
    // accumulate distance statistics over the legal-response coord cloud.
    Move responses[CFX_MAX_MOVES];
    int n_pseudo = generate_moves(&child, responses);
    int legal_count = 0;
    float sum_d = 0.0f;
    float sum_d2 = 0.0f;
    float max_d = 0.0f;

    for (int j = 0; j < n_pseudo; ++j) {
        Position grandchild = child;
        Undo undo2;
        make_move(&grandchild, responses[j], &undo2);
        if (in_check(&grandchild, 1 - grandchild.side)) continue;

        Coord3D resp_coord = compute_coord3d(&grandchild);
        float dx = resp_coord.x - cand_coord.x;
        float dy = resp_coord.y - cand_coord.y;
        float dz = resp_coord.z - cand_coord.z;
        float d = sqrtf(dx * dx + dy * dy + dz * dz);

        sum_d  += d;
        sum_d2 += d * d;
        if (d > max_d) max_d = d;
        legal_count++;
    }

    int idx = fen_idx * CFX_MAX_MOVES + cand_idx;
    n_resp_out[idx] = legal_count;
    if (legal_count > 0) {
        float mean = sum_d / float(legal_count);
        float var  = sum_d2 / float(legal_count) - mean * mean;
        if (var < 0.0f) var = 0.0f;
        mean_dist_out[idx] = mean;
        std_dist_out[idx]  = sqrtf(var);
        max_dist_out[idx]  = max_d;
    } else {
        // Terminal — checkmate or stalemate after candidate.
        mean_dist_out[idx] = 0.0f;
        std_dist_out[idx]  = 0.0f;
        max_dist_out[idx]  = 0.0f;
    }
}

static inline void cuda_must(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in %s: %s\n", what, cudaGetErrorString(err));
        exit(2);
    }
}

int main(int /*argc*/, char** /*argv*/) {
    // Read FENs from stdin, one per line. Empty lines and comments (#) skipped.
    Position* h_parents = (Position*)calloc(CFX_MAX_FENS, sizeof(Position));
    char** fen_strs = (char**)calloc(CFX_MAX_FENS, sizeof(char*));
    int n_fens = 0;

    char line[512];
    while (fgets(line, sizeof(line), stdin) != NULL) {
        size_t len = strlen(line);
        while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r')) {
            line[--len] = '\0';
        }
        if (len == 0 || line[0] == '#') continue;
        if (n_fens >= CFX_MAX_FENS) {
            fprintf(stderr, "exceeded CFX_MAX_FENS=%d\n", CFX_MAX_FENS);
            return 1;
        }
        fen_strs[n_fens] = strdup(line);
        parse_fen(line, &h_parents[n_fens]);
        n_fens++;
    }
    if (n_fens == 0) {
        fprintf(stderr, "no FENs on stdin\n");
        return 1;
    }

    // Device side
    Position* d_parents = nullptr;
    int*   d_n_resp    = nullptr;
    float* d_mean_dist = nullptr;
    float* d_std_dist  = nullptr;
    float* d_max_dist  = nullptr;
    Move*  d_legal     = nullptr;
    int*   d_n_legal   = nullptr;
    cuda_must(cudaMalloc(&d_parents, n_fens * sizeof(Position)), "malloc parents");
    cuda_must(cudaMalloc(&d_n_resp, n_fens * CFX_MAX_MOVES * sizeof(int)),
              "malloc n_resp");
    cuda_must(cudaMalloc(&d_mean_dist, n_fens * CFX_MAX_MOVES * sizeof(float)),
              "malloc mean_dist");
    cuda_must(cudaMalloc(&d_std_dist, n_fens * CFX_MAX_MOVES * sizeof(float)),
              "malloc std_dist");
    cuda_must(cudaMalloc(&d_max_dist, n_fens * CFX_MAX_MOVES * sizeof(float)),
              "malloc max_dist");
    cuda_must(cudaMalloc(&d_legal, n_fens * CFX_MAX_MOVES * sizeof(Move)),
              "malloc legal");
    cuda_must(cudaMalloc(&d_n_legal, n_fens * sizeof(int)), "malloc n_legal");

    cuda_must(cudaMemcpy(d_parents, h_parents, n_fens * sizeof(Position),
                         cudaMemcpyHostToDevice),
              "memcpy parents");

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    cfx_batch_kernel<<<n_fens, CFX_MAX_MOVES>>>(
        d_parents, d_n_resp, d_mean_dist, d_std_dist, d_max_dist,
        d_legal, d_n_legal
    );
    cudaEventRecord(stop);
    cuda_must(cudaEventSynchronize(stop), "kernel sync");
    cuda_must(cudaGetLastError(), "kernel launch");

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    int*   h_n_resp    = (int*)malloc(n_fens * CFX_MAX_MOVES * sizeof(int));
    float* h_mean_dist = (float*)malloc(n_fens * CFX_MAX_MOVES * sizeof(float));
    float* h_std_dist  = (float*)malloc(n_fens * CFX_MAX_MOVES * sizeof(float));
    float* h_max_dist  = (float*)malloc(n_fens * CFX_MAX_MOVES * sizeof(float));
    Move*  h_legal     = (Move*)malloc(n_fens * CFX_MAX_MOVES * sizeof(Move));
    int*   h_n_legal   = (int*)malloc(n_fens * sizeof(int));
    cuda_must(cudaMemcpy(h_n_resp, d_n_resp, n_fens * CFX_MAX_MOVES * sizeof(int),
                         cudaMemcpyDeviceToHost),
              "memcpy n_resp");
    cuda_must(cudaMemcpy(h_mean_dist, d_mean_dist, n_fens * CFX_MAX_MOVES * sizeof(float),
                         cudaMemcpyDeviceToHost),
              "memcpy mean_dist");
    cuda_must(cudaMemcpy(h_std_dist, d_std_dist, n_fens * CFX_MAX_MOVES * sizeof(float),
                         cudaMemcpyDeviceToHost),
              "memcpy std_dist");
    cuda_must(cudaMemcpy(h_max_dist, d_max_dist, n_fens * CFX_MAX_MOVES * sizeof(float),
                         cudaMemcpyDeviceToHost),
              "memcpy max_dist");
    cuda_must(cudaMemcpy(h_legal, d_legal, n_fens * CFX_MAX_MOVES * sizeof(Move),
                         cudaMemcpyDeviceToHost),
              "memcpy legal");
    cuda_must(cudaMemcpy(h_n_legal, d_n_legal, n_fens * sizeof(int),
                         cudaMemcpyDeviceToHost),
              "memcpy n_legal");

    // Output per FEN. Header: ">>> <fen> <n_legal>"
    // Per candidate: <uci> <n_resp> <mean_dist> <std_dist> <max_dist>
    long long total_candidates = 0;
    for (int i = 0; i < n_fens; ++i) {
        int n = h_n_legal[i];
        printf(">>> %s %d\n", fen_strs[i], n);
        for (int j = 0; j < n; ++j) {
            int idx = i * CFX_MAX_MOVES + j;
            char uci[8] = {0};
            move_to_uci(h_legal[idx], uci);
            printf("%s %d %.6f %.6f %.6f\n",
                   uci,
                   h_n_resp[idx],
                   h_mean_dist[idx], h_std_dist[idx], h_max_dist[idx]);
        }
        total_candidates += n;
    }

    fprintf(stderr,
        "[cfx_batch] %d FENs, %lld candidates, kernel %.2fms (%.3f us/cand)\n",
        n_fens, total_candidates, ms,
        ms * 1000.0 / (total_candidates ? total_candidates : 1));

    return 0;
}
