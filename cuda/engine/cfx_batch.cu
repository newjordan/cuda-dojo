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

using namespace engine;

#ifndef CFX_MAX_MOVES
#define CFX_MAX_MOVES 256
#endif
#ifndef CFX_MAX_FENS
#define CFX_MAX_FENS 4096
#endif

// One block per FEN, one thread per candidate.
// Output layout: n_resp_out[fen_idx * CFX_MAX_MOVES + cand_idx],
//                legal_out[fen_idx * CFX_MAX_MOVES + cand_idx],
//                n_legal_out[fen_idx].
__global__ void cfx_batch_kernel(
    const Position* parents,
    int* n_resp_out,
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

    // Apply candidate → child position. Enumerate partner pseudo-legal,
    // count those that don't leave the partner's king in check.
    Position child = parent;
    Undo undo;
    make_move(&child, cand, &undo);

    Move responses[CFX_MAX_MOVES];
    int n_pseudo = generate_moves(&child, responses);
    int legal_count = 0;
    for (int j = 0; j < n_pseudo; ++j) {
        Position grandchild = child;
        Undo undo2;
        make_move(&grandchild, responses[j], &undo2);
        if (!in_check(&grandchild, 1 - grandchild.side)) {
            legal_count++;
        }
    }
    n_resp_out[fen_idx * CFX_MAX_MOVES + cand_idx] = legal_count;
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
    int* d_n_resp = nullptr;
    Move* d_legal = nullptr;
    int* d_n_legal = nullptr;
    cuda_must(cudaMalloc(&d_parents, n_fens * sizeof(Position)), "malloc parents");
    cuda_must(cudaMalloc(&d_n_resp, n_fens * CFX_MAX_MOVES * sizeof(int)),
              "malloc n_resp");
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
        d_parents, d_n_resp, d_legal, d_n_legal
    );
    cudaEventRecord(stop);
    cuda_must(cudaEventSynchronize(stop), "kernel sync");
    cuda_must(cudaGetLastError(), "kernel launch");

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    int* h_n_resp = (int*)malloc(n_fens * CFX_MAX_MOVES * sizeof(int));
    Move* h_legal = (Move*)malloc(n_fens * CFX_MAX_MOVES * sizeof(Move));
    int* h_n_legal = (int*)malloc(n_fens * sizeof(int));
    cuda_must(cudaMemcpy(h_n_resp, d_n_resp, n_fens * CFX_MAX_MOVES * sizeof(int),
                         cudaMemcpyDeviceToHost),
              "memcpy n_resp");
    cuda_must(cudaMemcpy(h_legal, d_legal, n_fens * CFX_MAX_MOVES * sizeof(Move),
                         cudaMemcpyDeviceToHost),
              "memcpy legal");
    cuda_must(cudaMemcpy(h_n_legal, d_n_legal, n_fens * sizeof(int),
                         cudaMemcpyDeviceToHost),
              "memcpy n_legal");

    // Output: one section per FEN. Header line: ">>> <fen> <n_legal>"
    long long total_candidates = 0;
    for (int i = 0; i < n_fens; ++i) {
        int n = h_n_legal[i];
        printf(">>> %s %d\n", fen_strs[i], n);
        for (int j = 0; j < n; ++j) {
            char uci[8] = {0};
            move_to_uci(h_legal[i * CFX_MAX_MOVES + j], uci);
            printf("%s %d\n", uci, h_n_resp[i * CFX_MAX_MOVES + j]);
        }
        total_candidates += n;
    }

    fprintf(stderr,
        "[cfx_batch] %d FENs, %lld candidates, kernel %.2fms (%.3f us/cand)\n",
        n_fens, total_candidates, ms,
        ms * 1000.0 / (total_candidates ? total_candidates : 1));

    return 0;
}
