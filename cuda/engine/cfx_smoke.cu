// =============================================================================
// cfx_smoke.cu
//
// Minimum GPU CFX kernel: for a given FEN, computes n_resp (partner legal-
// response count) for every legal candidate move ON THE GPU.
//
// The chess principle this captures: a "forcing move" leaves the opponent
// with few legal replies. Per the audit's Req 5 partner-counterfactual,
// smaller n_resp = more forcing = more likely to be the played move in
// middlegame and endgame.
//
// Verification path: for any FEN, the per-candidate n_resp values must
// match what python-chess produces for the same moves. The CUDA engine's
// rules are already perft-verified (engine_test_perft passes 6/6 canonical
// chessprogramming.org positions to depth 4).
// =============================================================================
#include <stdio.h>
#include <string.h>

#include <cuda_runtime.h>

#include "movegen.cuh"
#include "make_unmake.cuh"
#include "attacks.cuh"

using namespace engine;

#ifndef CFX_MAX_MOVES
#define CFX_MAX_MOVES 256
#endif

// Kernel: one thread per legal candidate move at the parent position.
// Each thread applies its candidate, generates pseudo-legal partner moves,
// filters for legality (partner king not in check), and writes the count.
__global__ void cfx_n_resp_kernel(
    const Position* parent_pos,
    const Move* candidates,
    int n_candidates,
    int* n_resp_out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_candidates) return;

    Position child = *parent_pos;
    Undo undo;
    make_move(&child, candidates[idx], &undo);

    Move responses[CFX_MAX_MOVES];
    int n_pseudo = generate_moves(&child, responses);

    int legal = 0;
    for (int j = 0; j < n_pseudo; ++j) {
        Position grandchild = child;
        Undo undo2;
        make_move(&grandchild, responses[j], &undo2);
        // After the response, grandchild.side has flipped to the candidate-mover.
        // The piece that just moved (the partner) was on side (1 - grandchild.side).
        // If THAT side's king is attacked, the response left their own king in
        // check — illegal.
        if (!in_check(&grandchild, 1 - grandchild.side)) {
            legal++;
        }
    }

    n_resp_out[idx] = legal;
}

static inline void cuda_must(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in %s: %s\n", what, cudaGetErrorString(err));
        exit(2);
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr,
            "usage: %s <fen>\n"
            "  prints per-legal-candidate: <uci> <n_resp_on_gpu>\n",
            argv[0]);
        return 1;
    }

    // Parse FEN (host)
    Position pos;
    parse_fen(argv[1], &pos);

    // Generate pseudo-legal candidates (host), filter to legal.
    Move pseudo[CFX_MAX_MOVES];
    int n_pseudo = generate_moves(&pos, pseudo);
    Move legal[CFX_MAX_MOVES];
    int n_legal = 0;
    for (int i = 0; i < n_pseudo; ++i) {
        Position child = pos;
        make_move(&child, pseudo[i]);
        if (!in_check(&child, 1 - child.side)) {
            legal[n_legal++] = pseudo[i];
        }
    }

    // Device allocation
    Position* d_pos = nullptr;
    Move* d_candidates = nullptr;
    int* d_n_resp = nullptr;
    cuda_must(cudaMalloc(&d_pos, sizeof(Position)), "cudaMalloc d_pos");
    cuda_must(cudaMalloc(&d_candidates, n_legal * sizeof(Move)),
              "cudaMalloc d_candidates");
    cuda_must(cudaMalloc(&d_n_resp, n_legal * sizeof(int)),
              "cudaMalloc d_n_resp");

    cuda_must(cudaMemcpy(d_pos, &pos, sizeof(Position), cudaMemcpyHostToDevice),
              "memcpy pos");
    cuda_must(cudaMemcpy(d_candidates, legal, n_legal * sizeof(Move),
                         cudaMemcpyHostToDevice),
              "memcpy candidates");

    // Launch
    int tpb = 64;
    int blocks = (n_legal + tpb - 1) / tpb;
    cfx_n_resp_kernel<<<blocks, tpb>>>(d_pos, d_candidates, n_legal, d_n_resp);
    cuda_must(cudaDeviceSynchronize(), "kernel sync");
    cuda_must(cudaGetLastError(), "kernel launch");

    int n_resp_host[CFX_MAX_MOVES];
    cuda_must(cudaMemcpy(n_resp_host, d_n_resp, n_legal * sizeof(int),
                         cudaMemcpyDeviceToHost),
              "memcpy results");

    // Output: <uci> <n_resp>
    for (int i = 0; i < n_legal; ++i) {
        char uci[8] = {0};
        move_to_uci(legal[i], uci);
        printf("%s %d\n", uci, n_resp_host[i]);
    }

    cudaFree(d_pos);
    cudaFree(d_candidates);
    cudaFree(d_n_resp);
    return 0;
}
