/* validate_frostmatrix_debug.cu — instrumented CUDA forward pass for parity debugging. */
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define USE_FROSTMATRIX_EVAL
#include "frostmatrix_eval.cuh"

static void print_host(const char *label, float *d_buf, int n) {
    float *h = (float*)malloc(n * sizeof(float));
    cudaMemcpy(h, d_buf, n * sizeof(float), cudaMemcpyDeviceToHost);
    printf("%s: ", label);
    for (int i = 0; i < n && i < 8; i++) printf("%.6f ", h[i]);
    printf("\n");
    free(h);
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <weights_dir>\n", argv[0]); return 1; }
    if (!frostmatrix_init(argv[1])) { fprintf(stderr, "init failed\n"); return 2; }

    int pieces[64] = {
        4,2,3,5,6,3,2,4,
        1,1,1,1,1,1,1,1,
        0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,
        9,9,9,9,9,9,9,9,
        12,10,11,13,14,11,10,12
    };
    int tokens[64];
    for (int sq = 0; sq < 64; sq++) tokens[sq] = fm_convert_piece(pieces[sq]);

    float from[64], to[64], value;
    frostmatrix_forward(tokens, 1, 0, from, to, &value);

    printf("VALUE: %.8f\n", value);
    printf("FROM: ");
    for (int i = 0; i < 64; i++) printf("%.8f ", from[i]);
    printf("\nTO: ");
    for (int i = 0; i < 64; i++) printf("%.8f ", to[i]);
    printf("\n");

    // Read pooled vector from device for comparison
    print_host("POOLED", fm_buf_pooled, 128);

    // Read x buffer (first 8 tokens of sequence) 
    print_host("X[0:8,fam0]", fm_buf_x, 8);
    print_host("X[65:8,brd0]", fm_buf_x + 65*128, 8);

    return 0;
}
