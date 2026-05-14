/* ═══════════════════════════════════════════════════════════════════════════
 * validate_frostmatrix.cu — tiny harness that loads trained FrostMatrix V3
 * weights and runs forward pass on a board, printing raw policy/value.
 *
 *   nvcc -arch=native -O3 -DUSE_FROSTMATRIX_EVAL validate_frostmatrix.cu \
 *        -o validate_frostmatrix -lcublas
 *
 * Input: piece-encoded board as 64 ints (stdin)
 *        piece encoding: 0=empty, 1-6=white P,N,B,R,Q,K,  7-12=black p,n,b,r,q,k
 * Output: 64 from_sq logits (space separated floats)
 *         64 to_sq logits
 *         value (single float)
 * ═══════════════════════════════════════════════════════════════════════════ */

#include <stdio.h>
#include <stdlib.h>

#define USE_FROSTMATRIX_EVAL
#include "frostmatrix_eval.cuh"

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <weights_dir>\n", argv[0]);
        fprintf(stderr, "  reads 64 piece-type ints from stdin (0=empty, 1-6=white, 7-12=black)\n");
        return 1;
    }
    const char *weights_dir = argv[1];

    if (!frostmatrix_init(weights_dir)) {
        fprintf(stderr, "validate_frostmatrix: init failed\n");
        return 2;
    }

    // Read 64 piece ints from stdin
    int pieces[64];
    int got = 0;
    while (got < 64) {
        if (scanf("%d", &pieces[got]) != 1) break;
        got++;
    }
    if (got != 64) {
        fprintf(stderr, "validate_frostmatrix: need 64 piece ints on stdin, got %d\n", got);
        return 3;
    }

    // Convert forge-style pieces to frostrmatrix tokens
    int tokens[64];
    for (int sq = 0; sq < 64; sq++) tokens[sq] = fm_convert_piece(pieces[sq]);

    float from_logits[64], to_logits[64], value;
    frostmatrix_forward(tokens, 1, 0, from_logits, to_logits, &value);

    // Print: from logits (64), to logits (64), value
    for (int i = 0; i < 64; i++) printf("%.8f%c", from_logits[i], (i<63)?' ':'\n');
    for (int i = 0; i < 64; i++) printf("%.8f%c", to_logits[i],   (i<63)?' ':'\n');
    printf("%.8f\n", value);

    return 0;
}
