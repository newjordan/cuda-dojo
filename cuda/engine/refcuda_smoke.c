// Smoke test for librefcuda.so — calls the GPU referee from plain C.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

// Forward decls from refcuda.cu's extern "C" API.
typedef struct Position Position;
extern Position* refc_position_new(void);
extern void refc_position_free(Position*);
extern int refc_parse_fen(const char* fen, Position* out);
extern int refc_legal_moves(const Position* pos, int32_t* moves_out, int max_n);
extern int refc_make_move(const Position* pos, int32_t mv, Position* out);
extern int refc_is_check(const Position* pos);
extern int refc_legal_count(const Position* pos);
extern int refc_is_checkmate(const Position* pos);
extern int refc_is_stalemate(const Position* pos);
extern int refc_coord3d(const Position* pos, float* xyz_out, int* octant_out);
extern void refc_move_to_uci(int32_t mv, char* uci_out);
extern int refc_position_size(void);

int main(void) {
    printf("Position size: %d bytes\n", refc_position_size());

    Position* p = refc_position_new();
    if (!p) { fprintf(stderr, "alloc fail\n"); return 1; }

    const char* startpos = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    if (refc_parse_fen(startpos, p) != 0) {
        fprintf(stderr, "parse_fen fail\n"); return 1;
    }
    printf("Parsed startpos. is_check=%d, legal_count=%d, mate=%d, stalemate=%d\n",
           refc_is_check(p), refc_legal_count(p),
           refc_is_checkmate(p), refc_is_stalemate(p));

    int32_t moves[256];
    int n = refc_legal_moves(p, moves, 256);
    printf("Legal moves at startpos: %d\n", n);
    for (int i = 0; i < n; ++i) {
        char uci[8] = {0};
        refc_move_to_uci(moves[i], uci);
        printf("  %s", uci);
    }
    printf("\n");

    float xyz[3]; int oct = 0;
    refc_coord3d(p, xyz, &oct);
    printf("startpos coord3d: x=%.4f y=%.4f z=%.4f octant=%d\n",
           xyz[0], xyz[1], xyz[2], oct);

    // Apply e2e4 (UCI 32-bit encoding requires looking it up among legal moves).
    int32_t e2e4 = -1;
    for (int i = 0; i < n; ++i) {
        char uci[8] = {0};
        refc_move_to_uci(moves[i], uci);
        if (strcmp(uci, "e2e4") == 0) { e2e4 = moves[i]; break; }
    }
    if (e2e4 == -1) { fprintf(stderr, "e2e4 not found\n"); return 1; }

    Position* p2 = refc_position_new();
    refc_make_move(p, e2e4, p2);
    int n2 = refc_legal_count(p2);
    printf("After e2e4: legal_count=%d (expected 20 for black)\n", n2);
    refc_coord3d(p2, xyz, &oct);
    printf("post-e2e4 coord3d: x=%.4f y=%.4f z=%.4f octant=%d\n",
           xyz[0], xyz[1], xyz[2], oct);

    // Test a checkmate position. Fool's mate post-position (white mated).
    Position* pm = refc_position_new();
    const char* mated = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3";
    refc_parse_fen(mated, pm);
    printf("Fool's-mate position: is_check=%d, legal_count=%d, mate=%d\n",
           refc_is_check(pm), refc_legal_count(pm), refc_is_checkmate(pm));

    refc_position_free(p);
    refc_position_free(p2);
    refc_position_free(pm);
    return 0;
}
