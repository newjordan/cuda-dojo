// =============================================================================
// engine_test_perft
//
// Reproduces the 6 canonical chessprogramming.org perft results at depth 4
// using ONLY the engine/ headers — no calls into gpu_fighter.
//
// Gold values from https://www.chessprogramming.org/Perft_Results
// =============================================================================
#include <stdio.h>
#include <string.h>

#include "../movegen.cuh"
#include "../make_unmake.cuh"
#include "../attacks.cuh"

using namespace engine;

struct PerftCase {
    const char* name;
    const char* fen;
    int         depth;
    long long   expected;
};

// Depth-4 perft expectations from chessprogramming.org "Perft Results".
// All 6 canonical positions; same set the gpu_fighter movegen was verified on.
static const PerftCase CASES[] = {
    { "Position 1 (Initial)",
      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
      4, 197281LL },

    { "Position 2 (Kiwipete)",
      "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
      4, 4085603LL },

    { "Position 3 (endgame)",
      "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
      4, 43238LL },

    { "Position 4 (mirror — promotions)",
      "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
      4, 422333LL },

    { "Position 5",
      "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
      4, 2103487LL },

    { "Position 6 (Steven Edwards)",
      "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
      4, 3894594LL },
};
static const int N_CASES = sizeof(CASES) / sizeof(CASES[0]);

int main(int /*argc*/, char** /*argv*/) {
    int failures = 0;
    long long total_nodes = 0;

    printf("engine_test_perft: 6 canonical positions, depth 4\n");
    printf("------------------------------------------------------------\n");

    for (int i = 0; i < N_CASES; i++) {
        Position p;
        parse_fen(CASES[i].fen, &p);
        long long got = perft(&p, CASES[i].depth);
        bool ok = (got == CASES[i].expected);
        printf("[%s] %-40s depth=%d  got=%lld  expected=%lld\n",
               ok ? "PASS" : "FAIL",
               CASES[i].name, CASES[i].depth, got, CASES[i].expected);
        if (!ok) failures++;
        total_nodes += got;
    }

    printf("------------------------------------------------------------\n");
    printf("total_nodes=%lld  failures=%d/%d\n",
           total_nodes, failures, N_CASES);

    return failures == 0 ? 0 : 1;
}
