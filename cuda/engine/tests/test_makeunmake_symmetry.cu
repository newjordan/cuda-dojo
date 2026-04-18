// =============================================================================
// test_makeunmake_symmetry
//
// Contract: for any reachable Position P and legal move M with paired
// Undo U from make_move(P, M, &U), unmake_move(P, M, &U) restores P to a
// byte-identical copy of its pre-make state.
//
// The harness explores N=100 random positions × N=100 random moves
// using a self-driving random walker rooted at the start position. Every
// time make/unmake disagrees byte-wise it logs the diff and counts a
// failure.
// =============================================================================
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "../movegen.cuh"
#include "../make_unmake.cuh"
#include "../attacks.cuh"

using namespace engine;

static const char* START_FEN =
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

// Pick a uniformly-random LEGAL move from `s`. Returns 0 if no legal moves.
static int pick_legal_move(const Position* s, unsigned int* rng) {
    Move buf[MAX_MOVES];
    int n = generate_moves(s, buf);
    Move legal[MAX_MOVES];
    int  m = 0;
    for (int i = 0; i < n; i++) {
        Position c = *s;
        make_move(&c, buf[i]);
        if (in_check(&c, 1 - c.side)) continue;
        legal[m++] = buf[i];
    }
    if (m == 0) return 0;
    *rng = (*rng) * 1664525u + 1013904223u;
    return legal[(*rng) % m];
}

static bool position_eq_bytes(const Position* a, const Position* b) {
    return memcmp(a, b, sizeof(Position)) == 0;
}

static void dump_position(const Position* p, const char* tag) {
    fprintf(stderr, "  [%s] side=%d castle=%d ep=%d hm=%d fm=%d king=(%d,%d)\n",
            tag, p->side, p->castle, p->ep, p->halfmove, p->fullmove,
            p->kingPos[0], p->kingPos[1]);
    fprintf(stderr, "  [%s] board:", tag);
    for (int i = 0; i < 64; i++) fprintf(stderr, " %d", p->board[i]);
    fprintf(stderr, "\n");
}

int main(int /*argc*/, char** /*argv*/) {
    const int N_POSITIONS  = 100;
    const int N_MOVES      = 100;

    unsigned int rng = 0xdeadbeef;

    int total_trials = 0;
    int failures     = 0;
    int positions_used = 0;
    int trials_per_position[N_POSITIONS];

    Position root;
    parse_fen(START_FEN, &root);

    // Outer loop: produce N_POSITIONS distinct positions by random self-play
    // walks of varying length; reset to root if we run out of legal moves.
    Position current = root;
    int walk_len = 0;

    for (int p = 0; p < N_POSITIONS; p++) {
        // Walk 1..16 random plies forward to vary the position distribution.
        int target = (rng = rng * 1664525u + 1013904223u) % 16 + 1;
        for (int step = 0; step < target; step++) {
            Move m = pick_legal_move(&current, &rng);
            if (m == 0) {
                // Game over (mate / stalemate). Restart.
                current = root;
                walk_len = 0;
                break;
            }
            make_move(&current, m);
            walk_len++;
            // Hard reset every 60 plies so we don't get stuck near 50-move-rule
            // territory in long random walks.
            if (walk_len > 60) {
                current = root;
                walk_len = 0;
                break;
            }
        }
        positions_used++;

        // Inner loop: try N_MOVES random legal moves, each tested for symmetry.
        Move buf[MAX_MOVES];
        int n = generate_moves(&current, buf);
        // Filter to legal moves to keep the test focused on real game tree.
        Move legal[MAX_MOVES];
        int  m_legal = 0;
        for (int i = 0; i < n; i++) {
            Position c = current;
            make_move(&c, buf[i]);
            if (in_check(&c, 1 - c.side)) continue;
            legal[m_legal++] = buf[i];
        }

        int trials_here = 0;

        if (m_legal == 0) {
            // Terminal node — record 0 trials; outer loop continues.
            trials_per_position[p] = 0;
            continue;
        }

        for (int t = 0; t < N_MOVES; t++) {
            rng = rng * 1664525u + 1013904223u;
            Move mv = legal[rng % m_legal];

            Position before = current;
            Position work   = current;

            Undo u;
            make_move(&work, mv, &u);
            unmake_move(&work, mv, &u);

            total_trials++;
            trials_here++;
            if (!position_eq_bytes(&before, &work)) {
                failures++;
                if (failures <= 3) {
                    char uci[8] = {0};
                    move_to_uci(mv, uci);
                    fprintf(stderr,
                            "MISMATCH (failure #%d) at position %d, trial %d, move=%s (raw=0x%x)\n",
                            failures, p, t, uci, mv);
                    dump_position(&before, "before");
                    dump_position(&work,   "after_unmake");
                }
            }
        }

        trials_per_position[p] = trials_here;
    }

    int positions_with_trials = 0;
    for (int i = 0; i < positions_used; i++)
        if (trials_per_position[i] > 0) positions_with_trials++;

    printf("test_makeunmake_symmetry\n");
    printf("------------------------------------------------------------\n");
    printf("positions_visited = %d\n", positions_used);
    printf("positions_with_legal_moves = %d\n", positions_with_trials);
    printf("total_trials = %d\n", total_trials);
    printf("failures = %d\n", failures);
    printf("pass_rate = %.4f%%\n",
           total_trials > 0 ? (100.0 * (total_trials - failures) / total_trials) : 0.0);
    return failures == 0 ? 0 : 1;
}
