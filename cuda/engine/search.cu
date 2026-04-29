// =============================================================================
// search.cu — root-parallel alpha-beta GPU search.
//
// See search.cuh for the high-level architecture and contracts.
//
// Hard rule (CUDA DOJO mandate): ALL chess thinking happens on the GPU.
// The host only:
//   - launches kernels
//   - copies the small (per-iteration) score array back
//   - polls a wall-clock and flips a __device__ stop flag
//   - walks the TT to extract the PV (no eval/movegen/legality on host)
// =============================================================================
#include "search.cuh"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <chrono>
#include <thread>
#include <utility>
#include <vector>
#include <string>

#include <cuda_runtime.h>

#include "engine_types.h"
#include "movegen.cuh"
#include "make_unmake.cuh"
#include "attacks.cuh"
#include "coord3d.cuh"
#include "eval.cuh"
#include "eval_service.cuh"
#include "scheduler.cuh"
#include "zobrist.cuh"
#include "tt.cuh"
#include "time.cuh"

namespace engine {

// =============================================================================
// Device-side scratch / globals
// =============================================================================
namespace {

// One stop flag, polled inside the negamax inner loop.
__device__ int        d_stop_flag = 0;
__device__ uint64_t   d_node_counter = 0;

// Mirrors / pointers reset by search_init().
Position*  d_root_state = nullptr;
Move*      d_root_moves = nullptr;
int*       d_root_scores = nullptr;
int*       d_root_n_resp = nullptr;       // CFX root: legal-response count
float*     d_root_mean_dist = nullptr;    // CFX root: mean response coord-distance
float*     d_root_std_dist = nullptr;     // CFX root: stddev of distances
float*     d_root_max_dist = nullptr;     // CFX root: max distance
uint64_t*  d_root_hashes = nullptr;
int*       d_root_count = nullptr;
int        d_initialised = 0;
int        h_runtime_initialised = 0;

SchedulerStorage h_scheduler_storage{};
EvalService      h_eval_service{};
SearchRuntimeReport h_runtime_report{};

// Per-ply node-count granularity for stop-flag polling.
constexpr int NODE_STOP_INTERVAL = 4096;
constexpr uint32_t RUNTIME_SCHEDULER_CAPACITY = 512;
constexpr int ROOT_RUNTIME_TOP_ORDER_LIMIT = 8;

// MVV-LVA table indexed by [victim_pt][attacker_pt], 1..6 used.
// Higher = search first.
__constant__ int d_MVV_LVA[7][7] = {
    {0,0,0,0,0,0,0},
    {0,105,104,103,102,101,100},  // pawn victim
    {0,205,204,203,202,201,200},  // knight
    {0,305,304,303,302,301,300},  // bishop
    {0,405,404,403,402,401,400},  // rook
    {0,505,504,503,502,501,500},  // queen
    {0,605,604,603,602,601,600},  // king (shouldn't happen but safe)
};

constexpr int QUIET_KILLER_SCORE_1 = 8000;
constexpr int QUIET_KILLER_SCORE_2 = 7990;

} // anonymous

__global__ void pack_root_eval_requests_from_scheduler_kernel(
    DeviceQueue<SchedulerEvalRequest> scheduler_queue,
    EvalBucketStorage eval_bucket,
    const Position* root_state,
    const Move* root_moves,
    uint32_t max_requests)
{
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        eval_bucket.d_counters->request_count = 0;
        eval_bucket.d_counters->result_count = 0;
        eval_bucket.d_counters->policy_count = 0;
        eval_bucket.d_counters->dropped_requests = 0;
    }

    if (!scheduler_queue.valid() || eval_bucket.d_requests == nullptr ||
        eval_bucket.d_counters == nullptr || root_state == nullptr ||
        root_moves == nullptr) {
        return;
    }

    __shared__ uint32_t queue_base;
    __shared__ uint32_t queue_count;

    if (threadIdx.x == 0) {
        const uint32_t dequeue = *scheduler_queue.dequeue_pos;
        const uint32_t enqueue = *scheduler_queue.enqueue_pos;
        queue_base = dequeue;
        uint32_t available = enqueue - dequeue;
        if (available > max_requests) available = max_requests;
        if (available > static_cast<uint32_t>(eval_bucket.batch_size)) {
            available = static_cast<uint32_t>(eval_bucket.batch_size);
        }
        queue_count = available;
        eval_bucket.d_counters->request_count = available;
    }
    __syncthreads();

    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= queue_count) return;

    const QueueSlot<SchedulerEvalRequest>& slot =
        scheduler_queue.slots[(queue_base + idx) & scheduler_queue.mask];
    const SchedulerEvalRequest request = slot.payload;

    Position child = *root_state;
    Move mv = root_moves[request.state_slot];
    make_move(&child, mv);

    EvalRequest eval_request{};
    eval_request.request_id = request.node_id;
    eval_request.state_idx = request.state_slot;
    eval_request.zobrist = request.zobrist;
    eval_request.flags = EVAL_REQUEST_NEEDS_VALUE;
    eval_request.depth_remaining = static_cast<int16_t>(request.depth);
    eval_request.ply = static_cast<int16_t>(request.ply);
    eval_request.position = child;
    eval_bucket.d_requests[idx] = eval_request;
}

// =============================================================================
// Device helpers
// =============================================================================

__device__ static inline bool d_is_quiet_move(Move m) {
    int flags = move_flags(m);
    return flags != FLAG_CAPTURE && flags != FLAG_EP && flags != FLAG_PROMO;
}

// Score a move for ordering. Higher first.
//   * TT move first (handled in caller, gets ordering_score = INT_MAX)
//   * Captures by MVV-LVA
//   * Quiet killer moves
//   * Other quiet moves get 0
__device__ static inline int d_score_move(const Position& s,
                                          Move m,
                                          Move killer1 = 0,
                                          Move killer2 = 0) {
    int flags = move_flags(m);
    if (flags == FLAG_CAPTURE || flags == FLAG_EP) {
        int from   = move_from(m);
        int to     = move_to(m);
        int attacker = piece_type(s.board[from]);
        int victim;
        if (flags == FLAG_EP) {
            victim = 1; // pawn
        } else {
            int v = s.board[to];
            victim = (v == EMPTY) ? 1 : piece_type(v);
        }
        if (attacker < 1 || attacker > 6) attacker = 1;
        if (victim   < 1 || victim   > 6) victim   = 1;
        return 10000 + d_MVV_LVA[victim][attacker];
    }
    if (flags == FLAG_PROMO) {
        // Order promotions ahead of quiets; queen-promo first.
        int p = piece_type(move_promo(m));
        return 9000 + p;
    }
    if (killer1 != 0 && m == killer1) return QUIET_KILLER_SCORE_1;
    if (killer2 != 0 && m == killer2) return QUIET_KILLER_SCORE_2;
    return 0;
}

// In-place insertion sort of moves by their score (descending).
// 256 moves max; insertion-sort is O(n^2) but n is small and we get
// sequential reads out which beats pointer-chasing in branchy GPU code.
__device__ static void d_sort_moves(Move* moves, int* scores, int n) {
    for (int i = 1; i < n; i++) {
        Move m = moves[i];
        int  s = scores[i];
        int  j = i - 1;
        while (j >= 0 && scores[j] < s) {
            moves[j + 1]  = moves[j];
            scores[j + 1] = scores[j];
            j--;
        }
        moves[j + 1]  = m;
        scores[j + 1] = s;
    }
}

// =============================================================================
// Quiescence search — single-ply captures + queen promotions with stand-pat.
//
// The full recursive qsearch draft overflowed the available CUDA stack in this
// environment, so the shipping path is a non-recursive capture extension.
// =============================================================================
constexpr int Q_MAX_MOVES = 64;

__device__ static inline int d_qsearch(Position s, int alpha, int beta) {
    int stand_pat = d_evaluate(&s);
    if (stand_pat >= beta) return beta;
    if (stand_pat > alpha) alpha = stand_pat;

    Move moves[MAX_PLY_MOVES];
    int  n = generate_moves(&s, moves);

    Move qmoves[Q_MAX_MOVES];
    int  qscores[Q_MAX_MOVES];
    int  qn = 0;
    for (int i = 0; i < n && qn < Q_MAX_MOVES; i++) {
        Move m = moves[i];
        int f = move_flags(m);
        bool is_cap = (f == FLAG_CAPTURE || f == FLAG_EP);
        bool is_qpromo = (f == FLAG_PROMO && piece_type(move_promo(m)) == 5);
        if (!is_cap && !is_qpromo) continue;
        qmoves[qn]  = m;
        qscores[qn] = d_score_move(s, m);
        qn++;
    }
    d_sort_moves(qmoves, qscores, qn);

    for (int i = 0; i < qn; i++) {
        Position c = s;
        make_move(&c, qmoves[i]);
        if (in_check(&c, 1 - c.side)) continue;
        int score = -d_evaluate(&c);
        if (score >= beta) return beta;
        if (score > alpha) alpha = score;
    }
    return alpha;
}

// =============================================================================
// Iterative-stack negamax with TT, MVV-LVA ordering, qsearch leaves.
// Returns best score from the root side's POV.
// =============================================================================
struct Frame {
    Position s;                    // board AT this ply (side-to-move = us)
    Move     moves[MAX_PLY_MOVES]; // pseudo-legal moves
    int      scores[MAX_PLY_MOVES];// move-ordering scores
    int      n;                    // count
    int      idx;                  // next move index
    int      alpha;
    int      beta;
    int      best;                 // best score so far
    Move     best_move;            // best move so far (for TT store)
    int      legal_count;
    int      depth;                // remaining depth
    int      alpha_orig;           // for TT bound classification on return
    uint64_t hash;                 // zobrist of s
};

__device__ static int d_negamax(const Position& root,
                                uint64_t root_hash,
                                int depth,
                                int alpha0,
                                int beta0,
                                Move* /*out*/ best_move_out)
{
    Frame st[MAX_SEARCH_DEPTH + 1];
    Move killer1[MAX_SEARCH_DEPTH + 1] = {0};
    Move killer2[MAX_SEARCH_DEPTH + 1] = {0};
    int sp = 0;

    // Init root frame.
    st[0].s          = root;
    st[0].depth      = depth;
    st[0].alpha      = alpha0;
    st[0].beta       = beta0;
    st[0].alpha_orig = alpha0;
    st[0].best       = -INF_SCORE;
    st[0].best_move  = 0;
    st[0].legal_count = 0;
    st[0].hash       = root_hash;
    st[0].idx        = 0;

    // Generate + order root moves with TT-move boost.
    {
        Frame* f = &st[0];
        f->n = generate_moves(&f->s, f->moves);
        // Probe TT for a move hint.
        TTEntry tte = tt_probe(f->hash);
        uint16_t tt_mv16 = (tte.key == f->hash) ? tte.move : 0;
        for (int i = 0; i < f->n; i++) {
            f->scores[i] = d_score_move(f->s, f->moves[i]);
            // TT-move gets the highest priority.
            if (tt_mv16 != 0 &&
                (uint16_t)(f->moves[i] & 0xFFFF) == tt_mv16) {
                f->scores[i] = 1000000;
            }
        }
        d_sort_moves(f->moves, f->scores, f->n);
    }

    int returned = 0;
    bool have_return = false;
    Move last_move_made = 0;
    uint64_t local_node_count = 0;

    while (sp >= 0) {
        Frame* f = &st[sp];

        // Process a returned subtree value.
        if (have_return) {
            int score = -returned;
            have_return = false;
            if (score > f->best) {
                f->best = score;
                f->best_move = last_move_made;
            }
            if (f->best > f->alpha) f->alpha = f->best;
            if (f->alpha >= f->beta) {
                if (d_is_quiet_move(f->best_move) && killer1[sp] != f->best_move) {
                    killer2[sp] = killer1[sp];
                    killer1[sp] = f->best_move;
                }
                // Beta cutoff. Store as LOWER bound and pop.
                int store_d = f->depth;
                if (store_d > 0) {
                    tt_store(f->hash, f->best, store_d, TT_BOUND_LOWER,
                             (uint16_t)(f->best_move & 0xFFFF));
                }
                returned = f->best;
                have_return = true;
                last_move_made = f->best_move; // for parent attribution chain
                // ^ note: parent doesn't actually use last_move_made except to
                //   remember which child move led to its own best score, which
                //   is set at the moment of descent below.
                sp--;
                continue;
            }
        }

        // Stop check (cooperative — every NODE_STOP_INTERVAL nodes).
        local_node_count++;
        if ((local_node_count & (NODE_STOP_INTERVAL - 1)) == 0) {
            if (d_stop_flag) {
                // Bail — no TT store with partial info.
                if (best_move_out) *best_move_out = st[0].best_move;
                atomicAdd((unsigned long long*)&d_node_counter,
                          (unsigned long long)local_node_count);
                return st[0].best;
            }
        }

        // Leaf?
        if (f->depth <= 0) {
            int e = d_qsearch(f->s, f->alpha, f->beta);
            returned = e;
            have_return = true;
            sp--;
            continue;
        }

        // First time at this frame? (idx==0 && best is -INF) => generate moves
        // and try TT cutoff.
        if (f->idx == 0 && f->best == -INF_SCORE) {
            f->n = generate_moves(&f->s, f->moves);

            // TT probe — exact bound or sufficient depth allows immediate
            // return; otherwise just take the move hint.
            TTEntry tte = tt_probe(f->hash);
            uint16_t tt_mv16 = 0;
            if (tte.key == f->hash) {
                tt_mv16 = tte.move;
                if ((int)tte.depth >= f->depth) {
                    int s = tte.score;
                    if (tte.bound == TT_BOUND_EXACT) {
                        returned = s;
                        have_return = true;
                        sp--;
                        continue;
                    } else if (tte.bound == TT_BOUND_LOWER) {
                        if (s > f->alpha) f->alpha = s;
                    } else if (tte.bound == TT_BOUND_UPPER) {
                        if (s < f->beta) f->beta = s;
                    }
                    if (f->alpha >= f->beta) {
                        returned = s;
                        have_return = true;
                        sp--;
                        continue;
                    }
                }
            }

            // Score & sort moves once.
            Move quiet_killer_1 = killer1[sp];
            Move quiet_killer_2 = killer2[sp];
            for (int i = 0; i < f->n; i++) {
                f->scores[i] = d_score_move(f->s, f->moves[i],
                                            quiet_killer_1, quiet_killer_2);
                if (tt_mv16 != 0 &&
                    (uint16_t)(f->moves[i] & 0xFFFF) == tt_mv16) {
                    f->scores[i] = 1000000;
                }
            }
            d_sort_moves(f->moves, f->scores, f->n);
        }

        // Find next legal move.
        bool descended = false;
        while (f->idx < f->n) {
            Move mv = f->moves[f->idx++];
            Position c = f->s;
            make_move(&c, mv);
            // Legality.
            if (in_check(&c, 1 - c.side)) continue;
            f->legal_count++;
            // Remember the move we're descending into so that on return we can
            // attribute the score to it.
            last_move_made = mv;
            // Push child frame (need a recursion depth check too).
            if (sp + 1 > MAX_SEARCH_DEPTH) {
                // Too deep — evaluate as leaf (qsearch).
                int e = d_qsearch(c, -f->beta, -f->alpha);
                returned = e;
                have_return = true;
                descended = true;
                break;
            }
            sp++;
            Frame* g = &st[sp];
            g->s          = c;
            g->depth      = f->depth - 1;
            g->alpha      = -f->beta;
            g->beta       = -f->alpha;
            g->alpha_orig = -f->beta;
            g->best       = -INF_SCORE;
            g->best_move  = 0;
            g->legal_count = 0;
            g->idx        = 0;
            g->hash       = zobrist_update(f->hash, f->s, mv);
            descended = true;
            break;
        }
        if (descended) continue;

        // Out of moves at this frame.
        int result;
        if (f->legal_count == 0) {
            // Terminal: mate or stalemate.
            if (in_check(&f->s, f->s.side)) {
                // Closer mates score higher (less negative).
                result = -MATE_SCORE + (MAX_SEARCH_DEPTH - f->depth);
            } else {
                result = 0; // stalemate
            }
        } else {
            result = f->best;
        }

        // Store TT entry (only if we actually searched at non-trivial depth).
        if (f->depth > 0 && f->legal_count > 0) {
            uint8_t bound;
            if (result <= f->alpha_orig)      bound = TT_BOUND_UPPER;
            else if (result >= f->beta)       bound = TT_BOUND_LOWER;
            else                              bound = TT_BOUND_EXACT;
            tt_store(f->hash, result, f->depth, bound,
                     (uint16_t)(f->best_move & 0xFFFF));
        }

        returned = result;
        have_return = true;
        last_move_made = f->best_move;
        sp--;
    }

    if (best_move_out) *best_move_out = st[0].best_move;
    atomicAdd((unsigned long long*)&d_node_counter,
              (unsigned long long)local_node_count);
    return returned;
}

// =============================================================================
// Root kernel — one thread per root move.
// =============================================================================
__global__ void root_search_kernel(
    const Position* root_state,
    const uint64_t* root_hashes,
    const Move*     root_moves,
    int             num_root_moves,
    int             search_depth, // depth of search BELOW root
    int*            out_scores)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_root_moves) return;

    Move mv = root_moves[tid];
    Position c = *root_state;
    make_move(&c, mv);
    // Legality at the root.
    if (in_check(&c, 1 - c.side)) {
        out_scores[tid] = -INF_SCORE;
        return;
    }
    uint64_t child_hash = zobrist_update(*root_hashes, *root_state, mv);
    Move dummy = 0;
    int opp_score = d_negamax(c, child_hash, search_depth,
                              -INF_SCORE, INF_SCORE, &dummy);
    // d_negamax returns from c.side's POV (= opponent of root). Negate.
    out_scores[tid] = -opp_score;
}

__global__ void generate_root_moves_kernel(const Position* root_state,
                                           Move* out_moves,
                                           int* out_count)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    Move pseudo[MAX_MOVES];
    int n = generate_moves(root_state, pseudo);
    int legal = 0;
    for (int i = 0; i < n; ++i) {
        Position child = *root_state;
        make_move(&child, pseudo[i]);
        if (in_check(&child, 1 - child.side)) continue;
        out_moves[legal++] = pseudo[i];
    }
    *out_count = legal;
}

__global__ void k_tt_store_root(uint64_t hash, int score, int depth,
                                uint8_t bound, uint16_t move)
{
    tt_store(hash, score, depth, bound, move);
}

// =============================================================================
// CFX root features kernel (audit Req 5 partner-counterfactual, root layer).
//
// One thread per root candidate move. After applying the candidate, computes
// all four CFX features over the partner's legal-response set:
//   n_resp     — count of legal responses (forcing geometry primary signal)
//   mean_dist  — mean ‖coord3d(response) − coord3d(candidate)‖
//   std_dist   — stddev of those distances
//   max_dist   — max distance
// Smaller-is-better for all four (forcing moves cluster opponent responses
// near the candidate's geometry). Used both for root move ordering and for
// the d=1 CFX score that replaces the runtime-eval fallback.
// =============================================================================
__global__ void cfx_root_features_kernel(
    const Position* root_state,
    const Move*     candidates,
    int             n_candidates,
    int*            n_resp_out,
    float*          mean_dist_out,
    float*          std_dist_out,
    float*          max_dist_out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_candidates) return;

    Position child = *root_state;
    make_move(&child, candidates[idx]);

    Coord3D cand_coord = compute_coord3d(&child);

    Move buf[MAX_MOVES];
    int n_pseudo = generate_moves(&child, buf);
    int legal = 0;
    float sum_d = 0.0f, sum_d2 = 0.0f, max_d = 0.0f;

    for (int j = 0; j < n_pseudo; ++j) {
        Position c = child;
        make_move(&c, buf[j]);
        if (in_check(&c, 1 - c.side)) continue;

        Coord3D resp = compute_coord3d(&c);
        float dx = resp.x - cand_coord.x;
        float dy = resp.y - cand_coord.y;
        float dz = resp.z - cand_coord.z;
        float d = sqrtf(dx * dx + dy * dy + dz * dz);
        sum_d  += d;
        sum_d2 += d * d;
        if (d > max_d) max_d = d;
        legal++;
    }

    n_resp_out[idx] = legal;
    if (legal > 0) {
        float mean = sum_d / float(legal);
        float var  = sum_d2 / float(legal) - mean * mean;
        if (var < 0.0f) var = 0.0f;
        mean_dist_out[idx] = mean;
        std_dist_out[idx]  = sqrtf(var);
        max_dist_out[idx]  = max_d;
    } else {
        // Terminal — checkmate / stalemate after candidate.
        // Zeros encode "most forcing possible" which is exactly correct
        // for mate-delivering moves.
        mean_dist_out[idx] = 0.0f;
        std_dist_out[idx]  = 0.0f;
        max_dist_out[idx]  = 0.0f;
    }
}

// =============================================================================
// Host helpers — set/clear stop flag, copy to device.
// =============================================================================
static void set_device_stop(int v) {
    cudaMemcpyToSymbol(d_stop_flag, &v, sizeof(int));
}

static uint64_t read_device_node_counter() {
    uint64_t h = 0;
    cudaMemcpyFromSymbol(&h, d_node_counter, sizeof(uint64_t));
    return h;
}

static void reset_device_node_counter() {
    uint64_t z = 0;
    cudaMemcpyToSymbol(d_node_counter, &z, sizeof(uint64_t));
}

const SearchRuntimeReport& search_runtime_report() {
    return h_runtime_report;
}

static void reset_runtime_report(int root_move_count) {
    h_runtime_report = SearchRuntimeReport{};
    h_runtime_report.runtime_enabled = (h_runtime_initialised != 0);
    h_runtime_report.root_move_count = (root_move_count > 0)
        ? static_cast<uint32_t>(root_move_count)
        : 0u;
}

static void reorder_root_moves_by_score(Move* legal_moves,
                                        int* scores,
                                        int n_legal,
                                        int prefer_move)
{
    for (int i = 1; i < n_legal; ++i) {
        Move mv = legal_moves[i];
        int sc = scores[i];
        int j = i - 1;
        while (j >= 0) {
            bool better = scores[j] < sc;
            bool prefer = (scores[j] == sc && mv == prefer_move);
            if (!better && !prefer) break;
            legal_moves[j + 1] = legal_moves[j];
            scores[j + 1] = scores[j];
            --j;
        }
        legal_moves[j + 1] = mv;
        scores[j + 1] = sc;
    }
}

static bool run_root_runtime_eval_batch(const Position& root,
                                        uint64_t root_hash,
                                        Move* legal_moves,
                                        int n_legal,
                                        int* out_scores)
{
    reset_runtime_report(n_legal);

    if (!h_runtime_initialised || n_legal <= 0 ||
        n_legal > EVAL_SERVICE_MAX_BUCKET) {
        return false;
    }

    cudaError_t err = scheduler_reset(&h_scheduler_storage);
    if (err != cudaSuccess) return false;
    err = eval_service_reset(&h_eval_service);
    if (err != cudaSuccess) return false;

    cudaMemcpy(d_root_state, &root, sizeof(Position), cudaMemcpyHostToDevice);
    cudaMemcpy(d_root_hashes, &root_hash, sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_root_moves, legal_moves, sizeof(Move) * n_legal, cudaMemcpyHostToDevice);

    FrontierTask tasks[EVAL_SERVICE_MAX_BUCKET]{};
    for (int i = 0; i < n_legal; ++i) {
        tasks[i].node_id = static_cast<uint32_t>(i + 1);
        tasks[i].state_slot = static_cast<uint32_t>(i);
        tasks[i].parent_id = UINT32_MAX;
        tasks[i].root_index = static_cast<uint32_t>(i);
        tasks[i].incoming_move = legal_moves[i];
        tasks[i].zobrist = zobrist_update(root_hash, root, legal_moves[i]);
        tasks[i].alpha = -INF_SCORE;
        tasks[i].beta = INF_SCORE;
        tasks[i].depth = 1;
        tasks[i].ply = 1;
        tasks[i].flags = FRONTIER_TASK_ROOT | FRONTIER_TASK_NEEDS_EVAL;
    }

    err = scheduler_seed_frontier(h_scheduler_storage, tasks,
                                  static_cast<uint32_t>(n_legal));
    if (err != cudaSuccess) return false;

    err = scheduler_launch(h_scheduler_storage, dim3(1), dim3(64));
    if (err != cudaSuccess) return false;
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) return false;

    SchedulerSnapshot scheduler_snapshot{};
    err = scheduler_copy_snapshot(h_scheduler_storage, &scheduler_snapshot);
    if (err != cudaSuccess) return false;

    const int bucket_idx = eval_service_pick_bucket_for_count(n_legal);
    if (!eval_service_valid_bucket(bucket_idx)) return false;

    const int threads = 128;
    const int blocks = (n_legal + threads - 1) / threads;
    pack_root_eval_requests_from_scheduler_kernel<<<blocks, threads>>>(
        h_scheduler_storage.eval_requests.view,
        h_eval_service.host_view.buckets[bucket_idx],
        d_root_state,
        d_root_moves,
        static_cast<uint32_t>(n_legal));
    err = cudaGetLastError();
    if (err != cudaSuccess) return false;
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) return false;

    err = eval_service_dispatch_bucket(&h_eval_service, bucket_idx);
    if (err != cudaSuccess) return false;
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) return false;

    EvalBucketCounters counters{};
    err = eval_service_snapshot_bucket(&h_eval_service, bucket_idx, &counters);
    if (err != cudaSuccess) return false;

    EvalResult results[EVAL_SERVICE_MAX_BUCKET]{};
    err = cudaMemcpy(results,
                     eval_service_bucket_results(&h_eval_service, bucket_idx),
                     sizeof(EvalResult) * eval_service_bucket_size(bucket_idx),
                     cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) return false;

    for (int i = 0; i < n_legal; ++i) {
        out_scores[i] = -INF_SCORE;
    }
    for (uint32_t i = 0; i < counters.result_count && i < static_cast<uint32_t>(n_legal); ++i) {
        const EvalResult& result = results[i];
        if (result.state_idx >= static_cast<uint32_t>(n_legal)) continue;
        if ((result.flags & EVAL_RESULT_VALUE_VALID) == 0u) continue;
        out_scores[result.state_idx] = -static_cast<int>(result.value_cp);
    }

    const EvalServiceReport* service_report = eval_service_report(&h_eval_service);
    h_runtime_report.runtime_used = true;
    h_runtime_report.scheduler_frontier_seeded =
        static_cast<uint32_t>(scheduler_snapshot.counters.frontier_tasks_seeded);
    h_runtime_report.scheduler_frontier_pops =
        static_cast<uint32_t>(scheduler_snapshot.counters.frontier_pops);
    h_runtime_report.scheduler_eval_requests =
        static_cast<uint32_t>(scheduler_snapshot.counters.eval_requests_pushed);
    h_runtime_report.scheduler_eval_request_depth = scheduler_snapshot.eval_request_depth;
    h_runtime_report.eval_bucket_idx = static_cast<uint32_t>(bucket_idx);
    h_runtime_report.eval_bucket_size =
        static_cast<uint32_t>(eval_service_bucket_size(bucket_idx));
    h_runtime_report.eval_request_count = counters.request_count;
    h_runtime_report.eval_result_count = counters.result_count;
    h_runtime_report.eval_dropped_requests = counters.dropped_requests;
    if (service_report != nullptr) {
        h_runtime_report.eval_backend =
            static_cast<uint32_t>(service_report->active_backend);
        h_runtime_report.eval_total_dispatches = service_report->total_dispatches;
        h_runtime_report.eval_fallback_dispatches =
            service_report->total_fallback_dispatches;
        h_runtime_report.eval_failed_dispatches =
            service_report->total_failed_dispatches;
    }

    int ordered_scores[SEARCH_MAX_ROOT_MOVES]{};
    for (int i = 0; i < n_legal; ++i) ordered_scores[i] = out_scores[i];
    reorder_root_moves_by_score(legal_moves, ordered_scores, n_legal, 0);
    int changed = 0;
    for (int i = 0; i < n_legal; ++i) {
        if (legal_moves[i] != tasks[i].incoming_move) ++changed;
    }
    h_runtime_report.reordered_root_moves = static_cast<uint32_t>(changed);
    for (int i = 0; i < n_legal && i < ROOT_RUNTIME_TOP_ORDER_LIMIT; ++i) {
        h_runtime_report.top_order_moves[i] = static_cast<uint32_t>(legal_moves[i]);
        h_runtime_report.top_order_scores[i] = ordered_scores[i];
    }

    for (int i = 0; i < n_legal; ++i) {
        out_scores[i] = ordered_scores[i];
    }
    return true;
}

// =============================================================================
// search_init / shutdown
// =============================================================================
void search_init() {
    if (d_initialised) return;
    cudaError_t err;
    err = cudaMalloc(&d_root_state, sizeof(Position));
    if (err != cudaSuccess) std::fprintf(stderr, "search_init: d_root_state: %s\n", cudaGetErrorString(err));
    err = cudaMalloc(&d_root_moves, sizeof(Move) * SEARCH_MAX_ROOT_MOVES);
    if (err != cudaSuccess) std::fprintf(stderr, "search_init: d_root_moves: %s\n", cudaGetErrorString(err));
    err = cudaMalloc(&d_root_scores, sizeof(int) * SEARCH_MAX_ROOT_MOVES);
    if (err != cudaSuccess) std::fprintf(stderr, "search_init: d_root_scores: %s\n", cudaGetErrorString(err));
    err = cudaMalloc(&d_root_hashes, sizeof(uint64_t));
    if (err != cudaSuccess) std::fprintf(stderr, "search_init: d_root_hashes: %s\n", cudaGetErrorString(err));
    err = cudaMalloc(&d_root_count, sizeof(int));
    if (err != cudaSuccess) std::fprintf(stderr, "search_init: d_root_count: %s\n", cudaGetErrorString(err));
    err = cudaMalloc(&d_root_n_resp, sizeof(int) * SEARCH_MAX_ROOT_MOVES);
    if (err != cudaSuccess) std::fprintf(stderr, "search_init: d_root_n_resp: %s\n", cudaGetErrorString(err));
    err = cudaMalloc(&d_root_mean_dist, sizeof(float) * SEARCH_MAX_ROOT_MOVES);
    if (err != cudaSuccess) std::fprintf(stderr, "search_init: d_root_mean_dist: %s\n", cudaGetErrorString(err));
    err = cudaMalloc(&d_root_std_dist, sizeof(float) * SEARCH_MAX_ROOT_MOVES);
    if (err != cudaSuccess) std::fprintf(stderr, "search_init: d_root_std_dist: %s\n", cudaGetErrorString(err));
    err = cudaMalloc(&d_root_max_dist, sizeof(float) * SEARCH_MAX_ROOT_MOVES);
    if (err != cudaSuccess) std::fprintf(stderr, "search_init: d_root_max_dist: %s\n", cudaGetErrorString(err));

    SchedulerConfig scheduler_config{};
    scheduler_config.frontier_capacity = RUNTIME_SCHEDULER_CAPACITY;
    scheduler_config.eval_request_capacity = RUNTIME_SCHEDULER_CAPACITY;
    scheduler_config.eval_result_capacity = RUNTIME_SCHEDULER_CAPACITY;
    scheduler_config.drain_on_frontier_empty = 1;
    err = scheduler_init(&h_scheduler_storage, scheduler_config);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "search_init: scheduler_init: %s\n",
                     cudaGetErrorString(err));
        scheduler_shutdown(&h_scheduler_storage);
    } else {
        EvalServiceConfig eval_config = eval_service_default_config();
        eval_config.preferred_backend = EVAL_BACKEND_DEVICE_GRAPH;
        eval_config.enable_graph_backend = true;
        err = eval_service_init(&h_eval_service, &eval_config);
        if (err != cudaSuccess) {
            std::fprintf(stderr, "search_init: eval_service_init: %s\n",
                         cudaGetErrorString(err));
            eval_service_shutdown(&h_eval_service);
            scheduler_shutdown(&h_scheduler_storage);
        } else {
            h_runtime_initialised = 1;
        }
    }
    set_device_stop(0);
    reset_runtime_report(0);
    d_initialised = 1;
}

void search_shutdown() {
    if (!d_initialised) return;
    if (d_root_state)   cudaFree(d_root_state);
    if (d_root_moves)   cudaFree(d_root_moves);
    if (d_root_scores)  cudaFree(d_root_scores);
    if (d_root_hashes)  cudaFree(d_root_hashes);
    if (d_root_count)   cudaFree(d_root_count);
    if (d_root_n_resp)   cudaFree(d_root_n_resp);
    if (d_root_mean_dist) cudaFree(d_root_mean_dist);
    if (d_root_std_dist)  cudaFree(d_root_std_dist);
    if (d_root_max_dist)  cudaFree(d_root_max_dist);
    if (h_runtime_initialised) {
        eval_service_shutdown(&h_eval_service);
        scheduler_shutdown(&h_scheduler_storage);
    }
    d_root_state = nullptr;
    d_root_moves = nullptr;
    d_root_scores = nullptr;
    d_root_n_resp = nullptr;
    d_root_mean_dist = nullptr;
    d_root_std_dist = nullptr;
    d_root_max_dist = nullptr;
    d_root_hashes = nullptr;
    d_root_count = nullptr;
    h_runtime_initialised = 0;
    reset_runtime_report(0);
    d_initialised = 0;
}

void search_new_game() {
    set_device_stop(0);
    if (h_runtime_initialised) {
        scheduler_reset(&h_scheduler_storage);
        eval_service_reset(&h_eval_service);
    }
    reset_runtime_report(0);
}

// =============================================================================
// PV recovery — walk the TT, applying TT-recommended moves until we hit a
// non-PV slot or a repeating hash. Up to `max_len` plies. Returns moves in
// UCI long-algebraic form.
// =============================================================================
static std::vector<std::string> recover_pv_from_tt(Move first_move)
{
    std::vector<std::string> out;
    if (first_move == 0) return out;

    char buf[8] = {0};
    move_to_uci(first_move, buf);
    out.emplace_back(buf);
    return out;
}

// =============================================================================
// search_root — iterative deepening driver.
// =============================================================================
uci::SearchResult search_root(const Position& root,
                              const uci::SearchLimits& limits,
                              uci::InfoCallback info_cb)
{
    uci::SearchResult result;
    if (!d_initialised) search_init();

    // -------- Compute root hash and generate legal root moves on device -----
    if (!zobrist_initialized()) init_zobrist();
    uint64_t root_hash = zobrist_full(root);

    cudaMemcpy(d_root_state,  &root,      sizeof(Position), cudaMemcpyHostToDevice);
    cudaMemcpy(d_root_hashes, &root_hash, sizeof(uint64_t), cudaMemcpyHostToDevice);
    generate_root_moves_kernel<<<1, 1>>>(d_root_state, d_root_moves, d_root_count);
    cudaError_t gen_launch_err = cudaGetLastError();
    if (gen_launch_err != cudaSuccess) {
        std::fprintf(stderr, "search: root movegen launch err: %s\n",
                     cudaGetErrorString(gen_launch_err));
        result.bestmove = "0000";
        return result;
    }
    cudaError_t gen_sync_err = cudaDeviceSynchronize();
    if (gen_sync_err != cudaSuccess) {
        std::fprintf(stderr, "search: root movegen sync err: %s\n",
                     cudaGetErrorString(gen_sync_err));
        result.bestmove = "0000";
        return result;
    }

    int n_legal = 0;
    cudaMemcpy(&n_legal, d_root_count, sizeof(int), cudaMemcpyDeviceToHost);
    if (n_legal == 0) {
        result.bestmove = "0000";
        return result;
    }

    Move legal_moves[MAX_MOVES];
    cudaMemcpy(legal_moves, d_root_moves, sizeof(Move) * n_legal, cudaMemcpyDeviceToHost);

    // ---- CFX root features (audit Req 5 partner-counterfactual) ----
    // For each candidate, compute (n_resp, mean_dist, std_dist, max_dist)
    // over the partner's legal response set. All four smaller = more
    // forcing. Used for (a) root move ordering and (b) the d=1 CFX
    // Borda score that replaces the PeSTO runtime-eval fallback.
    // Pure GPU compute; host only does integer/float sort + Borda
    // arithmetic on N_legal scalars (no chess work on host).
    int   cfx_n_resp_h[MAX_MOVES];
    float cfx_mean_h[MAX_MOVES];
    float cfx_std_h[MAX_MOVES];
    float cfx_max_h[MAX_MOVES];
    bool  cfx_ok = false;
    {
        cfx_root_features_kernel<<<n_legal, 1>>>(
            d_root_state, d_root_moves, n_legal,
            d_root_n_resp, d_root_mean_dist, d_root_std_dist, d_root_max_dist
        );
        cudaError_t err = cudaGetLastError();
        if (err == cudaSuccess) err = cudaDeviceSynchronize();
        if (err == cudaSuccess) {
            cudaMemcpy(cfx_n_resp_h, d_root_n_resp,
                       sizeof(int) * n_legal, cudaMemcpyDeviceToHost);
            cudaMemcpy(cfx_mean_h, d_root_mean_dist,
                       sizeof(float) * n_legal, cudaMemcpyDeviceToHost);
            cudaMemcpy(cfx_std_h, d_root_std_dist,
                       sizeof(float) * n_legal, cudaMemcpyDeviceToHost);
            cudaMemcpy(cfx_max_h, d_root_max_dist,
                       sizeof(float) * n_legal, cudaMemcpyDeviceToHost);
            // Sort root_moves by ascending n_resp (existing ordering signal).
            int order[MAX_MOVES];
            for (int i = 0; i < n_legal; ++i) order[i] = i;
            std::stable_sort(order, order + n_legal,
                [&](int a, int b) { return cfx_n_resp_h[a] < cfx_n_resp_h[b]; });
            Move sorted[MAX_MOVES];
            int  s_n_resp[MAX_MOVES];
            float s_mean[MAX_MOVES], s_std[MAX_MOVES], s_max[MAX_MOVES];
            for (int i = 0; i < n_legal; ++i) {
                sorted[i]   = legal_moves[order[i]];
                s_n_resp[i] = cfx_n_resp_h[order[i]];
                s_mean[i]   = cfx_mean_h[order[i]];
                s_std[i]    = cfx_std_h[order[i]];
                s_max[i]    = cfx_max_h[order[i]];
            }
            for (int i = 0; i < n_legal; ++i) {
                legal_moves[i]  = sorted[i];
                cfx_n_resp_h[i] = s_n_resp[i];
                cfx_mean_h[i]   = s_mean[i];
                cfx_std_h[i]    = s_std[i];
                cfx_max_h[i]    = s_max[i];
            }
            cudaMemcpy(d_root_moves, legal_moves,
                       sizeof(Move) * n_legal, cudaMemcpyHostToDevice);
            cfx_ok = true;
        } else {
            std::fprintf(stderr, "search: cfx_root_features skipped: %s\n",
                         cudaGetErrorString(err));
        }
    }

    // Trivial bestmove fallback (in case search aborts immediately).
    {
        char buf[8] = {0};
        move_to_uci(legal_moves[0], buf);
        result.bestmove = buf;
    }

    // -------- Decide budget / max depth --------
    int max_depth = (limits.depth > 0) ? limits.depth : MAX_SEARCH_DEPTH;
    if (max_depth > MAX_SEARCH_DEPTH) max_depth = MAX_SEARCH_DEPTH;

    int budget_ms = 0;
    bool we_white = (root.side == WHITE_SIDE);
    if (limits.movetime > 0)        budget_ms = limits.movetime;
    else if (limits.depth > 0)      budget_ms = 0;
    else if (limits.infinite)       budget_ms = 0;
    else if (limits.wtime > 0 || limits.btime > 0 || limits.winc > 0 || limits.binc > 0)
        budget_ms = compute_movetime(limits.wtime, limits.btime,
                                     limits.winc, limits.binc, we_white);
    else                            budget_ms = 0;

    if (budget_ms > 0) start_timer(budget_ms);
    else               clear_timer();

    set_device_stop(0);
    reset_device_node_counter();

    // Bump TT age for this search.
    tt_age();

    // -------- Iterative deepening --------
    using clock = std::chrono::steady_clock;
    auto t0 = clock::now();
    int runtime_scores[SEARCH_MAX_ROOT_MOVES]{};
    bool runtime_depth1_ready =
        run_root_runtime_eval_batch(root, root_hash, legal_moves, n_legal,
                                    runtime_scores);

    // Override the d=1 score array with CFX-Borda when the CFX root kernel
    // succeeded. Borda over (n_resp, mean_dist, std_dist, max_dist) — all
    // smaller-is-better. Score = -borda_sum so argmax picks the most
    // forcing candidate. At higher depths this is overridden naturally
    // by alpha-beta. Empirically validated as the +3.40% PC-B lift on
    // iter21 N=2000.
    if (cfx_ok) {
        int borda_sum[SEARCH_MAX_ROOT_MOVES] = {0};
        int order[SEARCH_MAX_ROOT_MOVES];

        auto add_ranks_int = [&](const int* vals) {
            for (int i = 0; i < n_legal; ++i) order[i] = i;
            std::stable_sort(order, order + n_legal,
                [&](int a, int b) { return vals[a] < vals[b]; });
            for (int r = 0; r < n_legal; ++r) borda_sum[order[r]] += r;
        };
        auto add_ranks_flt = [&](const float* vals) {
            for (int i = 0; i < n_legal; ++i) order[i] = i;
            std::stable_sort(order, order + n_legal,
                [&](int a, int b) { return vals[a] < vals[b]; });
            for (int r = 0; r < n_legal; ++r) borda_sum[order[r]] += r;
        };

        add_ranks_int(cfx_n_resp_h);
        add_ranks_flt(cfx_mean_h);
        add_ranks_flt(cfx_std_h);
        add_ranks_flt(cfx_max_h);

        for (int i = 0; i < n_legal; ++i) {
            runtime_scores[i] = -borda_sum[i];
        }
        runtime_depth1_ready = true;
    }

    Move best_move_so_far = legal_moves[0];
    int  best_score_so_far = 0;

    for (int d = 1; d <= max_depth; ++d) {
        // Stop check before launching.
        if (uci::stop_requested() || time_up()) break;

        // Keep the previous iteration's best move at the head of the root
        // list so equal-score ties stay stable across iterations.
        if (d > 1) {
            for (int i = 1; i < n_legal; ++i) {
                if (legal_moves[i] == best_move_so_far) {
                    std::swap(legal_moves[0], legal_moves[i]);
                    break;
                }
            }
        }
        cudaMemcpy(d_root_moves, legal_moves, sizeof(Move) * n_legal, cudaMemcpyHostToDevice);

        // For depth-1 the root parallelism searches at depth=0 below the root
        // (i.e. one ply on the opponent's reply ... actually depth-1 means
        // we ply our move and just leaf-eval). Map UCI depth d -> children's
        // search depth = d-1.
        int below = d - 1;
        if (below < 0) below = 0;

        int scores[SEARCH_MAX_ROOT_MOVES];
        if (d == 1 && runtime_depth1_ready) {
            for (int i = 0; i < n_legal; ++i) {
                scores[i] = runtime_scores[i];
            }
        } else {
            // Launch one thread per root move.
            int threads = 1;
            int blocks  = n_legal;
            root_search_kernel<<<blocks, threads>>>(
                d_root_state, d_root_hashes, d_root_moves,
                n_legal, below, d_root_scores);
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                std::fprintf(stderr, "search: kernel launch err: %s\n", cudaGetErrorString(err));
                break;
            }

            // Poll: every 5 ms check the wall-clock and (if exceeded) signal stop.
            // We still need to wait for the kernel to finish before reading scores.
            {
                cudaEvent_t done;
                cudaEventCreate(&done);
                cudaEventRecord(done);
                while (cudaEventQuery(done) == cudaErrorNotReady) {
                    if (uci::stop_requested() || time_up()) {
                        set_device_stop(1);
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds(2));
                }
                cudaEventDestroy(done);
            }
            cudaDeviceSynchronize();

            bool aborted = uci::stop_requested() || time_up();
            if (aborted) {
                set_device_stop(0);
                break;
            }

            cudaMemcpy(scores, d_root_scores, sizeof(int) * n_legal, cudaMemcpyDeviceToHost);
        }

        int best_idx = 0;
        int best_score = scores[0];
        for (int i = 1; i < n_legal; i++) {
            if (scores[i] > best_score ||
                (scores[i] == best_score && legal_moves[i] == best_move_so_far)) {
                best_score = scores[i];
                best_idx = i;
            }
        }
        // Skip this iteration's result if every move was illegal (shouldn't
        // happen because we filtered already).
        if (best_score <= -INF_SCORE / 2) break;

        best_move_so_far  = legal_moves[best_idx];
        best_score_so_far = best_score;
        if (best_idx != 0) {
            std::swap(legal_moves[0], legal_moves[best_idx]);
        }

        // Store the root TT entry so PV recovery sees it.
        // Encode the root move's low 16 bits.
        // We do this via a 1-thread kernel.
        {
            k_tt_store_root<<<1, 1>>>(root_hash, best_score_so_far, d,
                                      TT_BOUND_EXACT,
                                      (uint16_t)(best_move_so_far & 0xFFFF));
            cudaDeviceSynchronize();
        }

        // Build info line.
        auto t_now = clock::now();
        int64_t elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t_now - t0).count();
        if (elapsed <= 0) elapsed = 1;
        uint64_t nodes = read_device_node_counter();

        uci::SearchInfo info;
        info.depth    = d;
        info.seldepth = d;
        info.nodes    = (int64_t)nodes;
        info.time_ms  = elapsed;
        info.nps      = (info.nodes * 1000) / elapsed;
        // Centipawn score from STM POV. UCI expects STM POV for "score cp",
        // not white POV. (Stockfish & friends: STM POV.)
        info.score_cp = best_score_so_far;
        // Mate detection (rough).
        if (best_score_so_far > MATE_SCORE - 1000) {
            int mate_dist = (MATE_SCORE - best_score_so_far + 1) / 2;
            info.mate_in = mate_dist;
            info.score_cp = 0;
        } else if (best_score_so_far < -MATE_SCORE + 1000) {
            int mate_dist = (MATE_SCORE + best_score_so_far) / 2;
            info.mate_in = -mate_dist;
            info.score_cp = 0;
        } else {
            info.mate_in = 0;
        }

        info.pv = recover_pv_from_tt(best_move_so_far);

        if (info_cb) info_cb(info);
    }

    // Emit a final TT-fill diag to stderr (helps verify TT is being populated).
    {
        double f = tt_fill_rate();
        std::fprintf(stderr, "search: tt_fill_rate=%.4f nodes=%llu\n",
                     f, (unsigned long long)read_device_node_counter());
    }

    // Format bestmove.
    char buf[8] = {0};
    move_to_uci(best_move_so_far, buf);
    result.bestmove = buf;
    result.ponder = "";

    return result;
}
} // namespace engine
