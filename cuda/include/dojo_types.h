// =============================================================================
// CUDA DOJO shared types — the coordination spine across kernels.
// Every fused_* kernel in this lab agrees on these layouts.
// =============================================================================
// CompactBoard: 64 squares + side-to-move. Matches batch_eval.cu.
// KnobConfig:   mirror of the JS fighter's FORGE_KNOBS. Hot-loadable via
//               cudaMemcpyToSymbol into __constant__ memory between launches.
// =============================================================================
#ifndef DOJO_TYPES_H
#define DOJO_TYPES_H

#include <stdint.h>

// -----------------------------------------------------------------------------
// Piece codes (must match batch_eval.cu / chess_mcts.cu)
// -----------------------------------------------------------------------------
#define DOJO_EMPTY   0
#define DOJO_WPAWN   1
#define DOJO_WKNIGHT 2
#define DOJO_WBISHOP 3
#define DOJO_WROOK   4
#define DOJO_WQUEEN  5
#define DOJO_WKING   6
#define DOJO_BPAWN   7
#define DOJO_BKNIGHT 8
#define DOJO_BBISHOP 9
#define DOJO_BROOK   10
#define DOJO_BQUEEN  11
#define DOJO_BKING   12

#define DOJO_WHITE 0
#define DOJO_BLACK 1

// piece_type ∈ [1..6]; IS_WHITE(p) = (p>=1 && p<=6)
#define DOJO_PTYPE(p) ((p) >= DOJO_WPAWN && (p) <= DOJO_WKING ? (p) : (p) - 6)
#define DOJO_IS_WHITE(p) ((p) >= DOJO_WPAWN && (p) <= DOJO_WKING)
#define DOJO_COLOR(p) (DOJO_IS_WHITE(p) ? DOJO_WHITE : DOJO_BLACK)

#define DOJO_MAX_POSITIONS 65536

// -----------------------------------------------------------------------------
// CompactBoard — the one true board layout on GPU.
// Square index 0 = a8 (top-left in FEN order). Square 63 = h1.
//   rank = sq >> 3   (0 = 8th rank, 7 = 1st rank)
//   file = sq & 7    (0 = a-file, 7 = h-file)
// -----------------------------------------------------------------------------
struct CompactBoard {
    int8_t board[64];     // piece codes per square
    int8_t side;          // DOJO_WHITE=0, DOJO_BLACK=1 (side to move)
    int8_t padding[3];    // alignment
};

// -----------------------------------------------------------------------------
// KnobConfig — the fighter's tunable surface, mirrored from FORGE_KNOBS in
// baseline_frozen_2026_04_16.js. All values are int32 for GPU-friendliness.
//
// Layout is stable; add new fields at the end. When this struct changes,
// bump KNOB_CONFIG_VERSION and every kernel/tool must be rebuilt.
// -----------------------------------------------------------------------------
#define KNOB_CONFIG_VERSION 1

struct KnobConfig {
    // --- Layered weights (per-depth bias on 6 piece-domain brains) ---
    //   index: 0=pawn 1=king 2=queen 3=rook 4=minor 5=tempo
    int32_t layer0[6];
    int32_t layer1[6];
    int32_t layer2[6];
    int32_t layer3[6];
    int32_t aggression[4];      // depth-indexed aggression level
    int32_t layerBleed;

    // --- Position-class overlays (6 piece-domains, all additive to layer weights) ---
    int32_t posOpening[6];
    int32_t posMidOpen[6];
    int32_t posMidClosed[6];
    int32_t posEndgame[6];
    int32_t posQueensOn[6];
    int32_t posNoQueens[6];

    // --- Pawn structure ---
    int32_t passedPawnMG[8];    // bonus by rank (rankIdx 0..7, 7 closest to promotion)
    int32_t passedPawnEG[8];
    int32_t doubledPawnMG;
    int32_t doubledPawnEG;
    int32_t isolatedPawnMG;
    int32_t isolatedPawnEG;
    int32_t connectedPawnMG;
    int32_t connectedPawnEG;

    // --- King safety ---
    int32_t shieldBonus;
    int32_t shieldHole;
    int32_t safetyCurve[6];     // attacker-count -> king safety penalty

    // --- Rook features ---
    int32_t rookOpenFile;
    int32_t rookSemiOpen;
    int32_t rook7th;

    // --- Minor piece features ---
    int32_t bishopPair;

    // --- Queen features ---
    int32_t queenExposed;
    int32_t queenTrapped;

    // --- Sniper (opponent-adaptive) overlays — 6 piece-domains each ---
    int32_t sniperAttacker[6];
    int32_t sniperFortress[6];
    int32_t sniperGrinder[6];
    int32_t sniperGambit[6];
    int32_t sniperPositional[6];
    int32_t spyScale;

    // --- Tempo ---
    int32_t tempo;

    // --- Search knobs (not used by eval kernels but carried for search kernels) ---
    int32_t futilityMargin[4];
    int32_t lmpThreshold[5];
    int32_t razorMargin[3];
    int32_t aspWindow;
};

// -----------------------------------------------------------------------------
// Default knob values matching baseline_frozen_2026_04_16.js FORGE_KNOBS.
// Kernels that want the "baseline" fighter initialize d_knobs from this.
// -----------------------------------------------------------------------------
static const KnobConfig DOJO_DEFAULT_KNOBS = {
    /*layer0*/        {6,6,12,8,8,11},
    /*layer1*/        {10,10,8,8,7,8},
    /*layer2*/        {12,14,8,6,8,6},
    /*layer3*/        {14,6,4,12,8,10},
    /*aggression*/    {8,8,6,3},
    /*layerBleed*/    3,
    /*posOpening*/    {-2,4,2,-4,0,2},
    /*posMidOpen*/    {-2,0,0,4,2,0},
    /*posMidClosed*/  {4,0,0,-3,0,0},
    /*posEndgame*/    {4,-4,-4,4,0,2},
    /*posQueensOn*/   {0,4,4,0,0,0},
    /*posNoQueens*/   {0,-4,-8,2,0,0},
    /*passedPawnMG*/  {0,0,10,17,30,55,85,0},
    /*passedPawnEG*/  {0,0,20,35,55,90,140,0},
    /*doubledPawnMG*/ 10, /*doubledPawnEG*/ 20,
    /*isolatedPawnMG*/15, /*isolatedPawnEG*/ 20,
    /*connectedPawnMG*/7, /*connectedPawnEG*/ 12,
    /*shieldBonus*/   15, /*shieldHole*/ 12,
    /*safetyCurve*/   {0,8,21,34,45,50},
    /*rookOpenFile*/  25, /*rookSemiOpen*/ 12, /*rook7th*/ 25,
    /*bishopPair*/    45,
    /*queenExposed*/  35, /*queenTrapped*/ 25,
    /*sniperAttacker*/  {2,6,2,0,0,-2},
    /*sniperFortress*/  {-1,-2,0,2,0,4},
    /*sniperGrinder*/   {4,-2,-2,2,2,2},
    /*sniperGambit*/    {0,4,2,0,0,-2},
    /*sniperPositional*/{0,0,0,2,2,3},
    /*spyScale*/      1000,
    /*tempo*/         15,
    /*futilityMargin*/{0,180,400,700},
    /*lmpThreshold*/  {0,5,9,14,22},
    /*razorMargin*/   {0,250,450},
    /*aspWindow*/     50,
};

#endif // DOJO_TYPES_H
