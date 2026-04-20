# Board Compression / Geometry System

Isolated reference copy of the FrostMatrix board compression and geometry encoding system.
Source files are unchanged — this folder is a clean, labeled extraction for study and reuse.

---

## What board compression is and why

Standard chess evaluation treats all 64 squares equally. The FrostMatrix approach instead
asks: *where is this game actually happening?*

The answer is the **warmth centroid** — the strategic center of gravity of the position.
Each piece radiates "warmth" proportional to its value (queen=9, rook=5, bishop/knight=3,
pawn=1, king=4), and that warmth spreads 0.25x to neighboring squares. The weighted centroid
of this warmth map is the fold center.

With the fold center known, the board splits into two zones:

- **Strategic zone** (radius < 2.5 squares from centroid): squares that carry unique positional
  information. Each gets its own absolute encoding.
- **Quiet zone** (everything else): squares that are not where the game is being decided.
  Each quiet square is reflected across the fold center and shares its encoding with its mirror.
  Two squares → one canonical slot. Cost halved.

This is **variable-resolution positional encoding**: full detail where it matters, compression
everywhere else. Recomputed from scratch for every board position, O(64), no parameters.

The fold center is also the **raytrace origin**: the warmth centroid IS the point from which
all 17 geometry encoders measure the board's structure. The sequence of fold centers across
plies of a game is the "raytraced shape" of that game through space.

---

## The three fold variants

Three different axes of folding capture different aspects of positional asymmetry:

**Variant A — fold_vertical (file symmetry)**
Folds the board left-right along the d/e file boundary. Files a-d mirror h-e within each rank.
A symmetric pair (same piece type on both sides) compresses cleanly to one slot.
Broken symmetry (different pieces, or one side empty) produces an outlier.
*What it captures:* which flank is "unbalanced" — the outlier centroid tells you which file
has the asymmetric piece. Useful for detecting kingside vs queenside imbalances.

**Variant B — fold_horizontal (rank invasion)**
Folds the board top-bottom along the 4th/5th rank boundary. White pieces in ranks 1-4 are
expected to mirror black pieces in ranks 5-8. A piece that has crossed the center line
(advanced pawn, invading rook, deep knight) becomes an outlier.
*What it captures:* territorial invasion — how many pieces have crossed into enemy territory
and how deeply. Positively correlated with decisive game outcomes.

**Variant C — fold_relationship (4x4 zone co-occupancy)**
Divides the board into 16 symmetric 4x4 quadrant zones. For each zone, records which white
and black pieces co-occupy it. A zone with both colors present is a contested zone (outlier).
*What it captures:* the geometric distribution of inter-side contact — where the pieces of
both sides are physically overlapping in zone space.

---

## The 17 geometry encoders (G1-G17)

All encoders take board[64] int → List[float]. They are independent and orthogonal —
each measures a fundamentally different structural property of the position.

The warmth centroid (from the active fold system) is the conceptual origin for all of them:
each encoder is asking "what does the board look like as seen from the fold center?"

| # | Name | Dims | Chess concept |
|---|------|------|--------------|
| G1 | tension_gradient | 3 | Main axis of conflict — sum of piece attack force vectors |
| G2 | pawn_corridor | 3 | Open/closed file topology, space asymmetry between flanks |
| G3 | king_shell | 3 | King distance, safety zone overlap, shield asymmetry |
| G4 | mobility_field | 3 | Where pieces are effectively active; mobility imbalance |
| G5 | color_complex | 2 | Light/dark square strength differential (bad bishops, etc.) |
| G6 | outpost_gravity | 3 | Structural anchor squares safe from enemy pawns |
| G7 | phase_gradient | 3 | Spatial distribution of middlegame vs endgame material |
| G8 | diagonal_axis | 3 | Diagonal vs orthogonal character of the position |
| G9 | territorial_frontier | 3 | Where the control boundary runs; frontier fragmentation |
| G10 | resonance_cluster | 3 | Coordinated piece pairs (batteries, lifts) — attack machinery |
| G11 | vertical_fold | 3 | File symmetry signal (from fold_vertical) |
| G12 | horizontal_fold | 3 | Invasion signal (from fold_horizontal) |
| G13 | relationship_fold | 4 | Zone co-occupancy contact map (from fold_relationship) |
| G14 | pawn_chain | 3 | Pawn structure topology (chains, hedgehog, IQP) |
| G15 | pin_xray | 3 | Slider alignment with enemy king — latent pin/x-ray pressure |
| G16 | open_file | 3 | Rook/queen lane geometry — which files are available |
| G17 | knight_outpost | 3 | Knight occupation of structural outpost squares |

See `ENCODERS.md` for the full per-dimension breakdown.

---

## The full feature vector: 55 dims per ply

```
4 base dims (active_fold_encode):
  [0] centroid_r      — rank of warmth centroid, normalized to [0,1]
  [1] centroid_f      — file of warmth centroid, normalized to [0,1]
  [2] compression     — fraction of squares in quiet zone
  [3] outlier_frac    — fraction of placed pieces in strategic zone

51 extended dims (G1-G17, concatenated):
  G1(3) G2(3) G3(3) G4(3) G5(2) G6(3) G7(3) G8(3) G9(3) G10(3)
  G11(3) G12(3) G13(4) G14(3) G15(3) G16(3) G17(3)
  = 3+3+3+3+2+3+3+3+3+3+3+3+4+3+3+3+3 = 51

Total: 4 + 51 = 55 dimensions per board position
```

The `composite_encode(board)` function in `geometry_encoders.py` returns this full 55-dim vector.

---

## How this feeds into FrostMatrix (clustering space)

The render pipeline (`tools/render_pipeline.py`) builds a **game trajectory** as a sequence
of 55-dim vectors — one per ply up to max_depth (default 10). This gives a (11 × 55) = 605-dim
fingerprint per game.

The **PrefixAggregator** (L4) accumulates these trajectories grouped by opening prefix. Every
game that played 1.e4 contributes its full 605-dim trajectory to the "e2e4" bucket. The result
is a per-prefix mean trajectory matrix and variance matrix.

The **FamilyClusterer** (L5) runs k-means (default 33 clusters) on the mean trajectory
feature space. The resulting cluster centroids are the 33 FrostMatrix "families" — geometric
archetypes of how games with a given opening tend to evolve structurally across the first 10 plies.

The fold_signature bitmask (which squares are in the strategic zone at each ply) is accumulated
separately as `fold_mean` — a floating-point average across all games in the prefix. This is the
"mean strategic zone shape" for that opening, used as a secondary family signal.

The 33 family IDs are then projected back onto the opening tree (sankey nodes) and used to
build transition surfaces P[depth][family_i][family_j] — the probability that a game currently
in family i will be in family j one ply later.

---

## How this feeds into the fighter (centroid routing)

The JS file `centroid_routing.js` (also `dojo_skills/centroid_routing.js`) implements the same
warmth centroid formula in the fighter's evaluate() function.

At search time, for every board position the fighter evaluates, it:
1. Computes the warmth centroid in O(64) — same formula as Python's `find_fold_center`
2. Maps the centroid to one of 5 strategic zones (kingside / queenside / central / advance / defend)
3. Applies piece-proximity bonuses: pieces physically closer to the centroid get larger bonuses
4. Scales the whole thing by game phase (fades to zero in pure endgame)

This makes the fighter's static evaluation "centroid-aware": it rewards pieces for being near
where the game is actually happening, as diagnosed live rather than from a lookup table.

---

## The raytrace metaphor

The fold_signature for a single position is a 64-bit integer where each bit marks whether that
square is in the strategic zone. Across plies 0..10 of a game, the sequence of fold_signatures
is the **raytraced shape** of that game — a time-series of "which part of the board mattered"
at each step.

Two games that look superficially different in piece placement may trace identical or very similar
strategic shapes across plies. That structural similarity is what the FrostMatrix family clustering
is finding: games that share a geometric trajectory, not just an opening name.

The centroid path (centroid_r, centroid_f at each ply) is the game's strategic center of gravity
moving through the board over time. A game where the centroid jumps from the queenside to the
kingside in two plies is structurally different from one where it stays fixed — even if the
material count is the same.

---

## Files in this folder

| File | Contents |
|------|---------|
| `board_fold.py` | Self-contained Python: warmth, centroid, active_fold, all 3 fold variants, demo |
| `geometry_encoders.py` | Self-contained Python: all 17 encoders, composite_encode(), demo |
| `centroid_routing.js` | Fighter-level JS centroid routing skill (dojo injection) |
| `ENCODERS.md` | Reference table: one row per encoder with dims and chess concept |
| `README.md` | This file |
