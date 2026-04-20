# FrostMatrix Geometry Encoders — Reference Table

All 17 encoders (G1-G17) produce the 51 extended channels of the 55-dim feature vector.
The 4 base channels come from the active fold system (centroid_r, centroid_f, compression, outlier_frac).

Dims are counts of floats output per board position. All values are normalized to roughly [0,1] or [-1,1].

| # | Name | Dims | What it measures | Chess concept captured |
|---|------|------|-----------------|----------------------|
| G1 | tension_gradient | 3 | Sum of piece force vectors along attack rays; dominant eigenvector direction + magnitude | Main axis of conflict — which direction forces are pushing across the board |
| G2 | pawn_corridor | 3 | Open/semi-open file counts, structural asymmetry between flanks, overall closure index | Whether the position is open, closed, or semi-open; which wing has space |
| G3 | king_shell | 3 | Chebyshev distance between kings, overlap of king safety zones, shield piece asymmetry | King safety, proximity of kings to each other, defensive shell integrity |
| G4 | mobility_field | 3 | Mobility-weighted piece centroid (r, f) and white-vs-black mobility imbalance | Where pieces are effectively active (not just placed); activity advantage |
| G5 | color_complex | 2 | Strategic weight on light squares vs dark squares, per side differential | Bad bishops, color weakness, bishop-of-opposite-color structures |
| G6 | outpost_gravity | 3 | Centroid and density of squares safe from enemy pawns and defended by friendly pawns | Structural outpost anchors; gravitational pull of advanced defended squares |
| G7 | phase_gradient | 3 | Per-quadrant material phase weight; gradient vector from endgame toward middlegame zone | Spatial distribution of middlegame vs endgame material; lopsided liquidation |
| G8 | diagonal_axis | 3 | Heaviest NE (a1-h8) and NW (a8-h1) diagonal loading, dominance of one axis over the other | Diagonal vs orthogonal character of the position; bishop/queen diagonal pressure |
| G9 | territorial_frontier | 3 | Centroid and fragmentation of the control-boundary between white and black territory | Where the strategic front line runs; fragmented = multiple islands of contact |
| G10 | resonance_cluster | 3 | Same-side piece pairs aligned on rank/file/diagonal (batteries, lifts); centroid + strength | Coordinated piece pairs — where the attack machinery is assembled |
| G11 | vertical_fold | 3 | File symmetry ratio (a-d vs h-e mirror); asymmetry score; centroid of broken-symmetry files | File imbalance — which flank has asymmetric piece placement |
| G12 | horizontal_fold | 3 | Rank symmetry ratio (white/black halves mirror); invasion piece count; boundary pressure depth | Territorial invasion — pieces crossing the center; how deep and how many |
| G13 | relationship_fold | 4 | Contested 4x4 quadrant zones (both colors present); conflict density; zone centroid | Spatial distribution of direct piece contact across the board's quadrants |
| G14 | pawn_chain | 3 | Fraction of pawns diagonally defending another pawn; max chain length; pawn centroid rank | Pawn structure topology — hedgehog, Maroczy bind, IQP, passed pawn chains |
| G15 | pin_xray | 3 | Sliders (R/B/Q) aligned with enemy king on rank/file/diagonal; combined axis + density | Latent pin and x-ray pressure without full move generation |
| G16 | open_file | 3 | Centroid of fully open files; white-vs-black semi-open file imbalance; pawn span width | Rook/queen lane geometry — which files are available and where they cluster |
| G17 | knight_outpost | 3 | Occupied outpost count per side (knight on square unreachable by enemy pawns); centroid rank | Knight outpost occupation — classic static advantage in pawn structures |

## Feature vector layout (55 dims total)

```
[0]    centroid_r       — rank of warmth centroid [0,7], normalized /7
[1]    centroid_f       — file of warmth centroid [0,7], normalized /7
[2]    compression      — fraction of 64 squares in quiet (compressed) zone
[3]    outlier_frac     — fraction of placed pieces in strategic zone
[4:7]  G1  tension_gradient
[7:10] G2  pawn_corridor
[10:13] G3 king_shell
[13:16] G4 mobility_field
[16:18] G5 color_complex
[18:21] G6 outpost_gravity
[21:24] G7 phase_gradient
[24:27] G8 diagonal_axis
[27:30] G9 territorial_frontier
[30:33] G10 resonance_cluster
[33:36] G11 vertical_fold
[36:39] G12 horizontal_fold
[39:43] G13 relationship_fold
[43:46] G14 pawn_chain
[46:49] G15 pin_xray
[49:52] G16 open_file
[52:55] G17 knight_outpost
```

## Per-encoder dim breakdown

| Encoder | Dim 0 | Dim 1 | Dim 2 | Dim 3 |
|---------|-------|-------|-------|-------|
| G1 tension_gradient | axis_r (signed, ~[-1,1]) | axis_f (signed, ~[-1,1]) | strength [0,∞] |  |
| G2 pawn_corridor | open_centroid_f [0,1] | semi_imbalance [-1,1] | closure [0,1] |  |
| G3 king_shell | king_dist [0,1] | overlap [0,1] | shield_asym [-1,1] |  |
| G4 mobility_field | mob_centroid_r [0,1] | mob_centroid_f [0,1] | imbalance [-1,1] |  |
| G5 color_complex | light_sq_adv [-1,1] | dark_sq_adv [-1,1] | — |  |
| G6 outpost_gravity | centroid_r [0,1] | centroid_f [0,1] | density [0,1] |  |
| G7 phase_gradient | gradient_r [-1,1] | gradient_f [-1,1] | phase_range [0,1] |  |
| G8 diagonal_axis | ne_load [0,1] | nw_load [0,1] | dominance [0,1] |  |
| G9 territorial_frontier | frontier_r [0,1] | frontier_f [0,1] | fragmentation [0,1] |  |
| G10 resonance_cluster | centroid_r [0,1] | centroid_f [0,1] | strength [0,1] |  |
| G11 vertical_fold | compression [0,1] | asym_score [0,1] | outlier_f [0,1] |  |
| G12 horizontal_fold | compression [0,1] | invasion_density [0,1] | rank_pressure [0,1] |  |
| G13 relationship_fold | conflict_density [0,1] | max_conflict [0,1] | zone_centroid_r [0,1] | zone_centroid_f [0,1] |
| G14 pawn_chain | chain_density [0,1] | max_len_norm [0,1] | centroid_r [0,1] |  |
| G15 pin_xray | axis_r [0,1] | axis_f [0,1] | pin_density [0,1] |  |
| G16 open_file | open_centroid_f [0,1] | semi_imbalance [-1,1] | file_span [0,1] |  |
| G17 knight_outpost | w_outpost [0,1] | b_outpost [0,1] | centroid_r [0,1] |  |
