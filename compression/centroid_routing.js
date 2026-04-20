/**
 * centroid_routing.js — Fighter-level JS implementation of centroid-based routing
 *
 * Extracted from: dojo_skills/centroid_routing.js (unmodified copy)
 *
 * ── Connection to the Python geometry system ────────────────────────────────
 *
 * The Python side (geometry_encoders.py / board_fold.py) computes the warmth
 * centroid offline, once per board position, as part of the FrostMatrix render
 * pipeline. That centroid is the "center of strategic gravity" — a weighted
 * average of all piece positions, where queen=9, rook=5, bishop/knight=3,
 * pawn=1, king=4, with 0.25x neighbor spread baked in.
 *
 * This JS file re-implements exactly the same centroid formula (_crCentroid)
 * inline inside the fighter's evaluate() function, so it runs at search time
 * without needing any precomputed tables. This makes the fighter "centroid-aware"
 * during self-play.
 *
 * The connection is:
 *   Python find_fold_center(warmth)  ←→  JS _crCentroid(board)
 *   Both compute: Σ(piece_weight × spread × square_coords) / Σ(weights)
 *
 * The JS version then maps the centroid to one of 5 strategic zones and applies
 * piece-proximity bonuses — this is the fighter-level action taken on the
 * Python-derived insight that "the game is happening HERE."
 *
 * The 5 zones:
 *   0 KINGSIDE  — centroid.f >= 5 (files f-h)
 *   1 QUEENSIDE — centroid.f <= 2 (files a-c)
 *   2 CENTRAL   — centroid in center files, mid-board rank
 *   3 ADVANCE   — centroid in center files, pushed deep (attacking)
 *   4 DEFEND    — centroid in center files, pulled back (defending)
 *
 * Zone bonus tables (_CR_ZONE_BONUS[zone][piece_type]) give each piece type a
 * bonus scaled by proximity to the centroid. Pieces closer to where the game
 * is decided are rewarded more.
 *
 * ── Usage ────────────────────────────────────────────────────────────────────
 *
 * This is a dojo skill module. The dojo_code_mutator.js injects HELPERS before
 * evaluate() and replaces the evaluate() body with EVALUATE_BODY. The inject()
 * function performs this transformation given a fighter source string.
 *
 * Compatible with any fighter that has:
 *   - pos.board[64] as {color, type} objects or null
 *   - pos.phase as integer (TOTAL_PHASE = 24)
 *   - WHITE=0, BLACK=1, PAWN=1..KING=6 constants
 *   - function evaluate(pos) { ... }
 */

export const name = 'centroid_routing';
export const description = 'Warmth centroid strategy zone routing — routes piece bonuses to where the game is being decided';
export const version = '1.0.0';
export const targetFn = 'evaluate';
export const requiredConstants = ['WHITE', 'BLACK', 'PAWN', 'KNIGHT', 'BISHOP', 'ROOK', 'QUEEN', 'KING', 'TOTAL_PHASE'];

// The helper code block injected before evaluate()
export const HELPERS = `
// ============================================================================
// SKILL: centroid_routing v1.0
// ============================================================================
const _CR_PIECE_W = [0, 1, 3, 3, 5, 9, 4];
const _CR_ZONE_BONUS = [
  [0,  5, 12, 10,  8,  4,  0], // KINGSIDE
  [0,  5, 10, 12,  8,  4,  0], // QUEENSIDE
  [0,  3,  8, 10, 10,  6,  0], // CENTRAL
  [0,  8,  6,  4,  6,  2,  0], // ADVANCE
  [0,  2,  4,  6,  8,  2,  0], // DEFEND
];

function _crCentroid(board) {
  let wr = 0, wf = 0, tw = 0;
  for (let sq = 0; sq < 64; sq++) {
    const p = board[sq];
    if (!p) continue;
    const w = _CR_PIECE_W[p.type];
    if (!w) continue;
    const r = sq >> 3, f = sq & 7;
    let spread = 0;
    for (let dr = -1; dr <= 1; dr++) for (let df = -1; df <= 1; df++) {
      if (dr === 0 && df === 0) continue;
      const nr = r + dr, nf = f + df;
      if (nr < 0 || nr > 7 || nf < 0 || nf > 7) continue;
      const nb = board[nr * 8 + nf];
      if (nb) spread += _CR_PIECE_W[nb.type] * 0.25;
    }
    const eff = w + spread;
    wr += r * eff; wf += f * eff; tw += eff;
  }
  return tw === 0 ? { r: 3.5, f: 3.5 } : { r: wr / tw, f: wf / tw };
}

function _crBonus(board, centroid, side, phase) {
  const mgFrac = Math.min(phase, TOTAL_PHASE) / TOTAL_PHASE;
  if (mgFrac < 0.1) return 0;
  let zone;
  if (centroid.f >= 5) zone = 0;
  else if (centroid.f <= 2) zone = 1;
  else {
    const adv = side === WHITE ? centroid.r : 7 - centroid.r;
    zone = adv < 2.5 ? 3 : adv > 5.5 ? 4 : 2;
  }
  const bt = _CR_ZONE_BONUS[zone];
  let bonus = 0;
  for (let sq = 0; sq < 64; sq++) {
    const p = board[sq];
    if (!p || p.type === KING) continue;
    const r = sq >> 3, f = sq & 7;
    const dist = Math.max(Math.abs(r - centroid.r), Math.abs(f - centroid.f));
    const prox = 1 / (dist + 1);
    const b = bt[p.type] * prox;
    if (p.color === side) bonus += b; else bonus -= b;
  }
  return bonus * mgFrac;
}
`;

// The new evaluate() body — replaces existing evaluate
export const EVALUATE_BODY = `function evaluate(pos) {
  const ph = Math.min(pos.phase, TOTAL_PHASE);
  const egPh = TOTAL_PHASE - ph;
  const base = (pos.mgScore * ph + pos.egScore * egPh) / TOTAL_PHASE;
  const centroid = _crCentroid(pos.board);
  const spatial = _crBonus(pos.board, centroid, pos.side, pos.phase);
  return pos.side === WHITE ? base + spatial : -base + spatial;
}`;

/**
 * inject(src) — takes fighter source string, returns modified source string.
 * Finds the existing evaluate() function, prepends helpers just before it,
 * and replaces the body.
 */
export function inject(src) {
  // Find evaluate function bounds
  const fnRe = /^function\s+evaluate\s*\(/m;
  const m = fnRe.exec(src);
  if (!m) throw new Error('centroid_routing: could not find function evaluate()');

  const start = m.index;
  const braceStart = src.indexOf('{', start);
  let depth = 0, i = braceStart;
  for (; i < src.length; i++) {
    const c = src[i];
    if (c === '{') depth++;
    else if (c === '}') { depth--; if (depth === 0) { i++; break; } }
  }

  // Guard: don't double-inject
  if (src.includes('_crCentroid')) throw new Error('centroid_routing: already injected');

  return src.slice(0, start) + HELPERS + '\n' + EVALUATE_BODY + '\n' + src.slice(i);
}
