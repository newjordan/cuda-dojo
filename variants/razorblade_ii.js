const { readFileSync } = require('fs');

// ============================================================================
// CONSTANTS
// ============================================================================

const INFINITY = 999999;
const MATE_SCORE = 100000;
const MAX_DEPTH = 64;

// Piece types
const PAWN = 1, KNIGHT = 2, BISHOP = 3, ROOK = 4, QUEEN = 5, KING = 6;
const PIECE_CHARS = { p: PAWN, n: KNIGHT, b: BISHOP, r: ROOK, q: QUEEN, k: KING };
const PROMO_CHARS = { q: QUEEN, r: ROOK, b: BISHOP, n: KNIGHT };
const PROMO_UCI = { [QUEEN]: 'q', [ROOK]: 'r', [BISHOP]: 'b', [KNIGHT]: 'n' };

const WHITE = 0, BLACK = 1;
const FILES = 'abcdefgh';

// Move flags
const FLAG_NONE = 0, FLAG_CAPTURE = 1, FLAG_EP = 2, FLAG_CASTLE_K = 3;
const FLAG_CASTLE_Q = 4, FLAG_DOUBLE_PAWN = 5, FLAG_PROMO = 6, FLAG_PROMO_CAP = 7;

// ============================================================================
// PIECE-SQUARE TABLES (PeSTO-style, from White's perspective, index 0 = a8)
// Middlegame and endgame tables for tapered evaluation
// ============================================================================

const MG_PAWN_VAL = 82, EG_PAWN_VAL = 94;
const MG_KNIGHT_VAL = 337, EG_KNIGHT_VAL = 281;
const MG_BISHOP_VAL = 365, EG_BISHOP_VAL = 297;
const MG_ROOK_VAL = 477, EG_ROOK_VAL = 512;
const MG_QUEEN_VAL = 1025, EG_QUEEN_VAL = 936;
const MG_KING_VAL = 0, EG_KING_VAL = 0;

const MG_PIECE_VAL = [0, MG_PAWN_VAL, MG_KNIGHT_VAL, MG_BISHOP_VAL, MG_ROOK_VAL, MG_QUEEN_VAL, MG_KING_VAL];
const EG_PIECE_VAL = [0, EG_PAWN_VAL, EG_KNIGHT_VAL, EG_BISHOP_VAL, EG_ROOK_VAL, EG_QUEEN_VAL, EG_KING_VAL];

// Phase weights for tapered eval
const PHASE_WEIGHTS = [0, 0, 1, 1, 2, 4, 0];
const TOTAL_PHASE = 24;

// PST arrays [pieceType][squareIndex] - from white's perspective (a8=0, h1=63)
const MG_PST = [
  null, // 0 = no piece
  // PAWN MG
  [0,0,0,0,0,0,0,0,98,134,61,95,68,126,34,-11,
   -6,7,26,31,65,56,25,-20,-14,13,6,21,23,12,17,-23,
   -27,-2,-5,12,17,6,10,-25,-26,-4,-4,-10,3,3,33,-12,
   -35,-1,-20,-23,-15,24,38,-22,0,0,0,0,0,0,0,0],
  // KNIGHT MG
  [-167,-89,-34,-49,61,-97,50,-73,-73,-41,72,36,23,62,7,-17,
   -47,60,37,65,84,129,73,44,-9,17,19,53,37,69,18,22,
   -13,4,16,13,28,19,21,-8,-23,-9,12,10,19,17,25,-16,
   -29,-53,-12,-3,-1,18,-14,-19,-105,-21,-58,-33,-17,-28,-19,-23],
  // BISHOP MG
  [-29,4,-82,-37,-25,-42,7,-8,-26,16,-18,-13,30,59,18,-47,
   -16,37,43,40,35,50,37,-2,-4,5,19,50,37,37,7,-2,
   -6,13,13,26,34,12,10,4,0,15,15,15,14,27,18,10,
   4,15,16,0,7,21,33,1,-33,-3,-14,-21,-13,-12,-39,-21],
  // ROOK MG
  [32,42,32,51,63,9,31,43,27,32,58,62,80,67,26,44,
   -5,19,26,36,17,45,61,16,-24,-11,7,26,24,35,-8,-20,
   -36,-26,-12,-1,9,-7,6,-23,-45,-25,-16,-17,3,0,-5,-33,
   -44,-16,-20,-9,-1,11,-6,-71,-19,-13,1,17,16,7,-37,-26],
  // QUEEN MG
  [-28,0,29,12,59,44,43,45,-24,-39,-5,1,-16,57,28,54,
   -13,-17,7,8,29,56,47,57,-27,-27,-16,-16,-1,17,-2,1,
   -9,-26,-9,-10,-2,-4,3,-3,-14,-2,-11,-2,-5,2,14,5,
   -35,-8,11,2,8,15,-3,1,-1,-18,-9,10,-15,-25,-31,-50],
  // KING MG
  [-65,23,16,-15,-56,-34,2,13,29,-1,-20,-7,-8,-4,-38,-29,
   -9,24,2,-16,-20,6,22,-22,-17,-20,-12,-27,-30,-25,-14,-36,
   -49,-1,-27,-39,-46,-44,-33,-51,-14,-14,-22,-46,-44,-30,-15,-27,
   1,7,-8,-64,-43,-16,9,8,-15,36,12,-54,8,-28,24,14],
];

const EG_PST = [
  null,
  // PAWN EG
  [0,0,0,0,0,0,0,0,178,173,158,134,147,132,165,187,
   94,100,85,67,56,53,82,84,32,24,13,5,-2,4,17,17,
   13,9,-3,-7,-7,-8,3,-1,4,7,-6,1,0,-5,-1,-8,
   13,8,8,10,13,0,2,-7,0,0,0,0,0,0,0,0],
  // KNIGHT EG
  [-58,-38,-13,-28,-31,-27,-63,-99,-25,-8,-25,-2,-9,-25,-24,-52,
   -24,-20,10,9,-1,-9,-19,-41,-17,3,22,22,22,11,8,-18,
   -18,-6,16,25,16,17,4,-18,-23,-3,-1,15,10,-3,-20,-22,
   -42,-20,-10,-5,-2,-20,-23,-44,-29,-51,-23,-15,-22,-18,-50,-64],
  // BISHOP EG
  [-14,-21,-11,-8,-7,-9,-17,-24,-8,-4,7,-12,-3,-13,-4,-14,
   2,-8,0,-1,-2,6,0,4,-3,9,12,9,14,10,3,2,
   -6,3,13,19,7,10,-3,-9,-12,-3,8,10,13,3,-7,-15,
   -14,-18,-7,-1,4,-9,-15,-27,-23,-9,-23,-5,-9,-16,-5,-17],
  // ROOK EG
  [13,10,18,15,12,12,8,5,11,13,13,11,-3,7,7,8,
   7,7,7,5,4,-3,-5,3,4,3,13,1,2,1,-1,2,
   3,5,8,4,-5,-6,-8,-11,-4,0,-5,-1,-7,-12,-8,-16,
   -6,-6,0,2,-9,-9,-11,-3,-9,2,3,-1,-5,-13,4,-20],
  // QUEEN EG
  [-9,22,22,27,27,19,10,20,-17,20,32,41,58,25,30,0,
   -20,6,9,49,47,35,19,9,3,22,24,45,57,40,57,36,
   -18,28,19,47,31,34,39,23,-16,-27,15,6,9,17,10,5,
   -22,-23,-30,-16,-16,-23,-36,-32,-33,-28,-22,-43,-5,-32,-20,-41],
  // KING EG
  [-74,-35,-18,-18,-11,15,4,-17,-12,17,14,17,17,38,23,11,
   10,17,23,15,20,45,44,13,-8,22,24,27,26,33,26,3,
   -18,-4,21,24,27,23,9,-11,-19,-3,11,21,23,16,7,-9,
   -27,-11,4,13,14,4,-5,-17,-53,-34,-21,-11,-28,-14,-24,-43],
];

// Mirror table: flips index for black pieces (a8 -> a1, etc.)
const MIRROR = [];
for (let r = 0; r < 8; r++) {
  for (let f = 0; f < 8; f++) {
    MIRROR[r * 8 + f] = (7 - r) * 8 + f;
  }
}

// ============================================================================
// ZOBRIST HASHING
// ============================================================================

// Deterministic pseudo-random number generator for reproducible Zobrist keys
let _seed = 1070372;
function xorshift32() {
  _seed ^= _seed << 13;
  _seed ^= _seed >> 17;
  _seed ^= _seed << 5;
  return _seed >>> 0;
}

// Generate Zobrist keys: [color][pieceType][square]
const ZOBRIST_PIECE = [];
for (let c = 0; c < 2; c++) {
  ZOBRIST_PIECE[c] = [];
  for (let p = 1; p <= 6; p++) {
    ZOBRIST_PIECE[c][p] = [];
    for (let sq = 0; sq < 64; sq++) {
      ZOBRIST_PIECE[c][p][sq] = xorshift32();
    }
  }
}
const ZOBRIST_SIDE = xorshift32();
const ZOBRIST_CASTLE = [];
for (let i = 0; i < 16; i++) ZOBRIST_CASTLE[i] = xorshift32();
const ZOBRIST_EP = [];
for (let i = 0; i < 8; i++) ZOBRIST_EP[i] = xorshift32();

// ============================================================================
// TRANSPOSITION TABLE
// ============================================================================

const TT_EXACT = 0, TT_LOWER = 1, TT_UPPER = 2;
const TT_SIZE = 1 << 20; // ~1M entries
const TT_MASK = TT_SIZE - 1;

// Each entry: { key, depth, score, flag, move }
const TT = new Array(TT_SIZE);

function ttProbe(key, depth, alpha, beta, ply) {
  const entry = TT[key & TT_MASK];
  if (!entry || entry.key !== key) return null;
  if (entry.depth >= depth) {
    let score = entry.score;
    // Adjust mate scores for ply
    if (score > MATE_SCORE - 100) score -= ply;
    else if (score < -MATE_SCORE + 100) score += ply;
    if (entry.flag === TT_EXACT) return { score, move: entry.move };
    if (entry.flag === TT_LOWER && score >= beta) return { score, move: entry.move };
    if (entry.flag === TT_UPPER && score <= alpha) return { score, move: entry.move };
  }
  return { score: null, move: entry.move }; // return move for ordering even if score unusable
}

function ttStore(key, depth, score, flag, move, ply) {
  const idx = key & TT_MASK;
  const existing = TT[idx];
  // Always replace if deeper or same position
  if (!existing || existing.key === key || existing.depth <= depth) {
    let adjScore = score;
    if (adjScore > MATE_SCORE - 100) adjScore += ply;
    else if (adjScore < -MATE_SCORE + 100) adjScore -= ply;
    TT[idx] = { key, depth, score: adjScore, flag, move };
  }
}

// ============================================================================
// POSITION
// ============================================================================

function parseFen(fen) {
  const [placement, side, castling, ep, halfmove, fullmove] = fen.trim().split(/\s+/);

  // board[sq] = { color, type } or null
  const board = new Array(64).fill(null);
  let sq = 0;
  for (const ch of placement) {
    if (ch === '/') continue;
    if (ch >= '1' && ch <= '8') { sq += Number(ch); continue; }
    const color = ch === ch.toUpperCase() ? WHITE : BLACK;
    const type = PIECE_CHARS[ch.toLowerCase()];
    board[sq] = { color, type };
    sq++;
  }

  // Castling rights as bitmask: bit0=K, bit1=Q, bit2=k, bit3=q
  let castleRights = 0;
  if (castling !== '-') {
    if (castling.includes('K')) castleRights |= 1;
    if (castling.includes('Q')) castleRights |= 2;
    if (castling.includes('k')) castleRights |= 4;
    if (castling.includes('q')) castleRights |= 8;
  }

  const epSq = ep === '-' ? -1 : (FILES.indexOf(ep[0]) + (8 - Number(ep[1])) * 8);

  // Compute Zobrist hash
  let hash = 0;
  for (let i = 0; i < 64; i++) {
    if (board[i]) hash ^= ZOBRIST_PIECE[board[i].color][board[i].type][i];
  }
  if (side === 'b') hash ^= ZOBRIST_SIDE;
  hash ^= ZOBRIST_CASTLE[castleRights];
  if (epSq >= 0) hash ^= ZOBRIST_EP[epSq % 8];

  // Compute material phase and incremental PST scores
  let mgScore = 0, egScore = 0, phase = 0;
  for (let i = 0; i < 64; i++) {
    const p = board[i];
    if (!p) continue;
    const pstIdx = p.color === WHITE ? i : MIRROR[i];
    const mg = MG_PIECE_VAL[p.type] + MG_PST[p.type][pstIdx];
    const eg = EG_PIECE_VAL[p.type] + EG_PST[p.type][pstIdx];
    if (p.color === WHITE) { mgScore += mg; egScore += eg; }
    else { mgScore -= mg; egScore -= eg; }
    phase += PHASE_WEIGHTS[p.type];
  }

  return {
    board,
    side: side === 'w' ? WHITE : BLACK,
    castleRights,
    epSq,
    halfmove: Number(halfmove || 0),
    fullmove: Number(fullmove || 1),
    hash,
    mgScore,
    egScore,
    phase,
  };
}

// ============================================================================
// MOVE GENERATION
// ============================================================================

const KNIGHT_OFFSETS = [
  [-2, -1], [-2, 1], [-1, -2], [-1, 2],
  [1, -2], [1, 2], [2, -1], [2, 1]
];
const BISHOP_DIRS = [[-1, -1], [-1, 1], [1, -1], [1, 1]];
const ROOK_DIRS = [[-1, 0], [1, 0], [0, -1], [0, 1]];
const QUEEN_DIRS = [...BISHOP_DIRS, ...ROOK_DIRS];
const KING_DIRS = QUEEN_DIRS;

function inBounds(r, c) { return r >= 0 && r < 8 && c >= 0 && c < 8; }

function generateMoves(pos, capturesOnly) {
  const moves = [];
  const { board, side, castleRights, epSq } = pos;
  const enemy = 1 - side;

  for (let sq = 0; sq < 64; sq++) {
    const piece = board[sq];
    if (!piece || piece.color !== side) continue;
    const r = sq >> 3, c = sq & 7;
    const pt = piece.type;

    if (pt === PAWN) {
      const dir = side === WHITE ? -1 : 1;
      const startRank = side === WHITE ? 6 : 1;
      const promoRank = side === WHITE ? 0 : 7;

      // Captures
      for (const dc of [-1, 1]) {
        const nr = r + dir, nc = c + dc;
        if (!inBounds(nr, nc)) continue;
        const to = nr * 8 + nc;
        const target = board[to];
        if (target && target.color === enemy) {
          if (nr === promoRank) {
            for (const promo of [QUEEN, ROOK, BISHOP, KNIGHT]) {
              moves.push({ from: sq, to, flag: FLAG_PROMO_CAP, captured: target.type, promo });
            }
          } else {
            moves.push({ from: sq, to, flag: FLAG_CAPTURE, captured: target.type });
          }
        }
        // En passant
        if (to === epSq) {
          moves.push({ from: sq, to, flag: FLAG_EP, captured: PAWN });
        }
      }

      if (capturesOnly) continue;

      // Forward moves
      const oneR = r + dir;
      const oneSq = oneR * 8 + c;
      if (inBounds(oneR, c) && !board[oneSq]) {
        if (oneR === promoRank) {
          for (const promo of [QUEEN, ROOK, BISHOP, KNIGHT]) {
            moves.push({ from: sq, to: oneSq, flag: FLAG_PROMO, promo });
          }
        } else {
          moves.push({ from: sq, to: oneSq, flag: FLAG_NONE });
          // Double push
          if (r === startRank) {
            const twoSq = (r + dir * 2) * 8 + c;
            if (!board[twoSq]) {
              moves.push({ from: sq, to: twoSq, flag: FLAG_DOUBLE_PAWN });
            }
          }
        }
      }
    } else if (pt === KNIGHT) {
      for (const [dr, dc] of KNIGHT_OFFSETS) {
        const nr = r + dr, nc = c + dc;
        if (!inBounds(nr, nc)) continue;
        const to = nr * 8 + nc;
        const target = board[to];
        if (target) {
          if (target.color === enemy) {
            if (!capturesOnly || true) moves.push({ from: sq, to, flag: FLAG_CAPTURE, captured: target.type });
          }
        } else if (!capturesOnly) {
          moves.push({ from: sq, to, flag: FLAG_NONE });
        }
      }
    } else if (pt === KING) {
      for (const [dr, dc] of KING_DIRS) {
        const nr = r + dr, nc = c + dc;
        if (!inBounds(nr, nc)) continue;
        const to = nr * 8 + nc;
        const target = board[to];
        if (target) {
          if (target.color === enemy) moves.push({ from: sq, to, flag: FLAG_CAPTURE, captured: target.type });
        } else if (!capturesOnly) {
          moves.push({ from: sq, to, flag: FLAG_NONE });
        }
      }
      // Castling
      if (!capturesOnly) {
        if (side === WHITE) {
          if ((castleRights & 1) && !board[61] && !board[62] &&
              board[63] && board[63].type === ROOK && board[63].color === WHITE &&
              !isSquareAttacked(pos, 60, enemy) && !isSquareAttacked(pos, 61, enemy) && !isSquareAttacked(pos, 62, enemy)) {
            moves.push({ from: 60, to: 62, flag: FLAG_CASTLE_K });
          }
          if ((castleRights & 2) && !board[59] && !board[58] && !board[57] &&
              board[56] && board[56].type === ROOK && board[56].color === WHITE &&
              !isSquareAttacked(pos, 60, enemy) && !isSquareAttacked(pos, 59, enemy) && !isSquareAttacked(pos, 58, enemy)) {
            moves.push({ from: 60, to: 58, flag: FLAG_CASTLE_Q });
          }
        } else {
          if ((castleRights & 4) && !board[5] && !board[6] &&
              board[7] && board[7].type === ROOK && board[7].color === BLACK &&
              !isSquareAttacked(pos, 4, enemy) && !isSquareAttacked(pos, 5, enemy) && !isSquareAttacked(pos, 6, enemy)) {
            moves.push({ from: 4, to: 6, flag: FLAG_CASTLE_K });
          }
          if ((castleRights & 8) && !board[3] && !board[2] && !board[1] &&
              board[0] && board[0].type === ROOK && board[0].color === BLACK &&
              !isSquareAttacked(pos, 4, enemy) && !isSquareAttacked(pos, 3, enemy) && !isSquareAttacked(pos, 2, enemy)) {
            moves.push({ from: 4, to: 2, flag: FLAG_CASTLE_Q });
          }
        }
      }
    } else {
      // Sliding pieces: BISHOP, ROOK, QUEEN
      const dirs = pt === BISHOP ? BISHOP_DIRS : pt === ROOK ? ROOK_DIRS : QUEEN_DIRS;
      for (const [dr, dc] of dirs) {
        let nr = r + dr, nc = c + dc;
        while (inBounds(nr, nc)) {
          const to = nr * 8 + nc;
          const target = board[to];
          if (target) {
            if (target.color === enemy) moves.push({ from: sq, to, flag: FLAG_CAPTURE, captured: target.type });
            break;
          }
          if (!capturesOnly) moves.push({ from: sq, to, flag: FLAG_NONE });
          nr += dr; nc += dc;
        }
      }
    }
  }
  return moves;
}

// ============================================================================
// ATTACK DETECTION
// ============================================================================

function isSquareAttacked(pos, sq, by) {
  const { board } = pos;
  const tr = sq >> 3, tc = sq & 7;

  // Pawn attacks
  const pawnDir = by === WHITE ? 1 : -1;
  for (const dc of [-1, 1]) {
    const pr = tr + pawnDir, pc = tc + dc;
    if (inBounds(pr, pc)) {
      const p = board[pr * 8 + pc];
      if (p && p.color === by && p.type === PAWN) return true;
    }
  }

  // Knight attacks
  for (const [dr, dc] of KNIGHT_OFFSETS) {
    const nr = tr + dr, nc = tc + dc;
    if (inBounds(nr, nc)) {
      const p = board[nr * 8 + nc];
      if (p && p.color === by && p.type === KNIGHT) return true;
    }
  }

  // Sliding attacks: bishop/queen diagonals, rook/queen straights
  for (const [dr, dc] of BISHOP_DIRS) {
    let nr = tr + dr, nc = tc + dc;
    while (inBounds(nr, nc)) {
      const p = board[nr * 8 + nc];
      if (p) {
        if (p.color === by && (p.type === BISHOP || p.type === QUEEN)) return true;
        break;
      }
      nr += dr; nc += dc;
    }
  }
  for (const [dr, dc] of ROOK_DIRS) {
    let nr = tr + dr, nc = tc + dc;
    while (inBounds(nr, nc)) {
      const p = board[nr * 8 + nc];
      if (p) {
        if (p.color === by && (p.type === ROOK || p.type === QUEEN)) return true;
        break;
      }
      nr += dr; nc += dc;
    }
  }

  // King attacks
  for (const [dr, dc] of KING_DIRS) {
    const nr = tr + dr, nc = tc + dc;
    if (inBounds(nr, nc)) {
      const p = board[nr * 8 + nc];
      if (p && p.color === by && p.type === KING) return true;
    }
  }

  return false;
}

function findKing(pos, color) {
  for (let i = 0; i < 64; i++) {
    const p = pos.board[i];
    if (p && p.color === color && p.type === KING) return i;
  }
  return -1;
}

function isInCheck(pos, color) {
  const kingSq = findKing(pos, color);
  if (kingSq < 0) return true;
  return isSquareAttacked(pos, kingSq, 1 - color);
}

// ============================================================================
// MAKE / UNMAKE MOVE (incremental update)
// ============================================================================

function makeMove(pos, move) {
  const { board, side, castleRights, hash, mgScore, egScore, phase } = pos;
  const enemy = 1 - side;
  const { from, to, flag, captured, promo } = move;
  const piece = board[from];

  // Save undo info
  const undo = {
    castleRights,
    epSq: pos.epSq,
    halfmove: pos.halfmove,
    hash: pos.hash,
    mgScore: pos.mgScore,
    egScore: pos.egScore,
    phase: pos.phase,
    capturedPiece: board[to],
    movedPiece: piece,
  };

  let newHash = hash;
  let newMg = mgScore;
  let newEg = egScore;
  let newPhase = phase;
  let newCastle = castleRights;
  let newEp = -1;
  let newHalfmove = pos.halfmove + 1;

  // Remove piece from source
  const fromPst = side === WHITE ? from : MIRROR[from];
  newHash ^= ZOBRIST_PIECE[side][piece.type][from];
  newMg -= (side === WHITE ? 1 : -1) * (MG_PIECE_VAL[piece.type] + MG_PST[piece.type][fromPst]);
  newEg -= (side === WHITE ? 1 : -1) * (EG_PIECE_VAL[piece.type] + EG_PST[piece.type][fromPst]);

  // Handle captures
  if (flag === FLAG_CAPTURE || flag === FLAG_PROMO_CAP) {
    const capPiece = board[to];
    const capPst = enemy === WHITE ? to : MIRROR[to];
    newHash ^= ZOBRIST_PIECE[enemy][capPiece.type][to];
    newMg += (side === WHITE ? 1 : -1) * (MG_PIECE_VAL[capPiece.type] + MG_PST[capPiece.type][capPst]);
    newEg += (side === WHITE ? 1 : -1) * (EG_PIECE_VAL[capPiece.type] + EG_PST[capPiece.type][capPst]);
    newPhase -= PHASE_WEIGHTS[capPiece.type];
    newHalfmove = 0;
  } else if (flag === FLAG_EP) {
    const epCapSq = side === WHITE ? to + 8 : to - 8;
    const capPst = enemy === WHITE ? epCapSq : MIRROR[epCapSq];
    newHash ^= ZOBRIST_PIECE[enemy][PAWN][epCapSq];
    newMg += (side === WHITE ? 1 : -1) * (MG_PIECE_VAL[PAWN] + MG_PST[PAWN][capPst]);
    newEg += (side === WHITE ? 1 : -1) * (EG_PIECE_VAL[PAWN] + EG_PST[PAWN][capPst]);
    board[epCapSq] = null;
    undo.epCapSq = epCapSq;
    newHalfmove = 0;
  }

  // Place piece on destination
  let placedType = piece.type;
  if (flag === FLAG_PROMO || flag === FLAG_PROMO_CAP) {
    placedType = promo;
    newPhase += PHASE_WEIGHTS[promo] - PHASE_WEIGHTS[PAWN];
  }

  const toPst = side === WHITE ? to : MIRROR[to];
  newHash ^= ZOBRIST_PIECE[side][placedType][to];
  newMg += (side === WHITE ? 1 : -1) * (MG_PIECE_VAL[placedType] + MG_PST[placedType][toPst]);
  newEg += (side === WHITE ? 1 : -1) * (EG_PIECE_VAL[placedType] + EG_PST[placedType][toPst]);

  board[to] = { color: side, type: placedType };
  board[from] = null;

  // Pawn moves reset halfmove
  if (piece.type === PAWN) newHalfmove = 0;

  // Double pawn push - set EP square
  if (flag === FLAG_DOUBLE_PAWN) {
    newEp = side === WHITE ? to + 8 : to - 8;
  }

  // Castling - move rook
  if (flag === FLAG_CASTLE_K) {
    const rookFrom = side === WHITE ? 63 : 7;
    const rookTo = side === WHITE ? 61 : 5;
    const rp = board[rookFrom];
    const rfPst = side === WHITE ? rookFrom : MIRROR[rookFrom];
    const rtPst = side === WHITE ? rookTo : MIRROR[rookTo];
    newHash ^= ZOBRIST_PIECE[side][ROOK][rookFrom];
    newHash ^= ZOBRIST_PIECE[side][ROOK][rookTo];
    newMg -= (side === WHITE ? 1 : -1) * (MG_PIECE_VAL[ROOK] + MG_PST[ROOK][rfPst]);
    newMg += (side === WHITE ? 1 : -1) * (MG_PIECE_VAL[ROOK] + MG_PST[ROOK][rtPst]);
    newEg -= (side === WHITE ? 1 : -1) * (EG_PIECE_VAL[ROOK] + EG_PST[ROOK][rfPst]);
    newEg += (side === WHITE ? 1 : -1) * (EG_PIECE_VAL[ROOK] + EG_PST[ROOK][rtPst]);
    board[rookTo] = rp;
    board[rookFrom] = null;
  } else if (flag === FLAG_CASTLE_Q) {
    const rookFrom = side === WHITE ? 56 : 0;
    const rookTo = side === WHITE ? 59 : 3;
    const rp = board[rookFrom];
    const rfPst = side === WHITE ? rookFrom : MIRROR[rookFrom];
    const rtPst = side === WHITE ? rookTo : MIRROR[rookTo];
    newHash ^= ZOBRIST_PIECE[side][ROOK][rookFrom];
    newHash ^= ZOBRIST_PIECE[side][ROOK][rookTo];
    newMg -= (side === WHITE ? 1 : -1) * (MG_PIECE_VAL[ROOK] + MG_PST[ROOK][rfPst]);
    newMg += (side === WHITE ? 1 : -1) * (MG_PIECE_VAL[ROOK] + MG_PST[ROOK][rtPst]);
    newEg -= (side === WHITE ? 1 : -1) * (EG_PIECE_VAL[ROOK] + EG_PST[ROOK][rfPst]);
    newEg += (side === WHITE ? 1 : -1) * (EG_PIECE_VAL[ROOK] + EG_PST[ROOK][rtPst]);
    board[rookTo] = rp;
    board[rookFrom] = null;
  }

  // Update castling rights
  newHash ^= ZOBRIST_CASTLE[castleRights];
  if (piece.type === KING) {
    if (side === WHITE) newCastle &= ~3;
    else newCastle &= ~12;
  }
  if (piece.type === ROOK) {
    if (from === 63) newCastle &= ~1;
    if (from === 56) newCastle &= ~2;
    if (from === 7) newCastle &= ~4;
    if (from === 0) newCastle &= ~8;
  }
  if (captured) {
    if (to === 63) newCastle &= ~1;
    if (to === 56) newCastle &= ~2;
    if (to === 7) newCastle &= ~4;
    if (to === 0) newCastle &= ~8;
  }
  newHash ^= ZOBRIST_CASTLE[newCastle];

  // Update EP hash
  if (pos.epSq >= 0) newHash ^= ZOBRIST_EP[pos.epSq % 8];
  if (newEp >= 0) newHash ^= ZOBRIST_EP[newEp % 8];

  // Flip side
  newHash ^= ZOBRIST_SIDE;

  pos.castleRights = newCastle;
  pos.epSq = newEp;
  pos.halfmove = newHalfmove;
  pos.hash = newHash;
  pos.side = enemy;
  pos.mgScore = newMg;
  pos.egScore = newEg;
  pos.phase = newPhase;

  return undo;
}

function unmakeMove(pos, move, undo) {
  const { from, to, flag, promo } = move;
  const side = 1 - pos.side; // side that made the move

  // Restore piece
  pos.board[from] = undo.movedPiece;

  if (flag === FLAG_CAPTURE || flag === FLAG_PROMO_CAP) {
    pos.board[to] = undo.capturedPiece;
  } else if (flag === FLAG_EP) {
    pos.board[to] = null;
    pos.board[undo.epCapSq] = { color: 1 - side, type: PAWN };
  } else {
    pos.board[to] = null;
  }

  // Undo castling rook
  if (flag === FLAG_CASTLE_K) {
    const rookFrom = side === WHITE ? 63 : 7;
    const rookTo = side === WHITE ? 61 : 5;
    pos.board[rookFrom] = pos.board[rookTo];
    pos.board[rookTo] = null;
  } else if (flag === FLAG_CASTLE_Q) {
    const rookFrom = side === WHITE ? 56 : 0;
    const rookTo = side === WHITE ? 59 : 3;
    pos.board[rookFrom] = pos.board[rookTo];
    pos.board[rookTo] = null;
  }

  pos.side = side;
  pos.castleRights = undo.castleRights;
  pos.epSq = undo.epSq;
  pos.halfmove = undo.halfmove;
  pos.hash = undo.hash;
  pos.mgScore = undo.mgScore;
  pos.egScore = undo.egScore;
  pos.phase = undo.phase;
}

// ============================================================================
// EVALUATION
// ============================================================================

function evaluate(pos) {
  const ph = Math.min(pos.phase, TOTAL_PHASE);
  const egPh = TOTAL_PHASE - ph;
  const score = (pos.mgScore * ph + pos.egScore * egPh) / TOTAL_PHASE;
  return pos.side === WHITE ? score : -score;
}

// ============================================================================
// MOVE ORDERING
// ============================================================================

// MVV-LVA table: [victim][attacker] -> score
const MVV_LVA = [];
for (let v = 0; v <= 6; v++) {
  MVV_LVA[v] = [];
  for (let a = 0; a <= 6; a++) {
    MVV_LVA[v][a] = v * 100 - a;
  }
}

// Killer moves: 2 per ply
const killers = [];
for (let i = 0; i < MAX_DEPTH; i++) killers[i] = [null, null];

// History heuristic: [color][from][to]
const history = [];
for (let c = 0; c < 2; c++) {
  history[c] = [];
  for (let f = 0; f < 64; f++) {
    history[c][f] = new Int32Array(64);
  }
}

function moveKey(m) { return (m.from << 6) | m.to | ((m.promo || 0) << 12); }

function scoreMove(move, ply, ttMove, side) {
  // TT move gets highest priority
  if (ttMove && move.from === ttMove.from && move.to === ttMove.to &&
      (move.promo || 0) === (ttMove.promo || 0)) return 10000000;

  if (move.flag === FLAG_PROMO || move.flag === FLAG_PROMO_CAP) {
    return 9000000 + (move.promo === QUEEN ? 1000 : 0);
  }

  if (move.captured) {
    return 1000000 + MVV_LVA[move.captured][move.movedType || PAWN];
  }

  const mk = moveKey(move);
  if (killers[ply][0] && moveKey(killers[ply][0]) === mk) return 900000;
  if (killers[ply][1] && moveKey(killers[ply][1]) === mk) return 800000;

  return history[side][move.from][move.to];
}

function orderMoves(moves, ply, ttMove, side) {
  // Annotate moved piece type for MVV-LVA
  for (const m of moves) {
    m._score = scoreMove(m, ply, ttMove, side);
  }
  moves.sort((a, b) => b._score - a._score);
}

// ============================================================================
// SEARCH
// ============================================================================

let nodes = 0;
let searchStartTime = 0;
let timeLimit = 0;
let searchAborted = false;

function timeUp() {
  if ((nodes & 1023) === 0) {
    if (Date.now() - searchStartTime >= timeLimit) {
      searchAborted = true;
      return true;
    }
  }
  return searchAborted;
}

function quiescence(pos, alpha, beta, ply) {
  if (searchAborted) return 0;
  nodes++;

  const standPat = evaluate(pos);
  if (standPat >= beta) return beta;
  // Delta pruning
  if (standPat + 1100 < alpha) return alpha;
  if (standPat > alpha) alpha = standPat;

  const moves = generateMoves(pos, true);
  const side = pos.side;
  // Simple MVV-LVA ordering for captures
  for (const m of moves) {
    m._score = m.captured ? MVV_LVA[m.captured][pos.board[m.from] ? pos.board[m.from].type : PAWN] : 0;
  }
  moves.sort((a, b) => b._score - a._score);

  for (const move of moves) {
    // Delta pruning per-move
    if (standPat + MG_PIECE_VAL[move.captured || 0] + 200 < alpha) continue;

    const undo = makeMove(pos, move);
    // Check legality
    if (isInCheck(pos, 1 - pos.side)) {
      unmakeMove(pos, move, undo);
      continue;
    }

    const score = -quiescence(pos, -beta, -alpha, ply + 1);
    unmakeMove(pos, move, undo);

    if (searchAborted) return 0;
    if (score >= beta) return beta;
    if (score > alpha) alpha = score;
  }
  return alpha;
}

// Futility pruning margins by depth
const FUTILITY_MARGIN = [0, 180, 400, 700];
// Late move pruning thresholds by depth
const LMP_THRESHOLD = [0, 5, 9, 14, 22];
// Razoring margins
const RAZOR_MARGIN = [0, 300, 550];

function search(pos, alpha, beta, depth, ply, doNull) {
  if (searchAborted || timeUp()) return 0;

  // Draw by 50-move rule
  if (pos.halfmove >= 100) return 0;

  const isRoot = ply === 0;
  const inCheck = isInCheck(pos, pos.side);

  // Check extension
  if (inCheck) depth++;

  if (depth <= 0) return quiescence(pos, alpha, beta, ply);

  nodes++;

  const isPV = beta - alpha > 1;

  // TT probe
  let ttMove = null;
  const ttResult = ttProbe(pos.hash, depth, alpha, beta, ply);
  if (ttResult) {
    ttMove = ttResult.move;
    if (ttResult.score !== null && !isRoot && !isPV) return ttResult.score;
  }

  // Static evaluation for pruning decisions
  const staticEval = inCheck ? -INFINITY : evaluate(pos);

  // Razoring: at low depth, if static eval is far below alpha, drop into qsearch
  if (!isPV && !inCheck && depth <= 2 && staticEval + RAZOR_MARGIN[depth] <= alpha) {
    const razorScore = quiescence(pos, alpha, beta, ply);
    if (razorScore <= alpha) return razorScore;
  }

  // Reverse futility pruning (static null move pruning)
  if (!isPV && !inCheck && depth <= 3 && doNull && staticEval - FUTILITY_MARGIN[depth] >= beta) {
    return staticEval;
  }

  // Null move pruning
  if (doNull && !inCheck && depth >= 3 && pos.phase > 2 && staticEval >= beta) {
    // Make null move
    const oldEp = pos.epSq;
    const oldHash = pos.hash;
    if (pos.epSq >= 0) pos.hash ^= ZOBRIST_EP[pos.epSq % 8];
    pos.epSq = -1;
    pos.side = 1 - pos.side;
    pos.hash ^= ZOBRIST_SIDE;

    const R = 3 + Math.min(Math.floor((staticEval - beta) / 200), 2) + (depth > 6 ? 1 : 0);
    const nullScore = -search(pos, -beta, -beta + 1, depth - R - 1, ply + 1, false);

    pos.side = 1 - pos.side;
    pos.epSq = oldEp;
    pos.hash = oldHash;

    if (searchAborted) return 0;
    if (nullScore >= beta) return beta;
  }

  const moves = generateMoves(pos, false);
  orderMoves(moves, ply, ttMove, pos.side);

  let bestScore = -INFINITY;
  let bestMove = null;
  let movesSearched = 0;
  let ttFlag = TT_UPPER;

  for (let i = 0; i < moves.length; i++) {
    const move = moves[i];
    const undo = makeMove(pos, move);

    // Legality check
    if (isInCheck(pos, 1 - pos.side)) {
      unmakeMove(pos, move, undo);
      continue;
    }

    const isCapture = move.flag === FLAG_CAPTURE || move.flag === FLAG_PROMO_CAP || move.flag === FLAG_EP;
    const isPromo = move.flag === FLAG_PROMO || move.flag === FLAG_PROMO_CAP;
    const givesCheck = isInCheck(pos, pos.side);

    // Late move pruning: skip late quiet moves at shallow depth
    if (!isPV && !inCheck && !givesCheck && !isCapture && !isPromo && depth <= 4 && movesSearched >= (LMP_THRESHOLD[depth] || 20)) {
      unmakeMove(pos, move, undo);
      continue;
    }

    // Futility pruning: at shallow depth, skip quiet moves that can't raise alpha
    if (!isPV && !inCheck && !givesCheck && !isCapture && !isPromo && depth <= 3 && movesSearched > 0) {
      if (staticEval + FUTILITY_MARGIN[depth] <= alpha) {
        unmakeMove(pos, move, undo);
        continue;
      }
    }

    let score;

    // LMR
    if (movesSearched >= 3 && depth >= 3 && !inCheck && !isCapture && !isPromo && !givesCheck) {
      let R = Math.floor(0.5 + Math.log(depth) * Math.log(movesSearched) / 2.0);
      if (!isPV) R++;
      R = Math.min(R, depth - 2);
      if (R < 1) R = 1;
      score = -search(pos, -alpha - 1, -alpha, depth - 1 - R, ply + 1, true);
      if (score > alpha) {
        score = -search(pos, -alpha - 1, -alpha, depth - 1, ply + 1, true);
      }
    } else if (movesSearched > 0) {
      // PVS zero-window
      score = -search(pos, -alpha - 1, -alpha, depth - 1, ply + 1, true);
    } else {
      score = alpha + 1; // force full window search
    }

    // Full window re-search
    if (score > alpha) {
      score = -search(pos, -beta, -alpha, depth - 1, ply + 1, true);
    }

    unmakeMove(pos, move, undo);
    movesSearched++;

    if (searchAborted) return bestScore !== -INFINITY ? bestScore : 0;

    if (score > bestScore) {
      bestScore = score;
      bestMove = move;

      if (score > alpha) {
        alpha = score;
        ttFlag = TT_EXACT;

        if (score >= beta) {
          ttFlag = TT_LOWER;
          // Update killer moves and history for quiet moves
          if (!isCapture) {
            if (!killers[ply][0] || moveKey(killers[ply][0]) !== moveKey(move)) {
              killers[ply][1] = killers[ply][0];
              killers[ply][0] = { from: move.from, to: move.to, promo: move.promo };
            }
            history[pos.side][move.from][move.to] += depth * depth;
            // Prevent overflow
            if (history[pos.side][move.from][move.to] > 100000) {
              for (let f = 0; f < 64; f++) for (let t = 0; t < 64; t++) history[pos.side][f][t] >>= 1;
            }
          }
          break;
        }
      }
    }
  }

  // Checkmate / stalemate
  if (movesSearched === 0) {
    bestScore = inCheck ? -MATE_SCORE + ply : 0;
  }

  ttStore(pos.hash, depth, bestScore, ttFlag, bestMove ? { from: bestMove.from, to: bestMove.to, promo: bestMove.promo } : null, ply);

  return bestScore;
}

// ============================================================================
// ITERATIVE DEEPENING
// ============================================================================

// Root search wrapper that captures the best move directly
let rootBestMove = null;

function searchRoot(pos, alpha, beta, depth) {
  const inCheck = isInCheck(pos, pos.side);
  if (inCheck) depth++;

  nodes++;
  let ttMove = null;
  const ttResult = ttProbe(pos.hash, depth, alpha, beta, 0);
  if (ttResult) ttMove = ttResult.move;

  // Null move pruning at root not useful, skip it

  const moves = generateMoves(pos, false);
  orderMoves(moves, 0, ttMove, pos.side);

  let bestScore = -INFINITY;
  let bestMove = null;
  let movesSearched = 0;
  let ttFlag = TT_UPPER;

  for (let i = 0; i < moves.length; i++) {
    const move = moves[i];
    const undo = makeMove(pos, move);
    if (isInCheck(pos, 1 - pos.side)) { unmakeMove(pos, move, undo); continue; }

    let score;
    const isCapture = move.flag === FLAG_CAPTURE || move.flag === FLAG_PROMO_CAP || move.flag === FLAG_EP;
    const isPromo = move.flag === FLAG_PROMO || move.flag === FLAG_PROMO_CAP;
    const givesCheck = isInCheck(pos, pos.side);

    if (movesSearched >= 3 && depth >= 3 && !inCheck && !isCapture && !isPromo && !givesCheck) {
      let R = Math.floor(0.5 + Math.log(depth) * Math.log(movesSearched) / 2.0);
      R = Math.min(R, depth - 2);
      if (R < 1) R = 1;
      score = -search(pos, -alpha - 1, -alpha, depth - 1 - R, 1, true);
      if (score > alpha) score = -search(pos, -alpha - 1, -alpha, depth - 1, 1, true);
    } else if (movesSearched > 0) {
      score = -search(pos, -alpha - 1, -alpha, depth - 1, 1, true);
    } else {
      score = alpha + 1;
    }
    if (score > alpha) score = -search(pos, -beta, -alpha, depth - 1, 1, true);

    unmakeMove(pos, move, undo);
    movesSearched++;

    if (searchAborted) { if (bestMove) rootBestMove = bestMove; return bestScore !== -INFINITY ? bestScore : 0; }

    if (score > bestScore) {
      bestScore = score;
      bestMove = move;
      if (score > alpha) {
        alpha = score;
        ttFlag = TT_EXACT;
        if (score >= beta) { ttFlag = TT_LOWER; break; }
      }
    }
  }

  if (movesSearched === 0) bestScore = inCheck ? -MATE_SCORE : 0;
  if (bestMove) rootBestMove = bestMove;
  ttStore(pos.hash, depth, bestScore, ttFlag, bestMove ? { from: bestMove.from, to: bestMove.to, promo: bestMove.promo } : null, 0);
  return bestScore;
}

function iterativeDeepening(pos) {
  searchStartTime = Date.now();
  // Time management tuned for a 5-second arena budget with a small safety margin.
  timeLimit = 4600;
  searchAborted = false;
  nodes = 0;
  rootBestMove = null;

  let prevScore = 0;
  const ASP_WINDOW = 50; // aspiration window size

  for (let depth = 1; depth <= MAX_DEPTH; depth++) {
    let score;

    // Aspiration windows: use narrow window around previous score at depth >= 5
    if (depth >= 5) {
      let alpha = prevScore - ASP_WINDOW;
      let beta = prevScore + ASP_WINDOW;
      score = searchRoot(pos, alpha, beta, depth);

      // Re-search with wider window on fail
      if (!searchAborted && (score <= alpha || score >= beta)) {
        // Widen significantly
        alpha = score <= alpha ? prevScore - ASP_WINDOW * 4 : alpha;
        beta = score >= beta ? prevScore + ASP_WINDOW * 4 : beta;
        score = searchRoot(pos, alpha, beta, depth);
      }
      // Full window if still failing
      if (!searchAborted && (score <= alpha || score >= beta)) {
        score = searchRoot(pos, -INFINITY, INFINITY, depth);
      }
    } else {
      score = searchRoot(pos, -INFINITY, INFINITY, depth);
    }

    if (searchAborted && depth > 1) break;

    prevScore = score;
    const elapsed = Date.now() - searchStartTime;

    // Leave room for the next depth and platform overhead.
    if (elapsed * 1.15 > timeLimit) break;

    // If we found a mate, stop searching
    if (Math.abs(score) > MATE_SCORE - 100) break;
  }

  return rootBestMove;
}

// ============================================================================
// MOVE TO UCI
// ============================================================================

function moveToUci(move) {
  const fromFile = FILES[move.from & 7];
  const fromRank = 8 - (move.from >> 3);
  const toFile = FILES[move.to & 7];
  const toRank = 8 - (move.to >> 3);
  let uci = `${fromFile}${fromRank}${toFile}${toRank}`;
  if (move.promo) uci += PROMO_UCI[move.promo];
  return uci;
}

// ============================================================================
// MAIN
// ============================================================================

const fen = readFileSync(0, 'utf8').trim();
const pos = parseFen(fen);

// Fallback: if search returns nothing, pick first legal move
let bestMove = iterativeDeepening(pos);

if (!bestMove) {
  const legal = generateMoves(pos, false).filter(m => {
    const undo = makeMove(pos, m);
    const legal = !isInCheck(pos, 1 - pos.side);
    unmakeMove(pos, m, undo);
    return legal;
  });
  if (legal.length > 0) bestMove = legal[0];
}

process.stdout.write(`${bestMove ? moveToUci(bestMove) : '0000'}\n`);
