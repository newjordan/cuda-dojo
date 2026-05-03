// ============================================================================
// THE CAVALRY — The Gambler's Instinct
//
// Walks a primary strategy but constantly scans for opportunistic edge cases.
// When it spots one, it calculates recovery confidence — "can I get back
// to safety if this doesn't work out?"
//
// High confidence → MORPH: take the shot, deviate from plan
// Low confidence → HOLD: stay on primary strategy
//
// This is NOT reckless. It's calculated aggression with a lifeline.
//
// The cavalry operates as an OVERLAY on the layered controller.
// It doesn't replace the eval — it ADJUSTS it when opportunity knocks.
//
// Three components:
//   1. OPPORTUNITY SCANNER — detects tactical edges in current position
//   2. RECOVERY ESTIMATOR — how safe is our fallback if the tactic fails?
//   3. MORPH DECISION — blend opportunistic vs conservative eval weights
//
// Integration: called once per root position, returns a morph factor (0.0-1.0)
// that blends between "stay on plan" (0.0) and "go for it" (1.0)
// ============================================================================

// === OPPORTUNITY TYPES ===
// Each has a reward estimate and a detection function
const OPP_NONE = 0;
const OPP_HANGING_PIECE = 1;     // enemy piece undefended
const OPP_FORK_AVAILABLE = 2;    // we can fork two pieces
const OPP_PIN_EXPLOITABLE = 3;   // enemy piece is pinned, we can pile on
const OPP_WEAK_KING = 4;         // enemy king exposed, attack possible
const OPP_PASSED_PAWN_RUSH = 5;  // we can push a passer and they can't stop it
const OPP_EXCHANGE_WIN = 6;      // favorable exchange sequence available
const OPP_BACK_RANK = 7;         // back rank threat
const OPP_PROMOTION = 8;         // pawn near promotion
const OPP_TRAPPED_PIECE = 9;     // enemy piece has no escape squares

// Reward estimates in centipawns (how much we gain if the tactic works)
const OPP_REWARD = [0, 300, 500, 350, 600, 400, 250, 800, 700, 400];

// Risk estimates (how much we lose if it fails and we can't recover)
const OPP_RISK = [0, 100, 200, 150, 400, 150, 100, 500, 200, 150];

// ============================================================================
// OPPORTUNITY SCANNER
// Fast detection of tactical edges (runs once, ~0.5ms)
// ============================================================================

export function scanOpportunities(board, side, phase) {
  const enemy = 1 - side;
  const opportunities = [];
  const pieceVal = [0, 100, 320, 330, 500, 900, 20000];

  // Track piece positions
  let ourKingSq = -1, theirKingSq = -1;
  const ourPieces = [], theirPieces = [];

  for (let sq = 0; sq < 64; sq++) {
    const p = board[sq];
    if (!p) continue;
    if (p.color === side) {
      if (p.type === 6) ourKingSq = sq; // KING=6
      ourPieces.push({ sq, type: p.type, val: pieceVal[p.type] });
    } else {
      if (p.type === 6) theirKingSq = sq;
      theirPieces.push({ sq, type: p.type, val: pieceVal[p.type] });
    }
  }

  // --- OPP_HANGING_PIECE: enemy piece with no defenders ---
  for (const tp of theirPieces) {
    if (tp.type === 6 || tp.type === 1) continue; // skip king and pawns
    let defended = false;
    // Quick check: is any friendly piece adjacent/defending?
    for (const tp2 of theirPieces) {
      if (tp2.sq === tp.sq) continue;
      const dr = Math.abs((tp.sq >> 3) - (tp2.sq >> 3));
      const dc = Math.abs((tp.sq & 7) - (tp2.sq & 7));
      // Rough: if another piece is within 2 squares, likely defended
      if (dr <= 2 && dc <= 2 && tp2.type !== 1) { defended = true; break; }
      // Pawn defense
      if (tp2.type === 1) {
        const pawnDir = enemy === 0 ? 1 : -1; // WHITE=0 goes up
        if ((tp2.sq >> 3) + pawnDir === (tp.sq >> 3) && Math.abs((tp2.sq & 7) - (tp.sq & 7)) === 1) {
          defended = true; break;
        }
      }
    }
    if (!defended) {
      // Can we attack it?
      for (const op of ourPieces) {
        if (op.val < tp.val) { // we attack with lesser piece = clear win
          opportunities.push({ type: OPP_HANGING_PIECE, reward: tp.val, risk: 50, sq: tp.sq });
          break;
        }
      }
    }
  }

  // --- OPP_WEAK_KING: enemy king with few defenders ---
  if (theirKingSq >= 0 && phase > 8) {
    const kr = theirKingSq >> 3, kc = theirKingSq & 7;
    let defenders = 0;
    let attackers = 0;
    for (let dr = -2; dr <= 2; dr++) for (let dc = -2; dc <= 2; dc++) {
      const r = kr + dr, c = kc + dc;
      if (r < 0 || r > 7 || c < 0 || c > 7) continue;
      const p = board[r * 8 + c];
      if (!p) continue;
      if (p.color === enemy && p.type !== 6) defenders++;
      if (p.color === side && p.type !== 6 && p.type !== 1) attackers++;
    }
    if (attackers >= 2 && defenders <= 1) {
      opportunities.push({ type: OPP_WEAK_KING, reward: 600, risk: 300, sq: theirKingSq });
    }
  }

  // --- OPP_PASSED_PAWN_RUSH: advanced passed pawn we can push ---
  for (const op of ourPieces) {
    if (op.type !== 1) continue; // PAWN=1
    const rank = side === 0 ? (7 - (op.sq >> 3)) : (op.sq >> 3);
    if (rank >= 5) {
      // Check if it's actually passed
      const file = op.sq & 7;
      let passed = true;
      for (const tp of theirPieces) {
        if (tp.type !== 1) continue;
        const tf = tp.sq & 7;
        if (Math.abs(tf - file) <= 1) {
          const tRank = enemy === 0 ? (7 - (tp.sq >> 3)) : (tp.sq >> 3);
          if ((side === 0 && (tp.sq >> 3) < (op.sq >> 3)) || (side === 1 && (tp.sq >> 3) > (op.sq >> 3))) {
            passed = false; break;
          }
        }
      }
      if (passed) {
        opportunities.push({ type: OPP_PASSED_PAWN_RUSH, reward: rank * 120, risk: 100, sq: op.sq });
      }
    }
  }

  // --- OPP_BACK_RANK: enemy king on back rank with no escape ---
  if (theirKingSq >= 0) {
    const backRank = enemy === 0 ? 7 : 0; // WHITE=0, rank 7 = rank 1
    if ((theirKingSq >> 3) === backRank) {
      const dir = enemy === 0 ? -1 : 1;
      let escape = 0;
      for (let dc = -1; dc <= 1; dc++) {
        const r = backRank + dir, c = (theirKingSq & 7) + dc;
        if (r >= 0 && r < 8 && c >= 0 && c < 8) {
          const p = board[r * 8 + c];
          if (!p || p.color === side) escape++;
        }
      }
      if (escape === 0) {
        // Check if we have a rook or queen that could deliver
        for (const op of ourPieces) {
          if (op.type === 4 || op.type === 5) { // ROOK=4, QUEEN=5
            opportunities.push({ type: OPP_BACK_RANK, reward: 900, risk: 200, sq: theirKingSq });
            break;
          }
        }
      }
    }
  }

  // --- OPP_TRAPPED_PIECE: enemy piece with 0-1 legal-looking squares ---
  for (const tp of theirPieces) {
    if (tp.type === 1 || tp.type === 6) continue;
    const tr = tp.sq >> 3, tc = tp.sq & 7;
    let escapes = 0;
    // Check all adjacent squares
    for (let dr = -1; dr <= 1; dr++) for (let dc = -1; dc <= 1; dc++) {
      if (dr === 0 && dc === 0) continue;
      const r = tr + dr, c = tc + dc;
      if (r < 0 || r > 7 || c < 0 || c > 7) continue;
      const p = board[r * 8 + c];
      if (!p || p.color === side) escapes++; // empty or capturable = escape
    }
    if (escapes <= 1 && tp.val >= 320) {
      opportunities.push({ type: OPP_TRAPPED_PIECE, reward: tp.val, risk: 100, sq: tp.sq });
    }
  }

  return opportunities;
}

// ============================================================================
// RECOVERY ESTIMATOR
// "If I deviate from plan, how hard is it to get back to safety?"
//
// Factors:
//   - Our material advantage (more material = easier recovery)
//   - Our king safety (safe king = can afford to attack)
//   - Piece activity (active pieces = flexible recovery)
//   - Pawn structure solidity (solid = safe base to return to)
//   - Time in game (early = more time to recover, late = less)
// ============================================================================

export function estimateRecovery(board, side, phase, opportunities) {
  if (opportunities.length === 0) return { confidence: 0, shouldMorph: false };

  const enemy = 1 - side;
  const pieceVal = [0, 100, 320, 330, 500, 900, 0];

  let ourMaterial = 0, theirMaterial = 0;
  let ourKingSafe = true;
  let ourPawnSolid = 0;
  let ourActivity = 0;
  let ourKingSq = -1;

  for (let sq = 0; sq < 64; sq++) {
    const p = board[sq];
    if (!p) continue;
    if (p.color === side) {
      ourMaterial += pieceVal[p.type];
      if (p.type === 6) ourKingSq = sq;
      // Activity: pieces in center or enemy half
      const r = side === 0 ? (sq >> 3) : (7 - (sq >> 3));
      if (r <= 4 && p.type !== 1 && p.type !== 6) ourActivity++;
    } else {
      theirMaterial += pieceVal[p.type];
    }
  }

  // King safety: check pawn shield
  if (ourKingSq >= 0) {
    const kr = ourKingSq >> 3, kc = ourKingSq & 7;
    const shieldDir = side === 0 ? -1 : 1;
    let shield = 0;
    for (let dc = -1; dc <= 1; dc++) {
      const r = kr + shieldDir, c = kc + dc;
      if (r >= 0 && r < 8 && c >= 0 && c < 8) {
        const p = board[r * 8 + c];
        if (p && p.type === 1 && p.color === side) shield++;
      }
    }
    if (shield <= 1) ourKingSafe = false;
  }

  // Pawn structure: count pawn islands (fewer = more solid)
  const pawnFiles = new Uint8Array(8);
  for (let sq = 0; sq < 64; sq++) {
    const p = board[sq];
    if (p && p.type === 1 && p.color === side) pawnFiles[sq & 7] = 1;
  }
  let islands = 0;
  let inIsland = false;
  for (let f = 0; f < 8; f++) {
    if (pawnFiles[f] && !inIsland) { islands++; inIsland = true; }
    if (!pawnFiles[f]) inIsland = false;
  }
  ourPawnSolid = Math.max(0, 4 - islands); // 1 island = 3, 4+ islands = 0

  // Recovery confidence factors (each 0-1)
  const materialFactor = Math.min(1, Math.max(0, (ourMaterial - theirMaterial + 300) / 600));
  const kingSafeFactor = ourKingSafe ? 0.8 : 0.3;
  const activityFactor = Math.min(1, ourActivity / 4);
  const pawnFactor = ourPawnSolid / 3;
  const phaseFactor = phase > 12 ? 0.7 : phase > 6 ? 0.5 : 0.3; // early game = more recovery time

  // Weighted recovery confidence
  const confidence = (
    materialFactor * 0.25 +
    kingSafeFactor * 0.30 +
    activityFactor * 0.20 +
    pawnFactor * 0.10 +
    phaseFactor * 0.15
  );

  // Pick the best opportunity (highest reward/risk ratio with recovery)
  let bestOpp = null;
  let bestScore = 0;
  for (const opp of opportunities) {
    const expectedValue = (opp.reward * confidence) - (opp.risk * (1 - confidence));
    if (expectedValue > bestScore) {
      bestScore = expectedValue;
      bestOpp = opp;
    }
  }

  // Morph threshold: only deviate if expected value is clearly positive
  // AND recovery confidence is above 0.4
  const shouldMorph = bestScore > 100 && confidence > 0.4;

  return {
    confidence,
    shouldMorph,
    bestOpportunity: bestOpp,
    expectedValue: bestScore,
    factors: { materialFactor, kingSafeFactor, activityFactor, pawnFactor, phaseFactor },
  };
}

// ============================================================================
// MORPH DECISION — Returns weight adjustments when morphing
//
// When shouldMorph is true, these adjustments push the eval toward
// the opportunity. When false, returns zeros (stay on plan).
//
// The key insight: we don't change the SEARCH, we change the EVAL WEIGHTS
// so the search naturally finds the tactical line.
// ============================================================================

export function getMorphWeights(morphResult) {
  if (!morphResult.shouldMorph || !morphResult.bestOpportunity) {
    return { pawnAdj: 0, kingAdj: 0, queenAdj: 0, rookAdj: 0, minorAdj: 0, tempoAdj: 0, aggAdj: 0 };
  }

  const opp = morphResult.bestOpportunity;
  const conf = morphResult.confidence;

  // Scale adjustments by confidence — higher confidence = bolder morph
  const scale = Math.min(conf * 1.5, 1.0);

  switch (opp.type) {
    case 1: // HANGING_PIECE — boost tempo to grab it
      return { pawnAdj: 0, kingAdj: 0, queenAdj: 0, rookAdj: 0, minorAdj: Math.round(2 * scale), tempoAdj: Math.round(4 * scale), aggAdj: Math.round(3 * scale) };

    case 2: // FORK_AVAILABLE — boost knight/minor eval
      return { pawnAdj: 0, kingAdj: 0, queenAdj: 0, rookAdj: 0, minorAdj: Math.round(4 * scale), tempoAdj: Math.round(3 * scale), aggAdj: Math.round(2 * scale) };

    case 3: // PIN_EXPLOITABLE — boost rook/queen activity
      return { pawnAdj: 0, kingAdj: 0, queenAdj: Math.round(2 * scale), rookAdj: Math.round(3 * scale), minorAdj: 0, tempoAdj: Math.round(2 * scale), aggAdj: Math.round(2 * scale) };

    case 4: // WEAK_KING — go all in on attack
      return { pawnAdj: -Math.round(2 * scale), kingAdj: -Math.round(3 * scale), queenAdj: Math.round(4 * scale), rookAdj: Math.round(3 * scale), minorAdj: Math.round(2 * scale), tempoAdj: Math.round(5 * scale), aggAdj: Math.round(4 * scale) };

    case 5: // PASSED_PAWN_RUSH — boost pawn eval hard
      return { pawnAdj: Math.round(5 * scale), kingAdj: 0, queenAdj: 0, rookAdj: Math.round(2 * scale), minorAdj: 0, tempoAdj: Math.round(3 * scale), aggAdj: Math.round(2 * scale) };

    case 6: // EXCHANGE_WIN — boost captures
      return { pawnAdj: 0, kingAdj: 0, queenAdj: 0, rookAdj: 0, minorAdj: 0, tempoAdj: Math.round(4 * scale), aggAdj: Math.round(3 * scale) };

    case 7: // BACK_RANK — maximum attack
      return { pawnAdj: 0, kingAdj: -Math.round(2 * scale), queenAdj: Math.round(3 * scale), rookAdj: Math.round(5 * scale), minorAdj: 0, tempoAdj: Math.round(4 * scale), aggAdj: Math.round(5 * scale) };

    case 8: // PROMOTION — push the pawn
      return { pawnAdj: Math.round(6 * scale), kingAdj: 0, queenAdj: 0, rookAdj: Math.round(2 * scale), minorAdj: 0, tempoAdj: Math.round(5 * scale), aggAdj: Math.round(3 * scale) };

    case 9: // TRAPPED_PIECE — exploit immobility
      return { pawnAdj: 0, kingAdj: 0, queenAdj: 0, rookAdj: 0, minorAdj: Math.round(3 * scale), tempoAdj: Math.round(4 * scale), aggAdj: Math.round(2 * scale) };

    default:
      return { pawnAdj: 0, kingAdj: 0, queenAdj: 0, rookAdj: 0, minorAdj: 0, tempoAdj: 0, aggAdj: 0 };
  }
}

// ============================================================================
// THE CAVALRY API — Single call from engine
// ============================================================================

export function cavalryAnalysis(board, side, phase) {
  const opportunities = scanOpportunities(board, side, phase);
  const recovery = estimateRecovery(board, side, phase, opportunities);
  const weights = getMorphWeights(recovery);

  return {
    morphing: recovery.shouldMorph,
    confidence: recovery.confidence,
    opportunity: recovery.bestOpportunity,
    expectedValue: recovery.expectedValue,
    weights,
    opportunityCount: opportunities.length,
  };
}
