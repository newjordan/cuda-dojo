import { briefEngine, detectPatterns, getCounterStrategy } from '../frostd4d/variants/the_un.js';
import { cavalryAnalysis } from './morph_target_brain.js';

const LINK_SIZE = 64;
const ROOT_STAT_SIZE = 64 * 64;

export const BUS = {
  PHASE: 0,
  VOLATILITY: 1,
  BOOK_HIT: 2,
  UN_EVAL: 3,
  CAV_CONF: 4,
  CAV_EV: 5,
  SNIPER_STYLE: 6,
  COUNTER_TEMPO: 7,
  COUNTER_SAFETY: 8,
  COUNTER_ACTIVITY: 9,
  LAYER0_TEMP: 12,
  LAYER1_TEMP: 13,
  LAYER2_TEMP: 14,
  LAYER3_TEMP: 15,
  LAYER4_TEMP: 16,
};

const STYLE_CODE = {
  fortress: -8,
  grinder: -2,
  positional: 2,
  attacker: 6,
  gambit: 8,
};

function clamp8(value) {
  return Math.max(-127, Math.min(127, Math.round(value)));
}

function clamp16(value) {
  return Math.max(-32768, Math.min(32767, Math.round(value)));
}

function depthBand(depth) {
  if (depth <= 1) return 0;
  if (depth <= 3) return 1;
  if (depth <= 5) return 2;
  if (depth <= 8) return 3;
  return 4;
}

function addHeat(squareHeat, sq, radius, amount) {
  if (sq < 0) return;
  const baseRank = sq >> 3;
  const baseFile = sq & 7;
  for (let dr = -radius; dr <= radius; dr++) {
    for (let df = -radius; df <= radius; df++) {
      const rank = baseRank + dr;
      const file = baseFile + df;
      if (rank < 0 || rank > 7 || file < 0 || file > 7) continue;
      const dist = Math.abs(dr) + Math.abs(df);
      const scaled = amount - dist * 2;
      if (scaled <= 0) continue;
      const idx = rank * 8 + file;
      squareHeat[idx] = clamp8(squareHeat[idx] + scaled);
    }
  }
}

function parseUciSquare(uci) {
  if (!uci || uci.length < 4) return -1;
  const file = uci.charCodeAt(2) - 97;
  const rank = 8 - Number(uci[3]);
  if (file < 0 || file > 7 || rank < 0 || rank > 7) return -1;
  return rank * 8 + file;
}

function classifySniperStyle(board, side, phase, threats, cavalry) {
  const enemy = 1 - side;
  let queensOn = 0;
  let enemyAdvanced = 0;
  let enemyPawnCount = 0;
  let enemyMinorCount = 0;
  let enemyKingSq = -1;

  for (let sq = 0; sq < 64; sq++) {
    const piece = board[sq];
    if (!piece || piece.color !== enemy) continue;
    if (piece.type === 5) queensOn++;
    if (piece.type === 1) enemyPawnCount++;
    if (piece.type === 2 || piece.type === 3) enemyMinorCount++;
    if (piece.type === 6) enemyKingSq = sq;
    const forward = enemy === 0 ? 7 - (sq >> 3) : (sq >> 3);
    if (piece.type !== 1 && piece.type !== 6 && forward >= 3) enemyAdvanced++;
  }

  let style = 'grinder';
  let profile = 'universal';

  if (cavalry.opportunityCount >= 3 || threats.includes('greek_gift') || threats.includes('smothered_mate')) {
    style = 'gambit';
    profile = 'tactical';
  } else if (queensOn > 0 && enemyAdvanced >= 3) {
    style = 'attacker';
    profile = 'aggressive';
  } else if (enemyPawnCount >= 7 && enemyAdvanced <= 1) {
    style = 'fortress';
    profile = 'defensive';
  } else if (phase > 12 && enemyMinorCount >= 3) {
    style = 'positional';
    profile = 'balanced';
  }

  return {
    style,
    profile,
    volatility: style === 'gambit' ? 15 : style === 'attacker' ? 10 : 2,
    styleCode: STYLE_CODE[style],
    enemyKingSq,
  };
}

function buildSquareHeat(state, briefing, cavalry, sniper, knobs) {
  const heat = state.squareHeat;
  // Center Control
  for (const sq of [27, 28, 35, 36]) heat[sq] += 4;
  // Opportunity Heat
  if (cavalry.bestTargetSq >= 0) addHeat(heat, cavalry.bestTargetSq, 2, knobs.opportunityHeat || 12);
  // King Ring Heat
  if (sniper.enemyKingSq >= 0) addHeat(heat, sniper.enemyKingSq, 2, knobs.kingRingHeat || 10);
  // Book Heat
  const bookSq = parseUciSquare(briefing.bookMove);
  if (bookSq >= 0) addHeat(heat, bookSq, 1, 6);
}

export function createTraincarState() {
  return {
    link: new Int8Array(LINK_SIZE),
    layerTemps: new Int8Array(5),
    squareHeat: new Int8Array(64),
    pieceHeat: new Int8Array(8),
    rootStats: new Int16Array(ROOT_STAT_SIZE),
    lastScores: new Int16Array(ROOT_STAT_SIZE),
    briefing: null,
    cavalry: null,
    counter: null,
    sniper: null,
    bookMove: null,
    trace: null,
  };
}

export function beginDojoTrace(state, fighter, fullmove, phase, bookMove) {
  state.trace = {
    fighter,
    fullmove,
    phase,
    bookMove,
    layerTemps: [0, 0, 0, 0, 0],
    maxDepth: 0,
    selectedScore: 0,
    selectedSource: 'none',
    selectedLayer: null,
    layers: [],
    iterations: [],
  };
}

export function noteDojoSearch(state, depth, tuning, source) {
  if (!state.trace) return;
  const index = depthBand(depth);
  if (!state.trace.layers[index]) {
    state.trace.layers[index] = { index, minDepth: depth, maxDepth: depth, tuning, source };
  } else {
    state.trace.layers[index].maxDepth = Math.max(state.trace.layers[index].maxDepth, depth);
  }
}

export function noteDojoIteration(state, depth, score, timeMs, nodes) {
  if (!state.trace) return;
  state.trace.maxDepth = Math.max(state.trace.maxDepth, depth);
  state.trace.iterations.push({ depth, score, timeMs, nodes, rootCandidates: [] });
}

export function noteDojoRootCandidate(state, depth, move, score, meta = {}) {
  if (!state.trace) return;
  const iter = state.trace.iterations.find((it) => it.depth === depth);
  if (!iter) return;
  const list = iter.rootCandidates;
  const existing = list.find((c) => c.move === move);
  const next = {
    move,
    score,
    order: list.length,
    isPv: Boolean(meta.isPv),
    isBest: Boolean(meta.isBest),
    givesCheck: Boolean(meta.givesCheck),
  };

  if (existing) {
    Object.assign(existing, next);
  } else {
    list.push(next);
  }

  list.sort((a, b) => {
    if (b.score !== a.score) return b.score - a.score;
    return a.order - b.order;
  });
  if (list.length > 16) list.length = 16;
}

export function noteDojoSelection(state, depth, score = 0, source = 'search') {
  const trace = state.trace;
  if (!trace) return;
  const index = depthBand(depth);
  trace.selectedSource = source;
  trace.selectedScore = Number.isFinite(score) ? score : 0;
  trace.selectedLayer = {
    index,
    depth,
    temp: Number(state.layerTemps[index] || 0),
  };
}

export function snapshotDojoTrace(state) {
  const trace = state.trace;
  if (!trace) return null;
  return {
    fighter: trace.fighter,
    fullmove: trace.fullmove,
    phase: trace.phase,
    bookMove: trace.bookMove,
    layerTemps: [...trace.layerTemps],
    maxDepth: trace.maxDepth,
    selectedScore: trace.selectedScore,
    selectedSource: trace.selectedSource,
    selectedLayer: trace.selectedLayer ? { ...trace.selectedLayer } : null,
    layers: trace.layers.map((layer) => ({ ...layer })),
    iterations: trace.iterations.map((item) => ({
      ...item,
      rootCandidates: Array.isArray(item.rootCandidates)
        ? item.rootCandidates.map((candidate) => ({ ...candidate }))
        : [],
    })),
  };
}

export function primeTraincars(state, fen, pos, knobs) {
  state.link.fill(0);
  state.layerTemps.fill(0);
  state.squareHeat.fill(0);
  state.pieceHeat.fill(0);
  state.rootStats.fill(0);
  state.lastScores.fill(0);

  const briefing = briefEngine(fen, pos.board, pos.side, pos.phase);
  const patterns = detectPatterns(pos.board, pos.side);
  const cavalry = cavalryAnalysis(pos.board, pos.side, pos.phase);
  const sniper = classifySniperStyle(pos.board, pos.side, pos.phase, patterns.threats, cavalry);
  const counter = getCounterStrategy(sniper.profile);

  state.briefing = briefing;
  state.cavalry = cavalry;
  state.counter = counter;
  state.sniper = sniper;
  state.bookMove = briefing.bookMove || null;

  state.link[BUS.PHASE] = clamp8(pos.phase - 12);
  state.link[BUS.VOLATILITY] = clamp8(sniper.volatility + Math.min(patterns.threats.length * 2, 12));
  state.link[BUS.BOOK_HIT] = briefing.bookMove ? 1 : 0;
  
  // Use busWeights for unEval
  const unEvalWeight = (knobs.busWeights?.unEval || 5) / 5;
  state.link[BUS.UN_EVAL] = clamp8((briefing.evalAdjustment / knobs.unEvalDivisor) * unEvalWeight);
  
  const cavWeight = (knobs.busWeights?.cavalry || 8) / 8;
  state.link[BUS.CAV_CONF] = clamp8((cavalry.confidence || 0) * knobs.cavalryConfidenceScale * cavWeight);
  state.link[BUS.CAV_EV] = clamp8(((cavalry.expectedValue || 0) / knobs.cavalryEvDivisor) * cavWeight);
  
  state.link[BUS.SNIPER_STYLE] = clamp8(sniper.styleCode);
  
  const tempoWeight = (knobs.busWeights?.tempo || 4) / 4;
  state.link[BUS.COUNTER_TEMPO] = clamp8((counter.weights[6] - 1) * knobs.counterScale * tempoWeight);
  
  const safetyWeight = (knobs.busWeights?.safety || 12) / 12;
  state.link[BUS.COUNTER_SAFETY] = clamp8((counter.weights[1] - 1) * knobs.counterScale * safetyWeight);
  state.link[BUS.COUNTER_ACTIVITY] = clamp8((counter.weights[2] - 1) * knobs.counterScale);

  for (let i = 0; i < state.layerTemps.length; i++) {
    const volatilityLift = Math.max(0, state.link[BUS.VOLATILITY]) / (knobs.layerVolatilityDivisor || 4);
    const confidenceLift = Math.max(0, state.link[BUS.CAV_CONF]) / (knobs.layerConfidenceDivisor || 5);
    state.layerTemps[i] = clamp8((knobs.layerTemps[i] || 0) + volatilityLift + confidenceLift);
    state.link[BUS.LAYER0_TEMP + i] = state.layerTemps[i];
  }

  buildSquareHeat(state, briefing, cavalry, sniper, knobs);
  return state;
}

export function computeTimeLimit(state, pos, knobs) {
  const volatility = Math.max(0, state.link[BUS.VOLATILITY]);
  const confidence = Math.max(0, state.link[BUS.CAV_CONF]);
  const endgameLift = pos.phase <= 8 ? knobs.timeEndgame : 0;
  const bookDiscount = state.link[BUS.BOOK_HIT] ? knobs.timeBookDiscount : 0;

  return Math.max(
    knobs.timeFloor,
    Math.min(
      knobs.timeCap,
      Math.round(
        knobs.timeBase +
          volatility * knobs.timeVolatility +
          confidence * knobs.timeConfidence +
          endgameLift -
          bookDiscount
      )
    )
  );
}

export function layerTemperature(state, depth) {
  return state.layerTemps[depthBand(depth)] || 0;
}

export function getSearchTuning(state, depth, knobs) {
  const band = depthBand(depth);
  const temp = state.layerTemps[band] || 0;
  const volatility = Math.max(0, state.link[BUS.VOLATILITY]);
  const confidence = Math.max(0, state.link[BUS.CAV_CONF]);
  const breadth = temp + Math.floor(volatility / 2) + Math.floor(confidence / 3);
  
  // Apply Layer Pruning Multiplier
  const pMult = (knobs.layerPruning && knobs.layerPruning[band] !== undefined) ? knobs.layerPruning[band] : 1.0;

  return {
    temp,
    aspiration: Math.round((knobs.aspirationBase + breadth * knobs.aspirationStep)),
    razor: Math.round(breadth * knobs.razorStep * pMult),
    futility: Math.round(breadth * knobs.futilityStep * pMult),
    lmp: Math.max(0, Math.floor(breadth / knobs.lmpDivisor * pMult)),
    lmr: Math.max(0, Math.floor(breadth / knobs.lmrDivisor * pMult)),
  };
}

export function getEvalBridge(state, ply, knobs) {
  if (ply > knobs.evalBridgePly) return 0;
  
  const band = depthBand(ply);
  const aggression = (knobs.layerAggression && knobs.layerAggression[band] !== undefined) ? knobs.layerAggression[band] : 1.0;

  const unEval = state.link[BUS.UN_EVAL] * (knobs.unEvalWeight || 3);
  const cavEval = state.link[BUS.CAV_EV] * (knobs.cavalryEvalWeight || 4);
  const safety = state.link[BUS.COUNTER_SAFETY] * (knobs.counterSafetyWeight || 2);
  const activity = state.link[BUS.COUNTER_ACTIVITY] * (knobs.counterActivityWeight || 2);
  const temp = (state.layerTemps[band] || 0) * (knobs.layerEvalWeight || 1);
  
  const base = Math.round((unEval + cavEval + safety + activity + temp) / knobs.evalBridgeDivisor);
  return Math.round(base * aggression);
}

export function getMoveOrderingBonus(state, move, pos, ply, knobs) {
  const piece = pos.board[move.from];
  const idx = (move.from << 6) | move.to;
  
  const spyWeight = (knobs.busWeights?.spy || 10) / 10;
  const squareBonus = state.squareHeat[move.to] * knobs.squareHeatWeight * spyWeight;
  
  const pieceBonus = piece ? state.pieceHeat[piece.type] * knobs.pieceHeatWeight : 0;
  const statBonus = Math.trunc(state.rootStats[idx] / knobs.rootStatDivisor);
  const tempBonus = Math.trunc(layerTemperature(state, ply + 2) * knobs.tempMoveWeight);
  
  const cavWeight = (knobs.busWeights?.cavalry || 8) / 8;
  const cavBonus = move.captured ? state.link[BUS.CAV_CONF] * knobs.captureWeight * cavWeight : 0;
  
  return Math.round(squareBonus + pieceBonus + statBonus + tempBonus + cavBonus);
}

export function noteRootScore(state, move, score, depth, knobs) {
  const idx = (move.from << 6) | move.to;
  const delta = clamp16(score / knobs.rootScoreDivisor + depth * knobs.rootDepthWeight);
  state.rootStats[idx] = clamp16(state.rootStats[idx] + delta);
  state.lastScores[idx] = clamp16(score);
}
