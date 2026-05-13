#!/usr/bin/env node
import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';

const ROW_SCHEMA = 'https://theforge.local/schemas/logic_ray_frontier_chrono_sidecar.schema.json';
const BUNDLE_SCHEMA = 'dojo.logic_ray_frontier_chrono_sidecar.bundle.v1';
const PROJECTION_SYSTEM = 'PZRG_CHRONO_O2';
const LAMBDA = 3722 / 2705;
const TAU_C = 1.0;
const T0 = 1.0;
const BETA = 0.0;

function usage() {
  return `Usage: node scripts/build_chrono_o2_frontier_sidecar.mjs --bridge <source_temporal_bridge.json> [--out <sidecar.json>]

Build a sidecar-only PZRG_CHRONO_O2 chronometric overlay using the documented
4-vector event state, log-time internal chronometer, bootstrap 4x4 response
tensors, and orthogonal contortion projector. This reads recorded GPU-derived
artifacts only; it does not generate legal moves, run search, or widen runtime
geometry.
`;
}

function parseArgs(argv) {
  const args = {
    bridge: null,
    out: null,
    conditionSource: '/home/frosty40/cuda_dojo/LOGIC_RAY_FRONTIER_RUNTIME_RECEIPT_2026-05-06.md',
  };
  for (let i = 0; i < argv.length; i += 1) {
    const token = argv[i];
    if (token === '--help' || token === '-h') {
      console.log(usage());
      process.exit(0);
    } else if (token === '--bridge') {
      args.bridge = argv[++i];
    } else if (token === '--out') {
      args.out = argv[++i];
    } else if (token === '--condition-source') {
      args.conditionSource = argv[++i];
    } else {
      throw new Error(`unknown argument: ${token}\n${usage()}`);
    }
  }
  if (!args.bridge) throw new Error(`missing --bridge\n${usage()}`);
  return args;
}

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, 'utf8'));
}

function asNumber(value, fallback = 0) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

function clamp(value, min = 0, max = 1) {
  return Math.max(min, Math.min(max, asNumber(value, 0)));
}

function round(value, digits = 6) {
  const factor = 10 ** digits;
  return Math.round(asNumber(value, 0) * factor) / factor;
}

function normalizeFen(fen) {
  return String(fen || '').trim().replace(/\s+/g, ' ');
}

function completeFen(fen) {
  const parts = normalizeFen(fen).split(/\s+/).filter(Boolean);
  if (parts.length === 2) return `${parts.join(' ')} - - 0 1`;
  if (parts.length === 3) return `${parts.join(' ')} - 0 1`;
  if (parts.length === 4) return `${parts.join(' ')} 0 1`;
  if (parts.length === 5) return `${parts.join(' ')} 1`;
  return parts.join(' ');
}

function sha256Text(text) {
  return crypto.createHash('sha256').update(String(text)).digest('hex');
}

function fenHash(fen) {
  return sha256Text(completeFen(fen));
}

function defaultOutPath(bridgePath) {
  const parsed = path.parse(bridgePath);
  const stem = parsed.base.replace(/\.pzrg_frostmatrix_bridge\.json$/, '').replace(/\.json$/, '');
  return path.join(parsed.dir, `${stem}.chrono_o2_sidecar.json`);
}

function bridgeRows(document) {
  return Array.isArray(document.rows)
    ? document.rows.filter((row) => row && typeof row === 'object')
    : [];
}

function frontierOf(row) {
  return row.logicRayFrontier || row.pzrgCandidate?.logicRayFrontier || {};
}

function sourceTemporalOf(row) {
  const frontier = frontierOf(row);
  return frontier.sourceTemporal || frontier.legacy?.sourceTemporal || frontier.legacy?.gpuForgePosition?.sourceTemporal || null;
}

function parseFenParts(fen) {
  const parts = completeFen(fen).split(/\s+/);
  const board = parts[0] || '';
  const side = parts[1] || 'w';
  const halfmove = asNumber(parts[4], 0);
  const fullmove = asNumber(parts[5], 1);
  let whiteMaterial = 0;
  let blackMaterial = 0;
  let pieceCount = 0;
  const values = { p: 1, n: 3, b: 3, r: 5, q: 9, k: 0 };
  for (const ch of board) {
    const lower = ch.toLowerCase();
    if (!values.hasOwnProperty(lower)) continue;
    pieceCount += 1;
    if (ch === lower) blackMaterial += values[lower];
    else whiteMaterial += values[lower];
  }
  return { board, side, halfmove, fullmove, whiteMaterial, blackMaterial, pieceCount };
}

function boardMapFromFen(fen) {
  const parts = completeFen(fen).split(/\s+/);
  const boardPart = parts[0] || '';
  const board = new Map();
  const ranks = boardPart.split('/');
  for (let row = 0; row < ranks.length; row += 1) {
    let file = 0;
    const rank = 8 - row;
    for (const ch of ranks[row]) {
      if (/\d/.test(ch)) {
        file += Number(ch);
      } else {
        board.set(`${String.fromCharCode('a'.charCodeAt(0) + file)}${rank}`, ch);
        file += 1;
      }
    }
  }
  return board;
}

function pieceName(piece) {
  if (!piece) return 'empty';
  const names = { p: 'pawn', n: 'knight', b: 'bishop', r: 'rook', q: 'queen', k: 'king' };
  return names[String(piece).toLowerCase()] || 'unknown';
}

function pieceColor(piece) {
  if (!piece) return 'none';
  return piece === String(piece).toUpperCase() ? 'white' : 'black';
}

function squareParts(square) {
  const text = String(square || '');
  if (!/^[a-h][1-8]$/.test(text)) return null;
  return {
    file: text.charCodeAt(0) - 'a'.charCodeAt(0),
    rank: Number(text[1]) - 1,
  };
}

function actionFeaturesFromFenMove(fen, move) {
  const text = String(move || '0000');
  if (!/^[a-h][1-8][a-h][1-8][qrbn]?$/.test(text)) {
    return {
      schemaVersion: 'dojo.chrono_o2_action_features.v1',
      source: 'rootFen+frontierMove',
      piece: 'unknown',
      pieceColor: 'none',
      family: 'invalid',
      capturedPiece: 'empty',
      flags: {
        capture: false,
        promotion: false,
        castle: false,
        diagonal: false,
        center4: false,
        center16: false,
        homeDeparture: false,
        forwardPositive: false,
        backwardOrLevel: false,
        sameFile: false,
        sameRank: false,
      },
      scalars: {
        dx: 0,
        dy: 0,
        forward: 0,
        distance: 0,
        targetFile: 0,
        targetRank: 0,
      },
      actionVector4: [0, 0, 0, 0],
    };
  }
  const parts = parseFenParts(fen);
  const board = boardMapFromFen(fen);
  const from = text.slice(0, 2);
  const to = text.slice(2, 4);
  const fromParts = squareParts(from);
  const toParts = squareParts(to);
  const piece = board.get(from) || null;
  const captured = board.get(to) || null;
  const dx = toParts.file - fromParts.file;
  const dy = toParts.rank - fromParts.rank;
  const sideSign = parts.side === 'b' ? -1 : 1;
  const forward = sideSign * dy;
  const promotion = text.length === 5;
  const castle = pieceName(piece) === 'king' && Math.abs(dx) === 2;
  const center4 = ['d4', 'e4', 'd5', 'e5'].includes(to);
  const center16 = toParts.file >= 2 && toParts.file <= 5 && toParts.rank >= 2 && toParts.rank <= 5;
  const homeDeparture = (
    (parts.side === 'w' && ['1', '2'].includes(from[1]))
    || (parts.side === 'b' && ['7', '8'].includes(from[1]))
  );
  let family = 'quiet';
  if (castle) family = 'castle';
  else if (promotion) family = 'promotion';
  else if (captured) family = 'capture';
  const capture = Boolean(captured);
  const diagonal = Math.abs(dx) === Math.abs(dy) && dx !== 0;
  const forwardPositive = forward > 0;
  return {
    schemaVersion: 'dojo.chrono_o2_action_features.v1',
    source: 'rootFen+frontierMove',
    piece: pieceName(piece),
    pieceColor: pieceColor(piece),
    family,
    capturedPiece: pieceName(captured),
    flags: {
      capture,
      promotion,
      castle,
      diagonal,
      center4,
      center16,
      homeDeparture,
      forwardPositive,
      backwardOrLevel: !forwardPositive,
      sameFile: dx === 0,
      sameRank: dy === 0,
    },
    scalars: {
      dx: round(dx),
      dy: round(dy),
      forward: round(forward),
      distance: round(Math.sqrt(dx ** 2 + dy ** 2)),
      targetFile: round(toParts.file / 7),
      targetRank: round(toParts.rank / 7),
    },
    actionVector4: [
      capture || promotion ? 1 : 0,
      diagonal ? 1 : 0,
      forwardPositive ? 1 : -1,
      center16 ? 1 : 0,
    ],
  };
}

function pressureCode(frontier) {
  const pressure = String(frontier.pzrg4d?.pressure || '');
  if (pressure.includes('danger')) return 1.0;
  if (pressure.includes('conversion')) return 0.75;
  if (pressure.includes('disagreement')) return 0.55;
  if (pressure.includes('agreement')) return 0.25;
  return 0.4;
}

function eventZFromFen(fen, sourceTemporal, frontier = {}) {
  const parts = parseFenParts(fen);
  const sourcePly = asNumber(sourceTemporal?.sourcePly, (Math.max(1, parts.fullmove) - 1) * 2 + (parts.side === 'b' ? 1 : 0));
  const tau = Math.max(0, sourcePly);
  const materialBalance = clamp((parts.whiteMaterial - parts.blackMaterial) / 39, -1, 1);
  const sideSign = parts.side === 'w' ? 1 : -1;
  const pieceDensity = clamp(parts.pieceCount / 32, 0, 1);
  const pressure = Object.keys(frontier).length ? pressureCode(frontier) : 0.4;
  return [
    round(tau),
    round(materialBalance),
    round(sideSign * pieceDensity),
    round(pressure),
  ];
}

function phaseFromFen(fen) {
  const { fullmove } = parseFenParts(fen);
  if (!Number.isFinite(fullmove) || fullmove <= 0) return 'root_branch';
  if (fullmove <= 10) return 'opening';
  if (fullmove <= 35) return 'middlegame';
  return 'endgame';
}

function sourceRecordsFromArtifact(filePath) {
  if (!filePath || !fs.existsSync(filePath)) return new Map();
  const text = fs.readFileSync(filePath, 'utf8');
  const records = new Map();
  const add = (fen, index) => {
    const complete = completeFen(fen);
    if (!complete.includes('/')) return;
    records.set(fenHash(complete), { fen: complete, index });
  };
  if (filePath.endsWith('.jsonl')) {
    let index = 0;
    for (const line of text.split('\n')) {
      if (!line.trim()) continue;
      try {
        const row = JSON.parse(line);
        if (row.fen) add(row.fen, index);
      } catch {
        // ignore bad source-artifact lines
      }
      index += 1;
    }
    return records;
  }
  try {
    const json = JSON.parse(text);
    let index = 0;
    for (const key of ['trainingCases', 'disagreements', 'rows']) {
      for (const row of Array.isArray(json[key]) ? json[key] : []) {
        if (row?.fen || row?.rootFen) add(row.fen || row.rootFen, index);
        index += 1;
      }
    }
  } catch {
    // Non-JSON source artifacts are not used for neighbor features.
  }
  return records;
}

function sourceRecordCache() {
  const cache = new Map();
  return function getRecord(filePath, hash) {
    if (!filePath || !hash) return null;
    if (!cache.has(filePath)) cache.set(filePath, sourceRecordsFromArtifact(filePath));
    return cache.get(filePath).get(hash) || null;
  };
}

function sourceTemporalForNeighbor(currentTemporal, neighborRecord, direction) {
  if (!neighborRecord) return null;
  return {
    ...(currentTemporal || {}),
    sourcePly: Math.max(0, asNumber(currentTemporal?.sourcePly, 0) + direction),
    sourceIndex: neighborRecord.index,
  };
}

function finiteDifferenceZ(frontier, sourceTemporal, getSourceRecord) {
  const currentFen = frontier.rootFen || '';
  const currentZ = eventZFromFen(currentFen, sourceTemporal, frontier);
  const previous = getSourceRecord(sourceTemporal?.sourceArtifact, sourceTemporal?.previousFenHash);
  const next = getSourceRecord(sourceTemporal?.sourceArtifact, sourceTemporal?.nextFenHash);
  const previousTemporal = sourceTemporalForNeighbor(sourceTemporal, previous, -1);
  const nextTemporal = sourceTemporalForNeighbor(sourceTemporal, next, 1);
  const previousZ = previous ? eventZFromFen(previous.fen, previousTemporal, {}) : null;
  const nextZ = next ? eventZFromFen(next.fen, nextTemporal, {}) : null;
  let u;
  let differenceMode;
  if (previousZ && nextZ) {
    u = nextZ.map((value, index) => (value - previousZ[index]) / 2);
    differenceMode = 'central_previous_next_source_artifact';
  } else if (nextZ) {
    u = nextZ.map((value, index) => value - currentZ[index]);
    differenceMode = 'forward_next_source_artifact';
  } else if (previousZ) {
    u = currentZ.map((value, index) => value - previousZ[index]);
    differenceMode = 'backward_previous_source_artifact';
  } else {
    u = [0, 0, 0, 0];
    differenceMode = 'missing_neighbor_zero_velocity_quarantine';
  }
  return {
    z: currentZ.map((value) => round(value)),
    u: u.map((value) => round(value)),
    differenceMode,
    hasNeighbor: Boolean(previousZ || nextZ),
  };
}

function moveToVector(move, side) {
  const text = String(move || '0000');
  if (!/^[a-h][1-8][a-h][1-8][qrbn]?$/.test(text)) return [0, 0, 0, 0];
  const file = (ch) => ch.charCodeAt(0) - 'a'.charCodeAt(0);
  const rank = (ch) => Number(ch) - 1;
  const fx = file(text[0]);
  const fy = rank(text[1]);
  const tx = file(text[2]);
  const ty = rank(text[3]);
  const sideSign = side === 'b' ? -1 : 1;
  const dx = sideSign * (tx - fx) / 7;
  const dy = sideSign * (ty - fy) / 7;
  const distance = Math.sqrt((tx - fx) ** 2 + (ty - fy) ** 2) / Math.sqrt(98);
  const promotion = text.length === 5 ? 1 : 0;
  const raw = [promotion, dx, dy, distance];
  const norm = Math.sqrt(raw.reduce((sum, value) => sum + value * value, 0)) || 1;
  return raw.map((value) => round(value / norm));
}

function dotMetric(a, b) {
  return -a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
}

function add4(a, b) {
  return a.map((value, index) => value + b[index]);
}

function scale4(a, scale) {
  return a.map((value) => value * scale);
}

function normEuclidean(a) {
  return Math.sqrt(a.reduce((sum, value) => sum + value * value, 0));
}

function projectOrthogonal(u, r) {
  const denom = dotMetric(u, u);
  if (Math.abs(denom) < 1e-9) return [0, 0, 0, 0];
  const factor = dotMetric(u, r) / denom;
  return r.map((value, index) => value - factor * u[index]);
}

function thetaFromTau(tau) {
  return ((2 * Math.PI) / Math.log(LAMBDA)) * Math.log((Math.max(0, tau) + TAU_C) / T0) + BETA;
}

function driftBucket(score) {
  if (score <= -0.2) return 'relieving';
  if (score < 0.2) return 'neutral';
  if (score < 0.55) return 'rising';
  return 'unstable';
}

function horizonBucket(uncertainty, sourcePly) {
  if (uncertainty >= 0.72) return 'quarantine';
  if (sourcePly <= 2) return 'immediate';
  if (sourcePly <= 8) return 'near';
  if (sourcePly <= 20) return 'mid';
  return 'far';
}

function rowUsefulSignal(row, frontier) {
  const injection = row.pzrgCandidate?.injection_relevance || {};
  const useful = clamp(
    injection.useful_injection_score
      ?? injection.useful_score
      ?? frontier.gate?.acceptedInjectionScore
      ?? frontier.utility,
    0,
    1,
  );
  return {
    useful,
    selected: Boolean(frontier.gate?.selectedMoveInFrontier),
    accepted: Boolean(injection.accepted_useful_injection || injection.promotion_gate_approved || frontier.gate?.acceptedUsefulInjection),
  };
}

function tensorSummaries() {
  return {
    bootstrapNotLearned: true,
    K0: { trace: 4, frobNorm: 2, maxAbsOffdiag: 0, det: 1 },
    Kc: { trace: 0, frobNorm: 0, maxAbsOffdiag: 0, det: 0 },
    Ks: { trace: 0, frobNorm: 0, maxAbsOffdiag: 0, det: 0 },
  };
}

function promotionReason(sourceOrderProxyRows, missingNeighborRows) {
  if (sourceOrderProxyRows || missingNeighborRows) {
    return 'PZRG_CHRONO_O2 bootstrap sidecar is source-order proxy evidence only; transition verification and comparison/calibration/independence gates are still required';
  }
  return 'PZRG_CHRONO_O2 bootstrap sidecar has GPU-verified temporal neighbors; learned response tensors and comparison/calibration/independence gates are still required';
}

function buildSidecarRow(row, bridgePath, generatedAt, getSourceRecord) {
  const frontier = frontierOf(row);
  const sourceTemporal = sourceTemporalOf(row);
  const rootFen = frontier.rootFen || row.rootFen || '';
  const move = frontier.move || row.move || '0000';
  const fenParts = parseFenParts(rootFen);
  const sourcePly = Math.max(0, asNumber(sourceTemporal?.sourcePly, frontier.rootIndex || 0));
  const tau = sourcePly;
  const theta = thetaFromTau(tau);
  const { z, u, differenceMode, hasNeighbor } = finiteDifferenceZ(frontier, sourceTemporal, getSourceRecord);
  const p = [...u];
  const n = moveToVector(move, fenParts.side);
  const actionFeatures = actionFeaturesFromFenMove(rootFen, move);
  const fExt = [
    0,
    n[1],
    n[2],
    n[3],
  ].map((value) => round(value));
  const speedSq = Math.abs(dotMetric(u, u));
  const projected = projectOrthogonal(u, n);
  const fCont = scale4(projected, speedSq).map((value) => round(value));
  const pNext = add4(add4(p, fExt), fCont).map((value) => round(value));
  const normBefore = round(dotMetric(u, u));
  const normAfter = round(dotMetric(pNext, pNext));
  const normDrift = round(normAfter - normBefore);
  const orthogonalityResidual = round(dotMetric(u, fCont));
  const sourceOrderProxy = sourceTemporal?.transitionVerified !== true;
  const missingNeighbor = !hasNeighbor;
  const uncertaintyScore = round(clamp((sourceOrderProxy ? 0.55 : 0.05) + (missingNeighbor ? 0.35 : 0.0)));
  const contortionScore = round(clamp(normEuclidean(fCont) / 2));
  const relationAlignment = clamp((dotMetric(u, n) + 1) / 2, 0, 1);
  const pressureScore = round(clamp(normEuclidean(fExt) / 1.5, 0, 1));
  const stabilityScore = round(clamp(
    1
      - Math.min(1, Math.abs(normDrift)) * 0.35
      - Math.min(1, Math.abs(orthogonalityResidual)) * 0.45
      - uncertaintyScore * 0.2,
    0,
    1,
  ));
  const signal = rowUsefulSignal(row, frontier);
  return {
    $schema: ROW_SCHEMA,
    schemaVersion: 'dojo.logic_ray_frontier_chrono_sidecar.v1',
    sidecarId: `${row.bridgeId || frontier.rootId}.${frontier.rank || row.rank || 1}.chrono_o2`,
    logicRayFrontierHash: row.logicRayFrontierHash,
    rootId: frontier.rootId || row.rootId,
    rootFen,
    rootIndex: Math.max(0, asNumber(frontier.rootIndex, 0)),
    move,
    rank: Math.max(1, asNumber(frontier.rank || row.rank, 1)),
    tau: round(tau),
    timePhase: {
      phase: phaseFromFen(frontier.rootFen || row.rootFen),
      theta: round(theta),
      thetaSin: round(Math.sin(theta)),
      thetaCos: round(Math.cos(theta)),
      tauSource: 'sourceTemporal.sourcePly',
    },
    pressureDrift: {
      score: pressureScore,
      bucket: driftBucket(pressureScore),
    },
    relationDrift: {
      score: round(relationAlignment),
      bucket: driftBucket(relationAlignment),
      pathProbability: round(relationAlignment),
      rankGradient: round(dotMetric(u, n)),
    },
    pathContortion: {
      score: contortionScore,
      bucket: driftBucket(contortionScore),
    },
    uncertainty: {
      score: uncertaintyScore,
      bucket: driftBucket(uncertaintyScore),
    },
    eventHorizon: {
      bucket: horizonBucket(uncertaintyScore, sourcePly),
      rank: Math.max(1, asNumber(frontier.rank || row.rank, 1)),
      rootWidth: Math.max(1, asNumber(frontier.rayfrontMetrics?.emittedWidth, 1)),
      pathDepth: Array.isArray(frontier.path) ? Math.max(1, frontier.path.length) : 1,
    },
    vectors: {
      z,
      u: u.map((value) => round(value)),
      p: p.map((value) => round(value)),
      n,
      F_ext: fExt,
      F_cont: fCont,
    },
    actionFeatures,
    diagnostics: {
      normBefore,
      normAfter,
      normDrift,
      orthogonalityResidual,
      stabilityScore,
    },
    runtimeChoiceSignal: {
      selectedMoveInFrontier: signal.selected,
      acceptedUsefulInjection: signal.accepted,
      usefulInjectionScore: round(signal.useful),
      promotionEligible: false,
    },
    responseTensors: tensorSummaries(),
    provenance: {
      generatedAt,
      sourceBridgePath: bridgePath,
      sourceOmniFoldManifestPath: null,
      derivation: 'PZRG_CHRONO_O2 bootstrap sidecar: z/u/p from sourceTemporal trajectory features, n from UCI move geometry, actionFeatures from recorded rootFen+move occupancy, theta from log-time phase, K0=I/Kc=Ks=0, F_cont=m*speed_sq(u)*project_orthogonal(u,K(theta)@n); no legal-move generation and no search',
      projectionSystem: PROJECTION_SYSTEM,
      noCoreGeometryWidening: true,
      hostRole: 'sidecar_build_validate_only',
      sourceTemporal: sourceTemporal || null,
      sourceOrderProxy,
      missingNeighbor,
      finiteDifferenceMode: differenceMode,
      phaseClock: {
        lambda: LAMBDA,
        tau_c: TAU_C,
        T0,
        beta: BETA,
      },
      metric: {
        kind: 'minkowski',
        signature: '(-, +, +, +)',
        c_sq: 1.0,
      },
      forceScaling: {
        m: 1.0,
        eps: 1.0,
        xi: 1.0,
      },
      bootstrapNotLearned: true,
    },
  };
}

function main() {
  const args = parseArgs(process.argv.slice(2));
  const bridgePath = path.resolve(args.bridge);
  const bridge = readJson(bridgePath);
  const rows = bridgeRows(bridge);
  const generatedAt = new Date().toISOString();
  const getSourceRecord = sourceRecordCache();
  const sidecarRows = rows.map((row) => buildSidecarRow(row, bridgePath, generatedAt, getSourceRecord));
  const rootIds = new Set(sidecarRows.map((row) => row.rootId));
  const sourceTemporalRows = sidecarRows.filter((row) => row.provenance.sourceTemporal).length;
  const missingNeighborRows = sidecarRows.filter((row) => row.provenance.missingNeighbor).length;
  const sourceOrderProxyRows = sidecarRows.filter((row) => row.provenance.sourceOrderProxy).length;
  const stableRows = sidecarRows.filter((row) => row.diagnostics.stabilityScore >= 0.5).length;
  const meanUncertainty = sidecarRows.length
    ? sidecarRows.reduce((sum, row) => sum + row.uncertainty.score, 0) / sidecarRows.length
    : 0;
  const output = {
    schemaVersion: BUNDLE_SCHEMA,
    generatedAt,
    rowSchema: ROW_SCHEMA,
    source: {
      bridgePath,
      bridgeSchemaVersion: bridge.schemaVersion || null,
    },
    projectionSystem: PROJECTION_SYSTEM,
    phaseClock: {
      lambda: LAMBDA,
      tau_c: TAU_C,
      T0,
      beta: BETA,
    },
    responseTensorPolicy: {
      bootstrapNotLearned: true,
      K0: 'I[4]',
      Kc: '0[4,4]',
      Ks: '0[4,4]',
      graduation: 'learned K0/Kc/Ks full 4x4 tensors only after PZRG_CHRONO_O2 clears gates',
    },
    condition: {
      source: path.resolve(args.conditionSource),
      runLabel: 'pzrg_chrono_o2_bootstrap_sidecar_on_source_temporal_frontier_bridge',
      changedFields: 'new sidecar artifact only; core frontier/runtime geometry unchanged',
    },
    rowCount: sidecarRows.length,
    stats: {
      rootCount: rootIds.size,
      sourceTemporalRows,
      sourceOrderProxyRows,
      missingNeighborRows,
      stableRows,
      stableRate: sidecarRows.length ? stableRows / sidecarRows.length : 0,
      meanUncertainty: round(meanUncertainty),
      eventHorizonCounts: Object.fromEntries(
        [...new Set(sidecarRows.map((row) => row.eventHorizon.bucket))]
          .sort()
          .map((bucket) => [bucket, sidecarRows.filter((row) => row.eventHorizon.bucket === bucket).length]),
      ),
      finiteDifferenceModeCounts: Object.fromEntries(
        [...new Set(sidecarRows.map((row) => row.provenance.finiteDifferenceMode))]
          .sort()
          .map((mode) => [mode, sidecarRows.filter((row) => row.provenance.finiteDifferenceMode === mode).length]),
      ),
    },
    promotionPolicy: {
      status: 'not_promoted',
      reason: promotionReason(sourceOrderProxyRows, missingNeighborRows),
      blockers: [
        ...(sourceOrderProxyRows ? ['source_order_proxy_transitions_unverified'] : []),
        ...(missingNeighborRows ? ['missing_neighbor_rows_present'] : []),
        'bootstrap_not_learned_response_tensors',
      ],
    },
    rows: sidecarRows,
  };
  const outPath = path.resolve(args.out || defaultOutPath(bridgePath));
  fs.mkdirSync(path.dirname(outPath), { recursive: true });
  fs.writeFileSync(outPath, `${JSON.stringify(output, null, 2)}\n`);
  console.log(JSON.stringify({
    ok: true,
    output: outPath,
    projectionSystem: PROJECTION_SYSTEM,
    rows: sidecarRows.length,
    rootCount: rootIds.size,
    sourceTemporalRows,
    sourceOrderProxyRows,
    missingNeighborRows,
    stableRows,
    meanUncertainty: output.stats.meanUncertainty,
    promotionStatus: output.promotionPolicy.status,
    blockers: output.promotionPolicy.blockers,
  }, null, 2));
}

try {
  main();
} catch (error) {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
}
