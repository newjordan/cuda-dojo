#!/usr/bin/env node
import fs from 'node:fs';
import path from 'node:path';

const SCHEMA_VERSION = 'dojo.chrono_independent_source_preflight.v1';

function usage() {
  return `Usage: node scripts/preflight_chrono_independent_source.mjs --bridge <bridge.json> [--out <preflight.json>]

Inspect a logicRayFrontier bridge artifact for independent temporal evidence
needed by a chrono sidecar. FEN clocks are reported, but they do not count as a
trajectory join key by themselves.
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
    }
    if (token === '--bridge') args.bridge = argv[++i];
    else if (token === '--out') args.out = argv[++i];
    else if (token === '--condition-source') args.conditionSource = argv[++i];
    else throw new Error(`unknown argument: ${token}\n${usage()}`);
  }
  if (!args.bridge) throw new Error(`missing --bridge\n${usage()}`);
  return args;
}

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, 'utf8'));
}

function defaultOutPath(bridgePath) {
  const parsed = path.parse(bridgePath);
  const stem = parsed.base.replace(/\.pzrg_frostmatrix_bridge\.json$/, '').replace(/\.json$/, '');
  return path.join(parsed.dir, `${stem}.chrono_independent_source_preflight.json`);
}

function parseFenClock(fen) {
  const parts = String(fen || '').trim().split(/\s+/);
  return {
    parseOk: parts.length >= 6,
    sideToMove: parts[1] || null,
    castling: parts[2] || null,
    enPassant: parts[3] || null,
    halfmove: Number.isFinite(Number(parts[4])) ? Number(parts[4]) : null,
    fullmove: Number.isFinite(Number(parts[5])) ? Number(parts[5]) : null,
  };
}

function firstPresent(obj, keys) {
  if (!obj || typeof obj !== 'object') return null;
  for (const key of keys) {
    const value = obj[key];
    if (value !== undefined && value !== null && value !== '') return value;
  }
  return null;
}

function rowTemporalFields(row) {
  const frontier = row.logicRayFrontier || row.pzrgCandidate?.logicRayFrontier || {};
  const frontierLegacy = frontier.legacy || {};
  const legacy = frontierLegacy.gpuForgePosition || {};
  const provenance = frontier.provenance || {};
  const pzrgEvidence = row.pzrgCandidate?.pzrg_4d_evidence || {};
  const sourceTemporal = frontier.sourceTemporal
    || frontierLegacy.sourceTemporal
    || legacy.sourceTemporal
    || pzrgEvidence.sourceTemporal
    || {};
  const fen = frontier.rootFen || row.rootFen || legacy.fen || '';
  const clock = parseFenClock(fen);
  const sourceGameId = firstPresent(sourceTemporal, ['sourceTrajectoryId', 'source_trajectory_id', 'sourceSequenceId', 'source_sequence_id', 'gameId', 'game_id', 'sourceGameId', 'source_game_id'])
    ?? firstPresent(frontier, ['gameId', 'game_id', 'sourceGameId', 'source_game_id', 'sourceSequenceId', 'source_sequence_id'])
    ?? firstPresent(legacy, ['gameId', 'game_id', 'sourceGameId', 'source_game_id'])
    ?? firstPresent(pzrgEvidence, ['gameId', 'game_id', 'sourceGameId', 'source_game_id']);
  const sourcePly = firstPresent(sourceTemporal, ['sourcePly', 'source_ply', 'plyNumber', 'ply_number', 'sequencePly', 'sequence_ply'])
    ?? firstPresent(frontier, ['plyNumber', 'ply_number', 'sourcePly', 'source_ply'])
    ?? firstPresent(legacy, ['plyNumber', 'ply_number', 'sourcePly', 'source_ply'])
    ?? firstPresent(pzrgEvidence, ['plyNumber', 'ply_number', 'sourcePly', 'source_ply']);
  const sourceIndex = firstPresent(sourceTemporal, ['sourceIndex', 'source_index', 'corpusIndex', 'corpus_index'])
    ?? firstPresent(frontier, ['sourceIndex', 'source_index', 'corpusIndex', 'corpus_index'])
    ?? firstPresent(legacy, ['sourceIndex', 'source_index', 'corpusIndex', 'corpus_index'])
    ?? firstPresent(provenance, ['sourceIndex', 'source_index', 'corpusIndex', 'corpus_index']);
  const previousHash = firstPresent(sourceTemporal, ['previousRootHash', 'previous_root_hash', 'previousFenHash', 'previous_fen_hash'])
    ?? firstPresent(frontier, ['previousRootHash', 'previous_root_hash', 'previousFenHash', 'previous_fen_hash']);
  const nextHash = firstPresent(sourceTemporal, ['nextRootHash', 'next_root_hash', 'nextFenHash', 'next_fen_hash'])
    ?? firstPresent(frontier, ['nextRootHash', 'next_root_hash', 'nextFenHash', 'next_fen_hash']);
  return {
    bridgeId: row.bridgeId,
    rootId: frontier.rootId || row.rootId || null,
    hash: row.logicRayFrontierHash || null,
    fen,
    fenClock: clock,
    hasFenClock: Boolean(clock.parseOk && clock.fullmove !== null && clock.halfmove !== null),
    sourceGameId,
    sourcePly,
    sourceIndex,
    previousHash,
    nextHash,
    hasTrajectoryJoinKey: Boolean(sourceGameId !== null && sourcePly !== null),
    hasNeighborLink: Boolean(previousHash || nextHash),
    hasSourceTemporal: Boolean(sourceTemporal && Object.keys(sourceTemporal).length),
    temporalEvidenceClass: sourceTemporal.temporalEvidenceClass || null,
    sourceOrderConfidence: sourceTemporal.sourceOrderConfidence || null,
    transitionVerified: sourceTemporal.transitionVerified ?? null,
    command: provenance.command || null,
  };
}

function countBy(rows, getter) {
  const counts = new Map();
  for (const row of rows) {
    const key = getter(row);
    counts.set(String(key), (counts.get(String(key)) || 0) + 1);
  }
  return [...counts.entries()]
    .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
    .map(([key, count]) => ({ key, count }));
}

function blockingRoots(rows) {
  const byRoot = new Map();
  for (const row of rows) {
    const reasons = [];
    if (!row.hasFenClock) reasons.push('fen_clock_missing');
    if (!row.hasTrajectoryJoinKey) reasons.push('trajectory_join_key_missing');
    if (!row.hasNeighborLink) reasons.push('neighbor_link_missing');
    if (!reasons.length) continue;
    const key = row.rootId || row.bridgeId || row.fen || '<unknown>';
    if (!byRoot.has(key)) {
      byRoot.set(key, {
        rootId: row.rootId,
        bridgeId: row.bridgeId,
        fen: row.fen,
        reasons,
        sourceGameId: row.sourceGameId,
        sourcePly: row.sourcePly,
        sourceIndex: row.sourceIndex,
        temporalEvidenceClass: row.temporalEvidenceClass,
        sourceOrderConfidence: row.sourceOrderConfidence,
        transitionVerified: row.transitionVerified,
        hasFenClock: row.hasFenClock,
        hasTrajectoryJoinKey: row.hasTrajectoryJoinKey,
        hasNeighborLink: row.hasNeighborLink,
      });
    }
  }
  return [...byRoot.values()];
}

function main() {
  const args = parseArgs(process.argv.slice(2));
  const bridgePath = path.resolve(args.bridge);
  const bridge = readJson(bridgePath);
  const rows = Array.isArray(bridge.rows) ? bridge.rows : [];
  const temporalRows = rows.map(rowTemporalFields);
  const uniqueRootIds = new Set(temporalRows.map((row) => row.rootId).filter(Boolean)).size;
  const rootsWithFenClock = new Set(temporalRows.filter((row) => row.hasFenClock).map((row) => row.rootId)).size;
  const rootsWithTrajectoryJoinKey = new Set(temporalRows.filter((row) => row.hasTrajectoryJoinKey).map((row) => row.rootId)).size;
  const rootsWithNeighborLink = new Set(temporalRows.filter((row) => row.hasNeighborLink).map((row) => row.rootId)).size;
  const rootsWithSourceTemporal = new Set(temporalRows.filter((row) => row.hasSourceTemporal).map((row) => row.rootId)).size;
  const fullmoveCounts = countBy(temporalRows, (row) => row.fenClock.fullmove ?? 'missing').slice(0, 24);
  const halfmoveCounts = countBy(temporalRows, (row) => row.fenClock.halfmove ?? 'missing').slice(0, 24);
  const temporalEvidenceClassCounts = countBy(temporalRows, (row) => row.temporalEvidenceClass || 'missing').slice(0, 24);
  const sourceOrderConfidenceCounts = countBy(temporalRows, (row) => row.sourceOrderConfidence || 'missing').slice(0, 24);
  const transitionVerifiedCounts = countBy(temporalRows, (row) => row.transitionVerified === true ? 'true' : (row.transitionVerified === false ? 'false' : 'missing')).slice(0, 24);
  const commandSamples = [...new Set(temporalRows.map((row) => JSON.stringify(row.command)).filter(Boolean))].slice(0, 4)
    .map((raw) => JSON.parse(raw));

  const checks = [
    {
      id: 'fen_clock_present',
      ok: rootsWithFenClock === uniqueRootIds && uniqueRootIds > 0,
      evidence: { rootsWithFenClock, uniqueRootIds },
      note: 'FEN halfmove/fullmove exists, but this alone is only a clock hint.',
    },
    {
      id: 'trajectory_join_key_present',
      ok: rootsWithTrajectoryJoinKey === uniqueRootIds && uniqueRootIds > 0,
      evidence: { rootsWithTrajectoryJoinKey, uniqueRootIds },
      note: 'Required for independent chrono: game_id/source trajectory id plus ply number.',
    },
    {
      id: 'neighbor_link_present',
      ok: rootsWithNeighborLink === uniqueRootIds && uniqueRootIds > 0,
      evidence: { rootsWithNeighborLink, uniqueRootIds },
      note: 'Previous/next position links are needed for temporal deltas without using frontier rank.',
    },
  ];
  const blockers = checks.filter((check) => !check.ok).map((check) => check.id);
  const verifiedTrajectoryRoots = new Set(temporalRows.filter((row) => row.transitionVerified === true).map((row) => row.rootId)).size;
  const rootBlockers = blockingRoots(temporalRows);
  const status = blockers.length
    ? 'blocked_missing_independent_temporal_source'
    : (verifiedTrajectoryRoots === uniqueRootIds
      ? 'ready_for_independent_chrono_derivation'
      : 'ready_for_source_order_chrono_derivation_unverified_transition');
  const output = {
    schemaVersion: SCHEMA_VERSION,
    generatedAt: new Date().toISOString(),
    condition: {
      source: path.resolve(args.conditionSource),
      runLabel: 'chrono_independent_source_preflight_on_documented_offset64_bridge',
      changedFields: 'none; read-only source inspection over existing bridge artifact',
    },
    sources: {
      bridgePath,
      bridgeSchemaVersion: bridge.schemaVersion || null,
      bridgeRows: rows.length,
      uniqueRootIds,
    },
    checks,
    summary: {
      status,
      blockers,
      rootsWithSourceTemporal,
      rootsWithFenClock,
      rootsWithTrajectoryJoinKey,
      rootsWithNeighborLink,
      verifiedTrajectoryRoots,
      fullmoveCounts,
      halfmoveCounts,
      temporalEvidenceClassCounts,
      sourceOrderConfidenceCounts,
      transitionVerifiedCounts,
      commandSamples,
    },
    blockingRoots: rootBlockers,
    requiredForNextChronoDerivation: [
      'sourceGameId or sourceTrajectoryId per root',
      'sourcePly or sourcePlyNumber per root',
      'previous/next root linkage or contiguous same-game sequence',
      'sidecar fields derived from those temporal joins, not from frontier rank/probability/risk/lockIn/utility',
    ],
    promotionPolicy: {
      status: 'not_promoted',
      reason: 'preflight only; current bridge artifact must carry independent temporal source fields before chrono v2 sidecar derivation',
    },
    sampleRows: temporalRows.slice(0, 8),
  };
  const outPath = path.resolve(args.out || defaultOutPath(bridgePath));
  fs.mkdirSync(path.dirname(outPath), { recursive: true });
  fs.writeFileSync(outPath, `${JSON.stringify(output, null, 2)}\n`);
  console.log(JSON.stringify({
    ok: true,
    output: outPath,
    status,
    blockers,
    bridgeRows: rows.length,
    uniqueRootIds,
    rootsWithFenClock,
    rootsWithTrajectoryJoinKey,
    rootsWithNeighborLink,
  }, null, 2));
  process.exit(blockers.length ? 2 : 0);
}

try {
  main();
} catch (error) {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
}
