#!/usr/bin/env node
import crypto from 'node:crypto';
import { execSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';

const BRIDGE_SCHEMA = 'dojo.logic_ray_frontier.pzrg_frostmatrix_bridge.v1';

function usage() {
  return `Usage: node scripts/bridge_logic_ray_frontier_to_pzrg_frostmatrix.mjs --input <frontier-bundle.json> [--out <bridge.json>] [--limit <n>]

Build a PZRG/FrostMatrix consumer bundle from schema-valid logicRayFrontier rows.
The full frontier row is embedded in each consumer payload and checked with a
stable SHA-256 round-trip hash so downstream code can use flat projections
without losing the original CUDA/Rayfront evidence row.
`;
}

function parseArgs(argv) {
  const args = {
    input: null,
    out: null,
    limit: 0,
  };
  for (let i = 0; i < argv.length; i += 1) {
    const token = argv[i];
    if (token === '--help' || token === '-h') {
      console.log(usage());
      process.exit(0);
    }
    if (token === '--input' || token === '--in') {
      args.input = argv[++i];
    } else if (token === '--out') {
      args.out = argv[++i];
    } else if (token === '--limit') {
      args.limit = Number(argv[++i] || 0);
    } else {
      throw new Error(`unknown argument: ${token}\n${usage()}`);
    }
  }
  if (!args.input) {
    throw new Error(`missing --input\n${usage()}`);
  }
  if (!Number.isFinite(args.limit) || args.limit < 0) {
    throw new Error('--limit must be a non-negative number');
  }
  return args;
}

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, 'utf8'));
}

function deepClone(value) {
  return JSON.parse(JSON.stringify(value));
}

function stableJson(value) {
  if (value === null || typeof value !== 'object') {
    return JSON.stringify(value);
  }
  if (Array.isArray(value)) {
    return `[${value.map(stableJson).join(',')}]`;
  }
  return `{${Object.keys(value).sort().map((key) => `${JSON.stringify(key)}:${stableJson(value[key])}`).join(',')}}`;
}

function sha256Stable(value) {
  return crypto.createHash('sha256').update(stableJson(value)).digest('hex');
}

function asRows(document) {
  if (Array.isArray(document)) {
    return document.filter((row) => row && typeof row === 'object' && !Array.isArray(row));
  }
  if (document && Array.isArray(document.rows)) {
    return document.rows.filter((row) => row && typeof row === 'object' && !Array.isArray(row));
  }
  if (document && document.schemaVersion === 'dojo.logic_ray_frontier.v1') {
    return [document];
  }
  throw new Error('input must be a logicRayFrontier row, row array, or bundle with rows[]');
}

function firstPresent(...values) {
  for (const value of values) {
    if (value !== undefined && value !== null && value !== '') {
      return value;
    }
  }
  return null;
}

function asList(value) {
  return Array.isArray(value) ? value.filter((item) => item !== null && item !== undefined && item !== '') : [];
}

function sanitize(text) {
  return String(text || 'row').replace(/[^a-zA-Z0-9_.:-]+/g, '_').replace(/^_+|_+$/g, '') || 'row';
}

function clamp01(value, fallback = 0) {
  const number = Number(value);
  if (!Number.isFinite(number)) {
    return fallback;
  }
  return Math.max(0, Math.min(1, number));
}

function normalizePzrg4d(row) {
  const raw = row.pzrg4d || {};
  const normalized = {
    schema: firstPresent(raw.schema, 'pzrg_4d_label_v1'),
    geometry: firstPresent(raw.geometry, raw.relationGeometry),
    pressure: firstPresent(raw.pressure, raw.pressureSemantics),
    chessExpression: firstPresent(raw.chessExpression, raw.expression, raw.chess_expression),
    actionGradient: firstPresent(raw.actionGradient, raw.gradient, raw.action_gradient),
    relationScope: firstPresent(raw.relationScope, raw.relation_scope, raw.scope),
    confidence: firstPresent(raw.confidence, 'medium'),
    evidence: raw.evidence && typeof raw.evidence === 'object' ? raw.evidence : {},
  };
  const missing = Object.entries(normalized)
    .filter(([key, value]) => key !== 'evidence' && value === null)
    .map(([key]) => key);
  if (missing.length) {
    throw new Error(`row ${row.rootId || row.id || '<unknown>'} is missing pzrg4d axes: ${missing.join(', ')}`);
  }
  return normalized;
}

function candidateTags(row, rowHash) {
  const labels = asList(row.labels).map(String);
  const tags = [
    ...labels,
    `logic_ray_frontier.sha256:${rowHash}`,
    `logic_ray_frontier.root_id:${row.rootId}`,
    `logic_ray_frontier.move:${row.move}`,
    `logic_ray_frontier.rank:${row.rank}`,
    `logic_ray_frontier.utility:${row.utility}`,
    'logic_ray_frontier.full_row:preserved',
  ];
  return [...new Set(tags.filter(Boolean))];
}

function buildPzrgCandidate(row, axes, rowHash, bridgeId) {
  const move = String(row.move || '0000');
  const line = asList(row.path).map(String);
  const useful = clamp01(row.gate?.acceptedInjectionScore ?? row.utility, 0);
  return {
    id: bridgeId,
    hardneg_source: 'cuda_dojo_logic_ray_frontier_v1',
    hardneg_class: `logic_ray_frontier_${axes.geometry}__${axes.pressure}`,
    hardneg_classes: [`logic_ray_frontier_${axes.geometry}__${axes.pressure}`],
    hardneg_reason: 'CUDA Dojo logicRayFrontier row preserved through PZRG bridge',
    fen: row.rootFen,
    chosen_move: move,
    move,
    accepted_expected_move: move,
    accepted_best_path: line.length ? line : [move],
    candidate_tags: candidateTags(row, rowHash),
    pzrg_4d_schema: axes.schema,
    pzrg_4d_geometry: axes.geometry,
    pzrg_4d_pressure: axes.pressure,
    pzrg_4d_chess_expression: axes.chessExpression,
    pzrg_4d_action_gradient: axes.actionGradient,
    pzrg_4d_relation_scope: axes.relationScope,
    pzrg_4d_confidence: axes.confidence,
    pzrg_4d_geometries: [axes.geometry],
    pzrg_4d_pressures: [axes.pressure],
    pzrg_4d_chess_expressions: [axes.chessExpression],
    pzrg_4d_action_gradients: [axes.actionGradient],
    pzrg_4d_relation_scopes: [axes.relationScope],
    pzrg_4d_evidence: {
      source: 'cuda_dojo_logic_ray_frontier_v1',
      logicRayFrontierRowHash: rowHash,
      rayfrontFamily: row.rayfrontFamily,
      gate: row.gate || {},
      pzrg4dEvidence: axes.evidence,
    },
    translation_space: 'pzrg_to_chess_meta_v1',
    translation_chess_concept: 'logic_ray_frontier_preserved_cuda_root_distribution',
    translation_connection: 'PZRG consumes flat axes while retaining the full logicRayFrontier row payload',
    translation_gradient: axes.actionGradient,
    time_truth: {
      bestPath: line.length ? line : [move],
      bestPathProbability: clamp01(row.pathProbability, 0),
      maxAttackPressure: clamp01(row.lockIn, 0),
      maxEscapeRisk: clamp01(row.risk, 0),
    },
    injection_relevance: {
      useful_score: useful,
      useful_injection_score: useful,
      promotion_gate_approved: Boolean(row.gate?.acceptedUsefulInjection),
      accepted_useful_injection: Boolean(row.gate?.acceptedUsefulInjection),
    },
    logicRayFrontierHash: rowHash,
    logicRayFrontier: deepClone(row),
  };
}

function buildFrostMatrixSideInput(row, axes, rowHash, bridgeId) {
  return {
    schemaVersion: 'dojo.frostmatrix.side_input.logic_ray_frontier.v1',
    rowId: bridgeId,
    source: 'cuda_dojo_logic_ray_frontier_v1',
    objective: 'accepted_useful_injections_per_wall_clock_hour_per_gpu_hour',
    tuple: {
      relationGeometry: axes.geometry,
      pressureSemantics: axes.pressure,
      chessExpression: axes.chessExpression,
      actionGradient: axes.actionGradient,
      relationScope: axes.relationScope,
      confidence: axes.confidence,
      evidence: [
        `logic_ray_frontier.sha256:${rowHash}`,
        `logic_ray_frontier.root_id:${row.rootId}`,
        `logic_ray_frontier.move:${row.move}`,
      ],
    },
    additiveLayer: {
      name: 'logic_ray_frontier',
      rayfrontFamily: row.rayfrontFamily,
      survivalBucket: row.survivalBucket,
      conversionBucket: row.conversionBucket,
      gateStatus: row.gate?.status || 'unknown',
      acceptedUsefulInjection: Boolean(row.gate?.acceptedUsefulInjection),
    },
    omnifoldFamily: deepClone(row.omnifoldFamily || {}),
    chronometricSidecar: deepClone(row.chrono || {}),
    metrics: {
      rank: row.rank,
      scoreCp: row.scoreCp,
      scoreGapFromBestCp: row.scoreGapFromBestCp,
      pathProbability: row.pathProbability,
      risk: row.risk,
      lockIn: row.lockIn,
      utility: row.utility,
      rayfrontMetrics: deepClone(row.rayfrontMetrics || {}),
    },
    logicRayFrontierHash: rowHash,
    logicRayFrontier: deepClone(row),
  };
}

function buildBridgeRow(row, index) {
  const sourceRow = deepClone(row);
  const rowHash = sha256Stable(sourceRow);
  const axes = normalizePzrg4d(sourceRow);
  const bridgeId = sanitize(`${sourceRow.rootId || `root_${index + 1}`}.${sourceRow.rank || index + 1}.${sourceRow.move || '0000'}`);
  const pzrgCandidate = buildPzrgCandidate(sourceRow, axes, rowHash, bridgeId);
  const frostMatrixSideInput = buildFrostMatrixSideInput(sourceRow, axes, rowHash, bridgeId);
  const proof = {
    algorithm: 'sha256(stable-json)',
    sourceRowHash: rowHash,
    bridgeRowHash: sha256Stable(sourceRow),
    pzrgEmbeddedRowHash: sha256Stable(pzrgCandidate.logicRayFrontier),
    frostMatrixEmbeddedRowHash: sha256Stable(frostMatrixSideInput.logicRayFrontier),
    exactPreservation: false,
    pzrgAxesPreserved: false,
    frostMatrixAxesPreserved: false,
  };
  proof.exactPreservation = (
    proof.sourceRowHash === proof.bridgeRowHash
    && proof.sourceRowHash === proof.pzrgEmbeddedRowHash
    && proof.sourceRowHash === proof.frostMatrixEmbeddedRowHash
  );
  proof.pzrgAxesPreserved = (
    pzrgCandidate.pzrg_4d_geometry === axes.geometry
    && pzrgCandidate.pzrg_4d_pressure === axes.pressure
    && pzrgCandidate.pzrg_4d_chess_expression === axes.chessExpression
    && pzrgCandidate.pzrg_4d_action_gradient === axes.actionGradient
    && pzrgCandidate.pzrg_4d_relation_scope === axes.relationScope
  );
  proof.frostMatrixAxesPreserved = (
    frostMatrixSideInput.tuple.relationGeometry === axes.geometry
    && frostMatrixSideInput.tuple.pressureSemantics === axes.pressure
    && frostMatrixSideInput.tuple.chessExpression === axes.chessExpression
    && frostMatrixSideInput.tuple.actionGradient === axes.actionGradient
    && frostMatrixSideInput.tuple.relationScope === axes.relationScope
  );
  return {
    bridgeId,
    rootId: sourceRow.rootId,
    rootFen: sourceRow.rootFen,
    move: sourceRow.move,
    rank: sourceRow.rank,
    logicRayFrontierHash: rowHash,
    logicRayFrontier: sourceRow,
    pzrgCandidate,
    frostMatrixSideInput,
    roundTripProof: proof,
  };
}

function defaultOutPath(inputPath) {
  const parsed = path.parse(inputPath);
  const stem = parsed.base.replace(/\.logic_ray_frontier\.json$/, '').replace(/\.json$/, '');
  return path.join(parsed.dir, `${stem}.pzrg_frostmatrix_bridge.json`);
}

function gitProvenance() {
  try {
    const commit = execSync('git rev-parse HEAD', { cwd: path.resolve(import.meta.dirname, '..'), encoding: 'utf8', timeout: 5000 }).trim();
    const branch = execSync('git rev-parse --abbrev-ref HEAD', { cwd: path.resolve(import.meta.dirname, '..'), encoding: 'utf8', timeout: 5000 }).trim();
    const dirty = execSync('git status --porcelain', { cwd: path.resolve(import.meta.dirname, '..'), encoding: 'utf8', timeout: 5000 }).trim();
    return { commit, branch, dirty: dirty.length > 0 };
  } catch {
    return { commit: null, branch: null, dirty: null };
  }
}

function main() {
  const args = parseArgs(process.argv.slice(2));
  const inputPath = path.resolve(args.input);
  const document = readJson(inputPath);
  const allRows = asRows(document);
  const rows = args.limit > 0 ? allRows.slice(0, args.limit) : allRows;
  const provenance = gitProvenance();
  const bridgeRows = rows.map((row, index) => buildBridgeRow(row, index));
  const failures = bridgeRows
    .filter((row) => !row.roundTripProof.exactPreservation || !row.roundTripProof.pzrgAxesPreserved || !row.roundTripProof.frostMatrixAxesPreserved)
    .map((row) => ({
      bridgeId: row.bridgeId,
      proof: row.roundTripProof,
    }));
  const generatedAt = new Date().toISOString();
  const output = {
    schemaVersion: BRIDGE_SCHEMA,
    generatedAt,
    source: {
      inputPath,
      inputSchemaVersion: document.schemaVersion || null,
      inputRowSchema: document.rowSchema || null,
      inputRows: allRows.length,
      emittedRows: bridgeRows.length,
      limitApplied: args.limit > 0 ? args.limit : null,
      provenance,
    },
    consumers: [
      'pzrg_candidate_jsonl_compatible',
      'frostmatrix_side_input_logic_ray_frontier',
    ],
    preservation: {
      mode: 'full_logicRayFrontier_row_embedded_in_bridge_pzrg_and_frostmatrix_payloads',
      hashAlgorithm: 'sha256(stable-json)',
      exactPreservationCount: bridgeRows.filter((row) => row.roundTripProof.exactPreservation).length,
      pzrgAxesPreservedCount: bridgeRows.filter((row) => row.roundTripProof.pzrgAxesPreserved).length,
      frostMatrixAxesPreservedCount: bridgeRows.filter((row) => row.roundTripProof.frostMatrixAxesPreserved).length,
      failures,
    },
    rowCount: bridgeRows.length,
    rows: bridgeRows,
  };
  const outPath = path.resolve(args.out || defaultOutPath(inputPath));
  fs.mkdirSync(path.dirname(outPath), { recursive: true });
  fs.writeFileSync(outPath, `${JSON.stringify(output, null, 2)}\n`);
  const ok = failures.length === 0;
  console.log(JSON.stringify({
    ok,
    rows: bridgeRows.length,
    sourceRows: allRows.length,
    output: outPath,
    exactPreservationCount: output.preservation.exactPreservationCount,
    pzrgAxesPreservedCount: output.preservation.pzrgAxesPreservedCount,
    frostMatrixAxesPreservedCount: output.preservation.frostMatrixAxesPreservedCount,
    firstRowHash: bridgeRows[0]?.logicRayFrontierHash || null,
    failures: failures.slice(0, 8),
  }, null, 2));
  process.exit(ok ? 0 : 1);
}

try {
  main();
} catch (error) {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
}
