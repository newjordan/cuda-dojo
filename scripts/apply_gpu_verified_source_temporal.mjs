#!/usr/bin/env node
import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';

const SCHEMA_VERSION = 'dojo.logic_ray_frontier.gpu_verified_source_temporal_subset.v1';

function usage() {
  return `Usage: node scripts/apply_gpu_verified_source_temporal.mjs --frontier <source_temporal_frontier.json> --verification <transition_gpu_verify.json> --out <frontier.verified.json> [--mode any|both]

Filter logicRayFrontier rows to roots with GPU-verified source transitions and
upgrade sourceTemporal to transitionVerified=true only for those roots. This
uses the verifier artifact; it does not run chess or infer missing transitions.
`;
}

function parseArgs(argv) {
  const args = {
    frontier: null,
    verification: null,
    out: null,
    mode: 'any',
    conditionSource: '/home/frosty40/cuda_dojo/LOGIC_RAY_FRONTIER_RUNTIME_RECEIPT_2026-05-06.md',
  };
  for (let i = 0; i < argv.length; i += 1) {
    const token = argv[i];
    if (token === '--help' || token === '-h') {
      console.log(usage());
      process.exit(0);
    }
    if (token === '--frontier') args.frontier = argv[++i];
    else if (token === '--verification') args.verification = argv[++i];
    else if (token === '--out') args.out = argv[++i];
    else if (token === '--mode') args.mode = argv[++i];
    else if (token === '--condition-source') args.conditionSource = argv[++i];
    else throw new Error(`unknown argument: ${token}\n${usage()}`);
  }
  if (!args.frontier) throw new Error(`missing --frontier\n${usage()}`);
  if (!args.verification) throw new Error(`missing --verification\n${usage()}`);
  if (!args.out) throw new Error(`missing --out\n${usage()}`);
  if (!['any', 'both'].includes(args.mode)) throw new Error('--mode must be any or both');
  return args;
}

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, 'utf8'));
}

function sha256File(filePath) {
  return crypto.createHash('sha256').update(fs.readFileSync(filePath)).digest('hex');
}

function asRows(document) {
  if (Array.isArray(document)) return document.filter((row) => row && typeof row === 'object');
  if (Array.isArray(document?.rows)) return document.rows.filter((row) => row && typeof row === 'object');
  if (document?.schemaVersion === 'dojo.logic_ray_frontier.v1') return [document];
  throw new Error('frontier input must be a row, row array, or bundle with rows[]');
}

function rowRootId(row) {
  return row.rootId || null;
}

function verificationByRoot(verification, mode) {
  const map = new Map();
  for (const row of Array.isArray(verification.rootTransitionVerification) ? verification.rootTransitionVerification : []) {
    const ok = mode === 'both'
      ? Boolean(row.gpuVerifiedBothNeighbors)
      : Boolean(row.gpuVerifiedAnyNeighbor);
    if (!ok || !row.rootId) continue;
    map.set(row.rootId, row);
  }
  return map;
}

function verifiedConfidence(info) {
  if (info.gpuVerifiedBothNeighbors) return 'gpu_refcuda_verified_both_neighbors';
  if (info.gpuVerifiedPreviousTransition) return 'gpu_refcuda_verified_previous_neighbor';
  if (info.gpuVerifiedNextTransition) return 'gpu_refcuda_verified_next_neighbor';
  return 'gpu_refcuda_verified_neighbor';
}

function updateSourceTemporal(row, info, verificationPath, verificationSha) {
  const original = row.sourceTemporal || {};
  return {
    ...original,
    previousFenHash: info.gpuVerifiedPreviousTransition ? original.previousFenHash : null,
    nextFenHash: info.gpuVerifiedNextTransition ? original.nextFenHash : null,
    temporalEvidenceClass: 'gpu_refcuda_verified_transition_neighbor',
    sourceOrderConfidence: verifiedConfidence(info),
    transitionVerified: true,
    gpuTransitionVerification: {
      schemaVersion: 'dojo.gpu_refcuda_transition_evidence.v1',
      verifierArtifact: verificationPath,
      verifierArtifactSha256: verificationSha,
      previousEdgeStatus: info.previousEdgeStatus,
      nextEdgeStatus: info.nextEdgeStatus,
      gpuVerifiedPreviousTransition: Boolean(info.gpuVerifiedPreviousTransition),
      gpuVerifiedNextTransition: Boolean(info.gpuVerifiedNextTransition),
      gpuVerifiedBothNeighbors: Boolean(info.gpuVerifiedBothNeighbors),
      clockPolicy: 'semantic_position_match_ignores_halfmove_fullmove',
    },
    note: 'Source transition was verified by applying the recorded source move through refcuda GPU legal-move/make-move and comparing the resulting position to the adjacent FEN.',
  };
}

function main() {
  const args = parseArgs(process.argv.slice(2));
  const frontierPath = path.resolve(args.frontier);
  const verificationPath = path.resolve(args.verification);
  const outPath = path.resolve(args.out);
  const frontier = readJson(frontierPath);
  const verification = readJson(verificationPath);
  const verificationSha = sha256File(verificationPath);
  const rootMap = verificationByRoot(verification, args.mode);
  const inputRows = asRows(frontier);
  const retainedRows = [];
  const excludedRootIds = new Set();
  for (const row of inputRows) {
    const rootId = rowRootId(row);
    const info = rootMap.get(rootId);
    if (!info) {
      if (rootId) excludedRootIds.add(rootId);
      continue;
    }
    const labels = new Set(Array.isArray(row.labels) ? row.labels : []);
    labels.add('chrono_source.gpu_refcuda_verified_transition_neighbor');
    labels.add(`chrono_source_confidence.${verifiedConfidence(info)}`);
    retainedRows.push({
      ...row,
      sourceTemporal: updateSourceTemporal(row, info, verificationPath, verificationSha),
      labels: [...labels],
    });
  }
  const retainedRootIds = new Set(retainedRows.map((row) => row.rootId).filter(Boolean));
  const output = {
    ...(Array.isArray(frontier) ? {} : frontier),
    schemaVersion: Array.isArray(frontier) ? 'dojo.logic_ray_frontier.bundle.v1' : (frontier.schemaVersion || 'dojo.logic_ray_frontier.bundle.v1'),
    generatedAt: new Date().toISOString(),
    gpuVerifiedSourceTemporalSubset: {
      schemaVersion: SCHEMA_VERSION,
      generatedAt: new Date().toISOString(),
      condition: {
        source: path.resolve(args.conditionSource),
        runLabel: `gpu_verified_source_temporal_${args.mode}_neighbor_subset`,
        changedFields: 'filtered to roots with GPU-verified source transition neighbors and upgraded sourceTemporal transitionVerified flags',
        labCondition: 'posthoc/subset; verifier artifact only',
      },
      frontierPath,
      frontierSha256: sha256File(frontierPath),
      verificationPath,
      verificationSha256: verificationSha,
      mode: args.mode,
      inputRows: inputRows.length,
      retainedRows: retainedRows.length,
      inputRootCount: new Set(inputRows.map((row) => row.rootId).filter(Boolean)).size,
      retainedRootCount: retainedRootIds.size,
      excludedRootIds: [...excludedRootIds].sort(),
    },
    rows: retainedRows,
  };
  fs.mkdirSync(path.dirname(outPath), { recursive: true });
  fs.writeFileSync(outPath, `${JSON.stringify(output, null, 2)}\n`);
  console.log(JSON.stringify({
    ok: retainedRows.length > 0,
    output: outPath,
    mode: args.mode,
    inputRows: inputRows.length,
    retainedRows: retainedRows.length,
    retainedRootCount: retainedRootIds.size,
    excludedRootCount: excludedRootIds.size,
    promote: false,
  }, null, 2));
  process.exit(retainedRows.length > 0 ? 0 : 2);
}

try {
  main();
} catch (error) {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
}
