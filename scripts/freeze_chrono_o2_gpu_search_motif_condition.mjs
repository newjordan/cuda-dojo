#!/usr/bin/env node
import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';

const SCHEMA_VERSION = 'dojo.frozen_chrono_o2_gpu_search_motif_condition.v1';

function usage() {
  return `Usage: node scripts/freeze_chrono_o2_gpu_search_motif_condition.mjs --patterns <gpu_search_patterns.json> [--motif-id <id>] [--out <condition.json>]

Freeze one GPU-search motif as a candidate heldout-gate condition. This only
records the selected motif, hashes, and required next evidence; it does not run
chess, search, training, or promotion.
`;
}

function parseArgs(argv) {
  const args = {
    patterns: null,
    motifId: null,
    out: null,
    conditionSource: '/home/frosty40/cuda_dojo/LOGIC_RAY_FRONTIER_RUNTIME_RECEIPT_2026-05-06.md',
  };
  for (let i = 0; i < argv.length; i += 1) {
    const token = argv[i];
    if (token === '--help' || token === '-h') {
      console.log(usage());
      process.exit(0);
    } else if (token === '--patterns') args.patterns = argv[++i];
    else if (token === '--motif-id') args.motifId = argv[++i];
    else if (token === '--out') args.out = argv[++i];
    else if (token === '--condition-source') args.conditionSource = argv[++i];
    else throw new Error(`unknown argument: ${token}\n${usage()}`);
  }
  if (!args.patterns) throw new Error(`missing --patterns\n${usage()}`);
  return args;
}

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, 'utf8'));
}

function sha256File(filePath) {
  return crypto.createHash('sha256').update(fs.readFileSync(filePath)).digest('hex');
}

function sha256Text(text) {
  return crypto.createHash('sha256').update(String(text)).digest('hex');
}

function defaultOutPath(patternsPath, motif) {
  const parsed = path.parse(patternsPath);
  const stem = parsed.base.replace(/\.json$/, '');
  return path.join(parsed.dir, `${stem}.${motif.id}.frozen_condition.json`);
}

function main() {
  const args = parseArgs(process.argv.slice(2));
  const patternsPath = path.resolve(args.patterns);
  const patterns = readJson(patternsPath);
  const candidates = Array.isArray(patterns.topGlobalPatterns) ? patterns.topGlobalPatterns : [];
  const motif = args.motifId
    ? candidates.find((item) => item.id === args.motifId)
    : candidates[0];
  if (!motif) throw new Error(`no motif found${args.motifId ? ` for --motif-id ${args.motifId}` : ''}`);
  const conditionKey = {
    sourcePatternArtifactSha256: sha256File(patternsPath),
    motifId: motif.id,
    motifTags: motif.tags,
    gpuSearchDepth: patterns.gpuSearchLabel?.depth ?? null,
    metric: 'root-fold top-k agreement with fixed-depth refcuda GPU search bestmove',
  };
  const output = {
    schemaVersion: SCHEMA_VERSION,
    generatedAt: new Date().toISOString(),
    conditionId: `chrono_o2_gpu_search_motif_${motif.id}`,
    conditionHash: sha256Text(JSON.stringify(conditionKey)),
    sourceCondition: {
      receipt: path.resolve(args.conditionSource),
      patternArtifact: patternsPath,
      patternArtifactSha256: conditionKey.sourcePatternArtifactSha256,
      patternSchemaVersion: patterns.schemaVersion || null,
      sourceBridgePath: patterns.sources?.bridgePath || null,
      sourceBridgeSha256: patterns.sources?.bridgeSha256 || null,
      sourceChronoPath: patterns.sources?.chronoPath || null,
      sourceChronoSha256: patterns.sources?.chronoSha256 || null,
      sourceOmnifoldPath: patterns.sources?.omnifoldPath || null,
      sourceOmnifoldSha256: patterns.sources?.omnifoldSha256 || null,
      libRefCudaPath: patterns.sources?.libRefCudaPath || null,
      libRefCudaSha256: patterns.sources?.libRefCudaSha256 || null,
      gpuSearchDepth: patterns.gpuSearchLabel?.depth ?? null,
      gpuSearchMovetimeMs: patterns.gpuSearchLabel?.movetimeMs ?? null,
      gpuSearchOkRoots: patterns.gpuSearchLabel?.searchOkRoots ?? null,
    },
    frozenMotif: {
      id: motif.id,
      tags: motif.tags,
      rowCount: motif.rowCount,
      rootCount: motif.rootCount,
      gpuBestRows: motif.gpuBestRows,
      gpuBestRootCount: motif.gpuBestRootCount,
      gpuBestRate: motif.gpuBestRate,
      examples: motif.examples || [],
    },
    observedEvidence: {
      baseline: patterns.baseline?.frontierRank || null,
      motifMetrics: motif.metrics || null,
      motifDeltaVsFrontierRank: motif.deltaVsFrontierRank || null,
      crossValidation: patterns.crossValidation ? {
        baseline: patterns.crossValidation.baseline,
        candidate: patterns.crossValidation.candidate,
        deltaVsFrontierRank: patterns.crossValidation.deltaVsFrontierRank,
        observedLift: patterns.crossValidation.observedLift,
      } : null,
    },
    promotionPolicy: {
      status: 'frozen_candidate_not_promoted',
      reason: 'motif showed post-hoc root-fold GPU-search lift; runtime promotion still requires heldout GPU gate or fight evidence under this frozen condition',
      blockers: [
        'heldout_gpu_gate_not_rerun_with_frozen_motif_condition',
        'accepted_useful_injection_lift_not_measured_for_frozen_condition',
        'trained_omnifold_delta_not_measured',
      ],
      requiredNextEvidence: [
        'run a heldout GPU gate using this exact conditionHash and motif tags',
        'measure accepted useful injection lift and GPU-search/fight stability',
        'resolve selected OmniFold family off-manifold audit before runtime promotion',
      ],
    },
  };
  const outPath = path.resolve(args.out || defaultOutPath(patternsPath, motif));
  fs.mkdirSync(path.dirname(outPath), { recursive: true });
  fs.writeFileSync(outPath, `${JSON.stringify(output, null, 2)}\n`);
  console.log(JSON.stringify({
    ok: true,
    output: outPath,
    conditionId: output.conditionId,
    conditionHash: output.conditionHash,
    motifId: motif.id,
    tags: motif.tags,
    status: output.promotionPolicy.status,
  }, null, 2));
}

try {
  main();
} catch (error) {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
}
