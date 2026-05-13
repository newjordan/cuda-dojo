#!/usr/bin/env node
import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';

const SCHEMA_VERSION = 'dojo.omnifold_fold_delta_frozen_condition.v1';

function usage() {
  return `Usage: node scripts/freeze_omnifold_fold_delta_condition.mjs --deltas <omnifold_fold_deltas.json> [--out <condition.json>]

Freeze the best fixed-artifact OmniFold fold-delta candidate so later GPU fight
or gate labels can evaluate the exact condition without retraining.
`;
}

function parseArgs(argv) {
  const args = { deltas: null, out: null };
  for (let i = 0; i < argv.length; i += 1) {
    const token = argv[i];
    if (token === '--help' || token === '-h') {
      console.log(usage());
      process.exit(0);
    } else if (token === '--deltas') {
      args.deltas = argv[++i];
    } else if (token === '--out') {
      args.out = argv[++i];
    } else {
      throw new Error(`unknown argument: ${token}\n${usage()}`);
    }
  }
  if (!args.deltas) throw new Error(`missing --deltas\n${usage()}`);
  return args;
}

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, 'utf8'));
}

function writeJson(filePath, payload) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, `${JSON.stringify(payload, null, 2)}\n`);
}

function sha256File(filePath) {
  return crypto.createHash('sha256').update(fs.readFileSync(filePath)).digest('hex');
}

function sha256Object(payload) {
  return crypto.createHash('sha256').update(JSON.stringify(payload)).digest('hex');
}

function safeId(text) {
  return String(text || 'unknown').replace(/[^A-Za-z0-9_.:-]+/g, '_');
}

function defaultOutPath(deltasPath, candidate) {
  const parsed = path.parse(deltasPath);
  return path.join(parsed.dir, `${parsed.name}.${safeId(candidate.spec.id)}.frozen_condition.json`);
}

function main() {
  const args = parseArgs(process.argv.slice(2));
  const deltasPath = path.resolve(args.deltas);
  const deltas = readJson(deltasPath);
  const candidate = deltas.fixedArtifact?.bestCandidate;
  if (!candidate?.spec || !candidate?.model) {
    throw new Error('deltas artifact has no fixedArtifact.bestCandidate.spec/model to freeze');
  }
  const frozenPayload = {
    sourceDeltasPath: deltasPath,
    sourceDeltasSha256: sha256File(deltasPath),
    sourceFightPath: deltas.sources?.fightPath || null,
    sourceFightSha256: deltas.sources?.fightSha256 || null,
    sourceOmnifoldPath: deltas.sources?.omnifoldPath || null,
    sourceOmnifoldSha256: deltas.sources?.omnifoldSha256 || null,
    conditionMetric: deltas.condition?.metric || null,
    candidateSpec: candidate.spec,
    candidateModel: candidate.model,
  };
  const conditionHash = sha256Object(frozenPayload);
  const output = {
    schemaVersion: SCHEMA_VERSION,
    generatedAt: new Date().toISOString(),
    conditionId: `omnifold_fold_delta_${safeId(candidate.spec.id)}`,
    conditionHash,
    status: 'frozen_candidate_not_promoted',
    condition: {
      runLabel: 'frozen_omnifold_fold_delta_candidate_from_shallow_gpu_fight_scout',
      source: deltas.condition?.source || null,
      metric: deltas.condition?.metric || null,
      changedFields: 'none; freeze selected fold-delta spec/model before deeper fight labels',
    },
    frozenPayload,
    sourceEvidence: {
      fixedArtifactDeltaVsFrontierRank: candidate.deltaVsFrontierRank || null,
      fixedArtifactMetrics: candidate.metrics || null,
      crossValidatedDeltaVsFrontierRank: deltas.crossValidation?.deltaVsFrontierRank || null,
      crossValidatedObservedLift: Boolean(deltas.crossValidation?.observedLift),
    },
    promotionPolicy: {
      status: 'not_promoted',
      reason: 'frozen fold-delta condition came from shallow GPU fight rollout labels; evaluate on deeper heldout GPU fight/gate labels before promotion',
      requiredNextEvidence: [
        'deeper frozen-condition GPU fight rollout evaluation without retraining',
        'accepted-useful or fight-score lift on the new label condition',
        'off-manifold audit resolution before runtime selection',
      ],
    },
  };
  const outPath = path.resolve(args.out || defaultOutPath(deltasPath, candidate));
  writeJson(outPath, output);
  console.log(JSON.stringify({
    ok: true,
    output: outPath,
    conditionId: output.conditionId,
    conditionHash,
    sourceFixedDelta: output.sourceEvidence.fixedArtifactDeltaVsFrontierRank,
    sourceCrossValidatedDelta: output.sourceEvidence.crossValidatedDeltaVsFrontierRank,
    promote: false,
  }, null, 2));
}

try {
  main();
} catch (error) {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
}
