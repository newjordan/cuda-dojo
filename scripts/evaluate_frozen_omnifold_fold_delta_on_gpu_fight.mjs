#!/usr/bin/env node
import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';

const SCHEMA_VERSION = 'dojo.omnifold_fold_delta_frozen_gpu_fight_eval.v1';

function usage() {
  return `Usage: node scripts/evaluate_frozen_omnifold_fold_delta_on_gpu_fight.mjs --condition <frozen_condition.json> --fight <gpu_fight_rollout_patterns.json> [--out <eval.json>]

Apply a frozen OmniFold fold-delta condition to a GPU fight rollout artifact.
No model retraining and no chess work are performed here.
`;
}

function parseArgs(argv) {
  const args = { condition: null, fight: null, out: null };
  for (let i = 0; i < argv.length; i += 1) {
    const token = argv[i];
    if (token === '--help' || token === '-h') {
      console.log(usage());
      process.exit(0);
    } else if (token === '--condition') {
      args.condition = argv[++i];
    } else if (token === '--fight') {
      args.fight = argv[++i];
    } else if (token === '--out') {
      args.out = argv[++i];
    } else {
      throw new Error(`unknown argument: ${token}\n${usage()}`);
    }
  }
  if (!args.condition) throw new Error(`missing --condition\n${usage()}`);
  if (!args.fight) throw new Error(`missing --fight\n${usage()}`);
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

function round(value, places = 6) {
  const number = Number(value);
  if (!Number.isFinite(number)) return null;
  const scale = 10 ** places;
  return Math.round(number * scale) / scale;
}

function asNumber(value, fallback = 0) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

function normalizeRows(fight) {
  return (Array.isArray(fight.rollouts) ? fight.rollouts : [])
    .filter((row) => Number.isFinite(Number(row.fightScore)))
    .map((row) => ({
      rootId: String(row.rootId),
      bridgeId: row.bridgeId || null,
      hash: row.hash || null,
      move: String(row.move || ''),
      rank: Math.max(1, Math.trunc(asNumber(row.rank, 1))),
      fightScore: asNumber(row.fightScore, 0),
      acceptedUsefulInjection: Boolean(row.acceptedUsefulInjection),
      selectedMoveInFrontier: Boolean(row.selectedMoveInFrontier),
      rolloutStatus: row.rollout?.status || null,
      tags: Array.isArray(row.tags) ? row.tags.map(String).sort() : [],
    }));
}

function groupByRoot(rows) {
  const groups = new Map();
  for (const row of rows) {
    if (!groups.has(row.rootId)) groups.set(row.rootId, []);
    groups.get(row.rootId).push(row);
  }
  return [...groups.entries()]
    .map(([rootId, groupRows]) => [rootId, groupRows.sort((a, b) => a.rank - b.rank || a.move.localeCompare(b.move))])
    .sort(([a], [b]) => a.localeCompare(b));
}

function scoreRow(row, model, spec) {
  const weightByTag = new Map((model.weights || []).map((entry) => [entry.tag, entry.weight]));
  const tagWeights = row.tags
    .map((tag) => asNumber(weightByTag.get(tag), 0))
    .filter((value) => value !== 0)
    .sort((a, b) => Math.abs(b) - Math.abs(a))
    .slice(0, spec.maxAppliedTags);
  const tagMean = tagWeights.length
    ? tagWeights.reduce((sum, value) => sum + value, 0) / Math.sqrt(tagWeights.length)
    : 0;
  return asNumber(model.baseMeanFightScore, 0)
    + asNumber(spec.tagScale, 1) * tagMean
    - asNumber(spec.rankBias, 0) * (row.rank - 1);
}

function publicTop(row, score) {
  return {
    bridgeId: row.bridgeId,
    hash: row.hash,
    move: row.move,
    rank: row.rank,
    score: round(score),
    fightScore: round(row.fightScore),
    acceptedUsefulInjection: row.acceptedUsefulInjection,
    selectedMoveInFrontier: row.selectedMoveInFrontier,
    rolloutStatus: row.rolloutStatus,
  };
}

function evaluate(groups, scorer, name) {
  let scoreSum = 0;
  let positiveTop1 = 0;
  let neutralTop1 = 0;
  let negativeTop1 = 0;
  let acceptedTop1 = 0;
  let selectedTop1 = 0;
  const roots = [];
  for (const [rootId, rows] of groups) {
    const ranked = rows
      .map((row) => ({ row, score: scorer(row) }))
      .sort((a, b) => b.score - a.score || a.row.rank - b.row.rank || a.row.move.localeCompare(b.row.move));
    const top = ranked[0];
    if (!top) continue;
    scoreSum += top.row.fightScore;
    if (top.row.fightScore > 0) positiveTop1 += 1;
    else if (top.row.fightScore < 0) negativeTop1 += 1;
    else neutralTop1 += 1;
    if (top.row.acceptedUsefulInjection) acceptedTop1 += 1;
    if (top.row.selectedMoveInFrontier) selectedTop1 += 1;
    roots.push({
      rootId,
      rowCount: rows.length,
      top1: publicTop(top.row, top.score),
    });
  }
  return {
    name,
    rootCount: roots.length,
    scoreSum: round(scoreSum),
    meanFightScore: roots.length ? round(scoreSum / roots.length) : null,
    positiveTop1,
    neutralTop1,
    negativeTop1,
    acceptedUsefulTop1: acceptedTop1,
    selectedMoveTop1: selectedTop1,
    roots,
  };
}

function metricsPublic(metrics) {
  return {
    rootCount: metrics.rootCount,
    meanFightScore: metrics.meanFightScore,
    positiveTop1: metrics.positiveTop1,
    neutralTop1: metrics.neutralTop1,
    negativeTop1: metrics.negativeTop1,
    acceptedUsefulTop1: metrics.acceptedUsefulTop1,
    selectedMoveTop1: metrics.selectedMoveTop1,
  };
}

function delta(candidate, baseline) {
  return {
    meanFightScore: candidate.meanFightScore != null && baseline.meanFightScore != null
      ? round(candidate.meanFightScore - baseline.meanFightScore)
      : null,
    positiveTop1: candidate.positiveTop1 - baseline.positiveTop1,
    acceptedUsefulTop1: candidate.acceptedUsefulTop1 - baseline.acceptedUsefulTop1,
    selectedMoveTop1: candidate.selectedMoveTop1 - baseline.selectedMoveTop1,
  };
}

function defaultOutPath(fightPath, condition) {
  const parsed = path.parse(fightPath);
  return path.join(parsed.dir, `${parsed.name}.${condition.conditionId || 'frozen_omnifold_fold_delta'}.eval.json`);
}

function main() {
  const args = parseArgs(process.argv.slice(2));
  const conditionPath = path.resolve(args.condition);
  const fightPath = path.resolve(args.fight);
  const condition = readJson(conditionPath);
  const fight = readJson(fightPath);
  const spec = condition.frozenPayload?.candidateSpec;
  const model = condition.frozenPayload?.candidateModel;
  if (!spec || !model) throw new Error('condition is missing frozenPayload.candidateSpec/candidateModel');
  const rows = normalizeRows(fight);
  const groups = groupByRoot(rows);
  const baseline = evaluate(groups, (row) => -row.rank, 'frontier_rank');
  const candidate = evaluate(groups, (row) => scoreRow(row, model, spec), condition.conditionId || spec.id);
  const resultDelta = delta(candidate, baseline);
  const observedLift = (
    resultDelta.meanFightScore != null && resultDelta.meanFightScore > 0
  ) || resultDelta.positiveTop1 > 0 || resultDelta.acceptedUsefulTop1 > 0;
  const output = {
    schemaVersion: SCHEMA_VERSION,
    generatedAt: new Date().toISOString(),
    condition: {
      conditionPath,
      conditionSha256: sha256File(conditionPath),
      conditionId: condition.conditionId || null,
      conditionHash: condition.conditionHash || null,
      runLabel: 'heldout_eval_frozen_omnifold_fold_delta_on_gpu_fight_rollout',
      changedFields: 'none; fixed fold-delta spec/model applied to separate GPU fight labels without retraining',
      metric: condition.condition?.metric || 'root-fold mean forced-candidate GPU self-play fight score',
    },
    sources: {
      fightPath,
      fightSha256: sha256File(fightPath),
      fightSchemaVersion: fight.schemaVersion || null,
      fightRunLabel: fight.condition?.runLabel || null,
      fightGpuWork: fight.gpuFightRolloutLabel?.gpuWork || [],
      fightDepth: fight.gpuFightRolloutLabel?.depth ?? null,
      fightMaxPliesAfterForcedMove: fight.gpuFightRolloutLabel?.maxPliesAfterForcedMove ?? null,
      fightMaxRank: fight.gpuFightRolloutLabel?.maxRank ?? null,
    },
    corpus: {
      rowCount: rows.length,
      rootCount: groups.length,
      acceptedRows: rows.filter((row) => row.acceptedUsefulInjection).length,
      selectedRows: rows.filter((row) => row.selectedMoveInFrontier).length,
      positiveRows: rows.filter((row) => row.fightScore > 0).length,
      neutralRows: rows.filter((row) => row.fightScore === 0).length,
      negativeRows: rows.filter((row) => row.fightScore < 0).length,
    },
    frozenSpec: spec,
    frozenModel: {
      familyId: model.familyId,
      foldFamily: model.foldFamily,
      orderSet: model.orderSet,
      baseMeanFightScore: model.baseMeanFightScore,
      trainRows: model.trainRows,
      eligibleTagCount: model.eligibleTagCount,
      weights: model.weights || [],
    },
    baseline: metricsPublic(baseline),
    candidate: metricsPublic(candidate),
    deltaVsFrontierRank: resultDelta,
    observedLift,
    promotionPolicy: {
      status: 'not_promoted',
      reason: observedLift
        ? 'frozen fold-delta improved this GPU fight label, but promotion still requires accepted-injection gate lift and off-manifold audit resolution'
        : 'frozen fold-delta did not improve the heldout GPU fight label',
      blockers: [
        'not_an_accepted_injection_gate',
        'off_manifold_audit_not_resolved',
        'runtime_selector_not_integrated',
      ],
    },
  };
  const outPath = path.resolve(args.out || defaultOutPath(fightPath, condition));
  writeJson(outPath, output);
  console.log(JSON.stringify({
    ok: true,
    output: outPath,
    conditionHash: output.condition.conditionHash,
    rootCount: output.corpus.rootCount,
    baselineMeanFightScore: output.baseline.meanFightScore,
    candidateMeanFightScore: output.candidate.meanFightScore,
    deltaVsFrontierRank: output.deltaVsFrontierRank,
    observedLift,
    promote: false,
  }, null, 2));
}

try {
  main();
} catch (error) {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
}
