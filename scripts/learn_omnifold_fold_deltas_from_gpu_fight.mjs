#!/usr/bin/env node
import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';

const SCHEMA_VERSION = 'dojo.omnifold_fold_delta_from_gpu_fight.v1';
const DEFAULT_CONDITION = '/home/frosty40/cuda_dojo/LOGIC_RAY_FRONTIER_RUNTIME_RECEIPT_2026-05-06.md';

function usage() {
  return `Usage: node scripts/learn_omnifold_fold_deltas_from_gpu_fight.mjs --omnifold <manifest.json> --fight <gpu_fight_rollout_patterns.json> [--out <deltas.json>]

Train standard-vs-OmniFold fold-family tag deltas from recorded GPU fight
rollout labels. This is a JSON/feature learner only; chess work remains in
the source fight artifact, which must report refcuda GPU calls.
`;
}

function parseArgs(argv) {
  const args = {
    omnifold: null,
    fight: null,
    out: null,
    folds: 4,
    minTagRows: 3,
    maxTagWeights: 10,
    conditionSource: DEFAULT_CONDITION,
  };
  for (let i = 0; i < argv.length; i += 1) {
    const token = argv[i];
    if (token === '--help' || token === '-h') {
      console.log(usage());
      process.exit(0);
    } else if (token === '--omnifold') {
      args.omnifold = argv[++i];
    } else if (token === '--fight') {
      args.fight = argv[++i];
    } else if (token === '--out') {
      args.out = argv[++i];
    } else if (token === '--folds') {
      args.folds = Number(argv[++i]);
    } else if (token === '--min-tag-rows') {
      args.minTagRows = Number(argv[++i]);
    } else if (token === '--max-tag-weights') {
      args.maxTagWeights = Number(argv[++i]);
    } else if (token === '--condition-source') {
      args.conditionSource = argv[++i];
    } else {
      throw new Error(`unknown argument: ${token}\n${usage()}`);
    }
  }
  if (!args.omnifold) throw new Error(`missing --omnifold\n${usage()}`);
  if (!args.fight) throw new Error(`missing --fight\n${usage()}`);
  if (!Number.isInteger(args.folds) || args.folds < 2) throw new Error('--folds must be an integer >= 2');
  if (!Number.isInteger(args.minTagRows) || args.minTagRows < 1) throw new Error('--min-tag-rows must be an integer >= 1');
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

function requireGpuFightArtifact(fight) {
  const work = fight.gpuFightRolloutLabel?.gpuWork;
  const required = [
    'refc_legal_moves',
    'refc_make_move',
    'refc_is_checkmate',
    'refc_is_stalemate',
    'refc_search_best_move',
  ];
  const missing = required.filter((name) => !Array.isArray(work) || !work.includes(name));
  if (missing.length) {
    throw new Error(`fight artifact is missing required refcuda GPU evidence: ${missing.join(', ')}`);
  }
}

function activeFamilies(omnifold) {
  return (Array.isArray(omnifold.foldFamilies) ? omnifold.foldFamilies : [])
    .filter((family) => family.status === 'active_frontier_attachable')
    .map((family) => ({
      id: family.id,
      foldFamily: family.foldFamily,
      orderSet: Array.isArray(family.orderSet) ? family.orderSet : [],
      activeVariantCount: asNumber(family.activeVariantCount, 0),
      activeVariants: Array.isArray(family.activeVariants) ? family.activeVariants : [],
    }));
}

function allowedTag(family, tag) {
  const maxOrder = Math.max(...family.orderSet, 0);
  if (maxOrder <= 2) {
    return tag.startsWith('pzrg:')
      || tag.startsWith('piece:')
      || tag.startsWith('family:')
      || tag.startsWith('chrono:relation:')
      || tag === 'chrono:stable_score';
  }
  if (maxOrder <= 4) {
    return tag.startsWith('pzrg:')
      || tag.startsWith('piece:')
      || tag.startsWith('family:')
      || tag.startsWith('action:')
      || tag.startsWith('tactical:')
      || tag.startsWith('chrono:relation:')
      || tag.startsWith('chrono:pressure:')
      || tag === 'chrono:stable_score';
  }
  return tag.startsWith('pzrg:')
    || tag.startsWith('piece:')
    || tag.startsWith('family:')
    || tag.startsWith('action:')
    || tag.startsWith('tactical:')
    || tag.startsWith('chrono:relation:')
    || tag.startsWith('chrono:pressure:')
    || tag.startsWith('chrono:contortion:')
    || tag.startsWith('chrono:uncertainty:')
    || tag === 'chrono:stable_score'
    || tag === 'chrono:unstable_score';
}

function normalizeRows(fight) {
  const rollouts = Array.isArray(fight.rollouts) ? fight.rollouts : [];
  return rollouts
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

function rootFolds(rootIds, foldCount) {
  return Array.from({ length: foldCount }, (_, fold) => (
    rootIds.filter((_, index) => index % foldCount === fold)
  )).filter((foldRoots) => foldRoots.length > 0);
}

function filterGroups(groups, rootSet) {
  return groups.filter(([rootId]) => rootSet.has(rootId));
}

function rowsFromGroups(groups) {
  return groups.flatMap(([, rows]) => rows);
}

function trainFamilyModel(family, rows, options) {
  const baseMean = rows.length
    ? rows.reduce((sum, row) => sum + row.fightScore, 0) / rows.length
    : 0;
  const tagStats = new Map();
  for (const row of rows) {
    for (const tag of row.tags) {
      if (!allowedTag(family, tag)) continue;
      if (!tagStats.has(tag)) tagStats.set(tag, { tag, rows: 0, scoreSum: 0 });
      const stat = tagStats.get(tag);
      stat.rows += 1;
      stat.scoreSum += row.fightScore;
    }
  }
  const weights = [...tagStats.values()]
    .filter((stat) => stat.rows >= options.minTagRows)
    .map((stat) => ({
      tag: stat.tag,
      rows: stat.rows,
      meanFightScore: round(stat.scoreSum / stat.rows),
      weight: round((stat.scoreSum / stat.rows) - baseMean),
    }))
    .filter((stat) => stat.weight !== 0)
    .sort((a, b) => Math.abs(b.weight) - Math.abs(a.weight) || b.rows - a.rows || a.tag.localeCompare(b.tag));
  return {
    familyId: family.id,
    foldFamily: family.foldFamily,
    orderSet: family.orderSet,
    baseMeanFightScore: round(baseMean),
    trainRows: rows.length,
    eligibleTagCount: weights.length,
    weights: weights.slice(0, options.maxTagWeights),
  };
}

function scoreRow(row, model, spec) {
  const weightByTag = new Map(model.weights.map((entry) => [entry.tag, entry.weight]));
  const tagWeights = row.tags
    .map((tag) => asNumber(weightByTag.get(tag), 0))
    .filter((value) => value !== 0)
    .sort((a, b) => Math.abs(b) - Math.abs(a))
    .slice(0, spec.maxAppliedTags);
  const tagMean = tagWeights.length
    ? tagWeights.reduce((sum, value) => sum + value, 0) / Math.sqrt(tagWeights.length)
    : 0;
  const rankPenalty = spec.rankBias * (row.rank - 1);
  return asNumber(model.baseMeanFightScore, 0) + spec.tagScale * tagMean - rankPenalty;
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

function objective(metrics) {
  return asNumber(metrics.meanFightScore, -999) * 10000
    + metrics.positiveTop1 * 100
    + metrics.acceptedUsefulTop1 * 10
    + metrics.selectedMoveTop1;
}

function modelSummary(model) {
  return {
    familyId: model.familyId,
    foldFamily: model.foldFamily,
    orderSet: model.orderSet,
    baseMeanFightScore: model.baseMeanFightScore,
    trainRows: model.trainRows,
    eligibleTagCount: model.eligibleTagCount,
    topPositiveWeights: model.weights.filter((entry) => entry.weight > 0).slice(0, 8),
    topNegativeWeights: model.weights.filter((entry) => entry.weight < 0).slice(0, 8),
  };
}

function candidateSpecs(families) {
  const rankBiases = [0, 0.02, 0.05, 0.1];
  const tagScales = [0.5, 1, 2];
  const maxAppliedTags = [3, 5, 8];
  const specs = [];
  for (const family of families) {
    for (const rankBias of rankBiases) {
      for (const tagScale of tagScales) {
        for (const maxTags of maxAppliedTags) {
          specs.push({
            id: `${family.id}.rank${String(rankBias).replace('.', '_')}.scale${String(tagScale).replace('.', '_')}.tags${maxTags}`,
            familyId: family.id,
            foldFamily: family.foldFamily,
            rankBias,
            tagScale,
            maxAppliedTags: maxTags,
          });
        }
      }
    }
  }
  return specs;
}

function trainModels(families, rows, options) {
  return Object.fromEntries(families.map((family) => [
    family.id,
    trainFamilyModel(family, rows, options),
  ]));
}

function evaluateSpec(spec, model, groups) {
  return evaluate(groups, (row) => scoreRow(row, model, spec), spec.id);
}

function pickBestSpec(specs, models, groups, baseline) {
  const scored = specs.map((spec) => {
    const model = models[spec.familyId];
    const metrics = evaluateSpec(spec, model, groups);
    return {
      spec,
      model,
      metrics,
      deltaVsFrontierRank: delta(metrics, baseline),
      objective: objective(metrics),
    };
  });
  scored.sort((a, b) => b.objective - a.objective || a.spec.id.localeCompare(b.spec.id));
  return scored[0] || null;
}

function aggregate(name, metricsList) {
  const rootCount = metricsList.reduce((sum, item) => sum + item.rootCount, 0);
  const scoreSum = metricsList.reduce((sum, item) => sum + asNumber(item.scoreSum, 0), 0);
  return {
    name,
    rootCount,
    scoreSum: round(scoreSum),
    meanFightScore: rootCount ? round(scoreSum / rootCount) : null,
    positiveTop1: metricsList.reduce((sum, item) => sum + item.positiveTop1, 0),
    neutralTop1: metricsList.reduce((sum, item) => sum + item.neutralTop1, 0),
    negativeTop1: metricsList.reduce((sum, item) => sum + item.negativeTop1, 0),
    acceptedUsefulTop1: metricsList.reduce((sum, item) => sum + item.acceptedUsefulTop1, 0),
    selectedMoveTop1: metricsList.reduce((sum, item) => sum + item.selectedMoveTop1, 0),
    roots: metricsList.flatMap((item) => item.roots),
  };
}

function crossValidate(rows, groups, families, specs, options) {
  const rootIds = groups.map(([rootId]) => rootId);
  const folds = rootFolds(rootIds, Math.min(options.folds, rootIds.length));
  const baselineEvals = [];
  const candidateEvals = [];
  const foldReports = [];
  for (let foldIndex = 0; foldIndex < folds.length; foldIndex += 1) {
    const evalSet = new Set(folds[foldIndex]);
    const trainSet = new Set(rootIds.filter((rootId) => !evalSet.has(rootId)));
    const trainGroups = filterGroups(groups, trainSet);
    const evalGroups = filterGroups(groups, evalSet);
    const trainRows = rows.filter((row) => trainSet.has(row.rootId));
    const trainModelsByFamily = trainModels(families, trainRows, options);
    const trainBaseline = evaluate(trainGroups, (row) => -row.rank, 'frontier_rank_train');
    const evalBaseline = evaluate(evalGroups, (row) => -row.rank, 'frontier_rank_eval');
    const selected = pickBestSpec(specs, trainModelsByFamily, trainGroups, trainBaseline);
    const trainCandidate = selected ? selected.metrics : trainBaseline;
    const evalCandidate = selected
      ? evaluateSpec(selected.spec, selected.model, evalGroups)
      : evalBaseline;
    baselineEvals.push(evalBaseline);
    candidateEvals.push(evalCandidate);
    foldReports.push({
      fold: foldIndex,
      trainRootCount: trainGroups.length,
      evalRootCount: evalGroups.length,
      selectedSpec: selected ? selected.spec : null,
      selectedModel: selected ? modelSummary(selected.model) : null,
      trainBaseline: metricsPublic(trainBaseline),
      trainCandidate: metricsPublic(trainCandidate),
      trainDeltaVsFrontierRank: delta(trainCandidate, trainBaseline),
      evalBaseline: metricsPublic(evalBaseline),
      evalCandidate: metricsPublic(evalCandidate),
      evalDeltaVsFrontierRank: delta(evalCandidate, evalBaseline),
    });
  }
  const baseline = aggregate('frontier_rank_cross_validation_baseline', baselineEvals);
  const candidate = aggregate('omnifold_fold_delta_cross_validation', candidateEvals);
  const aggregateDelta = delta(candidate, baseline);
  const observedLift = (
    aggregateDelta.meanFightScore != null && aggregateDelta.meanFightScore > 0
  ) || aggregateDelta.positiveTop1 > 0 || aggregateDelta.acceptedUsefulTop1 > 0;
  return {
    baseline,
    candidate,
    deltaVsFrontierRank: aggregateDelta,
    observedLift,
    folds: foldReports,
  };
}

function defaultOutPath(fightPath) {
  const parsed = path.parse(fightPath);
  return path.join(parsed.dir, `${parsed.name}.omnifold_fold_deltas.json`);
}

function main() {
  const args = parseArgs(process.argv.slice(2));
  const omnifoldPath = path.resolve(args.omnifold);
  const fightPath = path.resolve(args.fight);
  const outPath = path.resolve(args.out || defaultOutPath(fightPath));
  const omnifold = readJson(omnifoldPath);
  const fight = readJson(fightPath);
  requireGpuFightArtifact(fight);
  const fightLabel = fight.gpuFightRolloutLabel || {};
  const sourceConditionLabel = [
    `depth${fightLabel.depth ?? 'unknown'}`,
    `ply${fightLabel.maxPliesAfterForcedMove ?? 'unknown'}`,
    `rank${fightLabel.maxRank ?? 'unknown'}`,
  ].join('_');
  const families = activeFamilies(omnifold);
  const rows = normalizeRows(fight);
  const groups = groupByRoot(rows);
  const options = {
    folds: args.folds,
    minTagRows: args.minTagRows,
    maxTagWeights: args.maxTagWeights,
  };
  const fullModels = trainModels(families, rows, options);
  const specs = candidateSpecs(families);
  const baseline = evaluate(groups, (row) => -row.rank, 'frontier_rank');
  const best = pickBestSpec(specs, fullModels, groups, baseline);
  const crossValidation = crossValidate(rows, groups, families, specs, options);
  const activeFamilySummaries = families.map((family) => ({
    id: family.id,
    foldFamily: family.foldFamily,
    orderSet: family.orderSet,
    activeVariantCount: family.activeVariantCount,
    activeVariantIds: family.activeVariants.map((variant) => variant.id),
  }));
  const output = {
    schemaVersion: SCHEMA_VERSION,
    generatedAt: new Date().toISOString(),
    condition: {
      source: path.resolve(args.conditionSource),
      runLabel: 'scout_new_experiment_omnifold_standard_vs_fold_delta_from_gpu_fight_labels',
      changedFields: 'post-hoc trained OmniFold fold-family tag deltas over recorded GPU fight rollout labels only; no runtime behavior changed',
      labCondition: 'scout/new_experiment/uses_existing_gpu_fight_rollout_label_artifact',
      metric: 'root-fold mean forced-candidate GPU self-play fight score',
    },
    sources: {
      omnifoldPath,
      omnifoldSha256: sha256File(omnifoldPath),
      omnifoldSchemaVersion: omnifold.schemaVersion || null,
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
      activeFoldFamilies: families.length,
      candidateSpecCount: specs.length,
      acceptedRows: rows.filter((row) => row.acceptedUsefulInjection).length,
      selectedRows: rows.filter((row) => row.selectedMoveInFrontier).length,
      positiveRows: rows.filter((row) => row.fightScore > 0).length,
      neutralRows: rows.filter((row) => row.fightScore === 0).length,
      negativeRows: rows.filter((row) => row.fightScore < 0).length,
    },
    omnifoldFamilies: activeFamilySummaries,
    trainingPolicy: {
      algorithm: 'per_family_tag_mean_delta_with_root_fold_model_selection',
      minTagRows: args.minTagRows,
      maxTagWeights: args.maxTagWeights,
      rankBiases: [0, 0.02, 0.05, 0.1],
      tagScales: [0.5, 1, 2],
      maxAppliedTags: [3, 5, 8],
      noRuntimePromotion: true,
    },
    fixedArtifact: {
      baseline: metricsPublic(baseline),
      bestCandidate: best ? {
        spec: best.spec,
        metrics: metricsPublic(best.metrics),
        deltaVsFrontierRank: best.deltaVsFrontierRank,
        model: modelSummary(best.model),
      } : null,
      observedLift: best ? (
        best.deltaVsFrontierRank.meanFightScore > 0
        || best.deltaVsFrontierRank.positiveTop1 > 0
        || best.deltaVsFrontierRank.acceptedUsefulTop1 > 0
      ) : false,
    },
    crossValidation: {
      baseline: metricsPublic(crossValidation.baseline),
      candidate: metricsPublic(crossValidation.candidate),
      deltaVsFrontierRank: crossValidation.deltaVsFrontierRank,
      observedLift: crossValidation.observedLift,
      folds: crossValidation.folds,
    },
    promotionPolicy: {
      status: 'not_promoted',
      reason: crossValidation.observedLift
        ? 'trained fold deltas show scout lift, but the source fight labels are posthoc and require a frozen heldout GPU fight/gate before promotion'
        : 'trained fold deltas did not show root-fold lift over frontier rank on the source GPU fight labels',
      blockers: [
        crossValidation.observedLift
          ? `source_fight_rollout_condition_is_posthoc_scout_${sourceConditionLabel}`
          : `no_root_fold_lift_on_source_fight_rollout_${sourceConditionLabel}`,
        'all_omnifold_rows_still_off_manifold_in_manifest',
        'no_frozen_deeper_heldout_gpu_fight_or_gate_result',
      ],
    },
  };
  writeJson(outPath, output);
  console.log(JSON.stringify({
    ok: true,
    output: outPath,
    rowCount: output.corpus.rowCount,
    rootCount: output.corpus.rootCount,
    activeFoldFamilies: output.corpus.activeFoldFamilies,
    candidateSpecCount: output.corpus.candidateSpecCount,
    fixedArtifactDelta: output.fixedArtifact.bestCandidate?.deltaVsFrontierRank || null,
    crossValidatedDelta: output.crossValidation.deltaVsFrontierRank,
    observedLift: output.crossValidation.observedLift,
    promote: false,
  }, null, 2));
}

try {
  main();
} catch (error) {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
}
