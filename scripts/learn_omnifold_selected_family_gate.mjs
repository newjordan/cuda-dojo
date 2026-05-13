#!/usr/bin/env node
import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';

const SCHEMA_VERSION = 'dojo.omnifold_selected_family_gate_sweep.v1';
const DEFAULT_CONDITION = '/home/frosty40/cuda_dojo/LOGIC_RAY_FRONTIER_RUNTIME_RECEIPT_2026-05-06.md';

function usage() {
  return `Usage: node scripts/learn_omnifold_selected_family_gate.mjs --resolution <selected_family_resolution.json> [--out <gate_sweep.json>]

Sweep selected-family gate candidates over a recorded OmniFold selected-family
resolution artifact. This is JSON evaluation over CUDA-derived rows only; it
does not run chess logic and does not promote runtime selection.
`;
}

function parseArgs(argv) {
  const args = {
    resolution: null,
    out: null,
    folds: 4,
    conditionSource: DEFAULT_CONDITION,
  };
  for (let i = 0; i < argv.length; i += 1) {
    const token = argv[i];
    if (token === '--help' || token === '-h') {
      console.log(usage());
      process.exit(0);
    } else if (token === '--resolution') {
      args.resolution = argv[++i];
    } else if (token === '--out') {
      args.out = argv[++i];
    } else if (token === '--folds') {
      args.folds = Number(argv[++i]);
    } else if (token === '--condition-source') {
      args.conditionSource = argv[++i];
    } else {
      throw new Error(`unknown argument: ${token}\n${usage()}`);
    }
  }
  if (!args.resolution) throw new Error(`missing --resolution\n${usage()}`);
  if (!Number.isInteger(args.folds) || args.folds < 2) throw new Error('--folds must be an integer >= 2');
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

function normalizeRows(resolution) {
  return (Array.isArray(resolution.rowAssignments) ? resolution.rowAssignments : [])
    .filter((row) => row.rootId && row.move)
    .map((row) => {
      const selected = row.resolution?.selected || {};
      const groups = Array.isArray(selected.selectedVariantGroups)
        ? selected.selectedVariantGroups.map(String).sort()
        : [];
      const groupScores = row.resolution?.groupScores || {};
      const hasFightScore = row.fightScore !== null
        && row.fightScore !== undefined
        && Number.isFinite(Number(row.fightScore));
      return {
        rootId: String(row.rootId),
        bridgeId: row.bridgeId || null,
        hash: row.logicRayFrontierHash || null,
        move: String(row.move),
        rank: Math.max(1, Math.trunc(asNumber(row.rank, 1))),
        acceptedUsefulInjection: Boolean(row.acceptedUsefulInjection),
        selectedMoveInFrontier: Boolean(row.selectedMoveInFrontier),
        fightScore: hasFightScore ? asNumber(row.fightScore) : null,
        selectedFamily: row.selectedFamily || null,
        selectedFamilyId: row.selectedFamilyId || null,
        selectedVariantId: row.selectedVariantId || null,
        orderMax: Math.max(...(Array.isArray(selected.orderSet) ? selected.orderSet : [0]).map((value) => asNumber(value, 0))),
        confidence: asNumber(row.resolution?.confidence, 0),
        margin: asNumber(row.resolution?.margin, 0),
        selectedVariantGroups: groups,
        groupScores: Object.fromEntries(Object.entries(groupScores).map(([key, value]) => [key, asNumber(value, 0)])),
      };
    });
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

function publicTop(row, score) {
  return {
    bridgeId: row.bridgeId,
    hash: row.hash,
    move: row.move,
    rank: row.rank,
    score: round(score),
    selectedFamily: row.selectedFamily,
    selectedFamilyId: row.selectedFamilyId,
    selectedVariantId: row.selectedVariantId,
    confidence: round(row.confidence),
    margin: round(row.margin),
    fightScore: row.fightScore == null ? null : round(row.fightScore),
    acceptedUsefulInjection: row.acceptedUsefulInjection,
    selectedMoveInFrontier: row.selectedMoveInFrontier,
  };
}

function evaluate(groups, scorer, name) {
  let scoreSum = 0;
  let fightTop1Rows = 0;
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
    if (Number.isFinite(top.row.fightScore)) {
      fightTop1Rows += 1;
      scoreSum += top.row.fightScore;
      if (top.row.fightScore > 0) positiveTop1 += 1;
      else if (top.row.fightScore < 0) negativeTop1 += 1;
      else neutralTop1 += 1;
    }
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
    fightTop1Rows,
    scoreSum: round(scoreSum),
    meanFightScore: fightTop1Rows ? round(scoreSum / fightTop1Rows) : null,
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
    fightTop1Rows: metrics.fightTop1Rows,
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

function objective(surface, metrics) {
  if (surface === 'accepted_injection') {
    return metrics.acceptedUsefulTop1 * 10000
      + metrics.selectedMoveTop1 * 100
      + asNumber(metrics.meanFightScore, 0);
  }
  return asNumber(metrics.meanFightScore, -999) * 10000
    + metrics.positiveTop1 * 100
    + metrics.acceptedUsefulTop1 * 10
    + metrics.selectedMoveTop1;
}

function distinct(rows, getter) {
  return [...new Set(rows.map(getter).filter(Boolean).map(String))].sort();
}

function candidateSpecs(rows) {
  const familyIds = distinct(rows, (row) => row.selectedFamilyId);
  const variants = distinct(rows, (row) => row.selectedVariantId);
  const groups = distinct(rows.flatMap((row) => row.selectedVariantGroups), (group) => group);
  const orders = distinct(rows, (row) => row.orderMax).map((value) => Number(value)).filter((value) => value > 0);
  const targetSpecs = [
    ...familyIds.map((targetValue) => ({ targetType: 'family_id', targetValue })),
    ...variants.map((targetValue) => ({ targetType: 'variant_id', targetValue })),
    ...groups.map((targetValue) => ({ targetType: 'variant_group', targetValue })),
    ...orders.map((targetValue) => ({ targetType: 'min_order', targetValue })),
  ];
  const minConfidences = [0, 0.75, 0.82, 0.88];
  const minMargins = [0, 0.02, 0.05];
  const bonuses = [0.55, 1.05, 2.25, 4];
  const rankPenalties = [1, 0.25, 0.02];
  const confidenceWeights = [0, 0.25];
  const marginWeights = [0, 0.5];
  const specs = [];
  for (const target of targetSpecs) {
    for (const minConfidence of minConfidences) {
      for (const minMargin of minMargins) {
        for (const bonus of bonuses) {
          for (const rankPenalty of rankPenalties) {
            for (const confidenceWeight of confidenceWeights) {
              for (const marginWeight of marginWeights) {
                specs.push({
                  id: [
                    target.targetType,
                    safeId(target.targetValue),
                    `c${String(minConfidence).replace('.', '_')}`,
                    `m${String(minMargin).replace('.', '_')}`,
                    `b${String(bonus).replace('.', '_')}`,
                    `r${String(rankPenalty).replace('.', '_')}`,
                    `cw${String(confidenceWeight).replace('.', '_')}`,
                    `mw${String(marginWeight).replace('.', '_')}`,
                  ].join('.'),
                  targetType: target.targetType,
                  targetValue: target.targetValue,
                  minConfidence,
                  minMargin,
                  bonus,
                  rankPenalty,
                  confidenceWeight,
                  marginWeight,
                });
              }
            }
          }
        }
      }
    }
  }
  for (const rankPenalty of rankPenalties) {
    for (const confidenceWeight of [0.25, 0.75, 1.5]) {
      for (const marginWeight of [0, 0.5, 1]) {
        specs.push({
          id: `confidence_any.r${String(rankPenalty).replace('.', '_')}.cw${String(confidenceWeight).replace('.', '_')}.mw${String(marginWeight).replace('.', '_')}`,
          targetType: 'any',
          targetValue: 'any',
          minConfidence: 0,
          minMargin: 0,
          bonus: 0,
          rankPenalty,
          confidenceWeight,
          marginWeight,
        });
      }
    }
  }
  return specs;
}

function rowMatches(row, spec) {
  if (row.confidence < spec.minConfidence || row.margin < spec.minMargin) return false;
  if (spec.targetType === 'any') return true;
  if (spec.targetType === 'family_id') return row.selectedFamilyId === spec.targetValue;
  if (spec.targetType === 'variant_id') return row.selectedVariantId === spec.targetValue;
  if (spec.targetType === 'variant_group') return row.selectedVariantGroups.includes(spec.targetValue);
  if (spec.targetType === 'min_order') return row.orderMax >= Number(spec.targetValue);
  return false;
}

function scoreRow(row, spec) {
  return (rowMatches(row, spec) ? spec.bonus : 0)
    + spec.confidenceWeight * row.confidence
    + spec.marginWeight * row.margin
    - spec.rankPenalty * (row.rank - 1);
}

function evaluateSpec(spec, groups) {
  return evaluate(groups, (row) => scoreRow(row, spec), spec.id);
}

function pickBestSpec(surface, specs, groups, baseline) {
  const scored = specs.map((spec) => {
    const metrics = evaluateSpec(spec, groups);
    return {
      spec,
      metrics,
      deltaVsFrontierRank: delta(metrics, baseline),
      objective: objective(surface, metrics),
    };
  });
  scored.sort((a, b) => b.objective - a.objective || a.spec.id.localeCompare(b.spec.id));
  return {
    best: scored[0] || null,
    top: scored.slice(0, 12).map((candidate) => ({
      spec: candidate.spec,
      metrics: metricsPublic(candidate.metrics),
      deltaVsFrontierRank: candidate.deltaVsFrontierRank,
      objective: round(candidate.objective),
    })),
  };
}

function aggregate(name, metricsList) {
  const rootCount = metricsList.reduce((sum, item) => sum + item.rootCount, 0);
  const fightTop1Rows = metricsList.reduce((sum, item) => sum + item.fightTop1Rows, 0);
  const scoreSum = metricsList.reduce((sum, item) => sum + asNumber(item.scoreSum, 0), 0);
  return {
    name,
    rootCount,
    fightTop1Rows,
    scoreSum: round(scoreSum),
    meanFightScore: fightTop1Rows ? round(scoreSum / fightTop1Rows) : null,
    positiveTop1: metricsList.reduce((sum, item) => sum + item.positiveTop1, 0),
    neutralTop1: metricsList.reduce((sum, item) => sum + item.neutralTop1, 0),
    negativeTop1: metricsList.reduce((sum, item) => sum + item.negativeTop1, 0),
    acceptedUsefulTop1: metricsList.reduce((sum, item) => sum + item.acceptedUsefulTop1, 0),
    selectedMoveTop1: metricsList.reduce((sum, item) => sum + item.selectedMoveTop1, 0),
    roots: metricsList.flatMap((item) => item.roots),
  };
}

function observedLift(surface, resultDelta) {
  if (surface === 'accepted_injection') return resultDelta.acceptedUsefulTop1 > 0;
  return (
    (resultDelta.meanFightScore != null && resultDelta.meanFightScore > 0)
    || resultDelta.positiveTop1 > 0
    || resultDelta.acceptedUsefulTop1 > 0
  );
}

function crossValidate(surface, groups, specs, foldCount) {
  const rootIds = groups.map(([rootId]) => rootId);
  const folds = rootFolds(rootIds, Math.min(foldCount, rootIds.length));
  const baselineEvals = [];
  const candidateEvals = [];
  const foldReports = [];
  for (let foldIndex = 0; foldIndex < folds.length; foldIndex += 1) {
    const evalSet = new Set(folds[foldIndex]);
    const trainSet = new Set(rootIds.filter((rootId) => !evalSet.has(rootId)));
    const trainGroups = filterGroups(groups, trainSet);
    const evalGroups = filterGroups(groups, evalSet);
    const trainBaseline = evaluate(trainGroups, (row) => -(row.rank - 1), 'frontier_rank_train');
    const evalBaseline = evaluate(evalGroups, (row) => -(row.rank - 1), 'frontier_rank_eval');
    const selected = pickBestSpec(surface, specs, trainGroups, trainBaseline).best;
    const trainCandidate = selected ? selected.metrics : trainBaseline;
    const evalCandidate = selected ? evaluateSpec(selected.spec, evalGroups) : evalBaseline;
    baselineEvals.push(evalBaseline);
    candidateEvals.push(evalCandidate);
    foldReports.push({
      fold: foldIndex,
      trainRootCount: trainGroups.length,
      evalRootCount: evalGroups.length,
      selectedSpec: selected ? selected.spec : null,
      trainBaseline: metricsPublic(trainBaseline),
      trainCandidate: metricsPublic(trainCandidate),
      trainDeltaVsFrontierRank: selected ? selected.deltaVsFrontierRank : delta(trainCandidate, trainBaseline),
      evalBaseline: metricsPublic(evalBaseline),
      evalCandidate: metricsPublic(evalCandidate),
      evalDeltaVsFrontierRank: delta(evalCandidate, evalBaseline),
    });
  }
  const baseline = aggregate(`${surface}_frontier_rank_cross_validation_baseline`, baselineEvals);
  const candidate = aggregate(`${surface}_selected_family_gate_cross_validation`, candidateEvals);
  const aggregateDelta = delta(candidate, baseline);
  return {
    baseline,
    candidate,
    deltaVsFrontierRank: aggregateDelta,
    observedLift: observedLift(surface, aggregateDelta),
    folds: foldReports,
  };
}

function surfaceReport(surface, rows, specs, foldCount) {
  const groups = groupByRoot(rows);
  const baseline = evaluate(groups, (row) => -(row.rank - 1), 'frontier_rank');
  const picked = pickBestSpec(surface, specs, groups, baseline);
  const crossValidation = crossValidate(surface, groups, specs, foldCount);
  return {
    surface,
    corpus: {
      rowCount: rows.length,
      rootCount: groups.length,
      acceptedRows: rows.filter((row) => row.acceptedUsefulInjection).length,
      selectedRows: rows.filter((row) => row.selectedMoveInFrontier).length,
      fightRows: rows.filter((row) => Number.isFinite(row.fightScore)).length,
      positiveRows: rows.filter((row) => Number.isFinite(row.fightScore) && row.fightScore > 0).length,
      neutralRows: rows.filter((row) => Number.isFinite(row.fightScore) && row.fightScore === 0).length,
      negativeRows: rows.filter((row) => Number.isFinite(row.fightScore) && row.fightScore < 0).length,
    },
    fixedArtifact: {
      baseline: metricsPublic(baseline),
      bestCandidate: picked.best ? {
        spec: picked.best.spec,
        metrics: metricsPublic(picked.best.metrics),
        deltaVsFrontierRank: picked.best.deltaVsFrontierRank,
        objective: round(picked.best.objective),
      } : null,
      topCandidates: picked.top,
      observedLift: picked.best ? observedLift(surface, picked.best.deltaVsFrontierRank) : false,
    },
    crossValidation: {
      baseline: metricsPublic(crossValidation.baseline),
      candidate: metricsPublic(crossValidation.candidate),
      deltaVsFrontierRank: crossValidation.deltaVsFrontierRank,
      observedLift: crossValidation.observedLift,
      folds: crossValidation.folds,
    },
  };
}

function buildFrozenCandidate(acceptedReport, fightReport, resolutionPath, resolutionSha256) {
  const acceptedLift = acceptedReport.crossValidation.observedLift;
  const fightLift = fightReport.crossValidation.observedLift;
  if (!acceptedLift && !fightLift) return null;
  const source = acceptedLift ? acceptedReport : fightReport;
  const best = source.fixedArtifact.bestCandidate;
  const payload = {
    sourceResolutionPath: resolutionPath,
    sourceResolutionSha256: resolutionSha256,
    surface: source.surface,
    candidateSpec: best.spec,
    fixedArtifactDeltaVsFrontierRank: best.deltaVsFrontierRank,
    crossValidatedDeltaVsFrontierRank: source.crossValidation.deltaVsFrontierRank,
  };
  return {
    conditionId: `omnifold_selected_family_gate_${safeId(source.surface)}_${safeId(best.spec.id)}`,
    conditionHash: sha256Object(payload),
    status: 'frozen_candidate_not_promoted',
    frozenPayload: payload,
  };
}

function defaultOutPath(resolutionPath) {
  const parsed = path.parse(resolutionPath);
  return path.join(parsed.dir, `${parsed.name}.selected_family_gate_sweep.json`);
}

function main() {
  const args = parseArgs(process.argv.slice(2));
  const resolutionPath = path.resolve(args.resolution);
  const outPath = path.resolve(args.out || defaultOutPath(resolutionPath));
  const resolution = readJson(resolutionPath);
  const rows = normalizeRows(resolution);
  const fightRows = rows.filter((row) => Number.isFinite(row.fightScore));
  const specs = candidateSpecs(rows);
  const resolutionSha256 = sha256File(resolutionPath);
  const acceptedReport = surfaceReport('accepted_injection', rows, specs, args.folds);
  const fightReport = surfaceReport('gpu_fight', fightRows, specs, args.folds);
  const frozenCandidate = buildFrozenCandidate(acceptedReport, fightReport, resolutionPath, resolutionSha256);
  const output = {
    schemaVersion: SCHEMA_VERSION,
    generatedAt: new Date().toISOString(),
    condition: {
      source: path.resolve(args.conditionSource),
      runLabel: 'scout_new_experiment_omnifold_selected_family_gate_sweep',
      changedFields: 'post-hoc selected-family gate sweep over recorded CUDA-derived selected-family and GPU fight labels only; no runtime behavior changed',
      labCondition: 'scout/new_experiment/uses_existing_selected_family_resolution_artifact',
      metric: 'accepted-useful top1 and root-fold mean forced-candidate GPU self-play fight score',
    },
    sources: {
      resolutionPath,
      resolutionSha256,
      resolutionSchemaVersion: resolution.schemaVersion || null,
      resolutionRunLabel: resolution.condition?.runLabel || null,
      bridgePath: resolution.sources?.bridgePath || null,
      bridgeSha256: resolution.sources?.bridgeSha256 || null,
      chronoPath: resolution.sources?.chronoPath || null,
      chronoSha256: resolution.sources?.chronoSha256 || null,
      omnifoldPath: resolution.sources?.omnifoldPath || null,
      omnifoldSha256: resolution.sources?.omnifoldSha256 || null,
      fightPath: resolution.sources?.fightPath || null,
      fightSha256: resolution.sources?.fightSha256 || null,
      fightRunLabel: resolution.sources?.fightRunLabel || null,
    },
    gatePolicy: {
      algorithm: 'root_fold_selected_family_bonus_gate_sweep',
      hostRole: 'json_feature_projection_only',
      noChessRuntime: true,
      noRuntimePromotion: true,
      candidateSpecCount: specs.length,
      folds: args.folds,
      surfaces: ['accepted_injection', 'gpu_fight'],
    },
    acceptedInjectionGate: acceptedReport,
    gpuFightGate: fightReport,
    frozenCandidate,
    promotionPolicy: {
      status: 'not_promoted',
      reason: frozenCandidate
        ? 'selected-family gate has a frozen scout candidate, but runtime promotion still requires a heldout accepted-injection gate and consumer integration'
        : 'selected-family gate sweep did not find cross-validated accepted-injection or GPU-fight lift over frontier rank',
      blockers: frozenCandidate ? [
        'not_evaluated_on_new_heldout_gate',
        'runtime_selector_not_integrated',
        'accepted_injection_per_gpu_hour_not_proven',
      ] : [
        'no_cross_validated_selected_family_gate_lift',
        'runtime_selector_not_integrated',
        'accepted_injection_per_gpu_hour_not_proven',
      ],
    },
  };
  writeJson(outPath, output);
  console.log(JSON.stringify({
    ok: true,
    output: outPath,
    rowCount: rows.length,
    fightRows: fightRows.length,
    candidateSpecCount: specs.length,
    acceptedCrossValidatedDelta: output.acceptedInjectionGate.crossValidation.deltaVsFrontierRank,
    fightCrossValidatedDelta: output.gpuFightGate.crossValidation.deltaVsFrontierRank,
    frozenCandidate: frozenCandidate ? {
      conditionId: frozenCandidate.conditionId,
      conditionHash: frozenCandidate.conditionHash,
      surface: frozenCandidate.frozenPayload.surface,
    } : null,
    promote: false,
  }, null, 2));
}

try {
  main();
} catch (error) {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
}
