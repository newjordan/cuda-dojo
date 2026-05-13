#!/usr/bin/env node
import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';

const SCHEMA_VERSION = 'dojo.gpu_fight_overtake_selector_mining.v1';
const DEFAULT_CONDITION = '/home/frosty40/cuda_dojo/LOGIC_RAY_FRONTIER_RUNTIME_RECEIPT_2026-05-06.md';

function usage() {
  return `Usage: node scripts/mine_gpu_fight_overtake_selectors.mjs --fight <gpu_fight_rollout_patterns.json> [--out <selectors.json>]

Mine differential Rayfront/OmniFold/chrono/tactical tags for cases where a
recorded GPU fight rollout says a non-rank1 candidate overtakes frontier rank.
This is JSON analysis over recorded refcuda GPU labels only; it does not run
chess, generate moves, or change runtime behavior.
`;
}

function parseArgs(argv) {
  const args = {
    fight: null,
    out: null,
    folds: 4,
    maxSize: 3,
    minRoots: 2,
    minRows: 2,
    minDelta: 0,
    topK: 24,
    conditionSource: DEFAULT_CONDITION,
  };
  for (let i = 0; i < argv.length; i += 1) {
    const token = argv[i];
    if (token === '--help' || token === '-h') {
      console.log(usage());
      process.exit(0);
    } else if (token === '--fight') {
      args.fight = argv[++i];
    } else if (token === '--out') {
      args.out = argv[++i];
    } else if (token === '--folds') {
      args.folds = Number(argv[++i]);
    } else if (token === '--max-size') {
      args.maxSize = Number(argv[++i]);
    } else if (token === '--min-roots') {
      args.minRoots = Number(argv[++i]);
    } else if (token === '--min-rows') {
      args.minRows = Number(argv[++i]);
    } else if (token === '--min-delta') {
      args.minDelta = Number(argv[++i]);
    } else if (token === '--top-k') {
      args.topK = Number(argv[++i]);
    } else if (token === '--condition-source') {
      args.conditionSource = argv[++i];
    } else {
      throw new Error(`unknown argument: ${token}\n${usage()}`);
    }
  }
  if (!args.fight) throw new Error(`missing --fight\n${usage()}`);
  if (!Number.isInteger(args.folds) || args.folds < 2) throw new Error('--folds must be an integer >= 2');
  if (!Number.isInteger(args.maxSize) || args.maxSize < 1) throw new Error('--max-size must be an integer >= 1');
  if (!Number.isInteger(args.minRoots) || args.minRoots < 1) throw new Error('--min-roots must be an integer >= 1');
  if (!Number.isInteger(args.minRows) || args.minRows < 1) throw new Error('--min-rows must be an integer >= 1');
  if (!Number.isFinite(args.minDelta) || args.minDelta < 0) throw new Error('--min-delta must be a number >= 0');
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

function motifId(tags) {
  return `overtake_${crypto.createHash('sha256').update(tags.join('\0')).digest('hex').slice(0, 16)}`;
}

function combinations(values, maxSize) {
  const sorted = [...new Set(values)].sort();
  const out = [];
  function walk(start, combo) {
    if (combo.length > 0) out.push([...combo]);
    if (combo.length >= maxSize) return;
    for (let idx = start; idx < sorted.length; idx += 1) {
      combo.push(sorted[idx]);
      walk(idx + 1, combo);
      combo.pop();
    }
  }
  walk(0, []);
  return out;
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

function rootFolds(rootIds, foldCount) {
  return Array.from({ length: Math.min(foldCount, rootIds.length) }, (_, fold) => (
    rootIds.filter((_, index) => index % Math.min(foldCount, rootIds.length) === fold)
  )).filter((roots) => roots.length > 0);
}

function filterGroups(groups, rootSet) {
  return groups.filter(([rootId]) => rootSet.has(rootId));
}

function rowPublic(row, score = null) {
  return {
    bridgeId: row.bridgeId,
    hash: row.hash,
    move: row.move,
    rank: row.rank,
    selectorScore: score === null ? null : round(score),
    fightScore: round(row.fightScore),
    acceptedUsefulInjection: row.acceptedUsefulInjection,
    selectedMoveInFrontier: row.selectedMoveInFrontier,
    rolloutStatus: row.rolloutStatus,
  };
}

function rootSummary(rootId, rows, minDelta) {
  const baseline = rows.find((row) => row.rank === 1) || rows[0];
  const best = [...rows].sort((a, b) => b.fightScore - a.fightScore || a.rank - b.rank || a.move.localeCompare(b.move))[0];
  const delta = round(best.fightScore - baseline.fightScore);
  const overtaken = Boolean(best && baseline && best.rank !== baseline.rank && delta > minDelta);
  return {
    rootId,
    rowCount: rows.length,
    baseline,
    best,
    delta,
    overtaken,
  };
}

function overtakePairs(groups, minDelta) {
  return groups
    .map(([rootId, rows]) => rootSummary(rootId, rows, minDelta))
    .filter((root) => root.baseline && root.best)
    .map((root) => ({
      rootId: root.rootId,
      rowCount: root.rowCount,
      deltaVsRank1: root.delta,
      overtaken: root.overtaken,
      rank1: rowPublic(root.baseline),
      bestGpuFight: rowPublic(root.best),
      differentialTags: root.overtaken
        ? root.best.tags.filter((tag) => !new Set(root.baseline.tags).has(tag))
        : [],
    }));
}

function matches(row, motif) {
  const tagSet = new Set(row.tags);
  return motif.tags.every((tag) => tagSet.has(tag));
}

function chooseByMotif(rows, motif) {
  const baseline = rows.find((row) => row.rank === 1) || rows[0];
  if (!baseline) return null;
  if (matches(baseline, motif)) return { row: baseline, switched: false, reason: 'rank1_matches_motif' };
  const matching = rows
    .filter((row) => row.rank > 1 && matches(row, motif))
    .sort((a, b) => a.rank - b.rank || b.fightScore - a.fightScore || a.move.localeCompare(b.move))[0];
  if (!matching) return { row: baseline, switched: false, reason: 'no_non_rank1_match' };
  return { row: matching, switched: true, reason: 'non_rank1_differential_match' };
}

function evaluate(groups, chooser, name, minDelta) {
  let scoreSum = 0;
  let positiveTop1 = 0;
  let neutralTop1 = 0;
  let negativeTop1 = 0;
  let acceptedTop1 = 0;
  let selectedTop1 = 0;
  let changedRootCount = 0;
  let capturedOvertakes = 0;
  let missedOvertakes = 0;
  let falseSwitches = 0;
  const roots = [];

  for (const [rootId, rows] of groups) {
    const summary = rootSummary(rootId, rows, minDelta);
    const choice = chooser(rows, summary);
    if (!choice?.row) continue;
    const row = choice.row;
    const switched = Boolean(choice.switched);
    scoreSum += row.fightScore;
    if (row.fightScore > 0) positiveTop1 += 1;
    else if (row.fightScore < 0) negativeTop1 += 1;
    else neutralTop1 += 1;
    if (row.acceptedUsefulInjection) acceptedTop1 += 1;
    if (row.selectedMoveInFrontier) selectedTop1 += 1;
    if (switched) changedRootCount += 1;
    if (summary.overtaken && row.hash === summary.best.hash) capturedOvertakes += 1;
    if (summary.overtaken && row.hash !== summary.best.hash) missedOvertakes += 1;
    if (switched && (!summary.overtaken || row.hash !== summary.best.hash)) falseSwitches += 1;
    roots.push({
      rootId,
      rowCount: rows.length,
      overtaken: summary.overtaken,
      deltaVsRank1: summary.delta,
      choiceReason: choice.reason,
      selected: rowPublic(row, choice.score ?? null),
      rank1: rowPublic(summary.baseline),
      bestGpuFight: rowPublic(summary.best),
    });
  }
  const rootCount = roots.length;
  return {
    name,
    rootCount,
    scoreSum: round(scoreSum),
    meanFightScore: rootCount ? round(scoreSum / rootCount) : null,
    positiveTop1,
    neutralTop1,
    negativeTop1,
    acceptedUsefulTop1: acceptedTop1,
    selectedMoveTop1: selectedTop1,
    changedRootCount,
    capturedOvertakes,
    missedOvertakes,
    falseSwitches,
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
    changedRootCount: metrics.changedRootCount,
    capturedOvertakes: metrics.capturedOvertakes,
    missedOvertakes: metrics.missedOvertakes,
    falseSwitches: metrics.falseSwitches,
  };
}

function metricDelta(candidate, baseline) {
  return {
    meanFightScore: candidate.meanFightScore !== null && baseline.meanFightScore !== null
      ? round(candidate.meanFightScore - baseline.meanFightScore)
      : null,
    positiveTop1: candidate.positiveTop1 - baseline.positiveTop1,
    acceptedUsefulTop1: candidate.acceptedUsefulTop1 - baseline.acceptedUsefulTop1,
    selectedMoveTop1: candidate.selectedMoveTop1 - baseline.selectedMoveTop1,
    changedRootCount: candidate.changedRootCount - baseline.changedRootCount,
    capturedOvertakes: candidate.capturedOvertakes - baseline.capturedOvertakes,
    falseSwitches: candidate.falseSwitches - baseline.falseSwitches,
  };
}

function generateMotifs(groups, args) {
  const stats = new Map();
  for (const [rootId, rows] of groups) {
    const summary = rootSummary(rootId, rows, args.minDelta);
    if (!summary.overtaken) continue;
    const baselineTags = new Set(summary.baseline.tags);
    const differential = summary.best.tags.filter((tag) => !baselineTags.has(tag));
    for (const tags of combinations(differential, args.maxSize)) {
      const key = tags.join('\0');
      if (!stats.has(key)) {
        stats.set(key, {
          id: motifId(tags),
          tags,
          rowCount: 0,
          rootIds: new Set(),
          deltaSum: 0,
          examples: [],
        });
      }
      const stat = stats.get(key);
      stat.rowCount += 1;
      stat.rootIds.add(rootId);
      stat.deltaSum += summary.delta;
      if (stat.examples.length < 8) {
        stat.examples.push({
          rootId,
          deltaVsRank1: summary.delta,
          rank1: rowPublic(summary.baseline),
          bestGpuFight: rowPublic(summary.best),
        });
      }
    }
  }
  return [...stats.values()]
    .filter((stat) => stat.rowCount >= args.minRows && stat.rootIds.size >= args.minRoots)
    .map((stat) => ({
      id: stat.id,
      tags: stat.tags,
      size: stat.tags.length,
      rowCount: stat.rowCount,
      rootCount: stat.rootIds.size,
      meanOvertakeDelta: round(stat.deltaSum / stat.rowCount),
      examples: stat.examples,
    }));
}

function objective(metrics, delta) {
  return asNumber(delta.meanFightScore, -999) * 10000
    + delta.positiveTop1 * 250
    + delta.acceptedUsefulTop1 * 100
    + delta.selectedMoveTop1 * 50
    + metrics.capturedOvertakes * 25
    - metrics.falseSwitches * 250
    + metrics.changedRootCount;
}

function rankMotifs(motifs, groups, baseline, args) {
  return motifs
    .map((motif) => {
      const metrics = evaluate(
        groups,
        (rows) => {
          const choice = chooseByMotif(rows, motif);
          return choice ? { ...choice, score: choice.switched ? 1 : 0 } : null;
        },
        motif.id,
        args.minDelta,
      );
      const delta = metricDelta(metrics, baseline);
      return {
        motif,
        metrics,
        delta,
        score: objective(metrics, delta) + motif.meanOvertakeDelta * 100 + motif.rootCount,
      };
    })
    .sort((a, b) => b.score - a.score || a.motif.id.localeCompare(b.motif.id))
    .slice(0, args.topK)
    .map((entry) => ({
      ...entry.motif,
      metrics: metricsPublic(entry.metrics),
      deltaVsFrontierRank: entry.delta,
      selectorScore: round(entry.score),
    }));
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
    changedRootCount: metricsList.reduce((sum, item) => sum + item.changedRootCount, 0),
    capturedOvertakes: metricsList.reduce((sum, item) => sum + item.capturedOvertakes, 0),
    missedOvertakes: metricsList.reduce((sum, item) => sum + item.missedOvertakes, 0),
    falseSwitches: metricsList.reduce((sum, item) => sum + item.falseSwitches, 0),
    roots: metricsList.flatMap((item) => item.roots),
  };
}

function crossValidate(groups, args) {
  const rootIds = groups.map(([rootId]) => rootId);
  const folds = rootFolds(rootIds, args.folds);
  const baselineEvals = [];
  const candidateEvals = [];
  const reports = [];
  for (let fold = 0; fold < folds.length; fold += 1) {
    const evalSet = new Set(folds[fold]);
    const trainSet = new Set(rootIds.filter((rootId) => !evalSet.has(rootId)));
    const trainGroups = filterGroups(groups, trainSet);
    const evalGroups = filterGroups(groups, evalSet);
    const trainBaseline = evaluate(trainGroups, (rows) => ({ row: rows.find((row) => row.rank === 1) || rows[0], switched: false, reason: 'frontier_rank' }), 'frontier_rank_train', args.minDelta);
    const evalBaseline = evaluate(evalGroups, (rows) => ({ row: rows.find((row) => row.rank === 1) || rows[0], switched: false, reason: 'frontier_rank' }), 'frontier_rank_eval', args.minDelta);
    const motifs = generateMotifs(trainGroups, args);
    const ranked = rankMotifs(motifs, trainGroups, trainBaseline, args);
    const selected = ranked[0] || null;
    const trainCandidate = selected
      ? evaluate(trainGroups, (rows) => {
        const choice = chooseByMotif(rows, selected);
        return choice ? { ...choice, score: choice.switched ? 1 : 0 } : null;
      }, selected.id, args.minDelta)
      : trainBaseline;
    const evalCandidate = selected
      ? evaluate(evalGroups, (rows) => {
        const choice = chooseByMotif(rows, selected);
        return choice ? { ...choice, score: choice.switched ? 1 : 0 } : null;
      }, selected.id, args.minDelta)
      : evalBaseline;
    baselineEvals.push(evalBaseline);
    candidateEvals.push(evalCandidate);
    reports.push({
      fold,
      trainRootCount: trainGroups.length,
      evalRootCount: evalGroups.length,
      trainMotifCount: motifs.length,
      selectedMotif: selected,
      trainBaseline: metricsPublic(trainBaseline),
      trainCandidate: metricsPublic(trainCandidate),
      trainDeltaVsFrontierRank: metricDelta(trainCandidate, trainBaseline),
      evalBaseline: metricsPublic(evalBaseline),
      evalCandidate: metricsPublic(evalCandidate),
      evalDeltaVsFrontierRank: metricDelta(evalCandidate, evalBaseline),
    });
  }
  const baseline = aggregate('frontier_rank_cross_validation_baseline', baselineEvals);
  const candidate = aggregate('gpu_fight_overtake_selector_cross_validation', candidateEvals);
  const delta = metricDelta(candidate, baseline);
  return {
    baseline,
    candidate,
    deltaVsFrontierRank: delta,
    observedLift: (delta.meanFightScore !== null && delta.meanFightScore > 0) || delta.positiveTop1 > 0,
    folds: reports,
  };
}

function promotionStatus(fixedDelta, cvDelta, fixedMetrics, cvMetrics) {
  const fixedReady = fixedDelta.meanFightScore > 0
    && fixedDelta.positiveTop1 >= 0
    && fixedDelta.acceptedUsefulTop1 >= 0
    && fixedDelta.selectedMoveTop1 >= 0
    && fixedMetrics.falseSwitches === 0;
  const cvReady = cvDelta.meanFightScore > 0
    && cvDelta.positiveTop1 >= 0
    && cvDelta.acceptedUsefulTop1 >= 0
    && cvDelta.selectedMoveTop1 >= 0
    && cvMetrics.falseSwitches === 0;
  const blockers = [];
  if (!fixedReady) blockers.push('fixed_artifact_overtake_selector_not_safe');
  if (!cvReady) blockers.push('root_fold_overtake_selector_not_safe');
  blockers.push('source_fight_rollout_is_posthoc_scout_label');
  return {
    status: fixedReady && cvReady ? 'freeze_candidate_only' : 'not_promoted',
    fixedReady,
    crossValidatedReady: cvReady,
    blockers,
    noRuntimePromotion: true,
  };
}

function main() {
  const args = parseArgs(process.argv.slice(2));
  const fightPath = path.resolve(args.fight);
  const fight = readJson(fightPath);
  requireGpuFightArtifact(fight);
  const rows = normalizeRows(fight);
  const groups = groupByRoot(rows).filter(([, groupRows]) => groupRows.some((row) => row.rank === 1) && groupRows.length > 1);
  const pairs = overtakePairs(groups, args.minDelta);
  const baseline = evaluate(groups, (groupRows) => ({ row: groupRows.find((row) => row.rank === 1) || groupRows[0], switched: false, reason: 'frontier_rank' }), 'frontier_rank', args.minDelta);
  const oracle = evaluate(groups, (groupRows, summary) => ({ row: summary.best, switched: summary.best.rank !== summary.baseline.rank, reason: 'gpu_fight_oracle' }), 'gpu_fight_oracle_upper_bound', args.minDelta);
  const motifs = generateMotifs(groups, args);
  const topSelectors = rankMotifs(motifs, groups, baseline, args);
  const fixedBest = topSelectors[0] || null;
  const fixedMetrics = fixedBest
    ? evaluate(groups, (groupRows) => {
      const choice = chooseByMotif(groupRows, fixedBest);
      return choice ? { ...choice, score: choice.switched ? 1 : 0 } : null;
    }, fixedBest.id, args.minDelta)
    : baseline;
  const fixedDelta = metricDelta(fixedMetrics, baseline);
  const cv = crossValidate(groups, args);
  const promotionPolicy = promotionStatus(fixedDelta, cv.deltaVsFrontierRank, fixedMetrics, cv.candidate);
  const output = {
    schemaVersion: SCHEMA_VERSION,
    generatedAt: new Date().toISOString(),
    condition: {
      source: path.resolve(args.conditionSource),
      runLabel: 'scout_new_experiment_gpu_fight_overtake_selector_mining',
      changedFields: 'post-hoc differential selector mining over recorded GPU fight rollout labels only; no runtime behavior changed',
      labCondition: 'scout/new_experiment/source_temporal_proxy_top_rank_frontier_rows',
      metric: 'root-fold mean forced-candidate GPU self-play fight score with accepted/selected safety',
    },
    sources: {
      fightPath,
      fightSha256: sha256File(fightPath),
      fightSchemaVersion: fight.schemaVersion || null,
      fightGeneratedAt: fight.generatedAt || null,
      gpuFightRolloutLabel: fight.gpuFightRolloutLabel || null,
    },
    miningPolicy: {
      algorithm: 'differential_non_rank1_overtake_tag_conjunction_selector',
      maxMotifSize: args.maxSize,
      minRoots: args.minRoots,
      minRows: args.minRows,
      minDelta: args.minDelta,
      topK: args.topK,
      noRuntimePromotion: true,
    },
    corpus: {
      rowCount: rows.length,
      rootCount: groups.length,
      overtakeRootCount: pairs.filter((pair) => pair.overtaken).length,
      overtakePositiveBestCount: pairs.filter((pair) => pair.overtaken && pair.bestGpuFight.fightScore > 0).length,
      overtakeMeanDelta: round(
        pairs.filter((pair) => pair.overtaken).reduce((sum, pair) => sum + pair.deltaVsRank1, 0)
        / Math.max(1, pairs.filter((pair) => pair.overtaken).length),
      ),
      candidateSelectorCount: motifs.length,
    },
    overtakePairs: pairs.filter((pair) => pair.overtaken),
    baseline: {
      frontierRank: metricsPublic(baseline),
      gpuFightOracleUpperBound: metricsPublic(oracle),
      oracleDeltaVsFrontierRank: metricDelta(oracle, baseline),
    },
    fixedArtifact: {
      selectedMotif: fixedBest,
      metrics: metricsPublic(fixedMetrics),
      deltaVsFrontierRank: fixedDelta,
    },
    crossValidation: cv,
    topSelectors,
    promotionPolicy,
  };
  const outPath = args.out
    ? path.resolve(args.out)
    : fightPath.replace(/\.json$/u, '.overtake_selectors.json');
  writeJson(outPath, output);
  console.log(JSON.stringify({
    ok: true,
    output: outPath,
    rowCount: rows.length,
    rootCount: groups.length,
    overtakeRootCount: output.corpus.overtakeRootCount,
    candidateSelectorCount: motifs.length,
    fixedArtifactDelta: fixedDelta,
    crossValidatedDelta: cv.deltaVsFrontierRank,
    observedLift: cv.observedLift,
    promote: promotionPolicy.status !== 'not_promoted',
  }, null, 2));
}

main();
