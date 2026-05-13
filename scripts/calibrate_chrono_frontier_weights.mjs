#!/usr/bin/env node
import fs from 'node:fs';
import path from 'node:path';

const SCHEMA_VERSION = 'dojo.chrono_frontier_calibration.v1';

function usage() {
  return `Usage: node scripts/calibrate_chrono_frontier_weights.mjs --bridge <bridge.json> --chrono <chrono_sidecar.json> [--out <calibration.json>]

Calibration-only sweep over bounded frontier/chrono feature weights. This uses
accepted-useful-injection labels from the fixed artifact, so it is diagnosis and
candidate generation only, not promotion or runtime integration.
`;
}

function parseArgs(argv) {
  const args = {
    bridge: null,
    chrono: null,
    out: null,
    folds: 4,
    top: 20,
    conditionSource: '/home/frosty40/cuda_dojo/LOGIC_RAY_FRONTIER_RUNTIME_RECEIPT_2026-05-06.md',
    runLabel: 'calibration_only_chrono_frontier_weight_sweep_on_documented_offset64_artifact',
  };
  for (let i = 0; i < argv.length; i += 1) {
    const token = argv[i];
    if (token === '--help' || token === '-h') {
      console.log(usage());
      process.exit(0);
    }
    if (token === '--bridge') args.bridge = argv[++i];
    else if (token === '--chrono') args.chrono = argv[++i];
    else if (token === '--out') args.out = argv[++i];
    else if (token === '--folds') args.folds = Math.max(2, Number(argv[++i] || args.folds));
    else if (token === '--top') args.top = Math.max(1, Number(argv[++i] || args.top));
    else if (token === '--condition-source') args.conditionSource = argv[++i];
    else if (token === '--run-label') args.runLabel = argv[++i];
    else throw new Error(`unknown argument: ${token}\n${usage()}`);
  }
  if (!args.bridge) throw new Error(`missing --bridge\n${usage()}`);
  if (!args.chrono) throw new Error(`missing --chrono\n${usage()}`);
  return args;
}

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, 'utf8'));
}

function asNumber(value, fallback = 0) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

function clamp01(value, fallback = 0) {
  return Math.max(0, Math.min(1, asNumber(value, fallback)));
}

function round(value, digits = 6) {
  const factor = 10 ** digits;
  return Math.round(asNumber(value, 0) * factor) / factor;
}

function defaultOutPath(chronoPath) {
  const parsed = path.parse(chronoPath);
  const stem = parsed.base.replace(/\.chrono_sidecar\.json$/, '').replace(/\.json$/, '');
  return path.join(parsed.dir, `${stem}.chrono_calibration_sweep.json`);
}

function rowsByHash(bundle) {
  const map = new Map();
  for (const row of Array.isArray(bundle.rows) ? bundle.rows : []) {
    if (row.logicRayFrontierHash) map.set(row.logicRayFrontierHash, row);
  }
  return map;
}

function accepted(bridgeRow, chronoRow) {
  const frontier = bridgeRow.logicRayFrontier || bridgeRow.pzrgCandidate?.logicRayFrontier || {};
  const injection = bridgeRow.pzrgCandidate?.injection_relevance || {};
  return Boolean(
    frontier.gate?.acceptedUsefulInjection
      || injection.accepted_useful_injection
      || injection.promotion_gate_approved
      || chronoRow?.runtimeChoiceSignal?.acceptedUsefulInjection,
  );
}

function selected(bridgeRow, chronoRow) {
  const frontier = bridgeRow.logicRayFrontier || bridgeRow.pzrgCandidate?.logicRayFrontier || {};
  return Boolean(frontier.gate?.selectedMoveInFrontier || chronoRow?.runtimeChoiceSignal?.selectedMoveInFrontier);
}

function rowFeatures(bridgeRow, chronoRow) {
  const frontier = bridgeRow.logicRayFrontier || bridgeRow.pzrgCandidate?.logicRayFrontier || {};
  const rank = Math.max(1, asNumber(frontier.rank || bridgeRow.rank, 1));
  return {
    bridgeId: bridgeRow.bridgeId,
    hash: bridgeRow.logicRayFrontierHash,
    rootId: frontier.rootId || bridgeRow.rootId || 'root',
    move: frontier.move || bridgeRow.move || '0000',
    rank,
    accepted: accepted(bridgeRow, chronoRow),
    selected: selected(bridgeRow, chronoRow),
    rankScore: -rank,
    utility: asNumber(frontier.utility, 0),
    pathProbability: clamp01(frontier.pathProbability, 0),
    lockIn: clamp01(frontier.lockIn, 0),
    risk: clamp01(frontier.risk, 0),
    scoreGapNorm: clamp01(asNumber(frontier.scoreGapFromBestCp, 0) / 400, 0),
    stability: clamp01(chronoRow?.diagnostics?.stabilityScore, 0),
    uncertainty: clamp01(chronoRow?.uncertainty?.score, 0),
    normDriftAbs: Math.min(1, Math.abs(asNumber(chronoRow?.diagnostics?.normDrift, 0))),
    contortion: clamp01(chronoRow?.pathContortion?.score, 0),
  };
}

function groupRows(featureRows) {
  const groups = new Map();
  for (const row of featureRows) {
    if (!groups.has(row.rootId)) groups.set(row.rootId, []);
    groups.get(row.rootId).push(row);
  }
  return [...groups.entries()]
    .sort(([a], [b]) => String(a).localeCompare(String(b)))
    .map(([rootId, rows], index) => ({
      rootId,
      fold: index,
      rows: rows.sort((a, b) => a.rank - b.rank),
    }));
}

function makeConfigs() {
  const grid = {
    rank: [0, 0.5, 1, 2],
    utility: [0, 0.5, 1],
    pathProbability: [0, 0.5],
    lockIn: [0, 0.5],
    risk: [0, 0.5],
    stability: [0, 0.05, 0.15],
    uncertainty: [0, 0.1, 0.25],
    normDrift: [0, 0.1, 0.25],
    contortion: [0, 0.1],
    scoreGap: [0, 0.1],
  };
  const configs = [];
  let id = 0;
  for (const rank of grid.rank)
    for (const utility of grid.utility)
      for (const pathProbability of grid.pathProbability)
        for (const lockIn of grid.lockIn)
          for (const risk of grid.risk)
            for (const stability of grid.stability)
              for (const uncertainty of grid.uncertainty)
                for (const normDrift of grid.normDrift)
                  for (const contortion of grid.contortion)
                    for (const scoreGap of grid.scoreGap) {
                      const weights = {
                        rank,
                        utility,
                        pathProbability,
                        lockIn,
                        risk,
                        stability,
                        uncertainty,
                        normDrift,
                        contortion,
                        scoreGap,
                      };
                      const usesChrono = Boolean(stability || uncertainty || normDrift || contortion);
                      configs.push({
                        id: `chrono_cal_${String(++id).padStart(5, '0')}`,
                        usesChrono,
                        weights,
                      });
                    }
  return configs;
}

function scoreRow(row, weights) {
  return (
    weights.rank * row.rankScore
    + weights.utility * row.utility
    + weights.pathProbability * row.pathProbability
    + weights.lockIn * row.lockIn
    - weights.risk * row.risk
    + weights.stability * row.stability
    - weights.uncertainty * row.uncertainty
    - weights.normDrift * row.normDriftAbs
    - weights.contortion * row.contortion
    - weights.scoreGap * row.scoreGapNorm
  );
}

function evaluateConfig(config, groups, foldFilter = null) {
  const roots = foldFilter ? groups.filter((group) => foldFilter(group.fold)) : groups;
  let top1 = 0;
  let top3 = 0;
  let selectedTop1 = 0;
  let acceptedRankSum = 0;
  let acceptedRankCount = 0;
  const examples = [];
  for (const group of roots) {
    const ranked = [...group.rows].sort((a, b) => {
      const delta = scoreRow(b, config.weights) - scoreRow(a, config.weights);
      if (delta !== 0) return delta;
      return a.rank - b.rank;
    }).map((row, index) => ({
      ...row,
      comparisonRank: index + 1,
      score: round(scoreRow(row, config.weights)),
    }));
    const acceptedRows = ranked.filter((row) => row.accepted);
    if (ranked[0]?.accepted) top1 += 1;
    if (ranked.slice(0, 3).some((row) => row.accepted)) top3 += 1;
    if (ranked[0]?.selected) selectedTop1 += 1;
    if (acceptedRows[0]) {
      acceptedRankSum += acceptedRows[0].comparisonRank;
      acceptedRankCount += 1;
    }
    if (examples.length < 8) {
      examples.push({
        rootId: group.rootId,
        top1: {
          bridgeId: ranked[0]?.bridgeId,
          move: ranked[0]?.move,
          originalRank: ranked[0]?.rank,
          comparisonRank: ranked[0]?.comparisonRank,
          score: ranked[0]?.score,
          accepted: Boolean(ranked[0]?.accepted),
          selected: Boolean(ranked[0]?.selected),
        },
        acceptedBestRank: acceptedRows[0]?.comparisonRank || null,
      });
    }
  }
  const rootCount = roots.length;
  return {
    rootCount,
    top1AcceptedUsefulInjections: top1,
    top1AcceptedUsefulInjectionRate: rootCount ? top1 / rootCount : 0,
    top3AcceptedUsefulInjections: top3,
    top3AcceptedUsefulInjectionRate: rootCount ? top3 / rootCount : 0,
    selectedMoveTop1: selectedTop1,
    meanAcceptedCandidateRank: acceptedRankCount ? round(acceptedRankSum / acceptedRankCount) : null,
    examples,
  };
}

function compareMetrics(a, b) {
  const top1 = a.metrics.top1AcceptedUsefulInjections - b.metrics.top1AcceptedUsefulInjections;
  if (top1) return top1;
  const top3 = a.metrics.top3AcceptedUsefulInjections - b.metrics.top3AcceptedUsefulInjections;
  if (top3) return top3;
  const aRank = a.metrics.meanAcceptedCandidateRank ?? 1e9;
  const bRank = b.metrics.meanAcceptedCandidateRank ?? 1e9;
  if (aRank !== bRank) return bRank - aRank;
  return Number(a.config.usesChrono) - Number(b.config.usesChrono);
}

function summarizeResult(config, metrics, baseline) {
  return {
    config,
    metrics,
    deltaVsFrontierRank: {
      top1: metrics.top1AcceptedUsefulInjections - baseline.top1AcceptedUsefulInjections,
      top3: metrics.top3AcceptedUsefulInjections - baseline.top3AcceptedUsefulInjections,
      meanAcceptedCandidateRank: (
        metrics.meanAcceptedCandidateRank != null && baseline.meanAcceptedCandidateRank != null
          ? round(metrics.meanAcceptedCandidateRank - baseline.meanAcceptedCandidateRank)
          : null
      ),
    },
  };
}

function rankedRows(config, group) {
  return [...group.rows].sort((a, b) => {
    const delta = scoreRow(b, config.weights) - scoreRow(a, config.weights);
    if (delta !== 0) return delta;
    return a.rank - b.rank;
  });
}

function equivalenceToBaseline(candidateConfig, baselineConfig, groups) {
  let sameTop1 = 0;
  let sameTop3Set = 0;
  const differences = [];
  for (const group of groups) {
    const candidate = rankedRows(candidateConfig, group);
    const baseline = rankedRows(baselineConfig, group);
    const candidateTop1 = candidate[0]?.hash || null;
    const baselineTop1 = baseline[0]?.hash || null;
    const candidateTop3 = candidate.slice(0, 3).map((row) => row.hash).sort();
    const baselineTop3 = baseline.slice(0, 3).map((row) => row.hash).sort();
    if (candidateTop1 === baselineTop1) sameTop1 += 1;
    if (candidateTop3.join('|') === baselineTop3.join('|')) sameTop3Set += 1;
    if ((candidateTop1 !== baselineTop1 || candidateTop3.join('|') !== baselineTop3.join('|')) && differences.length < 12) {
      differences.push({
        rootId: group.rootId,
        baselineTop1: {
          hash: baselineTop1,
          move: baseline[0]?.move,
          rank: baseline[0]?.rank,
          accepted: Boolean(baseline[0]?.accepted),
        },
        candidateTop1: {
          hash: candidateTop1,
          move: candidate[0]?.move,
          rank: candidate[0]?.rank,
          accepted: Boolean(candidate[0]?.accepted),
        },
      });
    }
  }
  return {
    rootCount: groups.length,
    sameTop1Roots: sameTop1,
    sameTop1Rate: groups.length ? sameTop1 / groups.length : 0,
    sameTop3SetRoots: sameTop3Set,
    sameTop3SetRate: groups.length ? sameTop3Set / groups.length : 0,
    differences,
  };
}

function crossValidate(configs, groups, folds, baselineConfig) {
  const foldRows = [];
  let aggregateTop1 = 0;
  let aggregateTop3 = 0;
  let aggregateRoots = 0;
  let aggregateRankSum = 0;
  let aggregateRankCount = 0;
  for (let fold = 0; fold < folds; fold += 1) {
    const trainFilter = (idx) => idx % folds !== fold;
    const evalFilter = (idx) => idx % folds === fold;
    const trainBaseline = evaluateConfig(baselineConfig, groups, trainFilter);
    const trainResults = configs
      .filter((config) => config.usesChrono)
      .map((config) => ({ config, metrics: evaluateConfig(config, groups, trainFilter) }))
      .sort((a, b) => compareMetrics(b, a));
    const selected = trainResults[0];
    const evalMetrics = evaluateConfig(selected.config, groups, evalFilter);
    const evalBaseline = evaluateConfig(baselineConfig, groups, evalFilter);
    aggregateTop1 += evalMetrics.top1AcceptedUsefulInjections;
    aggregateTop3 += evalMetrics.top3AcceptedUsefulInjections;
    aggregateRoots += evalMetrics.rootCount;
    if (evalMetrics.meanAcceptedCandidateRank != null) {
      aggregateRankSum += evalMetrics.meanAcceptedCandidateRank * evalMetrics.rootCount;
      aggregateRankCount += evalMetrics.rootCount;
    }
    foldRows.push({
      fold,
      trainBaseline,
      selectedConfig: selected.config,
      trainMetrics: selected.metrics,
      evalBaseline,
      evalMetrics,
      evalDeltaVsBaseline: {
        top1: evalMetrics.top1AcceptedUsefulInjections - evalBaseline.top1AcceptedUsefulInjections,
        top3: evalMetrics.top3AcceptedUsefulInjections - evalBaseline.top3AcceptedUsefulInjections,
        meanAcceptedCandidateRank: (
          evalMetrics.meanAcceptedCandidateRank != null && evalBaseline.meanAcceptedCandidateRank != null
            ? round(evalMetrics.meanAcceptedCandidateRank - evalBaseline.meanAcceptedCandidateRank)
            : null
        ),
      },
    });
  }
  const baselineAll = evaluateConfig(baselineConfig, groups);
  return {
    folds: foldRows,
    aggregate: {
      rootCount: aggregateRoots,
      top1AcceptedUsefulInjections: aggregateTop1,
      top1AcceptedUsefulInjectionRate: aggregateRoots ? aggregateTop1 / aggregateRoots : 0,
      top3AcceptedUsefulInjections: aggregateTop3,
      top3AcceptedUsefulInjectionRate: aggregateRoots ? aggregateTop3 / aggregateRoots : 0,
      meanAcceptedCandidateRank: aggregateRankCount ? round(aggregateRankSum / aggregateRankCount) : null,
      deltaVsFrontierRank: {
        top1: aggregateTop1 - baselineAll.top1AcceptedUsefulInjections,
        top3: aggregateTop3 - baselineAll.top3AcceptedUsefulInjections,
        meanAcceptedCandidateRank: (
          aggregateRankCount && baselineAll.meanAcceptedCandidateRank != null
            ? round((aggregateRankSum / aggregateRankCount) - baselineAll.meanAcceptedCandidateRank)
            : null
        ),
      },
    },
  };
}

function main() {
  const args = parseArgs(process.argv.slice(2));
  const bridgePath = path.resolve(args.bridge);
  const chronoPath = path.resolve(args.chrono);
  const bridge = readJson(bridgePath);
  const chrono = readJson(chronoPath);
  const chronoByHash = rowsByHash(chrono);
  const bridgeRows = Array.isArray(bridge.rows) ? bridge.rows : [];
  const featureRows = bridgeRows.map((row) => rowFeatures(row, chronoByHash.get(row.logicRayFrontierHash)));
  const groups = groupRows(featureRows);
  const configs = makeConfigs();
  const frontierRankConfig = {
    id: 'frontier_rank',
    usesChrono: false,
    weights: {
      rank: 1,
      utility: 0,
      pathProbability: 0,
      lockIn: 0,
      risk: 0,
      stability: 0,
      uncertainty: 0,
      normDrift: 0,
      contortion: 0,
      scoreGap: 0,
    },
  };
  const baseline = evaluateConfig(frontierRankConfig, groups);
  const evaluated = configs.map((config) => ({
    config,
    metrics: evaluateConfig(config, groups),
  }));
  const topOverall = evaluated
    .map((item) => summarizeResult(item.config, item.metrics, baseline))
    .sort((a, b) => compareMetrics({ config: a.config, metrics: a.metrics }, { config: b.config, metrics: b.metrics }) * -1)
    .slice(0, args.top);
  const topChrono = evaluated
    .filter((item) => item.config.usesChrono)
    .map((item) => summarizeResult(item.config, item.metrics, baseline))
    .sort((a, b) => compareMetrics({ config: a.config, metrics: a.metrics }, { config: b.config, metrics: b.metrics }) * -1)
    .slice(0, args.top);
  const bestChrono = topChrono[0] || null;
  const crossValidation = crossValidate(configs, groups, args.folds, frontierRankConfig);
  const bestChronoEquivalenceToFrontierRank = bestChrono
    ? equivalenceToBaseline(bestChrono.config, frontierRankConfig, groups)
    : null;
  const output = {
    schemaVersion: SCHEMA_VERSION,
    generatedAt: new Date().toISOString(),
    condition: {
      source: path.resolve(args.conditionSource),
      runLabel: args.runLabel,
      metric: 'accepted useful injection top-k ranking over documented offset-64 frontier roots',
      changedFields: 'none; calibration-only sweep over existing bridge and chrono sidecar artifacts',
      labelUse: 'accepted-useful-injection labels are used to select weights; this is not promotion evidence',
    },
    sources: {
      bridgePath,
      bridgeSchemaVersion: bridge.schemaVersion || null,
      chronoPath,
      chronoSchemaVersion: chrono.schemaVersion || null,
      bridgeRows: bridgeRows.length,
      chronoRows: Array.isArray(chrono.rows) ? chrono.rows.length : 0,
      rootCount: groups.length,
    },
    sweep: {
      configCount: configs.length,
      folds: args.folds,
      featureSet: [
        'rank',
        'utility',
        'pathProbability',
        'lockIn',
        'risk',
        'stability',
        'uncertainty',
        'normDriftAbs',
        'pathContortion',
        'scoreGap',
      ],
    },
    baseline: {
      config: frontierRankConfig,
      metrics: baseline,
    },
    bestChrono,
    bestChronoEquivalenceToFrontierRank,
    topOverall,
    topChrono,
    crossValidation,
    promotionPolicy: {
      status: 'not_promoted',
      reason: 'calibration uses labels on a fixed artifact; any candidate must be frozen and rerun as a fixed GPU gate condition before promotion',
      requiredNextEvidence: [
        'choose one calibrated formula without looking at new gate labels',
        'rerun heldout frontier gate with that formula as a fixed consumer',
        'show accepted useful injection lift without worsening strict/proxy gate interpretation',
      ],
    },
  };
  const outPath = path.resolve(args.out || defaultOutPath(chronoPath));
  fs.mkdirSync(path.dirname(outPath), { recursive: true });
  fs.writeFileSync(outPath, `${JSON.stringify(output, null, 2)}\n`);
  console.log(JSON.stringify({
    ok: true,
    output: outPath,
    rootCount: groups.length,
    configCount: configs.length,
    baselineTop1: baseline.top1AcceptedUsefulInjections,
    bestChronoTop1: bestChrono?.metrics.top1AcceptedUsefulInjections ?? null,
    bestChronoDelta: bestChrono?.deltaVsFrontierRank ?? null,
    bestChronoEquivalenceToFrontierRank,
    crossValidatedDelta: crossValidation.aggregate.deltaVsFrontierRank,
    promotionStatus: output.promotionPolicy.status,
  }, null, 2));
}

try {
  main();
} catch (error) {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
}
