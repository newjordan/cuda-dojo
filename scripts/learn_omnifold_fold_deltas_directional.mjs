#!/usr/bin/env node
/**
 * learn_omnifold_fold_deltas_directional.mjs
 *
 * Directional fold-delta learner — the prism axis.
 *
 * STANDARD (symmetric):
 *   Every row contributes equally to baseMean → fold is an average over all prisms.
 *   Equivalent to "flat light" through a prism network.
 *
 * DIRECTIONAL (--directional):
 *   Rows are weighted by prismConfidence = f(policyMargin, valueError, injection).
 *   X-strong prisms (high policy conviction, low value error, accepted injections)
 *   dominate the fold. Y-weak prisms contribute less.
 *
 *   Metaphor: "follow the X-highest conjunction pathway through the prism lightbeams"
 *   X = policy logit margin (how sure is this move?)
 *   Y = value prediction error (how wrong was the eval?)
 *   Confidence = tanh(policyMargin / (1 + |valueError|))  → [0, 1]
 *
 * When policy margin data isn't available (pre-FrostMatrix integration), falls back to
 * acceptedUsefulInjection as a binary proxy.
 *
 * Usage:
 *   node scripts/learn_omnifold_fold_deltas_directional.mjs \
 *     --omnifold manifest.json \
 *     --fight gpu_fight_rollout.json \
 *     --directional \
 *     --out directional_deltas.json
 */

import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';

const SCHEMA_VERSION = 'dojo.omnifold_fold_delta_directional.v1';
const DEFAULT_CONDITION = '/home/frosty40/cuda_dojo/LOGIC_RAY_FRONTIER_RUNTIME_RECEIPT_2026-05-06.md';

function usage() {
  return `Usage: node scripts/learn_omnifold_fold_deltas_directional.mjs
  --omnifold <manifest.json>
  --fight <gpu_fight_rollout_patterns.json>
  [--directional]                enable prism-weighted directional folds
  [--prism-data <jsonl>]         optional: per-root policyMargin + valueError from FrostMatrix
  [--prism-gamma <float>]         confidence curve steepness (default: 1.0)
  [--out <deltas.json>]
  [--folds <n>]
  [--min-tag-rows <n>]
  [--max-tag-weights <n>]
  [--condition-source <path>]
`;
}

function parseArgs(argv) {
  const args = {
    omnifold: null,
    fight: null,
    out: null,
    directional: false,
    prismData: null,
    prismGamma: 1.0,
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
    } else if (token === '--directional') {
      args.directional = true;
    } else if (token === '--prism-data') {
      args.prismData = argv[++i];
    } else if (token === '--prism-gamma') {
      args.prismGamma = Number(argv[++i]);
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

// ═══════════════════════════════════════════════════════════════════════════
// PRISM CONFIDENCE — the directional gate
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Compute prism confidence per row.
 *
 * With FrostMatrix data:
 *   X = policyMargin = max(logit) - runnerUp(logit)    [how sure is this move?]
 *   Y = valueError   = |valuePred - actualOutcome|      [how wrong was the eval?]
 *   confidence = tanh(X / (1 + Y))                      [0..1]
 *
 * Without FrostMatrix data (pre-integration), uses:
 *   confidence = acceptedUsefulInjection ? 1.0 : 0.3   [binary proxy]
 *
 * The proxy treats accepted injections as X-strong prisms (keep),
 * non-accepted as Y-weak prisms (dampen).
 */
function loadPrismData(prismDataPath) {
  if (!prismDataPath) return new Map();
  const content = fs.readFileSync(prismDataPath, 'utf8');
  const map = new Map();
  for (const line of content.trim().split('\n')) {
    if (!line.trim()) continue;
    const entry = JSON.parse(line);
    map.set(String(entry.rootId), {
      policyMargin: Number(entry.policyMargin) || 0,
      valueError: Number(entry.valueError) || 0,
      policyEntropy: Number(entry.policyEntropy) || null,
      policyEntropyNormalized: Number(entry.policyEntropyNormalized) || null,
      winConfidence: Number(entry.winConfidence) || null,
    });
  }
  return map;
}

/**
 * Prism confidence — three-axis conviction scoring.
 *
 * X = policy margin (how sure is this move vs runner-up?)
 * Y = value error (how wrong was the eval?)
 * H = policy entropy (how diffuse is the policy distribution?)
 *
 * Three formulations depending on available data:
 *
 * A. FULL (FrostMatrix policy + value + entropy):
 *    conviction = (1 - H_norm) * tanh(gamma * policyMargin / (1 + |valueError|))
 *    Low entropy amplifies the X/Y ratio; high entropy dampens it.
 *
 * B. PARTIAL (FrostMatrix policy + value, no entropy):
 *    conviction = tanh(gamma * policyMargin / (1 + |valueError|))
 *    Classic prism: X/N ratio only.
 *
 * C. PROXY (no FrostMatrix, entropy from frontier score distribution):
 *    conviction = acceptedUsefulInjection ? (1 - H_norm) : 0.3 * (1 - H_norm)
 *    Entropy still modulates the binary gate.
 *
 * D. FALLBACK (nothing available):
 *    conviction = acceptedUsefulInjection ? 1.0 : 0.3
 */
function prismConfidence(row, prismMap, gamma) {
  const entry = prismMap.get(String(row.rootId));

  if (entry) {
    const hasFrostMatrix = entry.policyMargin > 0 || entry.valueError > 0;
    const hasEntropy = entry.policyEntropyNormalized != null;
    const hasWinConfidence = entry.winConfidence != null;

    if (hasFrostMatrix && hasEntropy) {
      // FORMULA A: Full tri-axial prism — policy margin, value error, entropy
      const X = entry.policyMargin;
      const Y = Math.abs(entry.valueError);
      const H = entry.policyEntropyNormalized;          // [0, 1] — 0 = certain, 1 = chaotic
      const conviction = (1 - H);                        // [0, 1] — 1 = focused, 0 = noise
      const ratio = X / (1.0 + Y);
      return conviction * Math.tanh(gamma * ratio);     // entropy gates the prism beam
    }

    if (hasWinConfidence) {
      // Direct win confidence from frontier entropy enrichment
      return Math.min(0.99, Math.max(0.01, entry.winConfidence));
    }

    if (hasFrostMatrix) {
      // FORMULA B: Classic prism X/Y ratio (no entropy available yet)
      const X = entry.policyMargin;
      const Y = Math.abs(entry.valueError);
      const ratio = X / (1.0 + Y);
      return Math.tanh(gamma * ratio);
    }
  }

  // Pre-FrostMatrix: entropy from row-level policyEntropy if available
  const entropyNorm = typeof row.policyEntropyNormalized === 'number'
    ? row.policyEntropyNormalized
    : null;

  if (entropyNorm != null) {
    // FORMULA C: Binary gate modulated by entropy
    const conviction = 1 - entropyNorm;                // low entropy → high conviction
    const baseWeight = row.acceptedUsefulInjection ? 1.0 : 0.3;
    return conviction * baseWeight;                    // entropy gates the proxy
  }

  // FORMULA D: Pure binary proxy (legacy fallback)
  return row.acceptedUsefulInjection ? 1.0 : 0.3;
}

// ═══════════════════════════════════════════════════════════════════════════
// CORE — identical to symmetric, with confidence-weighted moments
// ═══════════════════════════════════════════════════════════════════════════

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
      || tag.startsWith('chrono:policyEntropy:')
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
    || tag.startsWith('chrono:policyEntropy:')
    || tag === 'chrono:stable_score'
    || tag === 'chrono:unstable_score';
}

function normalizeRows(fight, prismMap, gamma) {
  const rollouts = Array.isArray(fight.rollouts) ? fight.rollouts : [];
  return rollouts
    .filter((row) => Number.isFinite(Number(row.fightScore)))
    .map((row) => {
      const entropyNorm = asNumber(row.rayfrontMetrics?.policyEntropyNormalized) || null;
      const normalized = {
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
        policyEntropyNormalized: entropyNorm,
      };
      // Compute prism confidence using the normalized row (which has policyEntropyNormalized)
      normalized.prismConfidence = prismConfidence(normalized, prismMap, gamma);
      return normalized;
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

function rowsFromGroups(groups) {
  return groups.flatMap(([, rows]) => rows);
}

// ═══════════════════════════════════════════════════════════════════════════
// DIRECTIONAL TRAINING — confidence-weighted moments
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Weighted mean: sum(value * weight) / sum(weight)
 */
function weightedMean(values, weights) {
  if (!values.length) return 0;
  const totalWeight = weights.reduce((sum, w) => sum + w, 0);
  if (totalWeight === 0) return 0;
  return values.reduce((sum, v, i) => sum + v * weights[i], 0) / totalWeight;
}

function trainFamilyModelDirectional(family, rows, options) {
  if (!rows.length) {
    return {
      familyId: family.id,
      foldFamily: family.foldFamily,
      orderSet: family.orderSet,
      baseMeanFightScore: round(0),
      trainRows: 0,
      totalConfidence: 0,
      eligibleTagCount: 0,
      weights: [],
    };
  }

  const confidences = rows.map((row) => row.prismConfidence);
  const baseMean = weightedMean(rows.map((row) => row.fightScore), confidences);

  const tagStats = new Map();
  for (let i = 0; i < rows.length; i++) {
    const row = rows[i];
    const c = confidences[i];
    for (const tag of row.tags) {
      if (!allowedTag(family, tag)) continue;
      if (!tagStats.has(tag)) tagStats.set(tag, { tag, rows: 0, scoreWeightedSum: 0, confidenceSum: 0 });
      const stat = tagStats.get(tag);
      stat.rows += 1;
      stat.scoreWeightedSum += row.fightScore * c;
      stat.confidenceSum += c;
    }
  }

  const weights = [...tagStats.values()]
    .filter((stat) => stat.rows >= options.minTagRows && stat.confidenceSum > 0)
    .map((stat) => ({
      tag: stat.tag,
      rows: stat.rows,
      confidenceSum: round(stat.confidenceSum),
      meanFightScore: round(stat.scoreWeightedSum / stat.confidenceSum),
      weight: round((stat.scoreWeightedSum / stat.confidenceSum) - baseMean),
    }))
    .filter((stat) => stat.weight !== 0)
    .sort((a, b) => Math.abs(b.weight) - Math.abs(a.weight) || b.rows - a.rows || a.tag.localeCompare(b.tag));

  return {
    familyId: family.id,
    foldFamily: family.foldFamily,
    orderSet: family.orderSet,
    baseMeanFightScore: round(baseMean),
    trainRows: rows.length,
    totalConfidence: round(confidences.reduce((sum, c) => sum + c, 0)),
    eligibleTagCount: weights.length,
    weights: weights.slice(0, options.maxTagWeights),
  };
}

// ═══════════════════════════════════════════════════════════════════════════
// SCORING & EVALUATION (unchanged — model is still linear-additive)
// ═══════════════════════════════════════════════════════════════════════════

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
    prismConfidence: round(row.prismConfidence),
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
    roots.push({ rootId, rowCount: rows.length, top1: publicTop(top.row, top.score) });
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
    totalConfidence: model.totalConfidence ?? null,
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
    trainFamilyModelDirectional(family, rows, options),
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
  const candidate = aggregate('omnifold_directional_fold_delta_cross_validation', candidateEvals);
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
  return path.join(parsed.dir, `${parsed.name}.directional_fold_deltas.json`);
}

function computeConfidenceDistribution(groups) {
  const confidences = [];
  for (const [, rows] of groups) {
    for (const row of rows) {
      confidences.push(row.prismConfidence);
    }
  }
  confidences.sort((a, b) => a - b);
  if (!confidences.length) return null;
  const mean = confidences.reduce((sum, c) => sum + c, 0) / confidences.length;
  const p10 = confidences[Math.floor(confidences.length * 0.1)];
  const p50 = confidences[Math.floor(confidences.length * 0.5)];
  const p90 = confidences[Math.floor(confidences.length * 0.9)];
  return {
    count: confidences.length,
    min: round(confidences[0]),
    max: round(confidences[confidences.length - 1]),
    mean: round(mean),
    p10: round(p10),
    p50: round(p50),
    p90: round(p90),
  };
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

  // Load prism data (FrostMatrix policy/value signals per root)
  const prismMap = args.directional ? loadPrismData(args.prismData) : new Map();

  const rows = normalizeRows(fight, prismMap, args.prismGamma);
  const groups = groupByRoot(rows);
  const options = {
    folds: args.folds,
    minTagRows: args.minTagRows,
    maxTagWeights: args.maxTagWeights,
    directional: args.directional,
    prismGamma: args.prismGamma,
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

  const algorithm = args.directional
    ? 'per_family_tag_weighted_mean_delta_directional_prism_confidence'
    : 'per_family_tag_mean_delta_with_root_fold_model_selection';

  const output = {
    schemaVersion: SCHEMA_VERSION,
    generatedAt: new Date().toISOString(),
    directional: args.directional,
    condition: {
      source: path.resolve(args.conditionSource),
      runLabel: args.directional
        ? 'scout_new_experiment_omnifold_directional_fold_delta_prism_confidence'
        : 'scout_new_experiment_omnifold_standard_vs_fold_delta_from_gpu_fight_labels',
      changedFields: args.directional
        ? 'directional prism-weighting: X-strong prisms (accepted injections) dominate folds; Y-weak prisms dampened to 0.3x'
        : 'post-hoc trained OmniFold fold-family tag deltas over recorded GPU fight rollout labels only; no runtime behavior changed',
      labCondition: 'scout/new_experiment/uses_existing_gpu_fight_rollout_label_artifact',
      metric: 'root-fold mean forced-candidate GPU self-play fight score',
    },
    prismConfig: args.directional ? {
      enabled: true,
      gamma: args.prismGamma,
      confidenceFormula: 'tanh(gamma * policyMargin / (1 + |valueError|)) * (1 - policyEntropyNormalized)',
      withEntropy: prismMap.size > 0
        ? [...prismMap.values()].some((e) => e.policyEntropyNormalized != null)
        : false,
      preIntegrationProxy: 'acceptedUsefulInjection modulated by (1 - policyEntropyNormalized)',
      prismDataSource: args.prismData || null,
      entropyStats: rows.some((r) => r.policyEntropyNormalized != null) ? {
        min: round(Math.min(...rows.filter((r) => r.policyEntropyNormalized != null).map((r) => r.policyEntropyNormalized))),
        max: round(Math.max(...rows.filter((r) => r.policyEntropyNormalized != null).map((r) => r.policyEntropyNormalized))),
        mean: round(rows.filter((r) => r.policyEntropyNormalized != null).reduce((s, r) => s + r.policyEntropyNormalized, 0) / rows.filter((r) => r.policyEntropyNormalized != null).length),
        availableCount: rows.filter((r) => r.policyEntropyNormalized != null).length,
      } : null,
      distribution: computeConfidenceDistribution(groups),
    } : { enabled: false },
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
      algorithm,
      directional: args.directional,
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
        ? 'trained directional fold deltas show scout lift, but the source fight labels are posthoc and require a frozen heldout GPU fight/gate before promotion'
        : 'trained directional fold deltas did not show root-fold lift over frontier rank on the source GPU fight labels',
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
    directional: args.directional,
    output: outPath,
    rowCount: output.corpus.rowCount,
    rootCount: output.corpus.rootCount,
    activeFoldFamilies: output.corpus.activeFoldFamilies,
    candidateSpecCount: output.corpus.candidateSpecCount,
    prismConfidence: output.prismConfig.distribution,
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
