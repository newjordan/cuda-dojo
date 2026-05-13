#!/usr/bin/env node
import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';

const SCHEMA_VERSION = 'dojo.chrono_o2_omnifold_tactical_pattern_mining.v1';

function usage() {
  return `Usage: node scripts/mine_chrono_o2_omnifold_tactical_patterns.mjs --bridge <bridge.json> --chrono <chrono_o2_tactical_sidecar.json> --omnifold <omnifold_manifest.json> [--out <patterns.json>] [--folds 4] [--max-size 3] [--min-roots 3] [--top-k 24]

Mine cohesive tactical/action/chrono motifs from recorded GPU-derived
logicRayFrontier rows and attach them to OmniFold family attribution. This is
post-hoc pattern discovery only: it does not generate legal moves, run search,
train folds, or promote runtime behavior.
`;
}

function parseArgs(argv) {
  const args = {
    bridge: null,
    chrono: null,
    omnifold: null,
    out: null,
    folds: 4,
    maxSize: 3,
    minRoots: 3,
    minRows: 8,
    topK: 24,
    conditionSource: '/home/frosty40/cuda_dojo/LOGIC_RAY_FRONTIER_RUNTIME_RECEIPT_2026-05-06.md',
    runLabel: 'posthoc_chrono_o2_omnifold_tactical_pattern_mining',
  };
  for (let i = 0; i < argv.length; i += 1) {
    const token = argv[i];
    if (token === '--help' || token === '-h') {
      console.log(usage());
      process.exit(0);
    } else if (token === '--bridge') args.bridge = argv[++i];
    else if (token === '--chrono') args.chrono = argv[++i];
    else if (token === '--omnifold') args.omnifold = argv[++i];
    else if (token === '--out') args.out = argv[++i];
    else if (token === '--folds') args.folds = Number(argv[++i]);
    else if (token === '--max-size') args.maxSize = Number(argv[++i]);
    else if (token === '--min-roots') args.minRoots = Number(argv[++i]);
    else if (token === '--min-rows') args.minRows = Number(argv[++i]);
    else if (token === '--top-k') args.topK = Number(argv[++i]);
    else if (token === '--condition-source') args.conditionSource = argv[++i];
    else if (token === '--run-label') args.runLabel = argv[++i];
    else throw new Error(`unknown argument: ${token}\n${usage()}`);
  }
  if (!args.bridge) throw new Error(`missing --bridge\n${usage()}`);
  if (!args.chrono) throw new Error(`missing --chrono\n${usage()}`);
  if (!args.omnifold) throw new Error(`missing --omnifold\n${usage()}`);
  if (!Number.isInteger(args.folds) || args.folds < 2) throw new Error('--folds must be an integer >= 2');
  if (!Number.isInteger(args.maxSize) || args.maxSize < 1 || args.maxSize > 4) throw new Error('--max-size must be an integer in [1,4]');
  if (!Number.isInteger(args.minRoots) || args.minRoots < 1) throw new Error('--min-roots must be an integer >= 1');
  if (!Number.isInteger(args.minRows) || args.minRows < 1) throw new Error('--min-rows must be an integer >= 1');
  if (!Number.isInteger(args.topK) || args.topK < 1) throw new Error('--top-k must be an integer >= 1');
  return args;
}

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, 'utf8'));
}

function asNumber(value, fallback = 0) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

function clamp(value, min = 0, max = 1) {
  return Math.max(min, Math.min(max, asNumber(value, 0)));
}

function round(value, digits = 6) {
  const factor = 10 ** digits;
  return Math.round(asNumber(value, 0) * factor) / factor;
}

function sha256Text(text) {
  return crypto.createHash('sha256').update(String(text)).digest('hex');
}

function defaultOutPath(chronoPath) {
  const parsed = path.parse(chronoPath);
  const stem = parsed.base.replace(/\.json$/, '');
  return path.join(parsed.dir, `${stem}.omnifold_tactical_patterns.json`);
}

function rowsByHash(bundle) {
  const map = new Map();
  for (const row of Array.isArray(bundle.rows) ? bundle.rows : []) {
    if (row.logicRayFrontierHash) map.set(row.logicRayFrontierHash, row);
  }
  return map;
}

function frontierOf(row) {
  return row.logicRayFrontier || row.pzrgCandidate?.logicRayFrontier || {};
}

function accepted(bridgeRow, chronoRow) {
  const frontier = frontierOf(bridgeRow);
  const injection = bridgeRow.pzrgCandidate?.injection_relevance || {};
  return Boolean(
    frontier.gate?.acceptedUsefulInjection
      || injection.accepted_useful_injection
      || injection.promotion_gate_approved
      || chronoRow?.runtimeChoiceSignal?.acceptedUsefulInjection,
  );
}

function selected(bridgeRow, chronoRow) {
  const frontier = frontierOf(bridgeRow);
  return Boolean(frontier.gate?.selectedMoveInFrontier || chronoRow?.runtimeChoiceSignal?.selectedMoveInFrontier);
}

function tagSafe(text) {
  return String(text || 'unknown').replace(/[^a-zA-Z0-9_.:-]+/g, '_');
}

function bucketReplies(value) {
  const replies = asNumber(value, 0);
  if (replies <= 20) return 'reply_low';
  if (replies <= 36) return 'reply_mid';
  return 'reply_high';
}

function bucketBalance(value) {
  const balance = asNumber(value, 0);
  if (balance > 0) return 'contact_positive';
  if (balance < 0) return 'contact_negative';
  return 'contact_neutral';
}

function addTag(tags, tag, enabled = true) {
  if (enabled) tags.add(tag);
}

function rowTags(bridgeRow, chronoRow) {
  const frontier = frontierOf(bridgeRow);
  const tags = new Set();
  const action = chronoRow.actionFeatures || {};
  const actionFlags = action.flags || {};
  const actionScalars = action.scalars || {};
  const tactical = chronoRow.tacticalContact || {};
  const tacticalFlags = tactical.flags || {};
  const counts = tactical.counts || {};
  const diagnostics = chronoRow.diagnostics || {};

  addTag(tags, `piece:${tagSafe(action.piece)}`, Boolean(action.piece));
  addTag(tags, `family:${tagSafe(action.family)}`, Boolean(action.family));
  addTag(tags, `captured:${tagSafe(action.capturedPiece)}`, action.capturedPiece && action.capturedPiece !== 'empty');
  addTag(tags, 'action:capture', actionFlags.capture);
  addTag(tags, 'action:promotion', actionFlags.promotion);
  addTag(tags, 'action:diagonal', actionFlags.diagonal);
  addTag(tags, 'action:center16', actionFlags.center16);
  addTag(tags, 'action:forward', actionFlags.forwardPositive);
  addTag(tags, 'action:backward_or_level', actionFlags.backwardOrLevel);
  addTag(tags, 'action:home_departure', actionFlags.homeDeparture);
  addTag(tags, 'action:same_file', actionFlags.sameFile);
  addTag(tags, 'action:same_rank', actionFlags.sameRank);
  addTag(tags, 'action:long_move', asNumber(actionScalars.distance, 0) >= 2.5);

  addTag(tags, 'tactical:gpu_verified', tactical.gpuVerified === true);
  addTag(tags, 'tactical:defended_before', tacticalFlags.destinationDefendedBefore);
  addTag(tags, 'tactical:attacked_before', tacticalFlags.destinationAttackedBefore);
  addTag(tags, 'tactical:defended_after', tacticalFlags.destinationDefendedAfter);
  addTag(tags, 'tactical:attacked_after', tacticalFlags.destinationAttackedAfter);
  addTag(tags, 'tactical:safe_after', tacticalFlags.destinationDefendedAfter && !tacticalFlags.destinationAttackedAfter);
  addTag(tags, 'tactical:loose_after', tacticalFlags.destinationAttackedAfter && !tacticalFlags.destinationDefendedAfter);
  addTag(tags, 'tactical:gives_check', tacticalFlags.givesCheckAfter);
  addTag(tags, 'tactical:capture_like', tacticalFlags.captureLike);
  addTag(tags, 'tactical:defended_capture', tacticalFlags.captureLike && tacticalFlags.destinationDefendedAfter);
  addTag(tags, 'tactical:attacked_capture', tacticalFlags.captureLike && tacticalFlags.destinationAttackedAfter);
  addTag(tags, 'tactical:safe_capture', tacticalFlags.captureLike && tacticalFlags.destinationDefendedAfter && !tacticalFlags.destinationAttackedAfter);
  addTag(tags, 'tactical:mover_in_check', tacticalFlags.moverInCheckBefore);
  addTag(tags, `tactical:${bucketReplies(counts.opponentLegalRepliesAfter)}`);
  addTag(tags, `tactical:${bucketBalance(counts.contactBalanceAfter)}`);

  addTag(tags, `chrono:phase:${tagSafe(chronoRow.timePhase?.phase)}`, Boolean(chronoRow.timePhase?.phase));
  addTag(tags, `chrono:pressure:${tagSafe(chronoRow.pressureDrift?.bucket)}`, Boolean(chronoRow.pressureDrift?.bucket));
  addTag(tags, `chrono:relation:${tagSafe(chronoRow.relationDrift?.bucket)}`, Boolean(chronoRow.relationDrift?.bucket));
  addTag(tags, `chrono:contortion:${tagSafe(chronoRow.pathContortion?.bucket)}`, Boolean(chronoRow.pathContortion?.bucket));
  addTag(tags, `chrono:uncertainty:${tagSafe(chronoRow.uncertainty?.bucket)}`, Boolean(chronoRow.uncertainty?.bucket));
  addTag(tags, 'chrono:stable_score', asNumber(diagnostics.stabilityScore, 0) >= 0.7);
  addTag(tags, 'chrono:unstable_score', asNumber(diagnostics.stabilityScore, 0) < 0.4);

  addTag(tags, `pzrg:pressure:${tagSafe(frontier.pzrg4d?.pressure)}`, Boolean(frontier.pzrg4d?.pressure));
  addTag(tags, `pzrg:expression:${tagSafe(frontier.pzrg4d?.chessExpression)}`, Boolean(frontier.pzrg4d?.chessExpression));
  return [...tags].sort();
}

function activeFoldFamilies(omnifold) {
  return (Array.isArray(omnifold.foldFamilies) ? omnifold.foldFamilies : [])
    .filter((family) => family.status === 'active_frontier_attachable')
    .map((family) => ({
      id: family.id,
      foldFamily: family.foldFamily,
      orderSet: family.orderSet,
      activeVariantCount: family.activeVariantCount,
    }));
}

function joinRows(bridgeRows, chronoByHash) {
  const items = [];
  const missingChrono = [];
  for (let index = 0; index < bridgeRows.length; index += 1) {
    const bridgeRow = bridgeRows[index];
    const chronoRow = chronoByHash.get(bridgeRow.logicRayFrontierHash);
    const frontier = frontierOf(bridgeRow);
    if (!chronoRow) {
      missingChrono.push({ rowIndex: index, hash: bridgeRow.logicRayFrontierHash });
      continue;
    }
    const tags = rowTags(bridgeRow, chronoRow);
    items.push({
      bridgeRow,
      chronoRow,
      hash: bridgeRow.logicRayFrontierHash,
      bridgeId: bridgeRow.bridgeId,
      rootId: frontier.rootId || bridgeRow.rootId || chronoRow.rootId || 'root',
      move: frontier.move || bridgeRow.move || chronoRow.move,
      originalRank: Math.max(1, asNumber(frontier.rank || bridgeRow.rank || chronoRow.rank, 1)),
      accepted: accepted(bridgeRow, chronoRow),
      selected: selected(bridgeRow, chronoRow),
      tags,
      tagSet: new Set(tags),
    });
  }
  return { items, missingChrono };
}

function combinations(values, size, start = 0, prefix = [], out = []) {
  if (prefix.length === size) {
    out.push([...prefix]);
    return out;
  }
  for (let i = start; i <= values.length - (size - prefix.length); i += 1) {
    prefix.push(values[i]);
    combinations(values, size, i + 1, prefix, out);
    prefix.pop();
  }
  return out;
}

function motifId(tags) {
  return `motif_${sha256Text(tags.join('|')).slice(0, 16)}`;
}

function motifKey(tags) {
  return tags.join('|');
}

function generateCandidateMotifs(items, args) {
  const stats = new Map();
  for (const item of items) {
    for (let size = 1; size <= args.maxSize; size += 1) {
      for (const combo of combinations(item.tags, size)) {
        const key = motifKey(combo);
        if (!stats.has(key)) {
          stats.set(key, {
            id: motifId(combo),
            tags: combo,
            rows: 0,
            acceptedRows: 0,
            selectedRows: 0,
            rootIds: new Set(),
            acceptedRootIds: new Set(),
            examples: [],
          });
        }
        const entry = stats.get(key);
        entry.rows += 1;
        if (item.accepted) entry.acceptedRows += 1;
        if (item.selected) entry.selectedRows += 1;
        entry.rootIds.add(item.rootId);
        if (item.accepted) entry.acceptedRootIds.add(item.rootId);
        if (entry.examples.length < 8) {
          entry.examples.push({
            rootId: item.rootId,
            move: item.move,
            originalRank: item.originalRank,
            acceptedUsefulInjection: item.accepted,
            selectedMoveInFrontier: item.selected,
            hash: item.hash,
          });
        }
      }
    }
  }
  return [...stats.values()].filter((entry) => (
    entry.rows >= args.minRows
    && entry.rootIds.size >= args.minRoots
  )).map((entry) => ({
    ...entry,
    rootCount: entry.rootIds.size,
    acceptedRootCount: entry.acceptedRootIds.size,
  }));
}

function groupByRoot(items) {
  const groups = new Map();
  for (const item of items) {
    if (!groups.has(item.rootId)) groups.set(item.rootId, []);
    groups.get(item.rootId).push(item);
  }
  return [...groups.entries()].sort(([a], [b]) => String(a).localeCompare(String(b)));
}

function filterGroups(groups, rootIds) {
  const allowed = new Set(rootIds);
  return groups.filter(([rootId]) => allowed.has(rootId));
}

function matchesMotif(item, motif) {
  return motif.tags.every((tag) => item.tagSet.has(tag));
}

function evaluateRankedGroups(groups, scoreFn, name) {
  const roots = [];
  let top1Accepted = 0;
  let top3Accepted = 0;
  let selectedTop1 = 0;
  let acceptedRankSum = 0;
  let acceptedRankCount = 0;
  for (const [rootId, rows] of groups) {
    const ranked = rows.map((item) => ({ item, score: scoreFn(item) }))
      .sort((a, b) => {
        const delta = b.score - a.score;
        if (Math.abs(delta) > 1e-12) return delta;
        return a.item.originalRank - b.item.originalRank;
      })
      .map((entry, index) => ({ ...entry, comparisonRank: index + 1 }));
    const firstAccepted = ranked.find((entry) => entry.item.accepted);
    if (ranked[0]?.item.accepted) top1Accepted += 1;
    if (ranked.slice(0, 3).some((entry) => entry.item.accepted)) top3Accepted += 1;
    if (ranked[0]?.item.selected) selectedTop1 += 1;
    if (firstAccepted) {
      acceptedRankSum += firstAccepted.comparisonRank;
      acceptedRankCount += 1;
    }
    roots.push({
      rootId,
      rowCount: rows.length,
      top1: ranked[0] ? publicRankedEntry(ranked[0]) : null,
      acceptedCandidateBestRank: firstAccepted?.comparisonRank || null,
    });
  }
  return {
    name,
    rootCount: groups.length,
    top1AcceptedUsefulInjections: top1Accepted,
    top3AcceptedUsefulInjections: top3Accepted,
    selectedMoveTop1: selectedTop1,
    acceptedRankSum,
    acceptedRankCount,
    meanAcceptedCandidateRank: acceptedRankCount ? round(acceptedRankSum / acceptedRankCount) : null,
    roots,
  };
}

function publicRankedEntry(entry) {
  return {
    bridgeId: entry.item.bridgeId,
    hash: entry.item.hash,
    move: entry.item.move,
    originalRank: entry.item.originalRank,
    comparisonRank: entry.comparisonRank,
    score: round(entry.score),
    acceptedUsefulInjection: entry.item.accepted,
    selectedMoveInFrontier: entry.item.selected,
  };
}

function evaluateFrontier(groups) {
  return evaluateRankedGroups(groups, (item) => -item.originalRank, 'frontier_rank');
}

function evaluateMotif(motif, groups) {
  return evaluateRankedGroups(groups, (item) => (matchesMotif(item, motif) ? 1 : 0), motif.id);
}

function metricDelta(candidate, baseline) {
  return {
    top1: candidate.top1AcceptedUsefulInjections - baseline.top1AcceptedUsefulInjections,
    top3: candidate.top3AcceptedUsefulInjections - baseline.top3AcceptedUsefulInjections,
    meanAcceptedCandidateRank: (
      candidate.meanAcceptedCandidateRank != null && baseline.meanAcceptedCandidateRank != null
        ? round(candidate.meanAcceptedCandidateRank - baseline.meanAcceptedCandidateRank)
        : null
    ),
  };
}

function objective(metrics) {
  return (
    metrics.top1AcceptedUsefulInjections * 10000
    + metrics.top3AcceptedUsefulInjections * 1000
    - asNumber(metrics.meanAcceptedCandidateRank, 999) * 10
  );
}

function motifPublic(motif, baselineRows) {
  const acceptedRate = motif.rows ? motif.acceptedRows / motif.rows : 0;
  const baselineRate = baselineRows ? motif.acceptedRows / baselineRows : 0;
  return {
    id: motif.id,
    tags: motif.tags,
    size: motif.tags.length,
    rowCount: motif.rows,
    rootCount: motif.rootCount,
    acceptedRows: motif.acceptedRows,
    acceptedRootCount: motif.acceptedRootCount,
    acceptedRate: round(acceptedRate),
    corpusAcceptedShare: round(baselineRate),
    selectedRows: motif.selectedRows,
    examples: motif.examples,
  };
}

function rankMotifs(motifs, groups, baseline, limit, corpusRows) {
  return motifs.map((motif) => {
    const metrics = evaluateMotif(motif, groups);
    const delta = metricDelta(metrics, baseline);
    const enrichment = motif.rows ? motif.acceptedRows / motif.rows : 0;
    return {
      motif,
      metrics,
      delta,
      sortScore: objective(metrics) + enrichment * 100 + motif.rootCount,
    };
  }).sort((a, b) => b.sortScore - a.sortScore || a.motif.id.localeCompare(b.motif.id))
    .slice(0, limit)
    .map((entry) => ({
      ...motifPublic(entry.motif, corpusRows),
      metrics: publicMetrics(entry.metrics),
      deltaVsFrontierRank: entry.delta,
    }));
}

function publicMetrics(metrics) {
  return {
    rootCount: metrics.rootCount,
    top1AcceptedUsefulInjections: metrics.top1AcceptedUsefulInjections,
    top3AcceptedUsefulInjections: metrics.top3AcceptedUsefulInjections,
    selectedMoveTop1: metrics.selectedMoveTop1,
    meanAcceptedCandidateRank: metrics.meanAcceptedCandidateRank,
  };
}

function rootFolds(rootIds, foldCount) {
  return Array.from({ length: foldCount }, (_, fold) => (
    rootIds.filter((_, index) => index % foldCount === fold)
  )).filter((foldRoots) => foldRoots.length > 0);
}

function aggregateEvaluations(name, evaluations) {
  const rootCount = evaluations.reduce((sum, item) => sum + item.rootCount, 0);
  const top1 = evaluations.reduce((sum, item) => sum + item.top1AcceptedUsefulInjections, 0);
  const top3 = evaluations.reduce((sum, item) => sum + item.top3AcceptedUsefulInjections, 0);
  const selectedTop1 = evaluations.reduce((sum, item) => sum + item.selectedMoveTop1, 0);
  const acceptedRankSum = evaluations.reduce((sum, item) => sum + item.acceptedRankSum, 0);
  const acceptedRankCount = evaluations.reduce((sum, item) => sum + item.acceptedRankCount, 0);
  return {
    name,
    rootCount,
    top1AcceptedUsefulInjections: top1,
    top3AcceptedUsefulInjections: top3,
    selectedMoveTop1: selectedTop1,
    acceptedRankSum,
    acceptedRankCount,
    meanAcceptedCandidateRank: acceptedRankCount ? round(acceptedRankSum / acceptedRankCount) : null,
    roots: evaluations.flatMap((item) => item.roots || []),
  };
}

function selectBestTrainMotif(trainMotifs, trainGroups, trainBaseline) {
  const ranked = trainMotifs.map((motif) => {
    const metrics = evaluateMotif(motif, trainGroups);
    const delta = metricDelta(metrics, trainBaseline);
    return { motif, metrics, delta, score: objective(metrics) + motif.acceptedRows * 10 + motif.rootCount };
  }).sort((a, b) => b.score - a.score || a.motif.id.localeCompare(b.motif.id));
  return ranked[0] || null;
}

function topRootDiffs(candidateEval, baselineEval, limit = 12) {
  const baselineByRoot = new Map((baselineEval.roots || []).map((root) => [root.rootId, root]));
  return (candidateEval.roots || []).map((root) => {
    const baseline = baselineByRoot.get(root.rootId);
    return {
      rootId: root.rootId,
      baselineTop1: baseline?.top1 || null,
      candidateTop1: root.top1 || null,
      baselineAcceptedBestRank: baseline?.acceptedCandidateBestRank || null,
      candidateAcceptedBestRank: root.acceptedCandidateBestRank || null,
      acceptedRankDelta: (
        root.acceptedCandidateBestRank != null && baseline?.acceptedCandidateBestRank != null
          ? root.acceptedCandidateBestRank - baseline.acceptedCandidateBestRank
          : null
      ),
    };
  }).filter((item) => item.acceptedRankDelta !== 0 || item.baselineTop1?.acceptedUsefulInjection !== item.candidateTop1?.acceptedUsefulInjection)
    .slice(0, limit);
}

function runCrossValidation(items, allGroups, allMotifs, args) {
  const rootIds = allGroups.map(([rootId]) => rootId);
  const folds = rootFolds(rootIds, Math.min(args.folds, rootIds.length));
  const foldReports = [];
  const candidateEvals = [];
  const baselineEvals = [];
  for (let foldIndex = 0; foldIndex < folds.length; foldIndex += 1) {
    const evalRoots = folds[foldIndex];
    const trainRoots = rootIds.filter((rootId) => !evalRoots.includes(rootId));
    const trainGroups = filterGroups(allGroups, trainRoots);
    const evalGroups = filterGroups(allGroups, evalRoots);
    const trainRootSet = new Set(trainRoots);
    const trainItems = items.filter((item) => trainRootSet.has(item.rootId));
    const trainMotifs = generateCandidateMotifs(trainItems, args);
    const trainBaseline = evaluateFrontier(trainGroups);
    const evalBaseline = evaluateFrontier(evalGroups);
    const selected = selectBestTrainMotif(trainMotifs, trainGroups, trainBaseline);
    const trainCandidate = selected ? evaluateMotif(selected.motif, trainGroups) : trainBaseline;
    const evalCandidate = selected ? evaluateMotif(selected.motif, evalGroups) : evalBaseline;
    candidateEvals.push(evalCandidate);
    baselineEvals.push(evalBaseline);
    foldReports.push({
      fold: foldIndex,
      trainRootCount: trainRoots.length,
      evalRootCount: evalRoots.length,
      trainMotifCount: trainMotifs.length,
      selectedMotif: selected ? {
        ...motifPublic(selected.motif, trainItems.length),
        trainMetrics: publicMetrics(selected.metrics),
        trainDeltaVsFrontierRank: selected.delta,
      } : null,
      evalBaseline: publicMetrics(evalBaseline),
      evalCandidate: publicMetrics(evalCandidate),
      evalDeltaVsBaseline: metricDelta(evalCandidate, evalBaseline),
    });
  }
  const baseline = aggregateEvaluations('frontier_rank_cross_validation_baseline', baselineEvals);
  const candidate = aggregateEvaluations('omnifold_tactical_pattern_cross_validation', candidateEvals);
  return {
    baseline,
    candidate,
    deltaVsFrontierRank: metricDelta(candidate, baseline),
    observedLift: (
      candidate.top1AcceptedUsefulInjections > baseline.top1AcceptedUsefulInjections
      || candidate.top3AcceptedUsefulInjections > baseline.top3AcceptedUsefulInjections
      || (
        candidate.meanAcceptedCandidateRank != null
        && baseline.meanAcceptedCandidateRank != null
        && candidate.meanAcceptedCandidateRank < baseline.meanAcceptedCandidateRank
      )
    ),
    rootDiffs: topRootDiffs(candidate, baseline),
    folds: foldReports,
  };
}

function main() {
  const args = parseArgs(process.argv.slice(2));
  const bridgePath = path.resolve(args.bridge);
  const chronoPath = path.resolve(args.chrono);
  const omnifoldPath = path.resolve(args.omnifold);
  const bridge = readJson(bridgePath);
  const chrono = readJson(chronoPath);
  const omnifold = readJson(omnifoldPath);
  const bridgeRows = Array.isArray(bridge.rows) ? bridge.rows : [];
  const chronoByHash = rowsByHash(chrono);
  const { items, missingChrono } = joinRows(bridgeRows, chronoByHash);
  const allGroups = groupByRoot(items);
  const baseline = evaluateFrontier(allGroups);
  const motifs = generateCandidateMotifs(items, args);
  const topGlobalPatterns = rankMotifs(motifs, allGroups, baseline, args.topK, items.length);
  const crossValidation = runCrossValidation(items, allGroups, motifs, args);
  const activeFamilies = activeFoldFamilies(omnifold);
  const acceptedRows = items.filter((item) => item.accepted).length;
  const output = {
    schemaVersion: SCHEMA_VERSION,
    generatedAt: new Date().toISOString(),
    condition: {
      source: path.resolve(args.conditionSource),
      runLabel: args.runLabel,
      changedFields: 'post-hoc motif mining over recorded GPU-derived bridge/chrono/omnifold artifacts only; no runtime behavior changed',
      labCondition: 'posthoc/subset when input bridge/chrono is a subset',
      metric: 'motif accepted-useful enrichment and root-fold top-k accepted useful injections',
    },
    sources: {
      bridgePath,
      bridgeSchemaVersion: bridge.schemaVersion || null,
      bridgeRows: bridgeRows.length,
      chronoPath,
      chronoSchemaVersion: chrono.schemaVersion || null,
      chronoRows: Array.isArray(chrono.rows) ? chrono.rows.length : 0,
      omnifoldPath,
      omnifoldSchemaVersion: omnifold.schemaVersion || null,
      joinedRows: items.length,
      missingChronoRows: missingChrono.length,
    },
    miningPolicy: {
      algorithm: 'enumerate_tag_conjunctions_then_rank_by_root_group_topk_accepted_useful_injection',
      maxMotifSize: args.maxSize,
      minRoots: args.minRoots,
      minRows: args.minRows,
      candidateMotifs: motifs.length,
      topK: args.topK,
      noRuntimePromotion: true,
      excludedRuntimeBehavior: [
        'no legal move generation',
        'no search',
        'no CPU chess referee',
        'no trained OmniFold delta claim',
      ],
    },
    omnifoldAttribution: {
      activeFoldFamilies: activeFamilies,
      activeFoldFamilyCount: activeFamilies.length,
      blockedFoldFamilyCount: (Array.isArray(omnifold.foldFamilies) ? omnifold.foldFamilies : []).length - activeFamilies.length,
      invarianceStatus: 'frontier_attachable_only_untrained_omnifold_delta_not_claimed',
      offManifoldAuditRows: omnifold.aggregate?.offManifoldAuditRows ?? null,
    },
    corpus: {
      rowCount: items.length,
      rootCount: allGroups.length,
      acceptedRows,
      acceptedRate: items.length ? round(acceptedRows / items.length) : 0,
      selectedRows: items.filter((item) => item.selected).length,
      tacticalGpuVerifiedRows: items.filter((item) => item.tagSet.has('tactical:gpu_verified')).length,
    },
    baseline: {
      frontierRank: publicMetrics(baseline),
    },
    topGlobalPatterns,
    crossValidation: {
      baseline: publicMetrics(crossValidation.baseline),
      candidate: publicMetrics(crossValidation.candidate),
      deltaVsFrontierRank: crossValidation.deltaVsFrontierRank,
      observedLift: crossValidation.observedLift,
      rootDiffs: crossValidation.rootDiffs,
      folds: crossValidation.folds,
    },
    promotionPolicy: {
      status: 'not_promoted',
      reason: crossValidation.observedLift
        ? 'post-hoc motif miner found root-fold lift; freeze motif condition and rerun heldout GPU gate before promotion'
        : 'motif miner did not show root-fold lift over frontier rank',
      blockers: [
        'posthoc_pattern_mining_not_runtime_evidence',
        'trained_omnifold_delta_not_measured',
        ...(crossValidation.observedLift ? ['heldout_gpu_gate_not_rerun_with_frozen_motif_condition'] : ['pattern_miner_no_cross_validated_lift']),
      ],
      requiredNextEvidence: crossValidation.observedLift
        ? [
          'freeze selected motif condition without looking at new gate labels',
          'resolve off-manifold audit for selected OmniFold family',
          'rerun heldout GPU gate and measure accepted useful injection lift',
        ]
        : [
          'add trained standard-vs-OmniFold fold deltas or GPU fight-outcome labels',
          'rerun motif mining with those labels and require root-fold lift',
        ],
    },
  };
  const outPath = path.resolve(args.out || defaultOutPath(chronoPath));
  fs.mkdirSync(path.dirname(outPath), { recursive: true });
  fs.writeFileSync(outPath, `${JSON.stringify(output, null, 2)}\n`);
  console.log(JSON.stringify({
    ok: true,
    output: outPath,
    joinedRows: items.length,
    rootCount: allGroups.length,
    candidateMotifs: motifs.length,
    topGlobalPatterns: topGlobalPatterns.length,
    activeFoldFamilies: activeFamilies.length,
    baselineTop1: baseline.top1AcceptedUsefulInjections,
    crossValidatedTop1: crossValidation.candidate.top1AcceptedUsefulInjections,
    crossValidatedDelta: crossValidation.deltaVsFrontierRank,
    observedLift: crossValidation.observedLift,
    promote: false,
  }, null, 2));
}

try {
  main();
} catch (error) {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
}
