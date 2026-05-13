#!/usr/bin/env node
import fs from 'node:fs';
import path from 'node:path';

const SCHEMA_VERSION = 'dojo.chrono_o2_tactical_relation_objective_learning.v1';

function usage() {
  return `Usage: node scripts/learn_chrono_o2_tactical_relation_objective.mjs --bridge <bridge.json> --chrono <chrono_o2_tactical_sidecar.json> [--out <objective.json>] [--folds 4] [--rounds 4] [--beam-size 24]

Learn a named tactical-relation scoring objective from recorded GPU-derived
PZRG_CHRONO_O2 tacticalContact rows. This does not generate legal moves, run
search, verify transitions, or promote runtime behavior.
`;
}

function parseArgs(argv) {
  const args = {
    bridge: null,
    chrono: null,
    out: null,
    folds: 4,
    rounds: 4,
    beamSize: 24,
    conditionSource: '/home/frosty40/cuda_dojo/LOGIC_RAY_FRONTIER_RUNTIME_RECEIPT_2026-05-06.md',
    runLabel: 'posthoc_pzrg_chrono_o2_tactical_relation_objective_learning',
  };
  for (let i = 0; i < argv.length; i += 1) {
    const token = argv[i];
    if (token === '--help' || token === '-h') {
      console.log(usage());
      process.exit(0);
    } else if (token === '--bridge') args.bridge = argv[++i];
    else if (token === '--chrono') args.chrono = argv[++i];
    else if (token === '--out') args.out = argv[++i];
    else if (token === '--folds') args.folds = Number(argv[++i]);
    else if (token === '--rounds') args.rounds = Number(argv[++i]);
    else if (token === '--beam-size') args.beamSize = Number(argv[++i]);
    else if (token === '--condition-source') args.conditionSource = argv[++i];
    else if (token === '--run-label') args.runLabel = argv[++i];
    else throw new Error(`unknown argument: ${token}\n${usage()}`);
  }
  if (!args.bridge) throw new Error(`missing --bridge\n${usage()}`);
  if (!args.chrono) throw new Error(`missing --chrono\n${usage()}`);
  if (!Number.isInteger(args.folds) || args.folds < 2) throw new Error('--folds must be an integer >= 2');
  if (!Number.isInteger(args.rounds) || args.rounds < 1) throw new Error('--rounds must be an integer >= 1');
  if (!Number.isInteger(args.beamSize) || args.beamSize < 1) throw new Error('--beam-size must be an integer >= 1');
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

function defaultOutPath(chronoPath) {
  const parsed = path.parse(chronoPath);
  const stem = parsed.base.replace(/\.json$/, '');
  return path.join(parsed.dir, `${stem}.tactical_relation_objective.json`);
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

function groupByRoot(items) {
  const groups = new Map();
  for (const item of items) {
    const rootId = item.rootId || 'root';
    if (!groups.has(rootId)) groups.set(rootId, []);
    groups.get(rootId).push(item);
  }
  return [...groups.entries()].sort(([a], [b]) => String(a).localeCompare(String(b)));
}

function filterGroups(groups, rootIds) {
  const allowed = new Set(rootIds);
  return groups.filter(([rootId]) => allowed.has(rootId));
}

function boolFeature(value) {
  return value ? 1 : 0;
}

function oneHot(value, expected) {
  return value === expected ? 1 : 0;
}

function signedBool(positive, negative) {
  return boolFeature(positive) - boolFeature(negative);
}

function featureVector(bridgeRow, chronoRow) {
  const frontier = frontierOf(bridgeRow);
  const action = chronoRow.actionFeatures || {};
  const actionFlags = action.flags || {};
  const actionScalars = action.scalars || {};
  const tactical = chronoRow.tacticalContact || {};
  const tacticalFlags = tactical.flags || {};
  const counts = tactical.counts || {};
  const vectors = chronoRow.vectors || {};
  const diagnostics = chronoRow.diagnostics || {};
  const replies = asNumber(counts.opponentLegalRepliesAfter, 0);
  const lowReply = 1 - clamp(replies / 64, 0, 1);
  const contactBalance = clamp(asNumber(counts.contactBalanceAfter, 0), -1, 1);
  const normDrift = clamp(Math.abs(asNumber(diagnostics.normDrift, 0)) / 8, 0, 1);
  const fContNorm = Math.sqrt((vectors.F_cont || []).reduce((sum, value) => sum + asNumber(value, 0) ** 2, 0));
  const nDistance = asNumber(vectors.n?.[3], 0);
  return {
    chrono_stability: clamp(diagnostics.stabilityScore, 0, 1),
    chrono_relation: clamp(chronoRow.relationDrift?.score, 0, 1),
    chrono_pressure: clamp(chronoRow.pressureDrift?.score, 0, 1),
    chrono_low_uncertainty: 1 - clamp(chronoRow.uncertainty?.score, 0, 1),
    chrono_low_contortion: 1 - clamp(chronoRow.pathContortion?.score, 0, 1),
    chrono_low_norm_drift: 1 - normDrift,
    chrono_fcont_norm: clamp(fContNorm / 2, 0, 1),
    chrono_move_distance: clamp(nDistance, 0, 1),

    action_capture_or_promotion: boolFeature(actionFlags.capture || actionFlags.promotion),
    action_capture: boolFeature(actionFlags.capture),
    action_promotion: boolFeature(actionFlags.promotion),
    action_diagonal: boolFeature(actionFlags.diagonal),
    action_center16: boolFeature(actionFlags.center16),
    action_forward: boolFeature(actionFlags.forwardPositive),
    action_backward_or_level: boolFeature(actionFlags.backwardOrLevel),
    action_home_departure: boolFeature(actionFlags.homeDeparture),
    action_same_file: boolFeature(actionFlags.sameFile),
    action_same_rank: boolFeature(actionFlags.sameRank),
    action_distance: clamp(asNumber(actionScalars.distance, 0) / 8, 0, 1),
    action_piece_pawn: oneHot(action.piece, 'pawn'),
    action_piece_knight: oneHot(action.piece, 'knight'),
    action_piece_bishop: oneHot(action.piece, 'bishop'),
    action_piece_rook: oneHot(action.piece, 'rook'),
    action_piece_queen: oneHot(action.piece, 'queen'),
    action_piece_king: oneHot(action.piece, 'king'),
    action_captured_any: action.capturedPiece && action.capturedPiece !== 'empty' ? 1 : 0,

    tactical_defended_before: boolFeature(tacticalFlags.destinationDefendedBefore),
    tactical_attacked_before: boolFeature(tacticalFlags.destinationAttackedBefore),
    tactical_defended_after: boolFeature(tacticalFlags.destinationDefendedAfter),
    tactical_attacked_after: boolFeature(tacticalFlags.destinationAttackedAfter),
    tactical_safe_after: boolFeature(tacticalFlags.destinationDefendedAfter && !tacticalFlags.destinationAttackedAfter),
    tactical_loose_after: boolFeature(tacticalFlags.destinationAttackedAfter && !tacticalFlags.destinationDefendedAfter),
    tactical_contact_balance: contactBalance,
    tactical_low_opponent_replies: lowReply,
    tactical_gives_check: boolFeature(tacticalFlags.givesCheckAfter),
    tactical_capture_like: boolFeature(tacticalFlags.captureLike),
    tactical_promotion: boolFeature(tacticalFlags.promotion),
    tactical_mover_in_check: boolFeature(tacticalFlags.moverInCheckBefore),
    tactical_defended_capture: boolFeature(tacticalFlags.captureLike && tacticalFlags.destinationDefendedAfter),
    tactical_attacked_capture: boolFeature(tacticalFlags.captureLike && tacticalFlags.destinationAttackedAfter),
    tactical_check_low_reply: boolFeature(tacticalFlags.givesCheckAfter) * lowReply,
    tactical_safe_capture: boolFeature(tacticalFlags.captureLike && tacticalFlags.destinationDefendedAfter && !tacticalFlags.destinationAttackedAfter),
    tactical_pressure_balance: signedBool(tacticalFlags.destinationDefendedAfter, tacticalFlags.destinationAttackedAfter),

    relation_agreement_pressure: String(frontier.pzrg4d?.pressure || '').includes('agreement') ? 1 : 0,
    relation_conversion_pressure: String(frontier.pzrg4d?.pressure || '').includes('conversion') ? 1 : 0,
  };
}

function featureNamesFromItems(items) {
  const names = new Set();
  for (const item of items) {
    for (const name of Object.keys(item.features || {})) names.add(name);
  }
  return [...names].sort();
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
      tacticalGpuVerified: chronoRow.tacticalContact?.gpuVerified === true,
      features: featureVector(bridgeRow, chronoRow),
    });
  }
  return { items, missingChrono };
}

function makeSpec(id, description, weights = {}, history = []) {
  return {
    id,
    description,
    weights: Object.fromEntries(Object.entries(weights).map(([name, value]) => [name, asNumber(value, 0)])),
    history,
  };
}

function specKey(spec) {
  return JSON.stringify(Object.fromEntries(
    Object.entries(spec.weights).sort(([a], [b]) => a.localeCompare(b)).map(([name, value]) => [name, round(value, 4)]),
  ));
}

function seedSpecs() {
  return [
    makeSpec('zero_relation_objective', 'Zero tactical relation objective; frontier rank tie-break only.'),
    makeSpec('chrono_stability_relation', 'Prefer stable, low-uncertainty chronometric rows.', {
      chrono_stability: 1,
      chrono_low_uncertainty: 1,
      chrono_low_contortion: 0.5,
      chrono_low_norm_drift: 0.5,
    }),
    makeSpec('action_gap_relation', 'Feature-gap prior: capture, diagonal, forward, center.', {
      action_capture_or_promotion: 1,
      action_diagonal: 1,
      action_forward: 1,
      action_center16: 1,
      action_backward_or_level: -1,
    }),
    makeSpec('tactical_safety_relation', 'Prefer defended contact and low opponent replies.', {
      tactical_defended_after: 1,
      tactical_attacked_after: -1,
      tactical_contact_balance: 1,
      tactical_low_opponent_replies: 1,
    }),
    makeSpec('tactical_force_relation', 'Prefer forcing check/capture contact.', {
      tactical_gives_check: 1,
      tactical_capture_like: 1,
      tactical_check_low_reply: 1,
      tactical_attacked_capture: -0.5,
    }),
    makeSpec('bishop_capture_relation', 'Prior from feature-gap atlas: bishop/diagonal/capture/center.', {
      action_piece_bishop: 1,
      action_capture: 1,
      action_diagonal: 1,
      action_center16: 0.5,
      tactical_attacked_capture: -0.5,
    }),
  ];
}

function mutateSpec(spec, feature, delta) {
  const nextWeights = { ...spec.weights };
  nextWeights[feature] = round(asNumber(nextWeights[feature], 0) + delta);
  return makeSpec(
    `${spec.id}.${feature}${delta >= 0 ? 'p' : 'm'}${String(Math.abs(delta)).replace('.', '_')}`,
    `Beam mutation ${feature} ${delta >= 0 ? '+' : ''}${delta}`,
    nextWeights,
    [...spec.history, { feature, delta }],
  );
}

function scoreItem(item, spec) {
  let score = 0;
  for (const [name, weight] of Object.entries(spec.weights)) {
    score += asNumber(weight, 0) * asNumber(item.features?.[name], 0);
  }
  return score;
}

function evaluateRankedGroups(groups, scoreFn, name) {
  const roots = [];
  let top1Accepted = 0;
  let top3Accepted = 0;
  let selectedTop1 = 0;
  let acceptedRankSum = 0;
  let acceptedRankCount = 0;
  for (const [rootId, rows] of groups) {
    const scored = rows.map((item) => ({ item, score: scoreFn(item) }));
    const ranked = scored.sort((a, b) => {
      const delta = b.score - a.score;
      if (Math.abs(delta) > 1e-12) return delta;
      return a.item.originalRank - b.item.originalRank;
    }).map((entry, index) => ({
      ...entry,
      comparisonRank: index + 1,
    }));
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

function evaluateSpec(spec, groups) {
  return evaluateRankedGroups(groups, (item) => scoreItem(item, spec), spec.id);
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

function compareEvaluated(a, b) {
  const diff = objective(b.metrics) - objective(a.metrics);
  if (diff !== 0) return diff;
  return a.spec.id.localeCompare(b.spec.id);
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

function publicSpec(spec) {
  const weights = Object.entries(spec.weights)
    .filter(([, value]) => Math.abs(asNumber(value, 0)) > 1e-9)
    .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]) || a[0].localeCompare(b[0]))
    .map(([feature, weight]) => ({ feature, weight: round(weight, 6) }));
  return {
    id: spec.id,
    description: spec.description,
    nonZeroWeights: weights.length,
    weights,
    history: spec.history.slice(-20),
  };
}

function runBeamSearch(trainGroups, featureNames, args) {
  const deltas = [-2, -1, -0.5, 0.5, 1, 2];
  let beam = seedSpecs().map((spec) => ({ spec, metrics: evaluateSpec(spec, trainGroups) }));
  const seen = new Set(beam.map((item) => specKey(item.spec)));
  const rounds = [];
  for (let roundIndex = 0; roundIndex < args.rounds; roundIndex += 1) {
    const evaluated = [...beam];
    for (const item of beam) {
      for (const feature of featureNames) {
        for (const delta of deltas) {
          const next = mutateSpec(item.spec, feature, delta);
          const mutatedValue = asNumber(next.weights[feature], 0);
          if (Math.abs(mutatedValue) > 4) continue;
          const key = specKey(next);
          if (seen.has(key)) continue;
          seen.add(key);
          evaluated.push({ spec: next, metrics: evaluateSpec(next, trainGroups) });
        }
      }
    }
    evaluated.sort(compareEvaluated);
    beam = evaluated.slice(0, args.beamSize);
    rounds.push({
      round: roundIndex + 1,
      evaluatedCandidates: evaluated.length,
      retainedBeam: beam.length,
      best: {
        spec: publicSpec(beam[0].spec),
        metrics: publicMetrics(beam[0].metrics),
      },
    });
  }
  return { best: beam[0], rounds, seenCount: seen.size };
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

function rootFolds(rootIds, foldCount) {
  return Array.from({ length: foldCount }, (_, fold) => (
    rootIds.filter((_, index) => index % foldCount === fold)
  )).filter((foldRoots) => foldRoots.length > 0);
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

function featureCoverage(items, featureNames) {
  return featureNames.map((feature) => {
    const values = items.map((item) => asNumber(item.features?.[feature], 0));
    const nonZero = values.filter((value) => Math.abs(value) > 1e-9).length;
    const mean = values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : 0;
    return {
      feature,
      nonZeroRows: nonZero,
      nonZeroRate: round(values.length ? nonZero / values.length : 0),
      mean: round(mean),
    };
  });
}

function main() {
  const args = parseArgs(process.argv.slice(2));
  const bridgePath = path.resolve(args.bridge);
  const chronoPath = path.resolve(args.chrono);
  const bridge = readJson(bridgePath);
  const chrono = readJson(chronoPath);
  const bridgeRows = Array.isArray(bridge.rows) ? bridge.rows : [];
  const chronoByHash = rowsByHash(chrono);
  const { items, missingChrono } = joinRows(bridgeRows, chronoByHash);
  const featureNames = featureNamesFromItems(items);
  const allGroups = groupByRoot(items);
  const rootIds = allGroups.map(([rootId]) => rootId);
  const folds = rootFolds(rootIds, Math.min(args.folds, rootIds.length));
  const tacticalVerifiedRows = items.filter((item) => item.tacticalGpuVerified).length;

  const globalBaseline = evaluateFrontier(allGroups);
  const globalLearn = runBeamSearch(allGroups, featureNames, args);
  const globalLearnEval = evaluateSpec(globalLearn.best.spec, allGroups);

  const foldReports = [];
  const foldCandidateEvals = [];
  const foldBaselineEvals = [];
  for (let foldIndex = 0; foldIndex < folds.length; foldIndex += 1) {
    const evalRoots = folds[foldIndex];
    const trainRoots = rootIds.filter((rootId) => !evalRoots.includes(rootId));
    const trainGroups = filterGroups(allGroups, trainRoots);
    const evalGroups = filterGroups(allGroups, evalRoots);
    const trainBaseline = evaluateFrontier(trainGroups);
    const evalBaseline = evaluateFrontier(evalGroups);
    const learned = runBeamSearch(trainGroups, featureNames, args);
    const trainCandidate = evaluateSpec(learned.best.spec, trainGroups);
    const evalCandidate = evaluateSpec(learned.best.spec, evalGroups);
    foldCandidateEvals.push(evalCandidate);
    foldBaselineEvals.push(evalBaseline);
    foldReports.push({
      fold: foldIndex,
      trainRootCount: trainRoots.length,
      evalRootCount: evalRoots.length,
      selectedSpec: publicSpec(learned.best.spec),
      trainBaseline: publicMetrics(trainBaseline),
      trainCandidate: publicMetrics(trainCandidate),
      trainDeltaVsBaseline: metricDelta(trainCandidate, trainBaseline),
      evalBaseline: publicMetrics(evalBaseline),
      evalCandidate: publicMetrics(evalCandidate),
      evalDeltaVsBaseline: metricDelta(evalCandidate, evalBaseline),
      rounds: learned.rounds,
      searchedCandidates: learned.seenCount,
    });
  }
  const crossValidatedBaseline = aggregateEvaluations('frontier_rank_cross_validation_baseline', foldBaselineEvals);
  const crossValidatedCandidate = aggregateEvaluations('chrono_o2_tactical_relation_objective_cross_validation', foldCandidateEvals);
  const crossValidatedDelta = metricDelta(crossValidatedCandidate, crossValidatedBaseline);
  const fixedArtifactDelta = metricDelta(globalLearnEval, globalBaseline);
  const observedLift = (
    crossValidatedDelta.top1 > 0
    || crossValidatedDelta.top3 > 0
    || (crossValidatedDelta.meanAcceptedCandidateRank != null && crossValidatedDelta.meanAcceptedCandidateRank < 0)
  );
  const fixedArtifactLift = (
    fixedArtifactDelta.top1 > 0
    || fixedArtifactDelta.top3 > 0
    || (fixedArtifactDelta.meanAcceptedCandidateRank != null && fixedArtifactDelta.meanAcceptedCandidateRank < 0)
  );
  const output = {
    schemaVersion: SCHEMA_VERSION,
    generatedAt: new Date().toISOString(),
    condition: {
      source: path.resolve(args.conditionSource),
      runLabel: args.runLabel,
      changedFields: 'post-hoc tactical relation objective over recorded GPU-derived sidecar rows only; no runtime behavior changed',
      labCondition: 'posthoc/subset when input bridge/chrono is a subset',
      metric: 'fold-heldout top-k accepted useful injections per root over recorded GPU-derived frontier rows',
    },
    sources: {
      bridgePath,
      bridgeSchemaVersion: bridge.schemaVersion || null,
      bridgeRows: bridgeRows.length,
      chronoPath,
      chronoSchemaVersion: chrono.schemaVersion || null,
      chronoRows: Array.isArray(chrono.rows) ? chrono.rows.length : 0,
      joinedRows: items.length,
      missingChronoRows: missingChrono.length,
      rootCount: allGroups.length,
      tacticalVerifiedRows,
    },
    learner: {
      algorithm: 'deterministic_beam_coordinate_search_over_named_gpu_tactical_relation_features',
      featureSet: 'gpu_tactical_relation_v1_no_frontier_rank_or_score_features',
      featureCount: featureNames.length,
      featureNames,
      folds: folds.length,
      rounds: args.rounds,
      beamSize: args.beamSize,
      deltas: [-2, -1, -0.5, 0.5, 1, 2],
      objective: 'top1 accepted, then top3 accepted, then lower mean accepted rank',
      noRuntimePromotion: true,
      explicitlyExcludedFeatures: [
        'frontier_rank',
        'scoreCp',
        'scoreGapFromBestCp',
        'pathProbability',
        'utility',
      ],
    },
    featureCoverage: featureCoverage(items, featureNames),
    baseline: {
      frontierRank: publicMetrics(globalBaseline),
    },
    globalBest: {
      spec: publicSpec(globalLearn.best.spec),
      metrics: publicMetrics(globalLearnEval),
      deltaVsFrontierRank: fixedArtifactDelta,
      fixedArtifactLift,
      searchedCandidates: globalLearn.seenCount,
      rounds: globalLearn.rounds,
      rootDiffs: topRootDiffs(globalLearnEval, globalBaseline),
    },
    crossValidation: {
      baseline: publicMetrics(crossValidatedBaseline),
      candidate: publicMetrics(crossValidatedCandidate),
      deltaVsFrontierRank: crossValidatedDelta,
      observedLift,
      rootDiffs: topRootDiffs(crossValidatedCandidate, crossValidatedBaseline),
      folds: foldReports,
    },
    promotionPolicy: {
      status: 'not_promoted',
      reason: observedLift
        ? 'post-hoc tactical relation objective shows fold-heldout lift; freeze as a new fixed condition and rerun a heldout GPU gate before promotion'
        : 'tactical relation objective did not show fold-heldout lift over frontier rank',
      blockers: [
        'posthoc_relation_objective_not_runtime_evidence',
        ...(observedLift ? ['heldout_gpu_gate_not_rerun_with_frozen_relation_objective'] : ['tactical_relation_objective_no_cross_validated_lift']),
      ],
      requiredNextEvidence: observedLift
        ? [
          'freeze selected tactical relation objective without looking at new gate labels',
          'rerun heldout GPU gate with the frozen relation objective as a consumer',
          'show accepted useful injection lift without weakening strict/proxy gate interpretation',
        ]
        : [
          'derive richer labels from tacticalContact, OmniFold invariants, or GPU runtime fight outcomes',
          'show positive fold-heldout lift before any frozen GPU gate condition',
        ],
    },
  };
  const outPath = path.resolve(args.out || defaultOutPath(chronoPath));
  fs.mkdirSync(path.dirname(outPath), { recursive: true });
  fs.writeFileSync(outPath, `${JSON.stringify(output, null, 2)}\n`);
  console.log(JSON.stringify({
    ok: true,
    output: outPath,
    rootCount: allGroups.length,
    bridgeRows: bridgeRows.length,
    joinedRows: items.length,
    featureCount: featureNames.length,
    folds: folds.length,
    rounds: args.rounds,
    beamSize: args.beamSize,
    baselineTop1: globalBaseline.top1AcceptedUsefulInjections,
    globalBestTop1: globalLearnEval.top1AcceptedUsefulInjections,
    globalBestDelta: fixedArtifactDelta,
    crossValidatedTop1: crossValidatedCandidate.top1AcceptedUsefulInjections,
    crossValidatedDelta,
    observedLift,
    promote: false,
  }, null, 2));
}

main();
