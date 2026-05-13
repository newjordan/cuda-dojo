#!/usr/bin/env node
import fs from 'node:fs';
import path from 'node:path';

const SCHEMA_VERSION = 'dojo.chrono_o2_response_tensor_learning.v1';
const AXES = [0, 1, 2, 3];

function usage() {
  return `Usage: node scripts/learn_chrono_o2_response_tensors.mjs --bridge <bridge.json> --chrono <chrono_o2_sidecar.json> [--out <learning.json>] [--folds 4] [--rounds 4] [--beam-size 24] [--basis diagonal|full|action-diagonal|action-full|tactical-action-diagonal|tactical-action-full]

Learn explicit non-bootstrap PZRG_CHRONO_O2 4x4 response tensors from recorded
GPU-derived bridge/sidecar artifacts. This does not run chess, generate legal
moves, verify transitions, or promote runtime behavior.
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
    basis: 'diagonal',
    conditionSource: '/home/frosty40/cuda_dojo/LOGIC_RAY_FRONTIER_RUNTIME_RECEIPT_2026-05-06.md',
    runLabel: 'posthoc_pzrg_chrono_o2_response_tensor_learning',
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
    else if (token === '--basis') args.basis = argv[++i];
    else if (token === '--condition-source') args.conditionSource = argv[++i];
    else if (token === '--run-label') args.runLabel = argv[++i];
    else throw new Error(`unknown argument: ${token}\n${usage()}`);
  }
  if (!args.bridge) throw new Error(`missing --bridge\n${usage()}`);
  if (!args.chrono) throw new Error(`missing --chrono\n${usage()}`);
  if (!Number.isInteger(args.folds) || args.folds < 2) throw new Error('--folds must be an integer >= 2');
  if (!Number.isInteger(args.rounds) || args.rounds < 1) throw new Error('--rounds must be an integer >= 1');
  if (!Number.isInteger(args.beamSize) || args.beamSize < 1) throw new Error('--beam-size must be an integer >= 1');
  const basisModes = ['diagonal', 'full', 'action-diagonal', 'action-full', 'tactical-action-diagonal', 'tactical-action-full'];
  if (!basisModes.includes(args.basis)) {
    throw new Error(`--basis must be one of: ${basisModes.join(', ')}`);
  }
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
  const stem = parsed.base.replace(/\.json$/, '');
  return path.join(parsed.dir, `${stem}.response_tensor_learning.json`);
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

function groupByRoot(rows) {
  const groups = new Map();
  for (const row of rows) {
    const frontier = frontierOf(row);
    const rootId = frontier.rootId || row.rootId || 'root';
    if (!groups.has(rootId)) groups.set(rootId, []);
    groups.get(rootId).push(row);
  }
  return [...groups.entries()].sort(([a], [b]) => String(a).localeCompare(String(b)));
}

function filterGroups(groups, rootIds) {
  const allowed = new Set(rootIds);
  return groups.filter(([rootId]) => allowed.has(rootId));
}

function dotMetric(a, b) {
  return -asNumber(a?.[0], 0) * asNumber(b?.[0], 0)
    + asNumber(a?.[1], 0) * asNumber(b?.[1], 0)
    + asNumber(a?.[2], 0) * asNumber(b?.[2], 0)
    + asNumber(a?.[3], 0) * asNumber(b?.[3], 0);
}

function add4(a, b) {
  return AXES.map((index) => asNumber(a?.[index], 0) + asNumber(b?.[index], 0));
}

function scale4(a, scale) {
  return AXES.map((index) => asNumber(a?.[index], 0) * scale);
}

function normEuclidean(a) {
  return Math.sqrt(AXES.reduce((sum, index) => sum + asNumber(a?.[index], 0) ** 2, 0));
}

function projectOrthogonal(u, r) {
  const denom = dotMetric(u, u);
  if (Math.abs(denom) < 1e-9) return [0, 0, 0, 0];
  const factor = dotMetric(u, r) / denom;
  return AXES.map((index) => asNumber(r?.[index], 0) - factor * asNumber(u?.[index], 0));
}

function matrixFromDiag(diag) {
  return AXES.map((row) => AXES.map((col) => (row === col ? asNumber(diag[row], 0) : 0)));
}

function zeroMatrix() {
  return matrixFromDiag([0, 0, 0, 0]);
}

function cloneMatrix(matrix) {
  return AXES.map((row) => AXES.map((col) => asNumber(matrix?.[row]?.[col], 0)));
}

function matAdd(a, b) {
  return AXES.map((row) => AXES.map((col) => asNumber(a?.[row]?.[col], 0) + asNumber(b?.[row]?.[col], 0)));
}

function matScale(a, scale) {
  return AXES.map((row) => AXES.map((col) => asNumber(a?.[row]?.[col], 0) * scale));
}

function matVec(a, v) {
  return AXES.map((row) => (
    AXES.reduce((sum, col) => sum + asNumber(a?.[row]?.[col], 0) * asNumber(v?.[col], 0), 0)
  ));
}

function tensorAt(spec, theta) {
  return matAdd(
    matAdd(spec.K0, matScale(spec.Kc, Math.cos(theta))),
    matScale(spec.Ks, Math.sin(theta)),
  );
}

function matrixTrace(a) {
  return AXES.reduce((sum, index) => sum + asNumber(a?.[index]?.[index], 0), 0);
}

function matrixFrob(a) {
  return Math.sqrt(AXES.reduce((sum, row) => (
    sum + AXES.reduce((inner, col) => inner + asNumber(a?.[row]?.[col], 0) ** 2, 0)
  ), 0));
}

function matrixSummary(a) {
  let maxAbsOffdiag = 0;
  for (const row of AXES) {
    for (const col of AXES) {
      if (row !== col) maxAbsOffdiag = Math.max(maxAbsOffdiag, Math.abs(asNumber(a?.[row]?.[col], 0)));
    }
  }
  return {
    trace: round(matrixTrace(a)),
    frobNorm: round(matrixFrob(a)),
    maxAbsOffdiag: round(maxAbsOffdiag),
  };
}

function tensorSummaries(spec) {
  return {
    K0: matrixSummary(spec.K0),
    Kc: matrixSummary(spec.Kc),
    Ks: matrixSummary(spec.Ks),
  };
}

function makeSpec(
  id,
  description,
  K0,
  Kc = zeroMatrix(),
  Ks = zeroMatrix(),
  history = [],
  actionWeights = [0, 0, 0, 0],
  tacticalWeights = [0, 0, 0, 0],
) {
  return {
    id,
    description,
    K0: cloneMatrix(K0),
    Kc: cloneMatrix(Kc),
    Ks: cloneMatrix(Ks),
    actionWeights: AXES.map((index) => asNumber(actionWeights?.[index], 0)),
    tacticalWeights: AXES.map((index) => asNumber(tacticalWeights?.[index], 0)),
    history,
  };
}

function specKey(spec) {
  return JSON.stringify({
    K: [spec.K0, spec.Kc, spec.Ks].map((matrix) => (
      matrix.map((row) => row.map((value) => round(value, 4)))
    )),
    A: AXES.map((index) => round(spec.actionWeights?.[index], 4)),
    T: AXES.map((index) => round(spec.tacticalWeights?.[index], 4)),
  });
}

function mutateSpec(spec, target, row, col, delta) {
  const next = makeSpec(
    `${spec.id}.${target}${row}${col}${delta >= 0 ? 'p' : 'm'}${String(Math.abs(delta)).replace('.', '_')}`,
    `Beam mutation ${target}[${row},${col}] ${delta >= 0 ? '+' : ''}${delta}`,
    spec.K0,
    spec.Kc,
    spec.Ks,
    [...spec.history, { target, row, col, delta }],
    spec.actionWeights,
    spec.tacticalWeights,
  );
  if (target === 'A') next.actionWeights[row] = round(next.actionWeights[row] + delta);
  else if (target === 'T') next.tacticalWeights[row] = round(next.tacticalWeights[row] + delta);
  else next[target][row][col] = round(next[target][row][col] + delta);
  return next;
}

function seedSpecs() {
  return [
    makeSpec('identity_bootstrap_reference', 'Bootstrap reference K0=I, Kc=Ks=0.', matrixFromDiag([1, 1, 1, 1])),
    makeSpec('zero_response', 'Zero base response tensor.', zeroMatrix()),
    makeSpec('quiet_spatial', 'Dampened spatial diagonal response.', matrixFromDiag([0.25, 0.5, 0.5, 0.5])),
    makeSpec('distance_emphasis', 'Distance-heavy diagonal response.', matrixFromDiag([1, 0.5, 0.5, 2])),
    makeSpec('file_rank_flip', 'File/rank sign-flipped diagonal response.', matrixFromDiag([1, -1, -1, 1])),
    makeSpec('action_capture_forward_seed', 'Bootstrap tensor with capture/diagonal/forward/center action input.', matrixFromDiag([1, 1, 1, 1]), zeroMatrix(), zeroMatrix(), [], [1, 1, 1, 1]),
    makeSpec('action_forward_center_seed', 'Bootstrap tensor with forward and center action input.', matrixFromDiag([1, 1, 1, 1]), zeroMatrix(), zeroMatrix(), [], [0, 0, 1, 1]),
    makeSpec('tactical_contact_seed', 'Bootstrap tensor with defended/attacked/check/reply tactical contact input.', matrixFromDiag([1, 1, 1, 1]), zeroMatrix(), zeroMatrix(), [], [0, 0, 0, 0], [1, -1, 1, -1]),
    makeSpec('tactical_safety_seed', 'Bootstrap tensor with defended-square tactical safety input.', matrixFromDiag([1, 1, 1, 1]), zeroMatrix(), zeroMatrix(), [], [0, 0, 0, 0], [1, -1, 0, 0]),
  ];
}

function basisEntries(mode = 'diagonal') {
  const result = [];
  const matrixMode = mode.endsWith('full') ? 'full' : 'diagonal';
  for (const target of ['K0', 'Kc', 'Ks']) {
    for (const row of AXES) {
      for (const col of AXES) {
        if (matrixMode === 'diagonal' && row !== col) continue;
        result.push({ target, row, col });
      }
    }
  }
  if (mode.startsWith('action-') || mode.startsWith('tactical-action-')) {
    for (const row of AXES) result.push({ target: 'A', row, col: 0 });
  }
  if (mode.startsWith('tactical-action-')) {
    for (const row of AXES) result.push({ target: 'T', row, col: 0 });
  }
  return result;
}

function responseInputVector(chronoRow, spec) {
  const n = chronoRow.vectors?.n || [0, 0, 0, 0];
  const action = chronoRow.actionFeatures?.actionVector4 || [0, 0, 0, 0];
  const tactical = chronoRow.tacticalContact?.contactVector4 || [0, 0, 0, 0];
  const actionWeights = spec.actionWeights || [0, 0, 0, 0];
  const tacticalWeights = spec.tacticalWeights || [0, 0, 0, 0];
  return AXES.map((index) => (
    asNumber(n?.[index], 0)
    + asNumber(actionWeights?.[index], 0) * asNumber(action?.[index], 0)
    + asNumber(tacticalWeights?.[index], 0) * asNumber(tactical?.[index], 0)
  ));
}

function recomputeO2(chronoRow, spec) {
  const u = chronoRow.vectors?.u || [0, 0, 0, 0];
  const p = chronoRow.vectors?.p || [0, 0, 0, 0];
  const fExt = chronoRow.vectors?.F_ext || [0, 0, 0, 0];
  const theta = asNumber(chronoRow.timePhase?.theta, 0);
  const response = matVec(tensorAt(spec, theta), responseInputVector(chronoRow, spec));
  const speedSq = Math.abs(dotMetric(u, u));
  const fCont = scale4(projectOrthogonal(u, response), speedSq).map((value) => round(value));
  const pNext = add4(add4(p, fExt), fCont).map((value) => round(value));
  const normBefore = round(dotMetric(u, u));
  const normAfter = round(dotMetric(pNext, pNext));
  const normDrift = round(normAfter - normBefore);
  const orthogonalityResidual = round(dotMetric(u, fCont));
  const uncertaintyScore = clamp01(chronoRow.uncertainty?.score, 0);
  const pathContortion = round(clamp01(normEuclidean(fCont) / 2, 0));
  const stabilityScore = round(clamp01(
    1
      - Math.min(1, Math.abs(normDrift)) * 0.35
      - Math.min(1, Math.abs(orthogonalityResidual)) * 0.45
      - uncertaintyScore * 0.2,
    0,
  ));
  const relationDrift = clamp01(chronoRow.relationDrift?.score, 0);
  const pressureDrift = clamp01(chronoRow.pressureDrift?.score, 0);
  const scoreRaw = (
    0.45 * stabilityScore
    + 0.25 * relationDrift
    + 0.15 * pressureDrift
    - 0.35 * uncertaintyScore
    - 0.2 * pathContortion
    - 0.15 * Math.min(1, Math.abs(normDrift))
  );
  return {
    scoreRaw,
    score: round(scoreRaw),
    stabilityScore,
    pathContortion,
    normDrift,
    orthogonalityResidual,
    F_cont: fCont,
  };
}

function evaluateFrontier(groups, chronoByHash) {
  return evaluateRankedGroups(groups, chronoByHash, (row) => -asNumber(frontierOf(row).rank || row.rank, 1), 'frontier_rank');
}

function evaluateSpec(spec, groups, chronoByHash) {
  return evaluateRankedGroups(groups, chronoByHash, (row) => (
    recomputeO2(chronoByHash.get(row.logicRayFrontierHash) || {}, spec).scoreRaw
  ), spec.id);
}

function evaluateRankedGroups(groups, chronoByHash, scoreFn, name) {
  const roots = [];
  let top1Accepted = 0;
  let top3Accepted = 0;
  let selectedTop1 = 0;
  let acceptedRankSum = 0;
  let acceptedRankCount = 0;
  for (const [rootId, rows] of groups) {
    const ranked = [...rows].sort((a, b) => {
      const delta = scoreFn(b) - scoreFn(a);
      if (delta !== 0) return delta;
      return asNumber(frontierOf(a).rank || a.rank, 1) - asNumber(frontierOf(b).rank || b.rank, 1);
    }).map((row, index) => {
      const chronoRow = chronoByHash.get(row.logicRayFrontierHash);
      return {
        row,
        comparisonRank: index + 1,
        score: round(scoreFn(row)),
        accepted: accepted(row, chronoRow),
        selected: selected(row, chronoRow),
      };
    });
    const firstAccepted = ranked.find((item) => item.accepted);
    if (ranked[0]?.accepted) top1Accepted += 1;
    if (ranked.slice(0, 3).some((item) => item.accepted)) top3Accepted += 1;
    if (ranked[0]?.selected) selectedTop1 += 1;
    if (firstAccepted) {
      acceptedRankSum += firstAccepted.comparisonRank;
      acceptedRankCount += 1;
    }
    roots.push({
      rootId,
      rowCount: rows.length,
      top1: ranked[0] ? publicRankedRow(ranked[0]) : null,
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

function publicRankedRow(item) {
  const frontier = frontierOf(item.row);
  return {
    bridgeId: item.row.bridgeId,
    hash: item.row.logicRayFrontierHash,
    move: item.row.move || frontier.move,
    originalRank: asNumber(frontier.rank || item.row.rank, 1),
    comparisonRank: item.comparisonRank,
    score: item.score,
    acceptedUsefulInjection: item.accepted,
    selectedMoveInFrontier: item.selected,
  };
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

function publicSpec(spec) {
  return {
    id: spec.id,
    description: spec.description,
    learnedNonBootstrap: spec.id !== 'identity_bootstrap_reference',
    tensorSummaries: tensorSummaries(spec),
    K0: spec.K0,
    Kc: spec.Kc,
    Ks: spec.Ks,
    actionWeights: AXES.map((index) => round(spec.actionWeights?.[index], 6)),
    tacticalWeights: AXES.map((index) => round(spec.tacticalWeights?.[index], 6)),
    history: spec.history.slice(-16),
  };
}

function runBeamSearch(trainGroups, chronoByHash, args) {
  const entries = basisEntries(args.basis);
  const deltas = [-2, -1, -0.5, 0.5, 1, 2];
  let beam = seedSpecs().map((spec) => ({ spec, metrics: evaluateSpec(spec, trainGroups, chronoByHash) }));
  const seen = new Set(beam.map((item) => specKey(item.spec)));
  const rounds = [];
  for (let roundIndex = 0; roundIndex < args.rounds; roundIndex += 1) {
    const evaluated = [...beam];
    for (const item of beam) {
      for (const entry of entries) {
        for (const delta of deltas) {
          const next = mutateSpec(item.spec, entry.target, entry.row, entry.col, delta);
          const mutatedValue = entry.target === 'A'
            ? next.actionWeights[entry.row]
            : entry.target === 'T'
              ? next.tacticalWeights[entry.row]
              : next[entry.target][entry.row][entry.col];
          if (Math.abs(mutatedValue) > 4) continue;
          const key = specKey(next);
          if (seen.has(key)) continue;
          seen.add(key);
          evaluated.push({ spec: next, metrics: evaluateSpec(next, trainGroups, chronoByHash) });
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

function publicMetrics(metrics) {
  return {
    rootCount: metrics.rootCount,
    top1AcceptedUsefulInjections: metrics.top1AcceptedUsefulInjections,
    top3AcceptedUsefulInjections: metrics.top3AcceptedUsefulInjections,
    selectedMoveTop1: metrics.selectedMoveTop1,
    meanAcceptedCandidateRank: metrics.meanAcceptedCandidateRank,
  };
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

function main() {
  const args = parseArgs(process.argv.slice(2));
  const bridgePath = path.resolve(args.bridge);
  const chronoPath = path.resolve(args.chrono);
  const bridge = readJson(bridgePath);
  const chrono = readJson(chronoPath);
  const bridgeRows = Array.isArray(bridge.rows) ? bridge.rows : [];
  const chronoByHash = rowsByHash(chrono);
  const allGroups = groupByRoot(bridgeRows);
  const rootIds = allGroups.map(([rootId]) => rootId);
  const folds = rootFolds(rootIds, Math.min(args.folds, rootIds.length));

  const globalBaseline = evaluateFrontier(allGroups, chronoByHash);
  const globalLearn = runBeamSearch(allGroups, chronoByHash, args);
  const globalLearnEval = evaluateSpec(globalLearn.best.spec, allGroups, chronoByHash);

  const foldReports = [];
  const foldCandidateEvals = [];
  const foldBaselineEvals = [];
  for (let foldIndex = 0; foldIndex < folds.length; foldIndex += 1) {
    const evalRoots = folds[foldIndex];
    const trainRoots = rootIds.filter((rootId) => !evalRoots.includes(rootId));
    const trainGroups = filterGroups(allGroups, trainRoots);
    const evalGroups = filterGroups(allGroups, evalRoots);
    const trainBaseline = evaluateFrontier(trainGroups, chronoByHash);
    const evalBaseline = evaluateFrontier(evalGroups, chronoByHash);
    const learned = runBeamSearch(trainGroups, chronoByHash, args);
    const trainCandidate = evaluateSpec(learned.best.spec, trainGroups, chronoByHash);
    const evalCandidate = evaluateSpec(learned.best.spec, evalGroups, chronoByHash);
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
  const crossValidatedCandidate = aggregateEvaluations('chrono_o2_learned_tensor_cross_validation', foldCandidateEvals);
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
      changedFields: 'post-hoc learned K0/Kc/Ks plus optional action/tactical response-input weights; no runtime behavior changed',
      labCondition: 'posthoc/subset when input bridge is a subset',
      metric: 'fold-heldout top-k accepted useful injections per root over recorded GPU-derived frontier rows',
    },
    sources: {
      bridgePath,
      bridgeSchemaVersion: bridge.schemaVersion || null,
      bridgeRows: bridgeRows.length,
      chronoPath,
      chronoSchemaVersion: chrono.schemaVersion || null,
      chronoRows: Array.isArray(chrono.rows) ? chrono.rows.length : 0,
      rootCount: allGroups.length,
    },
    learner: {
      algorithm: 'deterministic_beam_coordinate_search_over_K0_Kc_Ks_4x4_and_optional_action_tactical_input_weights',
      folds: folds.length,
      rounds: args.rounds,
      beamSize: args.beamSize,
      basisEntries: basisEntries(args.basis).length,
      basis: args.basis,
      deltas: [-2, -1, -0.5, 0.5, 1, 2],
      objective: 'top1 accepted, then top3 accepted, then lower mean accepted rank',
      noRuntimePromotion: true,
    },
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
        ? 'learned response tensors show post-hoc fold-heldout lift; freeze as a new fixed condition and rerun a heldout GPU gate before promotion'
        : 'learned response tensors did not show fold-heldout lift over frontier rank',
      blockers: [
        'posthoc_tensor_learning_not_runtime_evidence',
        ...(observedLift ? ['heldout_gpu_gate_not_rerun_with_frozen_tensor_condition'] : ['learned_tensor_no_cross_validated_lift']),
      ],
      requiredNextEvidence: observedLift
        ? [
          'freeze selected K0/Kc/Ks tensor condition without looking at new gate labels',
          'rerun heldout GPU gate with the frozen tensor condition as a consumer',
          'show accepted useful injection lift without weakening strict/proxy gate interpretation',
        ]
        : [
          'improve tensor learner or add stronger GPU-derived temporal features',
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

try {
  main();
} catch (error) {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
}
