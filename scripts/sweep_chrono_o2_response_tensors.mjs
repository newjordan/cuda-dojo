#!/usr/bin/env node
import fs from 'node:fs';
import path from 'node:path';

const SCHEMA_VERSION = 'dojo.chrono_o2_response_tensor_sweep.v1';

function usage() {
  return `Usage: node scripts/sweep_chrono_o2_response_tensors.mjs --bridge <bridge.json> --chrono <chrono_o2_sidecar.json> [--out <sweep.json>]

Post-hoc sweep of explicit 4x4 O2 response-tensor candidates. This recomputes
the internal O2 projector from existing sidecar vectors only; it does not run
chess, generate legal moves, verify transitions, or promote runtime behavior.
`;
}

function parseArgs(argv) {
  const args = {
    bridge: null,
    chrono: null,
    out: null,
    conditionSource: '/home/frosty40/cuda_dojo/LOGIC_RAY_FRONTIER_RUNTIME_RECEIPT_2026-05-06.md',
    runLabel: 'posthoc_pzrg_chrono_o2_response_tensor_sweep',
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
  const stem = parsed.base.replace(/\.json$/, '');
  return path.join(parsed.dir, `${stem}.response_tensor_sweep.json`);
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

function dotMetric(a, b) {
  return -asNumber(a?.[0], 0) * asNumber(b?.[0], 0)
    + asNumber(a?.[1], 0) * asNumber(b?.[1], 0)
    + asNumber(a?.[2], 0) * asNumber(b?.[2], 0)
    + asNumber(a?.[3], 0) * asNumber(b?.[3], 0);
}

function add4(a, b) {
  return [0, 1, 2, 3].map((index) => asNumber(a?.[index], 0) + asNumber(b?.[index], 0));
}

function scale4(a, scale) {
  return [0, 1, 2, 3].map((index) => asNumber(a?.[index], 0) * scale);
}

function normEuclidean(a) {
  return Math.sqrt([0, 1, 2, 3].reduce((sum, index) => sum + asNumber(a?.[index], 0) ** 2, 0));
}

function projectOrthogonal(u, r) {
  const denom = dotMetric(u, u);
  if (Math.abs(denom) < 1e-9) return [0, 0, 0, 0];
  const factor = dotMetric(u, r) / denom;
  return [0, 1, 2, 3].map((index) => asNumber(r?.[index], 0) - factor * asNumber(u?.[index], 0));
}

function matrixFromDiag(diag) {
  return [0, 1, 2, 3].map((row) => [0, 1, 2, 3].map((col) => (row === col ? diag[row] : 0)));
}

function zeroMatrix() {
  return matrixFromDiag([0, 0, 0, 0]);
}

function matAdd(a, b) {
  return [0, 1, 2, 3].map((row) => [0, 1, 2, 3].map((col) => asNumber(a?.[row]?.[col], 0) + asNumber(b?.[row]?.[col], 0)));
}

function matScale(a, scale) {
  return [0, 1, 2, 3].map((row) => [0, 1, 2, 3].map((col) => asNumber(a?.[row]?.[col], 0) * scale));
}

function matVec(a, v) {
  return [0, 1, 2, 3].map((row) => (
    [0, 1, 2, 3].reduce((sum, col) => sum + asNumber(a?.[row]?.[col], 0) * asNumber(v?.[col], 0), 0)
  ));
}

function matrixTrace(a) {
  return [0, 1, 2, 3].reduce((sum, index) => sum + asNumber(a?.[index]?.[index], 0), 0);
}

function matrixFrob(a) {
  return Math.sqrt([0, 1, 2, 3].reduce((sum, row) => (
    sum + [0, 1, 2, 3].reduce((inner, col) => inner + asNumber(a?.[row]?.[col], 0) ** 2, 0)
  ), 0));
}

function matrixSummary(a) {
  let maxAbsOffdiag = 0;
  for (const row of [0, 1, 2, 3]) {
    for (const col of [0, 1, 2, 3]) {
      if (row !== col) maxAbsOffdiag = Math.max(maxAbsOffdiag, Math.abs(asNumber(a?.[row]?.[col], 0)));
    }
  }
  return {
    trace: round(matrixTrace(a)),
    frobNorm: round(matrixFrob(a)),
    maxAbsOffdiag: round(maxAbsOffdiag),
  };
}

function candidate(id, description, k0, kc = zeroMatrix(), ks = zeroMatrix()) {
  return {
    id,
    description,
    K0: k0,
    Kc: kc,
    Ks: ks,
    summaries: {
      K0: matrixSummary(k0),
      Kc: matrixSummary(kc),
      Ks: matrixSummary(ks),
    },
  };
}

function candidates() {
  const result = [
    candidate('identity_bootstrap', 'Current bootstrap K0=I, Kc=Ks=0.', matrixFromDiag([1, 1, 1, 1])),
    candidate('spatial_xy_mirror', 'Mirror file/rank move-vector response.', matrixFromDiag([1, -1, -1, 1])),
    candidate('file_mirror', 'Mirror file-axis response only.', matrixFromDiag([1, -1, 1, 1])),
    candidate('rank_mirror', 'Mirror rank-axis response only.', matrixFromDiag([1, 1, -1, 1])),
    candidate('distance_emphasis', 'Emphasize move-distance response.', matrixFromDiag([1, 0.5, 0.5, 2])),
    candidate('file_emphasis', 'Emphasize lateral/file response.', matrixFromDiag([1, 2, 0.5, 1])),
    candidate('rank_emphasis', 'Emphasize forward/rank response.', matrixFromDiag([1, 0.5, 2, 1])),
    candidate('quiet_spatial', 'Dampen spatial response.', matrixFromDiag([0.25, 0.5, 0.5, 0.5])),
  ];
  for (const amp of [-0.5, 0.5, 1.0]) {
    result.push(candidate(
      `cos_spatial_amp_${String(amp).replace('-', 'neg_').replace('.', '_')}`,
      `Identity plus cosine spatial modulation amp ${amp}.`,
      matrixFromDiag([1, 1, 1, 1]),
      matrixFromDiag([0, amp, amp, amp]),
    ));
    result.push(candidate(
      `sin_spatial_amp_${String(amp).replace('-', 'neg_').replace('.', '_')}`,
      `Identity plus sine spatial modulation amp ${amp}.`,
      matrixFromDiag([1, 1, 1, 1]),
      zeroMatrix(),
      matrixFromDiag([0, amp, amp, amp]),
    ));
  }
  for (const [name, diag] of [
    ['cos_file_rank_opposition', [0, 0.75, -0.75, 0]],
    ['cos_rank_distance_opposition', [0, 0, 0.75, -0.75]],
    ['sin_file_distance_opposition', [0, 0.75, 0, -0.75]],
  ]) {
    result.push(candidate(
      name,
      `Identity plus ${name.replaceAll('_', ' ')} phase modulation.`,
      matrixFromDiag([1, 1, 1, 1]),
      name.startsWith('cos') ? matrixFromDiag(diag) : zeroMatrix(),
      name.startsWith('sin') ? matrixFromDiag(diag) : zeroMatrix(),
    ));
  }
  return result;
}

function tensorAt(candidateSpec, theta) {
  return matAdd(
    matAdd(candidateSpec.K0, matScale(candidateSpec.Kc, Math.cos(theta))),
    matScale(candidateSpec.Ks, Math.sin(theta)),
  );
}

function recomputeO2(chronoRow, candidateSpec) {
  const u = chronoRow.vectors?.u || [0, 0, 0, 0];
  const n = chronoRow.vectors?.n || [0, 0, 0, 0];
  const p = chronoRow.vectors?.p || [0, 0, 0, 0];
  const fExt = chronoRow.vectors?.F_ext || [0, 0, 0, 0];
  const theta = asNumber(chronoRow.timePhase?.theta, 0);
  const tensor = tensorAt(candidateSpec, theta);
  const response = matVec(tensor, n);
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
    stabilityScore: round(stabilityScore),
    pathContortion: round(pathContortion),
    normDrift: round(normDrift),
    orthogonalityResidual: round(orthogonalityResidual),
    F_cont: fCont,
  };
}

function evaluateFrontierRank(groups, chronoByHash) {
  let top1Accepted = 0;
  let top3Accepted = 0;
  let selectedTop1 = 0;
  let acceptedRankSum = 0;
  let acceptedRankCount = 0;
  for (const [, rows] of groups) {
    const ranked = [...rows].sort((a, b) => asNumber(frontierOf(a).rank || a.rank, 1) - asNumber(frontierOf(b).rank || b.rank, 1));
    const annotated = ranked.map((row, index) => {
      const chronoRow = chronoByHash.get(row.logicRayFrontierHash);
      return {
        row,
        rank: index + 1,
        accepted: accepted(row, chronoRow),
        selected: selected(row, chronoRow),
      };
    });
    if (annotated[0]?.accepted) top1Accepted += 1;
    if (annotated.slice(0, 3).some((row) => row.accepted)) top3Accepted += 1;
    if (annotated[0]?.selected) selectedTop1 += 1;
    const firstAccepted = annotated.find((row) => row.accepted);
    if (firstAccepted) {
      acceptedRankSum += firstAccepted.rank;
      acceptedRankCount += 1;
    }
  }
  return {
    name: 'frontier_rank',
    rootCount: groups.length,
    top1AcceptedUsefulInjections: top1Accepted,
    top3AcceptedUsefulInjections: top3Accepted,
    selectedMoveTop1: selectedTop1,
    meanAcceptedCandidateRank: acceptedRankCount ? round(acceptedRankSum / acceptedRankCount) : null,
  };
}

function storedO2Score(chronoRow) {
  return (
    0.45 * clamp01(chronoRow?.diagnostics?.stabilityScore, 0)
    + 0.25 * clamp01(chronoRow?.relationDrift?.score, 0)
    + 0.15 * clamp01(chronoRow?.pressureDrift?.score, 0)
    - 0.35 * clamp01(chronoRow?.uncertainty?.score, 0)
    - 0.2 * clamp01(chronoRow?.pathContortion?.score, 0)
    - 0.15 * Math.min(1, Math.abs(asNumber(chronoRow?.diagnostics?.normDrift, 0)))
  );
}

function evaluateStoredO2Identity(groups, chronoByHash) {
  let top1Accepted = 0;
  let top3Accepted = 0;
  let selectedTop1 = 0;
  let acceptedRankSum = 0;
  let acceptedRankCount = 0;
  for (const [, rows] of groups) {
    const ranked = [...rows].sort((a, b) => {
      const delta = storedO2Score(chronoByHash.get(b.logicRayFrontierHash)) - storedO2Score(chronoByHash.get(a.logicRayFrontierHash));
      if (delta !== 0) return delta;
      return asNumber(frontierOf(a).rank || a.rank, 1) - asNumber(frontierOf(b).rank || b.rank, 1);
    }).map((row, index) => {
      const chronoRow = chronoByHash.get(row.logicRayFrontierHash);
      return {
        row,
        rank: index + 1,
        accepted: accepted(row, chronoRow),
        selected: selected(row, chronoRow),
      };
    });
    if (ranked[0]?.accepted) top1Accepted += 1;
    if (ranked.slice(0, 3).some((row) => row.accepted)) top3Accepted += 1;
    if (ranked[0]?.selected) selectedTop1 += 1;
    const firstAccepted = ranked.find((row) => row.accepted);
    if (firstAccepted) {
      acceptedRankSum += firstAccepted.rank;
      acceptedRankCount += 1;
    }
  }
  return {
    name: 'stored_chrono_o2_internal_projector',
    rootCount: groups.length,
    top1AcceptedUsefulInjections: top1Accepted,
    top3AcceptedUsefulInjections: top3Accepted,
    selectedMoveTop1: selectedTop1,
    meanAcceptedCandidateRank: acceptedRankCount ? round(acceptedRankSum / acceptedRankCount) : null,
  };
}

function evaluateTensorCandidate(candidateSpec, groups, chronoByHash) {
  const roots = [];
  let top1Accepted = 0;
  let top3Accepted = 0;
  let selectedTop1 = 0;
  let acceptedRankSum = 0;
  let acceptedRankCount = 0;
  let meanPathContortion = 0;
  let meanStability = 0;
  let featureCount = 0;
  for (const [rootId, rows] of groups) {
    const ranked = [...rows].map((row) => {
      const chronoRow = chronoByHash.get(row.logicRayFrontierHash);
      const recomputed = recomputeO2(chronoRow || {}, candidateSpec);
      meanPathContortion += recomputed.pathContortion;
      meanStability += recomputed.stabilityScore;
      featureCount += 1;
      return {
        row,
        chronoRow,
        recomputed,
        accepted: accepted(row, chronoRow),
        selected: selected(row, chronoRow),
      };
    }).sort((a, b) => {
      const delta = b.recomputed.scoreRaw - a.recomputed.scoreRaw;
      if (delta !== 0) return delta;
      return asNumber(frontierOf(a.row).rank || a.row.rank, 1) - asNumber(frontierOf(b.row).rank || b.row.rank, 1);
    }).map((item, index) => ({ ...item, comparisonRank: index + 1 }));
    if (ranked[0]?.accepted) top1Accepted += 1;
    if (ranked.slice(0, 3).some((row) => row.accepted)) top3Accepted += 1;
    if (ranked[0]?.selected) selectedTop1 += 1;
    const firstAccepted = ranked.find((row) => row.accepted);
    if (firstAccepted) {
      acceptedRankSum += firstAccepted.comparisonRank;
      acceptedRankCount += 1;
    }
    roots.push({
      rootId,
      rowCount: rows.length,
      top1: ranked[0] ? {
        bridgeId: ranked[0].row.bridgeId,
        hash: ranked[0].row.logicRayFrontierHash,
        move: ranked[0].row.move || frontierOf(ranked[0].row).move,
        originalRank: asNumber(frontierOf(ranked[0].row).rank || ranked[0].row.rank, 1),
        score: ranked[0].recomputed.score,
        acceptedUsefulInjection: ranked[0].accepted,
        selectedMoveInFrontier: ranked[0].selected,
        stabilityScore: ranked[0].recomputed.stabilityScore,
        pathContortion: ranked[0].recomputed.pathContortion,
        normDrift: ranked[0].recomputed.normDrift,
      } : null,
      acceptedCandidateBestRank: firstAccepted?.comparisonRank || null,
    });
  }
  return {
    id: candidateSpec.id,
    description: candidateSpec.description,
    tensorSummaries: candidateSpec.summaries,
    rootCount: groups.length,
    top1AcceptedUsefulInjections: top1Accepted,
    top3AcceptedUsefulInjections: top3Accepted,
    selectedMoveTop1: selectedTop1,
    meanAcceptedCandidateRank: acceptedRankCount ? round(acceptedRankSum / acceptedRankCount) : null,
    meanPathContortion: featureCount ? round(meanPathContortion / featureCount) : null,
    meanStabilityScore: featureCount ? round(meanStability / featureCount) : null,
    roots,
  };
}

function delta(candidateEval, baselineEval) {
  return {
    top1: candidateEval.top1AcceptedUsefulInjections - baselineEval.top1AcceptedUsefulInjections,
    top3: candidateEval.top3AcceptedUsefulInjections - baselineEval.top3AcceptedUsefulInjections,
    meanAcceptedCandidateRank: (
      candidateEval.meanAcceptedCandidateRank != null && baselineEval.meanAcceptedCandidateRank != null
        ? round(candidateEval.meanAcceptedCandidateRank - baselineEval.meanAcceptedCandidateRank)
        : null
    ),
  };
}

function publicCandidate(evaluation, recomputedIdentity, sourceO2, frontier) {
  return {
    id: evaluation.id,
    description: evaluation.description,
    tensorSummaries: evaluation.tensorSummaries,
    top1AcceptedUsefulInjections: evaluation.top1AcceptedUsefulInjections,
    top3AcceptedUsefulInjections: evaluation.top3AcceptedUsefulInjections,
    selectedMoveTop1: evaluation.selectedMoveTop1,
    meanAcceptedCandidateRank: evaluation.meanAcceptedCandidateRank,
    meanPathContortion: evaluation.meanPathContortion,
    meanStabilityScore: evaluation.meanStabilityScore,
    deltaVsRecomputedIdentity: delta(evaluation, recomputedIdentity),
    deltaVsStoredO2Identity: delta(evaluation, sourceO2),
    deltaVsFrontierRank: delta(evaluation, frontier),
    topRegressions: evaluation.roots
      .filter((root) => root.acceptedCandidateBestRank === 1 && !root.top1?.acceptedUsefulInjection)
      .slice(0, 8),
  };
}

function main() {
  const args = parseArgs(process.argv.slice(2));
  const bridgePath = path.resolve(args.bridge);
  const chronoPath = path.resolve(args.chrono);
  const bridge = readJson(bridgePath);
  const chrono = readJson(chronoPath);
  const bridgeRows = Array.isArray(bridge.rows) ? bridge.rows : [];
  const chronoByHash = rowsByHash(chrono);
  const groups = groupByRoot(bridgeRows);
  const frontierEval = evaluateFrontierRank(groups, chronoByHash);
  const sourceO2Eval = evaluateStoredO2Identity(groups, chronoByHash);
  const candidateSpecs = candidates();
  const evaluations = candidateSpecs.map((spec) => evaluateTensorCandidate(spec, groups, chronoByHash));
  const identityEval = evaluations.find((item) => item.id === 'identity_bootstrap') || evaluations[0];
  const best = [...evaluations].sort((a, b) => {
    const top1 = b.top1AcceptedUsefulInjections - a.top1AcceptedUsefulInjections;
    if (top1) return top1;
    const top3 = b.top3AcceptedUsefulInjections - a.top3AcceptedUsefulInjections;
    if (top3) return top3;
    return (a.meanAcceptedCandidateRank ?? 1e9) - (b.meanAcceptedCandidateRank ?? 1e9);
  })[0];
  const chronoBlockers = Array.isArray(chrono.promotionPolicy?.blockers) ? chrono.promotionPolicy.blockers : [];
  const needsTransitionVerification = chronoBlockers.includes('source_order_proxy_transitions_unverified')
    || chronoBlockers.includes('missing_neighbor_rows_present');
  const output = {
    schemaVersion: SCHEMA_VERSION,
    generatedAt: new Date().toISOString(),
    condition: {
      source: path.resolve(args.conditionSource),
      runLabel: args.runLabel,
      changedFields: 'post-hoc response tensor sweep over existing O2 vectors; no runtime behavior changed',
      labCondition: 'posthoc/proxy/subset when input bridge is a subset',
    },
    sources: {
      bridgePath,
      bridgeSchemaVersion: bridge.schemaVersion || null,
      bridgeRows: bridgeRows.length,
      chronoPath,
      chronoSchemaVersion: chrono.schemaVersion || null,
      chronoRows: Array.isArray(chrono.rows) ? chrono.rows.length : 0,
      rootCount: groups.length,
    },
    baseline: {
      frontierRank: frontierEval,
      storedO2Identity: sourceO2Eval,
      recomputedIdentityBootstrap: publicCandidate(identityEval, identityEval, sourceO2Eval, frontierEval),
    },
    bestCandidate: publicCandidate(best, identityEval, sourceO2Eval, frontierEval),
    candidateCount: evaluations.length,
    candidates: evaluations.map((item) => publicCandidate(item, identityEval, sourceO2Eval, frontierEval)),
    promotionPolicy: {
      status: 'not_promoted',
      reason: needsTransitionVerification
        ? 'response tensor sweep is fixed-artifact diagnosis only; transition-verified temporal source and heldout GPU gate are still required'
        : 'response tensor sweep is fixed-artifact diagnosis only; learned response tensors and heldout GPU gate are still required',
      blockers: [
        'posthoc_tensor_sweep_not_runtime_evidence',
        ...chronoBlockers,
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
    bridgeRows: bridgeRows.length,
    candidateCount: evaluations.length,
    frontierTop1: frontierEval.top1AcceptedUsefulInjections,
    storedO2IdentityTop1: sourceO2Eval.top1AcceptedUsefulInjections,
    recomputedIdentityTop1: identityEval.top1AcceptedUsefulInjections,
    bestCandidate: best.id,
    bestTop1: best.top1AcceptedUsefulInjections,
    bestTop3: best.top3AcceptedUsefulInjections,
    bestDeltaVsRecomputedIdentity: delta(best, identityEval),
    bestDeltaVsStoredO2Identity: delta(best, sourceO2Eval),
    bestDeltaVsFrontierRank: delta(best, frontierEval),
    promote: false,
  }, null, 2));
}

try {
  main();
} catch (error) {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
}
