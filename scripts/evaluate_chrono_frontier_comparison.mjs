#!/usr/bin/env node
import fs from 'node:fs';
import path from 'node:path';

const SCHEMA_VERSION = 'dojo.chrono_frontier_comparison.v1';

function usage() {
  return `Usage: node scripts/evaluate_chrono_frontier_comparison.mjs --bridge <bridge.json> --chrono <chrono_sidecar.json> [--out <comparison.json>]

Compare no-chrono frontier ranking against chrono-adjusted ranking on the same
GPU-derived frontier rows. This is post-hoc analysis only: it does not generate
legal moves, run search, or promote a runtime choice path.
`;
}

function parseArgs(argv) {
  const args = {
    bridge: null,
    chrono: null,
    out: null,
    conditionSource: '/home/frosty40/cuda_dojo/LOGIC_RAY_FRONTIER_RUNTIME_RECEIPT_2026-05-06.md',
    runLabel: 'posthoc_chrono_sidecar_comparison_on_documented_offset64_frontier_gate',
  };
  for (let i = 0; i < argv.length; i += 1) {
    const token = argv[i];
    if (token === '--help' || token === '-h') {
      console.log(usage());
      process.exit(0);
    }
    if (token === '--bridge') {
      args.bridge = argv[++i];
    } else if (token === '--chrono') {
      args.chrono = argv[++i];
    } else if (token === '--out') {
      args.out = argv[++i];
    } else if (token === '--condition-source') {
      args.conditionSource = argv[++i];
    } else if (token === '--run-label') {
      args.runLabel = argv[++i];
    } else {
      throw new Error(`unknown argument: ${token}\n${usage()}`);
    }
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
  return path.join(parsed.dir, `${stem}.chrono_frontier_comparison.json`);
}

function rowsByHash(bundle) {
  const result = new Map();
  for (const row of Array.isArray(bundle.rows) ? bundle.rows : []) {
    const hash = row.logicRayFrontierHash;
    if (hash) result.set(hash, row);
  }
  return result;
}

function groupByRoot(rows) {
  const groups = new Map();
  for (const row of rows) {
    const frontier = row.logicRayFrontier || row.pzrgCandidate?.logicRayFrontier || {};
    const rootId = frontier.rootId || row.rootId || 'root';
    if (!groups.has(rootId)) groups.set(rootId, []);
    groups.get(rootId).push(row);
  }
  return groups;
}

function accepted(row, chrono) {
  const frontier = row.logicRayFrontier || row.pzrgCandidate?.logicRayFrontier || {};
  const injection = row.pzrgCandidate?.injection_relevance || {};
  return Boolean(
    frontier.gate?.acceptedUsefulInjection
      || injection.accepted_useful_injection
      || injection.promotion_gate_approved
      || chrono?.runtimeChoiceSignal?.acceptedUsefulInjection,
  );
}

function selected(row, chrono) {
  const frontier = row.logicRayFrontier || row.pzrgCandidate?.logicRayFrontier || {};
  return Boolean(frontier.gate?.selectedMoveInFrontier || chrono?.runtimeChoiceSignal?.selectedMoveInFrontier);
}

function useful(row, chrono) {
  const frontier = row.logicRayFrontier || row.pzrgCandidate?.logicRayFrontier || {};
  const injection = row.pzrgCandidate?.injection_relevance || {};
  return clamp01(
    chrono?.runtimeChoiceSignal?.usefulInjectionScore
      ?? injection.useful_injection_score
      ?? injection.useful_score
      ?? frontier.gate?.acceptedInjectionScore
      ?? frontier.utility,
    0,
  );
}

function rankers(chronoByHash) {
  return {
    frontier_rank: {
      family: 'no_chrono',
      description: 'Existing GPU frontier order: lowest rank wins.',
      score: (row) => -asNumber((row.logicRayFrontier || row.pzrgCandidate?.logicRayFrontier || {}).rank, 1),
    },
    frontier_utility: {
      family: 'no_chrono',
      description: 'Existing GPU frontier utility only.',
      score: (row) => asNumber((row.logicRayFrontier || row.pzrgCandidate?.logicRayFrontier || {}).utility, 0),
    },
    chrono_stability_adjusted: {
      family: 'chrono',
      description: 'Frontier utility adjusted by sidecar stability and uncertainty, without using accepted labels.',
      score: (row) => {
        const frontier = row.logicRayFrontier || row.pzrgCandidate?.logicRayFrontier || {};
        const chrono = chronoByHash.get(row.logicRayFrontierHash) || {};
        return (
          asNumber(frontier.utility, 0)
          + 0.35 * clamp01(chrono.diagnostics?.stabilityScore, 0)
          - 0.25 * clamp01(chrono.uncertainty?.score, 0)
          - 0.15 * Math.min(1, Math.abs(asNumber(chrono.diagnostics?.normDrift, 0)))
        );
      },
    },
    chrono_pressure_horizon: {
      family: 'chrono',
      description: 'Path probability/lock-in/risk with sidecar stability, uncertainty, and path-contortion penalties.',
      score: (row) => {
        const frontier = row.logicRayFrontier || row.pzrgCandidate?.logicRayFrontier || {};
        const chrono = chronoByHash.get(row.logicRayFrontierHash) || {};
        return (
          clamp01(frontier.pathProbability, 0)
          + clamp01(frontier.lockIn, 0)
          - clamp01(frontier.risk, 0)
          + 0.25 * clamp01(chrono.diagnostics?.stabilityScore, 0)
          - 0.25 * clamp01(chrono.uncertainty?.score, 0)
          - 0.1 * clamp01(chrono.pathContortion?.score, 0)
        );
      },
    },
    chrono_o2_internal_projector: {
      family: 'chrono_o2',
      description: 'PZRG_CHRONO_O2 internal-clock/projector score without frontier utility, rank, probability, risk, or lock-in.',
      score: (row) => {
        const chrono = chronoByHash.get(row.logicRayFrontierHash) || {};
        return (
          0.45 * clamp01(chrono.diagnostics?.stabilityScore, 0)
          + 0.25 * clamp01(chrono.relationDrift?.score, 0)
          + 0.15 * clamp01(chrono.pressureDrift?.score, 0)
          - 0.35 * clamp01(chrono.uncertainty?.score, 0)
          - 0.2 * clamp01(chrono.pathContortion?.score, 0)
          - 0.15 * Math.min(1, Math.abs(asNumber(chrono.diagnostics?.normDrift, 0)))
        );
      },
    },
  };
}

function evaluateRanker(name, spec, groups, chronoByHash) {
  const roots = [];
  let top1Accepted = 0;
  let top3Accepted = 0;
  let selectedTop1 = 0;
  let selectedTop3 = 0;
  let acceptedCandidateRoots = 0;
  let meanAcceptedRank = 0;
  let acceptedRankCount = 0;
  for (const [rootId, rows] of groups.entries()) {
    const sorted = [...rows].sort((a, b) => {
      const delta = spec.score(b) - spec.score(a);
      if (delta !== 0) return delta;
      const aRank = asNumber((a.logicRayFrontier || a.pzrgCandidate?.logicRayFrontier || {}).rank, 1);
      const bRank = asNumber((b.logicRayFrontier || b.pzrgCandidate?.logicRayFrontier || {}).rank, 1);
      return aRank - bRank;
    });
    const annotated = sorted.map((row, index) => {
      const chrono = chronoByHash.get(row.logicRayFrontierHash) || null;
      return {
        bridgeId: row.bridgeId,
        logicRayFrontierHash: row.logicRayFrontierHash,
        move: row.move,
        originalRank: asNumber((row.logicRayFrontier || row.pzrgCandidate?.logicRayFrontier || {}).rank, 0),
        comparisonRank: index + 1,
        score: round(spec.score(row)),
        acceptedUsefulInjection: accepted(row, chrono),
        selectedMoveInFrontier: selected(row, chrono),
        usefulInjectionScore: useful(row, chrono),
      };
    });
    const rootAcceptedRows = annotated.filter((row) => row.acceptedUsefulInjection);
    if (rootAcceptedRows.length) {
      acceptedCandidateRoots += 1;
      meanAcceptedRank += rootAcceptedRows[0].comparisonRank;
      acceptedRankCount += 1;
    }
    if (annotated[0]?.acceptedUsefulInjection) top1Accepted += 1;
    if (annotated.slice(0, 3).some((row) => row.acceptedUsefulInjection)) top3Accepted += 1;
    if (annotated[0]?.selectedMoveInFrontier) selectedTop1 += 1;
    if (annotated.slice(0, 3).some((row) => row.selectedMoveInFrontier)) selectedTop3 += 1;
    roots.push({
      rootId,
      rowCount: rows.length,
      top1: annotated[0] || null,
      top3: annotated.slice(0, 3),
      acceptedCandidateCount: rootAcceptedRows.length,
      acceptedCandidateBestRank: rootAcceptedRows[0]?.comparisonRank || null,
    });
  }
  const rootCount = groups.size;
  return {
    name,
    family: spec.family,
    description: spec.description,
    rootCount,
    acceptedCandidateRoots,
    top1AcceptedUsefulInjections: top1Accepted,
    top1AcceptedUsefulInjectionRate: rootCount ? top1Accepted / rootCount : 0,
    top3AcceptedUsefulInjections: top3Accepted,
    top3AcceptedUsefulInjectionRate: rootCount ? top3Accepted / rootCount : 0,
    selectedMoveTop1: selectedTop1,
    selectedMoveTop3: selectedTop3,
    meanAcceptedCandidateRank: acceptedRankCount ? round(meanAcceptedRank / acceptedRankCount) : null,
    roots,
  };
}

function delta(name, candidate, baseline) {
  return {
    candidate: name,
    baseline: baseline.name,
    top1AcceptedUsefulInjectionDelta: candidate.top1AcceptedUsefulInjections - baseline.top1AcceptedUsefulInjections,
    top3AcceptedUsefulInjectionDelta: candidate.top3AcceptedUsefulInjections - baseline.top3AcceptedUsefulInjections,
    meanAcceptedCandidateRankDelta: (
      candidate.meanAcceptedCandidateRank != null && baseline.meanAcceptedCandidateRank != null
        ? round(candidate.meanAcceptedCandidateRank - baseline.meanAcceptedCandidateRank)
        : null
    ),
  };
}

function rowFeatureSummary(annotatedRow, bridgeByHash, chronoByHash) {
  if (!annotatedRow) return null;
  const bridge = bridgeByHash.get(annotatedRow.logicRayFrontierHash) || {};
  const chrono = chronoByHash.get(annotatedRow.logicRayFrontierHash) || {};
  const frontier = bridge.logicRayFrontier || bridge.pzrgCandidate?.logicRayFrontier || {};
  return {
    bridgeId: annotatedRow.bridgeId,
    hash: annotatedRow.logicRayFrontierHash,
    move: annotatedRow.move,
    originalRank: annotatedRow.originalRank,
    comparisonRank: annotatedRow.comparisonRank,
    score: annotatedRow.score,
    acceptedUsefulInjection: annotatedRow.acceptedUsefulInjection,
    selectedMoveInFrontier: annotatedRow.selectedMoveInFrontier,
    frontier: {
      utility: round(frontier.utility),
      pathProbability: round(frontier.pathProbability),
      risk: round(frontier.risk),
      lockIn: round(frontier.lockIn),
      scoreGapFromBestCp: round(frontier.scoreGapFromBestCp, 3),
    },
    chrono: {
      stabilityScore: round(chrono.diagnostics?.stabilityScore),
      uncertainty: round(chrono.uncertainty?.score),
      pathContortion: round(chrono.pathContortion?.score),
      normDrift: round(chrono.diagnostics?.normDrift),
      orthogonalityResidual: round(chrono.diagnostics?.orthogonalityResidual),
      eventHorizon: chrono.eventHorizon?.bucket || null,
    },
  };
}

function meanOf(rows, getter) {
  const values = rows.map(getter).filter((value) => Number.isFinite(value));
  if (!values.length) return null;
  return round(values.reduce((sum, value) => sum + value, 0) / values.length);
}

function featureMeans(rows) {
  return {
    count: rows.length,
    utility: meanOf(rows, (row) => row.frontier.utility),
    pathProbability: meanOf(rows, (row) => row.frontier.pathProbability),
    risk: meanOf(rows, (row) => row.frontier.risk),
    lockIn: meanOf(rows, (row) => row.frontier.lockIn),
    scoreGapFromBestCp: meanOf(rows, (row) => row.frontier.scoreGapFromBestCp),
    stabilityScore: meanOf(rows, (row) => row.chrono.stabilityScore),
    uncertainty: meanOf(rows, (row) => row.chrono.uncertainty),
    pathContortion: meanOf(rows, (row) => row.chrono.pathContortion),
    normDriftAbs: meanOf(rows, (row) => Math.abs(row.chrono.normDrift)),
  };
}

function buildDiagnostics(baselineEval, chronoEval, bridgeByHash, chronoByHash) {
  const baselineByRoot = new Map(baselineEval.roots.map((root) => [root.rootId, root]));
  const chronoByRoot = new Map(chronoEval.roots.map((root) => [root.rootId, root]));
  const regressions = [];
  const gains = [];
  const sharedSuccesses = [];
  const sharedFailures = [];
  for (const [rootId, baselineRoot] of baselineByRoot.entries()) {
    const chronoRoot = chronoByRoot.get(rootId);
    if (!chronoRoot) continue;
    const baselineTop = rowFeatureSummary(baselineRoot.top1, bridgeByHash, chronoByHash);
    const chronoTop = rowFeatureSummary(chronoRoot.top1, bridgeByHash, chronoByHash);
    const item = {
      rootId,
      rowCount: baselineRoot.rowCount,
      acceptedCandidateCount: baselineRoot.acceptedCandidateCount,
      baselineTop1: baselineTop,
      chronoTop1: chronoTop,
      baselineAcceptedBestRank: baselineRoot.acceptedCandidateBestRank,
      chronoAcceptedBestRank: chronoRoot.acceptedCandidateBestRank,
    };
    if (baselineTop?.acceptedUsefulInjection && !chronoTop?.acceptedUsefulInjection) {
      regressions.push(item);
    } else if (!baselineTop?.acceptedUsefulInjection && chronoTop?.acceptedUsefulInjection) {
      gains.push(item);
    } else if (baselineTop?.acceptedUsefulInjection && chronoTop?.acceptedUsefulInjection) {
      sharedSuccesses.push(item);
    } else {
      sharedFailures.push(item);
    }
  }
  const regressionBaselineRows = regressions.map((item) => item.baselineTop1).filter(Boolean);
  const regressionChronoRows = regressions.map((item) => item.chronoTop1).filter(Boolean);
  const gainBaselineRows = gains.map((item) => item.baselineTop1).filter(Boolean);
  const gainChronoRows = gains.map((item) => item.chronoTop1).filter(Boolean);
  return {
    rankers: {
      baseline: baselineEval.name,
      chrono: chronoEval.name,
    },
    counts: {
      regressions: regressions.length,
      gains: gains.length,
      sharedSuccesses: sharedSuccesses.length,
      sharedFailures: sharedFailures.length,
    },
    featureMeans: {
      regressionBaselineTop1: featureMeans(regressionBaselineRows),
      regressionChronoTop1: featureMeans(regressionChronoRows),
      gainBaselineTop1: featureMeans(gainBaselineRows),
      gainChronoTop1: featureMeans(gainChronoRows),
    },
    topRegressions: regressions.slice(0, 12),
    topGains: gains.slice(0, 12),
    interpretation: regressions.length > gains.length
      ? 'chrono ranker is displacing accepted frontier top choices more often than it recovers missed accepted choices'
      : 'chrono ranker recovers at least as many accepted choices as it displaces on top1',
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
  const bridgeByHash = rowsByHash(bridge);
  const groups = groupByRoot(bridgeRows);
  const missingChrono = bridgeRows.filter((row) => !chronoByHash.has(row.logicRayFrontierHash)).length;
  const specs = rankers(chronoByHash);
  const evaluations = Object.fromEntries(
    Object.entries(specs).map(([name, spec]) => [name, evaluateRanker(name, spec, groups, chronoByHash)]),
  );
  const baseline = evaluations.frontier_rank;
  const chronoDeltas = Object.fromEntries(
    Object.entries(evaluations)
      .filter(([, evaluation]) => evaluation.family === 'chrono')
      .map(([name, evaluation]) => [name, delta(name, evaluation, baseline)]),
  );
  const bestChrono = Object.values(evaluations)
    .filter((evaluation) => evaluation.family === 'chrono')
    .sort((a, b) => {
      const top1 = b.top1AcceptedUsefulInjections - a.top1AcceptedUsefulInjections;
      if (top1) return top1;
      const top3 = b.top3AcceptedUsefulInjections - a.top3AcceptedUsefulInjections;
      if (top3) return top3;
      return (a.meanAcceptedCandidateRank ?? 1e9) - (b.meanAcceptedCandidateRank ?? 1e9);
    })[0] || null;
  const bestDelta = bestChrono ? delta(bestChrono.name, bestChrono, baseline) : null;
  const liftObserved = Boolean(bestDelta && (
    bestDelta.top1AcceptedUsefulInjectionDelta > 0
      || bestDelta.top3AcceptedUsefulInjectionDelta > 0
      || (bestDelta.meanAcceptedCandidateRankDelta != null && bestDelta.meanAcceptedCandidateRankDelta < 0)
  ));
  const diagnostics = bestChrono
    ? buildDiagnostics(baseline, bestChrono, bridgeByHash, chronoByHash)
    : null;
  const output = {
    schemaVersion: SCHEMA_VERSION,
    generatedAt: new Date().toISOString(),
    condition: {
      source: path.resolve(args.conditionSource),
      runLabel: args.runLabel,
      metric: 'top-k accepted useful injections per root on documented offset-64 frontier rows',
      changedFields: 'none; post-hoc comparison over existing bridge and chrono artifacts',
    },
    sources: {
      bridgePath,
      bridgeSchemaVersion: bridge.schemaVersion || null,
      chronoPath,
      chronoSchemaVersion: chrono.schemaVersion || null,
      bridgeRows: bridgeRows.length,
      chronoRows: Array.isArray(chrono.rows) ? chrono.rows.length : 0,
      missingChronoRows: missingChrono,
    },
    comparison: {
      baselineRanker: 'frontier_rank',
      chronoRankers: Object.keys(evaluations).filter((name) => evaluations[name].family === 'chrono'),
      bestChronoRanker: bestChrono?.name || null,
      liftObserved,
      bestDelta,
      interpretation: liftObserved
        ? 'chrono sidecar produced post-hoc ranking lift on this fixed artifact; this is evidence for a runtime gate, not promotion'
        : 'chrono sidecar did not improve the post-hoc ranking metric on this fixed artifact',
    },
    promotionPolicy: {
      status: 'not_promoted',
      reason: 'post-hoc artifact comparison only; runtime choice integration and fixed GPU gate rerun are still required',
      requiredNextEvidence: [
        'CUDA-side ranker integration or explicit GPU gate consumer',
        'same heldout command rerun with chrono ranker active',
        'accepted useful injection lift without weakening strict parity/proxy status',
      ],
    },
    evaluations,
    chronoDeltas,
    diagnostics,
  };
  const outPath = path.resolve(args.out || defaultOutPath(chronoPath));
  fs.mkdirSync(path.dirname(outPath), { recursive: true });
  fs.writeFileSync(outPath, `${JSON.stringify(output, null, 2)}\n`);
  console.log(JSON.stringify({
    ok: true,
    output: outPath,
    rootCount: groups.size,
    bridgeRows: bridgeRows.length,
    missingChronoRows: missingChrono,
    baselineTop1Accepted: baseline.top1AcceptedUsefulInjections,
    bestChronoRanker: bestChrono?.name || null,
    bestChronoTop1Accepted: bestChrono?.top1AcceptedUsefulInjections ?? null,
    bestDelta,
    liftObserved,
    diagnostics: diagnostics ? diagnostics.counts : null,
  }, null, 2));
}

try {
  main();
} catch (error) {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
}
