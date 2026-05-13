#!/usr/bin/env node
import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';

const SCHEMA_VERSION = 'dojo.chrono_o2_failure_atlas.v1';

function usage() {
  return `Usage: node scripts/analyze_chrono_o2_failure_atlas.mjs --bridge <bridge.json> --chrono <chrono_o2_sidecar.json> --comparison <comparison.json> [--gate <gate.json>] [--out <atlas.json>]

Build a post-hoc failure atlas for PZRG_CHRONO_O2 on fixed GPU-derived
frontier artifacts. This reads existing JSON only; it does not run chess,
generate moves, modify runtime choices, or promote a ranker.
`;
}

function parseArgs(argv) {
  const args = {
    bridge: null,
    chrono: null,
    comparison: null,
    gate: null,
    out: null,
    conditionSource: '/home/frosty40/cuda_dojo/LOGIC_RAY_FRONTIER_RUNTIME_RECEIPT_2026-05-06.md',
    runLabel: 'posthoc_pzrg_chrono_o2_failure_atlas_on_source_temporal_bridge',
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
    } else if (token === '--comparison') {
      args.comparison = argv[++i];
    } else if (token === '--gate') {
      args.gate = argv[++i];
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
  if (!args.comparison) throw new Error(`missing --comparison\n${usage()}`);
  return args;
}

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, 'utf8'));
}

function sha256File(filePath) {
  return crypto.createHash('sha256').update(fs.readFileSync(filePath)).digest('hex');
}

function asNumber(value, fallback = 0) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

function clamp01(value, fallback = 0) {
  return Math.max(0, Math.min(1, asNumber(value, fallback)));
}

function round(value, digits = 6) {
  const number = asNumber(value, 0);
  const factor = 10 ** digits;
  return Math.round(number * factor) / factor;
}

function maybeRound(value, digits = 6) {
  return Number.isFinite(Number(value)) ? round(value, digits) : null;
}

function roundVector(values, digits = 6) {
  if (!Array.isArray(values)) return null;
  return values.map((value) => maybeRound(value, digits));
}

function vectorDelta(a, b, digits = 6) {
  if (!Array.isArray(a) || !Array.isArray(b)) return null;
  const length = Math.min(a.length, b.length);
  return Array.from({ length }, (_, index) => round(asNumber(a[index], 0) - asNumber(b[index], 0), digits));
}

function rowsByHash(bundle) {
  const result = new Map();
  for (const row of Array.isArray(bundle.rows) ? bundle.rows : []) {
    if (row.logicRayFrontierHash) result.set(row.logicRayFrontierHash, row);
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

function frontierRow(row) {
  return row.logicRayFrontier || row.pzrgCandidate?.logicRayFrontier || {};
}

function accepted(row, chrono) {
  const frontier = frontierRow(row);
  const injection = row.pzrgCandidate?.injection_relevance || {};
  return Boolean(
    frontier.gate?.acceptedUsefulInjection
      || injection.accepted_useful_injection
      || injection.promotion_gate_approved
      || chrono?.runtimeChoiceSignal?.acceptedUsefulInjection,
  );
}

function selected(row, chrono) {
  const frontier = frontierRow(row);
  return Boolean(frontier.gate?.selectedMoveInFrontier || chrono?.runtimeChoiceSignal?.selectedMoveInFrontier);
}

function useful(row, chrono) {
  const frontier = frontierRow(row);
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

function frontierRank(row) {
  return asNumber(frontierRow(row).rank ?? row.rank, 1);
}

function o2Score(chrono) {
  return (
    0.45 * clamp01(chrono?.diagnostics?.stabilityScore, 0)
    + 0.25 * clamp01(chrono?.relationDrift?.score, 0)
    + 0.15 * clamp01(chrono?.pressureDrift?.score, 0)
    - 0.35 * clamp01(chrono?.uncertainty?.score, 0)
    - 0.2 * clamp01(chrono?.pathContortion?.score, 0)
    - 0.15 * Math.min(1, Math.abs(asNumber(chrono?.diagnostics?.normDrift, 0)))
  );
}

function sortByFrontierRank(rows) {
  return [...rows].sort((a, b) => frontierRank(a) - frontierRank(b));
}

function sortByO2(rows, chronoByHash) {
  return [...rows].sort((a, b) => {
    const scoreDelta = o2Score(chronoByHash.get(b.logicRayFrontierHash)) - o2Score(chronoByHash.get(a.logicRayFrontierHash));
    if (scoreDelta !== 0) return scoreDelta;
    return frontierRank(a) - frontierRank(b);
  });
}

function countBy(rows, getter) {
  const counts = {};
  for (const row of rows) {
    const key = getter(row) || 'unknown';
    counts[key] = (counts[key] || 0) + 1;
  }
  return Object.fromEntries(Object.entries(counts).sort(([a], [b]) => a.localeCompare(b)));
}

function rowSummary(row, chronoByHash, rankMaps) {
  if (!row) return null;
  const chrono = chronoByHash.get(row.logicRayFrontierHash) || {};
  const frontier = frontierRow(row);
  const sourceTemporal = frontier.sourceTemporal || chrono.provenance?.sourceTemporal || null;
  return {
    bridgeId: row.bridgeId || row.pzrgCandidate?.id || null,
    hash: row.logicRayFrontierHash || null,
    move: row.move || frontier.move || null,
    originalRank: frontierRank(row),
    frontierRank: rankMaps.frontier.get(row.logicRayFrontierHash) || null,
    o2Rank: rankMaps.o2.get(row.logicRayFrontierHash) || null,
    scores: {
      frontierRank: round(-frontierRank(row)),
      chronoO2InternalProjector: round(o2Score(chrono)),
    },
    acceptedUsefulInjection: accepted(row, chrono),
    selectedMoveInFrontier: selected(row, chrono),
    usefulInjectionScore: round(useful(row, chrono)),
    frontier: {
      utility: maybeRound(frontier.utility),
      pathProbability: maybeRound(frontier.pathProbability),
      risk: maybeRound(frontier.risk),
      lockIn: maybeRound(frontier.lockIn),
      scoreCp: maybeRound(frontier.scoreCp, 3),
      scoreGapFromBestCp: maybeRound(frontier.scoreGapFromBestCp, 3),
      survivalBucket: frontier.survivalBucket || null,
      conversionBucket: frontier.conversionBucket || null,
    },
    chronoO2: {
      timePhase: chrono.timePhase ? {
        phase: chrono.timePhase.phase || null,
        theta: maybeRound(chrono.timePhase.theta),
        tau: maybeRound(chrono.tau),
      } : null,
      stabilityScore: maybeRound(chrono.diagnostics?.stabilityScore),
      uncertainty: maybeRound(chrono.uncertainty?.score),
      relationDrift: maybeRound(chrono.relationDrift?.score),
      pressureDrift: maybeRound(chrono.pressureDrift?.score),
      pathContortion: maybeRound(chrono.pathContortion?.score),
      normDrift: maybeRound(chrono.diagnostics?.normDrift),
      orthogonalityResidual: maybeRound(chrono.diagnostics?.orthogonalityResidual),
      eventHorizon: chrono.eventHorizon?.bucket || null,
      finiteDifferenceMode: chrono.provenance?.finiteDifferenceMode || null,
      sourceOrderProxy: Boolean(chrono.provenance?.sourceOrderProxy),
      missingNeighbor: Boolean(chrono.provenance?.missingNeighbor),
      bootstrapNotLearned: Boolean(chrono.responseTensors?.bootstrapNotLearned || chrono.provenance?.bootstrapNotLearned),
      vectors: {
        z: roundVector(chrono.vectors?.z),
        u: roundVector(chrono.vectors?.u),
        p: roundVector(chrono.vectors?.p),
        n: roundVector(chrono.vectors?.n),
        F_ext: roundVector(chrono.vectors?.F_ext),
        F_cont: roundVector(chrono.vectors?.F_cont),
      },
      sourceTemporal: sourceTemporal ? {
        sourceFamily: sourceTemporal.sourceFamily || null,
        sourceSequenceId: sourceTemporal.sourceSequenceId || null,
        sourceIndex: sourceTemporal.sourceIndex ?? null,
        sourcePly: sourceTemporal.sourcePly ?? null,
        transitionVerified: Boolean(sourceTemporal.transitionVerified),
      } : null,
    },
  };
}

function evaluationSnapshot(comparison, name) {
  const evaluation = comparison.evaluations?.[name] || null;
  if (!evaluation) return null;
  return {
    name: evaluation.name || name,
    family: evaluation.family || null,
    rootCount: evaluation.rootCount ?? null,
    top1AcceptedUsefulInjections: evaluation.top1AcceptedUsefulInjections ?? null,
    top3AcceptedUsefulInjections: evaluation.top3AcceptedUsefulInjections ?? null,
    selectedMoveTop1: evaluation.selectedMoveTop1 ?? null,
    selectedMoveTop3: evaluation.selectedMoveTop3 ?? null,
    meanAcceptedCandidateRank: evaluation.meanAcceptedCandidateRank ?? null,
  };
}

function buildRootRecord(rootId, rows, chronoByHash) {
  const frontierSorted = sortByFrontierRank(rows);
  const o2Sorted = sortByO2(rows, chronoByHash);
  const frontierRankMap = new Map(frontierSorted.map((row, index) => [row.logicRayFrontierHash, index + 1]));
  const o2RankMap = new Map(o2Sorted.map((row, index) => [row.logicRayFrontierHash, index + 1]));
  const rankMaps = { frontier: frontierRankMap, o2: o2RankMap };
  const annotated = rows.map((row) => rowSummary(row, chronoByHash, rankMaps));
  const acceptedRows = rows.filter((row) => accepted(row, chronoByHash.get(row.logicRayFrontierHash)));
  const acceptedByFrontier = sortByFrontierRank(acceptedRows);
  const acceptedByO2 = sortByO2(acceptedRows, chronoByHash);
  const baselineTop1 = frontierSorted[0] || null;
  const o2Top1 = o2Sorted[0] || null;
  const baselineTop1Summary = rowSummary(baselineTop1, chronoByHash, rankMaps);
  const o2Top1Summary = rowSummary(o2Top1, chronoByHash, rankMaps);
  const acceptedFrontierBest = acceptedByFrontier[0] || null;
  const acceptedO2Best = acceptedByO2[0] || null;
  const finiteDifferenceModeCounts = countBy(rows, (row) => chronoByHash.get(row.logicRayFrontierHash)?.provenance?.finiteDifferenceMode);
  const sourceOrderProxyRows = rows.filter((row) => chronoByHash.get(row.logicRayFrontierHash)?.provenance?.sourceOrderProxy).length;
  const missingNeighborRows = rows.filter((row) => chronoByHash.get(row.logicRayFrontierHash)?.provenance?.missingNeighbor).length;
  const bootstrapNotLearnedRows = rows.filter((row) => {
    const chrono = chronoByHash.get(row.logicRayFrontierHash) || {};
    return chrono.responseTensors?.bootstrapNotLearned || chrono.provenance?.bootstrapNotLearned;
  }).length;
  const baselineAcceptedBestRank = acceptedFrontierBest ? frontierRankMap.get(acceptedFrontierBest.logicRayFrontierHash) : null;
  const o2AcceptedBestRank = acceptedO2Best ? o2RankMap.get(acceptedO2Best.logicRayFrontierHash) : null;
  const classes = [];
  if (baselineTop1Summary?.acceptedUsefulInjection && !o2Top1Summary?.acceptedUsefulInjection) {
    classes.push('o2_demotes_frontier_accepted_top1');
  }
  if (!baselineTop1Summary?.acceptedUsefulInjection && o2Top1Summary?.acceptedUsefulInjection) {
    classes.push('o2_recovers_frontier_miss');
  }
  if (baselineAcceptedBestRank != null && baselineAcceptedBestRank <= 3 && (o2AcceptedBestRank == null || o2AcceptedBestRank > 3)) {
    classes.push('accepted_candidate_demoted_beyond_top3');
  }
  if (baselineAcceptedBestRank === 1 && o2AcceptedBestRank !== 1) {
    classes.push('accepted_candidate_lost_top1');
  }
  if (missingNeighborRows > 0) {
    classes.push('missing_neighbor_quarantine_root');
  }
  if (sourceOrderProxyRows === rows.length) {
    classes.push('source_order_proxy_all_rows');
  }
  if (bootstrapNotLearnedRows > 0) {
    classes.push('bootstrap_k_not_learned');
  }
  return {
    rootId,
    rootFen: frontierRow(frontierSorted[0] || {}).rootFen || frontierSorted[0]?.rootFen || null,
    rowCount: rows.length,
    classes,
    baselineTop1: baselineTop1Summary,
    o2Top1: o2Top1Summary,
    acceptedCandidateCount: acceptedRows.length,
    baselineAcceptedBestRank,
    o2AcceptedBestRank,
    acceptedRankDelta: baselineAcceptedBestRank != null && o2AcceptedBestRank != null
      ? o2AcceptedBestRank - baselineAcceptedBestRank
      : null,
    finiteDifferenceModeCounts,
    sourceOrderProxyRows,
    missingNeighborRows,
    bootstrapNotLearnedRows,
    topAcceptedCandidates: acceptedByO2.slice(0, 5).map((row) => rowSummary(row, chronoByHash, rankMaps)),
    topO2Candidates: o2Sorted.slice(0, 5).map((row) => rowSummary(row, chronoByHash, rankMaps)),
    topFrontierCandidates: frontierSorted.slice(0, 5).map((row) => rowSummary(row, chronoByHash, rankMaps)),
    _rankMaps: rankMaps,
    _rows: rows,
    _acceptedByFrontier: acceptedByFrontier,
    _o2Top1: o2Top1,
  };
}

function publicRootRecord(record) {
  const { _rankMaps, _rows, _acceptedByFrontier, _o2Top1, ...rest } = record;
  void _rankMaps;
  void _rows;
  void _acceptedByFrontier;
  void _o2Top1;
  return rest;
}

function learningTarget(record, chronoByHash) {
  const acceptedBest = record._acceptedByFrontier[0] || null;
  const o2Top = record._o2Top1 || null;
  if (!acceptedBest || !o2Top) return null;
  const acceptedChrono = chronoByHash.get(acceptedBest.logicRayFrontierHash) || {};
  const o2Chrono = chronoByHash.get(o2Top.logicRayFrontierHash) || {};
  return {
    targetType: 'posthoc_learning_target_not_promoted',
    rootId: record.rootId,
    rootFen: record.rootFen,
    class: record.classes.includes('o2_demotes_frontier_accepted_top1')
      ? 'o2_demotes_frontier_accepted_top1'
      : 'accepted_candidate_o2_rank_lag',
    acceptedMove: acceptedBest.move,
    o2TopMove: o2Top.move,
    baselineAcceptedBestRank: record.baselineAcceptedBestRank,
    o2AcceptedBestRank: record.o2AcceptedBestRank,
    acceptedMinusO2Score: round(o2Score(acceptedChrono) - o2Score(o2Chrono)),
    deltaN: vectorDelta(acceptedChrono.vectors?.n, o2Chrono.vectors?.n),
    deltaFCont: vectorDelta(acceptedChrono.vectors?.F_cont, o2Chrono.vectors?.F_cont),
    accepted: {
      hash: acceptedBest.logicRayFrontierHash,
      move: acceptedBest.move,
      vectors: {
        n: roundVector(acceptedChrono.vectors?.n),
        F_cont: roundVector(acceptedChrono.vectors?.F_cont),
      },
      diagnostics: {
        stabilityScore: maybeRound(acceptedChrono.diagnostics?.stabilityScore),
        uncertainty: maybeRound(acceptedChrono.uncertainty?.score),
        normDrift: maybeRound(acceptedChrono.diagnostics?.normDrift),
      },
    },
    o2Top: {
      hash: o2Top.logicRayFrontierHash,
      move: o2Top.move,
      vectors: {
        n: roundVector(o2Chrono.vectors?.n),
        F_cont: roundVector(o2Chrono.vectors?.F_cont),
      },
      diagnostics: {
        stabilityScore: maybeRound(o2Chrono.diagnostics?.stabilityScore),
        uncertainty: maybeRound(o2Chrono.uncertainty?.score),
        normDrift: maybeRound(o2Chrono.diagnostics?.normDrift),
      },
    },
  };
}

function quarantineTarget(record, chronoByHash) {
  const missingRows = record._rows.filter((row) => chronoByHash.get(row.logicRayFrontierHash)?.provenance?.missingNeighbor);
  if (!missingRows.length) return null;
  return {
    targetType: 'chrono_o2_transition_quarantine',
    reason: 'missing_neighbor_zero_velocity_quarantine',
    rootId: record.rootId,
    rootFen: record.rootFen,
    missingNeighborRows: missingRows.length,
    rows: missingRows.map((row) => {
      const chrono = chronoByHash.get(row.logicRayFrontierHash) || {};
      const sourceTemporal = chrono.provenance?.sourceTemporal || frontierRow(row).sourceTemporal || {};
      return {
        hash: row.logicRayFrontierHash,
        bridgeId: row.bridgeId || null,
        move: row.move || null,
        sourceSequenceId: sourceTemporal.sourceSequenceId || null,
        sourceIndex: sourceTemporal.sourceIndex ?? null,
        sourcePly: sourceTemporal.sourcePly ?? null,
        finiteDifferenceMode: chrono.provenance?.finiteDifferenceMode || null,
      };
    }),
  };
}

function buildSummary(rootRecords, bridgeRows, chronoRows, comparison, chrono, gate) {
  const baseline = evaluationSnapshot(comparison, 'frontier_rank');
  const o2 = evaluationSnapshot(comparison, 'chrono_o2_internal_projector');
  const classCounts = {};
  for (const record of rootRecords) {
    for (const klass of record.classes) classCounts[klass] = (classCounts[klass] || 0) + 1;
  }
  const sourceOrderProxyRows = chronoRows.filter((row) => row.provenance?.sourceOrderProxy).length;
  const missingNeighborRows = chronoRows.filter((row) => row.provenance?.missingNeighbor).length;
  const bootstrapNotLearnedRows = chronoRows.filter((row) => row.responseTensors?.bootstrapNotLearned || row.provenance?.bootstrapNotLearned).length;
  return {
    rootCount: rootRecords.length,
    bridgeRows: bridgeRows.length,
    chronoRows: chronoRows.length,
    baseline,
    chronoO2: o2,
    top1Regressions: classCounts.o2_demotes_frontier_accepted_top1 || 0,
    top1Gains: classCounts.o2_recovers_frontier_miss || 0,
    sharedTop1Successes: rootRecords.filter((record) => (
      record.baselineTop1?.acceptedUsefulInjection && record.o2Top1?.acceptedUsefulInjection
    )).length,
    sharedTop1Failures: rootRecords.filter((record) => (
      !record.baselineTop1?.acceptedUsefulInjection && !record.o2Top1?.acceptedUsefulInjection
    )).length,
    acceptedDemotedBeyondTop3Roots: classCounts.accepted_candidate_demoted_beyond_top3 || 0,
    sourceOrderProxyRows,
    sourceOrderProxyRoots: rootRecords.filter((record) => record.sourceOrderProxyRows > 0).length,
    missingNeighborRows,
    missingNeighborRoots: rootRecords.filter((record) => record.missingNeighborRows > 0).length,
    bootstrapNotLearnedRows,
    bootstrapNotLearnedRoots: rootRecords.filter((record) => record.bootstrapNotLearnedRows > 0).length,
    sidecarPromotionBlockers: chrono.promotionPolicy?.blockers || [],
    gateStatus: gate?.decision?.status || gate?.promotionPolicy?.status || gate?.status || null,
    gatePromote: gate?.decision?.promote ?? gate?.promotionPolicy?.promote ?? gate?.promote ?? null,
    gateBlockingCheckCount: (
      gate?.decision?.blockingCheckCount
        ?? gate?.promotionPolicy?.blockingCheckCount
        ?? gate?.blockingCheckCount
        ?? null
    ),
    classCounts: Object.fromEntries(Object.entries(classCounts).sort(([a], [b]) => a.localeCompare(b))),
  };
}

function defaultOutPath(chronoPath) {
  const parsed = path.parse(chronoPath);
  const stem = parsed.base.replace(/\.chrono_o2_sidecar\.json$/, '').replace(/\.json$/, '');
  return path.join(parsed.dir, `${stem}.chrono_o2_failure_atlas.json`);
}

function main() {
  const args = parseArgs(process.argv.slice(2));
  const bridgePath = path.resolve(args.bridge);
  const chronoPath = path.resolve(args.chrono);
  const comparisonPath = path.resolve(args.comparison);
  const gatePath = args.gate ? path.resolve(args.gate) : null;
  const bridge = readJson(bridgePath);
  const chrono = readJson(chronoPath);
  const comparison = readJson(comparisonPath);
  const gate = gatePath ? readJson(gatePath) : null;
  const bridgeRows = Array.isArray(bridge.rows) ? bridge.rows : [];
  const chronoRows = Array.isArray(chrono.rows) ? chrono.rows : [];
  const chronoByHash = rowsByHash(chrono);
  const rootRecordsWithInternals = Array.from(groupByRoot(bridgeRows).entries())
    .map(([rootId, rows]) => buildRootRecord(rootId, rows, chronoByHash));
  const learningTargets = rootRecordsWithInternals
    .filter((record) => record.acceptedCandidateCount > 0 && (
      record.classes.includes('o2_demotes_frontier_accepted_top1')
        || (record.o2AcceptedBestRank != null && record.baselineAcceptedBestRank != null && record.o2AcceptedBestRank > record.baselineAcceptedBestRank)
    ))
    .map((record) => learningTarget(record, chronoByHash))
    .filter(Boolean)
    .sort((a, b) => (b.o2AcceptedBestRank - b.baselineAcceptedBestRank) - (a.o2AcceptedBestRank - a.baselineAcceptedBestRank));
  const quarantineTargets = rootRecordsWithInternals
    .map((record) => quarantineTarget(record, chronoByHash))
    .filter(Boolean);
  const responseTensorTargets = chrono.promotionPolicy?.blockers?.includes('bootstrap_not_learned_response_tensors')
    ? [{
      targetType: 'chrono_o2_response_tensor_learning_gate',
      reason: 'bootstrap_not_learned_response_tensors',
      requiredEvidence: [
        ...(chrono.promotionPolicy?.blockers?.includes('source_order_proxy_transitions_unverified')
          ? ['transition-verified chrono source, not source-order proxy alone']
          : []),
        'learned or recorded K0/Kc/Ks tensors with non-bootstrap provenance',
        'fixed-artifact comparison lift and heldout GPU gate rerun before promotion',
      ],
      currentTensorSummary: chrono.responseTensorPolicy || chrono.responseTensors || null,
    }]
    : [];
  const rootRecords = rootRecordsWithInternals.map(publicRootRecord);
  const summary = buildSummary(rootRecords, bridgeRows, chronoRows, comparison, chrono, gate);
  const requiredNextEvidence = [
    ...(summary.sourceOrderProxyRows
      ? ['replace source-order proxy with transition-verified temporal source']
      : []),
    ...(summary.missingNeighborRows
      ? ['remove missing-neighbor quarantine rows or prove they do not affect candidate ordering']
      : []),
    'learn or record non-bootstrap K0/Kc/Ks response tensors',
    'show positive fixed-artifact and heldout GPU-gate lift before runtime integration',
  ];
  const output = {
    schemaVersion: SCHEMA_VERSION,
    generatedAt: new Date().toISOString(),
    projectionSystem: chrono.projectionSystem || 'PZRG_CHRONO_O2',
    condition: {
      source: path.resolve(args.conditionSource),
      runLabel: args.runLabel,
      metric: 'root-level failure attribution between frontier_rank and chrono_o2_internal_projector on fixed source-temporal bridge rows',
      changedFields: 'none; post-hoc artifact-only atlas over existing GPU-derived bridge, sidecar, comparison, and gate reports',
      labCondition: 'posthoc/proxy/subset; not a runtime promotion gate',
    },
    sources: {
      bridgePath,
      bridgeSchemaVersion: bridge.schemaVersion || null,
      bridgeSha256: sha256File(bridgePath),
      chronoPath,
      chronoSchemaVersion: chrono.schemaVersion || null,
      chronoSha256: sha256File(chronoPath),
      comparisonPath,
      comparisonSchemaVersion: comparison.schemaVersion || null,
      comparisonSha256: sha256File(comparisonPath),
      gatePath,
      gateSchemaVersion: gate?.schemaVersion || null,
      gateSha256: gatePath ? sha256File(gatePath) : null,
    },
    summary,
    failureClasses: {
      o2_demotes_frontier_accepted_top1: 'frontier rank top1 was an accepted useful injection, but the O2 internal projector top1 was not',
      o2_recovers_frontier_miss: 'frontier rank top1 missed an accepted useful injection, but the O2 internal projector top1 selected one',
      accepted_candidate_demoted_beyond_top3: 'an accepted candidate available in the frontier top3 fell outside O2 top3',
      accepted_candidate_lost_top1: 'an accepted frontier top1 was not O2 top1',
      missing_neighbor_quarantine_root: 'one or more O2 rows used missing-neighbor zero-velocity quarantine',
      source_order_proxy_all_rows: 'all rows in the root used recorded source order proxy evidence with transitions unverified',
      bootstrap_k_not_learned: 'one or more rows used bootstrap K0/Kc/Ks response tensors',
    },
    rootRecords,
    learningTargets: learningTargets.slice(0, 64),
    quarantineTargets,
    responseTensorTargets,
    promotionPolicy: {
      status: 'not_promoted',
      reason: 'failure atlas only; O2 remains blocked by internal sidecar blockers and no fixed-artifact/cross-validated lift',
      requiredNextEvidence,
    },
  };
  const outPath = path.resolve(args.out || defaultOutPath(chronoPath));
  fs.mkdirSync(path.dirname(outPath), { recursive: true });
  fs.writeFileSync(outPath, `${JSON.stringify(output, null, 2)}\n`);
  console.log(JSON.stringify({
    ok: true,
    output: outPath,
    projectionSystem: output.projectionSystem,
    rootCount: output.summary.rootCount,
    bridgeRows: output.summary.bridgeRows,
    chronoRows: output.summary.chronoRows,
    top1Regressions: output.summary.top1Regressions,
    top1Gains: output.summary.top1Gains,
    acceptedDemotedBeyondTop3Roots: output.summary.acceptedDemotedBeyondTop3Roots,
    missingNeighborRoots: output.summary.missingNeighborRoots,
    learningTargets: output.learningTargets.length,
    quarantineTargets: output.quarantineTargets.length,
    promote: false,
  }, null, 2));
}

try {
  main();
} catch (error) {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
}
