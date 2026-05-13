#!/usr/bin/env node
import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';

const SCHEMA_VERSION = 'dojo.omnifold_narrowed_runtime_guard_sweep.v1';
const DEFAULT_CONDITION = '/home/frosty40/cuda_dojo/LOGIC_RAY_FRONTIER_RUNTIME_RECEIPT_2026-05-06.md';
const DEFAULT_VARIANT = 'R4_piece_zone_move_family_ray_relation_fold2x2_negative_yspace';

function usage() {
  return `Usage: node scripts/sweep_omnifold_narrowed_runtime_guards.mjs --resolution <selected_family_resolution.json> [--preflight <preflight.json>] [--chrono <chrono_o2_tactical_sidecar.json>] [--out <guard_sweep.json>]

Sweep runtime-visible guards for the narrowed R4 negative-yspace OmniFold
consumer. This reads recorded CUDA-derived rows only. It does not run chess
logic, does not use CPU legality, and does not change runtime behavior.
`;
}

function parseArgs(argv) {
  const args = {
    resolution: null,
    preflight: null,
    chrono: null,
    out: null,
    variant: DEFAULT_VARIANT,
    conditionSource: DEFAULT_CONDITION,
  };
  for (let i = 0; i < argv.length; i += 1) {
    const token = argv[i];
    if (token === '--help' || token === '-h') {
      console.log(usage());
      process.exit(0);
    } else if (token === '--resolution') {
      args.resolution = argv[++i];
    } else if (token === '--preflight') {
      args.preflight = argv[++i];
    } else if (token === '--chrono') {
      args.chrono = argv[++i];
    } else if (token === '--out') {
      args.out = argv[++i];
    } else if (token === '--variant') {
      args.variant = argv[++i];
    } else if (token === '--condition-source') {
      args.conditionSource = argv[++i];
    } else {
      throw new Error(`unknown argument: ${token}\n${usage()}`);
    }
  }
  if (!args.resolution) throw new Error(`missing --resolution\n${usage()}`);
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

function moveGeometry(move) {
  const text = String(move || '');
  if (text.length < 4) {
    return {
      df: null,
      dr: null,
      knightLike: false,
      edgeTarget: false,
      centerTarget: false,
      longMove: false,
    };
  }
  const fromFile = text.charCodeAt(0) - 97;
  const fromRank = Number(text[1]) - 1;
  const toFile = text.charCodeAt(2) - 97;
  const toRank = Number(text[3]) - 1;
  const df = Math.abs(toFile - fromFile);
  const dr = Math.abs(toRank - fromRank);
  const distance = Math.sqrt(df * df + dr * dr);
  return {
    df,
    dr,
    knightLike: Math.min(df, dr) === 1 && Math.max(df, dr) === 2,
    edgeTarget: toFile === 0 || toFile === 7 || toRank === 0 || toRank === 7,
    centerTarget: toFile >= 2 && toFile <= 5 && toRank >= 2 && toRank <= 5,
    longMove: distance >= 2.5,
  };
}

function chronoByHash(chrono) {
  const rows = Array.isArray(chrono?.rows) ? chrono.rows : [];
  return new Map(rows
    .filter((row) => row.logicRayFrontierHash)
    .map((row) => [row.logicRayFrontierHash, row]));
}

function chronoFeatures(chronoRow) {
  const tactical = chronoRow?.tacticalContact || {};
  const flags = tactical.flags || {};
  const counts = tactical.counts || {};
  return {
    available: Boolean(chronoRow),
    pressureBucket: chronoRow?.pressureDrift?.bucket || null,
    relationBucket: chronoRow?.relationDrift?.bucket || null,
    contortionBucket: chronoRow?.pathContortion?.bucket || null,
    uncertaintyBucket: chronoRow?.uncertainty?.bucket || null,
    stabilityScore: asNumber(chronoRow?.diagnostics?.stabilityScore, 0),
    normDrift: asNumber(chronoRow?.diagnostics?.normDrift, 0),
    tacticalGpuVerified: Boolean(tactical.gpuVerified),
    destinationDefendedAfter: Boolean(flags.destinationDefendedAfter),
    destinationAttackedAfter: Boolean(flags.destinationAttackedAfter),
    destinationContestedAfter: Boolean(flags.destinationDefendedAfter && flags.destinationAttackedAfter),
    moverInCheckBefore: Boolean(flags.moverInCheckBefore),
    givesCheckAfter: Boolean(flags.givesCheckAfter),
    captureLike: Boolean(flags.captureLike),
    promotion: Boolean(flags.promotion),
    opponentLegalRepliesAfter: asNumber(counts.opponentLegalRepliesAfter, 0),
    contactBalanceAfter: asNumber(counts.contactBalanceAfter, 0),
  };
}

function normalizeRows(resolution, chronoRows = new Map()) {
  return (Array.isArray(resolution.rowAssignments) ? resolution.rowAssignments : [])
    .filter((row) => row.rootId && row.move)
    .map((row) => {
      const selected = row.resolution?.selected || {};
      const groups = Array.isArray(selected.selectedVariantGroups)
        ? selected.selectedVariantGroups.map(String).sort()
        : [];
      const hasFightScore = row.fightScore !== null
        && row.fightScore !== undefined
        && Number.isFinite(Number(row.fightScore));
      return {
        rootId: String(row.rootId),
        bridgeId: row.bridgeId || null,
        hash: row.logicRayFrontierHash || null,
        move: String(row.move),
        geometry: moveGeometry(row.move),
        rank: Math.max(1, Math.trunc(asNumber(row.rank, 1))),
        acceptedUsefulInjection: Boolean(row.acceptedUsefulInjection),
        selectedMoveInFrontier: Boolean(row.selectedMoveInFrontier),
        fightScore: hasFightScore ? asNumber(row.fightScore) : null,
        selectedFamily: row.selectedFamily || null,
        selectedFamilyId: row.selectedFamilyId || null,
        selectedVariantId: row.selectedVariantId || null,
        confidence: asNumber(row.resolution?.confidence, 0),
        margin: asNumber(row.resolution?.margin, 0),
        selectedVariantGroups: groups,
        groupScores: row.resolution?.groupScores || {},
        chrono: chronoFeatures(chronoRows.get(row.logicRayFrontierHash)),
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

function baselineChoice(rows) {
  return rows.slice().sort((a, b) => a.rank - b.rank || a.move.localeCompare(b.move))[0] || null;
}

function guardAllows(row, baseline, guard) {
  if (!baseline) return false;
  if (row.selectedVariantId !== guard.variant) return false;
  if (row.rank > guard.maxRank) return false;
  if (row.confidence < guard.minCandidateConfidence) return false;
  if (row.margin < guard.minCandidateMargin) return false;
  if ((row.rank - baseline.rank) > guard.maxRankJump) return false;
  if ((baseline.confidence - row.confidence) > guard.maxConfidenceDrop) return false;
  if (guard.protectBaselineSelected && baseline.selectedMoveInFrontier) return false;
  if (guard.requireCandidateSelected && !row.selectedMoveInFrontier) return false;
  if (guard.requireKnightLike && !row.geometry.knightLike) return false;
  if (guard.requireEdgeTarget && !row.geometry.edgeTarget) return false;
  if (guard.requireCenterTarget && !row.geometry.centerTarget) return false;
  if (guard.requireIntersection && asNumber(row.groupScores.fold2x2_intersection, 0) < 0.9) return false;
  if (guard.requireLocalContact && asNumber(row.groupScores.fold2x2_local_contact, 0) < 0.9) return false;
  if (guard.blockWeakRay && asNumber(row.groupScores.ray_only, 0) < guard.minRayScore) return false;
  if (guard.requireChronoAvailable && !row.chrono.available) return false;
  if (guard.requireTacticalGpuVerified && !row.chrono.tacticalGpuVerified) return false;
  if (guard.requireCaptureLike && !row.chrono.captureLike) return false;
  if (guard.forbidCaptureLike && row.chrono.captureLike) return false;
  if (guard.requireMoverInCheckBefore && !row.chrono.moverInCheckBefore) return false;
  if (guard.forbidMoverInCheckBefore && row.chrono.moverInCheckBefore) return false;
  if (guard.requireDestinationContestedAfter && !row.chrono.destinationContestedAfter) return false;
  if (guard.requireContactBalanceNonNegative && row.chrono.contactBalanceAfter < 0) return false;
  if (guard.requireContactBalancePositive && row.chrono.contactBalanceAfter <= 0) return false;
  if (guard.maxOpponentLegalRepliesAfter !== null
    && row.chrono.opponentLegalRepliesAfter > guard.maxOpponentLegalRepliesAfter) return false;
  if (guard.requireUnstablePressure && row.chrono.pressureBucket !== 'unstable') return false;
  if (guard.requireRisingContortion && row.chrono.contortionBucket !== 'rising') return false;
  if (row.chrono.stabilityScore < guard.minStabilityScore) return false;
  return true;
}

function scoreRow(row, baseline, guard) {
  return (guardAllows(row, baseline, guard) ? guard.bonus : 0)
    + guard.confidenceWeight * row.confidence
    + guard.marginWeight * row.margin
    - guard.rankPenalty * (row.rank - 1);
}

function publicChoice(row, score, baseline) {
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
    changedFromFrontierRank: baseline ? row.hash !== baseline.hash : false,
  };
}

function evaluate(groups, guard, name) {
  let scoreSum = 0;
  let fightTop1Rows = 0;
  let positiveTop1 = 0;
  let neutralTop1 = 0;
  let negativeTop1 = 0;
  let acceptedTop1 = 0;
  let selectedTop1 = 0;
  let changedRootCount = 0;
  let guardMatchedRows = 0;
  const changedRoots = [];
  const roots = [];

  for (const [rootId, rows] of groups) {
    const baseline = baselineChoice(rows);
    if (!baseline) continue;
    const ranked = rows
      .map((row) => {
        const matched = guardAllows(row, baseline, guard);
        if (matched) guardMatchedRows += 1;
        return { row, score: scoreRow(row, baseline, guard), matched };
      })
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

    const topPublic = publicChoice(top.row, top.score, baseline);
    roots.push({ rootId, rowCount: rows.length, top1: topPublic });
    if (top.row.hash !== baseline.hash) {
      changedRootCount += 1;
      changedRoots.push({
        rootId,
        baselineTop1: publicChoice(baseline, -(baseline.rank - 1), baseline),
        candidateTop1: topPublic,
        deltaFightScore: baseline.fightScore != null && top.row.fightScore != null
          ? round(top.row.fightScore - baseline.fightScore)
          : null,
        acceptedDelta: Number(top.row.acceptedUsefulInjection) - Number(baseline.acceptedUsefulInjection),
        selectedDelta: Number(top.row.selectedMoveInFrontier) - Number(baseline.selectedMoveInFrontier),
      });
    }
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
    changedRootCount,
    guardMatchedRows,
    changedRoots,
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
    changedRootCount: metrics.changedRootCount,
    guardMatchedRows: metrics.guardMatchedRows,
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
    changedRootCount: candidate.changedRootCount,
  };
}

function evaluateSurface(name, rows, guard) {
  const groups = groupByRoot(rows);
  const baselineGuard = {
    ...guard,
    variant: '__never_match__',
    bonus: 0,
    confidenceWeight: 0,
    marginWeight: 0,
    rankPenalty: 1,
  };
  const baseline = evaluate(groups, baselineGuard, `${name}_frontier_rank`);
  const candidate = evaluate(groups, guard, `${name}_guarded_consumer`);
  return {
    name,
    corpus: {
      rowCount: rows.length,
      rootCount: groups.length,
      acceptedRows: rows.filter((row) => row.acceptedUsefulInjection).length,
      selectedRows: rows.filter((row) => row.selectedMoveInFrontier).length,
      fightRows: rows.filter((row) => Number.isFinite(row.fightScore)).length,
    },
    baseline: metricsPublic(baseline),
    candidate: metricsPublic(candidate),
    deltaVsFrontierRank: delta(candidate, baseline),
    changedRoots: candidate.changedRoots,
  };
}

function isFreshGateEligible(allRowsSurface, fightRowsSurface) {
  const acceptedSafe = allRowsSurface.deltaVsFrontierRank.acceptedUsefulTop1 >= 0
    && fightRowsSurface.deltaVsFrontierRank.acceptedUsefulTop1 >= 0;
  const selectedSafe = allRowsSurface.deltaVsFrontierRank.selectedMoveTop1 >= 0
    && fightRowsSurface.deltaVsFrontierRank.selectedMoveTop1 >= 0;
  const fightLift = (
    fightRowsSurface.deltaVsFrontierRank.meanFightScore != null
      && fightRowsSurface.deltaVsFrontierRank.meanFightScore > 0
  ) || fightRowsSurface.deltaVsFrontierRank.positiveTop1 > 0;
  const changed = allRowsSurface.deltaVsFrontierRank.changedRootCount > 0
    || fightRowsSurface.deltaVsFrontierRank.changedRootCount > 0;
  return {
    acceptedSafe,
    selectedSafe,
    fightLift,
    changed,
    ready: acceptedSafe && selectedSafe && fightLift && changed,
  };
}

function guardId(guard) {
  return sha256Object(guard).slice(0, 16);
}

function scoreCandidate(result) {
  const fightDelta = asNumber(result.fightLabeledSurface.deltaVsFrontierRank.meanFightScore, -1);
  const positiveDelta = asNumber(result.fightLabeledSurface.deltaVsFrontierRank.positiveTop1, 0);
  const acceptedDelta = asNumber(result.allRowsAcceptedSurface.deltaVsFrontierRank.acceptedUsefulTop1, 0)
    + asNumber(result.fightLabeledSurface.deltaVsFrontierRank.acceptedUsefulTop1, 0);
  const selectedDelta = asNumber(result.allRowsAcceptedSurface.deltaVsFrontierRank.selectedMoveTop1, 0)
    + asNumber(result.fightLabeledSurface.deltaVsFrontierRank.selectedMoveTop1, 0);
  const changed = asNumber(result.allRowsAcceptedSurface.deltaVsFrontierRank.changedRootCount, 0)
    + asNumber(result.fightLabeledSurface.deltaVsFrontierRank.changedRootCount, 0);
  return round(fightDelta * 1000 + positiveDelta * 10 + acceptedDelta * 5 + selectedDelta * 3 + changed * 0.01);
}

function compactResult(result) {
  return {
    guardId: result.guardId,
    score: result.score,
    guard: result.guard,
    eligibility: result.eligibility,
    allRowsDelta: result.allRowsAcceptedSurface.deltaVsFrontierRank,
    fightRowsDelta: result.fightLabeledSurface.deltaVsFrontierRank,
    allRowsChangedRootCount: result.allRowsAcceptedSurface.changedRoots.length,
    fightRowsChangedRootCount: result.fightLabeledSurface.changedRoots.length,
    changedRootsPreview: {
      allRows: result.allRowsAcceptedSurface.changedRoots.slice(0, 8),
      fightRows: result.fightLabeledSurface.changedRoots.slice(0, 8),
    },
  };
}

function pushTop(list, item, limit) {
  list.push(item);
  list.sort((a, b) => b.score - a.score || b.guard.bonus - a.guard.bonus);
  if (list.length > limit) list.length = limit;
}

function guardGrid(variant, includeChrono) {
  const guards = [];
  const base = {
    variant,
    confidenceWeight: 0.25,
    marginWeight: 0,
    requireCandidateSelected: false,
    requireCenterTarget: false,
    requireLocalContact: true,
    blockWeakRay: false,
    minRayScore: 0,
    requireChronoAvailable: false,
    requireTacticalGpuVerified: false,
    requireCaptureLike: false,
    forbidCaptureLike: false,
    requireMoverInCheckBefore: false,
    forbidMoverInCheckBefore: false,
    requireDestinationContestedAfter: false,
    requireContactBalanceNonNegative: false,
    requireContactBalancePositive: false,
    maxOpponentLegalRepliesAfter: null,
    requireUnstablePressure: false,
    requireRisingContortion: false,
    minStabilityScore: 0,
  };
  const bonuses = includeChrono ? [0.48, 0.55] : [0.28, 0.38, 0.48, 0.55];
  const rankPenalties = includeChrono ? [0.02, 0.05] : [0.02, 0.05, 0.08];
  const maxRanks = includeChrono ? [2] : [2, 3];
  const minConfidences = includeChrono ? [0, 0.72, 0.74] : [0, 0.72, 0.74, 0.76];
  const minMargins = includeChrono ? [0, 0.004, 0.04] : [0, 0.004, 0.04];
  const maxRankJumps = includeChrono ? [1, 2] : [1, 2, 99];
  const maxConfidenceDrops = includeChrono ? [99, 0.08, 0.05] : [99, 0.08, 0.05, 0.02];
  const booleanPairs = [
    { protectBaselineSelected: false, requireKnightLike: false, requireEdgeTarget: false, requireIntersection: false },
    { protectBaselineSelected: true, requireKnightLike: false, requireEdgeTarget: false, requireIntersection: false },
    { protectBaselineSelected: false, requireKnightLike: true, requireEdgeTarget: false, requireIntersection: false },
    { protectBaselineSelected: false, requireKnightLike: true, requireEdgeTarget: true, requireIntersection: false },
    { protectBaselineSelected: false, requireKnightLike: true, requireEdgeTarget: false, requireIntersection: true },
    { protectBaselineSelected: true, requireKnightLike: true, requireEdgeTarget: false, requireIntersection: false },
  ];
  const chronoModes = includeChrono ? [
    {},
    { requireTacticalGpuVerified: true },
    { requireCaptureLike: true },
    { forbidMoverInCheckBefore: true },
    { requireMoverInCheckBefore: true },
    { requireDestinationContestedAfter: true },
    { requireCaptureLike: true, requireContactBalanceNonNegative: true },
    { requireCaptureLike: true, maxOpponentLegalRepliesAfter: 33 },
    { forbidMoverInCheckBefore: true, maxOpponentLegalRepliesAfter: 33 },
    { requireUnstablePressure: true, requireRisingContortion: true },
  ].map((mode) => ({ requireChronoAvailable: true, ...mode })) : [{}];

  for (const bonus of bonuses) {
    for (const rankPenalty of rankPenalties) {
      for (const maxRank of maxRanks) {
        for (const minCandidateConfidence of minConfidences) {
          for (const minCandidateMargin of minMargins) {
            for (const maxRankJump of maxRankJumps) {
              for (const maxConfidenceDrop of maxConfidenceDrops) {
                for (const booleans of booleanPairs) {
                  for (const chronoMode of chronoModes) {
                    guards.push({
                      ...base,
                      ...booleans,
                      ...chronoMode,
                      bonus,
                      rankPenalty,
                      maxRank,
                      minCandidateConfidence,
                      minCandidateMargin,
                      maxRankJump,
                      maxConfidenceDrop,
                    });
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  return guards;
}

function defaultOutPath(resolutionPath) {
  const parsed = path.parse(resolutionPath);
  return path.join(parsed.dir, `${parsed.name}.narrowed_runtime_guard_sweep.json`);
}

function main() {
  const args = parseArgs(process.argv.slice(2));
  const resolutionPath = path.resolve(args.resolution);
  const preflightPath = args.preflight ? path.resolve(args.preflight) : null;
  const chronoPath = args.chrono ? path.resolve(args.chrono) : null;
  const resolution = readJson(resolutionPath);
  const preflight = preflightPath ? readJson(preflightPath) : null;
  const chrono = chronoPath ? readJson(chronoPath) : null;
  const rows = normalizeRows(resolution, chronoByHash(chrono));
  const fightRows = rows.filter((row) => Number.isFinite(row.fightScore));
  const guards = guardGrid(args.variant, Boolean(chronoPath));
  let eligibleGuardCount = 0;
  const eligibleGuards = [];
  const bestBlockedGuards = [];
  const noAcceptedRegression = [];
  const fightLiftOnly = [];

  for (const guard of guards) {
    const allRowsAcceptedSurface = evaluateSurface('all_rows', rows, guard);
    const fightLabeledSurface = evaluateSurface('fight_labeled_rows', fightRows, guard);
    const eligibility = isFreshGateEligible(allRowsAcceptedSurface, fightLabeledSurface);
    const result = {
      guardId: guardId(guard),
      guard,
      eligibility,
      allRowsAcceptedSurface,
      fightLabeledSurface,
    };
    result.score = scoreCandidate(result);
    const compact = compactResult(result);
    if (result.eligibility.ready) {
      eligibleGuardCount += 1;
      pushTop(eligibleGuards, compact, 24);
    } else {
      pushTop(bestBlockedGuards, compact, 32);
    }
    if (result.eligibility.acceptedSafe && result.eligibility.selectedSafe) {
      pushTop(noAcceptedRegression, compact, 32);
    }
    if (result.eligibility.fightLift && result.eligibility.changed) {
      pushTop(fightLiftOnly, compact, 32);
    }
  }

  const output = {
    schemaVersion: SCHEMA_VERSION,
    generatedAt: new Date().toISOString(),
    condition: {
      source: path.resolve(args.conditionSource),
      runLabel: 'runtime_visible_guard_sweep_omnifold_r4_negative_yspace',
      changedFields: 'post-hoc guard sweep over recorded CUDA-derived selected-family rows only; no runtime behavior changed',
      labCondition: 'guard_sweep/uses_existing_selected_family_resolution_and_optional_preflight',
      metric: 'fresh-gate eligibility requires accepted-useful non-regression, selected-move non-regression, and fight lift',
    },
    sources: {
      resolutionPath,
      resolutionSha256: sha256File(resolutionPath),
      resolutionSchemaVersion: resolution.schemaVersion || null,
      preflightPath,
      preflightSha256: preflightPath ? sha256File(preflightPath) : null,
      preflightStatus: preflight?.preflight?.status || null,
      chronoPath,
      chronoSha256: chronoPath ? sha256File(chronoPath) : null,
      chronoSchemaVersion: chrono?.schemaVersion || null,
      chronoRowCount: Array.isArray(chrono?.rows) ? chrono.rows.length : null,
      fightPath: resolution.sources?.fightPath || null,
      fightSha256: resolution.sources?.fightSha256 || null,
    },
    searchSpace: {
      guardCount: guards.length,
      targetVariantId: args.variant,
      runtimeVisibleFields: [
        'candidate selectedVariantId',
        'candidate rank',
        'candidate confidence',
        'candidate margin',
        'candidate move geometry',
        'candidate OmniFold groupScores',
        'baseline frontier-rank top1 confidence',
        'baseline selectedMoveInFrontier gate field',
        ...(chronoPath ? [
          'chrono pressure/relation/contortion buckets',
          'GPU tactical contact flags',
          'GPU tactical contact counts',
          'chrono stability score',
        ] : []),
      ],
      excludedAsGuardFields: [
        'acceptedUsefulInjection',
        'fightScore',
      ],
      corpus: {
        rowCount: rows.length,
        rootCount: groupByRoot(rows).length,
        fightRowCount: fightRows.length,
        fightRootCount: groupByRoot(fightRows).length,
      },
    },
    promotionPolicy: {
      status: eligibleGuardCount ? 'candidate_ready_for_fresh_gate' : 'not_promoted',
      reason: eligibleGuardCount
        ? 'at least one runtime-visible guarded condition passed recorded-label preflight'
        : 'no runtime-visible guarded condition produced accepted-useful non-regression, selected-move non-regression, and fight lift together',
      requiredNextEvidence: eligibleGuardCount ? [
        'freeze the top guard into a condition artifact',
        'run a fresh fixed-condition accepted-injection GPU gate',
        'prove GPU legality/search unchanged in the runtime consumer',
      ] : [
        'do not launch the R4 negative-yspace consumer on this condition',
        'mine another independent GPU-derived runtime-visible signal before spending a fresh gate',
      ],
    },
    eligibleGuardCount,
    eligibleGuards,
    bestBlockedGuards,
    acceptedAndSelectedSafeButNoFreshGate: noAcceptedRegression,
    fightLiftButUnsafe: fightLiftOnly,
  };

  const outPath = path.resolve(args.out || defaultOutPath(resolutionPath));
  writeJson(outPath, output);
  console.log(JSON.stringify({
    ok: true,
    output: outPath,
    guardCount: guards.length,
    eligibleGuardCount: output.eligibleGuardCount,
    promotionStatus: output.promotionPolicy.status,
    bestBlocked: output.bestBlockedGuards[0] ? {
      guardId: output.bestBlockedGuards[0].guardId,
      allRowsDelta: output.bestBlockedGuards[0].allRowsDelta,
      fightRowsDelta: output.bestBlockedGuards[0].fightRowsDelta,
      eligibility: output.bestBlockedGuards[0].eligibility,
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
