#!/usr/bin/env node
import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';

const SCHEMA_VERSION = 'dojo.omnifold_selected_family_resolution.v1';
const DEFAULT_CONDITION = '/home/frosty40/cuda_dojo/LOGIC_RAY_FRONTIER_RUNTIME_RECEIPT_2026-05-06.md';

function usage() {
  return `Usage: node scripts/resolve_omnifold_selected_family.mjs --bridge <bridge.json> --chrono <chrono_o2_tactical_sidecar.json> --omnifold <manifest.json> [--fight <gpu_fight_rollout_patterns.json>] [--out <resolution.json>]

Resolve candidate OmniFold selected-family membership for recorded
logicRayFrontier rows using manifest active variant groups plus CUDA-derived
PZRG/chrono/tactical fields. This does not run chess logic or promote runtime
selection.
`;
}

function parseArgs(argv) {
  const args = {
    bridge: null,
    chrono: null,
    omnifold: null,
    fight: null,
    out: null,
    conditionSource: DEFAULT_CONDITION,
  };
  for (let i = 0; i < argv.length; i += 1) {
    const token = argv[i];
    if (token === '--help' || token === '-h') {
      console.log(usage());
      process.exit(0);
    } else if (token === '--bridge') {
      args.bridge = argv[++i];
    } else if (token === '--chrono') {
      args.chrono = argv[++i];
    } else if (token === '--omnifold') {
      args.omnifold = argv[++i];
    } else if (token === '--fight') {
      args.fight = argv[++i];
    } else if (token === '--out') {
      args.out = argv[++i];
    } else if (token === '--condition-source') {
      args.conditionSource = argv[++i];
    } else {
      throw new Error(`unknown argument: ${token}\n${usage()}`);
    }
  }
  if (!args.bridge) throw new Error(`missing --bridge\n${usage()}`);
  if (!args.chrono) throw new Error(`missing --chrono\n${usage()}`);
  if (!args.omnifold) throw new Error(`missing --omnifold\n${usage()}`);
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

function clamp(value, lo = 0, hi = 1) {
  return Math.max(lo, Math.min(hi, asNumber(value, lo)));
}

function frontierOf(row) {
  return row.logicRayFrontier || row.pzrgCandidate?.logicRayFrontier || {};
}

function chronoByHash(chrono) {
  return Object.fromEntries((Array.isArray(chrono.rows) ? chrono.rows : [])
    .filter((row) => row.logicRayFrontierHash)
    .map((row) => [row.logicRayFrontierHash, row]));
}

function fightByHash(fight) {
  if (!fight) return {};
  return Object.fromEntries((Array.isArray(fight.rollouts) ? fight.rollouts : [])
    .filter((row) => row.hash)
    .map((row) => [row.hash, row]));
}

function acceptedUseful(row, chronoRow) {
  const frontier = frontierOf(row);
  const injection = row.pzrgCandidate?.injection_relevance || {};
  return Boolean(
    frontier.gate?.acceptedUsefulInjection
      || injection.accepted_useful_injection
      || injection.promotion_gate_approved
      || chronoRow?.runtimeChoiceSignal?.acceptedUsefulInjection,
  );
}

function selectedMove(row, chronoRow) {
  const frontier = frontierOf(row);
  return Boolean(
    frontier.gate?.selectedMoveInFrontier
      || chronoRow?.runtimeChoiceSignal?.selectedMoveInFrontier,
  );
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

function featureScores(row, chronoRow) {
  const frontier = frontierOf(row);
  const move = String(frontier.move || row.move || chronoRow?.move || '');
  const fromFile = move.length >= 4 ? move.charCodeAt(0) - 97 : -1;
  const fromRank = move.length >= 4 ? Number(move[1]) - 1 : -1;
  const toFile = move.length >= 4 ? move.charCodeAt(2) - 97 : -1;
  const toRank = move.length >= 4 ? Number(move[3]) - 1 : -1;
  const df = Math.abs(toFile - fromFile);
  const dr = Math.abs(toRank - fromRank);
  const moveDistance = Math.sqrt(df * df + dr * dr);
  const sameFile = df === 0 && dr > 0;
  const sameRank = dr === 0 && df > 0;
  const diagonal = df === dr && df > 0;
  const center16 = toFile >= 2 && toFile <= 5 && toRank >= 2 && toRank <= 5;
  const pzrg = frontier.pzrg4d || {};
  const action = chronoRow?.actionFeatures || {};
  const actionFlags = action.flags || {};
  const actionScalars = action.scalars || {};
  const tactical = chronoRow?.tacticalContact || {};
  const tacticalFlags = tactical.flags || {};
  const counts = tactical.counts || {};
  const sourceOrderProxy = chronoRow?.provenance?.sourceOrderProxy === true;
  const pressure = String(pzrg.pressure || '');
  const expression = String(pzrg.chessExpression || '');
  const relationBucket = chronoRow?.relationDrift?.bucket || '';
  const contortionBucket = chronoRow?.pathContortion?.bucket || '';
  const uncertaintyBucket = chronoRow?.uncertainty?.bucket || '';
  const stability = asNumber(chronoRow?.diagnostics?.stabilityScore, 0);
  const utility = clamp(frontier.utility, 0, 2) / 2;
  const pathProbability = clamp(frontier.pathProbability, 0, 1);
  const lockIn = clamp(frontier.lockIn, 0, 1);
  const risk = clamp(frontier.risk, 0, 1);
  const contactBalance = asNumber(counts.contactBalanceAfter, 0);
  const replies = asNumber(counts.opponentLegalRepliesAfter, 0);
  const pieceKnown = Boolean(
    (action.piece && action.piece !== 'empty')
      || asNumber(tactical.pieces?.movingPiece, 0) !== 0
  );
  const familyKnown = Boolean(
    (action.family && action.family !== 'unknown')
      || sameFile
      || sameRank
      || diagonal
      || moveDistance > 0
  );
  const relationKnown = Boolean(pressure || expression || relationBucket);
  const contact = Boolean(
    tactical.gpuVerified
      || tacticalFlags.captureLike
      || tacticalFlags.destinationDefendedAfter
      || tacticalFlags.destinationAttackedAfter
      || tacticalFlags.destinationDefendedBefore
      || tacticalFlags.destinationAttackedBefore,
  );
  const geometry = Boolean(
    actionFlags.longMove
      || actionFlags.diagonal
      || actionFlags.center16
      || actionFlags.sameFile
      || actionFlags.sameRank
      || diagonal
      || center16
      || sameFile
      || sameRank
      || Math.max(asNumber(actionScalars.distance, 0), moveDistance) >= 2.5,
  );
  const intersection = Boolean(
    tacticalFlags.captureLike
      && (tacticalFlags.destinationDefendedAfter || tacticalFlags.destinationAttackedAfter)
  );
  const anchorSpan = Boolean(
    Math.max(asNumber(actionScalars.distance, 0), moveDistance) >= 2.5
      || frontier.rank <= 2
      || expression === 'candidate_cohesion'
      || pathProbability >= 0.2
      || lockIn >= 0.5
  );
  const matrix4x4 = Boolean(
    (chronoRow?.responseTensors && chronoRow.responseTensors.bootstrapNotLearned !== true)
      || relationBucket === 'unstable'
      || contortionBucket === 'unstable'
  );
  const negativeY = Boolean(
    contortionBucket === 'unstable'
      || relationBucket === 'unstable'
      || (!sourceOrderProxy && uncertaintyBucket === 'unstable')
      || pressure === 'root_disagreement_pressure'
      || expression === 'candidate_tension'
      || tacticalFlags.moverInCheckBefore
      || tacticalFlags.destinationAttackedAfter
      || contactBalance < 0
      || replies > 36
      || risk >= 0.45
      || stability < 0.45
  );
  return {
    piece_only: pieceKnown ? 0.75 : 0.25,
    move_family_only: familyKnown ? 0.75 : 0.25,
    zone_only: actionFlags.center16 || center16 ? 0.85 : 0.35,
    ray_only: clamp((utility + pathProbability + lockIn) / 3, 0.15, 0.9),
    relation_only: relationKnown ? 0.8 : 0.25,
    fold2x2_local_contact: contact ? 0.9 : 0.15,
    fold2x2_geometric: geometry ? 0.85 : 0.2,
    fold2x2_intersection: intersection ? 0.95 : 0.15,
    fold2x4_anchor_span: anchorSpan ? 0.9 : 0.2,
    fold4x4_matrix: matrix4x4 ? 0.85 : 0.1,
    negative_yspace_local: negativeY ? 1.0 : 0.05,
  };
}

function variantScore(variant, scores) {
  const groups = Array.isArray(variant.groups) ? variant.groups : [];
  if (!groups.length) return 0;
  const raw = groups.reduce((sum, group) => sum + asNumber(scores[group], 0), 0) / groups.length;
  const readiness = variant.trainerReady ? 0.05 : -0.1;
  const dimBonus = Math.min(0.08, asNumber(variant.omnifoldDim, 0) / 4000);
  return clamp(raw + readiness + dimBonus, 0, 1);
}

function selectFamily(row, chronoRow, families) {
  const scores = featureScores(row, chronoRow);
  const candidates = [];
  for (const family of families) {
    let bestVariant = null;
    let bestScore = -Infinity;
    for (const variant of family.activeVariants) {
      const score = variantScore(variant, scores);
      if (score > bestScore) {
        bestScore = score;
        bestVariant = variant;
      }
    }
    const selectedGroups = Array.isArray(bestVariant?.groups) ? bestVariant.groups : [];
    const maxOrder = Math.max(...family.orderSet, 0);
    const duplicateMatrixVariantPenalty = (
      maxOrder >= 6
        && selectedGroups.includes('fold2x4_anchor_span')
        && selectedGroups.includes('fold4x4_matrix')
        && !selectedGroups.includes('negative_yspace_local')
        && scores.negative_yspace_local < 0.75
    ) ? 0.12 : 0;
    const orderBonus = maxOrder * 0.003;
    candidates.push({
      familyId: family.id,
      foldFamily: family.foldFamily,
      orderSet: family.orderSet,
      activeVariantCount: family.activeVariantCount,
      selectedVariantId: bestVariant?.id || null,
      selectedVariantGroups: selectedGroups,
      score: round(clamp(bestScore + orderBonus - duplicateMatrixVariantPenalty, 0, 1)),
    });
  }
  candidates.sort((a, b) => (
    b.score - a.score
      || Math.max(...b.orderSet, 0) - Math.max(...a.orderSet, 0)
      || a.foldFamily.localeCompare(b.foldFamily)
  ));
  const selected = candidates[0] || null;
  const runnerUp = candidates[1] || null;
  const margin = selected && runnerUp ? round(selected.score - runnerUp.score) : null;
  const confidence = selected ? round(clamp((selected.score + Math.max(0, margin || 0)) / 1.1, 0, 1)) : 0;
  return {
    selected,
    candidates,
    margin,
    confidence,
    groupScores: scores,
  };
}

function familySummary(assignments, fightAvailable) {
  const byFamily = new Map();
  for (const row of assignments) {
    const family = row.selectedFamily || 'unresolved';
    if (!byFamily.has(family)) {
      byFamily.set(family, {
        selectedFamily: family,
        rows: 0,
        acceptedRows: 0,
        selectedRows: 0,
        fightRows: 0,
        fightScoreSum: 0,
        positiveFightRows: 0,
        confidenceSum: 0,
      });
    }
    const item = byFamily.get(family);
    item.rows += 1;
    if (row.acceptedUsefulInjection) item.acceptedRows += 1;
    if (row.selectedMoveInFrontier) item.selectedRows += 1;
    item.confidenceSum += row.resolution.confidence;
    if (Number.isFinite(row.fightScore)) {
      item.fightRows += 1;
      item.fightScoreSum += row.fightScore;
      if (row.fightScore > 0) item.positiveFightRows += 1;
    }
  }
  return [...byFamily.values()].map((item) => ({
    selectedFamily: item.selectedFamily,
    rows: item.rows,
    rowShare: round(item.rows / assignments.length),
    acceptedRows: item.acceptedRows,
    acceptedRate: round(item.acceptedRows / item.rows),
    selectedMoveRows: item.selectedRows,
    selectedMoveRate: round(item.selectedRows / item.rows),
    meanConfidence: round(item.confidenceSum / item.rows),
    fightRows: fightAvailable ? item.fightRows : null,
    meanFightScore: fightAvailable && item.fightRows ? round(item.fightScoreSum / item.fightRows) : null,
    positiveFightRows: fightAvailable ? item.positiveFightRows : null,
  })).sort((a, b) => b.rows - a.rows || a.selectedFamily.localeCompare(b.selectedFamily));
}

function rootTopEvaluation(assignments, scoreSelector) {
  const groups = new Map();
  for (const row of assignments) {
    if (!groups.has(row.rootId)) groups.set(row.rootId, []);
    groups.get(row.rootId).push(row);
  }
  let acceptedTop1 = 0;
  let selectedTop1 = 0;
  let fightRows = 0;
  let fightScoreSum = 0;
  let positiveFightTop1 = 0;
  const roots = [];
  for (const [rootId, rows] of [...groups.entries()].sort(([a], [b]) => a.localeCompare(b))) {
    const ranked = rows
      .map((row) => ({ row, score: scoreSelector(row) }))
      .sort((a, b) => b.score - a.score || a.row.rank - b.row.rank || a.row.move.localeCompare(b.row.move));
    const top = ranked[0];
    if (!top) continue;
    if (top.row.acceptedUsefulInjection) acceptedTop1 += 1;
    if (top.row.selectedMoveInFrontier) selectedTop1 += 1;
    if (Number.isFinite(top.row.fightScore)) {
      fightRows += 1;
      fightScoreSum += top.row.fightScore;
      if (top.row.fightScore > 0) positiveFightTop1 += 1;
    }
    roots.push({
      rootId,
      rowCount: rows.length,
      top1: {
        bridgeId: top.row.bridgeId,
        hash: top.row.logicRayFrontierHash,
        move: top.row.move,
        rank: top.row.rank,
        score: round(top.score),
        selectedFamily: top.row.selectedFamily,
        confidence: top.row.resolution.confidence,
        acceptedUsefulInjection: top.row.acceptedUsefulInjection,
        selectedMoveInFrontier: top.row.selectedMoveInFrontier,
        fightScore: Number.isFinite(top.row.fightScore) ? round(top.row.fightScore) : null,
      },
    });
  }
  return {
    rootCount: roots.length,
    acceptedUsefulTop1: acceptedTop1,
    selectedMoveTop1: selectedTop1,
    fightTop1Rows: fightRows,
    meanFightScore: fightRows ? round(fightScoreSum / fightRows) : null,
    positiveFightTop1,
    roots,
  };
}

function defaultOutPath(bridgePath) {
  const parsed = path.parse(bridgePath);
  const stem = parsed.base.replace(/\.pzrg_frostmatrix_bridge\.json$/, '').replace(/\.json$/, '');
  return path.join(parsed.dir, `${stem}.omnifold_selected_family_resolution.json`);
}

function main() {
  const args = parseArgs(process.argv.slice(2));
  const bridgePath = path.resolve(args.bridge);
  const chronoPath = path.resolve(args.chrono);
  const omnifoldPath = path.resolve(args.omnifold);
  const fightPath = args.fight ? path.resolve(args.fight) : null;
  const bridge = readJson(bridgePath);
  const chrono = readJson(chronoPath);
  const omnifold = readJson(omnifoldPath);
  const fight = fightPath ? readJson(fightPath) : null;
  const rows = Array.isArray(bridge.rows) ? bridge.rows : [];
  const chronoRows = chronoByHash(chrono);
  const fightRows = fightByHash(fight);
  const families = activeFamilies(omnifold);
  const assignments = rows.map((row, index) => {
    const frontier = frontierOf(row);
    const rowHash = row.logicRayFrontierHash;
    const chronoRow = chronoRows[rowHash] || null;
    const fightRow = fightRows[rowHash] || null;
    const selection = selectFamily(row, chronoRow, families);
    const selected = selection.selected;
    return {
      bridgeId: row.bridgeId,
      logicRayFrontierHash: rowHash,
      rowIndex: index,
      rootId: frontier.rootId || row.rootId,
      move: frontier.move || row.move,
      rank: Math.max(1, asNumber(frontier.rank || row.rank, 1)),
      acceptedUsefulInjection: acceptedUseful(row, chronoRow),
      selectedMoveInFrontier: selectedMove(row, chronoRow),
      fightScore: fightRow && Number.isFinite(Number(fightRow.fightScore)) ? asNumber(fightRow.fightScore) : null,
      selectedFamily: selected?.foldFamily || null,
      selectedFamilyId: selected?.familyId || null,
      selectedVariantId: selected?.selectedVariantId || null,
      resolvedOmnifoldFamily: {
        status: selected ? 'attached' : 'quarantine',
        families: families.map((family) => family.foldFamily),
        selectedFamily: selected?.foldFamily || null,
        offManifoldAudit: {
          offManifold: !selected,
          previousStatus: frontier.omnifoldFamily?.status || null,
          previousSelectedFamily: frontier.omnifoldFamily?.selectedFamily || null,
          resolutionMode: 'deterministic_manifest_group_selector',
          promotionEligible: false,
          reason: selected
            ? 'selected active OmniFold family candidate from manifest variant group fit over CUDA-derived frontier/chrono/tactical fields'
            : 'no active OmniFold family candidate selected',
        },
      },
      resolution: {
        confidence: selection.confidence,
        margin: selection.margin,
        selected,
        candidates: selection.candidates,
        groupScores: Object.fromEntries(Object.entries(selection.groupScores).map(([key, value]) => [key, round(value)])),
      },
    };
  });
  const fightAvailable = Boolean(fightPath);
  const beforeOffManifold = rows.filter((row) => {
    const omnifoldFamily = frontierOf(row).omnifoldFamily || {};
    return !omnifoldFamily.selectedFamily || omnifoldFamily.status === 'placeholder';
  }).length;
  const afterOffManifold = assignments.filter((row) => row.resolvedOmnifoldFamily.offManifoldAudit.offManifold).length;
  const baselineEval = rootTopEvaluation(assignments, (row) => -row.rank);
  const confidenceEval = rootTopEvaluation(assignments, (row) => row.resolution.confidence - row.rank * 0.001);
  const foldDepthEval = rootTopEvaluation(assignments, (row) => (
    Math.max(...(row.resolution.selected?.orderSet || [0])) * 0.01
      + row.resolution.confidence
      - row.rank * 0.001
  ));
  const output = {
    schemaVersion: SCHEMA_VERSION,
    generatedAt: new Date().toISOString(),
    condition: {
      source: path.resolve(args.conditionSource),
      runLabel: 'scout_new_experiment_omnifold_selected_family_resolution',
      changedFields: 'post-hoc selected-family/off-manifold resolution artifact only; source bridge and runtime behavior unchanged',
      labCondition: 'scout/new_experiment/uses_recorded_cuda_derived_frontier_chrono_tactical_rows',
      metric: 'selected-family coverage plus accepted-useful and optional GPU fight attribution',
    },
    sources: {
      bridgePath,
      bridgeSha256: sha256File(bridgePath),
      bridgeRows: rows.length,
      chronoPath,
      chronoSha256: sha256File(chronoPath),
      chronoRows: Array.isArray(chrono.rows) ? chrono.rows.length : 0,
      omnifoldPath,
      omnifoldSha256: sha256File(omnifoldPath),
      omnifoldSchemaVersion: omnifold.schemaVersion || null,
      fightPath,
      fightSha256: fightPath ? sha256File(fightPath) : null,
      fightSchemaVersion: fight?.schemaVersion || null,
      fightRunLabel: fight?.condition?.runLabel || null,
    },
    activeFamilies: families.map((family) => ({
      id: family.id,
      foldFamily: family.foldFamily,
      orderSet: family.orderSet,
      activeVariantCount: family.activeVariantCount,
      activeVariantIds: family.activeVariants.map((variant) => variant.id),
    })),
    selectorPolicy: {
      algorithm: 'score_imported_omnifold_variant_groups_from_pzrg_chrono_tactical_features',
      hostRole: 'json_feature_projection_only',
      noChessRuntime: true,
      noRuntimePromotion: true,
      selectedFamilySemantics: 'candidate selected-family membership for off-manifold audit resolution; runtime gate still required',
    },
    aggregate: {
      rowCount: assignments.length,
      activeFoldFamilies: families.length,
      offManifoldBefore: beforeOffManifold,
      offManifoldAfterCandidateResolution: afterOffManifold,
      resolvedRows: assignments.length - afterOffManifold,
      acceptedRows: assignments.filter((row) => row.acceptedUsefulInjection).length,
      selectedMoveRows: assignments.filter((row) => row.selectedMoveInFrontier).length,
      fightRows: assignments.filter((row) => Number.isFinite(row.fightScore)).length,
    },
    familySummaries: familySummary(assignments, fightAvailable),
    rootTopEvaluations: {
      frontierRank: {
        acceptedUsefulTop1: baselineEval.acceptedUsefulTop1,
        selectedMoveTop1: baselineEval.selectedMoveTop1,
        fightTop1Rows: baselineEval.fightTop1Rows,
        meanFightScore: baselineEval.meanFightScore,
        positiveFightTop1: baselineEval.positiveFightTop1,
      },
      confidenceSelector: {
        acceptedUsefulTop1: confidenceEval.acceptedUsefulTop1,
        selectedMoveTop1: confidenceEval.selectedMoveTop1,
        fightTop1Rows: confidenceEval.fightTop1Rows,
        meanFightScore: confidenceEval.meanFightScore,
        positiveFightTop1: confidenceEval.positiveFightTop1,
        deltaVsFrontierRank: {
          acceptedUsefulTop1: confidenceEval.acceptedUsefulTop1 - baselineEval.acceptedUsefulTop1,
          selectedMoveTop1: confidenceEval.selectedMoveTop1 - baselineEval.selectedMoveTop1,
          meanFightScore: confidenceEval.meanFightScore != null && baselineEval.meanFightScore != null
            ? round(confidenceEval.meanFightScore - baselineEval.meanFightScore)
            : null,
          positiveFightTop1: confidenceEval.positiveFightTop1 - baselineEval.positiveFightTop1,
        },
      },
      foldDepthSelector: {
        acceptedUsefulTop1: foldDepthEval.acceptedUsefulTop1,
        selectedMoveTop1: foldDepthEval.selectedMoveTop1,
        fightTop1Rows: foldDepthEval.fightTop1Rows,
        meanFightScore: foldDepthEval.meanFightScore,
        positiveFightTop1: foldDepthEval.positiveFightTop1,
        deltaVsFrontierRank: {
          acceptedUsefulTop1: foldDepthEval.acceptedUsefulTop1 - baselineEval.acceptedUsefulTop1,
          selectedMoveTop1: foldDepthEval.selectedMoveTop1 - baselineEval.selectedMoveTop1,
          meanFightScore: foldDepthEval.meanFightScore != null && baselineEval.meanFightScore != null
            ? round(foldDepthEval.meanFightScore - baselineEval.meanFightScore)
            : null,
          positiveFightTop1: foldDepthEval.positiveFightTop1 - baselineEval.positiveFightTop1,
        },
      },
    },
    rowAssignments: assignments,
    promotionPolicy: {
      status: 'not_promoted',
      reason: 'selected-family resolution is an off-manifold audit candidate, not a runtime fold gate',
      requiredNextEvidence: [
        'accepted-injection lift under a frozen selected-family gate',
        'runtime consumer that reads selectedFamily without changing chess legality/search semantics',
        'heldout GPU fight/gate result showing selected-family effect',
      ],
    },
  };
  const outPath = path.resolve(args.out || defaultOutPath(bridgePath));
  writeJson(outPath, output);
  console.log(JSON.stringify({
    ok: true,
    output: outPath,
    rowCount: output.aggregate.rowCount,
    offManifoldBefore: output.aggregate.offManifoldBefore,
    offManifoldAfterCandidateResolution: output.aggregate.offManifoldAfterCandidateResolution,
    familySummaries: output.familySummaries.map((family) => ({
      selectedFamily: family.selectedFamily,
      rows: family.rows,
      acceptedRate: family.acceptedRate,
      meanFightScore: family.meanFightScore,
    })),
    confidenceSelectorDelta: output.rootTopEvaluations.confidenceSelector.deltaVsFrontierRank,
    foldDepthSelectorDelta: output.rootTopEvaluations.foldDepthSelector.deltaVsFrontierRank,
    promote: false,
  }, null, 2));
}

try {
  main();
} catch (error) {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
}
