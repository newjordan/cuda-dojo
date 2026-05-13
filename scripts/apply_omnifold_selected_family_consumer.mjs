#!/usr/bin/env node
import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';

const SCHEMA_VERSION = 'dojo.omnifold_selected_family_consumer_eval.v1';
const DEFAULT_CONDITION = '/home/frosty40/cuda_dojo/LOGIC_RAY_FRONTIER_RUNTIME_RECEIPT_2026-05-06.md';

function usage() {
  return `Usage: node scripts/apply_omnifold_selected_family_consumer.mjs --gate <selected_family_gate_sweep.json> [--resolution <selected_family_resolution.json>] [--out <consumer_eval.json>]

Apply a frozen selected-family gate condition as a measured consumer over
recorded selected-family rows. This is a JSON reranker only; it does not run
chess logic and does not change GPU legality/search semantics.
`;
}

function parseArgs(argv) {
  const args = {
    gate: null,
    resolution: null,
    out: null,
    conditionSource: DEFAULT_CONDITION,
  };
  for (let i = 0; i < argv.length; i += 1) {
    const token = argv[i];
    if (token === '--help' || token === '-h') {
      console.log(usage());
      process.exit(0);
    } else if (token === '--gate') {
      args.gate = argv[++i];
    } else if (token === '--resolution') {
      args.resolution = argv[++i];
    } else if (token === '--out') {
      args.out = argv[++i];
    } else if (token === '--condition-source') {
      args.conditionSource = argv[++i];
    } else {
      throw new Error(`unknown argument: ${token}\n${usage()}`);
    }
  }
  if (!args.gate) throw new Error(`missing --gate\n${usage()}`);
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

function normalizeRows(resolution) {
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
        rank: Math.max(1, Math.trunc(asNumber(row.rank, 1))),
        acceptedUsefulInjection: Boolean(row.acceptedUsefulInjection),
        selectedMoveInFrontier: Boolean(row.selectedMoveInFrontier),
        fightScore: hasFightScore ? asNumber(row.fightScore) : null,
        selectedFamily: row.selectedFamily || null,
        selectedFamilyId: row.selectedFamilyId || null,
        selectedVariantId: row.selectedVariantId || null,
        orderMax: Math.max(...(Array.isArray(selected.orderSet) ? selected.orderSet : [0]).map((value) => asNumber(value, 0))),
        confidence: asNumber(row.resolution?.confidence, 0),
        margin: asNumber(row.resolution?.margin, 0),
        selectedVariantGroups: groups,
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

function rowMatches(row, spec) {
  if (row.confidence < asNumber(spec.minConfidence, 0) || row.margin < asNumber(spec.minMargin, 0)) return false;
  if (spec.targetType === 'any') return true;
  if (spec.targetType === 'family_id') return row.selectedFamilyId === spec.targetValue;
  if (spec.targetType === 'variant_id') return row.selectedVariantId === spec.targetValue;
  if (spec.targetType === 'variant_group') return row.selectedVariantGroups.includes(spec.targetValue);
  if (spec.targetType === 'min_order') return row.orderMax >= Number(spec.targetValue);
  return false;
}

function scoreRow(row, spec) {
  return (rowMatches(row, spec) ? asNumber(spec.bonus, 0) : 0)
    + asNumber(spec.confidenceWeight, 0) * row.confidence
    + asNumber(spec.marginWeight, 0) * row.margin
    - asNumber(spec.rankPenalty, 1) * (row.rank - 1);
}

function publicChoice(row, score, baselineRow = null) {
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
    changedFromFrontierRank: baselineRow ? row.hash !== baselineRow.hash : false,
  };
}

function evaluate(groups, scorer, name) {
  let scoreSum = 0;
  let fightTop1Rows = 0;
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
    if (Number.isFinite(top.row.fightScore)) {
      fightTop1Rows += 1;
      scoreSum += top.row.fightScore;
      if (top.row.fightScore > 0) positiveTop1 += 1;
      else if (top.row.fightScore < 0) negativeTop1 += 1;
      else neutralTop1 += 1;
    }
    if (top.row.acceptedUsefulInjection) acceptedTop1 += 1;
    if (top.row.selectedMoveInFrontier) selectedTop1 += 1;
    roots.push({
      rootId,
      rowCount: rows.length,
      top1: publicChoice(top.row, top.score),
    });
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
    roots,
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

function changedRoots(baseline, consumer) {
  const byRoot = new Map(baseline.roots.map((root) => [root.rootId, root.top1]));
  return consumer.roots.map((root) => {
    const baselineTop = byRoot.get(root.rootId);
    return {
      rootId: root.rootId,
      changed: baselineTop?.hash !== root.top1.hash,
      baselineTop1: baselineTop || null,
      consumerTop1: {
        ...root.top1,
        changedFromFrontierRank: baselineTop?.hash !== root.top1.hash,
      },
      deltaFightScore: baselineTop?.fightScore != null && root.top1.fightScore != null
        ? round(root.top1.fightScore - baselineTop.fightScore)
        : null,
      acceptedDelta: Number(root.top1.acceptedUsefulInjection) - Number(Boolean(baselineTop?.acceptedUsefulInjection)),
      selectedDelta: Number(root.top1.selectedMoveInFrontier) - Number(Boolean(baselineTop?.selectedMoveInFrontier)),
    };
  }).filter((root) => root.changed);
}

function evaluateSurface(name, rows, spec) {
  const groups = groupByRoot(rows);
  const baseline = evaluate(groups, (row) => -(row.rank - 1), `${name}_frontier_rank`);
  const consumer = evaluate(groups, (row) => scoreRow(row, spec), `${name}_selected_family_consumer`);
  const resultDelta = delta(consumer, baseline);
  const changes = changedRoots(baseline, consumer);
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
    consumer: metricsPublic(consumer),
    deltaVsFrontierRank: resultDelta,
    changedRootCount: changes.length,
    changedRoots: changes,
  };
}

function defaultOutPath(gatePath, conditionId) {
  const parsed = path.parse(gatePath);
  const safeId = String(conditionId || 'selected_family_consumer').replace(/[^A-Za-z0-9_.:-]+/g, '_');
  return path.join(parsed.dir, `${parsed.name}.${safeId}.consumer_eval.json`);
}

function main() {
  const args = parseArgs(process.argv.slice(2));
  const gatePath = path.resolve(args.gate);
  const gate = readJson(gatePath);
  const frozen = gate.frozenCandidate;
  if (!frozen?.frozenPayload?.candidateSpec) {
    throw new Error('gate artifact has no frozenCandidate.frozenPayload.candidateSpec');
  }
  const frozenPayload = frozen.frozenPayload;
  const recomputedHash = sha256Object(frozenPayload);
  if (frozen.conditionHash && recomputedHash !== frozen.conditionHash) {
    throw new Error(`frozen condition hash mismatch: expected ${frozen.conditionHash}, recomputed ${recomputedHash}`);
  }
  const resolutionPath = path.resolve(args.resolution || frozenPayload.sourceResolutionPath);
  const resolution = readJson(resolutionPath);
  const rows = normalizeRows(resolution);
  const spec = frozenPayload.candidateSpec;
  const allRowsSurface = evaluateSurface('all_rows', rows, spec);
  const fightRowsSurface = evaluateSurface('fight_labeled_rows', rows.filter((row) => Number.isFinite(row.fightScore)), spec);
  const acceptedLift = allRowsSurface.deltaVsFrontierRank.acceptedUsefulTop1 > 0;
  const fightLift = (
    fightRowsSurface.deltaVsFrontierRank.meanFightScore != null
      && fightRowsSurface.deltaVsFrontierRank.meanFightScore > 0
  ) || fightRowsSurface.deltaVsFrontierRank.positiveTop1 > 0;
  const output = {
    schemaVersion: SCHEMA_VERSION,
    generatedAt: new Date().toISOString(),
    condition: {
      source: path.resolve(args.conditionSource),
      runLabel: 'measured_consumer_omnifold_selected_family_condition',
      changedFields: 'post-hoc consumer rerank over recorded CUDA-derived selected-family rows only; GPU legality/search unchanged',
      labCondition: 'consumer_eval/uses_frozen_selected_family_condition',
      metric: 'accepted-useful top1 plus forced-candidate GPU fight score',
    },
    sources: {
      gatePath,
      gateSha256: sha256File(gatePath),
      gateSchemaVersion: gate.schemaVersion || null,
      resolutionPath,
      resolutionSha256: sha256File(resolutionPath),
      expectedResolutionSha256: frozenPayload.sourceResolutionSha256 || null,
      fightPath: gate.sources?.fightPath || resolution.sources?.fightPath || null,
      fightSha256: gate.sources?.fightSha256 || resolution.sources?.fightSha256 || null,
    },
    frozenCondition: {
      conditionId: frozen.conditionId,
      conditionHash: frozen.conditionHash,
      recomputedConditionHash: recomputedHash,
      status: frozen.status,
      surface: frozenPayload.surface,
      candidateSpec: spec,
      fixedArtifactDeltaVsFrontierRank: frozenPayload.fixedArtifactDeltaVsFrontierRank,
      crossValidatedDeltaVsFrontierRank: frozenPayload.crossValidatedDeltaVsFrontierRank,
    },
    consumerPolicy: {
      algorithm: 'apply_frozen_selected_family_bonus_condition',
      hostRole: 'json_consumer_reranker_only',
      noChessRuntime: true,
      noGpuLegalityOrSearchChange: true,
      noRuntimePromotion: true,
    },
    surfaces: {
      allRowsAcceptedSurface: allRowsSurface,
      fightLabeledSurface: fightRowsSurface,
    },
    corpus: allRowsSurface.corpus,
    baseline: allRowsSurface.baseline,
    consumer: allRowsSurface.consumer,
    deltaVsFrontierRank: allRowsSurface.deltaVsFrontierRank,
    changedRootCount: allRowsSurface.changedRootCount,
    changedRoots: allRowsSurface.changedRoots,
    promotionPolicy: {
      status: 'not_promoted',
      reason: acceptedLift
        ? 'consumer improves accepted-useful top1 on this artifact, but still needs a fresh heldout accepted-injection gate before runtime promotion'
        : 'consumer does not improve accepted-useful top1 and must not be wired into runtime selection',
      blockers: acceptedLift ? [
        'not_evaluated_on_fresh_heldout_accepted_injection_gate',
        'not_integrated_in_runtime_selector',
      ] : [
        'accepted_useful_top1_not_improved',
        'not_evaluated_on_fresh_heldout_accepted_injection_gate',
        'not_integrated_in_runtime_selector',
      ],
    },
  };
  const outPath = path.resolve(args.out || defaultOutPath(gatePath, frozen.conditionId));
  writeJson(outPath, output);
  console.log(JSON.stringify({
    ok: true,
    output: outPath,
    conditionHash: frozen.conditionHash,
    allRows: {
      rootCount: allRowsSurface.corpus.rootCount,
      changedRootCount: allRowsSurface.changedRootCount,
      deltaVsFrontierRank: allRowsSurface.deltaVsFrontierRank,
    },
    fightLabeledRows: {
      rootCount: fightRowsSurface.corpus.rootCount,
      changedRootCount: fightRowsSurface.changedRootCount,
      deltaVsFrontierRank: fightRowsSurface.deltaVsFrontierRank,
    },
    observedFightLift: fightLift,
    promote: false,
  }, null, 2));
}

try {
  main();
} catch (error) {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
}
