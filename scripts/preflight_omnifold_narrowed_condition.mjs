#!/usr/bin/env node
import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';

const SCHEMA_VERSION = 'dojo.omnifold_narrowed_condition_preflight.v1';
const DEFAULT_CONDITION = '/home/frosty40/cuda_dojo/LOGIC_RAY_FRONTIER_RUNTIME_RECEIPT_2026-05-06.md';
const DEFAULT_VARIANT = 'R4_piece_zone_move_family_ray_relation_fold2x2_negative_yspace';

function usage() {
  return `Usage: node scripts/preflight_omnifold_narrowed_condition.mjs --resolution <selected_family_resolution.json> [--contrast <contrast.json>] [--out <preflight.json>]

Fail-closed preflight for the narrowed R4 negative-yspace rank<=2 condition.
This is JSON evaluation over recorded CUDA-derived rows only; it does not run
chess logic and should block a fresh gate if accepted-useful top1 regresses.
`;
}

function parseArgs(argv) {
  const args = {
    resolution: null,
    contrast: null,
    out: null,
    variant: DEFAULT_VARIANT,
    maxRank: 2,
    bonus: 0.55,
    rankPenalty: 0.02,
    conditionSource: DEFAULT_CONDITION,
  };
  for (let i = 0; i < argv.length; i += 1) {
    const token = argv[i];
    if (token === '--help' || token === '-h') {
      console.log(usage());
      process.exit(0);
    } else if (token === '--resolution') {
      args.resolution = argv[++i];
    } else if (token === '--contrast') {
      args.contrast = argv[++i];
    } else if (token === '--out') {
      args.out = argv[++i];
    } else if (token === '--variant') {
      args.variant = argv[++i];
    } else if (token === '--max-rank') {
      args.maxRank = Number(argv[++i]);
    } else if (token === '--bonus') {
      args.bonus = Number(argv[++i]);
    } else if (token === '--rank-penalty') {
      args.rankPenalty = Number(argv[++i]);
    } else if (token === '--condition-source') {
      args.conditionSource = argv[++i];
    } else {
      throw new Error(`unknown argument: ${token}\n${usage()}`);
    }
  }
  if (!args.resolution) throw new Error(`missing --resolution\n${usage()}`);
  if (!Number.isInteger(args.maxRank) || args.maxRank < 1) throw new Error('--max-rank must be an integer >= 1');
  if (!Number.isFinite(args.bonus)) throw new Error('--bonus must be numeric');
  if (!Number.isFinite(args.rankPenalty)) throw new Error('--rank-penalty must be numeric');
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

function normalizeRows(resolution) {
  return (Array.isArray(resolution.rowAssignments) ? resolution.rowAssignments : [])
    .filter((row) => row.rootId && row.move)
    .map((row) => {
      const selected = row.resolution?.selected || {};
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
        confidence: asNumber(row.resolution?.confidence, 0),
        margin: asNumber(row.resolution?.margin, 0),
        selectedVariantGroups: Array.isArray(selected.selectedVariantGroups)
          ? selected.selectedVariantGroups.map(String).sort()
          : [],
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

function matches(row, condition) {
  return row.selectedVariantId === condition.variant && row.rank <= condition.maxRank;
}

function conditionScore(row, condition) {
  return (matches(row, condition) ? condition.bonus : 0)
    - condition.rankPenalty * (row.rank - 1);
}

function publicChoice(row, score) {
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
    conditionMatched: matches(row, conditionFromChoice),
  };
}

let conditionFromChoice = null;

function evaluate(groups, scorer, name, condition) {
  conditionFromChoice = condition;
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

function changedRoots(baseline, candidate) {
  const baselineByRoot = new Map(baseline.roots.map((root) => [root.rootId, root.top1]));
  return candidate.roots.map((root) => {
    const baselineTop = baselineByRoot.get(root.rootId);
    return {
      rootId: root.rootId,
      changed: baselineTop?.hash !== root.top1.hash,
      baselineTop1: baselineTop || null,
      candidateTop1: root.top1,
      deltaFightScore: baselineTop?.fightScore != null && root.top1.fightScore != null
        ? round(root.top1.fightScore - baselineTop.fightScore)
        : null,
      acceptedDelta: Number(root.top1.acceptedUsefulInjection) - Number(Boolean(baselineTop?.acceptedUsefulInjection)),
      selectedDelta: Number(root.top1.selectedMoveInFrontier) - Number(Boolean(baselineTop?.selectedMoveInFrontier)),
    };
  }).filter((root) => root.changed);
}

function evaluateSurface(name, rows, condition) {
  const groups = groupByRoot(rows);
  const baseline = evaluate(groups, (row) => -(row.rank - 1), `${name}_frontier_rank`, condition);
  const candidate = evaluate(groups, (row) => conditionScore(row, condition), `${name}_narrowed_condition`, condition);
  const resultDelta = delta(candidate, baseline);
  const changes = changedRoots(baseline, candidate);
  return {
    name,
    corpus: {
      rowCount: rows.length,
      rootCount: groups.length,
      candidateRows: rows.filter((row) => matches(row, condition)).length,
      acceptedRows: rows.filter((row) => row.acceptedUsefulInjection).length,
      selectedRows: rows.filter((row) => row.selectedMoveInFrontier).length,
      fightRows: rows.filter((row) => Number.isFinite(row.fightScore)).length,
    },
    baseline: metricsPublic(baseline),
    candidate: metricsPublic(candidate),
    deltaVsFrontierRank: resultDelta,
    changedRootCount: changes.length,
    changedRoots: changes,
  };
}

function defaultOutPath(resolutionPath) {
  const parsed = path.parse(resolutionPath);
  return path.join(parsed.dir, `${parsed.name}.narrowed_r4_rank2_preflight.json`);
}

function main() {
  const args = parseArgs(process.argv.slice(2));
  const resolutionPath = path.resolve(args.resolution);
  const contrastPath = args.contrast ? path.resolve(args.contrast) : null;
  const resolution = readJson(resolutionPath);
  const contrast = contrastPath ? readJson(contrastPath) : null;
  const rows = normalizeRows(resolution);
  const condition = {
    variant: args.variant,
    maxRank: args.maxRank,
    bonus: args.bonus,
    rankPenalty: args.rankPenalty,
  };
  const allRows = evaluateSurface('all_rows', rows, condition);
  const fightRows = evaluateSurface('fight_labeled_rows', rows.filter((row) => Number.isFinite(row.fightScore)), condition);
  const acceptedSafe = allRows.deltaVsFrontierRank.acceptedUsefulTop1 >= 0
    && fightRows.deltaVsFrontierRank.acceptedUsefulTop1 >= 0;
  const selectedSafe = allRows.deltaVsFrontierRank.selectedMoveTop1 >= 0
    && fightRows.deltaVsFrontierRank.selectedMoveTop1 >= 0;
  const fightLift = (
    fightRows.deltaVsFrontierRank.meanFightScore != null
      && fightRows.deltaVsFrontierRank.meanFightScore > 0
  ) || fightRows.deltaVsFrontierRank.positiveTop1 > 0;
  const output = {
    schemaVersion: SCHEMA_VERSION,
    generatedAt: new Date().toISOString(),
    condition: {
      source: path.resolve(args.conditionSource),
      runLabel: 'fail_closed_preflight_omnifold_r4_negative_yspace_rank2',
      changedFields: 'post-hoc narrowed condition preflight over recorded CUDA-derived rows only; no runtime behavior changed',
      labCondition: 'preflight/uses_existing_selected_family_resolution_and_optional_contrast',
      metric: 'accepted-useful non-regression before any fresh gate',
    },
    sources: {
      resolutionPath,
      resolutionSha256: sha256File(resolutionPath),
      resolutionSchemaVersion: resolution.schemaVersion || null,
      contrastPath,
      contrastSha256: contrastPath ? sha256File(contrastPath) : null,
      contrastRecommendation: contrast?.narrowedConditionRecommendation || null,
      fightPath: resolution.sources?.fightPath || null,
      fightSha256: resolution.sources?.fightSha256 || null,
    },
    narrowedCondition: condition,
    surfaces: {
      allRowsAcceptedSurface: allRows,
      fightLabeledSurface: fightRows,
    },
    preflight: {
      status: acceptedSafe && selectedSafe && fightLift ? 'candidate_ready_for_fresh_gate' : 'blocked_do_not_launch_fresh_gate',
      acceptedSafe,
      selectedSafe,
      fightLift,
      blockers: [
        ...(acceptedSafe ? [] : ['accepted_useful_top1_regresses']),
        ...(selectedSafe ? [] : ['selected_move_top1_regresses']),
        ...(fightLift ? [] : ['no_fight_lift']),
      ],
    },
    promotionPolicy: {
      status: 'not_promoted',
      reason: 'preflight only; no runtime selector or fresh heldout gate was run',
      requiredNextEvidence: acceptedSafe && selectedSafe && fightLift ? [
        'fresh accepted-injection gate using this exact narrowed condition',
        'GPU legality/search unchanged proof in runtime consumer',
      ] : [
        'derive a narrower runtime-visible guard before any fresh gate',
        'accepted-useful top1 non-regression on recorded labels',
      ],
    },
  };
  const outPath = path.resolve(args.out || defaultOutPath(resolutionPath));
  writeJson(outPath, output);
  console.log(JSON.stringify({
    ok: true,
    output: outPath,
    status: output.preflight.status,
    condition,
    allRowsDelta: allRows.deltaVsFrontierRank,
    fightRowsDelta: fightRows.deltaVsFrontierRank,
    acceptedSafe,
    selectedSafe,
    fightLift,
    promote: false,
  }, null, 2));
}

try {
  main();
} catch (error) {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
}
