#!/usr/bin/env node
import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';

const SCHEMA_VERSION = 'dojo.omnifold_selected_family_consumer_contrast.v1';

function usage() {
  return `Usage: node scripts/analyze_omnifold_selected_family_consumer_contrast.mjs --consumer <consumer_eval.json> [--resolution <selected_family_resolution.json>] [--out <contrast.json>]

Analyze why a frozen selected-family consumer helped the fight-labeled surface
but failed accepted-useful selection. This reads recorded JSON artifacts only.
`;
}

function parseArgs(argv) {
  const args = { consumer: null, resolution: null, out: null };
  for (let i = 0; i < argv.length; i += 1) {
    const token = argv[i];
    if (token === '--help' || token === '-h') {
      console.log(usage());
      process.exit(0);
    } else if (token === '--consumer') {
      args.consumer = argv[++i];
    } else if (token === '--resolution') {
      args.resolution = argv[++i];
    } else if (token === '--out') {
      args.out = argv[++i];
    } else {
      throw new Error(`unknown argument: ${token}\n${usage()}`);
    }
  }
  if (!args.consumer) throw new Error(`missing --consumer\n${usage()}`);
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

function moveGeometry(move) {
  const text = String(move || '');
  if (text.length < 4) return { df: null, dr: null, distance: null, longMove: false, edgeTarget: false };
  const fromFile = text.charCodeAt(0) - 97;
  const fromRank = Number(text[1]) - 1;
  const toFile = text.charCodeAt(2) - 97;
  const toRank = Number(text[3]) - 1;
  const df = Math.abs(toFile - fromFile);
  const dr = Math.abs(toRank - fromRank);
  return {
    df,
    dr,
    distance: round(Math.sqrt(df * df + dr * dr)),
    longMove: Math.sqrt(df * df + dr * dr) >= 2.5,
    edgeTarget: toFile === 0 || toFile === 7 || toRank === 0 || toRank === 7,
  };
}

function rowsByHash(resolution) {
  const out = new Map();
  for (const row of resolution.rowAssignments || []) {
    if (row.logicRayFrontierHash) out.set(row.logicRayFrontierHash, row);
  }
  return out;
}

function rowFeature(row) {
  const selected = row?.resolution?.selected || {};
  return {
    hash: row?.logicRayFrontierHash || null,
    bridgeId: row?.bridgeId || null,
    rootId: row?.rootId || null,
    move: row?.move || null,
    moveGeometry: moveGeometry(row?.move),
    rank: row?.rank ?? null,
    acceptedUsefulInjection: Boolean(row?.acceptedUsefulInjection),
    selectedMoveInFrontier: Boolean(row?.selectedMoveInFrontier),
    fightScore: row?.fightScore ?? null,
    selectedFamily: row?.selectedFamily || null,
    selectedFamilyId: row?.selectedFamilyId || null,
    selectedVariantId: row?.selectedVariantId || null,
    confidence: round(row?.resolution?.confidence),
    margin: round(row?.resolution?.margin),
    selectedVariantGroups: Array.isArray(selected.selectedVariantGroups)
      ? selected.selectedVariantGroups
      : [],
    groupScores: row?.resolution?.groupScores || {},
  };
}

function enrichChange(change, byHash) {
  const baselineRow = byHash.get(change.baselineTop1?.hash);
  const consumerRow = byHash.get(change.consumerTop1?.hash);
  return {
    rootId: change.rootId,
    deltaFightScore: change.deltaFightScore,
    acceptedDelta: change.acceptedDelta,
    selectedDelta: change.selectedDelta,
    baseline: rowFeature(baselineRow),
    consumer: rowFeature(consumerRow),
    rankJump: consumerRow && baselineRow ? asNumber(consumerRow.rank, 0) - asNumber(baselineRow.rank, 0) : null,
    consumerHasFightLabel: consumerRow?.fightScore !== null && consumerRow?.fightScore !== undefined,
  };
}

function summarizeChanges(changes) {
  const count = changes.length;
  const fightKnown = changes.filter((change) => change.consumerHasFightLabel);
  const fightImproved = changes.filter((change) => (
    change.deltaFightScore !== null
      && change.deltaFightScore !== undefined
      && Number.isFinite(Number(change.deltaFightScore))
      && Number(change.deltaFightScore) > 0
  ));
  const acceptedLost = changes.filter((change) => change.acceptedDelta < 0);
  const selectedLost = changes.filter((change) => change.selectedDelta < 0);
  const rankJumps = changes.map((change) => asNumber(change.rankJump, 0));
  const highRankJumps = changes.filter((change) => asNumber(change.rankJump, 0) > 1);
  const edgeTargets = changes.filter((change) => change.consumer.moveGeometry.edgeTarget);
  const longMoves = changes.filter((change) => change.consumer.moveGeometry.longMove);
  const byVariant = new Map();
  for (const change of changes) {
    const variant = change.consumer.selectedVariantId || 'unknown';
    if (!byVariant.has(variant)) byVariant.set(variant, { variant, count: 0, acceptedDelta: 0, selectedDelta: 0, fightDeltaSum: 0, fightDeltaCount: 0 });
    const item = byVariant.get(variant);
    item.count += 1;
    item.acceptedDelta += asNumber(change.acceptedDelta, 0);
    item.selectedDelta += asNumber(change.selectedDelta, 0);
    if (
      change.deltaFightScore !== null
      && change.deltaFightScore !== undefined
      && Number.isFinite(Number(change.deltaFightScore))
    ) {
      item.fightDeltaSum += Number(change.deltaFightScore);
      item.fightDeltaCount += 1;
    }
  }
  return {
    count,
    fightKnownCount: fightKnown.length,
    fightImprovedCount: fightImproved.length,
    acceptedLostCount: acceptedLost.length,
    selectedLostCount: selectedLost.length,
    meanRankJump: count ? round(rankJumps.reduce((sum, value) => sum + value, 0) / count) : null,
    highRankJumpCount: highRankJumps.length,
    edgeTargetCount: edgeTargets.length,
    longMoveCount: longMoves.length,
    variantSummary: [...byVariant.values()].map((item) => ({
      ...item,
      meanFightDelta: item.fightDeltaCount ? round(item.fightDeltaSum / item.fightDeltaCount) : null,
    })).sort((a, b) => b.count - a.count || a.variant.localeCompare(b.variant)),
  };
}

function defaultOutPath(consumerPath) {
  const parsed = path.parse(consumerPath);
  return path.join(parsed.dir, `${parsed.name}.contrast.json`);
}

function main() {
  const args = parseArgs(process.argv.slice(2));
  const consumerPath = path.resolve(args.consumer);
  const consumer = readJson(consumerPath);
  const resolutionPath = path.resolve(args.resolution || consumer.sources?.resolutionPath);
  const resolution = readJson(resolutionPath);
  const byHash = rowsByHash(resolution);
  const fightChanges = (consumer.surfaces?.fightLabeledSurface?.changedRoots || []).map((change) => enrichChange(change, byHash));
  const allChanges = (consumer.surfaces?.allRowsAcceptedSurface?.changedRoots || []).map((change) => enrichChange(change, byHash));
  const highRankBad = allChanges
    .filter((change) => asNumber(change.rankJump, 0) > 1)
    .filter((change) => change.acceptedDelta < 0 || change.selectedDelta < 0);
  const fightHelpful = fightChanges
    .filter((change) => asNumber(change.deltaFightScore, 0) > 0);
  const output = {
    schemaVersion: SCHEMA_VERSION,
    generatedAt: new Date().toISOString(),
    sources: {
      consumerPath,
      consumerSha256: sha256File(consumerPath),
      resolutionPath,
      resolutionSha256: sha256File(resolutionPath),
      conditionHash: consumer.frozenCondition?.conditionHash || null,
      conditionId: consumer.frozenCondition?.conditionId || null,
    },
    condition: {
      runLabel: 'analysis_omnifold_selected_family_consumer_success_failure_contrast',
      changedFields: 'analysis only over measured consumer artifact; no runtime behavior changed',
      metric: 'contrast fight-labeled helpful changes against all-row accepted-useful losses',
    },
    summaries: {
      allRowsAcceptedSurface: summarizeChanges(allChanges),
      fightLabeledSurface: summarizeChanges(fightChanges),
    },
    fightHelpfulChanges: fightHelpful,
    highRankAcceptedDamageExamples: highRankBad.slice(0, 24),
    narrowedConditionRecommendation: {
      status: 'candidate_pattern_not_gate',
      keep: [
        'R4 negative-yspace remains interesting only when the consumer target is already in the fight-labeled rank<=2 surface',
        'fight-positive cases are small nudges from rank1 to rank2, not broad high-rank promotion',
      ],
      reject: [
        'do not allow the condition to promote arbitrary high-rank R4 negative-yspace rows',
        'do not use fight-score lift without accepted-useful protection',
      ],
      nextGateShape: [
        'targetVariantId == R4_piece_zone_move_family_ray_relation_fold2x2_negative_yspace',
        'rank <= 2',
        'require acceptedUsefulTop1 not to regress before any runtime consumer test',
        'prefer a fresh accepted-injection gate before deeper fight rollouts',
      ],
    },
    promotionPolicy: {
      status: 'not_promoted',
      reason: 'contrast analysis identifies a narrow gameplay-pattern clue but no accepted-useful gate',
      blockers: [
        'accepted_useful_top1_not_improved',
        'high_rank_negative_yspace_promotions_are_destructive',
        'fresh_accepted_injection_gate_missing',
      ],
    },
  };
  const outPath = path.resolve(args.out || defaultOutPath(consumerPath));
  writeJson(outPath, output);
  console.log(JSON.stringify({
    ok: true,
    output: outPath,
    conditionHash: output.sources.conditionHash,
    fightHelpfulChanges: output.fightHelpfulChanges.length,
    allRowChangedRoots: allChanges.length,
    highRankAcceptedDamageExamples: highRankBad.length,
    nextGateShape: output.narrowedConditionRecommendation.nextGateShape,
    promote: false,
  }, null, 2));
}

try {
  main();
} catch (error) {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
}
