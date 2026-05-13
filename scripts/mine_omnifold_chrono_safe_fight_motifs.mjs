#!/usr/bin/env node
import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';

const SCHEMA_VERSION = 'dojo.omnifold_chrono_safe_fight_motif_mining.v1';
const DEFAULT_CONDITION = '/home/frosty40/cuda_dojo/LOGIC_RAY_FRONTIER_RUNTIME_RECEIPT_2026-05-06.md';

function usage() {
  return `Usage: node scripts/mine_omnifold_chrono_safe_fight_motifs.mjs --resolution <selected_family_resolution.json> --chrono <chrono_o2_tactical_sidecar.json> [--out <motifs.json>]

Mine new GPU-derived OmniFold/chrono/tactical motifs against recorded
forced-candidate GPU fight labels. A motif is eligible only if it changes at
least one root, improves fight labels, and does not regress accepted-useful or
selected-move top1. This is post-hoc JSON analysis over recorded CUDA-derived
artifacts only; it does not run chess logic or change runtime behavior.
`;
}

function parseArgs(argv) {
  const args = {
    resolution: null,
    chrono: null,
    out: null,
    folds: 4,
    maxSize: 3,
    minRoots: 2,
    minRows: 2,
    maxMotifs: 384,
    topK: 24,
    conditionSource: DEFAULT_CONDITION,
  };
  for (let i = 0; i < argv.length; i += 1) {
    const token = argv[i];
    if (token === '--help' || token === '-h') {
      console.log(usage());
      process.exit(0);
    } else if (token === '--resolution') {
      args.resolution = argv[++i];
    } else if (token === '--chrono') {
      args.chrono = argv[++i];
    } else if (token === '--out') {
      args.out = argv[++i];
    } else if (token === '--folds') {
      args.folds = Number(argv[++i]);
    } else if (token === '--max-size') {
      args.maxSize = Number(argv[++i]);
    } else if (token === '--min-roots') {
      args.minRoots = Number(argv[++i]);
    } else if (token === '--min-rows') {
      args.minRows = Number(argv[++i]);
    } else if (token === '--max-motifs') {
      args.maxMotifs = Number(argv[++i]);
    } else if (token === '--top-k') {
      args.topK = Number(argv[++i]);
    } else if (token === '--condition-source') {
      args.conditionSource = argv[++i];
    } else {
      throw new Error(`unknown argument: ${token}\n${usage()}`);
    }
  }
  if (!args.resolution) throw new Error(`missing --resolution\n${usage()}`);
  if (!args.chrono) throw new Error(`missing --chrono\n${usage()}`);
  for (const [name, min, max] of [
    ['folds', 2, 16],
    ['maxSize', 1, 4],
    ['minRoots', 1, 64],
    ['minRows', 1, 4096],
    ['maxMotifs', 1, 20000],
    ['topK', 1, 512],
  ]) {
    if (!Number.isInteger(args[name]) || args[name] < min || args[name] > max) {
      throw new Error(`--${name.replace(/[A-Z]/g, (c) => `-${c.toLowerCase()}`)} must be an integer in [${min},${max}]`);
    }
  }
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

function tagSafe(text) {
  return String(text ?? 'unknown').replace(/[^A-Za-z0-9_.:-]+/g, '_');
}

function safeId(text) {
  return String(text ?? 'unknown').replace(/[^A-Za-z0-9_.:-]+/g, '_');
}

function moveGeometry(move) {
  const text = String(move || '');
  if (text.length < 4) {
    return {
      df: null,
      dr: null,
      distance: null,
      knightLike: false,
      diagonal: false,
      sameFile: false,
      sameRank: false,
      centerTarget: false,
      edgeTarget: false,
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
    distance,
    knightLike: Math.min(df, dr) === 1 && Math.max(df, dr) === 2,
    diagonal: df === dr && df > 0,
    sameFile: df === 0 && dr > 0,
    sameRank: dr === 0 && df > 0,
    centerTarget: toFile >= 2 && toFile <= 5 && toRank >= 2 && toRank <= 5,
    edgeTarget: toFile === 0 || toFile === 7 || toRank === 0 || toRank === 7,
    longMove: distance >= 2.5,
  };
}

function chronoByHash(chrono) {
  return new Map((Array.isArray(chrono.rows) ? chrono.rows : [])
    .filter((row) => row.logicRayFrontierHash)
    .map((row) => [row.logicRayFrontierHash, row]));
}

function addTag(tags, tag, enabled = true) {
  if (enabled) tags.add(tag);
}

function bucketReplies(value) {
  const replies = asNumber(value, 0);
  if (replies <= 20) return 'low';
  if (replies <= 36) return 'mid';
  return 'high';
}

function bucketBalance(value) {
  const balance = asNumber(value, 0);
  if (balance > 0) return 'positive';
  if (balance < 0) return 'negative';
  return 'neutral';
}

function bucketTheta(sinValue, cosValue) {
  const sin = asNumber(sinValue, 0);
  const cos = asNumber(cosValue, 0);
  if (sin >= 0 && cos >= 0) return 'q1';
  if (sin >= 0 && cos < 0) return 'q2';
  if (sin < 0 && cos < 0) return 'q3';
  return 'q4';
}

function rowTags(row, chronoRow) {
  const tags = new Set();
  const selected = row.resolution?.selected || {};
  const groups = Array.isArray(selected.selectedVariantGroups)
    ? selected.selectedVariantGroups.map(String).sort()
    : [];
  const groupScores = row.resolution?.groupScores || {};
  const geometry = moveGeometry(row.move);
  const tactical = chronoRow?.tacticalContact || {};
  const flags = tactical.flags || {};
  const counts = tactical.counts || {};
  const pieces = tactical.pieces || {};
  const diagnostics = chronoRow?.diagnostics || {};

  addTag(tags, `omnifold:family:${tagSafe(row.selectedFamily)}`, row.selectedFamily);
  addTag(tags, `omnifold:family_id:${tagSafe(row.selectedFamilyId)}`, row.selectedFamilyId);
  addTag(tags, `omnifold:variant:${tagSafe(row.selectedVariantId)}`, row.selectedVariantId);
  for (const group of groups) addTag(tags, `omnifold:group:${tagSafe(group)}`);
  for (const [key, rawValue] of Object.entries(groupScores)) {
    const value = asNumber(rawValue, 0);
    addTag(tags, `omnifold:score:${tagSafe(key)}:high`, value >= 0.85);
    addTag(tags, `omnifold:score:${tagSafe(key)}:mid`, value >= 0.7 && value < 0.85);
    addTag(tags, `omnifold:score:${tagSafe(key)}:low`, value < 0.25);
  }

  addTag(tags, 'rank:1', asNumber(row.rank, 0) === 1);
  addTag(tags, 'rank:2', asNumber(row.rank, 0) === 2);
  addTag(tags, 'rank:lte2', asNumber(row.rank, 99) <= 2);
  addTag(tags, 'rank:lte3', asNumber(row.rank, 99) <= 3);
  addTag(tags, 'move:knight_like', geometry.knightLike);
  addTag(tags, 'move:diagonal', geometry.diagonal);
  addTag(tags, 'move:same_file', geometry.sameFile);
  addTag(tags, 'move:same_rank', geometry.sameRank);
  addTag(tags, 'move:center_target', geometry.centerTarget);
  addTag(tags, 'move:edge_target', geometry.edgeTarget);
  addTag(tags, 'move:long', geometry.longMove);

  addTag(tags, `chrono:phase:${tagSafe(chronoRow?.timePhase?.phase)}`, chronoRow?.timePhase?.phase);
  addTag(tags, `chrono:theta:${bucketTheta(chronoRow?.timePhase?.thetaSin, chronoRow?.timePhase?.thetaCos)}`, chronoRow);
  addTag(tags, `chrono:pressure:${tagSafe(chronoRow?.pressureDrift?.bucket)}`, chronoRow?.pressureDrift?.bucket);
  addTag(tags, `chrono:relation:${tagSafe(chronoRow?.relationDrift?.bucket)}`, chronoRow?.relationDrift?.bucket);
  addTag(tags, `chrono:contortion:${tagSafe(chronoRow?.pathContortion?.bucket)}`, chronoRow?.pathContortion?.bucket);
  addTag(tags, `chrono:uncertainty:${tagSafe(chronoRow?.uncertainty?.bucket)}`, chronoRow?.uncertainty?.bucket);
  addTag(tags, 'chrono:stable_score', asNumber(diagnostics.stabilityScore, 0) >= 0.7);
  addTag(tags, 'chrono:low_stability', asNumber(diagnostics.stabilityScore, 0) < 0.55);
  addTag(tags, 'chrono:norm_drift_high', asNumber(diagnostics.normDrift, 0) >= 4);

  addTag(tags, 'tactical:gpu_verified', tactical.gpuVerified === true);
  addTag(tags, 'tactical:defended_before', flags.destinationDefendedBefore);
  addTag(tags, 'tactical:attacked_before', flags.destinationAttackedBefore);
  addTag(tags, 'tactical:defended_after', flags.destinationDefendedAfter);
  addTag(tags, 'tactical:attacked_after', flags.destinationAttackedAfter);
  addTag(tags, 'tactical:contested_after', flags.destinationDefendedAfter && flags.destinationAttackedAfter);
  addTag(tags, 'tactical:safe_after', flags.destinationDefendedAfter && !flags.destinationAttackedAfter);
  addTag(tags, 'tactical:loose_after', flags.destinationAttackedAfter && !flags.destinationDefendedAfter);
  addTag(tags, 'tactical:mover_in_check', flags.moverInCheckBefore);
  addTag(tags, 'tactical:gives_check', flags.givesCheckAfter);
  addTag(tags, 'tactical:capture_like', flags.captureLike);
  addTag(tags, 'tactical:quiet_like', !flags.captureLike);
  addTag(tags, 'tactical:defended_capture', flags.captureLike && flags.destinationDefendedAfter);
  addTag(tags, 'tactical:attacked_capture', flags.captureLike && flags.destinationAttackedAfter);
  addTag(tags, `tactical:replies:${bucketReplies(counts.opponentLegalRepliesAfter)}`, chronoRow);
  addTag(tags, `tactical:contact_balance:${bucketBalance(counts.contactBalanceAfter)}`, chronoRow);
  addTag(tags, `tactical:moving_piece:${tagSafe(pieces.movingPiece)}`, pieces.movingPiece !== undefined);
  addTag(tags, `tactical:captured_piece:${tagSafe(pieces.capturedPiece)}`, asNumber(pieces.capturedPiece, 0) !== 0);

  return [...tags].sort();
}

function normalizeRows(resolution, chronoRows) {
  return (Array.isArray(resolution.rowAssignments) ? resolution.rowAssignments : [])
    .filter((row) => row.rootId && row.move)
    .map((row) => {
      const chronoRow = chronoRows.get(row.logicRayFrontierHash);
      const selected = row.resolution?.selected || {};
      const groups = Array.isArray(selected.selectedVariantGroups)
        ? selected.selectedVariantGroups.map(String).sort()
        : [];
      const hasFightScore = row.fightScore !== null
        && row.fightScore !== undefined
        && Number.isFinite(Number(row.fightScore));
      const tags = rowTags(row, chronoRow);
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
        selectedVariantGroups: groups,
        tags,
        tagSet: new Set(tags),
        chronoJoined: Boolean(chronoRow),
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

function rootFolds(rootIds, foldCount) {
  return Array.from({ length: foldCount }, (_, fold) => (
    rootIds.filter((_, index) => index % foldCount === fold)
  )).filter((foldRoots) => foldRoots.length > 0);
}

function filterGroups(groups, rootSet) {
  return groups.filter(([rootId]) => rootSet.has(rootId));
}

function combinations(values, size, start = 0, prefix = [], out = []) {
  if (prefix.length === size) {
    out.push([...prefix]);
    return out;
  }
  for (let i = start; i <= values.length - (size - prefix.length); i += 1) {
    prefix.push(values[i]);
    combinations(values, size, i + 1, prefix, out);
    prefix.pop();
  }
  return out;
}

function motifId(tags) {
  return `safe_fight_motif_${sha256Object(tags).slice(0, 16)}`;
}

function motifKey(tags) {
  return tags.join('|');
}

function generateMotifs(rows, args) {
  const stats = new Map();
  for (const row of rows) {
    for (let size = 1; size <= args.maxSize; size += 1) {
      for (const combo of combinations(row.tags, size)) {
        const key = motifKey(combo);
        if (!stats.has(key)) {
          stats.set(key, {
            id: motifId(combo),
            tags: combo,
            rows: 0,
            fightRows: 0,
            fightScoreSum: 0,
            positiveFightRows: 0,
            acceptedRows: 0,
            selectedRows: 0,
            rootIds: new Set(),
            examples: [],
          });
        }
        const entry = stats.get(key);
        entry.rows += 1;
        entry.rootIds.add(row.rootId);
        if (Number.isFinite(row.fightScore)) {
          entry.fightRows += 1;
          entry.fightScoreSum += row.fightScore;
          if (row.fightScore > 0) entry.positiveFightRows += 1;
        }
        if (row.acceptedUsefulInjection) entry.acceptedRows += 1;
        if (row.selectedMoveInFrontier) entry.selectedRows += 1;
        if (entry.examples.length < 8) {
          entry.examples.push({
            rootId: row.rootId,
            bridgeId: row.bridgeId,
            move: row.move,
            rank: row.rank,
            fightScore: row.fightScore == null ? null : round(row.fightScore),
            acceptedUsefulInjection: row.acceptedUsefulInjection,
            selectedMoveInFrontier: row.selectedMoveInFrontier,
            hash: row.hash,
          });
        }
      }
    }
  }
  return [...stats.values()]
    .filter((entry) => entry.rows >= args.minRows && entry.rootIds.size >= args.minRoots)
    .map((entry) => ({
      ...entry,
      rootCount: entry.rootIds.size,
      meanFightScore: entry.fightRows ? entry.fightScoreSum / entry.fightRows : null,
      acceptedRate: entry.rows ? entry.acceptedRows / entry.rows : 0,
      selectedRate: entry.rows ? entry.selectedRows / entry.rows : 0,
      positiveFightRate: entry.fightRows ? entry.positiveFightRows / entry.fightRows : 0,
    }))
    .sort((a, b) => (
      (b.positiveFightRate - a.positiveFightRate)
        || (asNumber(b.meanFightScore, -999) - asNumber(a.meanFightScore, -999))
        || (b.acceptedRate - a.acceptedRate)
        || (b.rootCount - a.rootCount)
        || a.id.localeCompare(b.id)
    ))
    .slice(0, args.maxMotifs);
}

function motifMatches(row, motif) {
  return motif.tags.every((tag) => row.tagSet.has(tag));
}

function publicTop(row, score, baselineRow = null) {
  return {
    bridgeId: row.bridgeId,
    hash: row.hash,
    move: row.move,
    rank: row.rank,
    score: round(score),
    selectedFamily: row.selectedFamily,
    selectedFamilyId: row.selectedFamilyId,
    selectedVariantId: row.selectedVariantId,
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
    const baselineRow = rows.slice().sort((a, b) => a.rank - b.rank || a.move.localeCompare(b.move))[0] || null;
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
      top1: publicTop(top.row, top.score, baselineRow),
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

function changedRoots(candidate, baseline, limit = 16) {
  const baselineByRoot = new Map((baseline.roots || []).map((root) => [root.rootId, root.top1]));
  return (candidate.roots || [])
    .map((root) => {
      const base = baselineByRoot.get(root.rootId);
      const changed = base?.hash !== root.top1?.hash;
      return {
        rootId: root.rootId,
        changed,
        baselineTop1: base || null,
        candidateTop1: root.top1 || null,
        deltaFightScore: base?.fightScore != null && root.top1?.fightScore != null
          ? round(root.top1.fightScore - base.fightScore)
          : null,
        acceptedDelta: Number(Boolean(root.top1?.acceptedUsefulInjection)) - Number(Boolean(base?.acceptedUsefulInjection)),
        selectedDelta: Number(Boolean(root.top1?.selectedMoveInFrontier)) - Number(Boolean(base?.selectedMoveInFrontier)),
      };
    })
    .filter((root) => root.changed)
    .slice(0, limit);
}

function candidateSpecs(motifs) {
  const bonuses = [0.28, 0.55, 1.05];
  const rankPenalties = [0.02, 0.05, 0.12, 0.25];
  const maxRanks = [2, 3, 99];
  const confidenceWeights = [0, 0.25];
  const specs = [];
  for (const motif of motifs) {
    for (const bonus of bonuses) {
      for (const rankPenalty of rankPenalties) {
        for (const maxRank of maxRanks) {
          for (const confidenceWeight of confidenceWeights) {
            specs.push({
              id: [
                motif.id,
                `b${String(bonus).replace('.', '_')}`,
                `r${String(rankPenalty).replace('.', '_')}`,
                `mr${maxRank}`,
                `cw${String(confidenceWeight).replace('.', '_')}`,
              ].join('.'),
              motifId: motif.id,
              tags: motif.tags,
              bonus,
              rankPenalty,
              maxRank,
              confidenceWeight,
            });
          }
        }
      }
    }
  }
  return specs;
}

function rowMatchesSpec(row, spec) {
  return row.rank <= spec.maxRank && spec.tags.every((tag) => row.tagSet.has(tag));
}

function scoreRow(row, spec) {
  return (rowMatchesSpec(row, spec) ? spec.bonus : 0)
    + spec.confidenceWeight * row.confidence
    - spec.rankPenalty * (row.rank - 1);
}

function evaluateSpec(groups, spec, name = spec.id) {
  return evaluate(groups, (row) => scoreRow(row, spec), name);
}

function eligibility(allRowsDelta, fightRowsDelta, changedCount) {
  const acceptedSafe = allRowsDelta.acceptedUsefulTop1 >= 0 && fightRowsDelta.acceptedUsefulTop1 >= 0;
  const selectedSafe = allRowsDelta.selectedMoveTop1 >= 0 && fightRowsDelta.selectedMoveTop1 >= 0;
  const fightLift = (
    fightRowsDelta.meanFightScore != null && fightRowsDelta.meanFightScore > 0
  ) || fightRowsDelta.positiveTop1 > 0;
  const changed = changedCount > 0;
  return {
    acceptedSafe,
    selectedSafe,
    fightLift,
    changed,
    ready: acceptedSafe && selectedSafe && fightLift && changed,
  };
}

function scoreCandidate(allRowsDelta, fightRowsDelta, eligibilityResult) {
  return round(
    asNumber(fightRowsDelta.meanFightScore, -1) * 1000
      + fightRowsDelta.positiveTop1 * 10
      + allRowsDelta.acceptedUsefulTop1 * 8
      + fightRowsDelta.acceptedUsefulTop1 * 8
      + allRowsDelta.selectedMoveTop1 * 4
      + fightRowsDelta.selectedMoveTop1 * 4
      + (eligibilityResult.ready ? 10000 : 0),
  );
}

function evaluateCandidate(spec, allGroups, fightGroups, baselines = null) {
  const allBaseline = baselines?.allBaseline || evaluate(allGroups, (row) => -(row.rank - 1), 'all_rows_frontier_rank');
  const fightBaseline = baselines?.fightBaseline || evaluate(fightGroups, (row) => -(row.rank - 1), 'fight_rows_frontier_rank');
  const allCandidate = evaluateSpec(allGroups, spec, 'all_rows_candidate');
  const fightCandidate = evaluateSpec(fightGroups, spec, 'fight_rows_candidate');
  const allRowsDelta = delta(allCandidate, allBaseline);
  const fightRowsDelta = delta(fightCandidate, fightBaseline);
  const allChanged = changedRoots(allCandidate, allBaseline, 64);
  const fightChanged = changedRoots(fightCandidate, fightBaseline, 64);
  const eligibilityResult = eligibility(allRowsDelta, fightRowsDelta, allChanged.length + fightChanged.length);
  return {
    spec,
    eligibility: eligibilityResult,
    score: scoreCandidate(allRowsDelta, fightRowsDelta, eligibilityResult),
    allRowsSurface: {
      baseline: metricsPublic(allBaseline),
      candidate: metricsPublic(allCandidate),
      deltaVsFrontierRank: allRowsDelta,
      changedRootCount: allChanged.length,
      changedRoots: allChanged.slice(0, 12),
    },
    fightRowsSurface: {
      baseline: metricsPublic(fightBaseline),
      candidate: metricsPublic(fightCandidate),
      deltaVsFrontierRank: fightRowsDelta,
      changedRootCount: fightChanged.length,
      changedRoots: fightChanged.slice(0, 12),
    },
  };
}

function compactCandidate(candidate) {
  return {
    spec: candidate.spec,
    eligibility: candidate.eligibility,
    score: candidate.score,
    allRowsDelta: candidate.allRowsSurface.deltaVsFrontierRank,
    fightRowsDelta: candidate.fightRowsSurface.deltaVsFrontierRank,
    allRowsChangedRootCount: candidate.allRowsSurface.changedRootCount,
    fightRowsChangedRootCount: candidate.fightRowsSurface.changedRootCount,
    changedRootsPreview: {
      allRows: candidate.allRowsSurface.changedRoots.slice(0, 8),
      fightRows: candidate.fightRowsSurface.changedRoots.slice(0, 8),
    },
  };
}

function pickBest(specs, allGroups, fightGroups) {
  let best = null;
  const topEligible = [];
  const topBlocked = [];
  const baselines = {
    allBaseline: evaluate(allGroups, (row) => -(row.rank - 1), 'all_rows_frontier_rank'),
    fightBaseline: evaluate(fightGroups, (row) => -(row.rank - 1), 'fight_rows_frontier_rank'),
  };
  for (const spec of specs) {
    const candidate = evaluateCandidate(spec, allGroups, fightGroups, baselines);
    if (!best || candidate.score > best.score || (candidate.score === best.score && spec.id.localeCompare(best.spec.id) < 0)) {
      best = candidate;
    }
    const compact = compactCandidate(candidate);
    const list = candidate.eligibility.ready ? topEligible : topBlocked;
    list.push(compact);
    list.sort((a, b) => b.score - a.score || a.spec.id.localeCompare(b.spec.id));
    if (list.length > 24) list.length = 24;
  }
  return { best, topEligible, topBlocked };
}

function aggregate(name, metricsList) {
  const rootCount = metricsList.reduce((sum, item) => sum + item.rootCount, 0);
  const fightTop1Rows = metricsList.reduce((sum, item) => sum + item.fightTop1Rows, 0);
  const scoreSum = metricsList.reduce((sum, item) => sum + asNumber(item.scoreSum, 0), 0);
  return {
    name,
    rootCount,
    fightTop1Rows,
    scoreSum: round(scoreSum),
    meanFightScore: fightTop1Rows ? round(scoreSum / fightTop1Rows) : null,
    positiveTop1: metricsList.reduce((sum, item) => sum + item.positiveTop1, 0),
    neutralTop1: metricsList.reduce((sum, item) => sum + item.neutralTop1, 0),
    negativeTop1: metricsList.reduce((sum, item) => sum + item.negativeTop1, 0),
    acceptedUsefulTop1: metricsList.reduce((sum, item) => sum + item.acceptedUsefulTop1, 0),
    selectedMoveTop1: metricsList.reduce((sum, item) => sum + item.selectedMoveTop1, 0),
    roots: metricsList.flatMap((item) => item.roots || []),
  };
}

function crossValidate(rows, args) {
  const allGroups = groupByRoot(rows);
  const rootIds = allGroups.map(([rootId]) => rootId);
  const folds = rootFolds(rootIds, Math.min(args.folds, rootIds.length));
  const allBaselineEvals = [];
  const allCandidateEvals = [];
  const fightBaselineEvals = [];
  const fightCandidateEvals = [];
  const foldReports = [];

  for (let foldIndex = 0; foldIndex < folds.length; foldIndex += 1) {
    const evalSet = new Set(folds[foldIndex]);
    const trainSet = new Set(rootIds.filter((rootId) => !evalSet.has(rootId)));
    const trainRows = rows.filter((row) => trainSet.has(row.rootId));
    const trainFightRows = trainRows.filter((row) => Number.isFinite(row.fightScore));
    const trainMotifs = generateMotifs(trainFightRows, args);
    const trainSpecs = candidateSpecs(trainMotifs);
    const trainAllGroups = filterGroups(allGroups, trainSet);
    const trainFightGroups = groupByRoot(trainFightRows);
    const evalAllGroups = filterGroups(allGroups, evalSet);
    const evalFightGroups = groupByRoot(rows.filter((row) => evalSet.has(row.rootId) && Number.isFinite(row.fightScore)));
    const selected = trainSpecs.length ? pickBest(trainSpecs, trainAllGroups, trainFightGroups).best : null;

    const allBaseline = evaluate(evalAllGroups, (row) => -(row.rank - 1), 'eval_all_frontier_rank');
    const fightBaseline = evaluate(evalFightGroups, (row) => -(row.rank - 1), 'eval_fight_frontier_rank');
    const allCandidate = selected ? evaluateSpec(evalAllGroups, selected.spec, 'eval_all_candidate') : allBaseline;
    const fightCandidate = selected ? evaluateSpec(evalFightGroups, selected.spec, 'eval_fight_candidate') : fightBaseline;
    const allRowsDelta = delta(allCandidate, allBaseline);
    const fightRowsDelta = delta(fightCandidate, fightBaseline);
    const foldEligibility = eligibility(
      allRowsDelta,
      fightRowsDelta,
      changedRoots(allCandidate, allBaseline, 64).length + changedRoots(fightCandidate, fightBaseline, 64).length,
    );

    allBaselineEvals.push(allBaseline);
    allCandidateEvals.push(allCandidate);
    fightBaselineEvals.push(fightBaseline);
    fightCandidateEvals.push(fightCandidate);
    foldReports.push({
      fold: foldIndex,
      trainRootCount: trainAllGroups.length,
      evalRootCount: evalAllGroups.length,
      trainMotifCount: trainMotifs.length,
      trainSpecCount: trainSpecs.length,
      selectedTrainCandidate: selected ? compactCandidate(selected) : null,
      evalEligibility: foldEligibility,
      evalAllRowsDelta: allRowsDelta,
      evalFightRowsDelta: fightRowsDelta,
      evalAllChangedRoots: changedRoots(allCandidate, allBaseline, 8),
      evalFightChangedRoots: changedRoots(fightCandidate, fightBaseline, 8),
    });
  }

  const allBaseline = aggregate('cross_validation_all_rows_frontier_rank', allBaselineEvals);
  const allCandidate = aggregate('cross_validation_all_rows_candidate', allCandidateEvals);
  const fightBaseline = aggregate('cross_validation_fight_rows_frontier_rank', fightBaselineEvals);
  const fightCandidate = aggregate('cross_validation_fight_rows_candidate', fightCandidateEvals);
  const allRowsDelta = delta(allCandidate, allBaseline);
  const fightRowsDelta = delta(fightCandidate, fightBaseline);
  const aggregateEligibility = eligibility(
    allRowsDelta,
    fightRowsDelta,
    changedRoots(allCandidate, allBaseline, 64).length + changedRoots(fightCandidate, fightBaseline, 64).length,
  );
  return {
    allRowsSurface: {
      baseline: metricsPublic(allBaseline),
      candidate: metricsPublic(allCandidate),
      deltaVsFrontierRank: allRowsDelta,
    },
    fightRowsSurface: {
      baseline: metricsPublic(fightBaseline),
      candidate: metricsPublic(fightCandidate),
      deltaVsFrontierRank: fightRowsDelta,
    },
    eligibility: aggregateEligibility,
    observedLift: aggregateEligibility.ready,
    folds: foldReports,
  };
}

function motifPublic(motif) {
  return {
    id: motif.id,
    tags: motif.tags,
    rowCount: motif.rows,
    rootCount: motif.rootCount,
    fightRows: motif.fightRows,
    meanFightScore: motif.meanFightScore == null ? null : round(motif.meanFightScore),
    positiveFightRows: motif.positiveFightRows,
    positiveFightRate: round(motif.positiveFightRate),
    acceptedRows: motif.acceptedRows,
    acceptedRate: round(motif.acceptedRate),
    selectedRows: motif.selectedRows,
    selectedRate: round(motif.selectedRate),
    examples: motif.examples,
  };
}

function buildFrozenCandidate(fixedBest, crossValidation, resolutionPath, resolutionSha256, chronoPath, chronoSha256) {
  if (!fixedBest?.eligibility?.ready || !crossValidation?.eligibility?.ready) return null;
  const payload = {
    sourceResolutionPath: resolutionPath,
    sourceResolutionSha256: resolutionSha256,
    sourceChronoPath: chronoPath,
    sourceChronoSha256: chronoSha256,
    candidateSpec: fixedBest.spec,
    fixedArtifactAllRowsDelta: fixedBest.allRowsSurface.deltaVsFrontierRank,
    fixedArtifactFightRowsDelta: fixedBest.fightRowsSurface.deltaVsFrontierRank,
    crossValidatedAllRowsDelta: crossValidation.allRowsSurface.deltaVsFrontierRank,
    crossValidatedFightRowsDelta: crossValidation.fightRowsSurface.deltaVsFrontierRank,
  };
  return {
    conditionId: `omnifold_chrono_safe_fight_motif_${safeId(fixedBest.spec.id)}`,
    conditionHash: sha256Object(payload),
    status: 'frozen_candidate_not_promoted',
    frozenPayload: payload,
  };
}

function defaultOutPath(resolutionPath) {
  const parsed = path.parse(resolutionPath);
  return path.join(parsed.dir, `${parsed.name}.omnifold_chrono_safe_fight_motifs.json`);
}

function main() {
  const args = parseArgs(process.argv.slice(2));
  const resolutionPath = path.resolve(args.resolution);
  const chronoPath = path.resolve(args.chrono);
  const outPath = path.resolve(args.out || defaultOutPath(resolutionPath));
  const resolution = readJson(resolutionPath);
  const chrono = readJson(chronoPath);
  const resolutionSha256 = sha256File(resolutionPath);
  const chronoSha256 = sha256File(chronoPath);
  const rows = normalizeRows(resolution, chronoByHash(chrono));
  const fightRows = rows.filter((row) => Number.isFinite(row.fightScore));
  const motifs = generateMotifs(fightRows, args);
  const specs = candidateSpecs(motifs);
  const allGroups = groupByRoot(rows);
  const fightGroups = groupByRoot(fightRows);
  const fixed = pickBest(specs, allGroups, fightGroups);
  const crossValidation = crossValidate(rows, args);
  const frozenCandidate = buildFrozenCandidate(fixed.best, crossValidation, resolutionPath, resolutionSha256, chronoPath, chronoSha256);

  const output = {
    schemaVersion: SCHEMA_VERSION,
    generatedAt: new Date().toISOString(),
    condition: {
      source: path.resolve(args.conditionSource),
      runLabel: 'posthoc_safe_gpu_fight_motif_mining_over_omnifold_chrono_tactical_rows',
      changedFields: 'post-hoc motif mining over recorded CUDA-derived selected-family, chrono/tactical, and GPU fight artifacts only; no runtime behavior changed',
      labCondition: 'scout/new_experiment/uses_existing_depth5_ply12_gpu_fight_labels',
      metric: 'root-fold fight lift with accepted-useful and selected-move top1 non-regression',
    },
    sources: {
      resolutionPath,
      resolutionSha256,
      resolutionSchemaVersion: resolution.schemaVersion || null,
      chronoPath,
      chronoSha256,
      chronoSchemaVersion: chrono.schemaVersion || null,
      fightPath: resolution.sources?.fightPath || null,
      fightSha256: resolution.sources?.fightSha256 || null,
      sourceBridgePath: resolution.sources?.bridgePath || null,
      sourceBridgeSha256: resolution.sources?.bridgeSha256 || null,
    },
    miningPolicy: {
      algorithm: 'gpu_fight_label_safe_motif_sweep_with_root_fold_validation',
      hostRole: 'json_feature_projection_only',
      noChessRuntime: true,
      noRuntimePromotion: true,
      excludedGuardFields: ['acceptedUsefulInjection', 'fightScore', 'selectedMoveInFrontier'],
      rows: rows.length,
      chronoJoinedRows: rows.filter((row) => row.chronoJoined).length,
      fightRows: fightRows.length,
      rootCount: allGroups.length,
      fightRootCount: fightGroups.length,
      candidateMotifCount: motifs.length,
      candidateSpecCount: specs.length,
      folds: args.folds,
      maxMotifSize: args.maxSize,
      minRoots: args.minRoots,
      minRows: args.minRows,
      maxMotifs: args.maxMotifs,
    },
    motifAtlas: {
      topFightMotifs: motifs.slice(0, args.topK).map(motifPublic),
    },
    fixedArtifact: {
      bestCandidate: fixed.best ? compactCandidate(fixed.best) : null,
      topEligibleCandidates: fixed.topEligible.slice(0, args.topK),
      topBlockedCandidates: fixed.topBlocked.slice(0, args.topK),
    },
    crossValidation,
    frozenCandidate,
    promotionPolicy: {
      status: 'not_promoted',
      reason: frozenCandidate
        ? 'safe GPU-fight motif has a frozen scout candidate, but runtime promotion still requires a fresh fixed-condition accepted-injection GPU gate'
        : 'no motif passed both fixed-artifact and root-fold safe fight-lift criteria',
      requiredNextEvidence: frozenCandidate ? [
        'freeze condition artifact is present but not promoted',
        'fresh accepted-injection GPU gate under this exact conditionHash',
        'runtime consumer proof with GPU legality/search unchanged',
      ] : [
        'mine a stronger independent GPU-derived outcome signal',
        'or increase fight label quality before another runtime consumer test',
      ],
    },
  };

  writeJson(outPath, output);
  console.log(JSON.stringify({
    ok: true,
    output: outPath,
    rows: rows.length,
    fightRows: fightRows.length,
    candidateMotifCount: motifs.length,
    candidateSpecCount: specs.length,
    fixedEligibility: fixed.best?.eligibility || null,
    fixedFightRowsDelta: fixed.best?.fightRowsSurface?.deltaVsFrontierRank || null,
    crossValidatedEligibility: crossValidation.eligibility,
    crossValidatedFightRowsDelta: crossValidation.fightRowsSurface.deltaVsFrontierRank,
    frozenCandidate: frozenCandidate ? {
      conditionId: frozenCandidate.conditionId,
      conditionHash: frozenCandidate.conditionHash,
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
