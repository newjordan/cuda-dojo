#!/usr/bin/env node
// =============================================================================
// oracle_enrich.mjs
//
// External baseline enrichment for dojo JSONL corpora.
// Adds Stockfish oracle labels and optional Lozza move agreement, then writes a
// compact sigma-style report for disagreement analysis.
// =============================================================================

import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { execFile } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { Stockfish } from '../gpu_spine/reference.mjs';

const __dirname = dirname(fileURLToPath(import.meta.url));
const DEFAULT_INPUT = resolve(__dirname, 'self_play_data.jsonl');
const DEFAULT_OUTPUT = resolve(__dirname, 'artifacts', 'oracle_enrich', 'enriched.jsonl');
const DEFAULT_REPORT = resolve(__dirname, 'artifacts', 'oracle_enrich', 'report.json');
const LOZZA_REFERENCE = resolve(__dirname, 'lozza_reference.mjs');

function usage() {
  console.error(
    'usage: node oracle_enrich.mjs --input <jsonl> [--output <jsonl>] [--report <json>] [--limit N] [--depth N] [--move-key key] [--score-key key] [--with-lozza]'
  );
}

function parseArgs(argv) {
  const args = {
    input: DEFAULT_INPUT,
    output: DEFAULT_OUTPUT,
    report: DEFAULT_REPORT,
    limit: 0,
    depth: 12,
    moveKey: '',
    scoreKey: '',
    withLozza: false,
    withPv: true,
    lozzaTimeoutMs: 8000,
    topDisagreements: 12,
  };

  for (let i = 2; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--input') args.input = resolve(argv[++i]);
    else if (arg === '--output') args.output = resolve(argv[++i]);
    else if (arg === '--report') args.report = resolve(argv[++i]);
    else if (arg === '--limit') args.limit = parseInt(argv[++i], 10) || 0;
    else if (arg === '--depth') args.depth = parseInt(argv[++i], 10) || 12;
    else if (arg === '--move-key') args.moveKey = argv[++i] || '';
    else if (arg === '--score-key') args.scoreKey = argv[++i] || '';
    else if (arg === '--with-lozza') args.withLozza = true;
    else if (arg === '--no-pv') args.withPv = false;
    else if (arg === '--lozza-timeout-ms') args.lozzaTimeoutMs = parseInt(argv[++i], 10) || 8000;
    else if (arg === '--top-disagreements') args.topDisagreements = parseInt(argv[++i], 10) || 12;
    else if (arg === '--help' || arg === '-h') {
      usage();
      process.exit(0);
    } else {
      console.error(`unknown arg: ${arg}`);
      usage();
      process.exit(1);
    }
  }
  return args;
}

function normalizeFen(raw) {
  const parts = String(raw || '').trim().split(/\s+/);
  if (parts.length >= 6) return parts.slice(0, 6).join(' ');
  if (parts.length === 4) return parts.join(' ') + ' 0 1';
  if (parts.length === 3) return parts.join(' ') + ' - 0 1';
  if (parts.length === 2) return parts.join(' ') + ' - - 0 1';
  throw new Error(`unparseable FEN: ${raw}`);
}

function loadJsonl(path, limit = 0) {
  const rows = [];
  const lines = readFileSync(path, 'utf8').split('\n');
  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    rows.push(JSON.parse(trimmed));
    if (limit && rows.length >= limit) break;
  }
  return rows;
}

function chooseField(row, explicitKey, candidates) {
  if (explicitKey) {
    return { key: explicitKey, value: row[explicitKey] ?? null };
  }
  for (const key of candidates) {
    if (row[key] !== undefined && row[key] !== null) return { key, value: row[key] };
  }
  return { key: null, value: null };
}

function toIntOrNull(value) {
  if (value === null || value === undefined || value === '') return null;
  const parsed = Number.parseInt(value, 10);
  return Number.isFinite(parsed) ? parsed : null;
}

function moveMatch(a, b) {
  if (!a || !b) return null;
  return String(a).toLowerCase() === String(b).toLowerCase();
}

function signMatch(a, b) {
  if (a === null || a === undefined || b === null || b === undefined) return null;
  return Math.sign(a) === Math.sign(b);
}

function scoreBucket(absDelta) {
  if (absDelta === null || absDelta === undefined) return null;
  if (absDelta <= 25) return 'lte_25';
  if (absDelta <= 50) return 'lte_50';
  if (absDelta <= 100) return 'lte_100';
  if (absDelta <= 200) return 'lte_200';
  return 'gt_200';
}

function percentile(sorted, q) {
  if (!sorted.length) return null;
  if (sorted.length === 1) return sorted[0];
  const idx = Math.min(sorted.length - 1, Math.max(0, Math.floor(q * (sorted.length - 1))));
  return sorted[idx];
}

function mean(nums) {
  if (!nums.length) return null;
  return nums.reduce((sum, n) => sum + n, 0) / nums.length;
}

function median(sorted) {
  if (!sorted.length) return null;
  const mid = Math.floor(sorted.length / 2);
  if (sorted.length % 2) return sorted[mid];
  return (sorted[mid - 1] + sorted[mid]) / 2;
}

function compactExamples(items, limit) {
  return items.slice(0, limit);
}

function sigmaScore({ absScoreDelta, refVsSfMoveMatch, refVsSfSignMatch, lozzaVsSfMoveMatch }) {
  const cpTerm = absScoreDelta === null ? 0.0 : Math.min(absScoreDelta / 100.0, 5.0);
  const movePenalty = refVsSfMoveMatch === false ? 1.0 : 0.0;
  const signPenalty = refVsSfSignMatch === false ? 1.0 : 0.0;
  const oracleSplitPenalty = lozzaVsSfMoveMatch === false ? 0.5 : 0.0;
  return +(cpTerm + movePenalty + signPenalty + oracleSplitPenalty).toFixed(4);
}

function sigmaBucket(score) {
  if (score >= 4.5) return 'severe';
  if (score >= 3.0) return 'bad';
  if (score >= 1.5) return 'watch';
  return 'ok';
}

function rate(numerator, denominator) {
  return denominator ? numerator / denominator : null;
}

function summarizeRows(rows) {
  const summary = {
    count: rows.length,
    reference_vs_sf_move_match_rate: null,
    reference_vs_sf_sign_match_rate: null,
    lozza_vs_sf_move_match_rate: null,
    reference_vs_lozza_move_match_rate: null,
    mean_abs_score_delta_cp: null,
    median_abs_score_delta_cp: null,
    p90_abs_score_delta_cp: null,
    oracle_split_rate: null,
    sigma_bad_rate: null,
  };
  if (!rows.length) return summary;

  let refMoveRows = 0;
  let refMoveMatch = 0;
  let refSignRows = 0;
  let refSignMatchCount = 0;
  let lozzaRows = 0;
  let lozzaMatch = 0;
  let refLozzaRows = 0;
  let refLozzaMatch = 0;
  let oracleSplit = 0;
  let sigmaBad = 0;
  const deltas = [];

  for (const row of rows) {
    if (row.sigma_ref_vs_sf_move_match !== null && row.sigma_ref_vs_sf_move_match !== undefined) {
      refMoveRows += 1;
      if (row.sigma_ref_vs_sf_move_match) refMoveMatch += 1;
    }
    if (row.sigma_ref_vs_sf_sign_match !== null && row.sigma_ref_vs_sf_sign_match !== undefined) {
      refSignRows += 1;
      if (row.sigma_ref_vs_sf_sign_match) refSignMatchCount += 1;
    }
    if (row.sigma_lozza_vs_sf_move_match !== null && row.sigma_lozza_vs_sf_move_match !== undefined) {
      lozzaRows += 1;
      if (row.sigma_lozza_vs_sf_move_match) lozzaMatch += 1;
      if (row.sigma_lozza_vs_sf_move_match === false) oracleSplit += 1;
    }
    if (row.sigma_ref_vs_lozza_move_match !== null && row.sigma_ref_vs_lozza_move_match !== undefined) {
      refLozzaRows += 1;
      if (row.sigma_ref_vs_lozza_move_match) refLozzaMatch += 1;
    }
    if (row.sigma_ref_vs_sf_abs_score_delta_cp !== null && row.sigma_ref_vs_sf_abs_score_delta_cp !== undefined) {
      deltas.push(row.sigma_ref_vs_sf_abs_score_delta_cp);
    }
    if (row.sigma_bucket === 'bad' || row.sigma_bucket === 'severe') sigmaBad += 1;
  }

  deltas.sort((a, b) => a - b);
  summary.reference_vs_sf_move_match_rate = rate(refMoveMatch, refMoveRows);
  summary.reference_vs_sf_sign_match_rate = rate(refSignMatchCount, refSignRows);
  summary.lozza_vs_sf_move_match_rate = rate(lozzaMatch, lozzaRows);
  summary.reference_vs_lozza_move_match_rate = rate(refLozzaMatch, refLozzaRows);
  summary.mean_abs_score_delta_cp = mean(deltas);
  summary.median_abs_score_delta_cp = median(deltas);
  summary.p90_abs_score_delta_cp = percentile(deltas, 0.90);
  summary.oracle_split_rate = rate(oracleSplit, lozzaRows);
  summary.sigma_bad_rate = rate(sigmaBad, rows.length);
  return summary;
}

function summarizeSlices(rows, key) {
  const buckets = new Map();
  for (const row of rows) {
    const value = row[key];
    if (value === null || value === undefined || value === '') continue;
    if (!buckets.has(value)) buckets.set(value, []);
    buckets.get(value).push(row);
  }
  const out = {};
  for (const [value, bucketRows] of buckets.entries()) {
    out[value] = summarizeRows(bucketRows);
  }
  return out;
}

function runLozza(fen, timeoutMs) {
  return new Promise((resolve) => {
    execFile(
      process.execPath,
      [LOZZA_REFERENCE, fen],
      {
        cwd: __dirname,
        timeout: timeoutMs + 4000,
        maxBuffer: 2 * 1024 * 1024,
      },
      (err, stdout) => {
        if (err) {
          resolve(null);
          return;
        }
        try {
          const parsed = JSON.parse(String(stdout || '{}'));
          resolve(parsed && parsed.bestmove ? parsed : null);
        } catch {
          resolve(null);
        }
      }
    );
  });
}

async function main() {
  const args = parseArgs(process.argv);
  const rows = loadJsonl(args.input, args.limit);
  mkdirSync(dirname(args.output), { recursive: true });
  mkdirSync(dirname(args.report), { recursive: true });

  const sf = new Stockfish({ depth: args.depth });
  const outputLines = [];
  const enrichedRows = [];

  let refMoveRows = 0;
  let refScoreRows = 0;
  let refSfMoveMatches = 0;
  let refSfSignRows = 0;
  let refSfSignMatches = 0;
  let lozzaSfMoveMatches = 0;
  let lozzaSfSignRows = 0;
  let lozzaSfSignMatches = 0;
  let refLozzaMoveMatches = 0;
  let lozzaRows = 0;
  const absScoreDeltas = [];
  const scoreBucketCounts = {};
  const sigmaBucketCounts = {};
  const highDisagreement = [];

  try {
    for (let i = 0; i < rows.length; i++) {
      const row = rows[i];
      const fen = normalizeFen(row.fen);
      const refMove = chooseField(row, args.moveKey, ['move', 'engine_bestmove', 'bestmove', 'sf_bestmove']);
      const refScore = chooseField(row, args.scoreKey, ['engine_score_cp', 'score_cp', 'sf_score_cp']);
      const refScoreCp = toIntOrNull(refScore.value);

      const sfLabel = await sf.label(fen, args.depth);
      let lozzaLabel = null;
      if (args.withLozza) {
        lozzaLabel = await runLozza(fen, args.lozzaTimeoutMs);
        lozzaRows += 1;
      }

      const refVsSfMoveMatch = moveMatch(refMove.value, sfLabel.bestmove);
      const refVsSfSignMatch = signMatch(refScoreCp, sfLabel.score_cp);
      const lozzaVsSfMoveMatch = moveMatch(lozzaLabel?.bestmove, sfLabel.bestmove);
      const lozzaVsSfSignMatch = signMatch(lozzaLabel?.score_cp ?? null, sfLabel.score_cp);
      const refVsLozzaMoveMatch = moveMatch(refMove.value, lozzaLabel?.bestmove ?? null);
      const signedScoreDelta = refScoreCp === null ? null : refScoreCp - sfLabel.score_cp;
      const absScoreDelta = signedScoreDelta === null ? null : Math.abs(signedScoreDelta);
      const deltaBucket = scoreBucket(absScoreDelta);
      const sigma = sigmaScore({
        absScoreDelta,
        refVsSfMoveMatch,
        refVsSfSignMatch,
        lozzaVsSfMoveMatch,
      });
      const sigmaClass = sigmaBucket(sigma);

      if (refMove.value) {
        refMoveRows += 1;
        if (refVsSfMoveMatch) refSfMoveMatches += 1;
      }
      if (refVsSfSignMatch !== null) {
        refSfSignRows += 1;
        if (refVsSfSignMatch) refSfSignMatches += 1;
      }
      if (lozzaLabel?.bestmove) {
        if (lozzaVsSfMoveMatch) lozzaSfMoveMatches += 1;
      }
      if (lozzaVsSfSignMatch !== null) {
        lozzaSfSignRows += 1;
        if (lozzaVsSfSignMatch) lozzaSfSignMatches += 1;
      }
      if (refMove.value && lozzaLabel?.bestmove && refVsLozzaMoveMatch) {
        refLozzaMoveMatches += 1;
      }
      if (refScoreCp !== null) {
        refScoreRows += 1;
        absScoreDeltas.push(absScoreDelta);
        scoreBucketCounts[deltaBucket] = (scoreBucketCounts[deltaBucket] || 0) + 1;
      }
      sigmaBucketCounts[sigmaClass] = (sigmaBucketCounts[sigmaClass] || 0) + 1;

      if (
        absScoreDelta !== null &&
        (absScoreDelta > 200 || refVsSfMoveMatch === false || (lozzaLabel?.bestmove && lozzaVsSfMoveMatch === false))
      ) {
        highDisagreement.push({
          fen,
          ref_move: refMove.value,
          oracle_sf_bestmove: sfLabel.bestmove,
          oracle_lozza_bestmove: lozzaLabel?.bestmove ?? null,
          ref_score_cp: refScoreCp,
          oracle_sf_score_cp: sfLabel.score_cp,
          oracle_lozza_score_cp: lozzaLabel?.score_cp ?? null,
          sigma_ref_vs_sf_abs_score_delta_cp: absScoreDelta,
          sigma_ref_vs_sf_move_match: refVsSfMoveMatch,
          sigma_lozza_vs_sf_move_match: lozzaVsSfMoveMatch,
          sigma_score: sigma,
          sigma_bucket: sigmaClass,
        });
      }

      const enriched = {
        ...row,
        fen,
        oracle_sf_bestmove: sfLabel.bestmove,
        oracle_sf_score_cp: sfLabel.score_cp,
        oracle_sf_depth: sfLabel.depth,
        sigma_ref_move_key: refMove.key,
        sigma_ref_score_key: refScore.key,
        sigma_ref_vs_sf_move_match: refVsSfMoveMatch,
        sigma_ref_vs_sf_sign_match: refVsSfSignMatch,
        sigma_ref_vs_sf_signed_score_delta_cp: signedScoreDelta,
        sigma_ref_vs_sf_abs_score_delta_cp: absScoreDelta,
        sigma_ref_vs_sf_score_bucket: deltaBucket,
        sigma_score: sigma,
        sigma_bucket: sigmaClass,
      };
      if (args.withPv) enriched.oracle_sf_pv = sfLabel.pv;
      if (lozzaLabel?.bestmove) {
        enriched.oracle_lozza_bestmove = lozzaLabel.bestmove;
        enriched.oracle_lozza_score_cp = lozzaLabel.score_cp;
        enriched.oracle_lozza_depth = lozzaLabel.depth;
        if (args.withPv) enriched.oracle_lozza_pv = lozzaLabel.pv;
        enriched.sigma_lozza_vs_sf_move_match = lozzaVsSfMoveMatch;
        enriched.sigma_lozza_vs_sf_sign_match = lozzaVsSfSignMatch;
        enriched.sigma_ref_vs_lozza_move_match = refVsLozzaMoveMatch;
      }
      outputLines.push(JSON.stringify(enriched, null, 0));
      enrichedRows.push(enriched);
      if ((i + 1) % 10 === 0 || i === rows.length - 1) {
        process.stderr.write(`\r[oracle_enrich] ${i + 1}/${rows.length}`);
      }
    }
    process.stderr.write('\n');
  } finally {
    sf.close();
  }

  writeFileSync(args.output, outputLines.join('\n') + (outputLines.length ? '\n' : ''), 'utf8');

  absScoreDeltas.sort((a, b) => a - b);
  highDisagreement.sort((a, b) => (b.sigma_ref_vs_sf_abs_score_delta_cp || 0) - (a.sigma_ref_vs_sf_abs_score_delta_cp || 0));

  const report = {
    input: args.input,
    output: args.output,
    rows: rows.length,
    depth: args.depth,
    with_lozza: args.withLozza,
    move_key: args.moveKey || 'auto',
    score_key: args.scoreKey || 'auto',
    reference_move_rows: refMoveRows,
    reference_score_rows: refScoreRows,
    lozza_rows: lozzaRows,
    reference_vs_sf_move_match_rate: rate(refSfMoveMatches, refMoveRows),
    reference_vs_sf_sign_match_rate: rate(refSfSignMatches, refSfSignRows),
    lozza_vs_sf_move_match_rate: rate(lozzaSfMoveMatches, lozzaRows),
    lozza_vs_sf_sign_match_rate: rate(lozzaSfSignMatches, lozzaSfSignRows),
    reference_vs_lozza_move_match_rate: rate(refLozzaMoveMatches, Math.min(refMoveRows, lozzaRows)),
    abs_score_delta_cp: {
      mean: mean(absScoreDeltas),
      median: median(absScoreDeltas),
      p90: percentile(absScoreDeltas, 0.90),
      max: absScoreDeltas.length ? absScoreDeltas[absScoreDeltas.length - 1] : null,
    },
    score_bucket_counts: scoreBucketCounts,
    sigma_bucket_counts: sigmaBucketCounts,
    oracle_split_rate: rate(
      enrichedRows.filter((row) => row.sigma_lozza_vs_sf_move_match === false).length,
      lozzaRows
    ),
    sigma_bad_rate: rate(
      enrichedRows.filter((row) => row.sigma_bucket === 'bad' || row.sigma_bucket === 'severe').length,
      enrichedRows.length
    ),
    by_source: summarizeSlices(enrichedRows, 'source'),
    by_phase_bucket: summarizeSlices(enrichedRows, 'phase_bucket'),
    by_pressure_band: summarizeSlices(enrichedRows, 'pressure_band'),
    top_disagreements: compactExamples(highDisagreement, args.topDisagreements),
  };
  writeFileSync(args.report, JSON.stringify(report, null, 2) + '\n', 'utf8');
  console.log(JSON.stringify(report, null, 2));
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
