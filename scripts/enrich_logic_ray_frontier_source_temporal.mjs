#!/usr/bin/env node
import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';

const SCRIPT_VERSION = 'dojo.logic_ray_frontier.source_temporal_enrichment.v1';
const REPO_ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');

function usage() {
  return `Usage: node scripts/enrich_logic_ray_frontier_source_temporal.mjs --input <frontier.json> [--out <frontier.enriched.json>]

Attach recorded source-order temporal provenance to logicRayFrontier rows.
This is host-side artifact enrichment only: it reads existing corpus/book files
and does not generate moves, call a chess engine, or validate legal transitions.
`;
}

function parseArgs(argv) {
  const args = {
    input: null,
    out: null,
    conditionSource: '/home/frosty40/cuda_dojo/LOGIC_RAY_FRONTIER_RUNTIME_RECEIPT_2026-05-06.md',
  };
  for (let i = 0; i < argv.length; i += 1) {
    const token = argv[i];
    if (token === '--help' || token === '-h') {
      console.log(usage());
      process.exit(0);
    } else if (token === '--input' || token === '--in') {
      args.input = argv[++i];
    } else if (token === '--out') {
      args.out = argv[++i];
    } else if (token === '--condition-source') {
      args.conditionSource = argv[++i];
    } else {
      throw new Error(`unknown argument: ${token}\n${usage()}`);
    }
  }
  if (!args.input) throw new Error(`missing --input\n${usage()}`);
  return args;
}

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, 'utf8'));
}

function normalizeFen(fen) {
  return String(fen || '').trim().replace(/\s+/g, ' ');
}

function completeFen(fen) {
  const parts = normalizeFen(fen).split(/\s+/).filter(Boolean);
  if (parts.length === 2) return `${parts.join(' ')} - - 0 1`;
  if (parts.length === 3) return `${parts.join(' ')} - 0 1`;
  if (parts.length === 4) return `${parts.join(' ')} 0 1`;
  if (parts.length === 5) return `${parts.join(' ')} 1`;
  return parts.join(' ');
}

function sha256Text(text) {
  return crypto.createHash('sha256').update(String(text)).digest('hex');
}

function sha256File(filePath) {
  return sha256Text(fs.readFileSync(filePath));
}

function fenHash(fen) {
  return sha256Text(completeFen(fen));
}

function sideToMove(fen) {
  const parts = completeFen(fen).split(/\s+/);
  return parts[1] || null;
}

function defaultOutPath(inputPath) {
  const parsed = path.parse(inputPath);
  const stem = parsed.base.replace(/\.logic_ray_frontier\.json$/, '').replace(/\.json$/, '');
  return path.join(parsed.dir, `${stem}.source_temporal.logic_ray_frontier.json`);
}

function asRows(document) {
  if (Array.isArray(document)) return document;
  if (document && Array.isArray(document.rows)) return document.rows;
  if (document && document.schemaVersion === 'dojo.logic_ray_frontier.v1') return [document];
  throw new Error('input must be a logicRayFrontier row, row array, or bundle with rows[]');
}

function finalizeSequence(records, target, sourceSequenceId, plyBasis, confidence, evidenceClass, transitionVerified, note) {
  records.forEach((record, index) => {
    const enriched = {
      ...record,
      sourceSequenceId,
      sourcePly: index,
      sourcePlyBasis: plyBasis,
      previousFenHash: index > 0 ? fenHash(records[index - 1].fen) : null,
      nextFenHash: index + 1 < records.length ? fenHash(records[index + 1].fen) : null,
      temporalEvidenceClass: evidenceClass,
      sourceOrderConfidence: confidence,
      transitionVerified,
      note,
    };
    const key = completeFen(enriched.fen);
    if (!target.has(key)) target.set(key, enriched);
  });
}

function addBatchRecords(target) {
  const candidates = [
    path.join(REPO_ROOT, 'fighter_accuracy', 'corpus', 'cuda_dojo_batch_latest.json'),
    path.join(REPO_ROOT, 'runtime', 'reports', 'cuda_dojo_batch_latest.json'),
  ];
  const batchPath = candidates.find((candidate) => fs.existsSync(candidate));
  if (!batchPath) return;
  const batch = readJson(batchPath);
  const artifactSha = sha256File(batchPath);
  for (const section of ['trainingCases', 'disagreements']) {
    const rows = Array.isArray(batch[section]) ? batch[section] : [];
    const records = rows
      .map((item, index) => ({
        fen: completeFen(item?.fen),
        sourceFamily: `fighter_accuracy_batch_${section}`,
        sourceArtifact: batchPath,
        sourceArtifactSha256: artifactSha,
        sourceIndex: index,
      }))
      .filter((record) => record.fen.includes('/'));
    if (!records.length) continue;
    const sequenceId = `fighter_accuracy_batch:${artifactSha.slice(0, 16)}:${batch.timestamp || 'unknown_timestamp'}:${section}`;
    finalizeSequence(
      records,
      target,
      sequenceId,
      'recorded_json_array_order',
      'recorded_order_only',
      'recorded_source_sequence',
      false,
      'Batch order is recorded source order; chess transition legality is not revalidated by this artifact pass.',
    );
  }
}

function addLozzaRecords(target) {
  const lozzaPath = path.join(REPO_ROOT, 'trainers', 'lozza', 'lozza_raw.js');
  if (!fs.existsSync(lozzaPath)) return;
  const raw = fs.readFileSync(lozzaPath, 'utf8');
  const artifactSha = sha256File(lozzaPath);
  const regex = /\['fen\s+([^']+?)'\s*,/g;
  const records = [];
  let match;
  while ((match = regex.exec(raw))) {
    const fen = completeFen(match[1]);
    if (!fen.includes('/')) continue;
    records.push({
      fen,
      sourceFamily: 'lozza_bench_fen_list',
      sourceArtifact: lozzaPath,
      sourceArtifactSha256: artifactSha,
      sourceIndex: records.length,
    });
  }
  finalizeSequence(
    records,
    target,
    `lozza_bench:${artifactSha.slice(0, 16)}:regex_fen_order`,
    'recorded_regex_fen_order',
    'recorded_order_only',
    'recorded_source_order',
    false,
    'Lozza FEN order is recorded input order only, not explicit game trajectory.',
  );
}

function addGpuSpineRecords(target) {
  const bookPath = path.join(REPO_ROOT, 'gpu_spine', 'book.jsonl');
  if (!fs.existsSync(bookPath)) return;
  const artifactSha = sha256File(bookPath);
  const lines = fs.readFileSync(bookPath, 'utf8').split('\n');
  const segments = [];
  let current = [];
  let previousSide = null;
  let lineIndex = 0;
  for (const line of lines) {
    if (!line.trim()) continue;
    try {
      const row = JSON.parse(line);
      if (!row.fen || row.sf_bestmove === '(none)') continue;
      const fen = completeFen(row.fen);
      const side = sideToMove(fen);
      if (current.length && side && previousSide && side === previousSide) {
        segments.push(current);
        current = [];
      }
      current.push({
        fen,
        sourceFamily: 'gpu_spine_book_jsonl',
        sourceArtifact: bookPath,
        sourceArtifactSha256: artifactSha,
        sourceIndex: lineIndex,
      });
      previousSide = side;
      lineIndex += 1;
    } catch {
      lineIndex += 1;
    }
  }
  if (current.length) segments.push(current);
  segments.forEach((records, segmentIndex) => {
    finalizeSequence(
      records,
      target,
      `gpu_spine_book:${artifactSha.slice(0, 16)}:side_toggle_segment_${segmentIndex}`,
      'jsonl_line_order_side_toggle_segment_offset',
      'side_toggle_segment',
      'recorded_source_sequence',
      false,
      'GPU spine book order is recorded and segmented on side-to-move toggles; legal transition verification is intentionally not performed here.',
    );
  });
}

function addVariantBookRecords(target) {
  const variantDir = path.join(REPO_ROOT, 'variants');
  if (!fs.existsSync(variantDir)) return;
  const files = fs.readdirSync(variantDir)
    .filter((name) => name.endsWith('.js') || name.endsWith('.src.js'))
    .sort();
  for (const file of files) {
    const filePath = path.join(variantDir, file);
    const artifactSha = sha256File(filePath);
    const raw = fs.readFileSync(filePath, 'utf8');
    const regex = /\bB\(\s*["']([^"']+\/[^"']+)["']\s*,/g;
    const records = [];
    let match;
    while ((match = regex.exec(raw))) {
      const fen = completeFen(match[1]);
      if (!fen.includes('/')) continue;
      records.push({
        fen,
        sourceFamily: 'variant_book_entries',
        sourceArtifact: filePath,
        sourceArtifactSha256: artifactSha,
        sourceIndex: records.length,
      });
    }
    finalizeSequence(
      records,
      target,
      `variant_book:${artifactSha.slice(0, 16)}:${file}`,
      'variant_B_entry_order',
      'recorded_order_only',
      'recorded_source_order',
      false,
      'Variant B() entry order is a recorded book order, not explicit game trajectory.',
    );
  }
}

function buildTemporalIndex() {
  const target = new Map();
  addBatchRecords(target);
  addLozzaRecords(target);
  addGpuSpineRecords(target);
  addVariantBookRecords(target);
  return target;
}

function countBy(items, getter) {
  const counts = new Map();
  for (const item of items) {
    const key = getter(item);
    counts.set(String(key), (counts.get(String(key)) || 0) + 1);
  }
  return Object.fromEntries([...counts.entries()].sort((a, b) => a[0].localeCompare(b[0])));
}

function main() {
  const args = parseArgs(process.argv.slice(2));
  const inputPath = path.resolve(args.input);
  const outPath = path.resolve(args.out || defaultOutPath(inputPath));
  const input = readJson(inputPath);
  const rows = asRows(input);
  const temporalIndex = buildTemporalIndex();
  let matchedRows = 0;
  let unmatchedRows = 0;
  const enrichedRows = rows.map((row) => {
    const record = temporalIndex.get(completeFen(row.rootFen));
    if (!record) {
      unmatchedRows += 1;
      return row;
    }
    matchedRows += 1;
    const labels = new Set(Array.isArray(row.labels) ? row.labels : []);
    labels.add(`source_temporal.${record.sourceFamily}`);
    labels.add(`chrono_source.${record.temporalEvidenceClass}`);
    return {
      ...row,
      sourceTemporal: {
        sourceFamily: record.sourceFamily,
        sourceArtifact: record.sourceArtifact,
        sourceArtifactSha256: record.sourceArtifactSha256,
        sourceSequenceId: record.sourceSequenceId,
        sourceIndex: record.sourceIndex,
        sourcePly: record.sourcePly,
        sourcePlyBasis: record.sourcePlyBasis,
        previousFenHash: record.previousFenHash,
        nextFenHash: record.nextFenHash,
        temporalEvidenceClass: record.temporalEvidenceClass,
        sourceOrderConfidence: record.sourceOrderConfidence,
        transitionVerified: record.transitionVerified,
        note: record.note,
      },
      labels: [...labels],
    };
  });

  const output = Array.isArray(input)
    ? enrichedRows
    : {
      ...input,
      generatedAt: input.generatedAt || new Date().toISOString(),
      sourceTemporalEnrichment: {
        schemaVersion: SCRIPT_VERSION,
        generatedAt: new Date().toISOString(),
        condition: {
          source: path.resolve(args.conditionSource),
          runLabel: 'posthoc_source_temporal_enrichment_on_documented_frontier_artifact',
          changedFields: 'sourceTemporal metadata added from recorded corpus/book source order only',
        },
        inputPath,
        matchedRows,
        unmatchedRows,
        sourceFamilyCounts: countBy(enrichedRows.filter((row) => row.sourceTemporal), (row) => row.sourceTemporal.sourceFamily),
        temporalEvidenceClassCounts: countBy(enrichedRows.filter((row) => row.sourceTemporal), (row) => row.sourceTemporal.temporalEvidenceClass),
        transitionVerified: false,
      },
      rows: enrichedRows,
    };
  fs.mkdirSync(path.dirname(outPath), { recursive: true });
  fs.writeFileSync(outPath, `${JSON.stringify(output, null, 2)}\n`);
  console.log(JSON.stringify({
    ok: unmatchedRows === 0,
    output: outPath,
    rows: rows.length,
    matchedRows,
    unmatchedRows,
    sourceFamilyCounts: countBy(enrichedRows.filter((row) => row.sourceTemporal), (row) => row.sourceTemporal.sourceFamily),
  }, null, 2));
  process.exit(unmatchedRows === 0 ? 0 : 2);
}

try {
  main();
} catch (error) {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
}
