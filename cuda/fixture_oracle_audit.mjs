#!/usr/bin/env node
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { basename, dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { Stockfish } from '../gpu_spine/reference.mjs';
import { Lozza } from './lozza_reference.mjs';

const __dirname = dirname(fileURLToPath(import.meta.url));
const DEFAULT_OUTPUT = resolve(__dirname, 'artifacts', 'fixture_oracle_audit', 'report.json');

function usage() {
  console.error(
    'usage: node fixture_oracle_audit.mjs --fixture <json> [--fixture <json> ...] [--depth N] [--output <json>] [--without-lozza]'
  );
}

function parseArgs(argv) {
  const args = {
    fixtures: [],
    depth: 12,
    output: DEFAULT_OUTPUT,
    withLozza: true,
  };
  for (let i = 2; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--fixture') args.fixtures.push(resolve(argv[++i]));
    else if (arg === '--depth') args.depth = parseInt(argv[++i], 10) || 12;
    else if (arg === '--output') args.output = resolve(argv[++i]);
    else if (arg === '--without-lozza') args.withLozza = false;
    else if (arg === '--help' || arg === '-h') {
      usage();
      process.exit(0);
    } else {
      console.error(`unknown arg: ${arg}`);
      usage();
      process.exit(1);
    }
  }
  if (!args.fixtures.length) {
    args.fixtures.push(resolve(__dirname, 'queen_pressure_cases.json'));
  }
  return args;
}

function loadCases(path) {
  const parsed = JSON.parse(readFileSync(path, 'utf8'));
  return Array.isArray(parsed) ? parsed : parsed.cases || parsed.rows || [];
}

function sameMove(a, b) {
  if (!a || !b) return false;
  return String(a).toLowerCase() === String(b).toLowerCase();
}

function classify(expected, sfMove, lozzaMove) {
  const expectedSf = sameMove(expected, sfMove);
  const expectedLozza = sameMove(expected, lozzaMove);
  const sfLozzaAgree = sameMove(sfMove, lozzaMove);

  if (expectedSf && expectedLozza) return 'confirmed_both';
  if (expectedSf && !expectedLozza) return sfLozzaAgree ? 'confirmed_stockfish' : 'split_stockfish';
  if (!expectedSf && expectedLozza) return sfLozzaAgree ? 'confirmed_lozza' : 'split_lozza';
  if (sfLozzaAgree) return 'stale_fixture';
  return 'ambiguous_split';
}

async function main() {
  const args = parseArgs(process.argv);
  const sf = new Stockfish({ depth: args.depth });
  const lozza = args.withLozza ? new Lozza({ movetimeMs: 800 }) : null;
  const rows = [];
  const byFixture = {};
  const classificationCounts = {};

  try {
    for (const fixture of args.fixtures) {
      const cases = loadCases(fixture);
      byFixture[basename(fixture)] = {
        cases: cases.length,
        classifications: {},
      };
      for (const row of cases) {
        const sfLabel = await sf.label(row.fen, args.depth);
        const lozzaLabel = lozza ? await lozza.label(row.fen) : { bestmove: null, score_cp: null };
        const classification = classify(
          row.expected_bestmove,
          sfLabel.bestmove,
          lozzaLabel.bestmove
        );

        const out = {
          fixture: basename(fixture),
          id: row.id,
          fen: row.fen,
          expected_bestmove: row.expected_bestmove,
          oracle_sf_bestmove: sfLabel.bestmove,
          oracle_lozza_bestmove: lozzaLabel.bestmove,
          oracle_sf_score_cp: sfLabel.score_cp,
          oracle_lozza_score_cp: lozzaLabel.score_cp,
          classification,
          oracle_split: lozza ? !sameMove(sfLabel.bestmove, lozzaLabel.bestmove) : null,
        };
        rows.push(out);
        classificationCounts[classification] = (classificationCounts[classification] || 0) + 1;
        byFixture[basename(fixture)].classifications[classification] =
          (byFixture[basename(fixture)].classifications[classification] || 0) + 1;
      }
    }
  } finally {
    sf.close();
    if (lozza) lozza.close();
  }

  const report = {
    depth: args.depth,
    fixtures: args.fixtures,
    by_fixture: byFixture,
    classification_counts: classificationCounts,
    stale_or_ambiguous: rows.filter((row) =>
      row.classification === 'stale_fixture' || row.classification === 'ambiguous_split'
    ),
    rows,
  };

  mkdirSync(dirname(args.output), { recursive: true });
  writeFileSync(args.output, JSON.stringify(report, null, 2) + '\n');
  console.log(JSON.stringify(report, null, 2));
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
