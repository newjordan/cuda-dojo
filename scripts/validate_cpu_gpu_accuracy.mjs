#!/usr/bin/env node
import { existsSync, mkdirSync, readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { basename, dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { execFileSync } from 'node:child_process';
import { compileAgent, getMoveFromFn, resetAgentCache } from '../dojo_runtime.js';
import { START_FEN, generateLegalMoves, parseFen } from '../dojo_chess.js';

const __dirname = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = resolve(__dirname, '..');
const REPORT_DIR = join(REPO_ROOT, 'runtime', 'reports');
const GPU_FORGE_BIN = join(REPO_ROOT, 'cuda', 'gpu_forge');
const CPU_HARNESS_CONDITION = 'standalone_fresh_process';

function parseArgs(argv) {
  const options = {
    fighter: '',
    fighterBlob: '',
    samples: 24,
    configs: 16,
    sims: 8,
    minAccuracy: 0.65,
    minCoverage: 0.8,
    timeoutMs: 300000,
    slug: '',
    gpuDepth: null,
    gpuFullQeval: false,
    gpuFilterLegal: false,
    gpuTraincarEval: false,
    gpuTraincarBook: false,
    gpuSerialRoot: false,
    gpuRootOrder: false,
    gpuFamilyDispatch: false,
    gpuTimeoutRootProxy: false,
  };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--fighter') options.fighter = argv[++i] || '';
    else if (arg === '--fighter-blob') options.fighterBlob = argv[++i] || '';
    else if (arg === '--samples') options.samples = Math.max(4, Number(argv[++i] || options.samples));
    else if (arg === '--configs') options.configs = Math.max(4, Number(argv[++i] || options.configs));
    else if (arg === '--sims') options.sims = Math.max(4, Number(argv[++i] || options.sims));
    else if (arg === '--min-accuracy') options.minAccuracy = Number(argv[++i] || options.minAccuracy);
    else if (arg === '--min-coverage') options.minCoverage = Number(argv[++i] || options.minCoverage);
    else if (arg === '--timeout-ms') options.timeoutMs = Math.max(0, Number(argv[++i] || options.timeoutMs));
    else if (arg === '--no-timeout') options.timeoutMs = 0;
    else if (arg === '--slug') options.slug = argv[++i] || '';
    else if (arg === '--gpu-depth') options.gpuDepth = Math.max(1, Math.min(12, Number(argv[++i] || 0)));
    else if (arg === '--gpu-full-qeval') options.gpuFullQeval = true;
    else if (arg === '--gpu-filter-legal') options.gpuFilterLegal = true;
    else if (arg === '--gpu-traincar-eval') options.gpuTraincarEval = true;
    else if (arg === '--gpu-traincar-book') options.gpuTraincarBook = true;
    else if (arg === '--gpu-serial-root') options.gpuSerialRoot = true;
    else if (arg === '--gpu-root-order') options.gpuRootOrder = true;
    else if (arg === '--gpu-family-dispatch') options.gpuFamilyDispatch = true;
    else if (arg === '--gpu-timeout-root-proxy') options.gpuTimeoutRootProxy = true;
  }

  if (!options.fighter) {
    throw new Error('Usage: node scripts/validate_cpu_gpu_accuracy.mjs --fighter variants/<name>.js [--fighter-blob path] [--samples N] [--configs N] [--sims N] [--min-accuracy R] [--min-coverage R] [--slug id]');
  }
  return options;
}

function normalizeFen(fen) {
  return String(fen || '').trim().replace(/\s+/g, ' ');
}

function collectBatchFens(limit = 12) {
  const batchPath = [
    join(REPO_ROOT, 'fighter_accuracy', 'corpus', 'cuda_dojo_batch_latest.json'),
    join(REPO_ROOT, 'runtime', 'reports', 'cuda_dojo_batch_latest.json'),
  ].find((path) => existsSync(path));
  if (!batchPath) return [];
  const batch = JSON.parse(readFileSync(batchPath, 'utf8'));
  const seen = new Set();
  const out = [];
  const add = (fen) => {
    const n = normalizeFen(fen);
    if (!n.includes('/')) return;
    if (seen.has(n)) return;
    seen.add(n);
    out.push(n);
  };
  for (const item of batch.trainingCases || []) {
    add(item.fen);
    if (out.length >= limit) break;
  }
  if (out.length < limit) {
    for (const item of batch.disagreements || []) {
      add(item.fen);
      if (out.length >= limit) break;
    }
  }
  return out;
}

function collectLozzaBenchFens(limit = 12) {
  const path = join(REPO_ROOT, 'trainers', 'lozza', 'lozza_raw.js');
  if (!existsSync(path)) return [];
  const raw = readFileSync(path, 'utf8');
  const regex = /\['fen\s+([^']+?)'\s*,/g;
  const out = [];
  const seen = new Set();
  let match;
  while ((match = regex.exec(raw))) {
    const fen = normalizeFen(match[1]);
    if (!fen.includes('/')) continue;
    if (seen.has(fen)) continue;
    seen.add(fen);
    out.push(fen);
    if (out.length >= limit) break;
  }
  return out;
}

function buildCorpus(maxSamples) {
  const seen = new Set();
  const out = [];
  const add = (fen) => {
    const n = normalizeFen(fen);
    if (!n || seen.has(n)) return;
    seen.add(n);
    out.push(n);
  };
  add(START_FEN);
  for (const fen of collectBatchFens()) add(fen);
  for (const fen of collectLozzaBenchFens()) add(fen);
  return out.slice(0, maxSamples);
}

function deriveBlobPath(fighterPath, explicitBlob) {
  if (explicitBlob) return explicitBlob;
  const candidates = [];
  if (fighterPath.endsWith('.src.js')) {
    candidates.push(fighterPath.replace(/\.src\.js$/, '.cuda_fighter_blob.json'));
  } else {
    candidates.push(fighterPath.replace(/\.js$/, '.cuda_fighter_blob.json'));
    candidates.push(fighterPath.replace(/\.js$/, '.src.cuda_fighter_blob.json'));
    candidates.push(fighterPath.replace(/\.js$/, '.src.js').replace(/\.src\.js$/, '.cuda_fighter_blob.json'));
  }

  const dir = dirname(fighterPath);
  const base = basename(fighterPath).replace(/\.src\.js$/, '').replace(/\.js$/, '');
  try {
    const siblingMatches = readdirSync(dir)
      .filter((name) =>
        name.startsWith(`${base}_`) &&
        name.endsWith('.cuda_fighter_blob.json'),
      )
      .sort();
    for (const name of siblingMatches) candidates.push(join(dir, name));
  } catch {
    // ignore directory scan failures
  }

  for (const candidate of candidates) {
    if (candidate && existsSync(candidate)) return candidate;
  }
  return candidates[0] || '';
}

function buildCpuInputLines(fighterPath, fens) {
  const usable = [];
  const skipped = [];
  for (const fen of fens) {
    resetAgentCache(fighterPath);
    const fn = compileAgent(fighterPath, {
      forceReload: true,
      deterministicClock: true,
      clockStartMs: 1700000000000,
      clockStepMs: 25,
    });
    if (!fn) throw new Error(`Could not compile fighter: ${fighterPath}`);

    const legal = generateLegalMoves(parseFen(fen));
    if (!legal.length) {
      skipped.push({ fen, reason: 'no_legal_moves' });
      continue;
    }
    const { move } = getMoveFromFn(fn, fen);
    if (!move || move === '0000' || move === '__FAIL__') {
      skipped.push({ fen, reason: 'cpu_invalid_move', move: move || '' });
      continue;
    }
    if (!legal.includes(move)) {
      skipped.push({ fen, reason: 'cpu_illegal_move', move });
      continue;
    }
    usable.push({ fen, move, legalCsv: legal.join(',') });
  }

  const input = usable.map((item) => `${item.fen}\t${item.move}\t${item.legalCsv}`).join('\n') + (usable.length ? '\n' : '');
  return { usable, skipped, input };
}

function runGpuForge(configs, sims, blobPath, input, timeoutMs, gpuDepth, gpuFullQeval, gpuFilterLegal, gpuTraincarEval, gpuTraincarBook, gpuSerialRoot, gpuRootOrder, gpuFamilyDispatch, gpuTimeoutRootProxy) {
  if (!existsSync(GPU_FORGE_BIN)) throw new Error(`Missing gpu_forge binary: ${GPU_FORGE_BIN}`);
  if (!existsSync(blobPath)) throw new Error(`Missing fighter blob: ${blobPath}`);
  const forgeArgs = [String(configs), String(sims), '--fighter-blob', blobPath];
  if (gpuDepth != null) forgeArgs.push('--depth', String(gpuDepth));
  if (gpuFullQeval) forgeArgs.push('--full-qeval');
  if (gpuFilterLegal) forgeArgs.push('--filter-legal');
  if (gpuTraincarEval) forgeArgs.push('--traincar-eval');
  if (gpuTraincarBook) forgeArgs.push('--traincar-book');
  if (gpuSerialRoot) forgeArgs.push('--serial-root');
  if (gpuRootOrder) forgeArgs.push('--root-order');
  if (gpuFamilyDispatch) forgeArgs.push('--family-dispatch');
  if (gpuTimeoutRootProxy) forgeArgs.push('--timeout-root-proxy');
  const command = timeoutMs > 0
    ? ['timeout', '--kill-after=10s', `${Math.ceil(timeoutMs / 1000)}s`, GPU_FORGE_BIN, ...forgeArgs]
    : [GPU_FORGE_BIN, ...forgeArgs];
  const raw = execFileSync(
    command[0],
    command.slice(1),
    { cwd: REPO_ROOT, input, encoding: 'utf8', maxBuffer: 32 * 1024 * 1024 },
  );
  return JSON.parse(raw);
}

function buildMarkdown(report) {
  const lines = [];
  lines.push('# CPU-GPU Accuracy');
  lines.push('');
  lines.push(`Generated: ${report.generatedAt}`);
  lines.push(`Fighter: ${report.fighter}`);
  lines.push(`Blob: ${report.fighterBlob}`);
  lines.push(`Condition: ${report.condition.label}`);
  lines.push('');
  lines.push(`Verdict: **${report.ok ? 'PASS' : 'FAIL'}**`);
  lines.push('');
  lines.push(`- Comparable positions: ${report.gpuSummary.comparablePositions}`);
  lines.push(`- Agreements: ${report.gpuSummary.agreements}`);
  lines.push(`- Disagreements: ${report.gpuSummary.disagreements}`);
  lines.push(`- Agreement rate: ${(report.gpuSummary.agreementRate * 100).toFixed(1)}%`);
  lines.push(`- Coverage: ${(report.gpuSummary.coverage * 100).toFixed(1)}%`);
  if (report.failures.length) lines.push(`- Failures: ${report.failures.join(', ')}`);
  if (report.warnings.length) lines.push(`- Warnings: ${report.warnings.join(', ')}`);
  return lines.join('\n') + '\n';
}

function main() {
  const options = parseArgs(process.argv.slice(2));
  const fighterPath = resolve(REPO_ROOT, options.fighter);
  const fighterBlob = resolve(REPO_ROOT, deriveBlobPath(fighterPath, options.fighterBlob));
  const slug = options.slug || basename(fighterPath).replace(/\.js$/, '');

  const corpus = buildCorpus(options.samples);
  const { usable, skipped, input } = buildCpuInputLines(fighterPath, corpus);
  const gpu = runGpuForge(options.configs, options.sims, fighterBlob, input, options.timeoutMs, options.gpuDepth, options.gpuFullQeval, options.gpuFilterLegal, options.gpuTraincarEval, options.gpuTraincarBook, options.gpuSerialRoot, options.gpuRootOrder, options.gpuFamilyDispatch, options.gpuTimeoutRootProxy);
  const summary = gpu.summary || {};
  const gpuCondition = summary.timeoutRootProxy ? 'proxy' : (summary.traincarBook ? 'strict_traincar_book' : 'strict');
  const conditionLabel = `${CPU_HARNESS_CONDITION}+${gpuCondition}`;

  const comparable = Number(summary.comparablePositions || 0);
  const agreements = Number(summary.agreements || 0);
  const disagreements = Number(summary.disagreements || summary.positions || 0);
  const agreementRate = comparable > 0 ? agreements / comparable : 0;
  const coverage = usable.length > 0 ? comparable / usable.length : 0;

  const failures = [];
  const warnings = [];
  if (usable.length === 0) failures.push('no_usable_cpu_positions');
  if (comparable === 0) failures.push('no_comparable_positions');
  if (coverage < options.minCoverage) failures.push(`coverage_below_${options.minCoverage}`);
  if (agreementRate < options.minAccuracy) failures.push(`agreement_below_${options.minAccuracy}`);
  if (skipped.length > 0) warnings.push(`cpu_skipped_${skipped.length}`);
  if (Number(summary.skippedEngineMoveMissing || 0) > 0) warnings.push('gpu_missing_engine_move_for_some_positions');
  if (summary.timeoutRootProxy) warnings.push('timeout_root_proxy_condition_not_strict_parity');
  if (options.gpuTraincarBook && !summary.traincarBook) warnings.push('traincar_book_requested_but_not_active');

  const report = {
    ok: failures.length === 0,
    generatedAt: new Date().toISOString(),
    fighter: fighterPath,
    fighterBlob,
    options: {
      samples: options.samples,
      configs: options.configs,
      sims: options.sims,
      minAccuracy: options.minAccuracy,
      minCoverage: options.minCoverage,
      timeoutMs: options.timeoutMs,
      gpuDepth: options.gpuDepth,
      gpuFullQeval: options.gpuFullQeval,
      gpuFilterLegal: options.gpuFilterLegal,
      gpuTraincarEval: options.gpuTraincarEval,
      gpuTraincarBook: options.gpuTraincarBook,
      gpuSerialRoot: options.gpuSerialRoot,
      gpuRootOrder: options.gpuRootOrder,
      gpuFamilyDispatch: options.gpuFamilyDispatch,
      gpuTimeoutRootProxy: options.gpuTimeoutRootProxy,
    },
    corpus: {
      size: corpus.length,
      usableCpuPositions: usable.length,
      skippedCpuPositions: skipped.length,
    },
    condition: {
      label: conditionLabel,
      cpuHarness: CPU_HARNESS_CONDITION,
      gpu: gpuCondition,
    },
    gpuSummary: {
      inputLines: Number(summary.inputLines || 0),
      parsedLines: Number(summary.parsedLines || 0),
      comparablePositions: comparable,
      agreements,
      disagreements,
      agreementRate,
      coverage,
      fixRate: Number(summary.fixRate || 0),
      searchDepth: Number(summary.searchDepth || 0),
      fighterFamily: summary.fighterFamily || 'unknown',
      familyDispatch: Boolean(summary.familyDispatch),
      timeoutRootProxy: Boolean(summary.timeoutRootProxy),
      traincarBook: Boolean(summary.traincarBook),
      traincarBookEntries: Number(summary.traincarBookEntries || 0),
      traincarBookOverrides: Number(summary.traincarBookOverrides || 0),
      traincarBookMisses: Number(summary.traincarBookMisses || 0),
      skippedNoMoves: Number(summary.skippedNoMoves || 0),
      skippedEngineMoveMissing: Number(summary.skippedEngineMoveMissing || 0),
    },
    sampleDisagreements: Array.isArray(gpu.positions) ? gpu.positions.slice(0, 10) : [],
    skippedCpuSample: skipped.slice(0, 10),
    failures,
    warnings,
  };

  mkdirSync(REPORT_DIR, { recursive: true });
  const jsonPath = join(REPORT_DIR, `${slug}.cpu_gpu_accuracy.json`);
  const mdPath = join(REPORT_DIR, `${slug}.cpu_gpu_accuracy.md`);
  const latestJson = join(REPORT_DIR, 'cpu_gpu_accuracy_latest.json');
  const latestMd = join(REPORT_DIR, 'cpu_gpu_accuracy_latest.md');
  const jsonText = JSON.stringify(report, null, 2) + '\n';
  writeFileSync(jsonPath, jsonText);
  writeFileSync(mdPath, buildMarkdown(report));
  writeFileSync(latestJson, jsonText);
  writeFileSync(latestMd, buildMarkdown(report));

  process.stdout.write(JSON.stringify({
    ok: report.ok,
    fighter: report.fighter,
    fighterBlob: report.fighterBlob,
    comparablePositions: report.gpuSummary.comparablePositions,
    agreementRate: report.gpuSummary.agreementRate,
    coverage: report.gpuSummary.coverage,
    fighterFamily: report.gpuSummary.fighterFamily,
    familyDispatch: report.gpuSummary.familyDispatch,
    timeoutRootProxy: report.gpuSummary.timeoutRootProxy,
    traincarBook: report.gpuSummary.traincarBook,
    traincarBookOverrides: report.gpuSummary.traincarBookOverrides,
    condition: report.condition,
    failures: report.failures,
    warnings: report.warnings,
    jsonPath,
    mdPath,
  }, null, 2) + '\n');

  if (!report.ok) process.exitCode = 1;
}

main();
