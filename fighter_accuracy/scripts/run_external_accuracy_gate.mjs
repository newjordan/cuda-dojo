#!/usr/bin/env node
import { createHash } from 'node:crypto';
import {
  copyFileSync,
  existsSync,
  mkdirSync,
  readFileSync,
  readdirSync,
  writeFileSync,
} from 'node:fs';
import { basename, dirname, join, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';

const __dirname = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = resolve(__dirname, '..', '..');

const FIGHTERS = [
  { name: 'Razor X', base: 'razor_x', fighter: 'variants/razor_x.js' },
  { name: 'Queensguard', base: 'queensguard', fighter: 'variants/queensguard.js' },
  { name: 'Firebird', base: 'firebird', fighter: 'variants/firebird.js' },
  { name: 'Fortress', base: 'fortress', fighter: 'variants/fortress.js' },
  { name: 'RazorBlade II', base: 'razorblade_ii', fighter: 'variants/razorblade_ii.js' },
];

function parseArgs(argv) {
  const options = {
    labDir: resolve(REPO_ROOT, '..', '.docker_labs', 'dojo_conversion_lab'),
    receiptDir: '',
    samples: 24,
    configs: 16,
    sims: 8,
    minAccuracy: 0.55,
    minCoverage: 0.75,
    timeoutMs: 300000,
    expectFail: false,
    skipRun: false,
  };

  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--lab-dir') options.labDir = resolve(argv[++i] || options.labDir);
    else if (arg === '--receipt-dir') options.receiptDir = resolve(REPO_ROOT, argv[++i] || '');
    else if (arg === '--samples') options.samples = Math.max(4, Number(argv[++i] || options.samples));
    else if (arg === '--configs') options.configs = Math.max(4, Number(argv[++i] || options.configs));
    else if (arg === '--sims') options.sims = Math.max(4, Number(argv[++i] || options.sims));
    else if (arg === '--min-accuracy') options.minAccuracy = Number(argv[++i] || options.minAccuracy);
    else if (arg === '--min-coverage') options.minCoverage = Number(argv[++i] || options.minCoverage);
    else if (arg === '--timeout-ms') options.timeoutMs = Math.max(0, Number(argv[++i] || options.timeoutMs));
    else if (arg === '--no-timeout') options.timeoutMs = 0;
    else if (arg === '--expect-fail') options.expectFail = true;
    else if (arg === '--skip-run') options.skipRun = true;
    else throw new Error(`Unknown argument: ${arg}`);
  }

  if (!options.receiptDir) {
    options.receiptDir = join(REPO_ROOT, 'fighter_accuracy', 'receipts', timestamp());
  }

  return options;
}

function timestamp() {
  return new Date().toISOString().replace(/[-:]/g, '').replace(/\.\d{3}Z$/, 'Z');
}

function hashFile(path) {
  const data = readFileSync(path);
  return createHash('sha256').update(data).digest('hex');
}

function maybeHash(path, labDir) {
  if (!existsSync(path)) {
    return {
      path,
      relativePath: relative(labDir, path),
      exists: false,
    };
  }
  return {
    path,
    relativePath: relative(labDir, path),
    exists: true,
    sha256: hashFile(path),
    bytes: readFileSync(path).byteLength,
  };
}

function runGit(labDir, args) {
  const result = spawnSync('git', ['-C', labDir, ...args], {
    cwd: REPO_ROOT,
    encoding: 'utf8',
  });
  return {
    status: result.status,
    stdout: String(result.stdout || '').trim(),
    stderr: String(result.stderr || '').trim(),
  };
}

function variantMatches(labDir, base) {
  const variantsDir = join(labDir, 'variants');
  if (!existsSync(variantsDir)) return [];
  return readdirSync(variantsDir)
    .filter((name) =>
      name === `${base}.js` ||
      name === `${base}.src.js` ||
      name.startsWith(`${base}_`) ||
      name.startsWith(`${base}.`) ||
      name.endsWith(`${base}.cuda_fighter_blob.json`))
    .filter((name) =>
      name.endsWith('.js') ||
      name.endsWith('.json') ||
      name.endsWith('.h'))
    .sort()
    .map((name) => join(variantsDir, name));
}

function collectConditionFiles(labDir) {
  const required = [
    'scripts/run_cpu_gpu_accuracy_matrix.mjs',
    'scripts/validate_cpu_gpu_accuracy.mjs',
    'cuda/gpu_forge.cu',
    'cuda/gpu_forge',
    'cuda/generated/dojo_active_fighter_legacy.h',
    'dojo_runtime.js',
    'dojo_chess.js',
    'trainers/lozza/lozza_raw.js',
    'runtime/reports/cuda_dojo_batch_latest.json',
  ].map((path) => join(labDir, path));

  const fighterFiles = FIGHTERS.flatMap((fighter) => [
    join(labDir, fighter.fighter),
    ...variantMatches(labDir, fighter.base),
  ]);

  return [...new Set([...required, ...fighterFiles])].sort();
}

function copyIfExists(source, destDir) {
  if (!source || !existsSync(source)) return null;
  const dest = join(destDir, basename(source));
  copyFileSync(source, dest);
  return dest;
}

function loadJsonIfExists(path) {
  if (!existsSync(path)) return null;
  return JSON.parse(readFileSync(path, 'utf8'));
}

function copyReports(labDir, receiptDir) {
  const reportDir = join(labDir, 'runtime', 'reports');
  const copied = [];
  const copy = (source) => {
    const dest = copyIfExists(source, receiptDir);
    if (dest) copied.push({ source, dest });
  };

  const matrixJson = join(reportDir, 'cpu_gpu_accuracy_matrix_latest.json');
  const matrixMd = join(reportDir, 'cpu_gpu_accuracy_matrix_latest.md');
  const conversionJson = join(reportDir, 'conversion_matrix_latest.json');
  const conversionMd = join(reportDir, 'conversion_matrix_latest.md');
  copy(matrixJson);
  copy(matrixMd);
  copy(conversionJson);
  copy(conversionMd);

  const matrix = loadJsonIfExists(matrixJson);
  for (const result of matrix?.results || []) {
    copy(result.jsonPath);
    copy(result.mdPath);
  }

  return { matrix, copied };
}

function summarizeMatrix(matrix) {
  if (!matrix) return null;
  return {
    ok: Boolean(matrix.ok),
    generatedAt: matrix.generatedAt,
    total: matrix.total,
    passCount: matrix.passCount,
    failed: matrix.failed || [],
    results: (matrix.results || []).map((result) => ({
      name: result.name,
      ok: Boolean(result.ok),
      comparablePositions: result.comparablePositions,
      agreementRate: result.agreementRate,
      coverage: result.coverage,
      fighter: result.fighter,
      fighterBlob: result.fighterBlob,
      failures: result.failures || [],
      warnings: result.warnings || [],
    })),
  };
}

function main() {
  const options = parseArgs(process.argv.slice(2));
  const validator = join(options.labDir, 'scripts', 'run_cpu_gpu_accuracy_matrix.mjs');
  if (!existsSync(validator)) {
    throw new Error(`Missing strict accuracy matrix runner: ${validator}`);
  }

  mkdirSync(options.receiptDir, { recursive: true });

  const matrixCommand = [
    process.execPath,
    validator,
    '--samples', String(options.samples),
    '--configs', String(options.configs),
    '--sims', String(options.sims),
    '--min-accuracy', String(options.minAccuracy),
    '--min-coverage', String(options.minCoverage),
  ];
  const command = options.timeoutMs > 0
    ? ['timeout', '--kill-after=10s', `${Math.ceil(options.timeoutMs / 1000)}s`, ...matrixCommand]
    : matrixCommand;

  let run = {
    skipped: true,
    status: null,
    signal: null,
  };

  if (!options.skipRun) {
    const result = spawnSync(command[0], command.slice(1), {
      cwd: options.labDir,
      encoding: 'utf8',
      maxBuffer: 64 * 1024 * 1024,
    });
    run = {
      skipped: false,
      status: result.status,
      signal: result.signal,
    };
    writeFileSync(join(options.receiptDir, 'stdout.txt'), String(result.stdout || ''));
    writeFileSync(join(options.receiptDir, 'stderr.txt'), String(result.stderr || ''));
  }

  const reports = copyReports(options.labDir, options.receiptDir);
  const matrixSummary = summarizeMatrix(reports.matrix);
  const condition = {
    generatedAt: new Date().toISOString(),
    runLabel: options.skipRun ? 'read_existing_external_receipt' : 'external_strict_lab_bridge',
    repoRoot: REPO_ROOT,
    labDir: options.labDir,
    receiptDir: options.receiptDir,
    options: {
      samples: options.samples,
      configs: options.configs,
      sims: options.sims,
      minAccuracy: options.minAccuracy,
      minCoverage: options.minCoverage,
      timeoutMs: options.timeoutMs,
      expectFail: options.expectFail,
      skipRun: options.skipRun,
    },
    command,
    run,
    labGit: {
      head: runGit(options.labDir, ['rev-parse', 'HEAD']),
      status: runGit(options.labDir, ['status', '--porcelain=v1', '--branch']),
    },
    conditionFiles: collectConditionFiles(options.labDir).map((path) => maybeHash(path, options.labDir)),
    copiedReports: reports.copied,
    matrixSummary,
  };

  writeFileSync(join(options.receiptDir, 'condition.json'), JSON.stringify(condition, null, 2) + '\n');

  const ok = Boolean(matrixSummary?.ok);
  const expectedFailOk = options.expectFail && !ok;
  const response = {
    ok,
    expectedFailOk,
    receiptDir: options.receiptDir,
    matrix: matrixSummary,
  };
  process.stdout.write(JSON.stringify(response, null, 2) + '\n');

  if (!ok && !expectedFailOk) process.exitCode = 1;
}

main();
