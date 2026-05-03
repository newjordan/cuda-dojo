#!/usr/bin/env node
import { mkdirSync, writeFileSync } from 'node:fs';
import { basename, dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { execFileSync } from 'node:child_process';

const __dirname = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = resolve(__dirname, '..');
const REPORT_DIR = join(REPO_ROOT, 'runtime', 'reports');
const VALIDATOR = join(REPO_ROOT, 'scripts', 'validate_cpu_gpu_accuracy.mjs');

const FIGHTERS = [
  { name: 'Razor X', fighter: 'variants/razor_x.js', slug: 'razor_x_accuracy' },
  { name: 'Queensguard', fighter: 'variants/queensguard.js', slug: 'queensguard_accuracy' },
  { name: 'Firebird', fighter: 'variants/firebird.js', slug: 'firebird_accuracy' },
  { name: 'Fortress', fighter: 'variants/fortress.js', slug: 'fortress_accuracy' },
  { name: 'RazorBlade II', fighter: 'variants/razorblade_ii.js', slug: 'razorblade_ii_accuracy' },
];

function parseArgs(argv) {
  const options = {
    samples: 24,
    configs: 16,
    sims: 8,
    minAccuracy: 0.55,
    minCoverage: 0.75,
    timeoutMs: 300000,
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
    if (arg === '--samples') options.samples = Math.max(4, Number(argv[++i] || options.samples));
    else if (arg === '--configs') options.configs = Math.max(4, Number(argv[++i] || options.configs));
    else if (arg === '--sims') options.sims = Math.max(4, Number(argv[++i] || options.sims));
    else if (arg === '--min-accuracy') options.minAccuracy = Number(argv[++i] || options.minAccuracy);
    else if (arg === '--min-coverage') options.minCoverage = Number(argv[++i] || options.minCoverage);
    else if (arg === '--timeout-ms') options.timeoutMs = Math.max(0, Number(argv[++i] || options.timeoutMs));
    else if (arg === '--no-timeout') options.timeoutMs = 0;
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
  return options;
}

function runOne(entry, options) {
  const args = [
    VALIDATOR,
    '--fighter', entry.fighter,
    '--samples', String(options.samples),
    '--configs', String(options.configs),
    '--sims', String(options.sims),
    '--min-accuracy', String(options.minAccuracy),
    '--min-coverage', String(options.minCoverage),
    '--timeout-ms', String(options.timeoutMs),
    '--slug', entry.slug,
  ];
  if (options.gpuDepth != null) args.push('--gpu-depth', String(options.gpuDepth));
  if (options.gpuFullQeval) args.push('--gpu-full-qeval');
  if (options.gpuFilterLegal) args.push('--gpu-filter-legal');
  if (options.gpuTraincarEval) args.push('--gpu-traincar-eval');
  if (options.gpuTraincarBook) args.push('--gpu-traincar-book');
  if (options.gpuSerialRoot) args.push('--gpu-serial-root');
  if (options.gpuRootOrder) args.push('--gpu-root-order');
  if (options.gpuFamilyDispatch) args.push('--gpu-family-dispatch');
  if (options.gpuTimeoutRootProxy) args.push('--gpu-timeout-root-proxy');

  try {
    const raw = execFileSync(process.execPath, args, {
      cwd: REPO_ROOT,
      encoding: 'utf8',
      maxBuffer: 8 * 1024 * 1024,
    });
    const parsed = JSON.parse(raw);
    return {
      name: entry.name,
      fighter: entry.fighter,
      ...parsed,
    };
  } catch (error) {
    const stdout = error.stdout ? String(error.stdout).trim() : '';
    try {
      const parsed = JSON.parse(stdout);
      return {
        name: entry.name,
        fighter: entry.fighter,
        ...parsed,
      };
    } catch {
      return {
        name: entry.name,
        fighter: entry.fighter,
        ok: false,
        failures: [error.message],
        warnings: [],
      };
    }
  }
}

function buildMarkdown(summary) {
  const lines = [];
  lines.push('# CPU-GPU Accuracy Matrix');
  lines.push('');
  lines.push(`Generated: ${summary.generatedAt}`);
  lines.push('');
  lines.push('| Fighter | Status | Agreement | Coverage | Comparable | Condition | Report |');
  lines.push('|---|---:|---:|---:|---:|---|---|');
  for (const item of summary.results) {
    const status = item.ok ? 'PASS' : 'FAIL';
    const agreement = item.agreementRate == null ? 'n/a' : `${(item.agreementRate * 100).toFixed(1)}%`;
    const coverage = item.coverage == null ? 'n/a' : `${(item.coverage * 100).toFixed(1)}%`;
    const comparable = item.comparablePositions ?? 'n/a';
    const condition = item.condition?.label ||
      (item.gpuSummary?.timeoutRootProxy ||
      item.warnings?.includes('timeout_root_proxy_condition_not_strict_parity')
        ? 'proxy'
        : 'strict');
    const report = item.mdPath ? basename(item.mdPath) : 'n/a';
    lines.push(`| ${item.name} | ${status} | ${agreement} | ${coverage} | ${comparable} | ${condition} | ${report} |`);
  }
  lines.push('');
  lines.push(`Pass: ${summary.passCount}/${summary.total}`);
  if (summary.failed.length) lines.push(`Failed: ${summary.failed.join(', ')}`);
  return lines.join('\n') + '\n';
}

function main() {
  const options = parseArgs(process.argv.slice(2));
  const results = FIGHTERS.map((entry) => runOne(entry, options));
  const passCount = results.filter((item) => item.ok).length;
  const failed = results.filter((item) => !item.ok).map((item) => item.name);

  const summary = {
    ok: failed.length === 0,
    generatedAt: new Date().toISOString(),
    options,
    total: results.length,
    passCount,
    failed,
    results,
  };

  mkdirSync(REPORT_DIR, { recursive: true });
  const jsonPath = join(REPORT_DIR, 'cpu_gpu_accuracy_matrix_latest.json');
  const mdPath = join(REPORT_DIR, 'cpu_gpu_accuracy_matrix_latest.md');
  writeFileSync(jsonPath, JSON.stringify(summary, null, 2) + '\n');
  writeFileSync(mdPath, buildMarkdown(summary));

  process.stdout.write(JSON.stringify({
    ok: summary.ok,
    total: summary.total,
    passCount: summary.passCount,
    failed: summary.failed,
    jsonPath,
    mdPath,
  }, null, 2) + '\n');
  if (!summary.ok) process.exitCode = 1;
}

main();
