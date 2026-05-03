#!/usr/bin/env node
import { existsSync, mkdirSync, readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { basename, dirname, join, resolve } from 'node:path';
import { spawnSync } from 'node:child_process';
import { compileAgent, getMoveFromFn, resetAgentCache } from '../dojo_runtime.js';
import { START_FEN, generateLegalMoves, parseFen } from '../dojo_chess.js';

const REPO_ROOT = resolve(dirname(new URL(import.meta.url).pathname), '..');
const GPU_FORGE_BIN = join(REPO_ROOT, 'cuda', 'gpu_forge');

function parseArgs(argv) {
  const options = {
    fighter: 'variants/razor_x.js',
    fighterBlob: '',
    fen: START_FEN,
    configs: 4,
    sims: 4,
    timeoutMs: 30000,
    out: '',
  };

  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--fighter') options.fighter = argv[++i] || options.fighter;
    else if (arg === '--fighter-blob') options.fighterBlob = argv[++i] || '';
    else if (arg === '--fen') options.fen = argv[++i] || options.fen;
    else if (arg === '--configs') options.configs = Math.max(1, Number(argv[++i] || options.configs));
    else if (arg === '--sims') options.sims = Math.max(1, Number(argv[++i] || options.sims));
    else if (arg === '--timeout-ms') options.timeoutMs = Math.max(0, Number(argv[++i] || options.timeoutMs));
    else if (arg === '--no-timeout') options.timeoutMs = 0;
    else if (arg === '--out') options.out = argv[++i] || '';
    else throw new Error(`Unknown argument: ${arg}`);
  }

  return options;
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
      .filter((name) => name.startsWith(`${base}_`) && name.endsWith('.cuda_fighter_blob.json'))
      .sort();
    for (const name of siblingMatches) candidates.push(join(dir, name));
  } catch {
    // Leave candidates as-is.
  }

  return candidates.find((candidate) => existsSync(candidate)) || candidates[0] || '';
}

function buildCpuTrace(fighterPath, fen) {
  resetAgentCache(fighterPath);
  const fn = compileAgent(fighterPath, {
    forceReload: true,
    deterministicClock: true,
    clockStartMs: 1700000000000,
    clockStepMs: 25,
  });
  if (!fn) throw new Error(`Could not compile fighter: ${fighterPath}`);

  const legalMoves = generateLegalMoves(parseFen(fen));
  const result = getMoveFromFn(fn, fen);
  return {
    move: result.move || '',
    legalMoves,
    legal: legalMoves.includes(result.move || ''),
    metrics: result.metrics || {},
    trace: result.trace || null,
  };
}

function runGpuTrace(options, fighterBlob, fen, cpuMove, legalMoves) {
  const input = `${fen}\t${cpuMove}\t${legalMoves.join(',')}\n`;
  const forgeArgs = [String(options.configs), String(options.sims), '--fighter-blob', fighterBlob];
  const command = options.timeoutMs > 0
    ? ['timeout', '--kill-after=10s', `${Math.ceil(options.timeoutMs / 1000)}s`, GPU_FORGE_BIN, ...forgeArgs]
    : [GPU_FORGE_BIN, ...forgeArgs];

  const result = spawnSync(command[0], command.slice(1), {
    cwd: REPO_ROOT,
    input,
    encoding: 'utf8',
    maxBuffer: 32 * 1024 * 1024,
  });

  let parsed = null;
  let parseError = '';
  if (result.stdout) {
    try {
      parsed = JSON.parse(result.stdout);
    } catch (error) {
      parseError = error.message;
    }
  }

  return {
    command,
    status: result.status,
    signal: result.signal,
    stdout: result.stdout || '',
    stderr: result.stderr || '',
    parseError,
    parsed,
  };
}

function defaultOutPath(fighterPath) {
  const slug = basename(fighterPath).replace(/\.js$/, '');
  const stamp = new Date().toISOString().replace(/[-:]/g, '').replace(/\.\d{3}Z$/, 'Z');
  return join(REPO_ROOT, 'fighter_accuracy', 'traces', `${slug}_${stamp}.json`);
}

function main() {
  const options = parseArgs(process.argv.slice(2));
  const fighterPath = resolve(REPO_ROOT, options.fighter);
  const fighterBlob = resolve(REPO_ROOT, deriveBlobPath(fighterPath, options.fighterBlob));
  if (!existsSync(fighterPath)) throw new Error(`Missing fighter: ${fighterPath}`);
  if (!existsSync(fighterBlob)) throw new Error(`Missing fighter blob: ${fighterBlob}`);
  if (!existsSync(GPU_FORGE_BIN)) throw new Error(`Missing gpu_forge binary: ${GPU_FORGE_BIN}`);

  const cpu = buildCpuTrace(fighterPath, options.fen);
  const gpu = runGpuTrace(options, fighterBlob, options.fen, cpu.move, cpu.legalMoves);
  const firstDisagreement = Array.isArray(gpu.parsed?.positions) ? gpu.parsed.positions[0] || null : null;

  const report = {
    generatedAt: new Date().toISOString(),
    fighter: fighterPath,
    fighterBlob,
    fen: options.fen,
    options: {
      configs: options.configs,
      sims: options.sims,
      timeoutMs: options.timeoutMs,
    },
    cpu,
    gpu: {
      command: gpu.command,
      status: gpu.status,
      signal: gpu.signal,
      stderr: gpu.stderr,
      parseError: gpu.parseError,
      summary: gpu.parsed?.summary || null,
      positions: gpu.parsed?.positions || [],
    },
    firstDisagreement,
    nextDebugTarget: firstDisagreement
      ? `CPU chose ${firstDisagreement.engine}; GPU chose ${firstDisagreement.mcts}. Add score tracing for both moves.`
      : 'No disagreement emitted for this FEN; inspect summary skips/agreements.',
  };

  const outPath = options.out ? resolve(REPO_ROOT, options.out) : defaultOutPath(fighterPath);
  mkdirSync(dirname(outPath), { recursive: true });
  writeFileSync(outPath, JSON.stringify(report, null, 2) + '\n');

  process.stdout.write(JSON.stringify({
    outPath,
    cpuMove: cpu.move,
    legal: cpu.legal,
    gpuStatus: gpu.status,
    agreementRate: report.gpu.summary?.agreementRate ?? null,
    firstDisagreement,
    nextDebugTarget: report.nextDebugTarget,
  }, null, 2) + '\n');

  if (gpu.status !== 0 || firstDisagreement) process.exitCode = 1;
}

main();

