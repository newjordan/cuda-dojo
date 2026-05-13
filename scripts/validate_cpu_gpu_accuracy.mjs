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
const FRONTIER_VALIDATOR = join(REPO_ROOT, 'scripts', 'validate_logic_ray_frontier_schema.py');
const FRONTIER_ROW_SCHEMA = 'https://theforge.local/schemas/logic_ray_frontier.schema.json';
const CPU_HARNESS_CONDITION = 'standalone_fresh_process';

function parseArgs(argv) {
  const options = {
    fighter: '',
    fighterBlob: '',
    samples: 24,
    corpusOffset: 0,
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
    gpuCpuShapedSearch: false,
    gpuTraincarBook: false,
    gpuSerialRoot: false,
    gpuRootOrder: false,
    gpuTraincarRootTieBreak: false,
    gpuTraincarRunwayRoot: false,
    gpuTraincarRunwayMargin: 85,
    gpuTraincarRootPrior: false,
    gpuTraincarCpuOrder: false,
    gpuFamilyDispatch: false,
    gpuTimeoutRootProxy: false,
    gpuEmitAll: false,
    gpuFfnPolicy: '',
  };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--fighter') options.fighter = argv[++i] || '';
    else if (arg === '--fighter-blob') options.fighterBlob = argv[++i] || '';
    else if (arg === '--samples') options.samples = Math.max(4, Number(argv[++i] || options.samples));
    else if (arg === '--corpus-offset') options.corpusOffset = Math.max(0, Number(argv[++i] || options.corpusOffset));
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
    else if (arg === '--gpu-cpu-shaped-search') options.gpuCpuShapedSearch = true;
    else if (arg === '--gpu-traincar-book') options.gpuTraincarBook = true;
    else if (arg === '--gpu-serial-root') options.gpuSerialRoot = true;
    else if (arg === '--gpu-root-order') options.gpuRootOrder = true;
    else if (arg === '--gpu-traincar-root-tiebreak') options.gpuTraincarRootTieBreak = true;
    else if (arg === '--gpu-traincar-runway-root') options.gpuTraincarRunwayRoot = true;
    else if (arg === '--gpu-traincar-runway-margin') options.gpuTraincarRunwayMargin = Math.max(0, Number(argv[++i] || options.gpuTraincarRunwayMargin));
    else if (arg === '--gpu-traincar-root-prior') options.gpuTraincarRootPrior = true;
    else if (arg === '--gpu-traincar-cpu-order') options.gpuTraincarCpuOrder = true;
    else if (arg === '--gpu-family-dispatch') options.gpuFamilyDispatch = true;
    else if (arg === '--gpu-timeout-root-proxy') options.gpuTimeoutRootProxy = true;
    else if (arg === '--gpu-emit-all') options.gpuEmitAll = true;
    else if (arg === '--gpu-ffn-policy') options.gpuFfnPolicy = argv[++i] || '';
  }

  if (!options.fighter) {
    throw new Error('Usage: node scripts/validate_cpu_gpu_accuracy.mjs --fighter variants/<name>.js [--fighter-blob path] [--samples N] [--corpus-offset N] [--configs N] [--sims N] [--min-accuracy R] [--min-coverage R] [--slug id]');
  }
  return options;
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

function collectGpuSpineBookFens(limit = 256) {
  const path = join(REPO_ROOT, 'gpu_spine', 'book.jsonl');
  if (!existsSync(path)) return [];
  const out = [];
  const seen = new Set();
  for (const line of readFileSync(path, 'utf8').split('\n')) {
    if (!line.trim()) continue;
    try {
      const row = JSON.parse(line);
      if (!row.fen || row.sf_bestmove === '(none)') continue;
      const fen = completeFen(row.fen);
      if (seen.has(fen)) continue;
      const pos = parseFen(fen);
      if (pos.board.length !== 64 || generateLegalMoves(pos).length === 0) continue;
      seen.add(fen);
      out.push(fen);
      if (out.length >= limit) return out;
    } catch {
      continue;
    }
  }
  return out;
}

function collectVariantBookFens(limit = 256) {
  const dir = join(REPO_ROOT, 'variants');
  if (!existsSync(dir)) return [];
  const files = readdirSync(dir)
    .filter((name) => name.endsWith('.js') || name.endsWith('.src.js'))
    .sort();
  const out = [];
  const seen = new Set();
  const add = (fen) => {
    const n = completeFen(fen);
    if (!n.includes('/') || seen.has(n)) return;
    try {
      const pos = parseFen(n);
      if (pos.board.length !== 64 || generateLegalMoves(pos).length === 0) return;
    } catch {
      return;
    }
    seen.add(n);
    out.push(n);
  };
  for (const file of files) {
    const raw = readFileSync(join(dir, file), 'utf8');
    const regex = /\bB\(\s*["']([^"']+\/[^"']+)["']\s*,/g;
    let match;
    while ((match = regex.exec(raw))) {
      add(match[1]);
      if (out.length >= limit) return out;
    }
  }
  return out;
}

function buildCorpus(maxSamples, offset = 0) {
  const seen = new Set();
  const out = [];
  const add = (fen) => {
    const n = completeFen(fen);
    if (!n || seen.has(n)) return;
    seen.add(n);
    out.push(n);
  };
  add(START_FEN);
  for (const fen of collectBatchFens()) add(fen);
  for (const fen of collectLozzaBenchFens()) add(fen);
  for (const fen of collectGpuSpineBookFens()) add(fen);
  for (const fen of collectVariantBookFens()) add(fen);
  return out.slice(offset, offset + maxSamples);
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

function runGpuForge(configs, sims, blobPath, input, timeoutMs, gpuDepth, gpuFullQeval, gpuFilterLegal, gpuTraincarEval, gpuCpuShapedSearch, gpuTraincarBook, gpuSerialRoot, gpuRootOrder, gpuTraincarRootTieBreak, gpuTraincarRunwayRoot, gpuTraincarRunwayMargin, gpuTraincarRootPrior, gpuTraincarCpuOrder, gpuFamilyDispatch, gpuTimeoutRootProxy, gpuEmitAll, gpuFfnPolicy) {
  if (!existsSync(GPU_FORGE_BIN)) throw new Error(`Missing gpu_forge binary: ${GPU_FORGE_BIN}`);
  if (!existsSync(blobPath)) throw new Error(`Missing fighter blob: ${blobPath}`);
  const forgeArgs = [String(configs), String(sims), '--fighter-blob', blobPath];
  if (gpuDepth != null) forgeArgs.push('--depth', String(gpuDepth));
  if (gpuFullQeval) forgeArgs.push('--full-qeval');
  if (gpuFilterLegal) forgeArgs.push('--filter-legal');
  if (gpuTraincarEval) forgeArgs.push('--traincar-eval');
  if (gpuCpuShapedSearch) forgeArgs.push('--cpu-shaped-search');
  if (gpuTraincarBook) forgeArgs.push('--traincar-book');
  if (gpuSerialRoot) forgeArgs.push('--serial-root');
  if (gpuRootOrder) forgeArgs.push('--root-order');
  if (gpuTraincarRootTieBreak) forgeArgs.push('--traincar-root-tiebreak');
  if (gpuTraincarRunwayRoot) forgeArgs.push('--traincar-runway-root', '--traincar-runway-margin', String(gpuTraincarRunwayMargin));
  if (gpuTraincarRootPrior) forgeArgs.push('--traincar-root-prior');
  if (gpuTraincarCpuOrder) forgeArgs.push('--traincar-cpu-order');
  if (gpuFamilyDispatch) forgeArgs.push('--family-dispatch');
  if (gpuTimeoutRootProxy) forgeArgs.push('--timeout-root-proxy');
  if (gpuEmitAll) forgeArgs.push('--emit-all');
  if (gpuFfnPolicy) forgeArgs.push('--ffn-policy', resolve(REPO_ROOT, gpuFfnPolicy));
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

function clamp(value, min = 0, max = 1) {
  return Math.min(max, Math.max(min, Number(value) || 0));
}

function round(value, digits = 6) {
  const factor = 10 ** digits;
  return Math.round(Number(value || 0) * factor) / factor;
}

function softmax(items, temperatureCp = 80) {
  if (!items.length) return [];
  const scores = items.map((item) => Number(item.score || 0) * 100);
  const maxScore = Math.max(...scores);
  const weights = scores.map((score) => Math.exp((score - maxScore) / temperatureCp));
  const denom = weights.reduce((sum, value) => sum + value, 0);
  return weights.map((value) => denom > 0 ? value / denom : 1 / items.length);
}

function pzrgFromGateRoot(position, move, scoreCp, gapCp, pathProbability) {
  const agreed = Boolean(position.agree);
  let pressure = agreed ? 'root_agreement_pressure' : 'root_disagreement_pressure';
  let chessExpression = agreed ? 'candidate_cohesion' : 'candidate_tension';
  let actionGradient = agreed ? 'increase_accepted_root_choice' : 'decrease_unexplained_root_choice';
  let confidence = agreed ? 'high' : 'medium';

  if (gapCp >= 120) {
    pressure = 'danger_pressure';
    chessExpression = 'forced_alternative';
    actionGradient = 'increase_frontier_audit';
    confidence = 'medium';
  } else if (scoreCp >= 120) {
    pressure = 'conversion_pressure';
    chessExpression = 'conversion_candidate';
    actionGradient = 'increase_conversion_move';
  }

  return {
    schema: 'pzrg_4d_label_v1',
    geometry: 'root_move_relation',
    pressure,
    chessExpression,
    actionGradient,
    relationScope: 'cuda_fighter_accuracy_gate_root_distribution',
    confidence,
    evidence: {
      source: 'cuda_gpu_forge_accuracy_gate',
      fen: position.fen,
      move,
      gpuSelectedMove: position.mcts || null,
      referenceMove: position.engine || null,
      agree: agreed,
      scoreCp: round(scoreCp, 3),
      scoreGapFromBestCp: round(gapCp, 3),
      pathProbability: round(pathProbability),
      engineRank: Number(position.engine_rank || 0),
      legalCount: Number(position.legal_count || 0),
      fixable: Number(position.fixable || 0),
    },
  };
}

function frontierRowsFromGatePositions(positions, context) {
  const rows = [];
  for (const [positionIndex, position] of positions.entries()) {
    const root = Array.isArray(position.gpu_root)
      ? position.gpu_root.filter((item) => item?.move && item.move !== '0000')
      : [];
    if (!root.length) continue;

    const bestScoreCp = Math.max(...root.map((item) => Number(item.score || 0) * 100));
    const minScoreCp = Math.min(...root.map((item) => Number(item.score || 0) * 100));
    const scoreSpan = Math.max(1, bestScoreCp - minScoreCp);
    const probabilities = softmax(root, 80);
    const selectedMove = String(position.mcts || '');
    const emittedWidth = root.length;
    const frontierWidth = Math.max(Number(position.legal_count || emittedWidth), emittedWidth, 1);

    for (const [rankIndex, item] of root.entries()) {
      const move = String(item.move || '');
      const scoreCp = Number(item.score || 0) * 100;
      const gapCp = Math.max(0, bestScoreCp - scoreCp);
      const pathProbability = round(probabilities[rankIndex]);
      const rankPressure = emittedWidth <= 1 ? 0 : rankIndex / (emittedWidth - 1);
      const scorePressure = clamp(gapCp / scoreSpan);
      const risk = round(clamp(0.35 * rankPressure + 0.45 * scorePressure + 0.2 * (1 - pathProbability)));
      const lockIn = round(clamp(0.55 * pathProbability + 0.45 * (1 - scorePressure)));
      const utility = round((scoreCp / 100) + lockIn - risk);
      const pzrg4d = pzrgFromGateRoot(position, move, scoreCp, gapCp, pathProbability);
      const selectedMoveInFrontier = move === selectedMove;
      const acceptedUsefulInjection = selectedMoveInFrontier && Boolean(position.agree);

      rows.push({
        $schema: FRONTIER_ROW_SCHEMA,
        schemaVersion: 'dojo.logic_ray_frontier.v1',
        rootId: `${context.slug}.accuracy_gate.${positionIndex + 1}`,
        rootFen: String(position.fen || ''),
        rootIndex: positionIndex,
        sourceEngine: 'cuda_dojo',
        sourceKernel: 'cuda/gpu_forge.cu::searchKernel',
        move,
        rank: Number(item.rank || rankIndex + 1),
        path: [move],
        scoreCp: round(scoreCp, 3),
        scoreGapFromBestCp: round(gapCp, 3),
        pathProbability,
        risk,
        lockIn,
        utility,
        survivalBucket: risk >= 0.72 ? 'survival' : 'stable_development',
        conversionBucket: scoreCp >= 450 ? 'terminal' : (scoreCp >= 120 ? 'conversion' : (risk >= 0.72 ? 'survival_repair' : 'candidate')),
        pzrg4d,
        rayfrontFamily: 'cuda_fighter_accuracy_gate_v1',
        rayfrontMetrics: {
          frontierWidth,
          emittedWidth,
          rootOrderScoreCp: round(scoreCp, 3),
          scoreGapFromBestCp: round(gapCp, 3),
          softmaxTemperatureCp: 80,
          schedulerFrontierSeeded: frontierWidth,
          schedulerFrontierPops: emittedWidth,
          schedulerEvalRequests: emittedWidth,
          schedulerEvalRequestDepth: emittedWidth,
          evalBackend: 'gpu_forge_root_search',
          evalBucketIdx: 0,
          evalBucketSize: emittedWidth,
          evalRequestCount: emittedWidth,
          evalResultCount: emittedWidth,
          evalDroppedRequests: Math.max(0, frontierWidth - emittedWidth),
          evalFallbackDispatches: 0,
          evalFailedDispatches: 0,
          runtimeUsed: true,
          reorderedRootMoves: 0,
          rankCoverage: round(emittedWidth / frontierWidth),
        },
        omnifoldFamily: {
          status: 'placeholder',
          families: ['elite_2', 'elite_4', 'elite_6', 'elite_8'],
          selectedFamily: null,
          offManifoldAudit: null,
        },
        chrono: {
          status: 'placeholder',
          timePhase: null,
          pressureDrift: null,
          relationDrift: null,
          pathContortion: null,
          uncertainty: null,
          eventHorizon: null,
        },
        labels: [
          'source.cuda_dojo.gpu_forge',
          'source.cuda_fighter_accuracy_gate',
          'phase.logic_ray_frontier',
          'phase.cuda_root_search',
          position.agree ? 'gate.cpu_gpu_agreement' : 'gate.cpu_gpu_disagreement',
          selectedMoveInFrontier ? 'gate.selected_move_in_frontier' : 'gate.frontier_alternative',
          `pzrg4d.pressure.${pzrg4d.pressure}`,
          `pzrg4d.expression.${pzrg4d.chessExpression}`,
        ],
        provenance: {
          generatedAt: context.generatedAt,
          sourceType: 'cuda_fighter_accuracy_gate',
          enginePath: GPU_FORGE_BIN,
          command: context.command,
          cwd: REPO_ROOT,
          cudaOnly: false,
          cpuRuntimePath: false,
          hostRole: 'launch_parse_validate_only',
          probeKind: 'cpu_gpu_accuracy_gate',
          probeStatus: context.reportOk ? 'passed' : 'failed',
          stderr: '',
        },
        gate: {
          schemaValid: true,
          gpuTraceCannon: 'cpu_gpu_accuracy_gate',
          selectedMoveInFrontier,
          acceptedInjectionScore: utility,
          acceptedUsefulInjection,
          status: acceptedUsefulInjection ? 'accepted' : 'candidate',
          reason: acceptedUsefulInjection
            ? 'selected GPU root move agreed with the reference fighter under this gate condition'
            : 'candidate frontier evidence recorded; promotion still requires gate agreement and accepted-injection lift',
        },
        legacy: {
          gpuForgePosition: position,
        },
      });
    }
  }
  return rows;
}

function validateFrontierArtifact(path) {
  try {
    const raw = execFileSync('python3', [FRONTIER_VALIDATOR, path], {
      cwd: REPO_ROOT,
      encoding: 'utf8',
      maxBuffer: 8 * 1024 * 1024,
    });
    return JSON.parse(raw);
  } catch (error) {
    const stdout = error.stdout ? String(error.stdout).trim() : '';
    try {
      return JSON.parse(stdout);
    } catch {
      return {
        ok: false,
        rowCount: 0,
        failures: [{ message: error.message }],
      };
    }
  }
}

function writeFrontierEvidence(slug, positions, context) {
  const rows = frontierRowsFromGatePositions(positions, { ...context, slug });
  const bundle = {
    schemaVersion: 'dojo.logic_ray_frontier.bundle.v1',
    generatedAt: context.generatedAt,
    rowSchema: FRONTIER_ROW_SCHEMA,
    source: 'cuda_fighter_accuracy_gate',
    partial: !context.gpuEmitAll,
    rows,
  };
  mkdirSync(REPORT_DIR, { recursive: true });
  const jsonPath = join(REPORT_DIR, `${slug}.logic_ray_frontier.json`);
  const latestJson = join(REPORT_DIR, 'logic_ray_frontier_latest.json');
  const text = JSON.stringify(bundle, null, 2) + '\n';
  writeFileSync(jsonPath, text);
  writeFileSync(latestJson, text);

  const validation = validateFrontierArtifact(jsonPath);
  const selectedMoveRows = rows.filter((row) => row.gate.selectedMoveInFrontier).length;
  const acceptedUsefulInjections = rows.filter((row) => row.gate.acceptedUsefulInjection).length;
  const sourcePositionsWithFrontier = positions.filter((position) =>
    Array.isArray(position.gpu_root) && position.gpu_root.length > 0,
  ).length;
  return {
    available: rows.length > 0,
    partial: !context.gpuEmitAll,
    sourcePositionCount: positions.length,
    sourcePositionsWithFrontier,
    rowCount: rows.length,
    selectedMoveRows,
    selectedMoveInFrontierRate: rows.length ? selectedMoveRows / rows.length : 0,
    acceptedUsefulInjections,
    acceptedUsefulInjectionRate: rows.length ? acceptedUsefulInjections / rows.length : 0,
    schemaValidation: validation,
    rowSchema: FRONTIER_ROW_SCHEMA,
    jsonPath,
    latestJson,
  };
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
  lines.push(`- Frontier rows: ${report.frontierEvidence.rowCount}`);
  lines.push(`- Frontier schema valid: ${report.frontierEvidence.schemaValidation?.ok ? 'yes' : 'no'}`);
  lines.push(`- Accepted useful injections: ${report.frontierEvidence.acceptedUsefulInjections}`);
  if (report.failures.length) lines.push(`- Failures: ${report.failures.join(', ')}`);
  if (report.warnings.length) lines.push(`- Warnings: ${report.warnings.join(', ')}`);
  return lines.join('\n') + '\n';
}

function main() {
  const options = parseArgs(process.argv.slice(2));
  const fighterPath = resolve(REPO_ROOT, options.fighter);
  const fighterBlob = resolve(REPO_ROOT, deriveBlobPath(fighterPath, options.fighterBlob));
  const slug = options.slug || basename(fighterPath).replace(/\.js$/, '');

  const corpus = buildCorpus(options.samples, options.corpusOffset);
  const { usable, skipped, input } = buildCpuInputLines(fighterPath, corpus);
  const gpu = runGpuForge(options.configs, options.sims, fighterBlob, input, options.timeoutMs, options.gpuDepth, options.gpuFullQeval, options.gpuFilterLegal, options.gpuTraincarEval, options.gpuCpuShapedSearch, options.gpuTraincarBook, options.gpuSerialRoot, options.gpuRootOrder, options.gpuTraincarRootTieBreak, options.gpuTraincarRunwayRoot, options.gpuTraincarRunwayMargin, options.gpuTraincarRootPrior, options.gpuTraincarCpuOrder, options.gpuFamilyDispatch, options.gpuTimeoutRootProxy, options.gpuEmitAll, options.gpuFfnPolicy);
  const summary = gpu.summary || {};
  const gpuTags = [summary.timeoutRootProxy ? 'proxy' : 'strict'];
  if (summary.traincarBook) gpuTags.push('traincar_book');
  if (options.gpuTraincarEval) gpuTags.push('traincar_eval');
  if (options.gpuCpuShapedSearch) gpuTags.push('cpu_shaped_search');
  if (summary.traincarRootTieBreak) gpuTags.push('traincar_root_tiebreak');
  if (summary.traincarRunwayRoot) gpuTags.push('traincar_runway_root');
  if (summary.traincarRootPrior) gpuTags.push('traincar_root_prior');
  if (summary.traincarCpuOrder) gpuTags.push('traincar_cpu_order');
  if (summary.ffnPolicy) gpuTags.push('ffn_policy_proxy');
  const gpuCondition = gpuTags.join('_');
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
  if (summary.ffnPolicy) warnings.push('ffn_policy_condition_not_strict_parity');
  if (options.gpuTraincarBook && !summary.traincarBook) warnings.push('traincar_book_requested_but_not_active');
  if (!options.gpuEmitAll) warnings.push('frontier_evidence_partial_without_gpu_emit_all');

  const generatedAt = new Date().toISOString();
  const frontierPositions = Array.isArray(gpu.positions) ? gpu.positions : [];
  const frontierEvidence = writeFrontierEvidence(slug, frontierPositions, {
    generatedAt,
    command: [process.execPath, ...process.argv.slice(1)],
    gpuEmitAll: options.gpuEmitAll,
    reportOk: failures.length === 0,
  });
  if (frontierEvidence.rowCount > 0 && !frontierEvidence.schemaValidation?.ok) {
    failures.push('frontier_schema_validation_failed');
  }
  if (comparable > 0 && frontierEvidence.rowCount === 0) {
    warnings.push('frontier_no_gpu_root_rows_recorded_use_gpu_emit_all');
  }

  const report = {
    ok: failures.length === 0,
    generatedAt,
    fighter: fighterPath,
    fighterBlob,
    options: {
      samples: options.samples,
      corpusOffset: options.corpusOffset,
      configs: options.configs,
      sims: options.sims,
      minAccuracy: options.minAccuracy,
      minCoverage: options.minCoverage,
      timeoutMs: options.timeoutMs,
      gpuDepth: options.gpuDepth,
      gpuFullQeval: options.gpuFullQeval,
      gpuFilterLegal: options.gpuFilterLegal,
      gpuTraincarEval: options.gpuTraincarEval,
      gpuCpuShapedSearch: options.gpuCpuShapedSearch,
      gpuTraincarBook: options.gpuTraincarBook,
      gpuSerialRoot: options.gpuSerialRoot,
      gpuRootOrder: options.gpuRootOrder,
      gpuTraincarRootTieBreak: options.gpuTraincarRootTieBreak,
      gpuTraincarRunwayRoot: options.gpuTraincarRunwayRoot,
      gpuTraincarRunwayMargin: options.gpuTraincarRunwayMargin,
      gpuTraincarRootPrior: options.gpuTraincarRootPrior,
      gpuTraincarCpuOrder: options.gpuTraincarCpuOrder,
      gpuFamilyDispatch: options.gpuFamilyDispatch,
      gpuTimeoutRootProxy: options.gpuTimeoutRootProxy,
      gpuEmitAll: options.gpuEmitAll,
      gpuFfnPolicy: options.gpuFfnPolicy,
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
      cpuShapedSearch: Boolean(summary.cpuShapedSearch),
      traincarRootTieBreak: Boolean(summary.traincarRootTieBreak),
      traincarRunwayRoot: Boolean(summary.traincarRunwayRoot),
      traincarRunwayMargin: Number(summary.traincarRunwayMargin || 0),
      traincarRootPrior: Boolean(summary.traincarRootPrior),
      traincarCpuOrder: Boolean(summary.traincarCpuOrder),
      traincarBook: Boolean(summary.traincarBook),
      traincarBookEntries: Number(summary.traincarBookEntries || 0),
      traincarBookOverrides: Number(summary.traincarBookOverrides || 0),
      traincarBookMisses: Number(summary.traincarBookMisses || 0),
      emitAllPositions: Boolean(summary.emitAllPositions),
      ffnPolicy: Boolean(summary.ffnPolicy),
      ffnPolicyOverrides: Number(summary.ffnPolicyOverrides || 0),
      skippedNoMoves: Number(summary.skippedNoMoves || 0),
      skippedEngineMoveMissing: Number(summary.skippedEngineMoveMissing || 0),
    },
    frontierEvidence,
    sampleDisagreements: Array.isArray(gpu.positions) ? gpu.positions.slice(0, 10) : [],
    policySamples: options.gpuEmitAll && Array.isArray(gpu.positions) ? gpu.positions : [],
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
    cpuShapedSearch: report.gpuSummary.cpuShapedSearch,
    traincarRootTieBreak: report.gpuSummary.traincarRootTieBreak,
    traincarRunwayRoot: report.gpuSummary.traincarRunwayRoot,
    traincarRunwayMargin: report.gpuSummary.traincarRunwayMargin,
    traincarRootPrior: report.gpuSummary.traincarRootPrior,
    traincarCpuOrder: report.gpuSummary.traincarCpuOrder,
    traincarBookOverrides: report.gpuSummary.traincarBookOverrides,
    emitAllPositions: report.gpuSummary.emitAllPositions,
    ffnPolicy: report.gpuSummary.ffnPolicy,
    ffnPolicyOverrides: report.gpuSummary.ffnPolicyOverrides,
    frontierRows: report.frontierEvidence.rowCount,
    frontierSchemaValid: Boolean(report.frontierEvidence.schemaValidation?.ok),
    acceptedUsefulInjections: report.frontierEvidence.acceptedUsefulInjections,
    frontierArtifact: report.frontierEvidence.jsonPath,
    condition: report.condition,
    failures: report.failures,
    warnings: report.warnings,
    jsonPath,
    mdPath,
  }, null, 2) + '\n');

  if (!report.ok) process.exitCode = 1;
}

main();
