#!/usr/bin/env node
import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const REPO_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const DEFAULT_ENGINE = resolve(REPO_ROOT, 'cuda', 'engine', 'engine');
const ROW_SCHEMA = 'https://theforge.local/schemas/logic_ray_frontier.schema.json';
const START_FEN = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';

function parseArgs(argv) {
  const options = {
    engine: DEFAULT_ENGINE,
    out: '',
    rootId: 'startpos.search_runtime_probe',
    rootFen: START_FEN,
    rootIndex: 0,
    top: 1,
    temperatureCp: 80,
    bundle: false,
  };
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === '--engine') options.engine = resolve(argv[++i] || options.engine);
    else if (arg === '--out') options.out = argv[++i] || '';
    else if (arg === '--root-id') options.rootId = argv[++i] || options.rootId;
    else if (arg === '--root-index') options.rootIndex = Math.max(0, Math.floor(Number(argv[++i] || 0)));
    else if (arg === '--top') options.top = Math.max(1, Math.floor(Number(argv[++i] || options.top)));
    else if (arg === '--temperature-cp') options.temperatureCp = Math.max(1, Number(argv[++i] || options.temperatureCp));
    else if (arg === '--bundle') options.bundle = true;
    else if (arg === '--help' || arg === '-h') {
      printUsage();
      process.exit(0);
    } else {
      throw new Error(`unknown argument: ${arg}`);
    }
  }
  return options;
}

function printUsage() {
  process.stdout.write(`Usage: node scripts/emit_logic_ray_frontier_from_search_probe.mjs [options]

Runs cuda/engine/engine --search-runtime-probe and maps the GPU scheduler/eval
root-order evidence into canonical logicRayFrontier row(s). The host only
launches the engine, parses JSON, and writes the receipt artifact.

Options:
  --engine PATH             CUDA engine binary (default: ${DEFAULT_ENGINE})
  --out PATH                Write JSON output to PATH
  --root-id ID              Root id stamped on rows
  --root-index N            Root index stamped on rows
  --top N                   Emit top N root-order rows (default: 1)
  --temperature-cp N        Softmax temperature for pathProbability (default: 80)
  --bundle                  Wrap rows in a bundle object
`);
}

function extractJsonObject(text) {
  const start = text.indexOf('{');
  const end = text.lastIndexOf('}');
  if (start < 0 || end < start) {
    throw new Error(`probe did not emit JSON on stdout: ${text.slice(0, 240)}`);
  }
  return JSON.parse(text.slice(start, end + 1));
}

function runSearchProbe(enginePath) {
  const command = [enginePath, '--search-runtime-probe'];
  const result = spawnSync(command[0], command.slice(1), {
    cwd: REPO_ROOT,
    encoding: 'utf8',
    maxBuffer: 8 * 1024 * 1024,
  });
  if (result.error) throw result.error;
  if (result.status !== 0) {
    throw new Error([
      `search-runtime-probe exited ${result.status}`,
      result.stderr?.trim() || '',
      result.stdout?.trim() || '',
    ].filter(Boolean).join('\n'));
  }
  const probe = extractJsonObject(result.stdout || '');
  return {
    probe,
    command,
    stderr: (result.stderr || '').trim(),
  };
}

function clamp(value, min = 0, max = 1) {
  return Math.min(max, Math.max(min, Number(value) || 0));
}

function round(value, digits = 6) {
  const factor = 10 ** digits;
  return Math.round(Number(value || 0) * factor) / factor;
}

function softmax(items, temperatureCp) {
  const scores = items.map((item) => Number(item.score || 0));
  const maxScore = Math.max(...scores);
  const weights = scores.map((score) => Math.exp((score - maxScore) / temperatureCp));
  const denom = weights.reduce((sum, value) => sum + value, 0);
  return weights.map((value) => denom > 0 ? value / denom : 1 / Math.max(1, weights.length));
}

function classifyPzrg(move, scoreCp, gapCp, pathProbability, runtime) {
  let geometry = 'root_move_relation';
  let pressure = 'development_pressure';
  let chessExpression = 'initiative_development';
  let actionGradient = 'increase_plan_continuity';
  let relationScope = 'cuda_search_runtime_root_order';
  let confidence = 'medium';

  if (scoreCp >= 300) {
    pressure = 'terminal_pressure';
    chessExpression = 'killshot_conversion';
    actionGradient = 'increase_conversion_move';
    relationScope = 'death_vector_mirror';
    confidence = 'high';
  } else if (scoreCp >= 120) {
    pressure = 'material_pressure';
    chessExpression = 'conversion_pressure';
    actionGradient = 'increase_protected_kill';
  } else if (gapCp === 0 && pathProbability >= 0.2) {
    pressure = 'root_initiative_pressure';
    chessExpression = 'plan_seed';
  }

  return {
    schema: 'pzrg_4d_label_v1',
    geometry,
    pressure,
    chessExpression,
    actionGradient,
    relationScope,
    confidence,
    evidence: {
      source: 'cuda_dojo_search_runtime_probe',
      move,
      scoreCp: round(scoreCp, 3),
      scoreGapFromBestCp: round(gapCp, 3),
      pathProbability: round(pathProbability),
      rootMoveCount: Number(runtime.root_move_count || 0),
      evalBackend: runtime.eval_backend || 'unknown',
      runtimeUsed: Boolean(runtime.runtime_used),
    },
  };
}

function survivalBucket(scoreCp, risk) {
  if (scoreCp >= 300) return 'conversion_attack';
  if (risk >= 0.72) return 'survival';
  return 'stable_development';
}

function conversionBucket(scoreCp, risk) {
  if (scoreCp >= 450) return 'terminal';
  if (scoreCp >= 120) return 'conversion';
  if (risk >= 0.72) return 'survival_repair';
  return 'candidate';
}

function buildRows(probe, run, options) {
  if (probe.kind !== 'search_runtime_probe') {
    throw new Error(`unsupported probe kind: ${probe.kind || 'missing'}`);
  }
  if (!probe.passed) {
    throw new Error(`search runtime probe did not pass: ${probe.status || 'unknown'}`);
  }
  const runtime = probe.runtime || {};
  const topOrder = Array.isArray(runtime.top_order) ? runtime.top_order : [];
  if (!topOrder.length) throw new Error('search runtime probe emitted no top_order rows');

  const emitted = topOrder.slice(0, Math.max(1, options.top));
  const bestScore = Math.max(...topOrder.map((item) => Number(item.score || 0)));
  const minEmittedScore = Math.min(...emitted.map((item) => Number(item.score || 0)));
  const scoreSpan = Math.max(1, bestScore - minEmittedScore);
  const probabilities = softmax(emitted, options.temperatureCp);
  const now = new Date().toISOString();

  return emitted.map((item, index) => {
    const move = String(item.move || '');
    const scoreCp = Number(item.score || 0);
    const gapCp = Math.max(0, bestScore - scoreCp);
    const pathProbability = round(probabilities[index]);
    const rankPressure = emitted.length <= 1 ? 0 : index / (emitted.length - 1);
    const scorePressure = clamp(gapCp / scoreSpan);
    const risk = round(clamp(0.35 * rankPressure + 0.45 * scorePressure + 0.2 * (1 - pathProbability)));
    const lockIn = round(clamp(0.55 * pathProbability + 0.45 * (1 - scorePressure)));
    const utility = round((scoreCp / 100) + lockIn - risk);
    const pzrg4d = classifyPzrg(move, scoreCp, gapCp, pathProbability, runtime);
    const selectedMoveInFrontier = move === probe.bestmove;

    return {
      $schema: ROW_SCHEMA,
      schemaVersion: 'dojo.logic_ray_frontier.v1',
      rootId: options.rootId,
      rootFen: options.rootFen,
      rootIndex: options.rootIndex,
      sourceEngine: 'cuda_dojo',
      sourceKernel: 'cuda/engine/search.cu::search_root + scheduler/eval_service',
      move,
      rank: index + 1,
      path: [move],
      scoreCp,
      scoreGapFromBestCp: round(gapCp, 3),
      pathProbability,
      risk,
      lockIn,
      utility,
      survivalBucket: survivalBucket(scoreCp, risk),
      conversionBucket: conversionBucket(scoreCp, risk),
      pzrg4d,
      rayfrontFamily: 'cuda_search_runtime_probe_v1',
      rayfrontMetrics: {
        frontierWidth: Number(runtime.root_move_count || topOrder.length),
        emittedWidth: emitted.length,
        rootOrderScoreCp: scoreCp,
        scoreGapFromBestCp: round(gapCp, 3),
        softmaxTemperatureCp: options.temperatureCp,
        // Shannon entropy placeholder — filled by enrich_frontier_policy_entropy.mjs
        // Once FrostMatrix is integrated into gpu_forge.cu, computed from actual
        // from_logits/to_logits policy head distributions.
        policyEntropy: null,
        policyEntropyNormalized: null,
        policyEntropyMax: null,
        policyEntropySource: null,
        winConfidence: null,
        schedulerFrontierSeeded: Number(runtime.scheduler_frontier_seeded || 0),
        schedulerFrontierPops: Number(runtime.scheduler_frontier_pops || 0),
        schedulerEvalRequests: Number(runtime.scheduler_eval_requests || 0),
        schedulerEvalRequestDepth: Number(runtime.scheduler_eval_request_depth || 0),
        evalBackend: runtime.eval_backend || 'unknown',
        evalBucketIdx: Number(runtime.eval_bucket_idx || 0),
        evalBucketSize: Number(runtime.eval_bucket_size || 0),
        evalRequestCount: Number(runtime.eval_request_count || 0),
        evalResultCount: Number(runtime.eval_result_count || 0),
        evalDroppedRequests: Number(runtime.eval_dropped_requests || 0),
        evalFallbackDispatches: Number(runtime.eval_fallback_dispatches || 0),
        evalFailedDispatches: Number(runtime.eval_failed_dispatches || 0),
        runtimeUsed: Boolean(runtime.runtime_used),
        reorderedRootMoves: Number(runtime.reordered_root_moves || 0),
        rankCoverage: round(emitted.length / Math.max(1, Number(runtime.root_move_count || topOrder.length))),
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
        'source.cuda_dojo.truth_engine',
        'source.cuda_search_runtime_probe',
        'phase.logic_ray_frontier',
        'phase.cuda_root_search',
        'runtime.scheduler_frontier',
        'runtime.eval_service',
        `eval.backend.${runtime.eval_backend || 'unknown'}`,
        selectedMoveInFrontier ? 'gate.selected_move_in_frontier' : 'gate.frontier_alternative',
        `pzrg4d.pressure.${pzrg4d.pressure}`,
        `pzrg4d.expression.${pzrg4d.chessExpression}`,
      ],
      provenance: {
        generatedAt: now,
        sourceType: 'cuda_search_runtime_probe',
        enginePath: resolve(options.engine),
        command: run.command,
        cwd: REPO_ROOT,
        cudaOnly: true,
        cpuRuntimePath: false,
        hostRole: 'launch_parse_validate_only',
        probeKind: probe.kind,
        probeStatus: probe.status,
        stderr: run.stderr,
      },
      gate: {
        schemaValid: true,
        gpuTraceCannon: 'search_runtime_probe',
        selectedMoveInFrontier,
        acceptedInjectionScore: utility,
        acceptedUsefulInjection: false,
        status: 'candidate',
        reason: 'runtime contract anchor only; promotion still requires fighter gate and accepted-injection accounting',
      },
      legacy: {
        searchRuntimeProbe: probe,
      },
    };
  });
}

function writeOutput(payload, outPath) {
  const text = `${JSON.stringify(payload, null, 2)}\n`;
  if (outPath) {
    mkdirSync(dirname(resolve(outPath)), { recursive: true });
    writeFileSync(outPath, text);
  }
  process.stdout.write(text);
}

const options = parseArgs(process.argv.slice(2));
const run = runSearchProbe(resolve(options.engine));
const rows = buildRows(run.probe, run, options);
const payload = options.bundle
  ? {
      schemaVersion: 'dojo.logic_ray_frontier.bundle.v1',
      generatedAt: new Date().toISOString(),
      rowSchema: ROW_SCHEMA,
      rows,
    }
  : (rows.length === 1 ? rows[0] : rows);

writeOutput(payload, options.out);
