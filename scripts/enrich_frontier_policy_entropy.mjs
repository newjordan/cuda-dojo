#!/usr/bin/env node
/**
 * enrich_frontier_policy_entropy.mjs
 *
 * Compute Shannon entropy over the frontier's move distribution and inject it
 * into rayfrontMetrics + chrono blocks. This gives the ray algos a conviction
 * axis orthogonal to the value head:
 *
 *   H(policy) = -sum(p_i * ln(p_i))
 *   winConfidence = exp(-H / H_max) * sigmoid(scoreCp / 100)
 *
 * Low entropy + high score → confident win  (narrow, certain)
 * High entropy + high score → fuzzy win      (optimistic but unsure)
 * Low entropy + low score → confident loss
 * High entropy + low score → chaos
 *
 * Input: frontier row JSON or bundle with rows[] containing pathProbability + scoreCp.
 * Output: enriched rows with rayfrontMetrics.policyEntropy and chrono.policyEntropy.
 *
 * Once FrostMatrix is integrated into the GPU forge, policyEntropy will be
 * computed from the actual from_logits/to_logits distributions (64-way each).
 * Until then, it uses the search score softmax distribution as a proxy.
 *
 * Usage:
 *   node scripts/enrich_frontier_policy_entropy.mjs \
 *     --input frontier.json \
 *     [--out frontier.entropy.json] \
 *     [--temperature 1.0]
 */

import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';

const SCRIPT_VERSION = 'dojo.rayfront.policy_entropy_enrichment.v1';

function usage() {
  return `Usage: node scripts/enrich_frontier_policy_entropy.mjs --input <frontier.json> [--out <enriched.json>]

Compute Shannon entropy over frontier move distributions and inject
policyEntropy into rayfrontMetrics and chrono blocks. Feeds the ray algos
a conviction axis orthogonal to value/score.

Options:
  --input, --in PATH      Frontier row(s) JSON (row, array, or bundle with rows[])
  --out PATH              Output path (default: input.entropy.json)
  --temperature FLOAT     Entropy temperature for probability sharpening (default: 1.0)
  --help, -h              Show this message
`;
}

function parseArgs(argv) {
  const args = {
    input: null,
    out: null,
    temperature: 1.0,
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
    } else if (token === '--temperature') {
      args.temperature = Math.max(0.1, Number(argv[++i]));
    } else if (token === '--condition-source') {
      args.conditionSource = argv[++i];
    } else {
      throw new Error(`unknown argument: ${token}\n${usage()}`);
    }
  }
  if (!args.input) throw new Error(`missing --input\n${usage()}`);
  return args;
}

// ═══════════════════════════════════════════════════════════════════════════
// ENTROPY COMPUTATION
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Shannon entropy: H = -sum(p_i * ln(p_i))
 * For 64-way policy head: H_max = ln(64) ≈ 4.159 per head, 8.318 total.
 * For the current score-softmax proxy: H_max = ln(N_moves).
 */
function shannonEntropy(probabilities) {
  let h = 0;
  for (const p of probabilities) {
    if (p > 0) h -= p * Math.log(p);
  }
  return h;
}

/**
 * Normalized entropy: H_norm = H / H_max  → [0, 1]
 * 0 = completely certain (one-hot), 1 = maximum entropy (uniform).
 */
function normalizedEntropy(entropy, maxEntropy) {
  if (maxEntropy <= 0) return 0;
  return Math.min(1, entropy / maxEntropy);
}

/**
 * Win confidence from entropy + score.
 * Low entropy + high score → high confidence.
 * confidence = exp(-H_norm) * sigmoid(scoreCp/100)
 */
function winConfidence(entropyNorm, scoreCp) {
  const conviction = Math.exp(-entropyNorm);          // [0.368, 1] — low H → near 1
  const scoreBelief = 1 / (1 + Math.exp(-scoreCp / 100));  // sigmoid
  return conviction * scoreBelief;
}

/**
 * Entropy bucket for tag generation.
 *
 * Calibrated thresholds:
 *   [0.00, 0.33) → low    — policy is sharply peaked, one move dominates (>2/3 mass)
 *   [0.33, 0.66) → medium — some spread, a few plausible moves
 *   [0.66, 1.00] → high   — policy is diffuse, no clear favorites
 *   null          → unknown — no entropy data available
 *
 * Reference examples (3-move distribution):
 *   [0.85, 0.10, 0.05] → H_norm = 0.47 → medium (dominant move at 85%)
 *   [0.50, 0.30, 0.20] → H_norm = 0.94 → high   (no clear winner)
 *   [0.95, 0.03, 0.02] → H_norm = 0.24 → low    (near-deterministic)
 */
function entropyBucket(normalizedEntropy) {
  if (normalizedEntropy == null) return 'unknown';
  if (normalizedEntropy < 0.33) return 'low';
  if (normalizedEntropy < 0.66) return 'medium';
  return 'high';
}

/**
 * Generate chrono:policyEntropy:* labels for a frontier row.
 * These labels are consumed by OmniFold fold-delta learners as pruning signals.
 */
function entropyLabels(entropyNorm) {
  const bucket = entropyBucket(entropyNorm);
  const labels = [
    `chrono:policyEntropy:bucket.${bucket}`,
  ];
  if (entropyNorm != null) {
    labels.push(
      `chrono:policyEntropy:${entropyNorm < 0.2 ? 'sharp' : entropyNorm < 0.5 ? 'focused' : 'diffuse'}`,
    );
  }
  return labels;
}

/**
 * Compute per-row entropy from the frontier row's pathProbability.
 *
 * When the frontier has a single row: entropy = 0 (certain).
 * When multiple rows: build a distribution from pathProbability values.
 *
 * For future FrostMatrix integration:
 *   - from_logits: [64] → softmax → H_from
 *   - to_logits: [64] → softmax → H_to
 *   - total entropy = H_from + H_to
 *   - max entropy = 2 * ln(64) ≈ 8.318
 */
function computeRowEntropy(row, allRows) {
  // If we have the full set of sibling frontier rows, compute joint entropy
  if (Array.isArray(allRows) && allRows.length > 1) {
    const probs = allRows
      .map((r) => (typeof r.pathProbability === 'number' ? r.pathProbability : 0))
      .filter((p) => p > 0);
    const totalProb = probs.reduce((sum, p) => sum + p, 0);
    const normalized = totalProb > 0 ? probs.map((p) => p / totalProb) : [1];
    const H = shannonEntropy(normalized);
    const Hmax = Math.log(Math.max(2, allRows.length));
    return { entropy: H, maxEntropy: Hmax, normalizedEntropy: normalizedEntropy(H, Hmax) };
  }

  // Single row: zero entropy (no distribution to measure)
  return { entropy: 0, maxEntropy: 0, normalizedEntropy: 0 };
}

/**
 * FROSTMATRIX READY: compute entropy from actual policy logits.
 * This is the target implementation once the FrostMatrix is integrated into gpu_forge.cu.
 *
 * @param {number[]} fromLogits - 64 logits from the from_sq policy head
 * @param {number[]} toLogits   - 64 logits from the to_sq policy head
 * @param {number} temperature  - softmax temperature
 * @returns {{ fromEntropy, toEntropy, totalEntropy, maxEntropy, normalizedEntropy }}
 */
function frostMatrixPolicyEntropy(fromLogits, toLogits, temperature = 1.0) {
  const maxLogit = Math.max(...fromLogits, ...toLogits, 0);
  
  const fromExp = fromLogits.map((l) => Math.exp((l - maxLogit) / temperature));
  const fromSum = fromExp.reduce((s, v) => s + v, 0);
  const fromProbs = fromSum > 0 ? fromExp.map((v) => v / fromSum) : new Array(64).fill(1 / 64);
  
  const toExp = toLogits.map((l) => Math.exp((l - maxLogit) / temperature));
  const toSum = toExp.reduce((s, v) => s + v, 0);
  const toProbs = toSum > 0 ? toExp.map((v) => v / toSum) : new Array(64).fill(1 / 64);

  const fromH = shannonEntropy(fromProbs);
  const toH = shannonEntropy(toProbs);
  const totalH = fromH + toH;
  const maxH = 2 * Math.log(64);  // ≈ 8.318

  return {
    fromEntropy: fromH,
    toEntropy: toH,
    totalEntropy: totalH,
    maxEntropy: maxH,
    normalizedEntropy: normalizedEntropy(totalH, maxH),
  };
}

// ═══════════════════════════════════════════════════════════════════════════
// ROW ENRICHMENT
// ═══════════════════════════════════════════════════════════════════════════

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

function round(value, places = 6) {
  const number = Number(value);
  if (!Number.isFinite(number)) return null;
  const scale = 10 ** places;
  return Math.round(number * scale) / scale;
}

function asRows(document) {
  if (Array.isArray(document)) return document;
  if (document && Array.isArray(document.rows)) return document.rows;
  if (document && document.schemaVersion === 'dojo.logic_ray_frontier.v1') return [document];
  throw new Error('input must be a logicRayFrontier row, row array, or bundle with rows[]');
}

/**
 * Group rows by rootId so we can compute per-root entropy distributions.
 */
function groupByRoot(rows) {
  const groups = new Map();
  for (const row of rows) {
    const key = String(row.rootId || 'unknown');
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(row);
  }
  return groups;
}

function enrichRow(row, allSiblingRows, options) {
  const scoreCp = typeof row.scoreCp === 'number' ? row.scoreCp : 0;
  const hasFrostMatrix = (
    Array.isArray(row.policyFromLogits) &&
    Array.isArray(row.policyToLogits)
  );

  let policyEntropy;
  if (hasFrostMatrix) {
    policyEntropy = frostMatrixPolicyEntropy(
      row.policyFromLogits,
      row.policyToLogits,
      options.temperature,
    );
  } else {
    // Proxy: use the search score softmax distribution
    const dist = computeRowEntropy(row, allSiblingRows);
    policyEntropy = {
      fromEntropy: null,       // not available without FrostMatrix
      toEntropy: null,
      totalEntropy: dist.entropy,
      maxEntropy: dist.maxEntropy,
      normalizedEntropy: dist.normalizedEntropy,
      source: 'score_softmax_proxy',
    };
  }

  const confidence = winConfidence(policyEntropy.normalizedEntropy, scoreCp);

  // Update rayfrontMetrics
  const rayfrontMetrics = {
    ...(row.rayfrontMetrics || {}),
    policyEntropy: round(policyEntropy.totalEntropy),
    policyEntropyNormalized: round(policyEntropy.normalizedEntropy),
    policyEntropyMax: round(policyEntropy.maxEntropy),
    policyEntropySource: policyEntropy.source || 'frostmatrix',
    winConfidence: round(confidence),
    // Future: fromEntropy, toEntropy when FrostMatrix is integrated
    ...(policyEntropy.fromEntropy != null ? {
      policyFromEntropy: round(policyEntropy.fromEntropy),
      policyToEntropy: round(policyEntropy.toEntropy),
    } : {}),
  };

  // Update chrono block
  const chrono = {
    ...(row.chrono || {}),
    policyEntropy: round(policyEntropy.totalEntropy),
    policyEntropyNormalized: round(policyEntropy.normalizedEntropy),
    winConfidence: round(confidence),
  };

  // Generate entropy labels for OmniFold fold-delta consumption
  const newEntropyLabels = entropyLabels(policyEntropy.normalizedEntropy);
  const labels = Array.isArray(row.labels) ? [...row.labels, ...newEntropyLabels] : [...newEntropyLabels];

  return {
    ...row,
    rayfrontMetrics,
    chrono,
    labels,
  };
}

function defaultOutPath(inputPath) {
  const parsed = path.parse(inputPath);
  return path.join(parsed.dir, `${parsed.name}.policy_entropy.json`);
}

function main() {
  const args = parseArgs(process.argv.slice(2));
  const inputPath = path.resolve(args.input);
  const outPath = path.resolve(args.out || defaultOutPath(inputPath));
  const document = readJson(inputPath);
  const rows = asRows(document);
  const groups = groupByRoot(rows);

  const enrichedRows = rows.map((row) => {
    const siblings = groups.get(String(row.rootId || 'unknown')) || [row];
    return enrichRow(row, siblings, args);
  });

  // Preserve document structure
  let output;
  if (Array.isArray(document)) {
    output = enrichedRows;
  } else if (document && Array.isArray(document.rows)) {
    output = { ...document, rows: enrichedRows };
  } else {
    output = enrichedRows[0] || enrichedRows;
  }

  // Add entropy provenance
  const provenance = {
    enrichmentKind: 'rayfront_policy_entropy',
    enrichmentVersion: SCRIPT_VERSION,
    enrichmentTimestamp: new Date().toISOString(),
    temperature: args.temperature,
    frostMatrixAvailable: rows.some(
      (r) => Array.isArray(r.policyFromLogits) && Array.isArray(r.policyToLogits),
    ),
    sourceFile: inputPath,
    sourceSha256: sha256File(inputPath),
    conditionSource: path.resolve(args.conditionSource),
  };

  if (!Array.isArray(output) && output.schemaVersion) {
    output.policyEntropyProvenance = provenance;
  } else if (output.rows) {
    output.policyEntropyProvenance = provenance;
  }

  writeJson(outPath, output);

  // Summary
  const entropies = enrichedRows.map((r) => r.rayfrontMetrics?.policyEntropyNormalized).filter((v) => v != null);
  const confidences = enrichedRows.map((r) => r.rayfrontMetrics?.winConfidence).filter((v) => v != null);
  
  console.log(JSON.stringify({
    ok: true,
    output: outPath,
    rowCount: enrichedRows.length,
    rootCount: groups.size,
    frostMatrixIntegrated: provenance.frostMatrixAvailable,
    entropyStats: entropies.length > 0 ? {
      count: entropies.length,
      min: round(Math.min(...entropies)),
      max: round(Math.max(...entropies)),
      mean: round(entropies.reduce((s, v) => s + v, 0) / entropies.length),
    } : null,
    confidenceStats: confidences.length > 0 ? {
      count: confidences.length,
      min: round(Math.min(...confidences)),
      max: round(Math.max(...confidences)),
      mean: round(confidences.reduce((s, v) => s + v, 0) / confidences.length),
    } : null,
  }, null, 2));
}

try {
  main();
} catch (error) {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
}
