#!/usr/bin/env node
import fs from 'node:fs';
import path from 'node:path';

const SCHEMA_VERSION = 'dojo.chrono_sidecar_independence_audit.v1';

function usage() {
  return `Usage: node scripts/audit_chrono_sidecar_independence.mjs --bridge <bridge.json> --chrono <chrono_sidecar.json> [--out <audit.json>]

Audit whether chrono sidecar fields are independent signals or proxies for
existing frontier rank/gap/utility fields. This is analysis over GPU-derived
artifacts only; it does not run chess search or change runtime behavior.
`;
}

function parseArgs(argv) {
  const args = {
    bridge: null,
    chrono: null,
    out: null,
    conditionSource: '/home/frosty40/cuda_dojo/LOGIC_RAY_FRONTIER_RUNTIME_RECEIPT_2026-05-06.md',
  };
  for (let i = 0; i < argv.length; i += 1) {
    const token = argv[i];
    if (token === '--help' || token === '-h') {
      console.log(usage());
      process.exit(0);
    }
    if (token === '--bridge') args.bridge = argv[++i];
    else if (token === '--chrono') args.chrono = argv[++i];
    else if (token === '--out') args.out = argv[++i];
    else if (token === '--condition-source') args.conditionSource = argv[++i];
    else throw new Error(`unknown argument: ${token}\n${usage()}`);
  }
  if (!args.bridge) throw new Error(`missing --bridge\n${usage()}`);
  if (!args.chrono) throw new Error(`missing --chrono\n${usage()}`);
  return args;
}

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, 'utf8'));
}

function asNumber(value, fallback = 0) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

function round(value, digits = 6) {
  const factor = 10 ** digits;
  return Math.round(asNumber(value, 0) * factor) / factor;
}

function defaultOutPath(chronoPath) {
  const parsed = path.parse(chronoPath);
  const stem = parsed.base.replace(/\.chrono_sidecar\.json$/, '').replace(/\.json$/, '');
  return path.join(parsed.dir, `${stem}.chrono_independence_audit.json`);
}

function rowsByHash(bundle) {
  const map = new Map();
  for (const row of Array.isArray(bundle.rows) ? bundle.rows : []) {
    if (row.logicRayFrontierHash) map.set(row.logicRayFrontierHash, row);
  }
  return map;
}

function mean(values) {
  return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : 0;
}

function pearson(xs, ys) {
  if (xs.length !== ys.length || xs.length < 2) return null;
  const mx = mean(xs);
  const my = mean(ys);
  let numerator = 0;
  let dx = 0;
  let dy = 0;
  for (let i = 0; i < xs.length; i += 1) {
    const x = xs[i] - mx;
    const y = ys[i] - my;
    numerator += x * y;
    dx += x * x;
    dy += y * y;
  }
  const denom = Math.sqrt(dx * dy);
  return denom > 0 ? numerator / denom : null;
}

function ranks(values) {
  const sorted = values
    .map((value, index) => ({ value, index }))
    .sort((a, b) => a.value - b.value);
  const result = new Array(values.length);
  for (let i = 0; i < sorted.length;) {
    let j = i + 1;
    while (j < sorted.length && sorted[j].value === sorted[i].value) j += 1;
    const rank = (i + j + 1) / 2;
    for (let k = i; k < j; k += 1) result[sorted[k].index] = rank;
    i = j;
  }
  return result;
}

function spearman(xs, ys) {
  if (xs.length !== ys.length || xs.length < 2) return null;
  return pearson(ranks(xs), ranks(ys));
}

function fieldStats(values) {
  const finite = values.filter((value) => Number.isFinite(value)).sort((a, b) => a - b);
  if (!finite.length) return null;
  const q = (p) => finite[Math.min(finite.length - 1, Math.max(0, Math.floor((finite.length - 1) * p)))];
  return {
    min: round(finite[0]),
    p25: round(q(0.25)),
    mean: round(mean(finite)),
    p75: round(q(0.75)),
    max: round(finite[finite.length - 1]),
  };
}

function main() {
  const args = parseArgs(process.argv.slice(2));
  const bridgePath = path.resolve(args.bridge);
  const chronoPath = path.resolve(args.chrono);
  const bridge = readJson(bridgePath);
  const chrono = readJson(chronoPath);
  const chronoByHash = rowsByHash(chrono);
  const joined = [];
  for (const row of Array.isArray(bridge.rows) ? bridge.rows : []) {
    const c = chronoByHash.get(row.logicRayFrontierHash);
    if (!c) continue;
    const frontier = row.logicRayFrontier || row.pzrgCandidate?.logicRayFrontier || {};
    joined.push({
      hash: row.logicRayFrontierHash,
      rootId: frontier.rootId || row.rootId,
      move: frontier.move || row.move,
      frontier_rank: asNumber(frontier.rank, 0),
      frontier_utility: asNumber(frontier.utility, 0),
      frontier_pathProbability: asNumber(frontier.pathProbability, 0),
      frontier_risk: asNumber(frontier.risk, 0),
      frontier_lockIn: asNumber(frontier.lockIn, 0),
      frontier_scoreGapFromBestCp: asNumber(frontier.scoreGapFromBestCp, 0),
      chrono_stabilityScore: asNumber(c.diagnostics?.stabilityScore, 0),
      chrono_uncertainty: asNumber(c.uncertainty?.score, 0),
      chrono_pathContortion: asNumber(c.pathContortion?.score, 0),
      chrono_normDriftAbs: Math.abs(asNumber(c.diagnostics?.normDrift, 0)),
      chrono_orthogonalityResidualAbs: Math.abs(asNumber(c.diagnostics?.orthogonalityResidual, 0)),
      chrono_pressureDrift: asNumber(c.pressureDrift?.score, 0),
      chrono_relationDrift: asNumber(c.relationDrift?.score, 0),
    });
  }
  const frontierFields = [
    'frontier_rank',
    'frontier_utility',
    'frontier_pathProbability',
    'frontier_risk',
    'frontier_lockIn',
    'frontier_scoreGapFromBestCp',
  ];
  const chronoFields = [
    'chrono_stabilityScore',
    'chrono_uncertainty',
    'chrono_pathContortion',
    'chrono_normDriftAbs',
    'chrono_orthogonalityResidualAbs',
    'chrono_pressureDrift',
    'chrono_relationDrift',
  ];
  const correlations = {};
  const proxyFlags = [];
  for (const chronoField of chronoFields) {
    correlations[chronoField] = {};
    const xs = joined.map((row) => row[chronoField]);
    for (const frontierField of frontierFields) {
      const ys = joined.map((row) => row[frontierField]);
      const p = pearson(xs, ys);
      const s = spearman(xs, ys);
      correlations[chronoField][frontierField] = {
        pearson: p == null ? null : round(p),
        spearman: s == null ? null : round(s),
      };
      if ((p != null && Math.abs(p) >= 0.95) || (s != null && Math.abs(s) >= 0.95)) {
        proxyFlags.push({
          chronoField,
          frontierField,
          pearson: p == null ? null : round(p),
          spearman: s == null ? null : round(s),
        });
      }
    }
  }
  const stats = Object.fromEntries([...frontierFields, ...chronoFields].map((field) => [
    field,
    fieldStats(joined.map((row) => row[field])),
  ]));
  const output = {
    schemaVersion: SCHEMA_VERSION,
    generatedAt: new Date().toISOString(),
    condition: {
      source: path.resolve(args.conditionSource),
      runLabel: 'chrono_sidecar_independence_audit_on_documented_offset64_artifact',
      changedFields: 'none; correlation audit over existing bridge and chrono artifacts',
    },
    sources: {
      bridgePath,
      bridgeRows: Array.isArray(bridge.rows) ? bridge.rows.length : 0,
      chronoPath,
      chronoRows: Array.isArray(chrono.rows) ? chrono.rows.length : 0,
      joinedRows: joined.length,
    },
    stats,
    correlations,
    proxyFlags,
    audit: {
      proxyFlagCount: proxyFlags.length,
      status: proxyFlags.length ? 'proxy_leakage_detected' : 'no_high_correlation_proxy_detected',
      interpretation: proxyFlags.length
        ? 'one or more chrono fields are high-correlation proxies for existing frontier fields and should not be treated as independent time-pressure signal'
        : 'no chrono field crossed the high-correlation proxy threshold on this artifact',
    },
    promotionPolicy: {
      status: 'not_promoted',
      reason: 'independence audit is diagnostic only; sidecar signals must survive fixed-condition ranking/fight gates before promotion',
    },
  };
  const outPath = path.resolve(args.out || defaultOutPath(chronoPath));
  fs.mkdirSync(path.dirname(outPath), { recursive: true });
  fs.writeFileSync(outPath, `${JSON.stringify(output, null, 2)}\n`);
  console.log(JSON.stringify({
    ok: true,
    output: outPath,
    joinedRows: joined.length,
    proxyFlagCount: proxyFlags.length,
    proxyFlags: proxyFlags.slice(0, 12),
    auditStatus: output.audit.status,
  }, null, 2));
}

try {
  main();
} catch (error) {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
}
