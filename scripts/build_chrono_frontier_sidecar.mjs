#!/usr/bin/env node
import fs from 'node:fs';
import path from 'node:path';

const ROW_SCHEMA = 'https://theforge.local/schemas/logic_ray_frontier_chrono_sidecar.schema.json';
const BUNDLE_SCHEMA = 'dojo.logic_ray_frontier_chrono_sidecar.bundle.v1';
const LAMBDA = 3722 / 2705;
const OMEGA = (2 * Math.PI) / Math.log(LAMBDA);

function usage() {
  return `Usage: node scripts/build_chrono_frontier_sidecar.mjs --bridge <pzrg_frostmatrix_bridge.json> [--omnifold-manifest <manifest.json>] [--out <sidecar.json>]

Build sidecar-only chronometric diagnostics for logicRayFrontier rows. This
does not run chess search, generate legal moves, or widen the core frontier
schema.
`;
}

function parseArgs(argv) {
  const args = {
    bridge: null,
    omnifoldManifest: null,
    out: null,
  };
  for (let i = 0; i < argv.length; i += 1) {
    const token = argv[i];
    if (token === '--help' || token === '-h') {
      console.log(usage());
      process.exit(0);
    }
    if (token === '--bridge') {
      args.bridge = argv[++i];
    } else if (token === '--omnifold-manifest') {
      args.omnifoldManifest = argv[++i];
    } else if (token === '--out') {
      args.out = argv[++i];
    } else {
      throw new Error(`unknown argument: ${token}\n${usage()}`);
    }
  }
  if (!args.bridge) {
    throw new Error(`missing --bridge\n${usage()}`);
  }
  return args;
}

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, 'utf8'));
}

function asNumber(value, fallback = 0) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

function clamp(value, min = 0, max = 1) {
  return Math.max(min, Math.min(max, asNumber(value, 0)));
}

function round(value, digits = 6) {
  const factor = 10 ** digits;
  return Math.round(asNumber(value, 0) * factor) / factor;
}

function defaultOutPath(bridgePath) {
  const parsed = path.parse(bridgePath);
  const stem = parsed.base.replace(/\.pzrg_frostmatrix_bridge\.json$/, '').replace(/\.json$/, '');
  return path.join(parsed.dir, `${stem}.chrono_sidecar.json`);
}

function phaseFromFen(fen) {
  const parts = String(fen || '').trim().split(/\s+/);
  const fullmove = Number(parts[5] || 0);
  if (!Number.isFinite(fullmove) || fullmove <= 0) {
    return 'root_branch';
  }
  if (fullmove <= 10) return 'opening';
  if (fullmove <= 35) return 'middlegame';
  return 'endgame';
}

function driftBucket(score) {
  if (score <= -0.2) return 'relieving';
  if (score < 0.2) return 'neutral';
  if (score < 0.55) return 'rising';
  return 'unstable';
}

function horizonBucket(rank, rootWidth, uncertainty) {
  if (uncertainty >= 0.72) return 'quarantine';
  if (rank <= 2) return 'immediate';
  if (rank <= Math.max(4, rootWidth * 0.25)) return 'near';
  if (rank <= Math.max(8, rootWidth * 0.5)) return 'mid';
  return 'far';
}

function metricDot(a, b) {
  return -a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
}

function add4(a, b) {
  return a.map((value, index) => value + b[index]);
}

function bridgeRows(document) {
  return Array.isArray(document.rows)
    ? document.rows.filter((row) => row && typeof row === 'object')
    : [];
}

function rootGroups(rows) {
  const groups = new Map();
  for (const row of rows) {
    const frontier = row.logicRayFrontier || row.pzrgCandidate?.logicRayFrontier || {};
    const key = frontier.rootId || row.rootId || 'root';
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(row);
  }
  for (const group of groups.values()) {
    group.sort((a, b) => asNumber((a.logicRayFrontier || a.pzrgCandidate?.logicRayFrontier || {}).rank, 0)
      - asNumber((b.logicRayFrontier || b.pzrgCandidate?.logicRayFrontier || {}).rank, 0));
  }
  return groups;
}

function rowUsefulSignal(row, frontier) {
  const injection = row.pzrgCandidate?.injection_relevance || {};
  const useful = clamp(
    injection.useful_injection_score
      ?? injection.useful_score
      ?? frontier.gate?.acceptedInjectionScore
      ?? frontier.utility,
    0,
    1,
  );
  const selected = Boolean(frontier.gate?.selectedMoveInFrontier);
  const accepted = Boolean(
    injection.accepted_useful_injection
      || injection.promotion_gate_approved
      || frontier.gate?.acceptedUsefulInjection,
  );
  return { useful, selected, accepted };
}

function buildSidecarRow(row, rootRows, rootIndexInGroup, sourceBridgePath, generatedAt, omnifoldManifestPath) {
  const frontier = row.logicRayFrontier || row.pzrgCandidate?.logicRayFrontier || {};
  const previous = rootRows[Math.max(0, rootIndexInGroup - 1)];
  const previousFrontier = previous?.logicRayFrontier || previous?.pzrgCandidate?.logicRayFrontier || frontier;
  const rootWidth = rootRows.length || asNumber(frontier.rayfrontMetrics?.emittedWidth, 1);
  const rank = Math.max(1, asNumber(frontier.rank || row.rank, 1));
  const tau = Math.max(0, asNumber(frontier.rootIndex, 0) + (rank / Math.max(1, rootWidth)));
  const theta = OMEGA * Math.log(tau + 1);
  const pathProbability = clamp(frontier.pathProbability, 0, 1);
  const risk = clamp(frontier.risk, 0, 1);
  const lockIn = clamp(frontier.lockIn, 0, 1);
  const scoreGap = Math.max(0, asNumber(frontier.scoreGapFromBestCp, 0));
  const scoreGapNorm = clamp(scoreGap / 400, 0, 1);
  const rankNorm = rootWidth <= 1 ? 0 : clamp((rank - 1) / (rootWidth - 1), 0, 1);
  const prevPath = clamp(previousFrontier.pathProbability, pathProbability, 1);
  const rankGradient = round(pathProbability - prevPath);
  const pressureScore = round(risk - lockIn);
  const relationScore = round(rankGradient);
  const contortionScore = round(clamp((scoreGapNorm + rankNorm + risk + (1 - lockIn)) / 4, 0, 1));
  const uncertaintyScore = round(clamp(0.45 * (1 - pathProbability) + 0.35 * risk + 0.2 * scoreGapNorm, 0, 1));
  // Shannon entropy from the policy distribution — orthogonal to score-based uncertainty
  const policyEntropy = asNumber(frontier.rayfrontMetrics?.policyEntropyNormalized) || null;
  // Blend: when policy entropy is available, replace path probability component with it
  const uncertaintyEntropy = policyEntropy != null
    ? round(clamp(0.45 * policyEntropy + 0.35 * risk + 0.2 * scoreGapNorm, 0, 1))
    : null;
  const z = [
    round(tau),
    round(pathProbability),
    round(lockIn),
    round(risk),
  ];
  const previousTau = previous === row ? Math.max(0, tau - (1 / Math.max(1, rootWidth))) : asNumber(previousFrontier.rootIndex, 0) + (Math.max(1, asNumber(previousFrontier.rank, 1)) / Math.max(1, rootWidth));
  const previousZ = [
    round(previousTau),
    round(clamp(previousFrontier.pathProbability, pathProbability, 1)),
    round(clamp(previousFrontier.lockIn, lockIn, 1)),
    round(clamp(previousFrontier.risk, risk, 1)),
  ];
  const u = z.map((value, index) => round(value - previousZ[index]));
  const p = [...u];
  const signal = rowUsefulSignal(row, frontier);
  const n = [
    1,
    round(rankNorm),
    round(scoreGapNorm),
    signal.selected ? 1 : 0,
  ];
  const fExt = [
    round(signal.useful),
    signal.selected ? 1 : 0,
    signal.accepted ? 1 : 0,
    round(pathProbability - risk),
  ];
  const fCont = [
    0,
    round(-pressureScore),
    round(relationScore),
    round(-contortionScore),
  ];
  const uNext = add4(add4(u, fExt), fCont);
  const normBefore = round(metricDot(u, u));
  const normAfter = round(metricDot(uNext, uNext));
  const normDrift = round(normAfter - normBefore);
  const orthogonalityResidual = round(metricDot(u, fCont));
  const stabilityScore = clamp(1 - Math.min(1, Math.abs(normDrift)) * 0.55 - Math.min(1, Math.abs(orthogonalityResidual)) * 0.45, 0, 1);
  return {
    $schema: ROW_SCHEMA,
    schemaVersion: 'dojo.logic_ray_frontier_chrono_sidecar.v1',
    sidecarId: `${row.bridgeId || frontier.rootId}.${rank}.chrono`,
    logicRayFrontierHash: row.logicRayFrontierHash,
    rootId: frontier.rootId || row.rootId,
    rootFen: frontier.rootFen || row.rootFen,
    rootIndex: Math.max(0, asNumber(frontier.rootIndex, 0)),
    move: frontier.move || row.move || '0000',
    rank,
    tau: round(tau),
    timePhase: {
      phase: phaseFromFen(frontier.rootFen || row.rootFen),
      theta: round(theta),
      thetaSin: round(Math.sin(theta)),
      thetaCos: round(Math.cos(theta)),
      tauSource: 'rootIndex + rank/rootWidth',
    },
    pressureDrift: {
      score: pressureScore,
      bucket: driftBucket(pressureScore),
    },
    relationDrift: {
      score: relationScore,
      bucket: driftBucket(relationScore),
      pathProbability: round(pathProbability),
      rankGradient,
    },
    pathContortion: {
      score: contortionScore,
      bucket: driftBucket(contortionScore),
    },
    uncertainty: {
      score: uncertaintyScore,
      bucket: driftBucket(uncertaintyScore),
      policyEntropy,
      policyEntropyScore: uncertaintyEntropy,
    },
    eventHorizon: {
      bucket: horizonBucket(rank, rootWidth, uncertaintyScore),
      rank,
      rootWidth,
      pathDepth: Array.isArray(frontier.path) ? Math.max(1, frontier.path.length) : 1,
    },
    vectors: {
      z,
      u,
      p,
      n,
      F_ext: fExt,
      F_cont: fCont,
    },
    diagnostics: {
      normBefore,
      normAfter,
      normDrift,
      orthogonalityResidual,
      stabilityScore: round(stabilityScore),
    },
    runtimeChoiceSignal: {
      selectedMoveInFrontier: signal.selected,
      acceptedUsefulInjection: signal.accepted,
      usefulInjectionScore: round(signal.useful),
      promotionEligible: false,
    },
    provenance: {
      generatedAt,
      sourceBridgePath,
      sourceOmniFoldManifestPath: omnifoldManifestPath || null,
      derivation: 'deterministic sidecar projection from existing logicRayFrontier rank/probability/risk/lockIn/utility fields; no legal-move generation and no search',
      noCoreGeometryWidening: true,
      hostRole: 'sidecar_build_validate_only',
    },
  };
}

function main() {
  const args = parseArgs(process.argv.slice(2));
  const bridgePath = path.resolve(args.bridge);
  const omnifoldManifestPath = args.omnifoldManifest ? path.resolve(args.omnifoldManifest) : null;
  const bridge = readJson(bridgePath);
  const rows = bridgeRows(bridge);
  const grouped = rootGroups(rows);
  const generatedAt = new Date().toISOString();
  const sidecarRows = [];
  for (const group of grouped.values()) {
    group.forEach((row, index) => {
      sidecarRows.push(buildSidecarRow(row, group, index, bridgePath, generatedAt, omnifoldManifestPath));
    });
  }
  const accepted = sidecarRows.filter((row) => row.runtimeChoiceSignal.acceptedUsefulInjection).length;
  const stable = sidecarRows.filter((row) => row.diagnostics.stabilityScore >= 0.5).length;
  const meanUncertainty = sidecarRows.length
    ? sidecarRows.reduce((sum, row) => sum + row.uncertainty.score, 0) / sidecarRows.length
    : 0;
  const output = {
    schemaVersion: BUNDLE_SCHEMA,
    generatedAt,
    rowSchema: ROW_SCHEMA,
    source: {
      bridgePath,
      bridgeSchemaVersion: bridge.schemaVersion || null,
      omnifoldManifestPath,
    },
    rowCount: sidecarRows.length,
    stats: {
      rootCount: grouped.size,
      acceptedUsefulInjections: accepted,
      stableRows: stable,
      stableRate: sidecarRows.length ? stable / sidecarRows.length : 0,
      meanUncertainty: round(meanUncertainty),
      eventHorizonCounts: Object.fromEntries(
        [...new Set(sidecarRows.map((row) => row.eventHorizon.bucket))]
          .sort()
          .map((bucket) => [bucket, sidecarRows.filter((row) => row.eventHorizon.bucket === bucket).length]),
      ),
    },
    promotionPolicy: {
      status: 'not_promoted',
      reason: 'sidecar schema/build proof only; no matched baseline or runtime-choice lift measured yet',
      requiredNextEvidence: [
        'fixed-condition comparison against no-chrono frontier choice',
        'accepted useful injection lift or preserved rate with lower instability',
        'orthogonality/norm-drift thresholds before any core geometry widening',
      ],
    },
    rows: sidecarRows,
  };
  const outPath = path.resolve(args.out || defaultOutPath(bridgePath));
  fs.mkdirSync(path.dirname(outPath), { recursive: true });
  fs.writeFileSync(outPath, `${JSON.stringify(output, null, 2)}\n`);
  console.log(JSON.stringify({
    ok: true,
    output: outPath,
    rows: sidecarRows.length,
    rootCount: grouped.size,
    acceptedUsefulInjections: accepted,
    stableRows: stable,
    meanUncertainty: output.stats.meanUncertainty,
  }, null, 2));
}

try {
  main();
} catch (error) {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
}
