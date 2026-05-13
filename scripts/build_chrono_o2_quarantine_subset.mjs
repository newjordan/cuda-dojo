#!/usr/bin/env node
import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';

const SCHEMA_VERSION = 'dojo.chrono_o2_quarantine_subset.v1';
const CHRONO_BUNDLE_SCHEMA = 'dojo.logic_ray_frontier_chrono_sidecar.bundle.v1';

function usage() {
  return `Usage: node scripts/build_chrono_o2_quarantine_subset.mjs --bridge <bridge.json> --chrono <chrono_o2_sidecar.json> [--preflight <preflight.json>] --out-bridge <bridge.subset.json> --out-chrono <chrono.subset.json> [--out-manifest <manifest.json>]

Create a paired bridge/chrono subset that quarantines roots with missing O2
temporal neighbors. This is post-hoc artifact filtering only: it does not run
chess, generate legal moves, verify transitions, or promote runtime behavior.
`;
}

function parseArgs(argv) {
  const args = {
    bridge: null,
    chrono: null,
    preflight: null,
    outBridge: null,
    outChrono: null,
    outManifest: null,
    conditionSource: '/home/frosty40/cuda_dojo/LOGIC_RAY_FRONTIER_RUNTIME_RECEIPT_2026-05-06.md',
    runLabel: 'posthoc_pzrg_chrono_o2_missing_neighbor_quarantine_subset',
  };
  for (let i = 0; i < argv.length; i += 1) {
    const token = argv[i];
    if (token === '--help' || token === '-h') {
      console.log(usage());
      process.exit(0);
    }
    if (token === '--bridge') args.bridge = argv[++i];
    else if (token === '--chrono') args.chrono = argv[++i];
    else if (token === '--preflight') args.preflight = argv[++i];
    else if (token === '--out-bridge') args.outBridge = argv[++i];
    else if (token === '--out-chrono') args.outChrono = argv[++i];
    else if (token === '--out-manifest') args.outManifest = argv[++i];
    else if (token === '--condition-source') args.conditionSource = argv[++i];
    else if (token === '--run-label') args.runLabel = argv[++i];
    else throw new Error(`unknown argument: ${token}\n${usage()}`);
  }
  if (!args.bridge) throw new Error(`missing --bridge\n${usage()}`);
  if (!args.chrono) throw new Error(`missing --chrono\n${usage()}`);
  if (!args.outBridge) throw new Error(`missing --out-bridge\n${usage()}`);
  if (!args.outChrono) throw new Error(`missing --out-chrono\n${usage()}`);
  return args;
}

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, 'utf8'));
}

function sha256File(filePath) {
  return crypto.createHash('sha256').update(fs.readFileSync(filePath)).digest('hex');
}

function round(value, digits = 6) {
  const number = Number(value);
  const finite = Number.isFinite(number) ? number : 0;
  const factor = 10 ** digits;
  return Math.round(finite * factor) / factor;
}

function bridgeRootId(row) {
  const frontier = row.logicRayFrontier || row.pzrgCandidate?.logicRayFrontier || {};
  return frontier.rootId || row.rootId || null;
}

function countBy(rows, getter) {
  const counts = {};
  for (const row of rows) {
    const key = getter(row) || 'unknown';
    counts[key] = (counts[key] || 0) + 1;
  }
  return Object.fromEntries(Object.entries(counts).sort(([a], [b]) => a.localeCompare(b)));
}

function rootFenById(rows) {
  const result = new Map();
  for (const row of rows) {
    const rootId = bridgeRootId(row);
    const frontier = row.logicRayFrontier || row.pzrgCandidate?.logicRayFrontier || {};
    if (rootId && !result.has(rootId)) result.set(rootId, frontier.rootFen || row.rootFen || null);
  }
  return result;
}

function quarantineRootsFromChrono(chronoRows) {
  const roots = new Set();
  for (const row of chronoRows) {
    if (
      row.provenance?.missingNeighbor
      || row.provenance?.finiteDifferenceMode === 'missing_neighbor_zero_velocity_quarantine'
      || row.eventHorizon?.bucket === 'quarantine'
    ) {
      roots.add(row.rootId);
    }
  }
  return roots;
}

function quarantineRootsFromPreflight(preflight) {
  const roots = new Set();
  for (const row of Array.isArray(preflight?.blockingRoots) ? preflight.blockingRoots : []) {
    if (Array.isArray(row.reasons) && row.reasons.includes('neighbor_link_missing') && row.rootId) {
      roots.add(row.rootId);
    }
  }
  return roots;
}

function updateBridge(bridge, keptRows, originalRows, manifest) {
  return {
    ...bridge,
    generatedAt: bridge.generatedAt || new Date().toISOString(),
    rowCount: keptRows.length,
    source: {
      ...(bridge.source || {}),
      inputRows: bridge.source?.inputRows ?? originalRows.length,
      emittedRows: keptRows.length,
      quarantineSubsetSourceRows: originalRows.length,
    },
    preservation: {
      ...(bridge.preservation || {}),
      exactPreservationCount: keptRows.length,
      pzrgAxesPreservedCount: keptRows.length,
      frostMatrixAxesPreservedCount: keptRows.length,
    },
    quarantineSubset: manifest,
    rows: keptRows,
  };
}

function updateChronoRows(rows, outBridgePath, manifest) {
  return rows.map((row) => ({
    ...row,
    provenance: {
      ...(row.provenance || {}),
      sourceBridgePath: outBridgePath,
      quarantineSubset: {
        status: 'retained',
        schemaVersion: SCHEMA_VERSION,
        sourceFullBridgePath: manifest.sources.bridgePath,
        sourceFullChronoPath: manifest.sources.chronoPath,
        excludedRootIds: manifest.excludedRootIds,
      },
    },
  }));
}

function updateChrono(chrono, keptRows, originalRows, outBridgePath, manifest) {
  const rootIds = new Set(keptRows.map((row) => row.rootId).filter(Boolean));
  const sourceTemporalRows = keptRows.filter((row) => row.provenance?.sourceTemporal).length;
  const sourceOrderProxyRows = keptRows.filter((row) => row.provenance?.sourceOrderProxy).length;
  const missingNeighborRows = keptRows.filter((row) => row.provenance?.missingNeighbor).length;
  const stableRows = keptRows.filter((row) => row.diagnostics?.stabilityScore >= 0.5).length;
  const meanUncertainty = keptRows.length
    ? keptRows.reduce((sum, row) => sum + Number(row.uncertainty?.score || 0), 0) / keptRows.length
    : 0;
  const retainedRows = updateChronoRows(keptRows, outBridgePath, manifest);
  const blockers = [
    ...(sourceOrderProxyRows ? ['source_order_proxy_transitions_unverified'] : []),
    ...(missingNeighborRows ? ['missing_neighbor_rows_present'] : []),
    'bootstrap_not_learned_response_tensors',
    'quarantine_subset_not_full_gate',
  ];
  return {
    ...chrono,
    generatedAt: chrono.generatedAt || new Date().toISOString(),
    source: {
      ...(chrono.source || {}),
      bridgePath: outBridgePath,
      sourceFullChronoPath: manifest.sources.chronoPath,
      sourceFullChronoRows: originalRows.length,
    },
    condition: {
      source: manifest.condition.source,
      runLabel: manifest.condition.runLabel,
      changedFields: 'post-hoc quarantine subset; missing-neighbor roots excluded from bridge and sidecar rows',
    },
    rowCount: retainedRows.length,
    stats: {
      rootCount: rootIds.size,
      sourceTemporalRows,
      sourceOrderProxyRows,
      missingNeighborRows,
      stableRows,
      stableRate: retainedRows.length ? stableRows / retainedRows.length : 0,
      meanUncertainty: round(meanUncertainty),
      eventHorizonCounts: countBy(retainedRows, (row) => row.eventHorizon?.bucket),
      finiteDifferenceModeCounts: countBy(retainedRows, (row) => row.provenance?.finiteDifferenceMode),
    },
    quarantineSubset: manifest,
    promotionPolicy: {
      status: 'not_promoted',
      reason: 'PZRG_CHRONO_O2 missing-neighbor roots are quarantined for diagnosis only; this subset is not a full runtime promotion gate',
      blockers,
    },
    rows: retainedRows,
  };
}

function main() {
  const args = parseArgs(process.argv.slice(2));
  const bridgePath = path.resolve(args.bridge);
  const chronoPath = path.resolve(args.chrono);
  const preflightPath = args.preflight ? path.resolve(args.preflight) : null;
  const outBridgePath = path.resolve(args.outBridge);
  const outChronoPath = path.resolve(args.outChrono);
  const outManifestPath = args.outManifest ? path.resolve(args.outManifest) : outChronoPath.replace(/\.json$/, '.manifest.json');
  const bridge = readJson(bridgePath);
  const chrono = readJson(chronoPath);
  const preflight = preflightPath ? readJson(preflightPath) : null;
  const bridgeRows = Array.isArray(bridge.rows) ? bridge.rows : [];
  const chronoRows = Array.isArray(chrono.rows) ? chrono.rows : [];
  const roots = new Set([
    ...quarantineRootsFromChrono(chronoRows),
    ...quarantineRootsFromPreflight(preflight),
  ].filter(Boolean));
  if (!roots.size) throw new Error('no missing-neighbor roots found to quarantine');

  const fens = rootFenById(bridgeRows);
  const keptBridgeRows = bridgeRows.filter((row) => !roots.has(bridgeRootId(row)));
  const keptChronoRows = chronoRows.filter((row) => !roots.has(row.rootId));
  const excludedBridgeRows = bridgeRows.length - keptBridgeRows.length;
  const excludedChronoRows = chronoRows.length - keptChronoRows.length;
  const manifest = {
    schemaVersion: SCHEMA_VERSION,
    generatedAt: new Date().toISOString(),
    condition: {
      source: path.resolve(args.conditionSource),
      runLabel: args.runLabel,
      changedFields: 'post-hoc quarantine subset; no chess runtime, no legal-move verification, no promotion',
      labCondition: 'posthoc/proxy/subset',
    },
    sources: {
      bridgePath,
      bridgeSha256: sha256File(bridgePath),
      chronoPath,
      chronoSha256: sha256File(chronoPath),
      preflightPath,
      preflightSha256: preflightPath ? sha256File(preflightPath) : null,
    },
    policy: {
      id: 'exclude_roots_with_missing_chrono_o2_neighbor',
      reason: 'missing_neighbor_zero_velocity_quarantine makes O2 finite-difference evidence non-comparable for that root',
      notPromotionReason: 'subset excludes a known blocked root and still uses unverified source-order proxy and bootstrap response tensors',
    },
    original: {
      bridgeRows: bridgeRows.length,
      chronoRows: chronoRows.length,
      rootCount: new Set(bridgeRows.map(bridgeRootId).filter(Boolean)).size,
    },
    retained: {
      bridgeRows: keptBridgeRows.length,
      chronoRows: keptChronoRows.length,
      rootCount: new Set(keptBridgeRows.map(bridgeRootId).filter(Boolean)).size,
    },
    excludedRootIds: [...roots].sort(),
    excludedRoots: [...roots].sort().map((rootId) => ({
      rootId,
      rootFen: fens.get(rootId) || null,
      bridgeRows: bridgeRows.filter((row) => bridgeRootId(row) === rootId).length,
      chronoRows: chronoRows.filter((row) => row.rootId === rootId).length,
    })),
    excludedBridgeRows,
    excludedChronoRows,
    promotionPolicy: {
      status: 'not_promoted',
      reason: 'quarantine subset only; use for blocker attribution, not runtime promotion',
    },
  };

  const bridgeOut = updateBridge(bridge, keptBridgeRows, bridgeRows, manifest);
  const chronoOut = updateChrono(chrono, keptChronoRows, chronoRows, outBridgePath, manifest);
  fs.mkdirSync(path.dirname(outBridgePath), { recursive: true });
  fs.mkdirSync(path.dirname(outChronoPath), { recursive: true });
  fs.mkdirSync(path.dirname(outManifestPath), { recursive: true });
  fs.writeFileSync(outBridgePath, `${JSON.stringify(bridgeOut, null, 2)}\n`);
  fs.writeFileSync(outChronoPath, `${JSON.stringify(chronoOut, null, 2)}\n`);
  fs.writeFileSync(outManifestPath, `${JSON.stringify(manifest, null, 2)}\n`);
  console.log(JSON.stringify({
    ok: true,
    outBridge: outBridgePath,
    outChrono: outChronoPath,
    outManifest: outManifestPath,
    excludedRootIds: manifest.excludedRootIds,
    retainedBridgeRows: keptBridgeRows.length,
    retainedChronoRows: keptChronoRows.length,
    missingNeighborRows: chronoOut.stats.missingNeighborRows,
    blockers: chronoOut.promotionPolicy.blockers,
    promote: false,
  }, null, 2));
}

try {
  main();
} catch (error) {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
}
