#!/usr/bin/env node
import fs from 'node:fs';
import path from 'node:path';

const SCHEMA_VERSION = 'dojo.omnifold_frontier_manifest.v1';
const DEFAULT_STANDARD_MANIFEST = '/home/frosty40/cuda_dojo/imports/academy_engine_moonshot_2026-05-06/runtime_transformer_root_metadata/standard_vs_omnifold_manifest-latest.json';
const DEFAULT_SOURCE_MANIFEST = '/home/frosty40/cuda_dojo/imports/academy_engine_moonshot_2026-05-06/runtime_transformer_root_metadata/omnifold_source_manifest-latest.json';

function usage() {
  return `Usage: node scripts/build_omnifold_frontier_manifest.mjs --bridge <pzrg_frostmatrix_bridge.json> [--out <manifest.json>]

Build a runtime-side OmniFold frontier manifest from a logicRayFrontier bridge
artifact and the imported Academy OmniFold manifests. The output defines the
elite 2 / 2+4 / 2+4+6 / 2+4+6+8 fold families, attaches frontier-row hashes to
eligible families, emits off-manifold audit rows, and reports accepted-useful-
injection attribution. It does not claim a trained OmniFold delta.
`;
}

function parseArgs(argv) {
  const args = {
    bridge: null,
    standardManifest: DEFAULT_STANDARD_MANIFEST,
    sourceManifest: DEFAULT_SOURCE_MANIFEST,
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
    } else if (token === '--standard-manifest') {
      args.standardManifest = argv[++i];
    } else if (token === '--source-manifest') {
      args.sourceManifest = argv[++i];
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

function clamp01(value, fallback = 0) {
  return Math.max(0, Math.min(1, asNumber(value, fallback)));
}

function defaultOutPath(bridgePath) {
  const parsed = path.parse(bridgePath);
  const stem = parsed.base.replace(/\.pzrg_frostmatrix_bridge\.json$/, '').replace(/\.json$/, '');
  return path.join(parsed.dir, `${stem}.omnifold_frontier_manifest.json`);
}

function variantGroups(variant) {
  return Array.isArray(variant?.omnifold?.groups) ? variant.omnifold.groups.map(String) : [];
}

function hasAnyGroup(variant, needles) {
  const groups = new Set(variantGroups(variant));
  return needles.some((needle) => groups.has(needle));
}

function hasFoldGroup(variant) {
  return hasAnyGroup(variant, [
    'fold2x2_local_contact',
    'fold2x2_geometric',
    'fold2x2_intersection',
    'fold2x4_anchor_span',
    'fold4x4_matrix',
  ]);
}

function sourceCurriculumById(sourceManifest) {
  const entries = Array.isArray(sourceManifest.curriculum) ? sourceManifest.curriculum : [];
  return Object.fromEntries(entries.map((entry) => [entry.id, entry]));
}

function summarizeVariant(variant) {
  return {
    id: variant.id,
    label: variant.label,
    trainerReady: Boolean(variant.trainerReady),
    omnifoldDim: asNumber(variant.launch?.omnifoldDim, null),
    groups: variantGroups(variant),
  };
}

function buildFoldFamilies(standardManifest, sourceManifest) {
  const variants = Array.isArray(standardManifest?.relationalStage?.activeVariants)
    ? standardManifest.relationalStage.activeVariants
    : [];
  const curriculum = sourceCurriculumById(sourceManifest);

  const specs = [
    {
      id: 'elite_order2_bootstrap',
      foldFamily: 'elite_2_ray_relation_core',
      orderSet: [2],
      selector: (variant) => !hasFoldGroup(variant) && !hasAnyGroup(variant, ['negative_yspace_local']),
      role: 'pair/ray/relation bootstrap',
    },
    {
      id: 'elite_order24_expansion',
      foldFamily: 'elite_2_4_anchor_matrix',
      orderSet: [2, 4],
      selector: (variant) => hasFoldGroup(variant) && !hasAnyGroup(variant, ['negative_yspace_local']),
      role: 'quad and local fold expansion',
    },
    {
      id: 'elite_order246_expansion',
      foldFamily: 'elite_2_4_6_monster_precursor',
      orderSet: [2, 4, 6],
      selector: (variant) => hasAnyGroup(variant, ['negative_yspace_local'])
        || hasAnyGroup(variant, ['fold2x4_anchor_span']) && hasAnyGroup(variant, ['fold4x4_matrix']),
      role: 'monster-group precursor with crossed fold surfaces',
    },
    {
      id: 'elite_order2468_omnifold',
      foldFamily: 'elite_2_4_6_8_monster_group',
      orderSet: [2, 4, 6, 8],
      selector: (variant) => hasAnyGroup(variant, ['fold8x8_monster_group', 'order8_monster_group']),
      role: 'full even-order sparse monster group target',
    },
  ];

  return specs.map((spec) => {
    const activeVariants = variants.filter(spec.selector).map(summarizeVariant);
    const curriculumEntry = curriculum[spec.id] || {};
    const status = activeVariants.length ? 'active_frontier_attachable' : 'blocked_no_active_variant';
    return {
      id: spec.id,
      foldFamily: spec.foldFamily,
      orderSet: spec.orderSet,
      sourceLaneId: curriculumEntry.sourceLaneId || 'stack_elite_2400_3299',
      priority: curriculumEntry.priority || null,
      role: spec.role,
      goal: curriculumEntry.goal || null,
      status,
      activeVariantCount: activeVariants.length,
      activeVariants,
      blockedReason: activeVariants.length ? null : 'imported standard_vs_omnifold manifest has no active render variant for this order set',
    };
  });
}

function bridgeRowMetrics(row) {
  const frontier = row.logicRayFrontier || row.pzrgCandidate?.logicRayFrontier || {};
  const injection = row.pzrgCandidate?.injection_relevance || {};
  const useful = clamp01(
    injection.useful_injection_score
      ?? injection.useful_score
      ?? frontier.gate?.acceptedInjectionScore
      ?? frontier.utility,
    0,
  );
  const accepted = Boolean(
    injection.accepted_useful_injection
      || injection.promotion_gate_approved
      || frontier.gate?.acceptedUsefulInjection,
  );
  const selectedFamily = frontier.omnifoldFamily?.selectedFamily ?? null;
  const offManifold = !selectedFamily || frontier.omnifoldFamily?.status === 'placeholder';
  return {
    bridgeId: row.bridgeId,
    rootId: row.rootId,
    rootFen: row.rootFen,
    move: row.move,
    rank: row.rank,
    logicRayFrontierHash: row.logicRayFrontierHash,
    pzrgClass: row.pzrgCandidate?.hardneg_class || null,
    acceptedUsefulInjection: accepted,
    usefulInjectionScore: useful,
    frontierUtility: asNumber(frontier.utility, 0),
    pathProbability: clamp01(frontier.pathProbability, 0),
    risk: clamp01(frontier.risk, 0),
    lockIn: clamp01(frontier.lockIn, 0),
    selectedFamily,
    offManifold,
    offManifoldReason: offManifold ? 'frontier row has no selected OmniFold family yet' : null,
  };
}

function summarizeFamily(family, metrics) {
  if (family.status !== 'active_frontier_attachable') {
    return {
      foldFamily: family.foldFamily,
      status: family.status,
      attachedRows: 0,
      frontierEvalRows: 0,
      acceptedUsefulInjections: 0,
      acceptedUsefulInjectionRate: 0,
      totalUsefulInjectionScore: 0,
      meanUsefulInjectionScore: 0,
      offManifoldAuditRows: metrics.length,
      effectStatus: 'blocked_no_active_variant',
      deltaAcceptedUsefulInjectionsVsStandard: null,
    };
  }
  const attachedRows = metrics.length;
  const accepted = metrics.filter((row) => row.acceptedUsefulInjection).length;
  const totalUseful = metrics.reduce((sum, row) => sum + row.usefulInjectionScore, 0);
  const offManifold = metrics.filter((row) => row.offManifold).length;
  return {
    foldFamily: family.foldFamily,
    status: family.status,
    activeVariantCount: family.activeVariantCount,
    attachedRows,
    frontierEvalRows: attachedRows,
    acceptedUsefulInjections: accepted,
    acceptedUsefulInjectionRate: attachedRows ? accepted / attachedRows : 0,
    totalUsefulInjectionScore: Number(totalUseful.toFixed(6)),
    meanUsefulInjectionScore: attachedRows ? Number((totalUseful / attachedRows).toFixed(6)) : 0,
    offManifoldAuditRows: offManifold,
    effectStatus: 'frontier_attribution_only_not_trained_fold_delta',
    deltaAcceptedUsefulInjectionsVsStandard: null,
  };
}

function buildRowAttachments(metrics, families) {
  return metrics.map((row) => ({
    bridgeId: row.bridgeId,
    logicRayFrontierHash: row.logicRayFrontierHash,
    rootId: row.rootId,
    move: row.move,
    rank: row.rank,
    acceptedUsefulInjection: row.acceptedUsefulInjection,
    usefulInjectionScore: row.usefulInjectionScore,
    offManifoldAudit: {
      offManifold: row.offManifold,
      selectedFamily: row.selectedFamily,
      reason: row.offManifoldReason,
    },
    foldAssignments: families.map((family) => ({
      foldFamily: family.foldFamily,
      orderSet: family.orderSet,
      status: family.status === 'active_frontier_attachable'
        ? 'attached_frontier_candidate'
        : 'blocked_no_active_variant',
      activeVariantCount: family.activeVariantCount,
      scoring: {
        usefulInjectionScore: row.usefulInjectionScore,
        acceptedUsefulInjection: row.acceptedUsefulInjection,
        deltaVsStandard: null,
        scoreSemantics: 'frontier row attribution before trained OmniFold-vs-standard delta',
      },
    })),
  }));
}

function main() {
  const args = parseArgs(process.argv.slice(2));
  const bridgePath = path.resolve(args.bridge);
  const standardManifestPath = path.resolve(args.standardManifest);
  const sourceManifestPath = path.resolve(args.sourceManifest);
  const bridge = readJson(bridgePath);
  const standardManifest = readJson(standardManifestPath);
  const sourceManifest = readJson(sourceManifestPath);
  const bridgeRows = Array.isArray(bridge.rows) ? bridge.rows : [];
  const metrics = bridgeRows.map(bridgeRowMetrics);
  const foldFamilies = buildFoldFamilies(standardManifest, sourceManifest);
  const familySummaries = foldFamilies.map((family) => summarizeFamily(family, metrics));
  const rowAttachments = buildRowAttachments(metrics, foldFamilies);
  const offManifoldAuditRows = metrics.filter((row) => row.offManifold).length;
  const acceptedUsefulInjections = metrics.filter((row) => row.acceptedUsefulInjection).length;
  const totalUseful = metrics.reduce((sum, row) => sum + row.usefulInjectionScore, 0);
  const output = {
    schemaVersion: SCHEMA_VERSION,
    generatedAt: new Date().toISOString(),
    objective: 'score OmniFold fold families by accepted useful injections per GPU-hour after runtime gates',
    sources: {
      bridgePath,
      bridgeSchemaVersion: bridge.schemaVersion || null,
      standardVsOmniFoldManifestPath: standardManifestPath,
      standardVsOmniFoldGeneratedAt: standardManifest.generatedAt || null,
      omnifoldSourceManifestPath: sourceManifestPath,
      omnifoldSourceGeneratedAt: sourceManifest.generatedAt || null,
      sourcePolicy: sourceManifest.sourcePolicy || null,
    },
    foldFamilies,
    aggregate: {
      frontierRows: metrics.length,
      bridgeRowsWithHashProof: bridge.preservation?.exactPreservationCount || 0,
      acceptedUsefulInjections,
      acceptedUsefulInjectionRate: metrics.length ? acceptedUsefulInjections / metrics.length : 0,
      totalUsefulInjectionScore: Number(totalUseful.toFixed(6)),
      meanUsefulInjectionScore: metrics.length ? Number((totalUseful / metrics.length).toFixed(6)) : 0,
      offManifoldAuditRows,
      activeFoldFamilies: foldFamilies.filter((family) => family.status === 'active_frontier_attachable').length,
      blockedFoldFamilies: foldFamilies.filter((family) => family.status !== 'active_frontier_attachable').length,
    },
    familySummaries,
    offManifoldAuditShape: {
      fields: [
        'bridgeId',
        'logicRayFrontierHash',
        'selectedFamily',
        'offManifold',
        'reason',
        'foldAssignments[].status',
      ],
      rule: 'a frontier row is off-manifold until an OmniFold family is selected by a runtime-facing fold gate',
    },
    promotionPolicy: {
      status: 'not_promoted',
      reason: 'manifest reports frontier attribution only; trained standard-vs-OmniFold deltas and fixed-condition GPU gates are still required',
      requiredNextEvidence: [
        'trained standard-vs-OmniFold eval rows per active fold family',
        'accepted useful injection delta vs standard under fixed gate condition',
        'off-manifold audit resolution for selected families',
      ],
    },
    rowAttachments,
  };
  const outPath = path.resolve(args.out || defaultOutPath(bridgePath));
  fs.mkdirSync(path.dirname(outPath), { recursive: true });
  fs.writeFileSync(outPath, `${JSON.stringify(output, null, 2)}\n`);
  console.log(JSON.stringify({
    ok: true,
    output: outPath,
    frontierRows: output.aggregate.frontierRows,
    activeFoldFamilies: output.aggregate.activeFoldFamilies,
    blockedFoldFamilies: output.aggregate.blockedFoldFamilies,
    acceptedUsefulInjections: output.aggregate.acceptedUsefulInjections,
    offManifoldAuditRows: output.aggregate.offManifoldAuditRows,
  }, null, 2));
}

try {
  main();
} catch (error) {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
}
