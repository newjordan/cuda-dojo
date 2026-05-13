#!/usr/bin/env node
import fs from 'node:fs';
import path from 'node:path';

const SCHEMA_VERSION = 'dojo.chrono_frontier_promotion_gate.v1';

function usage() {
  return `Usage: node scripts/gate_chrono_frontier_promotion.mjs --chrono <chrono_sidecar.json> --comparison <comparison.json> --calibration <calibration.json> --independence <audit.json> [--out <gate.json>]

Fail-closed promotion gate for chrono frontier sidecars. This gate does not run
chess search or change runtime behavior. It decides whether existing artifacts
cover the evidence required to promote chrono into a runtime consumer.
`;
}

function parseArgs(argv) {
  const args = {
    chrono: null,
    comparison: null,
    calibration: null,
    independence: null,
    out: null,
    conditionSource: '/home/frosty40/cuda_dojo/LOGIC_RAY_FRONTIER_RUNTIME_RECEIPT_2026-05-06.md',
  };
  for (let i = 0; i < argv.length; i += 1) {
    const token = argv[i];
    if (token === '--help' || token === '-h') {
      console.log(usage());
      process.exit(0);
    }
    if (token === '--chrono') args.chrono = argv[++i];
    else if (token === '--comparison') args.comparison = argv[++i];
    else if (token === '--calibration') args.calibration = argv[++i];
    else if (token === '--independence') args.independence = argv[++i];
    else if (token === '--out') args.out = argv[++i];
    else if (token === '--condition-source') args.conditionSource = argv[++i];
    else throw new Error(`unknown argument: ${token}\n${usage()}`);
  }
  for (const key of ['chrono', 'comparison', 'calibration', 'independence']) {
    if (!args[key]) throw new Error(`missing --${key}\n${usage()}`);
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

function defaultOutPath(comparisonPath) {
  const parsed = path.parse(comparisonPath);
  const stem = parsed.base.replace(/\.chrono_frontier_comparison\.json$/, '').replace(/\.json$/, '');
  return path.join(parsed.dir, `${stem}.chrono_promotion_gate.json`);
}

function deltaImproves(delta) {
  if (!delta || typeof delta !== 'object') return false;
  return (
    asNumber(delta.top1, 0) > 0
    || asNumber(delta.top1AcceptedUsefulInjectionDelta, 0) > 0
    || asNumber(delta.top3, 0) > 0
    || asNumber(delta.top3AcceptedUsefulInjectionDelta, 0) > 0
    || asNumber(delta.meanAcceptedCandidateRank, 0) < 0
    || asNumber(delta.meanAcceptedCandidateRankDelta, 0) < 0
  );
}

function addBlocker(blockers, id, evidence, required) {
  blockers.push({ id, evidence, required });
}

function main() {
  const args = parseArgs(process.argv.slice(2));
  const chronoPath = path.resolve(args.chrono);
  const comparisonPath = path.resolve(args.comparison);
  const calibrationPath = path.resolve(args.calibration);
  const independencePath = path.resolve(args.independence);
  const chrono = readJson(chronoPath);
  const comparison = readJson(comparisonPath);
  const calibration = readJson(calibrationPath);
  const independence = readJson(independencePath);
  const blockers = [];
  const warnings = [];

  if (chrono.schemaVersion !== 'dojo.logic_ray_frontier_chrono_sidecar.bundle.v1') {
    addBlocker(blockers, 'chrono_schema_version_unexpected', chrono.schemaVersion, 'dojo.logic_ray_frontier_chrono_sidecar.bundle.v1');
  }
  if (!Array.isArray(chrono.rows) || chrono.rows.length === 0) {
    addBlocker(blockers, 'chrono_no_rows', chrono.rowCount || 0, 'non-empty chrono sidecar rows');
  }
  if (chrono.promotionPolicy?.status !== 'not_promoted') {
    warnings.push({
      id: 'chrono_artifact_claims_nonstandard_promotion_status',
      evidence: chrono.promotionPolicy?.status || null,
    });
  }
  if (Array.isArray(chrono.promotionPolicy?.blockers) && chrono.promotionPolicy.blockers.length > 0) {
    addBlocker(
      blockers,
      'chrono_artifact_internal_blockers_present',
      chrono.promotionPolicy.blockers,
      'chrono sidecar artifact must clear its own derivation/source/projection blockers before runtime promotion',
    );
  }

  const comparisonLift = Boolean(comparison.comparison?.liftObserved);
  if (!comparisonLift) {
    addBlocker(
      blockers,
      'comparison_no_fixed_artifact_lift',
      comparison.comparison?.bestDelta || null,
      'chrono comparison must improve top1/top3 accepted useful injections or mean accepted rank',
    );
  }
  const bestDelta = comparison.comparison?.bestDelta || {};
  if (!deltaImproves(bestDelta)) {
    addBlocker(
      blockers,
      'comparison_best_delta_not_positive',
      bestDelta,
      'best chrono ranker delta must be positive before runtime integration',
    );
  }

  const cvDelta = calibration.crossValidation?.aggregate?.deltaVsFrontierRank || {};
  if (!deltaImproves(cvDelta)) {
    addBlocker(
      blockers,
      'calibration_no_cross_validated_lift',
      cvDelta,
      'calibrated chrono candidate must cross-validate above frontier rank',
    );
  }
  const equivalence = calibration.bestChronoEquivalenceToFrontierRank || {};
  if (asNumber(equivalence.sameTop1Rate, 0) >= 0.95 && asNumber(equivalence.sameTop3SetRate, 0) >= 0.95) {
    addBlocker(
      blockers,
      'calibration_best_config_frontier_equivalent',
      {
        sameTop1Rate: equivalence.sameTop1Rate,
        sameTop3SetRate: equivalence.sameTop3SetRate,
      },
      'best chrono config must not be merely frontier-rank equivalent',
    );
  }

  const proxyFlagCount = asNumber(independence.audit?.proxyFlagCount, independence.proxyFlags?.length || 0);
  if (proxyFlagCount > 0 || independence.audit?.status === 'proxy_leakage_detected') {
    addBlocker(
      blockers,
      'independence_proxy_leakage_detected',
      {
        proxyFlagCount,
        proxyFlags: (independence.proxyFlags || []).slice(0, 12),
      },
      'chrono sidecar fields must not be high-correlation proxies for existing frontier fields',
    );
  }

  const promote = blockers.length === 0;
  const output = {
    schemaVersion: SCHEMA_VERSION,
    generatedAt: new Date().toISOString(),
    condition: {
      source: path.resolve(args.conditionSource),
      runLabel: 'chrono_frontier_fail_closed_promotion_gate',
      changedFields: 'none; gate reads existing sidecar, comparison, calibration, and independence artifacts',
    },
    sources: {
      chronoPath,
      chronoSchemaVersion: chrono.schemaVersion || null,
      comparisonPath,
      comparisonSchemaVersion: comparison.schemaVersion || null,
      calibrationPath,
      calibrationSchemaVersion: calibration.schemaVersion || null,
      independencePath,
      independenceSchemaVersion: independence.schemaVersion || null,
    },
    evidence: {
      chronoRows: chrono.rowCount || (Array.isArray(chrono.rows) ? chrono.rows.length : 0),
      comparisonBestDelta: comparison.comparison?.bestDelta || null,
      comparisonLiftObserved: comparisonLift,
      calibrationCrossValidatedDelta: cvDelta,
      calibrationEquivalenceToFrontierRank: equivalence,
      independenceAudit: independence.audit || null,
    },
    decision: {
      status: promote ? 'promote_candidate' : 'blocked',
      promote,
      blockingCheckCount: blockers.length,
      blockers,
      warnings,
      nextRequiredAction: promote
        ? 'freeze the chrono formula and rerun the heldout GPU gate as a fixed condition before any operational promotion'
        : 'change chrono sidecar derivation to use independent temporal evidence, then rerun sidecar, comparison, calibration, independence audit, and this gate',
    },
  };
  const outPath = path.resolve(args.out || defaultOutPath(comparisonPath));
  fs.mkdirSync(path.dirname(outPath), { recursive: true });
  fs.writeFileSync(outPath, `${JSON.stringify(output, null, 2)}\n`);
  console.log(JSON.stringify({
    ok: true,
    output: outPath,
    promote,
    status: output.decision.status,
    blockingCheckCount: blockers.length,
    blockers: blockers.map((blocker) => blocker.id),
  }, null, 2));
  process.exit(promote ? 0 : 2);
}

try {
  main();
} catch (error) {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
}
