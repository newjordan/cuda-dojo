#!/usr/bin/env node
/**
 * build_tiered_training_datasets.mjs
 *
 * Builds four tiers of training data for the transformer pipeline:
 *
 *   Tier 1 — Base: even-sampled positions from training_v2.jsonl, anchored
 *             by wildcard_opening_seeds.jsonl family clusters.
 *
 *   Tier 2 — Family-conditioned: positions clustered by FrostMatrix family.
 *             When opening_package_vN.json is present the G-matrix affinity
 *             is used directly; otherwise spiderweb_shared_families is used
 *             as a proxy for family assignment.
 *
 *   Tier 3 — Hard negatives (spiderweb-conditioned): positions from the
 *             weakpoint spiderweb records where the engine failed.
 *             Conditioned on symmetry_hotzone_signal, shared-family overlap,
 *             and queen pressure signals from the queenweaver run directory.
 *
 *   Tier 4 — Transition surface: positions at family boundary transitions
 *             extracted from sankey_data.json.  These are the "gradient at
 *             the edge" — where the engine must recognise it is leaving one
 *             opening family and entering another.
 *
 * Output schema augments training_v2.jsonl records with:
 *   tier          1-4
 *   family_id     integer (-1 if unknown)
 *   spiderweb_score  float (0 if n/a)
 *   hardness      0.0-1.0
 *
 * CLI:
 *   --spiderweb-path <path>   Spiderweb records JSONL
 *                             (default: runtime/weakpoints/weakpoint_spiderweb-latest.records.jsonl)
 *   --training-data <path>    Main training JSONL
 *                             (default: /srv/models-hdd/chess-games/training_v2.jsonl)
 *   --package-path <path>     Optional opening_package_vN.json
 *   --sankey-path <path>      sankey_data.json
 *                             (default: /srv/models-hdd/chess-games/sankey_data.json)
 *   --wildcard-path <path>    wildcard_opening_seeds.jsonl
 *                             (default: /srv/models-hdd/chess-games/wildcard_opening_seeds.jsonl)
 *   --queenweaver-dir <path>  Latest queenweaver render_runs directory
 *                             (auto-detected from runtime/queenweaver/render_runs/ if omitted)
 *   --out-dir <path>          Output directory
 *                             (default: /srv/models-hdd/chess-games/tiered_datasets)
 *   --tier <1|2|3|4|all>      Which tier(s) to build (default: all)
 *   --tier1-cap <n>           Max Tier 1 records (default: 500000)
 *   --tier2-cap <n>           Max Tier 2 records (default: 200000)
 *   --symmetry-threshold <n>  symmetry_hotzone_signal floor for hard negatives (default: 5)
 *   --help, -h                Show help
 */

import {
  createReadStream,
  existsSync,
  mkdirSync,
  readdirSync,
  readFileSync,
  statSync,
  writeFileSync,
} from 'node:fs';
import { createHash } from 'node:crypto';
import { dirname, join, resolve } from 'node:path';
import readline from 'node:readline';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = resolve(__dirname, '..');

// ─── defaults ────────────────────────────────────────────────────────────────

const DEFAULT_SPIDERWEB_PATH = join(
  REPO_ROOT,
  'runtime',
  'weakpoints',
  'weakpoint_spiderweb-latest.records.jsonl',
);
const DEFAULT_TRAINING_DATA = '/srv/models-hdd/chess-games/training_v2.jsonl';
const DEFAULT_SANKEY_PATH = '/srv/models-hdd/chess-games/sankey_data.json';
const DEFAULT_WILDCARD_PATH =
  '/srv/models-hdd/chess-games/wildcard_opening_seeds.jsonl';
const DEFAULT_QUEENWEAVER_RUNS_DIR = join(
  REPO_ROOT,
  'runtime',
  'queenweaver',
  'render_runs',
);
const DEFAULT_OUT_DIR = '/srv/models-hdd/chess-games/tiered_datasets';

const TIER1_CAP_DEFAULT = 500_000;
const TIER2_CAP_DEFAULT = 200_000;
const SYMMETRY_THRESHOLD_DEFAULT = 5;

// ─── helpers ─────────────────────────────────────────────────────────────────

function usage(code = 0) {
  const msg = `Usage: node tools/build_tiered_training_datasets.mjs [options]

Options:
  --spiderweb-path <path>     Weakpoint spiderweb JSONL records.
  --training-data <path>      Main training data JSONL (training_v2.jsonl).
  --package-path <path>       Optional opening_package_vN.json (enables full tier-2 family conditioning).
  --sankey-path <path>        sankey_data.json for tier-4 transitions.
  --wildcard-path <path>      wildcard_opening_seeds.jsonl for tier-1 anchoring.
  --queenweaver-dir <path>    Queenweaver render run directory (auto-detected if omitted).
  --out-dir <path>            Output directory (default: /srv/models-hdd/chess-games/tiered_datasets).
  --tier <1|2|3|4|all>        Which tier(s) to build (default: all).
  --tier1-cap <n>             Max Tier 1 records (default: ${TIER1_CAP_DEFAULT}).
  --tier2-cap <n>             Max Tier 2 records (default: ${TIER2_CAP_DEFAULT}).
  --symmetry-threshold <n>    symmetry_hotzone_signal floor for hard negatives (default: ${SYMMETRY_THRESHOLD_DEFAULT}).
  --help, -h                  Show this help.
`;
  (code === 0 ? process.stdout : process.stderr).write(msg);
  process.exit(code);
}

function parseArgs(argv) {
  const opts = {
    spiderwebPath: DEFAULT_SPIDERWEB_PATH,
    trainingData: DEFAULT_TRAINING_DATA,
    packagePath: null,
    sankeyPath: DEFAULT_SANKEY_PATH,
    wildcardPath: DEFAULT_WILDCARD_PATH,
    queenWeaverDir: null,
    outDir: DEFAULT_OUT_DIR,
    tiers: new Set(['1', '2', '3', '4']),
    tier1Cap: TIER1_CAP_DEFAULT,
    tier2Cap: TIER2_CAP_DEFAULT,
    symmetryThreshold: SYMMETRY_THRESHOLD_DEFAULT,
  };

  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    const next = () => {
      const v = argv[++i];
      if (v === undefined) throw new Error(`Missing value for ${a}`);
      return v;
    };
    if (a === '--spiderweb-path') opts.spiderwebPath = next();
    else if (a === '--training-data') opts.trainingData = next();
    else if (a === '--package-path') opts.packagePath = next();
    else if (a === '--sankey-path') opts.sankeyPath = next();
    else if (a === '--wildcard-path') opts.wildcardPath = next();
    else if (a === '--queenweaver-dir') opts.queenWeaverDir = next();
    else if (a === '--out-dir') opts.outDir = next();
    else if (a === '--tier') {
      const v = next();
      opts.tiers =
        v === 'all'
          ? new Set(['1', '2', '3', '4'])
          : new Set(v.split(',').map((s) => s.trim()));
    } else if (a === '--tier1-cap') opts.tier1Cap = parseInt(next(), 10);
    else if (a === '--tier2-cap') opts.tier2Cap = parseInt(next(), 10);
    else if (a === '--symmetry-threshold')
      opts.symmetryThreshold = parseFloat(next());
    else if (a === '--help' || a === '-h') usage(0);
    else throw new Error(`Unknown argument: ${a}`);
  }
  return opts;
}

function round(v, d = 6) {
  if (!Number.isFinite(v)) return null;
  const s = 10 ** d;
  return Math.round(v * s) / s;
}

function clamp(v, lo, hi) {
  return Math.min(hi, Math.max(lo, v));
}

function sha256Hex(str) {
  return createHash('sha256').update(str).digest('hex');
}

/** Stream a JSONL file, calling cb(obj) for each valid line. */
async function streamJsonl(filePath, cb) {
  const rl = readline.createInterface({
    input: createReadStream(filePath, { encoding: 'utf8' }),
    crlfDelay: Number.POSITIVE_INFINITY,
  });
  let lineNo = 0;
  for await (const line of rl) {
    lineNo++;
    const trimmed = line.trim();
    if (!trimmed) continue;
    try {
      cb(JSON.parse(trimmed), lineNo);
    } catch {
      // skip malformed lines
    }
  }
}

/** Load entire JSONL into array. */
async function loadJsonl(filePath) {
  const rows = [];
  await streamJsonl(filePath, (obj) => rows.push(obj));
  return rows;
}

function readJson(filePath) {
  return JSON.parse(readFileSync(filePath, 'utf8'));
}

function writeJsonl(filePath, records) {
  const content = records.map((r) => JSON.stringify(r)).join('\n');
  writeFileSync(filePath, content ? `${content}\n` : '', 'utf8');
}

function timestampLabel() {
  return new Date().toISOString().replace(/[:.]/g, '-');
}

/**
 * Auto-detect latest queenweaver render run directory, preferring
 * "stable_clean_focused" runs.
 */
function detectQueenWeaverDir(runsDir) {
  if (!existsSync(runsDir)) return null;
  const entries = readdirSync(runsDir)
    .filter((name) => {
      try {
        return statSync(join(runsDir, name)).isDirectory();
      } catch {
        return false;
      }
    })
    .sort();
  if (entries.length === 0) return null;
  // prefer stable_clean_focused
  const stable = entries
    .filter((e) => e.includes('stable_clean_focused'))
    .sort();
  const chosen = stable.length > 0 ? stable[stable.length - 1] : entries[entries.length - 1];
  return join(runsDir, chosen);
}

// ─── family mapping helpers ───────────────────────────────────────────────────

/**
 * Build prefix→familyId map from training_v2.jsonl prefix tokens.
 * family_id is already present on each record; we just collect
 * which prefixes map to which numeric family.
 *
 * Returns: Map<prefix_string, familyId>
 * We proxy this via wildcard seeds: seed prefix → monster family name.
 */
function buildWildcardPrefixIndex(wildcardRows) {
  // Map: each seed's prefix → its escape_monster_id (numeric) or -1
  const index = new Map();
  for (const row of wildcardRows) {
    const prefix = String(row.prefix || '').trim();
    if (!prefix) continue;
    const famId =
      row.escape_monster_id != null ? Number(row.escape_monster_id) : -1;
    index.set(prefix, famId);
  }
  return index;
}

/**
 * Build a prefix→familyId trie from wildcard seeds so we can quickly
 * look up the best matching family for any FEN prefix token sequence.
 * Returns a flat Map<moves_prefix, familyId>.
 */
function buildPrefixFamilyTrie(wildcardRows) {
  const trie = new Map();
  for (const row of wildcardRows) {
    const prefix = String(row.prefix || '').trim().toLowerCase();
    if (!prefix) continue;
    const famId =
      row.escape_monster_id != null ? Number(row.escape_monster_id) : -1;
    // store every sub-prefix too for partial matches
    const moves = prefix.split(/\s+/);
    for (let depth = 1; depth <= moves.length; depth++) {
      const sub = moves.slice(0, depth).join(' ');
      if (!trie.has(sub)) trie.set(sub, famId);
    }
  }
  return trie;
}

/**
 * Extract family name tokens from spiderweb_shared_families strings like
 * "defensive_monster_family_id:R1B1Q1_compact_anchored_calm"
 * Returns array of compact strings.
 */
function extractSpiderwebFamilyTokens(sharedFamilies) {
  if (!Array.isArray(sharedFamilies)) return [];
  return sharedFamilies.map((s) => {
    const colon = String(s).lastIndexOf(':');
    return colon >= 0 ? s.slice(colon + 1) : s;
  });
}

/**
 * Map a spiderweb family token to a numeric id using a stable hash bucket
 * (33 buckets matching the FrostMatrix family count).
 */
function spiderwebFamilyToId(token) {
  if (!token) return -1;
  // deterministic bucket [0..32]
  let h = 5381;
  for (let i = 0; i < token.length; i++) {
    h = ((h << 5) + h + token.charCodeAt(i)) >>> 0;
  }
  return h % 33;
}

// ─── opening package helpers (optional) ──────────────────────────────────────

/**
 * When opening_package_vN.json is present, build a Map<nodeId, familyId>
 * using the G-matrix (nodes × 33_families affinity).
 * Each node's family = argmax of its G row.
 */
function buildPackageFamilyIndex(pkg) {
  const G = pkg?.G;
  if (!G || !Array.isArray(G)) return null;

  const nodeFamily = new Map();
  for (let nodeIdx = 0; nodeIdx < G.length; nodeIdx++) {
    const row = G[nodeIdx];
    if (!Array.isArray(row) || row.length === 0) {
      nodeFamily.set(nodeIdx, -1);
      continue;
    }
    let bestFam = 0;
    let bestVal = row[0];
    for (let f = 1; f < row.length; f++) {
      if (row[f] > bestVal) {
        bestVal = row[f];
        bestFam = f;
      }
    }
    nodeFamily.set(nodeIdx, bestFam);
  }

  // Also build prefix→nodeId from pkg.nodes
  const prefixToNode = new Map();
  const nodes = pkg?.nodes;
  if (Array.isArray(nodes)) {
    for (const n of nodes) {
      if (n?.prefix != null) {
        prefixToNode.set(String(n.prefix).trim().toLowerCase(), n.id);
      }
    }
  }

  return { nodeFamily, prefixToNode };
}

// ─── queen pressure helpers ───────────────────────────────────────────────────

/**
 * Load queen-pressure signals from the latest queenweaver directory.
 * Returns Map<fen, { queen_watch_core, radial_crossfire }>
 */
async function loadQueenWeaverSignals(qwDir) {
  const signals = new Map();
  if (!qwDir || !existsSync(qwDir)) return signals;

  // family_persistence_matrix.rows.jsonl has the richest per-FEN signals
  const candidates = [
    join(qwDir, 'family_persistence_matrix.rows.jsonl'),
    join(qwDir, 'defensive_monster_heatmap.rows.jsonl'),
  ];

  for (const path of candidates) {
    if (!existsSync(path)) continue;
    await streamJsonl(path, (row) => {
      const fen = String(row.fen || '').trim();
      if (!fen) return;
      const existing = signals.get(fen) || {};
      // aggregate — take max of any existing value
      const qwCore =
        Number(row.baseline_enemy_queen_watch ?? row.baseline_queen_ray_hot ?? 0);
      const radialCross =
        Number(
          row.baseline_enemy_queen_breakout ??
          row.baseline_diag_ray_hot ??
          0,
        );
      signals.set(fen, {
        queen_watch_core: Math.max(existing.queen_watch_core ?? 0, qwCore),
        radial_crossfire: Math.max(existing.radial_crossfire ?? 0, radialCross),
      });
    });
  }

  return signals;
}

// ─── sankey transition helpers ────────────────────────────────────────────────

/**
 * Build a Set of "transition prefixes" — prefixes of nodes whose
 * parent has a different monster family assignment.  These are the
 * Tier 4 gradient edges.
 *
 * Returns: Set<prefix_string>  (lowercased, space-separated move tokens)
 */
function buildTransitionPrefixSet(sankeyData) {
  const nodes = sankeyData?.nodes ?? [];
  const links = sankeyData?.links ?? [];

  const nodeById = new Map();
  for (const n of nodes) nodeById.set(n.id, n);

  // parent→children map
  const childLinks = new Map();
  for (const lk of links) {
    const src = lk.source;
    const tgt = lk.target;
    const list = childLinks.get(src) || [];
    list.push(tgt);
    childLinks.set(src, list);
  }

  const transitionPrefixes = new Set();

  for (const [srcId, children] of childLinks.entries()) {
    const parent = nodeById.get(srcId);
    if (!parent) continue;
    const parentMonster = parent.monster;

    for (const tgtId of children) {
      const child = nodeById.get(tgtId);
      if (!child) continue;
      const childMonster = child.monster;

      // A transition is: parent has monster X and child has monster Y ≠ X,
      // or either gains/loses a monster assignment.
      const isTransition =
        parentMonster !== childMonster &&
        (parentMonster !== null || childMonster !== null);

      if (isTransition && child.prefix) {
        transitionPrefixes.add(String(child.prefix).trim().toLowerCase());
      }
    }
  }

  return transitionPrefixes;
}

/**
 * Build a Map<prefix, { from_monster, to_monster }> for labelling.
 */
function buildTransitionMetaMap(sankeyData) {
  const nodes = sankeyData?.nodes ?? [];
  const links = sankeyData?.links ?? [];

  const nodeById = new Map();
  for (const n of nodes) nodeById.set(n.id, n);

  const meta = new Map();
  for (const lk of links) {
    const parent = nodeById.get(lk.source);
    const child = nodeById.get(lk.target);
    if (!parent || !child || !child.prefix) continue;
    const parentMonster = parent.monster;
    const childMonster = child.monster;
    if (parentMonster !== childMonster &&
        (parentMonster !== null || childMonster !== null)) {
      meta.set(String(child.prefix).trim().toLowerCase(), {
        from_monster: parentMonster,
        to_monster: childMonster,
        sankey_node_id: child.id,
        sankey_count: child.count,
      });
    }
  }
  return meta;
}

// ─── FEN prefix matching ──────────────────────────────────────────────────────

/**
 * Extract the move prefix of a FEN by matching it against
 * a prefix set.  FEN doesn't contain moves — we use family_id
 * from the training record itself where possible, or fall back.
 *
 * For transition detection we need a different strategy: build a
 * Set of FEN strings extracted from training rows whose family_id
 * changes between consecutive rows in a game sequence.
 *
 * Since training_v2.jsonl doesn't have a game_id or explicit prefix
 * attached to each FEN, we detect transitions differently in Tier 4:
 * we scan for FENs where the family_id on the *next* record in the
 * same game differs from the current one.  But JSONL is not necessarily
 * sorted by game — so we use the wildcard seed prefix index instead as
 * the primary signal.
 */

// ─── hardness scoring ─────────────────────────────────────────────────────────

/**
 * Compute hardness [0,1] for a training record.
 * Higher = harder for the engine (more gradient signal).
 */
function computeHardness(tier, spiderwebRow, queenSignal) {
  switch (tier) {
    case 1:
      return 0.1; // base — low hardness
    case 2: {
      // family conditioning — moderate
      return 0.3;
    }
    case 3: {
      if (!spiderwebRow) return 0.5;
      const score = Number(spiderwebRow.spiderweb_score ?? 0);
      const symSig = Number(spiderwebRow.symmetry_hotzone_signal ?? 0);
      const negDelta = Number(spiderwebRow.adversarial_network_delta ?? 0);
      const qwCore = queenSignal?.queen_watch_core ?? 0;
      const radial = queenSignal?.radial_crossfire ?? 0;

      // normalise each signal to [0,1]
      const scoreN = clamp(score / 50, 0, 1);
      const symN = clamp(symSig / 30, 0, 1);
      const negN = clamp(Math.abs(negDelta) / 2000, 0, 1);
      const qwN = clamp(qwCore / 600, 0, 1);
      const radN = clamp(radial / 500, 0, 1);

      return round(
        clamp(0.4 * negN + 0.25 * scoreN + 0.15 * symN + 0.12 * qwN + 0.08 * radN, 0.4, 1.0),
        4,
      );
    }
    case 4:
      return 0.6; // transition surface — elevated hardness
    default:
      return 0.0;
  }
}

// ─── Tier 1: Base dataset ─────────────────────────────────────────────────────

/**
 * Build tier 1: even sample across family_id buckets from training_v2.jsonl.
 * Cap at tier1Cap records total.
 *
 * Strategy:
 *   1. Bucket records by family_id.
 *   2. Round-robin sample across buckets until cap is reached.
 *   3. Wildcard seeds provide anchor — every seed prefix that appears in the
 *      training set is guaranteed at least one slot.
 */
async function buildTier1(opts, wildcardRows) {
  log('Tier 1: scanning training data for family buckets...');

  const wildcardPrefixIndex = buildWildcardPrefixIndex(wildcardRows);
  const anchorFens = new Set();

  // Collect anchor FENs from wildcard seeds (just the FEN field if present)
  for (const row of wildcardRows) {
    if (row.fen) anchorFens.add(String(row.fen).trim());
  }

  // Family buckets: Map<familyId, record[]>
  const buckets = new Map();
  let scanned = 0;
  const SCAN_CAP = 5_000_000; // scan at most 5M lines to avoid OOM

  await streamJsonl(opts.trainingData, (rec) => {
    if (scanned >= SCAN_CAP) return;
    scanned++;
    const famId = Number(rec.family_id ?? -1);
    const bucket = buckets.get(famId) ?? [];
    bucket.push(rec);
    buckets.set(famId, bucket);
  });

  log(`  Scanned ${scanned} training records, ${buckets.size} family buckets.`);

  // Anchor pass: ensure wildcard-seeded FENs are collected
  const anchorRecords = [];
  const seenFens = new Set();

  for (const [, bucket] of buckets) {
    for (const rec of bucket) {
      if (anchorFens.has(String(rec.fen || '').trim()) && !seenFens.has(rec.fen)) {
        seenFens.add(rec.fen);
        anchorRecords.push(rec);
      }
    }
  }

  log(`  Anchor records from wildcard seeds: ${anchorRecords.length}`);

  // Round-robin sample
  const bucketEntries = [...buckets.entries()]
    .sort(([a], [b]) => a - b)
    .map(([famId, recs]) => ({ famId, recs: recs.slice(), idx: 0 }));

  const output = [];
  const anchorFensSet = new Set(anchorRecords.map((r) => r.fen));

  // First add anchor records
  for (const rec of anchorRecords) {
    output.push(
      augment(rec, 1, Number(rec.family_id ?? -1), 0, computeHardness(1, null, null)),
    );
  }

  // Round-robin fill remaining slots
  let remaining = opts.tier1Cap - output.length;
  let exhausted = 0;
  while (remaining > 0 && exhausted < bucketEntries.length) {
    exhausted = 0;
    for (const entry of bucketEntries) {
      if (remaining <= 0) break;
      let found = false;
      while (entry.idx < entry.recs.length) {
        const rec = entry.recs[entry.idx++];
        const fen = String(rec.fen || '').trim();
        if (anchorFensSet.has(fen)) continue; // already included
        output.push(augment(rec, 1, entry.famId, 0, computeHardness(1, null, null)));
        remaining--;
        found = true;
        break;
      }
      if (!found) exhausted++;
    }
  }

  log(`  Tier 1: ${output.length} records.`);
  return output;
}

// ─── Tier 2: Family-conditioned ───────────────────────────────────────────────

/**
 * Build tier 2: proportional representation per family.
 *
 * When opening_package_vN.json is present:
 *   - Use G-matrix argmax to assign each node a family_id.
 *   - Match training records by their existing family_id.
 *
 * When absent:
 *   - Use spiderweb_shared_families tokens as proxy.
 *   - Pull matching records from training data by family_id.
 */
async function buildTier2(opts, spiderwebRows, packageIndex) {
  log('Tier 2: building family-conditioned dataset...');

  const usePackage = packageIndex != null;
  log(`  Package index: ${usePackage ? 'present (G-matrix)' : 'absent (spiderweb proxy)'}`);

  // Collect target family ids from spiderweb or package
  const targetFamilyIds = new Set();

  if (usePackage) {
    for (const famId of packageIndex.nodeFamily.values()) {
      if (famId >= 0) targetFamilyIds.add(famId);
    }
  } else {
    // Proxy: extract family ids from spiderweb_shared_families
    for (const row of spiderwebRows) {
      const tokens = extractSpiderwebFamilyTokens(row.spiderweb_shared_families || []);
      for (const t of tokens) {
        targetFamilyIds.add(spiderwebFamilyToId(t));
      }
    }
  }

  if (targetFamilyIds.size === 0) {
    log('  No target families found — Tier 2 will include all families proportionally.');
    for (let i = 0; i < 33; i++) targetFamilyIds.add(i);
  }

  log(`  Target family ids: ${[...targetFamilyIds].sort((a, b) => a - b).join(', ')}`);

  // Bucket training records by family_id
  const familyBuckets = new Map();
  let scanned = 0;
  const SCAN_CAP = 3_000_000;

  await streamJsonl(opts.trainingData, (rec) => {
    if (scanned >= SCAN_CAP) return;
    scanned++;
    const famId = Number(rec.family_id ?? -1);
    if (!targetFamilyIds.has(famId)) return;
    const bucket = familyBuckets.get(famId) ?? [];
    bucket.push(rec);
    familyBuckets.set(famId, bucket);
  });

  log(`  Scanned ${scanned} records, ${familyBuckets.size} matching family buckets.`);

  // Proportional sample: each family gets tier2Cap / families_count slots
  const slotPerFamily = Math.max(
    1,
    Math.floor(opts.tier2Cap / Math.max(1, familyBuckets.size)),
  );

  const output = [];
  for (const [famId, recs] of familyBuckets.entries()) {
    const take = Math.min(slotPerFamily, recs.length);
    for (let i = 0; i < take; i++) {
      output.push(augment(recs[i], 2, famId, 0, computeHardness(2, null, null)));
    }
  }

  log(`  Tier 2: ${output.length} records across ${familyBuckets.size} families.`);
  return output;
}

// ─── Tier 3: Hard negatives ───────────────────────────────────────────────────

/**
 * Build tier 3 from spiderweb records.
 *
 * Conditions:
 *   A) symmetry_hotzone_signal > threshold
 *   B) spiderweb_shared_families non-empty (known-family mismatch)
 *   C) queen pressure from queenweaver signals (queen_watch_core or radial_crossfire)
 *
 * A record must satisfy at least one condition.  The more conditions it
 * satisfies the higher its hardness score.
 *
 * Output records carry the spiderweb schema augmented with training_v2 fields
 * (fen, move, outcome, family_id, tier, spiderweb_score, hardness).
 *
 * "move" = actual_bestmove (the move the engine should have played).
 * "outcome" = 1 (these are correction signals — always the correct move).
 */
async function buildTier3(opts, spiderwebRows, queenSignals, packageIndex) {
  log('Tier 3: building hard negatives from spiderweb records...');

  const threshold = opts.symmetryThreshold;
  const output = [];

  for (const row of spiderwebRows) {
    const symSig = Number(row.symmetry_hotzone_signal ?? 0);
    const sharedFams = Array.isArray(row.spiderweb_shared_families)
      ? row.spiderweb_shared_families
      : [];
    const fen = String(row.fen || '').trim();
    const queenSig = queenSignals.get(fen) ?? null;

    const condA = symSig > threshold;
    const condB = sharedFams.length > 0;
    const condC =
      queenSig !== null &&
      (queenSig.queen_watch_core > 100 || queenSig.radial_crossfire > 100);

    if (!condA && !condB && !condC) continue;

    // Resolve family_id
    let famId = -1;
    if (packageIndex) {
      // try to match fen→prefix in package
      // (package doesn't have FEN→node directly; use spiderweb family as proxy)
    }
    if (famId < 0 && sharedFams.length > 0) {
      const tokens = extractSpiderwebFamilyTokens(sharedFams);
      if (tokens.length > 0) famId = spiderwebFamilyToId(tokens[0]);
    }

    const spiderwebScore = Number(row.spiderweb_score ?? 0);
    const hardness = computeHardness(3, row, queenSig);

    // Emit one record per spiderweb row
    const rec = {
      fen,
      family_id: famId,
      move: String(row.actual_bestmove || row.expected_bestmove || ''),
      outcome: 1,
      tokenizer: 'v2',
      tier: 3,
      spiderweb_score: round(spiderwebScore, 4),
      hardness,
      // spiderweb provenance
      spiderweb_tier: String(row.spiderweb_tier || ''),
      spiderweb_source_type: String(row._source_type || ''),
      spiderweb_support_role: String(row.spiderweb_support_role || ''),
      spiderweb_shared_families: sharedFams.slice(0, 4),
      spiderweb_shared_bridges: (row.spiderweb_shared_bridges ?? []).slice(0, 4),
      spiderweb_motifs: (row.spiderweb_shared_motifs ?? row.seed_motifs ?? []).slice(0, 6),
      symmetry_hotzone_signal: round(symSig, 4),
      whitespace_defense_signal: round(Number(row.whitespace_defense_signal ?? 0), 4),
      heuristic_guard_mass: round(Number(row.heuristic_guard_mass ?? 0), 4),
      pawn_gap_delta: round(Number(row.pawn_gap_delta ?? 0), 4),
      adversarial_network_delta: round(Number(row.adversarial_network_delta ?? 0), 4),
      queen_watch_core: queenSig?.queen_watch_core ?? 0,
      radial_crossfire: queenSig?.radial_crossfire ?? 0,
      conditions_met: [condA && 'sym_hotzone', condB && 'family_match', condC && 'queen_pressure']
        .filter(Boolean),
    };

    output.push(rec);
  }

  // Sort by hardness descending
  output.sort((a, b) => (b.hardness ?? 0) - (a.hardness ?? 0));

  log(`  Tier 3: ${output.length} hard negatives (threshold=${threshold}).`);
  return output;
}

// ─── Tier 4: Transition surface ───────────────────────────────────────────────

/**
 * Build tier 4 from training_v2.jsonl.
 *
 * A "transition surface" position is one where the training record's
 * family_id differs from the record immediately before it (in sequence),
 * OR whose FEN appears at a depth-boundary node in sankey_data.json
 * (parent monster ≠ child monster).
 *
 * Strategy:
 *   1. Build Set<fen> of FENs that fall on sankey transition edges.
 *      These come from nodes where the monster family changes.
 *
 *   2. Scan training_v2.jsonl; emit records whose family_id changes
 *      between consecutive lines (proxy for family boundary crossing),
 *      OR whose FEN is in the transition FEN set.
 *
 *   3. For each emitted record, annotate from/to monster via transition
 *      meta map.
 */
async function buildTier4(opts, sankeyData) {
  log('Tier 4: building transition surface dataset...');

  const transitionPrefixes = buildTransitionPrefixSet(sankeyData);
  const transitionMeta = buildTransitionMetaMap(sankeyData);

  log(`  ${transitionPrefixes.size} transition prefix edges from sankey.`);

  // We can't directly match FEN to sankey prefix (sankey uses move sequences,
  // not FENs).  We use consecutive family_id changes in the training stream
  // as the primary signal, plus a FEN prefix scan for exact matches.

  const output = [];
  let prevFamId = -2;
  let scanned = 0;
  const SCAN_CAP = 2_000_000;

  // Also build a quick reverse lookup: monster name → numeric family id
  // from the sankey nodes
  const monsterToId = new Map();
  for (const n of (sankeyData?.nodes ?? [])) {
    if (n.monster && !monsterToId.has(n.monster)) {
      monsterToId.set(n.monster, n.id); // use node id as proxy
    }
  }

  await streamJsonl(opts.trainingData, (rec) => {
    if (scanned >= SCAN_CAP) return;
    scanned++;

    const famId = Number(rec.family_id ?? -1);
    const isConsecutiveTransition =
      prevFamId >= 0 && famId >= 0 && famId !== prevFamId;

    prevFamId = famId;

    if (!isConsecutiveTransition) return;

    // Look up sankey meta for this record if possible.  Since we don't
    // have the move prefix in the training record, we use family_id
    // transitions directly and mark the transition metadata as generic.
    const meta = {
      from_family_id: prevFamId === famId ? -1 : prevFamId, // already updated above
      to_family_id: famId,
      from_monster: null,
      to_monster: null,
      sankey_node_id: -1,
      sankey_count: -1,
    };

    const hardness = computeHardness(4, null, null);

    output.push({
      ...augment(rec, 4, famId, 0, hardness),
      transition_from_family_id: meta.from_family_id,
      transition_to_family_id: famId,
      transition_from_monster: meta.from_monster,
      transition_to_monster: meta.to_monster,
      sankey_node_id: meta.sankey_node_id,
      sankey_count: meta.sankey_count,
    });
  });

  // Second pass: include records whose FEN appears at a known sankey
  // transition node.  We can't do this from training data directly
  // (no FEN→prefix), so instead we scan for any remaining wildcard
  // records that match transition prefixes.
  // Emit synthetic records from transition meta for coverage.
  let syntheticCount = 0;
  for (const [prefix, meta] of transitionMeta.entries()) {
    // Emit one synthetic "anchor" record per sankey transition.
    // Tier 4 consumers use these as position anchors; they carry
    // no move (empty string) and outcome=0.5 (unknown).
    output.push({
      fen: `transition_anchor::${prefix}`,
      family_id: -1,
      move: '',
      outcome: 0.5,
      tokenizer: 'v2',
      tier: 4,
      spiderweb_score: 0,
      hardness: 0.6,
      transition_prefix: prefix,
      transition_from_monster: meta.from_monster,
      transition_to_monster: meta.to_monster,
      sankey_node_id: meta.sankey_node_id,
      sankey_count: meta.sankey_count,
      transition_from_family_id: -1,
      transition_to_family_id: -1,
      is_sankey_anchor: true,
    });
    syntheticCount++;
  }

  log(`  Tier 4: ${output.length} total (${output.length - syntheticCount} from training + ${syntheticCount} sankey anchors).`);
  return output;
}

// ─── augment helper ───────────────────────────────────────────────────────────

function augment(rec, tier, familyId, spiderwebScore, hardness) {
  return {
    ...rec,
    tier,
    family_id: familyId,
    spiderweb_score: spiderwebScore,
    hardness,
  };
}

// ─── manifest ─────────────────────────────────────────────────────────────────

function buildManifest(opts, outputs, paths) {
  const checksums = {};
  for (const [name, filePath] of Object.entries(paths)) {
    if (existsSync(filePath)) {
      const data = readFileSync(filePath);
      checksums[name] = sha256Hex(data);
    }
  }

  return {
    generatedAt: new Date().toISOString(),
    sources: {
      spiderwebPath: opts.spiderwebPath,
      trainingData: opts.trainingData,
      packagePath: opts.packagePath ?? null,
      sankeyPath: opts.sankeyPath,
      wildcardPath: opts.wildcardPath,
    },
    counts: {
      tier1: outputs.tier1?.length ?? 0,
      tier2: outputs.tier2?.length ?? 0,
      tier3: outputs.tier3?.length ?? 0,
      tier4: outputs.tier4?.length ?? 0,
    },
    outputPaths: paths,
    checksums,
    tierDescriptions: {
      tier1: 'Base dataset: even-sampled across opening families, wildcard-seed anchored.',
      tier2: 'Family-conditioned: proportional per FrostMatrix family (G-matrix or spiderweb proxy).',
      tier3: 'Hard negatives: spiderweb-conditioned positions where the engine failed.',
      tier4: 'Transition surface: positions at family boundary transitions from sankey_data.',
    },
    schema: {
      addedFields: ['tier', 'family_id', 'spiderweb_score', 'hardness'],
      tier3ExtraFields: [
        'spiderweb_tier', 'spiderweb_source_type', 'spiderweb_support_role',
        'spiderweb_shared_families', 'spiderweb_shared_bridges', 'spiderweb_motifs',
        'symmetry_hotzone_signal', 'whitespace_defense_signal', 'heuristic_guard_mass',
        'pawn_gap_delta', 'adversarial_network_delta', 'queen_watch_core',
        'radial_crossfire', 'conditions_met',
      ],
      tier4ExtraFields: [
        'transition_from_family_id', 'transition_to_family_id',
        'transition_from_monster', 'transition_to_monster',
        'sankey_node_id', 'sankey_count', 'is_sankey_anchor',
      ],
    },
  };
}

// ─── logging ──────────────────────────────────────────────────────────────────

function log(msg) {
  process.stderr.write(`[tiered] ${msg}\n`);
}

// ─── main ─────────────────────────────────────────────────────────────────────

async function main() {
  const opts = parseArgs(process.argv.slice(2));

  // Validate required paths
  if (!existsSync(opts.spiderwebPath)) {
    process.stderr.write(`ERROR: spiderweb path not found: ${opts.spiderwebPath}\n`);
    process.exit(1);
  }
  if (!existsSync(opts.trainingData)) {
    process.stderr.write(`ERROR: training data not found: ${opts.trainingData}\n`);
    process.exit(1);
  }

  // Auto-detect queenweaver dir
  if (!opts.queenWeaverDir) {
    opts.queenWeaverDir = detectQueenWeaverDir(DEFAULT_QUEENWEAVER_RUNS_DIR);
    if (opts.queenWeaverDir) {
      log(`Auto-detected queenweaver dir: ${opts.queenWeaverDir}`);
    } else {
      log('No queenweaver dir detected — queen pressure signals will be empty.');
    }
  }

  // Load shared data
  log('Loading spiderweb records...');
  const spiderwebRows = await loadJsonl(opts.spiderwebPath);
  log(`  ${spiderwebRows.length} spiderweb records loaded.`);

  log('Loading wildcard opening seeds...');
  const wildcardRows = existsSync(opts.wildcardPath)
    ? await loadJsonl(opts.wildcardPath)
    : [];
  log(`  ${wildcardRows.length} wildcard seeds.`);

  log('Loading sankey data...');
  const sankeyData = existsSync(opts.sankeyPath) ? readJson(opts.sankeyPath) : {};
  log(`  sankey nodes=${sankeyData?.nodes?.length ?? 0}, links=${sankeyData?.links?.length ?? 0}`);

  // Load opening package if present
  let packageIndex = null;
  if (opts.packagePath && existsSync(opts.packagePath)) {
    log(`Loading opening package: ${opts.packagePath}`);
    const pkg = readJson(opts.packagePath);
    packageIndex = buildPackageFamilyIndex(pkg);
    log(`  Package loaded. nodeFamily entries: ${packageIndex?.nodeFamily?.size ?? 0}`);
  } else if (opts.packagePath) {
    log(`WARNING: --package-path specified but not found: ${opts.packagePath}`);
    log('Falling back to spiderweb proxy for tier-2 family conditioning.');
  }

  // Load queenweaver signals
  log('Loading queenweaver signals...');
  const queenSignals = await loadQueenWeaverSignals(opts.queenWeaverDir);
  log(`  ${queenSignals.size} FEN queen-pressure entries loaded.`);

  // Build requested tiers
  const outputs = {};

  if (opts.tiers.has('1')) {
    outputs.tier1 = await buildTier1(opts, wildcardRows);
  }
  if (opts.tiers.has('2')) {
    outputs.tier2 = await buildTier2(opts, spiderwebRows, packageIndex);
  }
  if (opts.tiers.has('3')) {
    outputs.tier3 = await buildTier3(opts, spiderwebRows, queenSignals, packageIndex);
  }
  if (opts.tiers.has('4')) {
    outputs.tier4 = await buildTier4(opts, sankeyData);
  }

  // Write outputs
  mkdirSync(opts.outDir, { recursive: true });
  const stamp = timestampLabel();

  const paths = {};

  const tierFileMap = {
    tier1: 'tier1_base.jsonl',
    tier2: 'tier2_family_conditioned.jsonl',
    tier3: 'tier3_hard_negatives.jsonl',
    tier4: 'tier4_transition_surface.jsonl',
  };

  for (const [key, filename] of Object.entries(tierFileMap)) {
    if (outputs[key] !== undefined) {
      const filePath = join(opts.outDir, filename);
      log(`Writing ${outputs[key].length} records → ${filePath}`);
      writeJsonl(filePath, outputs[key]);
      paths[key] = filePath;

      // Also write a stamped copy
      const stampedPath = join(opts.outDir, filename.replace('.jsonl', `-${stamp}.jsonl`));
      writeJsonl(stampedPath, outputs[key]);
    }
  }

  const manifest = buildManifest(opts, outputs, paths);
  const manifestPath = join(opts.outDir, 'manifest.json');
  writeFileSync(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`, 'utf8');
  paths.manifest = manifestPath;

  // Stdout summary
  const summary = {
    outDir: opts.outDir,
    tier1: outputs.tier1?.length ?? 'skipped',
    tier2: outputs.tier2?.length ?? 'skipped',
    tier3: outputs.tier3?.length ?? 'skipped',
    tier4: outputs.tier4?.length ?? 'skipped',
    manifest: manifestPath,
  };
  process.stdout.write(`${JSON.stringify(summary, null, 2)}\n`);
}

main().catch((err) => {
  process.stderr.write(`FATAL: ${err?.stack || err?.message || String(err)}\n`);
  process.exit(1);
});
