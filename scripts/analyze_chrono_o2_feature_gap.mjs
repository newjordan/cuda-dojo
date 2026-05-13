#!/usr/bin/env node
import fs from 'node:fs';
import path from 'node:path';

const SCHEMA_VERSION = 'dojo.chrono_o2_feature_gap_atlas.v1';

function usage() {
  return `Usage: node scripts/analyze_chrono_o2_feature_gap.mjs --atlas <chrono_o2_failure_atlas.json> [--out <feature_gap.json>]

Mine move/action feature gaps from an existing O2 failure atlas. This reads
recorded GPU-derived artifacts only; it does not generate legal moves, run
search, verify transitions, or promote runtime behavior.
`;
}

function parseArgs(argv) {
  const args = {
    atlas: null,
    out: null,
    conditionSource: '/home/frosty40/cuda_dojo/LOGIC_RAY_FRONTIER_RUNTIME_RECEIPT_2026-05-06.md',
    runLabel: 'posthoc_pzrg_chrono_o2_feature_gap_atlas',
  };
  for (let i = 0; i < argv.length; i += 1) {
    const token = argv[i];
    if (token === '--help' || token === '-h') {
      console.log(usage());
      process.exit(0);
    } else if (token === '--atlas') args.atlas = argv[++i];
    else if (token === '--out') args.out = argv[++i];
    else if (token === '--condition-source') args.conditionSource = argv[++i];
    else if (token === '--run-label') args.runLabel = argv[++i];
    else throw new Error(`unknown argument: ${token}\n${usage()}`);
  }
  if (!args.atlas) throw new Error(`missing --atlas\n${usage()}`);
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

function defaultOutPath(atlasPath) {
  const parsed = path.parse(atlasPath);
  const stem = parsed.base.replace(/\.json$/, '');
  return path.join(parsed.dir, `${stem}.feature_gap.json`);
}

function boardFromFen(fen) {
  const [boardPart = '', side = 'w'] = String(fen || '').trim().split(/\s+/);
  const board = new Map();
  const rows = boardPart.split('/');
  for (let row = 0; row < rows.length; row += 1) {
    let file = 0;
    const rank = 8 - row;
    for (const ch of rows[row]) {
      if (/\d/.test(ch)) {
        file += Number(ch);
      } else {
        const square = `${String.fromCharCode('a'.charCodeAt(0) + file)}${rank}`;
        board.set(square, ch);
        file += 1;
      }
    }
  }
  return { board, side };
}

function pieceName(piece) {
  if (!piece) return 'empty';
  const names = { p: 'pawn', n: 'knight', b: 'bishop', r: 'rook', q: 'queen', k: 'king' };
  return names[String(piece).toLowerCase()] || 'unknown';
}

function pieceColor(piece) {
  if (!piece) return 'none';
  return piece === String(piece).toUpperCase() ? 'white' : 'black';
}

function squareParts(square) {
  const text = String(square || '');
  if (!/^[a-h][1-8]$/.test(text)) return null;
  return {
    file: text.charCodeAt(0) - 'a'.charCodeAt(0),
    rank: Number(text[1]) - 1,
  };
}

function moveFeatures(fen, move) {
  const text = String(move || '0000');
  const parsed = /^[a-h][1-8][a-h][1-8][qrbn]?$/.test(text);
  if (!parsed) {
    return {
      move: text,
      validUciShape: false,
      piece: 'unknown',
      pieceColor: 'none',
      family: 'invalid',
    };
  }
  const { board, side } = boardFromFen(fen);
  const from = text.slice(0, 2);
  const to = text.slice(2, 4);
  const fromParts = squareParts(from);
  const toParts = squareParts(to);
  const piece = board.get(from) || null;
  const captured = board.get(to) || null;
  const dx = toParts.file - fromParts.file;
  const dy = toParts.rank - fromParts.rank;
  const sideSign = side === 'b' ? -1 : 1;
  const forward = sideSign * dy;
  const promotion = text.length === 5;
  const castle = pieceName(piece) === 'king' && Math.abs(dx) === 2;
  const center4 = ['d4', 'e4', 'd5', 'e5'].includes(to);
  const center16 = toParts.file >= 2 && toParts.file <= 5 && toParts.rank >= 2 && toParts.rank <= 5;
  const homeDeparture = (
    (side === 'w' && ['1', '2'].includes(from[1]))
    || (side === 'b' && ['7', '8'].includes(from[1]))
  );
  let family = 'quiet';
  if (castle) family = 'castle';
  else if (promotion) family = 'promotion';
  else if (captured) family = 'capture';
  return {
    move: text,
    validUciShape: true,
    side,
    from,
    to,
    piece: pieceName(piece),
    pieceColor: pieceColor(piece),
    capturedPiece: pieceName(captured),
    capturedColor: pieceColor(captured),
    family,
    capture: Boolean(captured),
    promotion,
    castle,
    dx,
    dy,
    forward,
    distance: round(Math.sqrt(dx ** 2 + dy ** 2)),
    center4,
    center16,
    homeDeparture,
    sameFile: dx === 0,
    sameRank: dy === 0,
    diagonal: Math.abs(dx) === Math.abs(dy),
  };
}

function inc(map, key, by = 1) {
  map.set(key, (map.get(key) || 0) + by);
}

function summarizeBoolean(items, getter) {
  if (!items.length) return { count: 0, trueCount: 0, rate: 0 };
  const trueCount = items.filter(getter).length;
  return {
    count: items.length,
    trueCount,
    rate: round(trueCount / items.length),
  };
}

function distribution(items, getter) {
  const counts = new Map();
  for (const item of items) inc(counts, getter(item));
  return Object.fromEntries([...counts.entries()].sort((a, b) => b[1] - a[1] || String(a[0]).localeCompare(String(b[0]))));
}

function mean(items, getter) {
  const values = items.map(getter).filter((value) => Number.isFinite(value));
  if (!values.length) return null;
  return round(values.reduce((sum, value) => sum + value, 0) / values.length);
}

function topPatternGaps(pairs) {
  const accepted = pairs.map((pair) => pair.accepted.features);
  const o2Top = pairs.map((pair) => pair.o2Top.features);
  const checks = [
    ['capture', (item) => item.capture],
    ['center4', (item) => item.center4],
    ['center16', (item) => item.center16],
    ['homeDeparture', (item) => item.homeDeparture],
    ['forwardPositive', (item) => item.forward > 0],
    ['backwardOrLevel', (item) => item.forward <= 0],
    ['diagonal', (item) => item.diagonal],
  ];
  return checks.map(([name, getter]) => {
    const a = summarizeBoolean(accepted, getter);
    const b = summarizeBoolean(o2Top, getter);
    return {
      feature: name,
      acceptedRate: a.rate,
      o2TopRate: b.rate,
      deltaAcceptedMinusO2Top: round(a.rate - b.rate),
      acceptedTrueCount: a.trueCount,
      o2TopTrueCount: b.trueCount,
    };
  }).sort((a, b) => Math.abs(b.deltaAcceptedMinusO2Top) - Math.abs(a.deltaAcceptedMinusO2Top));
}

function buildPairs(atlas) {
  const targets = Array.isArray(atlas.learningTargets) ? atlas.learningTargets : [];
  return targets.map((target) => {
    const acceptedMove = target.acceptedMove || target.accepted?.move;
    const o2TopMove = target.o2TopMove || target.o2Top?.move;
    return {
      rootId: target.rootId,
      class: target.class,
      rootFen: target.rootFen,
      baselineAcceptedBestRank: target.baselineAcceptedBestRank ?? null,
      o2AcceptedBestRank: target.o2AcceptedBestRank ?? null,
      acceptedMinusO2Score: target.acceptedMinusO2Score ?? null,
      accepted: {
        move: acceptedMove,
        features: moveFeatures(target.rootFen, acceptedMove),
      },
      o2Top: {
        move: o2TopMove,
        features: moveFeatures(target.rootFen, o2TopMove),
      },
      deltaN: target.deltaN || null,
      deltaFCont: target.deltaFCont || null,
    };
  }).filter((pair) => pair.accepted.move && pair.o2Top.move);
}

function main() {
  const args = parseArgs(process.argv.slice(2));
  const atlasPath = path.resolve(args.atlas);
  const atlas = readJson(atlasPath);
  const pairs = buildPairs(atlas);
  const acceptedFeatures = pairs.map((pair) => pair.accepted.features);
  const o2Features = pairs.map((pair) => pair.o2Top.features);
  const output = {
    schemaVersion: SCHEMA_VERSION,
    generatedAt: new Date().toISOString(),
    condition: {
      source: path.resolve(args.conditionSource),
      runLabel: args.runLabel,
      changedFields: 'post-hoc action-feature atlas over existing O2 failure atlas only',
      labCondition: 'posthoc/subset when input atlas is a subset',
      metric: 'feature gap between accepted O2 learning target moves and O2 top moves',
    },
    sources: {
      atlasPath,
      atlasSchemaVersion: atlas.schemaVersion || null,
      projectionSystem: atlas.projectionSystem || null,
      rootCount: atlas.summary?.rootCount ?? null,
      learningTargets: pairs.length,
    },
    summary: {
      learningTargetCount: pairs.length,
      classDistribution: distribution(pairs, (pair) => pair.class || 'unknown'),
      accepted: {
        pieceDistribution: distribution(acceptedFeatures, (item) => item.piece),
        familyDistribution: distribution(acceptedFeatures, (item) => item.family),
        capture: summarizeBoolean(acceptedFeatures, (item) => item.capture),
        center4: summarizeBoolean(acceptedFeatures, (item) => item.center4),
        center16: summarizeBoolean(acceptedFeatures, (item) => item.center16),
        homeDeparture: summarizeBoolean(acceptedFeatures, (item) => item.homeDeparture),
        forwardPositive: summarizeBoolean(acceptedFeatures, (item) => item.forward > 0),
        meanForward: mean(acceptedFeatures, (item) => item.forward),
        meanDistance: mean(acceptedFeatures, (item) => item.distance),
      },
      o2Top: {
        pieceDistribution: distribution(o2Features, (item) => item.piece),
        familyDistribution: distribution(o2Features, (item) => item.family),
        capture: summarizeBoolean(o2Features, (item) => item.capture),
        center4: summarizeBoolean(o2Features, (item) => item.center4),
        center16: summarizeBoolean(o2Features, (item) => item.center16),
        homeDeparture: summarizeBoolean(o2Features, (item) => item.homeDeparture),
        forwardPositive: summarizeBoolean(o2Features, (item) => item.forward > 0),
        meanForward: mean(o2Features, (item) => item.forward),
        meanDistance: mean(o2Features, (item) => item.distance),
      },
      featureGaps: topPatternGaps(pairs),
    },
    patternHypotheses: [
      'Add piece-family and capture/target occupancy channels to the O2 response basis before another K tensor learner.',
      'Separate home-rank development moves from already-developed piece moves in the temporal/action feature basis.',
      'Condition K response on center-target and forward/backward action classes instead of only normalized UCI geometry n.',
    ],
    pairs,
    promotionPolicy: {
      status: 'not_promoted',
      reason: 'feature-gap atlas is diagnostic only; it does not define a frozen tensor condition or rerun the heldout GPU gate',
      requiredNextEvidence: [
        'implement a new GPU-derived action feature basis using these pattern gaps',
        'rerun tensor learning with fold-heldout lift over frontier rank',
        'freeze the improved tensor condition and rerun heldout GPU gate before runtime promotion',
      ],
    },
  };
  const outPath = path.resolve(args.out || defaultOutPath(atlasPath));
  fs.mkdirSync(path.dirname(outPath), { recursive: true });
  fs.writeFileSync(outPath, `${JSON.stringify(output, null, 2)}\n`);
  console.log(JSON.stringify({
    ok: true,
    output: outPath,
    learningTargets: pairs.length,
    topFeatureGaps: output.summary.featureGaps.slice(0, 4),
    promote: false,
  }, null, 2));
}

try {
  main();
} catch (error) {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
}
