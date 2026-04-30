#!/usr/bin/env node
// run_arbiter_native.mjs — single-game driver that calls the
// AgentChess match-processor's playGame() directly. We DO NOT modify
// arbiter source; we just import its modules and run one game.
//
// Usage:
//   node run_arbiter_native.mjs \
//     --white /path/whiteFighter.js \
//     --black /path/blackFighter.js \
//     --match-id abc \
//     --max-plies 500 \
//     --move-timeout-ms 5500 \
//     [--out-json out.json] [--out-pgn out.pgn]
//
// Outputs JSON with the same shape live_match.py does:
//   { match_id, result, pgn_result, reason, plies, moves,
//     wall_seconds, max_plies, referee:"arbiter", pgn }

import { readFileSync, writeFileSync } from 'node:fs';
import { basename } from 'node:path';

const ARBITER_SRC = process.env.ARBITER_SRC
    || '/home/frosty40/AgentChess_Server/match-processor/src';

// By default we import combat_shipping's lifted-caps fork (rules from
// prod chess-engine.js, docker caps env-tunable via AGENT_CPUS /
// AGENT_MEMORY) so the arbiter side runs under the same caps as the
// cuda side in head-to-head comparison. Set ARBITER_USE_PROD=1 to
// import prod's sandboxed-referee.js verbatim instead.
const useProd = process.env.ARBITER_USE_PROD === '1';
const playgameUrl = useProd
    ? `${ARBITER_SRC}/sandboxed-referee.js`
    : new URL('../bridges/playgame_lifted.mjs', import.meta.url).href;
const { playGame } = await import(playgameUrl);

function parseArgs(argv) {
    const args = {};
    for (let i = 2; i < argv.length; i++) {
        const k = argv[i];
        if (!k.startsWith('--')) continue;
        const key = k.slice(2);
        const val = argv[i + 1];
        args[key] = val;
        i++;
    }
    return args;
}

const a = parseArgs(process.argv);
const whitePath = a.white;
const blackPath = a.black;
const matchId = a['match-id'] || null;
const maxPlies = parseInt(a['max-plies'] || '500');
const moveTimeoutMs = parseInt(a['move-timeout-ms'] || '5500');
const outJson = a['out-json'] || null;
const outPgn = a['out-pgn'] || null;
const whiteName = a['white-name'] || basename(whitePath, '.js');
const blackName = a['black-name'] || basename(blackPath, '.js');

const whiteCode = readFileSync(whitePath, 'utf-8');
const blackCode = readFileSync(blackPath, 'utf-8');

const t0 = process.hrtime.bigint();
let res;
try {
    res = playGame({
        matchId: matchId || undefined,
        whiteCode, whiteLang: 'js', whiteName,
        blackCode, blackLang: 'js', blackName,
        maxPlies,
        moveTimeoutMs,
    });
} catch (err) {
    console.error(JSON.stringify({ error: err.message, stack: err.stack }));
    process.exit(2);
}
const t1 = process.hrtime.bigint();
const wallSeconds = Number(t1 - t0) / 1e9;

const payload = {
    match_id: matchId,
    result: res.result,
    pgn_result: res.pgnResult,
    reason: res.reason,
    plies: res.plies,
    moves: res.moves,
    wall_seconds: Math.round(wallSeconds * 1000) / 1000,
    max_plies: maxPlies,
    referee: 'arbiter',
};

if (outPgn) writeFileSync(outPgn, res.pgn);
if (outJson) writeFileSync(outJson, JSON.stringify(payload, null, 2));
else console.log(JSON.stringify(payload, null, 2));
