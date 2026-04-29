#!/usr/bin/env node
// =============================================================================
// arbiter_referee.mjs — adapter that imports the AgentChess match-processor
// chess-engine.js without modifying it, and replays a UCI move list to verify
// each move's legality and the final terminal state.
//
// I/O: line-delimited JSON over stdin/stdout.
//   in:  {"id": int, "moves": ["e2e4", "e7e5", ...]}
//   out: {"id": int, "ok": bool, "plies_replayed": int,
//         "terminal": "checkmate"|"stalemate"|"threefold"|"fifty"|
//                     "insufficient"|"max_plies"|"undecided"|null,
//         "winner_after_terminal": "white"|"black"|"draw"|null,
//         "error": string|null}
//
// All chess work is performed by the arbiter's own chess-engine.js
// (CPU). This bridge only imports it and feeds it FENs/UCIs.
// =============================================================================

import { createInterface } from 'node:readline';
import { stdin, stdout } from 'node:process';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

// Path to arbiter source — read-only import. We don't modify it; we don't
// even bundle it into this package. Resolved via env var or default.
const __dirname = dirname(fileURLToPath(import.meta.url));
const ARBITER_SRC = process.env.ARBITER_SRC ||
    '/home/frosty40/AgentChess_Server/match-processor/src';

// Dynamic import so the path can be configured at runtime.
const chessEngineUrl = `file://${resolve(ARBITER_SRC, 'chess-engine.js')}`;
const ce = await import(chessEngineUrl);
const {
    parseFen, boardToFen, applyUciMove, generateLegalMoves,
    isInCheck, getBoardKey, insufficientMaterial, STARTING_FEN
} = ce;

function replayGame(uciMoves, maxPlies = 500) {
    let pos;
    try {
        pos = parseFen(STARTING_FEN);
    } catch (e) {
        return { ok: false, plies_replayed: 0, terminal: null,
                 winner_after_terminal: null, error: `parseFen: ${e.message}` };
    }
    const positionHistory = new Map();
    let plies = 0;

    for (const uci of uciMoves) {
        if (plies >= maxPlies) {
            return { ok: true, plies_replayed: plies, terminal: 'max_plies',
                     winner_after_terminal: 'draw', error: null };
        }

        // Pre-move terminal/draw checks (mirroring sandboxed-referee.js order)
        if (pos.halfmove >= 100) {
            return { ok: true, plies_replayed: plies, terminal: 'fifty',
                     winner_after_terminal: 'draw', error: null };
        }
        if (insufficientMaterial(pos.board)) {
            return { ok: true, plies_replayed: plies, terminal: 'insufficient',
                     winner_after_terminal: 'draw', error: null };
        }
        const key = getBoardKey(pos);
        const count = (positionHistory.get(key) || 0) + 1;
        positionHistory.set(key, count);
        if (count >= 3) {
            return { ok: true, plies_replayed: plies, terminal: 'threefold',
                     winner_after_terminal: 'draw', error: null };
        }
        const legalMoves = generateLegalMoves(pos);
        if (legalMoves.length === 0) {
            const inCheck = isInCheck(pos.board, pos.side);
            const winner = inCheck
                ? (pos.side === 'w' ? 'black' : 'white')
                : 'draw';
            return { ok: true, plies_replayed: plies,
                     terminal: inCheck ? 'checkmate' : 'stalemate',
                     winner_after_terminal: winner, error: null };
        }

        if (!legalMoves.includes(uci)) {
            return { ok: false, plies_replayed: plies, terminal: null,
                     winner_after_terminal: null,
                     error: `illegal move ${uci} at ply ${plies} ` +
                            `(side=${pos.side}, fen=${boardToFen(pos)})` };
        }

        try {
            pos = applyUciMove(pos, uci);
        } catch (e) {
            return { ok: false, plies_replayed: plies, terminal: null,
                     winner_after_terminal: null,
                     error: `applyUciMove(${uci}): ${e.message}` };
        }
        plies += 1;
    }

    // Post-move state — check terminal status one more time.
    // (Mirrors the start-of-next-iteration checks the arbiter would
    // have run if the game had continued.)
    if (pos.halfmove >= 100) {
        return { ok: true, plies_replayed: plies, terminal: 'fifty',
                 winner_after_terminal: 'draw', error: null };
    }
    if (insufficientMaterial(pos.board)) {
        return { ok: true, plies_replayed: plies, terminal: 'insufficient',
                 winner_after_terminal: 'draw', error: null };
    }
    const finalKey = getBoardKey(pos);
    const finalCount = (positionHistory.get(finalKey) || 0) + 1;
    if (finalCount >= 3) {
        return { ok: true, plies_replayed: plies, terminal: 'threefold',
                 winner_after_terminal: 'draw', error: null };
    }
    const legalAfter = generateLegalMoves(pos);
    if (legalAfter.length === 0) {
        const inCheck = isInCheck(pos.board, pos.side);
        const winner = inCheck
            ? (pos.side === 'w' ? 'black' : 'white')
            : 'draw';
        return { ok: true, plies_replayed: plies,
                 terminal: inCheck ? 'checkmate' : 'stalemate',
                 winner_after_terminal: winner, error: null };
    }
    return { ok: true, plies_replayed: plies, terminal: 'undecided',
             winner_after_terminal: null, error: null };
}

// JSON-line REPL.
const rl = createInterface({ input: stdin, terminal: false });
rl.on('line', (line) => {
    line = line.trim();
    if (!line) return;
    let req;
    try {
        req = JSON.parse(line);
    } catch (e) {
        stdout.write(JSON.stringify({ error: `bad json: ${e.message}` }) + '\n');
        return;
    }
    const out = replayGame(req.moves || [], req.max_plies || 500);
    out.id = req.id;
    stdout.write(JSON.stringify(out) + '\n');
});
rl.on('close', () => process.exit(0));

// Hello banner (so the harness knows we're ready).
process.stderr.write(
    `[arbiter_referee] loaded ${chessEngineUrl}\n` +
    `[arbiter_referee] ready\n`);
