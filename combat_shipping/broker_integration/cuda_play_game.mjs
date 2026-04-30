// cuda_play_game.mjs — drop-in replacement for match-processor's
// playGame() that delegates the chess work to the CUDA referee
// (live_arbiter/live_match.py + dojo_ref) while keeping the wire-format
// of the return value byte-compatible with prod's sandboxed-referee.js.
//
// Returns: { result, reason, plies, moves, pgn, pgnResult }
//
//   - result, reason, plies, moves, pgnResult come from live_match.py
//     (the CUDA referee).
//   - pgn is built via prod's pgn-builder.js (SAN movetext, same headers
//     prod produces) so the broker's submit endpoint sees an
//     indistinguishable PGN format. This uses prod's chess-engine.js
//     for SAN disambiguation only — same code prod uses today.
//
// Selection at runtime: game-worker.js imports either this module or
// sandboxed-referee.js based on USE_CUDA_REFEREE env.

import { execFileSync } from 'node:child_process';
import { writeFileSync, readFileSync, mkdtempSync, rmSync, existsSync, statSync } from 'node:fs';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { randomUUID } from 'node:crypto';
import net from 'node:net';

const ARBITER_SRC = process.env.ARBITER_SRC
    || '/home/frosty40/AgentChess_Server/match-processor/src';
const LIVE_MATCH_PY = process.env.LIVE_MATCH_PY
    || '/home/frosty40/VC_V1/cuda-dojo-public/combat_shipping/live_arbiter/live_match.py';
const PYTHON = process.env.CUDA_PYTHON || 'python3';

// If a daemon socket is available we route through it (no per-game python +
// dojo_ref startup). Otherwise fall back to subprocess-per-game.
const ARBITER_SOCKET = process.env.CUDA_ARBITER_SOCKET || '/tmp/cuda_arbiter.sock';
function daemonAvailable() {
    try {
        if (!existsSync(ARBITER_SOCKET)) return false;
        const s = statSync(ARBITER_SOCKET);
        return s.isSocket();
    } catch { return false; }
}

// Lazy-load prod's PGN builder + legal-move generator (used only for
// SAN formatting of the move list — not for any chess decision).
const { buildPgnSync } = await import(`${ARBITER_SRC}/pgn-builder.js`);
const { generateLegalMoves } = await import(`${ARBITER_SRC}/chess-engine.js`);


function runViaSubprocess({ matchId, whiteName, blackName, wPath, bPath, jsonOut, maxPlies, moveTimeoutMs }) {
    const overallTimeoutMs = maxPlies * (moveTimeoutMs + 1000) + 30000;
    execFileSync(PYTHON, [
        LIVE_MATCH_PY,
        '--white', wPath,
        '--black', bPath,
        '--match-id', matchId,
        '--white-name', whiteName,
        '--black-name', blackName,
        '--max-plies', String(maxPlies),
        '--move-timeout-ms', String(moveTimeoutMs),
        '--out-json', jsonOut,
    ], {
        stdio: ['pipe', 'pipe', 'pipe'],
        timeout: overallTimeoutMs,
        maxBuffer: 16 * 1024 * 1024,
    });
    return JSON.parse(readFileSync(jsonOut, 'utf-8'));
}

function runViaDaemon({ matchId, whiteName, blackName, whiteLang, blackLang, wPath, bPath, maxPlies, moveTimeoutMs }) {
    return new Promise((resolve, reject) => {
        const sock = net.createConnection(ARBITER_SOCKET);
        const overallTimeoutMs = maxPlies * (moveTimeoutMs + 1000) + 30000;
        const timer = setTimeout(() => {
            sock.destroy(new Error(`daemon timeout after ${overallTimeoutMs}ms`));
        }, overallTimeoutMs);
        let buf = '';
        sock.setEncoding('utf-8');
        sock.on('connect', () => {
            const req = {
                id: 1,
                match_id: matchId,
                white: wPath, black: bPath,
                white_lang: whiteLang, black_lang: blackLang,
                white_name: whiteName, black_name: blackName,
                max_plies: maxPlies, move_timeout_ms: moveTimeoutMs,
            };
            sock.write(JSON.stringify(req) + '\n');
        });
        sock.on('data', (chunk) => {
            buf += chunk;
            const nl = buf.indexOf('\n');
            if (nl >= 0) {
                const line = buf.slice(0, nl);
                clearTimeout(timer);
                sock.end();
                try { resolve(JSON.parse(line)); } catch (e) { reject(e); }
            }
        });
        sock.on('error', (err) => { clearTimeout(timer); reject(err); });
        sock.on('close', () => {
            if (!buf || buf.indexOf('\n') < 0) {
                clearTimeout(timer);
                reject(new Error('daemon closed without response'));
            }
        });
    });
}

export async function playGame(opts) {
    const {
        matchId = randomUUID().slice(0, 8),
        whiteCode, whiteLang = 'js', whiteName = 'White',
        blackCode, blackLang = 'js', blackName = 'Black',
        maxPlies = 500,
        moveTimeoutMs = 5500,
    } = opts;

    const wExt = whiteLang === 'py' ? '.py'
        : (whiteCode.includes('require(') && !whiteCode.includes('import ')) ? '.js' : '.mjs';
    const bExt = blackLang === 'py' ? '.py'
        : (blackCode.includes('require(') && !blackCode.includes('import ')) ? '.js' : '.mjs';

    const dir = mkdtempSync(join(tmpdir(), 'cuda-arb-'));
    const wPath = join(dir, `white${wExt}`);
    const bPath = join(dir, `black${bExt}`);
    const jsonOut = join(dir, 'result.json');

    writeFileSync(wPath, whiteCode);
    writeFileSync(bPath, blackCode);

    let payload;
    const callArgs = { matchId, whiteName, blackName, whiteLang, blackLang, wPath, bPath, jsonOut, maxPlies, moveTimeoutMs };
    try {
        if (daemonAvailable()) {
            payload = await runViaDaemon(callArgs);
        } else {
            payload = runViaSubprocess(callArgs);
        }
    } catch (err) {
        const stderrStr = err.stderr ? err.stderr.toString() : '';
        const stdoutStr = err.stdout ? err.stdout.toString() : '';
        const msg = err.message || String(err);
        // eslint-disable-next-line no-console
        console.error(
            `[cuda_play_game] match=${matchId} dispatch error: msg=${msg} ` +
            `| stderr_tail=${stderrStr.slice(-1500).replace(/\n/g, ' | ')} ` +
            `| stdout_tail=${stdoutStr.slice(-200).replace(/\n/g, ' | ')}`
        );
        payload = { result: 'draw', reason: 'crash', plies: 0, moves: [], pgn_result: '1/2-1/2' };
    } finally {
        try { rmSync(dir, { recursive: true, force: true }); } catch {}
    }

    // Format SAN PGN identical to prod's output (broker submits this).
    const pgn = buildPgnSync({
        whiteName,
        blackName,
        moves: payload.moves || [],
        result: payload.pgn_result,
        reason: payload.reason,
    }, generateLegalMoves);

    return {
        result: payload.result,
        reason: payload.reason,
        plies: payload.plies,
        moves: payload.moves || [],
        pgn,
        pgnResult: payload.pgn_result,
    };
}

export default { playGame };
