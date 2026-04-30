// playgame_lifted.mjs
//
// Fork of match-processor/src/sandboxed-referee.js's playGame with the
// docker resource caps made env-tunable:
//   AGENT_CPUS         (default "0.5", prod-equivalent)
//   AGENT_MEMORY       (default "256m", prod-equivalent)
//   AGENT_PIDS_LIMIT   (default "32")
//   AGENT_TMPFS_SIZE   (default "10m")
//
// Rules come from prod's chess-engine.js — single source of truth. The
// only divergence vs. prod sandboxed-referee.js is `docker run` flag
// values. Used by combat_shipping's live comparison so the arbiter side
// of the head-to-head runs under the same lifted caps as the cuda side
// (`live_arbiter/live_match.py`).

import { execFileSync, execSync } from 'node:child_process';
import { writeFileSync, unlinkSync, mkdirSync } from 'node:fs';
import { join } from 'node:path';
import { randomUUID } from 'node:crypto';

const ARBITER_SRC = process.env.ARBITER_SRC
    || '/home/frosty40/AgentChess_Server/match-processor/src';

const {
    parseFen, boardToFen, applyUciMove, generateLegalMoves,
    isInCheck, getBoardKey, insufficientMaterial, STARTING_FEN,
} = await import(`${ARBITER_SRC}/chess-engine.js`);
const { buildPgnSync } = await import(`${ARBITER_SRC}/pgn-builder.js`);
const config = (await import(`${ARBITER_SRC}/config.js`)).default;

const AGENT_CPUS = process.env.AGENT_CPUS || '0.5';
const AGENT_MEMORY = process.env.AGENT_MEMORY || config.agentMemoryLimit || '256m';
const AGENT_PIDS_LIMIT = process.env.AGENT_PIDS_LIMIT || '32';
const AGENT_TMPFS_SIZE = process.env.AGENT_TMPFS_SIZE || '10m';

const MOVE_REGEX = /^[a-h][1-8][a-h][1-8][qrbn]?$/;

function startContainer(matchId, color, agentCode, language) {
    const containerName = `lifted-${matchId}-${color}`;
    let ext;
    if (language === 'py') {
        ext = '.py';
    } else {
        ext = agentCode.includes('require(') && !agentCode.includes('import ') ? '.js' : '.mjs';
    }
    const tmpFile = join(config.dataDir, `${containerName}${ext}`);
    mkdirSync(config.dataDir, { recursive: true });
    writeFileSync(tmpFile, agentCode);

    try {
        execSync([
            'docker', 'run', '-d',
            '--name', containerName,
            '--network', 'none',
            '--read-only',
            '--memory', AGENT_MEMORY,
            '--cpus', AGENT_CPUS,
            '--cap-drop', 'ALL',
            '--security-opt', 'no-new-privileges',
            '--pids-limit', AGENT_PIDS_LIMIT,
            '--tmpfs', `/tmp:size=${AGENT_TMPFS_SIZE},nodev,nosuid`,
            config.sandboxImage,
            'sleep', 'infinity',
        ].join(' '), { stdio: ['pipe', 'pipe', 'pipe'], timeout: 10000 });

        execFileSync('docker', [
            'exec', '-i', containerName,
            'sh', '-c', `cat > /tmp/agent${ext}`
        ], {
            input: agentCode,
            timeout: 5000,
            stdio: ['pipe', 'pipe', 'pipe'],
        });
    } finally {
        try { unlinkSync(tmpFile); } catch {}
    }

    return { containerName, ext };
}

function getAgentMove(containerName, fen, language, timeoutMs, ext) {
    const runtime = language === 'py' ? 'python3' : 'node';
    ext = ext || (language === 'py' ? '.py' : '.js');
    const timeoutSec = Math.ceil(timeoutMs / 1000);
    try {
        const raw = execFileSync('docker', [
            'exec', '-i', containerName,
            'timeout', String(timeoutSec),
            runtime, `/tmp/agent${ext}`
        ], {
            input: fen + '\n',
            encoding: 'utf-8',
            timeout: timeoutMs + 2000,
            stdio: ['pipe', 'pipe', 'pipe'],
            maxBuffer: 1024 * 1024,
        });
        return String(raw).trim();
    } catch (e) {
        if (e.killed || e.signal === 'SIGTERM' || e.status === 124) return '__TIMEOUT__';
        if (e.status === 137) return '__OOM__';
        return '__CRASH__';
    }
}

function stopContainer(containerName) {
    try {
        execSync(`docker rm -f ${containerName}`, {
            stdio: ['pipe', 'pipe', 'pipe'], timeout: 10000,
        });
    } catch {}
}

export function playGame(opts) {
    const {
        matchId = randomUUID().slice(0, 8),
        whiteCode, whiteLang = 'js', whiteName = 'White',
        blackCode, blackLang = 'js', blackName = 'Black',
        maxPlies = 500,
        moveTimeoutMs = config.agentMoveTimeoutMs,
    } = opts;

    const white = startContainer(matchId, 'white', whiteCode, whiteLang);
    const black = startContainer(matchId, 'black', blackCode, blackLang);

    try {
        let pos = parseFen(STARTING_FEN);
        const positionHistory = new Map();
        const moveLog = [];

        for (let ply = 0; ply < maxPlies; ply++) {
            const fen = boardToFen(pos);
            const isWhiteTurn = pos.side === 'w';
            const agent = isWhiteTurn ? white : black;
            const lang = isWhiteTurn ? whiteLang : blackLang;

            if (pos.halfmove >= 100) {
                return buildResult({ result: 'draw', reason: '50-move', plies: ply, moves: moveLog, whiteName, blackName });
            }
            if (insufficientMaterial(pos.board)) {
                return buildResult({ result: 'draw', reason: 'insufficient', plies: ply, moves: moveLog, whiteName, blackName });
            }
            const boardKey = getBoardKey(pos);
            const count = (positionHistory.get(boardKey) || 0) + 1;
            positionHistory.set(boardKey, count);
            if (count >= 3) {
                return buildResult({ result: 'draw', reason: 'threefold', plies: ply, moves: moveLog, whiteName, blackName });
            }

            const legalMoves = generateLegalMoves(pos);
            if (legalMoves.length === 0) {
                if (isInCheck(pos.board, pos.side)) {
                    const winner = isWhiteTurn ? 'black' : 'white';
                    return buildResult({ result: winner, reason: 'checkmate', plies: ply, moves: moveLog, whiteName, blackName });
                }
                return buildResult({ result: 'draw', reason: 'stalemate', plies: ply, moves: moveLog, whiteName, blackName });
            }

            let uci = getAgentMove(agent.containerName, fen, lang, moveTimeoutMs, agent.ext);

            if (uci === '__CRASH__' || uci === '__OOM__') {
                const agentCode = isWhiteTurn ? whiteCode : blackCode;
                stopContainer(agent.containerName);
                const fresh = startContainer(matchId + 'r', isWhiteTurn ? 'white' : 'black', agentCode, lang);
                if (isWhiteTurn) { Object.assign(white, fresh); } else { Object.assign(black, fresh); }
                uci = getAgentMove(fresh.containerName, fen, lang, moveTimeoutMs, fresh.ext);
            }

            if (uci === '__TIMEOUT__') {
                const winner = isWhiteTurn ? 'black' : 'white';
                return buildResult({ result: winner, reason: 'timeout', plies: ply, moves: moveLog, whiteName, blackName });
            }
            if (uci === '__CRASH__' || uci === '__OOM__') {
                const winner = isWhiteTurn ? 'black' : 'white';
                return buildResult({ result: winner, reason: uci === '__OOM__' ? 'oom' : 'crash', plies: ply, moves: moveLog, whiteName, blackName });
            }
            if (!MOVE_REGEX.test(uci)) {
                const winner = isWhiteTurn ? 'black' : 'white';
                return buildResult({ result: winner, reason: 'invalid_format', plies: ply, moves: moveLog, whiteName, blackName });
            }
            if (!legalMoves.includes(uci)) {
                const winner = isWhiteTurn ? 'black' : 'white';
                return buildResult({ result: winner, reason: 'illegal', plies: ply, moves: moveLog, whiteName, blackName });
            }

            moveLog.push(uci);
            pos = applyUciMove(pos, uci);
        }

        return buildResult({ result: 'draw', reason: 'max_plies', plies: maxPlies, moves: moveLog, whiteName, blackName });
    } finally {
        stopContainer(white.containerName);
        stopContainer(black.containerName);
    }
}

function buildResult({ result, reason, plies, moves, whiteName, blackName }) {
    let pgnResult;
    if (result === 'white') pgnResult = '1-0';
    else if (result === 'black') pgnResult = '0-1';
    else pgnResult = '1/2-1/2';

    const pgn = buildPgnSync({ whiteName, blackName, moves, result: pgnResult, reason }, generateLegalMoves);
    return { result, reason, plies, moves, pgn, pgnResult };
}

export default { playGame };
