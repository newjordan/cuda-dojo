// =============================================================================
// reference.mjs — Stockfish-backed reference labeller for the oscillating book.
// =============================================================================
// Opens ONE long-lived Stockfish process, reuses it for every label request.
// Returns { bestmove, score_cp, depth, pv } per FEN. All scores are from the
// side-to-move's POV, in centipawns. Mate scores are converted to ±30000.
// =============================================================================
import { spawn } from 'node:child_process';
import { existsSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const SF_BIN = resolve(__dirname, '..', 'trainers', 'stockfish', 'stockfish_bin');
const MATE_CP = 30000;

export class Stockfish {
  constructor({ depth = 12, threads = 4, hashMB = 256 } = {}) {
    if (!existsSync(SF_BIN)) throw new Error(`Stockfish missing at ${SF_BIN}`);
    this.depth = depth;
    this.proc = spawn(SF_BIN, [], { stdio: ['pipe', 'pipe', 'pipe'] });
    this._buf = '';
    this._pending = null;
    this.proc.stdout.on('data', d => this._onData(d));
    this.proc.stderr.on('data', () => {}); // ignore
    this._send('uci');
    this._send(`setoption name Threads value ${threads}`);
    this._send(`setoption name Hash value ${hashMB}`);
    this._send('isready');
  }

  _send(line) { this.proc.stdin.write(line + '\n'); }

  _onData(chunk) {
    this._buf += chunk.toString();
    let idx;
    while ((idx = this._buf.indexOf('\n')) >= 0) {
      const line = this._buf.slice(0, idx).trim();
      this._buf = this._buf.slice(idx + 1);
      if (this._pending) this._pending.onLine(line);
    }
  }

  // Label one FEN. Returns { bestmove, score_cp (STM POV), depth, pv }.
  async label(fen, depth = this.depth) {
    return new Promise((resolve, reject) => {
      let lastScore = 0, lastDepth = 0, lastPv = [];
      let done = false;
      const onLine = (line) => {
        if (line.startsWith('info ') && line.includes(' depth ')) {
          const m = line.match(/ depth (\d+).*? score (cp|mate) (-?\d+)(?: .* pv (.+))?/);
          if (m) {
            lastDepth = parseInt(m[1]);
            if (m[2] === 'cp') lastScore = parseInt(m[3]);
            else lastScore = m[3].startsWith('-') ? -MATE_CP + parseInt(m[3]) : MATE_CP - parseInt(m[3]);
            lastPv = m[4] ? m[4].split(/\s+/) : lastPv;
          }
        } else if (line.startsWith('bestmove')) {
          if (done) return;
          done = true;
          const bm = line.split(/\s+/)[1];
          this._pending = null;
          resolve({ bestmove: bm, score_cp: lastScore, depth: lastDepth, pv: lastPv });
        }
      };
      this._pending = { onLine };
      this._send(`position fen ${fen}`);
      this._send(`go depth ${depth}`);
    });
  }

  // Label many FENs sequentially. Returns array parallel to input.
  async labelMany(fens, depth = this.depth, { onProgress } = {}) {
    const out = [];
    for (let i = 0; i < fens.length; i++) {
      out.push(await this.label(fens[i], depth));
      if (onProgress && (i % 10 === 0 || i === fens.length - 1)) onProgress(i + 1, fens.length);
    }
    return out;
  }

  close() {
    try { this._send('quit'); } catch {}
    this.proc.kill();
  }
}

// CLI smoke test: node reference.mjs "<fen>"
if (import.meta.url === `file://${process.argv[1]}`) {
  const fen = process.argv[2] || 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';
  const sf = new Stockfish({ depth: 12 });
  sf.label(fen).then(r => {
    console.log(JSON.stringify(r, null, 2));
    sf.close();
  }).catch(e => { console.error(e); sf.close(); process.exit(1); });
}
