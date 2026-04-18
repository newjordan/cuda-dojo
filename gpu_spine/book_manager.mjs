// =============================================================================
// book_manager.mjs — the oscillating book.
// =============================================================================
// A book is a set of positions with Stockfish reference labels. Labels are
// cached to disk (expensive to regenerate). The book has an "active window" —
// the working set currently being used by the training loop — plus a reserve
// pool that positions cycle in from.
// =============================================================================
// On-disk format (JSONL, one entry per line):
//   { "fen": "...", "sf_bestmove": "e2e4", "sf_score_cp": 48, "sf_depth": 12 }
// =============================================================================
import { readFileSync, writeFileSync, existsSync, appendFileSync, mkdirSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { Stockfish } from './reference.mjs';

const __dirname = dirname(fileURLToPath(import.meta.url));
const DEFAULT_BOOK_PATH = resolve(__dirname, 'book.jsonl');

function normalizeFen(raw) {
  const parts = raw.trim().split(/\s+/);
  if (parts.length >= 6) return parts.slice(0, 6).join(' ');
  if (parts.length === 4) return parts.join(' ') + ' 0 1';
  if (parts.length === 3) return parts.join(' ') + ' - 0 1';
  if (parts.length === 2) return parts.join(' ') + ' - - 0 1';
  throw new Error(`Unparseable FEN: ${raw}`);
}

export class Book {
  constructor(path = DEFAULT_BOOK_PATH) {
    this.path = path;
    this.entries = [];      // [{ fen, sf_bestmove, sf_score_cp, sf_depth }]
    this.byFen = new Map();
    if (existsSync(path)) this._load();
  }

  _load() {
    const lines = readFileSync(this.path, 'utf-8').split('\n').filter(Boolean);
    for (const line of lines) {
      try {
        const e = JSON.parse(line);
        if (!this.byFen.has(e.fen)) { this.entries.push(e); this.byFen.set(e.fen, e); }
      } catch { /* skip bad line */ }
    }
  }

  has(fen) { return this.byFen.has(normalizeFen(fen)); }
  get(fen) { return this.byFen.get(normalizeFen(fen)); }
  get size() { return this.entries.length; }

  // Add a fully-labeled entry. Appends to disk and memory.
  add(entry) {
    entry.fen = normalizeFen(entry.fen);
    if (this.byFen.has(entry.fen)) return false;
    this.entries.push(entry);
    this.byFen.set(entry.fen, entry);
    mkdirSync(dirname(this.path), { recursive: true });
    appendFileSync(this.path, JSON.stringify(entry) + '\n');
    return true;
  }

  // Label every FEN in `fens` that's not already in the book. Uses provided
  // Stockfish instance. Returns count added.
  async labelMissing(fens, sf, { depth = 12, onProgress } = {}) {
    const missing = [...new Set(fens.map(normalizeFen))].filter(f => !this.byFen.has(f));
    let added = 0;
    for (let i = 0; i < missing.length; i++) {
      const fen = missing[i];
      try {
        const r = await sf.label(fen, depth);
        this.add({
          fen,
          sf_bestmove: r.bestmove,
          sf_score_cp: r.score_cp,
          sf_depth: r.depth,
        });
        added++;
      } catch (e) {
        console.warn(`[book] label failed for ${fen}: ${e.message}`);
      }
      if (onProgress && (i % 5 === 0 || i === missing.length - 1)) {
        onProgress(i + 1, missing.length, added);
      }
    }
    return added;
  }

  // Return a working-set view (subset of entries). By default returns all.
  window(n = this.entries.length, offset = 0) {
    return this.entries.slice(offset, offset + n);
  }

  // Rotate: keep top-`keep` by disagreement (hardest), replace the bottom-N
  // with new entries from the reserve pool. `disagreement` is a parallel
  // array of |gpu_score - sf_score| per entry.
  rotate(disagreement, reservePool, keepFraction = 0.7) {
    if (disagreement.length !== this.entries.length) {
      throw new Error(`disagreement length ${disagreement.length} != book ${this.entries.length}`);
    }
    const keep = Math.floor(this.entries.length * keepFraction);
    const ranked = this.entries.map((e, i) => ({ e, d: disagreement[i] }))
                               .sort((a, b) => b.d - a.d)
                               .slice(0, keep);
    const newEntries = ranked.map(r => r.e);
    let added = 0;
    for (const e of reservePool) {
      if (newEntries.length >= this.entries.length) break;
      if (!newEntries.find(x => x.fen === e.fen)) { newEntries.push(e); added++; }
    }
    // Note: we don't persist rotation — book.jsonl is the master; rotation is
    // just working-set shaping in memory. If we want persistent curriculum,
    // write a separate "active_window.jsonl" snapshot.
    this.entries = newEntries;
    this.byFen = new Map(newEntries.map(e => [e.fen, e]));
    return { kept: keep, added };
  }
}

// CLI: seed the book from a FEN file (appends missing labels).
//   node book_manager.mjs seed <fen-file> [depth]
if (import.meta.url === `file://${process.argv[1]}`) {
  const [,, cmd, arg1, arg2] = process.argv;
  if (cmd === 'seed') {
    const file = arg1;
    const depth = parseInt(arg2 || '12');
    if (!file || !existsSync(file)) { console.error('usage: node book_manager.mjs seed <fen-file> [depth]'); process.exit(1); }
    const raw = readFileSync(file, 'utf-8').split('\n').filter(Boolean);
    const fens = raw.map(normalizeFen);
    const book = new Book();
    const sf = new Stockfish({ depth });
    console.log(`[seed] book currently has ${book.size} entries, ${fens.length} FENs incoming`);
    book.labelMissing(fens, sf, { depth, onProgress: (i, n, added) => {
      process.stdout.write(`\r[seed] labeled ${i}/${n} (${added} new)`);
    }}).then(added => {
      console.log(`\n[seed] done. added ${added}. book now has ${book.size} entries.`);
      sf.close();
    });
  } else if (cmd === 'stats') {
    const book = new Book();
    console.log(`book: ${book.size} entries at ${book.path}`);
    const depths = book.entries.map(e => e.sf_depth);
    if (depths.length) {
      console.log(`  sf_depth: min=${Math.min(...depths)} max=${Math.max(...depths)}`);
      const scores = book.entries.map(e => e.sf_score_cp);
      const sorted = [...scores].sort((a,b)=>a-b);
      console.log(`  scores: min=${sorted[0]} median=${sorted[sorted.length>>1]} max=${sorted[sorted.length-1]}`);
    }
  } else {
    console.log('usage:');
    console.log('  node book_manager.mjs seed <fen-file> [depth]');
    console.log('  node book_manager.mjs stats');
  }
}
