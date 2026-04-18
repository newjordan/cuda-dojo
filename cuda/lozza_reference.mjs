#!/usr/bin/env node
// =============================================================================
// lozza_reference.mjs
//
// VM-backed Lozza labeler exposing { bestmove, score_cp, depth, pv } from the
// engine's UCI info stream.
// =============================================================================

import { existsSync, readFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import vm from 'node:vm';

const __dirname = dirname(fileURLToPath(import.meta.url));
const LOZZA_RAW = resolve(__dirname, '..', 'trainers', 'lozza', 'lozza_raw.js');
const MATE_CP = 30000;

function buildSandbox(onLine) {
  return vm.createContext({
    postMessage(msg) {
      onLine(String(msg));
    },
    onmessage: null,
    self: {},
    console,
    Math, Date, Array, Object, String, Number, Boolean, Map, Set, WeakMap, WeakSet,
    RegExp, Error, TypeError, RangeError, SyntaxError, ReferenceError,
    JSON, parseInt, parseFloat, isNaN, isFinite, undefined, Infinity, NaN,
    Int8Array, Int16Array, Int32Array, Uint8Array, Uint16Array, Uint32Array,
    Float32Array, Float64Array, BigInt64Array, BigUint64Array,
    ArrayBuffer, SharedArrayBuffer, DataView, Symbol, Proxy, Reflect, Promise, BigInt,
    setTimeout, clearTimeout, setInterval, clearInterval,
    Buffer, TextEncoder, TextDecoder, URL, URLSearchParams,
    process: {
      stdout: { write() { return true; } },
      stderr: { write() { return true; } },
      exit() {},
      hrtime: process.hrtime,
    },
    globalThis: {},
    queueMicrotask,
    performance: { now: () => Date.now() },
    ...(typeof structuredClone !== 'undefined' ? { structuredClone } : {}),
    ...(typeof atob !== 'undefined' ? { atob, btoa } : {}),
  });
}

export class Lozza {
  constructor({ movetimeMs = 800 } = {}) {
    if (!existsSync(LOZZA_RAW)) throw new Error(`Lozza missing at ${LOZZA_RAW}`);
    this.movetimeMs = movetimeMs;
    this._collector = null;

    let lozzaCode = readFileSync(LOZZA_RAW, 'utf8');
    lozzaCode = lozzaCode.replace(
      /const nodeHost = \(typeof process\) != 'undefined';/,
      'const nodeHost = false;'
    );

    this.sandbox = buildSandbox((line) => this._handleLine(line));
    const script = new vm.Script(lozzaCode, { filename: 'lozza_raw.js' });
    script.runInContext(this.sandbox, { timeout: 30000 });
    this._send('uci');
    this._send('isready');
    this._send('ucinewgame');
  }

  _send(cmd) {
    if (typeof this.sandbox.onmessage === 'function') {
      this.sandbox.onmessage({ data: cmd });
    }
  }

  _handleLine(line) {
    if (this._collector) this._collector(line.trim());
  }

  async label(fen, movetimeMs = this.movetimeMs) {
    const state = {
      bestmove: '0000',
      score_cp: 0,
      depth: 0,
      pv: [],
    };

    const onLine = (line) => {
      if (!line) return;
      if (line.startsWith('info ')) {
        const info = line.match(/ depth (\d+).*? score (cp|mate) (-?\d+)(?: .* pv (.+))?/);
        if (info) {
          state.depth = parseInt(info[1], 10);
          if (info[2] === 'cp') {
            state.score_cp = parseInt(info[3], 10);
          } else {
            const matePly = parseInt(info[3], 10);
            state.score_cp = matePly < 0 ? -MATE_CP - matePly : MATE_CP - matePly;
          }
          state.pv = info[4] ? info[4].trim().split(/\s+/) : state.pv;
        }
      } else if (line.startsWith('bestmove')) {
        const parts = line.split(/\s+/);
        state.bestmove = parts[1] || state.bestmove;
      }
    };

    this._collector = onLine;
    this._send(`position fen ${fen}`);
    this._send(`go movetime ${movetimeMs}`);
    this._collector = null;
    return state;
  }

  close() {}
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const fen = process.argv[2] || 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';
  const lozza = new Lozza();
  lozza.label(fen).then((result) => {
    console.log(JSON.stringify(result, null, 2));
    lozza.close();
  }).catch((err) => {
    console.error(err);
    process.exit(1);
  });
}
