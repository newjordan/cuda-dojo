import { existsSync, readFileSync } from 'node:fs';
import { basename, dirname, join, resolve } from 'node:path';
import { createRequire } from 'node:module';
import { pathToFileURL } from 'node:url';
import vm from 'node:vm';

const nodeRequire = createRequire(import.meta.url);
const agentCache = new Map();
const IMPORT_RE = /^import\s+([\s\S]*?)\s+from\s+['"]([^'"]+)['"];?\s*$/gm;

function stripShebang(code) {
  return code.replace(/^#!.*\n/, '');
}

function replaceImportMeta(code, modulePath) {
  const metaUrl = pathToFileURL(modulePath).href;
  return code
    .replace(/import\.meta\.url/g, JSON.stringify(metaUrl))
    .replace(/import\.meta\.dirname/g, JSON.stringify(dirname(modulePath)))
    .replace(/import\.meta\.filename/g, JSON.stringify(modulePath));
}

function uniquePush(items, item) {
  if (!items.some((entry) => entry.key === item.key && entry.value === item.value)) {
    items.push(item);
  }
}

function collectExportEntries(code) {
  const exports = [];

  for (const match of code.matchAll(/^export\s+(?:const|let|var|function|class)\s+([A-Za-z_$][\w$]*)/gm)) {
    uniquePush(exports, { key: match[1], value: match[1] });
  }

  for (const match of code.matchAll(/^export\s*\{([\s\S]*?)\};?\s*$/gm)) {
    const parts = match[1].split(',').map((part) => part.trim()).filter(Boolean);
    for (const part of parts) {
      const aliasMatch = part.match(/^([A-Za-z_$][\w$]*)\s+as\s+([A-Za-z_$][\w$]*)$/);
      if (aliasMatch) {
        uniquePush(exports, { key: aliasMatch[2], value: aliasMatch[1] });
      } else {
        uniquePush(exports, { key: part, value: part });
      }
    }
  }

  return exports;
}

function stripExports(code) {
  return code
    .replace(/^export\s+const\s+/gm, 'const ')
    .replace(/^export\s+let\s+/gm, 'let ')
    .replace(/^export\s+var\s+/gm, 'var ')
    .replace(/^export\s+function\s+/gm, 'function ')
    .replace(/^export\s+class\s+/gm, 'class ')
    .replace(/^export\s+default\s+/gm, '')
    .replace(/^export\s*\{[\s\S]*?\};?\s*$/gm, '');
}

function resolveImportPath(fromPath, specifier) {
  let resolved = resolve(dirname(fromPath), specifier);
  if (existsSync(resolved)) return resolved;
  if (existsSync(`${resolved}.js`)) return `${resolved}.js`;
  const indexJs = join(resolved, 'index.js');
  if (existsSync(indexJs)) return indexJs;
  return resolved;
}

function buildImportBindings(specifierText, moduleExpr) {
  const text = specifierText.trim();
  const bindings = [];
  const bracesStart = text.indexOf('{');
  const bracesEnd = text.lastIndexOf('}');

  if (text.startsWith('* as ')) {
    return `const ${text.slice(5).trim()} = ${moduleExpr};`;
  }

  if (bracesStart !== -1 && bracesEnd !== -1 && bracesEnd > bracesStart) {
    const before = text.slice(0, bracesStart).replace(/,$/, '').trim();
    const inner = text.slice(bracesStart + 1, bracesEnd).trim();

    if (before) {
      bindings.push(`const ${before} = (${moduleExpr}.default ?? ${moduleExpr});`);
    }

    if (inner) {
      const props = inner
        .split(',')
        .map((part) => part.trim())
        .filter(Boolean)
        .map((part) => {
          const aliasMatch = part.match(/^([A-Za-z_$][\w$]*)\s+as\s+([A-Za-z_$][\w$]*)$/);
          return aliasMatch ? `${aliasMatch[1]}: ${aliasMatch[2]}` : part;
        });
      if (props.length) bindings.push(`const { ${props.join(', ')} } = ${moduleExpr};`);
    }

    return bindings.join('\n');
  }

  if (text) {
    return `const ${text} = (${moduleExpr}.default ?? ${moduleExpr});`;
  }

  return '';
}

function bundleModule(modulePath, state) {
  if (state.modules.has(modulePath)) return state.modules.get(modulePath);

  const moduleVar = `__dojo_mod_${state.nextModuleId++}`;
  state.modules.set(modulePath, moduleVar);

  let code = stripShebang(readFileSync(modulePath, 'utf8'));
  code = replaceImportMeta(code, modulePath);
  let prelude = '';

  code = code.replace(IMPORT_RE, (full, specifierText, source) => {
    const moduleExpr = source.startsWith('.')
      ? bundleModule(resolveImportPath(modulePath, source), state)
      : `require(${JSON.stringify(source)})`;
    const binding = buildImportBindings(specifierText, moduleExpr);
    if (binding) prelude += binding + '\n';
    return '';
  });

  const exportEntries = collectExportEntries(code);
  code = stripExports(code);
  const exportBody = exportEntries.map(({ key, value }) => `${key}: ${value}`).join(', ');

  state.chunks.push(`
const ${moduleVar} = (() => {
${prelude}${code}
return { ${exportBody} };
})();
`);

  return moduleVar;
}

function bundleSource(sourceCode, sourcePath) {
  const state = { modules: new Map(), chunks: [], nextModuleId: 0 };
  let code = stripShebang(sourceCode);
  code = replaceImportMeta(code, sourcePath);
  let prelude = '';

  code = code.replace(IMPORT_RE, (full, specifierText, source) => {
    const moduleExpr = source.startsWith('.')
      ? bundleModule(resolveImportPath(sourcePath, source), state)
      : `require(${JSON.stringify(source)})`;
    const binding = buildImportBindings(specifierText, moduleExpr);
    if (binding) prelude += binding + '\n';
    return '';
  });

  code = stripExports(code);

  return `${state.chunks.join('\n')}\n${prelude}${code}`;
}

function buildWrappedAgent(agentPath) {
  let code = stripShebang(readFileSync(agentPath, 'utf8'));

  const patternA = /^const fen = readFileSync\(0,\s*'utf8'\)\.trim\(\);/m;
  const idxA = code.search(patternA);

  let engineCode;
  let getMoveCall;

  if (idxA !== -1) {
    engineCode = code.substring(0, idxA);
    getMoveCall = `
  const start = performance.now();
  if (typeof currentFen !== 'undefined') currentFen = fen;
  const pos = parseFen(fen);
  const result = iterativeDeepening(pos);
  let bestMove = result && typeof result === 'object' ? result.move : result;
  const internalMetrics = result && typeof result === 'object' ? result.metrics : {};
  const internalTrace =
    result && typeof result === 'object' && result.trace
      ? result.trace
      : (typeof getDojoTrace === 'function' ? getDojoTrace() : undefined);

  if (!bestMove) {
    const legal = generateMoves(pos, false).filter(m => {
      const undo = makeMove(pos, m);
      const ok = !isInCheck(pos, 1 - pos.side);
      unmakeMove(pos, m, undo);
      return ok;
    });
    if (legal.length > 0) bestMove = legal[0];
  }
  const end = performance.now();
  return {
    move: bestMove ? (typeof bestMove === 'string' ? bestMove : moveToUci(bestMove)) : '0000',
    metrics: {
      ...internalMetrics,
      totalMs: end - start,
    },
    trace: internalTrace,
  };`;
  } else {
    const writeMatch = code.match(/process\.stdout\.write\([`'"]\$\{([^(]+)\(/);
    if (!writeMatch) return null;

    const funcName = writeMatch[1].trim();
    const mainIdx = code.search(/const fen\s*=/m);
    if (mainIdx <= 0) return null;

    engineCode = code.substring(0, mainIdx);
    getMoveCall = `
  const start = performance.now();
  const move = ${funcName}(fen);
  const end = performance.now();
  return {
    move,
    metrics: { totalMs: end - start },
    trace: typeof getDojoTrace === 'function' ? getDojoTrace() : undefined,
  };`;
  }

  engineCode = bundleSource(engineCode, agentPath)
    .replace(/^(module\.exports|export)\s*=\s*\{[^}]*\};?\s*$/gm, '');

  return `
(function() {
${engineCode}

return function __getMove(fen) {
${getMoveCall}
};
})();
`;
}

function buildSandboxDate(options = {}) {
  if (!options.deterministicClock) return Date;

  const RealDate = Date;
  const startMs = Number.isFinite(options.clockStartMs) ? options.clockStartMs : 1700000000000;
  const stepMs = Number.isFinite(options.clockStepMs) ? options.clockStepMs : 1;
  let tick = 0;

  function nextMs() {
    return startMs + stepMs * tick++;
  }

  class DojoDate extends RealDate {
    constructor(...args) {
      super(...(args.length ? args : [nextMs()]));
    }

    static now() {
      return nextMs();
    }

    static parse(value) {
      return RealDate.parse(value);
    }

    static UTC(...args) {
      return RealDate.UTC(...args);
    }
  }

  return DojoDate;
}

function createSandbox(options = {}) {
  const SandboxDate = buildSandboxDate(options);
  const clockStartMs = Number.isFinite(options.clockStartMs) ? options.clockStartMs : 1700000000000;
  return vm.createContext({
    readFileSync,
    require: nodeRequire,
    console: { log() {}, error() {}, warn() {}, info() {}, debug() {} },
    Math, Date: SandboxDate, Array, Object, String, Number, Boolean, Map, Set, WeakMap, WeakSet,
    RegExp, Error, TypeError, RangeError, SyntaxError, ReferenceError, URIError, EvalError,
    JSON, parseInt, parseFloat, isNaN, isFinite,
    undefined, Infinity, NaN,
    Int8Array, Int16Array, Int32Array, Uint8Array, Uint16Array, Uint32Array,
    Float32Array, Float64Array, BigInt64Array, BigUint64Array,
    ArrayBuffer, SharedArrayBuffer, DataView,
    Symbol, Proxy, Reflect, Promise, BigInt,
    setTimeout, clearTimeout, setInterval, clearInterval,
    Buffer, TextEncoder, TextDecoder, URL, URLSearchParams,
    process: {
      stdout: { write() { return true; } },
      stderr: { write() { return true; } },
      exit() {},
      env: {},
      argv: [],
    },
    performance: {
      now: options.deterministicClock
        ? () => SandboxDate.now() - clockStartMs
        : () => Date.now(),
    },
    globalThis: {},
    queueMicrotask,
    ...(typeof structuredClone !== 'undefined' ? { structuredClone } : {}),
    ...(typeof atob !== 'undefined' ? { atob, btoa } : {}),
  });
}

function compileAgent(agentPath, options = {}) {
  const { forceReload = false } = options;
  const cacheKey = JSON.stringify({
    agentPath,
    deterministicClock: Boolean(options.deterministicClock),
    clockStartMs: options.clockStartMs ?? null,
    clockStepMs: options.clockStepMs ?? null,
  });
  if (!forceReload && agentCache.has(cacheKey)) return agentCache.get(cacheKey);

  const wrappedCode = buildWrappedAgent(agentPath);
  if (!wrappedCode) {
    agentCache.set(cacheKey, null);
    return null;
  }

  try {
    const script = new vm.Script(wrappedCode, { filename: basename(agentPath) });
    const fn = script.runInContext(createSandbox(options), { timeout: 15000 });
    if (typeof fn !== 'function') {
      agentCache.set(cacheKey, null);
      return null;
    }
    agentCache.set(cacheKey, fn);
    return fn;
  } catch (error) {
    console.error(`[dojo_runtime] compile fail ${basename(agentPath)}: ${error.message}`);
    agentCache.set(cacheKey, null);
    return null;
  }
}

function resetAgentCache(agentPath = null) {
  if (!agentPath) {
    agentCache.clear();
    return;
  }
  for (const key of agentCache.keys()) {
    if (key === agentPath) {
      agentCache.delete(key);
      continue;
    }
    try {
      const parsed = JSON.parse(key);
      if (parsed?.agentPath === agentPath) agentCache.delete(key);
    } catch {
      if (String(key).includes(agentPath)) agentCache.delete(key);
    }
  }
}

function getMoveFromFn(fn, fen) {
  if (typeof fn !== 'function') return { move: '__FAIL__', metrics: { totalMs: 0 }, trace: null };
  try {
    const result = fn(fen);
    const move = typeof result === 'string' ? result : result.move;
    const metrics = typeof result === 'string' ? { totalMs: 0 } : result.metrics;
    const trace = typeof result === 'string' ? null : (result.trace || null);

    if (/^[a-h][1-8][a-h][1-8][qrbn]?$/.test(move)) return { move, metrics, trace };
    if (move === '0000') return { move, metrics, trace };
    return { move: '__FAIL__', metrics, trace };
  } catch {
    return { move: '__FAIL__', metrics: { totalMs: 0 }, trace: null };
  }
}

export { compileAgent, resetAgentCache, getMoveFromFn };
