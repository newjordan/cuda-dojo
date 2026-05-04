#!/usr/bin/env node
import { readFileSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = resolve(__dirname, '..');
const DEFAULT_BOOK_SOURCE = resolve(REPO_ROOT, 'frostd4d', 'variants', 'the_un.js');
const DEFAULT_BLOBS = [
  'variants/razor_x.cuda_fighter_blob.json',
  'variants/queensguard.cuda_fighter_blob.json',
  'variants/firebird_src_dojo_runtime1.cuda_fighter_blob.json',
  'variants/fortress_src_dojo_runtime1.cuda_fighter_blob.json',
];

function fnv1a(text) {
  let h = 0x811c9dc5;
  for (let i = 0; i < text.length; i++) {
    h ^= text.charCodeAt(i);
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h;
}

function fenToKey(fen) {
  const parts = String(fen || '').trim().split(/\s+/);
  return `${parts[0]} ${parts[1]} ${parts[2]}`;
}

function extractBook(sourceText) {
  const entries = new Map();
  const re = /B\(\s*"([^"]+)"\s*,\s*"([^"]+)"\s*\)/g;
  let match;
  while ((match = re.exec(sourceText))) {
    entries.set(fnv1a(fenToKey(match[1])), match[2]);
  }
  return [...entries.entries()]
    .map(([hash, move]) => ({ hash, move }))
    .sort((a, b) => a.hash - b.hash || a.move.localeCompare(b.move));
}

function main() {
  const args = process.argv.slice(2);
  const blobs = args.length ? args : DEFAULT_BLOBS;
  const book = extractBook(readFileSync(DEFAULT_BOOK_SOURCE, 'utf8'));
  if (book.length === 0) throw new Error(`No Traincar book entries found in ${DEFAULT_BOOK_SOURCE}`);

  const changed = [];
  for (const blobRel of blobs) {
    const blobPath = resolve(REPO_ROOT, blobRel);
    const blob = JSON.parse(readFileSync(blobPath, 'utf8'));
    blob.openingBook = book;
    writeFileSync(blobPath, JSON.stringify(blob, null, 2) + '\n');
    changed.push(blobRel);
  }

  process.stdout.write(JSON.stringify({
    bookSource: DEFAULT_BOOK_SOURCE,
    entries: book.length,
    changed,
  }, null, 2) + '\n');
}

main();
