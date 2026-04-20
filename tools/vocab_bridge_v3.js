'use strict';

/**
 * vocab_bridge_v3.js — Vocabulary bridge between the JS tokenizer stack and
 * transformer_v3_frostmatrix.cu.
 *
 * KEY INSIGHT (see docs/VOCAB_SIZE_PATCH.md for the full patch note):
 *
 *   The .cu hardcodes VOCAB_SIZE=513 claiming "13 piece types + 500 opening family IDs".
 *   Both numbers are wrong:
 *     - The opening vocab has 8,463 entries (not 500).
 *     - In V3 the opening context is carried by the 65-node family axis, which uses
 *       its own weight matrices (fax.family_embed, fax.unknown_embed).  Those are NOT
 *       looked up via token_embed.
 *     - token_embed therefore only needs to cover the 13 piece-type IDs used for the
 *       64 board-square tokens.
 *     - policy_W (the output head) is independent and should cover the full move
 *       target space, not the embedding vocabulary.
 *
 * V3 SEQUENCE LAYOUT (SEQ_LEN_V3 = 129):
 *
 *   positions [0 .. 64]  — 65 family-axis slots
 *     even positions (0, 2, 4, …, 64): family nodes   (family_id = slot/2, 0..32)
 *     odd  positions (1, 3, 5, …, 63): unknown nodes  (unknown_id = slot/2, 0..31)
 *     The .cu build_family_axis_kernel ignores the integer token values at these
 *     slots — it builds embeddings positionally from fax.family_embed/unknown_embed.
 *     The bridge still fills these slots with a meaningful integer (the matched
 *     openingTokenId or 0 for unmatched) so training code can use them if needed.
 *
 *   positions [65 .. 128] — 64 board-square tokens
 *     Piece-type IDs (0=empty, 1-6 white, 7-12 black) from FEN, a1→h1→…→a8→h8.
 *     These ARE looked up in token_embed.
 *
 * CORRECT CONSTANT VALUES (patch transformer_v3_frostmatrix.cu):
 *
 *   PIECE_VOCAB_SIZE   = 13    // input embedding for board squares (piece types only)
 *   POLICY_OUT_SIZE    = 4096  // output head (64 from-squares × 64 to-squares)
 *   OPENING_VOCAB_SIZE = 8463  // opening_branch_vocab.json entry count (for reference)
 *
 * Usage:
 *   const bridge = require('./vocab_bridge_v3');
 *   const { tokens } = bridge.encode({ fen, moveHistory });
 *   // tokens.length === 129  always
 */

const path = require('node:path');
const { loadOpeningBranchVocab, encodeOpeningBranch } = require('./opening_branch_encoder');
const { encodeFenTokens: encodeFenV1 } = require('./chess_transformer_tokenizer');

// ─── Constants ───────────────────────────────────────────────────────────────

const DEFAULT_VOCAB_PATH = '/srv/models-hdd/chess-games/opening_branch_vocab.json';

/** Number of family nodes in the V3 family axis (matching N_FAMILIES in .cu). */
const N_FAMILIES = 33;

/** Number of unknown-potential nodes (matching N_UNKNOWNS in .cu). */
const N_UNKNOWNS = 32;

/** Total family-axis slots = 33 families + 32 unknowns (matching N_FAM_NODES in .cu). */
const N_FAM_NODES = N_FAMILIES + N_UNKNOWNS; // 65

/** Board squares (matching N_BOARD_SQ in .cu). */
const N_BOARD_SQ = 64;

/** Full sequence length (matching SEQ_LEN_V3 in .cu). */
const SEQ_LEN_V3 = N_FAM_NODES + N_BOARD_SQ; // 129

/**
 * Correct input-embedding vocabulary size for V3.
 *
 * Board-square tokens are piece-type IDs 0–12 only.  In V3 the opening context
 * is handled by the separate family-axis weight matrices — it does NOT pass
 * through token_embed.  So token_embed only needs 13 rows.
 *
 * This is the CORRECT replacement for VOCAB_SIZE=513 in the .cu.
 */
const PIECE_VOCAB_SIZE = 13;

/**
 * Correct output-head vocabulary size (policy logits).
 *
 * V3 predicts a destination move as (from_square * 64 + to_square), covering
 * all 4 096 possible from→to combinations.  This is independent of the input
 * embedding size.
 */
const POLICY_OUT_SIZE = 4096; // 64 from × 64 to

/** Actual opening-family vocab size from opening_branch_vocab.json. */
const OPENING_VOCAB_SIZE = 8463; // loaded dynamically; this is the expected value

// ─── Vocab loader (lazy, singleton) ─────────────────────────────────────────

let _cachedVocab = null;
let _cachedVocabPath = null;

function getVocab(vocabPath) {
  const resolved = path.resolve(vocabPath || DEFAULT_VOCAB_PATH);
  if (_cachedVocab && _cachedVocabPath === resolved) return _cachedVocab;
  _cachedVocab = loadOpeningBranchVocab(resolved);
  _cachedVocabPath = resolved;
  return _cachedVocab;
}

// ─── Encoder ─────────────────────────────────────────────────────────────────

/**
 * Build the 65-element family-axis slot array for a given move history.
 *
 * The V3 family axis has 65 positional slots:
 *   slot 0, 2, 4, … 64 → family nodes (family_id = slot / 2, range 0..32)
 *   slot 1, 3, 5, … 63 → unknown-potential nodes (unknown_id = slot / 2, range 0..31)
 *
 * We fill each slot with a token value that reflects the opening context:
 *   - Family slots: the openingTokenId of the best-matched ancestor at that
 *     depth (or 0 for the root when no match exists).
 *   - Unknown slots: the interpolated openingTokenId between adjacent families
 *     (average of left and right family openingTokenIds), or 0 if unmatched.
 *
 * The .cu build_family_axis_kernel currently ignores these integer values and
 * builds embeddings positionally — but filling them correctly costs nothing
 * and makes the array useful for any future kernel that does use them.
 *
 * Returns an array of length N_FAM_NODES (65) containing integer token values.
 */
function buildFamilyAxisSlots(moveHistory, vocab) {
  const encoded = encodeOpeningBranch(moveHistory, vocab);
  const pathTokens = encoded.pathTokens || [];

  // Build a lookup: depth -> openingTokenId for matched prefixes
  const depthToTokenId = new Map();
  for (const entry of pathTokens) {
    depthToTokenId.set(entry.depth, entry.openingTokenId);
  }

  // Map matched depths onto family slots (family_id 0..32 → slot 0..64 step 2)
  // We spread matched path tokens across the 33 family positions proportionally.
  // If fewer than 33 path tokens matched, fill the remainder with the last matched id.
  const familyIds = new Array(N_FAMILIES).fill(0); // default = root family (id=0)
  const matched = pathTokens.slice(0, N_FAMILIES);
  for (let fi = 0; fi < matched.length; fi += 1) {
    familyIds[fi] = matched[fi].openingTokenId;
  }
  // Propagate last matched id forward (opening context persists)
  for (let fi = matched.length; fi < N_FAMILIES; fi += 1) {
    familyIds[fi] = matched.length > 0 ? matched[matched.length - 1].openingTokenId : 0;
  }

  // Interleave family and unknown slots
  const slots = new Array(N_FAM_NODES);
  for (let slot = 0; slot < N_FAM_NODES; slot += 1) {
    if (slot % 2 === 0) {
      // Family node
      slots[slot] = familyIds[slot / 2];
    } else {
      // Unknown-potential node: interpolate between adjacent family ids
      const leftId = familyIds[Math.floor(slot / 2)];
      const rightId = familyIds[Math.min(Math.ceil(slot / 2), N_FAMILIES - 1)];
      slots[slot] = Math.round((leftId + rightId) / 2);
    }
  }

  return {
    slots,
    matchedDepth: encoded.matchedDepth,
    longestToken: encoded.longestToken,
    pathTokens: encoded.pathTokens,
  };
}

/**
 * Encode a game record into the V3 token sequence.
 *
 * @param {object} record
 * @param {string}   record.fen          - FEN string for the current position.
 * @param {string[]|string} [record.moveHistory] - UCI move list up to this position
 *                                                  (e.g. ['e2e4','e7e5']).  May also
 *                                                  be a space-separated string.
 * @param {string}   [record.vocabPath]  - Override path to opening_branch_vocab.json.
 *
 * @returns {{
 *   tokens: number[],           // length = SEQ_LEN_V3 = 129
 *   familySlots: number[],      // length = N_FAM_NODES = 65  (positions 0..64)
 *   boardTokens: number[],      // length = N_BOARD_SQ  = 64  (positions 65..128)
 *   matchedDepth: number,       // deepest opening prefix matched (-1 = none)
 *   longestToken: object|null,  // vocab entry for the deepest match
 *   meta: object,               // FEN metadata (side, castling, ep, etc.)
 * }}
 */
function encode(record) {
  const { fen, vocabPath } = record;

  // Normalise moveHistory to string array
  let moveHistory = record.moveHistory || [];
  if (typeof moveHistory === 'string') {
    moveHistory = moveHistory.split(/[\s,]+/).map((s) => s.trim()).filter(Boolean);
  }

  if (!fen || typeof fen !== 'string') {
    throw new Error('vocab_bridge_v3.encode(): record.fen is required');
  }

  // ── Part 1: Board square tokens (positions 65..128) ──────────────────────
  const fenResult = encodeFenV1(fen);
  // squareTokens = 64 piece-type IDs (0-12); auxTokens = 7 aux IDs (13-39)
  // V3 board sequence uses ONLY the 64 square tokens (aux context is handled
  // structurally via FrostMatrix geometry and the family axis, not as sequence tokens)
  const boardTokens = fenResult.squareTokens; // length 64

  // ── Part 2: Family-axis slots (positions 0..64) ───────────────────────────
  const vocab = getVocab(vocabPath);
  const familyResult = buildFamilyAxisSlots(moveHistory, vocab);
  const familySlots = familyResult.slots; // length 65

  // ── Concatenate: [65 family | 64 board] = 129 ────────────────────────────
  const tokens = familySlots.concat(boardTokens);

  if (tokens.length !== SEQ_LEN_V3) {
    throw new Error(
      `vocab_bridge_v3: expected ${SEQ_LEN_V3} tokens, got ${tokens.length}`,
    );
  }

  return {
    tokens,
    familySlots,
    boardTokens,
    matchedDepth: familyResult.matchedDepth,
    longestToken: familyResult.longestToken,
    meta: fenResult.meta,
  };
}

// ─── Summary export ──────────────────────────────────────────────────────────

/**
 * Load the vocab and return a human-readable summary of the token space.
 * Useful for diagnostics and for verifying that the vocab file is accessible.
 *
 * @param {string} [vocabPath]
 * @returns {{
 *   vocab_size: number,
 *   token_types: object,
 *   encoding_description: string,
 * }}
 */
function summary(vocabPath) {
  const vocab = getVocab(vocabPath);

  return {
    vocab_size: vocab.entries.length,

    token_types: {
      piece_types: {
        count: PIECE_VOCAB_SIZE,
        range: '0–12',
        description:
          '0=empty square, 1=P 2=N 3=B 4=R 5=Q 6=K (white), ' +
          '7=p 8=n 9=b 10=r 11=q 12=k (black)',
        used_in: 'board-square tokens (sequence positions 65–128)',
      },
      opening_families: {
        count: vocab.entries.length,
        id_range: `0–${vocab.entries.length - 1}`,
        description:
          'Opening prefix tree entries from opening_branch_vocab.json. ' +
          'maxPly=' + vocab.maxPly + ', minCount=' + vocab.minCount,
        used_in:
          'family-axis slots (sequence positions 0–64) — ' +
          'family-axis embeddings are separate weight matrices in .cu (fax.family_embed / fax.unknown_embed), ' +
          'NOT looked up via token_embed',
      },
      policy_output: {
        count: POLICY_OUT_SIZE,
        description:
          'Policy head output logits: from_square * 64 + to_square (0–4095). ' +
          'Covers all 4 096 from→to square pairs.',
        used_in: 'policy_W output projection in .cu',
      },
    },

    encoding_description:
      `V3 sequence: [${N_FAM_NODES} family-axis slots | ${N_BOARD_SQ} board-square tokens] = ${SEQ_LEN_V3} tokens total. ` +
      `Family-axis even slots (0,2,…,64) = family nodes (openingTokenId of matched prefix). ` +
      `Family-axis odd slots (1,3,…,63) = unknown-potential interpolations. ` +
      `Board tokens (65–128) = piece-type IDs 0–12 for each square in a1→h1→…→a8→h8 order.`,

    // ── PATCH INFO ────────────────────────────────────────────────────────
    patch_required: {
      file: 'cuda/transformer_v3_frostmatrix.cu',
      wrong_line: '#define VOCAB_SIZE      513   // 13 piece types + 500 opening family IDs',
      replace_with: [
        `#define PIECE_VOCAB_SIZE   ${PIECE_VOCAB_SIZE}    // piece-type IDs for board-square token_embed (0=empty, 1-12=pieces)`,
        `#define OPENING_VOCAB_SIZE ${vocab.entries.length} // opening family entries in opening_branch_vocab.json`,
        `#define POLICY_OUT_SIZE    ${POLICY_OUT_SIZE}  // policy head output: from_sq*64+to_sq (all 64×64 move pairs)`,
      ],
      note:
        'token_embed must be sized PIECE_VOCAB_SIZE (13 rows). ' +
        'policy_W must be sized POLICY_OUT_SIZE (4096 cols). ' +
        'Opening family context is carried by the 65-node family axis via ' +
        'fax.family_embed[N_FAMILIES=33] and fax.unknown_embed[N_UNKNOWNS=32], ' +
        'which are independent of token_embed.',
    },
  };
}

// ─── Exports ─────────────────────────────────────────────────────────────────

module.exports = {
  // Constants
  N_FAMILIES,
  N_UNKNOWNS,
  N_FAM_NODES,
  N_BOARD_SQ,
  SEQ_LEN_V3,
  PIECE_VOCAB_SIZE,
  POLICY_OUT_SIZE,
  OPENING_VOCAB_SIZE,
  DEFAULT_VOCAB_PATH,

  // Functions
  encode,
  summary,

  // Internal helpers (exposed for testing)
  buildFamilyAxisSlots,
  getVocab,
};
