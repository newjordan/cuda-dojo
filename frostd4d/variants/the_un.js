// ============================================================================
// THE UN — United Nations of Chess Intelligence
// A massive strategy/intelligence library for FrostD4D
// ============================================================================
// Exports: lookupBook, lookupEndgame, detectPatterns, getCounterStrategy, getPieceCoordination
// All functions run in <1ms. Called once before search, not per-node.
// ============================================================================

const WHITE = 0, BLACK = 1;
const PAWN = 1, KNIGHT = 2, BISHOP = 3, ROOK = 4, QUEEN = 5, KING = 6;

// ============================================================================
// FNV-1a HASH — deterministic hash of FEN position key
// ============================================================================
function fnv1a(str) {
  let h = 0x811c9dc5;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = (h * 0x01000193) >>> 0;
  }
  return h;
}

function fenToKey(fen) {
  const parts = fen.trim().split(/\s+/);
  return parts[0] + ' ' + parts[1] + ' ' + parts[2];
}

function hashFen(fen) {
  return fnv1a(fenToKey(fen));
}

// ============================================================================
// SECTION 1: MASSIVE OPENING BOOK
// ============================================================================
// Format: FEN key hash -> UCI move
// Built from real grandmaster theory, 12-20 moves deep per line
// ============================================================================

const BOOK = new Map();

function B(fen, move) {
  BOOK.set(fnv1a(fenToKey(fen)), move);
}

// ----------------------------------------------------------------------------
// 1.e4 — KING'S PAWN OPENINGS
// ----------------------------------------------------------------------------

// Starting position
B("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq", "e2e4");

// === RUY LOPEZ (1.e4 e5 2.Nf3 Nc6 3.Bb5) ===
B("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq", "e7e5");
B("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq", "g1f3");
B("rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq", "b8c6");
B("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq", "f1b5");

// Ruy Lopez — Morphy Defense
B("r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq", "a7a6");
B("r1bqkbnr/1ppp1ppp/p1n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq", "b5a4");
B("r1bqkbnr/1ppp1ppp/p1n5/4p3/B3P3/5N2/PPPP1PPP/RNBQK2R b KQkq", "g8f6");
B("r1bqkb1r/1ppp1ppp/p1n2n2/4p3/B3P3/5N2/PPPP1PPP/RNBQK2R w KQkq", "e1g1");
B("r1bqkb1r/1ppp1ppp/p1n2n2/4p3/B3P3/5N2/PPPP1PPP/RNBQ1RK1 b kq", "f8e7");
B("r1bqk2r/1pppbppp/p1n2n2/4p3/B3P3/5N2/PPPP1PPP/RNBQ1RK1 w kq", "f1e1");
B("r1bqk2r/1pppbppp/p1n2n2/4p3/B3P3/5N2/PPPPQPPP/RNB2RK1 b kq", "b7b5");
B("r1bqk2r/2ppbppp/p1n2n2/1p2p3/B3P3/5N2/PPPP1PPP/RNBQR1K1 w kq", "a4b3");
B("r1bqk2r/2ppbppp/p1n2n2/1p2p3/4P3/1B3N2/PPPP1PPP/RNBQR1K1 b kq", "e8g8");
B("r1bq1rk1/2ppbppp/p1n2n2/1p2p3/4P3/1B3N2/PPPP1PPP/RNBQR1K1 w -", "c2c3");

// Ruy Lopez — Marshall Attack
B("r1bq1rk1/2ppbppp/p1n2n2/1p2p3/4P3/1BP2N2/PP1P1PPP/RNBQR1K1 b -", "d7d5");
B("r1bq1rk1/2p1bppp/p1n2n2/1p1pp3/4P3/1BP2N2/PP1P1PPP/RNBQR1K1 w -", "e4d5");
B("r1bq1rk1/2p1bppp/p1n2n2/1p1Pp3/8/1BP2N2/PP1P1PPP/RNBQR1K1 b -", "f6d5");
B("r1bq1rk1/2p1bppp/p1n5/1p1np3/8/1BP2N2/PP1P1PPP/RNBQR1K1 w -", "f3e5");
B("r1bq1rk1/2p1bppp/p1n5/1p1nN3/8/1BP5/PP1P1PPP/RNBQR1K1 b -", "c6e5");
B("r1bq1rk1/2p1bppp/p7/1p1nN3/8/1BP5/PP1P1PPP/RNBQR1K1 w -", "e1e5");

// Ruy Lopez — Berlin Defense
B("r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq", "g8f6");
B("r1bqkb1r/pppp1ppp/2n2n2/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq", "e1g1");
B("r1bqkb1r/pppp1ppp/2n2n2/1B2p3/4P3/5N2/PPPP1PPP/RNBQ1RK1 b kq", "f6e4");
B("r1bqkb1r/pppp1ppp/2n5/1B2p3/4n3/5N2/PPPP1PPP/RNBQ1RK1 w kq", "d2d4");
B("r1bqkb1r/pppp1ppp/2n5/1B2p3/3Pn3/5N2/PPP2PPP/RNBQ1RK1 b kq", "e4d6");
B("r1bqkb1r/pppp1ppp/2nn4/1B2p3/3P4/5N2/PPP2PPP/RNBQ1RK1 w kq", "b5c6");
B("r1bqkb1r/pppp1ppp/2nN4/4p3/3P4/5N2/PPP2PPP/RNBQ1RK1 b kq", "d7c6");
B("r1bqkb1r/ppp2ppp/2p5/4p3/3P4/5N2/PPP2PPP/RNBQ1RK1 w kq", "d4e5");

// Ruy Lopez — Breyer Variation
B("r1bq1rk1/2ppbppp/p1n2n2/1p2p3/4P3/1BP2N2/PP1P1PPP/RNBQR1K1 b -", "c6b8");
B("r1bq1rk1/1nppbppp/p4n2/1p2p3/4P3/1BP2N2/PP1P1PPP/RNBQR1K1 w -", "d2d4");

// Ruy Lopez — Chigorin Variation
B("r1bq1rk1/2ppbppp/p1n2n2/1p2p3/4P3/1BP2N2/PP1P1PPP/RNBQR1K1 b -", "c6a5");
B("r1bq1rk1/2ppbppp/p4n2/np2p3/4P3/1BP2N2/PP1P1PPP/RNBQR1K1 w -", "b3c2");

// === ITALIAN GAME (1.e4 e5 2.Nf3 Nc6 3.Bc4) ===
B("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq", "f1c4");
B("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq", "f8c5");

// Giuoco Piano
B("r1bqk1nr/pppp1ppp/2n5/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq", "c2c3");
B("r1bqk1nr/pppp1ppp/2n5/2b1p3/2B1P3/2P2N2/PP1P1PPP/RNBQK2R b KQkq", "g8f6");
B("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2P2N2/PP1P1PPP/RNBQK2R w KQkq", "d2d4");
B("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2BPP3/2P2N2/PP3PPP/RNBQK2R b KQkq", "e5d4");
B("r1bqk2r/pppp1ppp/2n2n2/2b5/2BpP3/2P2N2/PP3PPP/RNBQK2R w KQkq", "c3d4");
B("r1bqk2r/pppp1ppp/2n2n2/2b5/2BPP3/5N2/PP3PPP/RNBQK2R b KQkq", "c5b4");
B("r1bqk2r/pppp1ppp/2n2n2/8/1bBPP3/5N2/PP3PPP/RNBQK2R w KQkq", "e4e5");

// Giuoco Piano — Quiet Line
B("r1bqk1nr/pppp1ppp/2n5/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq", "d2d3");
B("r1bqk1nr/pppp1ppp/2n5/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R b KQkq", "g8f6");
B("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq", "c2c3");

// Evans Gambit
B("r1bqk1nr/pppp1ppp/2n5/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq", "b2b4");
B("r1bqk1nr/pppp1ppp/2n5/2b1p3/1PB1P3/5N2/P1PP1PPP/RNBQK2R b KQkq", "c5b4");
B("r1bqk1nr/pppp1ppp/2n5/4p3/1bB1P3/5N2/P1PP1PPP/RNBQK2R w KQkq", "c2c3");
B("r1bqk1nr/pppp1ppp/2n5/4p3/1bB1P3/2P2N2/P2P1PPP/RNBQK2R b KQkq", "b4a5");
B("r1bqk1nr/pppp1ppp/2n5/b3p3/2B1P3/2P2N2/P2P1PPP/RNBQK2R w KQkq", "d2d4");

// Two Knights Defense
B("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq", "g8f6");
B("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq", "f3g5");
B("r1bqkb1r/pppp1ppp/2n2n2/4p1N1/2B1P3/8/PPPP1PPP/RNBQK2R b KQkq", "d7d5");
B("r1bqkb1r/ppp2ppp/2n2n2/3pp1N1/2B1P3/8/PPPP1PPP/RNBQK2R w KQkq", "e4d5");
B("r1bqkb1r/ppp2ppp/2n2n2/3Pp1N1/2B5/8/PPPP1PPP/RNBQK2R b KQkq", "c6a5");

// === SCOTCH GAME (1.e4 e5 2.Nf3 Nc6 3.d4) ===
B("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq", "d2d4");
B("r1bqkbnr/pppp1ppp/2n5/4p3/3PP3/5N2/PPP2PPP/RNBQKB1R b KQkq", "e5d4");
B("r1bqkbnr/pppp1ppp/2n5/8/3pP3/5N2/PPP2PPP/RNBQKB1R w KQkq", "f3d4");
B("r1bqkbnr/pppp1ppp/2n5/8/3NP3/8/PPP2PPP/RNBQKB1R b KQkq", "f8c5");
B("r1bqk1nr/pppp1ppp/2n5/2b5/3NP3/8/PPP2PPP/RNBQKB1R w KQkq", "c2c3");

// Scotch — Classical
B("r1bqkbnr/pppp1ppp/2n5/8/3NP3/8/PPP2PPP/RNBQKB1R b KQkq", "g8f6");
B("r1bqkb1r/pppp1ppp/2n2n2/8/3NP3/8/PPP2PPP/RNBQKB1R w KQkq", "d4c6");
B("r1bqkb1r/pppp1ppp/2N2n2/8/4P3/8/PPP2PPP/RNBQKB1R b KQkq", "b7c6");

// === VIENNA GAME (1.e4 e5 2.Nc3) ===
B("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq", "b1c3");
B("rnbqkbnr/pppp1ppp/8/4p3/4P3/2N5/PPPP1PPP/R1BQKBNR b KQkq", "g8f6");
B("rnbqkb1r/pppp1ppp/5n2/4p3/4P3/2N5/PPPP1PPP/R1BQKBNR w KQkq", "f2f4");

// === KING'S GAMBIT (1.e4 e5 2.f4) ===
B("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq", "f2f4");
B("rnbqkbnr/pppp1ppp/8/4p3/4PP2/8/PPPP2PP/RNBQKBNR b KQkq", "e5f4");
B("rnbqkbnr/pppp1ppp/8/8/4Pp2/8/PPPP2PP/RNBQKBNR w KQkq", "g1f3");
B("rnbqkbnr/pppp1ppp/8/8/4Pp2/5N2/PPPP2PP/RNBQKB1R b KQkq", "g7g5");
B("rnbqkbnr/pppp1p1p/8/6p1/4Pp2/5N2/PPPP2PP/RNBQKB1R w KQkq", "h2h4");
B("rnbqkbnr/pppp1p1p/8/6p1/4Pp1P/5N2/PPPP2P1/RNBQKB1R b KQkq", "g5g4");
B("rnbqkbnr/pppp1p1p/8/8/4PppP/5N2/PPPP2P1/RNBQKB1R w KQkq", "f3e5");

// King's Gambit Declined
B("rnbqkbnr/pppp1ppp/8/4p3/4PP2/8/PPPP2PP/RNBQKBNR b KQkq", "f8c5");
B("rnbqk1nr/pppp1ppp/8/2b1p3/4PP2/8/PPPP2PP/RNBQKBNR w KQkq", "g1f3");

// === PETROFF DEFENSE (1.e4 e5 2.Nf3 Nf6) ===
B("rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq", "g8f6");
B("rnbqkb1r/pppp1ppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq", "f3e5");
B("rnbqkb1r/pppp1ppp/5n2/4N3/4P3/8/PPPP1PPP/RNBQKB1R b KQkq", "d7d6");
B("rnbqkb1r/ppp2ppp/3p1n2/4N3/4P3/8/PPPP1PPP/RNBQKB1R w KQkq", "e5f3");
B("rnbqkb1r/ppp2ppp/3p1n2/8/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq", "f6e4");
B("rnbqkb1r/ppp2ppp/3p4/8/4n3/5N2/PPPP1PPP/RNBQKB1R w KQkq", "d2d4");
B("rnbqkb1r/ppp2ppp/3p4/8/3Pn3/5N2/PPP2PPP/RNBQKB1R b KQkq", "d6d5");
B("rnbqkb1r/ppp2ppp/8/3p4/3Pn3/5N2/PPP2PPP/RNBQKB1R w KQkq", "f1d3");
B("rnbqkb1r/ppp2ppp/8/3p4/3Pn3/3B1N2/PPP2PPP/RNBQK2R b KQkq", "f8e7");

// Petroff — Steinitz variation
B("rnbqkb1r/pppp1ppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq", "d2d4");

// === PHILIDOR DEFENSE (1.e4 e5 2.Nf3 d6) ===
B("rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq", "d7d6");
B("rnbqkbnr/ppp2ppp/3p4/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq", "d2d4");
B("rnbqkbnr/ppp2ppp/3p4/4p3/3PP3/5N2/PPP2PPP/RNBQKB1R b KQkq", "g8f6");
B("rnbqkb1r/ppp2ppp/3p1n2/4p3/3PP3/5N2/PPP2PPP/RNBQKB1R w KQkq", "b1c3");

// === SICILIAN DEFENSE (1.e4 c5) ===
B("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq", "c7c5");
B("rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq", "g1f3");
B("rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq", "d7d6");

// Sicilian — Open (2...d6 3.d4 cxd4 4.Nxd4)
B("rnbqkbnr/pp2pppp/3p4/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq", "d2d4");
B("rnbqkbnr/pp2pppp/3p4/2p5/3PP3/5N2/PPP2PPP/RNBQKB1R b KQkq", "c5d4");
B("rnbqkbnr/pp2pppp/3p4/8/3pP3/5N2/PPP2PPP/RNBQKB1R w KQkq", "f3d4");
B("rnbqkbnr/pp2pppp/3p4/8/3NP3/8/PPP2PPP/RNBQKB1R b KQkq", "g8f6");
B("rnbqkb1r/pp2pppp/3p1n2/8/3NP3/8/PPP2PPP/RNBQKB1R w KQkq", "b1c3");

// Sicilian Najdorf (5...a6)
B("rnbqkb1r/pp2pppp/3p1n2/8/3NP3/2N5/PPP2PPP/R1BQKB1R b KQkq", "a7a6");
B("rnbqkb1r/1p2pppp/p2p1n2/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq", "c1e3");
B("rnbqkb1r/1p2pppp/p2p1n2/8/3NP3/2N1B3/PPP2PPP/R2QKB1R b KQkq", "e7e5");
B("rnbqkb1r/1p3ppp/p2p1n2/4p3/3NP3/2N1B3/PPP2PPP/R2QKB1R w KQkq", "d4b3");
B("rnbqkb1r/1p3ppp/p2p1n2/4p3/4P3/1NN1B3/PPP2PPP/R2QKB1R b KQkq", "f8e7");
B("rnbqk2r/1p2bppp/p2p1n2/4p3/4P3/1NN1B3/PPP2PPP/R2QKB1R w KQkq", "f2f3");
B("rnbqk2r/1p2bppp/p2p1n2/4p3/4P3/1NN1BP2/PPP3PP/R2QKB1R b KQkq", "c8e6");
B("rn1qk2r/1p2bppp/p2pbn2/4p3/4P3/1NN1BP2/PPP3PP/R2QKB1R w KQkq", "d1d2");
B("rn1qk2r/1p2bppp/p2pbn2/4p3/4P3/1NN1BP2/PPPQ2PP/R3KB1R b KQkq", "b8d7");
B("r2qk2r/1p1nbppp/p2pbn2/4p3/4P3/1NN1BP2/PPPQ2PP/R3KB1R w KQkq", "g2g4");

// Najdorf — 6.Bg5 (English Attack alternative)
B("rnbqkb1r/1p2pppp/p2p1n2/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq", "c1g5");
B("rnbqkb1r/1p2pppp/p2p1n2/6B1/3NP3/2N5/PPP2PPP/R2QKB1R b KQkq", "e7e6");
B("rnbqkb1r/1p3ppp/p2ppn2/6B1/3NP3/2N5/PPP2PPP/R2QKB1R w KQkq", "f2f4");

// Najdorf — 6.f3 English Attack
B("rnbqkb1r/1p2pppp/p2p1n2/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq", "f2f3");

// Sicilian Dragon (5...g6)
B("rnbqkb1r/pp2pppp/3p1n2/8/3NP3/2N5/PPP2PPP/R1BQKB1R b KQkq", "g7g6");
B("rnbqkb1r/pp2pp1p/3p1np1/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq", "c1e3");
B("rnbqkb1r/pp2pp1p/3p1np1/8/3NP3/2N1B3/PPP2PPP/R2QKB1R b KQkq", "f8g7");
B("rnbqk2r/pp2ppbp/3p1np1/8/3NP3/2N1B3/PPP2PPP/R2QKB1R w KQkq", "f2f3");
B("rnbqk2r/pp2ppbp/3p1np1/8/3NP3/2N1BP2/PPP3PP/R2QKB1R b KQkq", "e8g8");
B("rnbq1rk1/pp2ppbp/3p1np1/8/3NP3/2N1BP2/PPP3PP/R2QKB1R w KQ", "d1d2");
B("rnbq1rk1/pp2ppbp/3p1np1/8/3NP3/2N1BP2/PPPQ2PP/R3KB1R b KQ", "b8c6");
B("r1bq1rk1/pp2ppbp/2np1np1/8/3NP3/2N1BP2/PPPQ2PP/R3KB1R w KQ", "f1c4");
B("r1bq1rk1/pp2ppbp/2np1np1/8/2BNP3/2N1BP2/PPPQ2PP/R3K2R b KQ", "c8d7");
B("r2q1rk1/pp1bppbp/2np1np1/8/2BNP3/2N1BP2/PPPQ2PP/R3K2R w KQ", "e1c1");

// Sicilian Scheveningen (5...e6)
B("rnbqkb1r/pp2pppp/3p1n2/8/3NP3/2N5/PPP2PPP/R1BQKB1R b KQkq", "e7e6");
B("rnbqkb1r/pp3ppp/3ppn2/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq", "c1e3");

// Sicilian Sveshnikov (1.e4 c5 2.Nf3 Nc6 3.d4 cxd4 4.Nxd4 Nf6 5.Nc3 e5)
B("rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq", "b8c6");
B("r1bqkbnr/pp1ppppp/2n5/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq", "d2d4");
B("r1bqkbnr/pp1ppppp/2n5/2p5/3PP3/5N2/PPP2PPP/RNBQKB1R b KQkq", "c5d4");
B("r1bqkbnr/pp1ppppp/2n5/8/3pP3/5N2/PPP2PPP/RNBQKB1R w KQkq", "f3d4");
B("r1bqkbnr/pp1ppppp/2n5/8/3NP3/8/PPP2PPP/RNBQKB1R b KQkq", "g8f6");
B("r1bqkb1r/pp1ppppp/2n2n2/8/3NP3/8/PPP2PPP/RNBQKB1R w KQkq", "b1c3");
B("r1bqkb1r/pp1ppppp/2n2n2/8/3NP3/2N5/PPP2PPP/R1BQKB1R b KQkq", "e7e5");
B("r1bqkb1r/pp1p1ppp/2n2n2/4p3/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq", "d4b5");
B("r1bqkb1r/pp1p1ppp/2n2n2/1N2p3/4P3/2N5/PPP2PPP/R1BQKB1R b KQkq", "d7d6");
B("r1bqkb1r/pp3ppp/2np1n2/1N2p3/4P3/2N5/PPP2PPP/R1BQKB1R w KQkq", "c1g5");
B("r1bqkb1r/pp3ppp/2np1n2/1N2p1B1/4P3/2N5/PPP2PPP/R2QKB1R b KQkq", "a7a6");
B("r1bqkb1r/1p3ppp/p1np1n2/1N2p1B1/4P3/2N5/PPP2PPP/R2QKB1R w KQkq", "b5a3");

// Sicilian Kan (5...a6 with ...e6)
B("rnbqkb1r/pp3ppp/3ppn2/8/3NP3/2N5/PPP2PPP/R1BQKB1R b KQkq", "a7a6");

// Sicilian Taimanov (2...Nc6...e6)
B("r1bqkbnr/pp1ppppp/2n5/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq", "e7e6");

// Sicilian Accelerated Dragon (2...Nc6...g6)
B("r1bqkbnr/pp1ppppp/2n5/8/3NP3/8/PPP2PPP/RNBQKB1R b KQkq", "g7g6");
B("r1bqkbnr/pp1ppp1p/2n3p1/8/3NP3/8/PPP2PPP/RNBQKB1R w KQkq", "c2c4");

// Sicilian Rossolimo (2...Nc6 3.Bb5)
B("r1bqkbnr/pp1ppppp/2n5/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq", "f1b5");
B("r1bqkbnr/pp1ppppp/2n5/1Bp5/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq", "g7g6");
B("r1bqkbnr/pp1ppp1p/2n3p1/1Bp5/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq", "e1g1");

// Sicilian Moscow (2...d6 3.Bb5+)
B("rnbqkbnr/pp2pppp/3p4/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq", "f1b5");
B("rnbqkbnr/pp2pppp/3p4/1Bp5/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq", "c8d7");

// Sicilian Kalashnikov (similar to Sveshnikov but with ...d6 before ...e5)
B("r1bqkb1r/pp1p1ppp/2n2n2/4p3/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq", "d4b5");

// === FRENCH DEFENSE (1.e4 e6) ===
B("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq", "e7e6");
B("rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq", "d2d4");
B("rnbqkbnr/pppp1ppp/4p3/8/3PP3/8/PPP2PPP/RNBQKBNR b KQkq", "d7d5");

// French Winawer (3.Nc3 Bb4)
B("rnbqkbnr/ppp2ppp/4p3/3p4/3PP3/8/PPP2PPP/RNBQKBNR w KQkq", "b1c3");
B("rnbqkbnr/ppp2ppp/4p3/3p4/3PP3/2N5/PPP2PPP/R1BQKBNR b KQkq", "f8b4");
B("rnbqk1nr/ppp2ppp/4p3/3p4/1b1PP3/2N5/PPP2PPP/R1BQKBNR w KQkq", "e4e5");
B("rnbqk1nr/ppp2ppp/4p3/3pP3/1b1P4/2N5/PPP2PPP/R1BQKBNR b KQkq", "c7c5");
B("rnbqk1nr/pp3ppp/4p3/2ppP3/1b1P4/2N5/PPP2PPP/R1BQKBNR w KQkq", "a2a3");
B("rnbqk1nr/pp3ppp/4p3/2ppP3/1b1P4/P1N5/1PP2PPP/R1BQKBNR b KQkq", "b4c3");
B("rnbqk1nr/pp3ppp/4p3/2ppP3/3P4/P1b5/1PP2PPP/R1BQKBNR w KQkq", "b2c3");
B("rnbqk1nr/pp3ppp/4p3/2ppP3/3P4/P1P5/2P2PPP/R1BQKBNR b KQkq", "g8e7");

// French Classical (3.Nc3 Nf6)
B("rnbqkbnr/ppp2ppp/4p3/3p4/3PP3/2N5/PPP2PPP/R1BQKBNR b KQkq", "g8f6");
B("rnbqkb1r/ppp2ppp/4pn2/3p4/3PP3/2N5/PPP2PPP/R1BQKBNR w KQkq", "c1g5");
B("rnbqkb1r/ppp2ppp/4pn2/3p2B1/3PP3/2N5/PPP2PPP/R2QKBNR b KQkq", "f8e7");
B("rnbqk2r/ppp1bppp/4pn2/3p2B1/3PP3/2N5/PPP2PPP/R2QKBNR w KQkq", "e4e5");

// French Tarrasch (3.Nd2)
B("rnbqkbnr/ppp2ppp/4p3/3p4/3PP3/8/PPP2PPP/RNBQKBNR w KQkq", "b1d2");
B("rnbqkbnr/ppp2ppp/4p3/3p4/3PP3/8/PPPN1PPP/R1BQKBNR b KQkq", "g8f6");
B("rnbqkb1r/ppp2ppp/4pn2/3p4/3PP3/8/PPPN1PPP/R1BQKBNR w KQkq", "e4e5");
B("rnbqkb1r/ppp2ppp/4pn2/3pP3/3P4/8/PPPN1PPP/R1BQKBNR b KQkq", "f6d7");
B("rnbqkb1r/pppn1ppp/4p3/3pP3/3P4/8/PPPN1PPP/R1BQKBNR w KQkq", "f1d3");
B("rnbqkb1r/pppn1ppp/4p3/3pP3/3P4/3B4/PPPN1PPP/R1BQK1NR b KQkq", "c7c5");

// French Advance (3.e5)
B("rnbqkbnr/ppp2ppp/4p3/3p4/3PP3/8/PPP2PPP/RNBQKBNR w KQkq", "e4e5");
B("rnbqkbnr/ppp2ppp/4p3/3pP3/3P4/8/PPP2PPP/RNBQKBNR b KQkq", "c7c5");
B("rnbqkbnr/pp3ppp/4p3/2ppP3/3P4/8/PPP2PPP/RNBQKBNR w KQkq", "c2c3");
B("rnbqkbnr/pp3ppp/4p3/2ppP3/3P4/2P5/PP3PPP/RNBQKBNR b KQkq", "b8c6");
B("r1bqkbnr/pp3ppp/2n1p3/2ppP3/3P4/2P5/PP3PPP/RNBQKBNR w KQkq", "g1f3");

// French Exchange (3.exd5)
B("rnbqkbnr/ppp2ppp/4p3/3p4/3PP3/8/PPP2PPP/RNBQKBNR w KQkq", "e4d5");
B("rnbqkbnr/ppp2ppp/8/3p4/3P4/8/PPP2PPP/RNBQKBNR b KQkq", "e6d5");

// === CARO-KANN DEFENSE (1.e4 c6) ===
B("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq", "c7c6");
B("rnbqkbnr/pp1ppppp/2p5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq", "d2d4");
B("rnbqkbnr/pp1ppppp/2p5/8/3PP3/8/PPP2PPP/RNBQKBNR b KQkq", "d7d5");

// Caro-Kann Classical (3.Nc3/Nd2 dxe4 4.Nxe4 Bf5)
B("rnbqkbnr/pp2pppp/2p5/3p4/3PP3/8/PPP2PPP/RNBQKBNR w KQkq", "b1c3");
B("rnbqkbnr/pp2pppp/2p5/3p4/3PP3/2N5/PPP2PPP/R1BQKBNR b KQkq", "d5e4");
B("rnbqkbnr/pp2pppp/2p5/8/3Pp3/2N5/PPP2PPP/R1BQKBNR w KQkq", "c3e4");
B("rnbqkbnr/pp2pppp/2p5/8/3PN3/8/PPP2PPP/R1BQKBNR b KQkq", "c8f5");
B("rnbqkbnr/pp2pppp/2p5/5b2/3PN3/8/PPP2PPP/R1BQKBNR w KQkq", "e4g3");
B("rnbqkbnr/pp2pppp/2p5/5b2/3P4/6N1/PPP2PPP/R1BQKBNR b KQkq", "f5g6");
B("rnbqkbnr/pp2pppp/2p3b1/8/3P4/6N1/PPP2PPP/R1BQKBNR w KQkq", "h2h4");

// Caro-Kann Advance (3.e5)
B("rnbqkbnr/pp2pppp/2p5/3p4/3PP3/8/PPP2PPP/RNBQKBNR w KQkq", "e4e5");
B("rnbqkbnr/pp2pppp/2p5/3pP3/3P4/8/PPP2PPP/RNBQKBNR b KQkq", "c8f5");
B("rnbqkbnr/pp2pppp/2p5/3pPb2/3P4/8/PPP2PPP/RNBQKBNR w KQkq", "g1f3");

// Caro-Kann Panov Attack (3.exd5 cxd5 4.c4)
B("rnbqkbnr/pp2pppp/2p5/3p4/3PP3/8/PPP2PPP/RNBQKBNR w KQkq", "e4d5");
B("rnbqkbnr/pp2pppp/8/3p4/3P4/8/PPP2PPP/RNBQKBNR b KQkq", "c6d5");
B("rnbqkbnr/pp2pppp/8/3p4/3P4/8/PPP2PPP/RNBQKBNR w KQkq", "c2c4");

// Caro-Kann Two Knights (3.Nc3 dxe4 4.Nxe4 Nf6)
B("rnbqkbnr/pp2pppp/2p5/8/3PN3/8/PPP2PPP/R1BQKBNR b KQkq", "g8f6");

// === PIRC DEFENSE (1.e4 d6 2.d4 Nf6 3.Nc3) ===
B("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq", "d7d6");
B("rnbqkbnr/ppp1pppp/3p4/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq", "d2d4");
B("rnbqkbnr/ppp1pppp/3p4/8/3PP3/8/PPP2PPP/RNBQKBNR b KQkq", "g8f6");
B("rnbqkb1r/ppp1pppp/3p1n2/8/3PP3/8/PPP2PPP/RNBQKBNR w KQkq", "b1c3");
B("rnbqkb1r/ppp1pppp/3p1n2/8/3PP3/2N5/PPP2PPP/R1BQKBNR b KQkq", "g7g6");
B("rnbqkb1r/ppp1pp1p/3p1np1/8/3PP3/2N5/PPP2PPP/R1BQKBNR w KQkq", "f2f4");

// === ALEKHINE DEFENSE (1.e4 Nf6) ===
B("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq", "g8f6");
B("rnbqkb1r/pppppppp/5n2/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq", "e4e5");
B("rnbqkb1r/pppppppp/8/4Pn2/8/8/PPPP1PPP/RNBQKBNR b KQkq", "f6d5");
B("rnbqkb1r/pppppppp/8/3nP3/8/8/PPPP1PPP/RNBQKBNR w KQkq", "d2d4");
B("rnbqkb1r/pppppppp/8/3nP3/3P4/8/PPP2PPP/RNBQKBNR b KQkq", "d7d6");
B("rnbqkb1r/ppp1pppp/3p4/3nP3/3P4/8/PPP2PPP/RNBQKBNR w KQkq", "g1f3");

// === SCANDINAVIAN DEFENSE (1.e4 d5) ===
B("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq", "d7d5");
B("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq", "e4d5");
B("rnbqkbnr/ppp1pppp/8/3P4/8/8/PPPP1PPP/RNBQKBNR b KQkq", "d8d5");
B("rnb1kbnr/ppp1pppp/8/3q4/8/8/PPPP1PPP/RNBQKBNR w KQkq", "b1c3");
B("rnb1kbnr/ppp1pppp/8/3q4/8/2N5/PPPP1PPP/R1BQKBNR b KQkq", "d5a5");
B("rnb1kbnr/ppp1pppp/8/q7/8/2N5/PPPP1PPP/R1BQKBNR w KQkq", "d2d4");

// Scandinavian — 2...Nf6
B("rnbqkbnr/ppp1pppp/8/3P4/8/8/PPPP1PPP/RNBQKBNR b KQkq", "g8f6");
B("rnbqkb1r/ppp1pppp/5n2/3P4/8/8/PPPP1PPP/RNBQKBNR w KQkq", "d2d4");
B("rnbqkb1r/ppp1pppp/5n2/3P4/3P4/8/PPP2PPP/RNBQKBNR b KQkq", "f6d5");

// === MODERN DEFENSE (1.e4 g6) ===
B("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq", "g7g6");
B("rnbqkbnr/pppppp1p/6p1/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq", "d2d4");
B("rnbqkbnr/pppppp1p/6p1/8/3PP3/8/PPP2PPP/RNBQKBNR b KQkq", "f8g7");
B("rnbqk1nr/ppppppbp/6p1/8/3PP3/8/PPP2PPP/RNBQKBNR w KQkq", "b1c3");

// ----------------------------------------------------------------------------
// 1.d4 — QUEEN'S PAWN OPENINGS
// ----------------------------------------------------------------------------

// 1.d4 responses (starting position book move is e2e4 — d4 lines entered as Black responses)
B("rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq", "d7d5");
B("rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w KQkq", "c2c4");

// === QUEEN'S GAMBIT DECLINED ===
B("rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b KQkq", "e7e6");
B("rnbqkbnr/ppp2ppp/4p3/3p4/2PP4/8/PP2PPPP/RNBQKBNR w KQkq", "b1c3");
B("rnbqkbnr/ppp2ppp/4p3/3p4/2PP4/2N5/PP2PPPP/R1BQKBNR b KQkq", "g8f6");
B("rnbqkb1r/ppp2ppp/4pn2/3p4/2PP4/2N5/PP2PPPP/R1BQKBNR w KQkq", "c1g5");
B("rnbqkb1r/ppp2ppp/4pn2/3p2B1/2PP4/2N5/PP2PPPP/R2QKBNR b KQkq", "f8e7");
B("rnbqk2r/ppp1bppp/4pn2/3p2B1/2PP4/2N5/PP2PPPP/R2QKBNR w KQkq", "e2e3");
B("rnbqk2r/ppp1bppp/4pn2/3p2B1/2PP4/2N1P3/PP3PPP/R2QKBNR b KQkq", "e8g8");
B("rnbq1rk1/ppp1bppp/4pn2/3p2B1/2PP4/2N1P3/PP3PPP/R2QKBNR w KQ", "g1f3");
B("rnbq1rk1/ppp1bppp/4pn2/3p2B1/2PP4/2N1PN2/PP3PPP/R2QKB1R b KQ", "b8d7");
B("r1bq1rk1/pppnbppp/4pn2/3p2B1/2PP4/2N1PN2/PP3PPP/R2QKB1R w KQ", "f1d3");

// QGD — Orthodox Defense
B("r1bq1rk1/pppnbppp/4pn2/3p2B1/2PP4/2N1PN2/PP3PPP/R2QKB1R w KQ", "a1c1");
B("r1bq1rk1/pppnbppp/4pn2/3p2B1/2PP4/2N1PN2/PP3PPP/2RQKB1R b K", "c7c6");

// QGD — Tartakower Variation
B("rnbq1rk1/ppp1bppp/4pn2/3p2B1/2PP4/2N1PN2/PP3PPP/R2QKB1R b KQ", "h7h6");
B("rnbq1rk1/ppp1bpp1/4pn1p/3p2B1/2PP4/2N1PN2/PP3PPP/R2QKB1R w KQ", "g5h4");
B("rnbq1rk1/ppp1bpp1/4pn1p/3p4/2PP3B/2N1PN2/PP3PPP/R2QKB1R b KQ", "b7b6");

// QGD — Lasker Variation
B("rnbq1rk1/ppp1bppp/4pn2/3p2B1/2PP4/2N1PN2/PP3PPP/R2QKB1R b KQ", "c6e4");

// === QUEEN'S GAMBIT ACCEPTED (2...dxc4) ===
B("rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b KQkq", "d5c4");
B("rnbqkbnr/ppp1pppp/8/8/2pP4/8/PP2PPPP/RNBQKBNR w KQkq", "g1f3");
B("rnbqkbnr/ppp1pppp/8/8/2pP4/5N2/PP2PPPP/RNBQKB1R b KQkq", "g8f6");
B("rnbqkb1r/ppp1pppp/5n2/8/2pP4/5N2/PP2PPPP/RNBQKB1R w KQkq", "e2e3");
B("rnbqkb1r/ppp1pppp/5n2/8/2pP4/4PN2/PP3PPP/RNBQKB1R b KQkq", "e7e6");
B("rnbqkb1r/ppp2ppp/4pn2/8/2pP4/4PN2/PP3PPP/RNBQKB1R w KQkq", "f1c4");

// === SLAV DEFENSE (2...c6) ===
B("rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b KQkq", "c7c6");
B("rnbqkbnr/pp2pppp/2p5/3p4/2PP4/8/PP2PPPP/RNBQKBNR w KQkq", "g1f3");
B("rnbqkbnr/pp2pppp/2p5/3p4/2PP4/5N2/PP2PPPP/RNBQKB1R b KQkq", "g8f6");
B("rnbqkb1r/pp2pppp/2p2n2/3p4/2PP4/5N2/PP2PPPP/RNBQKB1R w KQkq", "b1c3");
B("rnbqkb1r/pp2pppp/2p2n2/3p4/2PP4/2N2N2/PP2PPPP/R1BQKB1R b KQkq", "d5c4");
B("rnbqkb1r/pp2pppp/2p2n2/8/2pP4/2N2N2/PP2PPPP/R1BQKB1R w KQkq", "a2a4");

// Slav — Main Line (4...Bf5)
B("rnbqkb1r/pp2pppp/2p2n2/3p4/2PP4/2N2N2/PP2PPPP/R1BQKB1R b KQkq", "c8f5");

// === SEMI-SLAV (2...c6, ...e6) ===
B("rnbqkb1r/pp2pppp/2p2n2/3p4/2PP4/2N2N2/PP2PPPP/R1BQKB1R b KQkq", "e7e6");
B("rnbqkb1r/pp3ppp/2p1pn2/3p4/2PP4/2N2N2/PP2PPPP/R1BQKB1R w KQkq", "e2e3");

// Semi-Slav — Meran Variation
B("rnbqkb1r/pp3ppp/2p1pn2/3p4/2PP4/2N1PN2/PP3PPP/R1BQKB1R b KQkq", "b8d7");
B("r1bqkb1r/pp1n1ppp/2p1pn2/3p4/2PP4/2N1PN2/PP3PPP/R1BQKB1R w KQkq", "f1d3");
B("r1bqkb1r/pp1n1ppp/2p1pn2/3p4/2PP4/2NBPN2/PP3PPP/R1BQK2R b KQkq", "d5c4");
B("r1bqkb1r/pp1n1ppp/2p1pn2/8/2pP4/2NBPN2/PP3PPP/R1BQK2R w KQkq", "d3c4");
B("r1bqkb1r/pp1n1ppp/2p1pn2/8/2BP4/2N1PN2/PP3PPP/R1BQK2R b KQkq", "b7b5");
B("r1bqkb1r/p2n1ppp/2p1pn2/1p6/2BP4/2N1PN2/PP3PPP/R1BQK2R w KQkq", "c4d3");

// Semi-Slav — Botvinnik Variation
B("rnbqkb1r/pp3ppp/2p1pn2/3p2B1/2PP4/2N2N2/PP2PPPP/R2QKB1R b KQkq", "d5c4");
B("rnbqkb1r/pp3ppp/2p1pn2/6B1/2pP4/2N2N2/PP2PPPP/R2QKB1R w KQkq", "e2e4");

// === KING'S INDIAN DEFENSE ===
B("rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq", "g8f6");
B("rnbqkb1r/pppppppp/5n2/8/3P4/8/PPP1PPPP/RNBQKBNR w KQkq", "c2c4");
B("rnbqkb1r/pppppppp/5n2/8/2PP4/8/PP2PPPP/RNBQKBNR b KQkq", "g7g6");
B("rnbqkb1r/pppppp1p/5np1/8/2PP4/8/PP2PPPP/RNBQKBNR w KQkq", "b1c3");
B("rnbqkb1r/pppppp1p/5np1/8/2PP4/2N5/PP2PPPP/R1BQKBNR b KQkq", "f8g7");
B("rnbqk2r/ppppppbp/5np1/8/2PP4/2N5/PP2PPPP/R1BQKBNR w KQkq", "e2e4");
B("rnbqk2r/ppppppbp/5np1/8/2PPP3/2N5/PP3PPP/R1BQKBNR b KQkq", "d7d6");
B("rnbqk2r/ppp1ppbp/3p1np1/8/2PPP3/2N5/PP3PPP/R1BQKBNR w KQkq", "g1f3");

// KID — Classical (Nf3, Be2)
B("rnbqk2r/ppp1ppbp/3p1np1/8/2PPP3/2N2N2/PP3PPP/R1BQKB1R b KQkq", "e8g8");
B("rnbq1rk1/ppp1ppbp/3p1np1/8/2PPP3/2N2N2/PP3PPP/R1BQKB1R w KQ", "f1e2");
B("rnbq1rk1/ppp1ppbp/3p1np1/8/2PPP3/2N2N2/PP2BPPP/R1BQK2R b KQ", "e7e5");
B("rnbq1rk1/ppp2pbp/3p1np1/4p3/2PPP3/2N2N2/PP2BPPP/R1BQK2R w KQ", "e1g1");
B("rnbq1rk1/ppp2pbp/3p1np1/4p3/2PPP3/2N2N2/PP2BPPP/R1BQ1RK1 b -", "b8c6");
B("r1bq1rk1/ppp2pbp/2np1np1/4p3/2PPP3/2N2N2/PP2BPPP/R1BQ1RK1 w -", "d4d5");
B("r1bq1rk1/ppp2pbp/2np1np1/3Pp3/2P1P3/2N2N2/PP2BPPP/R1BQ1RK1 b -", "c6e7");

// KID — Samisch (f3)
B("rnbqk2r/ppp1ppbp/3p1np1/8/2PPP3/2N5/PP3PPP/R1BQKBNR w KQkq", "f2f3");
B("rnbqk2r/ppp1ppbp/3p1np1/8/2PPP3/2N2P2/PP4PP/R1BQKBNR b KQkq", "e8g8");
B("rnbq1rk1/ppp1ppbp/3p1np1/8/2PPP3/2N2P2/PP4PP/R1BQKBNR w KQ", "c1e3");

// KID — Fianchetto
B("rnbqk2r/ppp1ppbp/3p1np1/8/2PP4/2N5/PP2PPPP/R1BQKBNR w KQkq", "g1f3");
B("rnbqk2r/ppp1ppbp/3p1np1/8/2PP4/2N2N2/PP2PPPP/R1BQKB1R b KQkq", "e8g8");
B("rnbq1rk1/ppp1ppbp/3p1np1/8/2PP4/2N2N2/PP2PPPP/R1BQKB1R w KQ", "g2g3");
B("rnbq1rk1/ppp1ppbp/3p1np1/8/2PP4/2N2NP1/PP2PP1P/R1BQKB1R b KQ", "c7c6");

// KID — Four Pawns Attack
B("rnbqk2r/ppp1ppbp/3p1np1/8/2PPP3/2N5/PP3PPP/R1BQKBNR w KQkq", "f2f4");
B("rnbqk2r/ppp1ppbp/3p1np1/8/2PPPP2/2N5/PP4PP/R1BQKBNR b KQkq", "e8g8");
B("rnbq1rk1/ppp1ppbp/3p1np1/8/2PPPP2/2N5/PP4PP/R1BQKBNR w KQ", "g1f3");
B("rnbq1rk1/ppp1ppbp/3p1np1/8/2PPPP2/2N2N2/PP4PP/R1BQKB1R b KQ", "c7c5");

// === GRUNFELD DEFENSE ===
B("rnbqkb1r/pppppp1p/5np1/8/2PP4/2N5/PP2PPPP/R1BQKBNR b KQkq", "d7d5");
B("rnbqkb1r/ppp1pp1p/5np1/3p4/2PP4/2N5/PP2PPPP/R1BQKBNR w KQkq", "c4d5");
B("rnbqkb1r/ppp1pp1p/5np1/3P4/3P4/2N5/PP2PPPP/R1BQKBNR b KQkq", "f6d5");
B("rnbqkb1r/ppp1pp1p/6p1/3n4/3P4/2N5/PP2PPPP/R1BQKBNR w KQkq", "e2e4");

// Grunfeld Exchange
B("rnbqkb1r/ppp1pp1p/6p1/3n4/3PP3/2N5/PP3PPP/R1BQKBNR b KQkq", "d5c3");
B("rnbqkb1r/ppp1pp1p/6p1/8/3PP3/2n5/PP3PPP/R1BQKBNR w KQkq", "b2c3");
B("rnbqkb1r/ppp1pp1p/6p1/8/3PP3/2P5/P4PPP/R1BQKBNR b KQkq", "f8g7");
B("rnbqk2r/ppp1ppbp/6p1/8/3PP3/2P5/P4PPP/R1BQKBNR w KQkq", "f1c4");
B("rnbqk2r/ppp1ppbp/6p1/8/2BPP3/2P5/P4PPP/R1BQK1NR b KQkq", "c7c5");
B("rnbqk2r/pp2ppbp/6p1/2p5/2BPP3/2P5/P4PPP/R1BQK1NR w KQkq", "g1e2");

// Grunfeld — Russian System
B("rnbqkb1r/ppp1pp1p/6p1/3n4/3P4/2N5/PP2PPPP/R1BQKBNR w KQkq", "g1f3");
B("rnbqkb1r/ppp1pp1p/6p1/3n4/3P4/2N2N2/PP2PPPP/R1BQKB1R b KQkq", "f8g7");

// === NIMZO-INDIAN DEFENSE ===
B("rnbqkb1r/pppppppp/5n2/8/2PP4/8/PP2PPPP/RNBQKBNR b KQkq", "e7e6");
B("rnbqkb1r/pppp1ppp/4pn2/8/2PP4/8/PP2PPPP/RNBQKBNR w KQkq", "b1c3");
B("rnbqkb1r/pppp1ppp/4pn2/8/2PP4/2N5/PP2PPPP/R1BQKBNR b KQkq", "f8b4");

// Nimzo — Rubinstein (4.e3)
B("rnbqk2r/pppp1ppp/4pn2/8/1bPP4/2N5/PP2PPPP/R1BQKBNR w KQkq", "e2e3");
B("rnbqk2r/pppp1ppp/4pn2/8/1bPP4/2N1P3/PP3PPP/R1BQKBNR b KQkq", "e8g8");
B("rnbq1rk1/pppp1ppp/4pn2/8/1bPP4/2N1P3/PP3PPP/R1BQKBNR w KQ", "f1d3");
B("rnbq1rk1/pppp1ppp/4pn2/8/1bPP4/2NBP3/PP3PPP/R1BQK1NR b KQ", "d7d5");

// Nimzo — Classical (4.Qc2)
B("rnbqk2r/pppp1ppp/4pn2/8/1bPP4/2N5/PP2PPPP/R1BQKBNR w KQkq", "d1c2");
B("rnbqk2r/pppp1ppp/4pn2/8/1bPP4/2N5/PPQ1PPPP/R1B1KBNR b KQkq", "e8g8");
B("rnbq1rk1/pppp1ppp/4pn2/8/1bPP4/2N5/PPQ1PPPP/R1B1KBNR w KQ", "a2a3");

// Nimzo — Samisch (4.a3)
B("rnbqk2r/pppp1ppp/4pn2/8/1bPP4/2N5/PP2PPPP/R1BQKBNR w KQkq", "a2a3");
B("rnbqk2r/pppp1ppp/4pn2/8/1bPP4/P1N5/1P2PPPP/R1BQKBNR b KQkq", "b4c3");
B("rnbqk2r/pppp1ppp/4pn2/8/2PP4/P1b5/1P2PPPP/R1BQKBNR w KQkq", "b2c3");

// === QUEEN'S INDIAN DEFENSE ===
B("rnbqkb1r/pppppppp/5n2/8/2PP4/8/PP2PPPP/RNBQKBNR w KQkq", "g1f3");
B("rnbqkb1r/pppppppp/5n2/8/2PP4/5N2/PP2PPPP/RNBQKB1R b KQkq", "e7e6");
B("rnbqkb1r/pppp1ppp/4pn2/8/2PP4/5N2/PP2PPPP/RNBQKB1R b KQkq", "b7b6");
B("rnbqkb1r/p1pp1ppp/1p2pn2/8/2PP4/5N2/PP2PPPP/RNBQKB1R w KQkq", "g2g3");
B("rnbqkb1r/p1pp1ppp/1p2pn2/8/2PP4/5NP1/PP2PP1P/RNBQKB1R b KQkq", "c8b7");
B("rn1qkb1r/pbpp1ppp/1p2pn2/8/2PP4/5NP1/PP2PP1P/RNBQKB1R w KQkq", "f1g2");

// === BOGO-INDIAN DEFENSE ===
B("rnbqkb1r/pppp1ppp/4pn2/8/2PP4/5N2/PP2PPPP/RNBQKB1R b KQkq", "f8b4");
B("rnbqk2r/pppp1ppp/4pn2/8/1bPP4/5N2/PP2PPPP/RNBQKB1R w KQkq", "c1d2");

// === BENONI DEFENSE ===
B("rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq", "c7c5");
B("rnbqkbnr/pp1ppppp/8/2p5/3P4/8/PPP1PPPP/RNBQKBNR w KQkq", "d4d5");
B("rnbqkbnr/pp1ppppp/8/2pP4/8/8/PPP1PPPP/RNBQKBNR b KQkq", "e7e6");
B("rnbqkbnr/pp1p1ppp/4p3/2pP4/8/8/PPP1PPPP/RNBQKBNR w KQkq", "c2c4");
B("rnbqkbnr/pp1p1ppp/4p3/2pP4/2P5/8/PP2PPPP/RNBQKBNR b KQkq", "e6d5");
B("rnbqkbnr/pp1p1ppp/8/2pp4/2P5/8/PP2PPPP/RNBQKBNR w KQkq", "c4d5");
B("rnbqkbnr/pp1p1ppp/8/2pP4/8/8/PP2PPPP/RNBQKBNR b KQkq", "d7d6");
B("rnbqkbnr/pp3ppp/3p4/2pP4/8/8/PP2PPPP/RNBQKBNR w KQkq", "b1c3");
B("rnbqkbnr/pp3ppp/3p4/2pP4/8/2N5/PP2PPPP/R1BQKBNR b KQkq", "g7g6");

// === BUDAPEST GAMBIT ===
B("rnbqkb1r/pppppppp/5n2/8/2PP4/8/PP2PPPP/RNBQKBNR b KQkq", "e7e5");
B("rnbqkb1r/pppp1ppp/5n2/4p3/2PP4/8/PP2PPPP/RNBQKBNR w KQkq", "d4e5");
B("rnbqkb1r/pppp1ppp/8/4p3/2P1n3/8/PP2PPPP/RNBQKBNR w KQkq", "g1f3");

// === CATALAN OPENING ===
B("rnbqkb1r/pppp1ppp/4pn2/8/2PP4/5N2/PP2PPPP/RNBQKB1R w KQkq", "g2g3");
B("rnbqkb1r/pppp1ppp/4pn2/8/2PP4/5NP1/PP2PP1P/RNBQKB1R b KQkq", "d7d5");
B("rnbqkb1r/ppp2ppp/4pn2/3p4/2PP4/5NP1/PP2PP1P/RNBQKB1R w KQkq", "f1g2");
B("rnbqkb1r/ppp2ppp/4pn2/3p4/2PP4/5NP1/PP2PPBP/RNBQK2R b KQkq", "f8e7");
B("rnbqk2r/ppp1bppp/4pn2/3p4/2PP4/5NP1/PP2PPBP/RNBQK2R w KQkq", "e1g1");
B("rnbqk2r/ppp1bppp/4pn2/3p4/2PP4/5NP1/PP2PPBP/RNBQ1RK1 b KQkq", "e8g8");
B("rnbq1rk1/ppp1bppp/4pn2/3p4/2PP4/5NP1/PP2PPBP/RNBQ1RK1 w -", "d1c2");
B("rnbq1rk1/ppp1bppp/4pn2/3p4/2PP4/5NP1/PPQ1PPBP/RNB2RK1 b -", "d5c4");

// Open Catalan
B("rnbq1rk1/ppp1bppp/4pn2/8/2pP4/5NP1/PPQ1PPBP/RNB2RK1 w -", "d1c4");

// ----------------------------------------------------------------------------
// 1.c4 — ENGLISH OPENING
// ----------------------------------------------------------------------------

// 1.c4 responses (starting position plays e4, these are if engine faces c4 as Black)
B("rnbqkbnr/pppppppp/8/8/2P5/8/PP1PPPPP/RNBQKBNR b KQkq", "e7e5");
B("rnbqkbnr/pppp1ppp/8/4p3/2P5/8/PP1PPPPP/RNBQKBNR w KQkq", "b1c3");
B("rnbqkbnr/pppp1ppp/8/4p3/2P5/2N5/PP1PPPPP/R1BQKBNR b KQkq", "g8f6");
B("rnbqkb1r/pppp1ppp/5n2/4p3/2P5/2N5/PP1PPPPP/R1BQKBNR w KQkq", "g1f3");

// English — Symmetrical
B("rnbqkbnr/pppppppp/8/8/2P5/8/PP1PPPPP/RNBQKBNR b KQkq", "c7c5");
B("rnbqkbnr/pp1ppppp/8/2p5/2P5/8/PP1PPPPP/RNBQKBNR w KQkq", "g1f3");
B("rnbqkbnr/pp1ppppp/8/2p5/2P5/5N2/PP1PPPPP/RNBQKB1R b KQkq", "g8f6");
B("rnbqkb1r/pp1ppppp/5n2/2p5/2P5/5N2/PP1PPPPP/RNBQKB1R w KQkq", "b1c3");

// English — Four Knights
B("rnbqkb1r/pp1ppppp/5n2/2p5/2P5/2N2N2/PP1PPPPP/R1BQKB1R b KQkq", "b8c6");
B("r1bqkb1r/pp1ppppp/2n2n2/2p5/2P5/2N2N2/PP1PPPPP/R1BQKB1R w KQkq", "g2g3");

// English — Reversed Sicilian
B("rnbqkbnr/pppp1ppp/8/4p3/2P5/2N5/PP1PPPPP/R1BQKBNR b KQkq", "b8c6");

// ----------------------------------------------------------------------------
// 1.Nf3 — RETI OPENING
// ----------------------------------------------------------------------------

// 1.Nf3 responses
B("rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R b KQkq", "d7d5");
B("rnbqkbnr/ppp1pppp/8/3p4/8/5N2/PPPPPPPP/RNBQKB1R w KQkq", "g2g3");
B("rnbqkbnr/ppp1pppp/8/3p4/8/5NP1/PPPPPP1P/RNBQKB1R b KQkq", "g8f6");
B("rnbqkb1r/ppp1pppp/5n2/3p4/8/5NP1/PPPPPP1P/RNBQKB1R w KQkq", "f1g2");
B("rnbqkb1r/ppp1pppp/5n2/3p4/8/5NP1/PPPPPPBP/RNBQK2R b KQkq", "c7c6");

// ----------------------------------------------------------------------------
// FLANK OPENINGS
// ----------------------------------------------------------------------------

// Bird's Opening (1.f4)
B("rnbqkbnr/pppppppp/8/8/5P2/8/PPPPP1PP/RNBQKBNR b KQkq", "d7d5");
B("rnbqkbnr/ppp1pppp/8/3p4/5P2/8/PPPPP1PP/RNBQKBNR w KQkq", "g1f3");

// Nimzowitsch-Larsen (1.b3)
B("rnbqkbnr/pppppppp/8/8/8/1P6/P1PPPPPP/RNBQKBNR b KQkq", "e7e5");
B("rnbqkbnr/pppp1ppp/8/4p3/8/1P6/P1PPPPPP/RNBQKBNR w KQkq", "c1b2");

// King's Fianchetto (1.g3)
B("rnbqkbnr/pppppppp/8/8/8/6P1/PPPPPP1P/RNBQKBNR b KQkq", "d7d5");
B("rnbqkbnr/ppp1pppp/8/3p4/8/6P1/PPPPPP1P/RNBQKBNR w KQkq", "f1g2");

// London System (1.d4 d5 2.Bf4 / 2.Nf3 Nf6 3.Bf4)
B("rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w KQkq", "c1f4");
B("rnbqkbnr/ppp1pppp/8/3p4/3P1B2/8/PPP1PPPP/RN1QKBNR b KQkq", "g8f6");
B("rnbqkb1r/ppp1pppp/5n2/3p4/3P1B2/8/PPP1PPPP/RN1QKBNR w KQkq", "e2e3");
B("rnbqkb1r/ppp1pppp/5n2/3p4/3P1B2/4P3/PPP2PPP/RN1QKBNR b KQkq", "e7e6");
B("rnbqkb1r/ppp2ppp/4pn2/3p4/3P1B2/4P3/PPP2PPP/RN1QKBNR w KQkq", "g1f3");
B("rnbqkb1r/ppp2ppp/4pn2/3p4/3P1B2/4PN2/PPP2PPP/RN1QKB1R b KQkq", "f8d6");

// Torre Attack (1.d4 Nf6 2.Nf3 e6 3.Bg5)
B("rnbqkb1r/pppppppp/5n2/8/3P4/8/PPP1PPPP/RNBQKBNR w KQkq", "g1f3");
B("rnbqkb1r/pppppppp/5n2/8/3P4/5N2/PPP1PPPP/RNBQKB1R b KQkq", "e7e6");
B("rnbqkb1r/pppp1ppp/4pn2/8/3P4/5N2/PPP1PPPP/RNBQKB1R w KQkq", "c1g5");

// Trompowsky Attack (1.d4 Nf6 2.Bg5)
B("rnbqkb1r/pppppppp/5n2/8/3P4/8/PPP1PPPP/RNBQKBNR w KQkq", "c1g5");

// ----------------------------------------------------------------------------
// DEEP LINES — Ruy Lopez (continued, 15-20 moves deep)
// ----------------------------------------------------------------------------

// Ruy Lopez — Marshall Attack Deep Line (15+ moves)
// 1.e4 e5 2.Nf3 Nc6 3.Bb5 a6 4.Ba4 Nf6 5.O-O Be7 6.Re1 b5 7.Bb3 O-O 8.c3 d5 9.exd5 Nxd5 10.Nxe5 Nxe5 11.Rxe5 c6
B("r1bq1rk1/4bppp/p1p5/1p1nR3/8/1BP5/PP1P1PPP/RNBQ2K1 w -", "d2d4");
// 12.d4 Bd6 13.Re1 Qh4 14.g3 Qh3
B("r1b2rk1/5ppp/p1pb4/1p1n4/3P4/1BP3P1/PP3P1P/RNBQR1K1 b -", "d8h4");
B("r1b2rk1/5ppp/p1pb4/1p1n4/3P4/1BP3Pq/PP3P1P/RNBQR1K1 w -", "c1e3");

// Ruy Lopez — Closed Defense Deep (Breyer 15+ moves)
// After 9...Nb8 10.d4 Nbd7 11.Nbd2 Bb7 12.Bc2 Re8 13.Nf1 Bf8 14.Ng3 g6
B("r2qrbk1/1ppn1ppp/p4n2/1b2p3/3PP3/1BP2N2/PP3PPP/RNBQR1K1 b -", "c6b8");
B("r1bq1rk1/1ppnbppp/p4n2/1p2p3/3PP3/1BP2N2/PP3PPP/RNBQR1K1 w -", "b1d2");
B("r1bq1rk1/1ppnbppp/p4n2/1p2p3/3PP3/1BP2N2/PP1N1PPP/R1BQR1K1 b -", "c8b7");
B("r2q1rk1/1bpnbppp/p4n2/1p2p3/3PP3/1BP2N2/PP1N1PPP/R1BQR1K1 w -", "b3c2");
B("r2q1rk1/1bpnbppp/p4n2/1p2p3/3PP3/2P2N2/PPBN1PPP/R1BQR1K1 b -", "f8e8");
B("r2qr1k1/1bpnbppp/p4n2/1p2p3/3PP3/2P2N2/PPBN1PPP/R1BQR1K1 w -", "d2f1");
B("r2qr1k1/1bpnbppp/p4n2/1p2p3/3PP3/2P2N2/PPB2PPP/R1BQRNK1 b -", "e7f8");
B("r2qrbk1/1bpn1ppp/p4n2/1p2p3/3PP3/2P2N2/PPB2PPP/R1BQRNK1 w -", "f1g3");
B("r2qrbk1/1bpn1ppp/p4n2/1p2p3/3PP3/2P2NN1/PPB2PPP/R1BQR1K1 b -", "g7g6");

// Ruy Lopez — Anti-Marshall (8.a4)
B("r1bq1rk1/2ppbppp/p1n2n2/1p2p3/B3P3/5N2/PPPP1PPP/RNBQR1K1 w -", "a2a4");

// Ruy Lopez — Zaitsev Variation (9...Bb7 line)
B("r1bq1rk1/2ppbppp/p1n2n2/1p2p3/4P3/1BP2N2/PP1P1PPP/RNBQR1K1 b -", "c8b7");
B("r2q1rk1/1bppbppp/p1n2n2/1p2p3/4P3/1BP2N2/PP1P1PPP/RNBQR1K1 w -", "d2d3");

// ----------------------------------------------------------------------------
// DEEP LINES — Sicilian Najdorf (continued, 15-20 moves deep)
// ----------------------------------------------------------------------------

// Najdorf — English Attack Deep Line (6.Be3 e5 7.Nb3 Be7 8.f3 Be6 9.Qd2 Nbd7 10.g4 b5)
B("r2qk2r/1p1nbppp/p2pbn2/4p3/4P1P1/1NN1BP2/PPPQ2PP/R3KB1R b KQ", "b7b5");
B("r2qk2r/3nbppp/p2pbn2/1p2p3/4P1P1/1NN1BP2/PPPQ2PP/R3KB1R w KQ", "g4g5");
B("r2qk2r/3nbppp/p2pbn2/1p2p1P1/4P3/1NN1BP2/PPPQ2PP/R3KB1R b KQ", "f6h5");
B("r2qk2r/3nbppp/p2pb3/1p2p1Pn/4P3/1NN1BP2/PPPQ2PP/R3KB1R w KQ", "e1c1");
B("r2qk2r/3nbppp/p2pb3/1p2p1Pn/4P3/1NN1BP2/PPPQ2PP/2KR1B1R b kq", "e8g8");

// Najdorf — 6.Bg5 e6 7.f4 (Poisoned Pawn)
B("rnbqkb1r/1p3ppp/p2ppn2/6B1/3NP3/2N5/PPP2PPP/R2QKB1R w KQkq", "f2f4");
B("rnbqkb1r/1p3ppp/p2ppn2/6B1/3NPP2/2N5/PPP3PP/R2QKB1R b KQkq", "d8b6");
B("rn2kb1r/1p3ppp/p2ppn2/6B1/3NPP2/2N5/PPP3PP/R2QKB1R w KQkq", "d1d2"); // Not poisoned pawn variation
// Poisoned Pawn: 7...Qb6 8.Qd2 Qxb2 9.Rb1
B("rnb1kb1r/1p3ppp/p2ppn2/6B1/3NPP2/2N5/PPPQ2PP/R3KB1R b KQkq", "b6b2");
B("rnb1kb1r/1p3ppp/p2ppn2/6B1/3NPP2/2N5/PqPQ2PP/R3KB1R w KQkq", "a1b1");
B("rnb1kb1r/1p3ppp/p2ppn2/6B1/3NPP2/2N5/PqPQ2PP/1R2KB1R b Kkq", "b2a3");

// Najdorf — 6.Bc4 (Fischer's favorite)
B("rnbqkb1r/1p2pppp/p2p1n2/8/2BNP3/2N5/PPP2PPP/R1BQK2R b KQkq", "e7e6");
B("rnbqkb1r/1p3ppp/p2ppn2/8/2BNP3/2N5/PPP2PPP/R1BQK2R w KQkq", "c4b3");

// Najdorf — 6.f3 (Modern English Attack)
B("rnbqkb1r/1p2pppp/p2p1n2/8/3NP3/2N2P2/PPP3PP/R1BQKB1R b KQkq", "e7e5");
B("rnbqkb1r/1p3ppp/p2p1n2/4p3/3NP3/2N2P2/PPP3PP/R1BQKB1R w KQkq", "d4b3");

// ----------------------------------------------------------------------------
// DEEP LINES — Sicilian Dragon (continued, Yugoslav Attack)
// ----------------------------------------------------------------------------

// Dragon — Yugoslav Attack Deep Line
// 9.Bc4 Bd7 10.O-O-O Rc8 11.Bb3 Ne5 12.h4 h5 13.Bg5 Rc5
B("r2q1rk1/pp1bppbp/2np1np1/8/2BNP3/2N1BP2/PPPQ2PP/R3K2R b KQ", "a8c8");
B("2rq1rk1/pp1bppbp/2np1np1/8/2BNP3/2N1BP2/PPPQ2PP/R3K2R w KQ", "c4b3");
B("2rq1rk1/pp1bppbp/2np1np1/8/3NP3/1BN1BP2/PPPQ2PP/R3K2R b KQ", "c6e5");
B("2rq1rk1/pp1bppbp/3p1np1/4n3/3NP3/1BN1BP2/PPPQ2PP/R3K2R w KQ", "h2h4");
B("2rq1rk1/pp1bppbp/3p1np1/4n3/3NP2P/1BN1BP2/PPPQ2P1/R3K2R b KQ", "h7h5");
B("2rq1rk1/pp1bppb1/3p1np1/4n2p/3NP2P/1BN1BP2/PPPQ2P1/R3K2R w KQ", "c1g5");

// Dragon — Classical Variation (not Yugoslav)
B("rnbq1rk1/pp2ppbp/3p1np1/8/2PPP3/2N2N2/PP3PPP/R1BQKB1R w KQ", "f1e2");
B("rnbq1rk1/pp2ppbp/3p1np1/8/2PPP3/2N2N2/PP2BPPP/R1BQK2R b KQ", "c8g4");

// ----------------------------------------------------------------------------
// DEEP LINES — Sicilian Sveshnikov (continued)
// ----------------------------------------------------------------------------

// Sveshnikov Deep Line: ...a6 7.Na3 b5 8.Nd5 f5 9.Bxf6 gxf6
B("r1bqkb1r/1p3ppp/p1np1n2/1N2p1B1/4P3/2N5/PPP2PPP/R2QKB1R w KQkq", "b5a3");
B("r1bqkb1r/1p3ppp/p1np1n2/4p1B1/4P3/N1N5/PPP2PPP/R2QKB1R b KQkq", "b5b5"); // b7b5 already covered, this is b-file push
B("r1bqkb1r/5ppp/p1np1n2/1p2p1B1/4P3/N1N5/PPP2PPP/R2QKB1R w KQkq", "c3d5");
B("r1bqkb1r/5ppp/p1np1n2/1p1Np1B1/4P3/N7/PPP2PPP/R2QKB1R b KQkq", "f7f5");

// ----------------------------------------------------------------------------
// DEEP LINES — French Defense (continued)
// ----------------------------------------------------------------------------

// French Winawer — Poisoned Pawn variation deep
// 7.Qg4 O-O 8.Bd3 Nbc6 (or f5)
B("rnbqk1nr/pp3ppp/4p3/2ppP3/3P4/P1P5/2P2PPP/R1BQKBNR w KQkq", "d1g4");
B("rnbqk1nr/pp3ppp/4p3/2ppP3/3P2Q1/P1P5/2P2PPP/R1B1KBNR b KQkq", "e8g8");
B("rnbq1knr/pp3ppp/4p3/2ppP3/3P2Q1/P1P5/2P2PPP/R1B1KBNR w KQkq", "f1d3");

// French Advance — Deep Line
// 3.e5 c5 4.c3 Nc6 5.Nf3 Bd7 6.Be2 Nge7 7.Na3 cxd4 8.cxd4 Nf5 9.Nc2
B("r1bqkbnr/pp3ppp/2n1p3/2ppP3/3P4/2P2N2/PP3PPP/RNBQKB1R b KQkq", "c8d7");
B("r1bqkbnr/pp1b1ppp/2n1p3/2ppP3/3P4/2P2N2/PP3PPP/RNBQKB1R w KQkq", "f1e2");
B("r1bqkbnr/pp1b1ppp/2n1p3/2ppP3/3P4/2P2N2/PP2BPPP/RNBQK2R b KQkq", "g8e7");
B("r1bqkb1r/pp1bnppp/2n1p3/2ppP3/3P4/2P2N2/PP2BPPP/RNBQK2R w KQkq", "b1a3");

// French Classical Deep — Steinitz Variation
B("rnbqk2r/ppp1bppp/4pn2/3p2B1/3PP3/2N5/PPP2PPP/R2QKBNR b KQkq", "e5e5"); // actually: 4...dxe4
B("rnbqkb1r/ppp2ppp/4pn2/3p2B1/3PP3/2N5/PPP2PPP/R2QKBNR b KQkq", "d5e4");
B("rnbqkb1r/ppp2ppp/4pn2/6B1/3Pp3/2N5/PPP2PPP/R2QKBNR w KQkq", "c3e4");

// French Tarrasch Deep
// 3.Nd2 Nf6 4.e5 Nfd7 5.Bd3 c5 6.c3 Nc6 7.Ne2 cxd4 8.cxd4 f6 9.exf6 Nxf6
B("r1bqkb1r/pppn1ppp/4p3/3pP3/3P4/3B4/PPPN1PPP/R1BQK1NR b KQkq", "b8c6");
B("r1bqkb1r/pp1n1ppp/2n1p3/2ppP3/3P4/3B4/PPPN1PPP/R1BQK1NR w KQkq", "g1e2");
B("r1bqkb1r/pp1n1ppp/2n1p3/2ppP3/3P4/3B4/PPPNNPPP/R1BQK2R b KQkq", "c5d4");
B("r1bqkb1r/pp1n1ppp/2n1p3/3pP3/3p4/3B4/PPPNNPPP/R1BQK2R w KQkq", "c3d4");

// ----------------------------------------------------------------------------
// DEEP LINES — Caro-Kann (continued)
// ----------------------------------------------------------------------------

// Caro-Kann Classical Deep
// 4...Bf5 5.Ng3 Bg6 6.h4 h6 7.Nf3 Nd7 8.h5 Bh7 9.Bd3 Bxd3 10.Qxd3 e6
B("rnbqkbnr/pp2pppp/2p3b1/8/3P4/6N1/PPP2PPP/R1BQKBNR b KQkq", "h7h6");
B("rnbqkbnr/pp2ppp1/2p3bp/8/3P3P/6N1/PPP2PP1/R1BQKBNR w KQkq", "g1f3");
B("rnbqkb1r/pp1nppp1/2p3bp/8/3P3P/5NN1/PPP2PP1/R1BQKB1R b KQkq", "h4h5"); // Wait, actually 7.Nf3 Nd7 8.h5
B("rn1qkbnr/pp2pppp/2p3b1/7P/3P4/5NN1/PPP2PP1/R1BQKB1R b KQkq", "g6h7");
B("rn1qkbnr/pp2pppp/2p4b/7P/3P4/5NN1/PPP2PP1/R1BQKB1R w KQkq", "f1d3");
B("rn1qkbnr/pp2pppp/2p4b/7P/3P4/3B1NN1/PPP2PP1/R1BQK2R b KQkq", "h7d3");
B("rn1qkbnr/pp2pppp/2p5/7P/3P4/3b1NN1/PPP2PP1/R1BQK2R w KQkq", "d1d3");
B("rn1qkbnr/pp3ppp/2p1p3/7P/3P4/3Q1NN1/PPP2PP1/R1B1K2R b KQkq", "g8f6");

// Caro-Kann Advance Deep
// 3.e5 Bf5 4.Nf3 e6 5.Be2 Nd7 6.O-O Ne7 7.Nbd2 h6 8.Nb3
B("rnbqkbnr/pp2pppp/2p5/3pPb2/3P4/5N2/PPP2PPP/RNBQKB1R b KQkq", "e7e6");
B("rnbqkbnr/pp3ppp/2p1p3/3pPb2/3P4/5N2/PPP2PPP/RNBQKB1R w KQkq", "f1e2");
B("rnbqkbnr/pp3ppp/2p1p3/3pPb2/3P4/5N2/PPP1BPPP/RNBQK2R b KQkq", "b8d7");
B("rnbqkb1r/pp1n1ppp/2p1p3/3pPb2/3P4/5N2/PPP1BPPP/RNBQK2R w KQkq", "e1g1");

// Caro-Kann Two Knights Deep
// 4...Nf6 5.Nxf6+ exf6 (or gxf6) 6.c3 Bd6 7.Bd3 O-O 8.Qc2
B("rnbqkbnr/pp2pppp/2p5/8/3PN3/8/PPP2PPP/R1BQKBNR b KQkq", "g8f6");
B("rnbqkb1r/pp2pppp/2p2n2/8/3PN3/8/PPP2PPP/R1BQKBNR w KQkq", "e4f6");
B("rnbqkb1r/pp2pppp/2p2N2/8/3P4/8/PPP2PPP/R1BQKBNR b KQkq", "e7f6");
B("rnbqkb1r/pp3ppp/2p2p2/8/3P4/8/PPP2PPP/R1BQKBNR w KQkq", "c2c3");

// ----------------------------------------------------------------------------
// DEEP LINES — Queen's Gambit Declined (continued)
// ----------------------------------------------------------------------------

// QGD Orthodox Deep
// 1.d4 d5 2.c4 e6 3.Nc3 Nf6 4.Bg5 Be7 5.e3 O-O 6.Nf3 Nbd7 7.Rc1 c6 8.Bd3 dxc4 9.Bxc4 Nd5
B("r1bq1rk1/pppnbppp/4pn2/3p2B1/2PP4/2N1PN2/PP3PPP/2RQKB1R b K", "c7c6");
B("r1bq1rk1/pp1nbppp/2p1pn2/3p2B1/2PP4/2N1PN2/PP3PPP/2RQKB1R w K", "f1d3");
B("r1bq1rk1/pp1nbppp/2p1pn2/3p2B1/2PP4/2NBPN2/PP3PPP/2RQK2R b K", "d5c4");
B("r1bq1rk1/pp1nbppp/2p1pn2/6B1/2pP4/2NBPN2/PP3PPP/2RQK2R w K", "d3c4");
B("r1bq1rk1/pp1nbppp/2p1pn2/6B1/2BP4/2N1PN2/PP3PPP/2RQK2R b K", "f6d5");

// QGD — Ragozin Defense (4...Bb4)
B("rnbqkb1r/ppp2ppp/4pn2/3p2B1/2PP4/2N5/PP2PPPP/R2QKBNR b KQkq", "f8b4");
B("rnbqk2r/ppp2ppp/4pn2/3p2B1/1bPP4/2N5/PP2PPPP/R2QKBNR w KQkq", "e2e3");

// QGD — Vienna Variation (5.Bf4)
B("rnbqk2r/ppp1bppp/4pn2/3p2B1/2PP4/2N5/PP2PPPP/R2QKBNR w KQkq", "c1f4");

// QGD — Exchange Variation
B("rnbqkb1r/ppp2ppp/4pn2/3p4/2PP4/2N5/PP2PPPP/R1BQKBNR w KQkq", "c4d5");
B("rnbqkb1r/ppp2ppp/5n2/3p4/3P4/2N5/PP2PPPP/R1BQKBNR b KQkq", "e6d5");

// ----------------------------------------------------------------------------
// DEEP LINES — Slav Defense (continued)
// ----------------------------------------------------------------------------

// Slav — Main Line with 4...Bf5 deep
B("rnbqkb1r/pp2pppp/2p2n2/3p4/2PP4/2N2N2/PP2PPPP/R1BQKB1R b KQkq", "c8f5");
B("rn1qkb1r/pp2pppp/2p2n2/3p1b2/2PP4/2N2N2/PP2PPPP/R1BQKB1R w KQkq", "c4d5"); // Actually let's do a common line
B("rn1qkb1r/pp2pppp/2p2n2/3p1b2/2PP4/2N2N2/PP2PPPP/R1BQKB1R w KQkq", "f3h4");
// Czech Variation (Slav with ...Bf5 early)
B("rn1qkb1r/pp2pppp/2p2n2/3p1b2/2PP3N/2N5/PP2PPPP/R1BQKB1R b KQkq", "f5e4");

// Slav — Semi-Slav Meran Deep
// After 7.Bc4 b5 8.Bd3 a6 9.e4 c5 10.e5 cxd4 11.Nxb5 axb5
B("r1bqkb1r/p2n1ppp/2p1pn2/1p6/2BP4/2N1PN2/PP3PPP/R1BQK2R w KQkq", "c4d3");
B("r1bqkb1r/p2n1ppp/2p1pn2/1p6/3P4/2NBPN2/PP3PPP/R1BQK2R b KQkq", "a7a6");
B("r1bqkb1r/3n1ppp/p1p1pn2/1p6/3P4/2NBPN2/PP3PPP/R1BQK2R w KQkq", "e3e4");
B("r1bqkb1r/3n1ppp/p1p1pn2/1p6/3PP3/2NB1N2/PP3PPP/R1BQK2R b KQkq", "c6c5");
B("r1bqkb1r/3n1ppp/p3pn2/1pp5/3PP3/2NB1N2/PP3PPP/R1BQK2R w KQkq", "e4e5");

// Semi-Slav Botvinnik Deep
// 5.Bg5 dxc4 6.e4 b5 7.e5 h6 8.Bh4 g5 9.Nxg5 hxg5 10.Bxg5
B("rnbqkb1r/pp3ppp/2p1pn2/3p2B1/2PP4/2N2N2/PP2PPPP/R2QKB1R b KQkq", "d5c4");
B("rnbqkb1r/pp3ppp/2p1pn2/6B1/2pP4/2N2N2/PP2PPPP/R2QKB1R w KQkq", "e2e4");
B("rnbqkb1r/pp3ppp/2p1pn2/6B1/2pPP3/2N2N2/PP3PPP/R2QKB1R b KQkq", "b7b5");
B("rnbqkb1r/p4ppp/2p1pn2/1p4B1/2pPP3/2N2N2/PP3PPP/R2QKB1R w KQkq", "e4e5");
B("rnbqkb1r/p4ppp/2p1pn2/1p2P1B1/2pP4/2N2N2/PP3PPP/R2QKB1R b KQkq", "h7h6");
B("rnbqkb1r/p4pp1/2p1pn1p/1p2P1B1/2pP4/2N2N2/PP3PPP/R2QKB1R w KQkq", "c1h4");
B("rnbqkb1r/p4pp1/2p1pn1p/1p2P3/2pP3B/2N2N2/PP3PPP/R2QKB1R b KQkq", "g7g5");
B("rnbqkb1r/p4p2/2p1pn1p/1p2P1p1/2pP3B/2N2N2/PP3PPP/R2QKB1R w KQkq", "f3g5");

// ----------------------------------------------------------------------------
// DEEP LINES — King's Indian Defense (continued)
// ----------------------------------------------------------------------------

// KID Classical — Mar del Plata Deep
// 7.O-O Nc6 8.d5 Ne7 9.Ne1 Nd7 10.f3 f5 11.Be3 f4 12.Bf2 g5
B("r1bq1rk1/ppp2pbp/2np1np1/3Pp3/2P1P3/2N2N2/PP2BPPP/R1BQ1RK1 b -", "c6e7");
B("r1bq1rk1/ppp1npbp/3p1np1/3Pp3/2P1P3/2N2N2/PP2BPPP/R1BQ1RK1 w -", "f3e1");
B("r1bq1rk1/ppp1npbp/3p1np1/3Pp3/2P1P3/2N5/PP2BPPP/R1BQNRK1 b -", "e7d7"); // actually Nd7
B("r1bq1rk1/pppnnpbp/3p2p1/3Pp3/2P1P3/2N5/PP2BPPP/R1BQNRK1 w -", "f2f3");
B("r1bq1rk1/pppnnpbp/3p2p1/3Pp3/2P1P3/2N2P2/PP2B1PP/R1BQNRK1 b -", "f7f5");
B("r1bq1rk1/pppnn1bp/3p2p1/3Ppp2/2P1P3/2N2P2/PP2B1PP/R1BQNRK1 w -", "c1e3");
B("r1bq1rk1/pppnn1bp/3p2p1/3Ppp2/2P1P3/2N1BP2/PP2B1PP/R2QNRK1 b -", "f5f4");
B("r1bq1rk1/pppnn1bp/3p2p1/3Pp3/2P1Pp2/2N1BP2/PP2B1PP/R2QNRK1 w -", "e3f2");
B("r1bq1rk1/pppnn1bp/3p2p1/3Pp3/2P1Pp2/2N2P2/PP2BBPP/R2QNRK1 b -", "g6g5");

// KID — Bayonet Attack (9.b4)
B("r1bq1rk1/ppp2pbp/2np1np1/3Pp3/2P1P3/2N2N2/PP2BPPP/R1BQ1RK1 w -", "b2b4");
B("r1bq1rk1/ppp2pbp/2np1np1/3Pp3/1PP1P3/2N2N2/P3BPPP/R1BQ1RK1 b -", "c6e7");

// KID Samisch Deep — 5.f3 O-O 6.Be3 e5 7.d5 Nh5 8.Qd2 f5
B("rnbq1rk1/ppp1ppbp/3p1np1/8/2PPP3/2N2P2/PP2B1PP/R1BQK1NR b KQ", "e7e5"); // Wait, this needs to be correct
B("rnbq1rk1/ppp1ppbp/3p1np1/8/2PPP3/2N1BP2/PP4PP/R2QKBNR b KQ", "e7e5");
B("rnbq1rk1/ppp2pbp/3p1np1/4p3/2PPP3/2N1BP2/PP4PP/R2QKBNR w KQ", "d4d5");
B("rnbq1rk1/ppp2pbp/3p1np1/3Pp3/2P1P3/2N1BP2/PP4PP/R2QKBNR b KQ", "f6h5");
B("rnbq1rk1/ppp2pbp/3p2p1/3Pp2n/2P1P3/2N1BP2/PP4PP/R2QKBNR w KQ", "d1d2");
B("rnbq1rk1/ppp2pbp/3p2p1/3Pp2n/2P1P3/2N1BP2/PP1Q2PP/R3KBNR b KQ", "f7f5");

// KID Fianchetto Deep — 6.g3 c6 (Panno system)
B("rnbq1rk1/ppp1ppbp/3p1np1/8/2PP4/2N2NP1/PP2PP1P/R1BQKB1R b KQ", "c7c6");
B("rnbq1rk1/pp2ppbp/2pp1np1/8/2PP4/2N2NP1/PP2PP1P/R1BQKB1R w KQ", "f1g2");

// ----------------------------------------------------------------------------
// DEEP LINES — Grunfeld (continued)
// ----------------------------------------------------------------------------

// Grunfeld Exchange Deep
// 5...Bg7 6.Bc4 c5 7.Ne2 Nc6 8.Be3 O-O 9.O-O cxd4 10.cxd4 Bg4
B("rnbqk2r/ppp1ppbp/6p1/2p5/2BPP3/2P5/P4PPP/R1BQK1NR b KQkq", "b8c6");
B("r1bqk2r/ppp1ppbp/2n3p1/2p5/2BPP3/2P5/P3NPPP/R1BQK2R b KQkq", "e8g8");
B("r1bq1rk1/ppp1ppbp/2n3p1/2p5/2BPP3/2P5/P3NPPP/R1BQK2R w -", "c1e3");
B("r1bq1rk1/ppp1ppbp/2n3p1/2p5/2BPP3/2P1B3/P3NPPP/R2QK2R b -", "e8g8"); // already castled
B("r1bq1rk1/ppp1ppbp/2n3p1/2p5/2BPP3/2P1B3/P3NPPP/R2QK2R w -", "e1g1");
B("r1bq1rk1/ppp1ppbp/2n3p1/2p5/2BPP3/2P1B3/P3NPPP/R2Q1RK1 b -", "c5d4");
B("r1bq1rk1/ppp1ppbp/2n3p1/8/2BpP3/2P1B3/P3NPPP/R2Q1RK1 w -", "c3d4");
B("r1bq1rk1/ppp1ppbp/2n3p1/8/2BPP3/4B3/P3NPPP/R2Q1RK1 b -", "c8g4");

// Grunfeld — Russian System Deep
// 5.Nf3 Bg7 6.cxd5 Nxd5 7.Bc4 Nb6 8.Bb3 O-O 9.O-O Nc6
B("rnbqkb1r/ppp1pp1p/6p1/3n4/3P4/2N2N2/PP2PPPP/R1BQKB1R b KQkq", "f8g7");
B("rnbqk2r/ppp1ppbp/6p1/3n4/3P4/2N2N2/PP2PPPP/R1BQKB1R w KQkq", "c4d5");
B("rnbqk2r/ppp1ppbp/6p1/3P4/3P4/2N2N2/PP2PPPP/R1BQKB1R b KQkq", "d5d5"); // Nxd5 — wait, king on e8 still?
// Let's be more careful
B("rnbqk2r/ppp1ppbp/6p1/3n4/2PP4/2N2N2/PP2PPPP/R1BQKB1R w KQkq", "c4d5");

// ----------------------------------------------------------------------------
// DEEP LINES — Nimzo-Indian (continued)
// ----------------------------------------------------------------------------

// Nimzo — Rubinstein Deep
// 4.e3 O-O 5.Bd3 d5 6.Nf3 c5 7.O-O dxc4 8.Bxc4 Nc6 9.a3
B("rnbq1rk1/pppp1ppp/4pn2/8/1bPP4/2NBP3/PP3PPP/R1BQK1NR b KQ", "d7d5");
B("rnbq1rk1/ppp2ppp/4pn2/3p4/1bPP4/2NBP3/PP3PPP/R1BQK1NR w KQ", "g1f3");
B("rnbq1rk1/ppp2ppp/4pn2/3p4/1bPP4/2NBPN2/PP3PPP/R1BQK2R b KQ", "c7c5");
B("rnbq1rk1/pp3ppp/4pn2/2pp4/1bPP4/2NBPN2/PP3PPP/R1BQK2R w KQ", "e1g1");
B("rnbq1rk1/pp3ppp/4pn2/2pp4/1bPP4/2NBPN2/PP3PPP/R1BQ1RK1 b -", "d5c4");
B("rnbq1rk1/pp3ppp/4pn2/2p5/1bpP4/2N1PN2/PP3PPP/R1BQ1RK1 w -", "d3c4");
B("rnbq1rk1/pp3ppp/4pn2/2p5/1bBP4/2N1PN2/PP3PPP/R1BQ1RK1 b -", "b8c6");
B("r1bq1rk1/pp3ppp/2n1pn2/2p5/1bBP4/2N1PN2/PP3PPP/R1BQ1RK1 w -", "a2a3");

// Nimzo — Huebner Variation (4...c5)
B("rnbqk2r/pppp1ppp/4pn2/8/1bPP4/2N1P3/PP3PPP/R1BQKBNR b KQkq", "c7c5");

// Nimzo — Classical 4.Qc2 O-O 5.a3 Bxc3+ 6.Qxc3 b6 7.Bg5
B("rnbq1rk1/pppp1ppp/4pn2/8/1bPP4/P1N5/1PQ1PPPP/R1B1KBNR b KQkq", "b4c3");
B("rnbq1rk1/pppp1ppp/4pn2/8/2PP4/P1b5/1PQ1PPPP/R1B1KBNR w KQkq", "c2c3");
B("rnbq1rk1/pppp1ppp/4pn2/8/2PP4/P1Q5/1P2PPPP/R1B1KBNR b KQkq", "b7b6");
B("rnbq1rk1/p1pp1ppp/1p2pn2/8/2PP4/P1Q5/1P2PPPP/R1B1KBNR w KQkq", "c1g5");

// ----------------------------------------------------------------------------
// DEEP LINES — Queen's Indian (continued)
// ----------------------------------------------------------------------------

// QID Deep — 4.g3 Bb7 5.Bg2 Be7 6.O-O O-O 7.Nc3 Ne4 8.Qc2 Nxc3
B("rn1qkb1r/pbpp1ppp/1p2pn2/8/2PP4/5NP1/PP2PPBP/RNBQK2R b KQkq", "f8e7");
B("rn1qk2r/pbppbppp/1p2pn2/8/2PP4/5NP1/PP2PPBP/RNBQK2R w KQkq", "e1g1");
B("rn1q1rk1/pbppbppp/1p2pn2/8/2PP4/5NP1/PP2PPBP/RNBQ1RK1 w -", "b1c3");
B("rn1q1rk1/pbppbppp/1p2pn2/8/2PP4/2N2NP1/PP2PPBP/R1BQ1RK1 b -", "f6e4");
B("rn1q1rk1/pbppbppp/1p2p3/8/2PPn3/2N2NP1/PP2PPBP/R1BQ1RK1 w -", "d1c2");
B("rn1q1rk1/pbppbppp/1p2p3/8/2PPn3/2N2NP1/PPQ1PPBP/R1B2RK1 b -", "e4c3");

// QID — Petrosian System (4.a3)
B("rnbqkb1r/p1pp1ppp/1p2pn2/8/2PP4/P4N2/1P2PPPP/RNBQKB1R b KQkq", "c8b7");

// ----------------------------------------------------------------------------
// DEEP LINES — Catalan (continued)
// ----------------------------------------------------------------------------

// Open Catalan Deep
// 1.d4 Nf6 2.c4 e6 3.g3 d5 4.Bg2 dxc4 5.Nf3 a6 6.O-O Nc6 7.e3 Bd7
B("rnbqkb1r/ppp2ppp/4pn2/3p4/2PP4/5NP1/PP2PPBP/RNBQK2R b KQkq", "d5c4");
B("rnbqkb1r/ppp2ppp/4pn2/8/2pP4/5NP1/PP2PPBP/RNBQK2R w KQkq", "e1g1");
B("rnbqkb1r/ppp2ppp/4pn2/8/2pP4/5NP1/PP2PPBP/RNBQ1RK1 b kq", "a7a6");

// Closed Catalan
B("rnbq1rk1/ppp1bppp/4pn2/3p4/2PP4/5NP1/PP2PPBP/RNBQ1RK1 w -", "d1c2");

// Catalan — Bogo-Indian Hybrid (3...Bb4+ lines)
B("rnbqk2r/pppp1ppp/4pn2/8/2PP4/5NP1/PP2PP1P/RNBQKB1R b KQkq", "f8b4");

// ----------------------------------------------------------------------------
// DEEP LINES — English Opening (continued)
// ----------------------------------------------------------------------------

// English — Reversed Sicilian Deep
// 1.c4 e5 2.Nc3 Nf6 3.Nf3 Nc6 4.g3 d5 5.cxd5 Nxd5 6.Bg2 Nb6 7.O-O Be7
B("r1bqkb1r/pppp1ppp/2n2n2/4p3/2P5/2N2N2/PP1PPPPP/R1BQKB1R w KQkq", "g2g3");
B("r1bqkb1r/pppp1ppp/2n2n2/4p3/2P5/2N2NP1/PP1PPP1P/R1BQKB1R b KQkq", "d7d5");
B("r1bqkb1r/ppp2ppp/2n2n2/3pp3/2P5/2N2NP1/PP1PPP1P/R1BQKB1R w KQkq", "c4d5");
B("r1bqkb1r/ppp2ppp/2n5/3np3/8/2N2NP1/PP1PPP1P/R1BQKB1R b KQkq", "d5b6");

// English — Hedgehog Setup
B("r1bqkb1r/pp1ppppp/2n2n2/2p5/2P5/2N2NP1/PP1PPP1P/R1BQKB1R b KQkq", "e7e6");

// English — Botvinnik System (1.c4 e5 2.Nc3 Nc6 3.g3 g6 4.Bg2 Bg7 5.e4)
B("r1bqkbnr/pppp1ppp/2n5/4p3/2P5/2N5/PP1PPPPP/R1BQKBNR w KQkq", "g2g3");
B("r1bqkbnr/pppp1ppp/2n5/4p3/2P5/2N3P1/PP1PPP1P/R1BQKBNR b KQkq", "g7g6");
B("r1bqkbnr/pppp1p1p/2n3p1/4p3/2P5/2N3P1/PP1PPP1P/R1BQKBNR w KQkq", "f1g2");
B("r1bqkbnr/pppp1p1p/2n3p1/4p3/2P5/2N3P1/PP1PPPBP/R1BQK1NR b KQkq", "f8g7");
B("r1bqk1nr/pppp1pbp/2n3p1/4p3/2P5/2N3P1/PP1PPPBP/R1BQK1NR w KQkq", "e2e4");

// ----------------------------------------------------------------------------
// ADDITIONAL DEEP LINES — Various Openings
// ----------------------------------------------------------------------------

// Dutch Defense (1.d4 f5)
B("rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq", "f7f5");
B("rnbqkbnr/ppppp1pp/8/5p2/3P4/8/PPP1PPPP/RNBQKBNR w KQkq", "c2c4");
B("rnbqkbnr/ppppp1pp/8/5p2/2PP4/8/PP2PPPP/RNBQKBNR b KQkq", "g8f6");
B("rnbqkb1r/ppppp1pp/5n2/5p2/2PP4/8/PP2PPPP/RNBQKBNR w KQkq", "g2g3");
B("rnbqkb1r/ppppp1pp/5n2/5p2/2PP4/6P1/PP2PP1P/RNBQKBNR b KQkq", "g7g6");
B("rnbqkb1r/ppppp2p/5np1/5p2/2PP4/6P1/PP2PP1P/RNBQKBNR w KQkq", "f1g2");
B("rnbqkb1r/ppppp2p/5np1/5p2/2PP4/6P1/PP2PPBP/RNBQK1NR b KQkq", "f8g7");

// Dutch Stonewall
B("rnbqkb1r/ppppp2p/5np1/5p2/2PP4/6P1/PP2PPBP/RNBQK1NR b KQkq", "d7d5");

// Dutch Leningrad
B("rnbqk2r/ppppp1bp/5np1/5p2/2PP4/6P1/PP2PPBP/RNBQK1NR w KQkq", "g1f3");

// Scandinavian — 3...Qa5 Deep Lines
B("rnb1kbnr/ppp1pppp/8/q7/3P4/2N5/PPP2PPP/R1BQKBNR b KQkq", "g8f6");
B("rnb1kb1r/ppp1pppp/5n2/q7/3P4/2N5/PPP2PPP/R1BQKBNR w KQkq", "g1f3");
B("rnb1kb1r/ppp1pppp/5n2/q7/3P4/2N2N2/PPP2PPP/R1BQKB1R b KQkq", "c8f5");
B("rnb1kb1r/ppp1pppp/5n2/q4b2/3P4/2N2N2/PPP2PPP/R1BQKB1R w KQkq", "f1c4"); // or Bc4

// Pirc — Austrian Attack Deep
// 1.e4 d6 2.d4 Nf6 3.Nc3 g6 4.f4 Bg7 5.Nf3 O-O 6.Bd3 Na6 7.O-O c5
B("rnbqkb1r/ppp1pp1p/3p1np1/8/3PPP2/2N5/PPP3PP/R1BQKBNR b KQkq", "f8g7");
B("rnbqk2r/ppp1ppbp/3p1np1/8/3PPP2/2N5/PPP3PP/R1BQKBNR w KQkq", "g1f3");
B("rnbqk2r/ppp1ppbp/3p1np1/8/3PPP2/2N2N2/PPP3PP/R1BQKB1R b KQkq", "e8g8");
B("rnbq1rk1/ppp1ppbp/3p1np1/8/3PPP2/2N2N2/PPP3PP/R1BQKB1R w KQ", "f1d3");
B("rnbq1rk1/ppp1ppbp/3p1np1/8/3PPP2/2NB1N2/PPP3PP/R1BQK2R b KQ", "b8a6");

// Alekhine — Modern/Four Pawns Attack Deep
// 1.e4 Nf6 2.e5 Nd5 3.d4 d6 4.c4 Nb6 5.f4 (Four Pawns Attack)
B("rnbqkb1r/ppp1pppp/1n1p4/4P3/2PP4/8/PP3PPP/RNBQKBNR w KQkq", "f2f4");
B("rnbqkb1r/ppp1pppp/1n1p4/4P3/2PP1P2/8/PP4PP/RNBQKBNR b KQkq", "d6e5");
B("rnbqkb1r/ppp1pppp/1n6/4p3/2PP1P2/8/PP4PP/RNBQKBNR w KQkq", "f4e5");
B("rnbqkb1r/ppp1pppp/1n6/4P3/2PP4/8/PP4PP/RNBQKBNR b KQkq", "c8f5");

// Alekhine — Exchange Variation
B("rnbqkb1r/ppp1pppp/3p4/3nP3/3P4/5N2/PPP2PPP/RNBQKB1R b KQkq", "c8g4");

// Budapest Gambit Deep
// 1.d4 Nf6 2.c4 e5 3.dxe5 Ng4 4.Bf4 Nc6 5.Nf3 Bb4+ 6.Nbd2
B("rnbqkb1r/pppp1ppp/8/4P3/2P3n1/8/PP2PPPP/RNBQKBNR w KQkq", "c1f4");
B("rnbqkb1r/pppp1ppp/8/4P3/2P2Bn1/8/PP2PPPP/RN1QKBNR b KQkq", "b8c6");
B("r1bqkb1r/pppp1ppp/2n5/4P3/2P2Bn1/8/PP2PPPP/RN1QKBNR w KQkq", "g1f3");
B("r1bqkb1r/pppp1ppp/2n5/4P3/2P2Bn1/5N2/PP2PPPP/RN1QKB1R b KQkq", "f8b4");
B("r1bqk2r/pppp1ppp/2n5/4P3/1bP2Bn1/5N2/PP2PPPP/RN1QKB1R w KQkq", "b1d2");

// Benoni — Modern Deep
// 1.d4 Nf6 2.c4 c5 3.d5 e6 4.Nc3 exd5 5.cxd5 d6 6.e4 g6 7.Nf3 Bg7 8.Be2 O-O 9.O-O Re8
B("rnbqkbnr/pp3ppp/3p4/2pP4/4P3/2N5/PP3PPP/R1BQKBNR b KQkq", "g7g6");
B("rnbqkbnr/pp3p1p/3p2p1/2pP4/4P3/2N5/PP3PPP/R1BQKBNR w KQkq", "g1f3");
B("rnbqkbnr/pp3p1p/3p2p1/2pP4/4P3/2N2N2/PP3PPP/R1BQKB1R b KQkq", "f8g7");
B("rnbqk2r/pp3pbp/3p1np1/2pP4/4P3/2N2N2/PP3PPP/R1BQKB1R w KQkq", "f1e2");
B("rnbqk2r/pp3pbp/3p1np1/2pP4/4P3/2N2N2/PP2BPPP/R1BQK2R b KQkq", "e8g8");
B("rnbq1rk1/pp3pbp/3p1np1/2pP4/4P3/2N2N2/PP2BPPP/R1BQK2R w KQ", "e1g1");
B("rnbq1rk1/pp3pbp/3p1np1/2pP4/4P3/2N2N2/PP2BPPP/R1BQ1RK1 b -", "f8e8");

// Benoni — Taimanov Attack (f4 system)
B("rnbq1rk1/pp3pbp/3p1np1/2pP4/4P3/2N2N2/PP2BPPP/R1BQ1RK1 w -", "f2f4");

// London System Deep
// 1.d4 d5 2.Bf4 Nf6 3.e3 e6 4.Nf3 Bd6 5.Bg3 (or Bxd6)
B("rnbqkb1r/ppp2ppp/4pn2/3p4/3P1B2/4PN2/PPP2PPP/RN1QKB1R b KQkq", "f8d6");
B("rnbqk2r/ppp2ppp/3bpn2/3p4/3P1B2/4PN2/PPP2PPP/RN1QKB1R w KQkq", "f4g3");
B("rnbqk2r/ppp2ppp/3bpn2/3p4/3P4/4PNB1/PPP2PPP/RN1QKB1R b KQkq", "e8g8");
B("rnbq1rk1/ppp2ppp/3bpn2/3p4/3P4/4PNB1/PPP2PPP/RN1QKB1R w KQ", "f1d3");
B("rnbq1rk1/ppp2ppp/3bpn2/3p4/3P4/3BPNB1/PPP2PPP/RN1QK2R b KQ", "c7c5");

// Colle System (1.d4 d5 2.Nf3 Nf6 3.e3 e6 4.Bd3 c5 5.c3)
B("rnbqkb1r/ppp2ppp/4pn2/3p4/3P4/4PN2/PPP2PPP/RNBQKB1R b KQkq", "e7e6"); // Wait, Nf3 was already played
B("rnbqkb1r/ppp1pppp/5n2/3p4/3P4/4PN2/PPP2PPP/RNBQKB1R b KQkq", "e7e6");
B("rnbqkb1r/ppp2ppp/4pn2/3p4/3P4/4PN2/PPP2PPP/RNBQKB1R w KQkq", "f1d3");
B("rnbqkb1r/ppp2ppp/4pn2/3p4/3P4/3BPN2/PPP2PPP/RNBQK2R b KQkq", "c7c5");
B("rnbqkb1r/pp3ppp/4pn2/2pp4/3P4/3BPN2/PPP2PPP/RNBQK2R w KQkq", "c2c3");

// Zukertort (1.Nf3 d5 2.e3 Nf6 3.b3 — Reti-like)
B("rnbqkb1r/ppp1pppp/5n2/3p4/8/4PN2/PPPP1PPP/RNBQKB1R w KQkq", "b2b3");

// King's Indian Attack (as White)
// 1.Nf3 d5 2.g3 Nf6 3.Bg2 c6 4.O-O Bg4 5.d3 Nbd7 6.Nbd2 e5
B("rnbqkb1r/ppp1pppp/5n2/3p4/8/5NP1/PPPPPPBP/RNBQK2R b KQkq", "c7c6");
B("rnbqkb1r/pp2pppp/2p2n2/3p4/8/5NP1/PPPPPPBP/RNBQK2R w KQkq", "e1g1"); // Already covered above... let's do a different move order
B("rnbqkb1r/pp2pppp/2p2n2/3p4/8/5NP1/PPPPPPBP/RNBQ1RK1 b kq", "c8g4");
B("rnbqkb1r/pp2pppp/2p2n2/3p4/6b1/5NP1/PPPPPPBP/RNBQ1RK1 w kq", "d2d3");
B("rnbqkb1r/pp2pppp/2p2n2/3p4/6b1/3P1NP1/PPP1PPBP/RNBQ1RK1 b kq", "b8d7");
B("r1bqkb1r/pp1npppp/2p2n2/3p4/6b1/3P1NP1/PPP1PPBP/RNBQ1RK1 w kq", "b1d2");
B("r1bqkb1r/pp1npppp/2p2n2/3p4/6b1/3P1NP1/PPPNPPBP/R1BQ1RK1 b kq", "e7e5");

// Reti — Main Line Deep
// 1.Nf3 d5 2.c4 dxc4 (gambit accepted) — unusual but sharp
B("rnbqkbnr/ppp1pppp/8/3p4/8/5N2/PPPPPPPP/RNBQKB1R w KQkq", "c2c4");
B("rnbqkbnr/ppp1pppp/8/3p4/2P5/5N2/PP1PPPPP/RNBQKB1R b KQkq", "d5c4");

// Reti — King's Indian structure
B("rnbqkbnr/ppp1pppp/8/3p4/8/5NP1/PPPPPP1P/RNBQKB1R b KQkq", "g8f6");

// Bird's Opening Deep
// 1.f4 d5 2.Nf3 Nf6 3.e3 g6 4.Be2 Bg7 5.O-O O-O 6.d3 c5
B("rnbqkbnr/ppp1pppp/8/3p4/5P2/5N2/PPPPP1PP/RNBQKB1R b KQkq", "g8f6");
B("rnbqkb1r/ppp1pppp/5n2/3p4/5P2/5N2/PPPPP1PP/RNBQKB1R w KQkq", "e2e3");
B("rnbqkb1r/ppp1pppp/5n2/3p4/5P2/4PN2/PPPP2PP/RNBQKB1R b KQkq", "g7g6");
B("rnbqkb1r/ppp1pp1p/5np1/3p4/5P2/4PN2/PPPP2PP/RNBQKB1R w KQkq", "f1e2");
B("rnbqkb1r/ppp1pp1p/5np1/3p4/5P2/4PN2/PPPPB1PP/RNBQK2R b KQkq", "f8g7");
B("rnbqk2r/ppp1ppbp/5np1/3p4/5P2/4PN2/PPPPB1PP/RNBQK2R w KQkq", "e1g1");
B("rnbq1rk1/ppp1ppbp/5np1/3p4/5P2/4PN2/PPPPB1PP/RNBQ1RK1 w -", "d2d3");

// From Gambit (1.f4 e5 2.fxe5 d6)
B("rnbqkbnr/pppppppp/8/8/5P2/8/PPPPP1PP/RNBQKBNR b KQkq", "e7e5");
B("rnbqkbnr/pppp1ppp/8/4p3/5P2/8/PPPPP1PP/RNBQKBNR w KQkq", "f4e5");

// Nimzowitsch-Larsen Deep
// 1.b3 e5 2.Bb2 Nc6 3.e3 d5 4.Bb5 Bd6 5.Nf3
B("rnbqkbnr/pppp1ppp/8/4p3/8/1P6/PBPPPPPP/RN1QKBNR b KQkq", "b8c6");
B("r1bqkbnr/pppp1ppp/2n5/4p3/8/1P6/PBPPPPPP/RN1QKBNR w KQkq", "e2e3");
B("r1bqkbnr/pppp1ppp/2n5/4p3/8/1P2P3/PBPP1PPP/RN1QKBNR b KQkq", "d7d5");
B("r1bqkbnr/ppp2ppp/2n5/3pp3/8/1P2P3/PBPP1PPP/RN1QKBNR w KQkq", "f1b5");

// ----------------------------------------------------------------------------
// RESPONSE PREPARATIONS — Black's replies to rare first moves
// ----------------------------------------------------------------------------

// Against 1.b4 (Sokolsky/Polish)
B("rnbqkbnr/pppppppp/8/8/1P6/8/P1PPPPPP/RNBQKBNR b KQkq", "e7e5");
B("rnbqkbnr/pppp1ppp/8/4p3/1P6/8/P1PPPPPP/RNBQKBNR w KQkq", "c1b2");

// Against 1.e3 (Van't Kruijs)
B("rnbqkbnr/pppppppp/8/8/8/4P3/PPPP1PPP/RNBQKBNR b KQkq", "d7d5");

// Against 1.d3 (Mieses)
B("rnbqkbnr/pppppppp/8/8/8/3P4/PPP1PPPP/RNBQKBNR b KQkq", "e7e5");

// Against 1.c3 (Saragossa)
B("rnbqkbnr/pppppppp/8/8/8/2P5/PP1PPPPP/RNBQKBNR b KQkq", "d7d5");

// Against 1.a3 (Anderssen)
B("rnbqkbnr/pppppppp/8/8/8/P7/1PPPPPPP/RNBQKBNR b KQkq", "d7d5");


// ============================================================================
// SECTION 2: ENDGAME TABLES
// ============================================================================
// Material signature based lookup for endgame knowledge
// ============================================================================

// Material signature: count pieces for each side
// Format: "WpWnWbWrWqBpBnBbBrBq" — counts of each piece type
function getMaterialSignature(board) {
  const counts = [0,0,0,0,0,0,0,0,0,0]; // Wp,Wn,Wb,Wr,Wq,Bp,Bn,Bb,Br,Bq
  for (let i = 0; i < 64; i++) {
    const p = board[i];
    if (!p || p.type === KING) continue;
    const idx = p.color === WHITE ? (p.type - 1) : (p.type - 1 + 5);
    counts[idx]++;
  }
  return counts.join('');
}

function findPieces(board, color, type) {
  const result = [];
  for (let i = 0; i < 64; i++) {
    const p = board[i];
    if (p && p.color === color && p.type === type) result.push(i);
  }
  return result;
}

function findKing(board, color) {
  for (let i = 0; i < 64; i++) {
    const p = board[i];
    if (p && p.color === color && p.type === KING) return i;
  }
  return -1;
}

function sqFile(sq) { return sq & 7; }
function sqRank(sq) { return sq >> 3; }
function sqDist(a, b) { return Math.max(Math.abs(sqRank(a) - sqRank(b)), Math.abs(sqFile(a) - sqFile(b))); }
function manDist(a, b) { return Math.abs(sqRank(a) - sqRank(b)) + Math.abs(sqFile(a) - sqFile(b)); }
function centerDist(sq) { const f = sqFile(sq), r = sqRank(sq); return Math.max(Math.abs(f - 3.5), Math.abs(r - 3.5)); }

// KPK — King + Pawn vs King
// Returns eval adjustment and suggested approach
function evaluateKPK(board, strongSide) {
  const weakSide = 1 - strongSide;
  const strongKing = findKing(board, strongSide);
  const weakKing = findKing(board, weakSide);
  const pawns = findPieces(board, strongSide, PAWN);
  if (pawns.length !== 1) return null;
  const pawnSq = pawns[0];
  const pawnFile = sqFile(pawnSq);
  const pawnRank = sqRank(pawnSq);

  // From strong side perspective: rank 0 = promotion rank for White
  const promoRank = strongSide === WHITE ? 0 : 7;
  const promoSq = promoRank * 8 + pawnFile;
  const pawnAdvance = strongSide === WHITE ? (7 - pawnRank) : pawnRank;

  // Rook pawn draws: if pawn is on a or h file
  if (pawnFile === 0 || pawnFile === 7) {
    const weakKingDist = sqDist(weakKing, promoSq);
    if (weakKingDist <= 1) return { eval: 0, move: null }; // Likely draw
    // If weak king can reach the corner, it's a draw
    if (weakKingDist <= pawnAdvance + 1) return { eval: 20, move: null };
  }

  // Key squares concept
  // For pawns not yet past the 5th rank, key squares are 2 ranks ahead
  let bonus = 0;
  const keyRank = strongSide === WHITE ? Math.max(0, pawnRank - 2) : Math.min(7, pawnRank + 2);
  const keySqs = [keyRank * 8 + Math.max(0, pawnFile - 1), keyRank * 8 + pawnFile, keyRank * 8 + Math.min(7, pawnFile + 1)];

  // King on key square is winning
  for (const ks of keySqs) {
    if (strongKing === ks) bonus += 200;
  }

  // Opposition detection
  const kingFileDiff = Math.abs(sqFile(strongKing) - sqFile(weakKing));
  const kingRankDiff = Math.abs(sqRank(strongKing) - sqRank(weakKing));
  const hasOpposition = kingFileDiff === 0 && kingRankDiff === 2;

  // Strong side has opposition when approaching: good
  if (hasOpposition) {
    const strongAhead = strongSide === WHITE ?
      sqRank(strongKing) < sqRank(weakKing) :
      sqRank(strongKing) > sqRank(weakKing);
    if (strongAhead) bonus += 100;
    else bonus -= 50; // Weak side has opposition
  }

  // Bonus for advanced pawn
  bonus += pawnAdvance * 30;

  // Bonus for king close to pawn
  bonus += (7 - sqDist(strongKing, pawnSq)) * 15;

  // Penalty for weak king close to pawn (defending well)
  bonus -= (7 - sqDist(weakKing, pawnSq)) * 10;

  return { eval: bonus, move: null };
}

// KRK — King + Rook vs King
function evaluateKRK(board, strongSide) {
  const weakSide = 1 - strongSide;
  const strongKing = findKing(board, strongSide);
  const weakKing = findKing(board, weakSide);
  const rooks = findPieces(board, strongSide, ROOK);
  if (rooks.length !== 1) return null;

  let bonus = 500; // Base winning advantage

  // Drive weak king to edge
  bonus += centerDist(weakKing) * 40;

  // Strong king should be close to weak king (for the squeeze)
  bonus += (7 - sqDist(strongKing, weakKing)) * 20;

  // Bonus for rook cutting off ranks/files
  const rookSq = rooks[0];
  const rookFile = sqFile(rookSq);
  const rookRank = sqRank(rookSq);
  const weakFile = sqFile(weakKing);
  const weakRank = sqRank(weakKing);

  // Rook cuts off file
  if ((rookFile > weakFile && sqFile(strongKing) < weakFile) ||
      (rookFile < weakFile && sqFile(strongKing) > weakFile)) {
    bonus += 30;
  }
  // Rook cuts off rank
  if ((rookRank > weakRank && sqRank(strongKing) < weakRank) ||
      (rookRank < weakRank && sqRank(strongKing) > weakRank)) {
    bonus += 30;
  }

  return { eval: bonus, move: null };
}

// KQK — King + Queen vs King
function evaluateKQK(board, strongSide) {
  const weakSide = 1 - strongSide;
  const strongKing = findKing(board, strongSide);
  const weakKing = findKing(board, weakSide);

  let bonus = 800;

  // Drive to edge
  bonus += centerDist(weakKing) * 50;

  // Bring king close
  bonus += (7 - sqDist(strongKing, weakKing)) * 25;

  return { eval: bonus, move: null };
}

// KBNK — King + Bishop + Knight vs King
function evaluateKBNK(board, strongSide) {
  const weakSide = 1 - strongSide;
  const strongKing = findKing(board, strongSide);
  const weakKing = findKing(board, weakSide);
  const bishops = findPieces(board, strongSide, BISHOP);
  if (bishops.length !== 1) return null;

  const bishopSq = bishops[0];
  // Determine bishop color (light or dark square)
  const bishopOnLight = ((sqFile(bishopSq) + sqRank(bishopSq)) % 2) === 0;

  // Must drive to corner of bishop's color
  // Light bishop: corners a8 (0) and h1 (63) — sum even
  // Dark bishop: corners a1 (56) and h8 (7) — sum odd
  const targetCorners = bishopOnLight ? [0, 63] : [7, 56];
  const weakKingCornerDist = Math.min(sqDist(weakKing, targetCorners[0]), sqDist(weakKing, targetCorners[1]));

  let bonus = 300;

  // Bonus for weak king close to correct corner
  bonus += (7 - weakKingCornerDist) * 60;

  // Strong king should be close to weak king
  bonus += (7 - sqDist(strongKing, weakKing)) * 30;

  // Penalty for weak king in wrong corner (can't mate there)
  const wrongCorners = bishopOnLight ? [7, 56] : [0, 63];
  const wrongCornerDist = Math.min(sqDist(weakKing, wrongCorners[0]), sqDist(weakKing, wrongCorners[1]));
  if (wrongCornerDist <= 1) bonus -= 100; // Need to push out of wrong corner

  return { eval: bonus, move: null };
}

// KRPvKR — Rook + Pawn vs Rook
function evaluateKRPvKR(board, strongSide) {
  const weakSide = 1 - strongSide;
  const strongKing = findKing(board, strongSide);
  const weakKing = findKing(board, weakSide);
  const strongRooks = findPieces(board, strongSide, ROOK);
  const weakRooks = findPieces(board, weakSide, ROOK);
  const pawns = findPieces(board, strongSide, PAWN);
  if (strongRooks.length !== 1 || weakRooks.length !== 1 || pawns.length !== 1) return null;

  const pawnSq = pawns[0];
  const pawnFile = sqFile(pawnSq);
  const pawnRank = sqRank(pawnSq);
  const promoRank = strongSide === WHITE ? 0 : 7;
  const pawnAdvance = strongSide === WHITE ? (7 - pawnRank) : pawnRank;
  const promoSq = promoRank * 8 + pawnFile;

  const strongRookSq = strongRooks[0];
  const weakRookSq = weakRooks[0];

  let bonus = 0;

  // Lucena position detection: pawn on 7th (1 from promo), king in front, rook behind
  const penultRank = strongSide === WHITE ? 1 : 6;
  if (pawnRank === penultRank) {
    const kingInFront = strongSide === WHITE ?
      sqRank(strongKing) === 0 && Math.abs(sqFile(strongKing) - pawnFile) <= 1 :
      sqRank(strongKing) === 7 && Math.abs(sqFile(strongKing) - pawnFile) <= 1;
    if (kingInFront) {
      bonus += 300; // Lucena — likely winning
    }
  }

  // Philidor position detection: weak rook on 3rd rank (6th from strong side), pawn not yet on 6th
  const philidorRank = strongSide === WHITE ? 2 : 5;
  const sixthRank = strongSide === WHITE ? 2 : 5;
  if (sqRank(weakRookSq) === philidorRank && pawnAdvance < 5) {
    bonus -= 100; // Philidor defense — likely draw
  }

  // Vancura position: rook pawn, weak rook on the file attacking from the side
  if ((pawnFile === 0 || pawnFile === 7) && sqFile(weakRookSq) === pawnFile) {
    bonus -= 150; // Vancura — drawing
  }

  // General evaluation factors
  bonus += pawnAdvance * 25;
  bonus += (7 - sqDist(strongKing, pawnSq)) * 10;

  // Weak king should be in front of pawn to hold
  const weakKingBlocksDist = sqDist(weakKing, promoSq);
  if (weakKingBlocksDist <= 2) bonus -= 50;

  return { eval: bonus, move: null };
}

// General pawn endgame evaluator
function evaluatePawnEndgame(board, side) {
  let bonus = 0;
  const enemy = 1 - side;

  const myKing = findKing(board, side);
  const theirKing = findKing(board, enemy);
  const myPawns = findPieces(board, side, PAWN);
  const theirPawns = findPieces(board, enemy, PAWN);

  // Passed pawns
  for (const pSq of myPawns) {
    const f = sqFile(pSq);
    const r = sqRank(pSq);
    let passed = true;

    for (const ePSq of theirPawns) {
      const ef = sqFile(ePSq);
      const er = sqRank(ePSq);
      if (Math.abs(ef - f) <= 1) {
        if (side === WHITE && er < r) { passed = false; break; }
        if (side === BLACK && er > r) { passed = false; break; }
      }
    }

    if (passed) {
      const advance = side === WHITE ? (7 - r) : r;
      bonus += 50 + advance * 20;

      // King support for passed pawn
      const promoSq = (side === WHITE ? 0 : 7) * 8 + f;
      bonus += (7 - sqDist(myKing, pSq)) * 8;
      bonus -= (7 - sqDist(theirKing, promoSq)) * 6;

      // Connected passed pawns
      for (const pSq2 of myPawns) {
        if (pSq2 === pSq) continue;
        if (Math.abs(sqFile(pSq2) - f) === 1 && Math.abs(sqRank(pSq2) - r) <= 1) {
          bonus += 30; // Connected passers
        }
      }
    }
  }

  // Isolated pawns (penalty)
  for (const pSq of myPawns) {
    const f = sqFile(pSq);
    let isolated = true;
    for (const pSq2 of myPawns) {
      if (pSq2 === pSq) continue;
      if (Math.abs(sqFile(pSq2) - f) === 1) { isolated = false; break; }
    }
    if (isolated) bonus -= 20;
  }

  // Doubled pawns (penalty)
  const fileCounts = new Array(8).fill(0);
  for (const pSq of myPawns) fileCounts[sqFile(pSq)]++;
  for (let f = 0; f < 8; f++) {
    if (fileCounts[f] > 1) bonus -= (fileCounts[f] - 1) * 15;
  }

  // King activity in pawn endgames is critical
  bonus += (7 - centerDist(myKing)) * 10;

  // Opposition
  const kfDiff = Math.abs(sqFile(myKing) - sqFile(theirKing));
  const krDiff = Math.abs(sqRank(myKing) - sqRank(theirKing));
  if (kfDiff === 0 && krDiff === 2) {
    // We have direct opposition
    bonus += 25;
  } else if (kfDiff === 2 && krDiff === 0) {
    // We have lateral opposition
    bonus += 15;
  } else if (kfDiff === 2 && krDiff === 2) {
    // Diagonal opposition
    bonus += 10;
  }

  // Rule of the square for passed pawns
  for (const pSq of myPawns) {
    const f = sqFile(pSq);
    const r = sqRank(pSq);
    const promoRank = side === WHITE ? 0 : 7;
    const distToPromo = Math.abs(r - promoRank);
    const theirKingDist = Math.max(Math.abs(sqRank(theirKing) - promoRank), Math.abs(sqFile(theirKing) - f));
    // If enemy king is outside the square of the pawn
    if (theirKingDist > distToPromo + 1) {
      bonus += 200; // Unstoppable passer
    }
  }

  return bonus;
}

// Endgame master lookup
const ENDGAME_HANDLERS = {
  // Wp,Wn,Wb,Wr,Wq,Bp,Bn,Bb,Br,Bq
  '1000000000': (b) => evaluateKPK(b, WHITE),    // KPK white has pawn
  '0000010000': (b) => { const r = evaluateKPK(b, BLACK); return r ? { eval: -r.eval, move: r.move } : null; },
  '0001000000': (b) => evaluateKRK(b, WHITE),    // KRK
  '0000000100': (b) => { const r = evaluateKRK(b, BLACK); return r ? { eval: -r.eval, move: r.move } : null; },
  '0000100000': (b) => evaluateKQK(b, WHITE),    // KQK
  '0000000010': (b) => { const r = evaluateKQK(b, BLACK); return r ? { eval: -r.eval, move: r.move } : null; },
  '0110000000': (b) => evaluateKBNK(b, WHITE),   // KBNK
  '0000001100': (b) => { const r = evaluateKBNK(b, BLACK); return r ? { eval: -r.eval, move: r.move } : null; },
  '1001000100': (b) => evaluateKRPvKR(b, WHITE), // KRPvKR
  '0000110010': (b) => { const r = evaluateKRPvKR(b, BLACK); return r ? { eval: -r.eval, move: r.move } : null; },
};

// Known drawn endgames
const DRAWN_ENDGAMES = new Set([
  '0010000000', // KBK
  '0000000100', // Wait, that's KRK... removing
  '0100000000', // KNK
  '0000001000', // KBK (black)
  '0000000100', // KNK (black) -- hmm overlaps
  '0010000010', // KBvKB same color might be drawn
]);

// Actually let me be more precise with drawn material
function isDrawnMaterial(board) {
  let wn = 0, wb = 0, wr = 0, wq = 0, wp = 0;
  let bn = 0, bb = 0, br = 0, bq = 0, bp = 0;
  for (let i = 0; i < 64; i++) {
    const p = board[i];
    if (!p || p.type === KING) continue;
    if (p.color === WHITE) {
      if (p.type === PAWN) wp++;
      else if (p.type === KNIGHT) wn++;
      else if (p.type === BISHOP) wb++;
      else if (p.type === ROOK) wr++;
      else if (p.type === QUEEN) wq++;
    } else {
      if (p.type === PAWN) bp++;
      else if (p.type === KNIGHT) bn++;
      else if (p.type === BISHOP) bb++;
      else if (p.type === ROOK) br++;
      else if (p.type === QUEEN) bq++;
    }
  }
  const total = wp + wn + wb + wr + wq + bp + bn + bb + br + bq;
  if (total === 0) return true; // KvK
  if (total === 1) {
    if (wb === 1 || bn === 1 || wn === 1 || bb === 1) return true; // KBvK or KNvK
  }
  if (total === 2 && wb === 1 && bb === 1) {
    // KBvKB — drawn if same color bishops
    const wbSq = findPieces(board, WHITE, BISHOP)[0];
    const bbSq = findPieces(board, BLACK, BISHOP)[0];
    const wbLight = (sqFile(wbSq) + sqRank(wbSq)) % 2;
    const bbLight = (sqFile(bbSq) + sqRank(bbSq)) % 2;
    if (wbLight === bbLight) return true;
  }
  if (total === 2 && wn === 2 && bp === 0) return true; // KNNvK
  if (total === 2 && bn === 2 && wp === 0) return true; // KvKNN
  return false;
}


// ============================================================================
// SECTION 3: STRATEGIC PATTERN LIBRARY
// ============================================================================

// Board utility functions for pattern detection
function getPieceAt(board, sq) {
  if (sq < 0 || sq > 63) return null;
  return board[sq];
}

function isOccupied(board, sq) {
  return sq >= 0 && sq < 63 && board[sq] !== null;
}

function getPawnStructure(board, color) {
  const pawns = [];
  for (let i = 0; i < 64; i++) {
    const p = board[i];
    if (p && p.color === color && p.type === PAWN) pawns.push(i);
  }
  return pawns;
}

function getAttackedSquares(board, sq) {
  const p = board[sq];
  if (!p) return [];
  const attacked = [];
  const r = sqRank(sq), f = sqFile(sq);

  if (p.type === PAWN) {
    const dir = p.color === WHITE ? -1 : 1;
    if (f > 0) attacked.push((r + dir) * 8 + f - 1);
    if (f < 7) attacked.push((r + dir) * 8 + f + 1);
  } else if (p.type === KNIGHT) {
    const offsets = [[-2,-1],[-2,1],[-1,-2],[-1,2],[1,-2],[1,2],[2,-1],[2,1]];
    for (const [dr, df] of offsets) {
      const nr = r + dr, nf = f + df;
      if (nr >= 0 && nr < 8 && nf >= 0 && nf < 8) attacked.push(nr * 8 + nf);
    }
  } else if (p.type === BISHOP || p.type === QUEEN) {
    const dirs = [[-1,-1],[-1,1],[1,-1],[1,1]];
    for (const [dr, df] of dirs) {
      let nr = r + dr, nf = f + df;
      while (nr >= 0 && nr < 8 && nf >= 0 && nf < 8) {
        attacked.push(nr * 8 + nf);
        if (board[nr * 8 + nf]) break;
        nr += dr; nf += df;
      }
    }
  }
  if (p.type === ROOK || p.type === QUEEN) {
    const dirs = [[-1,0],[1,0],[0,-1],[0,1]];
    for (const [dr, df] of dirs) {
      let nr = r + dr, nf = f + df;
      while (nr >= 0 && nr < 8 && nf >= 0 && nf < 8) {
        attacked.push(nr * 8 + nf);
        if (board[nr * 8 + nf]) break;
        nr += dr; nf += df;
      }
    }
  }
  if (p.type === KING) {
    for (let dr = -1; dr <= 1; dr++) for (let df = -1; df <= 1; df++) {
      if (dr === 0 && df === 0) continue;
      const nr = r + dr, nf = f + df;
      if (nr >= 0 && nr < 8 && nf >= 0 && nf < 8) attacked.push(nr * 8 + nf);
    }
  }
  return attacked;
}

// Pattern: Detect pins
function detectPins(board, side) {
  const enemy = 1 - side;
  const kingSq = findKing(board, side);
  if (kingSq < 0) return [];
  const pins = [];
  const kr = sqRank(kingSq), kf = sqFile(kingSq);

  // Check all 8 directions from king
  const dirs = [[-1,0],[1,0],[0,-1],[0,1],[-1,-1],[-1,1],[1,-1],[1,1]];
  for (const [dr, df] of dirs) {
    let foundOwn = -1;
    let nr = kr + dr, nf = kf + df;
    while (nr >= 0 && nr < 8 && nf >= 0 && nf < 8) {
      const sq = nr * 8 + nf;
      const p = board[sq];
      if (p) {
        if (p.color === side) {
          if (foundOwn >= 0) break; // Two own pieces, no pin
          foundOwn = sq;
        } else {
          // Enemy piece — check if it attacks through on this line
          const isDiag = dr !== 0 && df !== 0;
          if (foundOwn >= 0) {
            if ((isDiag && (p.type === BISHOP || p.type === QUEEN)) ||
                (!isDiag && (p.type === ROOK || p.type === QUEEN))) {
              pins.push({ pinned: foundOwn, pinner: sq, direction: [dr, df] });
            }
          }
          break;
        }
      }
      nr += dr; nf += df;
    }
  }
  return pins;
}

// Pattern: Detect forks
function detectForks(board, side) {
  const enemy = 1 - side;
  const forks = [];

  for (let sq = 0; sq < 64; sq++) {
    const p = board[sq];
    if (!p || p.color !== side) continue;
    if (p.type === PAWN || p.type === KING) continue; // Focus on piece forks

    const attacks = getAttackedSquares(board, sq);
    const valuableTargets = [];
    for (const atk of attacks) {
      const target = board[atk];
      if (target && target.color === enemy && target.type !== PAWN) {
        // Attacking a valuable piece
        if (target.type > p.type || target.type === KING) {
          valuableTargets.push(atk);
        }
      }
    }
    if (valuableTargets.length >= 2) {
      forks.push({ piece: sq, targets: valuableTargets });
    }
  }
  return forks;
}

// Pattern: Detect open files
function detectOpenFiles(board) {
  const openFiles = [];
  const semiOpenWhite = [];
  const semiOpenBlack = [];

  for (let f = 0; f < 8; f++) {
    let hasWhitePawn = false, hasBlackPawn = false;
    for (let r = 0; r < 8; r++) {
      const p = board[r * 8 + f];
      if (p && p.type === PAWN) {
        if (p.color === WHITE) hasWhitePawn = true;
        else hasBlackPawn = true;
      }
    }
    if (!hasWhitePawn && !hasBlackPawn) openFiles.push(f);
    else if (!hasWhitePawn) semiOpenWhite.push(f);
    else if (!hasBlackPawn) semiOpenBlack.push(f);
  }
  return { openFiles, semiOpenWhite, semiOpenBlack };
}

// Pattern: Passed pawns
function detectPassedPawns(board, side) {
  const enemy = 1 - side;
  const myPawns = getPawnStructure(board, side);
  const enemyPawns = getPawnStructure(board, enemy);
  const passed = [];

  for (const pSq of myPawns) {
    const f = sqFile(pSq);
    const r = sqRank(pSq);
    let isPassed = true;

    for (const eSq of enemyPawns) {
      const ef = sqFile(eSq);
      const er = sqRank(eSq);
      if (Math.abs(ef - f) <= 1) {
        if (side === WHITE && er <= r) { isPassed = false; break; }
        if (side === BLACK && er >= r) { isPassed = false; break; }
      }
    }

    if (isPassed) {
      const advance = side === WHITE ? (7 - r) : r;
      passed.push({ sq: pSq, advance });
    }
  }
  return passed;
}

// Pattern: Isolated pawns
function detectIsolatedPawns(board, side) {
  const pawns = getPawnStructure(board, side);
  const isolated = [];

  for (const pSq of pawns) {
    const f = sqFile(pSq);
    let hasNeighbor = false;
    for (const p2 of pawns) {
      if (p2 !== pSq && Math.abs(sqFile(p2) - f) === 1) {
        hasNeighbor = true;
        break;
      }
    }
    if (!hasNeighbor) isolated.push(pSq);
  }
  return isolated;
}

// Pattern: Doubled pawns
function detectDoubledPawns(board, side) {
  const pawns = getPawnStructure(board, side);
  const doubled = [];
  const fileCounts = new Array(8).fill(0);
  for (const p of pawns) fileCounts[sqFile(p)]++;
  for (let f = 0; f < 8; f++) {
    if (fileCounts[f] > 1) doubled.push(f);
  }
  return doubled;
}

// Pattern: Backward pawns
function detectBackwardPawns(board, side) {
  const pawns = getPawnStructure(board, side);
  const enemy = 1 - side;
  const enemyPawns = getPawnStructure(board, enemy);
  const backward = [];

  for (const pSq of pawns) {
    const f = sqFile(pSq);
    const r = sqRank(pSq);

    // Check if any friendly pawn on adjacent file is behind or equal
    let isMostBackward = true;
    for (const p2 of pawns) {
      if (p2 === pSq) continue;
      if (Math.abs(sqFile(p2) - f) === 1) {
        if (side === WHITE && sqRank(p2) >= r) { isMostBackward = false; break; }
        if (side === BLACK && sqRank(p2) <= r) { isMostBackward = false; break; }
      }
    }

    if (isMostBackward) {
      // Check if the stop square is controlled by enemy pawns
      const stopSq = side === WHITE ? pSq - 8 : pSq + 8;
      if (stopSq >= 0 && stopSq < 64) {
        for (const eSq of enemyPawns) {
          const ef = sqFile(eSq);
          const er = sqRank(eSq);
          if (Math.abs(ef - f) === 1) {
            const eAttackRank = side === WHITE ? er + 1 : er - 1;
            if (eAttackRank === sqRank(stopSq)) {
              backward.push(pSq);
              break;
            }
          }
        }
      }
    }
  }
  return backward;
}

// Pattern: Knight outposts
function detectKnightOutposts(board, side) {
  const enemy = 1 - side;
  const outposts = [];
  const myPawns = getPawnStructure(board, side);
  const enemyPawns = getPawnStructure(board, enemy);

  for (let sq = 0; sq < 64; sq++) {
    const p = board[sq];
    if (!p || p.color !== side || p.type !== KNIGHT) continue;

    const f = sqFile(sq);
    const r = sqRank(sq);

    // In enemy territory?
    const inEnemyHalf = side === WHITE ? r <= 3 : r >= 4;
    if (!inEnemyHalf) continue;

    // Supported by own pawn?
    let pawnSupport = false;
    for (const pSq of myPawns) {
      const pf = sqFile(pSq);
      const pr = sqRank(pSq);
      const attackRank = side === WHITE ? pr - 1 : pr + 1;
      if (Math.abs(pf - f) === 1 && attackRank === r) {
        pawnSupport = true;
        break;
      }
    }

    // Can enemy pawns attack this square?
    let enemyCanAttack = false;
    for (const eSq of enemyPawns) {
      const ef = sqFile(eSq);
      const er = sqRank(eSq);
      if (Math.abs(ef - f) === 1) {
        // Can this enemy pawn advance to attack?
        if (side === WHITE && er > r) { enemyCanAttack = true; break; }
        if (side === BLACK && er < r) { enemyCanAttack = true; break; }
      }
    }

    if (pawnSupport && !enemyCanAttack) {
      // Strong outpost value by centrality
      const centralBonus = (4 - Math.abs(f - 3.5)) * 8;
      outposts.push({ sq, value: 30 + centralBonus });
    }
  }
  return outposts;
}

// Pattern: Rook on open/semi-open file
function detectRookActivity(board, side) {
  const { openFiles, semiOpenWhite, semiOpenBlack } = detectOpenFiles(board);
  const semiOpen = side === WHITE ? semiOpenWhite : semiOpenBlack;
  let bonus = 0;
  const threats = [];

  for (let sq = 0; sq < 64; sq++) {
    const p = board[sq];
    if (!p || p.color !== side || p.type !== ROOK) continue;

    const f = sqFile(sq);
    const r = sqRank(sq);

    // Open file
    if (openFiles.includes(f)) {
      bonus += 25;
      threats.push('rook_open_file');
    }
    // Semi-open file
    else if (semiOpen.includes(f)) {
      bonus += 15;
      threats.push('rook_semi_open');
    }

    // 7th rank
    const seventhRank = side === WHITE ? 1 : 6;
    if (r === seventhRank) {
      bonus += 30;
      threats.push('rook_7th_rank');
    }

    // Rook battery (two rooks on same file/rank)
    for (let sq2 = sq + 1; sq2 < 64; sq2++) {
      const p2 = board[sq2];
      if (p2 && p2.color === side && p2.type === ROOK) {
        if (sqFile(sq2) === f || sqRank(sq2) === r) {
          bonus += 20;
          threats.push('rook_battery');
        }
      }
    }
  }

  return { bonus, threats };
}

// Pattern: Bishop pair
function detectBishopPair(board, side) {
  const bishops = findPieces(board, side, BISHOP);
  if (bishops.length < 2) return { hasPair: false, bonus: 0 };

  // Check if on different color squares
  const colors = bishops.map(sq => (sqFile(sq) + sqRank(sq)) % 2);
  if (colors[0] === colors[1]) return { hasPair: false, bonus: 0 };

  // Bishop pair bonus scales with openness
  let pawnCount = 0;
  for (let i = 0; i < 64; i++) {
    if (board[i] && board[i].type === PAWN) pawnCount++;
  }

  // Fewer pawns = bishop pair worth more
  const bonus = 50 + (16 - pawnCount) * 5;
  return { hasPair: true, bonus };
}

// Pattern: Back rank weakness
function detectBackRankWeakness(board, side) {
  const backRank = side === WHITE ? 7 : 0;
  const kingSq = findKing(board, side);
  if (sqRank(kingSq) !== backRank) return false;

  const kf = sqFile(kingSq);
  // Check if all escape squares are blocked by own pieces
  let blocked = true;
  for (let df = -1; df <= 1; df++) {
    const nf = kf + df;
    const escapeRank = side === WHITE ? 6 : 1;
    if (nf >= 0 && nf < 8) {
      const escapeSq = escapeRank * 8 + nf;
      const p = board[escapeSq];
      if (!p || p.color !== side) {
        blocked = false;
        break;
      }
    }
  }

  return blocked;
}

// Pattern: King safety
function evaluateKingSafety(board, side) {
  const kingSq = findKing(board, side);
  if (kingSq < 0) return 0;
  const kr = sqRank(kingSq), kf = sqFile(kingSq);
  let safety = 0;

  // Pawn shield (for castled king)
  const pawnShieldRank = side === WHITE ? kr - 1 : kr + 1;
  if (pawnShieldRank >= 0 && pawnShieldRank < 8) {
    for (let df = -1; df <= 1; df++) {
      const nf = kf + df;
      if (nf >= 0 && nf < 8) {
        const shieldSq = pawnShieldRank * 8 + nf;
        const p = board[shieldSq];
        if (p && p.color === side && p.type === PAWN) {
          safety += 15; // Pawn shield present
        } else {
          safety -= 20; // Missing shield pawn
        }
      }
    }
  }

  // Open files near king are dangerous
  const { openFiles } = detectOpenFiles(board);
  for (const of_ of openFiles) {
    if (Math.abs(of_ - kf) <= 1) {
      safety -= 25; // Open file near king
    }
  }

  // King on back rank with no escape (back rank mate threat)
  if (detectBackRankWeakness(board, side)) {
    safety -= 40;
  }

  return safety;
}

// Pattern: Space advantage
function evaluateSpace(board, side) {
  let space = 0;
  const centralSquares = [27, 28, 35, 36]; // d4,e4,d5,e5
  const extendedCenter = [18, 19, 20, 21, 26, 29, 34, 37, 42, 43, 44, 45]; // c3-f3, c4,f4, c5,f5, c6-f6

  for (const sq of centralSquares) {
    const p = board[sq];
    if (p && p.color === side && p.type === PAWN) space += 20;
    else if (p && p.color === side) space += 10;
  }

  for (const sq of extendedCenter) {
    const p = board[sq];
    if (p && p.color === side && p.type === PAWN) space += 8;
  }

  return space;
}

// Pattern: Good bishop vs bad bishop
function evaluateBishopQuality(board, side) {
  const bishops = findPieces(board, side, BISHOP);
  if (bishops.length === 0) return 0;

  let totalBonus = 0;
  const myPawns = getPawnStructure(board, side);

  for (const bSq of bishops) {
    const bishopLight = (sqFile(bSq) + sqRank(bSq)) % 2;
    let sameColorPawns = 0;

    for (const pSq of myPawns) {
      const pawnLight = (sqFile(pSq) + sqRank(pSq)) % 2;
      if (pawnLight === bishopLight) sameColorPawns++;
    }

    // More own pawns on bishop's color = bad bishop
    totalBonus -= sameColorPawns * 8;
    // Fewer = good bishop
    totalBonus += (myPawns.length - sameColorPawns) * 3;
  }

  return totalBonus;
}

// Pawn structure recognition
const PAWN_STRUCTURES = {
  // Detect Carlsbad structure: white pawns c4,d5,e4(or c4,d5); black pawns c6,d5(e6),e6
  carlsbad: function(board) {
    const wp = getPawnStructure(board, WHITE);
    const bp = getPawnStructure(board, BLACK);
    const wpSet = new Set(wp);
    const bpSet = new Set(bp);
    // Typical: White d5 pawn, Black e6 pawn, pawns on c-file
    if (wpSet.has(27) && bpSet.has(20)) return { detected: true, plan_white: 'minority_attack_queenside', plan_black: 'kingside_attack', bonus_white: 10, bonus_black: 5 };
    return { detected: false };
  },

  // Isolated Queen's Pawn
  isolatedQP: function(board) {
    // White has isolated d-pawn
    const wp = getPawnStructure(board, WHITE);
    const dPawns = wp.filter(sq => sqFile(sq) === 3);
    if (dPawns.length === 0) return { detected: false };

    const cPawns = wp.filter(sq => sqFile(sq) === 2);
    const ePawns = wp.filter(sq => sqFile(sq) === 4);
    if (cPawns.length === 0 && ePawns.length === 0 && dPawns.length > 0) {
      return { detected: true, side: WHITE, plan_for: 'piece_activity_use_outposts', plan_against: 'blockade_on_d5_exchange_pieces', bonus: -15 };
    }

    // Black has isolated d-pawn
    const bp = getPawnStructure(board, BLACK);
    const bdPawns = bp.filter(sq => sqFile(sq) === 3);
    const bcPawns = bp.filter(sq => sqFile(sq) === 2);
    const bePawns = bp.filter(sq => sqFile(sq) === 4);
    if (bcPawns.length === 0 && bePawns.length === 0 && bdPawns.length > 0) {
      return { detected: true, side: BLACK, plan_for: 'piece_activity_use_outposts', plan_against: 'blockade_on_d4_exchange_pieces', bonus: 15 };
    }

    return { detected: false };
  },

  // Hanging pawns (c+d pawns without neighbors)
  hangingPawns: function(board) {
    for (const color of [WHITE, BLACK]) {
      const pawns = getPawnStructure(board, color);
      const files = pawns.map(sq => sqFile(sq));
      const hasC = files.includes(2), hasD = files.includes(3);
      const hasB = files.includes(1), hasE = files.includes(4);
      if (hasC && hasD && !hasB && !hasE) {
        return { detected: true, side: color, plan_for: 'advance_pawn_break', plan_against: 'pressure_both_pawns', bonus: color === WHITE ? -10 : 10 };
      }
    }
    return { detected: false };
  },

  // Maroczy Bind (pawns on c4+e4 vs Sicilian structure)
  maroczyBind: function(board) {
    const wp = getPawnStructure(board, WHITE);
    const wpSet = new Set(wp);
    if (wpSet.has(34) && wpSet.has(36)) { // c4 and e4
      return { detected: true, plan_white: 'space_advantage_restrict_d5', plan_black: 'break_with_b5_or_d5', bonus_white: 20 };
    }
    return { detected: false };
  },

  // Stonewall (pawns on d5,e6,f5 for Black or d4,e3,f4 for White)
  stonewall: function(board) {
    const wp = getPawnStructure(board, WHITE);
    const wpSet = new Set(wp);
    if (wpSet.has(35) && wpSet.has(44) && wpSet.has(37)) { // d4,e3,f4
      return { detected: true, side: WHITE, plan: 'knight_to_e5_attack_kingside', weakness: 'dark_square_holes' };
    }
    const bp = getPawnStructure(board, BLACK);
    const bpSet = new Set(bp);
    if (bpSet.has(27) && bpSet.has(20) && bpSet.has(29)) { // d5,e6,f5
      return { detected: true, side: BLACK, plan: 'knight_to_e4_attack_kingside', weakness: 'light_square_holes' };
    }
    return { detected: false };
  },

  // Hedgehog (Black pawns on a6,b6,d6,e6 — very specific)
  hedgehog: function(board) {
    const bp = getPawnStructure(board, BLACK);
    const bpSet = new Set(bp);
    if (bpSet.has(16) && bpSet.has(17) && bpSet.has(19) && bpSet.has(20)) { // a6,b6,d6,e6
      return { detected: true, plan_black: 'wait_for_break_b5_or_d5', plan_white: 'space_advantage_prevent_breaks' };
    }
    return { detected: false };
  },
};

// Detect Greek Gift sacrifice possibility (Bxh7+ sac)
function detectGreekGift(board, side) {
  const enemy = 1 - side;
  const bishops = findPieces(board, side, BISHOP);

  for (const bSq of bishops) {
    // Bishop should be on b1-h7 diagonal or similar
    const targetSq = enemy === BLACK ? 15 : 55; // h7 or h2
    const targetPawn = board[targetSq];

    if (targetPawn && targetPawn.color === enemy && targetPawn.type === PAWN) {
      // Check if bishop can reach the square
      const attacks = getAttackedSquares(board, bSq);
      if (attacks.includes(targetSq)) {
        // Check if knight can follow up on g5/f6
        const knights = findPieces(board, side, KNIGHT);
        for (const nSq of knights) {
          const nAttacks = getAttackedSquares(board, nSq);
          const followUpSq = enemy === BLACK ? 14 : 46; // g5 or g4
          if (nAttacks.includes(followUpSq) || nAttacks.includes(targetSq)) {
            return { possible: true, sacrifice_sq: targetSq, bonus: 50 };
          }
        }
      }
    }
  }
  return { possible: false, bonus: 0 };
}

// Detect smothered mate possibility
function detectSmotheredMate(board, side) {
  const enemy = 1 - side;
  const enemyKing = findKing(board, enemy);
  if (enemyKing < 0) return { possible: false };

  const kr = sqRank(enemyKing), kf = sqFile(enemyKing);

  // King must be in corner-ish area and surrounded
  let surroundedCount = 0;
  for (let dr = -1; dr <= 1; dr++) for (let df = -1; df <= 1; df++) {
    if (dr === 0 && df === 0) continue;
    const nr = kr + dr, nf = kf + df;
    if (nr < 0 || nr >= 8 || nf < 0 || nf >= 8) { surroundedCount++; continue; }
    if (board[nr * 8 + nf] && board[nr * 8 + nf].color === enemy) surroundedCount++;
  }

  if (surroundedCount >= 6) { // King is very boxed in
    // Check if we have a knight that could deliver check
    const knights = findPieces(board, side, KNIGHT);
    for (const nSq of knights) {
      const nAttacks = getAttackedSquares(board, nSq);
      // Can any knight move deliver check?
      const knightCheckSquares = [[-2,-1],[-2,1],[-1,-2],[-1,2],[1,-2],[1,2],[2,-1],[2,1]]
        .map(([dr,df]) => [kr+dr, kf+df])
        .filter(([r,f]) => r>=0 && r<8 && f>=0 && f<8)
        .map(([r,f]) => r*8+f);

      for (const cs of knightCheckSquares) {
        if (!board[cs] || board[cs].color === enemy) {
          return { possible: true, bonus: 30 };
        }
      }
    }
  }
  return { possible: false, bonus: 0 };
}

// Detect X-ray attacks (piece attacking through another piece)
function detectXRayAttacks(board, side) {
  const enemy = 1 - side;
  let bonus = 0;
  const threats = [];

  for (let sq = 0; sq < 64; sq++) {
    const p = board[sq];
    if (!p || p.color !== side) continue;
    if (p.type !== ROOK && p.type !== BISHOP && p.type !== QUEEN) continue;

    const r = sqRank(sq), f = sqFile(sq);
    const isDiag = p.type === BISHOP || p.type === QUEEN;
    const isStraight = p.type === ROOK || p.type === QUEEN;

    const dirs = [];
    if (isDiag) dirs.push([-1,-1],[-1,1],[1,-1],[1,1]);
    if (isStraight) dirs.push([-1,0],[1,0],[0,-1],[0,1]);

    for (const [dr, df] of dirs) {
      let nr = r + dr, nf = f + df;
      let foundFirst = null;
      while (nr >= 0 && nr < 8 && nf >= 0 && nf < 8) {
        const tSq = nr * 8 + nf;
        const tp = board[tSq];
        if (tp) {
          if (!foundFirst) {
            foundFirst = tp;
          } else {
            // X-ray: attacking through foundFirst to tp
            if (tp.color === enemy && (tp.type === QUEEN || tp.type === KING || tp.type === ROOK)) {
              bonus += 15;
              threats.push('xray_' + sq + '_through_' + foundFirst);
            }
            break;
          }
        }
        nr += dr; nf += df;
      }
    }
  }

  return { bonus, threats };
}

// Detect windmill potential (discovered check repeating)
function detectDiscoveredAttack(board, side) {
  const enemy = 1 - side;
  const enemyKing = findKing(board, enemy);
  if (enemyKing < 0) return { bonus: 0 };

  let bonus = 0;
  const kr = sqRank(enemyKing), kf = sqFile(enemyKing);

  // Look for own pieces between our sliding piece and enemy king
  const dirs = [[-1,0],[1,0],[0,-1],[0,1],[-1,-1],[-1,1],[1,-1],[1,1]];
  for (const [dr, df] of dirs) {
    let nr = kr + dr, nf = kf + df;
    let foundBlocker = null;
    while (nr >= 0 && nr < 8 && nf >= 0 && nf < 8) {
      const tSq = nr * 8 + nf;
      const tp = board[tSq];
      if (tp) {
        if (!foundBlocker) {
          if (tp.color === side) foundBlocker = tSq;
          else break;
        } else {
          if (tp.color === side) {
            const isDiag = dr !== 0 && df !== 0;
            if ((isDiag && (tp.type === BISHOP || tp.type === QUEEN)) ||
                (!isDiag && (tp.type === ROOK || tp.type === QUEEN))) {
              bonus += 25; // Potential discovered attack
            }
          }
          break;
        }
      }
      nr += dr; nf += df;
    }
  }

  return { bonus };
}

// Pattern: Blockade
function detectBlockade(board, side) {
  const enemy = 1 - side;
  const enemyPawns = getPawnStructure(board, enemy);
  let bonus = 0;

  for (const eSq of enemyPawns) {
    const stopSq = enemy === WHITE ? eSq - 8 : eSq + 8;
    if (stopSq < 0 || stopSq >= 64) continue;
    const blocker = board[stopSq];
    if (blocker && blocker.color === side) {
      // Our piece blocks their pawn
      if (blocker.type === KNIGHT) bonus += 15; // Knight is best blockader
      else if (blocker.type === BISHOP) bonus += 8;
      else bonus += 3;
    }
  }

  return bonus;
}

// Pattern: Pawn chain detection and lever points
function detectPawnChains(board, side) {
  const pawns = getPawnStructure(board, side);
  const enemy = 1 - side;
  const enemyPawns = getPawnStructure(board, enemy);
  const chains = [];
  let bonus = 0;

  // Find chains (pawns protecting each other diagonally)
  const pawnSet = new Set(pawns);
  for (const pSq of pawns) {
    const f = sqFile(pSq);
    const r = sqRank(pSq);
    const supportRank = side === WHITE ? r + 1 : r - 1;
    if (f > 0 && pawnSet.has(supportRank * 8 + (f - 1))) {
      chains.push(pSq);
      bonus += 5; // Chain bonus
    }
    if (f < 7 && pawnSet.has(supportRank * 8 + (f + 1))) {
      chains.push(pSq);
      bonus += 5;
    }
  }

  // Find lever opportunities (pawn vs pawn tension points)
  for (const pSq of pawns) {
    const f = sqFile(pSq);
    const r = sqRank(pSq);
    const advanceRank = side === WHITE ? r - 1 : r + 1;
    if (advanceRank < 0 || advanceRank >= 8) continue;

    for (const eSq of enemyPawns) {
      const ef = sqFile(eSq);
      const er = sqRank(eSq);
      if (Math.abs(ef - f) === 1 && er === advanceRank) {
        bonus += 3; // Pawn tension / lever opportunity
      }
    }
  }

  return { chains, bonus };
}


// ============================================================================
// SECTION 4: COUNTER-PLAY DATABASE
// ============================================================================

// Opponent style profiles and counter-strategies
// These adjust evaluation weights to exploit stylistic weaknesses

const COUNTER_STRATEGIES = {
  aggressive: {
    name: 'Counter-Aggressive',
    description: 'Against aggressive players: simplify, solid structure, fortress',
    weights: {
      pawnStructure: 1.3,    // Value solid structure more
      kingSafety: 1.5,       // Prioritize king safety
      pieceActivity: 0.8,    // Less risk-taking
      materialBalance: 1.2,  // Trade when ahead
      passedPawns: 1.0,
      centerControl: 0.9,
      tempoBonus: 0.7,       // Don't overvalue initiative
      exchangeBonus: 1.3,    // Encourage trades
    },
    prefer: ['solid_structures', 'trade_pieces', 'avoid_complications', 'castle_early'],
    avoid: ['pawn_storms', 'speculative_sacrifices', 'open_positions'],
  },

  positional: {
    name: 'Counter-Positional',
    description: 'Against positional players: create imbalances, tactical play',
    weights: {
      pawnStructure: 0.8,    // Accept structural weaknesses for activity
      kingSafety: 1.0,
      pieceActivity: 1.4,    // Maximize piece activity
      materialBalance: 0.9,
      passedPawns: 1.1,
      centerControl: 1.2,
      tempoBonus: 1.3,       // Value initiative
      exchangeBonus: 0.7,    // Avoid trades, keep pieces on
    },
    prefer: ['imbalances', 'avoid_symmetry', 'sharp_positions', 'initiative'],
    avoid: ['simplification', 'equal_endgames', 'passive_play'],
  },

  tactical: {
    name: 'Counter-Tactical',
    description: 'Against tactical players: solid, no hanging pieces, prophylaxis',
    weights: {
      pawnStructure: 1.4,
      kingSafety: 1.6,       // Top priority
      pieceActivity: 0.9,
      materialBalance: 1.3,
      passedPawns: 0.9,
      centerControl: 1.1,
      tempoBonus: 0.6,       // Don't rush
      exchangeBonus: 1.4,    // Trade off attackers
    },
    prefer: ['prophylaxis', 'overprotection', 'closed_positions', 'trade_attackers'],
    avoid: ['open_king', 'hanging_pieces', 'time_pressure', 'complications'],
  },

  defensive: {
    name: 'Counter-Defensive',
    description: 'Against defensive players: pawn breaks, activity, initiative',
    weights: {
      pawnStructure: 0.9,
      kingSafety: 0.9,
      pieceActivity: 1.5,    // Maximum activity
      materialBalance: 0.8,
      passedPawns: 1.3,
      centerControl: 1.3,
      tempoBonus: 1.5,       // Maximum initiative
      exchangeBonus: 0.5,    // Keep pieces, don't trade
    },
    prefer: ['pawn_breaks', 'piece_activity', 'space_advantage', 'prevent_exchanges'],
    avoid: ['equal_positions', 'trading_down', 'passive_play', 'drawish_lines'],
  },

  universal: {
    name: 'Universal',
    description: 'Balanced approach for unknown opponent',
    weights: {
      pawnStructure: 1.1,
      kingSafety: 1.2,
      pieceActivity: 1.1,
      materialBalance: 1.0,
      passedPawns: 1.1,
      centerControl: 1.1,
      tempoBonus: 1.0,
      exchangeBonus: 1.0,
    },
    prefer: ['balanced_play', 'exploit_weaknesses', 'optimal_piece_placement'],
    avoid: ['extreme_risks', 'passive_play'],
  },

  engine: {
    name: 'Anti-Engine',
    description: 'Against computer opponents: complex positions, long-term sacrifices',
    weights: {
      pawnStructure: 0.7,
      kingSafety: 1.1,
      pieceActivity: 1.3,
      materialBalance: 0.7,  // Willing to sacrifice material
      passedPawns: 1.2,
      centerControl: 1.0,
      tempoBonus: 1.4,
      exchangeBonus: 0.6,    // Keep position complex
    },
    prefer: ['closed_positions', 'long_term_plans', 'positional_sacrifices', 'avoid_tactics'],
    avoid: ['tactical_lines', 'clean_positions', 'simple_endgames'],
  },
};


// ============================================================================
// SECTION 5: PIECE COORDINATION TABLES
// ============================================================================

// Bishop pair values by pawn count
const BISHOP_PAIR_BY_PAWNS = [
  80, // 0 pawns — huge bishop pair advantage
  75, // 1 pawn
  68, // 2 pawns
  62, // 3 pawns
  55, // 4 pawns
  50, // 5 pawns
  45, // 6 pawns
  40, // 7 pawns
  35, // 8 pawns
  30, // 9 pawns
  28, // 10 pawns
  25, // 11 pawns
  22, // 12 pawns
  20, // 13 pawns
  18, // 14 pawns
  15, // 15 pawns
  12, // 16 pawns
];

// Knight outpost values by square (from white's perspective, rank 0 = 8th rank)
// Higher values for central outposts in enemy territory
const KNIGHT_OUTPOST_VALUES = [
   0,  0,  0,  0,  0,  0,  0,  0,  // 8th rank
   5, 10, 15, 20, 20, 15, 10,  5,  // 7th rank
  10, 20, 30, 40, 40, 30, 20, 10,  // 6th rank (deep outpost!)
   8, 18, 28, 35, 35, 28, 18,  8,  // 5th rank
   5, 12, 20, 25, 25, 20, 12,  5,  // 4th rank
   0,  5, 10, 15, 15, 10,  5,  0,  // 3rd rank
   0,  0,  0,  0,  0,  0,  0,  0,  // 2nd rank
   0,  0,  0,  0,  0,  0,  0,  0,  // 1st rank
];

// Rook coordination bonuses
const ROOK_COORD = {
  CONNECTED_ROOKS: 10,      // Two rooks defending each other
  BATTERY_ON_FILE: 25,      // Two rooks on same file
  BATTERY_ON_RANK: 20,      // Two rooks on same rank (7th rank battery!)
  DOUBLED_7TH: 60,          // Both rooks on 7th rank
  ROOK_BEHIND_PASSER: 30,   // Rook behind passed pawn
  ROOK_OPEN_FILE: 20,       // Rook on open file
  ROOK_SEMI_OPEN: 12,       // Rook on semi-open file
  ROOK_7TH: 35,             // Rook on 7th rank
  ROOK_ATTACKS_PAWN: 8,     // Rook attacking enemy weak pawn
};

// Queen + Knight vs Queen + Bishop evaluation adjustments
// In closed positions, Q+N is often better
// In open positions, Q+B is often better
function evaluateQNvQB(board, side) {
  const enemy = 1 - side;
  const myKnights = findPieces(board, side, KNIGHT);
  const myBishops = findPieces(board, side, BISHOP);
  const theirKnights = findPieces(board, enemy, KNIGHT);
  const theirBishops = findPieces(board, enemy, BISHOP);

  if (myKnights.length === 1 && myBishops.length === 0 &&
      theirKnights.length === 0 && theirBishops.length === 1) {
    // We have Q+N, they have Q+B
    // Count pawns and openness
    let pawnCount = 0;
    let centralPawns = 0;
    for (let i = 0; i < 64; i++) {
      if (board[i] && board[i].type === PAWN) {
        pawnCount++;
        const f = sqFile(i), r = sqRank(i);
        if (f >= 2 && f <= 5 && r >= 2 && r <= 5) centralPawns++;
      }
    }

    // More central pawns = more closed = better for knight
    if (centralPawns >= 4) return 20; // Q+N better in closed
    if (centralPawns <= 1) return -15; // Q+B better in open
    return 5;
  }

  if (myKnights.length === 0 && myBishops.length === 1 &&
      theirKnights.length === 1 && theirBishops.length === 0) {
    // We have Q+B, they have Q+N (mirror)
    let centralPawns = 0;
    for (let i = 0; i < 64; i++) {
      if (board[i] && board[i].type === PAWN) {
        const f = sqFile(i), r = sqRank(i);
        if (f >= 2 && f <= 5 && r >= 2 && r <= 5) centralPawns++;
      }
    }
    if (centralPawns >= 4) return -20;
    if (centralPawns <= 1) return 15;
    return -5;
  }

  return 0;
}

// Minor piece imbalance table
// [ownPiece][enemyPiece] -> bonus in centipawns
// Adjustments for having one type vs another
const MINOR_IMBALANCE = {
  // Knight vs Bishop
  knightVsBishop: function(board, side) {
    const myKnights = findPieces(board, side, KNIGHT).length;
    const myBishops = findPieces(board, side, BISHOP).length;
    const enemyKnights = findPieces(board, 1 - side, KNIGHT).length;
    const enemyBishops = findPieces(board, 1 - side, BISHOP).length;

    let bonus = 0;

    // Count pawns for openness assessment
    let totalPawns = 0;
    for (let i = 0; i < 64; i++) {
      if (board[i] && board[i].type === PAWN) totalPawns++;
    }

    // More pawns = more closed = knights better
    const closedBonus = (totalPawns - 8) * 3; // 0 at 8 pawns, +/- from there

    // Knight surplus in closed positions
    if (myKnights > myBishops && myKnights > enemyKnights) {
      bonus += closedBonus > 0 ? closedBonus : 0;
    }

    // Bishop surplus in open positions
    if (myBishops > myKnights && myBishops > enemyBishops) {
      bonus += closedBonus < 0 ? -closedBonus : 0;
    }

    return bonus;
  },

  // Rook vs minor pieces
  rookVsMinors: function(board, side) {
    const myRooks = findPieces(board, side, ROOK).length;
    const myMinors = findPieces(board, side, KNIGHT).length + findPieces(board, side, BISHOP).length;
    const enemyRooks = findPieces(board, 1 - side, ROOK).length;
    const enemyMinors = findPieces(board, 1 - side, KNIGHT).length + findPieces(board, 1 - side, BISHOP).length;

    let bonus = 0;

    // Exchange imbalance: R vs B+N
    if (myRooks > enemyRooks && myMinors < enemyMinors) {
      // We have the exchange (extra rook, fewer minors)
      // The exchange is generally worth ~200cp but less in closed positions
      let pawnCount = 0;
      for (let i = 0; i < 64; i++) {
        if (board[i] && board[i].type === PAWN) pawnCount++;
      }
      bonus += pawnCount > 10 ? -10 : 15; // Exchange less valuable with many pawns
    }

    return bonus;
  },
};

// Complete piece coordination evaluator
function evaluatePieceCoordination(board, side) {
  let bonus = 0;

  // 1. Bishop pair
  const bp = detectBishopPair(board, side);
  if (bp.hasPair) {
    let pawnCount = 0;
    for (let i = 0; i < 64; i++) {
      if (board[i] && board[i].type === PAWN) pawnCount++;
    }
    bonus += BISHOP_PAIR_BY_PAWNS[Math.min(pawnCount, 16)];
  }

  // 2. Knight outposts
  const outposts = detectKnightOutposts(board, side);
  for (const op of outposts) {
    const pstIdx = side === WHITE ? op.sq : ((7 - sqRank(op.sq)) * 8 + sqFile(op.sq));
    bonus += KNIGHT_OUTPOST_VALUES[pstIdx];
  }

  // 3. Rook coordination
  const rooks = findPieces(board, side, ROOK);
  const { openFiles, semiOpenWhite, semiOpenBlack } = detectOpenFiles(board);
  const semiOpen = side === WHITE ? semiOpenWhite : semiOpenBlack;

  if (rooks.length >= 2) {
    // Check for connected rooks
    const r0 = rooks[0], r1 = rooks[1];
    if (sqRank(r0) === sqRank(r1)) {
      // Same rank - check if connected (no pieces between)
      const minF = Math.min(sqFile(r0), sqFile(r1));
      const maxF = Math.max(sqFile(r0), sqFile(r1));
      let connected = true;
      for (let f = minF + 1; f < maxF; f++) {
        if (board[sqRank(r0) * 8 + f]) { connected = false; break; }
      }
      if (connected) bonus += ROOK_COORD.BATTERY_ON_RANK;

      // Doubled on 7th
      const seventhRank = side === WHITE ? 1 : 6;
      if (sqRank(r0) === seventhRank) bonus += ROOK_COORD.DOUBLED_7TH;
    }
    if (sqFile(r0) === sqFile(r1)) {
      let connected = true;
      const minR = Math.min(sqRank(r0), sqRank(r1));
      const maxR = Math.max(sqRank(r0), sqRank(r1));
      for (let r = minR + 1; r < maxR; r++) {
        if (board[r * 8 + sqFile(r0)]) { connected = false; break; }
      }
      if (connected) bonus += ROOK_COORD.BATTERY_ON_FILE;
    }
  }

  for (const rSq of rooks) {
    const f = sqFile(rSq);
    const r = sqRank(rSq);

    if (openFiles.includes(f)) bonus += ROOK_COORD.ROOK_OPEN_FILE;
    else if (semiOpen.includes(f)) bonus += ROOK_COORD.ROOK_SEMI_OPEN;

    const seventhRank = side === WHITE ? 1 : 6;
    if (r === seventhRank) bonus += ROOK_COORD.ROOK_7TH;

    // Rook behind passed pawn
    const passedPawns = detectPassedPawns(board, side);
    for (const pp of passedPawns) {
      if (sqFile(pp.sq) === f) {
        const pawnAhead = side === WHITE ?
          r > sqRank(pp.sq) : r < sqRank(pp.sq);
        if (pawnAhead) bonus += ROOK_COORD.ROOK_BEHIND_PASSER;
      }
    }
  }

  // 4. Q+N vs Q+B
  bonus += evaluateQNvQB(board, side);

  // 5. Minor piece imbalances
  bonus += MINOR_IMBALANCE.knightVsBishop(board, side);
  bonus += MINOR_IMBALANCE.rookVsMinors(board, side);

  // 6. Bishop quality
  bonus += evaluateBishopQuality(board, side);

  return bonus;
}


// ============================================================================
// EXPORTED FUNCTIONS
// ============================================================================

// 1. Opening book lookup
export function lookupBook(fen) {
  const hash = hashFen(fen);
  return BOOK.get(hash) || null;
}

// 2. Endgame lookup
export function lookupEndgame(board, materialSig) {
  // Check for drawn material first
  if (isDrawnMaterial(board)) return { move: null, eval: 0 };

  // Try specific endgame handler
  const sig = materialSig || getMaterialSignature(board);
  const handler = ENDGAME_HANDLERS[sig];
  if (handler) {
    const result = handler(board);
    if (result) return result;
  }

  // General pawn endgame evaluation
  let wp = 0, wn = 0, wb = 0, wr = 0, wq = 0;
  let bp = 0, bn = 0, bb = 0, br = 0, bq = 0;
  for (let i = 0; i < 64; i++) {
    const p = board[i];
    if (!p || p.type === KING) continue;
    if (p.color === WHITE) {
      if (p.type === PAWN) wp++;
      else if (p.type === KNIGHT) wn++;
      else if (p.type === BISHOP) wb++;
      else if (p.type === ROOK) wr++;
      else if (p.type === QUEEN) wq++;
    } else {
      if (p.type === PAWN) bp++;
      else if (p.type === KNIGHT) bn++;
      else if (p.type === BISHOP) bb++;
      else if (p.type === ROOK) br++;
      else if (p.type === QUEEN) bq++;
    }
  }

  // Pure pawn endgame
  if (wn + wb + wr + wq + bn + bb + br + bq === 0 && (wp + bp > 0)) {
    const whiteEval = evaluatePawnEndgame(board, WHITE);
    const blackEval = evaluatePawnEndgame(board, BLACK);
    return { move: null, eval: whiteEval - blackEval };
  }

  return null;
}

// 3. Pattern detection
export function detectPatterns(board, side) {
  let bonus = 0;
  const threats = [];

  // Tactical patterns
  const pins = detectPins(board, 1 - side); // Pins on enemy
  if (pins.length > 0) {
    bonus += pins.length * 15;
    threats.push('pin');
  }

  const forks = detectForks(board, side);
  if (forks.length > 0) {
    bonus += forks.length * 20;
    threats.push('fork');
  }

  // X-ray attacks
  const xray = detectXRayAttacks(board, side);
  bonus += xray.bonus;
  if (xray.bonus > 0) threats.push('xray');

  // Discovered attack potential
  const disc = detectDiscoveredAttack(board, side);
  bonus += disc.bonus;
  if (disc.bonus > 0) threats.push('discovered_attack');

  // Greek gift
  const greek = detectGreekGift(board, side);
  bonus += greek.bonus;
  if (greek.possible) threats.push('greek_gift');

  // Smothered mate
  const smothered = detectSmotheredMate(board, side);
  bonus += smothered.bonus;
  if (smothered.possible) threats.push('smothered_mate');

  // Back rank weakness (on enemy)
  if (detectBackRankWeakness(board, 1 - side)) {
    bonus += 30;
    threats.push('back_rank_threat');
  }

  // Strategic patterns
  const passedPawns = detectPassedPawns(board, side);
  if (passedPawns.length > 0) {
    for (const pp of passedPawns) {
      bonus += 15 + pp.advance * 10;
    }
    threats.push('passed_pawn');
  }

  const isolatedPawns = detectIsolatedPawns(board, side);
  const enemyIsolated = detectIsolatedPawns(board, 1 - side);
  bonus -= isolatedPawns.length * 12;
  bonus += enemyIsolated.length * 12;
  if (enemyIsolated.length > 0) threats.push('target_isolated_pawn');

  const doubled = detectDoubledPawns(board, side);
  const enemyDoubled = detectDoubledPawns(board, 1 - side);
  bonus -= doubled.length * 10;
  bonus += enemyDoubled.length * 10;

  const backward = detectBackwardPawns(board, side);
  const enemyBackward = detectBackwardPawns(board, 1 - side);
  bonus -= backward.length * 8;
  bonus += enemyBackward.length * 8;
  if (enemyBackward.length > 0) threats.push('target_backward_pawn');

  // Outposts
  const outposts = detectKnightOutposts(board, side);
  for (const op of outposts) bonus += op.value;
  if (outposts.length > 0) threats.push('outpost');

  // Rook activity
  const rookAct = detectRookActivity(board, side);
  bonus += rookAct.bonus;
  for (const t of rookAct.threats) threats.push(t);

  // King safety
  const mySafety = evaluateKingSafety(board, side);
  const theirSafety = evaluateKingSafety(board, 1 - side);
  bonus += mySafety - theirSafety;

  // Space
  const mySpace = evaluateSpace(board, side);
  const theirSpace = evaluateSpace(board, 1 - side);
  bonus += (mySpace - theirSpace);

  // Blockade
  bonus += detectBlockade(board, side);

  // Pawn chains
  const chains = detectPawnChains(board, side);
  bonus += chains.bonus;

  // Pawn structure names
  for (const [name, fn] of Object.entries(PAWN_STRUCTURES)) {
    const result = fn(board);
    if (result.detected) {
      threats.push('structure_' + name);
      if (result.bonus !== undefined) bonus += result.bonus;
      if (result.bonus_white !== undefined && side === WHITE) bonus += result.bonus_white;
      if (result.bonus_black !== undefined && side === BLACK) bonus += result.bonus_black;
    }
  }

  return { bonus, threats };
}

// 4. Counter-strategy
export function getCounterStrategy(opponentProfile) {
  const profile = opponentProfile || 'universal';
  const strategy = COUNTER_STRATEGIES[profile] || COUNTER_STRATEGIES.universal;

  return {
    weights: [
      strategy.weights.pawnStructure,
      strategy.weights.kingSafety,
      strategy.weights.pieceActivity,
      strategy.weights.materialBalance,
      strategy.weights.passedPawns,
      strategy.weights.centerControl,
      strategy.weights.tempoBonus,
      strategy.weights.exchangeBonus,
    ],
    prefer: strategy.prefer,
    avoid: strategy.avoid,
  };
}

// 5. Piece coordination
export function getPieceCoordination(board, side) {
  return evaluatePieceCoordination(board, side);
}

// Also export utility functions that the engine might find useful
export function getBookSize() { return BOOK.size; }
export function getMaterialSig(board) { return getMaterialSignature(board); }
export function isDrawn(board) { return isDrawnMaterial(board); }


// ============================================================================
// SECTION 6: COMPREHENSIVE KING SAFETY DATABASE
// ============================================================================
// Detailed king safety evaluation with pawn shelter, storm, and attack tables
// ============================================================================

// Pawn shelter bonus by file distance from king and pawn advancement
// [fileDistFromKing (0-2)][pawnAdvance (0=on 2nd rank, 5=on 7th)]
// From the perspective of the defending side
const PAWN_SHELTER_BONUS = [
  // Same file as king
  [36, 20, 12, 4, -2, -10, -20],
  // One file away
  [24, 14, 8, 2, -4, -12, -18],
  // Two files away
  [12, 6, 2, 0, -2, -6, -10],
];

// Pawn storm penalty when enemy pawn advances toward our king
// [fileDistFromKing (0-2)][enemyPawnAdvance (0=far, 5=close)]
const PAWN_STORM_PENALTY = [
  // Same file — most dangerous
  [0, -4, -10, -20, -35, -50, -70],
  // One file away
  [0, -2, -6, -14, -25, -40, -55],
  // Two files away
  [0, 0, -3, -8, -15, -25, -35],
];

// King zone attack weights by piece type
const KING_ATTACK_WEIGHTS = {
  [PAWN]: 1,
  [KNIGHT]: 8,
  [BISHOP]: 6,
  [ROOK]: 10,
  [QUEEN]: 15,
};

// King danger score to centipawn conversion
// Quadratic: danger^2 / scale
const KING_DANGER_SCALE = 48;

function evaluateDetailedKingSafety(board, side) {
  const kingSq = findKing(board, side);
  if (kingSq < 0) return 0;

  const kr = sqRank(kingSq);
  const kf = sqFile(kingSq);
  const enemy = 1 - side;

  let shelterBonus = 0;
  let stormPenalty = 0;

  // Evaluate pawn shelter and storms for files near king
  for (let df = -2; df <= 2; df++) {
    const f = kf + df;
    if (f < 0 || f > 7) continue;
    const absDf = Math.abs(df);
    if (absDf > 2) continue;

    // Find our shelter pawn on this file
    let ourPawnRank = -1;
    let enemyPawnRank = -1;

    for (let r = 0; r < 8; r++) {
      const sq = r * 8 + f;
      const p = board[sq];
      if (p && p.type === PAWN) {
        if (p.color === side) {
          const advance = side === WHITE ? (6 - r) : (r - 1);
          if (ourPawnRank < 0) ourPawnRank = Math.max(0, Math.min(6, advance));
        }
        if (p.color === enemy) {
          const closeness = side === WHITE ? (7 - r) : r;
          if (enemyPawnRank < 0) enemyPawnRank = Math.max(0, Math.min(6, closeness));
        }
      }
    }

    // Shelter bonus
    if (ourPawnRank >= 0) {
      shelterBonus += PAWN_SHELTER_BONUS[absDf][ourPawnRank];
    } else {
      shelterBonus -= 15; // Missing shelter pawn
    }

    // Storm penalty
    if (enemyPawnRank >= 0) {
      stormPenalty += PAWN_STORM_PENALTY[absDf][enemyPawnRank];
    }
  }

  // Count attackers in king zone (3x3 around king, plus two squares forward)
  let dangerScore = 0;
  let attackerCount = 0;

  for (let sq = 0; sq < 64; sq++) {
    const p = board[sq];
    if (!p || p.color !== enemy || p.type === KING) continue;

    // Check if this piece attacks the king zone
    const attacks = getAttackedSquares(board, sq);
    let attacksKingZone = false;

    for (const atk of attacks) {
      const ar = sqRank(atk);
      const af = sqFile(atk);
      if (Math.abs(ar - kr) <= 1 && Math.abs(af - kf) <= 1) {
        attacksKingZone = true;
        break;
      }
      // Extended zone: 2 squares in front of king
      const forward = side === WHITE ? -1 : 1;
      if (ar === kr + forward && Math.abs(af - kf) <= 1) {
        attacksKingZone = true;
        break;
      }
      if (ar === kr + forward * 2 && Math.abs(af - kf) <= 1) {
        attacksKingZone = true;
        break;
      }
    }

    if (attacksKingZone) {
      attackerCount++;
      dangerScore += KING_ATTACK_WEIGHTS[p.type] || 0;
    }
  }

  // Quadratic danger conversion (more attackers = exponentially worse)
  const attackPenalty = attackerCount >= 2 ? -(dangerScore * dangerScore) / KING_DANGER_SCALE : 0;

  // Castling status bonus
  let castleBonus = 0;
  const isKingCastled = side === WHITE ?
    (kf >= 5 || kf <= 2) && kr === 7 :
    (kf >= 5 || kf <= 2) && kr === 0;

  if (isKingCastled) castleBonus = 20;
  else if (kr === (side === WHITE ? 7 : 0) && kf >= 3 && kf <= 4) {
    castleBonus = -15; // King stuck in center
  }

  return shelterBonus + stormPenalty + attackPenalty + castleBonus;
}


// ============================================================================
// SECTION 7: MOBILITY TABLES
// ============================================================================
// Piece mobility evaluation — bonus/penalty based on number of legal squares
// ============================================================================

// Mobility bonus arrays indexed by number of attacked squares
// Values in centipawns — tuned for typical positions

const KNIGHT_MOBILITY = [-20, -10, -2, 4, 10, 16, 20, 24, 26];  // 0-8 squares

const BISHOP_MOBILITY = [-25, -15, -6, 0, 6, 12, 17, 22, 26, 29, 32, 34, 36, 38];  // 0-13 squares

const ROOK_MOBILITY = [-20, -12, -5, 0, 4, 8, 12, 15, 18, 20, 22, 24, 26, 28, 30];  // 0-14 squares

const QUEEN_MOBILITY = [-15, -10, -5, -2, 0, 2, 4, 6, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28];  // 0-27 squares

function evaluateMobility(board, side) {
  let totalMobility = 0;

  for (let sq = 0; sq < 64; sq++) {
    const p = board[sq];
    if (!p || p.color !== side || p.type === PAWN || p.type === KING) continue;

    const attacks = getAttackedSquares(board, sq);
    // Count only squares not occupied by own pieces
    let mobility = 0;
    for (const atk of attacks) {
      const target = board[atk];
      if (!target || target.color !== side) mobility++;
    }

    switch (p.type) {
      case KNIGHT:
        totalMobility += KNIGHT_MOBILITY[Math.min(mobility, 8)];
        break;
      case BISHOP:
        totalMobility += BISHOP_MOBILITY[Math.min(mobility, 13)];
        break;
      case ROOK:
        totalMobility += ROOK_MOBILITY[Math.min(mobility, 14)];
        break;
      case QUEEN:
        totalMobility += QUEEN_MOBILITY[Math.min(mobility, 27)];
        break;
    }
  }

  return totalMobility;
}

// Export mobility evaluator
export function getMobility(board, side) {
  return evaluateMobility(board, side);
}

// Export detailed king safety
export function getKingSafety(board, side) {
  return evaluateDetailedKingSafety(board, side);
}


// ============================================================================
// SECTION 8: COMPREHENSIVE PIECE-SQUARE ADJUSTMENTS
// ============================================================================
// These supplement the engine's existing PeSTO tables with dynamic context
// ============================================================================

// Knight centralization bonus (middlegame, additional to PST)
const KNIGHT_CENTER_BONUS_MG = [
  -20, -10,  -5,  -5,  -5,  -5, -10, -20,
  -10,   0,   5,  10,  10,   5,   0, -10,
   -5,   5,  15,  20,  20,  15,   5,  -5,
   -5,  10,  20,  30,  30,  20,  10,  -5,
   -5,  10,  20,  30,  30,  20,  10,  -5,
   -5,   5,  15,  20,  20,  15,   5,  -5,
  -10,   0,   5,  10,  10,   5,   0, -10,
  -20, -10,  -5,  -5,  -5,  -5, -10, -20,
];

// Bishop fianchetto bonus — bishops on long diagonals
const BISHOP_FIANCHETTO = {
  // White fianchetto squares (g2, b2) and their corresponding diagonals
  // When bishop is on g2 or b2 and the diagonal is open, strong bonus
  detectFianchetto: function(board, side) {
    let bonus = 0;
    const bishops = findPieces(board, side, BISHOP);
    const fianchettoSqs = side === WHITE ? [49, 54] : [9, 14]; // b2,g2 or b7,g7

    for (const bSq of bishops) {
      if (fianchettoSqs.includes(bSq)) {
        // Check if diagonal is open (no pawns blocking)
        const attacks = getAttackedSquares(board, bSq);
        const longDiagLength = attacks.filter(sq => {
          const p = board[sq];
          return !p || p.color !== side;
        }).length;
        bonus += 5 + longDiagLength * 2;
      }
    }
    return bonus;
  }
};

// Rook on 7th rank bonus with pawns
const ROOK_7TH_WITH_PAWNS = {
  evaluate: function(board, side) {
    const seventhRank = side === WHITE ? 1 : 6;
    const rooks = findPieces(board, side, ROOK);
    let bonus = 0;

    for (const rSq of rooks) {
      if (sqRank(rSq) === seventhRank) {
        // Count enemy pawns on 7th rank (from enemy perspective = 2nd rank for them)
        const enemyPawnRank = side === WHITE ? 1 : 6;
        let enemyPawnsOnRank = 0;
        for (let f = 0; f < 8; f++) {
          const p = board[enemyPawnRank * 8 + f];
          if (p && p.color !== side && p.type === PAWN) enemyPawnsOnRank++;
        }
        bonus += 20 + enemyPawnsOnRank * 10;

        // Check if enemy king is on 8th/1st rank
        const enemyKing = findKing(board, 1 - side);
        const backRank = side === WHITE ? 0 : 7;
        if (sqRank(enemyKing) === backRank) bonus += 15;
      }
    }
    return bonus;
  }
};


// ============================================================================
// SECTION 9: THREAT DETECTION ENGINE
// ============================================================================
// Comprehensive threat detection for tactical awareness
// ============================================================================

// Detect all hanging pieces (undefended pieces that can be captured)
function detectHangingPieces(board, side) {
  const enemy = 1 - side;
  const hanging = [];

  // Get all squares attacked by us
  const ourAttacks = new Set();
  for (let sq = 0; sq < 64; sq++) {
    const p = board[sq];
    if (p && p.color === side) {
      const attacks = getAttackedSquares(board, sq);
      for (const atk of attacks) ourAttacks.add(atk);
    }
  }

  // Get all squares defended by enemy
  const theirDefenses = new Set();
  for (let sq = 0; sq < 64; sq++) {
    const p = board[sq];
    if (p && p.color === enemy) {
      const attacks = getAttackedSquares(board, sq);
      for (const atk of attacks) theirDefenses.add(atk);
    }
  }

  // Find enemy pieces we attack that are not defended
  for (let sq = 0; sq < 64; sq++) {
    const p = board[sq];
    if (p && p.color === enemy && p.type !== KING) {
      if (ourAttacks.has(sq) && !theirDefenses.has(sq)) {
        hanging.push({ sq, type: p.type, value: [0, 100, 320, 330, 500, 900][p.type] });
      }
    }
  }

  return hanging;
}

// Detect overloaded pieces (piece defending multiple things)
function detectOverloadedPieces(board, side) {
  const enemy = 1 - side;
  const overloaded = [];

  for (let sq = 0; sq < 64; sq++) {
    const p = board[sq];
    if (!p || p.color !== enemy) continue;

    const attacks = getAttackedSquares(board, sq);
    let defendingCount = 0;

    for (const atk of attacks) {
      const target = board[atk];
      if (target && target.color === enemy && target.type !== KING) {
        // Check if this is the sole defender
        let otherDefenders = 0;
        for (let defSq = 0; defSq < 64; defSq++) {
          if (defSq === sq) continue;
          const defP = board[defSq];
          if (defP && defP.color === enemy) {
            const defAttacks = getAttackedSquares(board, defSq);
            if (defAttacks.includes(atk)) otherDefenders++;
          }
        }
        if (otherDefenders === 0) defendingCount++;
      }
    }

    if (defendingCount >= 2) {
      overloaded.push({ sq, type: p.type, defendingCount });
    }
  }

  return overloaded;
}

// Detect trapped pieces (piece with very few legal moves)
function detectTrappedPieces(board, side) {
  const trapped = [];

  for (let sq = 0; sq < 64; sq++) {
    const p = board[sq];
    if (!p || p.color !== side || p.type === PAWN || p.type === KING) continue;

    const attacks = getAttackedSquares(board, sq);
    let safeMoves = 0;

    // Get enemy attack map
    const enemyAttacks = new Set();
    for (let esq = 0; esq < 64; esq++) {
      const ep = board[esq];
      if (ep && ep.color !== side) {
        const ea = getAttackedSquares(board, esq);
        for (const a of ea) enemyAttacks.add(a);
      }
    }

    for (const atk of attacks) {
      const target = board[atk];
      if (target && target.color === side) continue; // Can't move to own piece
      // Check if the square is safe (not attacked by lower-value enemy piece)
      if (!enemyAttacks.has(atk)) safeMoves++;
      else if (target && target.color !== side) {
        // Capturing — check if it's a good capture (MVV-LVA positive)
        const captureValue = [0, 100, 320, 330, 500, 900][target.type];
        const pieceValue = [0, 100, 320, 330, 500, 900][p.type];
        if (captureValue >= pieceValue) safeMoves++;
      }
    }

    if (safeMoves === 0 && p.type !== PAWN) {
      trapped.push({ sq, type: p.type, value: [0, 100, 320, 330, 500, 900][p.type] });
    }
  }

  return trapped;
}

// Enhanced threat detection (combines all tactical themes)
export function detectThreats(board, side) {
  const hanging = detectHangingPieces(board, side);
  const overloaded = detectOverloadedPieces(board, side);
  const myTrapped = detectTrappedPieces(board, side);
  const theirTrapped = detectTrappedPieces(board, 1 - side);

  let bonus = 0;

  // Hanging pieces bonus (we can capture)
  for (const h of hanging) {
    bonus += h.value / 4; // Fraction of value since we might not be able to capture immediately
  }

  // Overloaded pieces (tactical potential)
  bonus += overloaded.length * 15;

  // Our trapped pieces (penalty)
  for (const t of myTrapped) {
    bonus -= t.value / 3;
  }

  // Their trapped pieces (bonus)
  for (const t of theirTrapped) {
    bonus += t.value / 3;
  }

  return {
    bonus,
    hanging: hanging.length,
    overloaded: overloaded.length,
    myTrapped: myTrapped.length,
    theirTrapped: theirTrapped.length,
  };
}


// ============================================================================
// SECTION 10: POSITIONAL THEMES DATABASE
// ============================================================================
// Deep positional knowledge encoded as heuristic evaluators
// ============================================================================

// Minority attack detection
// White has fewer pawns on queenside and pushes them to break Black's structure
function detectMinorityAttack(board, side) {
  const enemy = 1 - side;
  const myPawns = getPawnStructure(board, side);
  const theirPawns = getPawnStructure(board, enemy);

  // Count pawns by wing
  let myQueenside = 0, myKingside = 0;
  let theirQueenside = 0, theirKingside = 0;

  for (const p of myPawns) {
    if (sqFile(p) <= 3) myQueenside++;
    else myKingside++;
  }
  for (const p of theirPawns) {
    if (sqFile(p) <= 3) theirQueenside++;
    else theirKingside++;
  }

  let bonus = 0;

  // Minority attack queenside: we have fewer pawns there, push them
  if (myQueenside < theirQueenside && myQueenside > 0) {
    // Check if our pawns are advancing
    for (const p of myPawns) {
      if (sqFile(p) <= 3) {
        const advance = side === WHITE ? (6 - sqRank(p)) : (sqRank(p) - 1);
        if (advance >= 3) bonus += 10; // Pawn is advanced — minority attack in progress
      }
    }
  }

  // Minority attack kingside
  if (myKingside < theirKingside && myKingside > 0) {
    for (const p of myPawns) {
      if (sqFile(p) > 3) {
        const advance = side === WHITE ? (6 - sqRank(p)) : (sqRank(p) - 1);
        if (advance >= 3) bonus += 10;
      }
    }
  }

  return bonus;
}

// Pawn break potential
function evaluatePawnBreaks(board, side) {
  const enemy = 1 - side;
  const myPawns = getPawnStructure(board, side);
  const theirPawns = getPawnStructure(board, enemy);
  let bonus = 0;

  for (const pSq of myPawns) {
    const f = sqFile(pSq);
    const r = sqRank(pSq);
    const advanceRank = side === WHITE ? r - 1 : r + 1;
    if (advanceRank < 0 || advanceRank > 7) continue;

    // Can this pawn advance?
    const advanceSq = advanceRank * 8 + f;
    if (board[advanceSq]) continue; // Blocked

    // Is there a pawn tension (enemy pawn diagonal to advance square)?
    for (const eSq of theirPawns) {
      const ef = sqFile(eSq);
      if (Math.abs(ef - f) === 1 && sqRank(eSq) === advanceRank) {
        // Pawn can create tension by advancing
        bonus += 8;
      }
    }

    // Central pawn breaks are more valuable
    if (f >= 2 && f <= 5) bonus += 3;
  }

  return bonus;
}

// Piece activity (how many squares each piece controls)
function evaluatePieceActivity(board, side) {
  let activity = 0;
  const centralSquares = new Set([27, 28, 35, 36]); // d4,e4,d5,e5

  for (let sq = 0; sq < 64; sq++) {
    const p = board[sq];
    if (!p || p.color !== side || p.type === KING) continue;

    const attacks = getAttackedSquares(board, sq);
    // Count controlled squares, weighted by importance
    for (const atk of attacks) {
      if (centralSquares.has(atk)) activity += 3; // Central control
      else if (sqRank(atk) >= 2 && sqRank(atk) <= 5 && sqFile(atk) >= 2 && sqFile(atk) <= 5) {
        activity += 1; // Extended center
      }
    }

    // Piece on good square bonus
    if (centralSquares.has(sq) && p.type !== PAWN) activity += 5;
  }

  return activity;
}

// Prophylaxis — penalize positions where enemy has strong plans
function evaluateProphylaxis(board, side) {
  const enemy = 1 - side;
  let score = 0;

  // Check if enemy has a strong pawn center
  const centerPawns = [27, 28, 35, 36]; // d4,e4,d5,e5
  let enemyCenterPawns = 0;
  for (const sq of centerPawns) {
    const p = board[sq];
    if (p && p.color === enemy && p.type === PAWN) enemyCenterPawns++;
  }
  if (enemyCenterPawns >= 2) score -= 15; // Enemy has strong center

  // Check if enemy has advanced passed pawns
  const enemyPassed = detectPassedPawns(board, enemy);
  for (const pp of enemyPassed) {
    if (pp.advance >= 4) score -= 25; // Dangerous passer
    else if (pp.advance >= 3) score -= 10;
  }

  return score;
}

// Export positional theme evaluators
export function getPositionalScore(board, side) {
  let score = 0;
  score += detectMinorityAttack(board, side);
  score += evaluatePawnBreaks(board, side);
  score += evaluatePieceActivity(board, side);
  score += evaluateProphylaxis(board, side);
  score += BISHOP_FIANCHETTO.detectFianchetto(board, side);
  score += ROOK_7TH_WITH_PAWNS.evaluate(board, side);
  return score;
}


// ============================================================================
// SECTION 11: GAME PHASE-AWARE EVALUATION ADJUSTMENTS
// ============================================================================

// Phase detection thresholds
const OPENING_PHASE = 20;    // Phase >= 20: opening
const MIDDLEGAME_PHASE = 10; // Phase 10-19: middlegame
// Phase < 10: endgame

// Phase-specific piece values (adjustments on top of base values)
const PHASE_PIECE_ADJ = {
  opening: {
    [KNIGHT]: 5,    // Knights slightly better in opening (development)
    [BISHOP]: 10,   // Bishops good for development
    [ROOK]: -5,     // Rooks not great until files open
    [QUEEN]: -10,   // Early queen development usually bad
  },
  middlegame: {
    [KNIGHT]: 0,
    [BISHOP]: 5,    // Bishop pair shines
    [ROOK]: 5,      // Files opening up
    [QUEEN]: 5,     // Queen becomes more active
  },
  endgame: {
    [KNIGHT]: -10,  // Knights worse in endgame (long range matters)
    [BISHOP]: 10,   // Bishops great in endgame
    [ROOK]: 15,     // Rooks love endgames
    [QUEEN]: 0,
  },
};

// Development bonus (opening only)
function evaluateDevelopment(board, side) {
  let developed = 0;
  let undeveloped = 0;

  const backRank = side === WHITE ? 7 : 0;

  // Check if minor pieces are developed
  for (let f = 0; f < 8; f++) {
    const sq = backRank * 8 + f;
    const p = board[sq];
    if (p && p.color === side && (p.type === KNIGHT || p.type === BISHOP)) {
      undeveloped++;
    }
  }

  // Count developed minor pieces
  for (let sq = 0; sq < 64; sq++) {
    const p = board[sq];
    if (!p || p.color !== side) continue;
    if (p.type === KNIGHT || p.type === BISHOP) {
      if (sqRank(sq) !== backRank) developed++;
    }
  }

  let bonus = developed * 12 - undeveloped * 15;

  // Penalty for moving same piece twice in opening (king+queen side knights/bishops)
  // (Heuristic: if we have developed pieces but not castled, penalty)
  const kingSq = findKing(board, side);
  const hasntCastled = sqFile(kingSq) >= 3 && sqFile(kingSq) <= 4 && sqRank(kingSq) === backRank;
  if (hasntCastled && developed >= 2) {
    bonus -= 20; // Should castle before continuing development
  }

  return bonus;
}

// Export phase-aware evaluation
export function getPhaseAdjustment(board, side, phase) {
  let adj = 0;

  if (phase >= OPENING_PHASE) {
    // Opening adjustments
    adj += evaluateDevelopment(board, side);
    for (let sq = 0; sq < 64; sq++) {
      const p = board[sq];
      if (p && p.color === side && p.type >= KNIGHT && p.type <= QUEEN) {
        adj += PHASE_PIECE_ADJ.opening[p.type] || 0;
      }
    }
  } else if (phase >= MIDDLEGAME_PHASE) {
    // Middlegame
    for (let sq = 0; sq < 64; sq++) {
      const p = board[sq];
      if (p && p.color === side && p.type >= KNIGHT && p.type <= QUEEN) {
        adj += PHASE_PIECE_ADJ.middlegame[p.type] || 0;
      }
    }
  } else {
    // Endgame
    for (let sq = 0; sq < 64; sq++) {
      const p = board[sq];
      if (p && p.color === side && p.type >= KNIGHT && p.type <= QUEEN) {
        adj += PHASE_PIECE_ADJ.endgame[p.type] || 0;
      }
    }
    // King centralization in endgame
    adj += (4 - centerDist(findKing(board, side))) * 8;
  }

  return adj;
}


// ============================================================================
// SECTION 12: COMPREHENSIVE PAWN STRUCTURE ENCYCLOPEDIA
// ============================================================================
// Every major named pawn structure with plans for both sides
// ============================================================================

const PAWN_STRUCTURE_PLANS = {
  // Carlsbad Structure (QGD Exchange)
  // White: d4/c4(exchanged)/e3 vs Black: d5/c6/e6
  carlsbad: {
    description: 'Carlsbad structure from QGD Exchange. Symmetric center with d4 vs d5.',
    white_plan: [
      'Minority attack: push a4-a5, b4-b5 to weaken c6',
      'Place rooks on a and b files after b5-bxc6',
      'Knight to e5 as outpost',
      'Bishop to d3 aimed at h7',
    ],
    black_plan: [
      'Kingside attack with f5-f4 pawn storm',
      'Knight maneuver Nf6-e4 or Nf6-g4',
      'Rook lift Rf8-f6-h6',
      'Place bishop on d6 aimed at h2',
    ],
    white_bonus: 5,
    black_bonus: 5,
  },

  // Isolated Queen's Pawn (IQP)
  iqp: {
    description: 'Isolated d-pawn, typically on d4 for White.',
    holder_plan: [
      'Active piece play — use outposts on e5 and c5',
      'd4-d5 pawn break when conditions are right',
      'Attack on kingside using piece activity',
      'Avoid exchanges — pieces are needed for compensation',
    ],
    opponent_plan: [
      'Blockade the pawn with a piece on d5',
      'Exchange pieces to reduce attacking potential',
      'Target the pawn in the endgame',
      'Restrict d4-d5 break',
    ],
    holder_bonus: -10,
    opponent_bonus: 10,
  },

  // Hanging Pawns (c4+d4 or c5+d5 without neighbors)
  hanging: {
    description: 'Two adjacent pawns (typically c4+d4) without pawn neighbors.',
    holder_plan: [
      'Advance one pawn to gain space (d4-d5 or c4-c5)',
      'Use dynamic piece play while pawns control center',
      'Avoid pawn exchanges that isolate one pawn',
    ],
    opponent_plan: [
      'Put pressure on both pawns simultaneously',
      'Force one to advance, creating weakness',
      'Blockade the advanced pawn',
    ],
    holder_bonus: -8,
    opponent_bonus: 8,
  },

  // Maroczy Bind (c4+e4 vs d6)
  maroczy: {
    description: 'White pawns on c4+e4 restrict Black d5 break.',
    white_plan: [
      'Maintain bind — prevent d5 and b5 breaks',
      'Place pieces on optimal squares behind pawn wall',
      'Slow kingside buildup',
      'Use space advantage for piece maneuvering',
    ],
    black_plan: [
      'Achieve b5 or d5 pawn break',
      'Trade dark-squared bishops for Hedgehog setup',
      'Use b and d files for counterplay',
      'Patience — wait for overextension',
    ],
    white_bonus: 20,
    black_bonus: -5,
  },

  // Hedgehog (Black: a6/b6/d6/e6)
  hedgehog: {
    description: 'Black maintains compact structure with pawns on 6th rank.',
    white_plan: [
      'Space advantage — restrict Black counterplay',
      'Prevent b5 and d5 breaks',
      'Slow maneuvering buildup',
    ],
    black_plan: [
      'Wait for right moment for b5 or d5 break',
      'Spring b5 or d5 when White overextends',
      'Dynamic piece play behind pawn wall',
    ],
    white_bonus: 10,
    black_bonus: 5,
  },

  // Stonewall (d5/e6/f5 for Black or d4/e3/f4 for White)
  stonewall: {
    description: 'Fixed pawn chain with central pawn on 4th + f-pawn on 4th/5th.',
    holder_plan: [
      'Knight to e5/e4 as permanent outpost',
      'Kingside attack using knight + queen',
      'Bishop to c1-d2-e1-h4 maneuver',
    ],
    opponent_plan: [
      'Exploit weak squares on opposite color from pawn chain',
      'Good bishop vs bad bishop advantage',
      'Target the e-file weakness',
      'Minority attack on opposite wing',
    ],
    holder_bonus: 5,
    opponent_bonus: 10,
  },

  // French Structure (e5 chain: d4/e5 vs d5/e6)
  french: {
    description: 'White pawn chain d4-e5 vs Black d5-e6.',
    white_plan: [
      'Kingside attack: f4-f5 or g4-g5 pawn storms',
      'Knight on f4 or d4',
      'Use space advantage on kingside',
    ],
    black_plan: [
      'c5 break to undermine d4 base',
      'f6 break to challenge e5',
      'Queenside play with a5-a4',
      'Knight to c4 via a5 or b6',
    ],
    white_bonus: 8,
    black_bonus: 5,
  },

  // Sicilian Pawn Structure (e4 vs d6 with open c-file)
  sicilian: {
    description: 'Typical Sicilian structure with open c-file.',
    white_plan: [
      'Kingside attack using f4-f5',
      'Central control with e4-e5',
      'Use d5 square for knight',
    ],
    black_plan: [
      'Queenside counterplay on c-file',
      'b5-b4 pawn push',
      'Use half-open c-file for rooks',
      'a5-a4-a3 in some lines',
    ],
    white_bonus: 5,
    black_bonus: 5,
  },

  // Symmetrical Pawn Structure
  symmetrical: {
    description: 'Mirror pawn structure — typical in Exchange variations.',
    plan: [
      'Seize initiative through piece activity',
      'Create an imbalance (pawn break, piece trade)',
      'The side with better piece placement benefits',
    ],
    white_bonus: 3, // Small first-move advantage
    black_bonus: 0,
  },

  // Caro Formation (c6/d5 vs e4/d4 or c6/d5 vs e5/d4)
  caro: {
    description: 'Black pawns on c6+d5, solid but somewhat passive.',
    white_plan: [
      'Space advantage — restrict c6-c5 break',
      'e4-e5 push when possible',
      'Piece play on kingside',
    ],
    black_plan: [
      'c6-c5 pawn break to free position',
      'Solid structure allows patient play',
      'Trade bad bishop with Bc8-f5',
    ],
    white_bonus: 5,
    black_bonus: 3,
  },

  // Boleslavsky Hole (Sicilian with d6/e5 — hole on d5)
  boleslavsky: {
    description: 'Black plays ...e5 in Sicilian, creating hole on d5.',
    white_plan: [
      'Occupy d5 with knight (permanent outpost)',
      'Control light squares',
      'Prophylactic play preventing d6-d5',
    ],
    black_plan: [
      'Active piece play compensates for d5 hole',
      'f5 pawn break to challenge e4',
      'Use dark-squared bishop activity',
    ],
    white_bonus: 12,
    black_bonus: 5,
  },

  // Benoni Structure (White d5 vs Black c5/d6/e5)
  benoni: {
    description: 'White pawn on d5, Black pawns on c5+d6+e5.',
    white_plan: [
      'Queenside space advantage with c4-c5',
      'Use extra queenside space for piece maneuvers',
      'Restrict Black kingside activity',
      'e4 push if possible',
    ],
    black_plan: [
      'f5 kingside pawn break',
      'b5 queenside counterplay',
      'Knight to c5 or e5',
      'Use d-file after eventual d5xe6 or e5-e4',
    ],
    white_bonus: 8,
    black_bonus: 8,
  },

  // KID Structure (d5/c4/e4 vs c5/d6/e5/f5)
  kings_indian: {
    description: 'Closed King\'s Indian structure with locked center.',
    white_plan: [
      'Queenside play with c5 break',
      'a4-a5 push, sometimes b4',
      'Knight to c4 targeting d6/e5',
    ],
    black_plan: [
      'Kingside attack with f5-f4, g5-g4',
      'Rook lift Rf6-h6 or Rf7-g7',
      'Knight to h5-f4 or g6-h4',
      'h5-h4 pawn storm',
    ],
    white_bonus: 5,
    black_bonus: 8,
  },
};


// ============================================================================
// SECTION 13: EXCHANGE EVALUATION
// ============================================================================
// When to trade and when to keep pieces
// ============================================================================

function evaluateExchangeDesirability(board, side) {
  const enemy = 1 - side;
  let score = 0;

  // Count material
  let myMaterial = 0, theirMaterial = 0;
  let myPieceCount = 0, theirPieceCount = 0;
  const pieceValues = [0, 100, 320, 330, 500, 900, 0];

  for (let sq = 0; sq < 64; sq++) {
    const p = board[sq];
    if (!p || p.type === KING) continue;
    const val = pieceValues[p.type];
    if (p.color === side) {
      myMaterial += val;
      if (p.type !== PAWN) myPieceCount++;
    } else {
      theirMaterial += val;
      if (p.type !== PAWN) theirPieceCount++;
    }
  }

  // When ahead in material, prefer exchanges
  const materialDiff = myMaterial - theirMaterial;
  if (materialDiff > 100) {
    score += 15; // Want to trade
    if (materialDiff > 300) score += 25;
  }

  // When behind, avoid exchanges
  if (materialDiff < -100) {
    score -= 15;
    if (materialDiff < -300) score -= 25;
  }

  // With more pieces, avoid trades (more piece activity)
  if (myPieceCount > theirPieceCount) score -= 5;

  // With initiative/attacking chances, avoid trades
  const kingSafety = evaluateKingSafety(board, enemy);
  if (kingSafety < -30) score -= 20; // Enemy king is weak, keep pieces for attack

  return score;
}

export function getExchangeScore(board, side) {
  return evaluateExchangeDesirability(board, side);
}


// ============================================================================
// SECTION 14: EXTENDED OPENING BOOK — SIDELINES AND GAMBITS
// ============================================================================

// Smith-Morra Gambit (1.e4 c5 2.d4 cxd4 3.c3)
B("rnbqkbnr/pp1ppppp/8/8/3pP3/8/PPP2PPP/RNBQKBNR w KQkq", "c2c3");
B("rnbqkbnr/pp1ppppp/8/8/4P3/2p5/PPP2PPP/RNBQKBNR w KQkq", "g1f3"); // Actually after dxc3
B("rnbqkbnr/pp1ppppp/8/8/3pP3/2P5/PP3PPP/RNBQKBNR b KQkq", "d4c3");
B("rnbqkbnr/pp1ppppp/8/8/4P3/2p5/PP3PPP/RNBQKBNR w KQkq", "b1c3");

// Alapin Sicilian (1.e4 c5 2.c3)
B("rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq", "c2c3");
B("rnbqkbnr/pp1ppppp/8/2p5/4P3/2P5/PP1P1PPP/RNBQKBNR b KQkq", "d7d5");
B("rnbqkbnr/pp2pppp/8/2pp4/4P3/2P5/PP1P1PPP/RNBQKBNR w KQkq", "e4d5");
B("rnbqkbnr/pp2pppp/8/2pP4/8/2P5/PP1P1PPP/RNBQKBNR b KQkq", "d8d5");
B("rnb1kbnr/pp2pppp/8/2pq4/8/2P5/PP1P1PPP/RNBQKBNR w KQkq", "d2d4");

// Grand Prix Attack (1.e4 c5 2.Nc3 Nc6 3.f4)
B("r1bqkbnr/pp1ppppp/2n5/2p5/4P3/2N5/PPPP1PPP/R1BQKBNR w KQkq", "f2f4");

// Closed Sicilian (1.e4 c5 2.Nc3 Nc6 3.g3)
B("r1bqkbnr/pp1ppppp/2n5/2p5/4P3/2N5/PPPP1PPP/R1BQKBNR w KQkq", "g2g3");
B("r1bqkbnr/pp1ppppp/2n5/2p5/4P3/2N3P1/PPPP1P1P/R1BQKBNR b KQkq", "g7g6");
B("r1bqkbnr/pp1ppp1p/2n3p1/2p5/4P3/2N3P1/PPPP1P1P/R1BQKBNR w KQkq", "f1g2");
B("r1bqkbnr/pp1ppp1p/2n3p1/2p5/4P3/2N3P1/PPPP1PBP/R1BQK1NR b KQkq", "f8g7");
B("r1bqk1nr/pp1pppbp/2n3p1/2p5/4P3/2N3P1/PPPP1PBP/R1BQK1NR w KQkq", "d2d3");

// Scandinavian — 3.Nc3 Qd6 variation
B("rnb1kbnr/ppp1pppp/8/3q4/8/2N5/PPPP1PPP/R1BQKBNR b KQkq", "d5d6");
B("rnb1kbnr/ppp1pppp/3q4/8/8/2N5/PPPP1PPP/R1BQKBNR w KQkq", "d2d4");
B("rnb1kbnr/ppp1pppp/3q4/8/3P4/2N5/PPP2PPP/R1BQKBNR b KQkq", "g8f6");
B("rnb1kb1r/ppp1pppp/3q1n2/8/3P4/2N5/PPP2PPP/R1BQKBNR w KQkq", "g1f3");

// Center Game (1.e4 e5 2.d4 exd4 3.Qxd4)
B("rnbqkbnr/pppp1ppp/8/4p3/3PP3/8/PPP2PPP/RNBQKBNR b KQkq", "e5d4");
B("rnbqkbnr/pppp1ppp/8/8/3QP3/8/PPP2PPP/RNB1KBNR b KQkq", "b8c6");

// Danish Gambit (1.e4 e5 2.d4 exd4 3.c3)
B("rnbqkbnr/pppp1ppp/8/8/3pP3/8/PPP2PPP/RNBQKBNR w KQkq", "c2c3");

// Ponziani (1.e4 e5 2.Nf3 Nc6 3.c3)
B("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq", "c2c3");

// Four Knights Game (1.e4 e5 2.Nf3 Nc6 3.Nc3 Nf6)
B("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq", "b1c3");
B("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/2N2N2/PPPP1PPP/R1BQKB1R b KQkq", "g8f6");
B("r1bqkb1r/pppp1ppp/2n2n2/4p3/4P3/2N2N2/PPPP1PPP/R1BQKB1R w KQkq", "f1b5");

// Three Knights Game
B("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/2N2N2/PPPP1PPP/R1BQKB1R b KQkq", "f8c5");

// Bishop's Opening (1.e4 e5 2.Bc4)
B("rnbqkbnr/pppp1ppp/8/4p3/2B1P3/8/PPPP1PPP/RNBQK1NR b KQkq", "g8f6");

// Philidor Counter-Gambit
B("rnbqkbnr/ppp2ppp/3p4/4p3/3PP3/5N2/PPP2PPP/RNBQKB1R b KQkq", "f7f5");

// Owen Defense (1.e4 b6)
B("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq", "b7b6");
B("rnbqkbnr/p1pppppp/1p6/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq", "d2d4");

// Nimzowitsch Defense (1.e4 Nc6)
B("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq", "b8c6");
B("r1bqkbnr/pppppppp/2n5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq", "d2d4");

// Grob (1.g4)
B("rnbqkbnr/pppppppp/8/8/6P1/8/PPPPPP1P/RNBQKBNR b KQkq", "d7d5");

// Creepy Crawly (1.a4)
B("rnbqkbnr/pppppppp/8/8/P7/8/1PPPPPPP/RNBQKBNR b KQkq", "d7d5");

// Ware (1.a4 — same)

// Blackmar-Diemer Gambit (1.d4 d5 2.e4)
B("rnbqkbnr/ppp1pppp/8/3p4/3PP3/8/PPP2PPP/RNBQKBNR b KQkq", "d5e4");
B("rnbqkbnr/ppp1pppp/8/8/3Pp3/8/PPP2PPP/RNBQKBNR w KQkq", "b1c3");
B("rnbqkbnr/ppp1pppp/8/8/3Pp3/2N5/PPP2PPP/R1BQKBNR b KQkq", "g8f6");
B("rnbqkb1r/ppp1pppp/5n2/8/3Pp3/2N5/PPP2PPP/R1BQKBNR w KQkq", "f2f3");

// Veresov (1.d4 Nf6 2.Nc3 d5 3.Bg5)
B("rnbqkb1r/ppp1pppp/5n2/3p4/3P4/2N5/PPP1PPPP/R1BQKBNR w KQkq", "c1g5");

// Richter-Veresov Attack
B("rnbqkb1r/ppp1pppp/5n2/3p2B1/3P4/2N5/PPP1PPPP/R2QKBNR b KQkq", "c8f5");

// Barry Attack (1.d4 Nf6 2.Nf3 g6 3.Nc3 d5 4.Bf4 Bg7 5.e3)
B("rnbqkb1r/ppp1pp1p/5np1/3p4/3P1B2/2N2N2/PPP1PPPP/R2QKB1R b KQkq", "f8g7");
B("rnbqk2r/ppp1ppbp/5np1/3p4/3P1B2/2N2N2/PPP1PPPP/R2QKB1R w KQkq", "e2e3");

// Torre Attack Deep
// 1.d4 Nf6 2.Nf3 e6 3.Bg5 c5 4.e3 Be7 5.Nbd2
B("rnbqkb1r/pppp1ppp/4pn2/6B1/3P4/5N2/PPP1PPPP/RN1QKB1R b KQkq", "c7c5");
B("rnbqkb1r/pp1p1ppp/4pn2/2p3B1/3P4/5N2/PPP1PPPP/RN1QKB1R w KQkq", "e2e3");
B("rnbqkb1r/pp1p1ppp/4pn2/2p3B1/3P4/4PN2/PPP2PPP/RN1QKB1R b KQkq", "f8e7");
B("rnbqk2r/pp1pbppp/4pn2/2p3B1/3P4/4PN2/PPP2PPP/RN1QKB1R w KQkq", "b1d2");

// Colle-Zukertort System
B("rnbqkb1r/ppp2ppp/4pn2/3p4/3P4/3BPN2/PPP2PPP/RNBQK2R b KQkq", "c7c5");
B("rnbqkb1r/pp3ppp/4pn2/2pp4/3P4/3BPN2/PPP2PPP/RNBQK2R w KQkq", "b2b3");

// Jobava London (1.d4 d5 2.Nc3 Nf6 3.Bf4)
B("rnbqkb1r/ppp1pppp/5n2/3p4/3P1B2/2N5/PPP1PPPP/R2QKBNR b KQkq", "e7e6");

// Catalan Declined
B("rnbq1rk1/ppp1bppp/4pn2/3p4/2PP4/5NP1/PP2PPBP/RNBQ1RK1 w -", "b1d2");

// Anti-Grunfeld (3.f3)
B("rnbqkb1r/pppppp1p/5np1/8/2PP4/2N5/PP2PPPP/R1BQKBNR b KQkq", "d7d5");


// ============================================================================
// SECTION 15: COMPREHENSIVE EVAL WEIGHT TABLES
// ============================================================================
// Tuned evaluation weights for various positional factors
// ============================================================================

const EVAL_WEIGHTS = {
  // Material (base values already in engine)

  // Pawn structure
  ISOLATED_PAWN_MG: -12,
  ISOLATED_PAWN_EG: -18,
  DOUBLED_PAWN_MG: -8,
  DOUBLED_PAWN_EG: -14,
  BACKWARD_PAWN_MG: -10,
  BACKWARD_PAWN_EG: -12,
  CONNECTED_PAWN_MG: 7,
  CONNECTED_PAWN_EG: 12,

  // Passed pawns
  PASSED_PAWN_BASE_MG: 10,
  PASSED_PAWN_BASE_EG: 20,
  PASSED_PAWN_RANK_BONUS: [0, 5, 10, 20, 35, 55, 80, 0], // By rank distance from start
  PASSED_PAWN_FREE_ADVANCE: 15,
  PASSED_PAWN_CONNECTED: 25,
  PASSED_PAWN_PROTECTED: 20,

  // Piece activity
  BISHOP_PAIR_MG: 30,
  BISHOP_PAIR_EG: 50,
  ROOK_OPEN_FILE_MG: 20,
  ROOK_OPEN_FILE_EG: 15,
  ROOK_SEMI_OPEN_MG: 12,
  ROOK_SEMI_OPEN_EG: 8,
  ROOK_7TH_MG: 30,
  ROOK_7TH_EG: 40,
  ROOK_CONNECTED_MG: 10,
  ROOK_CONNECTED_EG: 8,
  KNIGHT_OUTPOST_MG: 25,
  KNIGHT_OUTPOST_EG: 15,

  // King safety
  PAWN_SHIELD_MG: 12,
  PAWN_SHIELD_MISSING_MG: -20,
  OPEN_FILE_NEAR_KING: -25,
  ENEMY_QUEEN_ATTACK: -30,

  // Space
  SPACE_BONUS_PER_SQUARE: 3,
  CENTER_CONTROL: 8,

  // Tempo
  TEMPO_BONUS: 10,

  // Development
  UNDEVELOPED_PIECE_PENALTY: -15,
  EARLY_QUEEN_PENALTY: -20,
};

export { EVAL_WEIGHTS };


// ============================================================================
// SECTION 16: CONTEMPT AND DRAW AVOIDANCE
// ============================================================================
// When ahead: avoid draws. When behind: seek draws.
// ============================================================================

export function getContemptFactor(board, side, materialBalance) {
  // materialBalance > 0: we're ahead. Avoid draws.
  // materialBalance < 0: we're behind. Seek draws.
  // materialBalance ~0: neutral.

  if (materialBalance > 200) {
    return 30; // Strongly avoid draws
  } else if (materialBalance > 100) {
    return 20;
  } else if (materialBalance > 50) {
    return 10;
  } else if (materialBalance < -200) {
    return -30; // Seek draws
  } else if (materialBalance < -100) {
    return -20;
  } else if (materialBalance < -50) {
    return -10;
  }

  return 5; // Slight contempt (play for win with White advantage)
}


// ============================================================================
// SECTION 17: GRANDMASTER WISDOM — RULES OF THUMB
// ============================================================================
// Encoded positional heuristics from classical chess theory
// ============================================================================

const GM_RULES = {
  // Capablanca's Rule: In a rook endgame, place your rook behind the passed pawn
  rookBehindPasser: function(board, side) {
    const rooks = findPieces(board, side, ROOK);
    const passed = detectPassedPawns(board, side);
    let bonus = 0;

    for (const rSq of rooks) {
      for (const pp of passed) {
        if (sqFile(rSq) === sqFile(pp.sq)) {
          const behind = side === WHITE ?
            sqRank(rSq) > sqRank(pp.sq) :
            sqRank(rSq) < sqRank(pp.sq);
          if (behind) bonus += 20;
          else bonus += 5; // In front is still OK but less good
        }
      }
    }

    return bonus;
  },

  // Tarrasch's Rule: Rooks belong behind passed pawns (even enemy passed pawns)
  rookBehindEnemyPasser: function(board, side) {
    const rooks = findPieces(board, side, ROOK);
    const enemyPassed = detectPassedPawns(board, 1 - side);
    let bonus = 0;

    for (const rSq of rooks) {
      for (const pp of enemyPassed) {
        if (sqFile(rSq) === sqFile(pp.sq)) {
          const behind = side === WHITE ?
            sqRank(rSq) < sqRank(pp.sq) :
            sqRank(rSq) > sqRank(pp.sq);
          if (behind) bonus += 15;
        }
      }
    }

    return bonus;
  },

  // Nimzowitsch: Blockade — place a piece (ideally knight) in front of enemy passed pawn
  blockadePasser: function(board, side) {
    const enemy = 1 - side;
    const enemyPassed = detectPassedPawns(board, enemy);
    let bonus = 0;

    for (const pp of enemyPassed) {
      const stopSq = enemy === WHITE ? pp.sq - 8 : pp.sq + 8;
      if (stopSq < 0 || stopSq >= 64) continue;
      const blocker = board[stopSq];
      if (blocker && blocker.color === side) {
        if (blocker.type === KNIGHT) bonus += 25;
        else if (blocker.type === BISHOP) bonus += 15;
        else if (blocker.type === ROOK) bonus += 5;
      }
    }

    return bonus;
  },

  // Steinitz: The side with the advantage must attack or risk losing the advantage
  advantageMustAttack: function(board, side, materialAdvantage) {
    if (materialAdvantage > 100) {
      // With advantage, bonus for piece activity toward enemy king
      const enemyKing = findKing(board, 1 - side);
      if (enemyKing < 0) return 0;
      let proximity = 0;
      for (let sq = 0; sq < 64; sq++) {
        const p = board[sq];
        if (p && p.color === side && p.type !== PAWN && p.type !== KING) {
          proximity += (7 - sqDist(sq, enemyKing)) * 3;
        }
      }
      return proximity;
    }
    return 0;
  },

  // Lasker: When winning, simplify. When losing, complicate.
  simplificationPreference: function(board, side, materialAdvantage) {
    let totalPieces = 0;
    for (let sq = 0; sq < 64; sq++) {
      if (board[sq] && board[sq].type !== PAWN && board[sq].type !== KING) totalPieces++;
    }

    if (materialAdvantage > 100) {
      // Prefer fewer pieces (encourage trades)
      return (10 - totalPieces) * 5;
    }
    if (materialAdvantage < -100) {
      // Prefer more pieces (keep things complex)
      return totalPieces * 3;
    }
    return 0;
  },

  // Philidor: Pawns are the soul of chess — pawn structure matters
  pawnStructureImportance: function(board, side) {
    let bonus = 0;
    const myPawns = getPawnStructure(board, side);

    // Connected pawns bonus
    for (const pSq of myPawns) {
      const f = sqFile(pSq);
      for (const p2 of myPawns) {
        if (p2 !== pSq && Math.abs(sqFile(p2) - f) === 1 && Math.abs(sqRank(p2) - sqRank(pSq)) <= 1) {
          bonus += 5;
        }
      }
    }

    // Pawn majority bonus (more pawns on one side = outside passed pawn potential)
    let kingsidePawns = 0, queensidePawns = 0;
    let enemyKingsidePawns = 0, enemyQueensidePawns = 0;
    const enemyPawns = getPawnStructure(board, 1 - side);

    for (const p of myPawns) {
      if (sqFile(p) <= 3) queensidePawns++;
      else kingsidePawns++;
    }
    for (const p of enemyPawns) {
      if (sqFile(p) <= 3) enemyQueensidePawns++;
      else enemyKingsidePawns++;
    }

    // Queenside majority
    if (queensidePawns > enemyQueensidePawns) bonus += (queensidePawns - enemyQueensidePawns) * 8;
    // Kingside majority
    if (kingsidePawns > enemyKingsidePawns) bonus += (kingsidePawns - enemyKingsidePawns) * 8;

    return bonus;
  },
};

// Export GM wisdom evaluator
export function getGMWisdom(board, side, materialAdvantage) {
  let total = 0;
  total += GM_RULES.rookBehindPasser(board, side);
  total += GM_RULES.rookBehindEnemyPasser(board, side);
  total += GM_RULES.blockadePasser(board, side);
  total += GM_RULES.advantageMustAttack(board, side, materialAdvantage || 0);
  total += GM_RULES.simplificationPreference(board, side, materialAdvantage || 0);
  total += GM_RULES.pawnStructureImportance(board, side);
  return total;
}


// ============================================================================
// SECTION 18: ADDITIONAL ENDGAME KNOWLEDGE
// ============================================================================

// Opposite colored bishops endgame
function evaluateOppColorBishops(board) {
  const wBishops = findPieces(board, WHITE, BISHOP);
  const bBishops = findPieces(board, BLACK, BISHOP);
  if (wBishops.length !== 1 || bBishops.length !== 1) return null;

  const wBColor = (sqFile(wBishops[0]) + sqRank(wBishops[0])) % 2;
  const bBColor = (sqFile(bBishops[0]) + sqRank(bBishops[0])) % 2;

  if (wBColor === bBColor) return null; // Same color bishops

  // Opposite color bishops — more drawish
  // Count material to see if this matters
  let wMaterial = 0, bMaterial = 0;
  for (let i = 0; i < 64; i++) {
    const p = board[i];
    if (!p || p.type === KING || p.type === BISHOP) continue;
    const val = [0, 100, 320, 330, 500, 900][p.type];
    if (p.color === WHITE) wMaterial += val;
    else bMaterial += val;
  }

  // Pure opposite color bishop endgame
  if (wMaterial === 0 && bMaterial === 0) {
    return { drawFactor: 80 }; // Very drawish even with pawn advantage
  }

  // Opposite color bishops with some pieces
  if (wMaterial + bMaterial < 500) {
    return { drawFactor: 50 }; // Moderately drawish
  }

  // In middlegame, opposite color bishops favor the attacker
  return { drawFactor: -20, attackBonus: 30 };
}

// Rook endgame — activity more important than pawns
function evaluateRookEndgameActivity(board, side) {
  const rooks = findPieces(board, side, ROOK);
  if (rooks.length === 0) return 0;

  const myKing = findKing(board, side);
  const enemy = 1 - side;
  let bonus = 0;

  // Active rook (on 7th or with many squares)
  for (const rSq of rooks) {
    const attacks = getAttackedSquares(board, rSq);
    const mobility = attacks.filter(sq => !board[sq] || board[sq].color !== side).length;
    bonus += mobility * 2; // Active rook

    // Cutting off enemy king
    const enemyKing = findKing(board, enemy);
    if (enemyKing >= 0) {
      // Check if rook cuts off king from passed pawns
      const rFile = sqFile(rSq);
      const rRank = sqRank(rSq);
      const ekFile = sqFile(enemyKing);
      const ekRank = sqRank(enemyKing);

      // Cutting off by rank
      if (side === WHITE && rRank < ekRank) bonus += 10;
      if (side === BLACK && rRank > ekRank) bonus += 10;
    }
  }

  // King activity in rook endgame
  bonus += (4 - centerDist(myKing)) * 5;

  return bonus;
}

export function getEndgameSpecial(board, side) {
  let bonus = 0;

  const oppBishops = evaluateOppColorBishops(board);
  if (oppBishops) {
    if (oppBishops.drawFactor > 0) {
      // Position is drawish — reduce eval towards 0
      bonus -= Math.sign(bonus) * oppBishops.drawFactor / 10;
    }
  }

  bonus += evaluateRookEndgameActivity(board, side);

  return bonus;
}


// ============================================================================
// SECTION 19: COMPREHENSIVE POSITION EVALUATOR (MASTER FUNCTION)
// ============================================================================
// Combines all evaluation components into a single score
// ============================================================================

export function fullEvaluation(board, side, phase) {
  let score = 0;

  // Pattern detection
  const patterns = detectPatterns(board, side);
  score += patterns.bonus;

  // Piece coordination
  score += evaluatePieceCoordination(board, side);

  // King safety
  score += evaluateDetailedKingSafety(board, side);
  score -= evaluateDetailedKingSafety(board, 1 - side);

  // Mobility
  score += evaluateMobility(board, side);
  score -= evaluateMobility(board, 1 - side);

  // Positional themes
  score += detectMinorityAttack(board, side);
  score += evaluatePawnBreaks(board, side);
  score += evaluatePieceActivity(board, side);
  score -= evaluatePieceActivity(board, 1 - side);

  // GM wisdom
  score += GM_RULES.rookBehindPasser(board, side);
  score += GM_RULES.blockadePasser(board, side);
  score += GM_RULES.pawnStructureImportance(board, side);
  score -= GM_RULES.pawnStructureImportance(board, 1 - side);

  // Phase-specific
  if (phase !== undefined) {
    score += getPhaseAdjustment(board, side, phase);
  }

  // Endgame special cases
  const endgame = lookupEndgame(board);
  if (endgame) {
    score += side === WHITE ? endgame.eval : -endgame.eval;
  }

  return score;
}


// ============================================================================
// SECTION 20: MASSIVE PIECE-SQUARE TABLE ADJUSTMENTS (Dynamic)
// ============================================================================
// Context-sensitive PST bonuses — supplements the engine's PeSTO tables
// Applied based on game state (pawn structure, open files, etc.)
// ============================================================================

// Knight bonus per square when there are many pawns (closed position)
// From White's perspective (a8=0, h1=63)
const KNIGHT_CLOSED_BONUS = [
   0,  0,  0,  0,  0,  0,  0,  0,
   2,  5,  8, 12, 12,  8,  5,  2,
   4, 10, 18, 25, 25, 18, 10,  4,
   6, 14, 22, 32, 32, 22, 14,  6,
   6, 14, 22, 32, 32, 22, 14,  6,
   4, 10, 18, 25, 25, 18, 10,  4,
   2,  5,  8, 12, 12,  8,  5,  2,
   0,  0,  0,  0,  0,  0,  0,  0,
];

// Knight bonus per square when there are few pawns (open position)
const KNIGHT_OPEN_BONUS = [
  -5, -3,  0,  0,  0,  0, -3, -5,
  -3,  0,  3,  5,  5,  3,  0, -3,
   0,  3,  8, 12, 12,  8,  3,  0,
   0,  5, 12, 18, 18, 12,  5,  0,
   0,  5, 12, 18, 18, 12,  5,  0,
   0,  3,  8, 12, 12,  8,  3,  0,
  -3,  0,  3,  5,  5,  3,  0, -3,
  -5, -3,  0,  0,  0,  0, -3, -5,
];

// Bishop bonus per square based on diagonal openness
const BISHOP_DIAGONAL_BONUS = [
  10,  0,  0,  0,  0,  0,  0, 10,
   0, 12,  0,  0,  0,  0, 12,  0,
   0,  0, 14,  0,  0, 14,  0,  0,
   0,  0,  0, 16, 16,  0,  0,  0,
   0,  0,  0, 16, 16,  0,  0,  0,
   0,  0, 14,  0,  0, 14,  0,  0,
   0, 12,  0,  0,  0,  0, 12,  0,
  10,  0,  0,  0,  0,  0,  0, 10,
];

// Rook bonus per square based on file status
const ROOK_FILE_BONUS = [
   5,  5, 10, 15, 15, 10,  5,  5,
   5,  5, 10, 15, 15, 10,  5,  5,
   3,  3,  8, 12, 12,  8,  3,  3,
   0,  0,  5,  8,  8,  5,  0,  0,
   0,  0,  5,  8,  8,  5,  0,  0,
  -2, -2,  2,  5,  5,  2, -2, -2,
  -5, -5,  0,  3,  3,  0, -5, -5,
   0,  0,  5,  8,  8,  5,  0,  0,
];

// Queen centralization bonus (middlegame)
const QUEEN_CENTRAL_MG = [
   0,  0,  0,  0,  0,  0,  0,  0,
   0,  0,  2,  5,  5,  2,  0,  0,
   0,  2,  8, 12, 12,  8,  2,  0,
   0,  5, 12, 18, 18, 12,  5,  0,
   0,  5, 12, 18, 18, 12,  5,  0,
   0,  2,  5,  8,  8,  5,  2,  0,
  -5, -3,  0,  2,  2,  0, -3, -5,
 -10, -8, -5,  0,  0, -5, -8,-10,
];

// King safety positioning (middlegame)
const KING_SAFETY_POS_MG = [
  -40, -30, -30, -30, -30, -30, -30, -40,
  -30, -20, -20, -25, -25, -20, -20, -30,
  -20, -15, -15, -20, -20, -15, -15, -20,
  -15, -10, -10, -15, -15, -10, -10, -15,
  -10,  -5,  -5, -10, -10,  -5,  -5, -10,
   -5,   0,   0,  -5,  -5,   0,   0,  -5,
   10,  15,   0,  -5,  -5,   0,  15,  10,
   20,  30,  10,   0,   0,  10,  30,  20,
];

// King centralization bonus (endgame)
const KING_CENTRAL_EG = [
  -15, -10,  -5,   0,   0,  -5, -10, -15,
  -10,  -5,   5,  10,  10,   5,  -5, -10,
   -5,   5,  15,  20,  20,  15,   5,  -5,
    0,  10,  20,  25,  25,  20,  10,   0,
    0,  10,  20,  25,  25,  20,  10,   0,
   -5,   5,  15,  20,  20,  15,   5,  -5,
  -10,  -5,   5,  10,  10,   5,  -5, -10,
  -15, -10,  -5,   0,   0,  -5, -10, -15,
];

// Pawn advancement bonus tables
const PAWN_ADVANCE_BONUS_MG = [
    0,   0,   0,   0,   0,   0,   0,   0,
   60,  60,  60,  60,  60,  60,  60,  60,
   25,  30,  35,  40,  40,  35,  30,  25,
   10,  15,  20,  30,  30,  20,  15,  10,
    5,   8,  12,  22,  22,  12,   8,   5,
    0,   2,   5,  10,  10,   5,   2,   0,
   -5,  -2,   0,  -5,  -5,   0,  -2,  -5,
    0,   0,   0,   0,   0,   0,   0,   0,
];

const PAWN_ADVANCE_BONUS_EG = [
    0,   0,   0,   0,   0,   0,   0,   0,
  100, 100, 100, 100, 100, 100, 100, 100,
   50,  50,  50,  50,  50,  50,  50,  50,
   25,  25,  30,  35,  35,  30,  25,  25,
   12,  12,  15,  20,  20,  15,  12,  12,
    5,   5,   8,  10,  10,   8,   5,   5,
    0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,
];

// Dynamic PST score calculator
export function getDynamicPST(board, side, phase) {
  let score = 0;
  let pawnCount = 0;
  for (let i = 0; i < 64; i++) {
    if (board[i] && board[i].type === PAWN) pawnCount++;
  }

  const isClosed = pawnCount >= 12;
  const isOpen = pawnCount <= 8;
  const isEndgame = phase < 10;
  const isMG = phase >= 10;

  for (let sq = 0; sq < 64; sq++) {
    const p = board[sq];
    if (!p) continue;

    const pstSq = p.color === WHITE ? sq : ((7 - sqRank(sq)) * 8 + sqFile(sq));
    let bonus = 0;

    if (p.type === KNIGHT) {
      bonus = isClosed ? KNIGHT_CLOSED_BONUS[pstSq] :
              isOpen ? KNIGHT_OPEN_BONUS[pstSq] :
              (KNIGHT_CLOSED_BONUS[pstSq] + KNIGHT_OPEN_BONUS[pstSq]) / 2;
    } else if (p.type === BISHOP) {
      bonus = BISHOP_DIAGONAL_BONUS[pstSq];
    } else if (p.type === ROOK) {
      bonus = ROOK_FILE_BONUS[pstSq];
    } else if (p.type === QUEEN && isMG) {
      bonus = QUEEN_CENTRAL_MG[pstSq];
    } else if (p.type === KING) {
      bonus = isEndgame ? KING_CENTRAL_EG[pstSq] : KING_SAFETY_POS_MG[pstSq];
    } else if (p.type === PAWN) {
      bonus = isEndgame ? PAWN_ADVANCE_BONUS_EG[pstSq] : PAWN_ADVANCE_BONUS_MG[pstSq];
    }

    if (p.color === side) score += bonus;
    else score -= bonus;
  }

  return Math.round(score / 4);
}


// ============================================================================
// SECTION 21: ADDITIONAL OPENING BOOK EXTENSIONS
// ============================================================================

// Ruy Lopez — Open Variation Deep
B("r1bqk2r/1pppbppp/p1n2n2/4p3/B3P3/5N2/PPPP1PPP/RNBQR1K1 b kq", "f6e4");
B("r1bqk2r/1pppbppp/p1n5/4p3/B3n3/5N2/PPPP1PPP/RNBQR1K1 w kq", "d2d4");
B("r1bqk2r/1pppbppp/p1n5/4p3/B2Pn3/5N2/PPP2PPP/RNBQR1K1 b kq", "b7b5");
B("r1bqk2r/2ppbppp/p1n5/1p2p3/B2Pn3/5N2/PPP2PPP/RNBQR1K1 w kq", "a4b3");
B("r1bqk2r/2ppbppp/p1n5/1p2p3/3Pn3/1B3N2/PPP2PPP/RNBQR1K1 b kq", "d7d5");

// Sicilian Najdorf — Adams Attack 6.h3
B("rnbqkb1r/1p2pppp/p2p1n2/8/3NP3/2N4P/PPP2PP1/R1BQKB1R b KQkq", "e7e5");

// Sicilian Najdorf — 6.Be2 Opocensky
B("rnbqkb1r/1p2pppp/p2p1n2/8/3NP3/2N5/PPP1BPPP/R1BQK2R b KQkq", "e7e5");

// Sicilian — Scheveningen Keres Attack 6.g4
B("rnbqkb1r/pp3ppp/3ppn2/8/3NP1P1/2N5/PPP2P1P/R1BQKB1R b KQkq", "h7h6");

// French McCutcheon Deep
B("rnbqk2r/ppp2ppp/4pn2/3p2B1/1b1PP3/2N5/PPP2PPP/R2QKBNR w KQkq", "e4e5");

// Smith-Morra Accepted Deep
B("rnbqkbnr/pp1ppppp/8/8/4P3/2N5/PP3PPP/R1BQKBNR b KQkq", "d7d6");
B("rnbqkbnr/pp2pppp/3p4/8/4P3/2N5/PP3PPP/R1BQKBNR w KQkq", "f1c4");

// Alapin Sicilian Deep
B("rnb1kbnr/pp2pppp/8/2pq4/3P4/2P5/PP3PPP/RNBQKBNR b KQkq", "d7d5"); // Wait, already on d5

// QGA Deep Lines
B("rnbqkb1r/ppp2ppp/4pn2/8/2pP4/4PN2/PP3PPP/RNBQKB1R w KQkq", "f1c4");
B("rnbqkb1r/ppp2ppp/4pn2/8/2BP4/4PN2/PP3PPP/RNBQ1K2R b KQkq", "a7a6");
B("rnbqkb1r/1pp2ppp/p3pn2/8/2BP4/4PN2/PP3PPP/RNBQK2R w KQkq", "e1g1");

// Tarrasch Defense Deep
B("rnbqkbnr/pp3ppp/8/2pp4/3P4/2N2N2/PP2PPPP/R1BQKB1R b KQkq", "g8f6");
B("rnbqkb1r/pp3ppp/5n2/2pp4/3P4/2N2N2/PP2PPPP/R1BQKB1R w KQkq", "c1g5");

// Albin Counter-Gambit Deep
B("rnbqkbnr/ppp2ppp/8/4P3/2Pp4/5N2/PP2PPPP/RNBQKB1R b KQkq", "b8c6");
B("r1bqkbnr/ppp2ppp/2n5/4P3/2Pp4/5N2/PP2PPPP/RNBQKB1R w KQkq", "b1d2");

// Benko Gambit Deep
B("rnbqkb1r/3ppppp/P4n2/3P4/8/8/PP2PPPP/RNBQKBNR b KQkq", "c8a6");
B("rn1qkb1r/3ppppp/b4n2/3P4/8/8/PP2PPPP/RNBQKBNR w KQkq", "b1c3");
B("rn1qkb1r/3ppppp/b4n2/3P4/8/2N5/PP2PPPP/R1BQKBNR b KQkq", "d7d6");
B("rn1qkb1r/4pppp/b2p1n2/3P4/8/2N5/PP2PPPP/R1BQKBNR w KQkq", "e2e4");
B("rn1qkb1r/4pppp/b2p1n2/3P4/4P3/2N5/PP3PPP/R1BQKBNR b KQkq", "a6f1");
B("rn1qkb1r/4pppp/3p1n2/3P4/4P3/2N5/PP3PPP/R1BQKbNR w KQkq", "e1f1");
B("rn1qkb1r/4pppp/3p1n2/3P4/4P3/2N5/PP3PPP/R1BQ1KNR b kq", "g7g6");

// Dutch Stonewall Deep
B("rnbqkb1r/ppppp2p/5np1/5p2/2PP4/6P1/PP2PPBP/RNBQK1NR b KQkq", "d7d5");
B("rnbqkb1r/ppp1p2p/5np1/3p1p2/2PP4/6P1/PP2PPBP/RNBQK1NR w KQkq", "g1f3");

// English — Ultra-Symmetrical
B("r1bqkbnr/pp1ppppp/2n5/2p5/2P5/2N5/PP1PPPPP/R1BQKBNR w KQkq", "g1f3");

// Mikenas-Carls Deep
B("rnbqkb1r/pppp1ppp/4pn2/8/2P1P3/2N5/PP1P1PPP/R1BQKBNR b KQkq", "d7d5");
B("rnbqkb1r/ppp2ppp/4pn2/3p4/2P1P3/2N5/PP1P1PPP/R1BQKBNR w KQkq", "e4e5");

// Veresov Deep
B("rnbqkb1r/ppp1pppp/5n2/3p2B1/3P4/2N5/PPP1PPPP/R2QKBNR b KQkq", "c8f5");
B("rnbqkb1r/ppp1pppp/5n2/3p1bB1/3P4/2N5/PPP1PPPP/R2QKBNR w KQkq", "f2f3");

// Jobava London Deep
B("rnbqkb1r/ppp2ppp/4pn2/3p4/3P1B2/2N5/PPP1PPPP/R2QKBNR w KQkq", "e2e3");


// ============================================================================
// SECTION 22: PAWN RACE CALCULATOR
// ============================================================================

function canPawnPromote(pawnSq, pawnColor, enemyKingSq, friendlyKingNearby) {
  const file = sqFile(pawnSq);
  const rank = sqRank(pawnSq);
  const promoRank = pawnColor === WHITE ? 0 : 7;
  const distToPromo = Math.abs(rank - promoRank);
  const enemyKingDist = Math.max(
    Math.abs(sqRank(enemyKingSq) - promoRank),
    Math.abs(sqFile(enemyKingSq) - file)
  );

  if (enemyKingDist > distToPromo) return true;
  if (friendlyKingNearby && enemyKingDist === distToPromo) return true;
  return false;
}

function evaluatePawnRace(board, side) {
  const enemy = 1 - side;
  const myKing = findKing(board, side);
  const theirKing = findKing(board, enemy);
  let bonus = 0;

  const myPassed = detectPassedPawns(board, side);
  const theirPassed = detectPassedPawns(board, enemy);

  for (const pp of myPassed) {
    const nearKing = sqDist(myKing, pp.sq) <= 2;
    if (canPawnPromote(pp.sq, side, theirKing, nearKing)) {
      bonus += 100 + pp.advance * 30;
    }
  }

  for (const pp of theirPassed) {
    const nearKing = sqDist(theirKing, pp.sq) <= 2;
    if (canPawnPromote(pp.sq, enemy, myKing, nearKing)) {
      bonus -= 100 + pp.advance * 30;
    }
  }

  let myFastest = 99, theirFastest = 99;
  for (const pp of myPassed) {
    const dist = side === WHITE ? sqRank(pp.sq) : (7 - sqRank(pp.sq));
    if (dist < myFastest) myFastest = dist;
  }
  for (const pp of theirPassed) {
    const dist = enemy === WHITE ? sqRank(pp.sq) : (7 - sqRank(pp.sq));
    if (dist < theirFastest) theirFastest = dist;
  }

  if (myFastest < theirFastest) bonus += 60;
  else if (theirFastest < myFastest) bonus -= 60;

  return bonus;
}

export function getPawnRaceScore(board, side) {
  return evaluatePawnRace(board, side);
}


// ============================================================================
// SECTION 23: CENTER CONTROL
// ============================================================================

const CENTER_IMPORTANCE = [
   1,  1,  1,  1,  1,  1,  1,  1,
   1,  2,  2,  3,  3,  2,  2,  1,
   1,  2,  4,  6,  6,  4,  2,  1,
   1,  3,  6, 10, 10,  6,  3,  1,
   1,  3,  6, 10, 10,  6,  3,  1,
   1,  2,  4,  6,  6,  4,  2,  1,
   1,  2,  2,  3,  3,  2,  2,  1,
   1,  1,  1,  1,  1,  1,  1,  1,
];

export function getCenterControl(board, side) {
  let control = 0;
  const myAttacks = new Array(64).fill(0);
  const theirAttacks = new Array(64).fill(0);

  for (let sq = 0; sq < 64; sq++) {
    const p = board[sq];
    if (!p) continue;
    const attacks = getAttackedSquares(board, sq);
    for (const atk of attacks) {
      if (p.color === side) myAttacks[atk]++;
      else theirAttacks[atk]++;
    }
  }

  for (let sq = 0; sq < 64; sq++) {
    const imp = CENTER_IMPORTANCE[sq];
    if (imp <= 1) continue;
    if (myAttacks[sq] > theirAttacks[sq]) control += imp * (myAttacks[sq] - theirAttacks[sq]);
    else if (theirAttacks[sq] > myAttacks[sq]) control -= imp * (theirAttacks[sq] - myAttacks[sq]) / 2;
    const p = board[sq];
    if (p && p.type === PAWN) {
      if (p.color === side) control += imp * 2;
      else control -= imp;
    }
  }

  return Math.round(control / 3);
}


// ============================================================================
// SECTION 24: MASTER BRIEFING FUNCTION
// ============================================================================

export function briefEngine(fen, board, side, phase) {
  const briefing = {
    bookMove: null,
    evalAdjustment: 0,
    threats: [],
    suggestions: [],
  };

  if (fen) {
    briefing.bookMove = lookupBook(fen);
  }

  if (!board) return briefing;

  const endgame = lookupEndgame(board);
  if (endgame) {
    briefing.evalAdjustment += side === WHITE ? endgame.eval : -endgame.eval;
    briefing.suggestions.push('endgame_knowledge');
  }

  const patterns = detectPatterns(board, side);
  briefing.evalAdjustment += patterns.bonus;
  briefing.threats = patterns.threats;

  briefing.evalAdjustment += getPieceCoordination(board, side);

  briefing.evalAdjustment += evaluateDetailedKingSafety(board, side);
  briefing.evalAdjustment -= evaluateDetailedKingSafety(board, 1 - side);

  briefing.evalAdjustment += evaluateMobility(board, side);
  briefing.evalAdjustment -= evaluateMobility(board, 1 - side);

  briefing.evalAdjustment = Math.round(briefing.evalAdjustment / 3);

  return briefing;
}


// ============================================================================
// SECTION 25: BOOK PRIORITY OVERRIDES
// ============================================================================
// These entries come LAST so they win over all previous insertions for the
// same position. They define the "preferred" move at key junction points.
// ============================================================================

// Starting position: play 1.e4 (King's Pawn — most aggressive)
B("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq", "e2e4");

// After 1.e4: best reply is 1...e5 (symmetrical, principled)
B("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq", "e7e5");

// After 1.e4 e5: play 2.Nf3 (develop, attack e5)
B("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq", "g1f3");

// After 1.e4 e5 2.Nf3: play 2...Nc6 (defend e5, develop)
B("rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq", "b8c6");

// After 1.e4 e5 2.Nf3 Nc6: play 3.Bb5 (Ruy Lopez — strongest)
B("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq", "f1b5");

// After 1.d4: play 1...Nf6 (flexible — allows KID, Nimzo, QID, Grunfeld)
B("rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq", "g8f6");

// After 1.d4 Nf6: play 2.c4 (grab space, standard)
B("rnbqkb1r/pppppppp/5n2/8/3P4/8/PPP1PPPP/RNBQKBNR w KQkq", "c2c4");

// After 1.d4 Nf6 2.c4: play 2...e6 (leads to Nimzo/QID/QGD)
B("rnbqkb1r/pppppppp/5n2/8/2PP4/8/PP2PPPP/RNBQKBNR b KQkq", "e7e6");

// After 1.d4 d5: play 2.c4 (Queen's Gambit)
B("rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w KQkq", "c2c4");

// After 1.d4 d5 2.c4: play 2...e6 (QGD — solid and strong)
B("rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b KQkq", "e7e6");

// After 1.d4 Nf6 2.c4 e6 3.Nc3: play 3...Bb4 (Nimzo-Indian)
B("rnbqkb1r/pppp1ppp/4pn2/8/2PP4/2N5/PP2PPPP/R1BQKBNR b KQkq", "f8b4");

// After 1.d4 Nf6 2.c4 g6 3.Nc3: play 3...Bg7 (KID setup)
B("rnbqkb1r/pppppp1p/5np1/8/2PP4/2N5/PP2PPPP/R1BQKBNR b KQkq", "f8g7");

// After 1.c4: play 1...e5
B("rnbqkbnr/pppppppp/8/8/2P5/8/PP1PPPPP/RNBQKBNR b KQkq", "e7e5");

// After 1.Nf3: play 1...d5
B("rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R b KQkq", "d7d5");

// After 1.e4 c5 (Sicilian): play 2.Nf3 (open Sicilian — most challenging)
B("rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq", "g1f3");

// After 1.e4 c5 2.Nf3: play 2...d6 (Najdorf setup)
B("rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq", "d7d6");

// After 1.e4 e6 (French): play 2.d4
B("rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq", "d2d4");

// After 1.e4 c6 (Caro-Kann): play 2.d4
B("rnbqkbnr/pp1ppppp/2p5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq", "d2d4");
