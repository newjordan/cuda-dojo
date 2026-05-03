const FILES = 'abcdefgh';
const START_FEN = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';

function sqToIdx(sq) {
  return (8 - Number(sq[1])) * 8 + FILES.indexOf(sq[0]);
}

function idxToSq(i) {
  return FILES[i & 7] + (8 - (i >> 3));
}

function opp(side) {
  return side === 'w' ? 'b' : 'w';
}

function colorOf(piece) {
  if (!piece || piece === '.') return null;
  return piece === piece.toUpperCase() ? 'w' : 'b';
}

function inBounds(r, c) {
  return r >= 0 && r < 8 && c >= 0 && c < 8;
}

function parseFen(fen) {
  const [placement, side, castling, ep, halfmove, fullmove] = fen.trim().split(/\s+/);
  const board = [];
  for (const ch of placement) {
    if (ch === '/') continue;
    if (ch >= '1' && ch <= '8') {
      for (let i = 0; i < Number(ch); i++) board.push('.');
    } else {
      board.push(ch);
    }
  }
  return {
    board,
    side: side || 'w',
    castling: castling || '-',
    ep: ep || '-',
    halfmove: Number(halfmove || 0),
    fullmove: Number(fullmove || 1),
  };
}

function boardToFen(pos) {
  let fen = '';
  for (let r = 0; r < 8; r++) {
    let empty = 0;
    for (let c = 0; c < 8; c++) {
      const piece = pos.board[r * 8 + c];
      if (piece === '.') {
        empty++;
      } else {
        if (empty) {
          fen += empty;
          empty = 0;
        }
        fen += piece;
      }
    }
    if (empty) fen += empty;
    if (r < 7) fen += '/';
  }
  return `${fen} ${pos.side} ${pos.castling} ${pos.ep} ${pos.halfmove} ${pos.fullmove}`;
}

function applyUci(pos, uci) {
  const from = sqToIdx(uci.slice(0, 2));
  const to = sqToIdx(uci.slice(2, 4));
  const promo = uci[4] || null;
  const board = [...pos.board];
  const piece = board[from];
  const target = board[to];
  const side = pos.side;
  let castling = pos.castling;
  let ep = '-';
  let halfmove = pos.halfmove + 1;

  board[from] = '.';

  if (piece.toLowerCase() === 'p' && uci.slice(2, 4) === pos.ep) {
    board[side === 'w' ? to + 8 : to - 8] = '.';
    halfmove = 0;
  }

  if (piece.toLowerCase() === 'k' && Math.abs(to - from) === 2) {
    if (to === 62) { board[61] = board[63]; board[63] = '.'; }
    if (to === 58) { board[59] = board[56]; board[56] = '.'; }
    if (to === 6)  { board[5] = board[7]; board[7] = '.'; }
    if (to === 2)  { board[3] = board[0]; board[0] = '.'; }
  }

  board[to] = promo ? (side === 'w' ? promo.toUpperCase() : promo.toLowerCase()) : piece;
  if (piece.toLowerCase() === 'p' || target !== '.') halfmove = 0;

  if (piece.toLowerCase() === 'p' && Math.abs(to - from) === 16) {
    ep = idxToSq((from + to) / 2);
  }

  if (piece === 'K') castling = castling.replace(/[KQ]/g, '');
  if (piece === 'k') castling = castling.replace(/[kq]/g, '');
  if (from === 63 || to === 63) castling = castling.replace('K', '');
  if (from === 56 || to === 56) castling = castling.replace('Q', '');
  if (from === 7 || to === 7) castling = castling.replace('k', '');
  if (from === 0 || to === 0) castling = castling.replace('q', '');
  if (!castling) castling = '-';

  return {
    board,
    side: opp(side),
    castling,
    ep,
    halfmove,
    fullmove: pos.fullmove + (side === 'b' ? 1 : 0),
  };
}

function findKing(board, side) {
  return board.indexOf(side === 'w' ? 'K' : 'k');
}

function isSquareAttacked(board, sq, by) {
  const tr = sq >> 3;
  const tc = sq & 7;

  const pawnRow = by === 'w' ? tr + 1 : tr - 1;
  for (const dc of [-1, 1]) {
    const nr = pawnRow;
    const nc = tc + dc;
    if (inBounds(nr, nc)) {
      const p = board[nr * 8 + nc];
      if (p !== '.' && colorOf(p) === by && p.toLowerCase() === 'p') return true;
    }
  }

  for (const [dr, dc] of [[-2,-1],[-2,1],[-1,-2],[-1,2],[1,-2],[1,2],[2,-1],[2,1]]) {
    const nr = tr + dr;
    const nc = tc + dc;
    if (inBounds(nr, nc)) {
      const p = board[nr * 8 + nc];
      if (p !== '.' && colorOf(p) === by && p.toLowerCase() === 'n') return true;
    }
  }

  for (const [dr, dc] of [[-1,-1],[-1,1],[1,-1],[1,1]]) {
    let nr = tr + dr;
    let nc = tc + dc;
    while (inBounds(nr, nc)) {
      const p = board[nr * 8 + nc];
      if (p !== '.') {
        if (colorOf(p) === by && (p.toLowerCase() === 'b' || p.toLowerCase() === 'q')) return true;
        break;
      }
      nr += dr;
      nc += dc;
    }
  }

  for (const [dr, dc] of [[-1,0],[1,0],[0,-1],[0,1]]) {
    let nr = tr + dr;
    let nc = tc + dc;
    while (inBounds(nr, nc)) {
      const p = board[nr * 8 + nc];
      if (p !== '.') {
        if (colorOf(p) === by && (p.toLowerCase() === 'r' || p.toLowerCase() === 'q')) return true;
        break;
      }
      nr += dr;
      nc += dc;
    }
  }

  for (let dr = -1; dr <= 1; dr++) {
    for (let dc = -1; dc <= 1; dc++) {
      if (dr === 0 && dc === 0) continue;
      const nr = tr + dr;
      const nc = tc + dc;
      if (inBounds(nr, nc)) {
        const p = board[nr * 8 + nc];
        if (p !== '.' && colorOf(p) === by && p.toLowerCase() === 'k') return true;
      }
    }
  }

  return false;
}

function isInCheck(board, side) {
  const kingSq = findKing(board, side);
  if (kingSq < 0) return true;
  return isSquareAttacked(board, kingSq, opp(side));
}

function generateLegalMoves(pos) {
  const moves = [];
  const { board, side, castling, ep } = pos;
  const enemy = opp(side);

  for (let i = 0; i < 64; i++) {
    const piece = board[i];
    if (piece === '.' || colorOf(piece) !== side) continue;
    const r = i >> 3;
    const c = i & 7;
    const lower = piece.toLowerCase();

    if (lower === 'p') {
      const dir = side === 'w' ? -1 : 1;
      const startRank = side === 'w' ? 6 : 1;
      const promoRank = side === 'w' ? 0 : 7;
      const oneR = r + dir;

      if (inBounds(oneR, c) && board[oneR * 8 + c] === '.') {
        if (oneR === promoRank) {
          for (const promo of ['q', 'r', 'b', 'n']) moves.push(idxToSq(i) + idxToSq(oneR * 8 + c) + promo);
        } else {
          moves.push(idxToSq(i) + idxToSq(oneR * 8 + c));
          if (r === startRank) {
            const twoR = r + dir * 2;
            if (board[twoR * 8 + c] === '.') moves.push(idxToSq(i) + idxToSq(twoR * 8 + c));
          }
        }
      }

      for (const dc of [-1, 1]) {
        const nr = r + dir;
        const nc = c + dc;
        if (!inBounds(nr, nc)) continue;
        const to = nr * 8 + nc;
        const toSq = idxToSq(to);
        if ((board[to] !== '.' && colorOf(board[to]) === enemy) || toSq === ep) {
          if (nr === promoRank) {
            for (const promo of ['q', 'r', 'b', 'n']) moves.push(idxToSq(i) + toSq + promo);
          } else {
            moves.push(idxToSq(i) + toSq);
          }
        }
      }
    } else if (lower === 'n') {
      for (const [dr, dc] of [[-2,-1],[-2,1],[-1,-2],[-1,2],[1,-2],[1,2],[2,-1],[2,1]]) {
        const nr = r + dr;
        const nc = c + dc;
        if (!inBounds(nr, nc)) continue;
        const t = board[nr * 8 + nc];
        if (t === '.' || colorOf(t) === enemy) moves.push(idxToSq(i) + idxToSq(nr * 8 + nc));
      }
    } else if (lower === 'k') {
      for (let dr = -1; dr <= 1; dr++) {
        for (let dc = -1; dc <= 1; dc++) {
          if (dr === 0 && dc === 0) continue;
          const nr = r + dr;
          const nc = c + dc;
          if (!inBounds(nr, nc)) continue;
          const t = board[nr * 8 + nc];
          if (t === '.' || colorOf(t) === enemy) moves.push(idxToSq(i) + idxToSq(nr * 8 + nc));
        }
      }

      const kingSq = side === 'w' ? 60 : 4;
      if (i === kingSq) {
        if (side === 'w') {
          if (
            castling.includes('K') &&
            board[61] === '.' &&
            board[62] === '.' &&
            board[63] === 'R' &&
            !isInCheck(board, side) &&
            !isSquareAttacked(board, 61, enemy) &&
            !isSquareAttacked(board, 62, enemy)
          ) moves.push('e1g1');

          if (
            castling.includes('Q') &&
            board[59] === '.' &&
            board[58] === '.' &&
            board[57] === '.' &&
            board[56] === 'R' &&
            !isInCheck(board, side) &&
            !isSquareAttacked(board, 59, enemy) &&
            !isSquareAttacked(board, 58, enemy)
          ) moves.push('e1c1');
        } else {
          if (
            castling.includes('k') &&
            board[5] === '.' &&
            board[6] === '.' &&
            board[7] === 'r' &&
            !isInCheck(board, side) &&
            !isSquareAttacked(board, 4, enemy) &&
            !isSquareAttacked(board, 5, enemy) &&
            !isSquareAttacked(board, 6, enemy)
          ) moves.push('e8g8');

          if (
            castling.includes('q') &&
            board[3] === '.' &&
            board[2] === '.' &&
            board[1] === '.' &&
            board[0] === 'r' &&
            !isInCheck(board, side) &&
            !isSquareAttacked(board, 4, enemy) &&
            !isSquareAttacked(board, 3, enemy) &&
            !isSquareAttacked(board, 2, enemy)
          ) moves.push('e8c8');
        }
      }
    } else {
      const dirs = lower === 'b'
        ? [[-1,-1],[-1,1],[1,-1],[1,1]]
        : lower === 'r'
          ? [[-1,0],[1,0],[0,-1],[0,1]]
          : [[-1,-1],[-1,1],[1,-1],[1,1],[-1,0],[1,0],[0,-1],[0,1]];

      for (const [dr, dc] of dirs) {
        let nr = r + dr;
        let nc = c + dc;
        while (inBounds(nr, nc)) {
          const t = board[nr * 8 + nc];
          if (t === '.') {
            moves.push(idxToSq(i) + idxToSq(nr * 8 + nc));
          } else {
            if (colorOf(t) === enemy) moves.push(idxToSq(i) + idxToSq(nr * 8 + nc));
            break;
          }
          nr += dr;
          nc += dc;
        }
      }
    }
  }

  return moves.filter((uci) => {
    const next = applyUci(pos, uci);
    return !isInCheck(next.board, side);
  });
}

function insuffMat(board) {
  const pieces = board.filter((p) => p !== '.');
  if (pieces.length <= 2) return true;
  if (pieces.length === 3 && pieces.some((p) => p.toLowerCase() === 'b' || p.toLowerCase() === 'n')) return true;
  return false;
}

export {
  START_FEN,
  sqToIdx,
  idxToSq,
  opp,
  parseFen,
  boardToFen,
  applyUci,
  generateLegalMoves,
  isInCheck,
  insuffMat,
};
