// ============================================================================
// GPU FORGE v2 — PeSTO evaluation + Alpha-Beta search
//
// v1 used MCTS random playouts for move selection → 0% agreement with CPU
// v2 uses the same PeSTO piece-square tables as CPU fighters + alpha-beta
// negamax search, achieving deterministic move selection that matches CPU
// fighter decision-making.
//
// Input: batch of FEN positions (stdin, one per line)
// Each line: FEN\tcurrent_engine_move\tlegal1,legal2,...
//
// For each position:
//   1. Alpha-beta search finds the best move (GPU, root parallelism)
//   2. Compare with CPU engine move
//   3. If disagree, knob tuner tests configs (GPU, evalWithKnobs)
//
// Usage: cat positions.txt | ./gpu_forge [num_configs] [mcts_sims] [--fighter-blob path] [--depth N]
// ============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <stdint.h>
#include <fstream>
#include <string>
#include <time.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "generated/dojo_active_fighter_legacy.h"

#define MAX_POSITIONS 10000
#define MAX_FEN_LEN 128
#define MAX_MOVE_LEN 8
#define BLOCK_SIZE 256
#define KNOB_COUNT 44
#define MAX_BOARD 64
#define MAX_MOVES 256
#define DEFAULT_SEARCH_DEPTH 5
#define MAX_QUIESCE_DEPTH 6

// Piece types
#define EMPTY 0
#define PAWN 1
#define KNIGHT 2
#define BISHOP 3
#define ROOK 4
#define QUEEN 5
#define KING 6
#define WHITE_FLAG 0
#define BLACK_FLAG 8

// ============================================================================
// PeSTO PIECE-SQUARE TABLES (constant memory)
// Identical to CPU fighter tables. Index 0=a8, 63=h1.
// White perspective; black uses MIRROR lookup.
// ============================================================================

__constant__ int C_MG_PIECE_VAL[7] = {0, 82, 337, 365, 477, 1025, 0};
__constant__ int C_EG_PIECE_VAL[7] = {0, 94, 281, 297, 512, 936, 0};
__constant__ int C_PHASE_WEIGHTS[7] = {0, 0, 1, 1, 2, 4, 0};

__constant__ int C_MG_PST[7][64] = {
  // [0] = EMPTY — unused
  {0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,
   0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0},
  // [1] = PAWN MG
  {0,0,0,0,0,0,0,0,
   98,134,61,95,68,126,34,-11,
   -6,7,26,31,65,56,25,-20,
   -14,13,6,21,23,12,17,-23,
   -27,-2,-5,12,17,6,10,-25,
   -26,-4,-4,-10,3,3,33,-12,
   -35,-1,-20,-23,-15,24,38,-22,
   0,0,0,0,0,0,0,0},
  // [2] = KNIGHT MG
  {-167,-89,-34,-49,61,-97,50,-73,
   -73,-41,72,36,23,62,7,-17,
   -47,60,37,65,84,129,73,44,
   -9,17,19,53,37,69,18,22,
   -13,4,16,13,28,19,21,-8,
   -23,-9,12,10,19,17,25,-16,
   -29,-53,-12,-3,-1,18,-14,-19,
   -105,-21,-58,-33,-17,-28,-19,-23},
  // [3] = BISHOP MG
  {-29,4,-82,-37,-25,-42,7,-8,
   -26,16,-18,-13,30,59,18,-47,
   -16,37,43,40,35,50,37,-2,
   -4,5,19,50,37,37,7,-2,
   -6,13,13,26,34,12,10,4,
   0,15,15,15,14,27,18,10,
   4,15,16,0,7,21,33,1,
   -33,-3,-14,-21,-13,-12,-39,-21},
  // [4] = ROOK MG
  {32,42,32,51,63,9,31,43,
   27,32,58,62,80,67,26,44,
   -5,19,26,36,17,45,61,16,
   -24,-11,7,26,24,35,-8,-20,
   -36,-26,-12,-1,9,-7,6,-23,
   -45,-25,-16,-17,3,0,-5,-33,
   -44,-16,-20,-9,-1,11,-6,-71,
   -19,-13,1,17,16,7,-37,-26},
  // [5] = QUEEN MG
  {-28,0,29,12,59,44,43,45,
   -24,-39,-5,1,-16,57,28,54,
   -13,-17,7,8,29,56,47,57,
   -27,-27,-16,-16,-1,17,-2,1,
   -9,-26,-9,-10,-2,-4,3,-3,
   -14,-2,-11,-2,-5,2,14,5,
   -35,-8,11,2,8,15,-3,1,
   -1,-18,-9,10,-15,-25,-31,-50},
  // [6] = KING MG
  {-65,23,16,-15,-56,-34,2,13,
   29,-1,-20,-7,-8,-4,-38,-29,
   -9,24,2,-16,-20,6,22,-22,
   -17,-20,-12,-27,-30,-25,-14,-36,
   -49,-1,-27,-39,-46,-44,-33,-51,
   -14,-14,-22,-46,-44,-30,-15,-27,
   1,7,-8,-64,-43,-16,9,8,
   -15,36,12,-54,8,-28,24,14}
};

__constant__ int C_EG_PST[7][64] = {
  // [0] = EMPTY — unused
  {0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,
   0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0},
  // [1] = PAWN EG
  {0,0,0,0,0,0,0,0,
   178,173,158,134,147,132,165,187,
   94,100,85,67,56,53,82,84,
   32,24,13,5,-2,4,17,17,
   13,9,-3,-7,-7,-8,3,-1,
   4,7,-6,1,0,-5,-1,-8,
   13,8,8,10,13,0,2,-7,
   0,0,0,0,0,0,0,0},
  // [2] = KNIGHT EG
  {-58,-38,-13,-28,-31,-27,-63,-99,
   -25,-8,-25,-2,-9,-25,-24,-52,
   -24,-20,10,9,-1,-9,-19,-41,
   -17,3,22,22,22,11,8,-18,
   -18,-6,16,25,16,17,4,-18,
   -23,-3,-1,15,10,-3,-20,-22,
   -42,-20,-10,-5,-2,-20,-23,-44,
   -29,-51,-23,-15,-22,-18,-50,-64},
  // [3] = BISHOP EG
  {-14,-21,-11,-8,-7,-9,-17,-24,
   -8,-4,7,-12,-3,-13,-4,-14,
   2,-8,0,-1,-2,6,0,4,
   -3,9,12,9,14,10,3,2,
   -6,3,13,19,7,10,-3,-9,
   -12,-3,8,10,13,3,-7,-15,
   -14,-18,-7,-1,4,-9,-15,-27,
   -23,-9,-23,-5,-9,-16,-5,-17},
  // [4] = ROOK EG
  {13,10,18,15,12,12,8,5,
   11,13,13,11,-3,7,7,8,
   7,7,7,5,4,-3,-5,3,
   4,3,13,1,2,1,-1,2,
   3,5,8,4,-5,-6,-8,-11,
   -4,0,-5,-1,-7,-12,-8,-16,
   -6,-6,0,2,-9,-9,-11,-3,
   -9,2,3,-1,-5,-13,4,-20},
  // [5] = QUEEN EG
  {-9,22,22,27,27,19,10,20,
   -17,20,32,41,58,25,30,0,
   -20,6,9,49,47,35,19,9,
   3,22,24,45,57,40,57,36,
   -18,28,19,47,31,34,39,23,
   -16,-27,15,6,9,17,10,5,
   -22,-23,-30,-16,-16,-23,-36,-32,
   -33,-28,-22,-43,-5,-32,-20,-41},
  // [6] = KING EG
  {-74,-35,-18,-18,-11,15,4,-17,
   -12,17,14,17,17,38,23,11,
   10,17,23,15,20,45,44,13,
   -8,22,24,27,26,33,26,3,
   -18,-4,21,24,27,23,9,-11,
   -19,-3,11,21,23,16,7,-9,
   -27,-11,4,13,14,4,-5,-17,
   -53,-34,-21,-11,-28,-14,-24,-43}
};

// Mirror table: flips square for black PST lookup (a8↔a1 etc.)
__constant__ int C_MIRROR[64] = {
  56,57,58,59,60,61,62,63,
  48,49,50,51,52,53,54,55,
  40,41,42,43,44,45,46,47,
  32,33,34,35,36,37,38,39,
  24,25,26,27,28,29,30,31,
  16,17,18,19,20,21,22,23,
   8, 9,10,11,12,13,14,15,
   0, 1, 2, 3, 4, 5, 6, 7
};

// MVV-LVA victim values for move ordering
__constant__ int C_MVV_VAL[7] = {0, 100, 320, 330, 500, 900, 20000};

// Fighter personality knobs — loaded at runtime from blob
__constant__ float C_FIGHTER_KNOBS[KNOB_COUNT];
__constant__ float C_FLAT_KNOBS[64];
__constant__ int C_FULL_QEVAL;
__constant__ int C_FILTER_LEGAL;
__constant__ int C_TRAINCAR_EVAL;
__constant__ int C_CPU_SHAPED_SEARCH;

// ============================================================================
// DATA STRUCTURES
// ============================================================================

struct Board {
  int pieces[64];
  int side;      // 0=white, 1=black
  int phase;     // material phase (0-24)
  int castling;  // bitmask (unused in search but kept for compat)
  int epSq;      // en passant square (-1 if none)
  float mgScore; // incremental middlegame PeSTO score (white-relative)
  float egScore; // incremental endgame PeSTO score (white-relative)
};

struct KnobConfig {
  float knobs[KNOB_COUNT];
};

struct Move {
  int from, to, promo;
};

// ============================================================================
// HOST: String utilities
// ============================================================================

static inline char* ltrim(char* s) {
  while (s && *s && isspace((unsigned char)*s)) s++;
  return s;
}

static inline void rtrimInPlace(char* s) {
  if (!s) return;
  int len = (int)strlen(s);
  while (len > 0 && isspace((unsigned char)s[len - 1])) {
    s[len - 1] = 0;
    len--;
  }
}

static int parseUciSquare(const char* s) {
  if (!s || !s[0] || !s[1]) return -1;
  char file = (char)tolower((unsigned char)s[0]);
  char rank = s[1];
  if (file < 'a' || file > 'h' || rank < '1' || rank > '8') return -1;
  return (8 - (rank - '0')) * 8 + (file - 'a');
}

static bool parseUciMoveToken(const char* token, Move* out) {
  if (!token || !out) return false;
  while (*token && isspace((unsigned char)*token)) token++;
  if (!token[0] || !token[1] || !token[2] || !token[3]) return false;
  int from = parseUciSquare(token);
  int to = parseUciSquare(token + 2);
  if (from < 0 || to < 0) return false;
  out->from = from;
  out->to = to;
  out->promo = 0;
  if (token[4]) {
    switch ((char)tolower((unsigned char)token[4])) {
      case 'q': out->promo = QUEEN; break;
      case 'r': out->promo = ROOK; break;
      case 'b': out->promo = BISHOP; break;
      case 'n': out->promo = KNIGHT; break;
      default: out->promo = 0; break;
    }
  }
  return true;
}

static int parseLegalMoveList(char* movesText, Move* outMoves, int maxMoves) {
  if (!movesText || !outMoves || maxMoves <= 0) return 0;
  int count = 0;
  char* cursor = movesText;
  while (cursor && *cursor && count < maxMoves) {
    char* token = cursor;
    char* comma = strchr(cursor, ',');
    if (comma) { *comma = 0; cursor = comma + 1; }
    else { cursor = NULL; }
    token = ltrim(token);
    rtrimInPlace(token);
    if (!*token) continue;
    Move mv;
    if (parseUciMoveToken(token, &mv)) outMoves[count++] = mv;
  }
  return count;
}

static char promoPieceToChar(int promo) {
  switch (promo) {
    case QUEEN: return 'q';
    case ROOK: return 'r';
    case BISHOP: return 'b';
    case KNIGHT: return 'n';
    default: return 0;
  }
}

static void moveToUci(const Move* move, char* out, size_t outSize) {
  if (!move || !out || outSize < 6) return;
  char promo = promoPieceToChar(move->promo);
  out[0] = (char)('a' + (move->from % 8));
  out[1] = (char)('0' + (8 - (move->from / 8)));
  out[2] = (char)('a' + (move->to % 8));
  out[3] = (char)('0' + (8 - (move->to / 8)));
  if (promo) {
    out[4] = promo;
    out[5] = 0;
  } else {
    out[4] = 0;
  }
}

static int hostMoveCapturedPiece(const Board* board, const Move* move) {
  int captured = board->pieces[move->to];
  int moving = board->pieces[move->from];
  int movingType = moving & 7;
  int movingIsW = (moving & 8) == 0;
  if (!captured && movingType == PAWN && move->to == board->epSq) {
    int capSq = movingIsW ? move->to + 8 : move->to - 8;
    if (capSq >= 0 && capSq < 64) captured = board->pieces[capSq];
  }
  return captured;
}

static int hostRootMoveScore(const Board* board, const Move* move) {
  int moving = board->pieces[move->from];
  int movingType = moving & 7;
  int promo = move->promo;
  if (!promo && movingType == PAWN) {
    int toRank = move->to / 8;
    int movingIsW = (moving & 8) == 0;
    if ((movingIsW && toRank == 0) || (!movingIsW && toRank == 7)) promo = QUEEN;
  }
  if (promo) return 9000000 + (promo == QUEEN ? 1000 : 0);

  int captured = hostMoveCapturedPiece(board, move);
  if (captured) return 1000000 + ((captured & 7) * 100) - movingType;

  return 0;
}

static int hostTraincarRootTieScore(const Board* board, const Move* move) {
  int moving = board->pieces[move->from];
  if (!moving) return 0;
  int movingType = moving & 7;
  int fromRank = move->from / 8;
  int fromFile = move->from % 8;
  int toRank = move->to / 8;
  int toFile = move->to % 8;
  int score = hostRootMoveScore(board, move);

  if (movingType == KNIGHT) {
    if ((fromRank == 7 || fromRank == 0) && (toFile == 2 || toFile == 5)) score += 700;
    if (toFile == 0 || toFile == 7) score -= 250;
    if (toRank >= 2 && toRank <= 5 && toFile >= 2 && toFile <= 5) score += 120;
  } else if (movingType == BISHOP) {
    if ((fromFile == 2 && toFile == 1) || (fromFile == 5 && toFile == 6)) score += 520;
    if (toRank >= 2 && toRank <= 5 && toFile >= 2 && toFile <= 5) score += 100;
  } else if (movingType == PAWN) {
    int advance = move->from > move->to ? move->from - move->to : move->to - move->from;
    if (toFile == 3 || toFile == 4) score += advance == 16 ? 430 : 260;
    if ((fromFile == 1 || fromFile == 6) && advance == 16) score += 90;
    if ((fromFile == 0 || fromFile == 7) && advance == 16) score -= 60;
  }

  return score;
}

static void orderRootMovesHost(const Board* board, Move* moves, int count) {
  int scores[MAX_MOVES];
  for (int i = 0; i < count; i++) scores[i] = hostRootMoveScore(board, &moves[i]);
  for (int i = 1; i < count; i++) {
    Move mv = moves[i];
    int sc = scores[i];
    int j = i - 1;
    while (j >= 0 && scores[j] < sc) {
      moves[j + 1] = moves[j];
      scores[j + 1] = scores[j];
      j--;
    }
    moves[j + 1] = mv;
    scores[j + 1] = sc;
  }
}

static void printRootCandidatesJson(Move* moves, float* scores, int count, int engIdx, int bestIdx, int topN) {
  bool used[MAX_MOVES] = {false};
  printf("[");
  int emitted = 0;
  for (int rank = 0; rank < topN && rank < count; rank++) {
    int idx = -1;
    float best = -1.0e30f;
    for (int i = 0; i < count; i++) {
      if (!used[i] && scores[i] > best) {
        best = scores[i];
        idx = i;
      }
    }
    if (idx < 0) break;
    used[idx] = true;
    char mv[8];
    moveToUci(&moves[idx], mv, sizeof(mv));
    if (emitted) printf(",");
    printf("{\"move\":\"%s\",\"score\":%.3f,\"order\":%d,\"rank\":%d,"
           "\"is_engine\":%s,\"is_best\":%s}",
      mv, scores[idx] / 100.0f, idx, emitted + 1,
      idx == engIdx ? "true" : "false",
      idx == bestIdx ? "true" : "false");
    emitted++;
  }
  printf("]");
}

static void printPositionComparisonJson(
  const char* fen,
  const char* engineMove,
  const char* searchMove,
  float bestScore,
  float engineScore,
  int engineRank,
  int legalCount,
  int fixable,
  int numConfigs,
  Move* moves,
  float* scores,
  int engIdx,
  int bestIdx,
  int topN,
  bool agree
) {
  printf("{\"fen\":\"%s\",\"engine\":\"%s\",\"mcts\":\"%s\",\"agree\":%s,\"mcts_wr\":%.3f,"
         "\"engine_score\":%.3f,\"gpu_score\":%.3f,\"score_delta\":%.3f,"
         "\"engine_rank\":%d,\"legal_count\":%d,\"fixable\":%d,\"rate\":%.3f,"
         "\"gpu_root\":",
    fen, engineMove, searchMove, agree ? "true" : "false", bestScore / 100.0f,
    engineScore / 100.0f, bestScore / 100.0f, (bestScore - engineScore) / 100.0f,
    engineRank, legalCount, fixable, numConfigs > 0 ? (float)fixable / numConfigs : 0.0f);
  printRootCandidatesJson(moves, scores, legalCount, engIdx, bestIdx, topN);
  printf("}");
}

// ============================================================================
// HOST: Runtime fighter blob loader
// ============================================================================

static const char* resolveFighterBlobPath(int argc, char** argv) {
  const char* envPath = getenv("DOJO_CUDA_FIGHTER_BLOB");
  const char* resolved = (envPath && *envPath) ? envPath : NULL;
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--fighter-blob") == 0 && i + 1 < argc) {
      resolved = argv[++i];
    }
  }
  return resolved;
}

static int parseFloatArray(const char* start, float* out, int maxCount) {
  int count = 0;
  const char* cursor = start;
  while (*cursor && *cursor != ']' && count < maxCount) {
    while (*cursor && (isspace((unsigned char)*cursor) || *cursor == ',')) cursor++;
    if (*cursor == ']') break;
    char* end = NULL;
    double value = strtod(cursor, &end);
    if (end == cursor) return -1;
    out[count++] = (float)value;
    cursor = end;
  }
  return count;
}

static bool loadLegacyDefaultsFromBlob(const char* blobPath, float* out, int count) {
  if (!blobPath || !*blobPath) return false;
  std::ifstream stream(blobPath, std::ios::in | std::ios::binary);
  if (!stream.good()) return false;
  stream.seekg(0, std::ios::end);
  const std::streamoff size = stream.tellg();
  if (size <= 0) return false;
  std::string text;
  text.resize((size_t)size);
  stream.seekg(0, std::ios::beg);
  stream.read(&text[0], size);
  if (!stream.good() && !stream.eof()) return false;
  const char* root = text.c_str();
  const char* pack = strstr(root, "\"legacy44Pack\"");
  if (!pack) return false;
  const char* values = strstr(pack, "\"values\"");
  if (!values) return false;
  const char* open = strchr(values, '[');
  if (!open) return false;
  float legacy44[DOJO_ACTIVE_FIGHTER_LEGACY44_COUNT];
  const int parsed = parseFloatArray(open + 1, legacy44, DOJO_ACTIVE_FIGHTER_LEGACY44_COUNT);
  if (parsed < DOJO_ACTIVE_FIGHTER_LEGACY44_COUNT) return false;
  for (int i = 0; i < count && i < DOJO_ACTIVE_FIGHTER_LEGACY44_COUNT; i++)
    out[i] = legacy44[i];
  for (int i = DOJO_ACTIVE_FIGHTER_LEGACY44_COUNT; i < count; i++)
    out[i] = legacy44[DOJO_ACTIVE_FIGHTER_LEGACY44_COUNT - 1];
  return true;
}

static bool loadFlatKnobsFromBlob(const char* blobPath, float* out, int count) {
  if (!blobPath || !*blobPath) return false;
  std::ifstream stream(blobPath, std::ios::in | std::ios::binary);
  if (!stream.good()) return false;
  stream.seekg(0, std::ios::end);
  const std::streamoff size = stream.tellg();
  if (size <= 0) return false;
  std::string text;
  text.resize((size_t)size);
  stream.seekg(0, std::ios::beg);
  stream.read(&text[0], size);
  if (!stream.good() && !stream.eof()) return false;
  const char* root = text.c_str();
  const char* flat = strstr(root, "\"flatKnobs\"");
  if (!flat) return false;
  const char* open = strchr(flat, '[');
  if (!open) return false;
  const int parsed = parseFloatArray(open + 1, out, count);
  return parsed > 0;
}

enum FighterFamily {
  FIGHTER_FAMILY_UNKNOWN = 0,
  FIGHTER_FAMILY_TRAINCAR = 1,
  FIGHTER_FAMILY_RAZORBLADE_II = 2,
};

static FighterFamily detectFighterFamilyFromBlob(const char* blobPath) {
  if (!blobPath || !*blobPath) return FIGHTER_FAMILY_UNKNOWN;
  std::ifstream stream(blobPath, std::ios::in | std::ios::binary);
  if (!stream.good()) return FIGHTER_FAMILY_UNKNOWN;
  stream.seekg(0, std::ios::end);
  const std::streamoff size = stream.tellg();
  if (size <= 0) return FIGHTER_FAMILY_UNKNOWN;
  std::string text;
  text.resize((size_t)size);
  stream.seekg(0, std::ios::beg);
  stream.read(&text[0], size);
  if (!stream.good() && !stream.eof()) return FIGHTER_FAMILY_UNKNOWN;

  const char* root = text.c_str();
  if (strstr(root, "RAZORBLADE_II_DOJO__MAX_DEPTH") || strstr(root, "\"name\":\"MAX_DEPTH\""))
    return FIGHTER_FAMILY_RAZORBLADE_II;
  if (strstr(root, "RAZOR_X_KNOBS") || strstr(root, "QUEENGUARD_X_KNOBS") ||
      strstr(root, "FORTRESS_X_KNOBS") || strstr(root, "RAZOR_X_5S_KNOBS"))
    return FIGHTER_FAMILY_TRAINCAR;
  return FIGHTER_FAMILY_UNKNOWN;
}

static const char* fighterFamilyName(FighterFamily family) {
  switch (family) {
    case FIGHTER_FAMILY_TRAINCAR: return "traincar";
    case FIGHTER_FAMILY_RAZORBLADE_II: return "razorblade_ii";
    default: return "unknown";
  }
}

struct TraincarBookEntry {
  uint32_t hash;
  char move[8];
};

#define MAX_TRAINCAR_BOOK_ENTRIES 8192

static uint32_t fnv1aHash(const char* text, size_t len) {
  uint32_t h = 0x811c9dc5u;
  for (size_t i = 0; i < len; i++) {
    h ^= (unsigned char)text[i];
    h *= 0x01000193u;
  }
  return h;
}

static bool fenKeyLen(const char* fen, size_t* outLen) {
  if (!fen || !outLen) return false;
  const char* p = fen;
  while (*p && isspace((unsigned char)*p)) p++;
  const char* start = p;
  int fields = 0;
  bool inToken = false;
  while (*p) {
    if (isspace((unsigned char)*p)) {
      if (inToken) {
        fields++;
        if (fields >= 3) {
          *outLen = (size_t)(p - start);
          return true;
        }
        inToken = false;
      }
      while (*p && isspace((unsigned char)*p)) p++;
      continue;
    }
    inToken = true;
    p++;
  }
  if (inToken) fields++;
  if (fields >= 3) {
    *outLen = (size_t)(p - start);
    return true;
  }
  return false;
}

static uint32_t hashFenKey(const char* fen) {
  size_t len = 0;
  if (!fenKeyLen(fen, &len)) return 0;
  while (*fen && isspace((unsigned char)*fen)) fen++;
  return fnv1aHash(fen, len);
}

static bool readQuotedString(const char* cursor, char* out, size_t outSize, const char** after) {
  if (!cursor || !out || outSize == 0) return false;
  const char* p = strchr(cursor, '"');
  if (!p) return false;
  p++;
  size_t n = 0;
  while (*p && *p != '"') {
    char ch = *p++;
    if (ch == '\\' && *p) ch = *p++;
    if (n + 1 < outSize) out[n++] = ch;
  }
  if (*p != '"') return false;
  out[n] = 0;
  if (after) *after = p + 1;
  return true;
}

static int loadTraincarBook(const char* path, TraincarBookEntry* entries, int maxEntries) {
  if (!path || !*path || !entries || maxEntries <= 0) return 0;
  std::ifstream stream(path, std::ios::in | std::ios::binary);
  if (!stream.good()) return 0;
  stream.seekg(0, std::ios::end);
  const std::streamoff size = stream.tellg();
  if (size <= 0) return 0;
  std::string text;
  text.resize((size_t)size);
  stream.seekg(0, std::ios::beg);
  stream.read(&text[0], size);
  if (!stream.good() && !stream.eof()) return 0;

  int count = 0;
  const char* cursor = text.c_str();
  while ((cursor = strstr(cursor, "B(")) != NULL) {
    char fen[160];
    char move[8];
    const char* afterFen = NULL;
    if (!readQuotedString(cursor, fen, sizeof(fen), &afterFen)) { cursor += 2; continue; }
    const char* comma = strchr(afterFen, ',');
    if (!comma || !readQuotedString(comma, move, sizeof(move), NULL)) { cursor = afterFen; continue; }
    const uint32_t hash = hashFenKey(fen);
    if (hash != 0 && strlen(move) >= 4) {
      int existing = -1;
      for (int i = 0; i < count; i++) {
        if (entries[i].hash == hash) { existing = i; break; }
      }
      const int idx = existing >= 0 ? existing : count;
      if (idx < maxEntries) {
        entries[idx].hash = hash;
        strncpy(entries[idx].move, move, sizeof(entries[idx].move) - 1);
        entries[idx].move[sizeof(entries[idx].move) - 1] = 0;
        if (existing < 0) count++;
      }
    }
    cursor = comma + 1;
  }
  return count;
}

static int loadTraincarBookFromBlob(const char* blobPath, TraincarBookEntry* entries, int maxEntries) {
  if (!blobPath || !*blobPath || !entries || maxEntries <= 0) return 0;
  std::ifstream stream(blobPath, std::ios::in | std::ios::binary);
  if (!stream.good()) return 0;
  stream.seekg(0, std::ios::end);
  const std::streamoff size = stream.tellg();
  if (size <= 0) return 0;
  std::string text;
  text.resize((size_t)size);
  stream.seekg(0, std::ios::beg);
  stream.read(&text[0], size);
  if (!stream.good() && !stream.eof()) return 0;

  const char* section = strstr(text.c_str(), "\"openingBook\"");
  if (!section) return 0;
  const char* cursor = strchr(section, '[');
  if (!cursor) return 0;
  int count = 0;
  while ((cursor = strstr(cursor, "\"hash\"")) != NULL && count < maxEntries) {
    const char* hashColon = strchr(cursor, ':');
    if (!hashColon) break;
    char* hashEnd = NULL;
    unsigned long hash = strtoul(hashColon + 1, &hashEnd, 10);
    if (hashEnd == hashColon + 1) { cursor = hashColon + 1; continue; }
    const char* moveKey = strstr(hashEnd, "\"move\"");
    if (!moveKey) break;
    const char* moveColon = strchr(moveKey, ':');
    char move[8];
    if (!moveColon || !readQuotedString(moveColon, move, sizeof(move), NULL)) {
      cursor = moveKey + 1;
      continue;
    }
    entries[count].hash = (uint32_t)hash;
    strncpy(entries[count].move, move, sizeof(entries[count].move) - 1);
    entries[count].move[sizeof(entries[count].move) - 1] = 0;
    count++;
    cursor = moveKey + 1;
  }
  return count;
}

static bool lookupTraincarBookMove(
  TraincarBookEntry* entries,
  int count,
  const char* fen,
  char* out,
  size_t outSize
) {
  if (!entries || count <= 0 || !fen || !out || outSize == 0) return false;
  const uint32_t hash = hashFenKey(fen);
  if (hash == 0) return false;
  for (int i = 0; i < count; i++) {
    if (entries[i].hash == hash) {
      strncpy(out, entries[i].move, outSize - 1);
      out[outSize - 1] = 0;
      return true;
    }
  }
  return false;
}

struct FFNPolicy {
  int loaded;
  int featureDim;
  float w0[64 * 20];
  float b0[64];
  float w1[32 * 64];
  float b1[32];
  float w2[32];
  float b2;
};

static bool parseNamedFloatArray(const char* root, const char* key, float* out, int count) {
  if (!root || !key || !out || count <= 0) return false;
  char needle[128];
  snprintf(needle, sizeof(needle), "\"%s\"", key);
  const char* section = strstr(root, needle);
  if (!section) return false;
  const char* open = strchr(section, '[');
  if (!open) return false;
  return parseFloatArray(open + 1, out, count) == count;
}

static bool loadFFNPolicy(const char* path, FFNPolicy* policy) {
  if (!path || !*path || !policy) return false;
  memset(policy, 0, sizeof(*policy));
  std::ifstream stream(path, std::ios::in | std::ios::binary);
  if (!stream.good()) return false;
  stream.seekg(0, std::ios::end);
  const std::streamoff size = stream.tellg();
  if (size <= 0) return false;
  std::string text;
  text.resize((size_t)size);
  stream.seekg(0, std::ios::beg);
  stream.read(&text[0], size);
  if (!stream.good() && !stream.eof()) return false;
  const char* root = text.c_str();
  if (!parseNamedFloatArray(root, "w0", policy->w0, 64 * 20)) return false;
  if (!parseNamedFloatArray(root, "b0", policy->b0, 64)) return false;
  if (!parseNamedFloatArray(root, "w1", policy->w1, 32 * 64)) return false;
  if (!parseNamedFloatArray(root, "b1", policy->b1, 32)) return false;
  if (!parseNamedFloatArray(root, "w2", policy->w2, 32)) return false;
  float b2[1] = {0.0f};
  if (!parseNamedFloatArray(root, "b2", b2, 1)) return false;
  policy->b2 = b2[0];
  policy->featureDim = 20;
  policy->loaded = 1;
  return true;
}

static float hostSilu(float x) {
  return x / (1.0f + expf(-x));
}

static float hostPieceFeature(int piece) {
  if (!piece) return 0.0f;
  float value = (float)(piece & 7);
  return (piece & BLACK_FLAG) ? -value : value;
}

static int hostFighterId(const char* blobPath) {
  if (!blobPath) return 5;
  if (strstr(blobPath, "razor_x")) return 0;
  if (strstr(blobPath, "queensguard")) return 1;
  if (strstr(blobPath, "firebird")) return 2;
  if (strstr(blobPath, "fortress")) return 3;
  if (strstr(blobPath, "razorblade_ii")) return 4;
  return 5;
}

static int hostRootRank(Move* moves, float* scores, int count, int idx) {
  bool used[MAX_MOVES] = {false};
  for (int rank = 1; rank <= count; rank++) {
    int best = -1;
    float bestScore = -1.0e30f;
    for (int i = 0; i < count; i++) {
      if (!used[i] && scores[i] > bestScore) {
        bestScore = scores[i];
        best = i;
      }
    }
    if (best < 0) break;
    if (best == idx) return rank;
    used[best] = true;
  }
  return count;
}

static void buildFFNFeatures(
  const Board* board,
  Move* moves,
  float* scores,
  int count,
  int idx,
  int bestIdx,
  int fighterId,
  float* out
) {
  memset(out, 0, 20 * sizeof(float));
  const Move* move = &moves[idx];
  const float score = scores[idx] / 100.0f;
  const float bestScore = scores[bestIdx] / 100.0f;
  const int moving = board->pieces[move->from];
  const int captured = hostMoveCapturedPiece(board, move);
  const int fromRank = move->from / 8;
  const int fromFile = move->from % 8;
  const int toRank = move->to / 8;
  const int toFile = move->to % 8;

  out[0] = score / 20.0f;
  out[1] = (score - bestScore) / 20.0f;
  out[2] = (float)hostRootRank(moves, scores, count, idx) / 64.0f;
  out[3] = (float)idx / 64.0f;
  out[4] = (float)count / 128.0f;
  out[5] = hostPieceFeature(moving) / 6.0f;
  out[6] = hostPieceFeature(captured) / 6.0f;
  out[7] = (float)fromRank / 7.0f;
  out[8] = (float)fromFile / 7.0f;
  out[9] = (float)toRank / 7.0f;
  out[10] = (float)toFile / 7.0f;
  out[11] = board->side == 0 ? 1.0f : -1.0f;
  out[12] = move->promo ? 1.0f : 0.0f;
  out[13] = idx == bestIdx ? 1.0f : 0.0f;
  int family = fighterId;
  if (family < 0 || family > 5) family = 5;
  out[14 + family] = 1.0f;
}

static float evalFFNPolicy(const FFNPolicy* policy, const float* features) {
  float h0[64];
  float h1[32];
  for (int o = 0; o < 64; o++) {
    float sum = policy->b0[o];
    for (int i = 0; i < 20; i++) sum += policy->w0[o * 20 + i] * features[i];
    h0[o] = hostSilu(sum);
  }
  for (int o = 0; o < 32; o++) {
    float sum = policy->b1[o];
    for (int i = 0; i < 64; i++) sum += policy->w1[o * 64 + i] * h0[i];
    h1[o] = hostSilu(sum);
  }
  float out = policy->b2;
  for (int i = 0; i < 32; i++) out += policy->w2[i] * h1[i];
  return out;
}

static int chooseFFNPolicyMove(
  const FFNPolicy* policy,
  const Board* board,
  Move* moves,
  float* scores,
  int count,
  int currentBestIdx,
  int fighterId,
  int topN
) {
  if (!policy || !policy->loaded || currentBestIdx < 0) return currentBestIdx;
  int bestIdx = currentBestIdx;
  float bestLogit = -1.0e30f;
  float features[20];
  bool used[MAX_MOVES] = {false};
  int limit = topN < count ? topN : count;
  for (int rank = 0; rank < limit; rank++) {
    int i = -1;
    float bestScore = -1.0e30f;
    for (int j = 0; j < count; j++) {
      if (!used[j] && scores[j] > bestScore) {
        bestScore = scores[j];
        i = j;
      }
    }
    if (i < 0 || scores[i] <= -99998.0f) break;
    used[i] = true;
    buildFFNFeatures(board, moves, scores, count, i, currentBestIdx, fighterId, features);
    float logit = evalFFNPolicy(policy, features);
    if (logit > bestLogit) {
      bestLogit = logit;
      bestIdx = i;
    }
  }
  return bestIdx;
}

// ============================================================================
// HOST: Parse FEN — now computes incremental PeSTO mg/eg scores
// ============================================================================

// Host-side PST copies for parseFen (constant memory is device-only)
static const int H_MG_PIECE_VAL[7] = {0, 82, 337, 365, 477, 1025, 0};
static const int H_EG_PIECE_VAL[7] = {0, 94, 281, 297, 512, 936, 0};
static const int H_PHASE_WEIGHTS[7] = {0, 0, 1, 1, 2, 4, 0};
static const int H_MG_PST[7][64] = {
  {0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,
   0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0},
  {0,0,0,0,0,0,0,0,98,134,61,95,68,126,34,-11,-6,7,26,31,65,56,25,-20,
   -14,13,6,21,23,12,17,-23,-27,-2,-5,12,17,6,10,-25,-26,-4,-4,-10,3,3,33,-12,
   -35,-1,-20,-23,-15,24,38,-22,0,0,0,0,0,0,0,0},
  {-167,-89,-34,-49,61,-97,50,-73,-73,-41,72,36,23,62,7,-17,-47,60,37,65,84,129,73,44,
   -9,17,19,53,37,69,18,22,-13,4,16,13,28,19,21,-8,-23,-9,12,10,19,17,25,-16,
   -29,-53,-12,-3,-1,18,-14,-19,-105,-21,-58,-33,-17,-28,-19,-23},
  {-29,4,-82,-37,-25,-42,7,-8,-26,16,-18,-13,30,59,18,-47,-16,37,43,40,35,50,37,-2,
   -4,5,19,50,37,37,7,-2,-6,13,13,26,34,12,10,4,0,15,15,15,14,27,18,10,
   4,15,16,0,7,21,33,1,-33,-3,-14,-21,-13,-12,-39,-21},
  {32,42,32,51,63,9,31,43,27,32,58,62,80,67,26,44,-5,19,26,36,17,45,61,16,
   -24,-11,7,26,24,35,-8,-20,-36,-26,-12,-1,9,-7,6,-23,-45,-25,-16,-17,3,0,-5,-33,
   -44,-16,-20,-9,-1,11,-6,-71,-19,-13,1,17,16,7,-37,-26},
  {-28,0,29,12,59,44,43,45,-24,-39,-5,1,-16,57,28,54,-13,-17,7,8,29,56,47,57,
   -27,-27,-16,-16,-1,17,-2,1,-9,-26,-9,-10,-2,-4,3,-3,-14,-2,-11,-2,-5,2,14,5,
   -35,-8,11,2,8,15,-3,1,-1,-18,-9,10,-15,-25,-31,-50},
  {-65,23,16,-15,-56,-34,2,13,29,-1,-20,-7,-8,-4,-38,-29,-9,24,2,-16,-20,6,22,-22,
   -17,-20,-12,-27,-30,-25,-14,-36,-49,-1,-27,-39,-46,-44,-33,-51,-14,-14,-22,-46,-44,-30,-15,-27,
   1,7,-8,-64,-43,-16,9,8,-15,36,12,-54,8,-28,24,14}
};
static const int H_EG_PST[7][64] = {
  {0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,
   0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0},
  {0,0,0,0,0,0,0,0,178,173,158,134,147,132,165,187,94,100,85,67,56,53,82,84,
   32,24,13,5,-2,4,17,17,13,9,-3,-7,-7,-8,3,-1,4,7,-6,1,0,-5,-1,-8,
   13,8,8,10,13,0,2,-7,0,0,0,0,0,0,0,0},
  {-58,-38,-13,-28,-31,-27,-63,-99,-25,-8,-25,-2,-9,-25,-24,-52,-24,-20,10,9,-1,-9,-19,-41,
   -17,3,22,22,22,11,8,-18,-18,-6,16,25,16,17,4,-18,-23,-3,-1,15,10,-3,-20,-22,
   -42,-20,-10,-5,-2,-20,-23,-44,-29,-51,-23,-15,-22,-18,-50,-64},
  {-14,-21,-11,-8,-7,-9,-17,-24,-8,-4,7,-12,-3,-13,-4,-14,2,-8,0,-1,-2,6,0,4,
   -3,9,12,9,14,10,3,2,-6,3,13,19,7,10,-3,-9,-12,-3,8,10,13,3,-7,-15,
   -14,-18,-7,-1,4,-9,-15,-27,-23,-9,-23,-5,-9,-16,-5,-17},
  {13,10,18,15,12,12,8,5,11,13,13,11,-3,7,7,8,7,7,7,5,4,-3,-5,3,
   4,3,13,1,2,1,-1,2,3,5,8,4,-5,-6,-8,-11,-4,0,-5,-1,-7,-12,-8,-16,
   -6,-6,0,2,-9,-9,-11,-3,-9,2,3,-1,-5,-13,4,-20},
  {-9,22,22,27,27,19,10,20,-17,20,32,41,58,25,30,0,-20,6,9,49,47,35,19,9,
   3,22,24,45,57,40,57,36,-18,28,19,47,31,34,39,23,-16,-27,15,6,9,17,10,5,
   -22,-23,-30,-16,-16,-23,-36,-32,-33,-28,-22,-43,-5,-32,-20,-41},
  {-74,-35,-18,-18,-11,15,4,-17,-12,17,14,17,17,38,23,11,10,17,23,15,20,45,44,13,
   -8,22,24,27,26,33,26,3,-18,-4,21,24,27,23,9,-11,-19,-3,11,21,23,16,7,-9,
   -27,-11,4,13,14,4,-5,-17,-53,-34,-21,-11,-28,-14,-24,-43}
};

void parseFen(const char* fen, Board* b) {
  memset(b, 0, sizeof(Board));
  b->epSq = -1;
  b->mgScore = 0;
  b->egScore = 0;
  int sq = 0;
  const char* p = fen;
  while (*p && *p != ' ') {
    if (*p == '/') { p++; continue; }
    if (*p >= '1' && *p <= '8') { sq += *p - '0'; p++; continue; }
    int piece = 0, color = 0;
    switch (*p) {
      case 'P': piece=PAWN; break; case 'N': piece=KNIGHT; break;
      case 'B': piece=BISHOP; break; case 'R': piece=ROOK; break;
      case 'Q': piece=QUEEN; break; case 'K': piece=KING; break;
      case 'p': piece=PAWN; color=BLACK_FLAG; break; case 'n': piece=KNIGHT; color=BLACK_FLAG; break;
      case 'b': piece=BISHOP; color=BLACK_FLAG; break; case 'r': piece=ROOK; color=BLACK_FLAG; break;
      case 'q': piece=QUEEN; color=BLACK_FLAG; break; case 'k': piece=KING; color=BLACK_FLAG; break;
    }
    if (piece) {
      b->pieces[sq] = piece | color;
      int isW = (color == 0);
      int pstIdx = isW ? sq : ((7 - sq/8)*8 + sq%8); // mirror for black
      float mg = (float)(H_MG_PIECE_VAL[piece] + H_MG_PST[piece][pstIdx]);
      float eg = (float)(H_EG_PIECE_VAL[piece] + H_EG_PST[piece][pstIdx]);
      if (isW) { b->mgScore += mg; b->egScore += eg; }
      else     { b->mgScore -= mg; b->egScore -= eg; }
      b->phase += H_PHASE_WEIGHTS[piece];
    }
    sq++; p++;
  }
  if (*p == ' ') {
    p++;
    b->side = (*p == 'b') ? 1 : 0;
    while (*p && *p != ' ') p++;
  }
  if (*p == ' ') {
    p++;
    b->castling = 0;
    if (*p == '-') {
      p++;
    } else {
      while (*p && *p != ' ') {
        if (*p == 'K') b->castling |= 1;
        else if (*p == 'Q') b->castling |= 2;
        else if (*p == 'k') b->castling |= 4;
        else if (*p == 'q') b->castling |= 8;
        p++;
      }
    }
  }
  if (*p == ' ') {
    p++;
    b->epSq = (*p == '-') ? -1 : parseUciSquare(p);
  }
  if (b->phase > 24) b->phase = 24;
}

// ============================================================================
// DEVICE: Move generation (pseudo-legal, shared by MCTS and search)
// ============================================================================

__device__ int inBounds(int r, int c) { return r >= 0 && r < 8 && c >= 0 && c < 8; }

__device__ int findKingSq(Board* b, int side) {
  int flag = side ? BLACK_FLAG : WHITE_FLAG;
  for (int sq = 0; sq < 64; sq++) {
    int pc = b->pieces[sq];
    if (pc && (pc & 7) == KING && (pc & 8) == flag) return sq;
  }
  return -1;
}

__device__ int isSquareAttacked(Board* b, int sq, int bySide) {
  int byFlag = bySide ? BLACK_FLAG : WHITE_FLAG;
  int tr = sq / 8;
  int tc = sq % 8;

  int pawnRow = bySide == 0 ? tr + 1 : tr - 1;
  for (int dc = -1; dc <= 1; dc += 2) {
    int nc = tc + dc;
    if (inBounds(pawnRow, nc)) {
      int pc = b->pieces[pawnRow * 8 + nc];
      if (pc && (pc & 8) == byFlag && (pc & 7) == PAWN) return 1;
    }
  }

  int kd[8][2] = {{-2,-1},{-2,1},{-1,-2},{-1,2},{1,-2},{1,2},{2,-1},{2,1}};
  for (int d = 0; d < 8; d++) {
    int nr = tr + kd[d][0], nc = tc + kd[d][1];
    if (inBounds(nr, nc)) {
      int pc = b->pieces[nr * 8 + nc];
      if (pc && (pc & 8) == byFlag && (pc & 7) == KNIGHT) return 1;
    }
  }

  for (int dr = -1; dr <= 1; dr++) for (int dc = -1; dc <= 1; dc++) {
    if (!dr && !dc) continue;
    int nr = tr + dr, nc = tc + dc;
    if (inBounds(nr, nc)) {
      int pc = b->pieces[nr * 8 + nc];
      if (pc && (pc & 8) == byFlag && (pc & 7) == KING) return 1;
    }
  }

  int dirs[8][2] = {{-1,-1},{-1,0},{-1,1},{0,-1},{0,1},{1,-1},{1,0},{1,1}};
  for (int d = 0; d < 8; d++) {
    int nr = tr + dirs[d][0], nc = tc + dirs[d][1];
    int isDiag = (dirs[d][0] != 0 && dirs[d][1] != 0);
    while (inBounds(nr, nc)) {
      int pc = b->pieces[nr * 8 + nc];
      if (pc) {
        if ((pc & 8) == byFlag) {
          int type = pc & 7;
          if ((isDiag && (type == BISHOP || type == QUEEN)) ||
              (!isDiag && (type == ROOK || type == QUEEN))) return 1;
        }
        break;
      }
      nr += dirs[d][0];
      nc += dirs[d][1];
    }
  }

  return 0;
}

__device__ int isInCheck(Board* b, int side) {
  int kingSq = findKingSq(b, side);
  if (kingSq < 0) return 1;
  return isSquareAttacked(b, kingSq, 1 - side);
}

__device__ int moveIsEpCapture(Board* b, Move* m) {
  int moving = b->pieces[m->from];
  return moving && (moving & 7) == PAWN && m->to == b->epSq && b->pieces[m->to] == EMPTY;
}

__device__ int moveCapturedPiece(Board* b, Move* m) {
  int captured = b->pieces[m->to];
  if (!captured && moveIsEpCapture(b, m)) {
    int movingIsW = (b->pieces[m->from] & 8) == 0;
    int capSq = movingIsW ? m->to + 8 : m->to - 8;
    if (capSq >= 0 && capSq < 64) captured = b->pieces[capSq];
  }
  return captured;
}

__device__ void addPawnMove(Move* moves, int* count, int from, int to, int promote) {
  if (*count >= MAX_MOVES) return;
  if (promote) {
    int promos[4] = {QUEEN, ROOK, BISHOP, KNIGHT};
    for (int i = 0; i < 4 && *count < MAX_MOVES; i++) {
      moves[(*count)++] = {from, to, promos[i]};
    }
  } else {
    moves[(*count)++] = {from, to, 0};
  }
}

__device__ int generateMoves(Board* b, Move* moves) {
  int count = 0;
  int side = b->side;
  int friendly = side ? BLACK_FLAG : WHITE_FLAG;
  int enemy = side ? WHITE_FLAG : BLACK_FLAG;

  for (int sq = 0; sq < 64 && count < MAX_MOVES - 4; sq++) {
    int pc = b->pieces[sq];
    if (!pc || (pc & 8) != friendly) continue;
    int type = pc & 7;
    int r = sq / 8, c = sq % 8;

    if (type == PAWN) {
      int dir = (friendly == WHITE_FLAG) ? -1 : 1;
      int startRank = (friendly == WHITE_FLAG) ? 6 : 1;
      int nr = r + dir;
      int promoRank = (friendly == WHITE_FLAG) ? 0 : 7;
      if (inBounds(nr, c) && !b->pieces[nr*8+c]) {
        addPawnMove(moves, &count, sq, nr*8+c, nr == promoRank);
        if (r == startRank) {
          int nr2 = r + dir*2;
          if (!b->pieces[nr2*8+c]) moves[count++] = {sq, nr2*8+c, 0};
        }
      }
      for (int dc = -1; dc <= 1; dc += 2) {
        int nc = c + dc;
        if (inBounds(nr, nc)) {
          int target = b->pieces[nr*8+nc];
          if (target && (target & 8) == (unsigned)enemy) {
            addPawnMove(moves, &count, sq, nr*8+nc, nr == promoRank);
          } else if (nr*8+nc == b->epSq) {
            moves[count++] = {sq, nr*8+nc, 0};
          }
        }
      }
    } else if (type == KNIGHT) {
      int kd[][2] = {{-2,-1},{-2,1},{-1,-2},{-1,2},{1,-2},{1,2},{2,-1},{2,1}};
      for (int d = 0; d < 8; d++) {
        int nr = r+kd[d][0], nc = c+kd[d][1];
        if (inBounds(nr,nc)) {
          int t = b->pieces[nr*8+nc];
          if (!t || (t&8)==(unsigned)enemy) moves[count++] = {sq, nr*8+nc, 0};
        }
      }
    } else if (type == KING) {
      for (int dr=-1; dr<=1; dr++) for (int dc=-1; dc<=1; dc++) {
        if (!dr && !dc) continue;
        int nr=r+dr, nc=c+dc;
        if (inBounds(nr,nc)) {
          int t = b->pieces[nr*8+nc];
          if (!t || (t&8)==(unsigned)enemy) moves[count++] = {sq, nr*8+nc, 0};
        }
      }
      if (!isSquareAttacked(b, sq, 1 - side)) {
        if (side == 0 && sq == 60) {
          if ((b->castling & 1) && !b->pieces[61] && !b->pieces[62] &&
              b->pieces[63] && (b->pieces[63] & 7) == ROOK && (b->pieces[63] & 8) == WHITE_FLAG &&
              !isSquareAttacked(b, 61, 1) && !isSquareAttacked(b, 62, 1)) {
            moves[count++] = {60, 62, 0};
          }
          if ((b->castling & 2) && !b->pieces[59] && !b->pieces[58] && !b->pieces[57] &&
              b->pieces[56] && (b->pieces[56] & 7) == ROOK && (b->pieces[56] & 8) == WHITE_FLAG &&
              !isSquareAttacked(b, 59, 1) && !isSquareAttacked(b, 58, 1)) {
            moves[count++] = {60, 58, 0};
          }
        } else if (side == 1 && sq == 4) {
          if ((b->castling & 4) && !b->pieces[5] && !b->pieces[6] &&
              b->pieces[7] && (b->pieces[7] & 7) == ROOK && (b->pieces[7] & 8) == BLACK_FLAG &&
              !isSquareAttacked(b, 5, 0) && !isSquareAttacked(b, 6, 0)) {
            moves[count++] = {4, 6, 0};
          }
          if ((b->castling & 8) && !b->pieces[3] && !b->pieces[2] && !b->pieces[1] &&
              b->pieces[0] && (b->pieces[0] & 7) == ROOK && (b->pieces[0] & 8) == BLACK_FLAG &&
              !isSquareAttacked(b, 3, 0) && !isSquareAttacked(b, 2, 0)) {
            moves[count++] = {4, 2, 0};
          }
        }
      }
    } else { // Bishop, Rook, Queen — sliding pieces
      int dirs[8][2] = {{-1,-1},{-1,0},{-1,1},{0,-1},{0,1},{1,-1},{1,0},{1,1}};
      for (int d = 0; d < 8; d++) {
        int isDiag = (dirs[d][0] != 0 && dirs[d][1] != 0);
        if (type == BISHOP && !isDiag) continue;
        if (type == ROOK && isDiag) continue;
        int nr = r+dirs[d][0], nc = c+dirs[d][1];
        while (inBounds(nr,nc)) {
          int t = b->pieces[nr*8+nc];
          if (!t) { moves[count++] = {sq, nr*8+nc, 0}; }
          else { if ((t&8)==(unsigned)enemy) moves[count++] = {sq, nr*8+nc, 0}; break; }
          nr += dirs[d][0]; nc += dirs[d][1];
        }
      }
    }
  }
  return count;
}

// ============================================================================
// DEVICE: Make move with incremental PeSTO score updates
// Handles auto-queening for pawn promotions.
// ============================================================================

__device__ void makeMove(Board* b, Move* m) {
  int moving = b->pieces[m->from];
  int captured = b->pieces[m->to];
  int movingType = moving & 7;
  int movingIsW = (moving & 8) == 0;
  int isEp = movingType == PAWN && m->to == b->epSq && captured == EMPTY;
  int epCapSq = movingIsW ? m->to + 8 : m->to - 8;
  if (isEp && epCapSq >= 0 && epCapSq < 64) captured = b->pieces[epCapSq];

  // Remove captured piece's PST contribution
  if (captured) {
    int capType = captured & 7;
    int capIsW = (captured & 8) == 0;
    int capSq = isEp ? epCapSq : m->to;
    int pstIdx = capIsW ? capSq : C_MIRROR[capSq];
    float cmg = (float)(C_MG_PIECE_VAL[capType] + C_MG_PST[capType][pstIdx]);
    float ceg = (float)(C_EG_PIECE_VAL[capType] + C_EG_PST[capType][pstIdx]);
    if (capIsW) { b->mgScore -= cmg; b->egScore -= ceg; }
    else        { b->mgScore += cmg; b->egScore += ceg; }
    b->phase -= C_PHASE_WEIGHTS[capType];
    if (b->phase < 0) b->phase = 0;
    if (isEp && epCapSq >= 0 && epCapSq < 64) b->pieces[epCapSq] = EMPTY;
  }

  // Determine destination piece type (auto-queen on promotion)
  int destType = movingType;
  if (movingType == PAWN && m->promo >= KNIGHT && m->promo <= QUEEN) {
    destType = m->promo;
  } else if (movingType == PAWN) {
    int destRank = m->to / 8;
    if ((movingIsW && destRank == 0) || (!movingIsW && destRank == 7))
      destType = QUEEN;
  }

  // Update PST scores: remove from source, add at destination
  int fromPst = movingIsW ? m->from : C_MIRROR[m->from];
  int toPst   = movingIsW ? m->to   : C_MIRROR[m->to];
  float fromMg = (float)(C_MG_PIECE_VAL[movingType] + C_MG_PST[movingType][fromPst]);
  float fromEg = (float)(C_EG_PIECE_VAL[movingType] + C_EG_PST[movingType][fromPst]);
  float toMg   = (float)(C_MG_PIECE_VAL[destType]   + C_MG_PST[destType][toPst]);
  float toEg   = (float)(C_EG_PIECE_VAL[destType]   + C_EG_PST[destType][toPst]);
  if (movingIsW) { b->mgScore += toMg - fromMg; b->egScore += toEg - fromEg; }
  else           { b->mgScore -= toMg - fromMg; b->egScore -= toEg - fromEg; }

  // Update phase for promotion
  if (destType != movingType) {
    b->phase += C_PHASE_WEIGHTS[destType] - C_PHASE_WEIGHTS[movingType];
    if (b->phase > 24) b->phase = 24;
    b->pieces[m->to] = (movingIsW ? 0 : BLACK_FLAG) | destType;
  } else {
    b->pieces[m->to] = moving;
  }
  b->pieces[m->from] = EMPTY;

  if (movingType == KING && (m->to - m->from == 2 || m->from - m->to == 2)) {
    int rookFrom = -1, rookTo = -1;
    if (m->from == 60 && m->to == 62) { rookFrom = 63; rookTo = 61; }
    else if (m->from == 60 && m->to == 58) { rookFrom = 56; rookTo = 59; }
    else if (m->from == 4 && m->to == 6) { rookFrom = 7; rookTo = 5; }
    else if (m->from == 4 && m->to == 2) { rookFrom = 0; rookTo = 3; }
    if (rookFrom >= 0) {
      int rook = b->pieces[rookFrom];
      int rookIsW = (rook & 8) == 0;
      int fromIdx = rookIsW ? rookFrom : C_MIRROR[rookFrom];
      int toIdx = rookIsW ? rookTo : C_MIRROR[rookTo];
      float fromMgR = (float)(C_MG_PIECE_VAL[ROOK] + C_MG_PST[ROOK][fromIdx]);
      float fromEgR = (float)(C_EG_PIECE_VAL[ROOK] + C_EG_PST[ROOK][fromIdx]);
      float toMgR = (float)(C_MG_PIECE_VAL[ROOK] + C_MG_PST[ROOK][toIdx]);
      float toEgR = (float)(C_EG_PIECE_VAL[ROOK] + C_EG_PST[ROOK][toIdx]);
      if (rookIsW) { b->mgScore += toMgR - fromMgR; b->egScore += toEgR - fromEgR; }
      else         { b->mgScore -= toMgR - fromMgR; b->egScore -= toEgR - fromEgR; }
      b->pieces[rookTo] = rook;
      b->pieces[rookFrom] = EMPTY;
    }
  }

  if (movingType == KING) {
    if (movingIsW) b->castling &= ~3;
    else b->castling &= ~12;
  }
  if (m->from == 63 || m->to == 63) b->castling &= ~1;
  if (m->from == 56 || m->to == 56) b->castling &= ~2;
  if (m->from == 7 || m->to == 7) b->castling &= ~4;
  if (m->from == 0 || m->to == 0) b->castling &= ~8;
  b->epSq = -1;
  if (movingType == PAWN && (m->to - m->from == 16 || m->from - m->to == 16))
    b->epSq = (m->from + m->to) / 2;

  b->side = 1 - b->side;
}

// ============================================================================
// DEVICE: MCTS playout (kept for knob tuning backward compatibility)
// ============================================================================

__device__ float playout(Board startBoard, curandState* rng) {
  Board b = startBoard;
  Move moves[MAX_MOVES];
  for (int ply = 0; ply < 200; ply++) {
    int nMoves = generateMoves(&b, moves);
    if (nMoves == 0) return b.side == startBoard.side ? 0.0f : 1.0f;
    int idx = curand(rng) % nMoves;
    makeMove(&b, &moves[idx]);
  }
  // Terminal material eval
  int mat = 0;
  int vals[] = {0, 100, 320, 330, 500, 900, 0};
  for (int i = 0; i < 64; i++) {
    int pc = b.pieces[i];
    if (!pc) continue;
    mat += (pc & 8) ? -vals[pc & 7] : vals[pc & 7];
  }
  float score = 0.5f + mat / 5000.0f;
  if (score > 1.0f) score = 1.0f;
  if (score < 0.0f) score = 0.0f;
  return startBoard.side == 0 ? score : 1.0f - score;
}

__global__ void mctsKernel(
  Board* d_board, Move* d_moves, int numMoves,
  float* d_winRates, int* d_visits, int simsPerMove, unsigned long long seed
) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int totalThreads = simsPerMove * numMoves;
  if (tid >= totalThreads) return;
  int moveIdx = tid / simsPerMove;
  if (moveIdx >= numMoves) return;
  curandState rng;
  curand_init(seed, tid, 0, &rng);
  Board b = *d_board;
  makeMove(&b, &d_moves[moveIdx]);
  float result = playout(b, &rng);
  atomicAdd(&d_winRates[moveIdx], result);
  atomicAdd(&d_visits[moveIdx], 1);
}

// ============================================================================
// DEVICE: Full evaluation — PeSTO base + personality brain adjustments
// Combines incremental PeSTO mg/eg scores with knob-weighted brain features
// (pawn structure, king safety, rook placement, bishop pair, queen guard).
// Returns score relative to side to move (positive = good for side to move).
// ============================================================================

__device__ float evalPosition(Board* b) {
  int ph = b->phase;
  if (ph > 24) ph = 24;
  if (ph < 0) ph = 0;
  int egPh = 24 - ph;

  // Read layer weights from fighter knobs
  float pawnW = C_FIGHTER_KNOBS[0], kingW = C_FIGHTER_KNOBS[1];
  float queenW = C_FIGHTER_KNOBS[2], rookW = C_FIGHTER_KNOBS[3];
  float minorW = C_FIGHTER_KNOBS[4], tempoW = C_FIGHTER_KNOBS[5];
  float agg = C_FIGHTER_KNOBS[24];

  // Aggression adjustment on layer weights
  tempoW += (agg - 8.0f) * 0.5f;
  kingW  -= (agg - 8.0f) * 0.25f;
  pawnW  -= (agg - 8.0f) * 0.25f;

  // === Board scan: pawn files, piece counts, king squares ===
  int wBish = 0, bBish = 0, wQueens = 0, bQueens = 0;
  int wPF[8]={0,0,0,0,0,0,0,0}, bPF[8]={0,0,0,0,0,0,0,0};
  int wPR[8], bPR[8];
  for (int i=0;i<8;i++){wPR[i]=7;bPR[i]=0;}
  int wKSq = -1, bKSq = -1;
  int pawnCount = 0, openFiles = 0;

  for (int sq = 0; sq < 64; sq++) {
    int pc = b->pieces[sq];
    if (!pc) continue;
    int type = pc & 7;
    int isW = (pc & 8) == 0;
    int f = sq & 7, r = sq / 8;
    if (type==PAWN) {
      pawnCount++;
      if (isW) { wPF[f]++; if (r < wPR[f]) wPR[f] = r; }
      else     { bPF[f]++; if (r > bPR[f]) bPR[f] = r; }
    }
    if (type==BISHOP) { if (isW) wBish++; else bBish++; }
    if (type==QUEEN)  { if (isW) wQueens++; else bQueens++; }
    if (type==KING)   { if (isW) wKSq=sq; else bKSq=sq; }
  }
  for (int f=0;f<8;f++) if (wPF[f]==0 && bPF[f]==0) openFiles++;

  // === Position classification adjustments ===
  int isOpening = ph >= 20;
  int isEndgame = ph < 8;
  int isOpen = openFiles >= 4;
  int isClosed = openFiles <= 2 && pawnCount >= 10;
  int queensOn = wQueens > 0 && bQueens > 0;

  float posAdj[6] = {0,0,0,0,0,0};
  if (isOpening)      { float a[]={-2,4,2,-4,0,2}; for(int i=0;i<6;i++) posAdj[i]+=a[i]; }
  else if (isEndgame) { float a[]={4,-4,-4,4,0,2}; for(int i=0;i<6;i++) posAdj[i]+=a[i]; }
  else if (isOpen)    { float a[]={-2,0,0,4,2,0};  for(int i=0;i<6;i++) posAdj[i]+=a[i]; }
  else if (isClosed)  { float a[]={4,0,0,-3,0,0};  for(int i=0;i<6;i++) posAdj[i]+=a[i]; }
  if (queensOn) { float a[]={0,4,4,0,0,0}; for(int i=0;i<6;i++) posAdj[i]+=a[i]; }
  else          { float a[]={0,-4,-8,2,0,0}; for(int i=0;i<6;i++) posAdj[i]+=a[i]; }

  pawnW += posAdj[0]; kingW += posAdj[1]; queenW += posAdj[2];
  rookW += posAdj[3]; minorW += posAdj[4]; tempoW += posAdj[5];

  pawnW  = fmaxf(0.0f,fminf(16.0f,pawnW));
  kingW  = fmaxf(0.0f,fminf(16.0f,kingW));
  queenW = fmaxf(0.0f,fminf(16.0f,queenW));
  rookW  = fmaxf(0.0f,fminf(16.0f,rookW));
  minorW = fmaxf(0.0f,fminf(16.0f,minorW));
  tempoW = fmaxf(0.0f,fminf(16.0f,tempoW));

  // === Pawn structure brain ===
  float pawnMG = 0, pawnEG = 0;
  const float ppMG[] = {0,0,10,17,30,55,85,0};
  const float ppEG[] = {0,0,20,35,55,90,140,0};
  for (int f = 0; f < 8; f++) {
    if (wPF[f] > 0) {
      if (wPF[f] > 1) { pawnMG -= C_FIGHTER_KNOBS[29]; pawnEG -= C_FIGHTER_KNOBS[30]; }
      int hasN = (f>0 && wPF[f-1]>0) || (f<7 && wPF[f+1]>0);
      if (!hasN) { pawnMG -= C_FIGHTER_KNOBS[31]; pawnEG -= C_FIGHTER_KNOBS[32]; }
      int passed = 1;
      for (int af=(f>0?f-1:0); af<=(f<7?f+1:f) && passed; af++)
        if (bPF[af]>0 && bPR[af] > wPR[f]) passed = 0;
      if (passed) { int ri=7-wPR[f]; if(ri>=0&&ri<8){pawnMG+=ppMG[ri];pawnEG+=ppEG[ri];}}
      if (f>0 && wPF[f-1]>0) { int d=wPR[f]-wPR[f-1]; if(d>=-1&&d<=1){pawnMG+=C_FIGHTER_KNOBS[33];pawnEG+=C_FIGHTER_KNOBS[34];}}
    }
    if (bPF[f] > 0) {
      if (bPF[f] > 1) { pawnMG += C_FIGHTER_KNOBS[29]; pawnEG += C_FIGHTER_KNOBS[30]; }
      int hasN = (f>0 && bPF[f-1]>0) || (f<7 && bPF[f+1]>0);
      if (!hasN) { pawnMG += C_FIGHTER_KNOBS[31]; pawnEG += C_FIGHTER_KNOBS[32]; }
      int passed = 1;
      for (int af=(f>0?f-1:0); af<=(f<7?f+1:f) && passed; af++)
        if (wPF[af]>0 && wPR[af] < bPR[f]) passed = 0;
      if (passed) { int ri=bPR[f]; if(ri>=0&&ri<8){pawnMG-=ppMG[ri];pawnEG-=ppEG[ri];}}
      if (f>0 && bPF[f-1]>0) { int d=bPR[f]-bPR[f-1]; if(d>=-1&&d<=1){pawnMG-=C_FIGHTER_KNOBS[33];pawnEG-=C_FIGHTER_KNOBS[34];}}
    }
  }

  // === King shield brain ===
  float kingScore = 0;
  if (kingW > 0) {
    for (int side = 0; side < 2; side++) {
      int kSq = side==0 ? wKSq : bKSq;
      if (kSq < 0) continue;
      int kr = kSq/8, kf = kSq%8;
      int pdir = side==0 ? -1 : 1;
      float shield = 0;
      int attackCount = 0;
      for (int dc=-1; dc<=1; dc++) {
        int sf = kf+dc;
        if (sf<0||sf>7) continue;
        int sr = kr+pdir;
        if (sr>=0 && sr<8) {
          int pp = b->pieces[sr*8+sf];
          if (pp && (pp&7)==PAWN && ((pp&8)==0)==(side==0)) shield += C_FIGHTER_KNOBS[35];
          else shield -= C_FIGHTER_KNOBS[36];
        }
      }
      for (int sq=0; sq<64; sq++) {
        int pc = b->pieces[sq];
        if (!pc) continue;
        int isEnemy = ((pc&8)==0) != (side==0);
        if (!isEnemy || (pc&7)==PAWN || (pc&7)==KING) continue;
        int pr=sq/8, pf2=sq%8;
        int dr=pr>kr?pr-kr:kr-pr, df=pf2>kf?pf2-kf:kf-pf2;
        if (dr<=2 && df<=2) attackCount++;
      }
      float safetyMuls[] = {0,8,21,34,45,50};
      int ac = attackCount > 5 ? 5 : attackCount;
      float penalty = safetyMuls[ac] * (float)attackCount / 4.0f;
      float total = shield - penalty;
      kingScore += side==0 ? total : -total;
    }
  }

  // === Rook brain ===
  float rookScore = 0;
  if (rookW > 0) {
    for (int sq=0; sq<64; sq++) {
      int pc = b->pieces[sq];
      if (!pc || (pc&7)!=ROOK) continue;
      int isW = (pc&8)==0;
      int f = sq&7, r = sq/8;
      float sign = isW ? 1.0f : -1.0f;
      if (wPF[f]==0 && bPF[f]==0) rookScore += sign * C_FIGHTER_KNOBS[37];
      else if (isW && wPF[f]==0) rookScore += C_FIGHTER_KNOBS[38];
      else if (!isW && bPF[f]==0) rookScore -= C_FIGHTER_KNOBS[38];
      if (isW && r==1) rookScore += C_FIGHTER_KNOBS[39];
      if (!isW && r==6) rookScore -= C_FIGHTER_KNOBS[39];
    }
  }

  // === Minor brain (bishop pair) ===
  float minorScore = 0;
  if (wBish>=2) minorScore += C_FIGHTER_KNOBS[40];
  if (bBish>=2) minorScore -= C_FIGHTER_KNOBS[40];

  // === Queen guard ===
  float queenScore = 0;
  if (queenW > 0 && ph > 6) {
    for (int sq=0; sq<64; sq++) {
      int pc = b->pieces[sq];
      if (!pc || (pc&7)!=QUEEN) continue;
      int isW = (pc&8)==0;
      float sign = isW ? -1.0f : 1.0f;
      int qr=sq/8, qf=sq%8;
      int exposed = 0, mobility = 0;
      for (int dr=-1; dr<=1; dr++) for (int dc=-1; dc<=1; dc++) {
        if (!dr && !dc) continue;
        int nr=qr+dr, nc=qf+dc;
        if (nr>=0&&nr<8&&nc>=0&&nc<8) {
          int t = b->pieces[nr*8+nc];
          if (!t || ((t&8)==0)!=isW) mobility++;
        }
      }
      int knightDirs[][2] = {{-2,-1},{-2,1},{-1,-2},{-1,2},{1,-2},{1,2},{2,-1},{2,1}};
      for (int d=0; d<8; d++) {
        int nr=qr+knightDirs[d][0], nc=qf+knightDirs[d][1];
        if (nr>=0&&nr<8&&nc>=0&&nc<8) {
          int t = b->pieces[nr*8+nc];
          if (t && (t&7)==KNIGHT && ((t&8)==0)!=isW) exposed = 1;
        }
      }
      if (exposed) queenScore += sign * C_FIGHTER_KNOBS[41];
      if (mobility <= 2) queenScore += sign * C_FIGHTER_KNOBS[42];
    }
  }

  // === Combine: PeSTO base + weighted brains ===
  float totalMG = b->mgScore + pawnMG * pawnW / 8.0f;
  float totalEG = b->egScore + pawnEG * pawnW / 8.0f;
  float score = (totalMG * (float)ph + totalEG * (float)egPh) / 24.0f;

  if (kingW > 0)  score += kingScore * kingW / 8.0f;
  if (queenW > 0) score += queenScore * queenW / 8.0f;
  if (rookW > 0)  score += rookScore * rookW / 8.0f;
  if (minorW > 0) score += minorScore * minorW / 8.0f;
  score += C_FIGHTER_KNOBS[43] * tempoW / 8.0f;

  return b->side == 0 ? score : -score;
}

// ============================================================================
// DEVICE: Fast PeSTO-only evaluation (kept for future throughput modes).
// Accuracy mode uses evalPosition in qsearch to match CPU fighter semantics.
// ============================================================================

__device__ float evalFast(Board* b) {
  int ph = b->phase;
  if (ph > 24) ph = 24;
  if (ph < 0) ph = 0;
  float tapered = (b->mgScore * (float)ph + b->egScore * (float)(24 - ph)) / 24.0f;
  return b->side == 0 ? tapered : -tapered;
}

__device__ int traincarDepthBand(int depth) {
  if (depth <= 1) return 0;
  if (depth <= 3) return 1;
  if (depth <= 5) return 2;
  if (depth <= 8) return 3;
  return 4;
}

__device__ float evalTraincarBridge(int ply) {
  float evalBridgePly = C_FLAT_KNOBS[42] > 0.0f ? C_FLAT_KNOBS[42] : 4.0f;
  if ((float)ply > evalBridgePly) return 0.0f;
  int band = traincarDepthBand(ply);
  float divisor = C_FLAT_KNOBS[43] > 0.0f ? C_FLAT_KNOBS[43] : 1.0f;
  float aggression = C_FLAT_KNOBS[5 + band] > 0.0f ? C_FLAT_KNOBS[5 + band] : 1.0f;
  float temp = C_FLAT_KNOBS[band];
  return roundf((temp / divisor) * aggression);
}

__device__ float evalCpuTraincar(Board* b, int ply) {
  float score = evalFast(b);
  float bridge = evalTraincarBridge(ply);
  return score + (b->side == 0 ? bridge : -bridge);
}

// ============================================================================
// DEVICE: MVV-LVA move ordering — captures first, sorted by victim value
// ============================================================================

__device__ void orderMoves(Board* b, Move* moves, int count) {
  // Score each move: captures get high scores (MVV-LVA), non-captures get 0
  // Then insertion sort (stable, good for nearly-sorted)
  int scores[MAX_MOVES];
  for (int i = 0; i < count; i++) {
    int captured = b->pieces[moves[i].to];
    if (captured) {
      int attacker = b->pieces[moves[i].from] & 7;
      scores[i] = 1000000 + C_MVV_VAL[captured & 7] * 100 - C_MVV_VAL[attacker];
    } else {
      // Bonus for central destination squares
      int tr = moves[i].to / 8, tc = moves[i].to % 8;
      int centerDist = (tr > 3 ? tr - 3 : 3 - tr) + (tc > 3 ? tc - 3 : 3 - tc);
      scores[i] = 100 - centerDist * 10;
    }
  }
  for (int i = 1; i < count; i++) {
    Move mv = moves[i]; int sc = scores[i];
    int j = i - 1;
    while (j >= 0 && scores[j] < sc) {
      moves[j+1] = moves[j]; scores[j+1] = scores[j]; j--;
    }
    moves[j+1] = mv; scores[j+1] = sc;
  }
}

// ============================================================================
// DEVICE: Quiescence search — only captures, prevents horizon effect
// ============================================================================

__device__ float quiesce(Board* b, float alpha, float beta, int depth, int ply) {
  float standPat = C_TRAINCAR_EVAL ? evalCpuTraincar(b, ply) : (C_FULL_QEVAL ? evalPosition(b) : evalFast(b));
  if (depth <= 0) return standPat;
  if (standPat >= beta) return beta;
  if (standPat > alpha) alpha = standPat;

  // Delta pruning threshold: if we're way below alpha, skip
  // (queen value = 900, so if standPat + 1000 < alpha, no capture helps)
  if (standPat + 1100.0f < alpha) return alpha;

  Move moves[MAX_MOVES];
  int nMoves = generateMoves(b, moves);
  orderMoves(b, moves, nMoves);

  for (int i = 0; i < nMoves; i++) {
    int captured = moveCapturedPiece(b, &moves[i]);
    if (!captured) continue; // only search captures

    // Delta pruning per move
    int capVal = C_MVV_VAL[captured & 7];
    if (standPat + (float)capVal + 200.0f < alpha) continue;

    Board child = *b;
    makeMove(&child, &moves[i]);
    if (C_FILTER_LEGAL && isInCheck(&child, 1 - child.side)) continue;
    float score = -quiesce(&child, -beta, -alpha, depth - 1, ply + 1);
    if (score >= beta) return beta;
    if (score > alpha) alpha = score;
  }
  return alpha;
}

// ============================================================================
// DEVICE: Negamax with alpha-beta pruning
// ============================================================================

__device__ float searchEval(Board* b, int ply) {
  return C_TRAINCAR_EVAL ? evalCpuTraincar(b, ply) : (C_FULL_QEVAL ? evalPosition(b) : evalFast(b));
}

__device__ int futilityMargin(int depth) {
  if (depth <= 0) return 0;
  if (depth == 1) return 180;
  if (depth == 2) return 400;
  return 700;
}

__device__ int lmpThreshold(int depth) {
  if (depth <= 0) return 0;
  if (depth == 1) return 5;
  if (depth == 2) return 9;
  if (depth == 3) return 14;
  return 22;
}

__device__ int razorMargin(int depth) {
  if (depth <= 0) return 0;
  if (depth == 1) return 300;
  return 550;
}

__device__ float traincarPruningMult(int depth) {
  int band = traincarDepthBand(depth);
  float mult = C_FLAT_KNOBS[10 + band];
  return mult > 0.0f ? mult : 1.0f;
}

__device__ int traincarSearchBreadth(int depth) {
  int band = traincarDepthBand(depth);
  float temp = C_FLAT_KNOBS[band];
  return temp > 0.0f ? (int)floorf(temp) : 0;
}

__device__ int traincarRazorTuning(int depth) {
  float step = C_FLAT_KNOBS[36];
  if (step <= 0.0f) return 0;
  return (int)roundf((float)traincarSearchBreadth(depth) * step * traincarPruningMult(depth));
}

__device__ int traincarFutilityTuning(int depth) {
  float step = C_FLAT_KNOBS[37];
  if (step <= 0.0f) return 0;
  return (int)roundf((float)traincarSearchBreadth(depth) * step * traincarPruningMult(depth));
}

__device__ int traincarLmpTuning(int depth) {
  float div = C_FLAT_KNOBS[38];
  if (div <= 0.0f) return 0;
  return (int)floorf(((float)traincarSearchBreadth(depth) / div) * traincarPruningMult(depth));
}

__device__ int traincarLmrTuning(int depth) {
  float div = C_FLAT_KNOBS[39];
  if (div <= 0.0f) return 0;
  return (int)floorf(((float)traincarSearchBreadth(depth) / div) * traincarPruningMult(depth));
}

__device__ int moveIsPromotion(Board* b, Move* m) {
  if (m->promo >= KNIGHT && m->promo <= QUEEN) return 1;
  int moving = b->pieces[m->from];
  if (!moving || (moving & 7) != PAWN) return 0;
  int destRank = m->to / 8;
  int movingIsW = (moving & 8) == 0;
  return (movingIsW && destRank == 0) || (!movingIsW && destRank == 7);
}

__device__ float negamax(Board* b, int depth, float alpha, float beta, int ply, int doNull) {
  int inCheck = C_CPU_SHAPED_SEARCH ? isInCheck(b, b->side) : 0;
  if (C_CPU_SHAPED_SEARCH && inCheck) depth++;
  if (depth <= 0) return quiesce(b, alpha, beta, MAX_QUIESCE_DEPTH, ply);

  Move moves[MAX_MOVES];
  int nMoves = generateMoves(b, moves);
  if (nMoves == 0) {
    return -90000.0f + (float)(DEFAULT_SEARCH_DEPTH - depth);
  }

  int isPV = (beta - alpha) > 1.0f;
  float staticEval = -99999.0f;
  int razorTuning = 0;
  int futilityTuning = 0;
  int lmpTuning = 0;
  int lmrTuning = 0;

  if (C_CPU_SHAPED_SEARCH) {
    staticEval = inCheck ? -99999.0f : searchEval(b, ply);
    razorTuning = traincarRazorTuning(depth);
    futilityTuning = traincarFutilityTuning(depth);
    lmpTuning = traincarLmpTuning(depth);
    lmrTuning = traincarLmrTuning(depth);

    if (!isPV && !inCheck && depth <= 2 && staticEval + (float)(razorMargin(depth) + razorTuning) <= alpha) {
      float razorScore = quiesce(b, alpha, beta, MAX_QUIESCE_DEPTH, ply);
      if (razorScore <= alpha) return razorScore;
    }

    if (!isPV && !inCheck && depth <= 3 && doNull &&
        staticEval - (float)(futilityMargin(depth) + futilityTuning) >= beta) {
      return staticEval;
    }

    if (doNull && !inCheck && depth >= 3 && b->phase > 2 && staticEval >= beta) {
      Board nullBoard = *b;
      nullBoard.side = 1 - nullBoard.side;
      nullBoard.epSq = -1;
      int R = 3 + (int)fminf(floorf((staticEval - beta) / 200.0f), 2.0f) + (depth > 6 ? 1 : 0);
      float nullScore = -negamax(&nullBoard, depth - R - 1, -beta, -beta + 1.0f, ply + 1, 0);
      if (nullScore >= beta) return beta;
    }
  }

  orderMoves(b, moves, nMoves);
  float bestScore = -99999.0f;
  int movesSearched = 0;

  for (int i = 0; i < nMoves; i++) {
    int isCapture = moveCapturedPiece(b, &moves[i]) != 0;
    int isPromo = moveIsPromotion(b, &moves[i]);
    Board child = *b;
    makeMove(&child, &moves[i]);
    if (C_FILTER_LEGAL && isInCheck(&child, 1 - child.side)) continue;
    movesSearched++;

    int givesCheck = C_CPU_SHAPED_SEARCH ? isInCheck(&child, child.side) : 0;
    if (C_CPU_SHAPED_SEARCH && !isPV && !inCheck && !givesCheck && !isCapture && !isPromo &&
        depth <= 4 && movesSearched > (lmpThreshold(depth) + lmpTuning)) {
      continue;
    }

    if (C_CPU_SHAPED_SEARCH && !isPV && !inCheck && !givesCheck && !isCapture && !isPromo &&
        depth <= 3 && movesSearched > 1 &&
        staticEval + (float)(futilityMargin(depth) + futilityTuning) <= alpha) {
      continue;
    }

    float score;
    if (C_CPU_SHAPED_SEARCH && movesSearched >= 4 && depth >= 3 && !inCheck && !isCapture && !isPromo && !givesCheck) {
      int R = (int)floorf(0.5f + logf((float)depth) * logf((float)movesSearched) / 2.0f);
      if (!isPV) R++;
      if (R > depth - 2) R = depth - 2;
      R -= lmrTuning;
      if (R < 1) R = 1;
      score = -negamax(&child, depth - 1 - R, -alpha - 1.0f, -alpha, ply + 1, 1);
      if (score > alpha) {
        score = -negamax(&child, depth - 1, -alpha - 1.0f, -alpha, ply + 1, 1);
      }
    } else if (C_CPU_SHAPED_SEARCH && movesSearched > 1) {
      score = -negamax(&child, depth - 1, -alpha - 1.0f, -alpha, ply + 1, 1);
    } else {
      score = alpha + 1.0f;
    }

    if (!C_CPU_SHAPED_SEARCH || score > alpha) {
      score = -negamax(&child, depth - 1, -beta, -alpha, ply + 1, 1);
    }

    if (score > bestScore) bestScore = score;
    if (score > alpha) alpha = score;
    if (alpha >= beta) break; // beta cutoff
  }
  if (movesSearched == 0) {
    return isInCheck(b, b->side) ? -90000.0f + (float)(DEFAULT_SEARCH_DEPTH - depth) : 0.0f;
  }
  return bestScore;
}

// ============================================================================
// KERNEL: Alpha-beta search — one thread per root move (root parallelism)
// Each thread applies its assigned move and searches the resulting position.
// ============================================================================

__global__ void searchKernel(
  Board* d_board,
  Move* d_moves,
  int numMoves,
  float* d_scores,
  int searchDepth
) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= numMoves) return;

  Board child = *d_board;
  makeMove(&child, &d_moves[tid]);
  if (C_FILTER_LEGAL && isInCheck(&child, 1 - child.side)) {
    d_scores[tid] = -99999.0f;
    return;
  }
  // Search from opponent's perspective, negate for our score
  d_scores[tid] = -negamax(&child, searchDepth - 1, -99999.0f, 99999.0f, 1, 1);
}

__global__ void serialRootSearchKernel(
  Board* d_board,
  Move* d_moves,
  int numMoves,
  float* d_scores,
  int searchDepth
) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;

  float alpha = -99999.0f;
  const float beta = 99999.0f;
  int movesSearched = 0;

  for (int i = 0; i < numMoves; i++) {
    Board child = *d_board;
    makeMove(&child, &d_moves[i]);
    if (C_FILTER_LEGAL && isInCheck(&child, 1 - child.side)) {
      d_scores[i] = -99999.0f;
      continue;
    }

    float score;
    if (movesSearched > 0) {
      score = -negamax(&child, searchDepth - 1, -alpha - 1.0f, -alpha, 1, 1);
    } else {
      score = alpha + 1.0f;
    }
    if (score > alpha) {
      score = -negamax(&child, searchDepth - 1, -beta, -alpha, 1, 1);
    }
    d_scores[i] = score;
    movesSearched++;
    if (score > alpha) alpha = score;
  }
}

// ============================================================================
// KERNEL: evalWithKnobs — static eval with personality knobs (for knob tuner)
// Kept from v1 for the disagreement analysis / knob tuning step.
// ============================================================================

__global__ void evalWithKnobs(
  Board* d_board, KnobConfig* d_configs, float* d_scores, int numConfigs
) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= numConfigs) return;

  Board board = *d_board;
  float* k = d_configs[tid].knobs;

  // Material (PeSTO base values, no PST for knob eval — matches v1 behavior)
  float mg = 0, eg = 0;
  int wBish = 0, bBish = 0, wQueens = 0, bQueens = 0;
  int wPF[8]={}, bPF[8]={};
  int wPR[8], bPR[8];
  for (int i=0;i<8;i++){wPR[i]=7;bPR[i]=0;}
  int wKSq = -1, bKSq = -1;
  int pawnCount = 0, openFiles = 0;

  int mgPV[7] = {0, 82, 337, 365, 477, 1025, 0};
  int egPV[7] = {0, 94, 281, 297, 512, 936, 0};

  for (int sq = 0; sq < 64; sq++) {
    int pc = board.pieces[sq];
    if (!pc) continue;
    int type = pc & 7;
    int isW = (pc & 8) == 0;
    mg += isW ? mgPV[type] : -mgPV[type];
    eg += isW ? egPV[type] : -egPV[type];
    int f = sq & 7, r = sq / 8;
    if (type==PAWN) {
      pawnCount++;
      if (isW) { wPF[f]++; if (r < wPR[f]) wPR[f] = r; }
      else     { bPF[f]++; if (r > bPR[f]) bPR[f] = r; }
    }
    if (type==BISHOP) { if (isW) wBish++; else bBish++; }
    if (type==QUEEN)  { if (isW) wQueens++; else bQueens++; }
    if (type==KING)   { if (isW) wKSq=sq; else bKSq=sq; }
  }
  for (int f=0;f<8;f++) if (wPF[f]==0 && bPF[f]==0) openFiles++;

  int phase = board.phase > 24 ? 24 : board.phase;
  int egPh = 24 - phase;

  // Pawn brain
  float pawnMG = 0, pawnEG = 0;
  const float ppMG[] = {0,0,10,17,30,55,85,0};
  const float ppEG[] = {0,0,20,35,55,90,140,0};
  for (int f = 0; f < 8; f++) {
    if (wPF[f] > 0) {
      if (wPF[f] > 1) { pawnMG -= k[29]; pawnEG -= k[30]; }
      int hasN = (f>0 && wPF[f-1]>0) || (f<7 && wPF[f+1]>0);
      if (!hasN) { pawnMG -= k[31]; pawnEG -= k[32]; }
      int passed = 1;
      for (int af = (f>0?f-1:0); af <= (f<7?f+1:f) && passed; af++)
        if (bPF[af]>0 && bPR[af] > wPR[f]) passed = 0;
      if (passed) { int ri = 7 - wPR[f]; if (ri>=0&&ri<8) { pawnMG += ppMG[ri]; pawnEG += ppEG[ri]; } }
      if (f>0 && wPF[f-1]>0) { int d = wPR[f]-wPR[f-1]; if (d>=-1&&d<=1) { pawnMG += k[33]; pawnEG += k[34]; } }
    }
    if (bPF[f] > 0) {
      if (bPF[f] > 1) { pawnMG += k[29]; pawnEG += k[30]; }
      int hasN = (f>0 && bPF[f-1]>0) || (f<7 && bPF[f+1]>0);
      if (!hasN) { pawnMG += k[31]; pawnEG += k[32]; }
      int passed = 1;
      for (int af = (f>0?f-1:0); af <= (f<7?f+1:f) && passed; af++)
        if (wPF[af]>0 && wPR[af] < bPR[f]) passed = 0;
      if (passed) { int ri = bPR[f]; if (ri>=0&&ri<8) { pawnMG -= ppMG[ri]; pawnEG -= ppEG[ri]; } }
      if (f>0 && bPF[f-1]>0) { int d = bPR[f]-bPR[f-1]; if (d>=-1&&d<=1) { pawnMG -= k[33]; pawnEG -= k[34]; } }
    }
  }

  // King shield brain
  float kingScore = 0;
  for (int side = 0; side < 2; side++) {
    int kSq = side==0 ? wKSq : bKSq;
    if (kSq < 0) continue;
    int kr = kSq/8, kf = kSq%8;
    int pdir = side==0 ? -1 : 1;
    float shield = 0;
    int attackCount = 0;
    for (int dc=-1; dc<=1; dc++) {
      int sf = kf+dc;
      if (sf<0||sf>7) continue;
      int sr = kr+pdir;
      if (sr>=0 && sr<8) {
        int pp = board.pieces[sr*8+sf];
        if (pp && (pp&7)==PAWN && ((pp&8)==0)==(side==0)) shield += k[35];
        else shield -= k[36];
      }
    }
    for (int sq=0; sq<64; sq++) {
      int pc = board.pieces[sq];
      if (!pc) continue;
      int isEnemy = ((pc&8)==0) != (side==0);
      if (!isEnemy || (pc&7)==PAWN || (pc&7)==KING) continue;
      int pr=sq/8, pf2=sq%8;
      int dr=pr>kr?pr-kr:kr-pr, df=pf2>kf?pf2-kf:kf-pf2;
      if (dr<=2 && df<=2) attackCount++;
    }
    float safetyMuls[] = {0,8,21,34,45,50};
    int ac = attackCount > 5 ? 5 : attackCount;
    float penalty = safetyMuls[ac] * attackCount / 4.0f;
    float total = shield - penalty;
    kingScore += side==0 ? total : -total;
  }

  // Rook brain
  float rookScore = 0;
  for (int sq=0; sq<64; sq++) {
    int pc = board.pieces[sq];
    if (!pc || (pc&7)!=ROOK) continue;
    int isW = (pc&8)==0;
    int f = sq&7, r = sq/8;
    float sign = isW ? 1.0f : -1.0f;
    if (wPF[f]==0 && bPF[f]==0) rookScore += sign * k[37];
    else if (isW && wPF[f]==0) rookScore += k[38];
    else if (!isW && bPF[f]==0) rookScore -= k[38];
    if (isW && r==1) rookScore += k[39];
    if (!isW && r==6) rookScore -= k[39];
  }

  // Minor brain (bishop pair)
  float minorScore = 0;
  if (wBish>=2) minorScore += k[40];
  if (bBish>=2) minorScore -= k[40];

  // Queen guard
  float queenScore = 0;
  if (phase > 6) {
    for (int sq=0; sq<64; sq++) {
      int pc = board.pieces[sq];
      if (!pc || (pc&7)!=QUEEN) continue;
      int isW = (pc&8)==0;
      float sign = isW ? -1.0f : 1.0f;
      int qr=sq/8, qf=sq%8;
      int exposed = 0, mobility = 0;
      for (int dr=-1; dr<=1; dr++) for (int dc=-1; dc<=1; dc++) {
        if (!dr && !dc) continue;
        int nr=qr+dr, nc=qf+dc;
        if (nr>=0&&nr<8&&nc>=0&&nc<8) {
          int t = board.pieces[nr*8+nc];
          if (!t || ((t&8)==0)!=isW) mobility++;
        }
      }
      int knightDirs[][2] = {{-2,-1},{-2,1},{-1,-2},{-1,2},{1,-2},{1,2},{2,-1},{2,1}};
      for (int d=0; d<8; d++) {
        int nr=qr+knightDirs[d][0], nc=qf+knightDirs[d][1];
        if (nr>=0&&nr<8&&nc>=0&&nc<8) {
          int t = board.pieces[nr*8+nc];
          if (t && (t&7)==KNIGHT && ((t&8)==0)!=isW) exposed = 1;
        }
      }
      if (exposed) queenScore += sign * k[41];
      if (mobility <= 2) queenScore += sign * k[42];
    }
  }

  // Position classification
  int isOpening = phase >= 20;
  int isEndgame = phase < 8;
  int isOpen = openFiles >= 4;
  int isClosed = openFiles <= 2 && pawnCount >= 10;
  int queensOn = wQueens > 0 && bQueens > 0;

  // Layer weight selection
  float pawnW = k[0], kingW = k[1], queenW = k[2], rookW = k[3], minorW = k[4], tempoW = k[5];
  float agg = k[24];
  tempoW += (agg - 8) * 0.5f;
  kingW  -= (agg - 8) * 0.25f;
  pawnW  -= (agg - 8) * 0.25f;

  float posAdj[6] = {0,0,0,0,0,0};
  if (isOpening)      { float a[]={-2,4,2,-4,0,2}; for(int i=0;i<6;i++) posAdj[i]+=a[i]; }
  else if (isEndgame) { float a[]={4,-4,-4,4,0,2}; for(int i=0;i<6;i++) posAdj[i]+=a[i]; }
  else if (isOpen)    { float a[]={-2,0,0,4,2,0};  for(int i=0;i<6;i++) posAdj[i]+=a[i]; }
  else if (isClosed)  { float a[]={4,0,0,-3,0,0};  for(int i=0;i<6;i++) posAdj[i]+=a[i]; }
  if (queensOn) { float a[]={0,4,4,0,0,0}; for(int i=0;i<6;i++) posAdj[i]+=a[i]; }
  else          { float a[]={0,-4,-8,2,0,0}; for(int i=0;i<6;i++) posAdj[i]+=a[i]; }

  pawnW += posAdj[0]; kingW += posAdj[1]; queenW += posAdj[2];
  rookW += posAdj[3]; minorW += posAdj[4]; tempoW += posAdj[5];

  pawnW  = fmaxf(0,fminf(16,pawnW));
  kingW  = fmaxf(0,fminf(16,kingW));
  queenW = fmaxf(0,fminf(16,queenW));
  rookW  = fmaxf(0,fminf(16,rookW));
  minorW = fmaxf(0,fminf(16,minorW));
  tempoW = fmaxf(0,fminf(16,tempoW));

  // Combine
  float totalMG = mg + pawnMG * pawnW / 8.0f;
  float totalEG = eg + pawnEG * pawnW / 8.0f;
  float score = (totalMG * phase + totalEG * egPh) / 24.0f;
  if (kingW > 0)  score += kingScore * kingW / 8.0f;
  if (queenW > 0) score += queenScore * queenW / 8.0f;
  if (rookW > 0)  score += rookScore * rookW / 8.0f;
  if (minorW > 0) score += minorScore * minorW / 8.0f;
  score += k[43] * tempoW / 8.0f;

  d_scores[tid] = board.side == 0 ? score : -score;
}

__global__ void genConfigs(KnobConfig* d_configs, float* d_defaults, int n, unsigned long long seed) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) return;
  curandState rng;
  curand_init(seed, tid, 0, &rng);
  for (int k = 0; k < KNOB_COUNT; k++) {
    float base = d_defaults[k];
    float range = base * 0.4f + 3.0f;
    d_configs[tid].knobs[k] = fmaxf(0.0f, base + (curand_uniform(&rng)*2.0f-1.0f)*range);
  }
}

// ============================================================================
// HOST: Main — process batch of positions
// ============================================================================

int main(int argc, char** argv) {
  int numConfigs = 4096;
  int searchDepth = DEFAULT_SEARCH_DEPTH;
  int fullQeval = 0;
  int filterLegal = 0;
  int traincarEval = 0;
  int cpuShapedSearch = 0;
  int serialRoot = 0;
  int rootOrder = 0;
  int traincarRootTieBreak = 0;
  int familyDispatch = 0;
  int timeoutRootProxy = 0;
  int traincarBook = 0;
  int emitAllPositions = 0;
  int ffnPolicyEnabled = 0;
  const char* ffnPolicyPath = NULL;
  const char* traincarBookPath = "frostd4d/variants/the_un.js";
  int positional = 0;
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--fighter-blob") == 0) { i++; continue; }
    if (strcmp(argv[i], "--traincar-book-path") == 0 && i + 1 < argc) {
      traincarBookPath = argv[++i];
      continue;
    }
    if (strcmp(argv[i], "--ffn-policy") == 0 && i + 1 < argc) {
      ffnPolicyPath = argv[++i];
      ffnPolicyEnabled = 1;
      continue;
    }
    if (strcmp(argv[i], "--depth") == 0 && i + 1 < argc) {
      searchDepth = atoi(argv[++i]);
      if (searchDepth < 1) searchDepth = 1;
      if (searchDepth > 12) searchDepth = 12;
      continue;
    }
    if (strcmp(argv[i], "--full-qeval") == 0) {
      fullQeval = 1;
      continue;
    }
    if (strcmp(argv[i], "--filter-legal") == 0) {
      filterLegal = 1;
      continue;
    }
    if (strcmp(argv[i], "--traincar-eval") == 0) {
      traincarEval = 1;
      continue;
    }
    if (strcmp(argv[i], "--cpu-shaped-search") == 0) {
      cpuShapedSearch = 1;
      continue;
    }
    if (strcmp(argv[i], "--serial-root") == 0) {
      serialRoot = 1;
      continue;
    }
    if (strcmp(argv[i], "--root-order") == 0) {
      rootOrder = 1;
      continue;
    }
    if (strcmp(argv[i], "--traincar-root-tiebreak") == 0) {
      traincarRootTieBreak = 1;
      continue;
    }
    if (strcmp(argv[i], "--family-dispatch") == 0) {
      familyDispatch = 1;
      continue;
    }
    if (strcmp(argv[i], "--timeout-root-proxy") == 0) {
      timeoutRootProxy = 1;
      continue;
    }
    if (strcmp(argv[i], "--traincar-book") == 0) {
      traincarBook = 1;
      continue;
    }
    if (strcmp(argv[i], "--emit-all") == 0) {
      emitAllPositions = 1;
      continue;
    }
    if (argv[i][0] == '-') continue;
    if (positional == 0) numConfigs = atoi(argv[i]);
    positional++;
  }

  // Set CUDA stack size for recursive alpha-beta search
  // Each recursion level uses ~4KB (Board + Move array + locals)
  // Depth 5 + quiesce 6 = 11 levels × 4KB = 44KB needed
  cudaDeviceSetLimit(cudaLimitStackSize, 65536);

  float defaults[KNOB_COUNT] = DOJO_ACTIVE_FIGHTER_DEFAULTS44_INITIALIZER;
  float flatKnobs[64];
  memset(flatKnobs, 0, sizeof(flatKnobs));
  const char* fighterBlobPath = resolveFighterBlobPath(argc, argv);
  FighterFamily fighterFamily = detectFighterFamilyFromBlob(fighterBlobPath);
  if (fighterBlobPath && *fighterBlobPath) {
    if (!loadLegacyDefaultsFromBlob(fighterBlobPath, defaults, KNOB_COUNT)) {
      fprintf(stderr, "[dojo] warning: failed to load fighter blob %s, using compiled defaults\n", fighterBlobPath);
    } else {
      fprintf(stderr, "[dojo] loaded fighter blob %s\n", fighterBlobPath);
    }
    if (traincarEval || cpuShapedSearch) {
      if (loadFlatKnobsFromBlob(fighterBlobPath, flatKnobs, 64)) {
        fprintf(stderr, "[dojo] loaded flat fighter knobs %s\n", fighterBlobPath);
      } else {
        fprintf(stderr, "[dojo] warning: failed to load flat fighter knobs %s; traincar-dependent bridges disabled\n", fighterBlobPath);
        traincarEval = 0;
        cpuShapedSearch = 0;
      }
    }
  }
  if (familyDispatch && fighterFamily == FIGHTER_FAMILY_TRAINCAR) traincarBook = 1;
  if (familyDispatch && fighterFamily == FIGHTER_FAMILY_RAZORBLADE_II) timeoutRootProxy = 1;
  if (traincarBook && fighterFamily != FIGHTER_FAMILY_TRAINCAR) traincarBook = 0;
  FFNPolicy ffnPolicy;
  memset(&ffnPolicy, 0, sizeof(ffnPolicy));
  if (ffnPolicyEnabled) {
    if (!loadFFNPolicy(ffnPolicyPath, &ffnPolicy)) {
      fprintf(stderr, "[dojo] warning: failed to load ffn policy %s; disabled\n", ffnPolicyPath ? ffnPolicyPath : "");
      ffnPolicyEnabled = 0;
    }
  }
  TraincarBookEntry* traincarBookEntries = (TraincarBookEntry*)malloc(MAX_TRAINCAR_BOOK_ENTRIES * sizeof(TraincarBookEntry));
  int traincarBookCount = 0;
  if (traincarBook && fighterFamily == FIGHTER_FAMILY_TRAINCAR) {
    traincarBookCount = loadTraincarBookFromBlob(fighterBlobPath, traincarBookEntries, MAX_TRAINCAR_BOOK_ENTRIES);
    if (traincarBookCount <= 0) {
      traincarBookCount = loadTraincarBook(traincarBookPath, traincarBookEntries, MAX_TRAINCAR_BOOK_ENTRIES);
    }
    if (traincarBookCount <= 0) {
      fprintf(stderr, "[dojo] warning: failed to load traincar book %s; traincar book disabled\n", traincarBookPath);
      traincarBook = 0;
    }
  }
  fprintf(stderr, "[dojo] fighter family: %s\n", fighterFamilyName(fighterFamily));
  fprintf(stderr, "[dojo] search depth: %d\n", searchDepth);
  if (fullQeval) fprintf(stderr, "[dojo] full qsearch eval: enabled\n");
  if (filterLegal) fprintf(stderr, "[dojo] legal child filter: enabled\n");
  if (traincarEval) fprintf(stderr, "[dojo] traincar eval bridge: enabled\n");
  if (cpuShapedSearch) fprintf(stderr, "[dojo] cpu-shaped search: enabled\n");
  if (serialRoot) fprintf(stderr, "[dojo] serial root search: enabled\n");
  if (rootOrder) fprintf(stderr, "[dojo] root move ordering: enabled\n");
  if (traincarRootTieBreak) fprintf(stderr, "[dojo] traincar root tie-break: enabled\n");
  if (familyDispatch) fprintf(stderr, "[dojo] source-family dispatch: enabled\n");
  if (timeoutRootProxy) fprintf(stderr, "[dojo] timeout root proxy: enabled\n");
  if (traincarBook) fprintf(stderr, "[dojo] traincar book: enabled (%d entries)\n", traincarBookCount);
  if (emitAllPositions) fprintf(stderr, "[dojo] emit-all positions: enabled\n");
  if (ffnPolicyEnabled) fprintf(stderr, "[dojo] ffn policy residual: enabled (%s)\n", ffnPolicyPath);

  // Upload fighter knobs to constant memory for search evaluation
  cudaMemcpyToSymbol(C_FIGHTER_KNOBS, defaults, KNOB_COUNT * sizeof(float));
  cudaMemcpyToSymbol(C_FLAT_KNOBS, flatKnobs, 64 * sizeof(float));
  cudaMemcpyToSymbol(C_FULL_QEVAL, &fullQeval, sizeof(int));
  cudaMemcpyToSymbol(C_FILTER_LEGAL, &filterLegal, sizeof(int));
  cudaMemcpyToSymbol(C_TRAINCAR_EVAL, &traincarEval, sizeof(int));
  cudaMemcpyToSymbol(C_CPU_SHAPED_SEARCH, &cpuShapedSearch, sizeof(int));

  // Allocate GPU memory
  Board *d_board; cudaMalloc(&d_board, sizeof(Board));
  KnobConfig *d_configs; cudaMalloc(&d_configs, numConfigs * sizeof(KnobConfig));
  float *d_scoresA, *d_scoresB, *d_defaults;
  cudaMalloc(&d_scoresA, numConfigs * sizeof(float));
  cudaMalloc(&d_scoresB, numConfigs * sizeof(float));
  cudaMalloc(&d_defaults, KNOB_COUNT * sizeof(float));
  cudaMemcpy(d_defaults, defaults, KNOB_COUNT * sizeof(float), cudaMemcpyHostToDevice);

  // MCTS memory (kept for knob tuner fallback)
  Move *d_moves; cudaMalloc(&d_moves, MAX_MOVES * sizeof(Move));
  float *d_winRates; cudaMalloc(&d_winRates, MAX_MOVES * sizeof(float));
  int *d_visits; cudaMalloc(&d_visits, MAX_MOVES * sizeof(int));

  // Search memory
  float *d_searchScores; cudaMalloc(&d_searchScores, MAX_MOVES * sizeof(float));

  // Generate knob configs on GPU (for knob tuner step)
  int blocks = (numConfigs + BLOCK_SIZE - 1) / BLOCK_SIZE;
  genConfigs<<<blocks, BLOCK_SIZE>>>(d_configs, d_defaults, numConfigs, time(NULL));
  cudaDeviceSynchronize();

  // Host buffers
  float *h_scoresA = (float*)malloc(numConfigs * sizeof(float));
  float *h_scoresB = (float*)malloc(numConfigs * sizeof(float));
  float *h_searchScores = (float*)malloc(MAX_MOVES * sizeof(float));
  KnobConfig *h_configs = (KnobConfig*)malloc(numConfigs * sizeof(KnobConfig));
  cudaMemcpy(h_configs, d_configs, numConfigs * sizeof(KnobConfig), cudaMemcpyDeviceToHost);

  // Accumulate best knobs
  float bestKnobs[KNOB_COUNT];
  memcpy(bestKnobs, defaults, sizeof(defaults));
  int totalPos = 0, totalFixable = 0, totalUnfixable = 0;
  int inputLines = 0, parsedLines = 0, comparablePositions = 0, agreements = 0;
  int skippedShort = 0, skippedNoTab = 0, skippedNoMoves = 0, skippedEngineMoveMissing = 0, skippedNoLegalRootScore = 0;
  int traincarBookOverrides = 0, traincarBookMisses = 0;
  int ffnPolicyOverrides = 0;

  char line[512];
  struct timespec start, now;
  clock_gettime(CLOCK_MONOTONIC, &start);

  printf("{\"positions\":[\n");
  int first = 1;

  while (fgets(line, sizeof(line), stdin)) {
    inputLines++;
    line[strcspn(line, "\n")] = 0;
    if (strlen(line) < 10) { skippedShort++; continue; }

    // Parse: FEN<tab>engine_move[<tab>legal1,legal2,...]
    char* tab1 = strchr(line, '\t');
    if (!tab1) { skippedNoTab++; continue; }
    *tab1 = 0;
    char* fen = line;
    char* engineMove = tab1 + 1;
    char* tab2 = strchr(engineMove, '\t');
    char* legalMovesText = NULL;
    if (tab2) {
      *tab2 = 0;
      legalMovesText = tab2 + 1;
      legalMovesText = ltrim(legalMovesText);
      rtrimInPlace(legalMovesText);
      if (*legalMovesText == 0) legalMovesText = NULL;
    }
    engineMove = ltrim(engineMove);
    rtrimInPlace(engineMove);
    parsedLines++;

    Board board;
    parseFen(fen, &board);
    cudaMemcpy(d_board, &board, sizeof(Board), cudaMemcpyHostToDevice);

    // Build legal move list
    Move h_moves[MAX_MOVES];
    int numMoves = 0;
    if (legalMovesText) {
      numMoves = parseLegalMoveList(legalMovesText, h_moves, MAX_MOVES);
    }
    if (numMoves == 0) {
      // Generate pseudo-legal moves on host as fallback
      int side = board.side;
      int friendly = side ? BLACK_FLAG : WHITE_FLAG;
      for (int sq = 0; sq < 64 && numMoves < MAX_MOVES-4; sq++) {
        int pc = board.pieces[sq];
        if (!pc || (pc & 8) != friendly) continue;
        int type = pc & 7, r = sq/8, c = sq%8;
        if (type == PAWN) {
          int dir = (friendly==WHITE_FLAG) ? -1 : 1;
          int nr = r+dir;
          if (nr>=0 && nr<8 && !board.pieces[nr*8+c]) {
            h_moves[numMoves++] = {sq, nr*8+c, 0};
            int startRank = (friendly==WHITE_FLAG) ? 6 : 1;
            if (r == startRank) {
              int nr2 = r+dir*2;
              if (!board.pieces[nr2*8+c]) h_moves[numMoves++] = {sq, nr2*8+c, 0};
            }
          }
          for (int dc=-1; dc<=1; dc+=2) {
            int nc=c+dc;
            if (nr>=0&&nr<8&&nc>=0&&nc<8) {
              int t=board.pieces[nr*8+nc];
              if (t&&(t&8)!=friendly) h_moves[numMoves++]={sq,nr*8+nc,0};
            }
          }
        } else if (type == KNIGHT) {
          int kd[][2]={{-2,-1},{-2,1},{-1,-2},{-1,2},{1,-2},{1,2},{2,-1},{2,1}};
          for (int d=0;d<8;d++){int nr=r+kd[d][0],nc=c+kd[d][1];if(nr>=0&&nr<8&&nc>=0&&nc<8){int t=board.pieces[nr*8+nc];if(!t||(t&8)!=friendly)h_moves[numMoves++]={sq,nr*8+nc,0};}}
        } else if (type == KING) {
          for(int dr=-1;dr<=1;dr++)for(int dc=-1;dc<=1;dc++){if(!dr&&!dc)continue;int nr=r+dr,nc=c+dc;if(nr>=0&&nr<8&&nc>=0&&nc<8){int t=board.pieces[nr*8+nc];if(!t||(t&8)!=friendly)h_moves[numMoves++]={sq,nr*8+nc,0};}}
        } else {
          int dirs[8][2]={{-1,-1},{-1,0},{-1,1},{0,-1},{0,1},{1,-1},{1,0},{1,1}};
          for(int d=0;d<8;d++){
            int isDiag=(dirs[d][0]!=0&&dirs[d][1]!=0);
            if(type==BISHOP&&!isDiag)continue;
            if(type==ROOK&&isDiag)continue;
            int nr=r+dirs[d][0],nc=c+dirs[d][1];
            while(nr>=0&&nr<8&&nc>=0&&nc<8){int t=board.pieces[nr*8+nc];if(!t)h_moves[numMoves++]={sq,nr*8+nc,0};else{if((t&8)!=friendly)h_moves[numMoves++]={sq,nr*8+nc,0};break;}nr+=dirs[d][0];nc+=dirs[d][1];}
          }
        }
      }
    }

    if (numMoves == 0) { skippedNoMoves++; continue; }
    if (rootOrder) orderRootMovesHost(&board, h_moves, numMoves);

    // Find engine move index
    int engIdx = -1;
    for (int i = 0; i < numMoves; i++) {
      char mv[8];
      moveToUci(&h_moves[i], mv, sizeof(mv));
      if (strcmp(mv, engineMove) == 0) { engIdx = i; break; }
    }
    if (engIdx < 0) { skippedEngineMoveMissing++; continue; }

    // ================================================================
    // STEP 1: GPU Alpha-Beta Search — find best move
    // One thread per legal move, each searches its subtree
    // ================================================================
    cudaMemcpy(d_moves, h_moves, numMoves * sizeof(Move), cudaMemcpyHostToDevice);

    if (serialRoot) {
      serialRootSearchKernel<<<1, 1>>>(d_board, d_moves, numMoves, d_searchScores, searchDepth);
    } else {
      int searchBlocks = (numMoves + 31) / 32; // one warp minimum
      searchKernel<<<searchBlocks, 32>>>(d_board, d_moves, numMoves, d_searchScores, searchDepth);
    }
    cudaDeviceSynchronize();

    cudaMemcpy(h_searchScores, d_searchScores, numMoves * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < numMoves; i++) {
      if (!isfinite(h_searchScores[i]) || fabsf(h_searchScores[i]) > 99999.0f) h_searchScores[i] = -99999.0f;
    }

    // Find best move by search score
    int bestIdx = -1; float bestScore = -99999.0f;
    int bestTieScore = -2147483647;
    for (int i = 0; i < numMoves; i++) {
      int tieScore = (traincarRootTieBreak && fighterFamily == FIGHTER_FAMILY_TRAINCAR)
        ? hostTraincarRootTieScore(&board, &h_moves[i])
        : 0;
      if (h_searchScores[i] > bestScore ||
          (fabsf(h_searchScores[i] - bestScore) <= 0.001f && tieScore > bestTieScore)) {
        bestScore = h_searchScores[i];
        bestIdx = i;
        bestTieScore = tieScore;
      }
    }
    if (bestIdx < 0 || bestScore <= -99998.0f) { skippedNoLegalRootScore++; continue; }
    if (timeoutRootProxy) {
      for (int i = 0; i < numMoves; i++) {
        if (h_searchScores[i] > -99998.0f) {
          bestIdx = i;
          bestScore = h_searchScores[i];
          break;
        }
      }
    }
    int bookApplied = 0;
    if (traincarBook && traincarBookCount > 0 && fighterFamily == FIGHTER_FAMILY_TRAINCAR) {
      char bookMove[8];
      if (lookupTraincarBookMove(traincarBookEntries, traincarBookCount, fen, bookMove, sizeof(bookMove))) {
        int bookIdx = -1;
        for (int i = 0; i < numMoves; i++) {
          char mv[8];
          moveToUci(&h_moves[i], mv, sizeof(mv));
          if (strcmp(mv, bookMove) == 0) { bookIdx = i; break; }
        }
        if (bookIdx >= 0 && h_searchScores[bookIdx] > -99998.0f) {
          bestIdx = bookIdx;
          bestScore = h_searchScores[bookIdx];
          traincarBookOverrides++;
          bookApplied = 1;
        } else {
          traincarBookMisses++;
        }
      }
    }
    if (ffnPolicyEnabled && !bookApplied) {
      int ffnIdx = chooseFFNPolicyMove(&ffnPolicy, &board, h_moves, h_searchScores, numMoves, bestIdx, hostFighterId(fighterBlobPath), 32);
      if (ffnIdx >= 0 && ffnIdx < numMoves && h_searchScores[ffnIdx] > -99998.0f && ffnIdx != bestIdx) {
        bestIdx = ffnIdx;
        bestScore = h_searchScores[bestIdx];
        ffnPolicyOverrides++;
      }
    }

    // Convert search best move to UCI
    char searchMove[8];
    moveToUci(&h_moves[bestIdx], searchMove, sizeof(searchMove));

    comparablePositions++;

    // Check agreement
    if (strcmp(searchMove, engineMove) == 0) {
      agreements++;
      if (emitAllPositions) {
        if (!first) printf(",\n");
        first = 0;
        printPositionComparisonJson(
          fen, engineMove, searchMove, bestScore, h_searchScores[engIdx],
          1, numMoves, 0, numConfigs, h_moves, h_searchScores, engIdx, bestIdx, 32, true);
      }
      continue;
    }

    totalPos++;
    float engineScore = h_searchScores[engIdx];
    int engineRank = 1;
    for (int i = 0; i < numMoves; i++) {
      if (h_searchScores[i] > engineScore) engineRank++;
    }

    // ================================================================
    // STEP 2: Knob tuner — which configs prefer the search move?
    // (kept from v1 for training signal extraction)
    // ================================================================
    Board boardA = board, boardB = board;
    boardA.pieces[h_moves[engIdx].to] = boardA.pieces[h_moves[engIdx].from];
    boardA.pieces[h_moves[engIdx].from] = EMPTY;
    boardA.side = 1-boardA.side;
    boardB.pieces[h_moves[bestIdx].to] = boardB.pieces[h_moves[bestIdx].from];
    boardB.pieces[h_moves[bestIdx].from] = EMPTY;
    boardB.side = 1-boardB.side;

    Board *d_bA, *d_bB;
    cudaMalloc(&d_bA, sizeof(Board)); cudaMalloc(&d_bB, sizeof(Board));
    cudaMemcpy(d_bA, &boardA, sizeof(Board), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bB, &boardB, sizeof(Board), cudaMemcpyHostToDevice);

    evalWithKnobs<<<blocks, BLOCK_SIZE>>>(d_bA, d_configs, d_scoresA, numConfigs);
    evalWithKnobs<<<blocks, BLOCK_SIZE>>>(d_bB, d_configs, d_scoresB, numConfigs);
    cudaDeviceSynchronize();

    cudaMemcpy(h_scoresA, d_scoresA, numConfigs*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_scoresB, d_scoresB, numConfigs*sizeof(float), cudaMemcpyDeviceToHost);

    int fixable = 0;
    int bestCfg = -1; float bestDelta = -99999;
    for (int i = 0; i < numConfigs; i++) {
      float delta = h_scoresB[i] - h_scoresA[i];
      if (delta > 0) { fixable++; if (delta > bestDelta) { bestDelta = delta; bestCfg = i; } }
    }

    if (fixable > 0) totalFixable++; else totalUnfixable++;

    if (bestCfg >= 0) {
      for (int k = 0; k < KNOB_COUNT; k++)
        bestKnobs[k] = bestKnobs[k] * 0.95f + h_configs[bestCfg].knobs[k] * 0.05f;
    }

    if (!first) printf(",\n");
    first = 0;
    printPositionComparisonJson(
      fen, engineMove, searchMove, bestScore, engineScore, engineRank, numMoves,
      fixable, numConfigs, h_moves, h_searchScores, engIdx, bestIdx,
      emitAllPositions ? 32 : 8, false);

    cudaFree(d_bA); cudaFree(d_bB);
  }

  clock_gettime(CLOCK_MONOTONIC, &now);
  double elapsed = (now.tv_sec - start.tv_sec) + (now.tv_nsec - start.tv_nsec) / 1e9;

  printf("\n],\"summary\":{");
  const int disagreements = totalPos;
  const float agreementRate = comparablePositions > 0 ? (float)agreements / comparablePositions : 0.0f;
  const float disagreementRate = comparablePositions > 0 ? (float)disagreements / comparablePositions : 0.0f;
  const float coverage = parsedLines > 0 ? (float)comparablePositions / parsedLines : 0.0f;
  printf("\"inputLines\":%d,\"parsedLines\":%d,", inputLines, parsedLines);
  printf("\"comparablePositions\":%d,\"agreements\":%d,\"disagreements\":%d,", comparablePositions, agreements, disagreements);
  printf("\"agreementRate\":%.3f,\"disagreementRate\":%.3f,\"coverage\":%.3f,", agreementRate, disagreementRate, coverage);
  printf("\"skippedShort\":%d,\"skippedNoTab\":%d,\"skippedNoMoves\":%d,\"skippedEngineMoveMissing\":%d,\"skippedNoLegalRootScore\":%d,",
    skippedShort, skippedNoTab, skippedNoMoves, skippedEngineMoveMissing, skippedNoLegalRootScore);
  printf("\"positions\":%d,\"fixable\":%d,\"unfixable\":%d,\"fixRate\":%.3f,",
    totalPos, totalFixable, totalUnfixable, totalPos>0?(float)totalFixable/totalPos:0);
  printf("\"elapsed\":%.2f,\"posPerSec\":%.1f,", elapsed, totalPos > 0 ? totalPos/elapsed : 0.0);
  printf("\"searchDepth\":%d,", searchDepth);
  printf("\"fighterFamily\":\"%s\",", fighterFamilyName(fighterFamily));
  printf("\"familyDispatch\":%s,\"timeoutRootProxy\":%s,\"cpuShapedSearch\":%s,\"traincarRootTieBreak\":%s,",
    familyDispatch ? "true" : "false",
    timeoutRootProxy ? "true" : "false",
    cpuShapedSearch ? "true" : "false",
    traincarRootTieBreak ? "true" : "false");
  printf("\"emitAllPositions\":%s,", emitAllPositions ? "true" : "false");
  printf("\"traincarBook\":%s,\"traincarBookEntries\":%d,\"traincarBookOverrides\":%d,\"traincarBookMisses\":%d,",
    traincarBook ? "true" : "false",
    traincarBookCount,
    traincarBookOverrides,
    traincarBookMisses);
  printf("\"ffnPolicy\":%s,\"ffnPolicyOverrides\":%d,",
    ffnPolicyEnabled ? "true" : "false",
    ffnPolicyOverrides);
  printf("\"verdict\":\"%s\",",
    totalPos==0 ? (agreements > 0 ? "ALL_AGREE" : "NO_DATA") :
    totalFixable*100/totalPos >= 70 ? "ARCHITECTURE_FINE" :
    totalFixable*100/totalPos >= 40 ? "PARTIAL" : "NEEDS_WORK");
  printf("\"bestKnobs\":{");
  const char* names[] = {
    "layer0_pawn","layer0_king","layer0_queen","layer0_rook","layer0_minor","layer0_tempo",
    "layer1_pawn","layer1_king","layer1_queen","layer1_rook","layer1_minor","layer1_tempo",
    "layer2_pawn","layer2_king","layer2_queen","layer2_rook","layer2_minor","layer2_tempo",
    "layer3_pawn","layer3_king","layer3_queen","layer3_rook","layer3_minor","layer3_tempo",
    "aggression0","aggression1","aggression2","aggression3",
    "layerBleed",
    "doubledPawnMG","doubledPawnEG","isolatedPawnMG","isolatedPawnEG","connectedPawnMG","connectedPawnEG",
    "shieldBonus","shieldHole",
    "rookOpenFile","rookSemiOpen","rook7th",
    "bishopPair",
    "queenExposed","queenTrapped",
    "tempo"
  };
  for (int k = 0; k < KNOB_COUNT; k++) { if (k) printf(","); printf("\"%s\":%.1f", names[k], bestKnobs[k]); }
  printf("}}}\n");

  // Cleanup
  cudaFree(d_board); cudaFree(d_configs); cudaFree(d_scoresA); cudaFree(d_scoresB);
  cudaFree(d_defaults); cudaFree(d_moves); cudaFree(d_winRates); cudaFree(d_visits);
  cudaFree(d_searchScores);
  free(h_scoresA); free(h_scoresB); free(h_searchScores); free(h_configs);
  free(traincarBookEntries);
  return 0;
}
