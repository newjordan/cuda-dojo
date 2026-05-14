#!/usr/bin/env python3
"""selfplay_game.py — Two gpu_forge agents play a full game against each other.

Uses python-chess for move generation and FEN construction.
Pipes positions to gpu_forge (with FrostMatrix eval), extracts engine moves,
and plays them on the board. Supports CPU parity check via Stockfish.

Usage:
  python3 selfplay_game.py [--engine ./gpu_forge_fm] [--weights PATH]
                           [--depth 5] [--pgn game.pgn] [--stockfish sf]
  Saves PGN to selfplay_game.pgn by default.
"""
import subprocess, sys, os, json, time, argparse
import chess, chess.pgn

def build_input(fen, legal_moves):
    """Build tab-separated input line for gpu_forge.
    Uses first legal move as reference; --emit-all ensures output regardless."""
    legals = ",".join(legal_moves[:64])
    ref_move = legal_moves[0]
    return f"{fen}\t{ref_move}\t{legals}\n"

def parse_engine_move(output, fen):
    """Extract the engine's best move from gpu_forge JSON output."""
    try:
        data = json.loads(output)
    except json.JSONDecodeError:
        start = output.find('{"positions"')
        if start < 0: return None
        end = output.rfind('}')
        if end < 0: return None
        try:
            data = json.loads(output[start:end+1])
        except:
            return None

    positions = data.get("positions", [])
    if not positions: return None

    pos = positions[0]
    # Primary: the "mcts" field gives the engine's chosen move
    mcts = pos.get("mcts")
    if mcts: return mcts

    # Fallback: from gpu_root moves
    for m in pos.get("gpu_root", []):
        if m.get("is_best"):
            return m.get("move")
    return None

def play_game(engine_bin, weights_dir, depth, stockfish_bin=None, pgn_path="selfplay_game.pgn"):
    """Play a full game with the engine playing both sides."""
    board = chess.Board()
    pgn_game = chess.pgn.Game()
    pgn_game.headers["Event"] = "CUDA Dojo Selfplay"
    pgn_game.headers["White"] = "FrostMatrix V3 (white)"
    pgn_game.headers["Black"] = "FrostMatrix V3 (black)"
    pgn_game.headers["Depth"] = str(depth)
    node = pgn_game

    move_count = 0
    max_moves = 200
    sf_proc = None

    if stockfish_bin:
        sf_proc = subprocess.Popen(
            [stockfish_bin], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1
        )
        sf_proc.stdin.write("uci\n")
        sf_proc.stdin.write("setoption name Threads value 4\n")
        sf_proc.stdin.write("setoption name Hash value 256\n")
        sf_proc.stdin.flush()
        sf_proc.stdin.write("isready\n"); sf_proc.stdin.flush()
        for line in sf_proc.stdout:
            if "readyok" in line: break

    print(f"{'='*60}")
    print(f"Self-play game: FrostMatrix V3 depth={depth}")
    print(f"Engine: {engine_bin}")
    print(f"Weights: {weights_dir}")
    if sf_proc:
        print(f"CPU parity: Stockfish (analyzing alongside)")
    print(f"{'='*60}")

    while not board.is_game_over(claim_draw=True) and move_count < max_moves:
        fen = board.fen()
        legal_moves = [m.uci() for m in board.legal_moves]
        side = "White" if board.turn == chess.WHITE else "Black"

        input_line = build_input(fen, legal_moves)
        args = [engine_bin, "1", "--depth", str(depth), "--emit-all"]
        if weights_dir:
            args.extend(["--frostmatrix-weights", weights_dir])

        proc = subprocess.run(
            args,
            input=input_line,
            capture_output=True, text=True,
            timeout=60,
            cwd=os.path.dirname(engine_bin) or ".",
        )

        stderr_output = proc.stderr
        stdout_output = proc.stdout

        move_uci = parse_engine_move(stdout_output, fen)

        if move_uci is None:
            print(f"[{move_count+1}] {side}: ENGINE FAILED to return move")
            print(f"    stderr: {stderr_output[-200:]}")
            break

        try:
            move = chess.Move.from_uci(move_uci)
        except ValueError:
            print(f"[{move_count+1}] {side}: Invalid move: {move_uci}")
            break

        if move not in board.legal_moves:
            print(f"[{move_count+1}] {side}: Illegal move: {move_uci}")
            # Try legal moves from output
            data = json.loads(stdout_output)
            pos = data["positions"][0]
            all_moves = sorted(pos.get("moves", []), key=lambda m: m.get("rank", 999))
            for m in all_moves:
                try:
                    cm = chess.Move.from_uci(m["move"])
                    if cm in board.legal_moves:
                        move = cm
                        print(f"    Fallback to: {move_uci}")
                        break
                except:
                    continue
            else:
                break

        # Stockfish analysis for parity check
        sf_eval = None
        sf_best = None
        if sf_proc:
            sf_proc.stdin.write(f"position fen {fen}\n")
            sf_proc.stdin.write(f"go depth {depth}\n")
            sf_proc.stdin.flush()
            for line in sf_proc.stdout:
                if line.startswith("bestmove"):
                    sf_best = line.split()[1]
                    break
                if "score cp" in line:
                    sf_eval = line
            if sf_best:
                parity = "✓" if sf_best == move_uci else f"✗ (SF: {sf_best})"
            else:
                parity = "?"
        else:
            parity = "—"

        # Get SAN before pushing (SAN depends on board state pre-move)
        san = board.san(move)

        # Extract FrostMatrix root value from stderr
        fm_val = None
        for line in stderr_output.split("\n"):
            if "FrostMatrix" in line and "value" in line:
                fm_val = line.split("value:")[-1].strip()
                break

        board.push(move)
        node = node.add_variation(move)
        move_count += 1
        print(f"[{move_count:3d}] {side}: {san:6s} ({move_uci}) | parity: {parity} | FM root: {fm_val or '?'}")

    result = board.result(claim_draw=True)
    pgn_game.headers["Result"] = result
    print(f"\nGame over: {result} after {move_count} moves")
    print(board)
    if board.is_checkmate():
        print("Checkmate!")
    elif board.is_stalemate():
        print("Stalemate!")
    elif board.is_insufficient_material():
        print("Draw: insufficient material")

    # Save PGN
    with open(pgn_path, "w") as f:
        print(pgn_game, file=f, end="\n\n")
    print(f"PGN saved to: {pgn_path}")

    if sf_proc:
        sf_proc.stdin.write("quit\n"); sf_proc.stdin.flush()
        sf_proc.terminate()
        sf_proc.wait(timeout=5)

    return result, move_count

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CUDA Dojo self-play game")
    parser.add_argument("--engine", default="./gpu_forge_fm", help="Engine binary")
    parser.add_argument("--weights", default="/srv/models-hdd/chess-games/training_runs/v3_frostmatrix/export",
                        help="FrostMatrix weights directory")
    parser.add_argument("--depth", type=int, default=5, help="Search depth")
    parser.add_argument("--pgn", default="selfplay_game.pgn", help="PGN output path")
    parser.add_argument("--stockfish", default=None, help="Stockfish binary for CPU parity check")
    parser.add_argument("--max-moves", type=int, default=200, help="Maximum moves")
    args = parser.parse_args()

    play_game(args.engine, args.weights, args.depth,
              stockfish_bin=args.stockfish, pgn_path=args.pgn)
