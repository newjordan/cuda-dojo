//! Self-play game: cuda-dojo's GPU engine plays itself, all chess on GPU,
//! orchestrated entirely from Rust via the dojo-ref crate.
//!
//! Run from the dojo-ref crate dir:
//!     cargo run --release --example play_game
//!
//! No UCI subprocess. No python-chess. Rust holds typed Position and Move,
//! calls Engine.search_depth, applies the returned move, repeats until
//! mate / stalemate / max plies. Every chess decision is a CUDA kernel
//! launched via the FFI ABI in librefcuda.so.

use std::time::Instant;

use dojo_ref::{Engine, Position};

fn main() {
    let engine = Engine::new();
    engine.new_game();

    let mut pos = Position::startpos();
    let depth = 5u32;
    let max_plies = 80usize;

    println!("=== dojo-ref self-play game (Rust → GPU) ===");
    println!("search depth: {}, max plies: {}", depth, max_plies);

    let mut moves: Vec<String> = Vec::new();
    let mut per_move_ms: Vec<u128> = Vec::new();
    let game_start = Instant::now();

    for ply in 0..max_plies {
        if pos.is_terminal() {
            println!("ply {} TERMINAL: is_check={} (so {})",
                     ply,
                     pos.is_check(),
                     if pos.is_check() { "checkmate" } else { "stalemate" });
            break;
        }

        let t0 = Instant::now();
        let (mv, score) = match engine.search_depth(&pos, depth) {
            Some(x) => x,
            None => {
                println!("ply {} search returned None", ply);
                break;
            }
        };
        let dt_ms = t0.elapsed().as_millis();
        per_move_ms.push(dt_ms);

        let uci = mv.to_uci();
        let side = if ply % 2 == 0 { "W" } else { "B" };
        println!(
            "ply {:3}  {}  {:<6}  {:5}ms  score={:+}cp  legal_count={}",
            ply, side, uci, dt_ms, score, pos.legal_count(),
        );
        moves.push(uci);

        pos = match pos.make_move(mv) {
            Ok(p) => p,
            Err(e) => {
                println!("make_move failed: {}", e);
                break;
            }
        };
    }

    let total_s = game_start.elapsed().as_secs_f64();
    println!();
    println!("=== Game over ===");
    println!("plies played:  {}", moves.len());
    println!("total time:    {:.2}s", total_s);
    if !per_move_ms.is_empty() {
        let sum: u128 = per_move_ms.iter().sum();
        let avg = sum / per_move_ms.len() as u128;
        let max = *per_move_ms.iter().max().unwrap();
        println!("per move:      avg={}ms max={}ms  (total {}ms search)", avg, max, sum);
    }
    println!("final position FEN-equivalent state:");
    println!("  is_check={}  is_checkmate={}  is_stalemate={}  is_terminal={}",
             pos.is_check(), pos.is_checkmate(),
             pos.is_stalemate(), pos.is_terminal());
    let c = pos.coord3d();
    println!("  coord3d: x={:.4} y={:.4} z={:.4} octant={}", c.x, c.y, c.z, c.octant);
    println!();
    println!("move list:");
    for chunk in moves.chunks(16) {
        println!("  {}", chunk.join(" "));
    }
}
