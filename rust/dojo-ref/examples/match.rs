//! Match between two opponents on the GPU.
//!
//! White and Black get DIFFERENT search depths, so the engine plays an
//! asymmetric match against itself — proving the API actually pits two
//! distinct opponents. Same Engine instance backs both sides because
//! cuda-dojo's search subsystem is process-global; the asymmetry is in
//! the per-call depth argument.
//!
//! Run:
//!     cargo run --release --example match
//!     DOJO_W_DEPTH=5 DOJO_B_DEPTH=2 cargo run --release --example match
//!
//! Output: per-ply move + score, final game state, plies-to-terminate,
//! who won (mate detection on the terminal position).

use std::time::Instant;

use dojo_ref::{Engine, Position};

fn env_u32(name: &str, default: u32) -> u32 {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn main() {
    let white_depth = env_u32("DOJO_W_DEPTH", 5);
    let black_depth = env_u32("DOJO_B_DEPTH", 2);
    let max_plies = env_u32("DOJO_MAX_PLIES", 80) as usize;

    let engine = Engine::new();
    engine.new_game();

    let mut pos = Position::startpos();
    let mut moves: Vec<String> = Vec::new();
    let mut times_w: Vec<u128> = Vec::new();
    let mut times_b: Vec<u128> = Vec::new();
    let game_start = Instant::now();
    let mut termination = String::from("max_plies");

    println!("=== dojo-ref MATCH ===");
    println!("White: search_depth={}", white_depth);
    println!("Black: search_depth={}", black_depth);
    println!("max_plies={}", max_plies);
    println!();

    for ply in 0..max_plies {
        if pos.is_terminal() {
            termination = if pos.is_check() {
                "checkmate".to_string()
            } else {
                "stalemate".to_string()
            };
            break;
        }

        let white_to_move = ply % 2 == 0;
        let depth = if white_to_move { white_depth } else { black_depth };

        let t0 = Instant::now();
        let (mv, score) = match engine.search_depth(&pos, depth) {
            Some(x) => x,
            None => {
                termination = format!("search returned None at ply {}", ply);
                break;
            }
        };
        let dt_ms = t0.elapsed().as_millis();
        if white_to_move { times_w.push(dt_ms); } else { times_b.push(dt_ms); }

        let uci = mv.to_uci();
        let side = if white_to_move { "W" } else { "B" };
        println!(
            "ply {:3}  {} d{}  {:<6}  {:5}ms  score={:+}cp  legal_count={}",
            ply, side, depth, uci, dt_ms, score, pos.legal_count(),
        );
        moves.push(uci);

        pos = match pos.make_move(mv) {
            Ok(p) => p,
            Err(e) => {
                termination = format!("make_move failed: {}", e);
                break;
            }
        };
    }

    let total_s = game_start.elapsed().as_secs_f64();

    // Determine winner: if terminal+in-check, the side TO MOVE was mated,
    // so the OTHER side won. If terminal but not in check → stalemate (draw).
    // If not terminal (max_plies hit) → undecided (treated as draw).
    let winner = if pos.is_checkmate() {
        let side_to_move_white = moves.len() % 2 == 0;
        if side_to_move_white { "Black (mated White)" } else { "White (mated Black)" }
    } else if pos.is_stalemate() {
        "draw (stalemate)"
    } else {
        "undecided (max plies)"
    };

    println!();
    println!("=== Match over ===");
    println!("termination:    {}", termination);
    println!("plies played:   {}", moves.len());
    println!("total time:     {:.2}s", total_s);
    let stats = |t: &Vec<u128>| -> (u128, u128, u128) {
        if t.is_empty() { return (0, 0, 0); }
        let sum: u128 = t.iter().sum();
        let avg = sum / t.len() as u128;
        let max = *t.iter().max().unwrap();
        (sum, avg, max)
    };
    let (sw, aw, mw) = stats(&times_w);
    let (sb, ab, mb) = stats(&times_b);
    println!("White (d{}):    n={} sum={}ms avg={}ms max={}ms",
             white_depth, times_w.len(), sw, aw, mw);
    println!("Black (d{}):    n={} sum={}ms avg={}ms max={}ms",
             black_depth, times_b.len(), sb, ab, mb);
    println!("WINNER:         {}", winner);
    println!();
    println!("move list:");
    for chunk in moves.chunks(16) {
        println!("  {}", chunk.join(" "));
    }
}
