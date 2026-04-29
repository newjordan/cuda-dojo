//! Smoke test for dojo-ref — proves Rust ↔ GPU chess works end-to-end.
//!
//! Run from the dojo-ref crate dir:
//!     cargo run --example smoke

use dojo_ref::{Position, Move};

fn main() {
    println!("dojo-ref smoke");
    println!("Position size (FFI): {} bytes", dojo_ref::position_size());

    let p = Position::startpos();
    println!(
        "startpos: legal_count={}, is_check={}, mate={}, stalemate={}",
        p.legal_count(),
        p.is_check(),
        p.is_checkmate(),
        p.is_stalemate(),
    );

    let moves = p.legal_moves();
    print!("legal moves ({}): ", moves.len());
    for mv in &moves {
        print!("{} ", mv.to_uci());
    }
    println!();

    let c = p.coord3d();
    println!(
        "startpos coord3d: x={:.4} y={:.4} z={:.4} octant={}",
        c.x, c.y, c.z, c.octant,
    );

    // Find e2e4 and apply it.
    let e2e4 = moves
        .iter()
        .copied()
        .find(|mv| mv.to_uci() == "e2e4")
        .expect("e2e4 must be legal at startpos");
    let p2 = p.make_move(e2e4).expect("e2e4 is legal");
    println!(
        "after e2e4: legal_count={} (expected 20)",
        p2.legal_count(),
    );
    let c2 = p2.coord3d();
    println!(
        "after e2e4 coord3d: x={:.4} y={:.4} z={:.4} octant={}",
        c2.x, c2.y, c2.z, c2.octant,
    );

    // Fool's mate position — white is mated.
    let mated = Position::from_fen(
        "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3",
    )
    .expect("FEN parses");
    println!(
        "fool's mate: is_check={} legal_count={} mate={} stalemate={}",
        mated.is_check(),
        mated.legal_count(),
        mated.is_checkmate(),
        mated.is_stalemate(),
    );

    println!("OK");
}
