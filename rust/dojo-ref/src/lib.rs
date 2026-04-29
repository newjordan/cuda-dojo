//! GPU chess referee — Rust binding for cuda-dojo's `librefcuda.so`.
//!
//! Every chess operation dispatches to a CUDA kernel inside the dojo's
//! GPU engine. Rust handles type safety, ownership, and string conversion;
//! it does NO chess work on the host. This is the canonical referee that
//! chess_train, the omnifold pipeline, and any future engine should use
//! to talk to the GPU's chess rules layer.

use std::ffi::CString;
use std::os::raw::{c_char, c_float, c_int};

/// Opaque handle to a `cuda_dojo::engine::Position` (74 bytes mailbox + side
/// + castling + ep + clocks). Allocated by the C ABI; freed via `Drop`.
#[repr(C)]
pub struct PositionRaw {
    _data: [u8; 96], // 74 actual bytes + safety padding
}

extern "C" {
    fn refc_position_new() -> *mut PositionRaw;
    fn refc_position_free(p: *mut PositionRaw);
    fn refc_parse_fen(fen: *const c_char, out: *mut PositionRaw) -> c_int;
    fn refc_legal_moves(
        pos: *const PositionRaw,
        moves_out: *mut i32,
        max_n: c_int,
    ) -> c_int;
    fn refc_make_move(
        pos: *const PositionRaw,
        mv: i32,
        out: *mut PositionRaw,
    ) -> c_int;
    fn refc_is_check(pos: *const PositionRaw) -> c_int;
    fn refc_legal_count(pos: *const PositionRaw) -> c_int;
    fn refc_is_checkmate(pos: *const PositionRaw) -> c_int;
    fn refc_is_stalemate(pos: *const PositionRaw) -> c_int;
    fn refc_coord3d(
        pos: *const PositionRaw,
        xyz_out: *mut c_float,
        octant_out: *mut c_int,
    ) -> c_int;
    fn refc_move_to_uci(mv: i32, uci_out: *mut c_char);
    fn refc_position_size() -> c_int;

    fn refc_search_init();
    fn refc_search_new_game();
    fn refc_search_shutdown();
    fn refc_search_best_move(
        pos: *const PositionRaw,
        depth: c_int,
        movetime_ms: c_int,
        move_out: *mut i32,
        score_out: *mut c_int,
    ) -> c_int;
}

/// Owned chess position. Drop frees the underlying C allocation.
pub struct Position {
    raw: *mut PositionRaw,
}

unsafe impl Send for Position {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Move(pub i32);

impl Move {
    /// UCI long-algebraic notation (e.g. "e2e4", "e7e8q", "e1g1" for
    /// king-side castle in standard chess).
    pub fn to_uci(self) -> String {
        let mut buf = [0u8; 8];
        unsafe {
            refc_move_to_uci(self.0, buf.as_mut_ptr() as *mut c_char);
        }
        let len = buf.iter().position(|&b| b == 0).unwrap_or(8);
        std::str::from_utf8(&buf[..len]).unwrap_or("").to_string()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Coord3D {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub octant: i32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RefError {
    AllocFailed,
    InvalidFen,
    IllegalMove,
}

impl std::fmt::Display for RefError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RefError::AllocFailed => write!(f, "Position allocation failed"),
            RefError::InvalidFen => write!(f, "FEN parse failed"),
            RefError::IllegalMove => write!(f, "illegal move"),
        }
    }
}

impl std::error::Error for RefError {}

impl Position {
    /// Parse a FEN string into a fresh Position. Host-side parsing only;
    /// no chess work happens here — that starts when you call `legal_moves`,
    /// `make_move`, etc.
    pub fn from_fen(fen: &str) -> Result<Self, RefError> {
        let c_fen = CString::new(fen).map_err(|_| RefError::InvalidFen)?;
        unsafe {
            let raw = refc_position_new();
            if raw.is_null() {
                return Err(RefError::AllocFailed);
            }
            if refc_parse_fen(c_fen.as_ptr(), raw) != 0 {
                refc_position_free(raw);
                return Err(RefError::InvalidFen);
            }
            Ok(Position { raw })
        }
    }

    /// Standard starting position.
    pub fn startpos() -> Self {
        Self::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
            .expect("startpos FEN is valid")
    }

    /// All legal moves at this position. Computed on GPU.
    pub fn legal_moves(&self) -> Vec<Move> {
        let mut buf = vec![0i32; 256];
        let n = unsafe {
            refc_legal_moves(self.raw, buf.as_mut_ptr(), 256)
        };
        if n < 0 {
            return Vec::new();
        }
        buf.truncate(n as usize);
        buf.into_iter().map(Move).collect()
    }

    /// Number of legal moves. Computed on GPU.
    pub fn legal_count(&self) -> usize {
        unsafe { refc_legal_count(self.raw) as usize }
    }

    /// Side-to-move's king is currently attacked (in check). Computed on GPU.
    pub fn is_check(&self) -> bool {
        unsafe { refc_is_check(self.raw) != 0 }
    }

    pub fn is_checkmate(&self) -> bool {
        unsafe { refc_is_checkmate(self.raw) != 0 }
    }

    pub fn is_stalemate(&self) -> bool {
        unsafe { refc_is_stalemate(self.raw) != 0 }
    }

    /// No legal moves remain — mate or stalemate.
    pub fn is_terminal(&self) -> bool {
        self.legal_count() == 0
    }

    /// Apply a move and return the resulting position. The move must be
    /// legal at this position (caller's responsibility — pull moves from
    /// `legal_moves()`). GPU `make_move` kernel.
    pub fn make_move(&self, mv: Move) -> Result<Position, RefError> {
        unsafe {
            let out = refc_position_new();
            if out.is_null() {
                return Err(RefError::AllocFailed);
            }
            if refc_make_move(self.raw, mv.0, out) != 0 {
                refc_position_free(out);
                return Err(RefError::IllegalMove);
            }
            Ok(Position { raw: out })
        }
    }

    /// 3-axis continuous coordinates `(x, y, z)` and octant id, computed
    /// on GPU via `compute_coord3d`. Material balance + king safety +
    /// pawn structure + mobility (y), non-pawn material (z), piece-cluster
    /// asymmetry (x).
    pub fn coord3d(&self) -> Coord3D {
        let mut xyz = [0.0f32; 3];
        let mut oct = 0i32;
        unsafe {
            refc_coord3d(self.raw, xyz.as_mut_ptr(), &mut oct);
        }
        Coord3D {
            x: xyz[0],
            y: xyz[1],
            z: xyz[2],
            octant: oct,
        }
    }
}

impl Drop for Position {
    fn drop(&mut self) {
        unsafe { refc_position_free(self.raw) }
    }
}

/// Sanity check exposed at the FFI boundary.
pub fn position_size() -> usize {
    unsafe { refc_position_size() as usize }
}

/// GPU search engine — alpha-beta with TT, killers, qsearch, PeSTO eval.
/// All search work runs on the GPU. Exactly one `Engine` should exist
/// per process; constructing additional engines reinitialises shared
/// device buffers and is generally fine but unnecessary.
pub struct Engine {
    _private: (),
}

impl Engine {
    /// Initialise the search subsystem (TT, zobrist, root buffers).
    /// Idempotent across calls.
    pub fn new() -> Self {
        unsafe { refc_search_init() }
        Engine { _private: () }
    }

    /// Reset between games (clears the host-side stop flag mirror).
    pub fn new_game(&self) {
        unsafe { refc_search_new_game() }
    }

    /// Search to fixed depth. Returns (bestmove, score in centipawns
    /// from STM POV). Score is +mate when STM is mating.
    pub fn search_depth(&self, pos: &Position, depth: u32) -> Option<(Move, i32)> {
        let mut mv: i32 = 0;
        let mut sc: c_int = 0;
        let r = unsafe {
            refc_search_best_move(
                pos.raw,
                depth as c_int,
                0,
                &mut mv,
                &mut sc,
            )
        };
        if r < 0 {
            None
        } else {
            Some((Move(mv), sc as i32))
        }
    }

    /// Search with movetime budget (ms). Returns (bestmove, score).
    pub fn search_movetime(&self, pos: &Position, movetime_ms: u32) -> Option<(Move, i32)> {
        let mut mv: i32 = 0;
        let mut sc: c_int = 0;
        let r = unsafe {
            refc_search_best_move(
                pos.raw,
                0,
                movetime_ms as c_int,
                &mut mv,
                &mut sc,
            )
        };
        if r < 0 {
            None
        } else {
            Some((Move(mv), sc as i32))
        }
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        unsafe { refc_search_shutdown() }
    }
}

impl Default for Engine {
    fn default() -> Self {
        Self::new()
    }
}
