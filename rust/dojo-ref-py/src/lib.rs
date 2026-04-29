//! Python bindings for dojo-ref. All chess work runs on the GPU; Python
//! just holds typed Position / Move / Engine handles and gets results back.
//!
//! Build: `maturin develop` from this crate's dir installs a `dojo_ref`
//! Python module backed by librefcuda.so.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use dojoref as dr;

/// Wraps dr::Position. All methods dispatch chess work to GPU kernels.
#[pyclass(name = "Position")]
struct PyPosition {
    inner: dr::Position,
}

#[pymethods]
impl PyPosition {
    /// Standard starting position.
    #[new]
    fn new() -> Self {
        PyPosition { inner: dr::Position::startpos() }
    }

    /// Construct from a FEN string.
    #[staticmethod]
    fn from_fen(fen: &str) -> PyResult<Self> {
        dr::Position::from_fen(fen)
            .map(|p| PyPosition { inner: p })
            .map_err(|e| PyValueError::new_err(format!("from_fen: {}", e)))
    }

    /// Standard starting position (alias).
    #[staticmethod]
    fn startpos() -> Self {
        PyPosition { inner: dr::Position::startpos() }
    }

    /// All legal UCI moves at this position. GPU kernel.
    fn legal_moves(&self) -> Vec<String> {
        self.inner.legal_moves().into_iter().map(|m| m.to_uci()).collect()
    }

    /// Number of legal moves. GPU kernel.
    fn legal_count(&self) -> usize {
        self.inner.legal_count()
    }

    fn is_check(&self) -> bool { self.inner.is_check() }
    fn is_checkmate(&self) -> bool { self.inner.is_checkmate() }
    fn is_stalemate(&self) -> bool { self.inner.is_stalemate() }
    fn is_terminal(&self) -> bool { self.inner.is_terminal() }

    /// Apply a UCI move and return the resulting Position. The move must
    /// be legal at this position (caller responsibility — pull from
    /// legal_moves()). GPU kernel.
    fn make_move(&self, uci: &str) -> PyResult<PyPosition> {
        let mv = self.inner.legal_moves().into_iter()
            .find(|m| m.to_uci() == uci)
            .ok_or_else(|| PyValueError::new_err(format!("not a legal move: {}", uci)))?;
        self.inner.make_move(mv)
            .map(|p| PyPosition { inner: p })
            .map_err(|e| PyRuntimeError::new_err(format!("make_move: {}", e)))
    }

    /// 3D coord (x, y, z) + octant id. GPU kernel.
    fn coord3d(&self) -> (f32, f32, f32, i32) {
        let c = self.inner.coord3d();
        (c.x, c.y, c.z, c.octant)
    }
}

/// Wraps dr::Engine. Search subsystem (TT, alpha-beta, PeSTO) on GPU.
#[pyclass(name = "Engine")]
struct PyEngine {
    inner: dr::Engine,
}

#[pymethods]
impl PyEngine {
    #[new]
    fn new() -> Self {
        PyEngine { inner: dr::Engine::new() }
    }

    fn new_game(&self) {
        self.inner.new_game();
    }

    /// Search to fixed depth. Returns (bestmove_uci, score_cp_stm_pov).
    fn search_depth(&self, pos: &PyPosition, depth: u32) -> Option<(String, i32)> {
        self.inner.search_depth(&pos.inner, depth)
            .map(|(mv, sc)| (mv.to_uci(), sc))
    }

    /// Search with movetime budget (ms). Returns (bestmove_uci, score).
    fn search_movetime(&self, pos: &PyPosition, movetime_ms: u32) -> Option<(String, i32)> {
        self.inner.search_movetime(&pos.inner, movetime_ms)
            .map(|(mv, sc)| (mv.to_uci(), sc))
    }
}

#[pymodule]
fn dojo_ref(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPosition>()?;
    m.add_class::<PyEngine>()?;
    m.add("__doc__",
        "GPU chess referee — typed bindings around librefcuda.so. \
         Every chess operation dispatches to a CUDA kernel.")?;
    Ok(())
}
