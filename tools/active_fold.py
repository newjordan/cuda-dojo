"""Import path used by tools/render_pipeline.py.

The fold implementation lives in compression/board_fold.py. This module
re-exports that implementation so the pipeline keeps the import it was
written against. Nothing here changes the fold math.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from compression.board_fold import (  # noqa: E402
    active_fold_positions,
    compute_warmth,
    find_fold_center,
)

__all__ = [
    "active_fold_positions",
    "compute_warmth",
    "find_fold_center",
]
