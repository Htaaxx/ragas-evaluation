"""Shared path bootstrap for CLI scripts."""

from __future__ import annotations

import sys
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def bootstrap() -> Path:
    """Add ``src/`` to ``sys.path`` so ``rag_filtering`` imports work."""
    root = repo_root()
    src = root / "src"
    src_str = str(src)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)
    return root
