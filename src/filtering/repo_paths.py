"""Locate the repository root on local machines and Kaggle."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional


_MARKERS = (
    "scripts/run_deberta_nli_baseline.py",
    "configs/experiments/filter_training.yaml",
    "src/filtering/learned_filter.py",
)


def _iter_candidates(extra: Optional[Iterable[Path]] = None) -> list[Path]:
    candidates: list[Path] = []
    cwd = Path.cwd().resolve()
    candidates.extend([cwd, *cwd.parents])

    defaults = [
        Path("/kaggle/working"),
        Path("/kaggle/working/ragas-evaluation"),
        cwd / "ragas-evaluation",
        cwd.parent / "ragas-evaluation",
    ]
    if extra:
        defaults.extend(extra)

    for base in defaults:
        try:
            resolved = base.resolve()
        except OSError:
            continue
        if not resolved.exists():
            continue
        candidates.append(resolved)
        nested = resolved / "ragas-evaluation"
        if nested.exists():
            candidates.append(nested.resolve())
    return candidates


def find_repo_root(extra: Optional[Iterable[Path]] = None) -> Path:
    """Return the directory that contains ``scripts/`` + ``src/filtering/``."""
    seen: set[Path] = set()
    for cand in _iter_candidates(extra):
        cand = cand.resolve()
        if cand in seen:
            continue
        seen.add(cand)
        if all((cand / marker).is_file() for marker in _MARKERS):
            return cand

    raise FileNotFoundError(
        "Could not find the ragas-evaluation repo root. "
        f"cwd={Path.cwd()}. Expected a folder that contains "
        f"{_MARKERS[0]}. On Kaggle, clone/upload the full repo "
        "(e.g. /kaggle/working/ragas-evaluation) and re-run Setup."
    )


def deberta_train_script(repo_root: Optional[Path] = None) -> Path:
    """Absolute path to ``scripts/run_deberta_nli_baseline.py``."""
    root = repo_root or find_repo_root()
    script = root / "scripts" / "run_deberta_nli_baseline.py"
    if not script.is_file():
        raise FileNotFoundError(f"Missing training script: {script}")
    return script
