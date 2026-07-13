#!/usr/bin/env python
"""Sync Kaggle kernel outputs (and optionally notebooks) into this repo.

Prerequisites
-------------
1. Kaggle API token at ``%USERPROFILE%\\.kaggle\\access_token``
   (or set env ``KAGGLE_API_TOKEN``).
2. ``pip install kaggle``

Examples
--------
# List your kernels
python scripts/kaggle_sync.py list

# Pull outputs from a kernel into results/kaggle_runs/<slug>/
python scripts/kaggle_sync.py pull-output htaaxx/v4-thesis

# Pull outputs and copy predictions.csv into the Stage-09 expected path
python scripts/kaggle_sync.py pull-output htaaxx/v4-thesis --install-predictions

# Pull the notebook source back into notebooks/kaggle_pulled/
python scripts/kaggle_sync.py pull-notebook htaaxx/v4-thesis
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from bootstrap import bootstrap

ROOT = bootstrap()

DEFAULT_OUTPUT_ROOT = ROOT / "results" / "kaggle_runs"
DEFAULT_NOTEBOOK_DIR = ROOT / "notebooks" / "kaggle_pulled"
EXPECTED_PREDICTIONS = ROOT / "results" / "normal_rag" / "merged" / "predictions.csv"


def _load_token() -> None:
    """Ensure KAGGLE_API_TOKEN is set from ~/.kaggle/access_token if needed."""
    if os.getenv("KAGGLE_API_TOKEN"):
        return
    token_path = Path.home() / ".kaggle" / "access_token"
    if token_path.exists():
        os.environ["KAGGLE_API_TOKEN"] = token_path.read_text(encoding="utf-8").strip()
        return
    raise SystemExit(
        "Kaggle token not found. Create %USERPROFILE%\\.kaggle\\access_token "
        "or set KAGGLE_API_TOKEN."
    )


def _kaggle(*args: str) -> None:
    _load_token()
    cmd = [sys.executable, "-m", "kaggle", *args]
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def cmd_list(_: argparse.Namespace) -> None:
    _kaggle("kernels", "list", "--mine")


def cmd_pull_output(args: argparse.Namespace) -> None:
    slug = args.kernel.strip().rstrip("/")
    name = slug.split("/")[-1]
    out_dir = Path(args.out_dir) if args.out_dir else DEFAULT_OUTPUT_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)

    _kaggle("kernels", "output", slug, "-p", str(out_dir), "-o")
    print(f"Outputs saved to: {out_dir}")

    if args.install_predictions:
        candidates = list(out_dir.rglob("predictions.csv"))
        # Prefer normal_rag/merged/predictions.csv if present
        preferred = [
            p
            for p in candidates
            if "normal_rag" in p.as_posix() and "merged" in p.as_posix()
        ]
        src = preferred[0] if preferred else (candidates[0] if candidates else None)
        if src is None:
            raise SystemExit(f"No predictions.csv found under {out_dir}")
        EXPECTED_PREDICTIONS.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, EXPECTED_PREDICTIONS)
        print(f"Copied {src} -> {EXPECTED_PREDICTIONS}")


def cmd_pull_notebook(args: argparse.Namespace) -> None:
    slug = args.kernel.strip().rstrip("/")
    name = slug.split("/")[-1]
    out_dir = Path(args.out_dir) if args.out_dir else DEFAULT_NOTEBOOK_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    _kaggle("kernels", "pull", slug, "-p", str(out_dir))
    print(f"Notebook saved to: {out_dir}")
    print("Diff against notebooks/08_*.ipynb or 09_*.ipynb, then copy changes you want.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="List your Kaggle kernels")
    p_list.set_defaults(func=cmd_list)

    p_out = sub.add_parser("pull-output", help="Download kernel output files")
    p_out.add_argument("kernel", help="Kernel slug, e.g. htaaxx/v4-thesis")
    p_out.add_argument(
        "--out-dir",
        default=None,
        help=f"Destination (default: {DEFAULT_OUTPUT_ROOT}/<slug>)",
    )
    p_out.add_argument(
        "--install-predictions",
        action="store_true",
        help=f"Copy predictions.csv to {EXPECTED_PREDICTIONS}",
    )
    p_out.set_defaults(func=cmd_pull_output)

    p_nb = sub.add_parser("pull-notebook", help="Download kernel notebook source")
    p_nb.add_argument("kernel", help="Kernel slug, e.g. htaaxx/v4-thesis")
    p_nb.add_argument(
        "--out-dir",
        default=None,
        help=f"Destination (default: {DEFAULT_NOTEBOOK_DIR}/<slug>)",
    )
    p_nb.set_defaults(func=cmd_pull_notebook)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
