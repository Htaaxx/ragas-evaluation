#!/usr/bin/env python
"""Print instructions for running the ASQA RAG baseline notebook workflow."""

from __future__ import annotations

import argparse

from bootstrap import bootstrap

bootstrap()

from rag_filtering.config.loader import load_yaml, resolve_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RAG baseline helper (run notebook for full pipeline)",
    )
    parser.add_argument(
        "--config",
        default="configs/experiments/asqa_baseline.yaml",
        help="Baseline experiment config YAML",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    print("ASQA RAG baseline is run interactively in:")
    print("  notebooks/03_rag_asqa_baseline.ipynb")
    print()
    print("Expected outputs:")
    print(f"  predictions: {resolve_path(cfg['results']['predictions_csv'])}")
    print(f"  filtered:    {resolve_path(cfg['results']['filtered_csv'])}")
    print()
    print("After generating predictions, apply the filter with:")
    print("  python scripts/run_filter_on_rag.py --config configs/experiments/asqa_baseline.yaml")


if __name__ == "__main__":
    main()
