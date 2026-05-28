#!/usr/bin/env python
"""Train and evaluate the Self-RAG-style answer verifier."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from verifier_system import VerifierSystem  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Self-RAG-style generative answer verifier (Flan-T5)",
    )
    parser.add_argument(
        "--config",
        default="configs/experiments/rag_verifier.yaml",
        help="Verifier config YAML (repo-root relative)",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Optional override path to labeled_asqa.csv",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Fine-tune Flan-T5 verifier",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run evaluation after training or on a loaded checkpoint",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "test"],
        help="Split to evaluate",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the latest checkpoint",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s",
    )
    args = parse_args()

    if not args.train and not args.evaluate:
        raise SystemExit("Pass --train and/or --evaluate.")

    verifier = VerifierSystem(config_path=args.config)
    verifier.load_data(csv_path=args.csv)

    if args.train:
        verifier.train(resume_from_checkpoint=args.resume)
    else:
        verifier.load_model()

    if args.evaluate:
        metrics = verifier.evaluate(split=args.split)
        print(
            f"Evaluation complete: F1={metrics['f1']:.3f}, "
            f"FPR={metrics['fpr']:.3f}, recall={metrics['recall']:.3f}"
        )


if __name__ == "__main__":
    main()
