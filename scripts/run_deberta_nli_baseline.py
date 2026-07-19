#!/usr/bin/env python
"""Run DeBERTa faithfulness filter N times + zero-shot NLI on a frozen test set.

Uses the leakage-safe base-ID split (test_size=0.2, seed=42) on
``data/labeled_merged.csv``, persists ``data/labeled_merged_test.csv``,
trains DeBERTa ``n_runs`` times with the same split, evaluates each run,
then evaluates zero-shot NLI once. Writes ``results/deberta_nli/summary.json``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.filtering.config_loader import load_yaml, resolve_path
from src.filtering.data_split import load_and_split
from src.filtering.deberta_filter_evaluator import (
    FilterEvaluator,
    average_classification_reports,
    classification_report_by_dataset,
    select_threshold_min_fpr,
)
from src.filtering.learned_filter import (
    AnswerQualityClassifier,
    _extract_top1_context,
    train_classifier,
)
from src.filtering.nli_filter import NLIAnswerFilter

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s",
)
logger = logging.getLogger(__name__)

METRIC_KEYS = [
    "precision",
    "recall",
    "f1",
    "accuracy",
    "fpr",
    "rejection_recall",
    "rejection_rate",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DeBERTa/NLI baseline (N runs)")
    parser.add_argument(
        "--config",
        default="configs/experiments/filter_training.yaml",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip training; evaluate existing run_* checkpoints only",
    )
    parser.add_argument(
        "--skip-nli",
        action="store_true",
        help="Skip zero-shot NLI evaluation",
    )
    parser.add_argument(
        "--skip-overfit-gate",
        action="store_true",
        help="Skip overfit_sanity_check before full training",
    )
    return parser.parse_args()


def _result_to_dict(result: Any) -> Dict[str, Any]:
    tp = int(result.tp)
    tn = int(result.tn)
    fp = int(result.fp)
    fn = int(result.fn)
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    return {
        "precision": float(result.precision),
        "recall": float(result.recall),
        "f1": float(result.f1),
        "accuracy": float(result.accuracy),
        "fpr": fpr,
        "rejection_recall": float(result.rejection_recall),
        "rejection_rate": float(result.rejection_rate),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def _mean_std(run_metrics: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for key in METRIC_KEYS:
        vals = [float(m[key]) for m in run_metrics if key in m and m[key] == m[key]]
        if not vals:
            continue
        out[key] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=0)),
            "values": vals,
        }
    return out


def _evaluate_filter_on_split(
    clf: AnswerQualityClassifier,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    min_recall: float,
    results_dir: Path,
    evaluator: FilterEvaluator,
) -> Dict[str, Any]:
    results_dir.mkdir(parents=True, exist_ok=True)

    val_contexts = [_extract_top1_context(c) for c in val_df["context"].tolist()]
    val_decisions = clf.predict_batch(val_contexts, val_df["answer"].tolist())
    threshold_result = select_threshold_min_fpr(
        [d.confidence for d in val_decisions],
        val_df["label"].tolist(),
        min_recall=min_recall,
    )
    best_threshold = float(threshold_result["threshold"])
    clf.threshold = best_threshold

    with open(results_dir / "threshold_selection.json", "w", encoding="utf-8") as fh:
        json.dump(threshold_result, fh, indent=2, default=str)

    test_contexts = [_extract_top1_context(c) for c in test_df["context"].tolist()]
    test_decisions = clf.predict_batch(test_contexts, test_df["answer"].tolist())
    test_preds = [d.confidence >= best_threshold for d in test_decisions]
    learned_result = evaluator.evaluate(test_preds, test_df["label"].tolist())
    learned_result.save(results_dir / "learned_filter_test_results.json")

    rows = []
    for i, (_, sample) in enumerate(test_df.iterrows()):
        rows.append({
            "id": sample["id"],
            "question": sample["question"],
            "answer": str(sample["answer"])[:200],
            "label": int(sample["label"]),
            "predicted": int(bool(test_preds[i])),
            "confidence": round(test_decisions[i].confidence, 4),
            "dataset": sample["dataset"] if "dataset" in sample.index else "unknown",
        })
    pred_df = pd.DataFrame(rows)
    pred_df.to_csv(results_dir / "test_predictions.csv", index=False)

    report_df = classification_report_by_dataset(pred_df)
    report_df.to_csv(results_dir / "classification_report.csv", index=False)
    logger.info("Classification report:\n%s", report_df.to_string(index=False))

    metrics = _result_to_dict(learned_result)
    metrics["threshold"] = best_threshold
    metrics["threshold_selection"] = threshold_result
    metrics["classification_report"] = report_df.to_dict(orient="records")
    return metrics


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    data_cfg = cfg["data"]
    n_runs = int(cfg.get("n_runs", 3))
    min_recall = float(cfg.get("min_recall_for_threshold", 0.70))
    results_root = resolve_path(cfg["results_dir"])
    results_root.mkdir(parents=True, exist_ok=True)
    model_root = resolve_path(cfg["model_output"])

    test_csv = data_cfg.get("test_csv")
    train_df, val_df, test_df = load_and_split(
        csv_path=str(resolve_path(data_cfg["labeled_csv"])),
        test_ratio=data_cfg["test_ratio"],
        val_ratio=data_cfg["val_ratio"],
        seed=data_cfg["seed"],
        test_csv_path=str(resolve_path(test_csv)) if test_csv else None,
        reuse_frozen_test=bool(data_cfg.get("reuse_frozen_test", True)),
    )
    logger.info(
        "Eval holdout: %d rows from %s",
        len(test_df),
        test_csv or "(resampled test split)",
    )

    if not args.skip_overfit_gate and not args.skip_train:
        from src.filtering.learned_filter import overfit_sanity_check

        logger.info("Running overfit_sanity_check gate …")
        gate = overfit_sanity_check(train_df, n_pairs=16, epochs=50)
        gate = dict(gate)
        gate["passed"] = float(gate.get("train_f1", 0.0)) >= 0.95
        with open(results_root / "overfit_sanity_check.json", "w", encoding="utf-8") as fh:
            json.dump(gate, fh, indent=2, default=str)
        if not gate["passed"]:
            raise RuntimeError(
                f"overfit_sanity_check failed: {gate}. "
                "Do not scale to full training until this passes."
            )

    evaluator = FilterEvaluator()
    run_metrics: List[Dict[str, Any]] = []
    run_reports: List[pd.DataFrame] = []

    for run_id in range(1, n_runs + 1):
        run_model_dir = model_root / f"run_{run_id}"
        run_results_dir = results_root / f"run_{run_id}"
        logger.info("=== DeBERTa run %d/%d ===", run_id, n_runs)

        if not args.skip_train:
            train_classifier(
                train_df=train_df,
                val_df=val_df,
                output_dir=str(run_model_dir),
            )

        clf = AnswerQualityClassifier(str(run_model_dir))
        metrics = _evaluate_filter_on_split(
            clf=clf,
            val_df=val_df,
            test_df=test_df,
            min_recall=min_recall,
            results_dir=run_results_dir,
            evaluator=evaluator,
        )
        metrics["run_id"] = run_id
        run_metrics.append(metrics)
        run_reports.append(pd.DataFrame(metrics["classification_report"]))
        with open(run_results_dir / "metrics.json", "w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2, default=str)

    no_filter = evaluator.compute_no_filter_baseline(test_df["label"].tolist())
    no_filter_pred = test_df.copy()
    no_filter_pred["predicted"] = 1
    no_filter_pred["confidence"] = 1.0
    no_filter_report = classification_report_by_dataset(no_filter_pred)
    no_filter_report.to_csv(
        results_root / "no_filter_classification_report.csv", index=False,
    )

    summary_report = average_classification_reports(run_reports)
    summary_report_path = results_root / "summary_classification_report.csv"
    summary_report.to_csv(summary_report_path, index=False)
    logger.info("Wrote %s", summary_report_path)

    summary: Dict[str, Any] = {
        "dataset": str(data_cfg["labeled_csv"]),
        "test_csv": data_cfg.get("test_csv"),
        "test_ratio": data_cfg["test_ratio"],
        "val_ratio": data_cfg["val_ratio"],
        "seed": data_cfg["seed"],
        "n_runs": n_runs,
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_test": len(test_df),
        "no_filter": _result_to_dict(no_filter),
        "deberta_runs": run_metrics,
        "deberta_mean_std": _mean_std(run_metrics),
        "summary_classification_report": summary_report.to_dict(orient="records"),
    }

    if not args.skip_nli:
        logger.info("=== Zero-shot NLI baseline on frozen test set ===")
        nli = NLIAnswerFilter()
        test_contexts = [_extract_top1_context(c) for c in test_df["context"].tolist()]
        nli_decisions = nli.predict_batch(test_contexts, test_df["answer"].tolist())
        # Use same min-FPR threshold selection on val for fair comparison
        val_contexts = [_extract_top1_context(c) for c in val_df["context"].tolist()]
        val_nli = nli.predict_batch(val_contexts, val_df["answer"].tolist())
        nli_thresh = select_threshold_min_fpr(
            [d.confidence for d in val_nli],
            val_df["label"].tolist(),
            min_recall=min_recall,
        )
        nli_preds = [
            d.confidence >= nli_thresh["threshold"] for d in nli_decisions
        ]
        nli_result = evaluator.evaluate(nli_preds, test_df["label"].tolist())
        nli_dir = results_root / "nli_zeroshot"
        nli_dir.mkdir(parents=True, exist_ok=True)
        nli_result.save(nli_dir / "nli_test_results.json")
        with open(nli_dir / "threshold_selection.json", "w", encoding="utf-8") as fh:
            json.dump(nli_thresh, fh, indent=2, default=str)

        nli_pred_df = test_df.copy()
        nli_pred_df["predicted"] = [int(p) for p in nli_preds]
        nli_pred_df["confidence"] = [d.confidence for d in nli_decisions]
        nli_report = classification_report_by_dataset(nli_pred_df)
        nli_report.to_csv(nli_dir / "classification_report.csv", index=False)

        summary["nli_zeroshot"] = _result_to_dict(nli_result)
        summary["nli_zeroshot"]["threshold"] = float(nli_thresh["threshold"])
        summary["nli_zeroshot"]["classification_report"] = nli_report.to_dict(
            orient="records",
        )

    deberta_label = "DeBERTa" if n_runs == 1 else "DeBERTa (mean±std)"
    comparison = {
        "No Filter": summary["no_filter"],
        "NLI zero-shot": summary.get("nli_zeroshot"),
        deberta_label: {
            k: f"{v['mean']:.4f}±{v['std']:.4f}"
            for k, v in summary["deberta_mean_std"].items()
        },
    }
    summary["comparison_table"] = comparison

    out_path = results_root / "summary.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)
    logger.info("Wrote %s", out_path)
    print(json.dumps(comparison, indent=2))
    print("\n=== summary_classification_report (DeBERTa) ===")
    print(summary_report.to_string(index=False))


if __name__ == "__main__":
    main()
