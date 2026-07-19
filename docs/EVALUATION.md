# Evaluation conventions

## Filter task

Post-generation faithfulness check:

- **Premise** = retrieved `context`
- **Hypothesis** = generated `answer`
- Accept if the answer is supported by the context; reject otherwise

Retrieval quality is out of scope for the filter metrics below.

## North-star rule (DeBERTa / NLI)

Minimize **false positive rate** (accepting hallucinations) subject to **recall ≥ 0.70** on correct answers.

Implementation: `select_threshold_min_fpr(confidences, labels, min_recall=0.70)` in `src/filtering/deberta_filter_evaluator.py`.

- Threshold is chosen on the **validation** split.
- That threshold is applied to the frozen **test** split.
- Do **not** use argmax at 0.5 as the final decision rule.

Config key: `min_recall_for_threshold` in `configs/experiments/filter_training.yaml` / `configs/filtering/deberta_filter.yaml`.

## Required baselines (DeBERTa experiments)

| Strategy | Description |
|----------|-------------|
| No Filter | Accept everything |
| NLI zero-shot | `NLIAnswerFilter` with P(entailment), thresholded on val |
| Fine-tuned DeBERTa | `AnswerQualityClassifier` (`n_runs: 1` by default) |

## Shared classification table

All thesis methods should be able to emit (or average into) this CSV schema — same as RAGAS / LLM-judge summaries:

```text
dataset,num_samples,accepted,acceptance_rate,accuracy,precision,recall,f1,roc_auc
```

| Column | Definition |
|--------|------------|
| `dataset` | Display name (`ASQA`, `MS MARCO`, `WikiEval`, `Overall`) |
| `num_samples` | Rows in that slice |
| `accepted` | Count of filter accepts (`predicted == 1`) |
| `acceptance_rate` | `accepted / num_samples` |
| `accuracy` / `precision` / `recall` / `f1` | sklearn metrics vs `label` (positive = correct) |
| `roc_auc` | Prefer scores from `confidence` / P(faithful); fallback to hard preds if needed |

Helper: `classification_report_by_dataset()` in `src/filtering/deberta_filter_evaluator.py`.

DeBERTa 3-run average: `results/deberta_nli/summary_classification_report.csv`.

## Extra DeBERTa diagnostics

Also log when available:

- Confusion counts: TP, TN, FP, FN
- FPR, rejection recall / rejection rate
- Chosen threshold + val sweep (`threshold_selection.json`)

## Training gates (before full DeBERTa train)

See [DEBERTA_NLI_BASELINE.md](DEBERTA_NLI_BASELINE.md). Hard gate: `overfit_sanity_check` must reach train F1 ≥ 0.95 on 16 pairs. Keep `fp16: false`.
