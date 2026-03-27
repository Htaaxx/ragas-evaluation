I've created 6 focused Cursor rules in `.cursor/rules/`. Here's a summary of what each rule covers:

| Rule File | Scope | Key Enforcements |
|---|---|---|
| `project-structure.mdc` | **Always applies** | Pipeline-stage folder structure, separation of `/src` vs `/experiments` vs `/notebooks` |
| `python-standards.mdc` | `**/*.py` | PEP8, type hints, line/file/function limits, naming conventions, structured logging |
| `filtering-design.mdc` | `**/filtering/**` | Strategy pattern, `BaseFilter` abstract class, no hardcoded thresholds, swappable filters, decision logging |
| `notebook-rules.mdc` | `**/*.ipynb` | Code separation (import from `/src`, don't define core logic), required structure, descriptive naming, reproducibility |
| `config-and-reproducibility.mdc` | `**/configs/**` | YAML/JSON configs for all tunable values, seed fixing, dataset/model versioning, embedding caching |
| `evaluation-standards.mdc` | `**/evaluation/**`, `**/experiments/**` | Required metrics (P@K, R@K, MRR, nDCG, filter effectiveness), experiment isolation, unit testing |

**How they work:**
- `project-structure.mdc` is set to `alwaysApply: true` — it will guide every conversation
- The other 5 rules activate automatically when you're editing files matching their glob patterns (e.g., the filtering rules kick in when you touch files in `/filtering/`)

All rules follow the skill guidelines: concise (under 50 lines each), one concern per rule, actionable with concrete examples. The AI will now enforce these standards whenever you write or modify code in this project.