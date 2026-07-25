# AI Guardrail Service

CPU-first sensitive-entity detection for AI prompts.

The first proof of concept compares deterministic regex, a fine-tuned English
DistilBERT NER model, and Qwen3 0.6B zero-shot extraction through `llama.cpp`.
The initial Gateway integration is shadow-only.

## Documentation

- [Approved design](docs/superpowers/specs/2026-07-24-ai-guardrail-service-design.md)
- [Phase 1 implementation plan](docs/superpowers/plans/2026-07-25-offline-detector-evaluation.md)
- [Offline evaluation workflow](docs/offline-evaluation.md)

## Development

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,ml]"
python -m pytest
python -m ruff check .
```

Model weights, generated datasets, raw prompts, and full reports are not
committed to Git.
