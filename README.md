# AI Guardrail Service

CPU-first sensitive-entity detection for AI prompts.

The proof of concept compares deterministic regex, fine-tuned English
DistilBERT NER models, and Qwen3 0.6B zero-shot extraction through
`llama.cpp`. The NER workflow supports reproducible v1 and v2 dataset releases;
v2 increases synthetic template and catalog diversity while keeping the
training seed fixed for a controlled comparison.

The initial Gateway integration is shadow-only.

## Documentation

- [Approved design](docs/superpowers/specs/2026-07-24-ai-guardrail-service-design.md)
- [Phase 1 implementation plan](docs/superpowers/plans/2026-07-25-offline-detector-evaluation.md)
- [Offline evaluation workflow](docs/offline-evaluation.md)
- [Dataset policy and generation](datasets/README.md)

## Development

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,ml]"
python -m pytest
python -m ruff check .
```

Standard tests do not download model weights or access the network.

## NER v2 quick start

Generate the deterministic v2 dataset:

```powershell
python -c "from pathlib import Path; from ai_guardrail.synthetic.generator import generate_dataset; print(generate_dataset(Path('config/synthetic-v2.yaml'), Path('datasets/generated/v2')))"
```

Train the trusted v2 release on CPU:

```powershell
python -m ai_guardrail.ner.train `
  --release-version v2 `
  --train datasets/generated/v2/train.jsonl `
  --validation datasets/generated/v2/validation.jsonl `
  --output artifacts/ai-guardrail-ner-en-v2 `
  --seed 20260725
```

Select the decision threshold using only the v2 validation split:

```powershell
python -m ai_guardrail.evaluation.threshold_cli `
  --validation datasets/generated/v2/validation.jsonl `
  --ner-model artifacts/ai-guardrail-ner-en-v2 `
  --output artifacts/thresholds/ai-guardrail-ner-en-v2.selected-threshold.json
```

The training command downloads
`distilbert/distilbert-base-cased`. On the reference development machine,
training 4,000 v2 examples for three epochs on CPU took about 24 minutes. See
the [offline evaluation workflow](docs/offline-evaluation.md) for the complete
v1, v2, Qwen, and benchmark procedure.

## Current NER POC result

The initial privacy-safe A/B evaluation used the same eight clean records for
both releases. One of the nine frozen challenge records was excluded before
inference because its content hash overlapped the v1 training provenance.

| Metric | v1 | v2 |
| --- | ---: | ---: |
| Original global threshold | 0.55 | 0.50 |
| Strict span F1 | 0.522 | 0.846 |
| Strict span recall | 0.462 | 0.846 |
| Exact-record accuracy | 25% | 75% |
| Sensitive-character miss rate | 26.83% | 1.46% |
| Extra-mask rate | 0% | 0% |

These eight examples provide a directional POC signal, not a statistically
conclusive quality estimate. A larger independently reviewed challenge release
is required before making a production-readiness decision. The table records
the original global-threshold A/B; the current evaluation workflow selects a
separate validation threshold for each entity type.

### Current v2 per-entity thresholds

The validation-selected schema-v2 threshold artifact uses:

| Entity | Threshold |
| --- | ---: |
| PERSON | 0.35 |
| ADDRESS | 0.90 |
| EMAIL | 0.45 |
| API_KEY | 0.95 |
| CUSTOMER_ID | 0.95 |
| INTERNAL_PROJECT | 0.60 |

Compared with the original global `0.50` policy, per-entity calibration
produced the following aggregate POC results:

| Metric | Global 0.50 | Per entity |
| --- | ---: | ---: |
| Validation strict span F1 (800 examples) | 0.876 | 0.906 |
| Challenge strict span F1 (13 examples) | 0.778 | 0.857 |
| Challenge exact-record accuracy | 69.2% | 76.9% |
| Challenge PERSON F1 | 0.500 | 0.667 |
| Generic hard-negative predictions | 0 | 0 |

The 13 challenge examples comprise the nine frozen v1 seed records and four
reviewed incremental v2 records; all are isolated from v2 training and
validation provenance. Per-entity thresholds recover the observed borderline
PERSON case and suppress high-confidence type-confusion false positives.
Compound addresses with alphanumeric unit identifiers remain a known
full-span recall limitation that requires future training-data improvement,
not further threshold tuning.

## Local artifacts and privacy

Model weights, generated datasets, raw prompts, threshold artifacts, and full
evaluation reports are not committed to Git. They remain under the ignored
`artifacts/`, `datasets/generated/`, and `evaluation/reports/` paths.

Use fictitious synthetic values only. Never put production prompts, customer
identifiers, credentials, or internal project names in repository data or
logs.
