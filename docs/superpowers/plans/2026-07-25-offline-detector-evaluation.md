# Offline Detector Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible CPU-first offline experiment that generates synthetic English sensitive-entity data and compares regex, a fine-tuned DistilBERT NER model, and Qwen3 0.6B zero-shot extraction.

**Architecture:** A small installable Python package owns shared domain schemas, deterministic synthetic data, detector adapters, and evaluation metrics. Regex and DistilBERT produce authoritative span candidates; Qwen3 runs through a separately managed `llama.cpp` OpenAI-compatible endpoint and remains observational. Standard tests use fakes and never download model weights.

**Tech Stack:** Python 3.11, Pydantic 2, pytest, Ruff, PyTorch, Hugging Face Transformers, httpx, jsonschema, psutil, Qwen3 0.6B Q4_K_M through `llama.cpp`.

## Global Constraints

- Language is English only: `language="en"`.
- Supported entity types are exactly `PERSON`, `ADDRESS`, `EMAIL`, `API_KEY`, `CUSTOMER_ID`, and `INTERNAL_PROJECT`.
- Use `distilbert/distilbert-base-cased` as the NER base checkpoint.
- Export the fine-tuned artifact as `ai-guardrail-ner-en-v1`.
- Use thirteen BIO labels: `O` plus `B-` and `I-` for each of the six entity types.
- Use Qwen3 0.6B GGUF Q4_K_M through `llama.cpp`, zero-shot, non-thinking, `temperature=0`, maximum input 512 tokens, and maximum output 96 tokens.
- Qwen is observational and never contributes to authoritative masking spans in Phase 1.
- Regex is authoritative for `EMAIL`, `API_KEY`, and known-format `CUSTOMER_ID`.
- Raw prompt text, entity text, Qwen raw output, generated datasets, reports, and model weights must not appear in application logs.
- Git must not contain generated datasets, benchmark reports, `.gguf`, `.onnx`, `.safetensors`, or model `.bin` files.
- Training, validation, and challenge template families must not overlap.
- Standard CI tests must not make network calls or download model weights.
- NER CPU P95 target is at most 150 ms in a container limited to 2 vCPU and 2 GiB memory.
- Qwen CPU P95 target is at most 2 seconds in a container limited to 4 vCPU and 4 GiB memory.
- The proof-of-concept targets are learning criteria, not production safety guarantees.

---

## Plan Boundary

This is Plan 1 of 3:

1. **This plan:** offline data, detectors, training, and comparative evaluation.
2. **Later plan:** local Guardrail HTTP service, validation/merge/mask pipeline, health, readiness, and Docker Compose.
3. **Later plan:** non-blocking `ai-gateway-service` shadow integration.

Do not add FastAPI, Kubernetes, Gateway changes, production enforcement, GLiNER, multilingual support, or Qwen fine-tuning while executing this plan.

## File Map

```text
ai-guardrail-service/
├── pyproject.toml                         # Package metadata and dependency groups
├── src/ai_guardrail/
│   ├── __init__.py                        # Package version
│   ├── domain.py                          # Shared Pydantic schemas and enums
│   ├── io.py                              # Safe JSONL reading and writing
│   ├── detectors/
│   │   ├── __init__.py
│   │   ├── base.py                        # Detector protocol
│   │   ├── regex.py                       # Deterministic detector
│   │   ├── ner.py                         # DistilBERT inference adapter
│   │   └── qwen.py                        # llama.cpp client and output validation
│   ├── synthetic/
│   │   ├── __init__.py
│   │   ├── catalog.py                     # Fictitious values only
│   │   ├── templates.py                   # Split-isolated template families
│   │   └── generator.py                   # Deterministic span-safe generation
│   ├── ner/
│   │   ├── __init__.py
│   │   ├── labels.py                      # Stable BIO label mapping
│   │   ├── alignment.py                   # Character span ↔ token label conversion
│   │   ├── train.py                       # Fine-tuning entry point
│   │   └── manifest.py                    # Reproducibility metadata and checksums
│   └── evaluation/
│       ├── __init__.py
│       ├── metrics.py                     # Span, character, latency metrics
│       ├── threshold_cli.py               # Validation threshold selection
│       ├── runner.py                      # Comparable detector benchmark
│       └── report.py                      # JSON and Markdown reports
├── config/
│   ├── regex-patterns.yaml                # Versioned deterministic patterns
│   ├── synthetic-v1.yaml                  # Dataset sizes and seed
│   └── qwen-entity-schema.json            # Constrained output contract
├── datasets/
│   ├── README.md                          # Privacy and review rules
│   └── challenge/
│       └── en-v1.seed.jsonl               # Small reviewed, fictitious smoke set
├── tests/
│   ├── test_domain.py
│   ├── test_io.py
│   ├── detectors/
│   │   ├── test_regex.py
│   │   ├── test_ner.py
│   │   └── test_qwen.py
│   ├── synthetic/
│   │   └── test_generator.py
│   ├── ner/
│   │   ├── test_alignment.py
│   │   └── test_manifest.py
│   └── evaluation/
│       ├── test_metrics.py
│       └── test_runner.py
└── docs/
    └── offline-evaluation.md               # Reproducible operator workflow
```

### Task 1: Package Foundation and Domain Contracts

**Files:**
- Create: `pyproject.toml`
- Create: `src/ai_guardrail/__init__.py`
- Create: `src/ai_guardrail/domain.py`
- Create: `src/ai_guardrail/io.py`
- Create: `src/ai_guardrail/detectors/__init__.py`
- Create: `src/ai_guardrail/detectors/base.py`
- Create: `tests/test_domain.py`
- Create: `tests/test_io.py`

**Interfaces:**
- Produces: `EntityType`, `DetectionSource`, `DatasetSplit`, `EntitySpan`, `LabeledExample`, `CandidateDetection`, `DetectorOutput`, and `Detector` protocol.
- Produces: `read_jsonl(path: Path) -> list[LabeledExample]`.
- Produces: `write_jsonl(path: Path, records: Iterable[LabeledExample]) -> None`.
- Consumes: no application code.

- [ ] **Step 1: Add packaging metadata and dependency groups**

Create `pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=75", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "ai-guardrail-service"
version = "0.1.0"
description = "CPU-first sensitive entity detection and evaluation"
readme = "README.md"
requires-python = ">=3.11,<3.13"
dependencies = [
  "httpx>=0.28,<1",
  "jsonschema>=4.23,<5",
  "pydantic>=2.9,<3",
  "PyYAML>=6,<7",
]

[project.optional-dependencies]
ml = [
  "accelerate>=1.3,<2",
  "huggingface-hub>=0.28,<1",
  "psutil>=6,<8",
  "torch>=2.5,<3",
  "transformers>=4.49,<5",
]
dev = [
  "pytest>=8.3,<9",
  "pytest-asyncio>=0.25,<1",
  "ruff>=0.8,<1",
]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
addopts = "-ra"
testpaths = ["tests"]
asyncio_mode = "auto"

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP"]
```

Create `src/ai_guardrail/__init__.py`:

```python
"""AI Guardrail offline evaluation package."""

__version__ = "0.1.0"
```

- [ ] **Step 2: Write failing domain tests**

Create `tests/test_domain.py`:

```python
import pytest
from pydantic import ValidationError

from ai_guardrail.domain import (
    CandidateDetection,
    DetectionSource,
    EntitySpan,
    EntityType,
    LabeledExample,
)


def test_entity_span_rejects_empty_or_reversed_range() -> None:
    with pytest.raises(ValidationError):
        EntitySpan(type=EntityType.PERSON, start=4, end=4)


def test_labeled_example_accepts_english_synthetic_record() -> None:
    record = LabeledExample(
        id="train-000001",
        language="en",
        text="Email jane@example.test.",
        entities=[EntitySpan(type=EntityType.EMAIL, start=6, end=23)],
        template_family="email-contact-train",
        generator_version="v1",
        split="train",
    )
    assert record.text[record.entities[0].start : record.entities[0].end] == "jane@example.test"


def test_candidate_detection_requires_bounded_confidence() -> None:
    with pytest.raises(ValidationError):
        CandidateDetection(
            message_index=0,
            type=EntityType.PERSON,
            start=0,
            end=4,
            source=DetectionSource.NER,
            confidence=1.1,
        )
```

- [ ] **Step 3: Run domain tests and verify RED**

Run:

```powershell
python -m pip install -e ".[dev]"
python -m pytest tests/test_domain.py -v
```

Expected: collection fails with `ModuleNotFoundError: No module named 'ai_guardrail.domain'`.

- [ ] **Step 4: Implement domain contracts**

Create `src/ai_guardrail/domain.py`:

```python
from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class EntityType(StrEnum):
    PERSON = "PERSON"
    ADDRESS = "ADDRESS"
    EMAIL = "EMAIL"
    API_KEY = "API_KEY"
    CUSTOMER_ID = "CUSTOMER_ID"
    INTERNAL_PROJECT = "INTERNAL_PROJECT"


class DetectionSource(StrEnum):
    REGEX = "regex"
    NER = "ner"
    QWEN = "qwen"


DatasetSplit = Literal["train", "validation", "challenge"]


class EntitySpan(BaseModel):
    type: EntityType
    start: int = Field(ge=0)
    end: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_range(self) -> "EntitySpan":
        if self.end <= self.start:
            raise ValueError("end must be greater than start")
        return self


class LabeledExample(BaseModel):
    id: str = Field(min_length=1)
    language: Literal["en"]
    text: str = Field(min_length=1)
    entities: list[EntitySpan]
    template_family: str = Field(min_length=1)
    generator_version: Literal["v1"]
    split: DatasetSplit

    @model_validator(mode="after")
    def validate_entity_bounds(self) -> "LabeledExample":
        ordered = sorted(self.entities, key=lambda entity: (entity.start, entity.end))
        previous_end = 0
        for entity in ordered:
            if entity.end > len(self.text):
                raise ValueError("entity span exceeds text length")
            if entity.start < previous_end:
                raise ValueError("gold entity spans must not overlap")
            previous_end = entity.end
        return self


class CandidateDetection(EntitySpan):
    message_index: int = Field(ge=0)
    source: DetectionSource
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class DetectorOutput(BaseModel):
    detector: str
    model_version: str
    status: Literal["success", "error", "timeout"]
    candidates: list[CandidateDetection] = Field(default_factory=list)
    latency_ms: float = Field(ge=0.0)
    invalid_candidate_count: int = Field(default=0, ge=0)
    error_code: str | None = None
```

Create `src/ai_guardrail/detectors/base.py`:

```python
from typing import Protocol

from ai_guardrail.domain import DetectorOutput


class Detector(Protocol):
    async def detect(self, text: str, message_index: int = 0) -> DetectorOutput:
        """Return candidates without logging or persisting input text."""
        ...
```

Create `src/ai_guardrail/detectors/__init__.py` as an empty package marker.

- [ ] **Step 5: Run domain tests and verify GREEN**

Run:

```powershell
python -m pytest tests/test_domain.py -v
```

Expected: `3 passed`.

- [ ] **Step 6: Write failing JSONL tests**

Create `tests/test_io.py`:

```python
from pathlib import Path

from ai_guardrail.domain import EntitySpan, EntityType, LabeledExample
from ai_guardrail.io import read_jsonl, write_jsonl


def test_jsonl_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    records = [
        LabeledExample(
            id="challenge-000001",
            language="en",
            text="Contact Jane Cooper.",
            entities=[EntitySpan(type=EntityType.PERSON, start=8, end=19)],
            template_family="person-contact-challenge",
            generator_version="v1",
            split="challenge",
        )
    ]

    write_jsonl(path, records)

    assert read_jsonl(path) == records
```

- [ ] **Step 7: Run JSONL test and verify RED**

Run:

```powershell
python -m pytest tests/test_io.py -v
```

Expected: collection fails with `ModuleNotFoundError: No module named 'ai_guardrail.io'`.

- [ ] **Step 8: Implement safe JSONL helpers**

Create `src/ai_guardrail/io.py`:

```python
from collections.abc import Iterable
from pathlib import Path

from ai_guardrail.domain import LabeledExample


def write_jsonl(path: Path, records: Iterable[LabeledExample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(record.model_dump_json())
            handle.write("\n")


def read_jsonl(path: Path) -> list[LabeledExample]:
    records: list[LabeledExample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(LabeledExample.model_validate_json(stripped))
            except ValueError as exc:
                raise ValueError(f"invalid JSONL record at line {line_number}") from exc
    return records
```

- [ ] **Step 9: Run foundation checks**

Run:

```powershell
python -m pytest tests/test_domain.py tests/test_io.py -v
python -m ruff check src tests
```

Expected: `4 passed`; Ruff exits `0`.

- [ ] **Step 10: Commit foundation**

```powershell
git add pyproject.toml src tests
git commit -m "feat: add offline evaluation domain contracts"
```

### Task 2: Deterministic Synthetic Dataset Generator

**Files:**
- Create: `src/ai_guardrail/synthetic/__init__.py`
- Create: `src/ai_guardrail/synthetic/catalog.py`
- Create: `src/ai_guardrail/synthetic/templates.py`
- Create: `src/ai_guardrail/synthetic/generator.py`
- Create: `config/synthetic-v1.yaml`
- Create: `datasets/README.md`
- Create: `datasets/catalog-sources.md`
- Create: `datasets/challenge/en-v1.seed.jsonl`
- Create: `tests/synthetic/test_generator.py`

**Interfaces:**
- Consumes: `EntityType`, `EntitySpan`, `LabeledExample`, and `write_jsonl`.
- Produces: `Template`, `render_template()`, `generate_split()`, `generate_dataset()`.
- Produces: generated files under ignored `datasets/generated/`.

- [ ] **Step 1: Write failing template rendering tests**

Create `tests/synthetic/test_generator.py`:

```python
from ai_guardrail.domain import EntityType
from ai_guardrail.synthetic.generator import generate_split, render_template
from ai_guardrail.synthetic.templates import Template


def test_render_template_calculates_exact_multi_entity_spans() -> None:
    template = Template(
        family="record-transfer-train",
        split="train",
        text="Send {PERSON}'s {CUSTOMER_ID} record to {INTERNAL_PROJECT}.",
    )

    record = render_template(
        template=template,
        values={
            EntityType.PERSON: ["Jane Cooper"],
            EntityType.CUSTOMER_ID: ["CUST-93821"],
            EntityType.INTERNAL_PROJECT: ["Project Orion"],
        },
        record_id="train-000001",
    )

    extracted = {
        entity.type: record.text[entity.start : entity.end] for entity in record.entities
    }
    assert extracted == {
        EntityType.PERSON: "Jane Cooper",
        EntityType.CUSTOMER_ID: "CUST-93821",
        EntityType.INTERNAL_PROJECT: "Project Orion",
    }


def test_template_families_are_isolated_by_split() -> None:
    from ai_guardrail.synthetic.templates import TEMPLATES

    families_by_split = {
        split: {template.family for template in TEMPLATES if template.split == split}
        for split in ("train", "validation", "challenge")
    }
    assert families_by_split["train"].isdisjoint(families_by_split["validation"])
    assert families_by_split["train"].isdisjoint(families_by_split["challenge"])
    assert families_by_split["validation"].isdisjoint(families_by_split["challenge"])


def test_catalog_values_are_isolated_by_split() -> None:
    from ai_guardrail.synthetic.catalog import CATALOG

    for entity_type in EntityType:
        train = set(CATALOG["train"][entity_type])
        validation = set(CATALOG["validation"][entity_type])
        challenge = set(CATALOG["challenge"][entity_type])
        assert train.isdisjoint(validation)
        assert train.isdisjoint(challenge)
        assert validation.isdisjoint(challenge)
```

- [ ] **Step 2: Run generator tests and verify RED**

Run:

```powershell
python -m pytest tests/synthetic/test_generator.py -v
```

Expected: collection fails because `ai_guardrail.synthetic` does not exist.

- [ ] **Step 3: Add fictitious catalogs and split-isolated templates**

Create `src/ai_guardrail/synthetic/catalog.py`:

```python
from ai_guardrail.domain import EntityType


CATALOG: dict[str, dict[EntityType, tuple[str, ...]]] = {
    "train": {
        EntityType.PERSON: ("Jane Cooper", "Marcus Reed", "jane cooper"),
        EntityType.ADDRESS: (
            "18 Willow Lane, Northbridge",
            "440 Cedar Avenue, Lakeview",
        ),
        EntityType.EMAIL: (
            "jane.cooper@example.test",
            "marcus.reed@example.test",
        ),
        EntityType.API_KEY: (
            "sk-test-A1B2C3D4E5F6G7H8",
            "sk test A1B2 C3D4 E5F6 G7H8",
        ),
        EntityType.CUSTOMER_ID: ("CUST-93821", "CUST-10482"),
        EntityType.INTERNAL_PROJECT: (
            "Project Orion",
            "project falcon",
        ),
    },
    "validation": {
        EntityType.PERSON: ("Elena Brooks", "Noah Bennett"),
        EntityType.ADDRESS: (
            "72 Juniper Road, Westhaven",
            "9 Harbor Street, Eastford",
        ),
        EntityType.EMAIL: (
            "elena.brooks@example.test",
            "noah.bennett@example.test",
        ),
        EntityType.API_KEY: (
            "api_test_Z9Y8X7W6V5U4T3S2",
            "pk-test-1122334455667788",
        ),
        EntityType.CUSTOMER_ID: ("CUST-77190", "CUST-55014"),
        EntityType.INTERNAL_PROJECT: (
            "Project Lantern",
            "Project Northstar",
        ),
    },
    "challenge": {
        EntityType.PERSON: ("Priya Wallace", "Owen Parker"),
        EntityType.ADDRESS: (
            "31 Maple Crescent, Stonehaven",
            "6 Orchard Way, Clearford",
        ),
        EntityType.EMAIL: (
            "priya.wallace@example.test",
            "owen.parker@example.test",
        ),
        EntityType.API_KEY: (
            "sk-test-Q1W2E3R4T5Y6U7I8",
            "api_test_M9N8B7V6C5X4Z3L2",
        ),
        EntityType.CUSTOMER_ID: ("CUST-66241", "CUST-88370"),
        EntityType.INTERNAL_PROJECT: (
            "Project Meridian",
            "Project Ember",
        ),
    },
}
```

Create `src/ai_guardrail/synthetic/templates.py`:

```python
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Template:
    family: str
    split: Literal["train", "validation", "challenge"]
    text: str


TEMPLATES = (
    Template(
        "record-transfer-train",
        "train",
        "Send {PERSON}'s {CUSTOMER_ID} record to {INTERNAL_PROJECT}.",
    ),
    Template(
        "contact-update-train",
        "train",
        "Update {PERSON} at {EMAIL} and note the address {ADDRESS}.",
    ),
    Template(
        "credential-copy-train",
        "train",
        "Copy {API_KEY} into the deployment note for {INTERNAL_PROJECT}.",
    ),
    Template(
        "ordinary-summary-negative-train",
        "train",
        "Summarize the quarterly report without changing its wording.",
    ),
    Template(
        "falcon-animal-negative-train",
        "train",
        "The peregrine falcon is one of the fastest birds.",
    ),
    Template(
        "meeting-summary-validation",
        "validation",
        "The owner is {PERSON}; follow up at {EMAIL} about {INTERNAL_PROJECT}.",
    ),
    Template(
        "account-review-validation",
        "validation",
        "Review customer {CUSTOMER_ID}, located at {ADDRESS}.",
    ),
    Template(
        "secure-config-validation",
        "validation",
        "The temporary credential for testing is {API_KEY}.",
    ),
    Template(
        "orion-constellation-negative-validation",
        "validation",
        "Orion is visible in the winter night sky.",
    ),
    Template(
        "codename-context-challenge",
        "challenge",
        "Use the codename {INTERNAL_PROJECT} in the briefing for {PERSON}.",
    ),
    Template(
        "mixed-contact-challenge",
        "challenge",
        "Mail {PERSON} at {ADDRESS}, or use {EMAIL}.",
    ),
    Template(
        "lantern-object-negative-challenge",
        "challenge",
        "The lantern on the porch needs a new battery.",
    ),
)
```

Create empty `src/ai_guardrail/synthetic/__init__.py`.

- [ ] **Step 4: Implement span-safe rendering**

Create `src/ai_guardrail/synthetic/generator.py`:

```python
from __future__ import annotations

import random
import re
from collections import defaultdict
from pathlib import Path

import yaml

from ai_guardrail.domain import EntitySpan, EntityType, LabeledExample
from ai_guardrail.io import write_jsonl
from ai_guardrail.synthetic.catalog import CATALOG
from ai_guardrail.synthetic.templates import TEMPLATES, Template

PLACEHOLDER = re.compile(r"\{([A-Z_]+)\}")


def render_template(
    *,
    template: Template,
    values: dict[EntityType, list[str]],
    record_id: str,
) -> LabeledExample:
    pieces: list[str] = []
    entities: list[EntitySpan] = []
    occurrence: dict[EntityType, int] = defaultdict(int)
    cursor = 0
    output_length = 0

    for match in PLACEHOLDER.finditer(template.text):
        literal = template.text[cursor : match.start()]
        pieces.append(literal)
        output_length += len(literal)

        entity_type = EntityType(match.group(1))
        value_index = occurrence[entity_type]
        value = values[entity_type][value_index]
        occurrence[entity_type] += 1

        start = output_length
        pieces.append(value)
        output_length += len(value)
        entities.append(EntitySpan(type=entity_type, start=start, end=output_length))
        cursor = match.end()

    pieces.append(template.text[cursor:])
    return LabeledExample(
        id=record_id,
        language="en",
        text="".join(pieces),
        entities=entities,
        template_family=template.family,
        generator_version="v1",
        split=template.split,
    )


def generate_split(*, split: str, count: int, seed: int) -> list[LabeledExample]:
    rng = random.Random(seed)
    templates = [template for template in TEMPLATES if template.split == split]
    if not templates:
        raise ValueError(f"no templates configured for split {split}")

    records: list[LabeledExample] = []
    for index in range(count):
        template = templates[index % len(templates)]
        required = [EntityType(name) for name in PLACEHOLDER.findall(template.text)]
        values: dict[EntityType, list[str]] = defaultdict(list)
        for entity_type in required:
            values[entity_type].append(rng.choice(CATALOG[split][entity_type]))
        records.append(
            render_template(
                template=template,
                values=dict(values),
                record_id=f"{split}-{index + 1:06d}",
            )
        )
    rng.shuffle(records)
    return records


def generate_dataset(config_path: Path, output_dir: Path) -> dict[str, int]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    seed = int(config["seed"])
    counts = {name: int(value) for name, value in config["counts"].items()}
    for offset, split in enumerate(("train", "validation", "challenge")):
        records = generate_split(split=split, count=counts[split], seed=seed + offset)
        filename = "challenge.candidates.jsonl" if split == "challenge" else f"{split}.jsonl"
        write_jsonl(output_dir / filename, records)
    return counts
```

- [ ] **Step 5: Add deterministic dataset configuration**

Create `config/synthetic-v1.yaml`:

```yaml
generator_version: v1
seed: 20260725
counts:
  train: 2000
  validation: 400
  challenge: 200
```

Create `datasets/README.md`:

````markdown
# Dataset policy

Only fictitious synthetic values may be used in this repository.

- `datasets/generated/` is reproducible output and is ignored by Git.
- `datasets/challenge/en-v1.seed.jsonl` is a small reviewed smoke fixture.
- No production prompt, customer identifier, credential, or copied internal
  project name may be committed.
- Every generated span must be verified by slicing the source text.
- A challenge release is accepted only after every line is manually reviewed
  for entity type, exact offsets, ambiguity, and absence of real information.
````

Create `datasets/catalog-sources.md`:

```markdown
# Synthetic catalog provenance

All names, addresses, customer IDs, API keys, email addresses, and project
codenames in `src/ai_guardrail/synthetic/catalog.py` were authored specifically
for this proof of concept. They were not copied from production traffic,
customer records, credential stores, or an external name/address dataset.

The `.test` email domain and visibly synthetic credential prefixes prevent
accidental use as real contact or authentication data.
```

Create `datasets/challenge/en-v1.seed.jsonl` with these exact reviewed records:

```jsonl
{"id":"challenge-seed-000001","language":"en","text":"Use the codename Project Falcon in the briefing.","entities":[{"type":"INTERNAL_PROJECT","start":17,"end":31}],"template_family":"manual-codename-challenge","generator_version":"v1","split":"challenge"}
{"id":"challenge-seed-000002","language":"en","text":"The peregrine falcon is one of the fastest birds.","entities":[],"template_family":"manual-falcon-negative-challenge","generator_version":"v1","split":"challenge"}
{"id":"challenge-seed-000003","language":"en","text":"Contact Jane Cooper at jane.cooper@example.test.","entities":[{"type":"PERSON","start":8,"end":19},{"type":"EMAIL","start":23,"end":47}],"template_family":"manual-contact-challenge","generator_version":"v1","split":"challenge"}
{"id":"challenge-seed-000004","language":"en","text":"Rotate sk-test-A1B2C3D4E5F6G7H8 before the demo.","entities":[{"type":"API_KEY","start":7,"end":31}],"template_family":"manual-api-key-challenge","generator_version":"v1","split":"challenge"}
```

- [ ] **Step 6: Run generator tests and verify GREEN**

Run:

```powershell
python -m pytest tests/synthetic/test_generator.py -v
```

Expected: `3 passed`.

- [ ] **Step 7: Add generated-span and reproducibility tests**

Append to `tests/synthetic/test_generator.py`:

```python
def test_generated_spans_always_slice_nonempty_source_text() -> None:
    records = generate_split(split="train", count=100, seed=17)
    for record in records:
        for entity in record.entities:
            assert record.text[entity.start : entity.end]


def test_generation_is_reproducible_for_same_seed() -> None:
    first = generate_split(split="validation", count=20, seed=29)
    second = generate_split(split="validation", count=20, seed=29)
    assert first == second
```

- [ ] **Step 8: Verify full configured generation**

Run:

```powershell
python -c "from pathlib import Path; from ai_guardrail.synthetic.generator import generate_dataset; print(generate_dataset(Path('config/synthetic-v1.yaml'), Path('datasets/generated/v1')))"
python -c "from pathlib import Path; from ai_guardrail.io import read_jsonl; print({p.stem: len(read_jsonl(p)) for p in sorted(Path('datasets/generated/v1').glob('*.jsonl'))})"
```

Expected:

```text
{'train': 2000, 'validation': 400, 'challenge': 200}
{'challenge.candidates': 200, 'train': 2000, 'validation': 400}
```

- [ ] **Step 9: Review and promote the challenge candidates**

Open `datasets/generated/v1/challenge.candidates.jsonl` and inspect every one of
the 200 lines. For each line, verify:

- all values are fictitious;
- every entity type is correct in context;
- `text[start:end]` is the intended complete entity;
- hard negatives contain no gold entity;
- no template family appears in train or validation.

After all 200 records pass review, promote the exact reviewed bytes:

```powershell
Copy-Item -LiteralPath datasets/generated/v1/challenge.candidates.jsonl -Destination datasets/generated/v1/challenge.reviewed.jsonl
Get-FileHash -Algorithm SHA256 datasets/generated/v1/challenge.reviewed.jsonl
```

Expected: the reviewed file contains exactly 200 valid JSONL records. Record the
printed SHA-256 in the benchmark report; do not commit the reviewed data.

- [ ] **Step 10: Run tests and lint**

Run:

```powershell
python -m pytest tests/synthetic tests/test_domain.py tests/test_io.py -v
python -m ruff check src tests
```

Expected: all selected tests pass; Ruff exits `0`.

- [ ] **Step 11: Commit synthetic data tooling**

```powershell
git add config/synthetic-v1.yaml datasets/README.md datasets/catalog-sources.md datasets/challenge/en-v1.seed.jsonl src/ai_guardrail/synthetic tests/synthetic
git commit -m "feat: generate deterministic synthetic entity data"
```

### Task 3: Regex Baseline Detector

**Files:**
- Create: `config/regex-patterns.yaml`
- Create: `src/ai_guardrail/detectors/regex.py`
- Create: `tests/detectors/test_regex.py`

**Interfaces:**
- Consumes: `CandidateDetection`, `DetectionSource`, `DetectorOutput`, `EntityType`.
- Produces: `RegexDetector.from_yaml(path: Path) -> RegexDetector`.
- Produces: `await RegexDetector.detect(text, message_index=0) -> DetectorOutput`.

- [ ] **Step 1: Write failing regex detector tests**

Create `tests/detectors/test_regex.py`:

```python
from pathlib import Path

import pytest

from ai_guardrail.domain import EntityType
from ai_guardrail.detectors.regex import RegexDetector


@pytest.mark.asyncio
async def test_regex_detects_authoritative_entity_spans(tmp_path: Path) -> None:
    config = tmp_path / "patterns.yaml"
    config.write_text(
        """
version: regex-v1
patterns:
  EMAIL:
    - '(?i)\\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\\.[A-Z]{2,}\\b'
  API_KEY:
    - '\\b(?:sk|pk|api)[-_][A-Za-z0-9_-]{16,}\\b'
  CUSTOMER_ID:
    - '\\bCUST-[0-9]{5,10}\\b'
""".strip(),
        encoding="utf-8",
    )
    detector = RegexDetector.from_yaml(config)
    text = "Email jane@example.test about CUST-93821 and rotate sk-test-A1B2C3D4E5F6G7H8."

    output = await detector.detect(text)

    assert output.status == "success"
    assert {(candidate.type, text[candidate.start : candidate.end]) for candidate in output.candidates} == {
        (EntityType.EMAIL, "jane@example.test"),
        (EntityType.CUSTOMER_ID, "CUST-93821"),
        (EntityType.API_KEY, "sk-test-A1B2C3D4E5F6G7H8"),
    }
    assert all(candidate.confidence is None for candidate in output.candidates)


def test_invalid_regex_fails_at_startup(tmp_path: Path) -> None:
    config = tmp_path / "patterns.yaml"
    config.write_text("version: regex-v1\npatterns:\n  EMAIL:\n    - '['\n", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid regex"):
        RegexDetector.from_yaml(config)
```

- [ ] **Step 2: Run regex tests and verify RED**

Run:

```powershell
python -m pytest tests/detectors/test_regex.py -v
```

Expected: collection fails because `ai_guardrail.detectors.regex` does not exist.

- [ ] **Step 3: Add production regex configuration**

Create `config/regex-patterns.yaml`:

```yaml
version: regex-v1
patterns:
  EMAIL:
    - "(?i)\\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\\.[A-Z]{2,}\\b"
  API_KEY:
    - "\\b(?:sk|pk|api)[-_][A-Za-z0-9_-]{16,}\\b"
  CUSTOMER_ID:
    - "\\bCUST-[0-9]{5,10}\\b"
```

- [ ] **Step 4: Implement regex detector**

Create `src/ai_guardrail/detectors/regex.py`:

```python
from __future__ import annotations

import re
import time
from pathlib import Path

import yaml

from ai_guardrail.domain import (
    CandidateDetection,
    DetectionSource,
    DetectorOutput,
    EntityType,
)


class RegexDetector:
    def __init__(
        self,
        *,
        version: str,
        patterns: dict[EntityType, tuple[re.Pattern[str], ...]],
    ) -> None:
        self.version = version
        self.patterns = patterns

    @classmethod
    def from_yaml(cls, path: Path) -> "RegexDetector":
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        compiled: dict[EntityType, tuple[re.Pattern[str], ...]] = {}
        for raw_type, raw_patterns in config["patterns"].items():
            entity_type = EntityType(raw_type)
            try:
                compiled[entity_type] = tuple(re.compile(pattern) for pattern in raw_patterns)
            except re.error as exc:
                raise ValueError(f"invalid regex for {entity_type}") from exc
        return cls(version=str(config["version"]), patterns=compiled)

    async def detect(self, text: str, message_index: int = 0) -> DetectorOutput:
        started = time.perf_counter()
        candidates: list[CandidateDetection] = []
        seen: set[tuple[EntityType, int, int]] = set()
        for entity_type, patterns in self.patterns.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    key = (entity_type, match.start(), match.end())
                    if key in seen:
                        continue
                    seen.add(key)
                    candidates.append(
                        CandidateDetection(
                            message_index=message_index,
                            type=entity_type,
                            start=match.start(),
                            end=match.end(),
                            source=DetectionSource.REGEX,
                        )
                    )
        candidates.sort(key=lambda item: (item.start, item.end, item.type))
        return DetectorOutput(
            detector=DetectionSource.REGEX,
            model_version=self.version,
            status="success",
            candidates=candidates,
            latency_ms=(time.perf_counter() - started) * 1000,
        )
```

- [ ] **Step 5: Run regex tests and verify GREEN**

Run:

```powershell
python -m pytest tests/detectors/test_regex.py -v
python -m ruff check src/ai_guardrail/detectors/regex.py tests/detectors/test_regex.py
```

Expected: `2 passed`; Ruff exits `0`.

- [ ] **Step 6: Commit regex baseline**

```powershell
git add config/regex-patterns.yaml src/ai_guardrail/detectors/regex.py tests/detectors/test_regex.py
git commit -m "feat: add deterministic regex detector"
```

### Task 4: BIO Labels and Character-to-Token Alignment

**Files:**
- Create: `src/ai_guardrail/ner/__init__.py`
- Create: `src/ai_guardrail/ner/labels.py`
- Create: `src/ai_guardrail/ner/alignment.py`
- Create: `tests/ner/test_alignment.py`

**Interfaces:**
- Consumes: `EntitySpan`, `EntityType`, `CandidateDetection`, `DetectionSource`.
- Produces: stable `LABEL_TO_ID`, `ID_TO_LABEL`.
- Produces: `align_spans_to_bio(offsets, entities) -> list[int]`.
- Produces: `decode_bio_predictions(...) -> list[CandidateDetection]`.

- [ ] **Step 1: Write failing alignment tests**

Create `tests/ner/test_alignment.py`:

```python
from ai_guardrail.domain import EntitySpan, EntityType
from ai_guardrail.ner.alignment import align_spans_to_bio, decode_bio_predictions
from ai_guardrail.ner.labels import LABEL_TO_ID


def test_aligns_character_span_to_bio_tokens() -> None:
    offsets = [(0, 0), (0, 4), (5, 9), (10, 15), (15, 16), (0, 0)]
    entities = [EntitySpan(type=EntityType.PERSON, start=5, end=15)]

    labels = align_spans_to_bio(offsets, entities)

    assert labels == [
        -100,
        LABEL_TO_ID["O"],
        LABEL_TO_ID["B-PERSON"],
        LABEL_TO_ID["I-PERSON"],
        LABEL_TO_ID["O"],
        -100,
    ]


def test_decodes_entity_with_minimum_token_confidence() -> None:
    text = "Jane Cooper"
    offsets = [(0, 4), (5, 11)]
    label_ids = [LABEL_TO_ID["B-PERSON"], LABEL_TO_ID["I-PERSON"]]
    probabilities = [0.98, 0.87]

    detections = decode_bio_predictions(
        text=text,
        offsets=offsets,
        label_ids=label_ids,
        probabilities=probabilities,
        message_index=2,
    )

    assert len(detections) == 1
    assert detections[0].start == 0
    assert detections[0].end == 11
    assert detections[0].confidence == 0.87
```

- [ ] **Step 2: Run alignment tests and verify RED**

Run:

```powershell
python -m pytest tests/ner/test_alignment.py -v
```

Expected: collection fails because `ai_guardrail.ner.alignment` does not exist.

- [ ] **Step 3: Implement stable label mapping**

Create `src/ai_guardrail/ner/labels.py`:

```python
from ai_guardrail.domain import EntityType

LABELS = ["O"]
for entity_type in EntityType:
    LABELS.extend((f"B-{entity_type.value}", f"I-{entity_type.value}"))

LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}

assert len(LABELS) == 13
```

Create empty `src/ai_guardrail/ner/__init__.py`.

- [ ] **Step 4: Implement alignment and decoding**

Create `src/ai_guardrail/ner/alignment.py`:

```python
from __future__ import annotations

from ai_guardrail.domain import (
    CandidateDetection,
    DetectionSource,
    EntitySpan,
    EntityType,
)
from ai_guardrail.ner.labels import ID_TO_LABEL, LABEL_TO_ID


def align_spans_to_bio(
    offsets: list[tuple[int, int]],
    entities: list[EntitySpan],
) -> list[int]:
    labels: list[int] = []
    for token_start, token_end in offsets:
        if token_start == token_end == 0:
            labels.append(-100)
            continue
        matches = [
            entity
            for entity in entities
            if token_start < entity.end and token_end > entity.start
        ]
        if not matches:
            labels.append(LABEL_TO_ID["O"])
            continue
        if len(matches) != 1:
            raise ValueError("token overlaps multiple gold entities")
        entity = matches[0]
        if token_start < entity.start or token_end > entity.end:
            raise ValueError("entity boundary does not align to token boundary")
        prefix = "B" if token_start == entity.start else "I"
        labels.append(LABEL_TO_ID[f"{prefix}-{entity.type.value}"])
    return labels


def decode_bio_predictions(
    *,
    text: str,
    offsets: list[tuple[int, int]],
    label_ids: list[int],
    probabilities: list[float],
    message_index: int,
) -> list[CandidateDetection]:
    detections: list[CandidateDetection] = []
    current_type: EntityType | None = None
    current_start = 0
    current_end = 0
    current_probabilities: list[float] = []

    def flush() -> None:
        nonlocal current_type, current_start, current_end, current_probabilities
        if current_type is not None:
            detections.append(
                CandidateDetection(
                    message_index=message_index,
                    type=current_type,
                    start=current_start,
                    end=current_end,
                    source=DetectionSource.NER,
                    confidence=min(current_probabilities),
                )
            )
        current_type = None
        current_probabilities = []

    for offset, label_id, probability in zip(
        offsets, label_ids, probabilities, strict=True
    ):
        token_start, token_end = offset
        if token_start == token_end == 0:
            continue
        label = ID_TO_LABEL[label_id]
        if label == "O":
            flush()
            continue
        prefix, raw_type = label.split("-", maxsplit=1)
        entity_type = EntityType(raw_type)
        if prefix == "B" or current_type != entity_type:
            flush()
            current_type = entity_type
            current_start = token_start
            current_end = token_end
            current_probabilities = [probability]
        else:
            current_end = token_end
            current_probabilities.append(probability)
    flush()

    for detection in detections:
        if not text[detection.start : detection.end]:
            raise ValueError("decoded an empty entity span")
    return detections
```

- [ ] **Step 5: Run alignment tests and verify GREEN**

Run:

```powershell
python -m pytest tests/ner/test_alignment.py -v
python -m ruff check src/ai_guardrail/ner tests/ner
```

Expected: `2 passed`; Ruff exits `0`.

- [ ] **Step 6: Commit NER alignment**

```powershell
git add src/ai_guardrail/ner tests/ner
git commit -m "feat: add BIO alignment and span decoding"
```

### Task 5: DistilBERT Training, Manifest, and CPU Inference

**Files:**
- Create: `src/ai_guardrail/ner/manifest.py`
- Create: `src/ai_guardrail/ner/train.py`
- Create: `src/ai_guardrail/detectors/ner.py`
- Create: `tests/ner/test_manifest.py`
- Create: `tests/detectors/test_ner.py`

**Interfaces:**
- Consumes: JSONL datasets, label mappings, alignment helpers.
- Produces: `build_manifest()`, `sha256_file()`, `write_manifest()`.
- Produces: `NerDetector.load(model_path, threshold)`.
- Produces: `await NerDetector.detect(text, message_index=0) -> DetectorOutput`.
- Produces CLI: `python -m ai_guardrail.ner.train`.

- [ ] **Step 1: Write failing manifest test**

Create `tests/ner/test_manifest.py`:

```python
from pathlib import Path

from ai_guardrail.ner.manifest import sha256_file


def test_sha256_file_is_reproducible(tmp_path: Path) -> None:
    artifact = tmp_path / "weights.bin"
    artifact.write_bytes(b"guardrail-model")

    assert sha256_file(artifact) == (
        "64c172d264fcea853f3556031a49bb15710ccedb51712322ca7bd6d997c4cb1a"
    )
```

- [ ] **Step 2: Run manifest test and verify RED**

Run:

```powershell
python -m pytest tests/ner/test_manifest.py -v
```

Expected: collection fails because `ai_guardrail.ner.manifest` does not exist.

- [ ] **Step 3: Implement artifact manifest helpers**

Create `src/ai_guardrail/ner/manifest.py`:

```python
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(
    *,
    base_checkpoint: str,
    base_revision: str,
    dataset_version: str,
    seed: int,
    threshold: float | None,
    metrics: dict[str, float],
    hyperparameters: dict[str, int | float],
) -> dict[str, Any]:
    return {
        "artifact_name": "ai-guardrail-ner-en-v1",
        "base_checkpoint": base_checkpoint,
        "base_revision": base_revision,
        "dataset_version": dataset_version,
        "seed": seed,
        "threshold": threshold,
        "metrics": metrics,
        "hyperparameters": hyperparameters,
        "python_version": platform.python_version(),
        "library_versions": {
            name: importlib.metadata.version(name)
            for name in ("accelerate", "torch", "transformers")
        },
    }


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
```

- [ ] **Step 4: Run manifest test and verify GREEN**

Run:

```powershell
python -m pytest tests/ner/test_manifest.py -v
```

Expected: `1 passed`.

- [ ] **Step 5: Write failing NER detector test with fakes**

Create `tests/detectors/test_ner.py`:

```python
import pytest
import torch

from ai_guardrail.detectors.ner import NerDetector
from ai_guardrail.domain import EntityType
from ai_guardrail.ner.labels import LABEL_TO_ID


class FakeBatch(dict):
    def to(self, device: str) -> "FakeBatch":
        return self


class FakeTokenizer:
    def __call__(self, text: str, **kwargs: object) -> FakeBatch:
        return FakeBatch(
            input_ids=torch.tensor([[101, 1, 2, 102]]),
            attention_mask=torch.tensor([[1, 1, 1, 1]]),
            offset_mapping=torch.tensor([[[0, 0], [0, 4], [5, 11], [0, 0]]]),
        )


class FakeOutput:
    def __init__(self) -> None:
        logits = torch.zeros((1, 4, 13))
        logits[0, 1, LABEL_TO_ID["B-PERSON"]] = 8
        logits[0, 2, LABEL_TO_ID["I-PERSON"]] = 8
        self.logits = logits


class FakeModel:
    def eval(self) -> "FakeModel":
        return self

    def __call__(self, **kwargs: object) -> FakeOutput:
        return FakeOutput()


@pytest.mark.asyncio
async def test_ner_detector_decodes_model_logits() -> None:
    detector = NerDetector(
        tokenizer=FakeTokenizer(),
        model=FakeModel(),
        model_version="ai-guardrail-ner-en-v1",
        threshold=0.5,
    )

    output = await detector.detect("Jane Cooper")

    assert output.status == "success"
    assert len(output.candidates) == 1
    assert output.candidates[0].type == EntityType.PERSON
    assert output.candidates[0].start == 0
    assert output.candidates[0].end == 11
```

- [ ] **Step 6: Run NER detector test and verify RED**

Run:

```powershell
python -m pip install -e ".[dev,ml]"
python -m pytest tests/detectors/test_ner.py -v
```

Expected: collection fails because `ai_guardrail.detectors.ner` does not exist.

- [ ] **Step 7: Implement CPU NER inference adapter**

Create `src/ai_guardrail/detectors/ner.py`:

```python
from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from ai_guardrail.domain import DetectionSource, DetectorOutput
from ai_guardrail.ner.alignment import decode_bio_predictions


class NerDetector:
    def __init__(
        self,
        *,
        tokenizer: Any,
        model: Any,
        model_version: str,
        threshold: float,
    ) -> None:
        self.tokenizer = tokenizer
        self.model = model.eval()
        self.model_version = model_version
        self.threshold = threshold

    @classmethod
    def load(cls, model_path: Path, threshold: float) -> "NerDetector":
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        model = AutoModelForTokenClassification.from_pretrained(model_path)
        return cls(
            tokenizer=tokenizer,
            model=model,
            model_version=model_path.name,
            threshold=threshold,
        )

    def _detect_sync(self, text: str, message_index: int) -> DetectorOutput:
        started = time.perf_counter()
        encoded = self.tokenizer(
            text,
            return_offsets_mapping=True,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        offsets_tensor = encoded.pop("offset_mapping")
        with torch.inference_mode():
            logits = self.model(**encoded).logits[0]
            probabilities = torch.softmax(logits, dim=-1)
            best_probabilities, label_ids = probabilities.max(dim=-1)
        offsets = [tuple(pair) for pair in offsets_tensor[0].tolist()]
        candidates = decode_bio_predictions(
            text=text,
            offsets=offsets,
            label_ids=label_ids.tolist(),
            probabilities=best_probabilities.tolist(),
            message_index=message_index,
        )
        candidates = [
            candidate
            for candidate in candidates
            if candidate.confidence is not None and candidate.confidence >= self.threshold
        ]
        return DetectorOutput(
            detector=DetectionSource.NER,
            model_version=self.model_version,
            status="success",
            candidates=candidates,
            latency_ms=(time.perf_counter() - started) * 1000,
        )

    async def detect(self, text: str, message_index: int = 0) -> DetectorOutput:
        return await asyncio.to_thread(self._detect_sync, text, message_index)
```

- [ ] **Step 8: Run NER detector test and verify GREEN**

Run:

```powershell
python -m pytest tests/detectors/test_ner.py tests/ner -v
python -m ruff check src/ai_guardrail/ner src/ai_guardrail/detectors/ner.py tests/ner tests/detectors/test_ner.py
```

Expected: all selected tests pass; Ruff exits `0`.

- [ ] **Step 9: Implement the training entry point**

Create `src/ai_guardrail/ner/train.py`:

```python
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import model_info
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

from ai_guardrail.io import read_jsonl
from ai_guardrail.ner.alignment import align_spans_to_bio
from ai_guardrail.ner.labels import ID_TO_LABEL, LABEL_TO_ID
from ai_guardrail.ner.manifest import build_manifest, sha256_file, write_manifest


class NerDataset:
    def __init__(self, path: Path, tokenizer: object) -> None:
        self.rows = []
        for example in read_jsonl(path):
            encoded = tokenizer(
                example.text,
                return_offsets_mapping=True,
                truncation=True,
                max_length=512,
            )
            offsets = [tuple(pair) for pair in encoded.pop("offset_mapping")]
            encoded["labels"] = align_spans_to_bio(offsets, example.entities)
            self.rows.append(encoded)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.rows[index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument(
        "--base-checkpoint",
        default="distilbert/distilbert-base-cased",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    resolved_revision = model_info(args.base_checkpoint).sha
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_checkpoint,
        revision=resolved_revision,
        use_fast=True,
    )
    model = AutoModelForTokenClassification.from_pretrained(
        args.base_checkpoint,
        revision=resolved_revision,
        num_labels=len(LABEL_TO_ID),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )
    train_dataset = NerDataset(args.train, tokenizer)
    validation_dataset = NerDataset(args.validation, tokenizer)
    training_args = TrainingArguments(
        output_dir=str(args.output),
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=args.seed,
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
    )
    trainer.train()
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    evaluation = {
        key: float(value)
        for key, value in trainer.evaluate().items()
        if isinstance(value, (int, float))
    }
    manifest = build_manifest(
        base_checkpoint=args.base_checkpoint,
        base_revision=resolved_revision,
        dataset_version="v1",
        seed=args.seed,
        threshold=None,
        metrics=evaluation,
        hyperparameters={
            "learning_rate": 2e-5,
            "train_batch_size": 16,
            "eval_batch_size": 16,
            "epochs": 3,
            "weight_decay": 0.01,
            "max_length": 512,
        },
    )
    manifest["artifact_checksums"] = {
        path.name: sha256_file(path)
        for path in sorted(args.output.iterdir())
        if path.is_file()
    }
    write_manifest(args.output / "training-manifest.json", manifest)


if __name__ == "__main__":
    main()
```

- [ ] **Step 10: Run the training module help without downloading a model**

Run:

```powershell
python -m ai_guardrail.ner.train --help
```

Expected: exit `0` and display required `--train`, `--validation`, and `--output`.

- [ ] **Step 11: Commit NER training and inference**

```powershell
git add src/ai_guardrail/ner src/ai_guardrail/detectors/ner.py tests/ner tests/detectors/test_ner.py
git commit -m "feat: add DistilBERT training and CPU inference"
```

### Task 6: Qwen3 llama.cpp Adapter and Candidate Validation

**Files:**
- Create: `config/qwen-entity-schema.json`
- Create: `src/ai_guardrail/detectors/qwen.py`
- Create: `tests/detectors/test_qwen.py`

**Interfaces:**
- Consumes: `CandidateDetection`, `DetectionSource`, `DetectorOutput`, `EntityType`.
- Produces: `resolve_occurrence(text, value, occurrence) -> tuple[int, int] | None`.
- Produces: `QwenDetector(base_url, model_version, timeout_seconds)`.
- Produces: `await QwenDetector.detect(text, message_index=0) -> DetectorOutput`.

- [ ] **Step 1: Write failing Qwen validation tests**

Create `tests/detectors/test_qwen.py`:

```python
import json

import httpx
import pytest

from ai_guardrail.detectors.qwen import QwenDetector, resolve_occurrence
from ai_guardrail.domain import EntityType


def test_repeated_text_requires_explicit_occurrence() -> None:
    text = "Project Falcon replaced Project Falcon."
    assert resolve_occurrence(text, "Project Falcon", None) is None
    assert resolve_occurrence(text, "Project Falcon", 2) == (24, 38)


@pytest.mark.asyncio
async def test_qwen_accepts_only_validated_source_substrings() -> None:
    response_content = {
        "entities": [
            {"type": "PERSON", "text": "Jane Cooper"},
            {"type": "INTERNAL_PROJECT", "text": "Project Mirage"},
        ]
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"tokens": list(range(12))})
        body = json.loads(request.content)
        assert body["temperature"] == 0
        assert body["max_tokens"] == 96
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": json.dumps(response_content)}}
                ]
            },
        )

    detector = QwenDetector(
        base_url="http://qwen.test",
        model_version="qwen3-0.6b-q4_k_m",
        timeout_seconds=2,
        transport=httpx.MockTransport(handler),
    )

    output = await detector.detect("Send Jane Cooper's record.")

    assert len(output.candidates) == 1
    assert output.candidates[0].type == EntityType.PERSON
    assert output.invalid_candidate_count == 1
```

- [ ] **Step 2: Run Qwen tests and verify RED**

Run:

```powershell
python -m pytest tests/detectors/test_qwen.py -v
```

Expected: collection fails because `ai_guardrail.detectors.qwen` does not exist.

- [ ] **Step 3: Add constrained JSON schema**

Create `config/qwen-entity-schema.json`:

```json
{
  "type": "object",
  "additionalProperties": false,
  "required": ["entities"],
  "properties": {
    "entities": {
      "type": "array",
      "maxItems": 32,
      "items": {
        "type": "object",
        "additionalProperties": false,
        "required": ["type", "text"],
        "properties": {
          "type": {
            "enum": [
              "PERSON",
              "ADDRESS",
              "EMAIL",
              "API_KEY",
              "CUSTOMER_ID",
              "INTERNAL_PROJECT"
            ]
          },
          "text": {
            "type": "string",
            "minLength": 1,
            "maxLength": 256
          },
          "occurrence": {
            "type": "integer",
            "minimum": 1,
            "maximum": 32
          }
        }
      }
    }
  }
}
```

- [ ] **Step 4: Implement Qwen adapter without logging model content**

Create `src/ai_guardrail/detectors/qwen.py`:

```python
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import httpx
import jsonschema

from ai_guardrail.domain import (
    CandidateDetection,
    DetectionSource,
    DetectorOutput,
    EntityType,
)

SYSTEM_PROMPT = """You identify sensitive entities in English text.
Allowed types: PERSON, ADDRESS, EMAIL, API_KEY, CUSTOMER_ID, INTERNAL_PROJECT.
Return JSON only. Each entity text must be an exact input substring.
If the same substring occurs more than once, include its 1-based occurrence.
Do not rewrite, explain, or infer text that is absent from the input."""


def resolve_occurrence(
    text: str,
    value: str,
    occurrence: int | None,
) -> tuple[int, int] | None:
    starts: list[int] = []
    cursor = 0
    while True:
        index = text.find(value, cursor)
        if index < 0:
            break
        starts.append(index)
        cursor = index + 1
    if not starts:
        return None
    if occurrence is None:
        if len(starts) != 1:
            return None
        selected = starts[0]
    elif occurrence < 1 or occurrence > len(starts):
        return None
    else:
        selected = starts[occurrence - 1]
    return selected, selected + len(value)


class QwenDetector:
    def __init__(
        self,
        *,
        base_url: str,
        model_version: str,
        timeout_seconds: float,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_version = model_version
        self.timeout_seconds = timeout_seconds
        self.transport = transport
        schema_path = Path("config/qwen-entity-schema.json")
        self.schema = json.loads(schema_path.read_text(encoding="utf-8"))

    async def detect(self, text: str, message_index: int = 0) -> DetectorOutput:
        started = time.perf_counter()
        try:
            async with httpx.AsyncClient(
                transport=self.transport,
                timeout=self.timeout_seconds,
            ) as client:
                token_response = await client.post(
                    f"{self.base_url}/tokenize",
                    json={"content": text, "add_special": False},
                )
                token_response.raise_for_status()
                if len(token_response.json()["tokens"]) > 512:
                    return DetectorOutput(
                        detector=DetectionSource.QWEN,
                        model_version=self.model_version,
                        status="error",
                        latency_ms=(time.perf_counter() - started) * 1000,
                        error_code="input_too_long",
                    )
                response = await client.post(
                    f"{self.base_url}/v1/chat/completions",
                    json={
                        "model": self.model_version,
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": f"{text}\n/no_think"},
                        ],
                        "temperature": 0,
                        "max_tokens": 96,
                        "response_format": {
                            "type": "json_schema",
                            "schema": self.schema,
                        },
                    },
                )
                response.raise_for_status()
                content = response.json()["choices"][0]["message"]["content"]
                payload: dict[str, Any] = json.loads(content)
                jsonschema.validate(payload, self.schema)
        except httpx.TimeoutException:
            return DetectorOutput(
                detector=DetectionSource.QWEN,
                model_version=self.model_version,
                status="timeout",
                latency_ms=(time.perf_counter() - started) * 1000,
                error_code="detector_timeout",
            )
        except (
            httpx.HTTPError,
            jsonschema.ValidationError,
            KeyError,
            TypeError,
            ValueError,
        ):
            return DetectorOutput(
                detector=DetectionSource.QWEN,
                model_version=self.model_version,
                status="error",
                latency_ms=(time.perf_counter() - started) * 1000,
                error_code="invalid_detector_response",
            )

        candidates: list[CandidateDetection] = []
        invalid_count = 0
        for raw in payload.get("entities", []):
            try:
                entity_type = EntityType(raw["type"])
                value = str(raw["text"])
                occurrence = raw.get("occurrence")
                span = resolve_occurrence(text, value, occurrence)
                if span is None:
                    invalid_count += 1
                    continue
                start, end = span
                candidates.append(
                    CandidateDetection(
                        message_index=message_index,
                        type=entity_type,
                        start=start,
                        end=end,
                        source=DetectionSource.QWEN,
                    )
                )
            except (KeyError, TypeError, ValueError):
                invalid_count += 1

        return DetectorOutput(
            detector=DetectionSource.QWEN,
            model_version=self.model_version,
            status="success",
            candidates=candidates,
            latency_ms=(time.perf_counter() - started) * 1000,
            invalid_candidate_count=invalid_count,
        )
```

- [ ] **Step 5: Run Qwen tests and verify GREEN**

Run:

```powershell
python -m pytest tests/detectors/test_qwen.py -v
python -m ruff check src/ai_guardrail/detectors/qwen.py tests/detectors/test_qwen.py
```

Expected: `2 passed`; Ruff exits `0`.

- [ ] **Step 6: Add timeout and malformed JSON tests**

Append to `tests/detectors/test_qwen.py`:

```python
@pytest.mark.asyncio
async def test_qwen_returns_safe_error_for_malformed_json() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"tokens": [1, 2, 3]})
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "not-json"}}]},
        )

    detector = QwenDetector(
        base_url="http://qwen.test",
        model_version="qwen3-0.6b-q4_k_m",
        timeout_seconds=2,
        transport=httpx.MockTransport(handler),
    )

    output = await detector.detect("Contact Jane Cooper.")

    assert output.status == "error"
    assert output.error_code == "invalid_detector_response"
    assert output.candidates == []
```

- [ ] **Step 7: Run all Qwen tests**

Run:

```powershell
python -m pytest tests/detectors/test_qwen.py -v
```

Expected: `3 passed`.

- [ ] **Step 8: Commit Qwen adapter**

```powershell
git add config/qwen-entity-schema.json src/ai_guardrail/detectors/qwen.py tests/detectors/test_qwen.py
git commit -m "feat: add validated Qwen detector adapter"
```

### Task 7: Comparative Metrics, Benchmark Runner, and Reports

**Files:**
- Create: `src/ai_guardrail/evaluation/__init__.py`
- Create: `src/ai_guardrail/evaluation/metrics.py`
- Create: `src/ai_guardrail/evaluation/resources.py`
- Create: `src/ai_guardrail/evaluation/runner.py`
- Create: `src/ai_guardrail/evaluation/report.py`
- Create: `src/ai_guardrail/evaluation/threshold_cli.py`
- Create: `src/ai_guardrail/evaluation/cli.py`
- Create: `tests/evaluation/test_metrics.py`
- Create: `tests/evaluation/test_runner.py`

**Interfaces:**
- Consumes: `LabeledExample`, `CandidateDetection`, detector protocol.
- Produces: `score_spans()`, `score_partial_spans()`, `score_characters()`,
  `score_by_entity()`, `select_threshold()`, and `nearest_rank_percentile()`.
- Produces: `BenchmarkRunner.run(examples) -> BenchmarkResult`.
- Produces: `AuthoritativeUnionDetector` for the `regex+ner` comparison.
- Produces: JSON and Markdown report writers.

- [ ] **Step 1: Write failing metric tests**

Create `tests/evaluation/test_metrics.py`:

```python
from ai_guardrail.domain import CandidateDetection, DetectionSource, EntitySpan, EntityType
from ai_guardrail.evaluation.metrics import (
    nearest_rank_percentile,
    score_by_entity,
    score_characters,
    score_partial_spans,
    score_spans,
    select_threshold,
)


def test_strict_span_metrics_count_type_and_boundary() -> None:
    gold = [
        EntitySpan(type=EntityType.PERSON, start=0, end=11),
        EntitySpan(type=EntityType.EMAIL, start=15, end=32),
    ]
    predicted = [
        CandidateDetection(
            message_index=0,
            type=EntityType.PERSON,
            start=0,
            end=11,
            source=DetectionSource.NER,
            confidence=0.9,
        ),
        CandidateDetection(
            message_index=0,
            type=EntityType.EMAIL,
            start=15,
            end=31,
            source=DetectionSource.REGEX,
        ),
    ]

    metrics = score_spans(gold, predicted)

    assert metrics == {"true_positive": 1, "false_positive": 1, "false_negative": 1}


def test_character_metrics_measure_missed_and_extra_characters() -> None:
    gold = [EntitySpan(type=EntityType.PERSON, start=0, end=4)]
    predicted = [
        CandidateDetection(
            message_index=0,
            type=EntityType.PERSON,
            start=0,
            end=3,
            source=DetectionSource.NER,
            confidence=0.9,
        )
    ]

    assert score_characters(10, gold, predicted) == {
        "sensitive_characters": 4,
        "missed_sensitive_characters": 1,
        "masked_characters": 3,
        "extra_masked_characters": 0,
    }


def test_nearest_rank_percentile_is_deterministic() -> None:
    assert nearest_rank_percentile([10, 20, 30, 40], 95) == 40
```

- [ ] **Step 2: Run metric tests and verify RED**

Run:

```powershell
python -m pytest tests/evaluation/test_metrics.py -v
```

Expected: collection fails because `ai_guardrail.evaluation.metrics` does not exist.

- [ ] **Step 3: Implement strict span, character, and latency metrics**

Create `src/ai_guardrail/evaluation/metrics.py`:

```python
from __future__ import annotations

import math

from ai_guardrail.domain import CandidateDetection, EntitySpan


def score_spans(
    gold: list[EntitySpan],
    predicted: list[CandidateDetection],
) -> dict[str, int]:
    gold_keys = {(entity.type, entity.start, entity.end) for entity in gold}
    predicted_keys = {
        (entity.type, entity.start, entity.end) for entity in predicted
    }
    return {
        "true_positive": len(gold_keys & predicted_keys),
        "false_positive": len(predicted_keys - gold_keys),
        "false_negative": len(gold_keys - predicted_keys),
    }


def score_characters(
    text_length: int,
    gold: list[EntitySpan],
    predicted: list[CandidateDetection],
) -> dict[str, int]:
    gold_mask = [False] * text_length
    predicted_mask = [False] * text_length
    for entity in gold:
        gold_mask[entity.start : entity.end] = [True] * (entity.end - entity.start)
    for entity in predicted:
        predicted_mask[entity.start : entity.end] = [True] * (
            entity.end - entity.start
        )
    return {
        "sensitive_characters": sum(gold_mask),
        "missed_sensitive_characters": sum(
            gold_value and not predicted_value
            for gold_value, predicted_value in zip(
                gold_mask, predicted_mask, strict=True
            )
        ),
        "masked_characters": sum(predicted_mask),
        "extra_masked_characters": sum(
            predicted_value and not gold_value
            for gold_value, predicted_value in zip(
                gold_mask, predicted_mask, strict=True
            )
        ),
    }


def precision_recall_f1(counts: dict[str, int]) -> dict[str, float]:
    tp = counts["true_positive"]
    fp = counts["false_positive"]
    fn = counts["false_negative"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def nearest_rank_percentile(values: list[float], percentile: int) -> float:
    if not values:
        raise ValueError("cannot calculate percentile of empty values")
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile / 100 * len(ordered)))
    return ordered[rank - 1]
```

Create empty `src/ai_guardrail/evaluation/__init__.py`.

- [ ] **Step 4: Run metric tests and verify GREEN**

Run:

```powershell
python -m pytest tests/evaluation/test_metrics.py -v
```

Expected: `3 passed`.

- [ ] **Step 5: Write failing benchmark runner test**

Create `tests/evaluation/test_runner.py`:

```python
import pytest

from ai_guardrail.domain import (
    CandidateDetection,
    DetectionSource,
    DetectorOutput,
    EntitySpan,
    EntityType,
    LabeledExample,
)
from ai_guardrail.evaluation.runner import AuthoritativeUnionDetector, BenchmarkRunner


class FakeDetector:
    async def detect(self, text: str, message_index: int = 0) -> DetectorOutput:
        return DetectorOutput(
            detector=DetectionSource.NER,
            model_version="fake-ner",
            status="success",
            latency_ms=12,
            candidates=[
                CandidateDetection(
                    message_index=message_index,
                    type=EntityType.PERSON,
                    start=0,
                    end=11,
                    source=DetectionSource.NER,
                    confidence=0.9,
                )
            ],
        )


@pytest.mark.asyncio
async def test_benchmark_runner_aggregates_metrics_without_storing_text() -> None:
    example = LabeledExample(
        id="challenge-1",
        language="en",
        text="Jane Cooper",
        entities=[EntitySpan(type=EntityType.PERSON, start=0, end=11)],
        template_family="manual-person-challenge",
        generator_version="v1",
        split="challenge",
    )
    runner = BenchmarkRunner({"ner": FakeDetector()})

    result = await runner.run([example])

    assert result["detectors"]["ner"]["strict"]["f1"] == 1.0
    assert result["detectors"]["ner"]["latency_ms"]["p95"] == 12
    assert "Jane Cooper" not in str(result)
```

- [ ] **Step 6: Run runner test and verify RED**

Run:

```powershell
python -m pytest tests/evaluation/test_runner.py -v
```

Expected: collection fails because `ai_guardrail.evaluation.runner` does not exist.

- [ ] **Step 7: Implement detector benchmark aggregation**

Create `src/ai_guardrail/evaluation/runner.py`:

```python
from __future__ import annotations

from typing import Any

from ai_guardrail.detectors.base import Detector
from ai_guardrail.domain import LabeledExample
from ai_guardrail.evaluation.metrics import (
    nearest_rank_percentile,
    precision_recall_f1,
    score_characters,
    score_spans,
)


class BenchmarkRunner:
    def __init__(self, detectors: dict[str, Detector]) -> None:
        self.detectors = detectors

    async def run(self, examples: list[LabeledExample]) -> dict[str, Any]:
        result: dict[str, Any] = {"example_count": len(examples), "detectors": {}}
        for name, detector in self.detectors.items():
            span_counts = {"true_positive": 0, "false_positive": 0, "false_negative": 0}
            character_counts = {
                "sensitive_characters": 0,
                "missed_sensitive_characters": 0,
                "masked_characters": 0,
                "extra_masked_characters": 0,
            }
            latencies: list[float] = []
            invalid_candidates = 0
            errors = 0
            timeouts = 0
            for example in examples:
                output = await detector.detect(example.text)
                latencies.append(output.latency_ms)
                invalid_candidates += output.invalid_candidate_count
                errors += int(output.status == "error")
                timeouts += int(output.status == "timeout")
                current_spans = score_spans(example.entities, output.candidates)
                current_characters = score_characters(
                    len(example.text),
                    example.entities,
                    output.candidates,
                )
                for key, value in current_spans.items():
                    span_counts[key] += value
                for key, value in current_characters.items():
                    character_counts[key] += value
            result["detectors"][name] = {
                "strict": {
                    **span_counts,
                    **precision_recall_f1(span_counts),
                },
                "characters": character_counts,
                "latency_ms": {
                    "p50": nearest_rank_percentile(latencies, 50),
                    "p95": nearest_rank_percentile(latencies, 95),
                    "p99": nearest_rank_percentile(latencies, 99),
                },
                "invalid_candidate_count": invalid_candidates,
                "error_count": errors,
                "timeout_count": timeouts,
            }
        return result
```

- [ ] **Step 8: Implement safe report output**

Create `src/ai_guardrail/evaluation/report.py`:

```python
import json
from pathlib import Path
from typing import Any


def write_json_report(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_markdown_report(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Offline Detector Evaluation",
        "",
        f"Examples: {result['example_count']}",
        "",
        "| Detector | Strict F1 | Partial F1 | P95 ms | Peak MiB | "
        "Consistency | Parse rate | Invalid | Errors | Timeouts |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    entity_sections: list[str] = []
    for name, metrics in result["detectors"].items():
        strict = metrics["strict"]
        partial = metrics["partial"]
        latency = metrics["latency_ms"]
        parse_rate = metrics["json_parse_rate"]
        parse_text = "" if parse_rate is None else f"{parse_rate:.4f}"
        lines.append(
            f"| {name} | {strict['f1']:.4f} | {partial['f1']:.4f} | "
            f"{latency['p95']:.2f} | {metrics['peak_rss_mib']:.2f} | "
            f"{metrics['consistency_rate']:.4f} | {parse_text} | "
            f"{metrics['invalid_candidate_count']} | {metrics['error_count']} | "
            f"{metrics['timeout_count']} |"
        )
        entity_sections.extend(
            [
                "",
                f"## {name} per entity",
                "",
                "| Entity | Precision | Recall | F1 |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for entity_type, entity_metrics in metrics["per_entity"].items():
            entity_sections.append(
                f"| {entity_type} | {entity_metrics['precision']:.4f} | "
                f"{entity_metrics['recall']:.4f} | "
                f"{entity_metrics['f1']:.4f} |"
            )
    lines.extend(entity_sections)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
```

- [ ] **Step 9: Run evaluation tests and lint**

Run:

```powershell
python -m pytest tests/evaluation -v
python -m ruff check src/ai_guardrail/evaluation tests/evaluation
```

Expected: `4 passed`; Ruff exits `0`.

- [ ] **Step 10: Write failing coverage tests for partial spans, per-type metrics, threshold selection, consistency, and the combined detector**

Append to `tests/evaluation/test_metrics.py`:

```python
def test_partial_span_counts_same_type_overlap_once() -> None:
    gold = [EntitySpan(type=EntityType.ADDRESS, start=0, end=20)]
    predicted = [
        CandidateDetection(
            message_index=0,
            type=EntityType.ADDRESS,
            start=5,
            end=20,
            source=DetectionSource.NER,
            confidence=0.8,
        )
    ]
    assert score_partial_spans(gold, predicted) == {
        "true_positive": 1,
        "false_positive": 0,
        "false_negative": 0,
    }


def test_per_entity_metrics_keep_rare_types_visible() -> None:
    gold = [EntitySpan(type=EntityType.INTERNAL_PROJECT, start=0, end=14)]
    predicted: list[CandidateDetection] = []
    result = score_by_entity(gold, predicted)
    assert result["INTERNAL_PROJECT"]["recall"] == 0.0
    assert result["INTERNAL_PROJECT"]["false_negative"] == 1


def test_threshold_selection_maximizes_strict_f1() -> None:
    gold = [[EntitySpan(type=EntityType.PERSON, start=0, end=11)]]
    predicted = [[
        CandidateDetection(
            message_index=0,
            type=EntityType.PERSON,
            start=0,
            end=11,
            source=DetectionSource.NER,
            confidence=0.8,
        ),
        CandidateDetection(
            message_index=0,
            type=EntityType.ADDRESS,
            start=0,
            end=4,
            source=DetectionSource.NER,
            confidence=0.6,
        ),
    ]]
    assert select_threshold(gold, predicted, [0.5, 0.7, 0.9]) == 0.7
```

Append to `tests/evaluation/test_runner.py`:

```python
class FakeRegexDetector:
    async def detect(self, text: str, message_index: int = 0) -> DetectorOutput:
        return DetectorOutput(
            detector="regex",
            model_version="regex-v1",
            status="success",
            latency_ms=1,
            candidates=[
                CandidateDetection(
                    message_index=message_index,
                    type=EntityType.EMAIL,
                    start=0,
                    end=len(text),
                    source=DetectionSource.REGEX,
                )
            ],
        )


class FakeEmailNerDetector:
    async def detect(self, text: str, message_index: int = 0) -> DetectorOutput:
        return DetectorOutput(
            detector="ner",
            model_version="fake-ner",
            status="success",
            latency_ms=5,
            candidates=[
                CandidateDetection(
                    message_index=message_index,
                    type=EntityType.EMAIL,
                    start=0,
                    end=len(text) - 1,
                    source=DetectionSource.NER,
                    confidence=0.8,
                )
            ],
        )


@pytest.mark.asyncio
async def test_authoritative_union_uses_regex_for_deterministic_type() -> None:
    detector = AuthoritativeUnionDetector(
        regex=FakeRegexDetector(),
        ner=FakeEmailNerDetector(),
    )
    output = await detector.detect("jane@example.test")
    assert len(output.candidates) == 1
    assert output.candidates[0].source == DetectionSource.REGEX


@pytest.mark.asyncio
async def test_runner_reports_repeat_consistency() -> None:
    example = LabeledExample(
        id="challenge-1",
        language="en",
        text="Jane Cooper",
        entities=[EntitySpan(type=EntityType.PERSON, start=0, end=11)],
        template_family="manual-person-challenge",
        generator_version="v1",
        split="challenge",
    )
    runner = BenchmarkRunner({"ner": FakeDetector()}, repetitions=2)
    result = await runner.run([example])
    assert result["detectors"]["ner"]["consistency_rate"] == 1.0
    assert result["detectors"]["ner"]["success_rate"] == 1.0
```

- [ ] **Step 11: Run expanded evaluation tests and verify RED**

Run:

```powershell
python -m pytest tests/evaluation -v
```

Expected: failures report missing `score_partial_spans`, `score_by_entity`,
`select_threshold`, `AuthoritativeUnionDetector`, and the `repetitions`
parameter.

- [ ] **Step 12: Add the missing metric functions**

Append to `src/ai_guardrail/evaluation/metrics.py`:

```python
from ai_guardrail.domain import EntityType


def score_partial_spans(
    gold: list[EntitySpan],
    predicted: list[CandidateDetection],
) -> dict[str, int]:
    unmatched = list(gold)
    true_positive = 0
    false_positive = 0
    for candidate in sorted(predicted, key=lambda item: (item.start, item.end)):
        matches = [
            entity
            for entity in unmatched
            if entity.type == candidate.type
            and candidate.start < entity.end
            and candidate.end > entity.start
        ]
        if not matches:
            false_positive += 1
            continue
        selected = max(
            matches,
            key=lambda entity: min(entity.end, candidate.end)
            - max(entity.start, candidate.start),
        )
        unmatched.remove(selected)
        true_positive += 1
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": len(unmatched),
    }


def score_by_entity(
    gold: list[EntitySpan],
    predicted: list[CandidateDetection],
) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    for entity_type in EntityType:
        counts = score_spans(
            [item for item in gold if item.type == entity_type],
            [item for item in predicted if item.type == entity_type],
        )
        result[entity_type.value] = {**counts, **precision_recall_f1(counts)}
    return result


def select_threshold(
    gold_by_example: list[list[EntitySpan]],
    predicted_by_example: list[list[CandidateDetection]],
    thresholds: list[float],
) -> float:
    scored: list[tuple[float, float, float]] = []
    for threshold in thresholds:
        counts = {"true_positive": 0, "false_positive": 0, "false_negative": 0}
        for gold, predicted in zip(gold_by_example, predicted_by_example, strict=True):
            filtered = [
                item
                for item in predicted
                if item.confidence is not None and item.confidence >= threshold
            ]
            current = score_spans(gold, filtered)
            for key in counts:
                counts[key] += current[key]
        metrics = precision_recall_f1(counts)
        scored.append((metrics["f1"], metrics["recall"], threshold))
    if not scored:
        raise ValueError("threshold list must not be empty")
    return max(scored)[2]
```

- [ ] **Step 13: Add peak-RSS sampling**

Create `src/ai_guardrail/evaluation/resources.py`:

```python
from __future__ import annotations

import threading
import time

import psutil


class PeakRssSampler:
    def __init__(
        self,
        interval_seconds: float = 0.01,
        process_id: int | None = None,
    ) -> None:
        self.interval_seconds = interval_seconds
        self.process = psutil.Process(process_id)
        self.peak_bytes = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample(self) -> None:
        while not self._stop.is_set():
            self.peak_bytes = max(self.peak_bytes, self.process.memory_info().rss)
            time.sleep(self.interval_seconds)

    def __enter__(self) -> "PeakRssSampler":
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *args: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1)
        self.peak_bytes = max(self.peak_bytes, self.process.memory_info().rss)

    @property
    def peak_mebibytes(self) -> float:
        return self.peak_bytes / 1024 / 1024
```

- [ ] **Step 14: Replace the runner with complete comparative behavior**

Replace `src/ai_guardrail/evaluation/runner.py` with:

```python
from __future__ import annotations

import asyncio
from typing import Any

from ai_guardrail.detectors.base import Detector
from ai_guardrail.domain import (
    CandidateDetection,
    DetectionSource,
    DetectorOutput,
    EntityType,
    LabeledExample,
)
from ai_guardrail.evaluation.metrics import (
    nearest_rank_percentile,
    precision_recall_f1,
    score_by_entity,
    score_characters,
    score_partial_spans,
    score_spans,
)
from ai_guardrail.evaluation.resources import PeakRssSampler

REGEX_AUTHORITATIVE_TYPES = {
    EntityType.EMAIL,
    EntityType.API_KEY,
    EntityType.CUSTOMER_ID,
}


def candidate_key(candidate: CandidateDetection) -> tuple[EntityType, int, int]:
    return candidate.type, candidate.start, candidate.end


class AuthoritativeUnionDetector:
    def __init__(self, *, regex: Detector, ner: Detector) -> None:
        self.regex = regex
        self.ner = ner

    async def detect(self, text: str, message_index: int = 0) -> DetectorOutput:
        regex_output, ner_output = await asyncio.gather(
            self.regex.detect(text, message_index),
            self.ner.detect(text, message_index),
        )
        candidates = list(regex_output.candidates)
        seen = {candidate_key(candidate) for candidate in candidates}
        for candidate in ner_output.candidates:
            if candidate.type in REGEX_AUTHORITATIVE_TYPES:
                continue
            if candidate_key(candidate) not in seen:
                candidates.append(candidate)
                seen.add(candidate_key(candidate))
        return DetectorOutput(
            detector="regex+ner",
            model_version=f"{regex_output.model_version}+{ner_output.model_version}",
            status="success",
            candidates=sorted(candidates, key=lambda item: (item.start, item.end)),
            latency_ms=max(regex_output.latency_ms, ner_output.latency_ms),
            invalid_candidate_count=(
                regex_output.invalid_candidate_count + ner_output.invalid_candidate_count
            ),
        )


class BenchmarkRunner:
    def __init__(
        self,
        detectors: dict[str, Detector],
        repetitions: int = 1,
        resource_pids: dict[str, int] | None = None,
    ) -> None:
        if repetitions < 1:
            raise ValueError("repetitions must be at least one")
        self.detectors = detectors
        self.repetitions = repetitions
        self.resource_pids = resource_pids or {}

    async def run(self, examples: list[LabeledExample]) -> dict[str, Any]:
        result: dict[str, Any] = {"example_count": len(examples), "detectors": {}}
        for name, detector in self.detectors.items():
            span_counts = {"true_positive": 0, "false_positive": 0, "false_negative": 0}
            partial_counts = {
                "true_positive": 0,
                "false_positive": 0,
                "false_negative": 0,
            }
            character_counts = {
                "sensitive_characters": 0,
                "missed_sensitive_characters": 0,
                "masked_characters": 0,
                "extra_masked_characters": 0,
            }
            per_type_counts = {
                entity_type.value: {
                    "true_positive": 0,
                    "false_positive": 0,
                    "false_negative": 0,
                }
                for entity_type in EntityType
            }
            latencies: list[float] = []
            invalid_candidates = 0
            errors = 0
            timeouts = 0
            successes = 0
            consistent_comparisons = 0
            total_comparisons = 0
            with PeakRssSampler(process_id=self.resource_pids.get(name)) as memory:
                for example in examples:
                    outputs = [
                        await detector.detect(example.text)
                        for _ in range(self.repetitions)
                    ]
                    output = outputs[0]
                    baseline = {candidate_key(item) for item in output.candidates}
                    for repeated in outputs[1:]:
                        total_comparisons += 1
                        repeated_keys = {
                            candidate_key(item) for item in repeated.candidates
                        }
                        consistent_comparisons += int(repeated_keys == baseline)
                    latencies.extend(item.latency_ms for item in outputs)
                    invalid_candidates += sum(
                        item.invalid_candidate_count for item in outputs
                    )
                    errors += sum(item.status == "error" for item in outputs)
                    timeouts += sum(item.status == "timeout" for item in outputs)
                    successes += sum(item.status == "success" for item in outputs)
                    current_spans = score_spans(example.entities, output.candidates)
                    current_partial = score_partial_spans(
                        example.entities, output.candidates
                    )
                    current_characters = score_characters(
                        len(example.text),
                        example.entities,
                        output.candidates,
                    )
                    for key in span_counts:
                        span_counts[key] += current_spans[key]
                        partial_counts[key] += current_partial[key]
                    for key in character_counts:
                        character_counts[key] += current_characters[key]
                    current_by_entity = score_by_entity(
                        example.entities, output.candidates
                    )
                    for entity_type, metrics in current_by_entity.items():
                        for key in (
                            "true_positive",
                            "false_positive",
                            "false_negative",
                        ):
                            per_type_counts[entity_type][key] += int(metrics[key])
            attempts = len(examples) * self.repetitions
            per_entity = {
                entity_type: {
                    **counts,
                    **precision_recall_f1(counts),
                }
                for entity_type, counts in per_type_counts.items()
            }
            result["detectors"][name] = {
                "strict": {**span_counts, **precision_recall_f1(span_counts)},
                "partial": {**partial_counts, **precision_recall_f1(partial_counts)},
                "per_entity": per_entity,
                "characters": character_counts,
                "latency_ms": {
                    "p50": nearest_rank_percentile(latencies, 50),
                    "p95": nearest_rank_percentile(latencies, 95),
                    "p99": nearest_rank_percentile(latencies, 99),
                },
                "peak_rss_mib": memory.peak_mebibytes,
                "success_rate": successes / attempts,
                "json_parse_rate": successes / attempts if name == "qwen" else None,
                "consistency_rate": (
                    consistent_comparisons / total_comparisons
                    if total_comparisons
                    else 1.0
                ),
                "invalid_candidate_count": invalid_candidates,
                "error_count": errors,
                "timeout_count": timeouts,
            }
        return result
```

- [ ] **Step 15: Run expanded evaluation tests and verify GREEN**

Run:

```powershell
python -m pytest tests/evaluation -v
python -m ruff check src/ai_guardrail/evaluation tests/evaluation
```

Expected: `9 passed`; Ruff exits `0`.

- [ ] **Step 16: Add deterministic validation threshold selection**

Create `src/ai_guardrail/evaluation/threshold_cli.py`:

```python
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path

from ai_guardrail.detectors.ner import NerDetector
from ai_guardrail.evaluation.metrics import select_threshold
from ai_guardrail.io import read_jsonl

THRESHOLDS = [value / 100 for value in range(50, 100, 5)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--ner-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


async def run() -> None:
    args = parse_args()
    examples = read_jsonl(args.validation)
    detector = NerDetector.load(args.ner_model, threshold=0.0)
    predictions = [
        (await detector.detect(example.text)).candidates for example in examples
    ]
    threshold = select_threshold(
        [example.entities for example in examples],
        predictions,
        THRESHOLDS,
    )
    payload = {
        "model_version": args.ner_model.name,
        "validation_sha256": hashlib.sha256(args.validation.read_bytes()).hexdigest(),
        "candidate_thresholds": THRESHOLDS,
        "selected_threshold": threshold,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
```

Run:

```powershell
python -m ai_guardrail.evaluation.threshold_cli --help
```

Expected: exit `0` and display `--validation`, `--ner-model`, and `--output`.

- [ ] **Step 17: Add a runnable benchmark CLI**

Create `src/ai_guardrail/evaluation/cli.py`:

```python
from __future__ import annotations

import argparse
import asyncio
import hashlib
import platform
from pathlib import Path

from ai_guardrail.detectors.ner import NerDetector
from ai_guardrail.detectors.qwen import QwenDetector
from ai_guardrail.detectors.regex import RegexDetector
from ai_guardrail.evaluation.report import write_json_report, write_markdown_report
from ai_guardrail.evaluation.runner import AuthoritativeUnionDetector, BenchmarkRunner
from ai_guardrail.io import read_jsonl


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--challenge", type=Path, required=True)
    parser.add_argument("--regex-config", type=Path, required=True)
    parser.add_argument("--ner-model", type=Path, required=True)
    parser.add_argument("--ner-threshold", type=float, required=True)
    parser.add_argument("--qwen-url", required=True)
    parser.add_argument("--qwen-model", default="qwen3-0.6b-q4_k_m")
    parser.add_argument("--qwen-timeout", type=float, default=2.0)
    parser.add_argument("--qwen-pid", type=int, required=True)
    parser.add_argument("--qwen-sha256", required=True)
    parser.add_argument("--llama-version", required=True)
    parser.add_argument("--guardrail-cpu-limit", type=int, required=True)
    parser.add_argument("--guardrail-memory-limit-mib", type=int, required=True)
    parser.add_argument("--qwen-cpu-limit", type=int, required=True)
    parser.add_argument("--qwen-memory-limit-mib", type=int, required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


async def run() -> None:
    args = parse_args()
    regex = RegexDetector.from_yaml(args.regex_config)
    ner = NerDetector.load(args.ner_model, args.ner_threshold)
    qwen = QwenDetector(
        base_url=args.qwen_url,
        model_version=args.qwen_model,
        timeout_seconds=args.qwen_timeout,
    )
    runner = BenchmarkRunner(
        {
            "regex": regex,
            "ner": ner,
            "qwen": qwen,
            "regex+ner": AuthoritativeUnionDetector(regex=regex, ner=ner),
        },
        repetitions=args.repetitions,
        resource_pids={"qwen": args.qwen_pid},
    )
    result = await runner.run(read_jsonl(args.challenge))
    result["environment"] = {
        "host_processor": platform.processor(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "challenge_sha256": sha256(args.challenge),
        "ner_model_path": args.ner_model.name,
        "ner_manifest_sha256": sha256(
            args.ner_model / "training-manifest.json"
        ),
        "qwen_model": args.qwen_model,
        "qwen_sha256": args.qwen_sha256,
        "llama_version": args.llama_version,
        "guardrail_cpu_limit": args.guardrail_cpu_limit,
        "guardrail_memory_limit_mib": args.guardrail_memory_limit_mib,
        "qwen_cpu_limit": args.qwen_cpu_limit,
        "qwen_memory_limit_mib": args.qwen_memory_limit_mib,
        "qwen_context_tokens": 1024,
        "qwen_input_limit_tokens": 512,
        "qwen_output_tokens": 96,
        "repetitions": args.repetitions,
    }
    write_json_report(args.output_dir / "report.json", result)
    write_markdown_report(args.output_dir / "report.md", result)


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
```

- [ ] **Step 18: Verify benchmark CLI help without models**

Run:

```powershell
python -m ai_guardrail.evaluation.cli --help
```

Expected: exit `0` and display all required model, challenge, and output
arguments.

- [ ] **Step 19: Commit evaluation framework**

```powershell
git add src/ai_guardrail/evaluation tests/evaluation
git commit -m "feat: add comparable detector evaluation"
```

### Task 8: Operator Workflow, Privacy Check, and Phase 1 Verification

**Files:**
- Modify: `README.md`
- Create: `docs/offline-evaluation.md`
- Modify: `tests/detectors/test_qwen.py`

**Interfaces:**
- Consumes: all Phase 1 package interfaces.
- Produces: a reproducible documented workflow and final verification evidence.

- [ ] **Step 1: Add a log privacy regression test**

Append to `tests/detectors/test_qwen.py`:

```python
@pytest.mark.asyncio
async def test_qwen_detector_does_not_log_prompt_or_entity(
    caplog: pytest.LogCaptureFixture,
) -> None:
    secret_prompt = "Rotate sk-test-NOT-FOR-LOGGING-1234."

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"tokens": [1, 2, 3]})
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"entities":[]}'}}]},
        )

    detector = QwenDetector(
        base_url="http://qwen.test",
        model_version="qwen3-0.6b-q4_k_m",
        timeout_seconds=2,
        transport=httpx.MockTransport(handler),
    )
    await detector.detect(secret_prompt)

    assert secret_prompt not in caplog.text
    assert "NOT-FOR-LOGGING" not in caplog.text
```

- [ ] **Step 2: Run privacy test**

Run:

```powershell
python -m pytest tests/detectors/test_qwen.py::test_qwen_detector_does_not_log_prompt_or_entity -v
```

Expected: `1 passed`.

- [ ] **Step 3: Document the exact offline workflow**

Create `docs/offline-evaluation.md`:

````markdown
# Offline evaluation workflow

## 1. Create the environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev,ml]"
```

## 2. Generate synthetic data

```powershell
python -c "from pathlib import Path; from ai_guardrail.synthetic.generator import generate_dataset; print(generate_dataset(Path('config/synthetic-v1.yaml'), Path('datasets/generated/v1')))"
```

Generated data remains under `datasets/generated/` and is not committed.

## 3. Train DistilBERT

```powershell
python -m ai_guardrail.ner.train `
  --train datasets/generated/v1/train.jsonl `
  --validation datasets/generated/v1/validation.jsonl `
  --output artifacts/ai-guardrail-ner-en-v1 `
  --seed 20260725
```

Training downloads `distilbert/distilbert-base-cased`; standard tests do not.

## 4. Start Qwen separately

Provide `Qwen3-0.6B-Q4_K_M.gguf` through the approved artifact channel, then
start `llama-server` with a 1,024-token context, a 512-token detector input
limit, and prompt logging disabled:

```powershell
.\tools\llama-server.exe `
  -m .\models\Qwen3-0.6B-Q4_K_M.gguf `
  -c 1024 `
  --host 127.0.0.1 `
  --port 8080 `
  --parallel 1 `
  --log-disable
```

## 5. Select the NER threshold

```powershell
python -m ai_guardrail.evaluation.threshold_cli `
  --validation datasets/generated/v1/validation.jsonl `
  --ner-model artifacts/ai-guardrail-ner-en-v1 `
  --output artifacts/ai-guardrail-ner-en-v1/selected-threshold.json
```

The challenge set is not used for threshold selection.

## 6. Evaluate

```powershell
$threshold = (Get-Content artifacts/ai-guardrail-ner-en-v1/selected-threshold.json | ConvertFrom-Json).selected_threshold
$qwenProcess = Get-Process llama-server
$qwenHash = (Get-FileHash -Algorithm SHA256 models/Qwen3-0.6B-Q4_K_M.gguf).Hash.ToLowerInvariant()
$llamaVersion = (& .\tools\llama-server.exe --version | Select-Object -First 1)

python -m ai_guardrail.evaluation.cli `
  --challenge datasets/generated/v1/challenge.reviewed.jsonl `
  --regex-config config/regex-patterns.yaml `
  --ner-model artifacts/ai-guardrail-ner-en-v1 `
  --ner-threshold $threshold `
  --qwen-url http://127.0.0.1:8080 `
  --qwen-pid $qwenProcess.Id `
  --qwen-sha256 $qwenHash `
  --llama-version $llamaVersion `
  --guardrail-cpu-limit 2 `
  --guardrail-memory-limit-mib 2048 `
  --qwen-cpu-limit 4 `
  --qwen-memory-limit-mib 4096 `
  --repetitions 3 `
  --output-dir evaluation/reports/v1
```

Store JSON and Markdown reports under `evaluation/reports/`; the directory is
ignored by Git. Reports contain aggregate metrics only and no example text.

## 7. Acceptance review

- Qwen JSON parse rate is at least 99%.
- No invalid Qwen candidate enters normalized output.
- `PERSON` and `ADDRESS` strict-span F1 are at least 0.85.
- `CUSTOMER_ID` and `INTERNAL_PROJECT` strict-span F1 are at least 0.75.
- NER P95 is at most 150 ms with 2 vCPU and 2 GiB.
- Qwen P95 is at most 2 seconds with 4 vCPU and 4 GiB.
- Captured logs contain no prompt or entity text.

Failure of a target is an experiment result, not permission to alter the target
or enable masking.
````

- [ ] **Step 4: Update the project README**

Replace the design-only text in `README.md` with:

````markdown
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
````

- [ ] **Step 5: Run the full standard verification suite**

Run:

```powershell
python -m pytest -v
python -m ruff check .
git status --short
```

Expected:

- all tests pass with zero failures;
- Ruff exits `0`;
- Git shows only the intended Task 8 documentation and test changes.

- [ ] **Step 6: Verify ignored artifact boundaries**

Run:

```powershell
git check-ignore datasets/generated/sample.jsonl
git check-ignore evaluation/reports/report.json
git check-ignore artifacts/ai-guardrail-ner-en-v1/model.safetensors
git check-ignore models/Qwen3-0.6B-Q4_K_M.gguf
```

Expected: all four paths are printed, proving they are ignored.

- [ ] **Step 7: Commit Phase 1 documentation and privacy gate**

```powershell
git add README.md docs/offline-evaluation.md tests/detectors/test_qwen.py
git commit -m "docs: add offline evaluation workflow"
```

- [ ] **Step 8: Record final Phase 1 evidence**

Run:

```powershell
git log --oneline --decorate -8
git status --porcelain=v1
```

Expected:

- commits exist for domain contracts, synthetic data, regex, NER alignment,
  NER training, Qwen adapter, evaluation framework, and workflow;
- working tree output is empty.

## Phase 1 Completion Gate

Phase 1 implementation is complete only when:

- all eight tasks have their own passing test evidence and commit;
- the generated split counts are 2,000/400/200;
- template-family isolation tests pass;
- standard tests require no model downloads or network access;
- the reviewed challenge release is distinct from generated training and
  validation families;
- DistilBERT training produces `ai-guardrail-ner-en-v1`, a training manifest,
  and a validation-selected threshold artifact;
- Qwen output is schema-constrained and every candidate is source-validated;
- reports contain aggregate data only;
- measured metrics and resource evidence are recorded without changing the
  agreed acceptance targets.
