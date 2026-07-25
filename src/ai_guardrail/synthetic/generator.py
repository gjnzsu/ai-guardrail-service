from __future__ import annotations

import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, StrictInt, ValidationError

from ai_guardrail.domain import EntitySpan, EntityType, LabeledExample
from ai_guardrail.io import write_jsonl
from ai_guardrail.synthetic.catalog import CATALOG
from ai_guardrail.synthetic.templates import TEMPLATES, Template

PLACEHOLDER = re.compile(r"\{([A-Z_]+)\}")


class _SyntheticCounts(BaseModel):
    model_config = ConfigDict(extra="forbid")

    train: StrictInt = Field(gt=0)
    validation: StrictInt = Field(gt=0)
    challenge: StrictInt = Field(gt=0)


class _SyntheticConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    generator_version: Literal["v1"]
    seed: StrictInt
    counts: _SyntheticCounts


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
    try:
        config = _SyntheticConfig.model_validate(
            yaml.safe_load(config_path.read_text(encoding="utf-8"))
        )
    except (OSError, UnicodeError, yaml.YAMLError, ValidationError):
        raise ValueError("invalid synthetic dataset configuration") from None
    seed = config.seed
    counts = config.counts.model_dump()
    for offset, split in enumerate(("train", "validation", "challenge")):
        records = generate_split(split=split, count=counts[split], seed=seed + offset)
        filename = "challenge.candidates.jsonl" if split == "challenge" else f"{split}.jsonl"
        write_jsonl(output_dir / filename, records)
    return counts
