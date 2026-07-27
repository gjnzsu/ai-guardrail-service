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

    generator_version: Literal["v1", "v2"]
    seed: StrictInt
    counts: _SyntheticCounts


def render_template(
    *,
    template: Template,
    values: dict[EntityType, list[str]],
    record_id: str,
    generator_version: Literal["v1", "v2"] = "v1",
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
        generator_version=generator_version,
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


def _v2_values(
    *,
    template: Template,
    split: str,
    rng: random.Random,
) -> dict[EntityType, list[str]]:
    from ai_guardrail.synthetic.catalog_v2 import V2_CATALOG

    required = [EntityType(name) for name in PLACEHOLDER.findall(template.text)]
    values: dict[EntityType, list[str]] = defaultdict(list)
    for entity_type in required:
        values[entity_type].append(rng.choice(V2_CATALOG[split][entity_type]))
    return dict(values)


def _generate_v2_group(
    *,
    split: str,
    count: int,
    seed: int,
    entity_free: bool,
) -> list[LabeledExample]:
    from ai_guardrail.synthetic.templates_v2 import (
        V2_PREFIXES,
        V2_SUFFIXES,
        V2_TEMPLATES,
    )

    templates = [
        template
        for template in V2_TEMPLATES
        if template.split == split
        and (not PLACEHOLDER.search(template.text)) == entity_free
    ]
    if not templates:
        raise ValueError(f"no v2 templates configured for split {split}")

    rng = random.Random(seed)
    records: list[LabeledExample] = []
    seen_texts: set[str] = set()
    attempts = 0
    max_attempts = max(100, count * 100)
    while len(records) < count and attempts < max_attempts:
        template = templates[attempts % len(templates)]
        composed = Template(
            family=template.family,
            split=template.split,
            text=(
                rng.choice(V2_PREFIXES[split])
                + template.text
                + rng.choice(V2_SUFFIXES[split])
            ),
        )
        record = render_template(
            template=composed,
            values=_v2_values(template=composed, split=split, rng=rng),
            record_id=f"pending-{attempts:06d}",
            generator_version="v2",
        )
        attempts += 1
        if record.text in seen_texts:
            continue
        seen_texts.add(record.text)
        records.append(record)
    if len(records) != count:
        raise ValueError(f"v2 generation exhausted unique capacity for split {split}")
    return records


def generate_v2_split(*, split: str, count: int, seed: int) -> list[LabeledExample]:
    if split not in {"train", "validation", "challenge"}:
        raise ValueError(f"no v2 templates configured for split {split}")
    negative_count = count // 5
    positive = _generate_v2_group(
        split=split,
        count=count - negative_count,
        seed=seed,
        entity_free=False,
    )
    negative = _generate_v2_group(
        split=split,
        count=negative_count,
        seed=seed + 10_000,
        entity_free=True,
    )
    records = positive + negative
    random.Random(seed + 20_000).shuffle(records)
    return [
        record.model_copy(update={"id": f"v2-{split}-{index + 1:06d}"})
        for index, record in enumerate(records)
    ]


def _entity_values(
    records: list[LabeledExample],
) -> dict[EntityType, set[str]]:
    values: dict[EntityType, set[str]] = defaultdict(set)
    for record in records:
        for entity in record.entities:
            values[entity.type].add(record.text[entity.start : entity.end])
    return values


def _validate_v2_release(
    records_by_split: dict[str, list[LabeledExample]],
) -> None:
    for split, records in records_by_split.items():
        if len({record.text for record in records}) != len(records):
            raise ValueError(f"v2 duplicate text detected in split {split}")
        if {
            entity.type for record in records for entity in record.entities
        } != set(EntityType):
            raise ValueError(f"v2 entity coverage incomplete in split {split}")

    for left, right in (
        ("train", "validation"),
        ("train", "challenge"),
        ("validation", "challenge"),
    ):
        left_records = records_by_split[left]
        right_records = records_by_split[right]
        if not {record.text for record in left_records}.isdisjoint(
            record.text for record in right_records
        ):
            raise ValueError(f"v2 text leakage between splits {left} and {right}")
        if not {record.template_family for record in left_records}.isdisjoint(
            record.template_family for record in right_records
        ):
            raise ValueError(f"v2 template leakage between splits {left} and {right}")
        left_values = _entity_values(left_records)
        right_values = _entity_values(right_records)
        if any(
            not left_values[entity_type].isdisjoint(right_values[entity_type])
            for entity_type in EntityType
        ):
            raise ValueError(f"v2 catalog leakage between splits {left} and {right}")


def generate_dataset(config_path: Path, output_dir: Path) -> dict[str, int]:
    try:
        config = _SyntheticConfig.model_validate(
            yaml.safe_load(config_path.read_text(encoding="utf-8"))
        )
    except (OSError, UnicodeError, yaml.YAMLError, ValidationError):
        raise ValueError("invalid synthetic dataset configuration") from None
    seed = config.seed
    counts = config.counts.model_dump()
    if config.generator_version == "v2":
        records_by_split = {
            split: generate_v2_split(
                split=split,
                count=counts[split],
                seed=seed + offset,
            )
            for offset, split in enumerate(("train", "validation", "challenge"))
        }
        _validate_v2_release(records_by_split)
        for split, records in records_by_split.items():
            filename = (
                "challenge.candidates.jsonl"
                if split == "challenge"
                else f"{split}.jsonl"
            )
            write_jsonl(output_dir / filename, records)
        return counts

    for offset, split in enumerate(("train", "validation", "challenge")):
        records = generate_split(split=split, count=counts[split], seed=seed + offset)
        filename = "challenge.candidates.jsonl" if split == "challenge" else f"{split}.jsonl"
        write_jsonl(output_dir / filename, records)
    return counts
