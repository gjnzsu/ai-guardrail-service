from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

from ai_guardrail.domain import EntityType, LabeledExample
from ai_guardrail.io import read_jsonl
from ai_guardrail.synthetic.generator import generate_dataset

V2_CONFIG = Path("config/synthetic-v2.yaml")
FROZEN_CHALLENGE = Path("datasets/challenge/en-v1.seed.jsonl")


def _read_release(output_dir: Path) -> dict[str, list[LabeledExample]]:
    return {
        "train": read_jsonl(output_dir / "train.jsonl"),
        "validation": read_jsonl(output_dir / "validation.jsonl"),
        "challenge": read_jsonl(output_dir / "challenge.candidates.jsonl"),
    }


def _entity_values(records: list[LabeledExample]) -> dict[EntityType, set[str]]:
    values: dict[EntityType, set[str]] = defaultdict(set)
    for record in records:
        for entity in record.entities:
            values[entity.type].add(record.text[entity.start : entity.end])
    return values


def test_generate_v2_release_meets_quality_gates(tmp_path: Path) -> None:
    assert V2_CONFIG.is_file(), "v2 configuration must be committed"
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"

    counts = generate_dataset(V2_CONFIG, first_dir)
    assert generate_dataset(V2_CONFIG, second_dir) == counts
    assert counts == {"train": 4000, "validation": 800, "challenge": 200}

    filenames = ("train.jsonl", "validation.jsonl", "challenge.candidates.jsonl")
    assert {
        filename: (first_dir / filename).read_bytes() for filename in filenames
    } == {
        filename: (second_dir / filename).read_bytes() for filename in filenames
    }

    release = _read_release(first_dir)
    for split, records in release.items():
        assert len(records) == counts[split]
        assert all(record.generator_version == "v2" for record in records)
        assert all(record.split == split for record in records)
        assert len({record.id for record in records}) == len(records)
        assert len({record.text for record in records}) / len(records) >= 0.95
        assert all(
            record.text[entity.start : entity.end]
            for record in records
            for entity in record.entities
        )

    for split in ("train", "validation"):
        records = release[split]
        negative_ratio = sum(not record.entities for record in records) / len(records)
        assert 0.15 <= negative_ratio <= 0.25
        assert {
            entity.type for record in records for entity in record.entities
        } == set(EntityType)

    split_pairs = (
        ("train", "validation"),
        ("train", "challenge"),
        ("validation", "challenge"),
    )
    for left, right in split_pairs:
        assert {record.id for record in release[left]}.isdisjoint(
            record.id for record in release[right]
        )
        assert {record.text for record in release[left]}.isdisjoint(
            record.text for record in release[right]
        )
        assert {record.template_family for record in release[left]}.isdisjoint(
            record.template_family for record in release[right]
        )
        left_values = _entity_values(release[left])
        right_values = _entity_values(release[right])
        for entity_type in EntityType:
            assert left_values[entity_type].isdisjoint(right_values[entity_type])

    frozen_texts = {record.text for record in read_jsonl(FROZEN_CHALLENGE)}
    assert all(
        record.text not in frozen_texts
        for records in release.values()
        for record in records
    )


def test_v2_training_covers_observed_failure_families(tmp_path: Path) -> None:
    output_dir = tmp_path / "v2"
    generate_dataset(V2_CONFIG, output_dir)
    records = read_jsonl(output_dir / "train.jsonl")
    values = _entity_values(records)

    assert any(value.islower() for value in values[EntityType.PERSON])
    assert {
        value.split(maxsplit=1)[0]
        for value in values[EntityType.INTERNAL_PROJECT]
        if " " in value
    } >= {"Project", "project", "PROJECT"}
    assert any(
        value.split("@", maxsplit=1)[0].isdigit()
        for value in values[EntityType.EMAIL]
    )
    assert any(value.isdigit() for value in values[EntityType.CUSTOMER_ID])
    assert any("\n" in record.text for record in records)

    entity_counts = Counter(
        entity.type for record in records for entity in record.entities
    )
    assert all(entity_counts[entity_type] >= 400 for entity_type in EntityType)
