import re
from collections import defaultdict
from pathlib import Path

import pytest

from ai_guardrail.domain import EntityType
from ai_guardrail.io import read_jsonl
from ai_guardrail.synthetic.catalog import CATALOG
from ai_guardrail.synthetic.generator import generate_dataset, generate_split, render_template
from ai_guardrail.synthetic.templates import TEMPLATES, Template


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


def test_generated_spans_always_slice_nonempty_source_text() -> None:
    records = generate_split(split="train", count=100, seed=17)
    for record in records:
        for entity in record.entities:
            assert record.text[entity.start : entity.end]


def test_generation_is_reproducible_for_same_seed() -> None:
    first = generate_split(split="validation", count=20, seed=29)
    second = generate_split(split="validation", count=20, seed=29)
    assert first == second


def test_committed_challenge_seed_fixture_has_expected_spans() -> None:
    records = read_jsonl(Path("datasets/challenge/en-v1.seed.jsonl"))
    expected_spans = {
        "challenge-seed-000001": [(EntityType.INTERNAL_PROJECT, "Project Falcon")],
        "challenge-seed-000002": [],
        "challenge-seed-000003": [
            (EntityType.PERSON, "Jane Cooper"),
            (EntityType.EMAIL, "jane.cooper@example.test"),
        ],
        "challenge-seed-000004": [(EntityType.API_KEY, "sk-test-A1B2C3D4E5F6G7H8")],
        "challenge-seed-000005": [
            (EntityType.PERSON, "tommy"),
            (EntityType.CUSTOMER_ID, "123456"),
        ],
        "challenge-seed-000006": [
            (EntityType.INTERNAL_PROJECT, "project Apex"),
        ],
        "challenge-seed-000007": [
            (EntityType.INTERNAL_PROJECT, "Project Apex"),
            (EntityType.EMAIL, "30156758@example.test"),
        ],
        "challenge-seed-000008": [
            (EntityType.INTERNAL_PROJECT, "Project Apex"),
            (EntityType.EMAIL, "project.owner@example.test"),
        ],
        "challenge-seed-000009": [
            (EntityType.INTERNAL_PROJECT, "Project APEX"),
            (EntityType.EMAIL, "project.owner@example.test"),
        ],
    }

    assert {record.id for record in records} == set(expected_spans)
    for record in records:
        assert record.language == "en"
        assert all(entity.type in EntityType for entity in record.entities)
        assert [
            (entity.type, record.text[entity.start : entity.end]) for entity in record.entities
        ] == expected_spans[record.id]


def test_incremental_v2_challenge_fixture_has_expected_spans() -> None:
    records = read_jsonl(Path("datasets/challenge/en-v2.incremental.jsonl"))
    expected_spans = {
        "challenge-v2-000001": [
            (EntityType.PERSON, "Raymond"),
            (EntityType.EMAIL, "87654321@qq.example"),
        ],
        "challenge-v2-000002": [
            (EntityType.PERSON, "Emily"),
            (EntityType.CUSTOMER_ID, "7654321"),
        ],
        "challenge-v2-000003": [
            (
                EntityType.ADDRESS,
                "Northbridge District Lantern Plaza A8-2507",
            ),
        ],
        "challenge-v2-000004": [],
    }

    assert {record.id for record in records} == set(expected_spans)
    for record in records:
        assert record.language == "en"
        assert record.generator_version == "v2"
        assert record.split == "challenge"
        assert [
            (entity.type, record.text[entity.start : entity.end])
            for entity in record.entities
        ] == expected_spans[record.id]


def test_rendering_every_template_preserves_exact_catalog_values_and_boundaries() -> None:
    for template in TEMPLATES:
        values: dict[EntityType, list[str]] = defaultdict(list)
        expected_text_parts: list[str] = []
        expected_spans: list[tuple[EntityType, int, int, str]] = []
        occurrence: dict[EntityType, int] = defaultdict(int)
        cursor = 0
        output_length = 0

        for match in re.finditer(r"\{([A-Z_]+)\}", template.text):
            literal = template.text[cursor : match.start()]
            expected_text_parts.append(literal)
            output_length += len(literal)

            entity_type = EntityType(match.group(1))
            value = CATALOG[template.split][entity_type][occurrence[entity_type]]
            occurrence[entity_type] += 1
            values[entity_type].append(value)

            start = output_length
            expected_text_parts.append(value)
            output_length += len(value)
            expected_spans.append((entity_type, start, output_length, value))
            cursor = match.end()

        expected_text_parts.append(template.text[cursor:])
        record = render_template(
            template=template,
            values=dict(values),
            record_id=f"{template.split}-controlled",
        )

        assert record.text == "".join(expected_text_parts)
        assert [
            (entity.type, entity.start, entity.end, record.text[entity.start : entity.end])
            for entity in record.entities
        ] == expected_spans


def test_generate_dataset_writes_configured_counts_and_filenames(tmp_path: Path) -> None:
    output_dir = tmp_path / "generated"

    counts = generate_dataset(Path("config/synthetic-v1.yaml"), output_dir)

    assert counts == {"train": 2000, "validation": 400, "challenge": 200}
    assert {path.name for path in output_dir.iterdir()} == {
        "train.jsonl",
        "validation.jsonl",
        "challenge.candidates.jsonl",
    }
    assert len(read_jsonl(output_dir / "train.jsonl")) == counts["train"]
    assert len(read_jsonl(output_dir / "validation.jsonl")) == counts["validation"]
    assert len(read_jsonl(output_dir / "challenge.candidates.jsonl")) == counts["challenge"]


@pytest.mark.parametrize(
    "config",
    [
        "generator_version: v3\nseed: 7\ncounts:\n  train: 1\n  validation: 1\n  challenge: 1\n",
        "generator_version: v1\nseed: 7\ncounts:\n  train: 1\n  validation: 1\n",
        "generator_version: v1\nseed: 7\ncounts:\n  train: '1'\n  validation: 1\n  challenge: 1\n",
        "generator_version: v1\nseed: 7\ncounts:\n  train: 0\n  validation: 1\n  challenge: 1\n",
        "generator_version: v1\nseed: true\ncounts:\n  train: 1\n  validation: 1\n  challenge: 1\n",
    ],
)
def test_generate_dataset_rejects_unsupported_or_non_strict_config(
    tmp_path: Path,
    config: str,
) -> None:
    config_path = tmp_path / "synthetic.yaml"
    config_path.write_text(config, encoding="utf-8")

    with pytest.raises(ValueError, match="invalid synthetic dataset configuration"):
        generate_dataset(config_path, tmp_path / "output")
