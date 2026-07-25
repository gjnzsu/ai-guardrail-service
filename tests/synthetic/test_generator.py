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


def test_generated_spans_always_slice_nonempty_source_text() -> None:
    records = generate_split(split="train", count=100, seed=17)
    for record in records:
        for entity in record.entities:
            assert record.text[entity.start : entity.end]


def test_generation_is_reproducible_for_same_seed() -> None:
    first = generate_split(split="validation", count=20, seed=29)
    second = generate_split(split="validation", count=20, seed=29)
    assert first == second
