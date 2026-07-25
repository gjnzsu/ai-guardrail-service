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
