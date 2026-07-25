from pathlib import Path

from ai_guardrail.domain import EntitySpan, EntityType, LabeledExample
from ai_guardrail.io import read_jsonl, read_jsonl_snapshot, write_jsonl


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


def test_jsonl_snapshot_keeps_records_and_digest_from_one_read(
    tmp_path: Path,
) -> None:
    path = tmp_path / "records.jsonl"
    original = LabeledExample(
        id="train-1",
        language="en",
        text="Original snapshot text",
        entities=[],
        template_family="original-train",
        generator_version="v1",
        split="train",
    )
    replacement = original.model_copy(
        update={"text": "Replacement text"}
    )
    write_jsonl(path, [original])

    snapshot = read_jsonl_snapshot(path)
    original_bytes = path.read_bytes()
    write_jsonl(path, [replacement])

    assert snapshot.records == (original,)
    assert snapshot.content == original_bytes
    assert snapshot.sha256 != read_jsonl_snapshot(path).sha256
