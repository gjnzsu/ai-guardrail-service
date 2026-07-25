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


@pytest.mark.parametrize("value", [True, "1", 1.0])
def test_entity_offsets_require_strict_integers(value: object) -> None:
    with pytest.raises(ValidationError):
        EntitySpan(type=EntityType.PERSON, start=value, end=4)


@pytest.mark.parametrize("value", [True, "0", 0.0])
def test_candidate_message_index_requires_strict_integer(value: object) -> None:
    with pytest.raises(ValidationError):
        CandidateDetection(
            message_index=value,
            type=EntityType.PERSON,
            start=0,
            end=4,
            source=DetectionSource.NER,
        )
