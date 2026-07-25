import pytest

from ai_guardrail.domain import EntitySpan, EntityType
from ai_guardrail.ner.alignment import align_spans_to_bio, decode_bio_predictions
from ai_guardrail.ner.labels import ID_TO_LABEL, LABEL_TO_ID


def test_label_mapping_is_stable_and_complete() -> None:
    assert LABEL_TO_ID == {
        "O": 0,
        "B-PERSON": 1,
        "I-PERSON": 2,
        "B-ADDRESS": 3,
        "I-ADDRESS": 4,
        "B-EMAIL": 5,
        "I-EMAIL": 6,
        "B-API_KEY": 7,
        "I-API_KEY": 8,
        "B-CUSTOMER_ID": 9,
        "I-CUSTOMER_ID": 10,
        "B-INTERNAL_PROJECT": 11,
        "I-INTERNAL_PROJECT": 12,
    }
    assert ID_TO_LABEL == {index: label for label, index in LABEL_TO_ID.items()}


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


def test_alignment_uses_unicode_code_point_offsets_and_punctuation() -> None:
    text = "前缀 😀 Jane,"
    start = text.index("Jane")
    offsets = [(0, 2), (3, 4), (5, 9), (9, 10)]

    labels = align_spans_to_bio(
        offsets,
        [EntitySpan(type=EntityType.PERSON, start=start, end=start + len("Jane"))],
    )

    assert labels == [
        LABEL_TO_ID["O"],
        LABEL_TO_ID["O"],
        LABEL_TO_ID["B-PERSON"],
        LABEL_TO_ID["O"],
    ]


def test_alignment_rejects_partial_token_entity_boundaries() -> None:
    with pytest.raises(ValueError, match="entity boundary does not align"):
        align_spans_to_bio(
            [(0, 5)],
            [EntitySpan(type=EntityType.PERSON, start=1, end=5)],
        )


def test_alignment_rejects_entity_boundary_in_tokenizer_gap() -> None:
    with pytest.raises(ValueError, match="entity boundary does not align"):
        align_spans_to_bio(
            [(0, 2), (4, 6)],
            [EntitySpan(type=EntityType.PERSON, start=3, end=6)],
        )


def test_alignment_rejects_token_overlapping_multiple_entities() -> None:
    with pytest.raises(ValueError, match="token overlaps multiple gold entities"):
        align_spans_to_bio(
            [(0, 4)],
            [
                EntitySpan(type=EntityType.PERSON, start=0, end=4),
                EntitySpan(type=EntityType.EMAIL, start=0, end=4),
            ],
        )


@pytest.mark.parametrize(
    "offset",
    [(-1, 1), (3, 2), (2, 2), ("0", 1)],
)
def test_alignment_rejects_invalid_non_special_offset(
    offset: tuple[int, int],
) -> None:
    with pytest.raises(ValueError, match="invalid token offset"):
        align_spans_to_bio([offset], [])


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


def test_decode_treats_orphan_i_tag_as_a_new_entity() -> None:
    detections = decode_bio_predictions(
        text="Jane Cooper",
        offsets=[(0, 4), (5, 11)],
        label_ids=[LABEL_TO_ID["I-PERSON"], LABEL_TO_ID["I-PERSON"]],
        probabilities=[0.91, 0.82],
        message_index=0,
    )

    assert [(item.type, item.start, item.end, item.confidence) for item in detections] == [
        (EntityType.PERSON, 0, 11, 0.82)
    ]


def test_decode_splits_on_invalid_i_type_transition() -> None:
    detections = decode_bio_predictions(
        text="Jane jane@example.test",
        offsets=[(0, 4), (5, 22)],
        label_ids=[LABEL_TO_ID["B-PERSON"], LABEL_TO_ID["I-EMAIL"]],
        probabilities=[0.98, 0.75],
        message_index=0,
    )

    assert [(item.type, item.start, item.end, item.confidence) for item in detections] == [
        (EntityType.PERSON, 0, 4, 0.98),
        (EntityType.EMAIL, 5, 22, 0.75),
    ]


def test_decode_rejects_mismatched_sequence_lengths() -> None:
    with pytest.raises(ValueError, match=r"zip\(\) argument"):
        decode_bio_predictions(
            text="Jane",
            offsets=[(0, 4)],
            label_ids=[LABEL_TO_ID["B-PERSON"], LABEL_TO_ID["I-PERSON"]],
            probabilities=[0.95],
            message_index=0,
        )


def test_decode_rejects_zero_width_non_special_entity_token() -> None:
    with pytest.raises(ValueError, match="invalid token offset"):
        decode_bio_predictions(
            text="Jane",
            offsets=[(3, 3)],
            label_ids=[LABEL_TO_ID["B-PERSON"]],
            probabilities=[0.95],
            message_index=0,
        )


@pytest.mark.parametrize(
    "offset",
    [(0, 5), (-1, 1), (3, 2), (2, 2), ("0", 1)],
)
def test_decode_rejects_invalid_non_special_offset(
    offset: tuple[int, int],
) -> None:
    with pytest.raises(ValueError, match="invalid token offset"):
        decode_bio_predictions(
            text="Jane",
            offsets=[offset],
            label_ids=[LABEL_TO_ID["O"]],
            probabilities=[0.95],
            message_index=0,
        )


def test_decode_ignores_special_tokens() -> None:
    detections = decode_bio_predictions(
        text="Jane",
        offsets=[(0, 0), (0, 4), (0, 0)],
        label_ids=[
            LABEL_TO_ID["B-PERSON"],
            LABEL_TO_ID["B-PERSON"],
            LABEL_TO_ID["I-PERSON"],
        ],
        probabilities=[0.11, 0.95, 0.11],
        message_index=0,
    )

    assert [(item.start, item.end, item.confidence) for item in detections] == [(0, 4, 0.95)]
