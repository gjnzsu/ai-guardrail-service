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
    token_starts = {
        token_start for token_start, token_end in offsets if token_start < token_end
    }
    token_ends = {token_end for token_start, token_end in offsets if token_start < token_end}
    for entity in entities:
        if entity.start not in token_starts or entity.end not in token_ends:
            raise ValueError("entity boundary does not align to token boundary")

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
            if current_start >= current_end or not text[current_start:current_end]:
                raise ValueError("decoded an empty entity span")
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
