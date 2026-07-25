import math

import pytest

from ai_guardrail.domain import (
    CandidateDetection,
    DetectionSource,
    EntitySpan,
    EntityType,
)
from ai_guardrail.evaluation.metrics import (
    nearest_rank_percentile,
    score_by_entity,
    score_characters,
    score_partial_spans,
    score_spans,
    select_threshold,
)


def candidate(
    entity_type: EntityType,
    start: int,
    end: int,
    confidence: float | None = None,
) -> CandidateDetection:
    return CandidateDetection(
        message_index=0,
        type=entity_type,
        start=start,
        end=end,
        source=DetectionSource.NER,
        confidence=confidence,
    )


def test_strict_span_metrics_count_type_and_boundary() -> None:
    gold = [
        EntitySpan(type=EntityType.PERSON, start=0, end=11),
        EntitySpan(type=EntityType.EMAIL, start=15, end=32),
    ]
    predicted = [
        candidate(EntityType.PERSON, 0, 11, 0.9),
        candidate(EntityType.EMAIL, 15, 31),
    ]

    assert score_spans(gold, predicted) == {
        "true_positive": 1,
        "false_positive": 1,
        "false_negative": 1,
    }


def test_character_metrics_measure_missed_and_extra_characters() -> None:
    gold = [EntitySpan(type=EntityType.PERSON, start=0, end=4)]
    predicted = [candidate(EntityType.PERSON, 0, 3, 0.9)]

    assert score_characters(10, gold, predicted) == {
        "sensitive_characters": 4,
        "missed_sensitive_characters": 1,
        "masked_characters": 3,
        "extra_masked_characters": 0,
    }


def test_nearest_rank_percentile_is_deterministic() -> None:
    assert nearest_rank_percentile([10, 20, 30, 40], 95) == 40


def test_partial_span_counts_same_type_overlap_once() -> None:
    gold = [EntitySpan(type=EntityType.ADDRESS, start=0, end=20)]
    predicted = [candidate(EntityType.ADDRESS, 5, 20, 0.8)]

    assert score_partial_spans(gold, predicted) == {
        "true_positive": 1,
        "false_positive": 0,
        "false_negative": 0,
    }


def test_partial_span_matching_is_one_to_one() -> None:
    gold = [EntitySpan(type=EntityType.ADDRESS, start=0, end=20)]
    predicted = [
        candidate(EntityType.ADDRESS, 0, 10, 0.8),
        candidate(EntityType.ADDRESS, 10, 20, 0.8),
    ]

    assert score_partial_spans(gold, predicted) == {
        "true_positive": 1,
        "false_positive": 1,
        "false_negative": 0,
    }


def test_per_entity_metrics_keep_rare_types_visible() -> None:
    gold = [EntitySpan(type=EntityType.INTERNAL_PROJECT, start=0, end=14)]

    result = score_by_entity(gold, [])

    assert set(result) == {entity_type.value for entity_type in EntityType}
    assert result["INTERNAL_PROJECT"]["recall"] == 0.0
    assert result["INTERNAL_PROJECT"]["false_negative"] == 1


def test_threshold_selection_maximizes_strict_f1() -> None:
    gold = [[EntitySpan(type=EntityType.PERSON, start=0, end=11)]]
    predicted = [[
        candidate(EntityType.PERSON, 0, 11, 0.8),
        candidate(EntityType.ADDRESS, 0, 4, 0.6),
    ]]

    assert select_threshold(gold, predicted, [0.5, 0.7, 0.9]) == 0.7


def test_threshold_selection_uses_higher_threshold_as_final_tiebreaker() -> None:
    gold = [[EntitySpan(type=EntityType.PERSON, start=0, end=11)]]
    predicted = [[candidate(EntityType.PERSON, 0, 11, 0.9)]]

    assert select_threshold(gold, predicted, [0.5, 0.7]) == 0.7


@pytest.mark.parametrize("thresholds", [[], [math.nan], [math.inf], [-0.1], [1.1]])
def test_threshold_selection_rejects_missing_or_non_finite_candidates(
    thresholds: list[float],
) -> None:
    with pytest.raises(ValueError, match="threshold"):
        select_threshold([], [], thresholds)
