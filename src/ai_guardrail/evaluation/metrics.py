from __future__ import annotations

import math

from ai_guardrail.domain import (
    CandidateDetection,
    EntitySpan,
    EntityType,
)

CountMetrics = dict[str, int]


def score_spans(
    gold: list[EntitySpan],
    predicted: list[CandidateDetection],
) -> CountMetrics:
    gold_keys = {(entity.type, entity.start, entity.end) for entity in gold}
    predicted_keys = {
        (entity.type, entity.start, entity.end) for entity in predicted
    }
    return {
        "true_positive": len(gold_keys & predicted_keys),
        "false_positive": len(predicted_keys - gold_keys),
        "false_negative": len(gold_keys - predicted_keys),
    }


def score_partial_spans(
    gold: list[EntitySpan],
    predicted: list[CandidateDetection],
) -> CountMetrics:
    unmatched = list(gold)
    true_positive = 0
    false_positive = 0
    for candidate in sorted(predicted, key=lambda item: (item.start, item.end)):
        matches = [
            entity
            for entity in unmatched
            if entity.type == candidate.type
            and candidate.start < entity.end
            and candidate.end > entity.start
        ]
        if not matches:
            false_positive += 1
            continue
        selected = max(
            matches,
            key=lambda entity: min(entity.end, candidate.end)
            - max(entity.start, candidate.start),
        )
        unmatched.remove(selected)
        true_positive += 1
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": len(unmatched),
    }


def score_characters(
    text_length: int,
    gold: list[EntitySpan],
    predicted: list[CandidateDetection],
) -> CountMetrics:
    if text_length < 0:
        raise ValueError("text length must not be negative")
    gold_mask = [False] * text_length
    predicted_mask = [False] * text_length
    for entity in gold:
        for offset in range(entity.start, min(entity.end, text_length)):
            gold_mask[offset] = True
    for entity in predicted:
        for offset in range(entity.start, min(entity.end, text_length)):
            predicted_mask[offset] = True
    return {
        "sensitive_characters": sum(gold_mask),
        "missed_sensitive_characters": sum(
            gold_value and not predicted_value
            for gold_value, predicted_value in zip(
                gold_mask, predicted_mask, strict=True
            )
        ),
        "masked_characters": sum(predicted_mask),
        "extra_masked_characters": sum(
            predicted_value and not gold_value
            for gold_value, predicted_value in zip(
                gold_mask, predicted_mask, strict=True
            )
        ),
    }


def precision_recall_f1(counts: CountMetrics) -> dict[str, float]:
    true_positive = counts["true_positive"]
    false_positive = counts["false_positive"]
    false_negative = counts["false_negative"]
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def score_by_entity(
    gold: list[EntitySpan],
    predicted: list[CandidateDetection],
) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    for entity_type in EntityType:
        counts = score_spans(
            [item for item in gold if item.type == entity_type],
            [item for item in predicted if item.type == entity_type],
        )
        result[entity_type.value] = {**counts, **precision_recall_f1(counts)}
    return result


def select_threshold(
    gold_by_example: list[list[EntitySpan]],
    predicted_by_example: list[list[CandidateDetection]],
    thresholds: list[float],
) -> float:
    if not thresholds:
        raise ValueError("threshold list must not be empty")
    if any(
        not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0
        for threshold in thresholds
    ):
        raise ValueError("thresholds must be finite values between 0 and 1")
    if len(gold_by_example) != len(predicted_by_example):
        raise ValueError("gold and predicted example counts must match")

    scored: list[tuple[float, float, float]] = []
    for threshold in thresholds:
        counts = {
            "true_positive": 0,
            "false_positive": 0,
            "false_negative": 0,
        }
        for gold, predicted in zip(
            gold_by_example,
            predicted_by_example,
            strict=True,
        ):
            filtered = [
                item
                for item in predicted
                if item.confidence is not None and item.confidence >= threshold
            ]
            current = score_spans(gold, filtered)
            for key in counts:
                counts[key] += current[key]
        metrics = precision_recall_f1(counts)
        scored.append((metrics["f1"], metrics["recall"], threshold))
    return max(scored)[2]


def nearest_rank_percentile(values: list[float], percentile: int) -> float:
    if not values:
        raise ValueError("cannot calculate percentile of empty values")
    if not 0 < percentile <= 100:
        raise ValueError("percentile must be between 0 and 100")
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile / 100 * len(ordered)))
    return ordered[rank - 1]
