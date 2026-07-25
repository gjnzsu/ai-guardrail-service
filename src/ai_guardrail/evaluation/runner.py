from __future__ import annotations

import asyncio
from typing import Any

from ai_guardrail.detectors.base import Detector
from ai_guardrail.domain import (
    CandidateDetection,
    DetectorOutput,
    EntityType,
    LabeledExample,
)
from ai_guardrail.evaluation.metrics import (
    nearest_rank_percentile,
    precision_recall_f1,
    score_by_entity,
    score_characters,
    score_partial_spans,
    score_spans,
)
from ai_guardrail.evaluation.resources import PeakRssSampler

REGEX_AUTHORITATIVE_TYPES = {
    EntityType.EMAIL,
    EntityType.API_KEY,
    EntityType.CUSTOMER_ID,
}


def candidate_key(
    candidate: CandidateDetection,
) -> tuple[EntityType, int, int]:
    return candidate.type, candidate.start, candidate.end


def output_signature(
    output: DetectorOutput,
) -> tuple[str, frozenset[tuple[EntityType, int, int]]]:
    return (
        output.status,
        frozenset(
            candidate_key(item)
            for item in output.candidates
        ),
    )


class AuthoritativeUnionDetector:
    def __init__(self, *, regex: Detector, ner: Detector) -> None:
        self.regex = regex
        self.ner = ner

    async def detect(
        self,
        text: str,
        message_index: int = 0,
    ) -> DetectorOutput:
        regex_output, ner_output = await asyncio.gather(
            self.regex.detect(text, message_index),
            self.ner.detect(text, message_index),
        )
        candidates = list(regex_output.candidates)
        seen = {candidate_key(candidate) for candidate in candidates}
        for candidate in ner_output.candidates:
            if candidate.type in REGEX_AUTHORITATIVE_TYPES:
                continue
            if candidate_key(candidate) not in seen:
                candidates.append(candidate)
                seen.add(candidate_key(candidate))

        statuses = {regex_output.status, ner_output.status}
        if "timeout" in statuses:
            status = "timeout"
        elif "error" in statuses:
            status = "error"
        else:
            status = "success"
        return DetectorOutput(
            detector="regex+ner",
            model_version=(
                f"{regex_output.model_version}+{ner_output.model_version}"
            ),
            status=status,
            candidates=sorted(
                candidates,
                key=lambda item: (item.start, item.end),
            ),
            latency_ms=max(
                regex_output.latency_ms,
                ner_output.latency_ms,
            ),
            invalid_candidate_count=(
                regex_output.invalid_candidate_count
                + ner_output.invalid_candidate_count
            ),
            error_code=None if status == "success" else "combined_detector_failure",
        )


class BenchmarkRunner:
    def __init__(
        self,
        detectors: dict[str, Detector],
        repetitions: int = 1,
        resource_pids: dict[str, int] | None = None,
    ) -> None:
        if repetitions < 1:
            raise ValueError("repetitions must be at least one")
        self.detectors = detectors
        self.repetitions = repetitions
        self.resource_pids = resource_pids or {}

    async def run(
        self,
        examples: list[LabeledExample],
    ) -> dict[str, Any]:
        if not examples:
            raise ValueError("benchmark requires at least one example")
        result: dict[str, Any] = {
            "example_count": len(examples),
            "detectors": {},
        }
        for name, detector in self.detectors.items():
            result["detectors"][name] = await self._run_detector(
                name,
                detector,
                examples,
            )
        return result

    async def _run_detector(
        self,
        name: str,
        detector: Detector,
        examples: list[LabeledExample],
    ) -> dict[str, Any]:
        span_counts = _empty_counts()
        partial_counts = _empty_counts()
        character_counts = {
            "sensitive_characters": 0,
            "missed_sensitive_characters": 0,
            "masked_characters": 0,
            "extra_masked_characters": 0,
        }
        per_type_counts = {
            entity_type.value: _empty_counts()
            for entity_type in EntityType
        }
        latencies: list[float] = []
        invalid_candidates = 0
        errors = 0
        timeouts = 0
        successes = 0
        consistent_comparisons = 0
        total_comparisons = 0

        with PeakRssSampler(
            process_id=self.resource_pids.get(name),
        ) as memory:
            for example in examples:
                outputs = [
                    await detector.detect(example.text)
                    for _ in range(self.repetitions)
                ]
                output = outputs[0]
                baseline = output_signature(output)
                for repeated in outputs[1:]:
                    total_comparisons += 1
                    consistent_comparisons += int(
                        output_signature(repeated) == baseline
                    )
                latencies.extend(item.latency_ms for item in outputs)
                invalid_candidates += sum(
                    item.invalid_candidate_count
                    for item in outputs
                )
                errors += sum(
                    item.status == "error"
                    for item in outputs
                )
                timeouts += sum(
                    item.status == "timeout"
                    for item in outputs
                )
                successes += sum(
                    item.status == "success"
                    for item in outputs
                )
                _accumulate_example_metrics(
                    example,
                    output,
                    span_counts,
                    partial_counts,
                    character_counts,
                    per_type_counts,
                )

        attempts = len(examples) * self.repetitions
        per_entity = {
            entity_type: {
                **counts,
                **precision_recall_f1(counts),
            }
            for entity_type, counts in per_type_counts.items()
        }
        return {
            "strict": {
                **span_counts,
                **precision_recall_f1(span_counts),
            },
            "partial": {
                **partial_counts,
                **precision_recall_f1(partial_counts),
            },
            "per_entity": per_entity,
            "characters": character_counts,
            "latency_ms": {
                "p50": nearest_rank_percentile(latencies, 50),
                "p95": nearest_rank_percentile(latencies, 95),
                "p99": nearest_rank_percentile(latencies, 99),
            },
            "peak_rss_mib": memory.peak_mebibytes,
            "success_rate": successes / attempts,
            "json_parse_rate": (
                successes / attempts
                if name == "qwen"
                else None
            ),
            "consistency_rate": (
                consistent_comparisons / total_comparisons
                if total_comparisons
                else 1.0
            ),
            "invalid_candidate_count": invalid_candidates,
            "error_count": errors,
            "timeout_count": timeouts,
        }


def _empty_counts() -> dict[str, int]:
    return {
        "true_positive": 0,
        "false_positive": 0,
        "false_negative": 0,
    }


def _accumulate_example_metrics(
    example: LabeledExample,
    output: DetectorOutput,
    span_counts: dict[str, int],
    partial_counts: dict[str, int],
    character_counts: dict[str, int],
    per_type_counts: dict[str, dict[str, int]],
) -> None:
    candidates = (
        output.candidates
        if output.status == "success"
        else []
    )
    current_spans = score_spans(example.entities, candidates)
    current_partial = score_partial_spans(
        example.entities,
        candidates,
    )
    current_characters = score_characters(
        len(example.text),
        example.entities,
        candidates,
    )
    for key in span_counts:
        span_counts[key] += current_spans[key]
        partial_counts[key] += current_partial[key]
    for key in character_counts:
        character_counts[key] += current_characters[key]
    current_by_entity = score_by_entity(
        example.entities,
        candidates,
    )
    for entity_type, metrics in current_by_entity.items():
        for key in (
            "true_positive",
            "false_positive",
            "false_negative",
        ):
            per_type_counts[entity_type][key] += int(metrics[key])
