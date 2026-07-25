import pytest

from ai_guardrail.domain import (
    CandidateDetection,
    DetectionSource,
    DetectorOutput,
    EntitySpan,
    EntityType,
    LabeledExample,
)
from ai_guardrail.evaluation.runner import AuthoritativeUnionDetector, BenchmarkRunner


class FakePeakRssSampler:
    peak_mebibytes = 64.0

    def __init__(self, process_id: int | None = None) -> None:
        self.process_id = process_id

    def __enter__(self) -> "FakePeakRssSampler":
        return self

    def __exit__(self, *args: object) -> None:
        return None


@pytest.fixture(autouse=True)
def fake_peak_rss_sampler(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "ai_guardrail.evaluation.runner.PeakRssSampler",
        FakePeakRssSampler,
    )


def challenge_example() -> LabeledExample:
    return LabeledExample(
        id="challenge-1",
        language="en",
        text="Jane Cooper",
        entities=[EntitySpan(type=EntityType.PERSON, start=0, end=11)],
        template_family="manual-person-challenge",
        generator_version="v1",
        split="challenge",
    )


class FakeDetector:
    async def detect(self, text: str, message_index: int = 0) -> DetectorOutput:
        return DetectorOutput(
            detector=DetectionSource.NER,
            model_version="fake-ner",
            status="success",
            latency_ms=12,
            candidates=[
                CandidateDetection(
                    message_index=message_index,
                    type=EntityType.PERSON,
                    start=0,
                    end=11,
                    source=DetectionSource.NER,
                    confidence=0.9,
                )
            ],
        )


class FailureDetector:
    def __init__(self) -> None:
        self.calls = 0

    async def detect(self, text: str, message_index: int = 0) -> DetectorOutput:
        self.calls += 1
        status = "timeout" if self.calls == 1 else "error"
        return DetectorOutput(
            detector="qwen",
            model_version="fake-qwen",
            status=status,
            latency_ms=float(self.calls),
            candidates=[
                CandidateDetection(
                    message_index=message_index,
                    type=EntityType.PERSON,
                    start=0,
                    end=11,
                    source=DetectionSource.QWEN,
                )
            ],
            invalid_candidate_count=self.calls,
            error_code=f"safe_{status}",
        )


class FakeRegexDetector:
    async def detect(self, text: str, message_index: int = 0) -> DetectorOutput:
        return DetectorOutput(
            detector="regex",
            model_version="regex-v1",
            status="success",
            latency_ms=1,
            candidates=[
                CandidateDetection(
                    message_index=message_index,
                    type=EntityType.EMAIL,
                    start=0,
                    end=len(text),
                    source=DetectionSource.REGEX,
                )
            ],
        )


class FakeEmailNerDetector:
    async def detect(self, text: str, message_index: int = 0) -> DetectorOutput:
        return DetectorOutput(
            detector="ner",
            model_version="fake-ner",
            status="success",
            latency_ms=5,
            candidates=[
                CandidateDetection(
                    message_index=message_index,
                    type=EntityType.EMAIL,
                    start=0,
                    end=len(text) - 1,
                    source=DetectionSource.NER,
                    confidence=0.8,
                )
            ],
        )


@pytest.mark.asyncio
async def test_benchmark_runner_aggregates_metrics_without_storing_text() -> None:
    runner = BenchmarkRunner({"ner": FakeDetector()})

    result = await runner.run([challenge_example()])

    assert result["detectors"]["ner"]["strict"]["f1"] == 1.0
    assert result["detectors"]["ner"]["latency_ms"]["p95"] == 12
    assert "Jane Cooper" not in str(result)


@pytest.mark.asyncio
async def test_authoritative_union_uses_regex_for_deterministic_type() -> None:
    detector = AuthoritativeUnionDetector(
        regex=FakeRegexDetector(),
        ner=FakeEmailNerDetector(),
    )

    output = await detector.detect("jane@example.test")

    assert len(output.candidates) == 1
    assert output.candidates[0].source == DetectionSource.REGEX


@pytest.mark.asyncio
async def test_runner_reports_repeat_consistency() -> None:
    runner = BenchmarkRunner({"ner": FakeDetector()}, repetitions=2)

    result = await runner.run([challenge_example()])

    assert result["detectors"]["ner"]["consistency_rate"] == 1.0
    assert result["detectors"]["ner"]["success_rate"] == 1.0


@pytest.mark.asyncio
async def test_runner_reports_timeout_and_error_attempts_without_error_details() -> None:
    runner = BenchmarkRunner({"qwen": FailureDetector()}, repetitions=2)

    result = await runner.run([challenge_example()])

    metrics = result["detectors"]["qwen"]
    assert metrics["success_rate"] == 0.0
    assert metrics["json_parse_rate"] == 0.0
    assert metrics["timeout_count"] == 1
    assert metrics["error_count"] == 1
    assert metrics["invalid_candidate_count"] == 3
    assert metrics["strict"]["f1"] == 0.0
    assert metrics["consistency_rate"] == 0.0
    assert "safe_timeout" not in str(result)
    assert "safe_error" not in str(result)


def test_runner_rejects_non_positive_repetitions() -> None:
    with pytest.raises(ValueError, match="repetitions"):
        BenchmarkRunner({"ner": FakeDetector()}, repetitions=0)
