import argparse
import hashlib
import json
import math
from pathlib import Path

import pytest

from ai_guardrail.domain import (
    CandidateDetection,
    DetectionSource,
    DetectorOutput,
    EntitySpan,
    EntityType,
    LabeledExample,
)
from ai_guardrail.evaluation import threshold_cli
from ai_guardrail.io import write_jsonl


def write_manifest(model_path: Path) -> Path:
    model_path.mkdir()
    manifest_path = model_path / "training-manifest.json"
    manifest_path.write_text(
        json.dumps({"artifact_name": "ai-guardrail-ner-en-v1"}) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def example(split: str) -> LabeledExample:
    return LabeledExample(
        id=f"{split}-1",
        language="en",
        text="Jane Cooper",
        entities=[EntitySpan(type=EntityType.PERSON, start=0, end=11)],
        template_family=f"person-{split}",
        generator_version="v1",
        split=split,
    )


class FakeNerDetector:
    async def detect(self, text: str, message_index: int = 0) -> DetectorOutput:
        return DetectorOutput(
            detector="ner",
            model_version="fake-ner",
            status="success",
            latency_ms=1,
            candidates=[
                CandidateDetection(
                    message_index=message_index,
                    type=EntityType.PERSON,
                    start=0,
                    end=11,
                    source=DetectionSource.NER,
                    confidence=0.81,
                ),
                CandidateDetection(
                    message_index=message_index,
                    type=EntityType.ADDRESS,
                    start=0,
                    end=4,
                    source=DetectionSource.NER,
                    confidence=0.61,
                ),
            ],
        )


class FailedNerDetector:
    async def detect(
        self,
        text: str,
        message_index: int = 0,
    ) -> DetectorOutput:
        return DetectorOutput(
            detector="ner",
            model_version="fake-ner",
            status="error",
            latency_ms=1,
            error_code="safe_failure",
        )


@pytest.mark.asyncio
async def test_threshold_cli_writes_finite_validation_selected_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validation_path = tmp_path / "validation.jsonl"
    output_path = tmp_path / "selected-threshold.json"
    model_path = tmp_path / "ai-guardrail-ner-en-v1"
    write_jsonl(validation_path, [example("validation")])
    manifest_path = write_manifest(model_path)
    monkeypatch.setattr(
        threshold_cli,
        "parse_args",
        lambda: argparse.Namespace(
            validation=validation_path,
            ner_model=model_path,
            output=output_path,
        ),
    )
    monkeypatch.setattr(
        threshold_cli.NerDetector,
        "load",
        lambda _path, threshold: FakeNerDetector(),
    )

    await threshold_cli.run()

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload == {
        "candidate_thresholds": threshold_cli.THRESHOLDS,
        "model_version": "ai-guardrail-ner-en-v1",
        "ner_manifest_sha256": hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest(),
        "selected_threshold": 0.8,
        "validation_sha256": hashlib.sha256(validation_path.read_bytes()).hexdigest(),
    }
    assert math.isfinite(payload["selected_threshold"])


@pytest.mark.asyncio
async def test_threshold_cli_rejects_challenge_before_loading_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    challenge_path = tmp_path / "challenge.jsonl"
    write_jsonl(challenge_path, [example("challenge")])
    load_called = False

    def unexpected_load(_path: Path, threshold: float) -> FakeNerDetector:
        nonlocal load_called
        load_called = True
        return FakeNerDetector()

    monkeypatch.setattr(
        threshold_cli,
        "parse_args",
        lambda: argparse.Namespace(
            validation=challenge_path,
            ner_model=tmp_path / "model",
            output=tmp_path / "threshold.json",
        ),
    )
    monkeypatch.setattr(threshold_cli.NerDetector, "load", unexpected_load)

    with pytest.raises(ValueError, match="validation split"):
        await threshold_cli.run()

    assert load_called is False
    assert not (tmp_path / "threshold.json").exists()


@pytest.mark.asyncio
async def test_threshold_cli_does_not_write_artifact_after_detector_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validation_path = tmp_path / "validation.jsonl"
    output_path = tmp_path / "threshold.json"
    model_path = tmp_path / "ai-guardrail-ner-en-v1"
    write_jsonl(validation_path, [example("validation")])
    write_manifest(model_path)
    monkeypatch.setattr(
        threshold_cli,
        "parse_args",
        lambda: argparse.Namespace(
            validation=validation_path,
            ner_model=model_path,
            output=output_path,
        ),
    )
    monkeypatch.setattr(
        threshold_cli.NerDetector,
        "load",
        lambda _path, threshold: FailedNerDetector(),
    )

    with pytest.raises(
        RuntimeError,
        match="failed safely",
    ) as exc_info:
        await threshold_cli.run()

    assert "safe_failure" not in str(exc_info.value)
    assert not output_path.exists()
