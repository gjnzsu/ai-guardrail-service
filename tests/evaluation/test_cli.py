import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pytest

from ai_guardrail.domain import LabeledExample
from ai_guardrail.evaluation import cli
from ai_guardrail.evaluation.threshold_artifact import THRESHOLDS
from ai_guardrail.io import write_jsonl


def challenge_example() -> LabeledExample:
    return LabeledExample(
        id="challenge-1",
        language="en",
        text="Nothing sensitive here.",
        entities=[],
        template_family="negative-challenge",
        generator_version="v1",
        split="challenge",
    )


def validation_example() -> LabeledExample:
    return LabeledExample(
        id="validation-1",
        language="en",
        text="Nothing sensitive here.",
        entities=[],
        template_family="negative-validation",
        generator_version="v1",
        split="validation",
    )


def write_model_and_threshold_artifact(
    tmp_path: Path,
    *,
    selected_threshold: float = 0.8,
    manifest_hash: str | None = None,
    model_version: str = "ai-guardrail-ner-en-v1",
) -> tuple[Path, Path]:
    model_path = tmp_path / "ai-guardrail-ner-en-v1"
    model_path.mkdir()
    manifest_path = model_path / "training-manifest.json"
    manifest_path.write_text(
        json.dumps({"artifact_name": model_version}) + "\n",
        encoding="utf-8",
    )
    actual_manifest_hash = hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()
    threshold_path = tmp_path / "selected-threshold.json"
    threshold_path.write_text(
        json.dumps(
            {
                "candidate_thresholds": THRESHOLDS,
                "model_version": model_version,
                "ner_manifest_sha256": (
                    manifest_hash
                    if manifest_hash is not None
                    else actual_manifest_hash
                ),
                "selected_threshold": selected_threshold,
                "validation_sha256": "a" * 64,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return model_path, threshold_path


def benchmark_args(
    tmp_path: Path,
    challenge_path: Path,
    model_path: Path,
    threshold_path: Path,
) -> argparse.Namespace:
    return argparse.Namespace(
        challenge=challenge_path,
        regex_config=tmp_path / "regex.yaml",
        ner_model=model_path,
        ner_threshold_artifact=threshold_path,
        qwen_url="http://127.0.0.1:8080",
        qwen_model="qwen3-0.6b-q4_k_m",
        qwen_timeout=2.0,
        qwen_pid=123,
        qwen_sha256="model-hash",
        llama_version="test",
        guardrail_cpu_limit=2,
        guardrail_memory_limit_mib=2048,
        qwen_cpu_limit=4,
        qwen_memory_limit_mib=4096,
        repetitions=3,
        output_dir=tmp_path / "output",
    )


@pytest.mark.asyncio
async def test_benchmark_cli_rejects_validation_as_challenge_before_loading_detectors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_path = tmp_path / "not-challenge.jsonl"
    write_jsonl(dataset_path, [validation_example()])
    model_path, threshold_path = write_model_and_threshold_artifact(tmp_path)
    detector_loaded = False

    def unexpected_load(*args: object, **kwargs: object) -> None:
        nonlocal detector_loaded
        detector_loaded = True

    monkeypatch.setattr(
        cli,
        "parse_args",
        lambda: benchmark_args(
            tmp_path,
            dataset_path,
            model_path,
            threshold_path,
        ),
    )
    monkeypatch.setattr(cli.RegexDetector, "from_yaml", unexpected_load)

    with pytest.raises(ValueError, match="challenge split"):
        await cli.run()

    assert detector_loaded is False


@pytest.mark.asyncio
async def test_benchmark_cli_uses_validated_threshold_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    challenge_path = tmp_path / "challenge.jsonl"
    write_jsonl(challenge_path, [challenge_example()])
    model_path, threshold_path = write_model_and_threshold_artifact(tmp_path)
    loaded_thresholds: list[float] = []
    written_results: list[dict[str, Any]] = []

    class FakeRunner:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        async def run(
            self,
            examples: list[LabeledExample],
        ) -> dict[str, Any]:
            return {"example_count": len(examples), "detectors": {}}

    monkeypatch.setattr(
        cli,
        "parse_args",
        lambda: benchmark_args(
            tmp_path,
            challenge_path,
            model_path,
            threshold_path,
        ),
    )
    monkeypatch.setattr(
        cli.RegexDetector,
        "from_yaml",
        lambda _path: object(),
    )
    monkeypatch.setattr(
        cli.NerDetector,
        "load",
        lambda _path, threshold: (
            loaded_thresholds.append(threshold) or object()
        ),
    )
    monkeypatch.setattr(
        cli,
        "QwenDetector",
        lambda **kwargs: object(),
    )
    monkeypatch.setattr(
        cli,
        "AuthoritativeUnionDetector",
        lambda **kwargs: object(),
    )
    monkeypatch.setattr(cli, "BenchmarkRunner", FakeRunner)
    monkeypatch.setattr(
        cli,
        "write_json_report",
        lambda _path, result: written_results.append(result),
    )
    monkeypatch.setattr(
        cli,
        "write_markdown_report",
        lambda _path, _result: None,
    )

    await cli.run()

    assert loaded_thresholds == [0.8]
    environment = written_results[0]["environment"]
    assert environment["ner_threshold"] == 0.8
    assert environment["ner_threshold_artifact_sha256"] == hashlib.sha256(
        threshold_path.read_bytes()
    ).hexdigest()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("selected_threshold", "manifest_hash"),
    [
        (0.73, None),
        (0.8, "0" * 64),
        (math.nan, None),
    ],
)
async def test_benchmark_cli_rejects_forged_or_mismatched_threshold_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    selected_threshold: float,
    manifest_hash: str | None,
) -> None:
    challenge_path = tmp_path / "challenge.jsonl"
    write_jsonl(challenge_path, [challenge_example()])
    model_path, threshold_path = write_model_and_threshold_artifact(
        tmp_path,
        selected_threshold=selected_threshold,
        manifest_hash=manifest_hash,
    )
    detector_loaded = False

    def unexpected_load(*args: object, **kwargs: object) -> None:
        nonlocal detector_loaded
        detector_loaded = True

    monkeypatch.setattr(
        cli,
        "parse_args",
        lambda: benchmark_args(
            tmp_path,
            challenge_path,
            model_path,
            threshold_path,
        ),
    )
    monkeypatch.setattr(
        cli.RegexDetector,
        "from_yaml",
        unexpected_load,
    )

    with pytest.raises(
        ValueError,
        match="threshold artifact",
    ):
        await cli.run()

    assert detector_loaded is False
