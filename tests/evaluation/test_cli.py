import argparse
from pathlib import Path

import pytest

from ai_guardrail.domain import LabeledExample
from ai_guardrail.evaluation import cli
from ai_guardrail.io import write_jsonl


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


@pytest.mark.asyncio
async def test_benchmark_cli_rejects_validation_as_challenge_before_loading_detectors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_path = tmp_path / "not-challenge.jsonl"
    write_jsonl(dataset_path, [validation_example()])
    detector_loaded = False

    def unexpected_load(*args: object, **kwargs: object) -> None:
        nonlocal detector_loaded
        detector_loaded = True

    monkeypatch.setattr(
        cli,
        "parse_args",
        lambda: argparse.Namespace(
            challenge=dataset_path,
            regex_config=tmp_path / "regex.yaml",
            ner_model=tmp_path / "model",
            ner_threshold=0.8,
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
        ),
    )
    monkeypatch.setattr(cli.RegexDetector, "from_yaml", unexpected_load)

    with pytest.raises(ValueError, match="challenge split"):
        await cli.run()

    assert detector_loaded is False
