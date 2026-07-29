import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pytest

from ai_guardrail.domain import EntityType, LabeledExample
from ai_guardrail.evaluation import cli
from ai_guardrail.evaluation.threshold_artifact import (
    LEGACY_THRESHOLDS,
    THRESHOLDS,
)
from ai_guardrail.io import write_jsonl
from ai_guardrail.ner.labels import LABEL_TO_ID
from ai_guardrail.ner.manifest import (
    ARTIFACT_NAME,
    sha256_bytes,
    sha256_file,
    write_manifest,
)


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
    selected_thresholds: dict[str, float] | None = None,
    manifest_hash: str | None = None,
    model_version: str = "ai-guardrail-ner-en-v1",
) -> tuple[Path, Path]:
    model_path = tmp_path / ARTIFACT_NAME
    model_path.mkdir()
    config = {
        "label2id": LABEL_TO_ID,
        "id2label": {str(value): key for key, value in LABEL_TO_ID.items()},
    }
    (model_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (model_path / "model.safetensors").write_bytes(b"weights")
    (model_path / "tokenizer.json").write_bytes(b"tokenizer")
    checksums = {
        name: sha256_file(model_path / name)
        for name in ("config.json", "model.safetensors", "tokenizer.json")
    }
    datasets = {
        "train": {
            "sha256": "1" * 64,
            "record_count": 1,
            "record_id_hashes": [sha256_bytes(b"train-1")],
            "template_families": ["train-family"],
            "content_hashes": [sha256_bytes(b"Training text")],
        },
        "validation": {
            "sha256": "a" * 64,
            "record_count": 1,
            "record_id_hashes": [sha256_bytes(b"validation-1")],
            "template_families": ["negative-validation"],
            "content_hashes": [sha256_bytes(b"Validation text")],
        },
    }
    manifest_path = model_path / "training-manifest.json"
    write_manifest(
        manifest_path,
        {
            "manifest_schema_version": 2,
            "artifact_name": model_version,
            "base_checkpoint": "distilbert/distilbert-base-cased",
            "base_revision": "immutable",
            "dataset_version": "v1",
            "generator_version": "v1",
            "label_mapping": LABEL_TO_ID,
            "artifact_checksums": checksums,
            "datasets": datasets,
        },
    )
    actual_manifest_hash = hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()
    threshold_path = tmp_path / "selected-threshold.json"
    payload = {
        "candidate_thresholds": (
            THRESHOLDS
            if selected_thresholds is not None
            else LEGACY_THRESHOLDS
        ),
        "model_version": model_version,
        "ner_artifact_sha256": sha256_bytes(
            json.dumps(
                checksums,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ),
        "ner_manifest_sha256": (
            manifest_hash
            if manifest_hash is not None
            else actual_manifest_hash
        ),
        "validation_sha256": "a" * 64,
        "validation_provenance": datasets["validation"],
    }
    if selected_thresholds is None:
        payload["selected_threshold"] = selected_threshold
    else:
        payload["selected_thresholds"] = selected_thresholds
        payload["threshold_schema_version"] = 2
    threshold_path.write_text(
        json.dumps(payload)
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
    selected_thresholds = {
        entity_type.value: (
            0.35
            if entity_type == EntityType.PERSON
            else 0.6
        )
        for entity_type in EntityType
    }
    model_path, threshold_path = write_model_and_threshold_artifact(
        tmp_path,
        selected_thresholds=selected_thresholds,
    )
    loaded_thresholds: list[
        tuple[dict[EntityType, float], dict[str, object]]
    ] = []
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
        lambda _path, threshold, **kwargs: (
            loaded_thresholds.append((threshold, kwargs)) or object()
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

    threshold_payload = json.loads(threshold_path.read_text(encoding="utf-8"))
    assert loaded_thresholds == [
        (
            {
                entity_type: selected_thresholds[entity_type.value]
                for entity_type in EntityType
            },
            {
                "expected_artifact_sha256": threshold_payload[
                    "ner_artifact_sha256"
                ],
                "expected_manifest_sha256": threshold_payload[
                    "ner_manifest_sha256"
                ],
            },
        )
    ]
    environment = written_results[0]["environment"]
    assert environment["ner_thresholds"] == selected_thresholds
    assert "ner_threshold" not in environment
    assert environment["ner_threshold_artifact_sha256"] == hashlib.sha256(
        threshold_path.read_bytes()
    ).hexdigest()
    assert environment["ner_artifact_sha256"] == threshold_payload[
        "ner_artifact_sha256"
    ]


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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("record_id", "family", "text"),
    [
        ("challenge-1", "train-family", "Fresh challenge text"),
        ("challenge-1", "fresh-family", "Training text"),
        ("train-1", "fresh-family", "Fresh challenge text"),
    ],
)
async def test_benchmark_cli_rejects_challenge_overlap_before_loading_detectors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    record_id: str,
    family: str,
    text: str,
) -> None:
    challenge_path = tmp_path / "challenge.jsonl"
    write_jsonl(
        challenge_path,
        [
            LabeledExample(
                id=record_id,
                language="en",
                text=text,
                entities=[],
                template_family=family,
                generator_version="v1",
                split="challenge",
            )
        ],
    )
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
            challenge_path,
            model_path,
            threshold_path,
        ),
    )
    monkeypatch.setattr(cli.RegexDetector, "from_yaml", unexpected_load)

    with pytest.raises(ValueError, match="challenge provenance"):
        await cli.run()

    assert detector_loaded is False


@pytest.mark.asyncio
async def test_benchmark_cli_rejects_duplicate_challenge_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    challenge_path = tmp_path / "challenge.jsonl"
    record = challenge_example()
    write_jsonl(challenge_path, [record, record])
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
            challenge_path,
            model_path,
            threshold_path,
        ),
    )
    monkeypatch.setattr(cli.RegexDetector, "from_yaml", unexpected_load)

    with pytest.raises(ValueError, match="challenge"):
        await cli.run()

    assert detector_loaded is False


@pytest.mark.asyncio
async def test_benchmark_uses_one_challenge_snapshot_when_path_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    challenge_path = tmp_path / "challenge.jsonl"
    original = challenge_example()
    write_jsonl(challenge_path, [original])
    original_bytes = challenge_path.read_bytes()
    model_path, threshold_path = write_model_and_threshold_artifact(tmp_path)
    seen_examples: list[LabeledExample] = []
    written_results: list[dict[str, Any]] = []

    class FakeRunner:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        async def run(
            self,
            examples: list[LabeledExample],
        ) -> dict[str, Any]:
            seen_examples.extend(examples)
            return {"example_count": len(examples), "detectors": {}}

    def replace_challenge(_path: Path) -> object:
        write_jsonl(
            challenge_path,
            [
                LabeledExample(
                    id="replacement-1",
                    language="en",
                    text="Private replacement challenge",
                    entities=[],
                    template_family="replacement-challenge",
                    generator_version="v1",
                    split="challenge",
                )
            ],
        )
        return object()

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
    monkeypatch.setattr(cli.RegexDetector, "from_yaml", replace_challenge)
    monkeypatch.setattr(
        cli.NerDetector,
        "load",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(cli, "QwenDetector", lambda **kwargs: object())
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
    monkeypatch.setattr(cli, "write_markdown_report", lambda *_args: None)

    await cli.run()

    assert seen_examples == [original]
    assert written_results[0]["environment"]["challenge_sha256"] == (
        hashlib.sha256(original_bytes).hexdigest()
    )
