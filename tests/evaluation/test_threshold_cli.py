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
from ai_guardrail.io import read_jsonl_snapshot, write_jsonl
from ai_guardrail.ner.labels import LABEL_TO_ID
from ai_guardrail.ner.manifest import (
    ARTIFACT_NAME,
    build_dataset_provenance,
    sha256_bytes,
    sha256_file,
)
from ai_guardrail.ner.manifest import (
    write_manifest as write_training_manifest,
)


def write_manifest(
    model_path: Path,
    validation_path: Path,
    *,
    release_version: str = "v1",
) -> tuple[Path, str]:
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
    manifest_path = model_path / "training-manifest.json"
    manifest = {
        "manifest_schema_version": 2,
        "artifact_name": model_path.name,
        "base_checkpoint": "distilbert/distilbert-base-cased",
        "base_revision": "immutable",
        "dataset_version": release_version,
        "generator_version": release_version,
        "label_mapping": LABEL_TO_ID,
        "artifact_checksums": checksums,
        "datasets": {
            "train": {
                "sha256": "1" * 64,
                "record_count": 1,
                "record_id_hashes": ["2" * 64],
                "template_families": ["person-train"],
                "content_hashes": ["3" * 64],
            },
            "validation": build_dataset_provenance(
                read_jsonl_snapshot(validation_path),
            ),
        },
    }
    write_training_manifest(manifest_path, manifest)
    artifact_sha256 = sha256_bytes(
        json.dumps(checksums, sort_keys=True, separators=(",", ":")).encode()
    )
    return manifest_path, artifact_sha256


def example(
    split: str,
    *,
    generator_version: str = "v1",
) -> LabeledExample:
    return LabeledExample(
        id=f"{split}-1",
        language="en",
        text="Jane Cooper",
        entities=[EntitySpan(type=EntityType.PERSON, start=0, end=11)],
        template_family=f"person-{split}",
        generator_version=generator_version,
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
@pytest.mark.parametrize(
    ("release_version", "artifact_name"),
    [
        ("v1", ARTIFACT_NAME),
        ("v2", "ai-guardrail-ner-en-v2"),
    ],
)
async def test_threshold_cli_writes_finite_validation_selected_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    release_version: str,
    artifact_name: str,
) -> None:
    validation_path = tmp_path / "validation.jsonl"
    output_path = tmp_path / "selected-threshold.json"
    model_path = tmp_path / artifact_name
    write_jsonl(
        validation_path,
        [
            example(
                "validation",
                generator_version=release_version,
            )
        ],
    )
    manifest_path, artifact_sha256 = write_manifest(
        model_path,
        validation_path,
        release_version=release_version,
    )
    load_calls: list[tuple[Path, float, dict[str, object]]] = []
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
        lambda path, threshold, **kwargs: (
            load_calls.append((path, threshold, kwargs))
            or FakeNerDetector()
        ),
    )

    await threshold_cli.run()

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload == {
        "candidate_thresholds": threshold_cli.THRESHOLDS,
        "model_version": artifact_name,
        "ner_artifact_sha256": artifact_sha256,
        "ner_manifest_sha256": hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest(),
        "selected_threshold": 0.8,
        "validation_sha256": hashlib.sha256(validation_path.read_bytes()).hexdigest(),
        "validation_provenance": build_dataset_provenance(
            read_jsonl_snapshot(validation_path),
        ),
    }
    assert "Jane Cooper" not in output_path.read_text(encoding="utf-8")
    assert math.isfinite(payload["selected_threshold"])
    assert load_calls == [
        (
            model_path,
            0.0,
            {
                "expected_artifact_sha256": artifact_sha256,
                "expected_manifest_sha256": hashlib.sha256(
                    manifest_path.read_bytes()
                ).hexdigest(),
            },
        )
    ]


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
    write_manifest(model_path, validation_path)
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
        lambda _path, threshold, **kwargs: FailedNerDetector(),
    )

    with pytest.raises(
        RuntimeError,
        match="failed safely",
    ) as exc_info:
        await threshold_cli.run()

    assert "safe_failure" not in str(exc_info.value)
    assert not output_path.exists()


@pytest.mark.asyncio
async def test_threshold_cli_rejects_validation_not_bound_to_training_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validation_path = tmp_path / "validation.jsonl"
    output_path = tmp_path / "threshold.json"
    model_path = tmp_path / ARTIFACT_NAME
    write_jsonl(validation_path, [example("validation")])
    write_manifest(model_path, validation_path)
    changed = example("validation").model_copy(
        update={"text": "Different validation content"}
    )
    write_jsonl(validation_path, [changed])
    load_called = False

    def unexpected_load(*args: object, **kwargs: object) -> FakeNerDetector:
        nonlocal load_called
        load_called = True
        return FakeNerDetector()

    monkeypatch.setattr(
        threshold_cli,
        "parse_args",
        lambda: argparse.Namespace(
            validation=validation_path,
            ner_model=model_path,
            output=output_path,
        ),
    )
    monkeypatch.setattr(threshold_cli.NerDetector, "load", unexpected_load)

    with pytest.raises(ValueError, match="threshold selection provenance"):
        await threshold_cli.run()

    assert load_called is False
    assert not output_path.exists()


@pytest.mark.asyncio
async def test_threshold_cli_rejects_output_inside_model_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validation_path = tmp_path / "validation.jsonl"
    model_path = tmp_path / ARTIFACT_NAME
    output_path = model_path / "selected-threshold.json"
    write_jsonl(validation_path, [example("validation")])
    write_manifest(model_path, validation_path)
    load_called = False

    def unexpected_load(*args: object, **kwargs: object) -> FakeNerDetector:
        nonlocal load_called
        load_called = True
        return FakeNerDetector()

    monkeypatch.setattr(
        threshold_cli,
        "parse_args",
        lambda: argparse.Namespace(
            validation=validation_path,
            ner_model=model_path,
            output=output_path,
        ),
    )
    monkeypatch.setattr(threshold_cli.NerDetector, "load", unexpected_load)

    with pytest.raises(
        ValueError,
        match="outside the NER model artifact",
    ):
        await threshold_cli.run()

    assert load_called is False
    assert not output_path.exists()


@pytest.mark.asyncio
async def test_threshold_cli_rejects_model_replacement_after_detector_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validation_path = tmp_path / "validation.jsonl"
    output_path = tmp_path / "threshold.json"
    model_path = tmp_path / ARTIFACT_NAME
    write_jsonl(validation_path, [example("validation")])
    write_manifest(model_path, validation_path)

    def replace_model(
        _path: Path,
        threshold: float,
        **kwargs: object,
    ) -> FakeNerDetector:
        weights_path = model_path / "model.safetensors"
        weights_path.write_bytes(b"replacement weights")
        manifest_path = model_path / "training-manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["artifact_checksums"]["model.safetensors"] = sha256_file(
            weights_path
        )
        write_training_manifest(manifest_path, manifest)
        return FakeNerDetector()

    monkeypatch.setattr(
        threshold_cli,
        "parse_args",
        lambda: argparse.Namespace(
            validation=validation_path,
            ner_model=model_path,
            output=output_path,
        ),
    )
    monkeypatch.setattr(threshold_cli.NerDetector, "load", replace_model)

    with pytest.raises(
        ValueError,
        match="model artifact changed during threshold selection",
    ):
        await threshold_cli.run()

    assert not output_path.exists()
