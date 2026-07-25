import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from ai_guardrail.domain import EntitySpan, EntityType, LabeledExample
from ai_guardrail.evaluation import cli, threshold_cli
from ai_guardrail.evaluation.threshold_artifact import load_selected_threshold
from ai_guardrail.io import read_jsonl_snapshot, write_jsonl
from ai_guardrail.ner.labels import ID_TO_LABEL, LABEL_TO_ID
from ai_guardrail.ner.manifest import (
    ARTIFACT_NAME,
    build_dataset_provenance,
    sha256_file,
    write_manifest,
)


class SnapshotBatch(dict):
    def to(self, device: str) -> "SnapshotBatch":
        assert device == "cpu"
        return self


class SnapshotTokenizer:
    def __call__(self, text: str, **kwargs: object) -> SnapshotBatch:
        return SnapshotBatch(
            input_ids=torch.tensor([[101, 1, 2, 102]]),
            attention_mask=torch.tensor([[1, 1, 1, 1]]),
            offset_mapping=torch.tensor(
                [[[0, 0], [0, 4], [5, 11], [0, 0]]]
            ),
        )


class SnapshotModel:
    config = SimpleNamespace(
        label2id=LABEL_TO_ID,
        id2label=ID_TO_LABEL,
    )

    def eval(self) -> "SnapshotModel":
        return self

    def to(self, device: str) -> "SnapshotModel":
        assert device == "cpu"
        return self

    def __call__(self, **kwargs: object) -> SimpleNamespace:
        logits = torch.zeros((1, 4, len(LABEL_TO_ID)))
        logits[0, 1, LABEL_TO_ID["B-PERSON"]] = 8
        logits[0, 2, LABEL_TO_ID["I-PERSON"]] = 8
        return SimpleNamespace(logits=logits)


def _write_model_artifact(
    root: Path,
    validation_path: Path,
) -> Path:
    model_path = root / ARTIFACT_NAME
    model_path.mkdir(parents=True)
    config = {
        "label2id": LABEL_TO_ID,
        "id2label": {
            str(identifier): label
            for identifier, label in ID_TO_LABEL.items()
        },
    }
    (model_path / "config.json").write_text(
        json.dumps(config),
        encoding="utf-8",
    )
    (model_path / "model.safetensors").write_bytes(b"weights")
    (model_path / "tokenizer.json").write_bytes(b"tokenizer")
    checksums = {
        name: sha256_file(model_path / name)
        for name in ("config.json", "model.safetensors", "tokenizer.json")
    }
    write_manifest(
        model_path / "training-manifest.json",
        {
            "manifest_schema_version": 2,
            "artifact_name": ARTIFACT_NAME,
            "base_checkpoint": "distilbert/distilbert-base-cased",
            "base_revision": "immutable",
            "dataset_version": "v1",
            "generator_version": "v1",
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
                    read_jsonl_snapshot(validation_path)
                ),
            },
        },
    )
    return model_path


@pytest.mark.asyncio
async def test_filesystem_provenance_chain_from_threshold_to_benchmark(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validation_path = tmp_path / "datasets" / "validation.jsonl"
    challenge_path = tmp_path / "datasets" / "challenge.jsonl"
    threshold_path = (
        tmp_path / "artifacts" / "thresholds" / "selected-threshold.json"
    )
    write_jsonl(
        validation_path,
        [
            LabeledExample(
                id="validation-1",
                language="en",
                text="Jane Cooper",
                entities=[
                    EntitySpan(
                        type=EntityType.PERSON,
                        start=0,
                        end=11,
                    )
                ],
                template_family="person-validation",
                generator_version="v1",
                split="validation",
            )
        ],
    )
    write_jsonl(
        challenge_path,
        [
            LabeledExample(
                id="challenge-1",
                language="en",
                text="No sensitive entity.",
                entities=[],
                template_family="negative-challenge",
                generator_version="v1",
                split="challenge",
            )
        ],
    )
    model_path = _write_model_artifact(
        tmp_path / "artifacts",
        validation_path,
    )
    loaded_paths: list[Path] = []

    def load_tokenizer(path: Path, **kwargs: object) -> SnapshotTokenizer:
        loaded_paths.append(path)
        assert path != model_path
        assert (path / "model.safetensors").read_bytes() == b"weights"
        return SnapshotTokenizer()

    def load_model(path: Path, **kwargs: object) -> SnapshotModel:
        assert path == loaded_paths[-1]
        return SnapshotModel()

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", load_tokenizer)
    monkeypatch.setattr(
        AutoModelForTokenClassification,
        "from_pretrained",
        load_model,
    )
    monkeypatch.setattr(
        threshold_cli,
        "parse_args",
        lambda: argparse.Namespace(
            validation=validation_path,
            ner_model=model_path,
            output=threshold_path,
        ),
    )

    await threshold_cli.run()

    selected = load_selected_threshold(threshold_path, model_path)
    assert selected.value == 0.95
    assert threshold_path.parent != model_path

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
        lambda: argparse.Namespace(
            challenge=challenge_path,
            regex_config=Path("config/regex-patterns.yaml"),
            ner_model=model_path,
            ner_threshold_artifact=threshold_path,
            qwen_url="http://127.0.0.1:8080",
            qwen_model="qwen3-0.6b-q4_k_m",
            qwen_timeout=2.0,
            qwen_pid=123,
            qwen_sha256="qwen-digest",
            llama_version="test",
            guardrail_cpu_limit=2,
            guardrail_memory_limit_mib=2048,
            qwen_cpu_limit=4,
            qwen_memory_limit_mib=4096,
            repetitions=3,
            output_dir=tmp_path / "reports",
        ),
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

    assert written_results[0]["environment"]["ner_artifact_sha256"] == (
        selected.model_artifact_sha256
    )
    assert len(loaded_paths) == 2
    assert all(not path.exists() for path in loaded_paths)
