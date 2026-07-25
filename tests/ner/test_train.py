import json
import subprocess
import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ai_guardrail.domain import EntitySpan, EntityType, LabeledExample
from ai_guardrail.io import write_jsonl
from ai_guardrail.ner import manifest as manifest_module
from ai_guardrail.ner import train as train_module
from ai_guardrail.ner.labels import ID_TO_LABEL, LABEL_TO_ID
from ai_guardrail.ner.train import NerDataset, _artifact_checksums


def test_training_cli_help_requires_dataset_and_output_paths() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "ai_guardrail.ner.train", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--train" in result.stdout
    assert "--validation" in result.stdout
    assert "--output" in result.stdout


class FakeTrainingTokenizer:
    def __call__(self, text: str, **kwargs: object) -> dict[str, list[object]]:
        assert text == "Jane Cooper"
        assert kwargs == {
            "return_offsets_mapping": True,
            "truncation": True,
            "max_length": 512,
        }
        return {
            "input_ids": [101, 1, 2, 102],
            "attention_mask": [1, 1, 1, 1],
            "offset_mapping": [(0, 0), (0, 4), (5, 11), (0, 0)],
        }


def test_ner_dataset_aligns_jsonl_spans_to_bio_labels(tmp_path: Path) -> None:
    dataset_path = tmp_path / "train.jsonl"
    write_jsonl(
        dataset_path,
        [
            LabeledExample(
                id="example-1",
                language="en",
                text="Jane Cooper",
                entities=[EntitySpan(type=EntityType.PERSON, start=0, end=11)],
                template_family="person",
                generator_version="v1",
                split="train",
            )
        ],
    )

    dataset = NerDataset(dataset_path, FakeTrainingTokenizer())

    assert len(dataset) == 1
    assert dataset[0] == {
        "input_ids": [101, 1, 2, 102],
        "attention_mask": [1, 1, 1, 1],
        "labels": [
            -100,
            LABEL_TO_ID["B-PERSON"],
            LABEL_TO_ID["I-PERSON"],
            -100,
        ],
    }


def test_artifact_checksums_are_stable_and_exclude_manifest_and_checkpoints(
    tmp_path: Path,
) -> None:
    (tmp_path / "model.safetensors").write_bytes(b"model")
    (tmp_path / "tokenizer.json").write_bytes(b"tokenizer")
    (tmp_path / "training-manifest.json").write_text("old manifest", encoding="utf-8")
    checkpoint = tmp_path / "checkpoint-1"
    checkpoint.mkdir()
    (checkpoint / "optimizer.pt").write_bytes(b"optimizer")

    assert _artifact_checksums(tmp_path) == {
        "model.safetensors": "9372c470eeadd5ecd9c3c74c2b3cb633f8e2f2fad799250a0f70d652b6b825e4",
        "tokenizer.json": "5f97e3774c51edd1d63706c2ec3826c564a067794770cdab0f8c4797971cacf9",
    }


def test_training_rejects_output_tree_containing_input_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "artifact"
    output.mkdir()
    train_path = output / "train.jsonl"
    validation_path = tmp_path / "validation.jsonl"
    train_path.write_text("{}\n", encoding="utf-8")
    validation_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        train_module,
        "parse_args",
        lambda: Namespace(
            train=train_path,
            validation=validation_path,
            output=output,
            seed=7,
            base_checkpoint="checkpoint",
        ),
    )
    monkeypatch.setattr(
        train_module,
        "_load_ml_dependencies",
        lambda: pytest.fail("ML dependencies must not load for an unsafe output path"),
    )

    with pytest.raises(
        ValueError,
        match="output directory must not contain an input dataset",
    ):
        train_module.main()


class FakeSavedTokenizer(FakeTrainingTokenizer):
    def save_pretrained(self, output: Path) -> None:
        (output / "tokenizer.json").write_bytes(b"tokenizer")


class FakeAutoTokenizer:
    calls: list[tuple[str, dict[str, object]]] = []
    tokenizer = FakeSavedTokenizer()

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str,
        **kwargs: object,
    ) -> FakeSavedTokenizer:
        cls.calls.append((checkpoint, kwargs))
        return cls.tokenizer


class FakeAutoModel:
    calls: list[tuple[str, dict[str, object]]] = []
    model = object()

    @classmethod
    def from_pretrained(cls, checkpoint: str, **kwargs: object) -> object:
        cls.calls.append((checkpoint, kwargs))
        return cls.model


class FakeTrainingArguments:
    latest: dict[str, object] = {}

    def __init__(self, **kwargs: object) -> None:
        type(self).latest = kwargs


class FakeTrainer:
    latest: "FakeTrainer | None" = None

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.trained = False
        type(self).latest = self

    def train(self) -> None:
        self.trained = True

    def save_model(self, output: Path) -> None:
        assert self.trained
        output.mkdir(parents=True, exist_ok=True)
        (output / "model.safetensors").write_bytes(b"model")

    def evaluate(self) -> dict[str, object]:
        assert self.trained
        return {"eval_loss": 0.25, "eval_runtime": 1, "ignored": "not numeric"}


def test_training_main_pins_revision_and_writes_artifact_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train_path = tmp_path / "train.jsonl"
    validation_path = tmp_path / "validation.jsonl"
    output = tmp_path / "artifact"
    for path, split in ((train_path, "train"), (validation_path, "validation")):
        write_jsonl(
            path,
            [
                LabeledExample(
                    id=f"{split}-1",
                    language="en",
                    text="Jane Cooper",
                    entities=[EntitySpan(type=EntityType.PERSON, start=0, end=11)],
                    template_family="person",
                    generator_version="v1",
                    split=split,
                )
            ],
        )

    FakeAutoTokenizer.calls = []
    FakeAutoModel.calls = []
    FakeTrainer.latest = None
    monkeypatch.setattr(
        train_module,
        "parse_args",
        lambda: Namespace(
            train=train_path,
            validation=validation_path,
            output=output,
            seed=7,
            base_checkpoint="distilbert/distilbert-base-cased",
        ),
    )
    monkeypatch.setattr(
        train_module,
        "_load_ml_dependencies",
        lambda: (
            lambda checkpoint: SimpleNamespace(sha="immutable-revision"),
            FakeAutoModel,
            FakeAutoTokenizer,
            lambda tokenizer: ("collator", tokenizer),
            FakeTrainer,
            FakeTrainingArguments,
        ),
        raising=False,
    )
    monkeypatch.setattr(
        manifest_module.importlib.metadata,
        "version",
        lambda name: {
            "accelerate": "1.3.0",
            "torch": "2.5.0",
            "transformers": "4.49.0",
        }[name],
    )

    train_module.main()

    assert FakeAutoTokenizer.calls == [
        (
            "distilbert/distilbert-base-cased",
            {"revision": "immutable-revision", "use_fast": True},
        )
    ]
    assert FakeAutoModel.calls == [
        (
            "distilbert/distilbert-base-cased",
            {
                "revision": "immutable-revision",
                "num_labels": len(LABEL_TO_ID),
                "id2label": ID_TO_LABEL,
                "label2id": LABEL_TO_ID,
            },
        )
    ]
    assert FakeTrainingArguments.latest["seed"] == 7
    assert FakeTrainingArguments.latest["report_to"] == []
    assert FakeTrainer.latest is not None
    assert len(FakeTrainer.latest.kwargs["train_dataset"]) == 1
    assert len(FakeTrainer.latest.kwargs["eval_dataset"]) == 1
    manifest = json.loads((output / "training-manifest.json").read_text(encoding="utf-8"))
    assert manifest["base_revision"] == "immutable-revision"
    assert manifest["seed"] == 7
    assert manifest["threshold"] is None
    assert manifest["metrics"] == {"eval_loss": 0.25, "eval_runtime": 1.0}
    assert manifest["artifact_checksums"] == {
        "model.safetensors": (
            "9372c470eeadd5ecd9c3c74c2b3cb633f8e2f2fad799250a0f70d652b6b825e4"
        ),
        "tokenizer.json": (
            "5f97e3774c51edd1d63706c2ec3826c564a067794770cdab0f8c4797971cacf9"
        ),
    }
