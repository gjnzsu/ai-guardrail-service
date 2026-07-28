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
from ai_guardrail.ner.manifest import ARTIFACT_NAME, BASE_CHECKPOINT
from ai_guardrail.ner.train import (
    NerDataset,
    _artifact_checksums,
    validate_training_datasets,
)


def dataset_example(
    *,
    record_id: str,
    split: str,
    family: str,
    text: str,
    generator_version: str = "v1",
) -> LabeledExample:
    return LabeledExample(
        id=record_id,
        language="en",
        text=text,
        entities=[],
        template_family=family,
        generator_version=generator_version,
        split=split,
    )


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
    assert "--release-version" in result.stdout


def test_training_cli_help_does_not_import_optional_ml_modules() -> None:
    script = """
import builtins
import runpy
import sys

real_import = builtins.__import__
blocked = {"accelerate", "huggingface_hub", "torch", "transformers"}

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name.split(".", 1)[0] in blocked:
        raise AssertionError(f"optional ML import attempted: {name}")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import
sys.argv = ["ai_guardrail.ner.train", "--help"]
runpy.run_module("ai_guardrail.ner.train", run_name="__main__")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--train" in result.stdout


class FakeTrainingTokenizer:
    def __call__(self, text: str, **kwargs: object) -> dict[str, list[object]]:
        assert text.startswith("Jane Cooper")
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


def test_artifact_checksums_are_recursive_and_exclude_only_manifest(
    tmp_path: Path,
) -> None:
    (tmp_path / "model.safetensors").write_bytes(b"model")
    (tmp_path / "tokenizer.json").write_bytes(b"tokenizer")
    (tmp_path / "training-manifest.json").write_text("old manifest", encoding="utf-8")
    checkpoint = tmp_path / "checkpoint-1"
    checkpoint.mkdir()
    (checkpoint / "optimizer.pt").write_bytes(b"optimizer")

    assert _artifact_checksums(tmp_path) == {
        "checkpoint-1/optimizer.pt": (
            "23732d00643137c9b14b11bdb90f47a8e97b4da42cdfc0887912f4ff732c689c"
        ),
        "model.safetensors": "9372c470eeadd5ecd9c3c74c2b3cb633f8e2f2fad799250a0f70d652b6b825e4",
        "tokenizer.json": "5f97e3774c51edd1d63706c2ec3826c564a067794770cdab0f8c4797971cacf9",
    }


def test_training_dataset_validation_records_hashed_provenance(
    tmp_path: Path,
) -> None:
    train_path = tmp_path / "train.jsonl"
    validation_path = tmp_path / "validation.jsonl"
    write_jsonl(
        train_path,
        [dataset_example(record_id="train-1", split="train", family="train-a", text="A")],
    )
    write_jsonl(
        validation_path,
        [
            dataset_example(
                record_id="validation-1",
                split="validation",
                family="validation-a",
                text="B",
            )
        ],
    )

    validated = validate_training_datasets(train_path, validation_path)
    provenance = validated.provenance

    assert provenance["train"]["sha256"] == manifest_module.sha256_file(train_path)
    assert provenance["validation"]["sha256"] == manifest_module.sha256_file(
        validation_path
    )
    assert provenance["train"]["record_count"] == 1
    assert provenance["train"]["template_families"] == ["train-a"]
    assert provenance["validation"]["template_families"] == ["validation-a"]
    assert provenance["train"]["record_id_hashes"] == [
        manifest_module.sha256_bytes(b"train-1")
    ]
    assert provenance["validation"]["content_hashes"] == [
        manifest_module.sha256_bytes(b"B")
    ]


def test_training_dataset_validation_accepts_expected_v2_release(
    tmp_path: Path,
) -> None:
    train_path = tmp_path / "train.jsonl"
    validation_path = tmp_path / "validation.jsonl"
    write_jsonl(
        train_path,
        [
            dataset_example(
                record_id="v2-train-1",
                split="train",
                family="v2-train-family",
                text="V2 train",
                generator_version="v2",
            )
        ],
    )
    write_jsonl(
        validation_path,
        [
            dataset_example(
                record_id="v2-validation-1",
                split="validation",
                family="v2-validation-family",
                text="V2 validation",
                generator_version="v2",
            )
        ],
    )

    validated = validate_training_datasets(
        train_path,
        validation_path,
        expected_generator_version="v2",
    )

    assert validated.train.records[0].generator_version == "v2"
    assert validated.validation.records[0].generator_version == "v2"


@pytest.mark.parametrize(
    "case",
    [
        "empty-train",
        "wrong-train-label",
        "swapped",
        "same-path",
        "same-sha",
        "duplicate-id",
        "duplicate-train-id",
        "duplicate-validation-id",
        "family-overlap",
        "content-overlap",
    ],
)
def test_training_dataset_validation_rejects_contaminated_splits(
    tmp_path: Path,
    case: str,
) -> None:
    train_path = tmp_path / "train.jsonl"
    validation_path = tmp_path / "validation.jsonl"
    train = [
        dataset_example(
            record_id="train-1",
            split="train",
            family="train-family",
            text="train text",
        )
    ]
    validation = [
        dataset_example(
            record_id="validation-1",
            split="validation",
            family="validation-family",
            text="validation text",
        )
    ]
    if case == "empty-train":
        train = []
    elif case == "wrong-train-label":
        train[0] = train[0].model_copy(update={"split": "validation"})
    elif case == "swapped":
        train, validation = validation, train
    elif case == "duplicate-id":
        validation[0] = validation[0].model_copy(update={"id": "train-1"})
    elif case == "duplicate-train-id":
        train.append(train[0].model_copy())
    elif case == "duplicate-validation-id":
        validation.append(validation[0].model_copy())
    elif case == "family-overlap":
        validation[0] = validation[0].model_copy(
            update={"template_family": "train-family"}
        )
    elif case == "content-overlap":
        validation[0] = validation[0].model_copy(update={"text": "train text"})
    write_jsonl(train_path, train)
    write_jsonl(validation_path, validation)
    if case == "same-path":
        validation_path = train_path
    elif case == "same-sha":
        validation_path.write_bytes(train_path.read_bytes())

    with pytest.raises(ValueError, match="invalid training dataset provenance"):
        validate_training_datasets(train_path, validation_path)


def test_validated_training_snapshot_is_used_after_paths_change(
    tmp_path: Path,
) -> None:
    train_path = tmp_path / "train.jsonl"
    validation_path = tmp_path / "validation.jsonl"
    original_train = dataset_example(
        record_id="train-1",
        split="train",
        family="train-family",
        text="Jane Cooper",
    )
    write_jsonl(train_path, [original_train])
    write_jsonl(
        validation_path,
        [
            dataset_example(
                record_id="validation-1",
                split="validation",
                family="validation-family",
                text="Validation content",
            )
        ],
    )

    validated = validate_training_datasets(train_path, validation_path)
    write_jsonl(
        train_path,
        [
            dataset_example(
                record_id="train-replacement",
                split="train",
                family="replacement-family",
                text="Private replacement text",
            )
        ],
    )
    dataset = NerDataset(validated.train.records, FakeTrainingTokenizer())

    assert len(dataset) == 1
    assert validated.train.records == (original_train,)


def test_training_rejects_output_tree_containing_input_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / ARTIFACT_NAME
    train_path = output / "train.jsonl"
    validation_path = tmp_path / "validation.jsonl"
    validation_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        train_module,
        "parse_args",
        lambda: Namespace(
            train=train_path,
            validation=validation_path,
            output=output,
            seed=7,
            base_checkpoint=BASE_CHECKPOINT,
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


def test_training_rejects_wrong_base_checkpoint_before_ml_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        train_module,
        "parse_args",
        lambda: Namespace(
            train=tmp_path / "train.jsonl",
            validation=tmp_path / "validation.jsonl",
            output=tmp_path / ARTIFACT_NAME,
            seed=7,
            base_checkpoint="private/unapproved-checkpoint",
        ),
    )
    monkeypatch.setattr(
        train_module,
        "_load_ml_dependencies",
        lambda: pytest.fail("ML dependencies must not load for a wrong checkpoint"),
    )

    with pytest.raises(ValueError, match="invalid training configuration") as exc_info:
        train_module.main()

    assert "private/unapproved-checkpoint" not in str(exc_info.value)


def test_training_rejects_wrong_artifact_name_before_ml_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        train_module,
        "parse_args",
        lambda: Namespace(
            train=tmp_path / "train.jsonl",
            validation=tmp_path / "validation.jsonl",
            output=tmp_path / "private-renamed-artifact",
            seed=7,
            base_checkpoint=BASE_CHECKPOINT,
        ),
    )
    monkeypatch.setattr(
        train_module,
        "_load_ml_dependencies",
        lambda: pytest.fail("ML dependencies must not load for a wrong artifact name"),
    )

    with pytest.raises(ValueError, match="invalid training configuration") as exc_info:
        train_module.main()

    assert "private-renamed-artifact" not in str(exc_info.value)


def test_training_rejects_prepopulated_output_before_ml_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / ARTIFACT_NAME
    output.mkdir()
    (output / "unrelated-private-file.txt").write_text("stale", encoding="utf-8")
    monkeypatch.setattr(
        train_module,
        "parse_args",
        lambda: Namespace(
            train=tmp_path / "train.jsonl",
            validation=tmp_path / "validation.jsonl",
            output=output,
            seed=7,
            base_checkpoint=BASE_CHECKPOINT,
        ),
    )
    monkeypatch.setattr(
        train_module,
        "_load_ml_dependencies",
        lambda: pytest.fail("ML dependencies must not load for stale output"),
    )

    with pytest.raises(ValueError, match="output directory must be empty") as exc_info:
        train_module.main()

    assert "unrelated-private-file.txt" not in str(exc_info.value)


def test_training_rejects_unexpected_generator_version_before_ml_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train_path = tmp_path / "train.jsonl"
    validation_path = tmp_path / "validation.jsonl"
    write_jsonl(
        train_path,
        [
            LabeledExample(
                id="train-1",
                language="en",
                text="Jane Cooper",
                entities=[EntitySpan(type=EntityType.PERSON, start=0, end=11)],
                template_family="person",
                generator_version="v1",
                split="train",
            )
        ],
    )
    validation_record = {
        "id": "validation-private",
        "language": "en",
        "text": "private validation text",
        "entities": [],
        "template_family": "private-family",
        "generator_version": "private-generator-v2",
        "split": "validation",
    }
    validation_path.write_text(
        json.dumps(validation_record) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        train_module,
        "parse_args",
        lambda: Namespace(
            train=train_path,
            validation=validation_path,
            output=tmp_path / ARTIFACT_NAME,
            seed=7,
            base_checkpoint=BASE_CHECKPOINT,
        ),
    )
    monkeypatch.setattr(
        train_module,
        "_load_ml_dependencies",
        lambda: pytest.fail("ML dependencies must not load for invalid provenance"),
    )

    with pytest.raises(ValueError) as exc_info:
        train_module.main()

    error = str(exc_info.value)
    assert "private-generator-v2" not in error
    assert "private validation text" not in error
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__suppress_context__


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
        (output / "config.json").write_text(
            json.dumps(
                {
                    "label2id": LABEL_TO_ID,
                    "id2label": {
                        str(identifier): label
                        for identifier, label in ID_TO_LABEL.items()
                    },
                }
            ),
            encoding="utf-8",
        )
        (output / "model.safetensors").write_bytes(b"model")

    def evaluate(self) -> dict[str, object]:
        assert self.trained
        return {"eval_loss": 0.25, "eval_runtime": 1, "ignored": "not numeric"}


@pytest.mark.parametrize("precreate_output", [False, True])
@pytest.mark.parametrize(
    ("release_version", "artifact_name", "generator_version"),
    [
        ("v1", ARTIFACT_NAME, "v1"),
        ("v2", "ai-guardrail-ner-en-v2", "v2"),
    ],
)
def test_training_main_pins_revision_and_writes_artifact_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    precreate_output: bool,
    release_version: str,
    artifact_name: str,
    generator_version: str,
) -> None:
    train_path = tmp_path / "train.jsonl"
    validation_path = tmp_path / "validation.jsonl"
    output = tmp_path / artifact_name
    if precreate_output:
        output.mkdir()
    for path, split in ((train_path, "train"), (validation_path, "validation")):
        write_jsonl(
            path,
            [
                LabeledExample(
                    id=f"{split}-1",
                    language="en",
                    text=(
                        "Jane Cooper"
                        if split == "train"
                        else "Jane Cooper validation"
                    ),
                    entities=[EntitySpan(type=EntityType.PERSON, start=0, end=11)],
                    template_family=f"person-{split}",
                    generator_version=generator_version,
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
            release_version=release_version,
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
    assert manifest["artifact_name"] == artifact_name
    assert manifest["dataset_version"] == release_version
    assert manifest["generator_version"] == generator_version
    assert manifest["label_mapping"] == LABEL_TO_ID
    assert manifest["seed"] == 7
    assert manifest["threshold"] is None
    assert manifest["metrics"] == {"eval_loss": 0.25, "eval_runtime": 1.0}
    assert manifest["manifest_schema_version"] == 2
    assert manifest["datasets"]["train"]["sha256"] == manifest_module.sha256_file(
        train_path
    )
    assert manifest["datasets"]["validation"]["sha256"] == (
        manifest_module.sha256_file(validation_path)
    )
    assert manifest["datasets"]["train"]["template_families"] == ["person-train"]
    assert manifest["datasets"]["validation"]["template_families"] == [
        "person-validation"
    ]
    assert "Jane Cooper" not in json.dumps(manifest)
    assert manifest["artifact_checksums"] == {
        "config.json": manifest_module.sha256_file(output / "config.json"),
        "model.safetensors": (
            "9372c470eeadd5ecd9c3c74c2b3cb633f8e2f2fad799250a0f70d652b6b825e4"
        ),
        "tokenizer.json": (
            "5f97e3774c51edd1d63706c2ec3826c564a067794770cdab0f8c4797971cacf9"
        ),
    }
