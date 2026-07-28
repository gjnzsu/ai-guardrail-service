from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ai_guardrail.domain import LabeledExample
from ai_guardrail.io import JsonlSnapshot, read_jsonl_snapshot
from ai_guardrail.ner.alignment import align_spans_to_bio
from ai_guardrail.ner.labels import ID_TO_LABEL, LABEL_TO_ID
from ai_guardrail.ner.manifest import (
    BASE_CHECKPOINT,
    GENERATOR_VERSION,
    artifact_checksums,
    build_dataset_provenance,
    build_manifest,
    get_release_profile,
    sha256_bytes,
    write_manifest,
)

_MANIFEST_NAME = "training-manifest.json"


class NerDataset:
    def __init__(
        self,
        source: Path | tuple[LabeledExample, ...],
        tokenizer: Any,
    ) -> None:
        examples = (
            read_jsonl_snapshot(source).records
            if isinstance(source, Path)
            else source
        )
        self.rows: list[dict[str, Any]] = []
        for example in examples:
            encoded = tokenizer(
                example.text,
                return_offsets_mapping=True,
                truncation=True,
                max_length=512,
            )
            offsets = [tuple(pair) for pair in encoded.pop("offset_mapping")]
            encoded["labels"] = align_spans_to_bio(offsets, example.entities)
            self.rows.append(encoded)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


def _artifact_checksums(output: Path) -> dict[str, str]:
    try:
        return artifact_checksums(output)
    except ValueError:
        raise ValueError(
            "artifact directory must contain only regular local files"
        ) from None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument(
        "--release-version",
        choices=("v1", "v2"),
        default="v1",
    )
    parser.add_argument(
        "--base-checkpoint",
        default=BASE_CHECKPOINT,
    )
    return parser.parse_args()


def _load_ml_dependencies() -> tuple[Any, ...]:
    from huggingface_hub import model_info
    from transformers import (
        AutoModelForTokenClassification,
        AutoTokenizer,
        DataCollatorForTokenClassification,
        Trainer,
        TrainingArguments,
    )

    return (
        model_info,
        AutoModelForTokenClassification,
        AutoTokenizer,
        DataCollatorForTokenClassification,
        Trainer,
        TrainingArguments,
    )


@dataclass(frozen=True)
class ValidatedTrainingDatasets:
    train: JsonlSnapshot
    validation: JsonlSnapshot
    provenance: dict[str, Any]


def validate_training_datasets(
    train_path: Path,
    validation_path: Path,
    expected_generator_version: str = GENERATOR_VERSION,
) -> ValidatedTrainingDatasets:
    try:
        if train_path.resolve() == validation_path.resolve():
            raise ValueError
        train_snapshot = read_jsonl_snapshot(train_path)
        validation_snapshot = read_jsonl_snapshot(validation_path)
    except (OSError, ValueError):
        raise ValueError("invalid training dataset provenance") from None
    if (
        not train_snapshot.records
        or not validation_snapshot.records
        or train_snapshot.sha256 == validation_snapshot.sha256
        or any(example.split != "train" for example in train_snapshot.records)
        or any(
            example.split != "validation"
            for example in validation_snapshot.records
        )
        or any(
            example.generator_version != expected_generator_version
            for example in train_snapshot.records + validation_snapshot.records
        )
    ):
        raise ValueError("invalid training dataset provenance")
    train_ids = [example.id for example in train_snapshot.records]
    validation_ids = [example.id for example in validation_snapshot.records]
    train_families = {
        example.template_family for example in train_snapshot.records
    }
    validation_families = {
        example.template_family for example in validation_snapshot.records
    }
    train_content_hashes = {
        sha256_bytes(example.text.encode("utf-8"))
        for example in train_snapshot.records
    }
    validation_content_hashes = {
        sha256_bytes(example.text.encode("utf-8"))
        for example in validation_snapshot.records
    }
    if (
        len(set(train_ids)) != len(train_ids)
        or len(set(validation_ids)) != len(validation_ids)
        or not set(train_ids).isdisjoint(validation_ids)
        or not train_families.isdisjoint(validation_families)
        or not train_content_hashes.isdisjoint(validation_content_hashes)
    ):
        raise ValueError("invalid training dataset provenance")
    return ValidatedTrainingDatasets(
        train=train_snapshot,
        validation=validation_snapshot,
        provenance={
            "train": build_dataset_provenance(train_snapshot),
            "validation": build_dataset_provenance(validation_snapshot),
        },
    )


def main() -> None:
    args = parse_args()
    release_version = getattr(args, "release_version", "v1")
    profile = get_release_profile(release_version)
    if (
        args.base_checkpoint != BASE_CHECKPOINT
        or args.output.name != profile.artifact_name
    ):
        raise ValueError("invalid training configuration")
    if args.output.is_symlink() or (args.output.exists() and not args.output.is_dir()):
        raise ValueError("output must be a local directory")
    if args.output.exists() and next(args.output.iterdir(), None) is not None:
        raise ValueError("output directory must be empty")
    resolved_output = args.output.resolve()
    if any(
        dataset_path.resolve().is_relative_to(resolved_output)
        for dataset_path in (args.train, args.validation)
    ):
        raise ValueError("output directory must not contain an input dataset")
    validated_datasets = validate_training_datasets(
        args.train,
        args.validation,
        expected_generator_version=profile.generator_version,
    )
    args.output.mkdir(parents=True, exist_ok=True)

    (
        model_info,
        auto_model,
        auto_tokenizer,
        data_collator_type,
        trainer_type,
        training_arguments_type,
    ) = _load_ml_dependencies()

    resolved_revision = model_info(args.base_checkpoint).sha
    if not isinstance(resolved_revision, str) or not resolved_revision:
        raise RuntimeError("base checkpoint did not resolve to an immutable revision")
    tokenizer = auto_tokenizer.from_pretrained(
        args.base_checkpoint,
        revision=resolved_revision,
        use_fast=True,
    )
    model = auto_model.from_pretrained(
        args.base_checkpoint,
        revision=resolved_revision,
        num_labels=len(LABEL_TO_ID),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )
    train_dataset = NerDataset(validated_datasets.train.records, tokenizer)
    validation_dataset = NerDataset(
        validated_datasets.validation.records,
        tokenizer,
    )
    training_args = training_arguments_type(
        output_dir=str(args.output),
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=args.seed,
        report_to=[],
    )
    trainer = trainer_type(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        data_collator=data_collator_type(tokenizer),
    )
    trainer.train()
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    evaluation = {
        key: float(value)
        for key, value in trainer.evaluate().items()
        if isinstance(value, int | float)
    }
    manifest = build_manifest(
        base_checkpoint=args.base_checkpoint,
        base_revision=resolved_revision,
        dataset_version=profile.dataset_version,
        generator_version=profile.generator_version,
        label_mapping=LABEL_TO_ID,
        seed=args.seed,
        threshold=None,
        metrics=evaluation,
        hyperparameters={
            "learning_rate": 2e-5,
            "train_batch_size": 16,
            "eval_batch_size": 16,
            "epochs": 3,
            "weight_decay": 0.01,
            "max_length": 512,
        },
        datasets=validated_datasets.provenance,
        release_version=profile.version,
    )
    manifest["artifact_checksums"] = _artifact_checksums(args.output)
    write_manifest(args.output / _MANIFEST_NAME, manifest)


if __name__ == "__main__":
    main()
