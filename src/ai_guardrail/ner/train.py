from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from ai_guardrail.io import read_jsonl
from ai_guardrail.ner.alignment import align_spans_to_bio
from ai_guardrail.ner.labels import ID_TO_LABEL, LABEL_TO_ID
from ai_guardrail.ner.manifest import build_manifest, sha256_file, write_manifest

_MANIFEST_NAME = "training-manifest.json"


class NerDataset:
    def __init__(self, path: Path, tokenizer: Any) -> None:
        self.rows: list[dict[str, Any]] = []
        for example in read_jsonl(path):
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
    checksums: dict[str, str] = {}
    for path in sorted(output.iterdir(), key=lambda item: item.name):
        if path.is_symlink():
            raise ValueError("artifact directory must not contain symbolic links")
        if path.is_file() and path.name != _MANIFEST_NAME:
            checksums[path.name] = sha256_file(path)
    return checksums


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument(
        "--base-checkpoint",
        default="distilbert/distilbert-base-cased",
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


def main() -> None:
    args = parse_args()
    if args.output.is_symlink() or (args.output.exists() and not args.output.is_dir()):
        raise ValueError("output must be a local directory")
    resolved_output = args.output.resolve()
    if any(
        dataset_path.resolve().is_relative_to(resolved_output)
        for dataset_path in (args.train, args.validation)
    ):
        raise ValueError("output directory must not contain an input dataset")
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
    train_dataset = NerDataset(args.train, tokenizer)
    validation_dataset = NerDataset(args.validation, tokenizer)
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
        dataset_version="v1",
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
    )
    manifest["artifact_checksums"] = _artifact_checksums(args.output)
    write_manifest(args.output / _MANIFEST_NAME, manifest)


if __name__ == "__main__":
    main()
