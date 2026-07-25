import hashlib
import importlib.metadata
import json
import math
import platform
from pathlib import Path
from typing import Any

from ai_guardrail.ner.labels import LABEL_TO_ID

ARTIFACT_NAME = "ai-guardrail-ner-en-v1"
BASE_CHECKPOINT = "distilbert/distilbert-base-cased"
GENERATOR_VERSION = "v1"


def is_exact_label_mapping(label_mapping: object) -> bool:
    return (
        isinstance(label_mapping, dict)
        and label_mapping.keys() == LABEL_TO_ID.keys()
        and all(
            type(label_mapping[label]) is int
            and label_mapping[label] == expected_id
            for label, expected_id in LABEL_TO_ID.items()
        )
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(
    *,
    base_checkpoint: str,
    base_revision: str,
    dataset_version: str,
    generator_version: str,
    label_mapping: dict[str, int],
    seed: int,
    threshold: float | None,
    metrics: dict[str, float],
    hyperparameters: dict[str, int | float],
) -> dict[str, Any]:
    if generator_version != GENERATOR_VERSION:
        raise ValueError("generator version does not match expected version")
    if not is_exact_label_mapping(label_mapping):
        raise ValueError("label mapping does not match expected mapping")
    if threshold is not None and (
        not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0
    ):
        raise ValueError("threshold must be between 0 and 1")
    return {
        "artifact_name": ARTIFACT_NAME,
        "base_checkpoint": base_checkpoint,
        "base_revision": base_revision,
        "dataset_version": dataset_version,
        "generator_version": generator_version,
        "label_mapping": dict(label_mapping),
        "seed": seed,
        "threshold": threshold,
        "metrics": metrics,
        "hyperparameters": hyperparameters,
        "python_version": platform.python_version(),
        "library_versions": {
            name: importlib.metadata.version(name)
            for name in ("accelerate", "torch", "transformers")
        },
    }


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(manifest, indent=2, sort_keys=True))
        handle.write("\n")
