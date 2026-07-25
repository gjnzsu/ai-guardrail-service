import hashlib
import importlib.metadata
import json
import math
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ai_guardrail.domain import LabeledExample
from ai_guardrail.ner.labels import LABEL_TO_ID

ARTIFACT_NAME = "ai-guardrail-ner-en-v1"
BASE_CHECKPOINT = "distilbert/distilbert-base-cased"
GENERATOR_VERSION = "v1"
MANIFEST_SCHEMA_VERSION = 2
MANIFEST_FILENAME = "training-manifest.json"

_ARTIFACT_ERROR = "invalid NER model artifact"


@dataclass(frozen=True)
class VerifiedModelArtifact:
    manifest: dict[str, Any]
    manifest_sha256: str
    artifact_sha256: str


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


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def build_dataset_provenance(
    path: Path,
    examples: list[LabeledExample],
) -> dict[str, Any]:
    return {
        "sha256": sha256_file(path),
        "record_count": len(examples),
        "record_id_hashes": sorted(
            sha256_bytes(example.id.encode("utf-8"))
            for example in examples
        ),
        "template_families": sorted(
            {example.template_family for example in examples}
        ),
        "content_hashes": sorted(
            sha256_bytes(example.text.encode("utf-8"))
            for example in examples
        ),
    }


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_dataset_provenance(value: object) -> bool:
    if not isinstance(value, dict) or value.keys() != {"train", "validation"}:
        return False
    required = {
        "sha256",
        "record_count",
        "record_id_hashes",
        "template_families",
        "content_hashes",
    }
    for split in ("train", "validation"):
        item = value.get(split)
        if not isinstance(item, dict) or item.keys() != required:
            return False
        if (
            not _is_sha256(item["sha256"])
            or type(item["record_count"]) is not int
            or item["record_count"] <= 0
            or not isinstance(item["record_id_hashes"], list)
            or len(item["record_id_hashes"]) != item["record_count"]
            or len(set(item["record_id_hashes"])) != item["record_count"]
            or any(not _is_sha256(digest) for digest in item["record_id_hashes"])
            or not isinstance(item["template_families"], list)
            or not item["template_families"]
            or len(set(item["template_families"])) != len(item["template_families"])
            or any(
                not isinstance(family, str) or not family
                for family in item["template_families"]
            )
            or not isinstance(item["content_hashes"], list)
            or len(item["content_hashes"]) != item["record_count"]
            or any(not _is_sha256(digest) for digest in item["content_hashes"])
        ):
            return False
    return True


def _read_regular_file(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError(_ARTIFACT_ERROR)
    try:
        return path.read_bytes()
    except OSError:
        raise ValueError(_ARTIFACT_ERROR) from None


def _decode_json_object(content: bytes) -> dict[str, Any]:
    try:
        value = json.loads(content)
    except (UnicodeError, json.JSONDecodeError):
        raise ValueError(_ARTIFACT_ERROR) from None
    if not isinstance(value, dict):
        raise ValueError(_ARTIFACT_ERROR)
    return value


def _validate_config_labels(config: dict[str, Any]) -> None:
    if not is_exact_label_mapping(config.get("label2id")):
        raise ValueError(_ARTIFACT_ERROR)
    expected_id_to_label = {
        str(identifier): label for label, identifier in LABEL_TO_ID.items()
    }
    if config.get("id2label") != expected_id_to_label:
        raise ValueError(_ARTIFACT_ERROR)


def verify_model_artifact(model_path: Path) -> VerifiedModelArtifact:
    if (
        model_path.name != ARTIFACT_NAME
        or model_path.is_symlink()
        or not model_path.is_dir()
    ):
        raise ValueError(_ARTIFACT_ERROR)
    manifest_content = _read_regular_file(model_path / MANIFEST_FILENAME)
    manifest = _decode_json_object(manifest_content)
    if manifest.get("manifest_schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported NER training manifest schema")
    checksums = manifest.get("artifact_checksums")
    if (
        manifest.get("artifact_name") != ARTIFACT_NAME
        or manifest.get("base_checkpoint") != BASE_CHECKPOINT
        or manifest.get("generator_version") != GENERATOR_VERSION
        or not is_exact_label_mapping(manifest.get("label_mapping"))
        or not _is_dataset_provenance(manifest.get("datasets"))
        or not isinstance(checksums, dict)
        or not checksums
        or "config.json" not in checksums
    ):
        raise ValueError(_ARTIFACT_ERROR)
    actual_artifact_names: set[str] = set()
    try:
        for artifact_path in model_path.iterdir():
            if artifact_path.is_symlink():
                raise ValueError(_ARTIFACT_ERROR)
            if artifact_path.is_file() and artifact_path.name != MANIFEST_FILENAME:
                actual_artifact_names.add(artifact_path.name)
            elif not artifact_path.is_dir() and artifact_path.name != MANIFEST_FILENAME:
                raise ValueError(_ARTIFACT_ERROR)
    except OSError:
        raise ValueError(_ARTIFACT_ERROR) from None
    if set(checksums) != actual_artifact_names:
        raise ValueError(_ARTIFACT_ERROR)
    for name, expected_sha256 in checksums.items():
        if (
            not isinstance(name, str)
            or not name
            or Path(name).name != name
            or not _is_sha256(expected_sha256)
        ):
            raise ValueError(_ARTIFACT_ERROR)
        artifact_path = model_path / name
        content = _read_regular_file(artifact_path)
        if sha256_bytes(content) != expected_sha256:
            raise ValueError(_ARTIFACT_ERROR)
    config = _decode_json_object(_read_regular_file(model_path / "config.json"))
    _validate_config_labels(config)
    canonical_checksums = json.dumps(
        checksums,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return VerifiedModelArtifact(
        manifest=manifest,
        manifest_sha256=sha256_bytes(manifest_content),
        artifact_sha256=sha256_bytes(canonical_checksums),
    )


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
    datasets: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if generator_version != GENERATOR_VERSION:
        raise ValueError("generator version does not match expected version")
    if not is_exact_label_mapping(label_mapping):
        raise ValueError("label mapping does not match expected mapping")
    if threshold is not None and (
        not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0
    ):
        raise ValueError("threshold must be between 0 and 1")
    manifest = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
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
    if datasets is not None:
        if not _is_dataset_provenance(datasets):
            raise ValueError("invalid training dataset provenance")
        manifest["datasets"] = datasets
    return manifest


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(manifest, indent=2, sort_keys=True))
        handle.write("\n")
