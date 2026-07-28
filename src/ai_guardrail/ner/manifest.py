import hashlib
import importlib.metadata
import json
import math
import platform
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from ai_guardrail.io import JsonlSnapshot
from ai_guardrail.ner.labels import LABEL_TO_ID

BASE_CHECKPOINT = "distilbert/distilbert-base-cased"
MANIFEST_SCHEMA_VERSION = 2
MANIFEST_FILENAME = "training-manifest.json"

_ARTIFACT_ERROR = "invalid NER model artifact"


@dataclass(frozen=True)
class NerReleaseProfile:
    version: str
    artifact_name: str
    generator_version: str
    dataset_version: str


_RELEASE_PROFILES = {
    "v1": NerReleaseProfile(
        version="v1",
        artifact_name="ai-guardrail-ner-en-v1",
        generator_version="v1",
        dataset_version="v1",
    ),
    "v2": NerReleaseProfile(
        version="v2",
        artifact_name="ai-guardrail-ner-en-v2",
        generator_version="v2",
        dataset_version="v2",
    ),
}

ARTIFACT_NAME = _RELEASE_PROFILES["v1"].artifact_name
GENERATOR_VERSION = _RELEASE_PROFILES["v1"].generator_version


def get_release_profile(version: str) -> NerReleaseProfile:
    try:
        return _RELEASE_PROFILES[version]
    except (KeyError, TypeError):
        raise ValueError("unsupported NER release") from None


def get_release_profile_for_artifact(artifact_name: str) -> NerReleaseProfile:
    profiles = [
        profile
        for profile in _RELEASE_PROFILES.values()
        if profile.artifact_name == artifact_name
    ]
    if len(profiles) != 1:
        raise ValueError(_ARTIFACT_ERROR)
    return profiles[0]


@dataclass(frozen=True)
class VerifiedModelArtifact:
    manifest: dict[str, Any]
    manifest_sha256: str
    artifact_sha256: str


@dataclass(frozen=True)
class VerifiedModelSnapshot:
    path: Path
    verified: VerifiedModelArtifact


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


def artifact_checksums(root: Path) -> dict[str, str]:
    checksums: dict[str, str] = {}
    pending = [root]
    try:
        while pending:
            directory = pending.pop()
            for path in directory.iterdir():
                if path.is_symlink():
                    raise ValueError(_ARTIFACT_ERROR)
                if path.is_dir():
                    pending.append(path)
                elif path.is_file():
                    relative_name = path.relative_to(root).as_posix()
                    if relative_name != MANIFEST_FILENAME:
                        checksums[relative_name] = sha256_file(path)
                else:
                    raise ValueError(_ARTIFACT_ERROR)
    except OSError:
        raise ValueError(_ARTIFACT_ERROR) from None
    return dict(sorted(checksums.items()))


def build_dataset_provenance(
    snapshot: JsonlSnapshot,
) -> dict[str, Any]:
    return {
        "sha256": snapshot.sha256,
        "record_count": len(snapshot.records),
        "record_id_hashes": sorted(
            sha256_bytes(example.id.encode("utf-8"))
            for example in snapshot.records
        ),
        "template_families": sorted(
            {example.template_family for example in snapshot.records}
        ),
        "content_hashes": sorted(
            sha256_bytes(example.text.encode("utf-8"))
            for example in snapshot.records
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
    try:
        profile = get_release_profile_for_artifact(model_path.name)
    except ValueError:
        raise ValueError(_ARTIFACT_ERROR) from None
    if model_path.is_symlink() or not model_path.is_dir():
        raise ValueError(_ARTIFACT_ERROR)
    manifest_content = _read_regular_file(model_path / MANIFEST_FILENAME)
    manifest = _decode_json_object(manifest_content)
    if manifest.get("manifest_schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported NER training manifest schema")
    checksums = manifest.get("artifact_checksums")
    if (
        manifest.get("artifact_name") != profile.artifact_name
        or manifest.get("base_checkpoint") != BASE_CHECKPOINT
        or manifest.get("generator_version") != profile.generator_version
        or manifest.get("dataset_version") != profile.dataset_version
        or not is_exact_label_mapping(manifest.get("label_mapping"))
        or not _is_dataset_provenance(manifest.get("datasets"))
        or not isinstance(checksums, dict)
        or not checksums
        or "config.json" not in checksums
    ):
        raise ValueError(_ARTIFACT_ERROR)
    actual_checksums = artifact_checksums(model_path)
    if set(checksums) != set(actual_checksums):
        raise ValueError(_ARTIFACT_ERROR)
    for name, expected_sha256 in checksums.items():
        relative = PurePosixPath(name) if isinstance(name, str) else None
        if (
            not isinstance(name, str)
            or not name
            or relative is None
            or relative.is_absolute()
            or ".." in relative.parts
            or "\\" in name
            or not _is_sha256(expected_sha256)
        ):
            raise ValueError(_ARTIFACT_ERROR)
        if actual_checksums[name] != expected_sha256:
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


@contextmanager
def verified_model_snapshot(
    source_path: Path,
    *,
    expected_artifact_sha256: str | None = None,
    expected_manifest_sha256: str | None = None,
) -> Iterator[VerifiedModelSnapshot]:
    source = verify_model_artifact(source_path)
    if (
        expected_artifact_sha256 is not None
        and source.artifact_sha256 != expected_artifact_sha256
    ) or (
        expected_manifest_sha256 is not None
        and source.manifest_sha256 != expected_manifest_sha256
    ):
        raise ValueError(_ARTIFACT_ERROR)
    with tempfile.TemporaryDirectory(
        prefix="ai-guardrail-verified-",
    ) as temporary_directory:
        snapshot_path = (
            Path(temporary_directory)
            / source.manifest["artifact_name"]
        )
        snapshot_path.mkdir()
        try:
            shutil.copyfile(
                source_path / MANIFEST_FILENAME,
                snapshot_path / MANIFEST_FILENAME,
            )
            for name in source.manifest["artifact_checksums"]:
                relative = PurePosixPath(name)
                destination = snapshot_path.joinpath(*relative.parts)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(
                    source_path.joinpath(*relative.parts),
                    destination,
                )
        except OSError:
            raise ValueError(_ARTIFACT_ERROR) from None
        copied = verify_model_artifact(snapshot_path)
        if (
            copied.artifact_sha256 != source.artifact_sha256
            or copied.manifest_sha256 != source.manifest_sha256
        ):
            raise ValueError(_ARTIFACT_ERROR)
        snapshot = VerifiedModelSnapshot(
            path=snapshot_path,
            verified=copied,
        )
        try:
            yield snapshot
        finally:
            final = verify_model_artifact(snapshot_path)
            if (
                final.artifact_sha256 != copied.artifact_sha256
                or final.manifest_sha256 != copied.manifest_sha256
            ):
                raise ValueError(_ARTIFACT_ERROR)


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
    release_version: str = "v1",
) -> dict[str, Any]:
    profile = get_release_profile(release_version)
    if generator_version != profile.generator_version:
        raise ValueError("generator version does not match expected version")
    if dataset_version != profile.dataset_version:
        raise ValueError("dataset version does not match expected version")
    if not is_exact_label_mapping(label_mapping):
        raise ValueError("label mapping does not match expected mapping")
    if threshold is not None and (
        not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0
    ):
        raise ValueError("threshold must be between 0 and 1")
    manifest = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "artifact_name": profile.artifact_name,
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
