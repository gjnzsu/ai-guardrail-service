import json
import math
from pathlib import Path

import pytest

from ai_guardrail.ner import manifest as manifest_module
from ai_guardrail.ner.labels import LABEL_TO_ID
from ai_guardrail.ner.manifest import (
    ARTIFACT_NAME,
    build_manifest,
    sha256_file,
    verify_model_artifact,
    write_manifest,
)


def write_verified_artifact(tmp_path: Path) -> tuple[Path, dict[str, str]]:
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
    manifest = {
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
                "template_families": ["train-family"],
                "content_hashes": ["3" * 64],
            },
            "validation": {
                "sha256": "4" * 64,
                "record_count": 1,
                "record_id_hashes": ["5" * 64],
                "template_families": ["validation-family"],
                "content_hashes": ["6" * 64],
            },
        },
    }
    write_manifest(model_path / "training-manifest.json", manifest)
    return model_path, checksums


def test_sha256_file_is_reproducible(tmp_path: Path) -> None:
    artifact = tmp_path / "weights.bin"
    artifact.write_bytes(b"guardrail-model")

    assert sha256_file(artifact) == (
        "64c172d264fcea853f3556031a49bb15710ccedb51712322ca7bd6d997c4cb1a"
    )


@pytest.mark.parametrize("filename", ["model.safetensors", "config.json", "tokenizer.json"])
def test_verify_model_artifact_rejects_tampered_files(
    tmp_path: Path,
    filename: str,
) -> None:
    model_path, _ = write_verified_artifact(tmp_path)
    (model_path / filename).write_bytes(b"tampered-private-content")

    with pytest.raises(ValueError, match="invalid NER model artifact") as exc_info:
        verify_model_artifact(model_path)

    assert filename not in str(exc_info.value)
    assert "tampered-private-content" not in str(exc_info.value)


def test_verify_model_artifact_rejects_missing_file_without_path_leak(
    tmp_path: Path,
) -> None:
    model_path, _ = write_verified_artifact(tmp_path)
    (model_path / "model.safetensors").unlink()

    with pytest.raises(ValueError, match="invalid NER model artifact") as exc_info:
        verify_model_artifact(model_path)

    assert "model.safetensors" not in str(exc_info.value)


def test_verify_model_artifact_rejects_symlinked_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path, checksums = write_verified_artifact(tmp_path)
    target = tmp_path / "private-weights.bin"
    target.write_bytes(b"weights")
    artifact = model_path / "model.safetensors"
    artifact.unlink()
    try:
        artifact.symlink_to(target)
    except OSError:
        original = Path.is_symlink
        monkeypatch.setattr(
            Path,
            "is_symlink",
            lambda path: (
                path == artifact
                or original(path)
            ),
        )
    assert sha256_file(target) == checksums["model.safetensors"]

    with pytest.raises(ValueError, match="invalid NER model artifact") as exc_info:
        verify_model_artifact(model_path)

    assert "private-weights.bin" not in str(exc_info.value)


def test_verify_model_artifact_rejects_directory_checksum_target(
    tmp_path: Path,
) -> None:
    model_path, _ = write_verified_artifact(tmp_path)
    artifact = model_path / "model.safetensors"
    artifact.unlink()
    artifact.mkdir()

    with pytest.raises(ValueError, match="invalid NER model artifact"):
        verify_model_artifact(model_path)


def test_verify_model_artifact_rejects_unchecked_top_level_file(
    tmp_path: Path,
) -> None:
    model_path, _ = write_verified_artifact(tmp_path)
    (model_path / "unchecked-tokenizer-private.json").write_bytes(b"unchecked")

    with pytest.raises(ValueError, match="invalid NER model artifact") as exc_info:
        verify_model_artifact(model_path)

    assert "unchecked-tokenizer-private.json" not in str(exc_info.value)


def test_verify_model_artifact_rejects_unchecked_nested_file(
    tmp_path: Path,
) -> None:
    model_path, _ = write_verified_artifact(tmp_path)
    checkpoint = model_path / "checkpoint-1"
    checkpoint.mkdir()
    (checkpoint / "private.bin").write_bytes(b"unchecked")

    with pytest.raises(ValueError, match="invalid NER model artifact") as exc_info:
        verify_model_artifact(model_path)

    assert "private.bin" not in str(exc_info.value)


def test_verify_model_artifact_accepts_registered_nested_file(
    tmp_path: Path,
) -> None:
    model_path, _ = write_verified_artifact(tmp_path)
    checkpoint = model_path / "checkpoint-1"
    checkpoint.mkdir()
    nested = checkpoint / "trainer_state.json"
    nested.write_bytes(b"state")
    manifest_path = model_path / "training-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_checksums"]["checkpoint-1/trainer_state.json"] = (
        sha256_file(nested)
    )
    write_manifest(manifest_path, manifest)

    verified = verify_model_artifact(model_path)

    assert verified.artifact_sha256


def test_verify_model_artifact_rejects_nested_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path, _ = write_verified_artifact(tmp_path)
    checkpoint = model_path / "checkpoint-1"
    checkpoint.mkdir()
    target = tmp_path / "private.bin"
    target.write_bytes(b"private")
    nested = checkpoint / "optimizer.pt"
    try:
        nested.symlink_to(target)
    except OSError:
        original = Path.is_symlink
        nested.write_bytes(b"private")
        monkeypatch.setattr(
            Path,
            "is_symlink",
            lambda path: path == nested or original(path),
        )

    with pytest.raises(ValueError, match="invalid NER model artifact"):
        verify_model_artifact(model_path)


def test_verify_model_artifact_reports_unsupported_manifest_schema(
    tmp_path: Path,
) -> None:
    model_path, _ = write_verified_artifact(tmp_path)
    manifest_path = model_path / "training-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["manifest_schema_version"] = 1
    write_manifest(manifest_path, manifest)

    with pytest.raises(
        ValueError,
        match="unsupported NER training manifest schema",
    ):
        verify_model_artifact(model_path)


def test_verify_model_artifact_returns_digest_of_verified_artifacts(
    tmp_path: Path,
) -> None:
    model_path, checksums = write_verified_artifact(tmp_path)

    verified = verify_model_artifact(model_path)

    expected = manifest_module.sha256_bytes(
        json.dumps(checksums, sort_keys=True, separators=(",", ":")).encode()
    )
    assert verified.artifact_sha256 == expected


def test_verify_model_artifact_rejects_config_label_mismatch(
    tmp_path: Path,
) -> None:
    model_path, _ = write_verified_artifact(tmp_path)
    config_path = model_path / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["label2id"]["O"] = 99
    config_path.write_text(json.dumps(config), encoding="utf-8")
    manifest_path = model_path / "training-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_checksums"]["config.json"] = sha256_file(config_path)
    write_manifest(manifest_path, manifest)

    with pytest.raises(ValueError, match="invalid NER model artifact"):
        verify_model_artifact(model_path)


def test_build_manifest_records_reproducibility_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(manifest_module.platform, "python_version", lambda: "3.12.0")
    monkeypatch.setattr(
        manifest_module.importlib.metadata,
        "version",
        lambda name: {"accelerate": "1.3.0", "torch": "2.5.0", "transformers": "4.49.0"}[
            name
        ],
    )

    result = build_manifest(
        base_checkpoint="distilbert/distilbert-base-cased",
        base_revision="abc123",
        dataset_version="v1",
        generator_version="v1",
        label_mapping=LABEL_TO_ID,
        seed=7,
        threshold=None,
        metrics={"eval_loss": 0.25},
        hyperparameters={"epochs": 3, "learning_rate": 2e-5},
    )

    assert result == {
        "manifest_schema_version": 2,
        "artifact_name": "ai-guardrail-ner-en-v1",
        "base_checkpoint": "distilbert/distilbert-base-cased",
        "base_revision": "abc123",
        "dataset_version": "v1",
        "generator_version": "v1",
        "label_mapping": LABEL_TO_ID,
        "seed": 7,
        "threshold": None,
        "metrics": {"eval_loss": 0.25},
        "hyperparameters": {"epochs": 3, "learning_rate": 2e-5},
        "python_version": "3.12.0",
        "library_versions": {
            "accelerate": "1.3.0",
            "torch": "2.5.0",
            "transformers": "4.49.0",
        },
    }


@pytest.mark.parametrize("threshold", [-0.01, 1.01, math.nan])
def test_build_manifest_rejects_out_of_range_threshold(threshold: float) -> None:
    with pytest.raises(ValueError, match="threshold must be between 0 and 1"):
        build_manifest(
            base_checkpoint="checkpoint",
            base_revision="revision",
            dataset_version="v1",
            generator_version="v1",
            label_mapping=LABEL_TO_ID,
            seed=7,
            threshold=threshold,
            metrics={},
            hyperparameters={},
        )


def test_build_manifest_rejects_changed_label_mapping() -> None:
    changed_mapping = dict(LABEL_TO_ID)
    changed_mapping["O"] = 99

    with pytest.raises(ValueError, match="label mapping does not match"):
        build_manifest(
            base_checkpoint="distilbert/distilbert-base-cased",
            base_revision="revision",
            dataset_version="v1",
            generator_version="v1",
            label_mapping=changed_mapping,
            seed=7,
            threshold=None,
            metrics={},
            hyperparameters={},
        )


def test_build_manifest_rejects_boolean_label_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    changed_mapping = dict(LABEL_TO_ID)
    changed_mapping["B-PERSON"] = True
    monkeypatch.setattr(
        manifest_module.importlib.metadata,
        "version",
        lambda name: "test-version",
    )

    with pytest.raises(ValueError, match="label mapping does not match"):
        build_manifest(
            base_checkpoint="distilbert/distilbert-base-cased",
            base_revision="revision",
            dataset_version="v1",
            generator_version="v1",
            label_mapping=changed_mapping,
            seed=7,
            threshold=None,
            metrics={},
            hyperparameters={},
        )


def test_build_manifest_rejects_unexpected_generator_version() -> None:
    with pytest.raises(ValueError, match="generator version does not match"):
        build_manifest(
            base_checkpoint="distilbert/distilbert-base-cased",
            base_revision="revision",
            dataset_version="v1",
            generator_version="private-generator-v2",
            label_mapping=LABEL_TO_ID,
            seed=7,
            threshold=None,
            metrics={},
            hyperparameters={},
        )


def test_write_manifest_is_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "training-manifest.json"

    write_manifest(path, {"z": 1, "a": {"c": 3, "b": 2}})

    assert path.read_bytes() == b'{\n  "a": {\n    "b": 2,\n    "c": 3\n  },\n  "z": 1\n}\n'
