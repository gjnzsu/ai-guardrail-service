import math
from pathlib import Path

import pytest

from ai_guardrail.ner import manifest as manifest_module
from ai_guardrail.ner.labels import LABEL_TO_ID
from ai_guardrail.ner.manifest import build_manifest, sha256_file, write_manifest


def test_sha256_file_is_reproducible(tmp_path: Path) -> None:
    artifact = tmp_path / "weights.bin"
    artifact.write_bytes(b"guardrail-model")

    assert sha256_file(artifact) == (
        "64c172d264fcea853f3556031a49bb15710ccedb51712322ca7bd6d997c4cb1a"
    )


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
