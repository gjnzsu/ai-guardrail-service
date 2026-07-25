from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ai_guardrail.ner.manifest import (
    ARTIFACT_NAME,
    sha256_bytes,
    verify_model_artifact,
)

THRESHOLDS = [value / 100 for value in range(50, 100, 5)]


@dataclass(frozen=True)
class SelectedThreshold:
    value: float
    artifact_sha256: str
    manifest_sha256: str
    model_artifact_sha256: str
    validation_sha256: str
    training_provenance: dict[str, Any]


def _read_regular_file(path: Path, error_message: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError(error_message)
    try:
        return path.read_bytes()
    except OSError:
        raise ValueError(error_message) from None


def _decode_json_object(content: bytes, error_message: str) -> dict[str, Any]:
    try:
        payload = json.loads(content)
    except (UnicodeError, json.JSONDecodeError):
        raise ValueError(error_message) from None
    if not isinstance(payload, dict):
        raise ValueError(error_message)
    return payload


def build_threshold_artifact(
    *,
    model_path: Path,
    validation_provenance: dict[str, Any],
    candidate_thresholds: list[float],
    selected_threshold: float,
    expected_artifact_sha256: str,
    expected_manifest_sha256: str,
) -> dict[str, Any]:
    verified = verify_model_artifact(model_path)
    if (
        verified.artifact_sha256 != expected_artifact_sha256
        or verified.manifest_sha256 != expected_manifest_sha256
    ):
        raise ValueError(
            "model artifact changed during threshold selection"
        )
    expected_validation = verified.manifest["datasets"]["validation"]
    if validation_provenance != expected_validation:
        raise ValueError("invalid threshold selection provenance")
    return {
        "candidate_thresholds": candidate_thresholds,
        "model_version": ARTIFACT_NAME,
        "ner_artifact_sha256": verified.artifact_sha256,
        "ner_manifest_sha256": verified.manifest_sha256,
        "selected_threshold": selected_threshold,
        "validation_sha256": validation_provenance["sha256"],
        "validation_provenance": validation_provenance,
    }


def _is_finite_threshold(value: object) -> bool:
    return (
        type(value) in (int, float)
        and math.isfinite(value)
        and 0.0 <= value <= 1.0
    )


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def load_selected_threshold(
    artifact_path: Path,
    model_path: Path,
) -> SelectedThreshold:
    error_message = "invalid selected threshold artifact"
    content = _read_regular_file(artifact_path, error_message)
    payload = _decode_json_object(content, error_message)
    candidates = payload.get("candidate_thresholds")
    selected = payload.get("selected_threshold")
    if (
        payload.get("model_version") != ARTIFACT_NAME
        or not isinstance(candidates, list)
        or any(not _is_finite_threshold(value) for value in candidates)
        or candidates != THRESHOLDS
        or not _is_finite_threshold(selected)
        or selected not in candidates
        or not _is_sha256(payload.get("validation_sha256"))
        or not _is_sha256(payload.get("ner_manifest_sha256"))
    ):
        raise ValueError(error_message)
    try:
        verified = verify_model_artifact(model_path)
    except ValueError:
        raise ValueError(error_message) from None
    validation_provenance = payload.get("validation_provenance")
    expected_validation = verified.manifest["datasets"]["validation"]
    if (
        payload["ner_manifest_sha256"] != verified.manifest_sha256
        or not _is_sha256(payload.get("ner_artifact_sha256"))
        or payload["ner_artifact_sha256"] != verified.artifact_sha256
        or validation_provenance != expected_validation
        or payload["validation_sha256"] != expected_validation["sha256"]
    ):
        raise ValueError(error_message)
    return SelectedThreshold(
        value=float(selected),
        artifact_sha256=sha256_bytes(content),
        manifest_sha256=verified.manifest_sha256,
        model_artifact_sha256=verified.artifact_sha256,
        validation_sha256=payload["validation_sha256"],
        training_provenance=verified.manifest["datasets"],
    )
