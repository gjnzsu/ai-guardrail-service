from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ai_guardrail.ner.manifest import ARTIFACT_NAME

MANIFEST_FILENAME = "training-manifest.json"
THRESHOLDS = [value / 100 for value in range(50, 100, 5)]


@dataclass(frozen=True)
class SelectedThreshold:
    value: float
    artifact_sha256: str
    manifest_sha256: str
    validation_sha256: str


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


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


def model_manifest_sha256(model_path: Path) -> str:
    error_message = "invalid NER model artifact"
    if (
        model_path.name != ARTIFACT_NAME
        or model_path.is_symlink()
        or not model_path.is_dir()
    ):
        raise ValueError(error_message)
    content = _read_regular_file(
        model_path / MANIFEST_FILENAME,
        error_message,
    )
    manifest = _decode_json_object(content, error_message)
    if manifest.get("artifact_name") != ARTIFACT_NAME:
        raise ValueError(error_message)
    return sha256_bytes(content)


def build_threshold_artifact(
    *,
    model_path: Path,
    validation_sha256: str,
    candidate_thresholds: list[float],
    selected_threshold: float,
) -> dict[str, Any]:
    return {
        "candidate_thresholds": candidate_thresholds,
        "model_version": ARTIFACT_NAME,
        "ner_manifest_sha256": model_manifest_sha256(model_path),
        "selected_threshold": selected_threshold,
        "validation_sha256": validation_sha256,
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
    manifest_sha256 = model_manifest_sha256(model_path)
    if payload["ner_manifest_sha256"] != manifest_sha256:
        raise ValueError(error_message)
    return SelectedThreshold(
        value=float(selected),
        artifact_sha256=sha256_bytes(content),
        manifest_sha256=manifest_sha256,
        validation_sha256=payload["validation_sha256"],
    )
