from __future__ import annotations

import asyncio
import json
import math
import time
from pathlib import Path
from typing import Any

from ai_guardrail.domain import DetectionSource, DetectorOutput
from ai_guardrail.ner.alignment import decode_bio_predictions
from ai_guardrail.ner.labels import LABELS
from ai_guardrail.ner.manifest import (
    ARTIFACT_NAME,
    BASE_CHECKPOINT,
    GENERATOR_VERSION,
    is_exact_label_mapping,
)

_MANIFEST_NAME = "training-manifest.json"


def _validate_artifact_identity(model_path: Path) -> None:
    if (
        model_path.name != ARTIFACT_NAME
        or model_path.is_symlink()
        or not model_path.is_dir()
    ):
        raise ValueError("invalid NER model artifact")
    manifest_path = model_path / _MANIFEST_NAME
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("invalid NER model artifact")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise ValueError("invalid NER model artifact") from None
    if (
        not isinstance(manifest, dict)
        or manifest.get("artifact_name") != ARTIFACT_NAME
        or manifest.get("base_checkpoint") != BASE_CHECKPOINT
        or manifest.get("generator_version") != GENERATOR_VERSION
        or not is_exact_label_mapping(manifest.get("label_mapping"))
    ):
        raise ValueError("invalid NER model artifact")


class NerDetector:
    def __init__(
        self,
        *,
        tokenizer: Any,
        model: Any,
        model_version: str,
        threshold: float,
    ) -> None:
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        self.tokenizer = tokenizer
        self.model = model.eval()
        self.model_version = model_version
        self.threshold = threshold

    @classmethod
    def load(cls, model_path: Path, threshold: float) -> NerDetector:
        _validate_artifact_identity(model_path)

        from transformers import AutoModelForTokenClassification, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=True,
            local_files_only=True,
        )
        model = AutoModelForTokenClassification.from_pretrained(
            model_path,
            local_files_only=True,
        ).to("cpu")
        return cls(
            tokenizer=tokenizer,
            model=model,
            model_version=ARTIFACT_NAME,
            threshold=threshold,
        )

    def _detect_sync(self, text: str, message_index: int) -> DetectorOutput:
        import torch

        started = time.perf_counter()
        encoded = self.tokenizer(
            text,
            return_offsets_mapping=True,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        encoded = encoded.to("cpu")
        offsets_tensor = encoded.pop("offset_mapping")
        if (
            offsets_tensor.ndim != 3
            or offsets_tensor.shape[0] != 1
            or offsets_tensor.shape[2] != 2
        ):
            raise ValueError("unexpected tokenizer offset shape")
        with torch.inference_mode():
            logits_batch = self.model(**encoded).logits
            if (
                logits_batch.ndim != 3
                or logits_batch.shape[0] != 1
                or logits_batch.shape[1] != offsets_tensor.shape[1]
                or logits_batch.shape[2] != len(LABELS)
            ):
                raise ValueError("unexpected model logits shape")
            logits = logits_batch[0]
            probabilities = torch.softmax(logits, dim=-1)
            best_probabilities, label_ids = probabilities.max(dim=-1)
        offsets = [tuple(pair) for pair in offsets_tensor[0].tolist()]
        candidates = decode_bio_predictions(
            text=text,
            offsets=offsets,
            label_ids=label_ids.tolist(),
            probabilities=best_probabilities.tolist(),
            message_index=message_index,
        )
        candidates = [
            candidate
            for candidate in candidates
            if candidate.confidence is not None and candidate.confidence >= self.threshold
        ]
        return DetectorOutput(
            detector=DetectionSource.NER,
            model_version=self.model_version,
            status="success",
            candidates=candidates,
            latency_ms=(time.perf_counter() - started) * 1000,
        )

    async def detect(self, text: str, message_index: int = 0) -> DetectorOutput:
        started = time.perf_counter()
        try:
            return await asyncio.to_thread(self._detect_sync, text, message_index)
        except Exception:
            return DetectorOutput(
                detector=DetectionSource.NER,
                model_version=self.model_version,
                status="error",
                candidates=[],
                latency_ms=(time.perf_counter() - started) * 1000,
                error_code="ner_inference_failed",
            )
