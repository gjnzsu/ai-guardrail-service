from __future__ import annotations

import asyncio
import gc
import math
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ai_guardrail.domain import DetectionSource, DetectorOutput, EntityType
from ai_guardrail.ner.alignment import decode_bio_predictions
from ai_guardrail.ner.labels import ID_TO_LABEL, LABELS
from ai_guardrail.ner.manifest import (
    is_exact_label_mapping,
    verified_model_snapshot,
)


def _detach_model_storage(model: Any) -> Any:
    import torch

    if not isinstance(model, torch.nn.Module):
        return model
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.data = parameter.detach().clone(
                memory_format=torch.preserve_format
            )
        for buffer in model.buffers():
            buffer.data = buffer.detach().clone(
                memory_format=torch.preserve_format
            )
    gc.collect()
    return model


def _normalize_thresholds(
    threshold: float | Mapping[EntityType, float],
) -> dict[EntityType, float]:
    if isinstance(threshold, Mapping):
        if set(threshold) != set(EntityType):
            raise ValueError(
                "thresholds must cover every entity type"
            )
        thresholds = dict(threshold)
    else:
        thresholds = {
            entity_type: threshold
            for entity_type in EntityType
        }
    if any(
        not math.isfinite(value) or not 0.0 <= value <= 1.0
        for value in thresholds.values()
    ):
        raise ValueError("threshold must be between 0 and 1")
    return thresholds


class NerDetector:
    def __init__(
        self,
        *,
        tokenizer: Any,
        model: Any,
        model_version: str,
        threshold: float | Mapping[EntityType, float],
        artifact_sha256: str | None = None,
        manifest_sha256: str | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.model = model.eval()
        self.model_version = model_version
        self.threshold = threshold
        self.thresholds = _normalize_thresholds(threshold)
        self.artifact_sha256 = artifact_sha256
        self.manifest_sha256 = manifest_sha256

    @classmethod
    def load(
        cls,
        model_path: Path,
        threshold: float | Mapping[EntityType, float],
        *,
        expected_artifact_sha256: str | None = None,
        expected_manifest_sha256: str | None = None,
    ) -> NerDetector:
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        with verified_model_snapshot(
            model_path,
            expected_artifact_sha256=expected_artifact_sha256,
            expected_manifest_sha256=expected_manifest_sha256,
        ) as snapshot:
            tokenizer = AutoTokenizer.from_pretrained(
                snapshot.path,
                use_fast=True,
                local_files_only=True,
            )
            model = AutoModelForTokenClassification.from_pretrained(
                snapshot.path,
                local_files_only=True,
            ).to("cpu")
            if (
                not is_exact_label_mapping(getattr(model.config, "label2id", None))
                or getattr(model.config, "id2label", None) != ID_TO_LABEL
            ):
                raise ValueError("invalid NER model artifact")
            model = _detach_model_storage(model)
        return cls(
            tokenizer=tokenizer,
            model=model,
            model_version=snapshot.verified.manifest["artifact_name"],
            threshold=threshold,
            artifact_sha256=snapshot.verified.artifact_sha256,
            manifest_sha256=snapshot.verified.manifest_sha256,
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
            if candidate.confidence is not None
            and candidate.confidence
            >= self.thresholds[candidate.type]
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
