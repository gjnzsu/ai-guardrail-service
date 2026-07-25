from __future__ import annotations

import asyncio
import json
import time
from importlib import metadata
from pathlib import Path
from typing import Any

import httpx
import jsonschema

from ai_guardrail.domain import (
    CandidateDetection,
    DetectionSource,
    DetectorOutput,
    EntityType,
)

SYSTEM_PROMPT = """You identify sensitive entities in English text.
Allowed types: PERSON, ADDRESS, EMAIL, API_KEY, CUSTOMER_ID, INTERNAL_PROJECT.
Return JSON only. Each entity text must be an exact input substring.
If the same substring occurs more than once, include its 1-based occurrence.
Do not rewrite, explain, or infer text that is absent from the input."""

MAX_MODEL_CONTENT_BYTES = 16_384
SCHEMA_FILENAME = "qwen-entity-schema.json"
INSTALLED_SCHEMA_SUFFIX = f"share/ai-guardrail-service/{SCHEMA_FILENAME}"


def _default_schema_path() -> Path:
    source_path = (
        Path(__file__).resolve().parents[3] / "config" / SCHEMA_FILENAME
    )
    if source_path.is_file():
        return source_path

    try:
        distribution = metadata.distribution("ai-guardrail-service")
    except metadata.PackageNotFoundError:
        raise ValueError from None

    for package_path in distribution.files or ():
        normalized = str(package_path).replace("\\", "/")
        if normalized.endswith(INSTALLED_SCHEMA_SUFFIX):
            installed_path = Path(distribution.locate_file(package_path))
            if installed_path.is_file():
                return installed_path
    raise ValueError


def resolve_occurrence(
    text: str,
    value: str,
    occurrence: int | None,
) -> tuple[int, int] | None:
    if not value:
        return None
    if occurrence is not None and type(occurrence) is not int:
        return None

    starts: list[int] = []
    cursor = 0
    while True:
        index = text.find(value, cursor)
        if index < 0:
            break
        starts.append(index)
        cursor = index + 1
    if not starts:
        return None
    if occurrence is None:
        if len(starts) != 1:
            return None
        selected = starts[0]
    elif occurrence < 1 or occurrence > len(starts):
        return None
    else:
        selected = starts[occurrence - 1]
    return selected, selected + len(value)


class QwenDetector:
    def __init__(
        self,
        *,
        base_url: str,
        model_version: str,
        timeout_seconds: float,
        schema_path: Path | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_version = model_version
        self.timeout_seconds = timeout_seconds
        self.transport = transport
        try:
            selected_schema_path = (
                schema_path if schema_path is not None else _default_schema_path()
            )
            schema = json.loads(
                selected_schema_path.read_text(encoding="utf-8")
            )
            if not isinstance(schema, dict):
                raise ValueError
            jsonschema.Draft202012Validator.check_schema(schema)
        except (
            OSError,
            UnicodeError,
            json.JSONDecodeError,
            jsonschema.SchemaError,
            RecursionError,
            TypeError,
            ValueError,
        ):
            raise ValueError("invalid Qwen detector schema") from None
        self.schema: dict[str, Any] = schema

    async def detect(self, text: str, message_index: int = 0) -> DetectorOutput:
        started = time.perf_counter()
        try:
            async with asyncio.timeout(self.timeout_seconds):
                return await self._detect_within_deadline(
                    text,
                    message_index,
                    started,
                )
        except (TimeoutError, httpx.TimeoutException):
            return DetectorOutput(
                detector=DetectionSource.QWEN,
                model_version=self.model_version,
                status="timeout",
                latency_ms=(time.perf_counter() - started) * 1000,
                error_code="detector_timeout",
            )

    async def _detect_within_deadline(
        self,
        text: str,
        message_index: int,
        started: float,
    ) -> DetectorOutput:
        try:
            async with httpx.AsyncClient(
                transport=self.transport,
                timeout=self.timeout_seconds,
            ) as client:
                token_response = await client.post(
                    f"{self.base_url}/tokenize",
                    json={"content": text, "add_special": False},
                )
                token_response.raise_for_status()
                token_payload = token_response.json()
                if not isinstance(token_payload, dict):
                    raise ValueError("invalid token response")
                tokens = token_payload.get("tokens")
                if not isinstance(tokens, list) or any(type(token) is not int for token in tokens):
                    raise ValueError("invalid token response")
                if len(tokens) > 512:
                    return DetectorOutput(
                        detector=DetectionSource.QWEN,
                        model_version=self.model_version,
                        status="error",
                        latency_ms=(time.perf_counter() - started) * 1000,
                        error_code="input_too_long",
                    )
                response = await client.post(
                    f"{self.base_url}/v1/chat/completions",
                    json={
                        "model": self.model_version,
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": f"{text}\n/no_think"},
                        ],
                        "temperature": 0,
                        "max_tokens": 96,
                        "json_schema": self.schema,
                        "chat_template_kwargs": {"enable_thinking": False},
                        "reasoning_effort": "none",
                    },
                )
                response.raise_for_status()
                completion_payload = response.json()
                if not isinstance(completion_payload, dict):
                    raise ValueError("invalid completion response")
                choices = completion_payload.get("choices")
                if not isinstance(choices, list) or not choices:
                    raise ValueError("invalid completion response")
                choice = choices[0]
                if not isinstance(choice, dict):
                    raise ValueError("invalid completion response")
                message = choice.get("message")
                if not isinstance(message, dict):
                    raise ValueError("invalid completion response")
                content = message.get("content")
                if not isinstance(content, str):
                    raise ValueError("invalid completion response")
                if len(content.encode("utf-8")) > MAX_MODEL_CONTENT_BYTES:
                    raise ValueError("invalid completion response")
                payload: dict[str, Any] = json.loads(content)
                jsonschema.validate(payload, self.schema)
                for raw in payload["entities"]:
                    if "occurrence" in raw and type(raw["occurrence"]) is not int:
                        raise ValueError("invalid completion response")
        except httpx.TimeoutException:
            raise
        except (
            httpx.HTTPError,
            jsonschema.ValidationError,
            KeyError,
            TypeError,
            ValueError,
        ):
            return DetectorOutput(
                detector=DetectionSource.QWEN,
                model_version=self.model_version,
                status="error",
                latency_ms=(time.perf_counter() - started) * 1000,
                error_code="invalid_detector_response",
            )

        candidates: list[CandidateDetection] = []
        invalid_count = 0
        for raw in payload.get("entities", []):
            try:
                entity_type = EntityType(raw["type"])
                value = str(raw["text"])
                occurrence = raw.get("occurrence")
                span = resolve_occurrence(text, value, occurrence)
                if span is None:
                    invalid_count += 1
                    continue
                start, end = span
                candidates.append(
                    CandidateDetection(
                        message_index=message_index,
                        type=entity_type,
                        start=start,
                        end=end,
                        source=DetectionSource.QWEN,
                    )
                )
            except (KeyError, TypeError, ValueError):
                invalid_count += 1

        return DetectorOutput(
            detector=DetectionSource.QWEN,
            model_version=self.model_version,
            status="success",
            candidates=candidates,
            latency_ms=(time.perf_counter() - started) * 1000,
            invalid_candidate_count=invalid_count,
        )
