from __future__ import annotations

import re
import time
from collections.abc import Mapping
from pathlib import Path

import yaml

from ai_guardrail.domain import (
    CandidateDetection,
    DetectionSource,
    DetectorOutput,
    EntityType,
)

REQUIRED_REGEX_TYPES = frozenset(
    {EntityType.EMAIL, EntityType.API_KEY, EntityType.CUSTOMER_ID}
)


class RegexDetector:
    def __init__(
        self,
        *,
        version: str,
        patterns: dict[EntityType, tuple[re.Pattern[str], ...]],
    ) -> None:
        self.version = version
        self.patterns = patterns

    @classmethod
    def from_yaml(cls, path: Path) -> RegexDetector:
        try:
            config = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ValueError("invalid regex configuration") from exc

        if not isinstance(config, Mapping):
            raise ValueError("invalid regex configuration")

        version = config.get("version")
        if not isinstance(version, str) or not version.strip():
            raise ValueError("missing regex version")

        raw_patterns_by_type = config.get("patterns")
        if not isinstance(raw_patterns_by_type, Mapping):
            raise ValueError("invalid regex patterns")

        required_type_values = {entity_type.value for entity_type in REQUIRED_REGEX_TYPES}
        for raw_type in raw_patterns_by_type:
            if not isinstance(raw_type, str) or raw_type not in required_type_values:
                raise ValueError("unknown entity type in regex configuration")
        if any(
            entity_type.value not in raw_patterns_by_type
            for entity_type in REQUIRED_REGEX_TYPES
        ):
            raise ValueError("missing required regex pattern types")

        compiled: dict[EntityType, tuple[re.Pattern[str], ...]] = {}
        for entity_type in REQUIRED_REGEX_TYPES:
            raw_patterns = raw_patterns_by_type[entity_type.value]
            if not isinstance(raw_patterns, list) or not raw_patterns:
                raise ValueError("invalid regex pattern list")
            if any(not isinstance(pattern, str) or not pattern.strip() for pattern in raw_patterns):
                raise ValueError("invalid regex pattern")
            try:
                compiled[entity_type] = tuple(re.compile(pattern) for pattern in raw_patterns)
            except re.error as exc:
                raise ValueError(f"invalid regex for {entity_type}") from exc
        return cls(version=version, patterns=compiled)

    async def detect(self, text: str, message_index: int = 0) -> DetectorOutput:
        started = time.perf_counter()
        candidates: list[CandidateDetection] = []
        seen: set[tuple[EntityType, int, int]] = set()
        for entity_type, patterns in self.patterns.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    key = (entity_type, match.start(), match.end())
                    if key in seen:
                        continue
                    seen.add(key)
                    candidates.append(
                        CandidateDetection(
                            message_index=message_index,
                            type=entity_type,
                            start=match.start(),
                            end=match.end(),
                            source=DetectionSource.REGEX,
                        )
                    )
        candidates.sort(key=lambda item: (item.start, item.end, item.type))
        return DetectorOutput(
            detector=DetectionSource.REGEX,
            model_version=self.version,
            status="success",
            candidates=candidates,
            latency_ms=(time.perf_counter() - started) * 1000,
        )
