from __future__ import annotations

import re
import time
from pathlib import Path

import yaml

from ai_guardrail.domain import (
    CandidateDetection,
    DetectionSource,
    DetectorOutput,
    EntityType,
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
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        compiled: dict[EntityType, tuple[re.Pattern[str], ...]] = {}
        for raw_type, raw_patterns in config["patterns"].items():
            entity_type = EntityType(raw_type)
            try:
                compiled[entity_type] = tuple(re.compile(pattern) for pattern in raw_patterns)
            except re.error as exc:
                raise ValueError(f"invalid regex for {entity_type}") from exc
        return cls(version=str(config["version"]), patterns=compiled)

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
