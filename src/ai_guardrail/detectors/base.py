from typing import Protocol

from ai_guardrail.domain import DetectorOutput


class Detector(Protocol):
    async def detect(self, text: str, message_index: int = 0) -> DetectorOutput:
        """Return candidates without logging or persisting input text."""
        ...
