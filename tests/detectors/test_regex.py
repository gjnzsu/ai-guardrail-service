from pathlib import Path

import pytest

from ai_guardrail.detectors.regex import RegexDetector
from ai_guardrail.domain import EntityType


@pytest.mark.asyncio
async def test_regex_detects_authoritative_entity_spans(tmp_path: Path) -> None:
    config = tmp_path / "patterns.yaml"
    config.write_text(
        """
version: regex-v1
patterns:
  EMAIL:
    - '(?i)\\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\\.[A-Z]{2,}\\b'
  API_KEY:
    - '\\b(?:sk|pk|api)[-_][A-Za-z0-9_-]{16,}\\b'
  CUSTOMER_ID:
    - '\\bCUST-[0-9]{5,10}\\b'
""".strip(),
        encoding="utf-8",
    )
    detector = RegexDetector.from_yaml(config)
    text = "Email jane@example.test about CUST-93821 and rotate sk-test-A1B2C3D4E5F6G7H8."

    output = await detector.detect(text)

    assert output.status == "success"
    detected_spans = {
        (candidate.type, text[candidate.start : candidate.end]) for candidate in output.candidates
    }
    assert detected_spans == {
        (EntityType.EMAIL, "jane@example.test"),
        (EntityType.CUSTOMER_ID, "CUST-93821"),
        (EntityType.API_KEY, "sk-test-A1B2C3D4E5F6G7H8"),
    }
    assert all(candidate.confidence is None for candidate in output.candidates)


def test_invalid_regex_fails_at_startup(tmp_path: Path) -> None:
    config = tmp_path / "patterns.yaml"
    config.write_text("version: regex-v1\npatterns:\n  EMAIL:\n    - '['\n", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid regex"):
        RegexDetector.from_yaml(config)
