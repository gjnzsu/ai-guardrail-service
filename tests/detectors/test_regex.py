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
    config.write_text(
        """
version: regex-v1
patterns:
  EMAIL:
    - '['
  API_KEY:
    - api-key
  CUSTOMER_ID:
    - customer-id
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="invalid regex"):
        RegexDetector.from_yaml(config)


@pytest.mark.parametrize(
    ("config_text", "message", "unsafe_value"),
    [
        ("- regex-v1\n", "invalid regex configuration", "regex-v1"),
        (
            "version: [private-value\n",
            "invalid regex configuration",
            "private-value",
        ),
        ("patterns: {}\n", "missing regex version", "patterns"),
        ("version: ''\npatterns: {}\n", "missing regex version", "patterns"),
        (
            "version: regex-v1\npatterns: scalar-patterns\n",
            "invalid regex patterns",
            "scalar-patterns",
        ),
        (
            "version: regex-v1\npatterns:\n  EMAIL:\n    - email\n  API_KEY:\n    - api-key\n",
            "missing required regex pattern types",
            "API_KEY",
        ),
        (
            "version: regex-v1\npatterns:\n  EMAIL: email\n  API_KEY:\n    - api-key\n"
            "  CUSTOMER_ID:\n    - customer-id\n",
            "invalid regex pattern list",
            "email",
        ),
        (
            "version: regex-v1\npatterns:\n  EMAIL: []\n  API_KEY:\n    - api-key\n"
            "  CUSTOMER_ID:\n    - customer-id\n",
            "invalid regex pattern list",
            "EMAIL",
        ),
        (
            "version: regex-v1\npatterns:\n  EMAIL:\n    - 42\n  API_KEY:\n    - api-key\n"
            "  CUSTOMER_ID:\n    - customer-id\n",
            "invalid regex pattern",
            "42",
        ),
        (
            "version: regex-v1\npatterns:\n  EMAIL:\n    - email\n  API_KEY:\n    - api-key\n"
            "  CUSTOMER_ID:\n    - customer-id\n  SENSITIVE_UNKNOWN:\n    - secret\n",
            "unknown entity type in regex configuration",
            "SENSITIVE_UNKNOWN",
        ),
    ],
)
def test_regex_rejects_malformed_schema_without_echoing_input(
    tmp_path: Path,
    config_text: str,
    message: str,
    unsafe_value: str,
) -> None:
    config = tmp_path / "patterns.yaml"
    config.write_text(config_text, encoding="utf-8")

    with pytest.raises(ValueError, match=message) as exc_info:
        RegexDetector.from_yaml(config)

    assert unsafe_value not in str(exc_info.value)


@pytest.mark.asyncio
async def test_regex_deduplicates_duplicate_configured_patterns(tmp_path: Path) -> None:
    config = tmp_path / "patterns.yaml"
    config.write_text(
        """
version: regex-v1
patterns:
  EMAIL:
    - '\\b[A-Za-z]+@[A-Za-z]+\\.test\\b'
    - '\\b[A-Za-z]+@[A-Za-z]+\\.test\\b'
  API_KEY:
    - '\\bapi-[A-Za-z0-9]+\\b'
  CUSTOMER_ID:
    - '\\bCUST-[0-9]+\\b'
""".strip(),
        encoding="utf-8",
    )
    text = "email user@example.test"

    output = await RegexDetector.from_yaml(config).detect(text)

    detected_spans = [
        (candidate.type, candidate.start, candidate.end) for candidate in output.candidates
    ]
    assert detected_spans == [(EntityType.EMAIL, 6, 23)]


@pytest.mark.asyncio
async def test_regex_orders_candidates_by_offset_then_type(tmp_path: Path) -> None:
    config = tmp_path / "patterns.yaml"
    config.write_text(
        """
version: regex-v1
patterns:
  CUSTOMER_ID:
    - '\\bCUST-[0-9]+\\b'
  API_KEY:
    - '\\bapi-[A-Za-z0-9]+\\b'
  EMAIL:
    - '\\b[A-Za-z]+@[A-Za-z]+\\.test\\b'
""".strip(),
        encoding="utf-8",
    )
    text = "api-token then user@example.test then CUST-93821"

    output = await RegexDetector.from_yaml(config).detect(text)

    assert [candidate.type for candidate in output.candidates] == [
        EntityType.API_KEY,
        EntityType.EMAIL,
        EntityType.CUSTOMER_ID,
    ]


@pytest.mark.asyncio
async def test_regex_uses_exact_python_offsets_after_unicode(tmp_path: Path) -> None:
    config = tmp_path / "patterns.yaml"
    config.write_text(
        """
version: regex-v1
patterns:
  EMAIL:
    - '\\b[A-Za-z]+@[A-Za-z]+\\.test\\b'
  API_KEY:
    - '\\bapi-[A-Za-z0-9]+\\b'
  CUSTOMER_ID:
    - '\\bCUST-[0-9]+\\b'
""".strip(),
        encoding="utf-8",
    )
    text = "前缀 😀 user@example.test"

    output = await RegexDetector.from_yaml(config).detect(text)

    candidate = output.candidates[0]
    assert (candidate.start, candidate.end) == (len("前缀 😀 "), len(text))
    assert text[candidate.start : candidate.end] == "user@example.test"
