import json
from pathlib import Path

import pytest

from ai_guardrail.evaluation.smoke import run_regex_smoke


@pytest.mark.asyncio
async def test_regex_smoke_produces_aggregate_only_report(tmp_path: Path) -> None:
    output_path = tmp_path / "regex-smoke.json"

    summary = await run_regex_smoke(
        fixture_path=Path("datasets/challenge/en-v1.seed.jsonl"),
        regex_config=Path("config/regex-patterns.yaml"),
        output_path=output_path,
    )

    assert summary == {
        "examples": 4,
        "gold_spans": 2,
        "predicted_spans": 2,
        "true_positive": 2,
        "false_positive": 0,
        "false_negative": 0,
    }
    report_text = output_path.read_text(encoding="utf-8")
    assert json.loads(report_text) == summary
    assert "Jane Cooper" not in report_text
    assert "jane.cooper@example.test" not in report_text
    assert "sk-test-A1B2C3D4E5F6G7H8" not in report_text
