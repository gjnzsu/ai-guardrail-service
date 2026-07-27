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
        "examples": 9,
        "gold_spans": 6,
        "predicted_spans": 5,
        "true_positive": 5,
        "false_positive": 0,
        "false_negative": 1,
    }
    report_text = output_path.read_text(encoding="utf-8")
    assert json.loads(report_text) == summary
    assert "Jane Cooper" not in report_text
    assert "jane.cooper@example.test" not in report_text
    assert "sk-test-A1B2C3D4E5F6G7H8" not in report_text
    assert "tommy" not in report_text
    assert "123456" not in report_text
    assert "Project Apex" not in report_text
    assert "30156758@example.test" not in report_text
    assert "project.owner@example.test" not in report_text
