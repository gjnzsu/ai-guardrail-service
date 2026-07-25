import json
from pathlib import Path

from ai_guardrail.evaluation.report import write_json_report, write_markdown_report


def benchmark_result() -> dict[str, object]:
    return {
        "example_count": 1,
        "detectors": {
            "ner": {
                "strict": {"f1": 1.0},
                "partial": {"f1": 1.0},
                "per_entity": {
                    "PERSON": {"precision": 1.0, "recall": 1.0, "f1": 1.0}
                },
                "latency_ms": {"p95": 12.0},
                "peak_rss_mib": 100.0,
                "consistency_rate": 1.0,
                "json_parse_rate": None,
                "invalid_candidate_count": 0,
                "error_count": 0,
                "timeout_count": 0,
            }
        },
    }


def test_report_writers_create_aggregate_only_outputs(tmp_path: Path) -> None:
    json_path = tmp_path / "report.json"
    markdown_path = tmp_path / "report.md"

    write_json_report(json_path, benchmark_result())
    write_markdown_report(markdown_path, benchmark_result())

    assert json.loads(json_path.read_text(encoding="utf-8")) == benchmark_result()
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "# Offline Detector Evaluation" in markdown
    assert "| ner | 1.0000 | 1.0000 |" in markdown
