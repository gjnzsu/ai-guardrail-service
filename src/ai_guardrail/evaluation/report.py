import json
from pathlib import Path
from typing import Any


def write_json_report(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_markdown_report(
    path: Path,
    result: dict[str, Any],
) -> None:
    lines = [
        "# Offline Detector Evaluation",
        "",
        f"Examples: {result['example_count']}",
        "",
        "| Detector | Strict F1 | Partial F1 | P95 ms | Peak MiB | "
        "Consistency | Parse rate | Invalid | Partial failures | Errors | "
        "Timeouts |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
        "---: | ---: | ---: |",
    ]
    entity_sections: list[str] = []
    for name, metrics in result["detectors"].items():
        strict = metrics["strict"]
        partial = metrics["partial"]
        latency = metrics["latency_ms"]
        parse_rate = metrics["json_parse_rate"]
        parse_text = (
            ""
            if parse_rate is None
            else f"{parse_rate:.4f}"
        )
        lines.append(
            f"| {name} | {strict['f1']:.4f} | "
            f"{partial['f1']:.4f} | {latency['p95']:.2f} | "
            f"{metrics['peak_rss_mib']:.2f} | "
            f"{metrics['consistency_rate']:.4f} | "
            f"{parse_text} | {metrics['invalid_candidate_count']} | "
            f"{metrics.get('partial_failure_count', 0)} | "
            f"{metrics['error_count']} | {metrics['timeout_count']} |"
        )
        entity_sections.extend(
            [
                "",
                f"## {name} per entity",
                "",
                "| Entity | Precision | Recall | F1 |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for entity_type, entity_metrics in metrics["per_entity"].items():
            entity_sections.append(
                f"| {entity_type} | "
                f"{entity_metrics['precision']:.4f} | "
                f"{entity_metrics['recall']:.4f} | "
                f"{entity_metrics['f1']:.4f} |"
            )
    lines.extend(entity_sections)
    environment = result.get("environment")
    if isinstance(environment, dict):
        threshold = environment.get("ner_threshold")
        artifact_sha256 = environment.get(
            "ner_threshold_artifact_sha256"
        )
        if threshold is not None and artifact_sha256 is not None:
            lines.extend(
                [
                    "",
                    "## Reproducibility",
                    "",
                    f"Selected NER threshold: `{threshold}`",
                    "",
                    f"Threshold artifact SHA-256: `{artifact_sha256}`",
                ]
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
