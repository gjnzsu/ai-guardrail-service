from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from ai_guardrail.detectors.regex import RegexDetector
from ai_guardrail.domain import EntityType
from ai_guardrail.io import read_jsonl

AUTHORITATIVE_REGEX_TYPES = frozenset(
    {EntityType.EMAIL, EntityType.API_KEY, EntityType.CUSTOMER_ID}
)


async def run_regex_smoke(
    *,
    fixture_path: Path,
    regex_config: Path,
    output_path: Path,
) -> dict[str, int]:
    examples = read_jsonl(fixture_path)
    detector = RegexDetector.from_yaml(regex_config)
    gold_spans: set[tuple[str, EntityType, int, int]] = set()
    predicted_spans: set[tuple[str, EntityType, int, int]] = set()

    for example in examples:
        gold_spans.update(
            (example.id, entity.type, entity.start, entity.end)
            for entity in example.entities
            if entity.type in AUTHORITATIVE_REGEX_TYPES
        )
        result = await detector.detect(example.text)
        predicted_spans.update(
            (example.id, candidate.type, candidate.start, candidate.end)
            for candidate in result.candidates
            if candidate.type in AUTHORITATIVE_REGEX_TYPES
        )

    summary = {
        "examples": len(examples),
        "gold_spans": len(gold_spans),
        "predicted_spans": len(predicted_spans),
        "true_positive": len(gold_spans & predicted_spans),
        "false_positive": len(predicted_spans - gold_spans),
        "false_negative": len(gold_spans - predicted_spans),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--regex-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = asyncio.run(
        run_regex_smoke(
            fixture_path=args.fixture,
            regex_config=args.regex_config,
            output_path=args.output,
        )
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
