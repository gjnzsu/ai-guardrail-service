from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
from pathlib import Path

from ai_guardrail.detectors.ner import NerDetector
from ai_guardrail.domain import LabeledExample
from ai_guardrail.evaluation.metrics import select_threshold
from ai_guardrail.io import read_jsonl

THRESHOLDS = [value / 100 for value in range(50, 100, 5)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--validation",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--ner-model",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    return parser.parse_args()


def _require_validation_examples(
    examples: list[LabeledExample],
) -> None:
    if not examples or any(
        example.split != "validation"
        for example in examples
    ):
        raise ValueError(
            "threshold selection requires a non-empty validation split"
        )


async def run() -> None:
    args = parse_args()
    examples = read_jsonl(args.validation)
    _require_validation_examples(examples)
    detector = NerDetector.load(
        args.ner_model,
        threshold=0.0,
    )
    predictions = []
    for example in examples:
        output = await detector.detect(example.text)
        if output.status != "success":
            raise RuntimeError(
                "NER threshold selection failed safely"
            )
        predictions.append(output.candidates)
    threshold = select_threshold(
        [example.entities for example in examples],
        predictions,
        THRESHOLDS,
    )
    if not math.isfinite(threshold):
        raise RuntimeError(
            "NER threshold selection produced an invalid threshold"
        )
    payload = {
        "model_version": args.ner_model.name,
        "validation_sha256": hashlib.sha256(
            args.validation.read_bytes()
        ).hexdigest(),
        "candidate_thresholds": THRESHOLDS,
        "selected_threshold": threshold,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
