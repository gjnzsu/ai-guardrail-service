from __future__ import annotations

import argparse
import asyncio
import hashlib
import platform
from pathlib import Path

from ai_guardrail.detectors.ner import NerDetector
from ai_guardrail.detectors.qwen import QwenDetector
from ai_guardrail.detectors.regex import RegexDetector
from ai_guardrail.domain import LabeledExample
from ai_guardrail.evaluation.report import (
    write_json_report,
    write_markdown_report,
)
from ai_guardrail.evaluation.runner import (
    AuthoritativeUnionDetector,
    BenchmarkRunner,
)
from ai_guardrail.evaluation.threshold_artifact import (
    load_selected_threshold,
)
from ai_guardrail.io import read_jsonl
from ai_guardrail.ner.manifest import sha256_bytes


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--challenge",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--regex-config",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--ner-model",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--ner-threshold-artifact",
        type=Path,
        required=True,
    )
    parser.add_argument("--qwen-url", required=True)
    parser.add_argument(
        "--qwen-model",
        default="qwen3-0.6b-q4_k_m",
    )
    parser.add_argument(
        "--qwen-timeout",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--qwen-pid",
        type=int,
        required=True,
    )
    parser.add_argument("--qwen-sha256", required=True)
    parser.add_argument("--llama-version", required=True)
    parser.add_argument(
        "--guardrail-cpu-limit",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--guardrail-memory-limit-mib",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--qwen-cpu-limit",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--qwen-memory-limit-mib",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    return parser.parse_args()


def _require_challenge_examples(
    examples: list[LabeledExample],
) -> None:
    if not examples or any(
        example.split != "challenge"
        for example in examples
    ) or len({example.id for example in examples}) != len(examples):
        raise ValueError(
            "benchmark requires a non-empty challenge split"
        )


def _require_challenge_isolation(
    examples: list[LabeledExample],
    training_provenance: dict[str, object],
) -> None:
    prior_ids: set[str] = set()
    prior_families: set[str] = set()
    prior_content: set[str] = set()
    for split in ("train", "validation"):
        provenance = training_provenance[split]
        if not isinstance(provenance, dict):
            raise ValueError("invalid challenge provenance")
        prior_ids.update(provenance["record_id_hashes"])
        prior_families.update(provenance["template_families"])
        prior_content.update(provenance["content_hashes"])
    challenge_ids = {
        sha256_bytes(example.id.encode("utf-8"))
        for example in examples
    }
    challenge_families = {
        example.template_family
        for example in examples
    }
    challenge_content = {
        sha256_bytes(example.text.encode("utf-8"))
        for example in examples
    }
    if (
        not challenge_ids.isdisjoint(prior_ids)
        or not challenge_families.isdisjoint(prior_families)
        or not challenge_content.isdisjoint(prior_content)
    ):
        raise ValueError("invalid challenge provenance")


async def run() -> None:
    args = parse_args()
    examples = read_jsonl(args.challenge)
    _require_challenge_examples(examples)
    threshold = load_selected_threshold(
        args.ner_threshold_artifact,
        args.ner_model,
    )
    _require_challenge_isolation(
        examples,
        threshold.training_provenance,
    )

    regex = RegexDetector.from_yaml(args.regex_config)
    ner = NerDetector.load(
        args.ner_model,
        threshold.value,
    )
    qwen = QwenDetector(
        base_url=args.qwen_url,
        model_version=args.qwen_model,
        timeout_seconds=args.qwen_timeout,
    )
    runner = BenchmarkRunner(
        {
            "regex": regex,
            "ner": ner,
            "qwen": qwen,
            "regex+ner": AuthoritativeUnionDetector(
                regex=regex,
                ner=ner,
            ),
        },
        repetitions=args.repetitions,
        resource_pids={"qwen": args.qwen_pid},
    )
    result = await runner.run(examples)
    result["environment"] = {
        "host_processor": platform.processor(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "challenge_sha256": sha256(args.challenge),
        "ner_model_path": args.ner_model.name,
        "ner_manifest_sha256": threshold.manifest_sha256,
        "ner_artifact_sha256": threshold.model_artifact_sha256,
        "ner_threshold": threshold.value,
        "ner_threshold_artifact_sha256": (
            threshold.artifact_sha256
        ),
        "ner_threshold_validation_sha256": (
            threshold.validation_sha256
        ),
        "qwen_model": args.qwen_model,
        "qwen_sha256": args.qwen_sha256,
        "llama_version": args.llama_version,
        "guardrail_cpu_limit": args.guardrail_cpu_limit,
        "guardrail_memory_limit_mib": (
            args.guardrail_memory_limit_mib
        ),
        "qwen_cpu_limit": args.qwen_cpu_limit,
        "qwen_memory_limit_mib": args.qwen_memory_limit_mib,
        "qwen_context_tokens": 1024,
        "qwen_input_limit_tokens": 512,
        "qwen_output_tokens": 96,
        "repetitions": args.repetitions,
    }
    write_json_report(
        args.output_dir / "report.json",
        result,
    )
    write_markdown_report(
        args.output_dir / "report.md",
        result,
    )


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
