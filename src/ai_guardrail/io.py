import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from ai_guardrail.domain import LabeledExample


@dataclass(frozen=True)
class JsonlSnapshot:
    content: bytes
    records: tuple[LabeledExample, ...]
    sha256: str


def write_jsonl(path: Path, records: Iterable[LabeledExample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(record.model_dump_json())
            handle.write("\n")


def read_jsonl_snapshot(path: Path) -> JsonlSnapshot:
    content = path.read_bytes()
    records: list[LabeledExample] = []
    for line_number, line in enumerate(content.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            records.append(LabeledExample.model_validate_json(stripped))
        except ValueError as exc:
            raise ValueError(f"invalid JSONL record at line {line_number}") from exc
    return JsonlSnapshot(
        content=content,
        records=tuple(records),
        sha256=hashlib.sha256(content).hexdigest(),
    )


def read_jsonl(path: Path) -> list[LabeledExample]:
    return list(read_jsonl_snapshot(path).records)
