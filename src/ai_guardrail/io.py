from collections.abc import Iterable
from pathlib import Path

from ai_guardrail.domain import LabeledExample


def write_jsonl(path: Path, records: Iterable[LabeledExample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(record.model_dump_json())
            handle.write("\n")


def read_jsonl(path: Path) -> list[LabeledExample]:
    records: list[LabeledExample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(LabeledExample.model_validate_json(stripped))
            except ValueError as exc:
                raise ValueError(f"invalid JSONL record at line {line_number}") from exc
    return records
