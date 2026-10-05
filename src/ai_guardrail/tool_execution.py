"""Standalone local mock-tool experiment, separate from detector semantics."""

from collections.abc import Callable, Mapping, Set
from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class AuditRecord:
    tool_id: str
    decision: Literal["allow", "block"]
    reason: Literal["allowlisted", "not_allowlisted", "not_registered"]


class ToolExecutor:
    """Run injected synchronous mock tools through a local default-deny gate."""

    def __init__(self, registry: Mapping[str, Callable[..., Any]], allowlist: Set[str]) -> None:
        self._registry = dict(registry)
        self._allowlist = allowlist
        self._audit: list[AuditRecord] = []

    @property
    def audit(self) -> tuple[AuditRecord, ...]:
        return tuple(self._audit)

    def execute(self, tool_id: str, *args: Any, **kwargs: Any) -> Any:
        if tool_id not in self._allowlist:
            self._audit.append(AuditRecord(tool_id, "block", "not_allowlisted"))
            raise PermissionError("tool execution denied")
        if tool_id not in self._registry:
            self._audit.append(AuditRecord(tool_id, "block", "not_registered"))
            raise PermissionError("tool execution denied")
        tool = self._registry[tool_id]
        self._audit.append(AuditRecord(tool_id, "allow", "allowlisted"))
        return tool(*args, **kwargs)
