from dataclasses import asdict
from unittest.mock import Mock

import pytest

from ai_guardrail.tool_execution import ToolExecutor


def test_allowed_tool_runs_once_with_private_allow_record() -> None:
    tool = Mock(return_value="synthetic-result-secret")
    executor = ToolExecutor({"lookup": tool}, {"lookup"})

    assert executor.execute("lookup", "synthetic-argument-secret") == "synthetic-result-secret"

    tool.assert_called_once_with("synthetic-argument-secret")
    assert [asdict(record) for record in executor.audit] == [
        {"tool_id": "lookup", "decision": "allow", "reason": "allowlisted"}
    ]
    assert "synthetic-argument-secret" not in repr(executor.audit)
    assert "synthetic-result-secret" not in repr(executor.audit)


def test_two_step_flow_checks_each_tool_before_its_call() -> None:
    events = []

    def first(argument):
        events.append(("call", "first"))
        assert argument == "synthetic-input-secret"
        assert [record.tool_id for record in executor.audit] == ["first"]
        return "synthetic-intermediate-secret"

    def second(argument):
        events.append(("call", "second"))
        assert argument == "synthetic-intermediate-secret"
        assert [record.tool_id for record in executor.audit] == ["first", "second"]
        return "synthetic-output-secret"

    class ObservedAllowlist(set):
        def __contains__(self, tool_id):
            events.append(("check", tool_id))
            return super().__contains__(tool_id)

    executor = ToolExecutor(
        {"first": first, "second": second}, ObservedAllowlist({"first", "second"})
    )

    intermediate = executor.execute("first", "synthetic-input-secret")
    assert executor.execute("second", intermediate) == "synthetic-output-secret"

    assert events == [
        ("check", "first"),
        ("call", "first"),
        ("check", "second"),
        ("call", "second"),
    ]
    assert [asdict(record) for record in executor.audit] == [
        {"tool_id": "first", "decision": "allow", "reason": "allowlisted"},
        {"tool_id": "second", "decision": "allow", "reason": "allowlisted"},
    ]
    assert "secret" not in repr(executor.audit)


def test_registered_tool_omitted_from_allowlist_is_blocked_without_call() -> None:
    tool = Mock(return_value="synthetic-result-secret")
    executor = ToolExecutor({"restricted": tool}, set())

    with pytest.raises(PermissionError, match="tool execution denied") as error:
        executor.execute("restricted", "synthetic-argument-secret")

    tool.assert_not_called()
    assert [asdict(record) for record in executor.audit] == [
        {"tool_id": "restricted", "decision": "block", "reason": "not_allowlisted"}
    ]
    assert "secret" not in repr(executor.audit)
    assert "secret" not in str(error.value)
