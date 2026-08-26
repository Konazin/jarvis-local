from jarvis_local.tools.base import RiskLevel, Tool
from jarvis_local.tools.executor import ToolExecutor
from jarvis_local.tools.registry import ToolRegistry


def tool(name, risk, callback):
    return Tool(name, "test tool", {"type": "object"}, risk, callback)


def executor_for(test_tool, approval_handler=None):
    registry = ToolRegistry()
    registry.register(test_tool)
    return ToolExecutor(registry, approval_handler)


def test_safe_executes_without_approval() -> None:
    calls, approvals = [], []
    executor = executor_for(
        tool("safe", RiskLevel.SAFE, lambda **kwargs: calls.append(kwargs) or {"ok": True}), approvals.append
    )
    assert executor.execute("safe", {"value": 1}) == {"ok": True}
    assert calls == [{"value": 1}]
    assert approvals == []


def test_confirm_approved_executes_exact_defensive_arguments() -> None:
    calls = []

    def approve(request):
        request.arguments["nested"]["value"] = "mutated"
        return True

    executor = executor_for(
        tool("confirm", RiskLevel.CONFIRM, lambda **kwargs: calls.append(kwargs) or {"ok": True}), approve
    )
    arguments = {"nested": {"value": "approved"}}
    assert executor.execute("confirm", arguments) == {"ok": True}
    assert calls == [{"nested": {"value": "approved"}}]
    assert arguments == {"nested": {"value": "approved"}}


def test_confirm_rejection_unavailable_and_handler_failure_do_not_execute() -> None:
    calls = []
    confirm_tool = tool("confirm", RiskLevel.CONFIRM, lambda **kwargs: calls.append(kwargs) or {"ok": True})
    assert executor_for(confirm_tool).execute("confirm", {}) == {
        "status": "rejected",
        "reason": "confirmation_unavailable",
    }
    assert executor_for(confirm_tool, lambda _request: False).execute("confirm", {}) == {
        "status": "rejected",
        "reason": "user_rejected",
    }

    def broken(_request):
        raise RuntimeError("approval failed")

    assert executor_for(confirm_tool, broken).execute("confirm", {}) == {
        "status": "rejected",
        "reason": "confirmation_failed",
    }
    assert calls == []


def test_each_confirm_invocation_requires_new_approval() -> None:
    calls, approvals = [], []

    def approve(_request):
        approvals.append(True)
        return True

    executor = executor_for(tool("confirm", RiskLevel.CONFIRM, lambda **kwargs: calls.append(kwargs) or {}), approve)
    executor.execute("confirm", {})
    executor.execute("confirm", {})
    assert len(approvals) == len(calls) == 2


def test_dangerous_is_always_blocked_without_approval() -> None:
    calls, approvals = [], []
    executor = executor_for(
        tool("danger", RiskLevel.DANGEROUS, lambda **kwargs: calls.append(kwargs) or {}), approvals.append
    )
    assert executor.execute("danger", {}) == {"status": "blocked", "reason": "dangerous_tool"}
    assert calls == approvals == []


def test_unknown_tool_failures_and_non_serializable_results_are_structured() -> None:
    assert ToolExecutor(ToolRegistry()).execute("missing", {}) == {"status": "error", "reason": "unknown_tool"}
    failing = executor_for(
        tool("failing", RiskLevel.SAFE, lambda **kwargs: (_ for _ in ()).throw(RuntimeError("boom")))
    )
    assert failing.execute("failing", {}) == {"status": "error", "error": "boom"}
    invalid = executor_for(tool("invalid", RiskLevel.SAFE, lambda **kwargs: {1, 2}))
    assert invalid.execute("invalid", {})["reason"] == "non_serializable_result"
