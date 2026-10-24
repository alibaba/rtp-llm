"""合并编排保持请求所有权、独立超时与请求窗口判定。"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from cases.master_ha_failover import actions as ha
from cases.request_completion.actions import execute, validate
from document_io import load_document
from program_fixtures import compile_scenarios
from runtime.deadline import Deadline, StageTimeout
from scenario.catalog import handlers
from scenario.context import RuntimeContext
from scenario.program_compiler import configure_program
from scenario.runtime import execute_instance
from scenario.stage_execution import _validate_output
from test_scenario_runtime import Backend, Clock

ROOT = Path(__file__).resolve().parents[1]


def completion_parameters():
    data = load_document(ROOT / "config/scenarios/request_completion.yaml")["parameters"]
    request = {k: v for k, v in data["traffic"].items() if k != "kind"}
    return dict(request, checks=data["checks"], operation_timeout_s=4)


@pytest.mark.parametrize(
    "failure,expected",
    [
        (None, "PASS"),
        ("fail_submit", "ERROR"),
        ("timeout_wait", "TIMEOUT"),
        ("completed", "FAIL"),
        ("fail_cleanup", "ERROR"),
    ],
)
def test_completion_runs_real_executor_and_always_reclaims_owned_requests(tmp_path, failure, expected):
    config = load_document(ROOT / "config/scenarios/request_completion.yaml")
    config["execution"]["stage_timeout_s"] = 3
    config["profiles"] = ["single-nonbatch"]
    document = configure_program(config, "completion.yaml")
    catalog = handlers("request_completion")
    (plan,) = compile_scenarios([("completion.yaml", document)], handlers=catalog)
    clock = Clock()
    backend = Backend(clock)
    if failure:
        setattr(backend, failure, False if failure == "completed" else True)
    result = execute_instance(plan, backend, handlers=catalog, artifact_dir=tmp_path, clock=clock, sleeper=clock.sleep)
    assert result["status"] == expected
    assert backend.calls.count("request_cleanup") >= 1
    assert backend.calls.count("teardown") == 1
    if failure is None or failure == "completed":
        terminal = next(stage for stage in result["stages"] if stage["id"] == "terminal")
        assert {c["id"] for c in terminal["checks"]} == {"completed", "no_errors"}
    assert [stage["id"] for stage in result["stages"]] == ["setup", "terminal", "cleanup"]


def test_issue_and_wait_keep_independent_operation_budgets(tmp_path):
    clock = Clock()
    ctx = RuntimeContext({}, None, tmp_path, clock, clock.sleep)

    def issue(ctx, params, deadline):
        deadline.sleep(3)
        return ctx.register_resource("requests", [1])

    def wait(ctx, requests, deadline):
        deadline.sleep(3)
        return dict(completed=True, error_count=0)

    ctx.backend = SimpleNamespace(start_requests=issue, wait_requests=wait)
    result = execute(ctx, completion_parameters(), Deadline(8, clock, clock.sleep))
    assert [c.status for c in result.checks] == ["PASS", "PASS"]
    assert clock() == pytest.approx(6)
    clock.now = 0
    with pytest.raises(StageTimeout):
        execute(ctx, completion_parameters(), Deadline(5, clock, clock.sleep))


@pytest.mark.parametrize(
    "change", [dict(output="unknown"), dict(unit="seconds"), dict(expected=1), dict(op="ge"), dict(windows=["other"])]
)
def test_completion_reuses_strict_criterion_contract(change):
    params = completion_parameters()
    params["checks"]["completed"].update(change)
    with pytest.raises(ValueError):
        validate(params, SimpleNamespace(path="completion"))


def test_ha_batch_keeps_issue_time_windows_sample_errors_and_all_named_checks(tmp_path):
    ctx = RuntimeContext(dict(profile="batch-window"), None, tmp_path, lambda: 0, lambda _: None)
    rows = ctx.register_resource(
        "ha_rows",
        [
            dict(status=status, send_start_epoch_ms=stamp)
            for stamp, status in [(1000, "ok"), (2000, "schedule_error"), (3000, "ok")]
        ],
    )
    criterion = dict(metric="ha_gate/success_rate", op="eq", expected=1, min_samples=1)
    params = dict(
        rows=rows,
        windows=dict(first={"from": 1, "until": 2}, middle={"from": 2, "until": 3}, empty={"from": 4, "until": 5}),
        operation_timeout_s=3,
        checks=dict(
            good=dict(criterion, rows=rows, window="first"),
            bad=dict(criterion, rows=rows, window="middle"),
            advisory=dict(criterion, rows=rows, window="middle", warning_profiles=["batch-window"]),
            insufficient=dict(criterion, rows=rows, window="empty"),
        ),
    )
    result = ha._analyze(ctx, params, Deadline(30, lambda: 0))
    _validate_output(ctx, result, {}, params["checks"])
    assert [(c.id, c.status) for c in result.checks] == [
        ("good", "PASS"),
        ("bad", "FAIL"),
        ("advisory", "WARNING"),
        ("insufficient", "ERROR"),
    ]
    assert result.checks[0].evidence["observed_request_bounds"] == [1, 1]
    assert result.checks[1].evidence["labels"]["window"] == "middle"
    assert result.checks[-1].actual is None
    assert result.checks[-1].evidence["validity"] == "INVALID"


def test_ha_period_cannot_spend_one_operations_budget_on_another(tmp_path):
    clock = Clock()
    ctx = RuntimeContext({}, None, tmp_path, clock, clock.sleep, wall_clock=clock)
    with pytest.raises(StageTimeout):
        ha._period(
            ctx, dict(name="b", settle_s=4, duration_s=1, operation_timeout_s=3), Deadline(6, clock, clock.sleep)
        )
    assert not ctx.report_events
    clock.now = 0
    result = ha._period(
        ctx, dict(name="baseline", settle_s=0, duration_s=2, operation_timeout_s=3), Deadline(6, clock, clock.sleep)
    )
    assert [e["id"] for e in ctx.report_events] == ["baseline_end"]
    assert result.output["end"] == pytest.approx(2)


def test_child_deadline_inherits_cancellation_and_parent_limit():
    parent = Deadline(5, lambda: 1)
    child = parent.child(100)
    assert child.expires_at == 5
    assert child.cancelled is parent.cancelled
    parent.cancelled.set()
    with pytest.raises(StageTimeout):
        child.check()


def test_operation_rejects_a_backend_that_returns_after_its_own_budget(tmp_path):
    clock = Clock()
    ctx = RuntimeContext({}, None, tmp_path, clock, clock.sleep)

    def issue(ctx, params, deadline):
        clock.sleep(5)  # backend ignores its supplied deadline
        return ctx.register_resource("requests", [])

    ctx.backend = SimpleNamespace(
        start_requests=issue, wait_requests=lambda *_: pytest.fail("expired issue must not reach wait")
    )
    with pytest.raises(StageTimeout):
        execute(ctx, completion_parameters(), Deadline(8, clock, clock.sleep))
