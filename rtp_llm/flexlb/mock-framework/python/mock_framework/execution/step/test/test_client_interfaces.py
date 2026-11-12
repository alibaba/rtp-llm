import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
from mock_framework.config import ConfigError
from mock_framework.execution import ExecutionError, metrics
from mock_framework.execution.roles import client as client_runtime
from mock_framework.execution.roles.client import ClientRole
from mock_framework.execution.step.client_query_qps import client_query_qps
from mock_framework.execution.step.step_registry import StepRegistry
from mock_framework.execution.test.helpers import FRAMEWORK


def context():
    events = []
    nodes = [
        {"id": "master-0", "role": "master", "rpc": "master:1"},
        {"id": "client-0", "role": "client", "rpc": "c:1", "control": "http://c:1"},
        {"id": "client-1", "role": "client", "rpc": "c:2", "control": "http://c:2"},
    ]
    topology = SimpleNamespace(
        active=lambda role=None: [
            item for item in nodes if role is None or item["role"] == role
        ]
    )
    records = SimpleNamespace(emit=lambda **v: events.append(v))
    clients = ClientRole(
        role_config=None,
        launch_config=None,
        supervisor=None,
        topology=topology,
        records=records,
        monitoring=None,
    )
    return SimpleNamespace(
        runtime=SimpleNamespace(
            topology=topology,
            clients=clients,
            case=SimpleNamespace(directory=FRAMEWORK),
        ),
        nodes=nodes,
        query_time="2026-10-28T00:00:00Z",
        timeout=20,
        records=records,
        events=events,
    )


def args():
    return {
        "masters": ["master:1"],
        "requests_file": str(FRAMEWORK / "data/traffic/basic.mckidx"),
        "qps": 10,
        "total_concurrency": 2,
    }


def test_start_requires_collecting_previous_playback(monkeypatch):
    calls = []

    async def invoke(url, operation, args, timeout):
        calls.append(url)
        return {"accepted": True}

    monkeypatch.setattr(client_runtime, "invoke", invoke)
    ctx = context()
    start = StepRegistry.standard().get("client/start/playback")
    assert asyncio.run(start.invoke(ctx, args()))
    with pytest.raises(ExecutionError, match="previous"):
        asyncio.run(start.invoke(ctx, args()))
    assert len(calls) == 1


def test_unknown_start_response_is_still_cleaned_up(monkeypatch):
    async def invoke(*a):
        raise OSError("connection lost")

    monkeypatch.setattr(client_runtime, "invoke", invoke)
    ctx = context()
    with pytest.raises(OSError):
        asyncio.run(
            StepRegistry.standard().get("client/start/playback").invoke(ctx, args())
        )
    assert ctx.runtime.clients.pending


def state():
    return dict(
        done=True,
        stopped=False,
        produced=1,
        issued=1,
        finished=1,
        unsent=0,
        queued=0,
        inflight=0,
        outcomes={"success": 1, "error": 0, "timeout": 0, "cancelled": 0},
        errors=[],
    )


def test_stop_collects_all_original_clients_without_task_id(monkeypatch, tmp_path):
    calls = []

    async def invoke(url, operation, args, timeout):
        calls.append((url, operation, args, timeout))
        return {"clients": {"http://c:1": state(), "http://c:2": state()}, "errors": []}

    monkeypatch.setattr(client_runtime, "invoke", invoke)
    ctx = context()
    ctx.runtime.clients.pending = True
    ctx.records.directory = tmp_path
    result = asyncio.run(
        StepRegistry.standard().get("client/stop/playback").invoke(ctx, {})
    )
    assert set(result) == {"c:1", "c:2"} and not ctx.runtime.clients.pending
    assert calls == [("http://c:1", "client/stop", {}, ctx.timeout)]
    assert ctx.events[0]["event"] == "client_result"


@pytest.mark.parametrize("bad", ["missing", "overflow", "unfinished", "writer"])
def test_stop_does_not_hide_missing_workers_or_execution_errors(monkeypatch, bad):
    value = {"clients": {"http://c:1": state(), "http://c:2": state()}, "errors": []}
    if bad == "missing":
        del value["clients"]["http://c:2"]
    if bad == "overflow":
        value["errors"] = ["queue full"]
    if bad == "unfinished":
        value["clients"]["http://c:2"]["done"] = False
    if bad == "writer":
        value["clients"]["http://c:2"]["errors"] = ["Client evidence write failed"]

    async def invoke(*a):
        return value

    monkeypatch.setattr(client_runtime, "invoke", invoke)
    with pytest.raises(ExecutionError):
        asyncio.run(
            StepRegistry.standard().get("client/stop/playback").invoke(context(), {})
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"qps": 0},
        {"qps": float("inf")},
        {"masters": []},
        {"masters": ["a", "a"]},
        {"total_concurrency": 0},
        {"concurrency": 2},
        {"queue_capacity": 0},
        {"request_timeout": "0s"},
    ],
)
def test_invalid_plans_fail_at_python_boundary(changes):
    with pytest.raises(ConfigError):
        StepRegistry.standard().get("client/start/playback").arguments(args() | changes)


def test_qps_reuses_metric_aggregation_and_measures_issued_not_finished(monkeypatch):
    async def query_counter(ctx, selection, role, numerator):
        assert role == "client" and numerator == "mock_client_requests_issued_total"
        return 123.0

    monkeypatch.setattr(metrics, "query_counter", query_counter)
    assert asyncio.run(client_query_qps(context(), metrics.MetricArgs())) == 123.0


@pytest.mark.parametrize("failed", [False, True])
def test_teardown_collects_unstopped_playback_before_final_scrape(failed, tmp_path):
    from mock_framework.execution.environment import RunEnvironment

    order = []

    async def collect(*a):
        order.append("client")
        if failed:
            raise ExecutionError("queue full")
        return {"127.0.0.1:9000": {"done": True}}

    async def scrape():
        order.append("scrape")

    async def close_monitor():
        order.append("monitor stopped")

    async def exercise():
        runtime = RunEnvironment(
            SimpleNamespace(environment={}),
            {"components": {}},
            SimpleNamespace(directory=tmp_path, emit=lambda **event: None),
        )
        runtime.clients.pending = True
        runtime.clients.stop_playback = collect
        runtime.monitoring = SimpleNamespace(
            wait_for_final_scrape=scrape,
            close=close_monitor,
            write_collection_result=lambda errors: None,
        )
        return await runtime.close()

    errors = asyncio.run(exercise())
    assert order == ["client", "scrape", "monitor stopped"]
    assert bool(errors) is failed
    if failed:
        assert "queue full" in errors[0]


@pytest.mark.parametrize("selection", [None, ["master:1", "master:2"]])
def test_master_selection_is_forwarded_without_freezing_default_discovery(
    monkeypatch, selection
):
    calls = []

    async def invoke(url, operation, args, timeout):
        calls.append(args)
        return {"accepted": True}

    monkeypatch.setattr(client_runtime, "invoke", invoke)
    ctx = context()
    ctx.nodes.append({"id": "master-1", "role": "master", "rpc": "master:2"})
    values = args()
    values.pop("masters")
    if selection is not None:
        values["masters"] = selection
    assert asyncio.run(
        StepRegistry.standard().get("client/start/playback").invoke(ctx, values)
    )
    assert calls[0].get("masters") == selection
    if selection is None:
        assert "masters" not in calls[0]


def test_unknown_master_selection_is_rejected_before_start(monkeypatch):
    async def invoke(*args):
        raise AssertionError("must not contact Client")

    monkeypatch.setattr(client_runtime, "invoke", invoke)
    ctx = context()
    with pytest.raises(ExecutionError, match="Unknown Master"):
        asyncio.run(
            StepRegistry.standard()
            .get("client/start/playback")
            .invoke(ctx, args() | {"masters": ["master:1", "missing:2"]})
        )
    assert not ctx.runtime.clients.pending


@pytest.mark.parametrize(
    "source",
    ["", "https://example.com/data", None, "requests.jsonl", "basic.mckidx.json"],
)
def test_missing_or_invalid_input_source_fails_contract(source):
    value = args()
    value["requests_file"] = source
    with pytest.raises(ConfigError):
        StepRegistry.standard().get("client/start/playback").arguments(value)


def test_removed_inline_input_is_rejected():
    with pytest.raises(ConfigError):
        StepRegistry.standard().get("client/start/playback").arguments(
            args() | {"requests": [{"token_ids": [1], "max_new_tokens": 2}]}
        )


def test_non_file_source_is_rejected_before_marking_client_pending(tmp_path):
    ctx = context()
    ctx.runtime.case = SimpleNamespace(directory=tmp_path)
    directory = tmp_path / "directory.mckidx"
    directory.mkdir()
    value = args()
    value["requests_file"] = str(directory)
    with pytest.raises(ExecutionError, match="regular file"):
        asyncio.run(
            StepRegistry.standard().get("client/start/playback").invoke(ctx, value)
        )
    assert not ctx.runtime.clients.pending


def test_replay_speed_replaces_qps_and_is_forwarded_without_timestamp_conversion(
    monkeypatch,
):
    ctx = context()
    calls = []

    async def start(body, timeout):
        calls.append(body)
        return True

    monkeypatch.setattr(ctx.runtime.clients, "start_playback", start)
    values = args()
    values.pop("qps")
    values["replay_speed"] = 5
    assert asyncio.run(
        StepRegistry.standard().get("client/start/playback").invoke(ctx, values)
    )
    assert calls[0]["replay_speed"] == 5 and "qps" not in calls[0]


@pytest.mark.parametrize("rate", [{"qps": 10}, {"replay_speed": 2}])
def test_warmup_is_forwarded_with_either_rate(monkeypatch, rate):
    ctx = context()
    calls = []

    async def start(body, timeout):
        calls.append(body)
        return True

    monkeypatch.setattr(ctx.runtime.clients, "start_playback", start)
    values = args()
    values.pop("qps")
    values.update(rate, warmup={"duration": "1.25s", "rate_scale": 0.25})
    assert asyncio.run(
        StepRegistry.standard().get("client/start/playback").invoke(ctx, values)
    )
    assert calls[0]["warmup"] == {"duration_ms": 1250, "rate_scale": 0.25}
    assert all(calls[0][name] == value for name, value in rate.items())


@pytest.mark.parametrize(
    "warmup",
    [
        {"duration": "0s", "rate_scale": 0.25},
        {"duration": "1us", "rate_scale": 0.25},
        {"duration": "25h", "rate_scale": 0.25},
        {"duration": "1s", "rate_scale": 0},
        {"duration": "1s", "rate_scale": 1},
        {"duration": "1s", "rate_scale": float("nan")},
        {"duration": "1s", "qps": 1},
    ],
)
def test_invalid_warmup_fails_at_python_boundary(warmup):
    with pytest.raises(ConfigError):
        StepRegistry.standard().get("client/start/playback").arguments(
            args() | {"warmup": warmup}
        )


@pytest.mark.parametrize(
    "rate",
    [
        {},
        {"qps": 1, "replay_speed": 5},
        {"replay_speed": 0},
        {"replay_speed": float("inf")},
        {"replay_speed": float("nan")},
    ],
)
def test_playback_requires_exactly_one_finite_positive_rate(rate):
    values = args()
    values.pop("qps")
    values.update(rate)
    with pytest.raises(ConfigError):
        StepRegistry.standard().get("client/start/playback").arguments(values)


def test_shared_requests_resolve_independently_of_case_directory(monkeypatch, tmp_path):
    ctx = context()
    ctx.runtime.case.directory = tmp_path / "cases/example"
    ctx.runtime.case.shared_root = tmp_path / "shared"
    source = ctx.runtime.case.shared_root / "data/traffic/example.mckidx"
    source.parent.mkdir(parents=True)
    source.touch()

    async def start(body, timeout):
        assert body["requests_file"] == str(source)
        return True

    monkeypatch.setattr(ctx.runtime.clients, "start_playback", start)
    values = args() | {"requests_file": "data/traffic/example.mckidx"}
    assert asyncio.run(
        StepRegistry.standard().get("client/start/playback").invoke(ctx, values)
    )
