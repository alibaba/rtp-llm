import asyncio
from types import SimpleNamespace

import pytest
from mock_framework.config import ConfigError
from mock_framework.execution import ExecutionError
from mock_framework.execution.roles.engine import EngineRole
from mock_framework.execution.step.step_registry import StepRegistry


def engine_role(active=("prefill-0", "prefill-1")):
    stops = []
    nodes = {
        name: {
            "id": name,
            "role": name.rsplit("-", 1)[0],
            "rpc": f"127.0.0.1:{1000 + index}",
        }
        for index, name in enumerate(active)
    }
    topology = SimpleNamespace(
        list_active_endpoints=lambda role=None: [
            node for node in nodes.values() if role is None or node["role"] == role
        ]
    )

    async def stop_intentionally(process_name, *, mode="terminate", grace=3):
        stops.append((process_name, mode))
        return []

    supervisor = SimpleNamespace(stop_intentionally=stop_intentionally)
    events = []
    records = SimpleNamespace(emit=lambda **event: events.append(event))
    role = EngineRole(
        supervisor=supervisor,
        topology=topology,
        records=records,
        environment={},
        launch_by_role={},
        monitoring=None,
    )
    return role, stops, events, nodes


def test_stop_kills_the_process_but_keeps_the_discovery_entry():
    role, stops, events, nodes = engine_role()
    result = asyncio.run(role.stop("prefill", ["127.0.0.1:1001"], None, "kill"))
    assert stops == [("prefill-1", "kill")]
    assert result == ["127.0.0.1:1001"]
    assert nodes.keys() == {"prefill-0", "prefill-1"}
    assert events == [
        {
            "event": "engine_stopped",
            "node": "prefill-1",
            "rpc": "127.0.0.1:1001",
            "role": "prefill",
            "mode": "kill",
        }
    ]


def test_stop_selects_the_first_active_instances_by_count():
    role, stops, _, _ = engine_role()
    result = asyncio.run(role.stop("prefill", None, 1, "terminate"))
    assert stops == [("prefill-0", "terminate")]
    assert result == ["127.0.0.1:1000"]


def test_stop_reports_supervisor_errors():
    role, _, _, _ = engine_role()

    async def failing(process_name, *, mode="terminate", grace=3):
        return ["boom"]

    role.supervisor = SimpleNamespace(stop_intentionally=failing)
    with pytest.raises(ExecutionError):
        asyncio.run(role.stop("prefill", ["127.0.0.1:1000"], None, "kill"))


@pytest.mark.parametrize(
    "targets,count,mode",
    [
        ([], None, "kill"),
        (["127.0.0.1:1000", "127.0.0.1:1000"], None, "kill"),
        (["127.0.0.1:9999"], None, "kill"),
        (["127.0.0.1:1000"], None, "restart"),
        (None, 3, "kill"),
    ],
)
def test_stop_rejects_bad_targets_counts_and_modes(targets, count, mode):
    role, _, _, _ = engine_role()
    with pytest.raises(ExecutionError):
        asyncio.run(role.stop("prefill", targets, count, mode))


def test_stop_step_arguments_require_targets_or_count():
    definition = StepRegistry.standard().get("engine/stop")
    with pytest.raises(ConfigError):
        definition.build_arguments({"role": "prefill"})
    with pytest.raises(ConfigError):
        definition.build_arguments({"targets": ["a"], "count": 1})
    with pytest.raises(ConfigError):
        definition.build_arguments({"targets": ["a"], "mode": "ignite"})
    parsed = definition.build_arguments({"targets": ["a"]})
    assert parsed.role == "prefill"
    assert parsed.mode == "kill"
    parsed = definition.build_arguments({"count": 2, "role": "decode"})
    assert parsed.targets is None
    assert parsed.count == 2
