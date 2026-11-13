import asyncio
import json
import os
import socket
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml
from mock_framework.config import ConfigError, compile_case
from mock_framework.execution import ExecutionError
from mock_framework.execution.launch import prepare_launch
from mock_framework.execution.runner import run_case
from mock_framework.execution.step.engine_scale import EngineScaleArgs, engine_scale
from mock_framework.execution.step.master_list_engines import (
    MasterListEnginesArgs,
    master_list_engines,
)
from mock_framework.execution.step.step_registry import StepRegistry, register_step
from mock_framework.execution.test.helpers import FRAMEWORK
from mock_framework.execution.test.helpers.input import write_inputs
from mock_framework.execution.test.helpers.zookeeper import command as zookeeper_command
from mock_framework.execution.transport import invoke

ROOT = FRAMEWORK


@dataclass(frozen=True)
class Args:
    pass


def setup_case(tmp_path, operation, *, invalid=False, traffic=False):
    registry = StepRegistry.standard()
    registry.register(operation)
    config = {
        "seq_size_per_block": 16,
        "test_block_num": 9,
        "reserve_block_ratio": 0,
        "max_seq_len": 4096,
        "mock.compute.forward.expression": "1",
    }
    env = {
        role: {
            "engine": "rtp",
            "deployment": {"replicas": 2 if role == "prefill" else 1},
            "config": dict(config),
        }
        for role in ("prefill", "decode")
    }
    if invalid:
        env["decode"]["config"]["test_block_num"] = 1
    if traffic:
        from mock_framework.execution.test.test_master import configuration

        config = configuration()
        config["FLEXLB_CONFIG"] = json.dumps(
            {
                "scheduler": {"type": "DIRECT"},
                "dispatcher": {"type": "NON_BATCH"},
                "requestLifecycle": {"request": {"timeoutMs": 3000}},
                "grpcServer": {
                    "executorCoreSize": 4,
                    "executorMaxSize": 8,
                    "executorQueueSize": 64,
                    "shutdownQuietPeriodMs": 10,
                },
            }
        )
        env["master"] = {"config": config}
        env["client"] = {}
    source = tmp_path / "case.yaml"
    source.write_text(
        yaml.safe_dump(
            {
                "id": "lifecycle",
                "describe": "Engine lifecycle",
                "environment": env,
                "phases": [
                    {
                        "id": "phase",
                        "describe": "instance operations",
                        "timeout": "60s",
                        "steps": [
                            {
                                "id": "exercise",
                                "describe": "exercise",
                                "interface": operation.definition.name,
                            }
                        ],
                    }
                ],
            }
        )
    )
    case = compile_case(source, registry)
    java = Path(os.environ["JAVA_HOME"]) / "bin/java"
    command = [
        str(java),
        "-cp",
        str(ROOT / "java/engine/target/mock-engine-service-1.0-SNAPSHOT-all.jar"),
        "org.flexlb.mock.engine.EngineMain",
    ]
    raw = {
        "components": {
            role: {"command": command} for role in env if role in ("prefill", "decode")
        }
    }
    if traffic:
        classpath = os.pathsep.join(
            [
                str(ROOT / "java/master/target/classes"),
                (ROOT / "java/master/target/dependencies.txt").read_text().strip(),
            ]
        )
        prefix = [str(java), "-XX:ActiveProcessorCount=2", "-cp", classpath]
        raw["components"]["master"] = {
            "command": prefix + ["org.flexlb.MasterMain"],
            "startup_timeout": "60s",
        }
        raw["zookeeper"] = {"command": zookeeper_command(ROOT, java)}
        raw["components"]["client"] = {
            "command": [
                str(java),
                "-jar",
                str(ROOT / "java/client/target/mock-client-1.0-SNAPSHOT-all.jar"),
            ]
        }
    return case, prepare_launch(raw, case, tmp_path)


def no_children(output):
    assert not (output / "topology-history.json").exists()
    events = [
        json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()
    ]
    for event in events:
        if event["event"] == "process_started":
            with pytest.raises(ProcessLookupError):
                os.kill(event["pid"], 0)
    return events


@pytest.mark.parametrize(
    "args",
    [
        {},
        {"direction": "other"},
        {"direction": "out", "targets": ["p:1"]},
        {"direction": "out", "count": 1025},
    ],
)
def test_scale_rejects_invalid_arguments(args):
    with pytest.raises(ConfigError):
        StepRegistry.standard().get("engine/scale").build_arguments(args)


def test_scale_in_accepts_count_above_start_limit():
    step = StepRegistry.standard().get("engine/scale")
    assert step.build_arguments({"direction": "in", "count": 1025}).count == 1025


def test_initial_engine_start_shares_slots_across_roles(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from mock_framework.execution.environment import EndpointRegistry
    from mock_framework.execution.processes import ProcessIdentity, ProcessSupervisor
    from mock_framework.execution.roles import ENGINE_ROLES
    from mock_framework.execution.roles.engine import (
        INITIAL_START_CONCURRENCY,
        EngineRole,
    )

    async def exercise():
        records = SimpleNamespace(directory=tmp_path, emit=lambda **_: None)
        count = INITIAL_START_CONCURRENCY // len(ENGINE_ROLES) + 2
        topology = EndpointRegistry(tmp_path)
        engine = EngineRole(
            supervisor=ProcessSupervisor(records),
            topology=topology,
            records=records,
            environment={
                role: {"deployment": {"replicas": count}} for role in ENGINE_ROLES
            },
            launch_by_role={role: {"startup_timeout": "2s"} for role in ENGINE_ROLES},
            monitoring=None,
        )
        active = peak = 0
        starts = []
        ready = []
        refilled = asyncio.Event()

        async def start_instance(role, index):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            starts.append(role)
            if len(starts) > INITIAL_START_CONCURRENCY:
                refilled.set()
            if (role, index) == ("prefill", 0):
                await refilled.wait()
            await asyncio.sleep(0)
            active -= 1
            ready.append((role, index))
            slot = ENGINE_ROLES.index(role) * count + index
            control = f"http://localhost:{10000 + slot}"
            return (
                ProcessIdentity(role, index),
                slot,
                control,
                {
                    "id": f"{role}-{index}",
                    "role": role,
                    "control": control,
                    "rpc": f"localhost:{20000 + slot}",
                },
            )

        monkeypatch.setattr(engine, "_start_instance", start_instance)
        await asyncio.wait_for(engine.start(), timeout=2)
        assert peak == INITIAL_START_CONCURRENCY
        assert set(starts[:INITIAL_START_CONCURRENCY]) == set(ENGINE_ROLES)
        assert ready[0] != ("prefill", 0)
        for role in ENGINE_ROLES:
            assert [node["id"] for node in topology.list_active_endpoints(role)] == [
                f"{role}-{i}" for i in range(count)
            ]

    asyncio.run(exercise())


def test_scale_updates_topology_and_preserves_instance_artifacts(tmp_path):
    @register_step("probe/lifecycle", effect="write")
    async def exercise(ctx, args: Args) -> bool:
        original = ctx.list_rpc_addresses()["prefill"]
        retired = ctx.records.directory / "components/prefill-0"
        endpoints = (retired / "endpoints.json").read_bytes()

        removed = await engine_scale(ctx, EngineScaleArgs(direction="in", count=1))
        assert json.loads(endpoints)["rpc"] == removed[0]
        assert set(ctx.list_rpc_addresses()["prefill"]) == set(original) - set(removed)
        assert not (retired / "ready.json").exists()
        with pytest.raises(ExecutionError):
            await engine_scale(ctx, EngineScaleArgs(direction="in", targets=removed))

        created = await engine_scale(ctx, EngineScaleArgs(direction="out", count=2))
        assert len(created) == 2 and not set(created) & set(original)
        assert (retired / "endpoints.json").read_bytes() == endpoints
        assert len(ctx.list_rpc_addresses()["prefill"]) == 3
        assert len(await engine_scale(ctx, EngineScaleArgs(direction="in"))) == 3
        assert ctx.list_rpc_addresses()["prefill"] == []
        assert len(await engine_scale(ctx, EngineScaleArgs(direction="out"))) == 1
        return True

    case, launch = setup_case(tmp_path, exercise)
    output = tmp_path / "result"
    result = asyncio.run(run_case(case, launch, output))
    assert result["status"] == "PASS", result
    no_children(output)
    assert json.loads((output / "topology.json").read_text()) == []


@pytest.mark.parametrize("invalid", [False, True])
def test_process_failure_or_partial_startup_fails_the_run_and_cleans_every_process(
    tmp_path, invalid
):
    @register_step("probe/crash", effect="write")
    async def crash(ctx, args: Args) -> bool:
        ctx.runtime.supervisor.child("prefill-0").kill()
        await asyncio.sleep(20)
        return True

    case, launch = setup_case(tmp_path, crash, invalid=invalid)
    output = tmp_path / "result"
    result = asyncio.run(run_case(case, launch, output))
    assert result["status"] == "ERROR"
    events = no_children(output)
    assert len([event for event in events if event["event"] == "process_started"]) == 3
    if not invalid:
        assert "prefill-0" in result["error"]


def test_allocator_exhaustion_fails_without_reusing_a_retired_listener(monkeypatch):
    from mock_framework.execution.roles.engine import EngineRole

    owner = EngineRole(
        supervisor=None,
        topology=None,
        records=None,
        environment={},
        launch_by_role={},
        monitoring=None,
    )
    owner.used_ports.add(12345)

    class Probe:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def bind(self, address):
            pass

        def getsockname(self):
            return ("127.0.0.1", 12345)

    monkeypatch.setattr(socket, "socket", Probe)
    with pytest.raises(ExecutionError, match="fresh Engine listener"):
        owner._port()


@pytest.mark.parametrize("operation", ["create", "stop", "stop_after_exit"])
def test_partial_scale_failure_is_not_replayed_or_reported_as_success(
    tmp_path, operation
):
    @register_step("probe/partial", effect="write")
    async def exercise(ctx, args: Args) -> bool:
        calls = []
        owner = ctx.runtime.engines if operation == "create" else ctx.runtime.supervisor
        method = "_start_instance" if operation == "create" else "stop_intentionally"
        original = getattr(owner, method)

        async def fail_second(*args):
            calls.append(args[1] if operation == "create" else args[0])
            if len(calls) == 2:
                ctx.records.emit(
                    event="injected_failure", operation=operation, calls=calls
                )
                if operation == "stop_after_exit":
                    assert await original(*args) == []
                    return ["injected process failure"]
                raise ExecutionError("injected process failure")
            return await original(*args)

        setattr(owner, method, fail_second)
        if operation == "create":
            await engine_scale(ctx, EngineScaleArgs(direction="out", count=2))
        else:
            try:
                await engine_scale(ctx, EngineScaleArgs(direction="in", count=2))
            except ExecutionError:
                assert len(ctx.runtime.topology.list_active_endpoints("prefill")) == (
                    0 if operation == "stop_after_exit" else 1
                )
                raise
        raise AssertionError("Partial failure was swallowed")

    case, launch = setup_case(tmp_path, exercise)
    output = tmp_path / "result"
    result = asyncio.run(run_case(case, launch, output))
    assert result["status"] == "ERROR", result
    events = no_children(output)
    injected = [e for e in events if e["event"] == "injected_failure"]
    assert len(injected) == 1 and len(injected[0]["calls"]) == 2
    assert len([e for e in events if e["event"] == "instance_started"]) == (
        4 if operation == "create" else 3
    )
    assert len([e for e in events if e["event"] == "instance_stopped"]) == (
        {"create": 0, "stop": 1, "stop_after_exit": 2}[operation]
    )
    assert json.loads((output / "topology.json").read_text()) == []


def test_real_master_routes_client_traffic_after_engine_scale(tmp_path):
    source = tmp_path / "traffic.mckidx"
    write_inputs(
        source,
        [(3, 0, {"max_output_tokens": 3}, 50, [])] * 320,
        chains=[(1, 0, 3)],
    )

    @register_step("probe/live_scale", effect="write")
    async def exercise(ctx, args: Args) -> bool:
        runtime = ctx.runtime
        (master,) = ctx.list_rpc_addresses()["masters"]
        (client,) = runtime.topology.list_active_endpoints("client")
        await runtime.clients.start_playback(
            {
                "requests_file": str(source),
                "qps": 8,
                "total_concurrency": 4,
                "queue_capacity": 16,
                "request_timeout_ms": 3000,
                "drain_timeout_ms": 5000,
            },
            ctx.timeout,
        )

        async def state():
            value = await invoke(
                client["control"], "worker/state", {"sequence": 1}, ctx.timeout
            )
            assert not value["errors"], value
            return value

        async with asyncio.timeout(20):
            while (before := await state())["outcomes"]["success"] < 5:
                await asyncio.sleep(0.1)

            original = ctx.list_rpc_addresses()["prefill"]
            created = await engine_scale(ctx, EngineScaleArgs(direction="out"))
            removed = await engine_scale(
                ctx, EngineScaleArgs(direction="in", targets=original)
            )
            assert set(removed) == set(original)
            assert ctx.list_rpc_addresses()["prefill"] == created

            while await master_list_engines(ctx, MasterListEnginesArgs()) != {
                master: created
            }:
                await asyncio.sleep(0.1)
            assert (await state())["issued"] > before["issued"]
            successful = (await state())["outcomes"]["success"]
            while (await state())["outcomes"]["success"] < successful + 5:
                await asyncio.sleep(0.1)
        return True

    case, launch = setup_case(tmp_path, exercise, traffic=True)
    output = tmp_path / "result"
    result = asyncio.run(run_case(case, launch, output))
    assert result["status"] == "PASS", result
    no_children(output)
