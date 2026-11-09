import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import pytest
import yaml
from mock_framework.config import compile_case
from mock_framework.execution import ExecutionError
from mock_framework.execution.launch import prepare_launch
from mock_framework.execution.runner import run_case
from mock_framework.execution.step.registry import Bounds, Registry, register_step
from mock_framework.execution.step.step import Reference, Step, resolve_refs


@dataclass(frozen=True)
class Args:
    value: Annotated[int, Bounds(0, 10)] = 2
    delay: float = 0
    invalid: bool = False
    crash: bool = False


@register_step("probe/read", effect="read")
class read(Step):
    @staticmethod
    async def execute(ctx, args: Args) -> int:
        await asyncio.sleep(args.delay)
        return "invalid" if args.invalid else args.value


@register_step("probe/write", effect="write")
class write(Step):
    @staticmethod
    async def execute(ctx, args: Args) -> int:
        ctx.records.emit(event="mutation")
        if args.crash:
            ctx.runtime.supervisor.children[0][1].kill()
        await asyncio.sleep(args.delay)
        return args.value


def step(name="read", **kw):
    return {"id": name, "describe": name, "interface": "probe/" + name, **kw}


@pytest.fixture
def setup(tmp_path):
    def make(steps, timeout="2s", child="ready"):
        registry = Registry.standard()
        registry.register(read)
        registry.register(write)
        # Real subprocess ownership and teardown, independent of any Java Case catalog.
        script = tmp_path / "peer.py"
        script.write_text(
            "import os,json,pathlib,time,sys\n"
            + (
                "sys.exit(19)\n"
                if child == "exit"
                else (
                    'pathlib.Path(os.environ["MOCK_READY_FILE"]).write_text(json.dumps({"url":"http://127.0.0.1:1"}))\ntime.sleep(30)\n'
                    if child == "ready"
                    else "time.sleep(30)\n"
                )
            )
        )
        document = {
            "id": "case",
            "describe": "runtime",
            "environment": {"master": {"config": {}}},
            "phases": [
                {"id": "phase", "describe": "phase", "timeout": timeout, "steps": steps}
            ],
        }
        path = tmp_path / "case.yaml"
        path.write_text(yaml.safe_dump(document))
        case = compile_case(path, registry)
        launch = prepare_launch(
            {
                "components": {
                    "master": {
                        "command": [sys.executable, str(script)],
                        "startup_timeout": "150ms" if child != "ready" else "3s",
                    }
                }
            },
            case,
            tmp_path,
        )
        return case, registry, launch

    return make


def events(out, name="events"):
    return [
        json.loads(line) for line in (out / (name + ".jsonl")).read_text().splitlines()
    ]


def cleaned(out):
    starts = [e for e in events(out) if e["event"] == "process_started"]
    stops = [e for e in events(out) if e["event"] == "process_stopped"]
    assert len(starts) == len(stops) == 1
    for e in starts:
        with pytest.raises(ProcessLookupError):
            os.kill(e["pid"], 0)


def test_cli_sigterm_cancels_and_cleans_owned_process(setup, tmp_path):
    case, _, launch = setup(
        [{"id": "leader", "describe": "leader", "interface": "master/get"}],
        child="hung",
    )
    launch["components"]["master"]["startup_timeout"] = "30s"
    recipe = tmp_path / "launch.yaml"
    recipe.write_text(yaml.safe_dump(launch))
    out = tmp_path / "terminated"
    child = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "mock_framework.case",
            "run",
            str(case.source),
            "--launch",
            str(recipe),
            "--output",
            str(out),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if (out / "events.jsonl").exists() and any(
                e["event"] == "process_started" for e in events(out)
            ):
                break
            assert child.poll() is None
            time.sleep(0.02)
        else:
            pytest.fail("CLI did not start its owned process")
        child.send_signal(signal.SIGTERM)
        _, stderr = child.communicate(timeout=10)
        assert child.returncode == 130, stderr
        assert json.loads((out / "result.json").read_text())["status"] == "CANCELLED"
        cleaned(out)
    finally:
        if child.poll() is None:
            child.kill()
            child.communicate(timeout=5)


def run(setup, tmp_path, steps, **kw):
    case, registry, launch = setup(steps, **kw)
    out = tmp_path / "out"
    result = asyncio.run(run_case(case, registry, launch, out))
    cleaned(out)
    return result, out


def test_reads_writes_refs_and_gates_share_one_path(setup, tmp_path):
    result, out = run(
        setup,
        tmp_path,
        [
            step(),
            step("write", args={"value": {"ref": "steps.read"}}, expect={"eq": 2}),
        ],
    )
    assert result["status"] == "PASS"
    assert len([e for e in events(out) if e["event"] == "mutation"]) == 1
    assert events(out, "gates")[0]["actual"] == 2
    assert (out / "interfaces.json").is_file()


@pytest.mark.parametrize(
    "args,expect,status",
    [({}, {"eq": 3}, "FAIL"), ({"invalid": True}, {"eq": 2}, "ERROR")],
)
def test_bad_result_or_gate_stops_following_steps(
    setup, tmp_path, args, expect, status
):
    result, out = run(setup, tmp_path, [step(args=args, expect=expect), step("write")])
    assert result["status"] == status
    assert not [e for e in events(out) if e["event"] == "mutation"]


def test_timed_out_write_is_not_retried(setup, tmp_path):
    result, out = run(
        setup, tmp_path, [step("write", args={"delay": 1}, within="30ms")]
    )
    assert result["status"] == "ERROR"
    assert len([e for e in events(out) if e["event"] == "mutation"]) == 1
    assert next(e for e in events(out) if e["event"] == "step_finished")[
        "outcome_unknown"
    ]


def test_failed_write_assertion_is_not_replayed_or_marked_unknown(setup, tmp_path):
    result, out = run(setup, tmp_path, [step("write", expect={"eq": 3})])
    assert result["status"] == "FAIL"
    assert len([e for e in events(out) if e["event"] == "mutation"]) == 1
    assert not next(e for e in events(out) if e["event"] == "step_finished")[
        "outcome_unknown"
    ]


def test_phase_budget_includes_earlier_steps(setup, tmp_path):
    result, out = run(
        setup,
        tmp_path,
        [step(args={"delay": 0.08}), step("write", args={"delay": 0.2}, within="5s")],
        timeout="150ms",
    )
    assert result["status"] == "ERROR"
    assert [e["status"] for e in events(out) if e["event"] == "step_finished"] == [
        "PASS",
        "ERROR",
    ]


@pytest.mark.parametrize("child", ["exit", "hang"])
def test_setup_failure_reclaims_process(setup, tmp_path, child):
    result, out = run(setup, tmp_path, [step()], child=child)
    assert result["status"] == "ERROR"
    assert not [e for e in events(out) if e["event"] == "step_started"]


def test_child_exit_interrupts_operation(setup, tmp_path):
    result, _ = run(setup, tmp_path, [step("write", args={"crash": True, "delay": 1})])
    assert result["status"] == "ERROR" and "unexpectedly" in result["error"]


def test_cancel_records_and_reclaims(setup, tmp_path):
    case, registry, launch = setup([step(args={"delay": 10})])
    out = tmp_path / "out"

    async def exercise():
        task = asyncio.create_task(run_case(case, registry, launch, out))
        for _ in range(500):
            if (out / "events.jsonl").exists() and any(
                e["event"] == "step_started" for e in events(out)
            ):
                break
            await asyncio.sleep(0.01)
        else:
            pytest.fail("Step did not start")
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(exercise())
    cleaned(out)
    assert json.loads((out / "result.json").read_text())["status"] == "CANCELLED"


def test_existing_output_is_not_overwritten(setup, tmp_path):
    case, registry, launch = setup([step()])
    with pytest.raises(FileExistsError):
        asyncio.run(run_case(case, registry, launch, tmp_path))


def test_refs_do_not_eval_and_require_actual_fields():
    assert resolve_refs(Reference("x", ("value",)), {"x": {"value": 2}}, {}) == 2
    assert resolve_refs(
        Reference("globals", ("masters",)), {}, {"masters": ["a:1"]}
    ) == ["a:1"]
    with pytest.raises(ExecutionError):
        resolve_refs(Reference("x", ("missing",)), {"x": {}}, {})


def test_cli_compiles_without_java_catalog(setup, tmp_path):
    case, _, _ = setup(
        [
            {
                "id": "leader",
                "describe": "leader",
                "interface": "master/get",
                "args": {"is_leader": True},
                "wait_after": "1ms",
            }
        ]
    )
    output = tmp_path / "compiled.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mock_framework.case",
            "compile",
            str(case.source),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert (
        json.loads(output.read_text())["phases"][0]["steps"][0]["wait_after"] == "1ms"
    )


def test_wait_after_preserves_refs_and_also_runs_on_last_step(setup, tmp_path):
    from datetime import datetime

    result, out = run(
        setup,
        tmp_path,
        [
            step(wait_after="40ms", expect={"eq": 2}),
            step("write", args={"value": {"ref": "steps.read"}}, wait_after="40ms"),
        ],
    )
    assert result["status"] == "PASS"
    log = events(out)
    for name in ("read", "write"):
        waiting = next(
            e for e in log if e["event"] == "wait_started" and e["step"] == name
        )
        finished = next(
            e for e in log if e["event"] == "step_finished" and e["step"] == name
        )
        assert finished["result"] == 2
        assert (
            datetime.fromisoformat(finished["recorded_at"])
            - datetime.fromisoformat(waiting["recorded_at"])
        ).total_seconds() >= 0.035
    assert [e["event"] for e in log if e.get("step") == "read"] == [
        "step_started",
        "wait_started",
        "step_finished",
    ]


@pytest.mark.parametrize(
    "args,expect,status", [({}, {"eq": 3}, "FAIL"), ({"invalid": True}, None, "ERROR")]
)
def test_failed_call_or_assertion_does_not_wait(setup, tmp_path, args, expect, status):
    options = {"expect": expect} if expect else {}
    result, out = run(
        setup, tmp_path, [step(args=args, wait_after="10s", **options)], timeout="1s"
    )
    assert result["status"] == status
    assert not any(e["event"] == "wait_started" for e in events(out))


@pytest.mark.parametrize("within,timeout", [("150ms", "2s"), (None, "150ms")])
def test_wait_timeout_preserves_successful_write_and_stops_next_step(
    setup, tmp_path, within, timeout
):
    result, out = run(
        setup,
        tmp_path,
        [step("write", args={"delay": 0.1}, wait_after="100ms", within=within), step()],
        timeout=timeout,
    )
    assert result["status"] == "ERROR"
    finished = [e for e in events(out) if e["event"] == "step_finished"]
    assert len(finished) == 1 and finished[0]["error_stage"] == "wait_after"
    assert finished[0]["result"] == 2 and not finished[0]["outcome_unknown"]
    assert len([e for e in events(out) if e["event"] == "mutation"]) == 1


@pytest.mark.parametrize("interrupt", ["cancel", "process_exit"])
def test_wait_is_interruptible_without_losing_completed_action(
    setup, tmp_path, interrupt
):
    case, registry, launch = setup(
        [step("write", wait_after="10s"), step()], timeout="20s"
    )
    out = tmp_path / "out"

    async def exercise():
        task = asyncio.create_task(run_case(case, registry, launch, out))
        async with asyncio.timeout(5):
            while not (out / "events.jsonl").exists() or not any(
                e["event"] == "wait_started" for e in events(out)
            ):
                await asyncio.sleep(0.01)
            if interrupt == "cancel":
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            else:
                import signal

                pid = next(
                    e["pid"] for e in events(out) if e["event"] == "process_started"
                )
                os.kill(pid, signal.SIGTERM)
                result = await task
                assert result["status"] == "ERROR" and "unexpectedly" in result["error"]

    asyncio.run(exercise())
    cleaned(out)
    finished = [e for e in events(out) if e["event"] == "step_finished"]
    assert len(finished) == 1 and finished[0]["error_stage"] == "wait_after"
    assert finished[0]["result"] == 2 and not finished[0]["outcome_unknown"]


def test_multiple_masters_always_use_distinct_processes(setup, tmp_path):
    case, registry, launch = setup([step()])
    case.environment["master"]["deployment"]["replicas"] = 2
    output = tmp_path / "out"
    assert asyncio.run(run_case(case, registry, launch, output))["status"] == "PASS"
    starts = [e for e in events(output) if e["event"] == "process_started"]
    assert {e["node"] for e in starts} == {"master-0", "master-1"}
    assert len({e["pid"] for e in starts}) == 2
    for event in starts:
        with pytest.raises(ProcessLookupError):
            os.kill(event["pid"], 0)


@pytest.mark.parametrize("exit_code", [None, 0])
def test_intentional_stop_retains_unreaped_process_for_final_cleanup(
    exit_code, monkeypatch
):
    from types import SimpleNamespace

    from mock_framework.execution.processes import ProcessSupervisor

    async def exercise():
        p = ProcessSupervisor(SimpleNamespace(emit=lambda **event: None))

        async def wait():
            child.returncode = exit_code
            if exit_code is None:
                raise TimeoutError()

        child = SimpleNamespace(returncode=None, pid=123, wait=wait)
        p._children["engine-prefill-0"] = child
        monkeypatch.setattr(
            "mock_framework.execution.processes.os.killpg",
            lambda *args: setattr(child, "returncode", exit_code),
        )
        original = p.stop

        async def stop(names, grace):
            return await original(names, 0)

        p.stop = stop
        errors = await p.stop_intentionally("engine-prefill-0")
        assert bool(p.children) == (exit_code is None)
        assert bool(errors) == (exit_code is None)
        assert "engine-prefill-0" in p._expected

    asyncio.run(exercise())


def test_master_ports_avoid_ephemeral_range_and_retry_busy_pair(monkeypatch):
    import socket

    from mock_framework.execution.roles import master as master_runtime

    first = master_runtime.master_port(1)
    assert 10000 <= first < 30000
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", first + 2))
        second = master_runtime.master_port(1)
        choices = iter([first - 10000, second - 10000])
        monkeypatch.setattr(
            master_runtime.secrets, "randbelow", lambda bound: next(choices)
        )
        assert master_runtime.master_port(1) == second


def test_runner_constructs_arguments_once_per_step(case_data, compile_data, tmp_path):
    constructions = []

    @dataclass(frozen=True)
    class Counted:
        value: int

        def __post_init__(self):
            constructions.append(self.value)

    @register_step("probe/counted", effect="read")
    class counted(Step):
        @staticmethod
        async def execute(ctx, args: Counted) -> int:
            return args.value

    registry = Registry((counted,))
    case_data["environment"] = {}
    case_data.pop("view", None)
    case_data["phases"][0]["steps"] = [
        {
            "id": "counted",
            "describe": "construct once",
            "interface": "probe/counted",
            "args": {"value": 3},
            "expect": {"eq": 3},
        }
    ]
    case = compile_data(case_data, interfaces=registry)
    constructions.clear()
    launch = prepare_launch({"components": {}}, case, tmp_path)
    result = asyncio.run(run_case(case, registry, launch, tmp_path / "out"))
    assert result["status"] == "PASS"
    assert constructions == [3]


def test_constructor_failure_after_reference_resolution_is_not_an_unknown_write(
    case_data, compile_data, tmp_path
):
    mutations = []

    @dataclass(frozen=True)
    class Positive:
        value: int

        def __post_init__(self):
            if self.value <= 0:
                raise ValueError("value must be positive")

    @register_step("probe/zero", effect="read")
    class zero(Step):
        @staticmethod
        async def execute(ctx, args: Args) -> int:
            return 0

    @register_step("probe/positive", effect="write")
    class positive(Step):
        @staticmethod
        async def execute(ctx, args: Positive) -> int:
            mutations.append(args.value)
            return args.value

    registry = Registry((zero, positive))
    case_data["environment"] = {}
    case_data.pop("view", None)
    case_data["phases"][0]["steps"] = [
        {"id": "zero", "describe": "source", "interface": "probe/zero"},
        {
            "id": "positive",
            "describe": "reject before write",
            "interface": "probe/positive",
            "args": {"value": {"ref": "steps.zero"}},
        },
    ]
    case = compile_data(case_data, interfaces=registry)
    launch = prepare_launch({"components": {}}, case, tmp_path)
    output = tmp_path / "out"
    result = asyncio.run(run_case(case, registry, launch, output))
    assert result["status"] == "ERROR"
    failed = [event for event in events(output) if event["event"] == "step_finished"][
        -1
    ]
    assert failed["error_stage"] == "arguments"
    assert failed["outcome_unknown"] is False
    assert "value must be positive" in failed["error"]
    assert mutations == []


def test_trace_correlates_steps_and_gates_across_streams(setup, tmp_path):
    case, registry, launch = setup([step(expect={"eq": 2})])
    output = tmp_path / "trace"
    assert asyncio.run(run_case(case, registry, launch, output))["status"] == "PASS"
    events = [
        json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()
    ]
    gates = [
        json.loads(line) for line in (output / "gates.jsonl").read_text().splitlines()
    ]
    result = json.loads((output / "result.json").read_text())
    records = events + gates
    assert {item["run_id"] for item in records} == {result["run_id"]}
    assert sorted(item["seq"] for item in records) == list(range(1, len(records) + 1))
    starts = {
        (event["phase_id"], event["step_id"], event["action"]): event["seq"]
        for event in events
        if event["event"] == "step_started"
    }
    assert gates
    for gate in gates:
        assert starts[gate["phase_id"], gate["step_id"], gate["action"]] < gate["seq"]


def test_compiler_rejects_reference_to_background_result(setup):
    from mock_framework.config import ConfigError

    with pytest.raises(ConfigError, match="Background Step 'read'"):
        setup(
            [
                step(background=True),
                step("write", args={"value": {"ref": "steps.read"}}),
            ]
        )


def test_background_failure_interrupts_foreground_and_reaps_process(setup, tmp_path):
    result, out = run(
        setup,
        tmp_path,
        [
            step(background=True, args={"delay": 0.05}, expect={"eq": 3}),
            step("write", args={"delay": 10}),
        ],
    )
    assert result["status"] == "FAIL"
    finished = {
        event["step_id"]: event["status"]
        for event in events(out)
        if event["event"] == "step_finished"
    }
    assert finished == {"read": "FAIL", "write": "CANCELLED"}


def test_phase_waits_for_background_to_finish(setup, tmp_path):
    result, out = run(
        setup, tmp_path, [step(background=True, args={"delay": 0.05}, expect={"eq": 2})]
    )
    assert result["status"] == "PASS"
    records = events(out)
    assert next(x["seq"] for x in records if x["event"] == "step_finished") < next(
        x["seq"] for x in records if x["event"] == "phase_finished"
    )


def test_engine_allocation_excludes_master_ports_before_master_listeners_start(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    from mock_framework.execution.environment import RunEnvironment
    from mock_framework.execution.roles import master

    monkeypatch.setattr(master, "master_port", lambda count: 26133)
    (tmp_path / "zookeeper").mkdir()

    async def check():
        case = SimpleNamespace(
            environment={"master": {"config": {}, "deployment": {"replicas": 2}}}
        )
        runtime = RunEnvironment(
            case, {"components": {"master": {}}}, SimpleNamespace(directory=tmp_path)
        )

        async def zookeeper_start():
            return "127.0.0.1:2181"

        async def engines_start():
            assert {26133, 26135} <= runtime.engines.used_ports

        async def nothing():
            pass

        runtime.zookeeper.start = zookeeper_start
        runtime.engines.start = engines_start
        runtime.masters.start = nothing
        runtime.clients.start_nodes = nothing
        await runtime.start()

    asyncio.run(check())
