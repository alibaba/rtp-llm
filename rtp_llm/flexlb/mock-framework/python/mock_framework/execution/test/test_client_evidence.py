import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from mock_framework.config import compile_case
from mock_framework.execution import ExecutionError
from mock_framework.execution.evidence.request_journal import (
    playback_directories,
    read_requests,
)
from mock_framework.execution.launch import prepare_launch
from mock_framework.execution.roles import client as client_runtime
from mock_framework.execution.runner import run_case
from mock_framework.execution.step.step_registry import StepRegistry
from mock_framework.execution.test.helpers import FRAMEWORK
from mock_framework.execution.test.helpers.input import write_inputs
from mock_framework.execution.test.helpers.zookeeper import command as zookeeper_command


def write_rows(directory, rows):
    (directory / "requests.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )


@pytest.fixture
def evidence(tmp_path):
    def create(
        name="client-0",
        playback=1,
        first=9007199254740993,
        index=0,
        stride=1,
        outcome="success",
    ):
        directory = tmp_path / "components" / name / "playback" / str(playback)
        directory.mkdir(parents=True)
        plan = dict(
            sequence=playback,
            first_request_id=first,
            first_index=index,
            stride=stride,
            count=1,
            concurrency=1,
        )
        (directory / "plan.json").write_text(json.dumps(plan))
        header = dict(
            version=1,
            client=name,
            playback=str(playback),
            first_request_id=str(first),
            first_index=str(index),
            stride=stride,
            planned="1",
            concurrency=1,
        )
        events = []
        for n, kind in enumerate(
            ("ISSUED", "SCHEDULE_START", "SCHEDULE_RESULT", "FIRST_TOKEN", "TERMINAL"),
            1,
        ):
            event = dict(
                sequence=str(n),
                request_id=str(first + index),
                epoch_ms=str(1000 + n),
                elapsed_nanos=str(n * 10),
                kind=kind,
            )
            if kind in ("SCHEDULE_START", "SCHEDULE_RESULT"):
                event.update(attempt=1, master="m:1")
            if kind == "SCHEDULE_RESULT":
                event.update(grpc_code="OK", error_code="200")
            if kind == "TERMINAL":
                event.update(
                    outcome=outcome, grpc_code="OK", ttft_nanos="40", output_tokens=3
                )
            events.append(event)
        rows = [header, *events, {"seal": {"issued": "1", "finished": "1"}}]
        result = dict(
            done=True,
            issued=1,
            finished=1,
            unsent=0,
            outcomes={
                v: int(v == outcome)
                for v in ("success", "error", "timeout", "cancelled")
            },
        )
        (directory / "result.json").write_text(json.dumps(result))
        write_rows(directory, rows)
        (tmp_path / "components/client-0/coordinator" / str(playback)).mkdir(
            parents=True, exist_ok=True
        )
        return directory, rows

    return create


def test_roundtrip_preserves_int64_identity_and_reconciles_summary(evidence):
    directory, _ = evidence()
    (result,) = read_requests(directory)
    assert result.request_id == 9007199254740993
    assert result.attempts[0].master == "m:1" and result.output_tokens == 3
    assert result.ttft_nanos == 40 and result.elapsed_nanos == 50


def test_reads_three_schedule_attempts_in_sealed_journal(evidence):
    directory, rows = evidence()
    header, issued, start, reply, first_token, terminal, seal = rows
    events = [
        issued,
        start,
        reply,
        dict(start, attempt=2),
        dict(reply, attempt=2, error_code="8409", response_success=False),
        dict(start, attempt=3),
        dict(reply, attempt=3, response_success=True),
        first_token,
        dict(terminal, ttft_nanos="80"),
    ]
    for sequence, event in enumerate(events, 1):
        event.update(
            sequence=str(sequence),
            epoch_ms=str(1000 + sequence),
            elapsed_nanos=str(sequence * 10),
        )
    write_rows(directory, [header, *events, seal])

    (result,) = read_requests(directory)
    assert [attempt.number for attempt in result.attempts] == [1, 2, 3]
    assert result.attempts[1].error_code == 8409
    assert result.outcome == "success"


def test_header_requires_explicit_first_index(evidence):
    directory, rows = evidence()
    del rows[0]["first_index"]
    write_rows(directory, rows)
    with pytest.raises(
        ExecutionError, match="Missing or unknown evidence header fields"
    ):
        list(read_requests(directory))


@pytest.mark.parametrize(
    "damage", ["missing_seal", "sequence", "duplicate_field", "trailing", "truncated"]
)
def test_invalid_jsonl_is_rejected(evidence, damage):
    directory, rows = evidence()
    rows = [json.dumps(row) for row in rows]
    if damage == "missing_seal":
        rows.pop()
    elif damage == "sequence":
        event = json.loads(rows[1])
        rows[1] = json.dumps({**event, "sequence": "99"})
    elif damage == "duplicate_field":
        rows[1] = rows[1][:-1] + ', "sequence": "1"}'
    elif damage == "trailing":
        rows.append("{}")
    newline = "" if damage == "truncated" else "\n"
    (directory / "requests.jsonl").write_text("\n".join(rows) + newline)
    with pytest.raises(ExecutionError, match=r"requests\.jsonl"):
        list(read_requests(directory))


@pytest.mark.parametrize(
    "damage",
    [
        "duplicate",
        "unfinished",
        "identity",
        "summary",
        "attempt",
        "first_token",
        "unknown_kind",
    ],
)
def test_damaged_evidence_never_passes(evidence, damage):
    directory, rows = evidence()
    if damage == "duplicate":
        rows[2]["kind"] = "ISSUED"
    elif damage == "unfinished":
        del rows[-2]
    elif damage == "identity":
        rows[0]["client"] = "client-7"
    elif damage == "summary":
        (directory / "result.json").write_text('{"done": false, "issued": 2}')
    elif damage == "attempt":
        rows[3]["attempt"] = 2
    elif damage == "first_token":
        del rows[-2]["ttft_nanos"]
    elif damage == "unknown_kind":
        rows[3]["kind"] = "UNKNOWN"
    write_rows(directory, rows)
    with pytest.raises(ExecutionError):
        list(read_requests(directory))


def test_query_defaults_to_all_clients_and_limits_details_not_counts(
    evidence, tmp_path
):
    evidence(index=0, stride=2)
    evidence("client-1", index=1, stride=2, outcome="timeout")
    topology = [
        dict(id=f"client-{i}", role="client", rpc=f"c:{i}", control=f"http://c:{i}")
        for i in range(2)
    ]
    ctx = SimpleNamespace(
        records=SimpleNamespace(directory=tmp_path),
        runtime=SimpleNamespace(topology=SimpleNamespace(active=lambda: topology)),
    )
    query = StepRegistry.standard().get("client/query/requests")
    result = asyncio.run(query.invoke(ctx, {"limit": 1}))
    assert (
        result["matched"] == 2 and result["truncated"] and len(result["records"]) == 1
    )
    assert result["outcomes"]["timeout"] == 1
    result = asyncio.run(
        query.invoke(
            ctx,
            {
                "targets": ["c:1"],
                "outcome": "timeout",
                "since_ms": 1001,
                "until_ms": 1002,
            },
        )
    )
    assert (
        result["matched"] == 1
        and result["records"][0]["request_id"] == 9007199254740994
    )
    assert asyncio.run(query.invoke(ctx, {"request_ids": [1]}))["matched"] == 0
    with pytest.raises(ExecutionError):
        asyncio.run(query.invoke(ctx, {"targets": ["missing"]}))
    path = tmp_path / "components/client-1/playback/1/requests.jsonl"
    path.write_bytes(path.read_bytes()[:-2])
    with pytest.raises(ExecutionError):
        asyncio.run(query.invoke(ctx, {"limit": 0, "request_ids": [1]}))


def test_stop_uses_rpc_summary_while_request_query_requires_files(
    monkeypatch, tmp_path
):
    from mock_framework.execution.step.test.test_client_interfaces import context, state

    ctx = context()
    ctx.records.directory = tmp_path
    ctx.runtime.clients.pending = True

    async def invoke(*args):
        return {"clients": {"http://c:1": state(), "http://c:2": state()}, "errors": []}

    monkeypatch.setattr(client_runtime, "invoke", invoke)
    result = asyncio.run(
        StepRegistry.standard().get("client/stop/playback").invoke(ctx, {})
    )
    assert set(result) == {"c:1", "c:2"} and not ctx.runtime.clients.pending
    assert list(tmp_path.iterdir()) == []
    with pytest.raises(ExecutionError, match="No Client playback"):
        asyncio.run(
            StepRegistry.standard().get("client/query/requests").invoke(ctx, {})
        )


def test_killed_java_writer_leaves_readable_lines_but_no_complete_evidence(tmp_path):
    import os
    import shutil
    import subprocess
    import time

    root = FRAMEWORK
    classpath = os.pathsep.join(
        str(root / path)
        for path in [
            "java/client/target/test-classes",
            "java/client/target/mock-client-1.0-SNAPSHOT-all.jar",
            "java/engine/target/mock-engine-service-1.0-SNAPSHOT-all.jar",
        ]
    )
    java = (
        str(Path(os.environ["JAVA_HOME"]) / "bin/java")
        if os.environ.get("JAVA_HOME")
        else shutil.which("java")
    )
    with (tmp_path / "probe.log").open("wb") as log:
        process = subprocess.Popen(
            [
                java,
                "-Xmx256m",
                "-cp",
                classpath,
                "org.flexlb.mock.client.EvidenceCrashProbe",
                str(tmp_path),
            ],
            stdout=log,
            stderr=log,
        )
        try:
            deadline = time.monotonic() + 15
            while (
                not (tmp_path / "written").exists()
                and process.poll() is None
                and time.monotonic() < deadline
            ):
                time.sleep(0.05)
            assert (tmp_path / "written").exists(), (tmp_path / "probe.log").read_text()
        finally:
            process.kill()
            process.wait(timeout=10)
    assert process.returncode != 0
    directory = tmp_path / "components/client-0/playback/1"
    assert any(
        json.loads(line).get("kind") == "FIRST_TOKEN"
        for line in (directory / "requests.jsonl").read_text().splitlines()
    )
    with pytest.raises(ExecutionError, match="Missing evidence seal"):
        list(read_requests(directory))


def test_history_selection_does_not_overwrite_or_mix_playbacks(evidence, tmp_path):
    first, _ = evidence()
    second, _ = evidence(playback=2, first=9007199254741000)
    topology = [dict(id="client-0", role="client", rpc="c:0", control="http://c:0")]
    assert playback_directories(tmp_path, topology) == [second]
    assert playback_directories(tmp_path, topology, playback=1) == [first]
    records = list(read_requests(first)) + list(read_requests(second))
    assert [record.request_id for record in records] == [
        9007199254740993,
        9007199254741000,
    ]


def test_cancelled_scan_stops_before_reading_next_line(evidence):
    import threading

    directory, _ = evidence()
    cancelled = threading.Event()
    cancelled.set()
    with pytest.raises(ExecutionError, match="scan cancelled"):
        list(read_requests(directory, cancelled))


@pytest.mark.parametrize("seal", [{}, {"issued": "0"}, {"finished": "0"}])
def test_seal_requires_explicit_counts(evidence, seal):
    directory, rows = evidence()
    write_rows(directory, [rows[0], {"seal": seal}])
    with pytest.raises(ExecutionError, match="Evidence seal does not reconcile"):
        list(read_requests(directory))


def test_zero_issued_round_can_seal_without_fabricating_terminal_events(evidence):
    directory, rows = evidence()
    write_rows(directory, [rows[0], {"seal": {"issued": "0", "finished": "0"}}])
    (directory / "result.json").write_text(
        json.dumps(
            dict(
                done=True,
                issued=0,
                finished=0,
                unsent=1,
                outcomes={v: 0 for v in ("success", "error", "timeout", "cancelled")},
            )
        )
    )
    assert list(read_requests(directory)) == []


@pytest.mark.parametrize(
    "replicas,large_input",
    [(2, False), (1, True), (2, True)],
    ids=["small-two", "large-one", "large-two"],
)
def test_client_playback_through_real_master(tmp_path, replicas, large_input):
    from mock_framework.execution.test.test_master import configuration

    root = FRAMEWORK
    java = str(Path(os.environ["JAVA_HOME"]) / "bin/java")
    master_classpath = os.pathsep.join(
        [
            str(root / "java/master/target/classes"),
            (root / "java/master/target/dependencies.txt").read_text().strip(),
        ]
    )
    native = configuration()
    # Match the sender's concurrency so this test measures playback, not DIRECT admission rejection.
    native["FLEXLB_CONFIG"] = json.dumps(
        {
            "scheduler": {"type": "DIRECT"},
            "dispatcher": {"type": "NON_BATCH", "maxInflightPerPrefillWorker": 16},
            "requestLifecycle": {"request": {"timeoutMs": 5000}},
            "grpcServer": {
                "executorCoreSize": 4,
                "executorMaxSize": 8,
                "executorQueueSize": 64,
                "shutdownQuietPeriodMs": 10,
            },
        }
    )
    engine = {
        "engine": "rtp",
        "config": {
            "seq_size_per_block": 16,
            "test_block_num": 4096,
            "reserve_block_ratio": 0,
            "max_seq_len": 4096,
            "mock.compute.forward.expression": "1 + batchSize",
        },
    }
    data = {
        "id": "client_playback",
        "describe": "Client0 coordinates native playback",
        "environment": {
            "master": {"config": native},
            "prefill": engine,
            "decode": engine,
            "client": {"deployment": {"replicas": replicas}, "config": {}},
        },
        "phases": [
            {
                "id": "traffic",
                "describe": "Play and collect",
                "timeout": "35s",
                "steps": [
                    {
                        "id": "start",
                        "describe": "Start global request plan",
                        "interface": "client/start/playback",
                        "args": {
                            "masters": {"ref": "globals.masters"},
                            "requests_file": "requests.mckidx",
                            "qps": 4,
                            "total_concurrency": 4,
                            "queue_capacity": 8,
                            "request_timeout": "5s",
                            "drain_timeout": "5s",
                        },
                        "wait_after": "6s",
                    },
                    {
                        "id": "stop",
                        "describe": "Collect every Client",
                        "interface": "client/stop/playback",
                    },
                ],
            }
        ],
    }
    count = 720 if large_input else 12
    source = tmp_path / "requests.mckidx"
    text_length = 1024 if large_input else 3
    write_inputs(
        source,
        (
            (
                (index + 1) * text_length,
                0,
                {"max_output_tokens": 3},
                50,
                [],
            )
            for index in range(count)
        ),
        chains=[(index * text_length + 1, 0, text_length) for index in range(count)],
    )
    if large_input:
        start = data["phases"][0]["steps"][0]
        start["args"].update(qps=24, total_concurrency=16, queue_capacity=32)
        start["wait_after"] = "24s"
        data["phases"][0]["timeout"] = "60s"
    data["phases"][0]["steps"].append(
        {
            "id": "records",
            "describe": "Read sealed Client evidence",
            "interface": "client/query/requests",
            "args": {"limit": 2},
        }
    )
    path = tmp_path / "case.yaml"
    path.write_text(yaml.safe_dump(data))
    registry = StepRegistry.standard()
    case = compile_case(path, registry)
    components = {
        "master": {
            "command": [
                java,
                "-XX:ActiveProcessorCount=2",
                "-cp",
                master_classpath,
                "org.flexlb.MasterMain",
            ],
            "startup_timeout": "60s",
        },
        "client": {
            "command": [
                java,
                "-jar",
                str(root / "java/client/target/mock-client-1.0-SNAPSHOT-all.jar"),
            ]
        },
    }
    for role in ("prefill", "decode"):
        components[role] = {
            "command": [
                java,
                "-cp",
                str(
                    root / "java/engine/target/mock-engine-service-1.0-SNAPSHOT-all.jar"
                ),
                "org.flexlb.mock.engine.EngineMain",
            ]
        }
    recipe = {
        "components": components,
        "zookeeper": {"command": zookeeper_command(root, java)},
    }
    if os.environ.get("PROMETHEUS_BIN"):
        recipe["prometheus"] = {
            "binary": os.environ["PROMETHEUS_BIN"],
            "scrape_interval": "250ms",
        }
        data["phases"][0]["steps"].insert(
            1,
            {
                "id": "qps",
                "describe": "Check aggregate issued rate",
                "interface": "client/query/qps",
                "args": {"range": ["-10s" if large_input else "-6s", "0s"]},
                "expect": (
                    {"gte": 21.6, "lte": 26.4}
                    if large_input
                    else {"gte": 1.5, "lte": 3}
                ),
                "wait_after": "10s" if large_input else "0s",
            },
        )
        data["phases"][0]["steps"].insert(
            2,
            {
                "id": "success",
                "describe": "Check aggregate terminal success rate",
                "interface": "client/query/success_rate",
                "args": {"range": ["-20s", "0s"]},
                "expect": {"eq": 1.0},
            },
        )
        path.write_text(yaml.safe_dump(data))
        case = compile_case(path, registry)
    elif large_input:
        data["phases"][0]["steps"][0]["wait_after"] = "34s"
        path.write_text(yaml.safe_dump(data))
        case = compile_case(path, registry)
    launch = prepare_launch(recipe, case, tmp_path)
    output = tmp_path / "result"
    result = asyncio.run(run_case(case, launch, output))
    assert result["status"] == "PASS", result
    configuration = json.loads((output / "configuration.json").read_text())
    assert configuration["case"]["environment"] == case.environment
    for role, component in case.environment.items():
        for index in range(component["deployment"]["replicas"]):
            instance_id = f"{role}-{index}"
            path = output / "components" / instance_id / "config.json"
            assert json.loads(path.read_text()) == {
                "component_role": role,
                "component": component,
                "id": instance_id,
            }
    if "prometheus" in recipe:
        gates = [
            json.loads(line)
            for line in (output / "gates.jsonl").read_text().splitlines()
        ]
        (success,) = [
            gate for gate in gates if gate["interface"] == "client/query/success_rate"
        ]
        assert success["actual"] == 1.0
    summary = json.loads(
        (output / "components/client-0/coordinator/1/result.json").read_text()
    )
    assert not summary["errors"] and len(summary["clients"]) == replicas
    assert sum(v["issued"] for v in summary["clients"].values()) == count
    assert all(
        v["finished"] == count // replicas
        and v["outcomes"]["success"] == count // replicas
        and v["unsent"] == 0
        and not v["stopped"]
        for v in summary["clients"].values()
    )
    pids = [
        json.loads(line)["pid"]
        for line in (output / "events.jsonl").read_text().splitlines()
        if json.loads(line).get("event") == "process_started"
        and json.loads(line).get("node", "").startswith("client-")
    ]
    assert len(set(pids)) == replicas
    manifest = json.loads(
        (output / "components/client-0/coordinator/1/input.json").read_text()
    )
    assert manifest["count"] == count and manifest["path"] == str(source)
    assert not (output / "components/client-0/coordinator/1/shards").exists()
    for i in range(replicas):
        plan = json.loads(
            (output / f"components/client-{i}/playback/1/plan.json").read_text()
        )
        assert (
            plan["requests_file"] == str(source)
            and plan["sha256"] == manifest["sha256"]
        )
        assert (plan["first_index"], plan["stride"]) == (i, replicas)
    assert not (output / "topology-history.json").exists()
    topology = [
        json.loads(path.read_text())
        for path in sorted(output.glob("components/client-*/endpoints.json"))
    ]
    records = [
        record
        for directory in playback_directories(output, topology)
        for record in read_requests(directory)
    ]
    assert len(records) == count and {record.request_id for record in records} == set(
        range(1, count + 1)
    )
    assert all(
        record.outcome == "success"
        and record.ttft_nanos is not None
        and record.output_tokens == 3
        for record in records
    )
    assert all(
        len(record.attempts) == 1
        and record.attempts[0].grpc_code == "OK"
        and record.attempts[0].response_success is True
        for record in records
    )
    (queried,) = [
        event["result"]
        for line in (output / "events.jsonl").read_text().splitlines()
        if (event := json.loads(line)).get("event") == "step_finished"
        and event.get("step_id") == "records"
    ]
    assert queried["matched"] == count and queried["outcomes"]["success"] == count
    assert len(queried["records"]) == 2 and queried["truncated"]
