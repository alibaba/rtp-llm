import asyncio
import json
import os
from pathlib import Path

import yaml
from mock_framework.config import compile_case
from mock_framework.execution.launch import prepare_launch
from mock_framework.execution.runner import run_case
from mock_framework.execution.step.step_registry import StepRegistry
from mock_framework.execution.test.helpers import FRAMEWORK
from mock_framework.execution.test.helpers.input import write_inputs
from mock_framework.execution.test.helpers.zookeeper import command as zookeeper_command


def test_client_playback_through_real_master(tmp_path):
    from mock_framework.execution.test.test_master import configuration

    root = FRAMEWORK
    replicas, count = 2, 12
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
    source = tmp_path / "requests.mckidx"
    write_inputs(
        source,
        (
            (
                (index + 1) * 3,
                0,
                {"max_output_tokens": 3},
                50,
                [],
            )
            for index in range(count)
        ),
        chains=[(index * 3 + 1, 0, 3) for index in range(count)],
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
    summary = json.loads(
        (output / "components/client-0/coordinator/1/result.json").read_text()
    )
    assert not summary["errors"] and len(summary["clients"]) == replicas
    assert sum(v["issued"] for v in summary["clients"].values()) == count
    assert all(
        v["finished"] == count // replicas
        and v["outcomes"]["success"] == count // replicas
        and v["unissued"] == 0
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
    coordinator = output / "components/client-0/coordinator/1"
    coordinator_plan = json.loads((coordinator / "plan.json").read_text())
    assert coordinator_plan["count"] == count
    assert coordinator_plan["config"]["requests_file"] == str(source)
    assert not (coordinator / "input.json").exists()
    assert not (coordinator / "shards").exists()
    for i in range(replicas):
        plan = json.loads(
            (output / f"components/client-{i}/playback/1/plan.json").read_text()
        )
        assert (
            plan["requests_file"] == str(source)
            and plan["sha256"] == coordinator_plan["sha256"]
        )
        assert (plan["first_index"], plan["stride"]) == (i, replicas)
    assert not (output / "topology-history.json").exists()
