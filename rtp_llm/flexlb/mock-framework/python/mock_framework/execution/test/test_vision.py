from mock_framework.execution.test.helpers import FRAMEWORK
from mock_framework.execution.test.helpers.zookeeper import command as zookeeper_command

"""Real Master plus independent V/P/D processes, using the standard binary playback path."""
import asyncio
import json
import os
from pathlib import Path

import pytest
import yaml
from mock_framework.config import compile_case
from mock_framework.execution.launch import prepare_launch
from mock_framework.execution.runner import run_case
from mock_framework.execution.step.step_registry import StepRegistry
from mock_framework.execution.test.helpers.input import pb, write_inputs
from test_master import configuration


def test_vision_playback_through_real_master(tmp_path):
    root = FRAMEWORK
    java = str(Path(os.environ["JAVA_HOME"]) / "bin/java")
    master_classpath = os.pathsep.join(
        [
            str(root / "java/master/target/classes"),
            (root / "java/master/target/dependencies.txt").read_text().strip(),
        ]
    )
    native = configuration()
    route = json.loads(native["MODEL_SERVICE_CONFIG"])
    route["role_endpoints"][0]["vit_endpoint"] = {
        "address": "v-domain",
        "protocol": "grpc",
    }
    native["MODEL_SERVICE_CONFIG"] = json.dumps(route)
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
            "image_token_id": 1000,
            "seq_size_per_block": 2,
            "test_block_num": 128,
            "reserve_block_ratio": 0,
            "max_seq_len": 128,
            "mock.compute.forward.expression": "1",
        },
    }
    vision = {
        "config": {
            "patch_size": 14,
            "downsample_ratio": 3,
            "min_pixels": 0,
            "max_image_tokens": 1024,
            "hidden_size": 4,
            "mock.vit.skip.download": True,
            "mm_cache_gpu_max_bytes": 1024,
            "mm_cache_cpu_max_bytes": 1024,
            "mock.compute.forward.expression": "2 + batchSize",
        }
    }
    data = {
        "id": "vision_playback",
        "describe": "Native V selection, metadata preparation, P/D generation",
        "environment": {
            "master": {"config": native},
            "prefill": engine,
            "decode": engine,
            "vit": vision,
            "client": {"config": {"image_token_id": 1000}},
        },
        "phases": [
            {
                "id": "traffic",
                "describe": "Play media requests",
                "timeout": "30s",
                "steps": [
                    {
                        "id": "start",
                        "describe": "Play image input",
                        "interface": "client/start",
                        "args": {
                            "requests_file": "requests.binpb.zst",
                            "qps": 4,
                            "concurrency": 4,
                            "queue_capacity": 8,
                            "request_timeout": "5s",
                            "drain_timeout": "5s",
                        },
                        "wait_after": "6s",
                    },
                    {
                        "id": "stop",
                        "describe": "Collect playback result",
                        "interface": "client/stop",
                    },
                ],
            }
        ],
    }
    (tmp_path / "configs").mkdir()
    for role in ("master", "prefill", "decode", "vit"):
        component = data["environment"][role]
        (tmp_path / "configs" / f"{role}.yaml").write_text(
            yaml.safe_dump(component["config"])
        )
        data["environment"][role] = {**component, "config": f"configs/{role}.yaml"}
    requests = [
        pb.Request(
            token_ids=[10, 0, 20],
            max_new_tokens=3,
            images=[
                pb.Image(
                    url="mock://image-" + str(i % 2), width=56, height=28, offset=1
                )
            ],
        )
        for i in range(8)
    ]
    write_inputs(tmp_path / "requests.binpb.zst", requests)
    path = tmp_path / "case.yaml"
    path.write_text(yaml.safe_dump(data))
    registry = StepRegistry.standard()
    case = compile_case(path, registry)
    host = root / "java/engine/target/mock-engine-service-1.0-SNAPSHOT-all.jar"
    components = {
        "master": {
            "command": [
                java,
                "-XX:ActiveProcessorCount=2",
                "-cp",
                master_classpath,
                "org.flexlb.MasterNode",
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
    for role in ("prefill", "decode", "vit"):
        components[role] = {
            "command": [
                java,
                "-XX:ActiveProcessorCount=2",
                "-cp",
                str(host),
                "org.flexlb.mock.engine.EngineMain",
            ],
            "env": {"MOCK_BIND_HOST": "0.0.0.0", "MOCK_ADVERTISE_HOST": "127.0.0.1"},
        }
    launch = prepare_launch(
        {
            "components": components,
            "zookeeper": {"command": zookeeper_command(root, java)},
        },
        case,
        tmp_path,
    )
    output = tmp_path / "result"
    result = asyncio.run(run_case(case, registry, launch, output))
    assert result["status"] == "PASS", result
    summary = json.loads(
        (output / "components/client-0/coordinator/1/result.json").read_text()
    )
    assert not summary["errors"], summary
    (client,) = summary["clients"].values()
    assert (
        client["issued"] == client["finished"] == client["outcomes"]["success"] == 8
    ), client
    events = [
        json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()
    ]
    engines = [
        event
        for event in events
        if event.get("event") == "process_started"
        and event.get("node") in {"prefill-0", "decode-0", "vit-0"}
    ]
    assert len({event["pid"] for event in engines}) == 3


def test_vit_is_a_shared_engine_without_backend_configuration():
    from mock_framework.config import ConfigError
    from mock_framework.config.compiler import compile_environment

    vision = {"config": {"mock.compute.forward.expression": "1"}}
    assert "engine" not in compile_environment({"vit": vision})["vit"]
    with pytest.raises(ConfigError):
        compile_environment({"vit": dict(vision, engine="rtp")})
