"""Performance framework behavior and failure regressions."""

from unittest.mock import Mock

from rtp_llm.test.perf_test.config_resolver import resolve_config
from rtp_llm.test.perf_test.config_schema import parse_config
from rtp_llm.test.perf_test.decode.model import (
    DecodeBatchMeasurement,
    DecodeCaseSpec,
    ExactBatchPolicy,
)
from rtp_llm.test.perf_test.decode.planner import plan_decode_cases
from rtp_llm.test.perf_test.errors import EngineLifecycleError
from rtp_llm.test.perf_test.execution.engine_session import EngineDiagnostics
from rtp_llm.test.perf_test.plan import build_execution_plan
from rtp_llm.test.perf_test.tests.support.fixtures import (
    fixture_config,
    fixture_engine,
    fixture_model,
    fixture_workload,
)
from rtp_llm.test.perf_test.tests.support.results import (
    default_executor_request_round as default_executor_request_round,
)


class engine_session_FakeHandle:

    def __init__(self, *, ready=True):
        self.ready = ready
        self.alive = True
        self.exit_code = None
        self.client = object()
        self.log_path = "/tmp/process.log"
        self.wait_ready_calls = []
        self.stop_calls = 0

    def wait_ready(self, timeout_seconds):
        self.wait_ready_calls.append(timeout_seconds)
        return self.ready

    def is_alive(self):
        return self.alive

    def stop(self):
        self.stop_calls += 1
        self.alive = False


class default_executor_Tokenizer:

    def encode(self, text):
        return text.split()


class default_executor_Client:
    port = 18000

    def configure_scheduler(self, batch_size, mode="prefill"):
        assert batch_size == 1
        assert mode == "prefill"

    def request(self, **_request):
        return {
            "success": True,
            "input_len": len(_request["input_ids"]),
            "output_len": 1,
            "reuse_len": (
                4
                if _request["case_id"].startswith("cache_probe_")
                and _request["phase"] == "measure"
                else 0
            ),
            "latency_ms": 10.0,
        }


class default_executor_Session:

    def __init__(self):
        self.client = default_executor_Client()
        self.entered = 0
        self.exited = 0
        self.diagnostics = Mock()

    def __enter__(self):
        self.entered += 1
        return self

    def __exit__(self, _type, _value, _traceback):
        self.exited += 1


class default_executor_FailingSession:
    diagnostics = EngineDiagnostics(log_path="failed-engine.log", exit_code=2)

    def __enter__(self):
        raise EngineLifecycleError("engine failed readiness check")

    def __exit__(self, _type, _value, _traceback):
        return None


def default_executor_source(result_root):
    return fixture_config(
        model=fixture_model(label="model"),
        engine=fixture_engine(max_seq_len=64, max_batch_tokens=64, concurrency_limit=1),
        workload=fixture_workload(
            max_input_len=16,
            cache_alignment=4,
            commit_tail_tokens=4,
            compute_steps=[4],
            cache_lengths=[0],
            measure_runs=1,
        ),
        output={"result_root": str(result_root)},
    )


def default_executor_decode_source_and_plan(root, workload=None):
    source = default_executor_source(root.parent)
    source["model"]["topology"] = {"tp": 1, "dp": 1, "pp": 1, "ep": 1}
    source["workload"] = workload or {
        "kind": "grid",
        "input_lengths": [32],
        "output_lengths": [8],
        "batch": {"strategy": "fixed", "sizes": [1]},
        "measure_runs": 1,
        "warmup_runs": 0,
        "profiling": {"enabled": False},
    }
    resolved = resolve_config(source)
    parsed = parse_config(source)
    cases = plan_decode_cases(
        parsed.workload,
        parsed.engine,
        dp_size=1,
        resolved_workload=resolved.resolved_config["workload"],
    )
    return (
        source,
        build_execution_plan(
            mode="decode",
            action="run",
            resolved=resolved,
            cases=cases,
            result_directory=str(root),
        ),
    )


def default_executor_measurement(
    batch, inputs, output, *, runs=1, throughput=100.0, latency=None
):
    return DecodeBatchMeasurement(
        batch_size=batch,
        input_lengths=inputs,
        worker_allocations=(),
        measurements=tuple(
            {
                "run": run,
                "throughput_tokens_per_second": throughput,
                "worker_allocations": [],
                "requests": list(
                    default_executor_request_round(inputs, output, latency=latency)
                ),
            }
            for run in range(runs)
        ),
    )


def local_executor_config(mode):
    workload = {
        "kind": "cache_grid",
        "max_input_len": 2048,
        "cache_alignment": 64,
        "commit_tail_tokens": 64,
        "compute_steps": [128],
        "batch_sizes": [1],
    }
    if mode == "decode":
        workload = {
            "kind": "grid",
            "input_lengths": [128],
            "output_lengths": [16],
            "batch": {"strategy": "fixed", "sizes": [1]},
        }
    return fixture_config(workload=workload)


def local_executor_plan(mode):
    identity = {"input_len": 512, "cache_len": 0, "batch_size": 1}
    if mode == "decode":
        identity = DecodeCaseSpec(
            workload_kind="grid",
            input_lengths=(128,),
            output_length=16,
            batch_policy=ExactBatchPolicy("fixed", 1),
            measure_runs=1,
            warmup_runs=0,
        ).to_identity()
    return build_execution_plan(
        mode=mode,
        action="run",
        resolved=resolve_config(local_executor_config(mode)),
        cases=[identity],
        result_directory=f"runs/{mode}",
    )


class local_executor_FakeSession:

    def __init__(self, enter_error=None):
        self.enter_error = enter_error
        self.enter_calls = 0
        self.exit_calls = 0
        self.diagnostics = EngineDiagnostics(log_path="process.log")

    def __enter__(self):
        self.enter_calls += 1
        if self.enter_error is not None:
            raise self.enter_error
        return self

    def __exit__(self, _type, _value, _traceback):
        self.exit_calls += 1


def runtime_adapter_plan(backend="local", transport="http_text", topology=None):
    if topology is None:
        topology = {"tp": 2, "dp": 1, "pp": 1, "ep": 2}
    world_size = topology.get("tp", 1) * topology.get("dp", 1) * topology.get("pp", 1)
    source = fixture_config(
        model=fixture_model(
            label="model",
            type="qwen_2",
            checkpoint="/models/qwen",
            tokenizer="/models/qwen-tokenizer",
            topology=topology,
        ),
        engine=fixture_engine(
            concurrency_limit=4,
            args={"reuse_cache": 1, "enable_cuda_graph": 0},
            env={"TOKENIZERS_PARALLELISM": "false"},
        ),
        workload=fixture_workload(
            transport=transport,
            max_input_len=1024,
            compute_steps=[256],
            cache_lengths=[0],
        ),
        execution=(
            {
                "backend": "local",
                "devices": [3, 5] if world_size == 2 else list(range(world_size)),
            }
            if backend == "local"
            else {
                "backend": "aone",
                "resource": "daily",
                "gpu_type": "H20_CU13",
                "max_parallel": 1,
                "shard_count": 1,
            }
        ),
        output={"result_root": "/tmp/results"},
    )
    return build_execution_plan(
        mode="prefill",
        action="run",
        resolved=resolve_config(source),
        cases=[
            {"input_len": 1024, "cache_len": 0, "compute_len": 1024, "batch_size": 1}
        ],
        result_directory="/tmp/results/run",
    )


class decode_transport_Response:

    def __init__(self, payload=None, *, status_code=200, text="ok"):
        self.status_code = status_code
        self.text = text
        self._payload = payload or {}

    def json(self):
        return self._payload
