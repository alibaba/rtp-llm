import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from rtp_llm.test.perf_test.aone.config_schema import parse_execution
from rtp_llm.test.perf_test.aone.package import generate_aone_package
from rtp_llm.test.perf_test.aone.plan import plan_resources
from rtp_llm.test.perf_test.aone.worker import RemoteShardExecutor, load_remote_inputs
from rtp_llm.test.perf_test.aone.worker import main as worker_main
from rtp_llm.test.perf_test.capabilities import validate_aone_capability
from rtp_llm.test.perf_test.config_resolver import resolve_config
from rtp_llm.test.perf_test.execution.engine_session import EngineDiagnostics
from rtp_llm.test.perf_test.modes import PrefillAdapter
from rtp_llm.test.perf_test.plan import build_execution_plan
from rtp_llm.test.perf_test.state import RunState


def prefill_source(
    *, model=None, engine=None, workload=None, execution=None, output=None
):
    """Fresh minimal remote prefill config; section overrides are explicit per case."""
    source = {
        "schema_version": 1,
        "model": {"label": "test", "type": "registered", "checkpoint": "/m"},
        "engine": {
            "max_seq_len": 4096,
            "max_batch_tokens": 8192,
            "concurrency_limit": 2,
        },
        "workload": {
            "kind": "cache_grid",
            "max_input_len": 2048,
            "cache_alignment": 64,
            "commit_tail_tokens": 64,
            "compute_steps": [256],
            "batch_sizes": [1],
        },
        "execution": {
            "backend": "aone",
            "resource": "daily",
            "gpu_type": "H20",
            "max_parallel": 2,
            "shard_count": 2,
        },
        "output": {},
        "regression": {},
        "publish": {},
    }
    for section, overrides in (
        ("model", model),
        ("engine", engine),
        ("workload", workload),
        ("execution", execution),
        ("output", output),
    ):
        if overrides is not None:
            source[section].update(overrides)
    return source


def _worker_plan(root):
    config = prefill_source(
        engine={"max_seq_len": 4097, "concurrency_limit": 1},
        workload={"cache_lengths": [0, 512]},
        execution={"max_parallel": 1, "shard_count": 1},
    )
    cases = [
        {
            "input_len": 512,
            "cache_len": 0,
            "compute_len": 512,
            "batch_size": 1,
            "measure_runs": 1,
            "seed": 1,
            "prefix_policy": "shared_seed",
        },
        {
            "input_len": 1024,
            "cache_len": 512,
            "compute_len": 512,
            "batch_size": 1,
            "measure_runs": 1,
            "seed": 1,
            "prefix_policy": "shared_seed",
        },
    ]
    return build_execution_plan(
        resource_planner=plan_resources,
        capability_validator=validate_aone_capability,
        mode="prefill",
        action="run",
        resolved=resolve_config(config, execution_parser=parse_execution),
        cases=cases,
        result_directory=str(root / "formal-results"),
    )


class _FakeSession:

    def __init__(self):
        self.enter_calls = 0
        self.exit_calls = 0
        self.diagnostics = EngineDiagnostics(log_path="remote-engine.log")

    def __enter__(self):
        self.enter_calls += 1
        return self

    def __exit__(self, _type, _value, _traceback):
        self.exit_calls += 1


def _worker_result(case):
    return {
        "case_id": case["case_id"],
        "identity": {key: value for (key, value) in case.items() if key != "case_id"},
        "status": "ok",
        "runs": 1,
        "errors": [],
        "metrics": {
            "input_length": case["input_len"],
            "expected_reuse_length": case["cache_len"],
            "observed_reuse_length": case["cache_len"],
            "compute_length": case["compute_len"],
            "latency_ms": 10.0,
            "throughput_tokens_per_second": 1000.0,
            "validation": {"shape": True, "reuse": True, "timing": True},
            "seed": 1,
            "prefix_policy": {"kind": "shared_seed"},
            "measurements": [{"run": 0, "latency_ms": 10.0}],
        },
    }


class AOneWorkerTest(unittest.TestCase):

    def test_remote_prefill_runner_uses_physical_cache_block_size(self):
        from rtp_llm.test.perf_test.aone.worker import _default_executor

        config = {
            "model": {"tokenizer": "test", "checkpoint": "test", "type": "test"},
            "engine": {"max_batch_tokens": 4096, "args": {"seq_size_per_block": 16}},
            "workload": {"cache_alignment": 64, "commit_tail_tokens": 64},
        }
        inputs = SimpleNamespace(config=config)
        with patch(
            "rtp_llm.test.perf_test.aone.worker.resolve_remote_runtime_inputs",
            return_value=inputs,
        ), patch(
            "rtp_llm.test.perf_test.test_util._load_tokenizer",
            return_value=SimpleNamespace(encode=lambda text: text.split()),
        ), patch(
            "rtp_llm.test.perf_test.aone.worker.RemoteShardExecutor"
        ) as executor:
            _default_executor(inputs)
            adapter = executor.call_args.kwargs["adapter"]
            runner = adapter._runner_factory(
                SimpleNamespace(session=SimpleNamespace(client=Mock()))
            )
        self.assertEqual(16, runner.cache_block_size)

    def test_worker_setup_failure_still_writes_an_owned_failure_result(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            plan = _worker_plan(root)
            package = generate_aone_package(
                root / "generated",
                plan,
                run_id="run-setup-failure",
                ownership=[[case.case_id for case in plan.cases]],
            )
            output = root / "remote-output"
            exit_code = worker_main(
                [
                    f"--resolved-config={package.resolved_config}",
                    f"--resolved-workload={package.resolved_workload}",
                    f"--shard-plan={package.shard_plan}",
                    "--shard-id=shard-0000",
                    f"--output-dir={output}",
                ],
                executor_factory=Mock(side_effect=RuntimeError("tokenizer missing")),
            )
            self.assertEqual(5, exit_code)
            result = json.loads((output / "shard_result.json").read_text())
            self.assertEqual("failed", result["status"])
            self.assertEqual("run-setup-failure", result["run_id"])
            self.assertEqual("shard-0000", result["shard_id"])
            self.assertEqual(
                [case.case_id for case in plan.cases], result["owned_case_ids"]
            )
            self.assertEqual([], result["completed_case_ids"])
            self.assertEqual([], result["cases"])
            self.assertEqual("RuntimeError", result["failure"]["type"])
            self.assertEqual("tokenizer missing", result["failure"]["message"])

    def test_multi_case_shard_uses_one_session_and_returns_atomic_claims(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            plan = _worker_plan(root)
            package = generate_aone_package(
                root / "generated",
                plan,
                run_id="run-remote",
                ownership=[[case.case_id for case in plan.cases]],
            )
            session = _FakeSession()
            session_factory = Mock(return_value=session)
            runner = Mock()
            runner.run_case.side_effect = _worker_result
            executor = RemoteShardExecutor(
                session_factory=session_factory,
                adapter=PrefillAdapter(lambda _context: runner),
            )
            output = root / "remote-output"
            factory = Mock(return_value=executor)
            code = worker_main(
                [
                    f"--resolved-config={package.resolved_config}",
                    f"--resolved-workload={package.resolved_workload}",
                    f"--shard-plan={package.shard_plan}",
                    "--shard-id=shard-0000",
                    f"--output-dir={output}",
                ],
                executor_factory=factory,
            )
            self.assertEqual(0, code)
            inputs = factory.call_args.args[0]
            result = SimpleNamespace(
                **json.loads((output / "shard_result.json").read_text())
            )
            self.assertEqual(RunState.COMPLETED, RunState(result.status))
            self.assertEqual("run-remote", result.run_id)
            self.assertEqual("shard-0000", result.shard_id)
            self.assertEqual(plan.config_fingerprint, result.config_fingerprint)
            self.assertEqual(plan.workload_fingerprint, result.workload_fingerprint)
            self.assertEqual(inputs.owned_case_ids, tuple(result.owned_case_ids))
            self.assertEqual(inputs.owned_case_ids, tuple(result.completed_case_ids))
            session_factory.assert_called_once()
            self.assertEqual(1, session.enter_calls)
            self.assertEqual(1, session.exit_calls)
            self.assertEqual(2, runner.run_case.call_count)
            self.assertEqual(
                2, len((output / "results/cases.jsonl").read_text().splitlines())
            )
            self.assertFalse((output / "results/cases").exists())
            persisted = json.loads((output / "shard_result.json").read_text())
            self.assertEqual(
                list(inputs.owned_case_ids), persisted["completed_case_ids"]
            )

    def test_worker_load_rejects_modified_frozen_runfiles(self):
        from rtp_llm.test.perf_test.errors import ConfigurationError

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = _worker_plan(root)
            package = generate_aone_package(
                root / "generated",
                plan,
                run_id="worker",
                ownership=[[case.case_id for case in plan.cases]],
            )
            package.resolved_workload.write_text("{}")
            with self.assertRaisesRegex(ConfigurationError, "fingerprint"):
                load_remote_inputs(
                    resolved_config=package.resolved_config,
                    resolved_workload=package.resolved_workload,
                    shard_plan=package.shard_plan,
                    shard_id="shard-0000",
                )

    def test_worker_uses_bazel_undeclared_output_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = _worker_plan(root)
            package = generate_aone_package(
                root / "generated",
                plan,
                run_id="worker",
                ownership=[[case.case_id for case in plan.cases]],
            )
            output = root / "undeclared"
            with patch.dict("os.environ", {"TEST_UNDECLARED_OUTPUTS_DIR": str(output)}):
                code = worker_main(
                    [
                        f"--resolved-config={package.resolved_config}",
                        f"--resolved-workload={package.resolved_workload}",
                        f"--shard-plan={package.shard_plan}",
                        "--shard-id=shard-0000",
                    ],
                    executor_factory=Mock(side_effect=RuntimeError("setup failed")),
                )
            self.assertEqual(5, code)
            result = json.loads(
                (output / "perf_test_shard-0000/shard_result.json").read_text()
            )
            self.assertEqual("worker", result["run_id"])
            self.assertEqual(plan.config_fingerprint, result["config_fingerprint"])

    def test_partial_worker_preserves_successful_case_and_failure_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = _worker_plan(root)
            package = generate_aone_package(
                root / "generated",
                plan,
                run_id="partial-worker",
                ownership=[[case.case_id for case in plan.cases]],
            )
            inputs = load_remote_inputs(
                resolved_config=package.resolved_config,
                resolved_workload=package.resolved_workload,
                shard_plan=package.shard_plan,
                shard_id="shard-0000",
            )
            calls = []

            class Runner:
                def run_case(self, case):
                    calls.append(case["case_id"])
                    if len(calls) == 2:
                        raise RuntimeError("request failed")
                    return _worker_result(case)

            result = RemoteShardExecutor(
                session_factory=lambda _plan: _FakeSession(),
                adapter=PrefillAdapter(lambda _context: Runner()),
            ).execute(inputs, root / "output")
            self.assertEqual(RunState.PARTIAL, result.state)
            self.assertEqual(5, result.exit_code)
            self.assertEqual(inputs.owned_case_ids, result.completed_case_ids)
            self.assertEqual(
                ["ok", "request_failed"], [case["status"] for case in result.cases]
            )
            self.assertEqual("CaseExecutionError", result.failure["type"])


if __name__ == "__main__":
    unittest.main()
