"""Core performance behavior contracts."""

import json
import signal
import struct
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from rtp_llm.test.perf_test.config_resolver import resolve_config
from rtp_llm.test.perf_test.config_schema import parse_config
from rtp_llm.test.perf_test.decode.model import (
    DecodeBatchMeasurement,
    DecodeCaseSpec,
    ExactBatchPolicy,
)
from rtp_llm.test.perf_test.decode.planner import plan_decode_cases
from rtp_llm.test.perf_test.errors import EngineLifecycleError
from rtp_llm.test.perf_test.execution.engine_session import (
    EngineDiagnostics,
    EngineInterrupted,
    EngineSession,
)
from rtp_llm.test.perf_test.execution.local import LocalExecutor, UnifiedLocalExecutor
from rtp_llm.test.perf_test.modes import ModeExecutionResult
from rtp_llm.test.perf_test.plan import build_execution_plan
from rtp_llm.test.perf_test.run_directory import initialize_run_directory
from rtp_llm.test.perf_test.state import RunState
from rtp_llm.test.perf_test.test_support import (
    fixture_engine,
    fixture_execution,
    fixture_model,
    fixture_workload,
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
    return {
        "schema_version": 1,
        "model": fixture_model(label="model"),
        "engine": fixture_engine(
            max_seq_len=64, max_batch_tokens=64, concurrency_limit=1
        ),
        "workload": fixture_workload(
            max_input_len=16,
            cache_alignment=4,
            commit_tail_tokens=4,
            compute_steps=[4],
            cache_lengths=[0],
            measure_runs=1,
        ),
        "execution": fixture_execution(),
        "output": {"result_root": str(result_root)},
        "regression": {},
        "publish": {},
    }


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


def default_executor_request_round(input_lengths, output_length):
    return tuple(
        (
            {
                "request_index": index,
                "success": True,
                "input_length": input_length,
                "output_length": output_length,
                "latency_ms": {
                    "wait": 1.0,
                    "prefill": 2.0,
                    "decode": 15.0,
                    "tpot": 1.0,
                    "total": 17.0,
                },
            }
            for (index, input_length) in enumerate(input_lengths)
        )
    )


def default_executor_measurement(batch, inputs, output):
    return DecodeBatchMeasurement(
        batch_size=batch,
        input_lengths=inputs,
        worker_allocations=(),
        measurements=(
            {
                "run": 0,
                "throughput_tokens_per_second": 100.0,
                "worker_allocations": [],
                "requests": list(default_executor_request_round(inputs, output)),
            },
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
    return {
        "schema_version": 1,
        "model": fixture_model(),
        "engine": fixture_engine(),
        "workload": workload,
        "execution": fixture_execution(),
        "output": {},
        "regression": {},
        "publish": {},
    }


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


class EngineSessionTest(unittest.TestCase):

    def _session(self, handle, **kwargs):
        launcher = Mock(return_value=handle)
        timelines = Mock(return_value=["traces/timeline.json"])
        coredumps = Mock(return_value={"count": 1, "files": ["core.123"]})
        session = EngineSession(
            launcher=launcher,
            readiness_timeout_seconds=30,
            timeline_collector=timelines,
            coredump_collector=coredumps,
            install_signal_handlers=False,
            **kwargs,
        )
        return (session, launcher, timelines, coredumps)

    def test_success_exposes_client_and_collects_diagnostics_on_teardown(self):
        handle = engine_session_FakeHandle()
        (session, launcher, timelines, coredumps) = self._session(handle)
        with session as active:
            self.assertIs(handle.client, active.client)
            self.assertEqual("/tmp/process.log", active.log_path)
            active.assert_healthy()
        launcher.assert_called_once_with()
        self.assertEqual([30], handle.wait_ready_calls)
        self.assertEqual(1, handle.stop_calls)
        timelines.assert_called_once_with()
        coredumps.assert_called_once_with()
        self.assertEqual(("traces/timeline.json",), session.diagnostics.timeline_files)
        self.assertEqual(1, session.diagnostics.coredumps["count"])

    def test_startup_failure_stops_and_preserves_diagnostics(self):
        handle = engine_session_FakeHandle(ready=False)
        handle.exit_code = 17
        (session, launcher, timelines, coredumps) = self._session(handle)
        with self.assertRaisesRegex(EngineLifecycleError, "failed readiness") as error:
            session.start()
        launcher.assert_called_once_with()
        self.assertEqual(1, handle.stop_calls)
        timelines.assert_called_once_with()
        coredumps.assert_called_once_with()
        self.assertEqual(17, error.exception.diagnostics.exit_code)
        session.close()
        self.assertEqual(1, handle.stop_calls)

    def test_interrupt_closes_session_and_carries_signal(self):
        handle = engine_session_FakeHandle()
        (session, _, _, _) = self._session(handle)
        session.start()
        with self.assertRaises(EngineInterrupted) as error:
            session.handle_signal(signal.SIGINT)
        self.assertEqual(signal.SIGINT, error.exception.signal_number)
        self.assertEqual(1, handle.stop_calls)
        session.close()
        self.assertEqual(1, handle.stop_calls)

    def test_multiple_cases_share_exactly_one_startup(self):
        handle = engine_session_FakeHandle()
        (session, launcher, _, _) = self._session(handle)
        with session:
            session.start()
            for _case in range(5):
                self.assertIs(handle.client, session.client)
                session.assert_healthy()
        launcher.assert_called_once_with()
        self.assertEqual([30], handle.wait_ready_calls)
        self.assertEqual(1, handle.stop_calls)


class DefaultExecutorTest(unittest.TestCase):

    def test_decode_run_uses_one_session_and_unified_case_executor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            (source, plan) = default_executor_decode_source_and_plan(root)
            initialize_run_directory(
                root,
                source_config=source,
                plan=plan,
                run_id="decode-local",
                code_version="test",
            )
            session = default_executor_Session()
            session.log_path = None
            batch_executor = Mock()
            batch_executor.measure.return_value = default_executor_measurement(
                batch=1, inputs=(32,), output=8
            )
            executor = UnifiedLocalExecutor(
                session_factory=Mock(return_value=session),
                tokenizer_loader=lambda _plan: default_executor_Tokenizer(),
                query_builder=lambda _plan, lengths: {
                    length: f"prompt-{length}" for length in lengths
                },
                decode_batch_executor_factory=Mock(return_value=batch_executor),
            )
            outcome = executor(plan)
            self.assertEqual(RunState.COMPLETED, outcome.state)
            self.assertEqual((1, 1), (session.entered, session.exited))
            self.assertEqual("ok", outcome.case_results[0]["status"])
            self.assertEqual(8, outcome.case_results[0]["metrics"]["output_length"])
            self.assertTrue((root / "reports" / "decode_report.json").is_file())

    def test_prefill_run_uses_one_session_and_finalizes_all_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            source = default_executor_source(root.parent)
            plan = build_execution_plan(
                mode="prefill",
                action="run",
                resolved=resolve_config(source),
                cases=[
                    {
                        "input_len": 16,
                        "cache_len": 0,
                        "compute_len": 16,
                        "batch_size": 1,
                        "measure_runs": 1,
                        "seed": 1,
                        "prefix_policy": "shared_seed",
                    }
                ],
                result_directory=str(root),
            )
            initialize_run_directory(
                root,
                source_config=source,
                plan=plan,
                run_id="run-local",
                code_version="test",
            )
            session = default_executor_Session()
            executor = UnifiedLocalExecutor(
                session_factory=Mock(return_value=session),
                tokenizer_loader=lambda _plan: default_executor_Tokenizer(),
                query_builder=lambda _plan, _lengths: {},
            )
            outcome = executor(plan)
            self.assertEqual(RunState.COMPLETED, outcome.state)
            self.assertEqual(1, session.entered)
            self.assertEqual(1, session.exited)
            self.assertTrue(
                (root / "checkpoints" / "prefill_cache_probe.json").is_file()
            )
            self.assertTrue((root / "results" / "result.json").is_file())
            self.assertTrue((root / "results" / "summary.json").is_file())
            self.assertTrue((root / "reports" / "prefill_report.json").is_file())
            self.assertTrue((root / "reports" / "latency.interactive.html").is_file())
            self.assertTrue((root / "reports" / "tpm.interactive.html").is_file())
            self.assertTrue(
                (root / "reports" / "formula" / "fit_report.json").is_file()
            )

    def test_engine_failure_preserves_typed_outcome_without_finalizing_cases(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            source = default_executor_source(root.parent)
            plan = build_execution_plan(
                mode="prefill",
                action="run",
                resolved=resolve_config(source),
                cases=[
                    {
                        "input_len": 16,
                        "cache_len": 0,
                        "compute_len": 16,
                        "batch_size": 1,
                        "measure_runs": 1,
                        "seed": 1,
                        "prefix_policy": "shared_seed",
                    }
                ],
                result_directory=str(root),
            )
            initialize_run_directory(
                root,
                source_config=source,
                plan=plan,
                run_id="run-engine-failure",
                code_version="test",
            )
            executor = UnifiedLocalExecutor(
                session_factory=Mock(return_value=default_executor_FailingSession()),
                tokenizer_loader=lambda _plan: default_executor_Tokenizer(),
                query_builder=lambda _plan, _lengths: {},
            )
            outcome = executor(plan)
            self.assertEqual(RunState.FAILED, outcome.state)
            self.assertEqual(4, outcome.exit_code)
            self.assertFalse((root / "results" / "result.json").exists())
            manifest = json.loads((root / "manifest.json").read_text())
            self.assertEqual("failed", manifest["state"])


class PrefillBackendIntegrationTest(unittest.TestCase):

    def test_batch_four_uses_shared_http_runner_across_backends(self):
        self._batch_four_across_backends("http_text")

    def _batch_four_across_backends(self, transport):
        from rtp_llm.test.perf_test.execution.runtime import HttpPrefillClient
        from rtp_llm.test.perf_test.prefill.planner import plan_prefill_cases

        for backend in ("local",):
            with self.subTest(
                backend=backend
            ), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                run_root = root / "run"
                source = default_executor_source(root)
                source["workload"]["transport"] = transport
                source["engine"]["concurrency_limit"] = 4
                source["workload"].update(batch_sizes=[4], compute_steps=[8, 4])
                parsed = parse_config(source)
                plan = build_execution_plan(
                    mode="prefill",
                    action="run",
                    resolved=resolve_config(source),
                    cases=plan_prefill_cases(parsed.workload, parsed.engine),
                    result_directory=str(run_root),
                )
                events = []
                wire_requests = []
                barrier = threading.Barrier(4, timeout=3)
                lock = threading.Lock()
                endpoint = "http://fake-prefill:18000"

                def post(url, *, json, **_kwargs):
                    with lock:
                        events.append((url, json))
                    if url.endswith("/update_scheduler_info"):
                        self.assertIn(
                            json,
                            (
                                {"batch_size": 4, "mode": "prefill"},
                                {"batch_size": 1, "mode": "prefill"},
                            ),
                        )
                        return Mock(status_code=200, json=lambda: {"status": "ok"})
                    self.assertEqual(endpoint, url)
                    is_probe = "cache_probe_" in json["prompt"]
                    if not is_probe:
                        barrier.wait()
                    return Mock(
                        status_code=200,
                        json=lambda: {
                            "aux_info": {
                                "input_len": len(json["prompt"].split()),
                                "output_len": 1,
                                "reuse_len": (
                                    4 if is_probe and "__run_" in json["prompt"] else 0
                                ),
                                "first_token_cost_time": 10.0,
                            }
                        },
                    )

                http_client = HttpPrefillClient(
                    endpoint, post=post, scheduler_attempts=1
                )
                client = http_client
                tokenizer = default_executor_Tokenizer()
                if transport == "dashsc_input_ids":
                    from rtp_llm.dash_sc.proto import predict_v2_pb2
                    from rtp_llm.test.perf_test.prefill.dashsc_transport import (
                        DashScPrefillClient,
                    )

                    class Tokenizer:

                        def __init__(self):
                            self.tokens = {}

                        def encode(self, text):
                            with lock:
                                return [
                                    self.tokens.setdefault(word, len(self.tokens) + 1)
                                    for word in text.split()
                                ]

                    tokenizer = Tokenizer()

                    class Stub:

                        def ModelStreamInfer(self, requests, **_kwargs):
                            (request,) = list(requests)
                            request = predict_v2_pb2.ModelInferRequest.FromString(
                                request.SerializeToString()
                            )
                            wire_requests.append(request)
                            ids = next(
                                (
                                    raw
                                    for (tensor, raw) in zip(
                                        request.inputs, request.raw_input_contents
                                    )
                                    if tensor.name == "input_ids"
                                )
                            )
                            values = struct.unpack("<" + "i" * (len(ids) // 4), ids)
                            with lock:
                                words = {
                                    value: word
                                    for (word, value) in tokenizer.tokens.items()
                                }
                            prompt = " ".join((words[value] for value in values))
                            payload = post(endpoint, json={"prompt": prompt}).json()[
                                "aux_info"
                            ]
                            response = predict_v2_pb2.ModelStreamInferResponse()
                            infer = response.infer_response
                            infer.parameters[
                                "engine_first_token_cost_time_us"
                            ].int64_param = 10000
                            for name, value in (
                                ("prompt_token_num", payload["input_len"]),
                                ("prompt_cached_token_num", payload["reuse_len"]),
                                ("generated_ids", 42),
                            ):
                                output = infer.outputs.add(name=name, datatype="INT32")
                                output.shape.append(1)
                                infer.raw_output_contents.append(
                                    struct.pack("<i", value)
                                )
                            return iter([response])

                    client = DashScPrefillClient(
                        http_client, channel=Mock(), stub=Stub()
                    )
                    client.wait_ready = Mock()
                session = default_executor_Session()
                session.client = client
                session.diagnostics = EngineDiagnostics()
                with patch(
                    "rtp_llm.test.perf_test.test_util._load_tokenizer",
                    return_value=tokenizer,
                ):
                    if backend == "local":
                        initialize_run_directory(
                            run_root,
                            source_config=source,
                            plan=plan,
                            run_id="integration",
                            code_version="test",
                        )
                        outcome = UnifiedLocalExecutor(
                            session_factory=lambda _plan: session,
                            tokenizer_loader=lambda _plan: tokenizer,
                        )(plan)
                        self.assertEqual(RunState.COMPLETED, outcome.state)
                        results = outcome.case_results
                self.assertCountEqual(
                    [case.case_id for case in plan.cases],
                    [result["case_id"] for result in results],
                )
                self.assertGreater(len(results), 1)
                self.assertEqual(len(plan.cases) * 5 + 3, len(events))
                if transport == "dashsc_input_ids":
                    self.assertEqual(len(plan.cases) * 4 + 2, len(wire_requests))
                self.assertEqual({"batch_size": 1, "mode": "prefill"}, events[0][1])
                self.assertTrue(
                    (run_root / "checkpoints" / "prefill_cache_probe.json").is_file()
                )
                events = events[3:]
                for offset in range(0, len(events), 5):
                    self.assertEqual(
                        endpoint + "/update_scheduler_info", events[offset][0]
                    )
                    self.assertEqual(
                        [endpoint] * 4,
                        [url for (url, _) in events[offset + 1 : offset + 5]],
                    )
                for result in results:
                    self.assertEqual("ok", result["status"])
                    self.assertEqual(
                        "concurrent_batch_v1",
                        result["identity"]["measurement_contract"],
                    )
                    records = result["metrics"]["measurements"]
                    self.assertEqual(
                        [0, 1, 2, 3], [record["request_index"] for record in records]
                    )
                    self.assertTrue(
                        all((record["round_elapsed_ms"] > 0 for record in records))
                    )


class LocalExecutorTest(unittest.TestCase):

    def test_prefill_and_decode_adapters_receive_one_session_and_frozen_plan(self):
        for mode in ("prefill", "decode"):
            with self.subTest(mode=mode):
                plan = local_executor_plan(mode)
                session = local_executor_FakeSession()
                adapter = Mock(
                    return_value=ModeExecutionResult(
                        state=RunState.COMPLETED,
                        case_results=({"case_id": plan.cases[0].case_id},),
                    )
                )
                other_adapter = Mock()
                adapters = {
                    mode: adapter,
                    "decode" if mode == "prefill" else "prefill": other_adapter,
                }
                session_factory = Mock(return_value=session)
                store = object()
                store_factory = Mock(return_value=store)
                executor = LocalExecutor(
                    session_factory=session_factory,
                    store_factory=store_factory,
                    adapters=adapters,
                )
                with patch(
                    "rtp_llm.test.perf_test.config_resolver.load_resolved_config",
                    side_effect=AssertionError("source config must not be parsed"),
                ) as source_loader:
                    outcome = executor.execute(plan)
                self.assertEqual(0, outcome.exit_code)
                self.assertEqual("process.log", outcome.engine_diagnostics.log_path)
                session_factory.assert_called_once_with(plan)
                store_factory.assert_called_once_with(plan.result_directory)
                adapter.assert_called_once()
                context = adapter.call_args.args[0]
                self.assertIs(plan, context.plan)
                self.assertIs(session, context.session)
                self.assertIs(store, context.store)
                self.assertFalse(hasattr(context, "source_config"))
                self.assertFalse(hasattr(context, "config_path"))
                other_adapter.assert_not_called()
                source_loader.assert_not_called()
                self.assertEqual(1, session.enter_calls)
                self.assertEqual(1, session.exit_calls)


if __name__ == "__main__":
    unittest.main()
