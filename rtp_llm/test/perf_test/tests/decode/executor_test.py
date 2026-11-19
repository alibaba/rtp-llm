from rtp_llm.test.perf_test.tests.support.execution import default_executor_measurement
from rtp_llm.test.perf_test.tests.support.execution import (
    default_executor_request_round as decode_executor_request_round,
)

"Performance framework behavior and failure regressions."
import json
import tempfile
import threading
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, call

from rtp_llm.test.perf_test.config_resolver import resolve_config
from rtp_llm.test.perf_test.config_schema import EngineConfig, parse_config
from rtp_llm.test.perf_test.decode.executor import (
    DecodeBatchExecutor,
    DecodeCapacityError,
    DecodeCaseExecutor,
)
from rtp_llm.test.perf_test.decode.model import (
    DecodeBatchMeasurement,
    DecodeCaseSpec,
    ExactBatchPolicy,
)
from rtp_llm.test.perf_test.decode.planner import plan_decode_cases
from rtp_llm.test.perf_test.decode.scheduler import schedule_decode_requests
from rtp_llm.test.perf_test.decode.transport import DecodeHttpTransport
from rtp_llm.test.perf_test.errors import PlanningError
from rtp_llm.test.perf_test.modes import DecodeAdapter, ModeExecutionContext
from rtp_llm.test.perf_test.plan import build_execution_plan
from rtp_llm.test.perf_test.state import RunState
from rtp_llm.test.perf_test.storage import RunStore
from rtp_llm.test.perf_test.tests.support.execution import decode_transport_Response
from rtp_llm.test.perf_test.tests.support.fixtures import fixture_config


def decode_fixed_config(input_lengths=None, batch_sizes=None, output_length=32):
    return fixture_config(
        workload={
            "kind": "grid",
            "input_lengths": input_lengths or [128, 256, 512],
            "output_lengths": [output_length],
            "batch": {"strategy": "fixed", "sizes": batch_sizes or [1, 2, 4, 8]},
            "measure_runs": 2,
            "warmup_runs": 1,
            "generate_config": {"top_k": 1},
        }
    )


def decode_fixed_measurement(case, batch, *, tpot=0.1, throughput=1000.0):
    return default_executor_measurement(
        batch,
        case.input_lengths[:batch],
        case.output_length,
        runs=case.measure_runs,
        throughput=throughput,
        latency={
            "wait": 1.0,
            "prefill": 2.0,
            "decode": 3.0,
            "tpot": tpot,
            "total": 5.0,
        },
    )


def decode_fixed_executor(measure):
    batch_executor = Mock()
    batch_executor.measure.side_effect = measure
    return DecodeCaseExecutor(
        batch_executor, topology={"tp": 1, "dp": 1, "pp": 1, "ep": 1}
    )


def decode_fixed_case(batch=4):
    return DecodeCaseSpec(
        workload_kind="grid",
        input_lengths=(128,) * batch,
        output_length=16,
        batch_policy=ExactBatchPolicy(strategy="fixed", size=batch),
        measure_runs=2,
        warmup_runs=0,
    )


def decode_distribution_payload():
    spec = DecodeCaseSpec(
        workload_kind="distribution",
        input_lengths=(64, 256),
        output_length=32,
        batch_policy=ExactBatchPolicy("fixed", 2),
        measure_runs=2,
        warmup_runs=0,
        distribution={"source_kind": "sequence_lengths"},
    )
    return {"case_id": "distribution", **spec.to_identity()}


def decode_executor_exact_case(*, batch, warmup=0, runs=1):
    return DecodeCaseSpec(
        workload_kind="grid",
        input_lengths=(128,) * batch,
        output_length=16,
        batch_policy=ExactBatchPolicy(strategy="fixed", size=batch),
        measure_runs=runs,
        warmup_runs=warmup,
        generate_config={"top_p": 0.8},
    )


class DecodeDistributionTest(unittest.TestCase):

    def test_fixed_distribution_measures_declared_lengths(self):
        measure = Mock(
            side_effect=lambda case, batch: decode_fixed_measurement(
                case, batch, tpot=5.0, throughput=100.0
            )
        )
        result = decode_fixed_executor(measure).execute(decode_distribution_payload())
        self.assertEqual("ok", result["status"])
        self.assertEqual(2, result["metrics"]["global_batch_size"])
        self.assertNotIn("search", result["metrics"])
        self.assertEqual(
            [((64, 256), 2)],
            [
                (item.args[0].input_lengths, item.args[1])
                for item in measure.call_args_list
            ],
        )
        self.assertEqual(
            [64, 256], result["metrics"]["distribution"]["actual_sequence_lengths"]
        )


class DecodeBatchExecutorTest(unittest.TestCase):

    def test_measure_configures_dp_then_runs_warmup_and_formal_rounds(self):
        transport = Mock()
        transport.run_round.side_effect = [
            decode_executor_request_round((128, 128), 16),
            decode_executor_request_round((128, 128), 16),
            decode_executor_request_round((128, 128), 16),
        ]
        executor = DecodeBatchExecutor(
            transport,
            engine=EngineConfig(
                max_seq_len=512, max_batch_tokens=512, concurrency_limit=2
            ),
            dp_size=2,
            monotonic=Mock(side_effect=[1.0, 1.1, 2.0, 2.2]),
        )
        measurement = executor.measure(
            decode_executor_exact_case(batch=2, warmup=1, runs=2), 2
        )
        self.assertEqual(2, measurement.batch_size)
        self.assertEqual(2, len(measurement.measurements))
        self.assertEqual(3, transport.run_round.call_count)
        allocations = transport.configure_scheduler.call_args.args[0]
        self.assertEqual([1, 1], [item["local_batch_size"] for item in allocations])
        self.assertEqual(
            [call.configure_scheduler(allocations)]
            + [
                call.run_round(
                    input_lengths=(128, 128),
                    output_length=16,
                    generate_config={"top_p": 0.8},
                )
            ]
            * 3,
            transport.method_calls,
        )
        self.assertEqual([0, 1], [item["run"] for item in measurement.measurements])
        self.assertAlmostEqual(
            320.0, measurement.measurements[0]["throughput_tokens_per_second"]
        )
        self.assertAlmostEqual(
            160.0, measurement.measurements[1]["throughput_tokens_per_second"]
        )

    def test_measure_supports_uniform_dp_and_rejects_idle_workers(self):
        for batch, dp, expected in ((9, 3, [3, 3, 3]), (2, 4, [1, 1, 0, 0])):
            with self.subTest(batch=batch, dp=dp):
                transport = Mock()
                transport.run_round.return_value = decode_executor_request_round(
                    (128,) * batch, 16
                )
                executor = DecodeBatchExecutor(
                    transport,
                    engine=EngineConfig(
                        max_seq_len=512, max_batch_tokens=5120, concurrency_limit=10
                    ),
                    dp_size=dp,
                )
                if batch < dp:
                    with self.assertRaisesRegex(DecodeCapacityError, "positive"):
                        executor.measure(decode_executor_exact_case(batch=batch), batch)
                    self.assertEqual([], transport.method_calls)
                    continue
                measurement = executor.measure(
                    decode_executor_exact_case(batch=batch), batch
                )
                self.assertEqual(
                    expected,
                    [
                        item["local_batch_size"]
                        for item in measurement.worker_allocations
                    ],
                )

    def test_uneven_active_dp_batches_fail_before_measurement_or_profiling(self):
        for batch, dp in ((3, 2), (10, 3), (1, 2), (2, 4)):
            for operation in ("measure", "profile"):
                with self.subTest(batch=batch, dp=dp, operation=operation):
                    transport = Mock()
                    transport.run_round.return_value = decode_executor_request_round(
                        (128,) * batch, 16
                    )
                    executor = DecodeBatchExecutor(
                        transport,
                        engine=EngineConfig(
                            max_seq_len=512, max_batch_tokens=5120, concurrency_limit=10
                        ),
                        dp_size=dp,
                    )
                    case = decode_executor_exact_case(batch=batch)
                    with self.assertRaisesRegex(DecodeCapacityError, "uniform"):
                        if operation == "measure":
                            executor.measure(case, batch)
                        else:
                            executor.profile(
                                case,
                                batch,
                                trace_name="unused",
                                steps=1,
                                enable_all_ranks=False,
                            )
                    self.assertEqual([], transport.method_calls)

        # Frozen cases and direct callers must fail before any scheduler/profile RPC.
        for steps, output, args, env, message in (
            (1001, 1600, {}, {}, "profiling.steps.*1000"),
            (1500, 1600, {}, {}, "profiling.steps.*1000"),
            (0, 16, {}, {}, "profiling.steps"),
            (True, 16, {}, {}, "profiling.steps"),
            (2, 16, {"sp_type": "mtp"}, {}, "SP/MTP.*steps=1"),
            (2, 16, {}, {"SP_TYPE": "dspark"}, "SP/MTP.*steps=1"),
        ):
            with self.subTest(steps=steps, args=args, env=env):
                transport = Mock()
                transport.run_round.return_value = (
                    {"success": True, "request_index": 0},
                )
                executor = DecodeBatchExecutor(
                    transport,
                    engine=EngineConfig(
                        max_seq_len=4096,
                        max_batch_tokens=8192,
                        concurrency_limit=16,
                        args=args,
                        env=env,
                    ),
                    dp_size=1,
                )
                case = replace(
                    decode_executor_exact_case(batch=1), output_length=output
                )
                with self.assertRaisesRegex(PlanningError, message):
                    executor.profile(
                        case,
                        1,
                        trace_name="invalid-window",
                        steps=steps,
                        enable_all_ranks=True,
                    )
                self.assertFalse(executor.profile_start_attempted)
                self.assertEqual([], transport.method_calls)

        for steps, args, env in (
            (1000, {}, {}),
            (1, {"sp_type": "mtp"}, {}),
            (1, {}, {"SP_TYPE": "dspark"}),
        ):
            with self.subTest(valid_steps=steps, args=args, env=env):
                transport = Mock()
                transport.run_round.return_value = (
                    {"success": True, "request_index": 0},
                )
                executor = DecodeBatchExecutor(
                    transport,
                    engine=EngineConfig(
                        max_seq_len=4096,
                        max_batch_tokens=8192,
                        concurrency_limit=16,
                        args=args,
                        env=env,
                    ),
                    dp_size=1,
                )
                case = replace(decode_executor_exact_case(batch=1), output_length=1600)
                case = replace(
                    case, profiling=replace(case.profiling, enabled=True, runs=2)
                )
                executor.profile(
                    case,
                    1,
                    trace_name="safe-window",
                    steps=steps,
                    enable_all_ranks=True,
                )
                self.assertEqual(2, transport.start_profile.call_count)
                self.assertEqual(2, transport.run_round.call_count)
                for rpc, request in zip(
                    transport.start_profile.call_args_list,
                    transport.run_round.call_args_list,
                ):
                    self.assertEqual(steps, rpc.kwargs["steps"])
                    self.assertEqual(
                        steps, request.kwargs["generate_config"]["profile_step"]
                    )
                # A failed re-arm must not launch that round's requests.
                transport.reset_mock()
                transport.start_profile.side_effect = [
                    None,
                    RuntimeError("re-arm failed"),
                ]
                with self.assertRaisesRegex(RuntimeError, "re-arm failed"):
                    executor.profile(
                        case,
                        1,
                        trace_name="lost-arm",
                        steps=steps,
                        enable_all_ranks=True,
                    )
                self.assertTrue(executor.profile_start_attempted)
                self.assertEqual(1, transport.run_round.call_count)

    def test_measure_rejects_capacity_before_calling_transport(self):
        for inputs, batch, concurrency, tokens, message in (
            ((128,), 2, 2, 512, "inputs"),
            ((128, 128, 128, 128), 4, 1, 512, "concurrency_limit"),
            ((200, 200, 100, 100), 4, 2, 300, "max_batch_tokens"),
        ):
            with self.subTest(message=message):
                transport = Mock()
                executor = DecodeBatchExecutor(
                    transport,
                    engine=EngineConfig(
                        max_seq_len=512,
                        max_batch_tokens=tokens,
                        concurrency_limit=concurrency,
                    ),
                    dp_size=2,
                )
                case = replace(
                    decode_executor_exact_case(batch=batch), input_lengths=inputs
                )
                with self.assertRaisesRegex(DecodeCapacityError, message) as error:
                    executor.measure(case, batch)
                self.assertIsInstance(error.exception, PlanningError)
                self.assertEqual([], transport.method_calls)

    def test_measure_counts_only_successful_output_tokens_and_retains_failures(self):
        outcomes = decode_executor_request_round((128, 128), 16)
        outcomes[1].update(success=False, error="HTTP 503")
        transport = Mock()
        transport.run_round.return_value = outcomes
        executor = DecodeBatchExecutor(
            transport,
            engine=EngineConfig(
                max_seq_len=512, max_batch_tokens=512, concurrency_limit=2
            ),
            dp_size=1,
            monotonic=Mock(side_effect=[1.0, 1.5]),
        )
        measurement = executor.measure(decode_executor_exact_case(batch=2), 2)
        self.assertEqual(
            32.0, measurement.measurements[0]["throughput_tokens_per_second"]
        )
        self.assertEqual(
            "HTTP 503", measurement.measurements[0]["requests"][1]["error"]
        )
        outcomes[0]["latency_ms"]["tpot"] = 100.0
        self.assertEqual(
            1.0, measurement.measurements[0]["requests"][0]["latency_ms"]["tpot"]
        )


class DecodeFixedTest(unittest.TestCase):

    def test_rank_batch_report_shows_rank_and_global_sizes(self):
        from rtp_llm.test.perf_test.decode.metrics import build_decode_case_result
        from rtp_llm.test.perf_test.decode.report import generate_decode_report

        spec = replace(decode_fixed_case(batch=4), batch_unit="per_dp_rank")
        case = {**spec.to_identity(), "case_id": "decode-rank-grid"}
        allocations = [
            worker.to_dict()
            for worker in schedule_decode_requests(
                spec.input_lengths, dp_size=2
            ).workers
        ]
        selected = decode_fixed_measurement(spec, 4)
        selected = replace(
            selected,
            worker_allocations=tuple(allocations),
            measurements=tuple(
                (
                    {**round_data, "worker_allocations": allocations}
                    for round_data in selected.measurements
                )
            ),
        )
        result = build_decode_case_result(
            case,
            selected=selected,
            topology={"tp": 1, "dp": 2, "pp": 1, "ep": 2},
            profiling={"status": "disabled", "manifest_path": None, "error": None},
        )
        report = generate_decode_report([result])
        self.assertEqual("formal", report["report_kind"])
        self.assertEqual(2, report["observations"][0]["rank_batch_size"])
        self.assertEqual(4, report["observations"][0]["global_batch_size"])
        self.assertEqual(2, report["matrices"][0]["rank_batch_size"])
        self.assertEqual(4, report["matrices"][0]["global_batch_size"])

    def test_rank_batch_result_rejects_missing_dp_allocation(self):
        from rtp_llm.test.perf_test.decode.metrics import build_decode_case_result
        from rtp_llm.test.perf_test.results import ResultSchemaError, parse_case_result

        spec = replace(decode_fixed_case(batch=4), batch_unit="per_dp_rank")
        case = {**spec.to_identity(), "case_id": "decode-rank-grid"}
        result = build_decode_case_result(
            case,
            selected=decode_fixed_measurement(spec, 4),
            topology={"tp": 1, "dp": 2, "pp": 1, "ep": 2},
            profiling={"status": "disabled", "manifest_path": None, "error": None},
        )
        with self.assertRaisesRegex(ResultSchemaError, "worker_allocations"):
            parse_case_result(0, "decode", result)

    def test_adapter_runs_every_fixed_case_and_checkpoints_through_executor(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = parse_config(
                decode_fixed_config(input_lengths=[128, 256], batch_sizes=[1, 2])
            )
            cases = plan_decode_cases(config.workload, config.engine, dp_size=1)
            plan = build_execution_plan(
                mode="decode",
                action="run",
                resolved=resolve_config(
                    decode_fixed_config(input_lengths=[128, 256], batch_sizes=[1, 2])
                ),
                cases=cases,
                result_directory=str(root),
            )
            (root / "checkpoints").mkdir(parents=True)
            (root / "results" / "cases").mkdir(parents=True)
            measure = Mock(side_effect=decode_fixed_measurement)
            adapter = DecodeAdapter(lambda _context: decode_fixed_executor(measure))
            outcome = adapter(
                ModeExecutionContext(plan=plan, session=object(), store=RunStore(root))
            )
            self.assertEqual(RunState.COMPLETED, outcome.state)
            self.assertEqual(4, measure.call_count)
            self.assertEqual(
                {32}, {call.args[0].output_length for call in measure.call_args_list}
            )
            self.assertEqual(4, len(RunStore(root).read_committed_cases()))
            checkpoint = json.loads(
                (root / "checkpoints" / "decode_progress.json").read_text()
            )
            self.assertEqual(
                [case.case_id for case in plan.cases], checkpoint["completed_case_ids"]
            )
            self.assertEqual({"ok"}, set(checkpoint["case_statuses"].values()))

    def test_adapter_converts_case_exception_and_continues_checkpointing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = decode_fixed_config(input_lengths=[128, 256], batch_sizes=[1])
            config = parse_config(source)
            plan = build_execution_plan(
                mode="decode",
                action="run",
                resolved=resolve_config(source),
                cases=plan_decode_cases(config.workload, config.engine, dp_size=1),
                result_directory=temp_dir,
            )

            def measure(spec, batch):
                if spec.input_lengths[0] == 128:
                    raise RuntimeError()
                return decode_fixed_measurement(spec, batch)

            store = RunStore(Path(temp_dir))
            (Path(temp_dir) / "results" / "cases").mkdir(parents=True)
            (Path(temp_dir) / "checkpoints").mkdir()
            outcome = DecodeAdapter(lambda _context: decode_fixed_executor(measure))(
                ModeExecutionContext(plan=plan, session=object(), store=store)
            )
            self.assertEqual(RunState.PARTIAL, outcome.state)
            self.assertEqual(
                ["request_failed", "ok"],
                [item["status"] for item in outcome.case_results],
            )
            self.assertIsNone(outcome.case_results[0]["metrics"])
            self.assertEqual(
                "RuntimeError",
                outcome.case_results[0]["errors"][0]["message"],
            )
            self.assertEqual(2, len(store.read_committed_cases()))
            checkpoint = json.loads(
                (Path(temp_dir) / "checkpoints" / "decode_progress.json").read_text()
            )
            self.assertEqual(2, len(checkpoint["completed_case_ids"]))


class DecodeTransportTest(unittest.TestCase):

    def test_scheduler_rejects_uneven_active_batches_before_http(self):
        post = Mock(return_value=Mock(status_code=200, json=lambda: {"status": "ok"}))
        transport = DecodeHttpTransport(
            "http://unused", prompt_for_length=str, post=post
        )
        self.addCleanup(transport.close)
        with self.assertRaisesRegex(PlanningError, "uniform"):
            transport.configure_scheduler(
                ({"local_batch_size": 2}, {"local_batch_size": 1})
            )
        post.assert_not_called()

    def test_scheduler_configures_uniform_batches_and_rejects_idle_workers(self):
        for local_batches, expected in (([2, 2], 2), ([1, 1, 0, 0], 1)):
            with self.subTest(local_batches=local_batches):
                post = Mock(
                    return_value=Mock(status_code=200, json=lambda: {"status": "ok"})
                )
                transport = DecodeHttpTransport(
                    "http://unused", prompt_for_length=str, post=post
                )
                self.addCleanup(transport.close)
                if 0 in local_batches:
                    with self.assertRaisesRegex(PlanningError, "positive"):
                        transport.configure_scheduler(
                            tuple(
                                ({"local_batch_size": size} for size in local_batches)
                            )
                        )
                    post.assert_not_called()
                    continue
                transport.configure_scheduler(
                    tuple(({"local_batch_size": size} for size in local_batches))
                )
                self.assertEqual(
                    {"batch_size": expected, "mode": "decode"},
                    post.call_args.kwargs["json"],
                )


class DecodeSchedulerTest(unittest.TestCase):

    def test_even_and_uneven_batches_cover_each_request_exactly_once(self):
        for global_batch, dp_size, local_batches in (
            (8, 2, [4, 4]),
            (5, 2, [3, 2]),
            (2, 4, [1, 1, 0, 0]),
        ):
            with self.subTest(global_batch=global_batch, dp_size=dp_size):
                lengths = tuple((100 + index for index in range(global_batch)))
                first = schedule_decode_requests(lengths, dp_size=dp_size)
                second = schedule_decode_requests(lengths, dp_size=dp_size)
                self.assertEqual(first, second)
                self.assertEqual(global_batch, first.global_batch_size)
                self.assertEqual(
                    local_batches, [item.local_batch_size for item in first.workers]
                )
                indices = [
                    index
                    for worker in first.workers
                    for index in worker.request_indices
                ]
                self.assertEqual(list(range(global_batch)), indices)
                self.assertEqual(len(indices), len(set(indices)))


class DecodeHttpTransportTest(unittest.TestCase):

    def test_measure_reuses_workers_prepared_before_the_timed_rounds(self):
        rounds = []
        barrier = threading.Barrier(2)

        def post(url, **kwargs):
            if url.endswith("/update_scheduler_info"):
                return decode_transport_Response({"status": "ok"})
            rounds.append(threading.current_thread())
            barrier.wait(timeout=5)
            return decode_transport_Response(
                {
                    "aux_info": {
                        "input_len": 128,
                        "output_len": 16,
                        "wait_time": 1,
                        "first_token_cost_time": 3,
                        "cost_time": 18,
                    }
                }
            )

        transport = DecodeHttpTransport(
            "http://127.0.0.1:18000", prompt_for_length=str, post=post
        )
        self.addCleanup(transport.close)
        before_rounds = []

        def clock():
            before_rounds.append(set(threading.enumerate()))
            return float(len(before_rounds))

        executor = DecodeBatchExecutor(
            transport,
            engine=EngineConfig(
                max_seq_len=512, max_batch_tokens=512, concurrency_limit=2
            ),
            dp_size=1,
            monotonic=clock,
        )
        executor.measure(decode_executor_exact_case(batch=2, warmup=0, runs=2), 2)
        self.assertEqual(set(rounds[:2]), set(rounds[2:]))
        self.assertTrue(set(rounds).issubset(before_rounds[0]))
        self.assertEqual(2, len(set(rounds)))
        transport.close()
        self.assertFalse(any(worker.is_alive() for worker in rounds))

    def test_worker_start_failure_releases_already_started_workers(self):
        from unittest.mock import patch

        transport = DecodeHttpTransport(
            "http://unused", prompt_for_length=str, post=Mock()
        )
        self.addCleanup(transport.close)
        started = []
        start = threading.Thread.start

        def fail_second_worker(worker):
            if started:
                raise RuntimeError("worker cannot start")
            start(worker)
            started.append(worker)

        with patch.object(threading.Thread, "start", fail_second_worker):
            with self.assertRaisesRegex(RuntimeError, "worker cannot start"):
                transport.configure_scheduler(({"local_batch_size": 2},))
        self.assertFalse(any(worker.is_alive() for worker in started))

    def test_rounds_reuse_http_connections(self):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        connections = []

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def setup(self):
                super().setup()
                connections.append(self.connection)

            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                payload = json.dumps(
                    {
                        "aux_info": {
                            "input_len": 128,
                            "output_len": 16,
                            "wait_time": 1,
                            "first_token_cost_time": 3,
                            "cost_time": 18,
                        }
                    }
                ).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        serving = threading.Thread(target=server.serve_forever, daemon=True)
        serving.start()
        transport = DecodeHttpTransport(
            f"http://127.0.0.1:{server.server_port}", prompt_for_length=str
        )
        self.addCleanup(transport.close)
        try:
            for _ in range(2):
                requests = transport.run_round(
                    input_lengths=(128,), output_length=16, generate_config={}
                )
                self.assertTrue(requests[0]["success"])
            self.assertEqual(1, len(connections))
        finally:
            transport.close()
            server.shutdown()
            server.server_close()
            serving.join(timeout=5)

    def test_round_forces_planned_output_after_user_generate_config(self):
        post = Mock(
            return_value=decode_transport_Response(
                {
                    "aux_info": {
                        "input_len": 128,
                        "output_len": 16,
                        "wait_time": 1.0,
                        "first_token_cost_time": 3.0,
                        "cost_time": 18.0,
                    }
                }
            )
        )
        transport = DecodeHttpTransport(
            endpoint="http://127.0.0.1:18000/",
            prompt_for_length=lambda length: f"prompt-{length}",
            post=post,
        )
        self.addCleanup(transport.close)
        config = {
            "min_new_tokens": 2,
            "max_new_tokens": 2,
            "top_p": 0.8,
            "top_k": 3,
            "force_sp_accept": False,
            "nested": {"value": 1},
        }
        outcomes = transport.run_round(
            input_lengths=(128,), output_length=16, generate_config=config
        )
        request = post.call_args.kwargs["json"]
        self.assertEqual("http://127.0.0.1:18000", post.call_args.args[0])
        self.assertEqual("prompt-128", request["prompt"])
        self.assertEqual(16, request["generate_config"]["min_new_tokens"])
        self.assertEqual(16, request["generate_config"]["max_new_tokens"])
        self.assertEqual(0.8, request["generate_config"]["top_p"])
        self.assertEqual(3, request["generate_config"]["top_k"])
        self.assertTrue(request["generate_config"]["force_sp_accept"])
        request["generate_config"]["nested"]["value"] = 2
        self.assertEqual(1, config["nested"]["value"])
        self.assertEqual(2, config["max_new_tokens"])
        self.assertTrue(outcomes[0]["success"])
        self.assertEqual(
            {"wait": 1.0, "prefill": 2.0, "decode": 15.0, "tpot": 1.0, "total": 17.0},
            outcomes[0]["latency_ms"],
        )

    def test_round_preserves_http_failure_as_request_outcome(self):
        transport = DecodeHttpTransport(
            endpoint="http://127.0.0.1:18000",
            prompt_for_length=lambda _length: "prompt",
            post=Mock(
                return_value=decode_transport_Response(
                    status_code=503, text="not ready"
                )
            ),
        )
        self.addCleanup(transport.close)
        outcome = transport.run_round(
            input_lengths=(128,), output_length=8, generate_config={}
        )[0]
        self.assertFalse(outcome["success"])
        self.assertIn("HTTP 503", outcome["error"])
        self.assertEqual(
            (0, 128, 8),
            tuple(
                (
                    outcome[key]
                    for key in ("request_index", "input_length", "output_length")
                )
            ),
        )

    def test_scheduler_retries_http_200_error_body(self):
        post = Mock(
            side_effect=[
                decode_transport_Response(
                    {"error": "Failed on some addresses: worker unavailable"}
                ),
                decode_transport_Response({"status": "ok"}),
            ]
        )
        sleep = Mock()
        transport = DecodeHttpTransport(
            endpoint="http://127.0.0.1:18000",
            prompt_for_length=str,
            post=post,
            sleep=sleep,
            scheduler_attempts=2,
        )
        self.addCleanup(transport.close)
        transport.configure_scheduler(({"local_batch_size": 2},))
        self.assertEqual(2, post.call_count)
        sleep.assert_called_once_with(3)
        from rtp_llm.test.perf_test.errors import EngineLifecycleError

        health = Mock(side_effect=EngineLifecycleError("engine exited"))
        post.reset_mock()
        broken = DecodeHttpTransport(
            endpoint="http://localhost",
            prompt_for_length=str,
            post=post,
            health_check=health,
        )
        self.addCleanup(broken.close)
        with self.assertRaises(EngineLifecycleError):
            broken.configure_scheduler(({"local_batch_size": 2},))
        post.assert_not_called()


if __name__ == "__main__":
    unittest.main()
