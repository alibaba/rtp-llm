"""Core performance behavior contracts."""

import json
import tempfile
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
    TargetTPOTPolicy,
)
from rtp_llm.test.perf_test.decode.planner import plan_decode_cases
from rtp_llm.test.perf_test.decode.scheduler import schedule_decode_requests
from rtp_llm.test.perf_test.decode.strategy import TargetTPOTStrategy, strategy_for
from rtp_llm.test.perf_test.errors import PlanningError
from rtp_llm.test.perf_test.modes import DecodeAdapter, ModeExecutionContext
from rtp_llm.test.perf_test.plan import build_execution_plan
from rtp_llm.test.perf_test.state import RunState
from rtp_llm.test.perf_test.storage import RunStore
from rtp_llm.test.perf_test.test_support import (
    fixture_engine,
    fixture_execution,
    fixture_model,
)


def decode_fixed_config(input_lengths=None, batch_sizes=None, output_length=32):
    return {
        "schema_version": 1,
        "model": fixture_model(),
        "engine": fixture_engine(),
        "workload": {
            "kind": "grid",
            "input_lengths": input_lengths or [128, 256, 512],
            "output_lengths": [output_length],
            "batch": {"strategy": "fixed", "sizes": batch_sizes or [1, 2, 4, 8]},
            "measure_runs": 2,
            "warmup_runs": 1,
            "generate_config": {"top_k": 1},
        },
        "execution": fixture_execution(),
        "output": {},
        "regression": {},
        "publish": {},
    }


def decode_fixed_measurement(case, batch, *, tpot=0.1):
    return DecodeBatchMeasurement(
        batch_size=batch,
        input_lengths=case.input_lengths[:batch],
        worker_allocations=(),
        measurements=tuple(
            (
                {
                    "run": run,
                    "throughput_tokens_per_second": 1000.0,
                    "worker_allocations": [],
                    "requests": [
                        {
                            "request_index": index,
                            "success": True,
                            "input_length": input_length,
                            "output_length": case.output_length,
                            "latency_ms": {
                                "wait": 1.0,
                                "prefill": 2.0,
                                "decode": 3.0,
                                "tpot": tpot,
                                "total": 5.0,
                            },
                        }
                        for (index, input_length) in enumerate(
                            case.input_lengths[:batch]
                        )
                    ],
                }
                for run in range(case.measure_runs)
            )
        ),
    )


def decode_fixed_executor(measure):
    batch_executor = Mock()
    batch_executor.measure.side_effect = measure
    return DecodeCaseExecutor(
        batch_executor, topology={"tp": 1, "dp": 1, "pp": 1, "ep": 1}
    )


def decode_strategy_exact_case(batch=4):
    return DecodeCaseSpec(
        workload_kind="grid",
        input_lengths=(128,) * batch,
        output_length=16,
        batch_policy=ExactBatchPolicy(strategy="fixed", size=batch),
        measure_runs=2,
        warmup_runs=0,
    )


def decode_strategy_tpot_case(max_size=16, step=4, target=10.0):
    return replace(
        decode_strategy_exact_case(max_size),
        batch_policy=TargetTPOTPolicy("target_tpot", max_size, step, target),
    )


def decode_strategy_measurement(batch, tpot, *, failed=False):
    return DecodeBatchMeasurement(
        batch_size=batch,
        input_lengths=(128,) * batch,
        worker_allocations=(),
        measurements=tuple(
            (
                {
                    "run": run,
                    "worker_allocations": [],
                    "throughput_tokens_per_second": 100.0 + run,
                    "requests": [
                        {
                            "request_index": index,
                            "success": not (failed and run == 1 and (index == 0)),
                            "input_length": 128,
                            "output_length": 16,
                            "latency_ms": {
                                "wait": 1.0,
                                "prefill": 2.0,
                                "decode": 15.0,
                                "tpot": tpot,
                                "total": 17.0,
                            },
                        }
                        for index in range(batch)
                    ],
                }
                for run in range(2)
            )
        ),
    )


def decode_distribution_executor(batch_runner):

    def measure(spec, batch):
        inputs = spec.input_lengths[:batch]
        raw = batch_runner(
            input_lengths=inputs,
            output_length=spec.output_length,
            measure_runs=spec.measure_runs,
            worker_allocations=[],
        )
        return DecodeBatchMeasurement(batch, inputs, (), tuple(raw["measurements"]))

    batch_executor = Mock()
    batch_executor.measure.side_effect = measure
    return DecodeCaseExecutor(
        batch_executor, topology={"tp": 1, "dp": 1, "pp": 1, "ep": 1}
    )


def decode_distribution_payload(*, search=False):
    spec = DecodeCaseSpec(
        workload_kind="distribution",
        input_lengths=(64, 128, 256, 512) if search else (64, 256),
        output_length=32,
        batch_policy=(
            TargetTPOTPolicy("target_tpot", 4, 2, 10.0)
            if search
            else ExactBatchPolicy("fixed", 2)
        ),
        measure_runs=1 if search else 2,
        warmup_runs=0,
        distribution={"source_kind": "sequence_lengths"},
    )
    return {"case_id": "distribution", **spec.to_identity()}


def decode_executor_request_round(input_lengths, output_length):
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

    def test_target_tpot_searches_distribution_prefix_batches(self):

        def execute(**kwargs):
            measurements = []
            for run in range(kwargs["measure_runs"]):
                measurements.append(
                    {
                        "run": run,
                        "throughput_tokens_per_second": 100.0,
                        "worker_allocations": kwargs["worker_allocations"],
                        "requests": [
                            {
                                "request_index": index,
                                "success": True,
                                "input_length": input_length,
                                "output_length": kwargs["output_length"],
                                "latency_ms": {
                                    "wait": 1.0,
                                    "prefill": 2.0,
                                    "decode": 3.0,
                                    "tpot": 5.0,
                                    "total": 5.0,
                                },
                            }
                            for (index, input_length) in enumerate(
                                kwargs["input_lengths"]
                            )
                        ],
                    }
                )
            return {"measurements": measurements}

        batch_runner = Mock(side_effect=execute)
        result = decode_distribution_executor(batch_runner).execute(
            decode_distribution_payload(search=True)
        )
        self.assertEqual(4, result["metrics"]["search"]["best_batch_size"])
        self.assertEqual(
            [(64, 128), (64, 128, 256, 512)],
            [call.kwargs["input_lengths"] for call in batch_runner.call_args_list],
        )
        self.assertEqual(
            [64, 128, 256, 512],
            result["metrics"]["distribution"]["actual_sequence_lengths"],
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

    def test_measure_supports_uneven_and_idle_dp_workers(self):
        for batch, dp, expected in ((10, 3, [4, 3, 3]), (2, 4, [1, 1, 0, 0])):
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

    def test_measure_rejects_capacity_before_calling_transport(self):
        for inputs, batch, concurrency, tokens, message in (
            ((128,), 2, 2, 512, "inputs"),
            ((128, 128, 128), 3, 1, 512, "concurrency_limit"),
            ((200, 200, 100), 3, 2, 300, "max_batch_tokens"),
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
                    raise RuntimeError("transport disconnected")
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
                "transport disconnected",
                outcome.case_results[0]["errors"][0]["message"],
            )
            self.assertEqual(2, len(store.read_committed_cases()))
            checkpoint = json.loads(
                (Path(temp_dir) / "checkpoints" / "decode_progress.json").read_text()
            )
            self.assertEqual(2, len(checkpoint["completed_case_ids"]))


class DecodeStrategyTest(unittest.TestCase):

    def test_tpot_promotes_cached_best_without_forcing_maximum(self):
        measured = {}

        def measure(_case, batch):
            self.assertNotIn(batch, measured)
            measured[batch] = decode_strategy_measurement(
                batch, 8.0 if batch <= 8 else 12.0
            )
            return measured[batch]

        case = decode_strategy_tpot_case()
        outcome = strategy_for(case.batch_policy).execute(case, measure)
        self.assertIsInstance(strategy_for(case.batch_policy), TargetTPOTStrategy)
        self.assertIs(measured[8], outcome.selected)
        self.assertEqual([8, 12], list(measured))
        self.assertEqual("found", outcome.search["outcome"])
        self.assertEqual(8, outcome.search["best_batch_size"])
        self.assertEqual(
            100.5, outcome.search["observations"][0]["throughput_tokens_per_second"]
        )

    def test_tpot_failure_retains_observations_without_formal_measurement(self):
        outcome = TargetTPOTStrategy().execute(
            decode_strategy_tpot_case(max_size=8, target=1.0),
            lambda _case, batch: decode_strategy_measurement(batch, 20.0),
        )
        self.assertIsNone(outcome.selected)
        self.assertEqual("target_tpot_not_met", outcome.error_code)
        self.assertEqual("no_satisfying_candidate", outcome.search["outcome"])
        self.assertEqual(
            [4, 1], [item["batch_size"] for item in outcome.search["observations"]]
        )

    def test_tpot_requires_success_in_every_formal_round(self):
        outcome = TargetTPOTStrategy().execute(
            decode_strategy_tpot_case(max_size=1),
            lambda _case, batch: decode_strategy_measurement(batch, 1.0, failed=True),
        )
        self.assertIsNone(outcome.selected)
        self.assertEqual(0.5, outcome.search["observations"][0]["success_rate"])


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


if __name__ == "__main__":
    unittest.main()
