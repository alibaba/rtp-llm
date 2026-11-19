"""Performance framework behavior and failure regressions."""

import asyncio
import json
import sys
import threading
import unittest
from unittest.mock import MagicMock, patch

from rtp_llm.test.perf_test.offline_runner import (
    OfflineBenchConfig,
    OfflineMetrics,
    OfflineRunner,
    WorkloadGenerator,
    _RequestRecord,
)


class TestWorkloadGenerator(unittest.TestCase):
    """Test WorkloadGenerator prompt generation with a fake tokenizer."""

    def _make_fake_tokenizer(self):
        tok = MagicMock()
        tok.encode = lambda text: list(range(len(text.split())))
        tok.decode = lambda ids: " ".join((str(i) for i in ids))
        return tok

    def test_basic_generation(self):
        from rtp_llm.test.perf_test.offline_runner import (
            OfflineBenchConfig,
            WorkloadGenerator,
        )

        tokenizer = self._make_fake_tokenizer()
        config = OfflineBenchConfig(
            input_len_min=6,
            input_len_max=6,
            output_len_min=2,
            output_len_max=4,
            prefix_len=0,
        )
        gen = WorkloadGenerator(config, tokenizer, seed=0)
        (prompt, out_len) = gen.next()
        self.assertIsInstance(prompt, str)
        self.assertEqual(len(tokenizer.encode(prompt)), 6)
        self.assertGreaterEqual(out_len, 2)
        self.assertLessEqual(out_len, 4)


class TestOfflineBenchMain(unittest.TestCase):

    def test_all_failed_requests_raise_from_entry_point_and_stop_server(self):
        import rtp_llm.test.perf_test.offline_bench_test as offline_bench
        import rtp_llm.test.perf_test.offline_runner as offline_runner_module
        from rtp_llm.test.perf_test.offline_runner import OfflineMetrics, OfflineRunner

        server = MagicMock(port=12345)
        failed_metrics = OfflineMetrics(
            total_submitted=2, success_requests=0, fail_requests=2
        )
        argv = [
            "prog",
            "--checkpoint_path",
            "/tmp/model",
            "--concurrency_limit",
            "2",
            "--existing-server-port",
            "23456",
        ]
        with patch.object(sys, "argv", argv), patch.object(
            offline_bench, "resolve_engine_paths", side_effect=lambda args: args
        ), patch.object(offline_bench, "EngineServer") as engine_server, patch.object(
            OfflineRunner, "_load_tokenizer"
        ), patch.object(
            OfflineRunner, "_run", return_value=failed_metrics
        ), patch.object(
            offline_runner_module, "WorkloadGenerator"
        ), patch.object(
            OfflineMetrics, "print_table"
        ), patch.object(
            OfflineMetrics, "save_json"
        ), patch.object(
            OfflineRunner, "_save_status_samples"
        ), patch.object(
            offline_bench, "summarize_and_cleanup_coredumps"
        ), patch.object(
            offline_bench.os, "makedirs"
        ), patch(
            "rtp_llm.config.log_config.setup_logging"
        ):
            engine_server.return_value = server
            with self.assertRaisesRegex(RuntimeError, "no requests succeeded"):
                offline_bench.main()
        self.assertEqual(23456, engine_server.call_args.args[0].existing_server_port)
        self.assertNotIn("--existing-server-port", engine_server.call_args.args[1])
        server.stop.assert_called_once()

    def test_manual_concurrency_is_forwarded_to_server_and_runner(self):
        import rtp_llm.test.perf_test.offline_bench_test as offline_bench

        server = MagicMock(port=12345)
        argv = [
            "prog",
            "--checkpoint_path",
            "/tmp/model",
            "--concurrency_limit",
            "37",
            "--duration",
            "0",
            "--test_block_num",
            "17",
        ]
        with patch.object(sys, "argv", argv), patch.object(
            offline_bench, "resolve_engine_paths", side_effect=lambda args: args
        ), patch.object(offline_bench, "EngineServer") as engine_server, patch.object(
            offline_bench, "OfflineRunner"
        ) as offline_runner, patch.object(
            offline_bench, "summarize_and_cleanup_coredumps"
        ), patch.object(
            offline_bench.os, "makedirs"
        ), patch(
            "rtp_llm.config.log_config.setup_logging"
        ):
            engine_server.return_value = server
            offline_bench.main()
        (parsed_args, engine_args) = engine_server.call_args.args
        self.assertEqual(parsed_args.concurrency_limit, 37)
        self.assertEqual(engine_args[engine_args.index("--test_block_num") + 1], "17")
        self.assertEqual(server.start.call_args.kwargs["max_concurrency"], 37)
        runner_config = offline_runner.call_args.kwargs["config"]
        self.assertEqual(runner_config.concurrency_limit, 37)
        self.assertEqual(runner_config.total_requests, 74)


class TestOfflineAnalyze(unittest.TestCase):
    """Test OfflineRunner._analyze with synthetic records."""

    def test_analyze_with_failures(self):
        from rtp_llm.test.perf_test.offline_runner import (
            OfflineBenchConfig,
            OfflineRunner,
            _RequestRecord,
        )

        config = OfflineBenchConfig()
        runner = OfflineRunner.__new__(OfflineRunner)
        runner._config = config
        runner._result_dir = "/tmp"
        runner._status_samples = []
        records = [
            _RequestRecord(
                submit_time=0.0,
                finish_time=1.0,
                response={"error_code": "MALLOC_FAILED", "message": "OOM"},
            )
        ]
        metrics = runner._analyze(
            records,
            wall_time=1.0,
            engine_status={},
            submitted_count=1,
            cancelled_count=0,
        )
        self.assertEqual(metrics.success_requests, 0)
        self.assertEqual(metrics.fail_requests, 1)

    def test_duration_workload_dump_contains_only_submitted_requests(self):
        import io
        import json
        import threading

        import rtp_llm.test.perf_test.offline_runner as offline_runner
        from rtp_llm.test.perf_test.offline_runner import (
            OfflineBenchConfig,
            OfflineRunner,
            WorkloadGenerator,
        )

        second_prompt_generated = threading.Event()

        class RecordingWorkloadGenerator(WorkloadGenerator):

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.generated_ids = []

            def _generate(self, spec, *args, **kwargs):
                prepared = super()._generate(spec, *args, **kwargs)
                self.generated_ids.append(spec.request_id)
                if spec.request_id == 1:
                    second_prompt_generated.set()
                return prepared

        class FastTokenizer:

            def encode(self, text):
                return list(range(len(text.split())))

            def decode(self, token_ids):
                return " ".join((str(token_id) for token_id in token_ids))

        class FakeClientSession:

            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

        async def run_dispatch():
            config = OfflineBenchConfig(
                input_len_min=1, input_len_max=1, output_len_min=1, output_len_max=1
            )
            dump_file = io.StringIO()
            gen = RecordingWorkloadGenerator(
                config, FastTokenizer(), seed=0, dump_file=dump_file
            )
            runner = OfflineRunner.__new__(OfflineRunner)
            runner._config = config
            runner._port = 0
            runner._profile = False
            runner._profile_steps = 1
            runner._status_samples = []
            runner._status_sample_errors = 0

            async def fake_status_sampler(*args, **kwargs):
                await asyncio.Event().wait()

            async def hold_first_request(session, prompt, output_len):
                while not second_prompt_generated.is_set():
                    await asyncio.sleep(0)
                await asyncio.sleep(0.05)
                return {"aux_info": [{"input_len": 1, "output_len": output_len}]}

            runner._status_sampler = fake_status_sampler
            runner._send_one = hold_first_request
            with patch.object(
                offline_runner.aiohttp, "ClientSession", FakeClientSession
            ):
                result = await runner._dispatch(
                    gen, concurrency_limit=1, duration_s=0.02, drain_timeout_s=1
                )
            dumped_ids = [
                json.loads(line)["id"] for line in dump_file.getvalue().splitlines()
            ]
            return (result, gen.generated_ids, dumped_ids)

        (result, generated_ids, dumped_ids) = asyncio.run(run_dispatch())
        self.assertIn(1, generated_ids)
        self.assertEqual(result.submitted_count, 1)
        self.assertEqual(len(result.records), 1)
        self.assertEqual(dumped_ids, [0])

    def test_run_uses_full_dispatch_window_for_tps_after_cancellation(self):
        from rtp_llm.test.perf_test.offline_runner import (
            OfflineBenchConfig,
            OfflineRunner,
            _DispatchResult,
            _RequestRecord,
        )

        runner = OfflineRunner.__new__(OfflineRunner)
        runner._config = OfflineBenchConfig(
            input_len_min=1,
            input_len_max=1,
            output_len_min=1,
            output_len_max=1,
            concurrency_limit=2,
            total_requests=2,
        )
        runner._status_samples = []
        dispatch_result = _DispatchResult(
            records=[
                _RequestRecord(
                    submit_time=20.0,
                    finish_time=30.0,
                    response={"aux_info": [{"input_len": 100, "output_len": 50}]},
                )
            ],
            cancelled_count=1,
            submitted_count=2,
            first_submit_time=10.0,
            terminal_time=30.0,
        )
        with patch.object(runner, "_query_status", return_value={}), patch.object(
            runner, "_dispatch_fixed", return_value=dispatch_result
        ):
            metrics = runner._run(MagicMock())
        self.assertEqual(metrics.cancelled_requests, 1)
        self.assertAlmostEqual(metrics.total_wall_time_s, 20.0)
        self.assertAlmostEqual(metrics.output_tps, 2.5)


class TestOfflineBenchConfig(unittest.TestCase):
    def test_rejects_invalid_workload_and_dispatch_settings(self):
        cases = [
            ({"input_len_min": 0}, "input_len_min"),
            ({"input_len_min": 2049}, "input_len_min"),
            ({"output_len_min": 0}, "output_len_min"),
            ({"output_len_min": 513}, "output_len_min"),
            ({"prefix_groups": 0}, "prefix_groups"),
            ({"prefix_len": -1}, "prefix_len"),
            ({"prefix_len": 513}, "prefix_len"),
            ({"num_return_sequences": 0}, "num_return_sequences"),
            ({"duration_s": 0, "total_requests": 0}, "duration_s"),
            ({"concurrency_limit": 0}, "concurrency_limit"),
        ]
        for overrides, invalid_field in cases:
            with self.subTest(overrides=overrides):
                settings = {"concurrency_limit": 1, **overrides}
                with self.assertRaisesRegex(ValueError, invalid_field):
                    OfflineBenchConfig(**settings).validate()

    def test_accepts_prefix_equal_to_input_and_fixed_mode_without_duration(self):
        for total_requests, duration_s in [(0, 1), (3, 0)]:
            with self.subTest(total_requests=total_requests):
                OfflineBenchConfig(
                    input_len_min=4,
                    input_len_max=4,
                    prefix_len=4,
                    concurrency_limit=1,
                    total_requests=total_requests,
                    duration_s=duration_s,
                ).validate()


class TestOfflineFailurePolicy(unittest.TestCase):
    def test_rejects_empty_failed_and_cancelled_workloads(self):
        for metrics, reason in [
            (OfflineMetrics(), "no requests were submitted"),
            (
                OfflineMetrics(total_submitted=2, fail_requests=2),
                "no requests succeeded",
            ),
            (
                OfflineMetrics(total_submitted=2, cancelled_requests=2),
                "no requests succeeded",
            ),
        ]:
            with self.subTest(metrics=metrics):
                with self.assertRaisesRegex(RuntimeError, reason):
                    metrics.raise_if_all_requests_failed()

    def test_accepts_success_with_failed_and_cancelled_requests(self):
        OfflineMetrics(
            total_submitted=3,
            success_requests=1,
            fail_requests=1,
            cancelled_requests=1,
        ).raise_if_all_requests_failed()


class TestExactWorkloadGeneration(unittest.TestCase):
    def test_groups_have_distinct_exact_prefixes_across_requests(self):
        class CharacterTokenizer:
            def encode(self, text):
                return [ord(char) for char in text]

            def decode(self, token_ids):
                return "".join(chr(token_id) for token_id in token_ids)

        tokenizer = CharacterTokenizer()
        config = OfflineBenchConfig(
            input_len_min=12,
            input_len_max=12,
            output_len_min=1,
            output_len_max=1,
            prefix_groups=3,
            prefix_len=3,
        )
        with patch("rtp_llm.test.perf_test.offline_runner.BASE_TEXTS", ["abcdefghij"]):
            generator = WorkloadGenerator(config, tokenizer, seed=0)
        expected_prefixes = [(97, 98, 99), (100, 101, 102), (103, 104, 105)]
        for request_id in range(9):
            prompt, output_len = generator.next()
            with self.subTest(request_id=request_id):
                token_ids = tokenizer.encode(prompt)
                self.assertEqual(len(token_ids), 12)
                self.assertEqual(
                    tuple(token_ids[:3]), expected_prefixes[request_id % 3]
                )
                self.assertEqual(output_len, 1)

    def test_boundary_retokenization_repairs_length_and_prefix(self):
        class BoundaryMergingTokenizer:
            def __init__(self):
                self.boundary_merges = 0

            def encode(self, text):
                result = []
                index = 0
                while index < len(text):
                    if text[index : index + 2] == "xy":
                        result.append(1000)
                        self.boundary_merges += 1
                        index += 2
                    else:
                        result.append(ord(text[index]))
                        index += 1
                return result

            def decode(self, token_ids):
                replacements = {ord("T"): "x", ord("h"): "y", 1000: "xy"}
                return "".join(
                    replacements.get(token_id, chr(token_id)) for token_id in token_ids
                )

        tokenizer = BoundaryMergingTokenizer()
        config = OfflineBenchConfig(
            input_len_min=6,
            input_len_max=6,
            output_len_min=1,
            output_len_max=1,
            prefix_len=1,
        )
        with patch(
            "rtp_llm.test.perf_test.offline_runner.BASE_TEXTS", ["The quick fox"]
        ):
            generator = WorkloadGenerator(config, tokenizer, seed=0)
        prompt, _ = generator.next()
        token_ids = tokenizer.encode(prompt)
        self.assertGreater(tokenizer.boundary_merges, 0)
        self.assertEqual(len(token_ids), 6)
        self.assertEqual(token_ids[0], ord("x"))

    def test_unrepresentable_exact_prefix_raises(self):
        tokenizer = MagicMock()
        tokenizer.encode.return_value = [0, 1, 2]
        tokenizer.decode.side_effect = lambda ids: " ".join(str(i) for i in ids)
        config = OfflineBenchConfig(input_len_min=5, input_len_max=8, prefix_len=5)
        with patch.object(WorkloadGenerator, "_MAX_TOKEN_REPAIR_ATTEMPTS", 2):
            with self.assertRaisesRegex(ValueError, "exact tokenizer-validated"):
                WorkloadGenerator(config, tokenizer)


class TestOfflineStatus(unittest.TestCase):
    def test_cache_status_preserves_tokens_and_aggregates_all_dp_shards(self):
        cases = [
            (
                {
                    "total_kv_cache": "2247680",
                    "available_kv_cache": "2231296",
                    "block_size": "1024",
                },
                2247680,
                2231296,
                2195,
                1024,
                [16 / 2195],
            ),
            (
                {
                    "results": [
                        {
                            "total_kv_cache": "100",
                            "available_kv_cache": "20",
                            "block_size": "10",
                        },
                        {
                            "total_kv_cache": "300",
                            "available_kv_cache": "240",
                            "block_size": "10",
                        },
                    ]
                },
                400,
                260,
                40,
                10,
                [0.8, 0.2],
            ),
            (
                {
                    "results": [
                        {
                            "total_kv_cache": "100",
                            "available_kv_cache": "20",
                            "block_size": "10",
                        },
                        {
                            "total_kv_cache": "300",
                            "available_kv_cache": "240",
                            "block_size": "20",
                        },
                    ]
                },
                400,
                260,
                25,
                0,
                [0.8, 0.2],
            ),
        ]
        runner = OfflineRunner.__new__(OfflineRunner)
        runner._port = 12345
        worker = {
            "frontend_concurrency_limit": 4,
            "frontend_available_concurrency": 1,
            "running_task_info": [{"is_waiting": False}, {"is_waiting": True}, {}],
        }
        for payload, total, available, blocks, block_size, utilizations in cases:
            with self.subTest(payload=payload):
                responses = [MagicMock(), MagicMock()]
                responses[0].json.return_value = payload
                responses[1].json.return_value = worker
                with patch(
                    "rtp_llm.test.perf_test.offline_runner.requests.get",
                    side_effect=responses,
                ):
                    status = runner._query_status()
                self.assertEqual(status["total_kv_cache"], total)
                self.assertEqual(status["available_kv_cache"], available)
                self.assertEqual(status["total_kv_cache_blocks"], blocks)
                self.assertEqual(status["block_size"], block_size)
                self.assertAlmostEqual(
                    status["kv_cache_max_dp_utilization"], max(utilizations)
                )
                self.assertEqual(len(status["per_dp_kv_cache"]), len(utilizations))
                for shard, utilization in zip(status["per_dp_kv_cache"], utilizations):
                    self.assertAlmostEqual(shard["utilization"], utilization)
                self.assertEqual(status["concurrency_limit"], 4)
                self.assertEqual(status["available_concurrency"], 1)
                self.assertEqual(status["running_streams"], 2)
                self.assertEqual(status["waiting_streams"], 1)

    def test_analyze_multi_return_counts_input_once_and_runtime_samples(self):
        runner = OfflineRunner.__new__(OfflineRunner)
        runner._config = OfflineBenchConfig(num_return_sequences=2)
        runner._status_samples = [
            {
                "kv_utilization": 0.2,
                "kv_cache_max_dp_utilization": 0.8,
                "concurrency_utilization": 0.25,
            },
            {
                "kv_utilization": 0.6,
                "kv_cache_max_dp_utilization": 0.7,
                "concurrency_utilization": 0.75,
            },
        ]
        records = [
            _RequestRecord(
                0,
                2,
                {
                    "aux_info": [
                        {
                            "input_len": 100,
                            "output_len": 3,
                            "cost_time": 500,
                            "first_token_cost_time": 100,
                        },
                        {
                            "input_len": 100,
                            "output_len": 5,
                            "cost_time": 600,
                            "first_token_cost_time": 120,
                        },
                    ]
                },
            ),
            _RequestRecord(0, 2, {"aux_info": [{"input_len": 100, "output_len": 3}]}),
        ]
        metrics = runner._analyze(
            records,
            wall_time=2,
            engine_status={
                "total_kv_cache": 400,
                "block_size": 10,
                "total_kv_cache_blocks": 40,
            },
            submitted_count=3,
            cancelled_count=1,
        )
        self.assertEqual(metrics.success_requests, 1)
        self.assertEqual(metrics.fail_requests, 1)
        self.assertEqual(metrics.cancelled_requests, 1)
        self.assertEqual(metrics.input_tps, 50)
        self.assertEqual(metrics.output_tps, 4)
        self.assertEqual(metrics.total_tps, 54)
        self.assertEqual(metrics.avg_input_len, 100)
        self.assertEqual(metrics.avg_output_len, 8)
        self.assertEqual(metrics.avg_tpot_ms, 160)
        self.assertEqual(metrics.avg_ttft_ms, 120)
        self.assertEqual(metrics.kv_cache_total_tokens, 400)
        self.assertEqual(metrics.kv_cache_total_blocks, 40)
        self.assertAlmostEqual(metrics.kv_cache_avg_utilization, 0.4)
        self.assertAlmostEqual(metrics.kv_cache_min_utilization, 0.2)
        self.assertAlmostEqual(metrics.kv_cache_max_utilization, 0.6)
        self.assertAlmostEqual(metrics.kv_cache_max_dp_utilization, 0.8)
        self.assertAlmostEqual(metrics.concurrency_avg_utilization, 0.5)
        self.assertAlmostEqual(metrics.concurrency_max_utilization, 0.75)


class TestOfflineFixedDispatch(unittest.TestCase):
    class ClientSession:
        def __init__(self, *, connector, **kwargs):
            self.connector = connector

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            await self.connector.close()

    class Generator:
        def __init__(self):
            self.count = 0
            self.lock = threading.Lock()

        def next(self):
            with self.lock:
                self.count += 1
                return f"prompt-{self.count}", 1

    def make_runner(self, send_one):
        runner = OfflineRunner.__new__(OfflineRunner)
        runner._config = OfflineBenchConfig()
        runner._port = 0
        runner._profile = False
        runner._status_samples = []
        runner._status_sample_errors = 0

        async def status_sampler(**kwargs):
            await asyncio.Event().wait()

        runner._status_sampler = status_sampler
        runner._send_one = send_one
        return runner

    def test_fixed_dispatch_submits_full_workload_before_drain_timeout(self):
        async def scenario():
            started = []
            drain_started = asyncio.Event()

            async def send_one(session, prompt, output_len):
                started.append(prompt)
                if prompt == "prompt-3":
                    await asyncio.Event().wait()
                return {"aux_info": [{"input_len": 1, "output_len": 1}]}

            runner = self.make_runner(send_one)
            real_drain = runner._drain_pending

            async def drain(pending, timeout):
                self.assertEqual(started, ["prompt-1", "prompt-2"])
                drain_started.set()
                return await real_drain(pending, timeout)

            runner._drain_pending = drain
            with patch(
                "rtp_llm.test.perf_test.offline_runner.aiohttp.ClientSession",
                self.ClientSession,
            ):
                result = await asyncio.wait_for(
                    runner._dispatch_fixed(
                        self.Generator(),
                        concurrency_limit=1,
                        total_requests=3,
                        drain_timeout_s=0.01,
                    ),
                    timeout=2,
                )
            self.assertTrue(drain_started.is_set())
            self.assertEqual(started, ["prompt-1", "prompt-2", "prompt-3"])
            return result

        result = asyncio.run(scenario())
        self.assertEqual(result.submitted_count, 3)
        self.assertEqual(len(result.records), 2)
        self.assertEqual(result.cancelled_count, 1)
        self.assertGreaterEqual(
            result.terminal_time, max(r.finish_time for r in result.records)
        )

    def test_cancelled_first_request_is_in_dispatch_window(self):
        async def scenario():
            async def send_one(session, prompt, output_len):
                if prompt == "prompt-1":
                    await asyncio.Event().wait()
                return {"aux_info": [{"input_len": 1, "output_len": 1}]}

            runner = self.make_runner(send_one)
            with patch(
                "rtp_llm.test.perf_test.offline_runner.aiohttp.ClientSession",
                self.ClientSession,
            ):
                return await asyncio.wait_for(
                    runner._dispatch_fixed(
                        self.Generator(),
                        concurrency_limit=2,
                        total_requests=2,
                        drain_timeout_s=0.01,
                    ),
                    timeout=2,
                )

        result = asyncio.run(scenario())
        self.assertEqual(result.submitted_count, 2)
        self.assertEqual(result.cancelled_count, 1)
        self.assertEqual(len(result.records), 1)
        self.assertLess(result.first_submit_time, result.records[0].submit_time)
        self.assertGreaterEqual(result.terminal_time, result.records[0].finish_time)

    def test_fixed_dispatch_replenishes_without_exceeding_concurrency(self):
        async def scenario():
            starts = asyncio.Queue()
            releases = asyncio.Queue()
            active = 0
            levels = []

            async def send_one(session, prompt, output_len):
                nonlocal active
                active += 1
                levels.append(active)
                starts.put_nowait(prompt)
                await releases.get()
                active -= 1
                return {"aux_info": [{"input_len": 1, "output_len": 1}]}

            runner = self.make_runner(send_one)
            with patch(
                "rtp_llm.test.perf_test.offline_runner.aiohttp.ClientSession",
                self.ClientSession,
            ):
                task = asyncio.create_task(
                    runner._dispatch_fixed(
                        self.Generator(),
                        concurrency_limit=2,
                        total_requests=5,
                        drain_timeout_s=0,
                    )
                )
                try:
                    for _ in range(2):
                        await asyncio.wait_for(starts.get(), timeout=2)
                    for _ in range(3):
                        releases.put_nowait(None)
                        await asyncio.wait_for(starts.get(), timeout=2)
                    for _ in range(2):
                        releases.put_nowait(None)
                    result = await asyncio.wait_for(task, timeout=2)
                finally:
                    if not task.done():
                        task.cancel()
                        await asyncio.gather(task, return_exceptions=True)
            return result, levels

        result, levels = asyncio.run(scenario())
        self.assertEqual(levels, [1, 2, 2, 2, 2])
        self.assertEqual(result.submitted_count, 5)
        self.assertEqual(len(result.records), 5)
        self.assertEqual(result.cancelled_count, 0)


if __name__ == "__main__":
    unittest.main()
