"""Core performance behavior contracts."""

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
        argv = ["prog", "--checkpoint_path", "/tmp/model", "--concurrency_limit", "2"]
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


if __name__ == "__main__":
    unittest.main()
