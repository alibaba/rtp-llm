import argparse
import ast
import os
import shlex
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from rtp_llm.test.perf_test import batch_decode_test, batch_perf_impl, grid_runner
from rtp_llm.test.perf_test.dataclass import ResponseInfo, TestResultMetrics
from rtp_llm.test.perf_test.server import EngineServer


class GridRunnerTest(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {}, clear=True)
        env.start()
        self.addCleanup(env.stop)
        for name in ("BatchPerfImpl", "create_metrics_table", "tqdm"):
            mock = patch.object(grid_runner, name)
            setattr(self, name, mock.start())
            self.addCleanup(mock.stop)
        self.BatchPerfImpl.return_value.run.return_value = TestResultMetrics(8, 8, 0)

    def runner(self, batches, lengths):
        return grid_runner.GridRunner(
            8088, 8, batches, lengths, {seq: str(seq) for seq in lengths}
        )

    def cells_sent(self):
        return [
            (call.args[2] // 8, int(call.args[3]))
            for call in self.BatchPerfImpl.call_args_list
        ]

    def test_capacity_guard_is_disabled_by_default(self):
        self.assertIsNone(grid_runner._kv_skip_reason(128, 131072))

    def test_capacity_guard_uses_local_batch_and_rounds_up(self):
        with patch.dict(
            os.environ,
            {
                "PERF_KV_TOTAL_BLOCKS": "10",
                "PERF_KV_GROUP_RESERVE": "2",
                "PERF_KV_SEQ_SIZE_PER_BLOCK": "256",
            },
        ):
            self.assertIsNone(grid_runner._kv_skip_reason(4, 257))
            self.assertIsNotNone(grid_runner._kv_skip_reason(5, 257))

    def test_reproduction_grid_keeps_fourteen_cells_and_skips_two(self):
        batches = [1, 2, 4, 8, 16, 32, 64, 128]
        lengths = [32768, 131072]
        with patch.dict(os.environ, {"PERF_KV_TOTAL_BLOCKS": "25000"}):
            runner = self.runner(batches, lengths)
            with patch.object(runner, "warmup"):
                metrics = runner.run()
        expected = [(bs, seq) for seq in lengths for bs in batches]
        self.assertEqual(self.cells_sent(), expected[:-2])
        self.assertEqual(len(metrics), 16)
        self.assertEqual([m.metrics.total_requests for m in metrics[-2:]], [0, 0])

    def test_failure_skip_is_sequence_local(self):
        self.BatchPerfImpl.return_value.run.side_effect = [
            TestResultMetrics(8, 8, 0),
            TestResultMetrics(16, 0, 16),
            TestResultMetrics(8, 8, 0),
            TestResultMetrics(16, 16, 0),
            TestResultMetrics(32, 32, 0),
        ]
        with patch.dict(os.environ, {"PERF_SKIP_ON_FAIL": "1"}):
            runner = self.runner([1, 2, 4], [32768, 131072])
            with patch.object(runner, "warmup"):
                metrics = runner.run()
        self.assertEqual(
            self.cells_sent(),
            [(1, 32768), (2, 32768), (1, 131072), (2, 131072), (4, 131072)],
        )
        self.assertEqual(metrics[1].metrics.fail_requests, 16)
        self.assertEqual(metrics[2].metrics.total_requests, 0)

    def test_failure_skip_is_opt_in(self):
        self.BatchPerfImpl.return_value.run.return_value = TestResultMetrics(8, 0, 8)
        runner = self.runner([1, 2, 4], [32768])
        with patch.object(runner, "warmup"):
            runner.run()
        self.assertEqual(len(self.cells_sent()), 3)

    def test_unsorted_batches_do_not_skip_smaller_supported_cells(self):
        with patch.dict(os.environ, {"PERF_KV_TOTAL_BLOCKS": "25000"}):
            runner = self.runner([64, 1, 128, 32], [131072])
            with patch.object(runner, "warmup"):
                runner.run()
        self.assertEqual(self.cells_sent(), [(1, 131072), (32, 131072)])

    def test_warmup_uses_a_supported_sequence(self):
        with patch.dict(os.environ, {"PERF_KV_TOTAL_BLOCKS": "1700"}):
            self.runner([1], [131072, 32768]).warmup()
        self.assertEqual(self.cells_sent(), [(1, 32768)])

    def test_warmup_does_not_submit_when_no_sequence_fits(self):
        with patch.dict(os.environ, {"PERF_KV_TOTAL_BLOCKS": "1"}):
            self.runner([1], [32768]).run()
        self.BatchPerfImpl.assert_not_called()

    def test_disabled_warmup_does_not_update_scheduler(self):
        with patch.dict(os.environ, {"PERF_GRID_WARMUP_RUNS": "0"}):
            self.runner([1], [32768]).warmup()
        self.BatchPerfImpl.assert_not_called()


class BatchPerfTest(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {}, clear=True)
        env.start()
        self.addCleanup(env.stop)
        pool = patch.object(batch_perf_impl, "ProcessPoolExecutor")
        pool.start()
        self.addCleanup(pool.stop)
        self.batch = batch_perf_impl.BatchPerfImpl(
            8088, 8, 8, "query", warmup_runs=2, measure_runs=1, profile_runs=1
        )
        self.batch._set_concurrency = MagicMock()
        self.batch._curl_server = MagicMock(return_value=TestResultMetrics(8, 0, 8))
        self.batch._curl_server_responses = MagicMock(return_value=[ResponseInfo({})])
        flush = patch.object(batch_perf_impl, "_wait_profile_flush")
        self.flush = flush.start()
        self.addCleanup(flush.stop)

    def test_failed_warmup_skips_measurement_and_profile(self):
        with patch.dict(os.environ, {"PERF_SKIP_ON_FAIL": "1"}):
            metrics = self.batch.run()
        self.assertEqual(metrics.total_requests, 0)
        self.batch._curl_server.assert_called_once_with()
        self.batch._curl_server_responses.assert_not_called()
        self.flush.assert_not_called()

    def test_failed_warmup_continues_when_skip_disabled(self):
        metrics = self.batch.run()
        self.assertEqual(metrics.success_requests, 1)
        self.batch._curl_server_responses.assert_called_once_with()
        self.batch._curl_server.assert_called_with(True)
        self.flush.assert_called_once_with("", 8)

    def test_successful_warmup_preserves_measurement_and_profile(self):
        self.batch._curl_server.return_value = TestResultMetrics(8, 8, 0)
        with patch.dict(os.environ, {"PERF_SKIP_ON_FAIL": "1"}):
            metrics = self.batch.run()
        self.assertEqual(metrics.success_requests, 1)
        self.assertEqual(self.batch._curl_server.call_count, 3)
        self.flush.assert_called_once()


class ProfileFlushTest(unittest.TestCase):
    def setUp(self):
        env = patch.dict(
            os.environ,
            {"TORCH_CUDA_PROFILER_DIR": "/traces", "PERF_PROFILE_FLUSH_SLEEP": "1"},
            clear=True,
        )
        env.start()
        self.addCleanup(env.stop)
        clock = patch.object(batch_perf_impl.time, "monotonic")
        self.clock = clock.start()
        self.addCleanup(clock.stop)
        self.now = 0.0
        self.clock.side_effect = lambda: self.now
        sleep = patch.object(batch_perf_impl.time, "sleep", side_effect=self.advance)
        self.sleep = sleep.start()
        self.addCleanup(sleep.stop)
        files = patch.object(batch_perf_impl.glob, "glob", return_value=[])
        self.files = files.start()
        self.addCleanup(files.stop)
        size = patch.object(batch_perf_impl.os.path, "getsize", return_value=2048)
        self.size = size.start()
        self.addCleanup(size.stop)

    def advance(self, duration):
        self.now += duration

    def test_ready_exports_return_before_timeout(self):
        self.files.return_value = [f"/traces/cell_wr{rank}.json" for rank in range(8)]
        batch_perf_impl._wait_profile_flush("cell", 8)
        self.assertAlmostEqual(self.now, 0.6)
        self.files.assert_called_with("/traces/cell_wr*.json")

    def test_world_size_overrides_dp_for_tensor_parallel_profiles(self):
        self.files.return_value = ["/traces/cell_wr0.json"]
        with patch.dict(os.environ, {"WORLD_SIZE": "4"}):
            with self.assertLogs(level="WARNING") as logs:
                batch_perf_impl._wait_profile_flush("cell", 1)
        self.assertIn("expected=4 got=1", logs.output[0])

    def test_explicit_rank_count_has_priority(self):
        self.files.return_value = ["/traces/cell_wr0.json"]
        with patch.dict(os.environ, {"WORLD_SIZE": "4", "PERF_PROFILE_RANKS": "1"}):
            batch_perf_impl._wait_profile_flush("cell", 8)
        self.assertAlmostEqual(self.now, 0.6)

    def test_incomplete_exports_timeout(self):
        self.files.return_value = ["/traces/cell_wr0.json"]
        self.size.return_value = 0
        with self.assertLogs(level="WARNING"):
            batch_perf_impl._wait_profile_flush("cell", 1)
        self.assertGreaterEqual(self.now, 1)

    def test_unnamed_profile_uses_timeout_fallback(self):
        batch_perf_impl._wait_profile_flush("", 8)
        self.sleep.assert_called_once_with(1.0)
        self.files.assert_not_called()

    def test_zero_timeout_disables_wait(self):
        with patch.dict(os.environ, {"PERF_PROFILE_FLUSH_SLEEP": "0"}):
            batch_perf_impl._wait_profile_flush("cell", 8)
        self.sleep.assert_not_called()


class PerfConfigurationTest(unittest.TestCase):
    def test_concurrency_is_per_rank_and_capped(self):
        args = argparse.Namespace(dp_size=8, concurrency_limit=128)
        self.assertEqual(batch_decode_test._local_concurrency(args, 128), 128)
        self.assertEqual(batch_decode_test._local_concurrency(args, 16), 16)
        args.concurrency_limit = 32
        self.assertEqual(batch_decode_test._local_concurrency(args, 128), 32)

    def test_all_sm120_targets_accept_supplied_checkpoint_paths(self):
        build = ast.parse(Path(__file__).with_name("BUILD").read_text())
        targets = []
        for node in build.body:
            if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
                continue
            kwargs = {kw.arg: kw.value for kw in node.value.keywords}
            if "name" in kwargs and ast.literal_eval(kwargs["name"]).startswith(
                "v4_flash_sm120_"
            ):
                targets.append(kwargs)
        self.assertEqual(len(targets), 4)
        for target in targets:
            argv = ast.literal_eval(target["args"])
            name = ast.literal_eval(target["name"])
            with self.subTest(target=name), patch.dict(os.environ, {}, clear=True):
                for key in ("checkpoint_path", "tokenizer_path", "sp_checkpoint_path"):
                    self.assertNotIn(f"--{key}", argv)
                argv += ["--checkpoint_path=/models/target checkpoint"]
                if name.endswith("_mtp"):
                    argv += ["--sp_checkpoint_path=/models/draft checkpoint"]
                with patch.object(sys, "argv", ["perf_test"] + argv):
                    args, remaining = batch_decode_test.parse_args()
                resolved = batch_decode_test.resolve_perf_engine_paths(remaining)
                EngineServer.propagate_engine_env(resolved)
                self.assertEqual(
                    os.environ["CHECKPOINT_PATH"], "/models/target checkpoint"
                )
                self.assertEqual(
                    os.environ["TOKENIZER_PATH"], "/models/target checkpoint"
                )
                cli = EngineServer(args, resolved)._build_engine_cli(131200, 128)
                self.assertEqual(
                    shlex.split(cli),
                    resolved
                    + [
                        "--dp_size",
                        str(args.dp_size),
                        "--max_seq_len",
                        "131200",
                        "--concurrency_limit",
                        "128",
                    ],
                )

    def test_explicit_tokenizer_is_preserved(self):
        with patch.dict(os.environ, {}, clear=True):
            EngineServer.propagate_engine_env(
                [
                    "--checkpoint_path",
                    "/models/target",
                    "--tokenizer_path",
                    "/models/tok",
                ]
            )
            self.assertEqual(os.environ["TOKENIZER_PATH"], "/models/tok")

    def test_server_and_profiler_use_same_output_directory(self):
        args = argparse.Namespace(partial=1, result_dir="/traces", dp_size=8)
        with patch.dict(os.environ, {}, clear=True):
            with patch("rtp_llm.test.perf_test.server.MagaServerManager") as manager:
                EngineServer(args, []).start(131200, 128)
                self.assertEqual(os.environ["TORCH_CUDA_PROFILER_DIR"], "/traces")
                self.assertEqual(
                    manager.call_args.kwargs["env_args"]["TORCH_CUDA_PROFILER_DIR"],
                    "/traces",
                )


if __name__ == "__main__":
    unittest.main()
