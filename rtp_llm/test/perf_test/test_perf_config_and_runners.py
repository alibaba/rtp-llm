"""Unit tests for perf test configuration, runner dispatch, and TPS binary search.

Tests:
- PerfTestConfig generation (grid/distribution, auto BS, TPS concurrency bump)
- Runner dispatch (_run_prefill, _run_decode for 4 modes)
- Helper functions (auto BS generation, KV cache filtering)
- TPS binary search BS candidate generation
"""

import argparse
import asyncio
import sys
import unittest
from unittest.mock import MagicMock, patch

from rtp_llm.test.perf_test.batch_decode_test import _run_decode, _run_prefill
from rtp_llm.test.perf_test.dataclass import PerfTestConfig
from rtp_llm.test.perf_test.perf_config import prepare_config as _prepare_config
from rtp_llm.test.perf_test.perf_utils import (
    auto_generate_bs_list as _auto_generate_bs_list,
)
from rtp_llm.test.perf_test.perf_utils import (
    filter_bs_by_kvcache as _filter_bs_by_kvcache,
)
from rtp_llm.test.perf_test.tps_runner import TpsBinarySearchRunner


def _make_args(**overrides):
    """Create a minimal argparse.Namespace with sensible defaults."""
    defaults = dict(
        batch_size="1,8,16",
        input_len="1024,4096",
        dataset_name="",
        dataset_path="",
        dataset="",
        test_json="",
        partial=0,
        generate_config="{}",
        result_dir="/tmp/test_result",
        decode_test_length=10,
        target_tpot=0,
        dp_size=1,
        max_seq_len=8192,
        concurrency_limit=64,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


# ---------------------------------------------------------------------------
# _prepare_config
# ---------------------------------------------------------------------------
class TestPrepareConfig(unittest.TestCase):
    """Test _prepare_config for grid and distribution modes."""

    def test_grid_mode_explicit_bs(self):
        with patch.object(sys, "argv", ["prog", "--batch_size", "1,4,8"]):
            args = _make_args(batch_size="1,4,8", input_len="128,256")
            config = _prepare_config(args, [])

        self.assertFalse(config.is_distribution)
        self.assertEqual(config.batch_size_list, [1, 4, 8])
        self.assertEqual(config.input_len_list, [128, 256])
        self.assertEqual(config.all_seq_lens, [128, 256])
        # max(input_len) + decode_test_length
        self.assertEqual(config.max_seq_len, 256 + 10)
        self.assertEqual(config.max_concurrency, 8)
        self.assertIsNone(config.test_config)

    def test_grid_mode_auto_bs(self):
        with patch.object(sys, "argv", ["prog"]):
            args = _make_args(concurrency_limit=32)
            config = _prepare_config(args, [])

        expected_bs = _auto_generate_bs_list(32)
        self.assertEqual(config.batch_size_list, expected_bs)
        self.assertEqual(config.max_concurrency, max(expected_bs))

    def test_grid_mode_tps_bumps_concurrency(self):
        with patch.object(sys, "argv", ["prog", "--batch_size", "1,8"]):
            args = _make_args(batch_size="1,8", target_tpot=30, concurrency_limit=64)
            config = _prepare_config(args, [])

        # max(8, concurrency_limit=64) = 64
        self.assertEqual(config.max_concurrency, 64)

    @patch("rtp_llm.test.perf_test.perf_config.prepare_distribution_config")
    def test_distribution_mode(self, mock_prepare):
        mock_prepare.return_value = {
            "batch_seq_len_map": {"4": [128, 256], "8": [128, 512]},
            "source": "test",
        }
        args = _make_args(dataset_name="sharegpt", max_seq_len=2048)
        config = _prepare_config(args, ["--tokenizer_path", "/tok"])

        self.assertTrue(config.is_distribution)
        self.assertEqual(config.all_seq_lens, [128, 256, 512])
        self.assertEqual(config.batch_size_list, [])
        self.assertEqual(config.input_len_list, [])
        # max(512 + 10, 2048) = 2048
        self.assertEqual(config.max_seq_len, 2048)
        self.assertEqual(config.max_concurrency, 8)
        self.assertIsNotNone(config.test_config)

    @patch("rtp_llm.test.perf_test.perf_config.prepare_distribution_config")
    def test_distribution_max_seq_len_expands(self, mock_prepare):
        """max_seq_len should expand when needed_seq_len > args.max_seq_len."""
        mock_prepare.return_value = {
            "batch_seq_len_map": {"2": [4000, 8000]},
        }
        args = _make_args(
            dataset_name="sharegpt", max_seq_len=2048, decode_test_length=10
        )
        config = _prepare_config(args, [])

        # needed = 8000 + 10 = 8010 > 2048
        self.assertEqual(config.max_seq_len, 8010)


# ---------------------------------------------------------------------------
# _run_prefill
# ---------------------------------------------------------------------------
class TestRunPrefill(unittest.TestCase):
    @patch("rtp_llm.test.perf_test.batch_decode_test.GridRunner")
    def test_grid_prefill(self, MockGridRunner):
        mock_runner = MagicMock()
        MockGridRunner.return_value = mock_runner

        config = PerfTestConfig(
            is_distribution=False,
            all_seq_lens=[128, 256],
            batch_size_list=[1, 8],
            input_len_list=[128, 256],
            max_seq_len=266,
            max_concurrency=8,
        )
        query_dict = {128: "q128", 256: "q256"}
        _run_prefill(8000, 1, config, query_dict, dump_json_path="/tmp")

        MockGridRunner.assert_called_once_with(
            8000,
            1,
            [1],
            [128, 256],
            query_dict,
            is_decode=False,
            dump_json_path="/tmp",
        )
        mock_runner.run.assert_called_once()

    @patch("rtp_llm.test.perf_test.batch_decode_test.GridRunner")
    def test_distribution_prefill_skipped(self, MockGridRunner):
        """Distribution mode: prefill is skipped (empty input_len_list)."""
        config = PerfTestConfig(
            is_distribution=True,
            all_seq_lens=[128, 256],
            batch_size_list=[],
            input_len_list=[],
            max_seq_len=266,
            max_concurrency=8,
        )
        _run_prefill(8000, 1, config, {128: "q"})
        MockGridRunner.assert_not_called()


# ---------------------------------------------------------------------------
# _run_decode — 4 modes
# ---------------------------------------------------------------------------
class TestRunDecode(unittest.TestCase):
    """Verify the correct runner is chosen for each of the 4 modes."""

    def _grid_config(self):
        return PerfTestConfig(
            is_distribution=False,
            all_seq_lens=[128, 256],
            batch_size_list=[1, 8, 16],
            input_len_list=[128, 256],
            max_seq_len=266,
            max_concurrency=16,
        )

    def _dist_config(self):
        return PerfTestConfig(
            is_distribution=True,
            all_seq_lens=[128, 256, 512],
            batch_size_list=[],
            input_len_list=[],
            max_seq_len=522,
            max_concurrency=8,
            test_config={"batch_seq_len_map": {"4": [128, 256], "8": [128, 512]}},
        )

    # --- Mode 1: decode × grid ---
    @patch("rtp_llm.test.perf_test.batch_decode_test.GridRunner")
    def test_decode_grid(self, MockGridRunner):
        mock_runner = MagicMock()
        MockGridRunner.return_value = mock_runner

        args = _make_args(target_tpot=0)
        config = self._grid_config()
        query_dict = {128: "q128", 256: "q256"}
        # max_kv_tokens=2048: 16*128=2048 fits, 16*256=4096 doesn't
        engine_meta = {"max_kv_tokens": 2048}

        _run_decode(
            8000, 1, args, config, query_dict, engine_meta, dump_json_path="/tmp"
        )

        self.assertEqual(MockGridRunner.call_count, 2)
        # input_len=128: all BS fit
        c1 = MockGridRunner.call_args_list[0]
        self.assertEqual(c1[0][2], [1, 8, 16])
        self.assertEqual(c1[0][3], [128])
        self.assertTrue(c1[1]["is_decode"])
        # input_len=256: only [1, 8]
        c2 = MockGridRunner.call_args_list[1]
        self.assertEqual(c2[0][2], [1, 8])
        self.assertEqual(c2[0][3], [256])

    @patch("rtp_llm.test.perf_test.batch_decode_test.GridRunner")
    def test_decode_grid_kv_skip(self, MockGridRunner):
        """All BS filtered out -> input_len is skipped entirely."""
        args = _make_args(target_tpot=0)
        config = PerfTestConfig(
            is_distribution=False,
            all_seq_lens=[4096],
            batch_size_list=[8, 16],
            input_len_list=[4096],
            max_seq_len=4106,
            max_concurrency=16,
        )
        # 8*4096=32768 > 1000
        _run_decode(
            8000, 1, args, config, {}, {"max_kv_tokens": 1000}, dump_json_path="/tmp"
        )
        MockGridRunner.assert_not_called()

    # --- Mode 2: decode × distribution ---
    @patch("rtp_llm.test.perf_test.batch_decode_test.DistributionRunner")
    def test_decode_distribution(self, MockDistRunner):
        mock_runner = MagicMock()
        MockDistRunner.return_value = mock_runner

        args = _make_args(target_tpot=0)
        config = self._dist_config()
        query_dict = {128: "q", 256: "q", 512: "q"}

        _run_decode(8000, 1, args, config, query_dict, None, dump_json_path="/tmp")

        MockDistRunner.assert_called_once()
        # test_config and input_query_dict passed to constructor
        call_args = MockDistRunner.call_args
        self.assertEqual(call_args[0][2], config.test_config)
        self.assertEqual(call_args[0][3], query_dict)
        mock_runner.run.assert_called_once()

    # --- Mode 3: tps × grid ---
    @patch("rtp_llm.test.perf_test.batch_decode_test.TpsBinarySearchRunner")
    def test_tps_grid(self, MockTpsRunner):
        mock_runner = MagicMock()
        MockTpsRunner.return_value = mock_runner

        args = _make_args(target_tpot=30, concurrency_limit=64)
        config = self._grid_config()
        query_dict = {128: "q128", 256: "q256"}
        # 16*256=4096 > 2048 so 256 only fits bs up to 8
        engine_meta = {"max_kv_tokens": 2048}

        _run_decode(
            8000, 1, args, config, query_dict, engine_meta, dump_json_path="/tmp"
        )

        MockTpsRunner.assert_called_once_with(
            8000,
            1,
            30,
            max_bs=64,
            dump_json_path="/tmp",
        )
        mock_runner.run_grid.assert_called_once()
        grid_call = mock_runner.run_grid.call_args
        self.assertEqual(grid_call[0][0], [128, 256])
        self.assertEqual(grid_call[0][1], query_dict)
        # max_bs_per_len: 128 -> max([1,8,16] all <=2048) = 16; 256 -> max([1,8]) = 8
        self.assertEqual(grid_call[0][2], {128: 16, 256: 8})

    # --- Mode 4: tps × distribution ---
    @patch("rtp_llm.test.perf_test.batch_decode_test.TpsBinarySearchRunner")
    def test_tps_distribution(self, MockTpsRunner):
        mock_runner = MagicMock()
        MockTpsRunner.return_value = mock_runner

        args = _make_args(target_tpot=30, concurrency_limit=64)
        config = self._dist_config()
        query_dict = {128: "q", 256: "q", 512: "q"}

        _run_decode(8000, 1, args, config, query_dict, None, dump_json_path="/tmp")

        mock_runner.run_distribution.assert_called_once_with(
            config.test_config,
            query_dict,
        )


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
class TestHelpers(unittest.TestCase):
    def test_auto_generate_bs_list(self):
        bs = _auto_generate_bs_list(128)
        self.assertEqual(bs[0], 1)
        self.assertIn(8, bs)
        self.assertIn(64, bs)
        self.assertIn(128, bs)
        # step should be 64 after 64
        idx_64 = bs.index(64)
        self.assertEqual(bs[idx_64 + 1], 128)

    def test_auto_generate_bs_list_small(self):
        bs = _auto_generate_bs_list(16)
        self.assertEqual(bs, [1, 8, 16])

    def test_filter_bs_by_kvcache(self):
        self.assertEqual(
            _filter_bs_by_kvcache([1, 8, 16, 32], 128, 2048),
            [1, 8, 16],  # 32*128=4096 > 2048
        )

    def test_filter_bs_by_kvcache_none_fit(self):
        self.assertEqual(
            _filter_bs_by_kvcache([8, 16], 4096, 1000),
            [],
        )


# ---------------------------------------------------------------------------
# TPS binary search BS candidates
# ---------------------------------------------------------------------------
class TestTpsBsCandidates(unittest.TestCase):
    def test_default_step4(self):
        candidates = TpsBinarySearchRunner._make_bs_candidates(16)
        self.assertEqual(candidates, [1, 4, 8, 12, 16])

    def test_step4_non_multiple(self):
        candidates = TpsBinarySearchRunner._make_bs_candidates(18)
        # [1, 4, 8, 12, 16, 18] — max_bs appended if not a multiple of step
        self.assertEqual(candidates, [1, 4, 8, 12, 16, 18])

    def test_step4_large(self):
        candidates = TpsBinarySearchRunner._make_bs_candidates(128)
        self.assertEqual(candidates[0], 1)
        self.assertEqual(candidates[1], 4)
        self.assertEqual(candidates[2], 8)
        self.assertIn(64, candidates)
        self.assertEqual(candidates[-1], 128)
        # All gaps should be 4 (except first gap 1->4)
        for i in range(2, len(candidates) - 1):
            self.assertEqual(candidates[i] - candidates[i - 1], 4)

    def test_max_bs_1(self):
        candidates = TpsBinarySearchRunner._make_bs_candidates(1)
        self.assertEqual(candidates, [1])

    def test_max_bs_4(self):
        candidates = TpsBinarySearchRunner._make_bs_candidates(4)
        self.assertEqual(candidates, [1, 4])


class TestOfflineBenchConfig(unittest.TestCase):
    """Lightweight tests for OfflineBenchConfig.validate."""

    def test_validate_ok(self):
        from rtp_llm.test.perf_test.offline_runner import OfflineBenchConfig
        config = OfflineBenchConfig(input_len_min=128, input_len_max=512,
                                    output_len_min=32, output_len_max=128)
        config.validate()

    def test_validate_input_len_inverted(self):
        from rtp_llm.test.perf_test.offline_runner import OfflineBenchConfig
        config = OfflineBenchConfig(input_len_min=512, input_len_max=128)
        with self.assertRaises(ValueError):
            config.validate()

    def test_validate_output_len_zero(self):
        from rtp_llm.test.perf_test.offline_runner import OfflineBenchConfig
        config = OfflineBenchConfig(output_len_min=0)
        with self.assertRaises(ValueError):
            config.validate()

    def test_validate_prefix_len_exceeds_input(self):
        from rtp_llm.test.perf_test.offline_runner import OfflineBenchConfig
        config = OfflineBenchConfig(input_len_min=64, prefix_len=128)
        with self.assertRaises(ValueError):
            config.validate()


class TestWorkloadGenerator(unittest.TestCase):
    """Test WorkloadGenerator prompt generation with a fake tokenizer."""

    def _make_fake_tokenizer(self):
        tok = MagicMock()
        tok.encode = lambda text: list(range(len(text.split())))
        tok.decode = lambda ids: " ".join(str(i) for i in ids)
        return tok

    def test_basic_generation(self):
        from rtp_llm.test.perf_test.offline_runner import OfflineBenchConfig, WorkloadGenerator
        config = OfflineBenchConfig(input_len_min=4, input_len_max=8,
                                    output_len_min=2, output_len_max=4, prefix_len=0)
        gen = WorkloadGenerator(config, self._make_fake_tokenizer(), seed=0)
        prompt, out_len = gen.next()
        self.assertIsInstance(prompt, str)
        self.assertGreaterEqual(out_len, 2)
        self.assertLessEqual(out_len, 4)

    def test_prefix_groups_have_different_prefixes(self):
        from rtp_llm.test.perf_test.offline_runner import OfflineBenchConfig, WorkloadGenerator
        config = OfflineBenchConfig(input_len_min=16, input_len_max=32,
                                    output_len_min=2, output_len_max=4,
                                    prefix_groups=3, prefix_len=4)
        gen = WorkloadGenerator(config, self._make_fake_tokenizer(), seed=0)
        prefixes = [g.prefix_text for g in gen._groups]
        self.assertEqual(len(prefixes), 3)
        self.assertEqual(len(set(prefixes)), 3)

    def test_prefix_groups_wrap_short_token_buffer(self):
        from rtp_llm.test.perf_test.offline_runner import OfflineBenchConfig, WorkloadGenerator

        tok = MagicMock()
        tok.encode = lambda text: [0, 1, 2]
        tok.decode = lambda ids: " ".join(str(i) for i in ids)
        config = OfflineBenchConfig(input_len_min=5, input_len_max=8,
                                    output_len_min=1, output_len_max=1,
                                    prefix_groups=4, prefix_len=5)
        gen = WorkloadGenerator(config, tok, seed=0)

        prefixes = [g.prefix_text.split() for g in gen._groups]
        self.assertEqual([len(p) for p in prefixes], [5, 5, 5, 5])

    def test_long_prompt_generation_does_not_resize_token_buffer(self):
        from rtp_llm.test.perf_test.offline_runner import OfflineBenchConfig, WorkloadGenerator

        tok = MagicMock()
        tok.encode = lambda text: [0, 1, 2]
        tok.decode = lambda ids: " ".join(str(i) for i in ids)
        config = OfflineBenchConfig(input_len_min=20, input_len_max=20,
                                    output_len_min=1, output_len_max=1)
        gen = WorkloadGenerator(config, tok, seed=0)
        initial_len = len(gen._big_tokens)

        for _ in range(5):
            prompt, output_len = gen.next()
            self.assertEqual(output_len, 1)
            self.assertTrue(prompt)
            self.assertEqual(len(gen._big_tokens), initial_len)


class TestOfflineAnalyze(unittest.TestCase):
    """Test OfflineRunner._analyze with synthetic records."""

    def test_analyze_basic(self):
        from rtp_llm.test.perf_test.offline_runner import OfflineBenchConfig, OfflineRunner, _RequestRecord
        config = OfflineBenchConfig(input_len_min=4, input_len_max=8,
                                    output_len_min=2, output_len_max=4)
        runner = OfflineRunner.__new__(OfflineRunner)
        runner._config = config
        runner._result_dir = "/tmp"
        runner._status_samples = []

        records = [
            _RequestRecord(
                submit_time=0.0, finish_time=1.0,
                response={
                    "aux_info": [{
                        "input_len": 100, "output_len": 50,
                        "cost_time": 500.0, "wait_time": 10.0,
                        "first_token_cost_time": 100.0,
                    }]
                },
            ),
            _RequestRecord(
                submit_time=0.5, finish_time=1.5,
                response={
                    "aux_info": [{
                        "input_len": 200, "output_len": 80,
                        "cost_time": 800.0, "wait_time": 20.0,
                        "first_token_cost_time": 150.0,
                    }]
                },
            ),
        ]

        metrics = runner._analyze(records, wall_time=2.0, engine_status={},
                                  submitted_count=2, cancelled_count=0)
        self.assertEqual(metrics.success_requests, 2)
        self.assertEqual(metrics.fail_requests, 0)
        self.assertAlmostEqual(metrics.output_tps, (50 + 80) / 2.0)
        self.assertAlmostEqual(metrics.input_tps, (100 + 200) / 2.0)

    def test_analyze_with_failures(self):
        from rtp_llm.test.perf_test.offline_runner import OfflineBenchConfig, OfflineRunner, _RequestRecord
        config = OfflineBenchConfig()
        runner = OfflineRunner.__new__(OfflineRunner)
        runner._config = config
        runner._result_dir = "/tmp"
        runner._status_samples = []

        records = [
            _RequestRecord(submit_time=0.0, finish_time=1.0,
                          response={"error_code": "MALLOC_FAILED", "message": "OOM"}),
        ]
        metrics = runner._analyze(records, wall_time=1.0, engine_status={},
                                  submitted_count=1, cancelled_count=0)
        self.assertEqual(metrics.success_requests, 0)
        self.assertEqual(metrics.fail_requests, 1)

    def test_dispatch_drains_only_inflight_fast_tasks(self):
        from rtp_llm.test.perf_test.offline_runner import OfflineBenchConfig, OfflineRunner
        import rtp_llm.test.perf_test.offline_runner as offline_runner

        class FakeClientSession:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

        class FakeGenerator:
            def __init__(self):
                self.count = 0

            def next(self):
                self.count += 1
                return "prompt", 1

        async def run_dispatch():
            config = OfflineBenchConfig(input_len_min=1, input_len_max=1,
                                        output_len_min=1, output_len_max=1)
            runner = OfflineRunner.__new__(OfflineRunner)
            runner._config = config
            runner._port = 0
            runner._profile = False
            runner._profile_steps = 1
            runner._status_samples = []
            runner._status_sample_errors = 0

            async def fake_status_sampler(*args, **kwargs):
                try:
                    while True:
                        await asyncio.sleep(3600)
                except asyncio.CancelledError:
                    raise

            async def fake_send_one(session, prompt, output_len):
                await asyncio.sleep(0)
                return {"aux_info": [{"input_len": 1, "output_len": output_len}]}

            runner._status_sampler = fake_status_sampler
            runner._send_one = fake_send_one

            with patch.object(offline_runner.aiohttp, "ClientSession", FakeClientSession):
                return await runner._dispatch(FakeGenerator(), concurrency_limit=2,
                                              duration_s=0.01, drain_timeout_s=1)

        loop = asyncio.new_event_loop()
        try:
            records, cancelled_count, submitted_count = loop.run_until_complete(run_dispatch())
        finally:
            loop.close()
        self.assertEqual(cancelled_count, 0)
        self.assertEqual(len(records), submitted_count)
        self.assertGreater(submitted_count, 2)


class TestParseOfflineArgs(unittest.TestCase):
    """Test parse_offline_args from offline_bench_test."""

    def test_default_args(self):
        from rtp_llm.test.perf_test.offline_bench_test import parse_offline_args
        with patch("sys.argv", ["prog", "--checkpoint_path", "/tmp/model"]):
            args, remaining = parse_offline_args()
            self.assertEqual(args.checkpoint_path, "/tmp/model")
            self.assertEqual(args.input_len_min, 512)

    def test_custom_args(self):
        from rtp_llm.test.perf_test.offline_bench_test import parse_offline_args
        with patch("sys.argv", ["prog", "--checkpoint_path", "/tmp/m",
                                "--input_len_min", "1024", "--duration", "60"]):
            args, remaining = parse_offline_args()
            self.assertEqual(args.input_len_min, 1024)
            self.assertEqual(args.duration, 60)


if __name__ == "__main__":
    unittest.main()
