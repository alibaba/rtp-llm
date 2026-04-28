"""Unit tests for perf test configuration, runner dispatch, and TPS binary search.

Tests:
- PerfTestConfig generation (grid/distribution, auto BS, TPS concurrency bump)
- Runner dispatch (_run_prefill, _run_decode for 4 modes)
- Helper functions (auto BS generation, KV cache filtering)
- TPS binary search BS candidate generation
"""

import argparse
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


if __name__ == "__main__":
    unittest.main()
