"""Unit tests for perf test configuration, runner dispatch, and TPS binary search.

Tests:
- PerfTestConfig generation (grid/distribution, auto BS, TPS concurrency bump)
- Runner dispatch (_run_prefill, _run_decode for 4 modes)
- Helper functions (auto BS generation, KV cache filtering)
- TPS binary search BS candidate generation
"""

import argparse
import asyncio
import os
import sys
import time
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

    def test_grid_mode_explicit_max_seq_len_preserved(self):
        with patch.object(sys, "argv", ["prog", "--max_seq_len", "409600"]):
            args = _make_args(input_len="128", max_seq_len=409600)
            config = _prepare_config(args, [])

        self.assertEqual(config.max_seq_len, 409600)

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

        config = OfflineBenchConfig(
            input_len_min=128,
            input_len_max=512,
            output_len_min=32,
            output_len_max=128,
            concurrency_limit=1,
        )
        config.validate()

    def test_validate_requires_concurrency_limit(self):
        from rtp_llm.test.perf_test.offline_runner import OfflineBenchConfig

        with self.assertRaisesRegex(ValueError, "concurrency_limit must be > 0"):
            OfflineBenchConfig().validate()

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

    def test_validate_num_return_sequences(self):
        from rtp_llm.test.perf_test.offline_runner import OfflineBenchConfig

        with self.assertRaisesRegex(
            ValueError, "num_return_sequences must be >= 1"
        ):
            OfflineBenchConfig(num_return_sequences=0).validate()


class TestOfflineFailurePolicy(unittest.TestCase):
    def test_rejects_zero_submitted_requests(self):
        from rtp_llm.test.perf_test.offline_runner import OfflineMetrics

        with self.assertRaisesRegex(RuntimeError, "no requests were submitted"):
            OfflineMetrics().raise_if_all_requests_failed()

    def test_rejects_zero_successful_requests(self):
        from rtp_llm.test.perf_test.offline_runner import OfflineMetrics

        metrics = OfflineMetrics(
            total_submitted=2, success_requests=0, fail_requests=2
        )
        with self.assertRaisesRegex(RuntimeError, "no requests succeeded"):
            metrics.raise_if_all_requests_failed()

    def test_accepts_partial_failures(self):
        from rtp_llm.test.perf_test.offline_runner import OfflineMetrics

        metrics = OfflineMetrics(
            total_submitted=10,
            success_requests=7,
            fail_requests=2,
            cancelled_requests=1,
        )
        metrics.raise_if_all_requests_failed()


class TestWorkloadGenerator(unittest.TestCase):
    """Test WorkloadGenerator prompt generation with a fake tokenizer."""

    def _make_fake_tokenizer(self):
        tok = MagicMock()
        tok.encode = lambda text: list(range(len(text.split())))
        tok.decode = lambda ids: " ".join(str(i) for i in ids)
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
        prompt, out_len = gen.next()
        self.assertIsInstance(prompt, str)
        self.assertEqual(len(tokenizer.encode(prompt)), 6)
        self.assertGreaterEqual(out_len, 2)
        self.assertLessEqual(out_len, 4)

    def test_prefix_groups_have_different_prefixes(self):
        from rtp_llm.test.perf_test.offline_runner import (
            OfflineBenchConfig,
            WorkloadGenerator,
        )

        config = OfflineBenchConfig(
            input_len_min=16,
            input_len_max=32,
            output_len_min=2,
            output_len_max=4,
            prefix_groups=3,
            prefix_len=4,
        )
        gen = WorkloadGenerator(config, self._make_fake_tokenizer(), seed=0)
        prefixes = [g.prefix_text for g in gen._groups]
        self.assertEqual(len(prefixes), 3)
        self.assertEqual(len(set(prefixes)), 3)

    def test_unrepresentable_exact_length_fails(self):
        from rtp_llm.test.perf_test.offline_runner import (
            OfflineBenchConfig,
            WorkloadGenerator,
        )

        tok = MagicMock()
        tok.encode = lambda text: [0, 1, 2]
        tok.decode = lambda ids: " ".join(str(i) for i in ids)
        config = OfflineBenchConfig(
            input_len_min=5,
            input_len_max=8,
            output_len_min=1,
            output_len_max=1,
            prefix_groups=4,
            prefix_len=5,
        )
        with patch.object(WorkloadGenerator, "_MAX_TOKEN_REPAIR_ATTEMPTS", 2):
            with self.assertRaisesRegex(ValueError, "exact tokenizer-validated"):
                WorkloadGenerator(config, tok, seed=0)

    def test_long_prompt_generation_does_not_resize_token_buffer(self):
        from rtp_llm.test.perf_test.offline_runner import (
            OfflineBenchConfig,
            WorkloadGenerator,
        )

        tok = self._make_fake_tokenizer()
        config = OfflineBenchConfig(
            input_len_min=20, input_len_max=20, output_len_min=1, output_len_max=1
        )
        gen = WorkloadGenerator(config, tok, seed=0)
        initial_len = len(gen._big_tokens)

        for _ in range(5):
            prompt, output_len = gen.next()
            self.assertEqual(output_len, 1)
            self.assertTrue(prompt)
            self.assertEqual(len(tok.encode(prompt)), 20)
            self.assertEqual(len(gen._big_tokens), initial_len)

    def test_group_requests_share_exact_token_prefix(self):
        from rtp_llm.test.perf_test.offline_runner import (
            OfflineBenchConfig,
            WorkloadGenerator,
        )

        tokenizer = self._make_fake_tokenizer()
        config = OfflineBenchConfig(
            input_len_min=12,
            input_len_max=12,
            output_len_min=1,
            output_len_max=1,
            prefix_groups=3,
            prefix_len=4,
        )
        gen = WorkloadGenerator(config, tokenizer, seed=0)

        for request_id in range(9):
            prompt, _ = gen.next()
            token_ids = tokenizer.encode(prompt)
            group = gen._groups[request_id % config.prefix_groups]
            self.assertEqual(len(token_ids), config.input_len_min)
            self.assertEqual(
                tuple(token_ids[: config.prefix_len]),
                group.prefix_token_ids,
            )

    def test_boundary_retokenization_is_repaired(self):
        from rtp_llm.test.perf_test.offline_runner import (
            OfflineBenchConfig,
            WorkloadGenerator,
        )

        class BoundaryMergingTokenizer:
            def __init__(self):
                self.boundary_merge_count = 0

            def encode(self, text):
                result = []
                index = 0
                while index < len(text):
                    if text[index : index + 2] == "xy":
                        result.append(1000)
                        self.boundary_merge_count += 1
                        index += 2
                    else:
                        result.append(ord(text[index]))
                        index += 1
                return result

            def decode(self, token_ids):
                decoded = []
                for token_id in token_ids:
                    if token_id == ord("T"):
                        decoded.append("x")
                    elif token_id == ord("h"):
                        decoded.append("y")
                    elif token_id == 1000:
                        decoded.append("xy")
                    else:
                        decoded.append(chr(token_id))
                return "".join(decoded)

        tokenizer = BoundaryMergingTokenizer()
        config = OfflineBenchConfig(
            input_len_min=6,
            input_len_max=6,
            output_len_min=1,
            output_len_max=1,
            prefix_len=1,
        )
        gen = WorkloadGenerator(config, tokenizer, seed=0)
        prompt, _ = gen.next()
        token_ids = tokenizer.encode(prompt)

        self.assertGreater(tokenizer.boundary_merge_count, 0)
        self.assertEqual(len(token_ids), 6)
        self.assertEqual(tuple(token_ids[:1]), gen._groups[0].prefix_token_ids)

    def test_qwen2_tokenizer_prompts_have_exact_length_and_prefix(self):
        from transformers.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer

        from rtp_llm.test.perf_test.offline_runner import (
            OfflineBenchConfig,
            WorkloadGenerator,
        )

        tokenizer = Qwen2Tokenizer.from_pretrained(
            "rtp_llm/test/tokenizer_test/testdata/qwen2_tokenizer"
        )
        config = OfflineBenchConfig(
            input_len_min=32,
            input_len_max=32,
            output_len_min=1,
            output_len_max=1,
            prefix_groups=2,
            prefix_len=8,
        )
        gen = WorkloadGenerator(config, tokenizer, seed=0)

        for request_id in range(6):
            prompt, _ = gen.next()
            token_ids = tokenizer.encode(prompt)
            group = gen._groups[request_id % config.prefix_groups]
            self.assertEqual(len(token_ids), 32)
            self.assertEqual(tuple(token_ids[:8]), group.prefix_token_ids)


class TestOfflineAnalyze(unittest.TestCase):
    """Test OfflineRunner._analyze with synthetic records."""

    def test_analyze_basic(self):
        from rtp_llm.test.perf_test.offline_runner import (
            OfflineBenchConfig,
            OfflineRunner,
            _RequestRecord,
        )

        config = OfflineBenchConfig(
            input_len_min=4, input_len_max=8, output_len_min=2, output_len_max=4
        )
        runner = OfflineRunner.__new__(OfflineRunner)
        runner._config = config
        runner._result_dir = "/tmp"
        total_kv_tokens = 2_247_680
        available_kv_tokens = 2_231_296
        expected_kv_utilization = 1.0 - available_kv_tokens / total_kv_tokens
        runner._status_samples = [
            {
                "kv_utilization": expected_kv_utilization,
                "kv_cache_max_dp_utilization": expected_kv_utilization,
            }
        ]

        records = [
            _RequestRecord(
                submit_time=0.0,
                finish_time=1.0,
                response={
                    "aux_info": [
                        {
                            "input_len": 100,
                            "output_len": 50,
                            "cost_time": 500.0,
                            "wait_time": 10.0,
                            "first_token_cost_time": 100.0,
                        }
                    ]
                },
            ),
            _RequestRecord(
                submit_time=0.5,
                finish_time=1.5,
                response={
                    "aux_info": [
                        {
                            "input_len": 200,
                            "output_len": 80,
                            "cost_time": 800.0,
                            "wait_time": 20.0,
                            "first_token_cost_time": 150.0,
                        }
                    ]
                },
            ),
        ]

        metrics = runner._analyze(
            records,
            wall_time=2.0,
            engine_status={
                "total_kv_cache": total_kv_tokens,
                "available_kv_cache": available_kv_tokens,
                "block_size": 1024,
            },
            submitted_count=2,
            cancelled_count=0,
        )
        self.assertEqual(metrics.success_requests, 2)
        self.assertEqual(metrics.fail_requests, 0)
        self.assertAlmostEqual(metrics.output_tps, (50 + 80) / 2.0)
        self.assertAlmostEqual(metrics.input_tps, (100 + 200) / 2.0)
        self.assertEqual(metrics.kv_cache_total_tokens, total_kv_tokens)
        self.assertEqual(metrics.kv_cache_total_blocks, 2_195)
        self.assertEqual(metrics.kv_cache_block_size, 1024)
        self.assertAlmostEqual(
            metrics.kv_cache_avg_utilization, expected_kv_utilization
        )
        self.assertAlmostEqual(
            metrics.kv_cache_max_dp_utilization, expected_kv_utilization
        )

    def test_analyze_multi_return_counts_input_once(self):
        from rtp_llm.test.perf_test.offline_runner import (
            OfflineBenchConfig,
            OfflineRunner,
            _RequestRecord,
        )

        runner = OfflineRunner.__new__(OfflineRunner)
        runner._config = OfflineBenchConfig(num_return_sequences=2)
        runner._status_samples = []
        records = [
            _RequestRecord(
                submit_time=0.0,
                finish_time=2.0,
                response={
                    "aux_info": [
                        {
                            "input_len": 100,
                            "output_len": 50,
                            "cost_time": 500.0,
                            "wait_time": 10.0,
                            "first_token_cost_time": 100.0,
                        },
                        {
                            "input_len": 100,
                            "output_len": 60,
                            "cost_time": 600.0,
                            "wait_time": 20.0,
                            "first_token_cost_time": 120.0,
                        },
                    ]
                },
            )
        ]

        metrics = runner._analyze(
            records,
            wall_time=2.0,
            engine_status={},
            submitted_count=1,
            cancelled_count=0,
        )

        self.assertEqual(metrics.success_requests, 1)
        self.assertEqual(metrics.fail_requests, 0)
        self.assertAlmostEqual(metrics.input_tps, 100 / 2.0)
        self.assertAlmostEqual(metrics.output_tps, (50 + 60) / 2.0)
        self.assertEqual(metrics.avg_input_len, 100)
        self.assertEqual(metrics.avg_output_len, 110)
        self.assertAlmostEqual(
            metrics.avg_tpot_ms,
            ((500.0 - 100.0) / 49 + (600.0 - 120.0) / 59) / 2,
        )

    def test_query_status_preserves_cache_status_token_units(self):
        from rtp_llm.test.perf_test.offline_runner import OfflineRunner

        runner = OfflineRunner.__new__(OfflineRunner)
        runner._port = 12345
        cache_response = MagicMock()
        cache_response.json.return_value = {
            "total_kv_cache": "2247680",
            "available_kv_cache": "2231296",
            "block_size": "1024",
            "results": [
                {
                    "total_kv_cache": "2247680",
                    "available_kv_cache": "2231296",
                    "block_size": "1024",
                }
            ],
        }
        worker_response = MagicMock()
        worker_response.json.return_value = {
            "frontend_concurrency_limit": 4,
            "frontend_available_concurrency": 3,
            "running_task_info": [],
        }

        with patch(
            "rtp_llm.test.perf_test.offline_runner.requests.get",
            side_effect=[cache_response, worker_response],
        ):
            status = runner._query_status()

        self.assertEqual(status["total_kv_cache"], 2_247_680)
        self.assertEqual(status["available_kv_cache"], 2_231_296)
        self.assertEqual(status["block_size"], 1024)
        self.assertEqual(len(status["per_dp_kv_cache"]), 1)

    def test_query_status_aggregates_all_dp_cache_shards(self):
        from rtp_llm.test.perf_test.offline_runner import OfflineRunner

        runner = OfflineRunner.__new__(OfflineRunner)
        runner._port = 12345
        cache_response = MagicMock()
        cache_response.json.return_value = {
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
            ],
        }
        worker_response = MagicMock()
        worker_response.json.return_value = {
            "frontend_concurrency_limit": 4,
            "frontend_available_concurrency": 3,
            "running_task_info": [],
        }

        with patch(
            "rtp_llm.test.perf_test.offline_runner.requests.get",
            side_effect=[cache_response, worker_response],
        ):
            status = runner._query_status()

        self.assertEqual(status["total_kv_cache"], 400)
        self.assertEqual(status["available_kv_cache"], 260)
        self.assertEqual(status["total_kv_cache_blocks"], 40)
        self.assertAlmostEqual(status["kv_cache_max_dp_utilization"], 0.8)
        self.assertAlmostEqual(status["per_dp_kv_cache"][0]["utilization"], 0.8)
        self.assertAlmostEqual(status["per_dp_kv_cache"][1]["utilization"], 0.2)

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
            ),
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

    def test_dispatch_drains_only_inflight_fast_tasks(self):
        import rtp_llm.test.perf_test.offline_runner as offline_runner
        from rtp_llm.test.perf_test.offline_runner import (
            OfflineBenchConfig,
            OfflineRunner,
        )

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
            config = OfflineBenchConfig(
                input_len_min=1, input_len_max=1, output_len_min=1, output_len_max=1
            )
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

            with patch.object(
                offline_runner.aiohttp, "ClientSession", FakeClientSession
            ):
                return await runner._dispatch(
                    FakeGenerator(),
                    concurrency_limit=2,
                    duration_s=0.01,
                    drain_timeout_s=1,
                )

        loop = asyncio.new_event_loop()
        try:
            dispatch_result = loop.run_until_complete(run_dispatch())
        finally:
            loop.close()
        self.assertEqual(dispatch_result.cancelled_count, 0)
        self.assertEqual(len(dispatch_result.records), dispatch_result.submitted_count)
        self.assertGreater(dispatch_result.submitted_count, 2)
        self.assertIsNotNone(dispatch_result.first_submit_time)
        self.assertGreaterEqual(
            dispatch_result.terminal_time, dispatch_result.records[-1].finish_time
        )

    def test_duration_report_does_not_wait_for_slow_tokenizer_thread(self):
        import threading

        import rtp_llm.test.perf_test.offline_runner as offline_runner
        from rtp_llm.test.perf_test.offline_runner import (
            OfflineBenchConfig,
            OfflineRunner,
            WorkloadGenerator,
        )

        tokenizer_started = threading.Event()
        release_tokenizer = threading.Event()
        tokenizer_finished = threading.Event()

        class SlowTokenizer:
            def encode(self, text):
                return list(range(len(text.split())))

            def decode(self, token_ids):
                tokenizer_started.set()
                release_tokenizer.wait(timeout=1.0)
                tokenizer_finished.set()
                return " ".join(str(token_id) for token_id in token_ids)

        class FakeClientSession:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

        config = OfflineBenchConfig(
            input_len_min=1,
            input_len_max=1,
            output_len_min=1,
            output_len_max=1,
            duration_s=0.05,
            concurrency_limit=1,
            total_requests=0,
        )
        gen = WorkloadGenerator(config, SlowTokenizer(), seed=0)
        runner = OfflineRunner.__new__(OfflineRunner)
        runner._config = config
        runner._port = 0
        runner._profile = False
        runner._profile_steps = 1
        runner._status_samples = []
        runner._status_sample_errors = 0

        async def fake_status_sampler(*args, **kwargs):
            await asyncio.Event().wait()

        runner._status_sampler = fake_status_sampler

        start = time.perf_counter()
        try:
            with patch.object(
                offline_runner.aiohttp, "ClientSession", FakeClientSession
            ), patch.object(runner, "_query_status", return_value={}):
                metrics = runner._run(gen)
            elapsed = time.perf_counter() - start
        finally:
            release_tokenizer.set()
            self.assertTrue(tokenizer_finished.wait(timeout=1.0))

        self.assertTrue(tokenizer_started.is_set())
        self.assertLess(elapsed, 0.5)
        self.assertEqual(metrics.total_submitted, 0)
        self.assertEqual(metrics.success_requests, 0)

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
                return " ".join(str(token_id) for token_id in token_ids)

        class FakeClientSession:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

        async def run_dispatch():
            config = OfflineBenchConfig(
                input_len_min=1,
                input_len_max=1,
                output_len_min=1,
                output_len_max=1,
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
            return result, gen.generated_ids, dumped_ids

        result, generated_ids, dumped_ids = asyncio.run(run_dispatch())

        self.assertIn(1, generated_ids)
        self.assertEqual(result.submitted_count, 1)
        self.assertEqual(len(result.records), 1)
        self.assertEqual(dumped_ids, [0])

    def test_fixed_dispatch_starts_drain_timeout_after_all_requests_submitted(self):
        import rtp_llm.test.perf_test.offline_runner as offline_runner
        from rtp_llm.test.perf_test.offline_runner import (
            OfflineBenchConfig,
            OfflineRunner,
        )

        class FakeClientSession:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

        class FakeGenerator:
            def next(self):
                return "prompt", 1

        async def run_dispatch():
            runner = OfflineRunner.__new__(OfflineRunner)
            runner._config = OfflineBenchConfig(
                input_len_min=1, input_len_max=1, output_len_min=1, output_len_max=1
            )
            runner._port = 0
            runner._profile = False
            runner._profile_steps = 1
            runner._status_samples = []
            runner._status_sample_errors = 0

            async def fake_status_sampler(*args, **kwargs):
                await asyncio.Event().wait()

            async def slow_send_one(session, prompt, output_len):
                await asyncio.sleep(0.01)
                return {"aux_info": [{"input_len": 1, "output_len": output_len}]}

            runner._status_sampler = fake_status_sampler
            runner._send_one = slow_send_one
            with patch.object(
                offline_runner.aiohttp, "ClientSession", FakeClientSession
            ):
                return await runner._dispatch_fixed(
                    FakeGenerator(),
                    concurrency_limit=1,
                    total_requests=3,
                    drain_timeout_s=0.005,
                )

        loop = asyncio.new_event_loop()
        try:
            dispatch_result = loop.run_until_complete(run_dispatch())
        finally:
            loop.close()

        self.assertEqual(dispatch_result.submitted_count, 3)
        self.assertEqual(len(dispatch_result.records), 2)
        self.assertEqual(dispatch_result.cancelled_count, 1)

    def test_fixed_dispatch_prefetch_maintains_concurrency_with_slow_generator(self):
        import rtp_llm.test.perf_test.offline_runner as offline_runner
        from rtp_llm.test.perf_test.offline_runner import (
            OfflineBenchConfig,
            OfflineRunner,
        )

        concurrency_limit = 3
        replacement_count = 3

        class FakeClientSession:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

        class SlowGenerator:
            def next(self):
                time.sleep(0.2)
                return "prompt", 1

        async def run_dispatch():
            runner = OfflineRunner.__new__(OfflineRunner)
            runner._config = OfflineBenchConfig(
                input_len_min=1, input_len_max=1, output_len_min=1, output_len_max=1
            )
            runner._port = 0
            runner._profile = False
            runner._profile_steps = 1
            runner._status_samples = []
            runner._status_sample_errors = 0

            inflight = 0
            full_event = asyncio.Event()
            release_queue = asyncio.Queue()
            request_start_levels = []

            async def fake_status_sampler(*args, **kwargs):
                await asyncio.Event().wait()

            async def controlled_send_one(session, prompt, output_len):
                nonlocal inflight
                inflight += 1
                request_start_levels.append(inflight)
                if inflight == concurrency_limit:
                    full_event.set()
                await release_queue.get()
                inflight -= 1
                return {"aux_info": [{"input_len": 1, "output_len": output_len}]}

            runner._status_sampler = fake_status_sampler
            runner._send_one = controlled_send_one
            with patch.object(
                offline_runner.aiohttp, "ClientSession", FakeClientSession
            ):
                dispatch_task = asyncio.create_task(
                    runner._dispatch_fixed(
                        SlowGenerator(),
                        concurrency_limit=concurrency_limit,
                        total_requests=concurrency_limit + replacement_count,
                        drain_timeout_s=1,
                    )
                )

                await asyncio.wait_for(full_event.wait(), timeout=2)
                # Give the bounded queue time to prepare one replacement per active request.
                await asyncio.sleep(0.3)

                for _ in range(replacement_count):
                    full_event.clear()
                    release_queue.put_nowait(None)
                    await asyncio.wait_for(full_event.wait(), timeout=0.1)

                for _ in range(concurrency_limit):
                    release_queue.put_nowait(None)
                dispatch_result = await dispatch_task

            return dispatch_result, request_start_levels

        loop = asyncio.new_event_loop()
        try:
            dispatch_result, request_start_levels = loop.run_until_complete(
                run_dispatch()
            )
        finally:
            loop.close()

        self.assertEqual(dispatch_result.submitted_count, 6)
        self.assertEqual(dispatch_result.cancelled_count, 0)
        self.assertEqual(len(dispatch_result.records), 6)
        self.assertEqual(
            request_start_levels[-replacement_count:],
            [concurrency_limit] * replacement_count,
        )

    def test_fixed_dispatch_window_starts_with_cancelled_first_request(self):
        import rtp_llm.test.perf_test.offline_runner as offline_runner
        from rtp_llm.test.perf_test.offline_runner import (
            OfflineBenchConfig,
            OfflineRunner,
        )

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
                return f"prompt-{self.count}", 1

        async def run_dispatch():
            runner = OfflineRunner.__new__(OfflineRunner)
            runner._config = OfflineBenchConfig(
                input_len_min=1, input_len_max=1, output_len_min=1, output_len_max=1
            )
            runner._port = 0
            runner._profile = False
            runner._profile_steps = 1
            runner._status_samples = []
            runner._status_sample_errors = 0

            async def fake_status_sampler(*args, **kwargs):
                await asyncio.Event().wait()

            async def send_first_slow(session, prompt, output_len):
                if prompt == "prompt-1":
                    await asyncio.Event().wait()
                return {"aux_info": [{"input_len": 1, "output_len": output_len}]}

            runner._status_sampler = fake_status_sampler
            runner._send_one = send_first_slow
            with patch.object(
                offline_runner.aiohttp, "ClientSession", FakeClientSession
            ):
                return await runner._dispatch_fixed(
                    FakeGenerator(),
                    concurrency_limit=2,
                    total_requests=2,
                    drain_timeout_s=0.005,
                )

        loop = asyncio.new_event_loop()
        try:
            dispatch_result = loop.run_until_complete(run_dispatch())
        finally:
            loop.close()

        self.assertEqual(dispatch_result.submitted_count, 2)
        self.assertEqual(dispatch_result.cancelled_count, 1)
        self.assertEqual(len(dispatch_result.records), 1)
        self.assertIsNotNone(dispatch_result.first_submit_time)
        self.assertLess(
            dispatch_result.first_submit_time, dispatch_result.records[0].submit_time
        )
        self.assertGreaterEqual(
            dispatch_result.terminal_time, dispatch_result.records[0].finish_time
        )

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


class TestParseOfflineArgs(unittest.TestCase):
    """Test parse_offline_args from offline_bench_test."""

    def test_default_args(self):
        from rtp_llm.test.perf_test.offline_bench_test import parse_offline_args

        with patch(
            "sys.argv",
            [
                "prog",
                "--checkpoint_path",
                "/tmp/model",
                "--concurrency_limit",
                "32",
            ],
        ):
            args, remaining = parse_offline_args()
            self.assertEqual(args.checkpoint_path, "/tmp/model")
            self.assertEqual(args.input_len_min, 512)
            self.assertEqual(args.server_start_timeout, 1600)
            self.assertEqual(args.concurrency_limit, 32)

    def test_concurrency_limit_is_required(self):
        from rtp_llm.test.perf_test.offline_bench_test import parse_offline_args

        with patch("sys.argv", ["prog", "--checkpoint_path", "/tmp/model"]):
            with self.assertRaises(SystemExit):
                parse_offline_args()

    def test_custom_args(self):
        from rtp_llm.test.perf_test.offline_bench_test import parse_offline_args

        with patch(
            "sys.argv",
            [
                "prog",
                "--checkpoint_path",
                "/tmp/m",
                "--concurrency_limit",
                "48",
                "--input_len_min",
                "1024",
                "--duration",
                "60",
                "--server_start_timeout",
                "3600",
                "--num_return_sequences",
                "4",
            ],
        ):
            args, remaining = parse_offline_args()
            self.assertEqual(args.input_len_min, 1024)
            self.assertEqual(args.duration, 60)
            self.assertEqual(args.server_start_timeout, 3600)
            self.assertEqual(args.concurrency_limit, 48)
            self.assertEqual(args.num_return_sequences, 4)

    def test_num_return_sequences_request_payload(self):
        from rtp_llm.test.perf_test.offline_runner import (
            OfflineBenchConfig,
            OfflineRunner,
        )

        class FakeResponse:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def json(self):
                return {"ok": True}

        class FakeSession:
            def post(self, url, json):
                self.url = url
                self.payload = json
                return FakeResponse()

        runner = OfflineRunner.__new__(OfflineRunner)
        runner._port = 12345
        runner._config = OfflineBenchConfig(num_return_sequences=4)
        session = FakeSession()

        result = asyncio.run(runner._send_one(session, "prompt", 32))

        self.assertEqual(result, {"ok": True})
        self.assertEqual(
            session.payload["generate_config"],
            {
                "max_new_tokens": 32,
                "min_new_tokens": 32,
                "top_k": 1,
                "num_beams": 1,
                "num_return_sequences": 4,
                "do_sample": True,
            },
        )


class TestOfflineBenchMain(unittest.TestCase):
    def test_all_failed_requests_raise_from_entry_point_and_stop_server(self):
        import rtp_llm.test.perf_test.offline_bench_test as offline_bench
        import rtp_llm.test.perf_test.offline_runner as offline_runner_module
        from rtp_llm.test.perf_test.offline_runner import (
            OfflineMetrics,
            OfflineRunner,
        )

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
        ]
        with patch.object(sys, "argv", argv), patch.object(
            offline_bench, "resolve_perf_engine_paths", side_effect=lambda args: args
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
            offline_bench, "resolve_perf_engine_paths", side_effect=lambda args: args
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

        parsed_args, engine_args = engine_server.call_args.args
        self.assertEqual(parsed_args.concurrency_limit, 37)
        self.assertEqual(engine_args[engine_args.index("--test_block_num") + 1], "17")
        self.assertEqual(server.start.call_args.kwargs["max_concurrency"], 37)

        runner_config = offline_runner.call_args.kwargs["config"]
        self.assertEqual(runner_config.concurrency_limit, 37)
        self.assertEqual(runner_config.total_requests, 74)

    def test_tokenizer_and_model_type_fall_back_to_environment(self):
        import rtp_llm.test.perf_test.offline_bench_test as offline_bench

        server = MagicMock(port=12345)
        argv = [
            "prog",
            "--checkpoint_path",
            "/tmp/checkpoint",
            "--concurrency_limit",
            "2",
        ]
        env = {
            "TOKENIZER_PATH": "/tmp/tokenizer",
            "MODEL_TYPE": "factory_model",
        }
        with patch.dict(os.environ, env, clear=True), patch.object(
            sys, "argv", argv
        ), patch.object(
            offline_bench, "resolve_perf_engine_paths", side_effect=lambda args: args
        ), patch.object(
            offline_bench, "EngineServer"
        ) as engine_server, patch.object(
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

        _, engine_args = engine_server.call_args.args
        self.assertEqual(
            engine_args[engine_args.index("--tokenizer_path") + 1],
            "/tmp/tokenizer",
        )
        self.assertEqual(
            offline_runner.call_args.kwargs["checkpoint_path"], "/tmp/checkpoint"
        )
        self.assertEqual(
            offline_runner.call_args.kwargs["tokenizer_path"], "/tmp/tokenizer"
        )
        self.assertEqual(offline_runner.call_args.kwargs["model_type"], "factory_model")

    def test_cli_tokenizer_overrides_environment(self):
        import rtp_llm.test.perf_test.offline_bench_test as offline_bench

        server = MagicMock(port=12345)
        argv = [
            "prog",
            "--checkpoint_path",
            "/tmp/checkpoint",
            "--tokenizer_path",
            "/tmp/cli-tokenizer",
            "--concurrency_limit",
            "2",
        ]
        with patch.dict(
            os.environ, {"TOKENIZER_PATH": "/tmp/env-tokenizer"}, clear=True
        ), patch.object(sys, "argv", argv), patch.object(
            offline_bench, "resolve_perf_engine_paths", side_effect=lambda args: args
        ), patch.object(
            offline_bench, "EngineServer"
        ) as engine_server, patch.object(
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

        self.assertEqual(
            offline_runner.call_args.kwargs["tokenizer_path"], "/tmp/cli-tokenizer"
        )


class TestEngineServerLifecycle(unittest.TestCase):
    @patch("rtp_llm.test.perf_test.server.MagaServerManager")
    def test_failed_start_stops_partial_server(self, MockMagaServerManager):
        from rtp_llm.test.perf_test.server import EngineServer

        manager = MockMagaServerManager.return_value
        manager.start_server.return_value = False
        server = EngineServer(_make_args(), [])

        with self.assertRaisesRegex(RuntimeError, "Engine server failed to start"):
            server.start(
                max_seq_len=8192,
                max_concurrency=8,
                server_start_timeout=1,
                use_batch_decode_scheduler=False,
            )

        manager.print_process_log.assert_called_once()
        manager.stop_server.assert_called_once()
        server.stop()
        manager.stop_server.assert_called_once()


if __name__ == "__main__":
    unittest.main()
