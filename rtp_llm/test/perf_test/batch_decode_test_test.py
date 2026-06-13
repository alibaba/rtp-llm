import argparse
import unittest

from rtp_llm.test.perf_test.batch_decode_test import (
    _effective_grid_max_seq_len,
    _parse_grid_cases,
    _positive_int_arg,
)
from rtp_llm.test.perf_test.test_util import target_reuse_len_for_hit_rate


class BatchDecodeTest(unittest.TestCase):
    def test_effective_grid_max_seq_len_uses_decode_need(self):
        args = argparse.Namespace(max_seq_len=8192, decode_test_length=30)
        self.assertEqual(_effective_grid_max_seq_len(args, [1024, 65536]), 65566)

    def test_effective_grid_max_seq_len_respects_explicit_headroom(self):
        args = argparse.Namespace(max_seq_len=65664, decode_test_length=30)
        self.assertEqual(_effective_grid_max_seq_len(args, [65536]), 65664)

    def test_parse_grid_cases_dedupes_in_order(self):
        self.assertEqual(
            _parse_grid_cases("1:8192, 2:262144,1:8192"),
            [(1, 8192), (2, 262144)],
        )

    def test_parse_grid_cases_empty_uses_cartesian_grid(self):
        self.assertIsNone(_parse_grid_cases(""))

    def test_parse_grid_cases_rejects_malformed_items(self):
        with self.assertRaises(ValueError):
            _parse_grid_cases("1:8192,bad")

    def test_parse_grid_cases_rejects_non_positive_values(self):
        with self.assertRaises(ValueError):
            _parse_grid_cases("0:8192")

    def test_positive_int_arg_parses_forwarded_server_args(self):
        self.assertEqual(
            _positive_int_arg(["--seq_size_per_block", "64"], "seq_size_per_block", 1),
            64,
        )
        self.assertEqual(_positive_int_arg([], "seq_size_per_block", 8), 8)

    def test_target_reuse_len_rounds_to_block(self):
        self.assertEqual(target_reuse_len_for_hit_rate(1048576, 0.85, 64), 891264)


if __name__ == "__main__":
    unittest.main()
