import argparse
import unittest

from rtp_llm.test.perf_test.batch_decode_test import _effective_grid_max_seq_len


class BatchDecodeTest(unittest.TestCase):
    def test_effective_grid_max_seq_len_uses_decode_need(self):
        args = argparse.Namespace(max_seq_len=8192, decode_test_length=30)
        self.assertEqual(_effective_grid_max_seq_len(args, [1024, 65536]), 65566)

    def test_effective_grid_max_seq_len_respects_explicit_headroom(self):
        args = argparse.Namespace(max_seq_len=65664, decode_test_length=30)
        self.assertEqual(_effective_grid_max_seq_len(args, [65536]), 65664)


if __name__ == "__main__":
    unittest.main()
