import argparse
import unittest

from rtp_llm.test.perf_test.batch_decode_test import (
    _effective_grid_max_seq_len,
    _ensure_default_role_type,
)


class BatchDecodeTest(unittest.TestCase):
    def test_effective_grid_max_seq_len_uses_decode_need(self):
        args = argparse.Namespace(max_seq_len=8192, decode_test_length=30)
        self.assertEqual(_effective_grid_max_seq_len(args, [1024, 65536]), 65566)

    def test_effective_grid_max_seq_len_respects_explicit_headroom(self):
        args = argparse.Namespace(max_seq_len=65664, decode_test_length=30)
        self.assertEqual(_effective_grid_max_seq_len(args, [65536]), 65664)

    def test_default_role_does_not_override_environment(self):
        args = []
        _ensure_default_role_type(args, {"ROLE_TYPE": "PREFILL"})
        self.assertEqual(args, [])

    def test_default_role_applies_without_cli_or_environment(self):
        args = []
        _ensure_default_role_type(args, {})
        self.assertEqual(args, ["--role_type", "PDFUSION"])

    def test_cli_role_wins_over_environment(self):
        args = ["--role_type", "DECODE"]
        _ensure_default_role_type(args, {"ROLE_TYPE": "PREFILL"})
        self.assertEqual(args, ["--role_type", "DECODE"])


if __name__ == "__main__":
    unittest.main()
