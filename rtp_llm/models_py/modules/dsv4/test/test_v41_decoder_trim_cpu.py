"""CPU tests for the R48 decoder-trim scaffold (flag parsing + CP4 zigzag tail
geometry + decoder split). No GPU needed — the geometry functions are pure.
"""

import importlib
import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[5]
for suffix in [
    "",
    ".models_py",
    ".models_py.modules",
    ".models_py.modules.dsv4",
]:
    name = "rtp_llm" + suffix
    module = types.ModuleType(name)
    module.__path__ = [str(ROOT.joinpath(*name.split(".")))]
    sys.modules.setdefault(name, module)

dt = importlib.import_module("rtp_llm.models_py.modules.dsv4._v41_decoder_trim")


class DecoderTrimFlagTest(unittest.TestCase):
    def test_default_off(self):
        env = dict(os.environ)
        env.pop("DSV41_DECODER_TRIM_KEEP", None)
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(dt.decoder_trim_keep(), 0)
            self.assertFalse(dt.decoder_trim_enabled())

    def test_keep_140(self):
        with mock.patch.dict(os.environ, {"DSV41_DECODER_TRIM_KEEP": "140"}):
            self.assertEqual(dt.decoder_trim_keep(), 140)
            self.assertTrue(dt.decoder_trim_enabled())

    def test_reject_negative(self):
        with mock.patch.dict(os.environ, {"DSV41_DECODER_TRIM_KEEP": "-5"}):
            with self.assertRaises(ValueError):
                dt.decoder_trim_keep()

    def test_reject_nonint(self):
        with mock.patch.dict(os.environ, {"DSV41_DECODER_TRIM_KEEP": "abc"}):
            with self.assertRaises(ValueError):
                dt.decoder_trim_keep()

    def test_reject_below_swa_window(self):
        # keep must exceed the 128-row SWA window
        with mock.patch.dict(os.environ, {"DSV41_DECODER_TRIM_KEEP": "64"}):
            with self.assertRaises(ValueError):
                dt.decoder_trim_keep()
        with mock.patch.dict(os.environ, {"DSV41_DECODER_TRIM_KEEP": "129"}):
            self.assertEqual(dt.decoder_trim_keep(), 129)


class DecoderSplitTest(unittest.TestCase):
    def test_boundary(self):
        self.assertFalse(dt.is_decoder_layer(20))
        self.assertTrue(dt.is_decoder_layer(21))
        self.assertTrue(dt.is_decoder_layer(39))
        self.assertFalse(dt.is_decoder_layer(0))


class ZigzagTailGeometryTest(unittest.TestCase):
    """The tail-row mapping under CP4 zigzag must match _v41_ced's canonical
    owner mapping (the R19-proven geometry)."""

    def test_tail_owner_rows_partition(self):
        # the CP4 zigzag is PER-CHUNK packed (1024 rows/rank over a 4096-token
        # chunk): the decoder tail of the final chunk = rows [3956, 4096)
        n, keep = 4096, 140
        owners = dt.tail_owner_rows(n, keep, cp=4)
        all_rows = sorted(r for rows in owners.values() for r in rows)
        self.assertEqual(all_rows, list(range(n - keep, n)))
        for rank, rows in owners.items():
            for p in rows:
                owner, _ = dt._zigzag_owner_local(p, n // (2 * 4), 4)
                self.assertEqual(owner, rank)

    def test_tail_row_counts_per_rank(self):
        # CP4 zigzag is PER-CHUNK packed: the final chunk's last 140 rows all
        # land in rank 0's back half (the zigzag's last pair). This is the
        # documented geometry (the decoder tail concentrates on rank 0).
        owners = dt.tail_owner_rows(4096, 140, cp=4)
        counts = {r: len(rows) for r, rows in owners.items()}
        self.assertEqual(sum(counts.values()), 140)
        self.assertEqual(counts[0], 140)          # all on rank 0 (back-half tail)
        for r in (1, 2, 3):
            self.assertEqual(counts[r], 0)


if __name__ == "__main__":
    unittest.main()
