"""CPU oracle for the SM120 prefill index-adapter ready path.

Reference = the current production chain (`canonical_topk` followed by the
consumer's `clamp_min_(0)`), executed on CPU.  Candidate =
`_ready_chunk` from `_sm120_prefill_indices`.  On producer-style tables
(dense valid prefix, zero tail, non-negative, bounded lengths) the two must
be BIT-EXACT, including the in-place lens-clamp side effect.  On tables that
violate the producer invariants (holes, stale nonzero tails) the two MUST
differ — that discrimination is what makes the equivalence claim falsifiable.

Rank/chunk-prefix coverage (all 4 CP ranks x 8 chunk prefixes x CSA/HCA) is
exercised by the G1 CUDA component test and the G3 full smoke; this CPU
oracle covers the width/length/layout/mutation contract space those runs
imply (HCA early widths 64->128 / 192->512 / 256->512, CSA 2048, window 128).
"""

import os
import subprocess
import sys
import unittest

import torch

from rtp_llm.models_py.modules.dsv4.fp8._sm120_prefill_indices import (
    Sm120PrefillIndices,
    _ready_chunk,
    index_adapter_mode,
)
from rtp_llm.models_py.modules.dsv4.fp8.sm120_sparse_mla import (
    SM120_EXTRA_TOPK_WIDTHS,
    canonical_topk,
)

SWA_SUPPORTED_BIG = (128, 512, 1024, 2048)  # rows > 64
SWA_SUPPORTED_SMALL = (128, 512, 1024)  # rows <= 64
WINDOW = 128


def reference_chain(indices, lens, supported):
    """The incumbent: canonical_topk then the consumer's clamp_min_(0)."""
    out_indices, out_lens = canonical_topk(indices, lens, supported)
    out_indices.clamp_min_(0)
    return out_indices, out_lens


def make_producer_table(rows, width, lens, seed):
    """Dense valid prefix + zero tail + non-negative, like the SM120 split."""
    g = torch.Generator().manual_seed(seed)
    table = torch.zeros((rows, width), dtype=torch.int32)
    for r in range(rows):
        n = int(lens[r])
        if n > 0:
            # Nonmonotonic ids with duplicates: order must be preserved.
            vals = torch.randint(0, 4096, (n,), generator=g, dtype=torch.int32)
            table[r, :n] = vals
    return table


class ReadyChunkEquivalenceTest(unittest.TestCase):
    def check_equal(self, rows, width, lens, supported, seed=0):
        table = make_producer_table(rows, width, lens, seed)
        lens_t = torch.tensor(lens, dtype=torch.int32)
        # Both paths may mutate the lens slice in place; give each its own.
        lens_ref = lens_t.clone()
        lens_new = lens_t.clone()
        ref_i, ref_l = reference_chain(table.clone(), lens_ref, supported)
        got = _ready_chunk(table, lens_new, 0, rows, supported)
        self.assertIsNotNone(got)
        new_i, new_l = got
        self.assertTrue(torch.equal(ref_i, new_i), f"indices differ {width=}")
        self.assertTrue(torch.equal(ref_l, new_l), f"lens differ {width=}")
        # The in-place clamp side effect on the producer lens must match.
        self.assertTrue(torch.equal(lens_ref, lens_new))
        self.assertEqual(new_i.dtype, torch.int32)
        self.assertTrue(new_i.is_contiguous())

    def test_swa_window_supported_widths(self):
        for supported in (SWA_SUPPORTED_BIG, SWA_SUPPORTED_SMALL):
            for rows in (1, 17, 64, 128, 1024):
                lens = [min(r + 1, WINDOW) for r in range(rows)]
                self.check_equal(rows, WINDOW, lens, supported, seed=rows)

    def test_extra_widths_and_native_padding(self):
        # HCA early chunks align to 64 then pad to 128; 192/256 pad to 512;
        # CSA 2048 is natively supported (no pad).
        for width in (64, 128, 192, 256, 2048):
            for rows in (1, 33, 256):
                lens = [min(2 * r + 1, width) for r in range(rows)]
                self.check_equal(rows, width, lens, SM120_EXTRA_TOPK_WIDTHS)

    def test_zero_and_full_lengths(self):
        rows = 64
        for lens in ([0] * rows, [WINDOW] * rows, [1] * rows):
            self.check_equal(rows, WINDOW, lens, SWA_SUPPORTED_BIG)
        # mixed zero/short/full
        lens = [0, 1, WINDOW // 2, WINDOW] * (rows // 4)
        self.check_equal(rows, WINDOW, lens, SWA_SUPPORTED_BIG)

    def test_oversized_and_negative_lengths_clamped(self):
        # The eligible producer never emits out-of-range lens (kernel bound),
        # and never emits a nonzero tail beyond the (clamped) lens.  Build the
        # producer-consistent content for these hypothetical lens values:
        # n = clamp(lens) valid entries per row, zero tail.
        rows = 8
        lens = [WINDOW + 5, 10**6, -3, 0, WINDOW, 1, WINDOW - 1, 2]
        clamped = [min(max(l, 0), WINDOW) for l in lens]
        table = make_producer_table(rows, WINDOW, clamped, 7)
        lens_i32 = dict(dtype=torch.int32)
        ref_i, ref_l = reference_chain(
            table.clone(), torch.tensor(lens, **lens_i32), SWA_SUPPORTED_BIG
        )
        new_i, new_l = _ready_chunk(
            table, torch.tensor(lens, **lens_i32), 0, rows, SWA_SUPPORTED_BIG
        )
        self.assertTrue(torch.equal(ref_i, new_i))
        self.assertTrue(torch.equal(ref_l, new_l))

    def test_chunk_slices(self):
        # Consumer slices [start:end] per Q chunk; verify a middle slice.
        rows, width = 1024, 256
        lens = [min(r, width) for r in range(rows)]
        table = make_producer_table(rows, width, lens, 3)
        lens_t = torch.tensor(lens, dtype=torch.int32)
        start, end = 512, 768
        ref_i, ref_l = reference_chain(
            table[start:end].clone(), lens_t[start:end].clone(), SM120_EXTRA_TOPK_WIDTHS
        )
        new_i, new_l = _ready_chunk(table, lens_t, start, end, SM120_EXTRA_TOPK_WIDTHS)
        self.assertTrue(torch.equal(ref_i, new_i))
        self.assertTrue(torch.equal(ref_l, new_l))

    def test_three_dimensional_input_squeezed(self):
        rows = 16
        lens = [1] * rows
        table = make_producer_table(rows, WINDOW, lens, 1).unsqueeze(1)
        got = _ready_chunk(
            table, torch.tensor(lens, dtype=torch.int32), 0, rows, SWA_SUPPORTED_BIG
        )
        self.assertIsNotNone(got)
        self.assertEqual(tuple(got[0].shape), (rows, WINDOW))


class ReadyChunkDiscriminationTest(unittest.TestCase):
    """Tables violating producer invariants MUST diverge from the reference."""

    def test_hole_inside_prefix_differs(self):
        # [5, -1, 7] with length 3: canonical compacts to [5, 7], ready keeps
        # the raw row.  Exact inequality is the required discrimination.
        indices = torch.tensor([[5, -1, 7, 0]], dtype=torch.int32)
        lens = torch.tensor([3], dtype=torch.int32)
        ref_i, ref_l = reference_chain(indices.clone(), lens.clone(), (4,))
        got = _ready_chunk(indices, lens, 0, 1, (4,))
        self.assertIsNotNone(got)
        self.assertFalse(torch.equal(ref_i, got[0]))

    def test_stale_nonzero_tail_differs(self):
        indices = torch.tensor([[5, 7, 9, 9]], dtype=torch.int32)
        lens = torch.tensor([2], dtype=torch.int32)
        ref_i, ref_l = reference_chain(indices.clone(), lens.clone(), (4,))
        got = _ready_chunk(indices, lens, 0, 1, (4,))
        self.assertIsNotNone(got)
        self.assertFalse(torch.equal(ref_i, got[0]))  # ref zeroes the tail


class ReadyChunkContractGateTest(unittest.TestCase):
    """Metadata mismatches return None -> caller uses the generic path."""

    def table(self, rows=4, width=WINDOW):
        return make_producer_table(rows, width, [1] * rows, 0)

    def test_int64_indices_fall_back(self):
        got = _ready_chunk(
            self.table().to(torch.int64),
            torch.ones(4, dtype=torch.int32),
            0,
            4,
            SWA_SUPPORTED_BIG,
        )
        self.assertIsNone(got)

    def test_noncontiguous_falls_back(self):
        wide = make_producer_table(4, WINDOW * 2, [1] * 4, 0)[:, ::2]
        self.assertFalse(wide.is_contiguous())
        got = _ready_chunk(
            wide, torch.ones(4, dtype=torch.int32), 0, 4, SWA_SUPPORTED_BIG
        )
        self.assertIsNone(got)

    def test_lens_dtype_mismatch_falls_back(self):
        got = _ready_chunk(
            self.table(), torch.ones(4, dtype=torch.int64), 0, 4, SWA_SUPPORTED_BIG
        )
        self.assertIsNone(got)

    def test_lens_noncontiguous_falls_back(self):
        # A stride-2 lens view: canonical_topk's to() would produce a
        # contiguous copy, so the ready path must refuse the different layout.
        wide_lens = torch.ones(8, dtype=torch.int32)[::2]
        self.assertFalse(wide_lens.is_contiguous())
        got = _ready_chunk(self.table(), wide_lens, 0, 4, SWA_SUPPORTED_BIG)
        self.assertIsNone(got)

    def test_lens_device_mismatch_falls_back(self):
        # Mixed-device pair (meta lens vs cpu indices): the generic path would
        # move the lens; the ready path must not silently pair them.
        meta_lens = torch.ones(4, dtype=torch.int32, device="meta")
        got = _ready_chunk(self.table(), meta_lens, 0, 4, SWA_SUPPORTED_BIG)
        self.assertIsNone(got)

    def test_lens_row_mismatch_falls_back(self):
        got = _ready_chunk(
            self.table(), torch.ones(3, dtype=torch.int32), 0, 4, SWA_SUPPORTED_BIG
        )
        self.assertIsNone(got)

    def test_width_beyond_supported_falls_back(self):
        got = _ready_chunk(
            self.table(width=9000),
            torch.ones(4, dtype=torch.int32),
            0,
            4,
            SM120_EXTRA_TOPK_WIDTHS,
        )
        self.assertIsNone(got)
        # ... and the generic path keeps its existing rejection.
        with self.assertRaises(RuntimeError):
            canonical_topk(
                self.table(width=9000),
                torch.ones(4, dtype=torch.int32),
                SM120_EXTRA_TOPK_WIDTHS,
            )


class ModeParsingTest(unittest.TestCase):
    def test_default_mode_is_off(self):
        self.assertEqual(index_adapter_mode(), "off")

    def test_invalid_mode_fails_at_import(self):
        code = "import rtp_llm.models_py.modules.dsv4.fp8._sm120_prefill_indices"
        env = dict(
            os.environ,
            DSV4_SM120_PREFILL_INDEX_ADAPTER="bogus",
            # Subprocess must see the same import roots as this test process.
            PYTHONPATH=os.pathsep.join(p for p in sys.path if p),
        )
        proc = subprocess.run(
            [sys.executable, "-c", code], env=env, capture_output=True, text=True
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("DSV4_SM120_PREFILL_INDEX_ADAPTER", proc.stderr)

    def test_direct_mode_fails_at_import(self):
        # 'direct' is not implemented in this revision; it must fail closed,
        # never silently follow the ready branch.
        code = "import rtp_llm.models_py.modules.dsv4.fp8._sm120_prefill_indices"
        env = dict(
            os.environ,
            DSV4_SM120_PREFILL_INDEX_ADAPTER="direct",
            PYTHONPATH=os.pathsep.join(p for p in sys.path if p),
        )
        proc = subprocess.run(
            [sys.executable, "-c", code], env=env, capture_output=True, text=True
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("not implemented", proc.stderr)


if __name__ == "__main__":
    unittest.main()
