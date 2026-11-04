"""CPU oracle for the SM120 prefill index-adapter ready path.

Reference = the current production chain (`canonical_topk` followed by the
consumer's `clamp_min_(0)`), executed on CPU.  Candidate =
`_ready_chunk` from `_sm120_prefill_indices`.  On producer-style tables
(dense valid prefix, zero tail, non-negative, bounded lengths) the two must
be BIT-EXACT and leave the producer's lengths unchanged. On tables that
violate the producer invariants (holes, stale nonzero tails) the two MUST
differ — that discrimination is what makes the equivalence claim falsifiable.

The CUDA producer matrix covers ranks, chunk prefixes and compression ratios.
This CPU oracle covers width, length, layout and input ownership, including
HCA early widths 64->128 / 192->512 / 256->512, CSA 2048 and window 128.
It also pins the masked-tail sentinel contract: every masked extra-table
tail reads extra row 0, so that row must stay finite in the repacked extra
cache (P1c NaN leak).  Finally it pins the paged-padding contract: on the
caller-supplied pool path, ``-1`` padding must read a live slot of its own
row, never the persistent pool's never-guaranteed-initialized block 0.
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
    remap_padding_to_first_entry,
    split_sm120_combined_tables,
    zero_extra_sentinel_row,
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
        lens_ref = lens_t.clone()
        lens_new = lens_t.clone()
        ref_i, ref_l = reference_chain(table.clone(), lens_ref, supported)
        got = _ready_chunk(table, lens_new, 0, rows, supported)
        self.assertIsNotNone(got)
        new_i, new_l = got
        self.assertTrue(torch.equal(ref_i, new_i), f"indices differ {width=}")
        self.assertTrue(torch.equal(ref_l, new_l), f"lens differ {width=}")
        self.assertTrue(torch.equal(lens_ref, lens_t))
        self.assertTrue(torch.equal(lens_new, lens_t))
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
        shared_lens = torch.tensor(lens, **lens_i32)
        new_i, new_l = _ready_chunk(table, shared_lens, 0, rows, SWA_SUPPORTED_BIG)
        self.assertTrue(torch.equal(ref_i, new_i))
        self.assertTrue(torch.equal(ref_l, new_l))
        self.assertEqual(shared_lens.tolist(), lens)
        new_l.zero_()
        self.assertEqual(shared_lens.tolist(), lens)

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


def build_producer_combined_tables(seq_lens, M, N, window, ratio, topk):
    """Synthesize ``combine_topk_swa_indices``-style combined tables on CPU.

    Per token at absolute position ``pos``: ``min((pos + 1) // ratio, topk)``
    compressed entries (local workspace rows ``[0, N_b)``) followed by the
    last ``min(pos + 1, window)`` SWA rows, then a -1 tail. The SWA buffer
    offsets are simplified (full-window gather); only the extra/compressed
    geometry matters for the sentinel test below.
    """
    width = topk + window
    rows = []
    lens = []
    for b, seq in enumerate(seq_lens):
        for pos in range(seq):
            topk_len = min((pos + 1) // ratio, topk)
            swa_len = min(pos + 1, window)
            row = torch.full((width,), -1, dtype=torch.int64)
            row[:topk_len] = b * M + torch.arange(topk_len)
            row[topk_len : topk_len + swa_len] = (
                b * M + N + window - swa_len + torch.arange(swa_len)
            )
            rows.append(row)
            lens.append(topk_len + swa_len)
    return (
        torch.stack(rows).to(torch.int32),
        torch.tensor(lens, dtype=torch.int32),
    )


def sm120_extra_value_load(extra_kv, extra_indices, extra_lens):
    """Emulate the SM120 sparse kernel's extra-cache value loader.

    Gathers rows by index, zero-fills only entries with index < 0, and weights
    each entry by the masked probability (1 inside the length, 0 in the tail),
    so ``0 * V`` must be finite for every tail entry.
    """
    idx = extra_indices.to(torch.int64)
    values = extra_kv[idx].to(torch.float32)
    values = torch.where((idx < 0).unsqueeze(-1), torch.zeros_like(values), values)
    cols = torch.arange(idx.shape[1]).unsqueeze(0)
    probs = (cols < extra_lens.to(torch.int64).unsqueeze(1)).to(torch.float32)
    return (probs.unsqueeze(-1) * values).sum(dim=1)


class ExtraSentinelFiniteTest(unittest.TestCase):
    """P1c: masked extra tails all read extra row 0, so it must be finite."""

    def run_pipeline(self, seq_lens, ratio, window, topk, apply_sentinel):
        """NaN-poisoned workspace -> gathers -> repack + split -> emulated
        kernel value load. Mirrors the production order in
        ``_attn_via_workspace``: sentinel zero first, gathers may overwrite."""
        per_req_rows = [seq // ratio for seq in seq_lens]
        N = max(per_req_rows)
        B = len(seq_lens)
        M = N + window + 64
        D = 8
        workspace = torch.full((B, M, D), float("nan"), dtype=torch.bfloat16)
        if apply_sentinel:
            zero_extra_sentinel_row(workspace, N)
        # Simulated cmp gather: finite rows [0, N_b) per request.
        for b, n_b in enumerate(per_req_rows):
            if n_b:
                workspace[b, :n_b, :] = torch.full(
                    (n_b, D), float(b + 1), dtype=torch.bfloat16
                )
        # Simulated SWA-prefix gather + fresh-K overlay: the SWA stride slice
        # is finite; only [0, N) is repacked as the extra cache.
        workspace[:, N:, :] = 0.5

        extra_kv = workspace[:, :N, :].contiguous().view(-1, D)
        combined, lens = build_producer_combined_tables(
            seq_lens, M, N, window, ratio, topk
        )
        _, _, extra_indices, extra_lens, _ = split_sm120_combined_tables(
            combined,
            lens,
            M=M,
            N=N,
            window_size=window,
            extra_width=max(topk, 1),
            ratio=ratio,
            device=torch.device("cpu"),
        )
        return sm120_extra_value_load(extra_kv, extra_indices, extra_lens)

    def test_hca_leading_request_without_compressed_rows_is_finite(self):
        # HCA ratio 128: request 0 (64 tokens) owns 0 compressed rows while
        # request 1 (128 tokens) owns 1, so N == 1 > 0 and extra row 0 is
        # never written by the gather — the P1c precondition.
        out = self.run_pipeline(
            seq_lens=[64, 128],
            ratio=128,
            window=128,
            topk=1,
            apply_sentinel=True,
        )
        self.assertTrue(torch.isfinite(out).all())

    def test_csa_short_leading_request_is_finite(self):
        # CSA ratio 4: request 0 (3 tokens) owns 0 rows, request 1 (8 tokens)
        # owns 2; extra_width comes from the indexer topk (>= N). Unaddressed
        # rows of request 0 stay NaN but are never indexed.
        out = self.run_pipeline(
            seq_lens=[3, 8],
            ratio=4,
            window=128,
            topk=4,
            apply_sentinel=True,
        )
        self.assertTrue(torch.isfinite(out).all())

    def test_without_sentinel_nan_leaks_through_masked_tails(self):
        # Falsifiability: the identical poisoned pipeline without the sentinel
        # zero must leak NaN (0 * NaN), proving the finite assertions above
        # test the fix rather than an unpoisoned input.
        out = self.run_pipeline(
            seq_lens=[64, 128],
            ratio=128,
            window=128,
            topk=1,
            apply_sentinel=False,
        )
        self.assertTrue(torch.isnan(out).any())

    def test_sentinel_zero_is_overwritten_by_real_compressed_row(self):
        # When request 0 owns a compressed row the gather overwrites the
        # zeroed sentinel — the fix must never clobber live values.
        workspace = torch.full((2, 1 + 128 + 64, 8), float("nan"), dtype=torch.bfloat16)
        zero_extra_sentinel_row(workspace, 1)
        self.assertTrue(
            torch.equal(workspace[0, 0], torch.zeros(8, dtype=torch.bfloat16))
        )
        workspace[0, 0, :] = 3.0
        extra_kv = workspace[:, :1, :].contiguous().view(-1, 8)
        self.assertEqual(extra_kv[0].tolist(), [3.0] * 8)

    def test_no_sentinel_when_extra_region_empty(self):
        workspace = torch.full((2, 128, 8), float("nan"), dtype=torch.bfloat16)
        zero_extra_sentinel_row(workspace, 0)
        self.assertTrue(torch.isnan(workspace[0, 0]).all())


def paged_value_load(pool, indices, lens):
    """Emulate the SM120 prefill kernel's paged value loader.

    ``idx < 0`` reads pool row 0 (never a zero-fill); masked tail entries
    still contribute ``0 * V`` to the PV product.  A zero-length row loads
    no tiles at all, so it contributes nothing.
    """
    idx = indices.to(torch.int64)
    idx = torch.where(idx < 0, torch.zeros_like(idx), idx)
    values = pool[idx].to(torch.float32)
    cols = torch.arange(idx.shape[1]).unsqueeze(0)
    probs = (cols < lens.to(torch.int64).unsqueeze(1)).to(torch.float32)
    out = (probs.unsqueeze(-1) * values).sum(dim=1)
    return torch.where((lens > 0).unsqueeze(-1), out, torch.zeros_like(out))


class PagedPaddingRemapTest(unittest.TestCase):
    """Caller-supplied paged pools never guarantee block 0 is initialized.

    The kernel maps ``idx < 0`` to pool slot 0 and still loads it, so the
    historical ``clamp_min_(0)`` lets ``0 * NaN`` poison rows that carry
    padding; the remap must point padding at a live slot of the same row.
    """

    def poisoned_pipeline(self, remap):
        # Pool block 0 stays uninitialized (NaN); the other rows hold live
        # tokens.  Padded rows come from canonical_topk, as on the paged path.
        pool = torch.full((8, 8), float("nan"), dtype=torch.bfloat16)
        pool[1:] = 1.0
        raw = torch.tensor(
            [[3, -1, -1, -1], [2, 5, -1, -1], [1, 2, 3, -1], [-1, -1, -1, -1]],
            dtype=torch.int32,
        )
        canonical, lens = canonical_topk(
            raw, torch.tensor([1, 2, 3, 0], dtype=torch.int32), (4,)
        )
        if remap:
            table = remap_padding_to_first_entry(canonical)
        else:
            table = canonical.clamp_min(0)
        return paged_value_load(pool, table, lens), canonical, lens

    def test_padding_reads_a_live_slot_of_the_same_row(self):
        out, canonical, lens = self.poisoned_pipeline(remap=True)
        self.assertTrue(torch.isfinite(out).all())
        # Lengths are untouched: the kernel still masks the remapped tail.
        self.assertTrue(
            torch.equal(lens, torch.tensor([1, 2, 3, 0], dtype=torch.int32))
        )

    def test_clamp_to_uninitialized_block_leaks_nan(self):
        # Falsifiability: the identical poisoned pool under the incumbent
        # clamp must leak NaN through the masked tail, proving the finite
        # assertion above tests the remap, not an unpoisoned pool.
        out, _, _ = self.poisoned_pipeline(remap=False)
        self.assertTrue(torch.isnan(out[:3]).all())

    def test_remapped_table_properties(self):
        canonical = torch.tensor(
            [[3, -1, -1, -1], [2, 5, -1, -1], [-1, -1, -1, -1]], dtype=torch.int32
        )
        original = canonical.clone()
        remapped = remap_padding_to_first_entry(canonical)
        self.assertTrue(
            torch.equal(remapped[0], torch.tensor([3, 3, 3, 3], dtype=torch.int32))
        )
        self.assertTrue(
            torch.equal(remapped[1], torch.tensor([2, 5, 2, 2], dtype=torch.int32))
        )
        # Zero-length rows load no tiles, so their -1 padding stays put.
        self.assertTrue(torch.equal(remapped[2], canonical[2]))
        # Out-of-place: the caller's (possibly shared) table is never mutated.
        self.assertTrue(torch.equal(canonical, original))
        self.assertEqual(remapped.dtype, torch.int32)
        self.assertTrue(remapped.is_contiguous())
        self.assertIsNot(remapped, canonical)

    def test_non_2d_input_rejected(self):
        with self.assertRaisesRegex(ValueError, "2D"):
            remap_padding_to_first_entry(torch.zeros((2, 1, 4), dtype=torch.int32))


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
