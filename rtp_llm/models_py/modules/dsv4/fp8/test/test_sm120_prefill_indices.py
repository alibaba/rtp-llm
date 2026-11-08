"""CPU oracle for the SM120 prefill index split helpers.

Pins the masked-tail sentinel contract: every masked extra-table tail reads
extra row 0, so that row must stay finite in the repacked extra cache (P1c
NaN leak).  Also pins the paged-padding contract: on the caller-supplied
pool path, ``-1`` padding must read a live slot of its own row, never the
persistent pool's never-guaranteed-initialized block 0.
"""

import unittest

import torch

from rtp_llm.models_py.modules.dsv4.fp8._sm120_prefill_indices import (
    remap_padding_to_first_entry,
    split_sm120_combined_tables,
    zero_extra_sentinel_row,
)
from rtp_llm.models_py.modules.dsv4.fp8.sm120_sparse_mla import (
    SM120_EXTRA_TOPK_WIDTHS,
    canonical_topk,
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


if __name__ == "__main__":
    unittest.main()
