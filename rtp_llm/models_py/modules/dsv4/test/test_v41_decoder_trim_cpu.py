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

import torch

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
        # R58c: gate PASSED (no additional error) but the net TTFT win washed
        # out (graph-replay suspension), so the trim stays default-off; set
        # DSV41_DECODER_TRIM_KEEP=140 to enable the validated rectangle.
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
        self.assertEqual(counts[0], 140)  # all on rank 0 (back-half tail)
        for r in (1, 2, 3):
            self.assertEqual(counts[r], 0)


class CompactTailLengthTest(unittest.TestCase):
    """R48d: the CP4-legal compact tail length (L = ceiling of KEEP)."""

    def test_off(self):
        with mock.patch.dict(os.environ, {"DSV41_DECODER_TRIM_KEEP": "0"}):
            self.assertEqual(dt.decoder_trim_tail(), 0)

    def test_keep_140_maps_to_144(self):
        with mock.patch.dict(os.environ, {"DSV41_DECODER_TRIM_KEEP": "140"}):
            tail = dt.decoder_trim_tail()
            self.assertEqual(tail, 144)
            self.assertEqual(tail % (2 * 4), 0)  # CP4 zigzag-legal
            self.assertEqual((tail // 4) % 2, 0)  # even rows/rank (36)
            self.assertEqual(dt.compact_local_rows(tail), 36)

    def test_keep_144_already_legal(self):
        with mock.patch.dict(os.environ, {"DSV41_DECODER_TRIM_KEEP": "144"}):
            self.assertEqual(dt.decoder_trim_tail(), 144)


class CompactZigzagTest(unittest.TestCase):
    """The compact view re-zigzags the TAIL=144 tail across all 4 CP ranks
    (36 rows/rank, non-zero everywhere -> no zero-row collectives), unlike the
    original per-chunk layout where the whole tail sits on rank 0."""

    def test_relative_roundtrip_restore(self):
        tail, cp = 144, 4
        local = dt.compact_local_rows(tail)  # 36
        # For each rank, relative[rank] maps local row -> canonical position;
        # unpad_restore maps canonical -> flat gather index; roundtrip must hold.
        restore = dt.zigzag_unpad_restore(local, tail, cp)
        self.assertEqual(len(restore), tail)
        self.assertEqual(sorted(restore), list(range(tail)))  # a permutation
        for rank in range(cp):
            rel = dt.zigzag_relative(rank, local, cp)
            self.assertEqual(len(rel), local)
            for i, canon in enumerate(rel):
                self.assertEqual(restore[canon], rank * local + i)

    def test_compact_covers_all_ranks(self):
        # every rank holds 36 rows (non-zero) in the compact view
        tail, cp = 144, 4
        local = dt.compact_local_rows(tail)
        for rank in range(cp):
            rel = dt.zigzag_relative(rank, local, cp)
            self.assertEqual(len(rel), local)
            self.assertTrue(all(0 <= p < tail for p in rel))

    def test_final_row_lands_on_rank0(self):
        # the request's final token = canonical position 143 -> rank 0 back half.
        tail, cp = 144, 4
        owner, _ = dt._zigzag_owner_local(tail - 1, tail // (2 * cp), cp)
        self.assertEqual(owner, 0)


class TrimSourceMapTest(unittest.TestCase):
    """The restore map handles the whole 144-tail sitting on rank 0's back half
    in the ORIGINAL per-chunk layout (the CP4-zigzag-divisibility relaxation
    that R48c documented as the blocker — the all-gather + re-zigzag makes the
    compact machinery tail-alignment-agnostic)."""

    def test_tail_all_on_rank0_original_layout(self):
        n, rows, tail = 4096, 1024, 144
        for rank in range(4):
            dst, src = dt.trim_tail_source_map(rank, rows, n, tail, cp=4)
            if rank == 0:
                self.assertEqual(len(dst), 144)  # all tail rows on rank 0
            else:
                self.assertEqual(dst, [])  # other ranks contribute nothing
                self.assertEqual(src, [])

    def test_src_covers_the_tail(self):
        n, rows, tail = 4096, 1024, 144
        dst, src = dt.trim_tail_source_map(0, rows, n, tail, cp=4)
        self.assertEqual(sorted(src), list(range(tail)))


class FinalChunkTest(unittest.TestCase):
    def test_is_final_chunk(self):
        # 32K request, 8 chunks of 4096: chunk 7 is final (28672+4096==32768).
        self.assertTrue(dt.is_final_chunk(28672, 4096, 32768))
        self.assertFalse(dt.is_final_chunk(24576, 4096, 32768))
        self.assertFalse(dt.is_final_chunk(0, 4096, 32768))


class MHCRowLocalityTest(unittest.TestCase):
    """The mHC delayed-residual collapse is row-local (sums over the hc/streams
    axis, not the token axis), so a poisoned row does NOT contaminate other rows.
    This is the crux that makes the compact tail safe (and validates that the
    extra real context rows in L=144 vs KEEP=140 do not perturb the final token).
    Uses torch on CPU (the pure-tensor collapse path)."""

    def test_collapse_delayed_row_local(self):
        torch.manual_seed(0)
        rows, streams, dim = 8, 4, 16
        x = torch.randn(rows, streams, dim, dtype=torch.float32)
        pre_mix = torch.randn(rows, streams, dtype=torch.float32)
        # poison row 0 (simulating a pad / an out-of-window row)
        x_poison = x.clone()
        x_poison[0] = float("nan")
        out_clean = (x.float() * pre_mix.unsqueeze(-1)).sum(dim=-2)
        out_poison = (x_poison.float() * pre_mix.unsqueeze(-1)).sum(dim=-2)
        # row 0 is poisoned; rows 1.. are bit-identical (row-local)
        self.assertTrue(torch.isnan(out_poison[0]).all())
        self.assertTrue(torch.equal(out_clean[1:], out_poison[1:]))


class MaskArmFlagTest(unittest.TestCase):
    """R58 mask-mode reference arm: flag parsing (DSV41_DECODER_TRIM_MASK +
    DSV41_DECODER_TRIM_MASK_EXACT)."""

    def test_default_off(self):
        env = dict(os.environ)
        env.pop("DSV41_DECODER_TRIM_MASK", None)
        env.pop("DSV41_DECODER_TRIM_MASK_EXACT", None)
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(dt.decoder_trim_mask_keep(), 0)
            self.assertEqual(dt.mask_tail(), 0)
            self.assertFalse(dt.decoder_trim_mask_exact())

    def test_mask_140(self):
        with mock.patch.dict(os.environ, {"DSV41_DECODER_TRIM_MASK": "140"}):
            self.assertEqual(dt.decoder_trim_mask_keep(), 140)
            self.assertEqual(dt.mask_tail(), 144)  # CP-legal ceiling, like compact

    def test_mask_rejects(self):
        for bad in ("-1", "abc", "64", "128"):
            with mock.patch.dict(os.environ, {"DSV41_DECODER_TRIM_MASK": bad}):
                with self.assertRaises(ValueError):
                    dt.decoder_trim_mask_keep()

    def test_mask_exact_flag(self):
        with mock.patch.dict(os.environ, {"DSV41_DECODER_TRIM_MASK_EXACT": "1"}):
            self.assertTrue(dt.decoder_trim_mask_exact())
        with mock.patch.dict(os.environ, {"DSV41_DECODER_TRIM_MASK_EXACT": "0"}):
            self.assertFalse(dt.decoder_trim_mask_exact())


class MaskFloorGeometryTest(unittest.TestCase):
    """The mask-arm SWA visibility floor: mask-matched (the compact arm's own
    tail start, the equivalence-gate reference) vs mask-exact (DeepInfer's
    keep-140 rectangle, rect_lo = data_len - keep_tokens)."""

    def test_matched_floor_is_the_compact_tail_start(self):
        # 32K needle, final chunk = [28672, 32768), tail 144: floor = 32624.
        floor = dt.mask_floor_global(28672, 4096, 144, 140, exact=False)
        self.assertEqual(floor, 28672 + 4096 - 144)
        self.assertEqual(floor, dt.mask_start_global(28672, 4096, 144))

    def test_exact_floor_is_the_keep_rectangle(self):
        # DeepInfer semantics: rect_lo = data_len - keep_tokens = 32768 - 140.
        floor = dt.mask_floor_global(28672, 4096, 144, 140, exact=True)
        self.assertEqual(floor, 32768 - 140)
        # = tail_start + (tail - keep): 4 rows above the matched floor.
        self.assertEqual(floor, dt.mask_start_global(28672, 4096, 144) + 4)

    def test_short_chunk_floors(self):
        # single-chunk 384-token prompt: matched floor 384-144=240, exact 244.
        self.assertEqual(dt.mask_floor_global(0, 384, 144, 140, False), 240)
        self.assertEqual(dt.mask_floor_global(0, 384, 144, 140, True), 244)


class MaskVisibilitySemanticsTest(unittest.TestCase):
    """Document + verify the trim semantics the GPU mask line implements:
    query at absolute position p sees SWA keys [max(p - 127, floor), p]
    (positions below the floor expand to -1, i.e. the row's SWA section is
    empty — DeepInfer yoco_swa_visibility: abs_lo = max(pos - window + 1,
    rect_lo)). Pure-python mirror of the eager swidx build in
    attention_v41._prefill_query."""

    @staticmethod
    def _visible(pos, floor, window=128):
        lo = max(pos - window + 1, floor)
        return list(range(lo, pos + 1)) if lo <= pos else []

    def test_final_row_keeps_full_window(self):
        floor = 32624  # matched floor for the 32K needle
        vis = self._visible(32767, floor)
        self.assertEqual(len(vis), 128)
        self.assertEqual(vis[0], 32767 - 127)

    def test_first_tail_row_sees_only_itself(self):
        self.assertEqual(self._visible(32624, 32624), [32624])

    def test_mid_tail_row_clipped_at_floor(self):
        vis = self._visible(32660, 32624)
        self.assertEqual(vis, list(range(32624, 32661)))  # 37 rows

    def test_late_tail_row_full_window(self):
        # rows >= floor + 127 keep the complete rolling window
        vis = self._visible(32624 + 127, 32624)
        self.assertEqual(len(vis), 128)

    def test_exact_floor_shaves_four_rows(self):
        # DeepInfer exact keep-140 vs the CP-legal 144: the first tail rows
        # lose 4 visible rows; rows with a full window are unaffected.
        matched, exact = 32624, 32628
        self.assertEqual(
            len(self._visible(32660, matched)) - len(self._visible(32660, exact)), 4
        )
        self.assertEqual(self._visible(32767, matched), self._visible(32767, exact))

    def test_mask_line_matches_semantics(self):
        # The eager index build masks `swpos >= max(swstart, mask_start)`; the
        # buffer index (swpos - swstart) is unchanged. Simulate the swidx row
        # for a tail query and check the visible set equals _visible().
        swstart, mask_start, ns, window = 28545, 32624, 32768, 128
        pos = 32767
        swpos = list(range(pos - window + 1, pos + 1))
        vis_start = max(swstart, mask_start)
        swidx = [(ns + p - swstart) if p >= vis_start else -1 for p in swpos]
        visible_positions = [p for p, i in zip(swpos, swidx) if i >= 0]
        self.assertEqual(visible_positions, self._visible(pos, mask_start, window))
        # buffer indices are within the full workspace and ordered
        idx = [i for i in swidx if i >= 0]
        self.assertEqual(idx, sorted(idx))
        self.assertTrue(all(ns <= i < ns + (127 + 4096) for i in idx))


if __name__ == "__main__":
    unittest.main()
