"""UT: DSV4 zero-SWA inverted-triangle compressor write-skip (Stage B guardian).

Directly exercises the *compute-reduction* logic without spinning up a server
or GPU kernels, so the subtle boundary cases that an end-to-end smoke run can
silently paper over are pinned here:

  * The skip window must be **block-aligned** to the cp-virtual cache block
    (``seq_size_per_block * cp_size``) and equal the C++
    ``full_cover = reuse_blocks_len + restore_blocks`` extension in
    ``HybridKVCacheAllocator::reuseCache`` *exactly*. Under-masking corrupts
    the shared cached blocks; over-masking drops compressor writes for fresh
    new-token blocks. (work_log Correction 1.)
  * Only ``kv_slots`` (paged FULL / compressed pools) are skipped; STATE pools
    are recomputed-as-scratch, so the helper never touches ``state_slots``.
    (work_log Correction 2.)
  * A fresh request (no prefix reuse) must skip nothing.
  * The enable gate mirrors C++: both env flags + ``fp8_kv_cache`` +
    ``n_layers > 1``; the 1-layer SWA-only MTP draft is a no-op.
"""

from __future__ import annotations

import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv4.fp8.compressor import (
    _apply_zero_swa_write_skip,
)
from rtp_llm.models_py.modules.dsv4.prefill.forward import (
    _dsv4_env_flag,
    _dsv4_zero_swa_write_skip_window,
)


def _cpp_full_cover_tokens(matched_blocks: int, restore_blocks: int, spb: int):
    """Reproduce the C++ reuseCache math (typical tail-anchor == capped case):

      capped           = max(matched - restore_blocks, 0)
      reuse_blocks_len = capped               (tail anchor lands at capped)
      full_cover       = min(capped + restore_blocks, matched)

    Returns ``(prefix_tokens, full_cover_tokens)``. When ``capped == 0`` the
    real allocator returns 0 *before* the FULL extension (no shared blocks), so
    prefix == 0 and the write-skip must be a no-op.
    """
    capped = max(matched_blocks - restore_blocks, 0)
    reuse_blocks_len = capped
    if reuse_blocks_len <= 0:
        return 0, 0  # reuse collapses -> all fresh, no shared FULL blocks
    full_cover = min(reuse_blocks_len + restore_blocks, matched_blocks)
    return reuse_blocks_len * spb, full_cover * spb


class ApplyWriteSkipTest(unittest.TestCase):
    def test_window_zero_is_identity(self):
        kv = torch.tensor([10, 11, 12, 13], dtype=torch.int64)
        pos = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
        bidx = torch.zeros(4, dtype=torch.int64)
        sp = torch.tensor([0], dtype=torch.int64)
        out = _apply_zero_swa_write_skip(kv, pos, bidx, sp, 0)
        self.assertTrue(torch.equal(out, kv))

    def test_none_kv_or_seqstart_is_noop(self):
        pos = torch.arange(4, dtype=torch.int64)
        bidx = torch.zeros(4, dtype=torch.int64)
        self.assertIsNone(
            _apply_zero_swa_write_skip(None, pos, bidx, torch.tensor([2]), 8)
        )
        kv = torch.arange(4, dtype=torch.int64)
        self.assertTrue(
            torch.equal(_apply_zero_swa_write_skip(kv, pos, bidx, None, 8), kv)
        )

    def test_fresh_request_skips_nothing(self):
        # seq_start == 0 -> reuse collapsed -> nothing reused -> no skip.
        kv = torch.arange(100, 108, dtype=torch.int64)
        pos = torch.arange(8, dtype=torch.int64)
        bidx = torch.zeros(8, dtype=torch.int64)
        sp = torch.tensor([0], dtype=torch.int64)
        out = _apply_zero_swa_write_skip(kv, pos, bidx, sp, window := 8)
        self.assertTrue(torch.equal(out, kv), msg=f"window={window}")

    def test_continued_request_boundary_exact(self):
        # prefix=4, window=8 -> skip positions in [4, 12); keep >= 12.
        prefix, window = 4, 8
        pos = torch.arange(prefix, prefix + 16, dtype=torch.int64)
        kv = torch.arange(200, 200 + 16, dtype=torch.int64)
        bidx = torch.zeros(16, dtype=torch.int64)
        sp = torch.tensor([prefix], dtype=torch.int64)
        out = _apply_zero_swa_write_skip(kv, pos, bidx, sp, window)
        for i, p in enumerate(pos.tolist()):
            if p < prefix + window:
                self.assertEqual(out[i].item(), -1, msg=f"pos {p} should skip")
            else:
                self.assertEqual(
                    out[i].item(), kv[i].item(), msg=f"pos {p} should write"
                )

    def test_block_alignment_gap_is_covered(self):
        # Correction 1: window is block-aligned and may exceed raw restore_tokens.
        # Positions in the gap [prefix+restore_tokens, prefix+window) MUST skip,
        # else those compressor writes land on shared cached blocks.
        prefix = 16
        restore_tokens = 10  # raw swa_window*n_layers (NOT block-aligned)
        spb = 8
        window = math.ceil(restore_tokens / spb) * spb  # = 16 (block-aligned)
        self.assertGreater(window, restore_tokens)
        pos = torch.arange(prefix, prefix + window + 4, dtype=torch.int64)
        kv = torch.arange(50, 50 + pos.numel(), dtype=torch.int64)
        bidx = torch.zeros(pos.numel(), dtype=torch.int64)
        sp = torch.tensor([prefix], dtype=torch.int64)
        out = _apply_zero_swa_write_skip(kv, pos, bidx, sp, window)
        # gap positions
        for i, p in enumerate(pos.tolist()):
            if prefix + restore_tokens <= p < prefix + window:
                self.assertEqual(
                    out[i].item(), -1, msg=f"gap pos {p} must be skipped"
                )

    def test_multi_request_independent_boundaries(self):
        # req0 prefix=4 window=4 -> skip [4,8); req1 fresh prefix=0 -> no skip.
        window = 4
        pos = torch.tensor([4, 5, 6, 7, 8, 0, 1, 2, 3], dtype=torch.int64)
        bidx = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.int64)
        kv = torch.arange(300, 300 + pos.numel(), dtype=torch.int64)
        sp = torch.tensor([4, 0], dtype=torch.int64)  # [B]
        out = _apply_zero_swa_write_skip(kv, pos, bidx, sp, window)
        expected_skip = [True, True, True, True, False, False, False, False, False]
        for i, want in enumerate(expected_skip):
            if want:
                self.assertEqual(out[i].item(), -1, msg=f"idx {i}")
            else:
                self.assertEqual(out[i].item(), kv[i].item(), msg=f"idx {i}")

    def test_does_not_mutate_input(self):
        kv = torch.arange(8, dtype=torch.int64)
        kv_copy = kv.clone()
        pos = torch.arange(8, dtype=torch.int64)
        bidx = torch.zeros(8, dtype=torch.int64)
        _apply_zero_swa_write_skip(kv, pos, bidx, torch.tensor([2]), 4)
        self.assertTrue(torch.equal(kv, kv_copy), "helper must not mutate kv_slots")

    def test_matches_cpp_full_cover(self):
        # The strongest guard for Correction 1: for a sweep of (matched, prefix),
        # prefix + window == full_cover * spb exactly, so a write at token p is
        # skipped iff its FULL block was reused (p < full_cover*spb).
        spb = 8
        for restore_blocks in (1, 3, 5):
            window = restore_blocks * spb  # cp_size == 1
            for matched in range(0, 20):
                prefix_tokens, full_cover_tokens = _cpp_full_cover_tokens(
                    matched, restore_blocks, spb
                )
                if prefix_tokens == 0:
                    continue  # reuse collapsed; helper is a no-op (sp==0 gate)
                # Build a token row spanning [prefix, matched) (the recomputed
                # input region) plus a couple of fresh tokens beyond `matched`.
                end = matched * spb + 2 * spb
                pos = torch.arange(prefix_tokens, end, dtype=torch.int64)
                kv = torch.arange(1000, 1000 + pos.numel(), dtype=torch.int64)
                bidx = torch.zeros(pos.numel(), dtype=torch.int64)
                sp = torch.tensor([prefix_tokens], dtype=torch.int64)
                out = _apply_zero_swa_write_skip(kv, pos, bidx, sp, window)
                boundary = prefix_tokens + window
                self.assertEqual(
                    boundary,
                    full_cover_tokens,
                    msg=f"matched={matched} rb={restore_blocks}: "
                    f"py boundary {boundary} != cpp full_cover {full_cover_tokens}",
                )
                for i, p in enumerate(pos.tolist()):
                    reused = p < full_cover_tokens
                    if reused:
                        self.assertEqual(out[i].item(), -1, msg=f"reused pos {p}")
                    else:
                        self.assertEqual(
                            out[i].item(), kv[i].item(), msg=f"fresh pos {p}"
                        )


class WriteSkipWindowGateTest(unittest.TestCase):
    @staticmethod
    def _v4(n_layers=4, window_size=128, fp8=True):
        return SimpleNamespace(
            fp8_kv_cache=fp8,
            args=SimpleNamespace(n_layers=n_layers, window_size=window_size),
        )

    @staticmethod
    def _kv(spb=64):
        return SimpleNamespace(seq_size_per_block=spb)

    def test_env_flag_truthiness_matches_cpp(self):
        for val in ("1", "true", "TRUE", "on", "ON", "yes"):
            with patch.dict("os.environ", {"X": val}, clear=False):
                self.assertTrue(_dsv4_env_flag("X"), val)
        for val in ("0", "false", "FALSE", "off", "OFF", ""):
            with patch.dict("os.environ", {"X": val}, clear=False):
                self.assertFalse(_dsv4_env_flag("X"), repr(val))
        with patch.dict("os.environ", {}, clear=True):
            self.assertFalse(_dsv4_env_flag("X_UNSET"))

    def test_disabled_when_envs_unset(self):
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(
                _dsv4_zero_swa_write_skip_window(self._v4(), self._kv(), None), 0
            )

    def test_requires_both_envs(self):
        with patch.dict(
            "os.environ", {"DSV4_ZERO_SWA_TRIM": "1"}, clear=True
        ):  # caching unset
            self.assertEqual(
                _dsv4_zero_swa_write_skip_window(self._v4(), self._kv(), None), 0
            )
        with patch.dict(
            "os.environ", {"DSV4_ZERO_SWA_CACHING": "1"}, clear=True
        ):  # trim unset
            self.assertEqual(
                _dsv4_zero_swa_write_skip_window(self._v4(), self._kv(), None), 0
            )

    def test_enabled_formula_block_aligned(self):
        env = {"DSV4_ZERO_SWA_CACHING": "1", "DSV4_ZERO_SWA_TRIM": "1"}
        with patch.dict("os.environ", env, clear=True):
            spb, n_layers, win = 64, 4, 128
            restore_tokens = win * n_layers  # 512
            got = _dsv4_zero_swa_write_skip_window(
                self._v4(n_layers=n_layers, window_size=win), self._kv(spb), None
            )
            want = math.ceil(restore_tokens / spb) * spb
            self.assertEqual(got, want)
            self.assertEqual(got % spb, 0)
            self.assertGreaterEqual(got, restore_tokens)

    def test_cp_size_scales_reuse_unit(self):
        env = {"DSV4_ZERO_SWA_CACHING": "1", "DSV4_ZERO_SWA_TRIM": "1"}
        with patch.dict("os.environ", env, clear=True):
            spb, n_layers, win, cp = 64, 4, 100, 2
            restore_tokens = win * n_layers  # 400
            reuse_unit = spb * cp  # 128
            cp_ctx = SimpleNamespace(cp_size=cp)
            got = _dsv4_zero_swa_write_skip_window(
                self._v4(n_layers=n_layers, window_size=win), self._kv(spb), cp_ctx
            )
            want = math.ceil(restore_tokens / reuse_unit) * reuse_unit
            self.assertEqual(got, want)
            self.assertEqual(got % reuse_unit, 0)

    def test_swa_window_zero_falls_back_to_128(self):
        env = {"DSV4_ZERO_SWA_CACHING": "1", "DSV4_ZERO_SWA_TRIM": "1"}
        with patch.dict("os.environ", env, clear=True):
            spb, n_layers = 64, 4
            got = _dsv4_zero_swa_write_skip_window(
                self._v4(n_layers=n_layers, window_size=0), self._kv(spb), None
            )
            want = math.ceil((128 * n_layers) / spb) * spb
            self.assertEqual(got, want)

    def test_mtp_single_layer_is_noop(self):
        # 1-layer SWA-only MTP draft: fails n_layers>1 gate -> no write-skip.
        env = {"DSV4_ZERO_SWA_CACHING": "1", "DSV4_ZERO_SWA_TRIM": "1"}
        with patch.dict("os.environ", env, clear=True):
            self.assertEqual(
                _dsv4_zero_swa_write_skip_window(
                    self._v4(n_layers=1, window_size=128), self._kv(), None
                ),
                0,
            )

    def test_disabled_when_not_fp8(self):
        env = {"DSV4_ZERO_SWA_CACHING": "1", "DSV4_ZERO_SWA_TRIM": "1"}
        with patch.dict("os.environ", env, clear=True):
            self.assertEqual(
                _dsv4_zero_swa_write_skip_window(
                    self._v4(fp8=False), self._kv(), None
                ),
                0,
            )


if __name__ == "__main__":
    unittest.main()
