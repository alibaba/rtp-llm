# -*- coding: utf-8 -*-
"""Contract tests for the interleaved MRoPE pure-torch path.

Locks the repo-wide mrope_interleaved=true contract implemented by
apply_mrope_interleaved: THW slot-cycling axis assignment (slots 0,3,6.. = T,
1,4,7.. = H, 2,5,8.. = W) and lo/hi half-split pairing within rope_dim,
against an independent reference mirroring the CUDA/ROCm kernels (see
rocm_impl/test/test_aiter_prefill_op.py::_apply_mrope). Pure torch: CPU-runnable.
"""

import unittest

import torch

from rtp_llm.models_py.modules.factory.attention.ascend_impl.ascend_rope_emb import (
    _gen_base_halves_cache,
    apply_mrope_interleaved,
    mrope_interleaved_axis_of_freq,
)

THETA = 10000.0


def _reference(q, k, position_ids, mrope_sections, rope_dim, theta=THETA):
    """Independent interleaved-MRoPE reference (kernel get_mrope_position_dim
    rule + lo/hi half-split pairing)."""
    rotary_pairs = rope_dim // 2
    tidx = torch.arange(rotary_pairs)
    axes = torch.zeros(rotary_pairs, dtype=torch.long)
    axes[(tidx % 3 == 1) & (tidx < 3 * mrope_sections[1])] = 1
    axes[(tidx % 3 == 2) & (tidx < 3 * mrope_sections[2])] = 2
    positions = position_ids[:, axes].to(torch.float32)
    inv_freq = theta ** (-2.0 * torch.arange(rotary_pairs, dtype=torch.float32) / rope_dim)
    angle = positions * inv_freq.unsqueeze(0)
    cos = torch.cos(angle).unsqueeze(1)
    sin = torch.sin(angle).unsqueeze(1)

    def rot(x):
        x_float = x.float()
        lo = x_float[..., :rotary_pairs]
        hi = x_float[..., rotary_pairs:rope_dim]
        tail = x_float[..., rope_dim:]
        return torch.cat([lo * cos - hi * sin, hi * cos + lo * sin, tail], dim=-1)

    return rot(q).to(q.dtype), rot(k).to(k.dtype)


class TestMropeInterleavedAxis(unittest.TestCase):
    def test_mod3_thw_cycling(self):
        axes = mrope_interleaved_axis_of_freq([24, 20, 20], 64, torch.device("cpu"))
        self.assertEqual(axes.numel(), 64)
        tidx = torch.arange(64)
        expected = torch.zeros(64, dtype=torch.long)
        expected[(tidx % 3 == 1) & (tidx < 60)] = 1
        expected[(tidx % 3 == 2) & (tidx < 60)] = 2
        self.assertTrue(torch.equal(axes, expected))
        self.assertEqual(int((axes == 0).sum()), 24)
        self.assertEqual(int((axes == 1).sum()), 20)
        self.assertEqual(int((axes == 2).sum()), 20)

    def test_qwen3_vl_sections(self):
        axes = mrope_interleaved_axis_of_freq([24, 20, 20], 64, torch.device("cpu"))
        self.assertEqual(int((axes == 1).sum()), 20)
        self.assertEqual(int((axes == 2).sum()), 20)

    def test_rejects_invalid_sections(self):
        with self.assertRaises(RuntimeError):
            mrope_interleaved_axis_of_freq([16, 24, 23], 64, torch.device("cpu"))  # sum != pairs
        with self.assertRaises(RuntimeError):
            mrope_interleaved_axis_of_freq([16, 24, 24], 64, torch.device("cpu"))  # H=24 exceeds strided capacity
        with self.assertRaises(RuntimeError):
            mrope_interleaved_axis_of_freq([16, 14, 10], 40, torch.device("cpu"))  # H=14 exceeds strided capacity


class TestApplyMropeInterleaved(unittest.TestCase):
    def _run_case(self, sections, rope_dim, head_dim, dtype=torch.float32):
        torch.manual_seed(0)
        t, heads = 37, 5
        max_pos = 512
        q = torch.randn(t, heads, head_dim, dtype=dtype)
        k = torch.randn(t, 2, head_dim, dtype=dtype)
        pos = torch.randint(0, max_pos, (t, 3), dtype=torch.long)
        cache = _gen_base_halves_cache(rope_dim, THETA, max_pos).cpu()
        q_ref, k_ref = _reference(q.clone(), k.clone(), pos, sections, rope_dim)
        apply_mrope_interleaved(q, k, pos, sections, cache)
        torch.testing.assert_close(q, q_ref, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(k, k_ref, rtol=1e-4, atol=1e-4)

    def test_parity_dense_t_sections(self):
        # Temporal-heavy layout: excess slots stay on T once H/W slots are filled
        self._run_case([32, 16, 16], rope_dim=128, head_dim=128)

    def test_parity_qwen3_vl_sections(self):
        self._run_case([24, 20, 20], rope_dim=128, head_dim=128)

    def test_parity_partial_rotary_with_tail(self):
        # head_dim > rope_dim: dims beyond rope_dim must pass through unchanged
        self._run_case([24, 20, 20], rope_dim=128, head_dim=192)

    def test_axis_contracts_differ_from_continuous_sections(self):
        # The axis map must be mod-3 cycled, NOT continuous [T*16 | H*24 | W*24]
        axes = mrope_interleaved_axis_of_freq([24, 20, 20], 64, torch.device("cpu"))
        self.assertEqual(int(axes[1]), 1)   # slot 1 -> H (continuous would give T)
        self.assertEqual(int(axes[62]), 0)  # slot 62 -> T (continuous would give W)


if __name__ == "__main__":
    unittest.main()
