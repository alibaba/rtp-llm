"""Correctness: fused conv1d_update + gdn_gating vs the two existing kernels."""

from __future__ import annotations

import math
import os
import unittest

import torch

from rtp_llm.models_py.triton_kernels.qwen35_decode_fusion.conv1d_gdn_gating import (
    conv1d_update_and_gdn_gating_ref,
    is_supported,
    maybe_fused_conv1d_update_gdn_gating,
)

_DIM = 12288
_WIDTH = 4
_HEADS = 64
_STATE_LEN = _WIDTH - 1
_SEQ_SIZE_PER_BLOCK = 128
_DTYPE = torch.bfloat16

# Existing conv1d decode UT uses these BF16 tolerances.
_RTOL_BF16 = 1e-2
_ATOL_BF16 = 5e-2
_RTOL_FP32 = 1e-4
_ATOL_FP32 = 1e-4


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        raise unittest.SkipTest("CUDA is required")


def _build_paged_decode(
    batch: int,
    sequence_lengths: torch.Tensor,
    dim: int = _DIM,
    heads: int = _HEADS,
    seq_size_per_block: int = _SEQ_SIZE_PER_BLOCK,
    device: str = "cuda",
    dtype: torch.dtype = _DTYPE,
    seed: int = 0,
):
    """Paged decode layout matching ``_conv1d`` / causal_conv1d_update tests."""
    torch.manual_seed(seed)
    if batch == 0:
        x = torch.empty(0, dim, 1, device=device, dtype=dtype)
        conv_state = torch.empty(0, dim, _STATE_LEN, device=device, dtype=dtype)
        weight = torch.randn(dim, _WIDTH, device=device, dtype=dtype)
        a = torch.empty(0, heads, device=device, dtype=dtype)
        b = torch.empty(0, heads, device=device, dtype=dtype)
        alog = torch.randn(heads, device=device, dtype=torch.float32)
        dt_bias = torch.randn(heads, device=device, dtype=dtype)
        block_map = torch.empty(0, 1, dtype=torch.int32, device=device)
        return {
            "x": x,
            "conv_state": conv_state,
            "weight": weight,
            "A_log": alog,
            "a": a,
            "b": b,
            "dt_bias": dt_bias,
            "block_map": block_map,
            "sequence_lengths": sequence_lengths,
            "seq_size_per_block": seq_size_per_block,
        }

    seq_list = sequence_lengths.tolist()
    block_nums = [max(math.ceil(seq / seq_size_per_block), 1) for seq in seq_list]
    max_block_num = max(block_nums)
    total_block_num = sum(block_nums)
    block_map = torch.zeros(batch, max_block_num, dtype=torch.int32, device=device)
    offset = 0
    for i, nblk in enumerate(block_nums):
        block_map[i, :nblk] = torch.arange(
            offset, offset + nblk, dtype=torch.int32, device=device
        )
        offset += nblk

    # [T, 1, dim] -> [T, dim, 1] like _conv1d reshape/transpose.
    x = torch.randn(batch, 1, dim, device=device, dtype=dtype).transpose(1, 2)
    origin_state = torch.randn(
        total_block_num, _STATE_LEN, dim, device=device, dtype=dtype
    )
    # Production: LinearCacheConverter [blocks, state, dim].transpose(1, 2).
    conv_state = origin_state.transpose(-1, -2)
    weight = torch.randn(dim, _WIDTH, device=device, dtype=dtype)
    a = torch.randn(batch, heads, device=device, dtype=dtype)
    b = torch.randn(batch, heads, device=device, dtype=dtype)
    alog = torch.randn(heads, device=device, dtype=torch.float32)
    dt_bias = torch.randn(heads, device=device, dtype=dtype)
    return {
        "x": x,
        "conv_state": conv_state,
        "weight": weight,
        "A_log": alog,
        "a": a,
        "b": b,
        "dt_bias": dt_bias,
        "block_map": block_map,
        "sequence_lengths": sequence_lengths,
        "seq_size_per_block": seq_size_per_block,
    }


def _clone_inputs(inputs: dict) -> dict:
    cloned = dict(inputs)
    cloned["x"] = inputs["x"].clone()
    cloned["conv_state"] = inputs["conv_state"].clone()
    cloned["a"] = inputs["a"].clone()
    cloned["b"] = inputs["b"].clone()
    return cloned


def _run_ref(inputs: dict):
    return conv1d_update_and_gdn_gating_ref(
        inputs["x"],
        inputs["conv_state"],
        inputs["weight"],
        inputs["A_log"],
        inputs["a"],
        inputs["b"],
        inputs["dt_bias"],
        block_map=inputs["block_map"],
        seq_size_per_block=inputs["seq_size_per_block"],
        sequence_lengths=inputs["sequence_lengths"],
    )


def _run_maybe(inputs: dict):
    return maybe_fused_conv1d_update_gdn_gating(
        inputs["x"],
        inputs["conv_state"],
        inputs["weight"],
        inputs["A_log"],
        inputs["a"],
        inputs["b"],
        inputs["dt_bias"],
        block_map=inputs["block_map"],
        seq_size_per_block=inputs["seq_size_per_block"],
        sequence_lengths=inputs["sequence_lengths"],
    )


def _assert_close_outputs(got, ref, conv_got, conv_ref):
    mixed_got, g_got, beta_got = got
    mixed_ref, g_ref, beta_ref = ref
    torch.testing.assert_close(mixed_got, mixed_ref, rtol=_RTOL_BF16, atol=_ATOL_BF16)
    torch.testing.assert_close(g_got, g_ref, rtol=_RTOL_FP32, atol=_ATOL_FP32)
    torch.testing.assert_close(beta_got, beta_ref, rtol=_RTOL_BF16, atol=_ATOL_BF16)
    torch.testing.assert_close(conv_got, conv_ref, rtol=_RTOL_BF16, atol=_ATOL_BF16)


class TestConv1dGdnGatingFusion(unittest.TestCase):
    def setUp(self) -> None:
        _require_cuda()
        os.environ["RTP_QWEN35_DECODE_FUSION"] = "1"

    def _case(self, batch: int, seq_lo: int = 16, seq_hi: int = 1024, seed: int = 0):
        device = "cuda"
        if batch == 0:
            sequence_lengths = torch.empty(0, dtype=torch.int32, device=device)
        else:
            sequence_lengths = torch.randint(
                seq_lo, seq_hi, (batch,), dtype=torch.int32, device=device
            )
        return _build_paged_decode(batch, sequence_lengths, seed=seed)

    def test_typical_batch_matches_old_path(self):
        for batch in (1, 8, 32, 128, 256):
            with self.subTest(batch=batch):
                ref_in = self._case(batch, seed=batch)
                fused_in = _clone_inputs(ref_in)
                self.assertTrue(is_supported(**_support_kwargs(ref_in)))
                ref = _run_ref(ref_in)
                got = _run_maybe(fused_in)
                self.assertIsNotNone(got)
                _assert_close_outputs(
                    got, ref, fused_in["conv_state"], ref_in["conv_state"]
                )

    def test_t0_empty_batch(self):
        ref_in = self._case(0)
        fused_in = _clone_inputs(ref_in)
        self.assertTrue(is_supported(**_support_kwargs(ref_in)))
        ref = _run_ref(ref_in)
        got = _run_maybe(fused_in)
        self.assertIsNotNone(got)
        self.assertEqual(got[0].shape, ref[0].shape)
        self.assertEqual(tuple(got[1].shape), (1, 0, _HEADS))
        self.assertEqual(tuple(got[2].shape), (1, 0, _HEADS))

    def test_t3_odd_batch(self):
        ref_in = self._case(3, seed=3)
        fused_in = _clone_inputs(ref_in)
        ref = _run_ref(ref_in)
        got = _run_maybe(fused_in)
        self.assertIsNotNone(got)
        _assert_close_outputs(got, ref, fused_in["conv_state"], ref_in["conv_state"])

    def test_width_4_only(self):
        ref_in = self._case(4, seed=4)
        bad_weight = torch.randn(_DIM, 3, device="cuda", dtype=_DTYPE)
        self.assertFalse(
            is_supported(
                ref_in["x"],
                ref_in["conv_state"],
                bad_weight,
                ref_in["A_log"],
                ref_in["a"],
                ref_in["b"],
                ref_in["dt_bias"],
                block_map=ref_in["block_map"],
                sequence_lengths=ref_in["sequence_lengths"],
            )
        )
        self.assertIsNone(
            maybe_fused_conv1d_update_gdn_gating(
                ref_in["x"],
                ref_in["conv_state"],
                bad_weight,
                ref_in["A_log"],
                ref_in["a"],
                ref_in["b"],
                ref_in["dt_bias"],
                block_map=ref_in["block_map"],
                sequence_lengths=ref_in["sequence_lengths"],
            )
        )

    def test_silu_only(self):
        ref_in = self._case(4, seed=5)
        self.assertFalse(
            is_supported(
                **_support_kwargs(ref_in),
                activation=None,
            )
        )
        self.assertIsNone(
            maybe_fused_conv1d_update_gdn_gating(
                ref_in["x"],
                ref_in["conv_state"],
                ref_in["weight"],
                ref_in["A_log"],
                ref_in["a"],
                ref_in["b"],
                ref_in["dt_bias"],
                block_map=ref_in["block_map"],
                sequence_lengths=ref_in["sequence_lengths"],
                activation=None,
            )
        )

    def test_conv_state_inplace_matches_old_path(self):
        ref_in = self._case(16, seed=6)
        fused_in = _clone_inputs(ref_in)
        before = fused_in["conv_state"].clone()
        _run_ref(ref_in)
        _run_maybe(fused_in)
        self.assertFalse(torch.equal(fused_in["conv_state"], before))
        torch.testing.assert_close(
            fused_in["conv_state"],
            ref_in["conv_state"],
            rtol=_RTOL_BF16,
            atol=_ATOL_BF16,
        )

    def test_ab_extremes(self):
        ref_in = self._case(8, seed=7)
        for a_val, b_val in (
            (40.0, 40.0),
            (-40.0, -40.0),
            (40.0, -40.0),
            (-40.0, 40.0),
        ):
            with self.subTest(a=a_val, b=b_val):
                ref_in["a"] = torch.full_like(ref_in["a"], a_val)
                ref_in["b"] = torch.full_like(ref_in["b"], b_val)
                fused_in = _clone_inputs(ref_in)
                ref = _run_ref(ref_in)
                got = _run_maybe(fused_in)
                self.assertIsNotNone(got)
                self.assertTrue(torch.isfinite(got[1]).all())
                self.assertTrue(torch.isfinite(got[2]).all())
                _assert_close_outputs(
                    got, ref, fused_in["conv_state"], ref_in["conv_state"]
                )

    def test_env_kill_switch(self):
        ref_in = self._case(2, seed=8)
        os.environ["RTP_QWEN35_DECODE_FUSION"] = "0"
        try:
            self.assertFalse(is_supported(**_support_kwargs(ref_in)))
            self.assertIsNone(_run_maybe(ref_in))
        finally:
            os.environ["RTP_QWEN35_DECODE_FUSION"] = "1"

    def test_seq_not_one_unsupported(self):
        ref_in = self._case(4, seed=9)
        x_tv = torch.randn(4, _DIM, 2, device="cuda", dtype=_DTYPE)
        self.assertFalse(
            is_supported(
                x_tv,
                ref_in["conv_state"],
                ref_in["weight"],
                ref_in["A_log"],
                ref_in["a"],
                ref_in["b"],
                ref_in["dt_bias"],
                block_map=ref_in["block_map"],
                sequence_lengths=ref_in["sequence_lengths"],
            )
        )


def _support_kwargs(inputs: dict) -> dict:
    return {
        "x": inputs["x"],
        "conv_state": inputs["conv_state"],
        "weight": inputs["weight"],
        "A_log": inputs["A_log"],
        "a": inputs["a"],
        "b": inputs["b"],
        "dt_bias": inputs["dt_bias"],
        "block_map": inputs["block_map"],
        "sequence_lengths": inputs["sequence_lengths"],
    }


if __name__ == "__main__":
    unittest.main()
