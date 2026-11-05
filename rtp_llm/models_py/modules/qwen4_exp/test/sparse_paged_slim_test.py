"""Slim-launch contract of the paged sparse GQA stack on SM90.

Covers the three §12.6 launch hotspots of the ``qwen4.qsa.fmha`` section:

1. the Q/K zero-centered RMSNorm Triton paths must engage on SM90 (the gates
   were hard-coded to SM120 only, forcing the 22-launch-per-layer torch
   fallback chain);
2. the attention output gate must fuse into the split/merge kernels for the
   eager incremental-prefill path (no separate ``sigmoid``/``mul`` launches);
3. the attention entry must accept a strided BTHD query view and write its
   output directly in BTHD layout (no transpose ``contiguous`` clones).

The RMSNorm parity itself (bit-exact Triton vs torch on this device) is
additionally pinned by ``prefill_qk_norm_test``/``grouped_qk_norm_test``,
which fail on SM90 until the gates open.
"""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.qwen4_exp.qsa.sparse_paged_fmha import (
    _grouped_split_config,
    sparse_paged_gqa_attn,
)
from rtp_llm.models_py.triton_kernels.common.zero_centered_rms_norm import (
    is_small_qk_supported,
    try_prefill_qk_rmsnorm_,
)

_DEVICE = "cuda"
_PRODUCTION_ENV = {"RTP_LLM_QSA_STRICT_VALIDATION": "0"}
_SM90 = torch.cuda.is_available() and torch.cuda.get_device_capability(0) == (9, 0)


def _paged_fixture(seed=41, query_len=3, selected_width=4, batch=1):
    torch.manual_seed(seed)
    q_heads, kv_heads, head_dim = 4, 2, 64
    page_size, cache_blocks = 4, 8
    q = torch.randn(
        batch, q_heads, query_len, head_dim, device=_DEVICE, dtype=torch.bfloat16
    )
    cache = torch.randn(
        cache_blocks,
        2,
        kv_heads,
        page_size,
        head_dim,
        device=_DEVICE,
        dtype=torch.bfloat16,
    )
    block_table = torch.tensor(
        [[1, 4, 6, 5]], dtype=torch.int32, device=_DEVICE
    ).repeat(batch, 1)
    kv_lens = torch.arange(
        1, query_len + 1, dtype=torch.int32, device=_DEVICE
    ).view(1, query_len) + torch.full((batch, 1), 9, dtype=torch.int32, device=_DEVICE)
    selected = torch.randint(
        0, 9, (batch, query_len, selected_width), device=_DEVICE, dtype=torch.int32
    )
    return q, cache, block_table, kv_lens, selected


def _kernel_names(fn):
    from torch.profiler import ProfilerActivity, profile

    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    return [e.key for e in prof.key_averages() if e.device_type.name == "CUDA"]


@unittest.skipUnless(_SM90, "SM90 required")
class Sm90RmsNormGateTest(unittest.TestCase):
    """Fix 1: the QK RMSNorm Triton paths must run on SM90."""

    @staticmethod
    def _reference(x, qg, kg, heads, eps=1e-6):
        out = x.clone().reshape(x.shape[0], heads + 2, 256)
        for start, stop, gamma in ((0, heads, qg), (heads, heads + 1, kg)):
            value = out[:, start:stop].float()
            out[:, start:stop].copy_(
                (value * torch.rsqrt(value.square().mean(-1, keepdim=True) + eps))
                * (1.0 + gamma.float())
            )
        return out.reshape_as(x)

    def test_small_qk_gate_accepts_sm90_and_matches_torch_bytes(self):
        torch.manual_seed(101)
        x = torch.randn(16, 14 * 256, device=_DEVICE, dtype=torch.bfloat16)
        qg = torch.randn(256, device=_DEVICE, dtype=torch.bfloat16) * 0.07
        kg = torch.randn_like(qg) * 0.11
        self.assertTrue(is_small_qk_supported(x, qg, kg, 12, 1, 256))
        expected = self._reference(x, qg, kg, 12)
        from rtp_llm.models_py.triton_kernels.common.zero_centered_rms_norm import (
            small_qk_rmsnorm_,
        )

        actual = small_qk_rmsnorm_(x, qg, kg, 12, 1, 256, 1e-6)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_prefill_qk_gate_accepts_sm90_and_matches_torch_bytes(self):
        torch.manual_seed(102)
        x = torch.randn(297, 14 * 256, device=_DEVICE, dtype=torch.bfloat16)
        qg = torch.randn(256, device=_DEVICE, dtype=torch.bfloat16) * 0.07
        kg = torch.randn_like(qg) * 0.11
        # The kernel normalises in place: take the reference from the pristine
        # input before the call.
        expected = self._reference(x, qg, kg, 12)
        result = try_prefill_qk_rmsnorm_(x, qg, kg, 12, 1, 256, 1e-6)
        self.assertIsNotNone(result, "prefill QK RMSNorm Triton path rejected SM90")
        torch.testing.assert_close(result, expected, rtol=0, atol=0)


@unittest.skipUnless(_SM90, "SM90 required")
class FusedOutputGateTest(unittest.TestCase):
    """Fix 3: the eager-prefill output gate must fuse into the kernels."""

    def test_gate_fuses_into_single_split_kernel(self):
        q, cache, block_table, kv_lens, selected = _paged_fixture(
            seed=51, query_len=3, selected_width=1
        )
        _, _, _, splits = _grouped_split_config(1 * 3, 2, 1)
        self.assertEqual(splits, 1, "fixture must exercise the single-split path")
        gate = torch.randn(
            1, 3, 4, 64, device=_DEVICE, dtype=torch.bfloat16
        )

        def run_gated():
            return sparse_paged_gqa_attn(
                q,
                cache,
                block_table,
                kv_lens,
                selected,
                page_size=4,
                kv_head_num=2,
                output_gate=gate,
            )

        with patch.dict(os.environ, _PRODUCTION_ENV):
            gated = run_gated()
            ungated = sparse_paged_gqa_attn(
                q,
                cache,
                block_table,
                kv_lens,
                selected,
                page_size=4,
                kv_head_num=2,
            )
        expected = ungated * torch.sigmoid(gate.transpose(1, 2))
        torch.testing.assert_close(gated, expected, rtol=0, atol=0)

        with patch.dict(os.environ, _PRODUCTION_ENV):
            names = _kernel_names(run_gated)
        self.assertFalse(
            [n for n in names if "sigmoid" in n],
            f"output gate ran as separate kernels: {names}",
        )

    def test_prefill_dispatch_routes_output_gate_to_forward(self):
        from rtp_llm.models_py.modules.qwen4_exp.qsa.sparse_gqa_impl import (
            SparseGqaFmhaImpl,
        )

        qkv = torch.randn(5, 4 * 64, device=_DEVICE, dtype=torch.bfloat16)
        gate = torch.randn(5, 4 * 64, device=_DEVICE, dtype=torch.bfloat16)
        result = torch.empty_like(gate)
        owner = SimpleNamespace(
            is_target_verify=False,
            is_prefill=True,
            _is_mtp_draft=False,
            _is_draft_graph=lambda: False,
            _is_prefix_reuse_prefill=lambda: True,
            q_width=4 * 64,
            attn_configs=SimpleNamespace(use_logn_attn=False),
            forward=Mock(return_value=result),
        )
        SparseGqaFmhaImpl.forward_with_output_gate(
            owner, qkv, "cache", 3, gate
        )
        owner.forward.assert_called_once_with(qkv, "cache", 3, output_gate=gate)


@unittest.skipUnless(_SM90, "SM90 required")
class BthdLayoutTest(unittest.TestCase):
    """Fix 2: strided BTHD queries in, BTHD output out, zero layout copies."""

    def test_bthd_output_layout_matches_standard_result(self):
        q, cache, block_table, kv_lens, selected = _paged_fixture(seed=61)

        def run_bthd():
            return sparse_paged_gqa_attn(
                q,
                cache,
                block_table,
                kv_lens,
                selected,
                page_size=4,
                kv_head_num=2,
                bthd_output=True,
            )

        with patch.dict(os.environ, _PRODUCTION_ENV):
            bthd = run_bthd()
            standard = sparse_paged_gqa_attn(
                q,
                cache,
                block_table,
                kv_lens,
                selected,
                page_size=4,
                kv_head_num=2,
            )
        self.assertEqual(tuple(bthd.shape), (1, 3, 4, 64))
        self.assertTrue(bthd.is_contiguous())
        torch.testing.assert_close(bthd, standard.transpose(1, 2), rtol=0, atol=0)

    def test_strided_query_view_avoids_layout_copy(self):
        torch.manual_seed(62)
        q_heads, kv_heads, head_dim, query_len = 4, 2, 64, 5
        qkv = torch.randn(
            query_len, (q_heads + 2 * kv_heads) * head_dim,
            device=_DEVICE, dtype=torch.bfloat16,
        )
        cache = torch.randn(
            8, 2, kv_heads, 4, head_dim, device=_DEVICE, dtype=torch.bfloat16
        )
        block_table = torch.tensor([[1, 4, 6, 5]], dtype=torch.int32, device=_DEVICE)
        kv_lens = (
            torch.arange(1, query_len + 1, dtype=torch.int32, device=_DEVICE)
            .view(1, query_len)
            + 9
        )
        selected = torch.randint(
            0, 9, (1, query_len, 6), device=_DEVICE, dtype=torch.int32
        )

        strided = qkv[:, : q_heads * head_dim].view(
            1, query_len, q_heads, head_dim
        ).transpose(1, 2)
        self.assertFalse(strided.is_contiguous())
        contiguous = strided.contiguous()

        def run_strided():
            return sparse_paged_gqa_attn(
                strided,
                cache,
                block_table,
                kv_lens,
                selected,
                page_size=4,
                kv_head_num=kv_heads,
                validate_selected=False,
            )

        with patch.dict(os.environ, _PRODUCTION_ENV):
            got = run_strided()
            expected = sparse_paged_gqa_attn(
                contiguous,
                cache,
                block_table,
                kv_lens,
                selected,
                page_size=4,
                kv_head_num=kv_heads,
                validate_selected=False,
            )
        torch.testing.assert_close(got, expected, rtol=0, atol=0)

        with patch.dict(os.environ, _PRODUCTION_ENV):
            names = _kernel_names(run_strided)
        copies = [n for n in names if "copy" in n.lower()]
        self.assertFalse(
            copies, f"strided query forced a layout copy kernel: {copies}"
        )

    def test_gate_and_bthd_and_strided_query_together(self):
        torch.manual_seed(63)
        q_heads, kv_heads, head_dim, query_len = 4, 2, 64, 3
        qkv = torch.randn(
            query_len, (q_heads + 2 * kv_heads) * head_dim,
            device=_DEVICE, dtype=torch.bfloat16,
        )
        cache = torch.randn(
            8, 2, kv_heads, 4, head_dim, device=_DEVICE, dtype=torch.bfloat16
        )
        block_table = torch.tensor([[1, 4, 6, 5]], dtype=torch.int32, device=_DEVICE)
        kv_lens = (
            torch.arange(1, query_len + 1, dtype=torch.int32, device=_DEVICE)
            .view(1, query_len)
            + 9
        )
        selected = torch.randint(
            0, 9, (1, query_len, 1), device=_DEVICE, dtype=torch.int32
        )
        gate = torch.randn(
            1, query_len, q_heads, head_dim, device=_DEVICE, dtype=torch.bfloat16
        )
        strided = qkv[:, : q_heads * head_dim].view(
            1, query_len, q_heads, head_dim
        ).transpose(1, 2)

        with patch.dict(os.environ, _PRODUCTION_ENV):
            fused = sparse_paged_gqa_attn(
                strided,
                cache,
                block_table,
                kv_lens,
                selected,
                page_size=4,
                kv_head_num=kv_heads,
                output_gate=gate,
                bthd_output=True,
            )
            ungated = sparse_paged_gqa_attn(
                strided,
                cache,
                block_table,
                kv_lens,
                selected,
                page_size=4,
                kv_head_num=kv_heads,
                bthd_output=True,
            )
        self.assertEqual(tuple(fused.shape), (1, query_len, q_heads, head_dim))
        expected = ungated * torch.sigmoid(gate)
        torch.testing.assert_close(fused, expected, rtol=0, atol=0)


@unittest.skipUnless(_SM90, "SM90 required")
class RaggedPathSlimTest(unittest.TestCase):
    """Fix 2+3 integration: the ragged prefill loop threads gate and layout."""

    @staticmethod
    def _owner(q_width=4 * 64):
        return SimpleNamespace(
            q_width=q_width,
            head_num=4,
            head_dim=64,
            kv_head_num=2,
            _is_mtp_draft=False,
            _paged_kv_scale_args=Mock(return_value={}),
        )

    @staticmethod
    def _plan(cache_blocks=8):
        # Two requests: rows [0, 3) and [3, 5) of a 5-row step.
        r0 = SimpleNamespace(
            start=0,
            end=3,
            kv_lens=torch.tensor([[10, 11, 12]], dtype=torch.int32, device=_DEVICE),
            selected_indices=torch.randint(
                0, 9, (1, 3, 4), device=_DEVICE, dtype=torch.int32
            ),
        )
        r1 = SimpleNamespace(
            start=3,
            end=5,
            kv_lens=torch.tensor([[11, 13]], dtype=torch.int32, device=_DEVICE),
            selected_indices=torch.randint(
                0, 9, (1, 2, 4), device=_DEVICE, dtype=torch.int32
            ),
        )
        return SimpleNamespace(
            tokens=5,
            page_size=4,
            block_table=torch.tensor(
                [[1, 4, 6, 5], [2, 4, 6, 5]], dtype=torch.int32, device=_DEVICE
            ),
            requests=[r0, r1],
        )

    def test_ragged_loop_fuses_gate_and_bthd(self):
        from rtp_llm.models_py.modules.qwen4_exp.qsa.sparse_gqa_impl import (
            SparseGqaFmhaImpl,
        )

        torch.manual_seed(71)
        owner = self._owner()
        plan = self._plan()
        q = torch.randn(5, owner.q_width, device=_DEVICE, dtype=torch.bfloat16)
        gate = torch.randn(5, owner.q_width, device=_DEVICE, dtype=torch.bfloat16)
        cache = torch.randn(
            8, 2, 2, 4, 64, device=_DEVICE, dtype=torch.bfloat16
        )

        with patch.dict(os.environ, _PRODUCTION_ENV):
            fused = SparseGqaFmhaImpl._forward_ragged_paged(
                owner,
                q,
                SimpleNamespace(kv_cache_base=cache),
                plan,
                validate_selected=False,
                output_gate=gate,
            )
            ungated = SparseGqaFmhaImpl._forward_ragged_paged(
                owner,
                q,
                SimpleNamespace(kv_cache_base=cache),
                plan,
                validate_selected=False,
            )
        self.assertEqual(tuple(fused.shape), (5, owner.q_width))
        expected = ungated * torch.sigmoid(gate)
        torch.testing.assert_close(fused, expected, rtol=0, atol=0)

        with patch.dict(os.environ, _PRODUCTION_ENV):
            names = _kernel_names(
                lambda: SparseGqaFmhaImpl._forward_ragged_paged(
                    owner,
                    q,
                    SimpleNamespace(kv_cache_base=cache),
                    plan,
                    validate_selected=False,
                    output_gate=gate,
                )
            )
        self.assertFalse(
            [n for n in names if "sigmoid" in n or "copy" in n or "direct_copy" in n],
            f"ragged loop still launches gate/layout kernels: {names}",
        )


if __name__ == "__main__":
    unittest.main()
