"""SM90 GDN preparation regressions on synthetic Qwen4 head geometry."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.model_desc.qwen3_next import Qwen3NextGatedDeltaNetPrefill
from rtp_llm.models_py.triton_kernels.common.scatter_qkv import (
    try_scatter_qkv_l2norm_sm90,
)
from rtp_llm.models_py.triton_kernels.fla import chunk, gdn_gating, gdn_post_conv


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class GdnSm90PreparationTest(unittest.TestCase):
    def setUp(self):
        if torch.version.hip or torch.cuda.get_device_capability() != (9, 0):
            self.skipTest("SM90 required")
        torch.manual_seed(1122)

    def assert_exact(self, actual, expected):
        self.assertEqual(len(actual), len(expected))
        for index, (value, reference) in enumerate(zip(actual, expected)):
            self.assertEqual(value.shape, reference.shape)
            self.assertEqual(value.dtype, reference.dtype)
            self.assertTrue(torch.isfinite(value).all().item(), index)
            self.assertTrue(
                torch.equal(
                    value.contiguous().view(torch.uint8),
                    reference.contiguous().view(torch.uint8),
                ),
                index,
            )

    def post_inputs(self, rows, h=8, hv=24):
        packed = torch.randn(
            rows, (2 * h + hv) * 128, device="cuda", dtype=torch.bfloat16
        )
        projected = torch.randn(
            rows,
            (2 * h + 2 * hv) * 128 + 2 * hv,
            device="cuda",
            dtype=torch.bfloat16,
        )
        b, a = projected[:, -2 * hv : -hv], projected[:, -hv:]
        alog = torch.randn(hv, device="cuda")
        bias = torch.randn(hv, device="cuda", dtype=torch.bfloat16)
        return packed, a, b, alog, bias

    def test_norm_exact_strided_gate_and_tail(self):
        from rtp_llm.models_py.modules.qwen4_exp.gdn_prefill_norm_triton import (
            Qwen4PrefillGatedNorm,
        )
        from rtp_llm.models_py.triton_kernels.common.layernorm_gated import RmsNormGated

        for rows in (2053, 32768):
            for dtype in (torch.bfloat16, torch.float32):
                weight = torch.randn(128, device="cuda", dtype=dtype)
                native = RmsNormGated(weight, group_size=128, activation="sigmoid")
                model = Qwen4PrefillGatedNorm(native, role="PREFILL")
                x = torch.randn(rows, 3072, device="cuda", dtype=torch.bfloat16)
                gate = torch.randn(rows, 8240, device="cuda", dtype=torch.bfloat16)[
                    :, 5120:8192
                ]
                self.assertTrue(model._supports_prefill(x, gate))
                self.assertTrue(torch.equal(model(x, gate), native(x, gate)))
                model.role = "DECODE"
                self.assertFalse(model._supports_prefill(x, gate))

    def test_norm_default_dispatch(self):
        from rtp_llm.models_py.modules.qwen4_exp.gdn_prefill_norm_triton import (
            Qwen4PrefillGatedNorm,
        )
        from rtp_llm.models_py.triton_kernels.common.layernorm_gated import RmsNormGated

        native = RmsNormGated(
            torch.ones(128, device="cuda", dtype=torch.bfloat16),
            group_size=128,
            activation="sigmoid",
        )
        model = Qwen4PrefillGatedNorm(native, role="PREFILL")
        x = torch.ones(2048, 3072, device="cuda", dtype=torch.bfloat16)
        self.assertTrue(model._supports_prefill(x, x))
        self.assertTrue(torch.equal(model(x, x), native(x, x)))

    def test_default_dispatch(self):
        self.assertTrue(gdn_post_conv.SM90_POST_CONV)
        self.assertTrue(chunk.SM90_CHUNK)
        inputs = self.post_inputs(2053)
        with patch.object(
            gdn_post_conv,
            "try_scatter_qkv_l2norm_sm90",
            wraps=try_scatter_qkv_l2norm_sm90,
        ) as scatter:
            prepared = gdn_post_conv.try_post_conv_gating(*inputs, 8, 24, 128, 128)
        self.assertIsNotNone(prepared)
        self.assertFalse(scatter.call_args.kwargs["copy_v"])
        data = self.chunk_inputs([2053])
        self.assertTrue(chunk._use_sm90_chunk(data["k"], data["v"], data["beta"]))

    def test_post_conv_exact(self):
        for rows in (2053, 8192, 32768):
            for h, hv in ((2, 6), (4, 12), (8, 24), (16, 48)):
                packed, a, b, alog, bias = self.post_inputs(rows, h, hv)
                qkv = try_scatter_qkv_l2norm_sm90(packed, h, hv, 128, 128)
                gb = gdn_gating.fused_gdn_gating(alog, a, b, bias)
                with patch.object(gdn_post_conv, "SM90_POST_CONV", True):
                    actual = gdn_post_conv.try_post_conv_gating(
                        packed, a, b, alog, bias, h, hv, 128, 128
                    )
                self.assert_exact(actual, (*qkv, *gb))

    def test_post_conv_graph_fresh_inputs(self):
        inputs = self.post_inputs(2053)
        packed, a, b, alog, bias = inputs
        saved = [value.clone() for value in inputs]
        with patch.object(gdn_post_conv, "SM90_POST_CONV", True):
            gdn_post_conv.try_post_conv_gating(*inputs, 8, 24, 128, 128)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual = gdn_post_conv.try_post_conv_gating(*inputs, 8, 24, 128, 128)
        for value, reference in zip(inputs, saved):
            self.assertTrue(torch.equal(value, reference))
        for _ in range(3):
            for value in inputs:
                value.normal_()
            graph.replay()
            qkv = try_scatter_qkv_l2norm_sm90(packed, 8, 24, 128, 128)
            gb = gdn_gating.fused_gdn_gating(alog, a, b, bias)
            self.assert_exact(actual, (*qkv, *gb))

    def chunk_inputs(self, lengths, h=8, hv=24, beta_dtype=torch.bfloat16):
        rows = sum(lengths)
        q, k = [
            torch.nn.functional.normalize(
                torch.randn(1, rows, h, 128, device="cuda"), dim=-1
            ).to(torch.bfloat16)
            for _ in range(2)
        ]
        cu = torch.tensor(
            [0, *torch.tensor(lengths).cumsum(0).tolist()],
            device="cuda",
            dtype=torch.int32,
        )
        return dict(
            q=q,
            k=k,
            v=torch.randn(1, rows, hv, 128, device="cuda", dtype=torch.bfloat16),
            g=-torch.nn.functional.softplus(torch.randn(1, rows, hv, device="cuda"))
            * 0.03,
            beta=torch.sigmoid(torch.randn(1, rows, hv, device="cuda")).to(beta_dtype),
            scale=128**-0.5,
            initial_state=torch.randn(len(lengths), hv, 128, 128, device="cuda") * 0.02,
            output_final_state=True,
            cu_seqlens=cu,
        )

    def test_chunk_long_and_ragged_exact(self):
        cases = (
            ([8192], 8, 24, torch.bfloat16),
            ([32768], 8, 24, torch.bfloat16),
            ([2053], 4, 12, torch.bfloat16),
            ([2053], 16, 48, torch.float32),
            ([1, 2, 3, 63, 64, 65, 1855], 2, 6, torch.float32),
        )
        for lengths, h, hv, dtype in cases:
            inputs = self.chunk_inputs(lengths, h, hv, dtype)
            saved = {
                name: value.clone()
                for name, value in inputs.items()
                if isinstance(value, torch.Tensor)
            }
            with patch.object(chunk, "SM90_CHUNK", False):
                expected = chunk.chunk_gated_delta_rule_fwd(**inputs)
            with patch.object(chunk, "SM90_CHUNK", True):
                with patch.object(
                    chunk,
                    "chunk_scaled_dot_kkt_fwd",
                    wraps=chunk.chunk_scaled_dot_kkt_fwd,
                ) as call:
                    actual = chunk.chunk_gated_delta_rule_fwd(**inputs)
                call.assert_called_once()
                self.assertEqual(call.call_args.kwargs["num_warps"], 2)
            self.assert_exact(actual, expected)
            for name, value in saved.items():
                self.assertTrue(torch.equal(inputs[name], value), name)

    def test_chunk_graph_fresh_inputs(self):
        inputs = self.chunk_inputs([2053], 2, 6)
        with patch.object(chunk, "SM90_CHUNK", True):
            chunk.chunk_gated_delta_rule_fwd(**inputs)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual = chunk.chunk_gated_delta_rule_fwd(**inputs)
        for _ in range(3):
            inputs["q"].normal_()
            inputs["v"].normal_()
            inputs["initial_state"].normal_().mul_(0.02)
            graph.replay()
            with patch.object(chunk, "SM90_CHUNK", False):
                expected = chunk.chunk_gated_delta_rule_fwd(**inputs)
            self.assert_exact(actual, expected)

    def test_prefill_entry_and_paged_state_exact(self):
        rows, h, hv = 2053, 8, 24
        packed, a, b, alog, bias = self.post_inputs(rows, h, hv)
        page_size, pages = 128, 20
        block_map = torch.arange(pages, device="cuda", dtype=torch.int32).flip(0)[
            None, :
        ]
        prefix = torch.tensor([128], device="cuda", dtype=torch.int32)
        initial_pool = torch.randn(pages, hv, 128, 128, device="cuda") * 0.02
        attn_inputs = SimpleNamespace(
            input_lengths=torch.tensor([rows], dtype=torch.int32),
            cu_seqlens_device=torch.tensor([0, rows], device="cuda", dtype=torch.int32),
            prefix_lengths_device=prefix,
            kv_cache_kernel_block_id_device=block_map,
        )
        meta = SimpleNamespace(get_aiter_gdn_prefill_metadata=lambda cu: None)
        for prefix_value in (0, 128):
            prefix.fill_(prefix_value)
            expected = None
            for post, tuned in (
                (False, False),
                (True, False),
                (False, True),
                (True, True),
            ):
                pool = initial_pool.clone()
                model = SimpleNamespace(
                    alog=alog,
                    dt_bias=bias,
                    local_num_k_heads=h,
                    local_num_v_heads=hv,
                    head_k_dim=128,
                    head_v_dim=128,
                    _get_ssm_states=lambda tensor: pool,
                    _initial_states_buffer=lambda batch, device: torch.empty(
                        batch, hv, 128, 128, device=device
                    ),
                )
                with patch.object(gdn_post_conv, "SM90_POST_CONV", post), patch.object(
                    chunk, "SM90_CHUNK", tuned
                ):
                    output = Qwen3NextGatedDeltaNetPrefill._fla(
                        model, packed, b, a, packed, page_size, attn_inputs, meta
                    )
                if expected is None:
                    expected = (output, pool)
                else:
                    self.assert_exact((output, pool), expected)
            self.assertTrue(
                torch.equal(pool[block_map[0, -1]], initial_pool[block_map[0, -1]])
            )

    def test_unsupported_inputs_fallback(self):
        inputs = self.post_inputs(2053)
        with patch.object(gdn_post_conv, "SM90_POST_CONV", False):
            self.assertIsNone(
                gdn_post_conv.try_post_conv_gating(*inputs, 8, 24, 128, 128)
            )
        with patch.object(gdn_post_conv, "SM90_POST_CONV", True):
            self.assertIsNone(
                gdn_post_conv.try_post_conv_gating(*inputs, 8, 24, 64, 128)
            )
        inputs = self.chunk_inputs([65], 2, 6)
        with patch.object(chunk, "SM90_CHUNK", True):
            self.assertFalse(
                chunk._use_sm90_chunk(inputs["k"], inputs["v"], inputs["beta"])
            )


if __name__ == "__main__":
    unittest.main()
