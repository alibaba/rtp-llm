import os
import unittest
from unittest import mock

import torch

from rtp_llm.models_py.model_desc.hy_v4_model import Hy4DecoderLayer
from rtp_llm.models_py.modules.hy_v4.ihc import Hy4IHCHead, Hy4IHCUnit
from rtp_llm.models_py.modules.hy_v4.ihc_triton import (
    maybe_fused_ihc_head,
    maybe_fused_ihc_post,
    maybe_fused_ihc_pre,
)
from rtp_llm.utils.model_weight import W

# DeepGEMM defaults to a relative ``.deep_gemm`` JIT directory. Bazel may run
# NVCC from a different working directory, which makes the generated
# ``kernel.cu`` path disappear from NVCC's point of view. Keep the test cache
# absolute before the availability probe imports deep_gemm.
os.environ.setdefault(
    "DG_JIT_CACHE_DIR",
    os.path.join(os.environ.get("TEST_TMPDIR", "/tmp"), "deep_gemm_cache"),
)


def _deepgemm_prenorm_available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        import deep_gemm

        return (
            torch.cuda.get_device_capability()[0] == 10
            and hasattr(deep_gemm, "tf32_hc_prenorm_gemm")
        )
    except ImportError:
        return False


class _TorchRMSNorm(torch.nn.Module):
    def __init__(self, weight: torch.Tensor, eps: float):
        super().__init__()
        self.register_buffer("weight", weight)
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        values = hidden_states.float()
        rstd = torch.rsqrt(
            values.square().mean(dim=-1, keepdim=True) + self.variance_epsilon
        )
        return (values * rstd * self.weight.float()).to(hidden_states.dtype)


class _AttentionOracleStub(torch.nn.Module):
    """Deterministic attention stand-in used to check decoder-layer wiring."""

    def __init__(self, topk_indices: torch.Tensor):
        super().__init__()
        self.topk_indices = topk_indices
        self.prev_topk_indices = None

    @staticmethod
    def block(hidden_states: torch.Tensor) -> torch.Tensor:
        values = hidden_states.float()
        return (0.625 * values + 0.25 * torch.sin(values)).to(hidden_states.dtype)

    def forward(
        self,
        *,
        hidden_states: torch.Tensor,
        fmha_impl,
        kv_cache=None,
        prev_topk_indices=None,
        return_topk=False,
    ):
        del fmha_impl, kv_cache
        self.prev_topk_indices = prev_topk_indices
        output = self.block(hidden_states)
        if return_topk:
            return output, self.topk_indices
        return output


class _MlpOracleStub(torch.nn.Module):
    """Deterministic MLP stand-in used to check decoder-layer wiring."""

    @staticmethod
    def block(hidden_states: torch.Tensor) -> torch.Tensor:
        values = hidden_states.float()
        return (-0.375 * values + 0.5 * torch.tanh(values)).to(
            hidden_states.dtype
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.block(hidden_states)


class Hy4IhcTest(unittest.TestCase):
    def _unit_weights(self, hidden: int, hc: int):
        return {
            W.hy4_ihc_attn_fn: torch.randn(2 * hc, hc * hidden).float(),
            W.hy4_ihc_attn_scale: torch.tensor([0.2, -0.1]).float(),
            W.hy4_ihc_attn_base: torch.randn(2 * hc).float(),
        }

    def _unit_weights_for_kind(self, hidden: int, hc: int, kind: str):
        prefix = "hy4_ihc_attn" if kind == "attn" else "hy4_ihc_mlp"
        return {
            getattr(W, f"{prefix}_fn"): torch.randn(2 * hc, hc * hidden).float(),
            getattr(W, f"{prefix}_scale"): torch.randn(2).float(),
            getattr(W, f"{prefix}_base"): torch.randn(2 * hc).float(),
        }

    @staticmethod
    def _vllm_pre_oracle(
        channels: torch.Tensor,
        fn_weight: torch.Tensor,
        scale: torch.Tensor,
        base: torch.Tensor,
        magnitude: float,
        hc_eps: float,
        norm_eps: float,
    ):
        """Source-equivalent oracle for vLLM HYV4HCPreLayer.forward."""
        hc = channels.size(1)
        flat = channels.flatten(1).float()
        rstd = torch.rsqrt(flat.square().mean(-1, keepdim=True) + norm_eps)
        mixes = torch.nn.functional.linear(flat, fn_weight) * rstd
        pre_raw, post_raw = mixes.split(hc, dim=-1)
        pre = torch.sigmoid(pre_raw * scale[0] + base[:hc]) + hc_eps
        post = (
            magnitude * torch.sigmoid(post_raw * scale[1] + base[hc:]) + hc_eps
        )
        reduced = torch.sum(pre.unsqueeze(-1) * channels.float(), dim=1)
        return reduced.to(channels.dtype), post

    @staticmethod
    def _vllm_post_oracle(
        block_output: torch.Tensor,
        residual: torch.Tensor,
        post_gate: torch.Tensor,
    ) -> torch.Tensor:
        """Source-equivalent oracle for vLLM HYV4HCPostLayer.forward."""
        output = residual.float() + post_gate.float().unsqueeze(-1) * (
            block_output.float().unsqueeze(1)
        )
        return output.to(block_output.dtype)

    @staticmethod
    def _vllm_head_oracle(
        channels: torch.Tensor,
        fn_weight: torch.Tensor,
        scale: torch.Tensor,
        base: torch.Tensor,
        hc_eps: float,
        norm_eps: float,
    ) -> torch.Tensor:
        """Source-equivalent oracle for vLLM HYV4HCHeadLayer.forward."""
        flat = channels.flatten(1).float()
        rstd = torch.rsqrt(flat.square().mean(-1, keepdim=True) + norm_eps)
        mixes = torch.nn.functional.linear(flat, fn_weight) * rstd
        pre = torch.sigmoid(mixes * scale.float() + base.float()) + hc_eps
        output = torch.sum(pre.unsqueeze(-1) * channels.float(), dim=1)
        return output.to(channels.dtype)

    def test_decoder_layer_and_head_match_vllm_source_oracle(self):
        """Compare RTP's real HY4 layer wiring with vLLM's source equations."""
        torch.manual_seed(37)
        tokens, hidden, hc = 5, 16, 4
        magnitude, hc_eps, norm_eps = 2.0, 1e-6, 1e-5
        attn_weights = self._unit_weights_for_kind(hidden, hc, "attn")
        mlp_weights = self._unit_weights_for_kind(hidden, hc, "mlp")
        all_weights = {**attn_weights, **mlp_weights}

        layer = object.__new__(Hy4DecoderLayer)
        torch.nn.Module.__init__(layer)
        layer.layer_idx = 0
        layer.attn_ihc = Hy4IHCUnit(
            all_weights,
            hidden_size=hidden,
            hc_mult=hc,
            magnitude=magnitude,
            hc_eps=hc_eps,
            norm_eps=norm_eps,
            kind="attn",
        )
        layer.mlp_ihc = Hy4IHCUnit(
            all_weights,
            hidden_size=hidden,
            hc_mult=hc,
            magnitude=magnitude,
            hc_eps=hc_eps,
            norm_eps=norm_eps,
            kind="mlp",
        )
        layer.input_layernorm = _TorchRMSNorm(
            torch.randn(hidden, dtype=torch.bfloat16), norm_eps
        )
        layer.post_attention_layernorm = _TorchRMSNorm(
            torch.randn(hidden, dtype=torch.bfloat16), norm_eps
        )
        topk_indices = torch.arange(tokens * 3).reshape(tokens, 3)
        layer.self_attn = _AttentionOracleStub(topk_indices)
        layer.mlp = _MlpOracleStub()

        channels = torch.randn(tokens, hc, hidden, dtype=torch.bfloat16)
        prev_topk_indices = torch.arange(tokens * 2).reshape(tokens, 2)
        with mock.patch.dict(os.environ, {"RTP_LLM_HY4_IHC_TRITON": "0"}):
            actual = layer(
                channels,
                fmha_impl=object(),
                kv_cache=object(),
                prev_topk_indices=prev_topk_indices,
            )

        attn_read, attn_post = self._vllm_pre_oracle(
            channels,
            attn_weights[W.hy4_ihc_attn_fn],
            attn_weights[W.hy4_ihc_attn_scale],
            attn_weights[W.hy4_ihc_attn_base],
            magnitude,
            hc_eps,
            norm_eps,
        )
        attn_input = layer.input_layernorm(attn_read)
        attn_output = _AttentionOracleStub.block(attn_input)
        after_attn = self._vllm_post_oracle(attn_output, channels, attn_post)
        mlp_read, mlp_post = self._vllm_pre_oracle(
            after_attn,
            mlp_weights[W.hy4_ihc_mlp_fn],
            mlp_weights[W.hy4_ihc_mlp_scale],
            mlp_weights[W.hy4_ihc_mlp_base],
            magnitude,
            hc_eps,
            norm_eps,
        )
        mlp_input = layer.post_attention_layernorm(mlp_read)
        mlp_output = _MlpOracleStub.block(mlp_input)
        expected_channels = self._vllm_post_oracle(
            mlp_output, after_attn, mlp_post
        )

        torch.testing.assert_close(
            actual.channels, expected_channels, rtol=0, atol=0
        )
        self.assertIs(actual.topk_indices, topk_indices)
        self.assertIs(layer.self_attn.prev_topk_indices, prev_topk_indices)

        head_weights = {
            W.hy4_ihc_head_fn: torch.randn(hc, hc * hidden).float(),
            W.hy4_ihc_head_scale: torch.randn(1).float(),
            W.hy4_ihc_head_base: torch.randn(hc).float(),
        }
        head = Hy4IHCHead(
            head_weights,
            hidden_size=hidden,
            hc_mult=hc,
            hc_eps=hc_eps,
            norm_eps=norm_eps,
        )
        final_norm = _TorchRMSNorm(
            torch.randn(hidden, dtype=torch.bfloat16), norm_eps
        )
        actual_hidden = final_norm(head(actual.channels))
        expected_hidden = final_norm(
            self._vllm_head_oracle(
                expected_channels,
                head_weights[W.hy4_ihc_head_fn],
                head_weights[W.hy4_ihc_head_scale],
                head_weights[W.hy4_ihc_head_base],
                hc_eps,
                norm_eps,
            )
        )
        torch.testing.assert_close(actual_hidden, expected_hidden, rtol=0, atol=0)

    def test_pre_post_match_fp32_reference(self):
        torch.manual_seed(7)
        hidden, hc = 5, 4
        weights = self._unit_weights(hidden, hc)
        unit = Hy4IHCUnit(
            weights,
            hidden_size=hidden,
            hc_mult=hc,
            magnitude=2.0,
            hc_eps=1e-6,
            norm_eps=1e-5,
            kind="attn",
            chunk_size=2,
        )
        channels = torch.randn(7, hc, hidden, dtype=torch.bfloat16)
        read, post = unit.pre(channels)

        flat = channels.flatten(1).float()
        rstd = torch.rsqrt(flat.square().mean(-1, keepdim=True) + 1e-5)
        mixes = torch.nn.functional.linear(flat, weights[W.hy4_ihc_attn_fn]) * rstd
        pre_raw, post_raw = mixes.chunk(2, dim=-1)
        pre_ref = (
            torch.sigmoid(
                pre_raw * weights[W.hy4_ihc_attn_scale][0]
                + weights[W.hy4_ihc_attn_base][:hc]
            )
            + 1e-6
        )
        post_ref = (
            2.0
            * torch.sigmoid(
                post_raw * weights[W.hy4_ihc_attn_scale][1]
                + weights[W.hy4_ihc_attn_base][hc:]
            )
            + 1e-6
        )
        read_ref = (pre_ref.unsqueeze(-1) * channels.float()).sum(1).bfloat16()
        torch.testing.assert_close(read, read_ref)
        torch.testing.assert_close(post, post_ref)

        block = torch.randn(7, hidden, dtype=torch.bfloat16)
        actual = unit.post(block, channels, post)
        expected = (
            channels.float() + post_ref.unsqueeze(-1) * block.float().unsqueeze(1)
        ).bfloat16()
        torch.testing.assert_close(actual, expected)

    def test_head_and_single_stream_expansion(self):
        torch.manual_seed(11)
        hidden, hc = 3, 4
        unit_weights = self._unit_weights(hidden, hc)
        unit = Hy4IHCUnit(
            unit_weights,
            hidden_size=hidden,
            hc_mult=hc,
            magnitude=2.0,
            hc_eps=1e-6,
            norm_eps=1e-6,
            kind="attn",
        )
        single = torch.randn(2, hidden)
        expanded = unit.prepare_input(single)
        self.assertEqual(tuple(expanded.shape), (2, hc, hidden))
        for idx in range(hc):
            torch.testing.assert_close(expanded[:, idx], single)

        head_weights = {
            W.hy4_ihc_head_fn: torch.randn(hc, hc * hidden).float(),
            W.hy4_ihc_head_scale: torch.tensor([0.15]).float(),
            W.hy4_ihc_head_base: torch.randn(hc).float(),
        }
        head = Hy4IHCHead(
            head_weights,
            hidden_size=hidden,
            hc_mult=hc,
            hc_eps=1e-6,
            norm_eps=1e-6,
            chunk_size=1,
        )
        actual = head(expanded.bfloat16())
        flat = expanded.flatten(1).float()
        rstd = torch.rsqrt(flat.square().mean(-1, keepdim=True) + 1e-6)
        logits = torch.nn.functional.linear(flat, head_weights[W.hy4_ihc_head_fn])
        gates = (
            torch.sigmoid(
                logits * rstd * head_weights[W.hy4_ihc_head_scale]
                + head_weights[W.hy4_ihc_head_base]
            )
            + 1e-6
        )
        expected = (gates.unsqueeze(-1) * expanded.float()).sum(1).bfloat16()
        torch.testing.assert_close(actual, expected)

    def test_triton_wrappers_reject_cpu_inputs(self):
        hidden, hc = 8, 4
        channels = torch.randn(2, hc, hidden, dtype=torch.bfloat16)
        fn_weight = torch.randn(2 * hc, hc * hidden, dtype=torch.float32)
        scale = torch.randn(2, dtype=torch.float32)
        base = torch.randn(2 * hc, dtype=torch.float32)
        block_output = torch.randn(2, hidden, dtype=torch.bfloat16)
        post_gate = torch.randn(2, hc, dtype=torch.float32)

        self.assertIsNone(
            maybe_fused_ihc_pre(
                channels,
                fn_weight,
                scale,
                base,
                magnitude=2.0,
                hc_eps=1e-6,
                norm_eps=1e-5,
            )
        )
        self.assertIsNone(maybe_fused_ihc_post(block_output, channels, post_gate))
        self.assertIsNone(
            maybe_fused_ihc_head(
                channels,
                fn_weight[:hc],
                scale[:1],
                base[:hc],
                hc_eps=1e-6,
                norm_eps=1e-5,
            )
        )

    def test_empty_inputs_preserve_output_contract(self):
        hidden, hc = 8, 4
        unit = Hy4IHCUnit(
            self._unit_weights(hidden, hc),
            hidden_size=hidden,
            hc_mult=hc,
            magnitude=2.0,
            hc_eps=1e-6,
            norm_eps=1e-5,
            kind="attn",
        )
        channels = torch.empty(0, hc, hidden, dtype=torch.bfloat16)
        read, post_gate = unit.pre(channels)
        output = unit.post(
            torch.empty(0, hidden, dtype=torch.bfloat16), channels, post_gate
        )

        self.assertEqual(tuple(read.shape), (0, hidden))
        self.assertEqual(read.dtype, torch.bfloat16)
        self.assertEqual(tuple(post_gate.shape), (0, hc))
        self.assertEqual(post_gate.dtype, torch.float32)
        self.assertEqual(tuple(output.shape), (0, hc, hidden))
        self.assertEqual(output.dtype, torch.bfloat16)

    def test_pre_normed_tries_split_preserving_grouped_path(self):
        hidden, hc = 8, 4
        unit = Hy4IHCUnit(
            self._unit_weights(hidden, hc),
            hidden_size=hidden,
            hc_mult=hc,
            magnitude=2.0,
            hc_eps=1e-6,
            norm_eps=1e-5,
            kind="attn",
            chunk_size=3,
        )
        channels = torch.randn(7, hc, hidden, dtype=torch.bfloat16)
        norm = _TorchRMSNorm(torch.randn(hidden, dtype=torch.bfloat16), 1e-5)
        expected = (
            torch.empty(7, hidden, dtype=torch.bfloat16),
            torch.empty(7, hc, dtype=torch.float32),
        )

        with mock.patch(
            "rtp_llm.models_py.modules.hy_v4.ihc.maybe_fused_ihc_pre_normed_grouped",
            return_value=expected,
        ) as fused:
            actual = unit.pre_normed(channels, norm)

        self.assertIs(actual, expected)
        fused.assert_called_once()
        self.assertIs(fused.call_args.args[0], channels)
        self.assertEqual(fused.call_args.kwargs["chunk_size"], 3)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA and Triton")
    def test_triton_pre_post_matches_eager_path(self):
        torch.manual_seed(17)
        device = torch.device("cuda")
        hidden, hc = 64, 4
        weights = {
            key: value.to(device)
            for key, value in self._unit_weights(hidden, hc).items()
        }
        unit = Hy4IHCUnit(
            weights,
            hidden_size=hidden,
            hc_mult=hc,
            magnitude=2.0,
            hc_eps=1e-6,
            norm_eps=1e-5,
            kind="attn",
            chunk_size=3,
        )
        channels = torch.randn(
            7, hc, hidden, dtype=torch.bfloat16, device=device
        )
        block_output = torch.randn(
            7, hidden, dtype=torch.bfloat16, device=device
        )

        with torch.no_grad(), mock.patch.dict(
            os.environ, {"RTP_LLM_HY4_IHC_TRITON": "0"}
        ):
            eager_read, eager_gate = unit.pre(channels)
            eager_post = unit.post(block_output, channels, eager_gate)
        with torch.no_grad(), mock.patch.dict(
            os.environ,
            {
                "RTP_LLM_HY4_IHC_TRITON": "1",
                "RTP_LLM_HY4_IHC_PRE_BACKEND": "triton",
            },
        ):
            fused_read, fused_gate = unit.pre(channels)
            fused_post = unit.post(block_output, channels, fused_gate)

        self.assertEqual(fused_read.dtype, torch.bfloat16)
        self.assertEqual(fused_gate.dtype, torch.float32)
        self.assertEqual(fused_post.dtype, torch.bfloat16)
        torch.testing.assert_close(fused_read, eager_read, rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(fused_gate, eager_gate, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(fused_post, eager_post, rtol=2e-2, atol=2e-2)

    @unittest.skipUnless(
        _deepgemm_prenorm_available(), "requires SM100 DeepGEMM prenorm"
    )
    def test_deepgemm_pre_rmsnorm_matches_eager_path(self):
        torch.manual_seed(29)
        device = torch.device("cuda")
        hidden, hc = 64, 4
        weights = {
            key: value.to(device)
            for key, value in self._unit_weights(hidden, hc).items()
        }
        unit = Hy4IHCUnit(
            weights,
            hidden_size=hidden,
            hc_mult=hc,
            magnitude=2.0,
            hc_eps=1e-6,
            norm_eps=1e-5,
            kind="attn",
            chunk_size=3,
        )
        norm = _TorchRMSNorm(
            torch.randn(hidden, dtype=torch.bfloat16, device=device), 1e-5
        )
        channels = torch.randn(
            7, hc, hidden, dtype=torch.bfloat16, device=device
        )

        with torch.no_grad(), mock.patch.dict(
            os.environ, {"RTP_LLM_HY4_IHC_TRITON": "0"}
        ):
            eager_read, eager_gate = unit.pre(channels)
            eager_normed = norm(eager_read)
        with torch.no_grad(), mock.patch.dict(
            os.environ,
            {
                "RTP_LLM_HY4_IHC_TRITON": "1",
                "RTP_LLM_HY4_IHC_PRE_BACKEND": "deepgemm",
            },
        ):
            fused_normed, fused_gate = unit.pre_normed(channels, norm)

        self.assertEqual(fused_normed.dtype, torch.bfloat16)
        self.assertEqual(fused_gate.dtype, torch.float32)
        torch.testing.assert_close(
            fused_normed, eager_normed, rtol=2e-2, atol=2e-2
        )
        torch.testing.assert_close(fused_gate, eager_gate, rtol=5e-4, atol=5e-5)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA and Triton")
    def test_triton_head_matches_eager_path(self):
        torch.manual_seed(23)
        device = torch.device("cuda")
        hidden, hc = 64, 4
        weights = {
            W.hy4_ihc_head_fn: torch.randn(
                hc, hc * hidden, dtype=torch.float32, device=device
            ),
            W.hy4_ihc_head_scale: torch.tensor(
                [0.15], dtype=torch.float32, device=device
            ),
            W.hy4_ihc_head_base: torch.randn(
                hc, dtype=torch.float32, device=device
            ),
        }
        head = Hy4IHCHead(
            weights,
            hidden_size=hidden,
            hc_mult=hc,
            hc_eps=1e-6,
            norm_eps=1e-5,
            chunk_size=3,
        )
        channels = torch.randn(
            7, hc, hidden, dtype=torch.bfloat16, device=device
        )

        with torch.no_grad(), mock.patch.dict(
            os.environ, {"RTP_LLM_HY4_IHC_TRITON": "0"}
        ):
            eager = head(channels)
        with torch.no_grad(), mock.patch.dict(
            os.environ, {"RTP_LLM_HY4_IHC_TRITON": "1"}
        ):
            fused = head(channels)

        self.assertEqual(fused.dtype, torch.bfloat16)
        torch.testing.assert_close(fused, eager, rtol=2e-2, atol=2e-2)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA and Triton")
    def test_production_shape_fused_chain_matches_eager_path(self):
        """Exercise HY4's real hidden=6144 geometry, including split-K paths."""
        torch.manual_seed(31)
        device = torch.device("cuda")
        hidden, hc, tokens = 6144, 4, 3
        weights = {
            key: value.to(device)
            for key, value in self._unit_weights(hidden, hc).items()
        }
        unit = Hy4IHCUnit(
            weights,
            hidden_size=hidden,
            hc_mult=hc,
            magnitude=2.0,
            hc_eps=1e-6,
            norm_eps=1e-5,
            kind="attn",
            chunk_size=2,
        )
        norm = _TorchRMSNorm(
            torch.randn(hidden, dtype=torch.bfloat16, device=device), 1e-5
        )
        head = Hy4IHCHead(
            {
                W.hy4_ihc_head_fn: torch.randn(
                    hc, hc * hidden, dtype=torch.float32, device=device
                ),
                W.hy4_ihc_head_scale: torch.tensor(
                    [0.15], dtype=torch.float32, device=device
                ),
                W.hy4_ihc_head_base: torch.randn(
                    hc, dtype=torch.float32, device=device
                ),
            },
            hidden_size=hidden,
            hc_mult=hc,
            hc_eps=1e-6,
            norm_eps=1e-5,
            chunk_size=2,
        )
        channels = torch.randn(
            tokens, hc, hidden, dtype=torch.bfloat16, device=device
        )
        block_output = torch.randn(
            tokens, hidden, dtype=torch.bfloat16, device=device
        )

        with torch.no_grad(), mock.patch.dict(
            os.environ, {"RTP_LLM_HY4_IHC_TRITON": "0"}
        ):
            eager_read, eager_gate = unit.pre_normed(channels, norm)
            eager_post = unit.post(block_output, channels, eager_gate)
            eager_head = head(eager_post)
        with torch.no_grad(), mock.patch.dict(
            os.environ,
            {
                "RTP_LLM_HY4_IHC_TRITON": "1",
                "RTP_LLM_HY4_IHC_PRE_BACKEND": "triton",
            },
        ):
            fused_read, fused_gate = unit.pre_normed(channels, norm)
            fused_post = unit.post(block_output, channels, fused_gate)
            fused_head = head(fused_post)

        torch.testing.assert_close(fused_read, eager_read, rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(fused_gate, eager_gate, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(fused_post, eager_post, rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(fused_head, eager_head, rtol=2e-2, atol=2e-2)


if __name__ == "__main__":
    unittest.main()
