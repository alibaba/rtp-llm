"""Smoke / sanity tests for ROCm FP8 fused-MoE executors.

PR #882 review (LLLLKKKK) flagged that ``RocmExpertsFp8PerBlock`` and the
refactored ``RocmExpertsFp8PerChannel`` had no end-to-end coverage. This file
runs each executor against a BF16 reference (computed from the dequantized
weights) and asserts shape, finiteness and approximate numerical agreement.
"""

import unittest
from dataclasses import replace
from unittest import SkipTest

import torch

try:
    from aiter.ops.shuffle import shuffle_weight

    from rtp_llm.config.model_config import ModelConfig
    from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
        MoEConfigAdapter,
    )
    from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
        ExpertForwardPayload,
        ExpertTokensMetadata,
    )
    from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
        FusedMoEQuantConfig,
    )
    from rtp_llm.models_py.modules.factory.fused_moe.impl.rocm._utils import (
        get_rocm_fp8_dtype,
    )
    from rtp_llm.models_py.modules.factory.fused_moe.impl.rocm.executors.deepep_normal_fused_moe_executor import (
        torch_moe_ref,
    )
    from rtp_llm.models_py.modules.factory.fused_moe.impl.rocm.executors.rocm_moe import (
        RocmExpertsFp8PerBlock,
        RocmExpertsFp8PerChannel,
    )
    from rtp_llm.ops import MoeConfig, ParallelismConfig
    from rtp_llm.utils.model_weight import W

    _IMPORT_ERROR = None
except ImportError as exc:  # stale librtp_compute_ops.so / missing aiter / etc.
    _IMPORT_ERROR = exc

FP8_E4M3FNUZ_MAX = 240.0  # max representable in float8_e4m3fnuz


def _per_channel_quant_fp8(w: torch.Tensor, fp8_dtype: torch.dtype):
    """Quantize per output channel. ``w``: [E, OUT, IN]; scale: [E, OUT]."""
    amax = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    scale = (amax / FP8_E4M3FNUZ_MAX).to(torch.float32)
    wq = (w / scale).to(fp8_dtype)
    return wq, scale.squeeze(-1)


def _per_block_quant_fp8(w: torch.Tensor, fp8_dtype: torch.dtype, block: int = 128):
    """Quantize per [block, block] block. ``w``: [E, OUT, IN]; scale: [E, OUT/b, IN/b]."""
    E, OUT, IN = w.shape
    assert OUT % block == 0 and IN % block == 0, "OUT/IN must be divisible by block"
    w_blk = w.reshape(E, OUT // block, block, IN // block, block)
    amax = w_blk.abs().amax(dim=(2, 4), keepdim=True).clamp(min=1e-8)
    scale = (amax / FP8_E4M3FNUZ_MAX).to(torch.float32)
    wq_blk = (w_blk / scale).to(fp8_dtype)
    wq = wq_blk.reshape(E, OUT, IN)
    return wq, scale.squeeze(2).squeeze(-1)


def _dequant_per_channel(wq: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return (wq.to(torch.float32) * scale.unsqueeze(-1)).to(torch.bfloat16)


def _dequant_per_block(
    wq: torch.Tensor, scale: torch.Tensor, block: int = 128
) -> torch.Tensor:
    E, OUT, IN = wq.shape
    wq_blk = wq.to(torch.float32).reshape(E, OUT // block, block, IN // block, block)
    deq = wq_blk * scale.unsqueeze(2).unsqueeze(-1)
    return deq.reshape(E, OUT, IN).to(torch.bfloat16)


def _make_parallelism_config():
    p = ParallelismConfig()
    p.ep_size = 1
    p.ep_rank = 0
    p.tp_size = 1
    p.tp_rank = 0
    p.dp_size = 1
    p.dp_rank = 0
    p.world_size = 1
    p.world_rank = 0
    p.local_rank = 0
    p.local_world_size = 1
    return p


def _make_config_adapter(expert_num: int, top_k: int, inter_dim: int):
    model_config = ModelConfig()
    model_config.attn_config.head_num = 4
    model_config.attn_config.size_per_head = 64
    model_config.num_layers = 2
    model_config.max_seq_len = 2048
    model_config.vocab_size = 32000
    model_config.expert_num = expert_num
    model_config.moe_k = top_k
    model_config.inter_size = inter_dim
    model_config.activation_type = "silu"

    return MoEConfigAdapter(
        model_config=model_config,
        parallelism_config=_make_parallelism_config(),
        moe_config=MoeConfig(),
    )


class _Fp8MoeBaseTest(unittest.TestCase):
    """Shared scaffolding for FP8 MoE executor tests."""

    def setUp(self):
        if _IMPORT_ERROR is not None:
            raise SkipTest(
                f"ROCm fused_moe deps unavailable (likely stale librtp_compute_ops.so): {_IMPORT_ERROR}"
            )
        if not torch.cuda.is_available():
            raise SkipTest("CUDA/HIP not available")
        self.device = "cuda"
        torch.set_default_device(self.device)
        torch.manual_seed(42)
        self.fp8_dtype = get_rocm_fp8_dtype()

    def _build_payload(self, M, K, E, top_k):
        hidden_states = (
            torch.randn(M, K, dtype=torch.bfloat16, device=self.device) * 0.05
        )
        topk_ids = torch.topk(
            torch.rand(M, E, device=self.device), top_k, dim=1
        ).indices.to(torch.int32)
        topk_weights = torch.softmax(
            torch.randn(M, top_k, device=self.device), dim=-1
        ).to(torch.float32)
        return ExpertForwardPayload(
            expert_x=hidden_states,
            expert_x_origin_dtype=hidden_states.dtype,
            expert_x_scale=None,
            expert_tokens_meta=ExpertTokensMetadata(None, None, None),
            expert_topk_ids=topk_ids,
            expert_topk_weights=topk_weights,
        )


class RocmExpertsFp8PerChannelTest(_Fp8MoeBaseTest):
    """Cover the refactored single-aiter-fused_moe path (review item #7)."""

    M, K, N, E, TOP_K = 16, 256, 256, 4, 2

    def _run(self, apply_router_weight_on_input: bool):
        payload = self._build_payload(self.M, self.K, self.E, self.TOP_K)
        # apply_router_weight_on_input requires top_k == 1.
        if apply_router_weight_on_input:
            payload.expert_topk_ids = payload.expert_topk_ids[:, :1]
            payload.expert_topk_weights = payload.expert_topk_weights[:, :1]

        # Random BF16 reference weights, then quantize per-channel.
        w1_ref = (
            torch.randn(
                self.E, 2 * self.N, self.K, dtype=torch.bfloat16, device=self.device
            )
            * 0.02
        )
        w2_ref = (
            torch.randn(
                self.E, self.K, self.N, dtype=torch.bfloat16, device=self.device
            )
            * 0.02
        )

        w1q, s1 = _per_channel_quant_fp8(w1_ref, self.fp8_dtype)
        w2q, s2 = _per_channel_quant_fp8(w2_ref, self.fp8_dtype)

        # Reference: torch_moe_ref against the dequantized weights so both
        # paths see the same numerical content (modulo fp8 rounding).
        w1_deq = _dequant_per_channel(w1q, s1)
        w2_deq = _dequant_per_channel(w2q, s2)
        reference_payload = payload
        if apply_router_weight_on_input:
            reference_payload = replace(
                payload,
                expert_x=payload.expert_x
                * payload.expert_topk_weights.to(payload.expert_x.dtype),
                expert_topk_weights=torch.ones_like(payload.expert_topk_weights),
            )
        ref_out = torch_moe_ref(
            payload=reference_payload,
            activation="silu",
            global_num_experts=self.E,
            expert_map=None,
            a2_scale=None,
            apply_router_weight_on_input=False,
            extra_expert_args=None,
            w1=w1_deq,
            w2=w2_deq,
        )

        # Build the executor.
        config_adapter = _make_config_adapter(self.E, self.TOP_K, 2 * self.N)
        weights = {
            W.moe_w1: shuffle_weight(w1q, (16, 16)),
            W.moe_w2: shuffle_weight(w2q, (16, 16)),
            W.moe_s1: s1,
            W.moe_s2: s2,
        }
        executor = RocmExpertsFp8PerChannel(
            config_adapter, FusedMoEQuantConfig(), weights
        )

        out = executor.execute(
            payload=payload,
            activation="silu",
            expert_map=None,
            a2_scale=None,
            apply_router_weight_on_input=apply_router_weight_on_input,
            extra_expert_args=None,
        ).fused_expert_output

        self.assertEqual(out.shape, (self.M, self.K))
        self.assertTrue(torch.isfinite(out).all().item(), "kernel produced non-finite")
        # FP8 + bf16 accumulation: loose tolerance, mainly a sanity bound.
        torch.testing.assert_close(out, ref_out, atol=5e-2, rtol=5e-2)

    def test_basic_forward(self):
        self._run(apply_router_weight_on_input=False)

    def test_apply_router_weight_on_input(self):
        # PR #882 review #7 specifically called out this newly-enabled path.
        self._run(apply_router_weight_on_input=True)


class RocmExpertsFp8PerChannelReferenceTest(_Fp8MoeBaseTest):
    """Real gfx942 kernels: independent numerical reference and graph replay."""

    def setUp(self):
        super().setUp()
        arch = torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName
        if arch.split(":")[0] != "gfx942":
            raise SkipTest("BF16 top-k accumulation regression is gfx942-specific")

    def _case(self, tokens, topk, input_weight=False, padded=False, graph=False):
        experts, width, inter = 16, 256, (512 if padded else 256)
        payload = self._build_payload(tokens, width, experts, topk)
        payload.expert_x.mul_(4)
        # A nonzero scale makes a zero-output implementation fail the L2 gate.
        w1 = (
            torch.randn(
                experts, 2 * inter, width, device=self.device, dtype=torch.bfloat16
            )
            * 0.15
        )
        w2 = (
            torch.randn(experts, width, inter, device=self.device, dtype=torch.bfloat16)
            * 0.15
        )
        if padded:
            w1[:, 384:inter].zero_()
            w1[:, inter + 384 :].zero_()
            w2[:, :, 384:].zero_()
        q1, s1 = _per_channel_quant_fp8(w1, self.fp8_dtype)
        q2, s2 = _per_channel_quant_fp8(w2, self.fp8_dtype)
        executor = RocmExpertsFp8PerChannel(
            _make_config_adapter(experts, topk, 2 * inter),
            FusedMoEQuantConfig(),
            {
                # Match device_impl.shuffle_moe_weight's physical layout.
                # q1 already has logical [gate, up] order for the reference.
                W.moe_w1: shuffle_weight(q1, (16, 16)),
                W.moe_w2: shuffle_weight(q2, (16, 16)),
                W.moe_s1: s1,
                W.moe_s2: s2,
            },
        )
        if topk == 1 and tokens:
            payload.expert_topk_weights.copy_(
                torch.linspace(0.25, 0.75, tokens, device=self.device)[:, None]
            )
        if tokens > 1:
            payload.expert_topk_weights[-1].zero_()

        def run():
            return executor.execute(
                payload, "silu", None, None, input_weight, None
            ).fused_expert_output

        hidden = payload.expert_x
        route = payload.expert_topk_weights
        if input_weight:
            hidden = hidden * route.to(hidden.dtype)
            route = torch.ones_like(route)
        hidden = hidden.float()
        reference = torch.zeros(tokens, width, dtype=torch.float32, device=self.device)
        # No AITER helper or expanded-token implementation in the reference.
        for expert in range(experts):
            gate, up = torch.nn.functional.linear(
                hidden, q1[expert].float() * s1[expert, :, None]
            ).chunk(2, dim=-1)
            contribution = torch.nn.functional.linear(
                torch.nn.functional.silu(gate) * up,
                q2[expert].float() * s2[expert, :, None],
            )
            weight = (route * (payload.expert_topk_ids == expert)).sum(dim=1)
            reference.add_(contribution * weight[:, None])
        run()
        actual = run().clone()
        self.assertEqual(actual.shape, (tokens, width))
        self.assertEqual(actual.dtype, torch.bfloat16)
        self.assertTrue(torch.isfinite(actual).all().item())
        norm = torch.linalg.vector_norm(reference)
        self.assertGreater(norm.item(), 0.1)
        error = torch.linalg.vector_norm(actual.float() - reference) / norm
        self.assertLess(error.item(), 0.05, "FP8 error exceeds 5% relative L2")
        if tokens > 1:
            self.assertEqual(torch.count_nonzero(actual[-1]).item(), 0)
        if graph:
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    run()
            torch.cuda.current_stream().wait_stream(stream)
            captured = torch.cuda.CUDAGraph()
            with torch.cuda.graph(captured):
                replay_output = run()
            for _ in range(3):
                captured.replay()
                replay_error = (
                    torch.linalg.vector_norm(replay_output.float() - reference) / norm
                )
                self.assertLess(replay_error.item(), 0.05)

    def test_one_token(self):
        self._case(1, 2)

    def test_unaligned_prefill(self):
        self._case(33, 8)

    def test_large_decode_batch(self):
        self._case(512, 8)

    def test_padded_intermediate(self):
        self._case(33, 8, padded=True)

    def test_graph_replay(self):
        self._case(33, 8, graph=True)

    def test_single_expert(self):
        self._case(17, 1)

    def test_input_router_weight(self):
        self._case(17, 1, input_weight=True)


class RocmExpertsFp8PerBlockTest(_Fp8MoeBaseTest):
    """Smoke / sanity for the brand-new PerBlock executor (review item #4)."""

    # Sizes must be divisible by 128 for per-128x128 quant.
    M, K, N, E, TOP_K = 16, 256, 256, 4, 2

    def _run(self, apply_router_weight_on_input: bool):
        payload = self._build_payload(self.M, self.K, self.E, self.TOP_K)
        if apply_router_weight_on_input:
            payload.expert_topk_ids = payload.expert_topk_ids[:, :1]
            payload.expert_topk_weights = payload.expert_topk_weights[:, :1]

        w1_ref = (
            torch.randn(
                self.E, 2 * self.N, self.K, dtype=torch.bfloat16, device=self.device
            )
            * 0.02
        )
        w2_ref = (
            torch.randn(
                self.E, self.K, self.N, dtype=torch.bfloat16, device=self.device
            )
            * 0.02
        )

        w1q, s1 = _per_block_quant_fp8(w1_ref, self.fp8_dtype)
        w2q, s2 = _per_block_quant_fp8(w2_ref, self.fp8_dtype)

        w1_deq = _dequant_per_block(w1q, s1)
        w2_deq = _dequant_per_block(w2q, s2)
        ref_out = torch_moe_ref(
            payload=payload,
            activation="silu",
            global_num_experts=self.E,
            expert_map=None,
            a2_scale=None,
            apply_router_weight_on_input=apply_router_weight_on_input,
            extra_expert_args=None,
            w1=w1_deq,
            w2=w2_deq,
        )

        config_adapter = _make_config_adapter(self.E, self.TOP_K, 2 * self.N)
        weights = {
            W.moe_w1: w1q,
            W.moe_w2: w2q,
            W.moe_s1: s1,
            W.moe_s2: s2,
        }
        executor = RocmExpertsFp8PerBlock(
            config_adapter, FusedMoEQuantConfig(), weights
        )

        out = executor.execute(
            payload=payload,
            activation="silu",
            expert_map=None,
            a2_scale=None,
            apply_router_weight_on_input=apply_router_weight_on_input,
            extra_expert_args=None,
        ).fused_expert_output

        self.assertEqual(out.shape, (self.M, self.K))
        self.assertTrue(torch.isfinite(out).all().item(), "kernel produced non-finite")
        torch.testing.assert_close(out, ref_out, atol=5e-2, rtol=5e-2)

    def test_basic_forward(self):
        self._run(apply_router_weight_on_input=False)

    def test_apply_router_weight_on_input(self):
        self._run(apply_router_weight_on_input=True)


if __name__ == "__main__":
    unittest.main()
