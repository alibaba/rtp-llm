"""Smoke / sanity tests for ROCm Quark MXFP4 fused-MoE on MI355.

This mirrors the existing FP8 ROCm smoke tests, but adapts the MXFP4
quantize/dequantize reference from aiter's own quant unit tests so we can do a
true numerical comparison against ``torch_moe_ref`` without relying on
unsupported ``bf16 -> float4`` PyTorch casts.
"""

import unittest
from unittest import SkipTest

import torch

try:
    from aiter.ops.shuffle import shuffle_weight  # noqa: F401

    from rtp_llm.config.model_config import ModelConfig
    from rtp_llm.device import get_current_device
    from rtp_llm.device.device_impl import is_gfx950
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
    from rtp_llm.models_py.modules.factory.fused_moe.impl.rocm.executors.deepep_normal_fused_moe_executor import (
        torch_moe_ref,
    )
    from rtp_llm.models_py.modules.factory.fused_moe.impl.rocm.executors.rocm_moe import (
        RocmExpertsMXFp4,
    )
    from rtp_llm.ops import MoeConfig, ParallelismConfig
    from rtp_llm.utils.model_weight import W

    _IMPORT_ERROR = None
except ImportError as exc:  # stale librtp_compute_ops.so / missing aiter / etc.
    _IMPORT_ERROR = exc


SCALE_GROUP_SIZE = 32


def _torch_dynamic_mxfp4_quant(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure-torch MXFP4 quantization adapted from aiter's quant tests."""
    assert x.dim() == 2, "helper expects [M, N]"
    assert x.shape[-1] % SCALE_GROUP_SIZE == 0, "last dim must be divisible by 32"

    EXP_BIAS_FP32 = 127
    EXP_BIAS_FP4 = 1
    EBITS_F32 = 8
    MBITS_F32 = 23
    EBITS_FP4 = 2
    MBITS_FP4 = 1
    MAX_NORMAL = 6
    MIN_NORMAL = 1
    SIGN_MASK = 1 << (EBITS_FP4 + MBITS_FP4)

    x_fp32 = x.contiguous().to(torch.float32)
    x_blocks = x_fp32.reshape(-1, x_fp32.shape[-1] // SCALE_GROUP_SIZE, SCALE_GROUP_SIZE)
    amax = torch.max(torch.abs(x_blocks), dim=-1).values
    amax = amax.view(torch.int32)
    amax = (amax + 0x200000) & 0xFF800000
    amax = amax.view(torch.float32)
    scale_e8m0_unbiased = torch.log2(amax).floor() - 2
    scale_e8m0_unbiased = torch.clamp(scale_e8m0_unbiased, min=-127, max=127)
    quant_scale = torch.exp2(-scale_e8m0_unbiased)

    qx = (x_blocks * quant_scale.unsqueeze(-1)).view(torch.int32)
    s = qx & 0x80000000
    qx = qx ^ s

    qx_fp32 = qx.view(torch.float32)
    saturate_mask = qx_fp32 >= MAX_NORMAL
    denormal_mask = torch.logical_and(~saturate_mask, qx_fp32 < MIN_NORMAL)
    normal_mask = ~(saturate_mask | denormal_mask)

    denorm_exp = (EXP_BIAS_FP32 - EXP_BIAS_FP4) + (MBITS_F32 - MBITS_FP4) + 1
    denorm_mask_int = denorm_exp << MBITS_F32
    denorm_mask_float = torch.tensor(
        denorm_mask_int, dtype=torch.int32, device=x.device
    ).view(torch.float32)

    denormal_x = (qx_fp32 + denorm_mask_float).view(torch.int32)
    denormal_x -= denorm_mask_int
    denormal_x = denormal_x.to(torch.uint8)

    normal_x = qx
    mant_odd = (normal_x >> (MBITS_F32 - MBITS_FP4)) & 1
    val_to_add = ((EXP_BIAS_FP4 - EXP_BIAS_FP32) << MBITS_F32) + (1 << 21) - 1
    normal_x += val_to_add
    normal_x += mant_odd
    normal_x = (normal_x >> (MBITS_F32 - MBITS_FP4)).to(torch.uint8)

    e2m1_value = torch.full_like(qx, 0x7, dtype=torch.uint8)
    e2m1_value = torch.where(normal_mask, normal_x, e2m1_value)
    e2m1_value = torch.where(denormal_mask, denormal_x, e2m1_value)

    sign_lp = s >> (MBITS_F32 + EBITS_F32 - MBITS_FP4 - EBITS_FP4)
    sign_lp = sign_lp.to(torch.uint8) & SIGN_MASK
    e2m1_value = e2m1_value | sign_lp

    x_mxfp4 = e2m1_value[..., ::2] | (e2m1_value[..., 1::2] << 4)
    x_mxfp4 = torch.flatten(x_mxfp4, -2, -1).reshape(x.shape[0], x.shape[1] // 2)
    bs_e8m0 = scale_e8m0_unbiased.to(torch.uint8) + 127
    return x_mxfp4, bs_e8m0


def _mxfp4_to_f32(x: torch.Tensor) -> torch.Tensor:
    unpacked = torch.empty(
        (*x.shape[:-1], x.shape[-1] * 2), dtype=torch.uint8, device=x.device
    )
    unpacked[..., ::2] = x & 0xF
    unpacked[..., 1::2] = x >> 4
    table = torch.tensor(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            -0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ],
        dtype=torch.float32,
        device=x.device,
    )
    return table[unpacked.long()]


def _e8m0_to_f32(x: torch.Tensor) -> torch.Tensor:
    x_f32 = 2 ** ((x - 127).to(torch.float32))
    x_f32[x_f32 == 128] = float("nan")
    return x_f32


def _dequant_mxfp4(x: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    x_f32 = _mxfp4_to_f32(x)
    scales = scales.repeat_interleave(SCALE_GROUP_SIZE, dim=-1).to(torch.float32)
    scales_f32 = _e8m0_to_f32(scales)[..., : x_f32.shape[-1]]
    return (x_f32 * scales_f32).to(torch.bfloat16)


def _swap_gate_up(w1: torch.Tensor) -> torch.Tensor:
    half = w1.shape[1] // 2
    return torch.cat([w1[:, half:, :], w1[:, :half, :]], dim=1)


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


class _MXFp4MoeBaseTest(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            raise SkipTest("CUDA/HIP not available")
        if not is_gfx950():
            raise SkipTest("Quark MXFP4 fused_moe test is MI355/gfx950 only")
        if _IMPORT_ERROR is not None:
            raise RuntimeError(
                "ROCm MXFP4 fused_moe deps unavailable on gfx950 "
                f"(likely stale librtp_compute_ops.so or missing aiter): {_IMPORT_ERROR}"
            ) from _IMPORT_ERROR
        self.device = "cuda"
        torch.set_default_device(self.device)
        torch.manual_seed(42)
        self.runtime_device = get_current_device()

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

    def _quantize_weight(self, w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        quantized = []
        scales = []
        for expert in range(w.shape[0]):
            q, s = _torch_dynamic_mxfp4_quant(w[expert])
            quantized.append(q)
            scales.append(s)
        return torch.stack(quantized), torch.stack(scales)

    def _prepare_weights(
        self, w1_ckpt: torch.Tensor, w2_ref: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        w1q, s1 = self._quantize_weight(w1_ckpt)
        w2q, s2 = self._quantize_weight(w2_ref)

        weights = {
            W.moe_w1: self.runtime_device.shuffle_moe_weight(
                w1q, torch.uint8, W.moe_w1
            ).contiguous(),
            W.moe_w2: self.runtime_device.shuffle_moe_weight(
                w2q, torch.uint8, W.moe_w2
            ).contiguous(),
            W.moe_s1: self.runtime_device.shuffle_moe_weight(
                s1, torch.uint8, W.moe_s1
            ).contiguous(),
            W.moe_s2: self.runtime_device.shuffle_moe_weight(
                s2, torch.uint8, W.moe_s2
            ).contiguous(),
        }
        w1_deq = _swap_gate_up(_dequant_mxfp4(w1q, s1))
        w2_deq = _dequant_mxfp4(w2q, s2)
        return weights, w1_deq, w2_deq


class RocmExpertsMXFp4Test(_MXFp4MoeBaseTest):
    M, K, N, E, TOP_K = 64, 4096, 256, 16, 10

    def _run(self, apply_router_weight_on_input: bool):
        payload = self._build_payload(self.M, self.K, self.E, self.TOP_K)
        if apply_router_weight_on_input:
            payload.expert_topk_ids = payload.expert_topk_ids[:, :1]
            payload.expert_topk_weights = payload.expert_topk_weights[:, :1]

        gate_ref = (
            torch.randn(
                self.E, self.N, self.K, dtype=torch.bfloat16, device=self.device
            )
            * 0.02
        )
        up_ref = (
            torch.randn(
                self.E, self.N, self.K, dtype=torch.bfloat16, device=self.device
            )
            * 0.02
        )
        # Quark ckpt stores w1 as [up, gate]; the ROCm loader swaps it back to
        # [gate, up] before running fused_moe.
        w1_ckpt = torch.cat([up_ref, gate_ref], dim=1)
        w2_ref = (
            torch.randn(
                self.E, self.K, self.N, dtype=torch.bfloat16, device=self.device
            )
            * 0.02
        )

        weights, w1_deq, w2_deq = self._prepare_weights(w1_ckpt, w2_ref)

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

        executor = RocmExpertsMXFp4(
            _make_config_adapter(self.E, self.TOP_K, 2 * self.N),
            FusedMoEQuantConfig(),
            weights,
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
        self.assertEqual(weights[W.moe_s1].dtype, torch.uint8)
        self.assertEqual(weights[W.moe_s2].dtype, torch.uint8)
        self.assertTrue(torch.isfinite(out).all().item(), "kernel produced non-finite")
        torch.testing.assert_close(out, ref_out, atol=2e-1, rtol=2e-1)

    def test_basic_forward(self):
        self._run(apply_router_weight_on_input=False)

    def test_apply_router_weight_on_input(self):
        self._run(apply_router_weight_on_input=True)

if __name__ == "__main__":
    unittest.main()
