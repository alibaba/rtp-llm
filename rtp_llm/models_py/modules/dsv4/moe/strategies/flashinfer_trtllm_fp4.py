"""FlashInfer TRTLLM MXFP4 routed-expert strategy for DSV4 precision checks.

This is an explicit opt-in path used to align RTP with the vLLM oracle that
selects FlashInfer TRTLLM MXFP4/MXFP8 MoE. It deliberately does not replace
the default DeepGEMM grouped FP4 production path.
"""

from __future__ import annotations

import os
from typing import Dict

import torch

from .base import MoeCfg, RoutedExpertsStrategy, register_strategy


def _opted_in() -> bool:
    return (
        os.environ.get("DSV4_MOE_STRATEGY", "").strip() == "flashinfer_trtllm_fp4"
        or os.environ.get("DSV4_USE_FLASHINFER_TRTLLM_FP4", "0").strip() == "1"
    )


def _has_flashinfer_trtllm_fp4() -> bool:
    if not _opted_in():
        return False
    try:
        from flashinfer import mxfp8_quantize, trtllm_fp4_block_scale_routed_moe  # noqa: F401
        from flashinfer.fp4_quantization import nvfp4_block_scale_interleave  # noqa: F401
        from flashinfer.fused_moe.core import get_w2_permute_indices_with_cache  # noqa: F401
    except Exception:
        return False
    if not torch.cuda.is_available():
        return False
    cap = torch.cuda.get_device_capability()
    return cap[0] == 10


def _pack_topk_ids_weights(topk_ids: torch.Tensor, topk_weights: torch.Tensor) -> torch.Tensor:
    """Pack top-k ids/weights exactly like vLLM's TRTLLM modular path."""
    return (topk_ids.to(torch.int32) << 16) | topk_weights.to(torch.bfloat16).view(torch.int16)


def _to_uint8_bytes(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.uint8)


def _get_w2_permute_indices_on_device(
    cache: dict,
    weight: torch.Tensor,
    epilogue_tile_m: int,
    num_elts_per_sf: int | None = None,
) -> torch.Tensor:
    """Call FlashInfer's helper outside RTP's meta-device construction scope."""
    from flashinfer.fused_moe.core import get_w2_permute_indices_with_cache

    # RTP builds modules under ``torch.device("meta")`` and later materializes
    # them. FlashInfer's helper creates temporary arange tensors without an
    # explicit device, so force those temporaries onto the real weight device.
    with torch.device(weight.device):
        return get_w2_permute_indices_with_cache(
            cache,
            weight,
            epilogue_tile_m,
            num_elts_per_sf=num_elts_per_sf,
        )


def _trtllm_transform_weights(
    w13: torch.Tensor,
    w2: torch.Tensor,
    s13: torch.Tensor,
    s2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert contiguous RTP FP4 expert tensors to FlashInfer TRTLLM layout.

    Mirrors vLLM ``convert_weight_to_mxfp4_moe_kernel_format`` for
    ``FLASHINFER_TRTLLM_MXFP4_MXFP8``:
      1. turn contiguous ``[w1, w3]`` rows into interleaved
         ``[w3_0, w1_0, w3_1, w1_1, ...]`` rows;
      2. apply FlashInfer's transposed-MMA row permutation;
      3. interleave NVFP4 block scales.
    """
    from flashinfer.fp4_quantization import nvfp4_block_scale_interleave

    num_experts = w13.shape[0]
    intermediate_size = w13.shape[1] // 2
    hidden_size = w13.shape[2] * 2
    sf_block_size = 32
    cache: dict[torch.Size, torch.Tensor] = {}
    epilogue_tile_m = 128

    w1 = w13[:, :intermediate_size, :]
    w3 = w13[:, intermediate_size:, :]
    w13 = torch.stack([w3, w1], dim=2).reshape(w13.shape)

    s1 = s13[:, :intermediate_size, :]
    s3 = s13[:, intermediate_size:, :]
    s13 = torch.stack([s3, s1], dim=2).reshape(s13.shape)

    w13_perm = _get_w2_permute_indices_on_device(
        cache,
        w13[0].view(torch.uint8),
        epilogue_tile_m,
    )
    w13 = w13.view(torch.uint8)[:, w13_perm].contiguous()

    s13_perm = _get_w2_permute_indices_on_device(
        cache,
        s13[0].view(torch.uint8),
        epilogue_tile_m,
        num_elts_per_sf=16,
    )
    s13_u8 = s13.view(torch.uint8)[:, s13_perm].contiguous()
    e, n_s, k_s = s13_u8.shape
    s13 = (
        nvfp4_block_scale_interleave(s13_u8.reshape(e * n_s, k_s))
        .reshape(num_experts, 2 * intermediate_size, hidden_size // sf_block_size)
        .view(torch.float8_e4m3fn)
    )

    w2_perm = _get_w2_permute_indices_on_device(
        cache,
        w2[0].view(torch.uint8),
        epilogue_tile_m,
    )
    w2 = w2.view(torch.uint8)[:, w2_perm].contiguous()

    s2_perm = _get_w2_permute_indices_on_device(
        cache,
        s2[0].view(torch.uint8),
        epilogue_tile_m,
        num_elts_per_sf=16,
    )
    s2_u8 = s2.view(torch.uint8)[:, s2_perm].contiguous()
    e2, n2_s, k2_s = s2_u8.shape
    s2 = (
        nvfp4_block_scale_interleave(s2_u8.reshape(e2 * n2_s, k2_s))
        .reshape(num_experts, hidden_size, intermediate_size // sf_block_size)
        .view(torch.float8_e4m3fn)
    )

    return w13, w2, s13, s2


@register_strategy
class FlashInferTrtllmFP4Strategy(RoutedExpertsStrategy):
    name = "flashinfer_trtllm_fp4"

    @classmethod
    def can_handle(cls, cfg: MoeCfg) -> bool:
        return (
            cfg.ep_size == 1
            and cfg.dim % 256 == 0
            and cfg.moe_inter_dim % 256 == 0
            and _has_flashinfer_trtllm_fp4()
        )

    def setup_weights(self, layer_weights: Dict) -> None:
        from rtp_llm.utils.model_weight import W

        cfg = self.cfg
        e, d, inter = cfg.n_routed_experts, cfg.dim, cfg.moe_inter_dim
        w1 = _to_uint8_bytes(layer_weights.pop(W.v4_routed_w1_w)).view(e, inter, d // 2)
        s1 = _to_uint8_bytes(layer_weights.pop(W.v4_routed_w1_s)).view(e, inter, d // 32)
        w2 = _to_uint8_bytes(layer_weights.pop(W.v4_routed_w2_w)).view(e, d, inter // 2)
        s2 = _to_uint8_bytes(layer_weights.pop(W.v4_routed_w2_s)).view(e, d, inter // 32)
        w3 = _to_uint8_bytes(layer_weights.pop(W.v4_routed_w3_w)).view(e, inter, d // 2)
        s3 = _to_uint8_bytes(layer_weights.pop(W.v4_routed_w3_s)).view(e, inter, d // 32)

        w13 = torch.empty((e, 2 * inter, d // 2), dtype=torch.uint8, device=w1.device)
        s13 = torch.empty((e, 2 * inter, d // 32), dtype=torch.uint8, device=s1.device)
        w13[:, :inter].copy_(w1)
        w13[:, inter:].copy_(w3)
        s13[:, :inter].copy_(s1)
        s13[:, inter:].copy_(s3)
        del w1, s1, w3, s3

        self._w13, self._w2, self._s13, self._s2 = _trtllm_transform_weights(w13, w2, s13, s2)
        del w13, w2, s13, s2

        if cfg.swiglu_limit and cfg.swiglu_limit > 0:
            self._gemm1_clamp_limit = torch.full(
                (e,),
                float(cfg.swiglu_limit),
                dtype=torch.float32,
                device=self._w13.device,
            )
        else:
            self._gemm1_clamp_limit = None
        torch.cuda.empty_cache()

    def forward(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        from flashinfer import mxfp8_quantize, trtllm_fp4_block_scale_routed_moe
        from rtp_llm.models_py.modules.dsv4 import _record_tensor as _rt

        cfg = self.cfg
        if x.dtype != torch.bfloat16:
            x = x.to(torch.bfloat16)

        x_quant, x_scale = mxfp8_quantize(
            x,
            is_sf_swizzled_layout=False,
            alignment=256,
        )
        x_scale = x_scale.view(torch.float8_e4m3fn).reshape(*x.shape[:-1], -1)
        packed_topk = _pack_topk_ids_weights(indices, weights)

        if os.environ.get("MOEDBG", "0") != "0" and _rt.should_record_layer(cfg.layer_id):
            _rt.record_if_level(2, f"L{cfg.layer_id:02d}_moe_routed_a1q", x)
            _rt.record_if_level(2, f"L{cfg.layer_id:02d}_moe_routed_prepared_topk_indices", indices)
            _rt.record_if_level(2, f"L{cfg.layer_id:02d}_moe_routed_prepared_topk_weights", weights)
            _rt.record_if_level(2, f"L{cfg.layer_id:02d}_moe_routed_x_quant", x_quant)
            _rt.record_if_level(2, f"L{cfg.layer_id:02d}_moe_routed_x_scale", x_scale)
            _rt.record_if_level(2, f"L{cfg.layer_id:02d}_moe_routed_packed_topk", packed_topk)

        out = torch.empty_like(x)
        trtllm_fp4_block_scale_routed_moe(
            topk_ids=packed_topk,
            routing_bias=None,
            hidden_states=x_quant,
            hidden_states_scale=x_scale,
            gemm1_weights=self._w13,
            gemm1_weights_scale=self._s13,
            gemm1_bias=None,
            gemm1_alpha=None,
            gemm1_beta=None,
            gemm1_clamp_limit=self._gemm1_clamp_limit,
            gemm2_weights=self._w2,
            gemm2_weights_scale=self._s2,
            gemm2_bias=None,
            output1_scale_scalar=None,
            output1_scale_gate_scalar=None,
            output2_scale_scalar=None,
            num_experts=cfg.n_routed_experts,
            top_k=indices.size(-1),
            n_group=None,
            topk_group=None,
            intermediate_size=cfg.moe_inter_dim,
            local_expert_offset=0,
            local_num_experts=cfg.n_routed_experts,
            routed_scaling_factor=None,
            routing_method_type=1,  # FlashInfer RoutingMethodType.Renormalize
            do_finalize=True,
            output=out,
            tune_max_num_tokens=max(int(cfg.max_tokens_per_rank), 1),
        )
        if os.environ.get("MOEDBG", "0") != "0" and _rt.should_record_layer(cfg.layer_id):
            _rt.record_if_level(2, f"L{cfg.layer_id:02d}_moe_routed_trtllm_out", out)
        return out.float()
