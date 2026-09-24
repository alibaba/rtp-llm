"""Gate-first DeepGEMM experts for the explicit SM120 decode transport."""

from __future__ import annotations

from typing import Dict, Optional

import torch

from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import (
    m_grouped_fp8_fp4_gemm_nt_contiguous,
    m_grouped_fp8_fp4_gemm_nt_masked,
)
from rtp_llm.models_py.kernels.cuda.fp8_kernel import sgl_per_token_group_quant_fp8
from rtp_llm.models_py.triton_kernels.common.activation import (
    create_packed_scale_tensor,
)
from rtp_llm.models_py.triton_kernels.moe.ep_kernels import (
    ep_gather,
    ep_scatter,
    ep_scatter_v2,
    recompute_topk_ids_sum_expert_count,
)
from rtp_llm.models_py.utils.arch import is_sm120
from rtp_llm.models_py.utils.math import align, ceil_div

from ...quant_layouts import FP4_BLOCK, FP8_BLOCK, prepare_fp4_weight_scale_for_deepgemm
from .._silu_mul_fp8_quant_triton import (
    silu_mul_fp8_quant_packed,
    silu_mul_masked_fp8_quant_packed,
)
from .base import RoutedExpertsStrategy

_GROUPED_ALIGNMENT = 128


class Sm120DecodeExperts(RoutedExpertsStrategy):
    name = "sm120_decode_experts"
    _sm120_masked_ws_cache: list = []

    def setup_weights(self, layer_weights: Dict) -> None:
        from rtp_llm.utils.model_weight import W

        cfg = self.cfg
        (E, D, inter) = (cfg.n_routed_experts, cfg.dim, cfg.moe_inter_dim)
        stacked_w1_w = layer_weights.pop(W.v4_routed_w1_w)
        stacked_w1_s = layer_weights.pop(W.v4_routed_w1_s)
        stacked_w2_w = layer_weights.pop(W.v4_routed_w2_w)
        stacked_w2_s = layer_weights.pop(W.v4_routed_w2_s)
        stacked_w3_w = layer_weights.pop(W.v4_routed_w3_w)
        stacked_w3_s = layer_weights.pop(W.v4_routed_w3_s)
        device = stacked_w1_w.device
        self._w13 = torch.empty((E, 2 * inter, D // 2), dtype=torch.int8, device=device)
        s13_raw = torch.empty(
            (E, 2 * inter, D // FP4_BLOCK), dtype=torch.float8_e8m0fnu, device=device
        )
        self._w2 = torch.empty((E, D, inter // 2), dtype=torch.int8, device=device)
        s2_raw = torch.empty(
            (E, D, inter // FP4_BLOCK), dtype=torch.float8_e8m0fnu, device=device
        )
        if not is_sm120(device):
            raise RuntimeError("SM120 decode weights require an exact SM120 device")
        self._w13[:, :inter].copy_(stacked_w1_w)
        s13_raw[:, :inter].copy_(stacked_w1_s)
        self._w13[:, inter:].copy_(stacked_w3_w)
        s13_raw[:, inter:].copy_(stacked_w3_s)
        self._w2.copy_(stacked_w2_w)
        s2_raw.copy_(stacked_w2_s)
        del stacked_w1_w, stacked_w1_s, stacked_w2_w, stacked_w2_s
        del stacked_w3_w, stacked_w3_s
        self._s13 = prepare_fp4_weight_scale_for_deepgemm(s13_raw, 2 * inter, D, E)
        self._s2 = prepare_fp4_weight_scale_for_deepgemm(s2_raw, D, inter, E)
        if self._s13.dtype != torch.int32 or self._s2.dtype != torch.int32:
            raise RuntimeError("SM120 decode requires packed DeepGEMM weight scales")
        del s13_raw, s2_raw
        torch.cuda.empty_cache()

    def _ensure_sm120_masked_workspace(
        self,
        n: int,
        topk: int,
        e: int,
        d: int,
        inter: int,
        alignment: int,
        device: torch.device,
    ) -> dict:
        key = (topk, e, d, inter, device.index)
        cache = Sm120DecodeExperts._sm120_masked_ws_cache
        (n_cap, a_cap) = (n, alignment)
        for cached in cache:
            if cached["key"] != key:
                continue
            if cached["n"] >= n and cached["alignment"] >= alignment:
                return cached
            n_cap = max(n_cap, int(cached["n"]))
            a_cap = max(a_cap, int(cached["alignment"]))
        packed = ceil_div(d // FP8_BLOCK, 4)
        ws = {
            "key": key,
            "n": n_cap,
            "alignment": a_cap,
            "expert_x": torch.empty(
                (e, a_cap, d), dtype=torch.float8_e4m3fn, device=device
            ),
            "expert_x_scale": torch.zeros(
                (e, packed, a_cap), dtype=torch.int32, device=device
            ).transpose(1, 2),
            "gate_up": torch.empty(
                (e, a_cap, 2 * inter), dtype=torch.bfloat16, device=device
            ),
            "down_in": torch.empty(
                (e, a_cap, inter), dtype=torch.float8_e4m3fn, device=device
            ),
            "down_in_scale": create_packed_scale_tensor(
                expert_num=e,
                token_num_padded=a_cap,
                hidden_dim=2 * inter,
                quant_group_size=FP8_BLOCK,
                device=device,
            ),
            "down_out": torch.empty((e, a_cap, d), dtype=torch.bfloat16, device=device),
            "start_loc": torch.empty((e,), dtype=torch.int32, device=device),
            "out_index": torch.empty((n_cap, topk), dtype=torch.int32, device=device),
            "adjusted": torch.empty((n_cap, topk), dtype=torch.int32, device=device),
            "masked_m": torch.empty((e,), dtype=torch.int32, device=device),
            "gather": torch.empty((n_cap, d), dtype=torch.float32, device=device),
        }
        cache.append(ws)
        return ws

    def _forward_sm120_deepgemm_masked(
        self,
        x,
        weights,
        indices,
        input_scale: Optional[torch.Tensor] = None,
        expert_start_id: int = 0,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        cfg = self.cfg
        (n, d) = x.shape
        (e, inter) = (cfg.n_routed_experts, cfg.moe_inter_dim)
        device = x.device
        try:
            n_int = int(n)
        except Exception:
            n_int = 0
        if n_int == 0:
            if out is not None:
                out.zero_()
                return out
            return torch.zeros((n, d), dtype=torch.float32, device=device)
        if self._s13 is None or self._s2 is None:
            raise RuntimeError(
                "SM120 DeepGEMM masked MoE requires DeepGEMM-format weight scales; setup_weights did not build _s13/_s2"
            )
        if input_scale is None:
            (a_fp8, a_scale) = sgl_per_token_group_quant_fp8(
                x if x.is_contiguous() else x.contiguous(),
                group_size=FP8_BLOCK,
                eps=0.0001,
                column_major_scales=True,
                scale_tma_aligned=True,
                scale_ue8m0=True,
            )
        else:
            (a_fp8, a_scale) = (x, input_scale)
        alignment = align(n_int, _GROUPED_ALIGNMENT)
        ws = self._ensure_sm120_masked_workspace(
            n_int, int(indices.size(1)), e, int(d), int(inter), alignment, device
        )
        alignment = int(ws["alignment"])
        expected_m = min(alignment, ceil_div(n_int * int(indices.size(1)), e))
        (adjusted_ids, masked_m) = recompute_topk_ids_sum_expert_count(
            indices,
            current_expert_start_id=int(expert_start_id),
            num_local_experts=e,
            weights=weights,
            adjusted_out=ws["adjusted"][:n_int],
            count_out=ws["masked_m"],
        )
        ep_scatter_v2(
            a_fp8,
            a_scale,
            adjusted_ids,
            alignment,
            ws["start_loc"],
            ws["expert_x"].view(e * alignment, d),
            ws["expert_x_scale"],
            ws["out_index"][:n_int],
            scale_ue8m0=True,
        )
        m_grouped_fp8_fp4_gemm_nt_masked(
            (ws["expert_x"], ws["expert_x_scale"]),
            (self._w13, self._s13),
            ws["gate_up"],
            masked_m,
            expected_m,
            recipe_a=(1, FP8_BLOCK),
            recipe_b=(1, FP4_BLOCK),
            disable_ue8m0_cast=False,
        )
        silu_mul_masked_fp8_quant_packed(
            ws["gate_up"],
            ws["down_in"],
            ws["down_in_scale"],
            masked_m,
            clamp_limit=cfg.swiglu_limit,
            group_size=FP8_BLOCK,
        )
        m_grouped_fp8_fp4_gemm_nt_masked(
            (ws["down_in"], ws["down_in_scale"]),
            (self._w2, self._s2),
            ws["down_out"],
            masked_m,
            expected_m,
            recipe_a=(1, FP8_BLOCK),
            recipe_b=(1, FP4_BLOCK),
            disable_ue8m0_cast=False,
        )
        gather_out = out if out is not None else ws["gather"][:n_int]
        ep_gather(
            ws["down_out"].view(e * alignment, d),
            adjusted_ids,
            weights,
            ws["out_index"][:n_int],
            gather_out,
        )
        return gather_out

    def _forward_sm120_deepgemm(
        self, x, weights, indices, input_scale: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        cfg = self.cfg
        (N, D) = x.shape
        (E, inter) = (cfg.n_routed_experts, cfg.moe_inter_dim)
        device = x.device
        routed_ids = torch.where(weights != 0, indices, torch.full_like(indices, -1))
        if input_scale is not None:
            s = (
                input_scale.reshape(N, D // FP4_BLOCK)
                .view(torch.float8_e8m0fnu)
                .to(torch.bfloat16)
            )
            x = (
                (
                    x.to(torch.bfloat16).view(N, D // FP4_BLOCK, FP4_BLOCK)
                    * s.unsqueeze(2)
                )
                .view(N, D)
                .contiguous()
            )
        (a_fp8, a_scale) = sgl_per_token_group_quant_fp8(
            x.contiguous(),
            group_size=FP8_BLOCK,
            eps=0.0001,
            column_major_scales=True,
            scale_tma_aligned=True,
            scale_ue8m0=True,
        )
        (adjusted_topk_ids, num_recv) = recompute_topk_ids_sum_expert_count(
            routed_ids, current_expert_start_id=0, num_local_experts=E
        )
        num_recv_cpu = num_recv.cpu().tolist()
        aligned_counts_list = [align(c, _GROUPED_ALIGNMENT) for c in num_recv_cpu]
        all_tokens = sum(aligned_counts_list)
        if all_tokens == 0:
            return torch.zeros((N, D), dtype=torch.float32, device=device)
        aligned_counts = torch.tensor(
            aligned_counts_list, dtype=torch.int32, pin_memory=True, device="cpu"
        ).to(device, non_blocking=True)
        scatter_out = torch.empty(
            (all_tokens, D), dtype=torch.float8_e4m3fn, device=device
        )
        scatter_out_scale = torch.zeros(
            [ceil_div(D // FP8_BLOCK, 4), all_tokens], device=device, dtype=torch.int
        ).transpose(0, 1)
        m_indices = torch.empty(all_tokens, dtype=torch.int32, device=device)
        output_index = torch.empty_like(adjusted_topk_ids)
        expert_start_loc = torch.empty_like(aligned_counts)
        ep_scatter(
            a_fp8,
            a_scale,
            adjusted_topk_ids,
            aligned_counts,
            expert_start_loc,
            scatter_out,
            scatter_out_scale,
            m_indices,
            output_index,
            scale_ue8m0=True,
        )
        m_indices.clamp_(min=0, max=E - 1)
        del a_fp8, a_scale
        gate_up = torch.empty(
            all_tokens, 2 * inter, device=device, dtype=torch.bfloat16
        )
        m_grouped_fp8_fp4_gemm_nt_contiguous(
            (scatter_out, scatter_out_scale),
            (self._w13, self._s13),
            gate_up,
            m_indices,
            recipe_a=(1, FP8_BLOCK),
            recipe_b=(1, FP4_BLOCK),
            disable_ue8m0_cast=False,
        )
        del scatter_out, scatter_out_scale
        (h_fp8, h_scale) = silu_mul_fp8_quant_packed(
            gate_up, clamp_limit=cfg.swiglu_limit, group_size=FP8_BLOCK
        )
        del gate_up
        down_out = torch.empty(all_tokens, D, device=device, dtype=torch.bfloat16)
        m_grouped_fp8_fp4_gemm_nt_contiguous(
            (h_fp8, h_scale),
            (self._w2, self._s2),
            down_out,
            m_indices,
            recipe_a=(1, FP8_BLOCK),
            recipe_b=(1, FP4_BLOCK),
            disable_ue8m0_cast=False,
        )
        del h_fp8, h_scale
        gather_out = torch.empty((N, D), dtype=torch.bfloat16, device=device)
        ep_gather(down_out, adjusted_topk_ids, weights, output_index, gather_out)
        return gather_out.float()

    def forward(self, x, weights, indices, input_scale=None):
        if input_scale is None and int(x.size(0)) <= 512:
            return self._forward_sm120_deepgemm_masked(x, weights, indices)
        return self._forward_sm120_deepgemm(x, weights, indices, input_scale)
