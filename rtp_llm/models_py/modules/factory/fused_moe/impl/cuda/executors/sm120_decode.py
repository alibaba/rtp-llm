"""Eager SM120 DP/EP decode MoE over the WORLD EP group.

All ranks exchange row counts, dispatch MXFP8 inputs, run local grouped
experts, then return and sum partial outputs. Uneven and idle ranks take
the same collective sequence. CUDA graphs and replicated TP are unsupported.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

import torch

from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import (
    m_grouped_fp8_fp4_gemm_nt_contiguous,
)
from rtp_llm.models_py.kernels.cuda.fp8_kernel import sgl_per_token_group_quant_fp8
from rtp_llm.models_py.kernels.cuda.quant_layouts import (
    FP4_BLOCK,
    FP8_BLOCK,
    prepare_fp4_weight_scale_for_deepgemm,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.sm120.ep_combine import (
    fp32_peer_sum,
    mxfp8_dequant_peer_sum,
)
from rtp_llm.models_py.triton_kernels.moe.ep_kernels import (
    ep_gather,
    ep_scatter,
    recompute_topk_ids_sum_expert_count,
)
from rtp_llm.models_py.triton_kernels.moe.silu_mul_fp8_quant import (
    silu_mul_fp8_quant_packed,
)
from rtp_llm.models_py.utils.arch import is_sm120
from rtp_llm.models_py.utils.math import align, ceil_div

from .fp8_fp4_base import Fp8Fp4ExecutorBase, normalize_moe_w13_gate_up
from .sm120_fused_moe import local_expert_config, validate_world_ep_topology

_logger = logging.getLogger(__name__)
_GROUPED_ALIGNMENT = 128
_A2A_PAYLOAD_TOKEN_BOUND = 65536
_DECODE_CHUNK_TOKENS = 4096


def _decode_chunk_tokens() -> int:
    value = os.environ.get("DSV4_MOE_CHUNK_TOKENS", str(_DECODE_CHUNK_TOKENS))
    try:
        tokens = int(value)
    except ValueError as exc:
        raise ValueError(
            f"DSV4_MOE_CHUNK_TOKENS must be a positive integer, got {value!r}"
        ) from exc
    if tokens <= 0:
        raise ValueError(
            f"DSV4_MOE_CHUNK_TOKENS must be a positive integer, got {value!r}"
        )
    return tokens


def _return_mode() -> str:
    mode = os.environ.get("DSV4_MOE_RETURN_DTYPE", "mxfp8")
    if mode not in ("mxfp8", "bf16", "fp32"):
        raise ValueError(
            f"DSV4_MOE_RETURN_DTYPE must be mxfp8, bf16, or fp32, got {mode!r}"
        )
    return mode


def _a2a_payload_bound_exceeded(recv_counts) -> bool:
    try:
        return sum(recv_counts) > _A2A_PAYLOAD_TOKEN_BOUND or any(
            (c < 0 or c > _A2A_PAYLOAD_TOKEN_BOUND for c in recv_counts)
        )
    except Exception:
        return True


def _validate_a2a_token_counts(
    recv_counts, *, world: int, rank: int, local_tokens: int, max_tokens: int
) -> None:
    # Validate collective metadata before using it for payload allocation.
    if (
        max_tokens <= 0
        or not 0 <= rank < world
        or len(recv_counts) != world
        or any(
            type(count) is not int or not 0 <= count <= max_tokens
            for count in recv_counts
        )
        or recv_counts[rank] != local_tokens
    ):
        raise RuntimeError(
            "SM120 MoE received invalid token counts: "
            f"counts={recv_counts!r}, world={world}, rank={rank}, "
            f"local_tokens={local_tokens}, max_metadata_tokens={max_tokens}; "
            "all EP ranks must use the same eager collective protocol"
        )


class Sm120DecodeExperts(torch.nn.Module):
    """Gate-first DeepGEMM FP8 x FP4 experts in a local expert-id space."""

    def __init__(self, cfg, layer_weights: Dict) -> None:
        super().__init__()
        self.cfg = cfg
        self._setup_weights(layer_weights)

    def _setup_weights(self, layer_weights: Dict) -> None:
        from rtp_llm.utils.model_weight import W

        cfg = self.cfg
        E, D, inter = cfg.n_routed_experts, cfg.dim, cfg.moe_inter_dim
        w13 = layer_weights.pop(W.moe_w1)
        s13_raw = layer_weights.pop(W.moe_s1)
        if not is_sm120(w13.device):
            raise RuntimeError("SM120 decode weights require an exact SM120 device")
        self._w13, s13_raw = normalize_moe_w13_gate_up(
            w13, s13_raw, inter, cfg.moe_w1_layout
        )
        self._w2 = layer_weights.pop(W.moe_w2)
        s2_raw = layer_weights.pop(W.moe_s2)
        self._s13 = prepare_fp4_weight_scale_for_deepgemm(s13_raw, 2 * inter, D, E)
        self._s2 = prepare_fp4_weight_scale_for_deepgemm(s2_raw, D, inter, E)
        if self._s13.dtype != torch.int32 or self._s2.dtype != torch.int32:
            raise RuntimeError("SM120 decode requires packed DeepGEMM weight scales")
        del s13_raw, s2_raw
        torch.cuda.empty_cache()

    def forward_contiguous(
        self, x, weights, indices, input_scale: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        cfg = self.cfg
        N, D = x.shape
        E, inter = cfg.n_routed_experts, cfg.moe_inter_dim
        device = x.device
        routed_ids = torch.where(weights != 0, indices, torch.full_like(indices, -1))
        if input_scale is not None:
            # MXFP8 dispatch payload: dequantize block-32 to BF16 before the
            # block-128 DeepGEMM requant.
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
        a_fp8, a_scale = sgl_per_token_group_quant_fp8(
            x.contiguous(),
            group_size=FP8_BLOCK,
            eps=0.0001,
            column_major_scales=True,
            scale_tma_aligned=True,
            scale_ue8m0=True,
        )
        adjusted_topk_ids, num_recv = recompute_topk_ids_sum_expert_count(
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
        h_fp8, h_scale = silu_mul_fp8_quant_packed(
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
        gather_dtype = torch.float32 if _return_mode() == "fp32" else torch.bfloat16
        gather_out = torch.empty((N, D), dtype=gather_dtype, device=device)
        ep_gather(down_out, adjusted_topk_ids, weights, output_index, gather_out)
        return gather_out.float()

    def forward(self, x, weights, indices, input_scale=None):
        return self.forward_contiguous(x, weights, indices, input_scale)


class Sm120DecodeExecutor(Fp8Fp4ExecutorBase):
    """Explicit SM120 DP/EP decode transport over the WORLD EP group."""

    # Every rank must enter the collectives, including ranks without tokens.
    execute_empty_inputs = True
    # One dispatch per layer on every rank: outer token chunking would let
    # DP ranks with different token counts issue different collective counts.
    single_round_dispatch = True

    @classmethod
    def check_conditions(cls, checker: Any, config) -> None:
        super().check_conditions(checker, config)
        checker.check(config.ep_size > 1)
        checker.check(getattr(config, "attn_tp_size", 1) == 1)
        checker.check(config.world_size == config.ep_size)
        checker.check(config.world_rank == config.ep_rank)
        checker.check(not getattr(config, "moe_cp_enabled", False))
        checker.check(torch.cuda.is_available() and is_sm120())

    def setup_weights(self, layer_weights: Dict) -> None:
        self._chunk_tokens = _decode_chunk_tokens()
        self._experts = Sm120DecodeExperts(local_expert_config(self.cfg), layer_weights)

    def _validate_group(self):
        dist = torch.distributed
        if not dist.is_initialized():
            raise RuntimeError("SM120 decode requires an initialized EP group")
        validate_world_ep_topology(self.cfg, dist, dist.group.WORLD)

    def forward(self, x, weights, indices):
        self._validate_group()
        if not is_sm120(x.device):
            raise RuntimeError("SM120 decode requires an exact SM120 device")
        if torch.cuda.is_current_stream_capturing() or isinstance(
            x.size(0), torch.SymInt
        ):
            raise RuntimeError(
                "sm120_decode requires eager execution; disable CUDA graphs"
            )
        if getattr(self.cfg, "attn_tp_size", 1) != 1:
            raise RuntimeError("sm120_decode requires TP1 with DP=EP=WORLD")
        try:
            return self._forward_all_to_all(x, weights, indices)
        except Exception:
            _logger.exception("SM120 MoE all-to-all failed")
            raise

    def _forward_all_to_all(self, x, weights, indices) -> torch.Tensor:
        from flashinfer import mxfp8_quantize

        dist = torch.distributed
        group = dist.group.WORLD
        world = dist.get_world_size(group)
        cfg = self.cfg
        experts_per_rank = cfg.n_routed_experts // world
        scale_cols = int(x.size(1)) // 32
        topk = weights.size(1)
        x_end = int(x.size(1))
        scale_end = x_end + scale_cols
        weight_end = scale_end + topk * 4
        payload_cols = int(weight_end + topk * 4)
        indices_i32 = indices.to(torch.int32)
        send_counts = [int(x.size(0))] * world
        local_count = torch.tensor([send_counts[0]], dtype=torch.int64, device=x.device)
        gathered = torch.empty(world, 1, dtype=torch.int64, device=x.device)
        dist.all_gather_into_tensor(gathered, local_count, group=group)
        recv_counts = [int(v) for v in gathered.view(-1).cpu().tolist()]
        _validate_a2a_token_counts(
            recv_counts,
            world=world,
            rank=dist.get_rank(group),
            local_tokens=int(x.size(0)),
            # Framework token counts are int32. max_tokens_per_rank can be the
            # chunk size and is not a bound on single-round dispatch rows.
            max_tokens=torch.iinfo(torch.int32).max,
        )
        if _a2a_payload_bound_exceeded(recv_counts):
            raise RuntimeError(
                f"SM120 eager decode supports at most {_A2A_PAYLOAD_TOKEN_BOUND} "
                f"aggregate EP tokens per step, got {sum(recv_counts)}; "
                "reduce the configured batch capacity"
            )
        if sum(recv_counts) == 0:
            return torch.empty_like(x, dtype=torch.float32)
        if x.size(0) == 0:
            x_fp8 = torch.empty_like(x, dtype=torch.float8_e4m3fn)
            x_scale = torch.empty((0, scale_cols), dtype=torch.uint8, device=x.device)
        else:
            x_fp8, x_scale = mxfp8_quantize(x.contiguous(), is_sf_swizzled_layout=False)
            x_scale = x_scale.reshape(x.size(0), scale_cols)
        destination = torch.div(
            indices, experts_per_rank, rounding_mode="floor"
        ).clamp_(0, world - 1)
        recv_payload = torch.empty(
            (sum(recv_counts), payload_cols), dtype=torch.uint8, device=x.device
        )
        send_payload_by_peer = torch.empty(
            (world, int(x.size(0)), payload_cols), dtype=torch.uint8, device=x.device
        )
        send_payload_by_peer[:, :, :x_end].copy_(x_fp8.view(torch.uint8))
        send_payload_by_peer[:, :, x_end:scale_end].copy_(x_scale)
        del x_fp8, x_scale
        send_weights = send_payload_by_peer[:, :, scale_end:weight_end].view(
            torch.float32
        )
        send_indices = send_payload_by_peer[:, :, weight_end:].view(torch.int32)
        for dst in range(world):
            owned = (destination == dst) & (indices >= 0)
            torch.where(
                owned, weights, torch.zeros_like(weights), out=send_weights[dst]
            )
            torch.where(
                owned,
                indices_i32,
                torch.full_like(indices_i32, -1),
                out=send_indices[dst],
            )
        send_payload = send_payload_by_peer.view(-1, payload_cols)
        dist.all_to_all_single(
            recv_payload,
            send_payload,
            output_split_sizes=recv_counts,
            input_split_sizes=send_counts,
            group=group,
        )
        del send_payload, send_payload_by_peer
        combine_payload = self._compute_received(
            recv_payload, sum(recv_counts), x, x_end, scale_end, weight_end
        )
        del recv_payload
        returned_payload = torch.empty(
            (world * int(x.size(0)), int(combine_payload.size(1))),
            dtype=combine_payload.dtype,
            device=x.device,
        )
        dist.all_to_all_single(
            returned_payload,
            combine_payload,
            output_split_sizes=send_counts,
            input_split_sizes=recv_counts,
            group=group,
        )
        del combine_payload
        if x.size(0) == 0:
            return torch.empty_like(x, dtype=torch.float32)
        if _return_mode() != "mxfp8":
            output = torch.empty_like(x, dtype=torch.float32)
            return fp32_peer_sum(returned_payload, world, output)
        return mxfp8_dequant_peer_sum(
            returned_payload, x.size(0), x.size(1), world, out_dtype=torch.float32
        )

    def _compute_received(self, recv_buf, n_rows, x, x_end, scale_end, weight_end):
        from flashinfer import mxfp8_quantize

        cfg = self.cfg
        recv_x = recv_buf[:, :x_end].contiguous().view(torch.float8_e4m3fn)
        recv_scale = recv_buf[:, x_end:scale_end].contiguous()
        recv_w = recv_buf[:, scale_end:weight_end].contiguous().view(torch.float32)
        recv_i = recv_buf[:, weight_end:].contiguous().view(torch.int32).to(torch.int64)
        local_i = recv_i - cfg.local_expert_start
        valid = (local_i >= 0) & (local_i < cfg.n_local_experts)
        local_w = recv_w * valid.to(recv_w.dtype)
        local_i = torch.where(valid, local_i, 0)
        chunk_tokens = self._chunk_tokens
        return_mode = _return_mode()
        partial_dtype = torch.float32 if return_mode == "fp32" else x.dtype
        out = torch.empty(
            (n_rows, int(x.size(1))), dtype=partial_dtype, device=x.device
        )
        for begin in range(0, n_rows, chunk_tokens):
            end = min(begin + chunk_tokens, n_rows)
            out[begin:end] = self._experts(
                recv_x[begin:end],
                local_w[begin:end],
                local_i[begin:end],
                input_scale=recv_scale[begin:end],
            )
        del recv_x, recv_scale, recv_w, recv_i, local_w, local_i, valid
        if return_mode != "mxfp8":
            return out.to(
                torch.float32 if return_mode == "fp32" else torch.bfloat16
            ).contiguous()
        combine_q, combine_scale = mxfp8_quantize(
            out.contiguous(), is_sf_swizzled_layout=False
        )
        combine_scale = combine_scale.reshape(n_rows, int(x.size(1)) // 32)
        return torch.cat(
            [combine_q.view(torch.uint8), combine_scale], dim=1
        ).contiguous()
