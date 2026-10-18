"""SM120 FP8-activation x MXFP4-weight grouped MoE executor (FlashInfer).

Consumer Blackwell (exact SM 12.0) has no DeepGEMM grouped FP8xFP4 path for
prefill, so this executor runs the rank-local experts on FlashInfer:

* eager: MXFP8 input quant -> Triton ``ep_scatter`` -> FlashInfer
  ``group_gemm_mxfp4_nt_groupwise`` (gate/up) -> fused SwiGLU + MXFP8 quant
  -> grouped down GEMM -> Triton ``ep_gather`` with router weights;
* CUDA graph capture and warmup: FlashInfer ``cutlass_fused_moe``.

Expert ids are in this executor's local space ``[0, n_routed_experts)``.
Zero router weights mark slots that do not belong to this executor.

"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from rtp_llm.models_py.kernels.cuda.quant_layouts import FP4_BLOCK, FP8_BLOCK
from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.warmup_sync import (
    cuda_graph_warmup_forward_enabled,
)
from rtp_llm.models_py.triton_kernels.moe.ep_kernels import (
    ep_gather,
    ep_scatter,
    recompute_topk_ids_sum_expert_count,
)
from rtp_llm.models_py.triton_kernels.moe.silu_mul_fp8_quant import (
    silu_mul_fp8_quant_packed_from_parts,
)
from rtp_llm.models_py.utils.arch import is_sm120
from rtp_llm.models_py.utils.math import align

from .fp8_fp4_base import Fp8Fp4ExecutorBase, split_moe_w13_gate_up

# Base layout -> stable-address power-of-two generations. Older generations
# stay alive because already-captured CUDA graphs may still reference them.
_SM120_FUSED_MOE_WORKSPACES: dict[tuple, list[tuple[int, torch.Tensor]]] = {}


def sm120_grouped_fp4_available() -> bool:
    """Probe the complete eager + CUDA-graph SM120 MoE dependency set."""
    if not torch.cuda.is_available() or not is_sm120():
        return False
    try:
        from flashinfer import block_scale_interleave, mxfp8_quantize
        from flashinfer.fused_moe import (
            cutlass_fused_moe,
            cutlass_fused_moe_workspace_size,
        )
        from flashinfer.fused_moe.core import ActivationType
        from flashinfer.gemm import group_gemm_mxfp4_nt_groupwise
    except (ImportError, AttributeError, OSError, RuntimeError):
        return False
    return (
        all(
            callable(symbol)
            for symbol in (
                block_scale_interleave,
                mxfp8_quantize,
                cutlass_fused_moe,
                cutlass_fused_moe_workspace_size,
                group_gemm_mxfp4_nt_groupwise,
            )
        )
        and getattr(ActivationType, "Swiglu", None) is not None
    )


def _swap_halves_(tensor: torch.Tensor, inter_dim: int) -> None:
    """Swap rows ``[:inter]`` and ``[inter:]`` along dim -2 in place."""
    first = tensor[..., :inter_dim, :]
    second = tensor[..., inter_dim:, :]
    saved = first.clone()
    first.copy_(second)
    second.copy_(saved)


class Sm120GroupedFp4Executor(Fp8Fp4ExecutorBase):
    """Rank-local SM120 MXFP4 experts in a local expert-id space."""

    @classmethod
    def check_conditions(cls, checker: Any, config) -> None:
        super().check_conditions(checker, config)
        checker.check(config.ep_size == 1)
        checker.check(config.dim % FP8_BLOCK == 0)
        checker.check(config.moe_inter_dim % FP8_BLOCK == 0)
        checker.check(sm120_grouped_fp4_available())

    def setup_weights(self, layer_weights: Dict) -> None:
        """Repack canonical ``W.moe_*`` stacks for the FlashInfer SM120 ABI.

        FlashInfer's SM120 MXFP8 x MXFP4 kernels consume ``[up | gate]`` rows
        and block-32 UE8M0 scales in ``block_scale_interleave`` layout.
        """
        from flashinfer import block_scale_interleave

        from rtp_llm.utils.model_weight import W

        cfg = self.cfg
        E, D, inter = cfg.n_routed_experts, cfg.dim, cfg.moe_inter_dim
        w13 = layer_weights.pop(W.moe_w1)
        s13 = layer_weights.pop(W.moe_s1)
        # Validate the declared layout before reordering.
        split_moe_w13_gate_up(w13, inter, cfg.moe_w1_layout)
        split_moe_w13_gate_up(s13, inter, cfg.moe_w1_layout)
        if not is_sm120(w13.device):
            raise RuntimeError(
                "SM120 grouped FP4 weights require an exact SM120 device"
            )
        if cfg.moe_w1_layout == "gate_up":
            _swap_halves_(w13, inter)
            s13 = s13.view(torch.uint8).clone()
            _swap_halves_(s13, inter)
        self._w13 = w13
        self._w2 = layer_weights.pop(W.moe_w2)
        s2 = layer_weights.pop(W.moe_s2)
        self._s13_sm120 = block_scale_interleave(s13.view(torch.uint8)).reshape(
            E, 2 * inter, D // FP4_BLOCK
        )
        self._s2_sm120 = block_scale_interleave(s2.view(torch.uint8)).reshape(
            E, D, inter // FP4_BLOCK
        )
        # Return the raw staging scales to the allocator before flushing its
        # cache so KV-cache capacity planning sees the freed memory.
        del s13, s2
        torch.cuda.empty_cache()

    def forward(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
        input_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return the ``[N, D]`` fp32 routed sum over this executor's experts."""
        if x.size(0) == 0:
            return torch.zeros(x.shape, dtype=torch.float32, device=x.device)
        if input_scale is None and (
            torch.cuda.is_current_stream_capturing()
            or cuda_graph_warmup_forward_enabled()
        ):
            return self._forward_capture(x, weights, indices)
        return self.forward_eager(x, weights, indices, input_scale)

    def forward_eager(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
        input_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Eager FlashInfer path, optionally consuming a pre-quantized input."""
        from flashinfer import block_scale_interleave, mxfp8_quantize
        from flashinfer.gemm import group_gemm_mxfp4_nt_groupwise

        cfg = self.cfg
        n, d = x.shape
        e, inter = cfg.n_routed_experts, cfg.moe_inter_dim
        device = x.device
        routed_ids = torch.where(weights != 0, indices, torch.full_like(indices, -1))
        adjusted_ids, counts = recompute_topk_ids_sum_expert_count(
            routed_ids, current_expert_start_id=0, num_local_experts=e
        )
        counts_list = counts.cpu().tolist()
        aligned_list = [align(int(count), 4) for count in counts_list]
        total_rows = sum(aligned_list)
        if total_rows == 0:
            return torch.zeros((n, d), dtype=torch.float32, device=device)
        aligned = torch.tensor(aligned_list, dtype=torch.int32, pin_memory=True).to(
            device, non_blocking=True
        )
        indptr = torch.zeros(e + 1, dtype=torch.int32, device=device)
        torch.cumsum(aligned, 0, dtype=torch.int32, out=indptr[1:])
        expert_start = torch.empty_like(aligned)
        m_indices = torch.empty(
            align(total_rows, 128), dtype=torch.int32, device=device
        )
        output_index = torch.empty_like(adjusted_ids)
        if input_scale is None:
            x_q, linear_scale = mxfp8_quantize(
                x.contiguous(), is_sf_swizzled_layout=False
            )
            linear_scale = linear_scale.reshape(n, d // FP4_BLOCK).view(torch.uint8)
        else:
            x_q = x
            linear_scale = input_scale.reshape(n, d // FP4_BLOCK).view(torch.uint8)
        routed_q = torch.empty(total_rows, d, dtype=x_q.dtype, device=device)
        routed_scale = torch.zeros(
            total_rows, d // FP4_BLOCK, dtype=torch.uint8, device=device
        )
        ep_scatter(
            x_q,
            linear_scale,
            adjusted_ids,
            aligned,
            expert_start,
            routed_q,
            routed_scale,
            m_indices,
            output_index,
            # FlashInfer's grouped GEMM walks ``indptr`` (align-4 counts);
            # the scatter layout must use the same per-expert alignment.
            align_m=4,
        )
        expert_ids = torch.bucketize(
            torch.arange(total_rows, device=device), indptr[1:], right=True
        ).clamp_max_(e - 1)
        group_ids = torch.arange(e + 1, dtype=torch.int32, device=device)
        sf_offsets = ((indptr + group_ids * 127) // 128) * 128
        scale_rows = torch.arange(total_rows, device=device) + (
            sf_offsets[:-1] - indptr[:-1]
        ).index_select(0, expert_ids)
        sf_rows = int(sf_offsets[-1].item())

        def pack_scale(linear: torch.Tensor) -> torch.Tensor:
            padded = torch.zeros(
                sf_rows, linear.size(1), dtype=torch.uint8, device=device
            )
            padded.index_copy_(0, scale_rows, linear)
            return block_scale_interleave(padded).reshape(sf_rows, linear.size(1))

        def gemm(inp_q, inp_scale, expert_weight, expert_scale):
            return group_gemm_mxfp4_nt_groupwise(
                inp_q,
                expert_weight,
                pack_scale(inp_scale),
                expert_scale,
                indptr,
                tile_n=128,
                out_dtype=torch.bfloat16,
            )

        gate_up = gemm(
            routed_q, routed_scale, self._w13.view(torch.uint8), self._s13_sm120
        )
        up, gate = gate_up[:, :inter], gate_up[:, inter:]
        hidden_q, hidden_scale_packed = silu_mul_fp8_quant_packed_from_parts(
            gate, up, clamp_limit=cfg.swiglu_limit, group_size=FP4_BLOCK
        )
        hidden_scale = (
            hidden_scale_packed.contiguous()
            .view(torch.uint8)
            .reshape(total_rows, inter // FP4_BLOCK)
        )
        down = gemm(hidden_q, hidden_scale, self._w2.view(torch.uint8), self._s2_sm120)
        output = torch.empty((n, d), dtype=torch.float32, device=device)
        ep_gather(down, adjusted_ids, weights, output_index, output)
        return output

    def _get_fused_moe_workspace(
        self, device: torch.device, max_tokens: Optional[int] = None
    ) -> torch.Tensor:
        from flashinfer.fused_moe import cutlass_fused_moe_workspace_size
        from flashinfer.fused_moe.core import ActivationType

        cfg = self.cfg
        # Size the FlashInfer workspace to the actual invocation. Startup CUDA
        # graph warmup can run the decode forward while max_tokens_per_rank
        # still holds the prefill chunk bound; reserving that would waste
        # multiple GiB during decode graph initialization.
        workspace_tokens = max(
            int(cfg.max_tokens_per_rank) if max_tokens is None else int(max_tokens),
            1,
        )
        key = (
            device.index,
            cfg.dim,
            cfg.moe_inter_dim,
            cfg.n_routed_experts,
            cfg.n_activated_experts,
        )
        generations = _SM120_FUSED_MOE_WORKSPACES.setdefault(key, [])
        reusable = [
            (capacity, workspace)
            for capacity, workspace in generations
            if capacity >= workspace_tokens
        ]
        if reusable:
            return min(reusable, key=lambda item: item[0])[1]

        if device.type == "cuda" and torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "SM120 fused MoE workspace must be materialized before CUDA "
                f"graph capture for at least {workspace_tokens} tokens"
            )

        workspace_capacity = (
            1 if workspace_tokens <= 1 else 1 << (workspace_tokens - 1).bit_length()
        )
        workspace_bytes = cutlass_fused_moe_workspace_size(
            workspace_capacity,
            cfg.dim,
            cfg.moe_inter_dim,
            cfg.n_routed_experts,
            cfg.n_activated_experts,
            x_dtype=torch.float8_e4m3fn,
            weight_dtype=torch.long,
            output_dtype=torch.bfloat16,
            activation_type=ActivationType.Swiglu,
            use_mxfp8_act_scaling=True,
            use_fused_finalize=False,
            device=device,
        )
        workspace = torch.empty(workspace_bytes, dtype=torch.uint8, device=device)
        generations.append((workspace_capacity, workspace))
        generations.sort(key=lambda item: item[0])
        return workspace

    def _forward_capture(self, x, weights, indices) -> torch.Tensor:
        from flashinfer import mxfp8_quantize
        from flashinfer.fused_moe import cutlass_fused_moe
        from flashinfer.fused_moe.core import ActivationType

        cfg = self.cfg
        num_experts = cfg.n_routed_experts
        # A CUDA-graph wrapper may retain a fixed-capacity input buffer larger
        # than the per-rank scheduling hint, so size the workspace and tuning
        # bound from this invocation.
        actual_tokens = max(int(x.size(0)), 1)
        fake_input_scale = torch.ones(num_experts, dtype=torch.float32, device=x.device)
        swiglu_limit = torch.full_like(fake_input_scale, cfg.swiglu_limit)
        output = torch.empty_like(x)
        kernel_input, input_sf = mxfp8_quantize(
            x.contiguous(), is_sf_swizzled_layout=True
        )
        # FlashInfer's MXFP8 x MXFP4 CUTLASS ABI consumes the raw packed FP4
        # weights; only the block-32 UE8M0 scales use block_scale_interleave.
        # The TRT-LLM NVFP4 weight shuffle is a block-16 format and must not be
        # applied to these MXFP4 weights.
        routed_ids = torch.where(weights != 0, indices, torch.full_like(indices, -1))
        cutlass_fused_moe(
            input=kernel_input,
            token_selected_experts=routed_ids.to(torch.int32).contiguous(),
            token_final_scales=weights.float().contiguous(),
            fc1_expert_weights=self._w13.view(torch.uint8).view(torch.long),
            fc2_expert_weights=self._w2.view(torch.uint8).view(torch.long),
            output_dtype=torch.bfloat16,
            quant_scales=[
                self._s13_sm120.view(torch.int32),
                fake_input_scale,
                self._s2_sm120.view(torch.int32),
                fake_input_scale,
            ],
            input_sf=input_sf,
            swiglu_limit=swiglu_limit,
            output=output,
            use_mxfp8_act_scaling=True,
            use_fused_finalize=False,
            enable_pdl=False,
            workspace_buffer=self._get_fused_moe_workspace(x.device, actual_tokens),
            tune_max_num_tokens=actual_tokens,
            activation_type=ActivationType.Swiglu,
        )
        return output.float()
