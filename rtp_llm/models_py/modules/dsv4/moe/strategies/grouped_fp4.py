"""GroupedFP4Strategy: single-card DeepGEMM FP8-act x FP4-weight MoE.

This mirrors vLLM's DeepSeek-V4 ``DeepGemmFP4Experts`` backend: route tokens
by expert, run grouped ``m_grouped_fp8_fp4_gemm_nt_contiguous`` for the
gate/up projection, fuse SiLU+mul+FP8 requant, run grouped down projection,
then gather/reduce by router weight.

Single-card + DeepGEMM ≥ 2.4 + SM100 selects this path by default. Set
``DSV4_USE_GROUPED_FP4=0`` or ``DSV4_MOE_STRATEGY=local_loop`` to opt out.

Forward is the 4-opt prefill path:
  (1) quant input ONCE pre-permute (vs. ×topk on padded buffer)
  (2) Triton ep_scatter (vs. argsort + bincount + cumsum + index_copy chain)
  (3) Fused silu+mul+fp8 quant kernel (vs. clamp + silu + mul + cast + quant)
  (4) Triton ep_gather with router-weight reduce (vs. index_select +
      fp32 [N,topk,D] materialize + sum)
"""

from __future__ import annotations

import os
from typing import Dict, Optional

import torch

from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import (
    fp8_fp4_gemm_nt,
    m_grouped_fp8_fp4_gemm_nt_contiguous,
)
from rtp_llm.models_py.kernels.cuda.fp8_kernel import sgl_per_token_group_quant_fp8
from rtp_llm.models_py.triton_kernels.moe.ep_kernels import (
    ep_gather,
    ep_scatter,
    recompute_topk_ids_sum_expert_count,
)
from rtp_llm.models_py.utils.math import align, ceil_div

from .base import MoeCfg, RoutedExpertsStrategy, register_strategy
from .._silu_mul_fp8_quant_triton import (
    silu_mul_fp8_quant_packed,
    silu_mul_fp8_quant_packed_from_parts,
)
from ..warmup_sync import cuda_graph_warmup_forward_enabled
from ...quant_layouts import FP4_BLOCK, FP8_BLOCK, prepare_fp4_weight_scale_for_deepgemm
from rtp_llm.models_py.utils.arch import is_sm120


# ep_scatter requires m_indices.shape[0] % BLOCK_E == 0 (BLOCK_E=128); also
# DeepGEMM contiguous requires per-expert M to be a multiple of the kernel's
# alignment (128 on SM100). We use the same constant.
_GROUPED_ALIGNMENT = 128
_SM120_FUSED_MOE_WORKSPACES = {}


def _has_fp8_fp4_grouped_kernel() -> bool:
    """True iff the grouped FP4 routed-expert path should be used.

    Requires deep_gemm ≥ 2.4 (ships ``m_grouped_fp8_fp4_gemm_nt_contiguous``)
    and an SM100 device.

    ``DSV4_USE_GROUPED_FP4`` semantics:
      - unset / "auto": enable when the runtime supports the vLLM-style
        DeepGEMM FP8xFP4 backend.
      - "0": disable and fall back to LocalLoopStrategy.
      - "1": request this strategy, but still require the kernel/device probe
        to pass.
    """
    flag = os.environ.get("DSV4_USE_GROUPED_FP4", "auto").strip().lower()
    if flag in ("0", "false", "off", "no"):
        return False
    if not torch.cuda.is_available():
        return False
    cap = torch.cuda.get_device_capability()
    if is_sm120():
        try:
            from flashinfer.gemm import group_gemm_mxfp4_nt_groupwise
            from flashinfer import mxfp8_quantize, block_scale_interleave
            return all((group_gemm_mxfp4_nt_groupwise, mxfp8_quantize,
                        block_scale_interleave))
        except Exception:
            return False
    try:
        import deep_gemm
    except Exception:
        return False
    if not hasattr(deep_gemm, "m_grouped_fp8_fp4_gemm_nt_contiguous"):
        return False
    if not hasattr(deep_gemm, "get_mk_alignment_for_contiguous_layout"):
        return False
    return cap[0] == 10


@register_strategy
class GroupedFP4Strategy(RoutedExpertsStrategy):
    name = "grouped_fp4"

    @classmethod
    def can_handle(cls, cfg: MoeCfg) -> bool:
        return (
            cfg.ep_size == 1
            and cfg.dim % FP8_BLOCK == 0
            and cfg.moe_inter_dim % FP8_BLOCK == 0
            and _has_fp8_fp4_grouped_kernel()
        )

    def setup_weights(self, layer_weights: Dict) -> None:
        """Stack EP-sliced routed-expert tensors into ``[E, ...]`` int8 +
        UE8M0 SF buffers in the layout DeepGEMM's contiguous kernel reads.

        Pops keys: ``W.v4_routed_w{1,2,3}_{w,s}`` from ``layer_weights``
        (each shaped ``[E_local, ...]``).

        Memory: pop the framework's stacked tensors so the only references
        kept alive are the repacked grouped buffers below, then bulk-copy
        in one `[:, :inter].copy_(stacked)` shot per slice (vs the legacy
        per-expert loop) — same allocation footprint, simpler code path.
        ``torch.cuda.empty_cache()`` after the copies returns the freed
        FP4 blocks to the CUDA driver so they don't sit in PyTorch's
        caching allocator while KV-pool sizing measures available HBM.
        """
        from rtp_llm.utils.model_weight import W

        cfg = self.cfg
        E, D, inter = cfg.n_routed_experts, cfg.dim, cfg.moe_inter_dim
        stacked_w1_w = layer_weights.pop(W.v4_routed_w1_w)
        stacked_w1_s = layer_weights.pop(W.v4_routed_w1_s)
        stacked_w2_w = layer_weights.pop(W.v4_routed_w2_w)
        stacked_w2_s = layer_weights.pop(W.v4_routed_w2_s)
        stacked_w3_w = layer_weights.pop(W.v4_routed_w3_w)
        stacked_w3_s = layer_weights.pop(W.v4_routed_w3_s)
        device = stacked_w1_w.device

        self._w13 = torch.empty(
            (E, 2 * inter, D // 2), dtype=torch.int8, device=device
        )
        s13_raw = torch.empty(
            (E, 2 * inter, D // FP4_BLOCK),
            dtype=torch.float8_e8m0fnu,
            device=device,
        )
        self._w2 = torch.empty((E, D, inter // 2), dtype=torch.int8, device=device)
        s2_raw = torch.empty(
            (E, D, inter // FP4_BLOCK),
            dtype=torch.float8_e8m0fnu,
            device=device,
        )
        # Bulk copy from stacked → repacked layout (one slice per dim,
        # no per-expert iteration).
        sm120 = is_sm120(device)
        if sm120:
            self._w13[:, :inter].copy_(stacked_w3_w)
            s13_raw[:, :inter].copy_(stacked_w3_s)
            self._w13[:, inter:].copy_(stacked_w1_w)
            s13_raw[:, inter:].copy_(stacked_w1_s)
        else:
            self._w13[:, :inter].copy_(stacked_w1_w)
            s13_raw[:, :inter].copy_(stacked_w1_s)
            self._w13[:, inter:].copy_(stacked_w3_w)
            s13_raw[:, inter:].copy_(stacked_w3_s)
        self._w2.copy_(stacked_w2_w)
        s2_raw.copy_(stacked_w2_s)
        del stacked_w1_w, stacked_w1_s, stacked_w2_w, stacked_w2_s
        del stacked_w3_w, stacked_w3_s
        if sm120:
            from flashinfer import block_scale_interleave
            self._s13_sm120 = block_scale_interleave(
                s13_raw.view(torch.uint8)
            ).reshape(E, 2 * inter, D // FP4_BLOCK)
            self._s2_sm120 = block_scale_interleave(
                s2_raw.view(torch.uint8)
            ).reshape(E, D, inter // FP4_BLOCK)
            self._s13 = self._s2 = None
            self._s13_dense_t = self._s2_dense_t = None
            torch.cuda.empty_cache()
            return

        self._s13 = prepare_fp4_weight_scale_for_deepgemm(
            s13_raw, 2 * inter, D, E
        )
        self._s2 = prepare_fp4_weight_scale_for_deepgemm(s2_raw, D, inter, E)
        s13_dense = prepare_fp4_weight_scale_for_deepgemm(
            s13_raw.reshape(E * 2 * inter, D // FP4_BLOCK),
            E * 2 * inter,
            D,
        )
        self._s13_dense_t = s13_dense.as_strided(
            (E, s13_dense.size(1), 2 * inter),
            (2 * inter, E * 2 * inter, 1),
        )
        s2_dense = prepare_fp4_weight_scale_for_deepgemm(
            s2_raw.reshape(E * D, inter // FP4_BLOCK),
            E * D,
            inter,
        )
        self._s2_dense_t = s2_dense.as_strided(
            (E, s2_dense.size(1), D),
            (D, E * D, 1),
        )
        del s13_raw, s2_raw

        # Return loader's freed FP4 blocks to CUDA so the KV-cache
        # planner sees the real residual HBM rather than what's
        # cached-but-unused inside PyTorch's allocator.
        torch.cuda.empty_cache()

    def forward(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """4-opt prefill path; returns ``[N, D] fp32``.

        Args:
          x: ``[N, D]`` BF16 flattened tokens (post-MoE-gate activation).
          weights: ``[N, topk]`` FP32 router weights.
          indices: ``[N, topk]`` int64 expert IDs.

        Returns:
          y: ``[N, D]`` float32 sum over (token, top-k) of
             ``weight * expert[idx](x)``.
        """
        cfg = self.cfg
        N, D = x.shape
        E = cfg.n_routed_experts
        inter = cfg.moe_inter_dim
        device = x.device

        if N == 0:
            return torch.zeros(N, D, dtype=torch.float32, device=device)
        if is_sm120(device):
            if torch.cuda.is_current_stream_capturing() or cuda_graph_warmup_forward_enabled():
                return self._forward_capture_sm120(x, weights, indices)
            return self._forward_sm120(x, weights, indices)
        if torch.cuda.is_current_stream_capturing():
            return self._forward_capture_topk(x, weights, indices)

        # (1) Quant input ONCE — column-major TMA-aligned UE8M0 packed scale,
        # shape compatible with both ep_scatter input and DeepGEMM contiguous.
        a_fp8, a_scale = sgl_per_token_group_quant_fp8(
            x.contiguous(),
            group_size=FP8_BLOCK,
            eps=1e-4,
            column_major_scales=True,
            scale_tma_aligned=True,
            scale_ue8m0=True,
        )

        # Per-expert counts in local index space (== global since ep_size==1).
        adjusted_topk_ids, num_recv = recompute_topk_ids_sum_expert_count(
            indices,
            current_expert_start_id=0,
            num_local_experts=E,
        )

        # Sum of aligned counts → all_tokens (CPU sync, ~E ints; same kind of
        # sync the framework's contiguous executor does at deepgemm_hybrid_executor.py:445).
        num_recv_cpu = num_recv.cpu().tolist()
        aligned_counts_list = [align(c, _GROUPED_ALIGNMENT) for c in num_recv_cpu]
        all_tokens = sum(aligned_counts_list)
        if all_tokens == 0:
            return torch.zeros((N, D), dtype=torch.float32, device=device)

        # ep_scatter's kernel_1 builds expert_start_loc as the EXCLUSIVE cumsum
        # of the per-expert counts it receives. For per-expert padded layout we
        # therefore must pass the ALIGNED counts (not the raw ``num_recv``) —
        # otherwise consecutive experts overlap each other's padding rows and
        # the GEMM reads garbage. Mirrors framework
        # ``deepgemm_hybrid_executor.py::execute_contiguous`` which builds a
        # GPU tensor of aligned counts before calling ep_scatter.
        aligned_counts = torch.tensor(
            aligned_counts_list,
            dtype=torch.int32, pin_memory=True, device="cpu",
        ).to(device, non_blocking=True)

        # (2) Triton ep_scatter: per-expert padded layout in 1 kernel pair.
        # Output scale is column-major TMA-aligned int32 (matches DeepGEMM
        # contiguous expectation when scale_ue8m0=True) — see framework's
        # deepgemm_hybrid_executor.py:427-432 for the same allocation pattern.
        scatter_out = torch.empty(
            (all_tokens, D), dtype=torch.float8_e4m3fn, device=device
        )
        scatter_out_scale = torch.zeros(
            [ceil_div(D // FP8_BLOCK, 4), all_tokens],
            device=device, dtype=torch.int,
        ).transpose(0, 1)
        # m_indices is fully overwritten by ep_scatter's kernel_1 (one expert_id
        # per row across the aligned region). Padding rows therefore tag a real
        # expert and DeepGEMM does (wasted) compute against it; ep_gather only
        # fetches the valid rows tracked in ``output_index`` so the wasted
        # output is discarded. Matches framework pattern.
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
        # Defensive clamp against any -1 leakage (e.g. if num_local_experts is
        # later split for EP > 1); matches framework safety guard.
        m_indices.clamp_(min=0, max=E - 1)
        del a_fp8, a_scale

        # GEMM 1: gate+up
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
        )
        del scatter_out, scatter_out_scale

        # (3) Fused SiLU+clamp+mul + per-token-group FP8 quant + UE8M0 packed scale.
        # Router weight is NOT applied here — the ep_gather below folds it
        # into the topk-reduce.
        h_fp8, h_scale = silu_mul_fp8_quant_packed(
            gate_up,
            clamp_limit=cfg.swiglu_limit,
            group_size=FP8_BLOCK,
        )
        del gate_up

        # GEMM 2: down
        down_out = torch.empty(
            all_tokens, D, device=device, dtype=torch.bfloat16
        )
        m_grouped_fp8_fp4_gemm_nt_contiguous(
            (h_fp8, h_scale),
            (self._w2, self._s2),
            down_out,
            m_indices,
            recipe_a=(1, FP8_BLOCK),
            recipe_b=(1, FP4_BLOCK),
        )
        del h_fp8, h_scale

        # (4) Triton ep_gather: per output token accumulates topk source rows
        # × router weight in fp32 register, single BF16 store. No
        # [N, topk, D] fp32 intermediate (legacy materialised ~700 MB at
        # N=4k, topk=6, D=7168).
        gather_out = torch.empty((N, D), dtype=torch.bfloat16, device=device)
        ep_gather(down_out, adjusted_topk_ids, weights, output_index, gather_out)
        return gather_out.float()

    def _forward_sm120(self, x, weights, indices,
                       input_scale: Optional[torch.Tensor] = None) -> torch.Tensor:
        from flashinfer import block_scale_interleave, mxfp8_quantize
        from flashinfer.gemm import group_gemm_mxfp4_nt_groupwise
        cfg = self.cfg
        n, d = x.shape
        e, inter = cfg.n_routed_experts, cfg.moe_inter_dim
        device = x.device
        routed_ids = torch.where(weights != 0, indices, torch.full_like(indices, -1))
        adjusted_ids, counts = recompute_topk_ids_sum_expert_count(
            routed_ids, current_expert_start_id=0, num_local_experts=e)
        counts_list = counts.cpu().tolist()
        aligned_list = [align(int(count), 4) for count in counts_list]
        total_rows = sum(aligned_list)
        if total_rows == 0:
            return torch.zeros((n, d), dtype=torch.float32, device=device)
        aligned = torch.tensor(aligned_list, dtype=torch.int32,
                               pin_memory=True).to(device, non_blocking=True)
        indptr = torch.cat((torch.zeros(1, dtype=torch.int32, device=device),
                            aligned.cumsum(0).to(torch.int32)))
        expert_start = torch.empty_like(aligned)
        m_indices = torch.empty(align(total_rows, 128),
                                dtype=torch.int32, device=device)
        output_index = torch.empty_like(adjusted_ids)
        if input_scale is None:
            x_q, linear_scale = mxfp8_quantize(x.contiguous(),
                                               is_sf_swizzled_layout=False)
            linear_scale = linear_scale.reshape(n, d // FP4_BLOCK).view(torch.uint8)
        else:
            x_q = x
            linear_scale = input_scale.reshape(n, d // FP4_BLOCK).view(torch.uint8)
        routed_q = torch.empty(total_rows, d, dtype=x_q.dtype, device=device)
        routed_scale = torch.zeros(total_rows, d // FP4_BLOCK,
                                   dtype=torch.uint8, device=device)
        ep_scatter(x_q, linear_scale, adjusted_ids, aligned, expert_start,
                   routed_q, routed_scale, m_indices, output_index)
        expert_ids = torch.bucketize(torch.arange(total_rows, device=device),
                                     indptr[1:], right=True)
        group_ids = torch.arange(e + 1, dtype=torch.int32, device=device)
        sf_offsets = ((indptr + group_ids * 127) // 128) * 128
        scale_rows = torch.arange(total_rows, device=device) + \
            (sf_offsets[:-1] - indptr[:-1]).index_select(0, expert_ids)
        sf_rows = int(sf_offsets[-1].item())
        def pack_scale(linear: torch.Tensor) -> torch.Tensor:
            padded = torch.zeros(sf_rows, linear.size(1),
                                 dtype=torch.uint8, device=device)
            padded.index_copy_(0, scale_rows, linear)
            return block_scale_interleave(padded).reshape(sf_rows, linear.size(1))
        def gemm(inp_q, inp_scale, expert_weight, expert_scale):
            return group_gemm_mxfp4_nt_groupwise(inp_q, expert_weight,
                pack_scale(inp_scale), expert_scale, indptr,
                tile_n=128, out_dtype=torch.bfloat16)
        gate_up = gemm(routed_q, routed_scale,
                       self._w13.view(torch.uint8), self._s13_sm120)
        up, gate = gate_up[:, :inter], gate_up[:, inter:]
        hidden_q, hidden_scale_packed = silu_mul_fp8_quant_packed_from_parts(
            gate, up, clamp_limit=cfg.swiglu_limit, group_size=FP4_BLOCK)
        hidden_scale = hidden_scale_packed.contiguous().view(torch.uint8) \
            .reshape(total_rows, inter // FP4_BLOCK)
        down = gemm(hidden_q, hidden_scale,
                    self._w2.view(torch.uint8), self._s2_sm120)
        output = torch.empty((n, d), dtype=torch.float32, device=device)
        ep_gather(down, adjusted_ids, weights, output_index, output)
        return output
    def _get_sm120_fused_moe_workspace(
        self, device, max_tokens: int | None = None
    ) -> torch.Tensor:
        from flashinfer.fused_moe import cutlass_fused_moe_workspace_size
        from flashinfer.fused_moe.core import ActivationType
        cfg = self.cfg
        # The startup CUDA-graph warmup can run through the decode forward
        # before the model role is marked as DECODE.  In that phase
        # ``cfg.max_tokens_per_rank`` still contains the prefill chunk bound
        # (typically 16384), although the graph input has only one token per
        # request.  Size the FlashInfer workspace to the actual invocation;
        # there is deliberately no separate hard capacity guard here.  This
        # avoids reserving a multi-GiB prefill workspace during decode graph
        # initialization and lets fixed-capacity graph buffers be processed.
        workspace_tokens = max(
            int(cfg.max_tokens_per_rank) if max_tokens is None else int(max_tokens),
            1,
        )
        key = (device.index, workspace_tokens, cfg.dim, cfg.moe_inter_dim,
               cfg.n_routed_experts, cfg.n_activated_experts)
        workspace = _SM120_FUSED_MOE_WORKSPACES.get(key)
        if workspace is None:
            workspace_bytes = cutlass_fused_moe_workspace_size(
                workspace_tokens, cfg.dim, cfg.moe_inter_dim, cfg.n_routed_experts,
                cfg.n_activated_experts, x_dtype=torch.float8_e4m3fn,
                weight_dtype=torch.long, output_dtype=torch.bfloat16,
                activation_type=ActivationType.Swiglu, use_mxfp8_act_scaling=True,
                use_fused_finalize=False, device=device)
            workspace = torch.empty(workspace_bytes, dtype=torch.uint8, device=device)
            _SM120_FUSED_MOE_WORKSPACES[key] = workspace
        return workspace
    def _forward_capture_sm120(self, x, weights, indices) -> torch.Tensor:
        from flashinfer import mxfp8_quantize
        from flashinfer.fused_moe import cutlass_fused_moe
        from flashinfer.fused_moe.core import ActivationType
        cfg = self.cfg
        num_experts = cfg.n_routed_experts
        # The CUDA-graph wrapper may retain a fixed-capacity input buffer
        # (especially for speculative/DP graphs), so ``x.size(0)`` can be
        # larger than the model's per-rank scheduling hint.  The FlashInfer
        # workspace and tuning bound are therefore derived from this actual
        # invocation rather than rejecting the call against that hint.
        actual_tokens = max(int(x.size(0)), 1)
        fake_input_scale = torch.ones(num_experts, dtype=torch.float32, device=x.device)
        swiglu_limit = torch.full_like(fake_input_scale, cfg.swiglu_limit)
        output = torch.empty_like(x)
        kernel_input, input_sf = mxfp8_quantize(x.contiguous(), is_sf_swizzled_layout=True)
        cutlass_fused_moe(input=kernel_input,
            token_selected_experts=indices.to(torch.int32).contiguous(),
            token_final_scales=weights.float().contiguous(),
            fc1_expert_weights=self._w13.view(torch.uint8).view(torch.long),
            fc2_expert_weights=self._w2.view(torch.uint8).view(torch.long), output_dtype=torch.bfloat16,
            quant_scales=[self._s13_sm120.view(torch.int32), fake_input_scale,
                self._s2_sm120.view(torch.int32), fake_input_scale],
            input_sf=input_sf, swiglu_limit=swiglu_limit, output=output,
            use_mxfp8_act_scaling=True, use_fused_finalize=False, enable_pdl=False,
            workspace_buffer=self._get_sm120_fused_moe_workspace(
                x.device, actual_tokens
            ),
            tune_max_num_tokens=actual_tokens,
            activation_type=ActivationType.Swiglu)
        return output.float()
    def _forward_capture_topk(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        cfg = self.cfg
        N, D = x.shape
        inter = cfg.moe_inter_dim
        K = indices.size(1)
        device = x.device

        x_2d = x.reshape(N, D).contiguous()
        if x_2d.dtype != torch.bfloat16:
            x_2d = x_2d.to(torch.bfloat16)

        y = torch.empty((N, D), dtype=torch.float32, device=device)
        y.zero_()
        for n in range(N):
            x_n = x_2d[n : n + 1].contiguous()
            x_fp8, x_scale = sgl_per_token_group_quant_fp8(
                x_n,
                group_size=FP8_BLOCK,
                eps=1e-4,
                column_major_scales=True,
                scale_tma_aligned=True,
                scale_ue8m0=True,
            )
            for k in range(K):
                eid_t = indices[n, k : k + 1]
                router_w = weights[n, k : k + 1, None]

                w13 = torch.index_select(self._w13, 0, eid_t).squeeze(0)
                s13 = (
                    torch.index_select(self._s13_dense_t, 0, eid_t)
                    .squeeze(0)
                    .transpose(0, 1)
                )
                gate_up = torch.empty(
                    1, 2 * inter, device=device, dtype=torch.bfloat16
                )
                fp8_fp4_gemm_nt(
                    (x_fp8, x_scale),
                    (w13, s13),
                    gate_up,
                    recipe_a=(1, FP8_BLOCK),
                    recipe_b=(1, FP4_BLOCK),
                )

                h_fp8, h_scale = silu_mul_fp8_quant_packed(
                    gate_up,
                    clamp_limit=cfg.swiglu_limit,
                    group_size=FP8_BLOCK,
                )
                w2 = torch.index_select(self._w2, 0, eid_t).squeeze(0)
                s2 = (
                    torch.index_select(self._s2_dense_t, 0, eid_t)
                    .squeeze(0)
                    .transpose(0, 1)
                )
                down_out = torch.empty(1, D, device=device, dtype=torch.bfloat16)
                fp8_fp4_gemm_nt(
                    (h_fp8, h_scale),
                    (w2, s2),
                    down_out,
                    recipe_a=(1, FP8_BLOCK),
                    recipe_b=(1, FP4_BLOCK),
                )
                y[n : n + 1].add_(down_out.float() * router_w)

        return y
