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
from .._silu_mul_fp8_quant_triton import silu_mul_fp8_quant_packed
from ..warmup_sync import cuda_graph_warmup_forward_enabled
from ...quant_layouts import FP4_BLOCK, FP8_BLOCK, prepare_fp4_weight_scale_for_deepgemm


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
    if cap[0] == 12:
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
        is_sm120 = torch.cuda.get_device_capability(device)[0] == 12
        if is_sm120:
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

        if is_sm120:
            # CUTLASS SM120 block-scaled MMA consumes the native UE8M0 values
            # in its 128x4 interleaved layout.  Each expert boundary is already
            # 128-row aligned for V4, so a single flattened conversion is safe.
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
        if torch.cuda.get_device_capability(device)[0] == 12:
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

    def _forward_sm120(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
        input_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """FlashInfer contiguous groupwise MXFP8 x MXFP4 MoE path.

        FlashInfer requires every ``m_indptr`` entry to be 4-row aligned and
        every expert's activation-scale segment to be padded independently to
        128 rows before the 128x4 scale swizzle.  Keeping those two layouts
        separate is essential: concatenating scales and padding only once lets
        the kernel read the following expert's scale rows.
        """
        from flashinfer import block_scale_interleave, mxfp8_quantize
        from flashinfer.gemm import group_gemm_mxfp4_nt_groupwise

        cfg = self.cfg
        n, d = x.shape
        topk = indices.size(1)
        num_experts = cfg.n_routed_experts
        inter = cfg.moe_inter_dim
        device = x.device

        flat_experts = indices.reshape(-1)
        flat_weights = weights.reshape(-1)
        valid_positions = torch.nonzero(flat_weights != 0, as_tuple=False).flatten()
        if valid_positions.numel() == 0:
            return torch.zeros((n, d), dtype=torch.float32, device=device)
        valid_experts = flat_experts.index_select(0, valid_positions)
        valid_order = torch.argsort(valid_experts, stable=True)
        order = valid_positions.index_select(0, valid_order)
        sorted_experts = flat_experts.index_select(0, order)
        counts = torch.bincount(sorted_experts, minlength=num_experts)
        counts_all = counts.cpu().tolist()
        active_experts = [i for i, count in enumerate(counts_all) if count]
        counts_list = [int(counts_all[i]) for i in active_experts]
        padded_counts = [align(int(count), 4) for count in counts_list]
        total_rows = int(sum(padded_counts))

        if input_scale is not None:
            assert x.dtype == torch.float8_e4m3fn
            assert input_scale.dtype == torch.uint8
            input_scale = input_scale.reshape(n, d // FP4_BLOCK)

        token_ids = torch.div(order, topk, rounding_mode="floor")
        routed_x = torch.zeros(total_rows, d, dtype=x.dtype, device=device)
        routed_input_scale = (
            torch.zeros(
                total_rows,
                d // FP4_BLOCK,
                dtype=torch.uint8,
                device=device,
            )
            if input_scale is not None
            else None
        )
        valid_rows = torch.empty(order.numel(), dtype=torch.int64, device=device)
        src_offset = 0
        dst_offset = 0
        for count, padded_count in zip(counts_list, padded_counts):
            count = int(count)
            if count:
                dst = torch.arange(dst_offset, dst_offset + count, device=device)
                source_x = x.index_select(
                    0, token_ids[src_offset : src_offset + count]
                )
                if x.dtype == torch.float8_e4m3fn:
                    routed_x.view(torch.uint8).index_copy_(
                        0, dst, source_x.view(torch.uint8)
                    )
                else:
                    routed_x.index_copy_(0, dst, source_x)
                if routed_input_scale is not None:
                    routed_input_scale.index_copy_(
                        0,
                        dst,
                        input_scale.index_select(
                            0, token_ids[src_offset : src_offset + count]
                        ),
                    )
                valid_rows[src_offset : src_offset + count] = dst
            src_offset += count
            dst_offset += padded_count

        def interleave_groupwise_scale(
            inp_scale: torch.Tensor,
        ) -> list[torch.Tensor]:
            scale_cols = inp_scale.size(1)
            scale_chunks = []
            offset = 0
            for padded_count in padded_counts:
                if padded_count:
                    chunk = block_scale_interleave(
                        inp_scale[offset : offset + padded_count].contiguous()
                    ).reshape(-1, scale_cols)
                    scale_chunks.append(chunk)
                offset += padded_count
            return scale_chunks

        def quantize_groupwise(inp: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            inp_q, inp_scale = mxfp8_quantize(
                inp.contiguous(), is_sf_swizzled_layout=False
            )
            # FlashInfer MXFP8 is fixed at one UE8M0 scale per 32 values;
            # RTP's ``FP8_BLOCK`` is 128 and belongs to the DeepGEMM path.
            inp_scale = inp_scale.reshape(inp.size(0), inp.size(1) // FP4_BLOCK)
            return inp_q, interleave_groupwise_scale(inp_scale)

        def run_single_group_gemms(
            inp_q: torch.Tensor,
            scale_chunks: list[torch.Tensor],
            expert_weight: torch.Tensor,
            expert_scale: torch.Tensor,
        ) -> torch.Tensor:
            outputs = []
            offset = 0
            for group_id, padded_count in enumerate(padded_counts):
                group_indptr = torch.tensor(
                    [0, padded_count], dtype=torch.int32, device=device
                )
                outputs.append(
                    group_gemm_mxfp4_nt_groupwise(
                        inp_q[offset : offset + padded_count],
                        expert_weight[group_id : group_id + 1],
                        scale_chunks[group_id],
                        expert_scale[group_id : group_id + 1],
                        group_indptr,
                        tile_n=128,
                        out_dtype=torch.bfloat16,
                    )
                )
                offset += padded_count
            return torch.cat(outputs, dim=0)

        if routed_input_scale is None:
            routed_q, routed_scale = quantize_groupwise(routed_x)
        else:
            routed_q = routed_x
            routed_scale = interleave_groupwise_scale(routed_input_scale)
        active_experts_t = torch.tensor(
            active_experts, dtype=torch.int64, device=device
        )
        gate_up = run_single_group_gemms(
            routed_q,
            routed_scale,
            self._w13.view(torch.uint8).index_select(0, active_experts_t),
            self._s13_sm120.index_select(0, active_experts_t),
        )
        # SM120 weights are kept in FlashInfer fused-MoE order [up, gate] so
        # the graph path can reuse them without a second model-sized copy.
        up = gate_up[:, :inter].float()
        gate = gate_up[:, inter:].float()
        if cfg.swiglu_limit > 0:
            gate.clamp_(max=cfg.swiglu_limit)
            up.clamp_(min=-cfg.swiglu_limit, max=cfg.swiglu_limit)
        hidden = (torch.nn.functional.silu(gate) * up).to(torch.bfloat16)
        hidden_q, hidden_scale = quantize_groupwise(hidden)
        down = run_single_group_gemms(
            hidden_q,
            hidden_scale,
            self._w2.view(torch.uint8).index_select(0, active_experts_t),
            self._s2_sm120.index_select(0, active_experts_t),
        )

        sorted_out = down.index_select(0, valid_rows)
        # Accumulate routed rows directly into their token output.  Materialising
        # [N, topk, D] in fp32 costs 768 MiB for the 8k startup-warmup shape and
        # leaves no headroom on the 72 GiB SM120 cards.  Chunking also bounds the
        # temporary created by the BF16 -> FP32 conversion.
        output = torch.zeros(n, d, dtype=torch.float32, device=device)
        reduce_chunk_rows = 1024
        sorted_weights = flat_weights.index_select(0, order)
        for begin in range(0, sorted_out.size(0), reduce_chunk_rows):
            end = min(begin + reduce_chunk_rows, sorted_out.size(0))
            output.index_add_(
                0,
                token_ids[begin:end],
                sorted_out[begin:end].float()
                * sorted_weights[begin:end].float().unsqueeze(-1),
            )
        return output

    def forward_sm120_mxfp8(
        self,
        x: torch.Tensor,
        input_scale: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """Run SM120 grouped MoE from linear-layout MXFP8 activations."""
        if os.environ.get("DSV4_SM120_FUSED_MOE_PREFILL", "1") != "0":
            from flashinfer import block_scale_interleave
            from flashinfer.fused_moe import cutlass_fused_moe
            from flashinfer.fused_moe.core import ActivationType

            cfg = self.cfg
            n = x.size(0)
            fake_input_scale = torch.ones(
                cfg.n_routed_experts, dtype=torch.float32, device=x.device
            )
            swiglu_limit = torch.full_like(fake_input_scale, cfg.swiglu_limit)
            output = torch.empty(
                (n, cfg.dim), dtype=torch.bfloat16, device=x.device
            )
            input_sf = block_scale_interleave(
                input_scale.reshape(n, cfg.dim // FP4_BLOCK).contiguous()
            )
            cutlass_fused_moe(
                input=x.contiguous(),
                token_selected_experts=indices.to(torch.int32).contiguous(),
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
                workspace_buffer=self._get_sm120_fused_moe_workspace(
                    x.device, max_tokens=n
                ),
                tune_max_num_tokens=n,
                activation_type=ActivationType.Swiglu,
            )
            return output.float()
        return self._forward_sm120(x, weights, indices, input_scale=input_scale)

    def _get_sm120_fused_moe_workspace(
        self, device: torch.device, max_tokens: Optional[int] = None
    ) -> torch.Tensor:
        from flashinfer.fused_moe import cutlass_fused_moe_workspace_size
        from flashinfer.fused_moe.core import ActivationType

        cfg = self.cfg
        max_tokens = (
            min(max(int(cfg.max_tokens_per_rank), 1), 512)
            if max_tokens is None
            else max(int(max_tokens), 1)
        )
        key = (device.index, max_tokens, cfg.dim, cfg.moe_inter_dim,
               cfg.n_routed_experts, cfg.n_activated_experts)
        workspace = _SM120_FUSED_MOE_WORKSPACES.get(key)
        if workspace is None:
            workspace_bytes = cutlass_fused_moe_workspace_size(
                max_tokens, cfg.dim, cfg.moe_inter_dim,
                cfg.n_routed_experts, cfg.n_activated_experts,
                x_dtype=torch.float8_e4m3fn, weight_dtype=torch.long,
                output_dtype=torch.bfloat16,
                activation_type=ActivationType.Swiglu,
                use_mxfp8_act_scaling=True,
                use_fused_finalize=False, device=device,
            )
            workspace = torch.empty(
                workspace_bytes, dtype=torch.uint8, device=device
            )
            _SM120_FUSED_MOE_WORKSPACES[key] = workspace
        return workspace

    def _forward_capture_sm120(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """Graph-safe fixed-shape fused path, warmed before capture."""
        from flashinfer import mxfp8_quantize
        from flashinfer.fused_moe import cutlass_fused_moe
        from flashinfer.fused_moe.core import ActivationType

        cfg = self.cfg
        num_experts = cfg.n_routed_experts
        fake_input_scale = torch.ones(
            num_experts, dtype=torch.float32, device=x.device
        )
        swiglu_limit = torch.full_like(fake_input_scale, cfg.swiglu_limit)
        output = torch.empty_like(x)
        kernel_input, input_sf = mxfp8_quantize(
            x.contiguous(), is_sf_swizzled_layout=True
        )
        cutlass_fused_moe(
            input=kernel_input,
            token_selected_experts=indices.to(torch.int32).contiguous(),
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
            workspace_buffer=self._get_sm120_fused_moe_workspace(x.device),
            tune_max_num_tokens=min(max(int(cfg.max_tokens_per_rank), 1), 512),
            activation_type=ActivationType.Swiglu,
        )
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
