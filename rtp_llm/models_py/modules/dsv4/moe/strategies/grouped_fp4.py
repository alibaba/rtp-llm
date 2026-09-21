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
    m_grouped_fp8_fp4_gemm_nt_masked,
)
from rtp_llm.models_py.kernels.cuda.fp8_kernel import sgl_per_token_group_quant_fp8
from rtp_llm.models_py.modules.dsv4.const_cache import cached_zeroed

# DSV4_GROUPED_FP4_ASYNC=1: shape-driven grouped GEMM — no counts.cpu() /
# sf_offsets.item() per layer per forward (each was a pipeline drain).
# Default off = legacy host-counts path (byte-identical to pre-flag code).
_GROUPED_FP4_ASYNC = os.environ.get("DSV4_GROUPED_FP4_ASYNC", "0") == "1"
from rtp_llm.models_py.triton_kernels.common.activation import (
    create_packed_scale_tensor,
)
from rtp_llm.models_py.triton_kernels.moe.ep_kernels import (
    ep_gather,
    ep_scatter,
    ep_scatter_v2,
    recompute_topk_ids_sum_expert_count,
)
from rtp_llm.models_py.utils.math import align, ceil_div

from .base import MoeCfg, RoutedExpertsStrategy, register_strategy
from .._silu_mul_fp8_quant_triton import (
    silu_mul_fp8_quant_packed,
    silu_mul_fp8_quant_packed_from_parts,
    silu_mul_masked_fp8_quant_packed,
)
from ..warmup_sync import cuda_graph_warmup_forward_enabled
from ...quant_layouts import FP4_BLOCK, FP8_BLOCK, prepare_fp4_weight_scale_for_deepgemm


# ep_scatter requires m_indices.shape[0] % BLOCK_E == 0 (BLOCK_E=128); also
# DeepGEMM contiguous requires per-expert M to be a multiple of the kernel's
# alignment (128 on SM100). We use the same constant.
_GROUPED_ALIGNMENT = 128
# DSV4_MOE_FP4_BACKEND=deepgemm (Sep 3, DSV4_DEEPGEMM_SWAP_RECONCILIATION_
# 20260903.md §4): SM120 expert compute on DeepGEMM's grouped mixed
# FP8×FP4 contiguous kernel instead of flashinfer group_gemm_mxfp4.
# Q5b measured the kernel at −12.3%/−18% vs flashinfer (under the 25% gate)
# — shipped per explicit user "worth a try" decision. Weight buffers stay
# in this class's native [E, N, K/2] per-32 UE8M0 packing (the layout
# DeepGEMM's SM120 kernel reads); only the scale tensors switch format.
_DG_BACKEND = os.environ.get("DSV4_MOE_FP4_BACKEND", "flashinfer").strip().lower()
_DG_BE_CT = [0]  # deepgemm-backend engagement proof (DSV4_DIAG, first 3 fires)
# DGV native wire (Sep 3, results_20260903_dg1/VERDICT.md CORRECTION): when the
# dispatch already quantized per-128 ue8m0 (deepep.py _DG_NATIVE_WIRE), the
# received x IS the per-128 fp8 activation and input_scale IS the plain
# per-token [N, D/128] ue8m0 bytes — skip the dequant+requant and rebuild
# DeepGEMM's TMA-aligned packed scale from the plain bytes (bench/q5e proved
# the transform bit-exact). Both modules read the same env → consistent gate.
_DG_NATIVE_WIRE = (
    int(os.environ.get("DSV4_DG_NATIVE_WIRE", "0")) and _DG_BACKEND == "deepgemm"
)


def _dg_repack_tma_scale(plain, n_rows):
    """Plain per-token [n_rows, k_groups] uint8 ue8m0 scales -> DeepGEMM's
    TMA-aligned column-major packed-int32 scale view
    [n_rows, ceil_align(k_groups,4)//4] — the exact tensor
    sgl_per_token_group_quant_fp8(scale_tma_aligned=True, scale_ue8m0=True)
    produces (4 ue8m0 bytes per int32, little-endian; token dim padded to
    ceil_align(n_rows,4)). bench/q5e: reconstruct(reference) == sgl BIT-EXACT."""
    k_groups = plain.shape[1]
    aligned_k = (k_groups + 3) // 4 * 4
    aligned_mn = (n_rows + 3) // 4 * 4
    ncol = aligned_k // 4
    pad = torch.zeros(n_rows, aligned_k, dtype=torch.uint8, device=plain.device)
    pad[:, :k_groups] = plain
    p = pad.view(n_rows, ncol, 4).to(torch.int64)
    packed = (p[..., 0] | (p[..., 1] << 8) | (p[..., 2] << 16)
              | (p[..., 3] << 24)).to(torch.int32)
    base = torch.zeros((ncol, aligned_mn), dtype=torch.int32, device=plain.device)
    base[:, :n_rows] = packed.t()
    return base.transpose(-1, -2)[:n_rows, :]


def _sm120_fused_moe_capacity(
    max_tokens_per_rank: int, input_rows: int, capacity_tokens: int = 0
) -> int:
    """Historical cutlass workspace capacity (kept for the CPU regression test).

    The SM120 capture path no longer sizes a flashinfer workspace — DeepGEMM
    masked grouped GEMM is shape-driven. This helper still encodes the
    post-gather tile contract the old ``cutlass_fused_moe`` path needed.
    """
    budget = min(max(int(max_tokens_per_rank), 1), 512)
    return max(int(input_rows or 0), int(capacity_tokens or 0), budget)


def _has_fp8_fp4_grouped_kernel() -> bool:
    """True iff the grouped FP4 routed-expert path should be used.

    Requires deep_gemm ≥ 2.4 (ships ``m_grouped_fp8_fp4_gemm_nt_contiguous``)
    and an SM100 device. SM120 also accepts the masked grouped kernel used
    by decode capture, or flashinfer as a last-resort probe.

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
            import deep_gemm
            if hasattr(deep_gemm, "m_grouped_fp8_fp4_gemm_nt_masked") or \
                    hasattr(deep_gemm, "m_grouped_fp8_fp4_gemm_nt_contiguous"):
                return True
        except Exception:
            pass
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
    # Grow-only workspaces shared across layers. Old entries stay so
    # captured graphs keep valid pointers.
    _sm120_masked_ws_cache: list = []

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
        # no per-expert iteration). Gate/up row order is gate-first:
        # DeepGEMM + silu_mul_fp8_quant_packed / masked silu all contract
        # "Gate in [:inter], up in [inter:]" (w1 then w3). The old SM120
        # flashinfer/cutlass path packed up-first; that path is gone.
        is_sm120 = torch.cuda.get_device_capability(device)[0] == 12
        self._w13[:, :inter].copy_(stacked_w1_w)
        s13_raw[:, :inter].copy_(stacked_w1_s)
        self._w13[:, inter:].copy_(stacked_w3_w)
        s13_raw[:, inter:].copy_(stacked_w3_s)
        self._w2.copy_(stacked_w2_w)
        s2_raw.copy_(stacked_w2_s)
        del stacked_w1_w, stacked_w1_s, stacked_w2_w, stacked_w2_s
        del stacked_w3_w, stacked_w3_s
        if is_sm120:
            # SM120 decode capture uses DeepGEMM masked grouped GEMM, and
            # eager SM120 uses the contiguous DeepGEMM body. Both need
            # DeepGEMM-format scales (never both formats — HBM law).
            self._s13 = prepare_fp4_weight_scale_for_deepgemm(
                s13_raw, 2 * inter, D, E
            )
            self._s2 = prepare_fp4_weight_scale_for_deepgemm(s2_raw, D, inter, E)
            self._s13_sm120 = self._s2_sm120 = None
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

    def _ensure_sm120_masked_workspace(
        self, n: int, topk: int, e: int, d: int, inter: int,
        alignment: int, device: torch.device,
    ) -> dict:
        """One grow-only masked workspace per device/shape, shared across layers.

        Capture walks 128→8 so a per-(n, alignment) cache would keep ~2 GiB.
        Larger buffers are valid for smaller n: scatter/GEMM use the stored
        alignment, token-indexed outputs are sliced to ``n``.
        """
        key = (topk, e, d, inter, device.index)
        cache = GroupedFP4Strategy._sm120_masked_ws_cache
        n_cap, a_cap = n, alignment
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
            # Zero once so unused TMA scale slots stay valid UE8M0 powers of 2.
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
            "down_out": torch.empty(
                (e, a_cap, d), dtype=torch.bfloat16, device=device
            ),
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
        """SM120 decode/capture MoE: DeepGEMM masked grouped FP8×FP4 GEMM.

        Replaces flashinfer ``cutlass_fused_moe``. Layout matches the bench
        ``DeepGemmFp4Fp8Experts._apply_masked``: cached ``[E, M_pad, K]``
        buffers, ``ep_scatter_v2``, masked GEMM, fused SiLU+clamp+quant,
        ``ep_gather`` (fp32 acc, store uses ``out`` dtype — bf16 on the
        decode A2A wire).
        """
        cfg = self.cfg
        n, d = x.shape
        e, inter = cfg.n_routed_experts, cfg.moe_inter_dim
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
                "SM120 DeepGEMM masked MoE requires DeepGEMM-format weight "
                "scales; setup_weights did not build _s13/_s2"
            )
        if input_scale is None:
            a_fp8, a_scale = sgl_per_token_group_quant_fp8(
                x if x.is_contiguous() else x.contiguous(),
                group_size=FP8_BLOCK,
                eps=1e-4,
                column_major_scales=True,
                scale_tma_aligned=True,
                scale_ue8m0=True,
            )
        else:
            a_fp8, a_scale = x, input_scale
        alignment = align(n_int, _GROUPED_ALIGNMENT)
        ws = self._ensure_sm120_masked_workspace(
            n_int, int(indices.size(1)), e, int(d), int(inter), alignment, device
        )
        alignment = int(ws["alignment"])
        expected_m = min(alignment, ceil_div(n_int * int(indices.size(1)), e))
        adjusted_ids, masked_m = recompute_topk_ids_sum_expert_count(
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

    def _forward_sm120_deepgemm(self, x, weights, indices,
                               input_scale: Optional[torch.Tensor] = None) -> torch.Tensor:
        """DSV4_MOE_FP4_BACKEND=deepgemm: SM120 expert compute on DeepGEMM's
        grouped mixed FP8×FP4 contiguous kernel.

        Body transplanted from this class's SM100 DeepGEMM path (proven in
        the engine) with two SM120/EP adaptations: (1) the C2/A2 wire already
        delivers FP8 + per-32 UE8M0 activations, so dequant once to BF16 and
        requant at DeepGEMM's per-128 UE8M0 recipe (the double-quant cost is
        the honest price of keeping the wire format unchanged); (2) routed_ids
        mask (weights==0 → −1) matching ``_forward_sm120`` semantics — the
        padding rows are discarded by ``ep_gather``. The SITU activation
        (``clamp_limit=cfg.swiglu_limit``) is preserved verbatim.
        """
        cfg = self.cfg
        N, D = x.shape
        E, inter = cfg.n_routed_experts, cfg.moe_inter_dim
        device = x.device
        routed_ids = torch.where(
            weights != 0, indices, torch.full_like(indices, -1)
        )
        if _DG_NATIVE_WIRE:
            # DGV native wire: x IS the per-128 fp8 activation and input_scale
            # IS the plain per-token [N, D/128] ue8m0 bytes from the dispatch —
            # rebuild DeepGEMM's TMA-aligned packed scale (bench/q5e: bit-exact)
            # and use x directly. NO dequant, NO requant — that double-quant was
            # the ~+115 ms which swamped the −28 ms GEMM1 win in the dg1 A/B.
            a_fp8 = x
            a_scale = _dg_repack_tma_scale(
                input_scale.reshape(N, D // FP8_BLOCK), N)
        else:
            if input_scale is not None:
                s = (
                    input_scale.reshape(N, D // FP4_BLOCK)
                    .view(torch.float8_e8m0fnu)
                    .to(torch.bfloat16)
                )
                x = (
                    x.to(torch.bfloat16).view(N, D // FP4_BLOCK, FP4_BLOCK)
                    * s.unsqueeze(2)
                ).view(N, D).contiguous()
            a_fp8, a_scale = sgl_per_token_group_quant_fp8(
                x.contiguous(),
                group_size=FP8_BLOCK,
                eps=1e-4,
                column_major_scales=True,
                scale_tma_aligned=True,
                scale_ue8m0=True,
            )
        if os.environ.get("DSV4_DIAG") and _DG_BE_CT[0] < 3:
            _DG_BE_CT[0] += 1
            import sys
            print(
                "[DG-BE] rank=%d N=%d E=%d backend=deepgemm native_wire=%d" % (
                    torch.distributed.get_rank()
                    if torch.distributed.is_initialized() else -1,
                    N, E, int(_DG_NATIVE_WIRE)),
                file=sys.stderr,
                flush=True,
            )
        adjusted_topk_ids, num_recv = recompute_topk_ids_sum_expert_count(
            routed_ids,
            current_expert_start_id=0,
            num_local_experts=E,
        )
        num_recv_cpu = num_recv.cpu().tolist()
        aligned_counts_list = [align(c, _GROUPED_ALIGNMENT) for c in num_recv_cpu]
        all_tokens = sum(aligned_counts_list)
        if all_tokens == 0:
            return torch.zeros((N, D), dtype=torch.float32, device=device)
        aligned_counts = torch.tensor(
            aligned_counts_list,
            dtype=torch.int32, pin_memory=True, device="cpu",
        ).to(device, non_blocking=True)
        scatter_out = torch.empty(
            (all_tokens, D), dtype=torch.float8_e4m3fn, device=device
        )
        scatter_out_scale = torch.zeros(
            [ceil_div(D // FP8_BLOCK, 4), all_tokens],
            device=device, dtype=torch.int,
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
        # Defensive clamp against any -1 leakage (padding rows must tag a
        # real expert for the grouped GEMM; ep_gather discards them).
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
        )
        del scatter_out, scatter_out_scale

        h_fp8, h_scale = silu_mul_fp8_quant_packed(
            gate_up,
            clamp_limit=cfg.swiglu_limit,
            group_size=FP8_BLOCK,
        )
        del gate_up

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

        gather_out = torch.empty((N, D), dtype=torch.bfloat16, device=device)
        ep_gather(down_out, adjusted_topk_ids, weights, output_index, gather_out)
        return gather_out.float()

    def _forward_sm120(self, x, weights, indices,
                       input_scale: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self._s13 is not None:
            # Decode-sized eager calls take the same masked path as capture
            # (no host count sync). Prefill / FP8-wire calls keep contiguous.
            if input_scale is None:
                try:
                    n_int = int(x.size(0))
                except Exception:
                    n_int = -1
                if 0 <= n_int <= 512:
                    return self._forward_sm120_deepgemm_masked(
                        x, weights, indices, input_scale=input_scale
                    )
            return self._forward_sm120_deepgemm(x, weights, indices, input_scale)
        from flashinfer import block_scale_interleave, mxfp8_quantize
        from flashinfer.gemm import group_gemm_mxfp4_nt_groupwise
        cfg = self.cfg
        n, d = x.shape
        e, inter = cfg.n_routed_experts, cfg.moe_inter_dim
        device = x.device
        routed_ids = torch.where(weights != 0, indices, torch.full_like(indices, -1))
        adjusted_ids, counts = recompute_topk_ids_sum_expert_count(
            routed_ids, current_expert_start_id=0, num_local_experts=e)
        topk = indices.shape[-1]
        if _GROUPED_FP4_ASYNC:
            # Shape-driven sizing, zero D2H syncs. sum(counts) == n*topk always
            # and 4-alignment adds <= 3 rows/expert, so `rows` overshoots the
            # true total by <= ~0.6% — extra GEMM rows are stale-expert garbage
            # discarded by ep_gather. The flashinfer SM120 group GEMM derives
            # per-group work from m_indptr on device (its exact-shape assert is
            # disabled "in consideration of performance").
            rows = n * topk + 4 * e
            aligned = (counts + 3) & ~3
            indptr = torch.zeros(e + 1, dtype=torch.int32, device=device)
            torch.cumsum(aligned, 0, dtype=torch.int32, out=indptr[1:])
            expert_start = torch.empty_like(aligned)
            # m_indices MUST be initialized (0 = a valid expert id): garbage
            # values would index out-of-range expert weights.
            m_indices = cached_zeroed((align(rows, 128),), dtype=torch.int32,
                                      device=device)
            # Host-side upper bound of sf_offsets[-1]: each expert's scale
            # rows round up by <= 127; block_scale_interleave additionally
            # pads rows to a 128-multiple, so keep sf_rows 128-aligned too.
            sf_rows = ((rows + 127 * e + 128 + 127) // 128) * 128
        else:
            counts_list = counts.cpu().tolist()
            aligned_list = [align(int(count), 4) for count in counts_list]
            rows = sum(aligned_list)
            if rows == 0:
                return torch.zeros((n, d), dtype=torch.float32, device=device)
            aligned = torch.tensor(aligned_list, dtype=torch.int32,
                                   pin_memory=True).to(device, non_blocking=True)
            # P1a tranche 2: was torch.cat((zeros(1), cumsum)) — 3 ops + 2 allocs
            # per layer per forward; now one small memset + one cumsum.
            indptr = torch.zeros(e + 1, dtype=torch.int32, device=device)
            torch.cumsum(aligned, 0, dtype=torch.int32, out=indptr[1:])
            expert_start = torch.empty_like(aligned)
            m_indices = torch.empty(align(rows, 128),
                                    dtype=torch.int32, device=device)
        output_index = torch.empty_like(adjusted_ids)
        if input_scale is None:
            x_q, linear_scale = mxfp8_quantize(x.contiguous(),
                                               is_sf_swizzled_layout=False)
            linear_scale = linear_scale.reshape(n, d // FP4_BLOCK).view(torch.uint8)
        else:
            x_q = x
            linear_scale = input_scale.reshape(n, d // FP4_BLOCK).view(torch.uint8)
        routed_q = torch.empty(rows, d, dtype=x_q.dtype, device=device)
        routed_scale = torch.zeros(rows, d // FP4_BLOCK,
                                   dtype=torch.uint8, device=device)
        ep_scatter(x_q, linear_scale, adjusted_ids, aligned, expert_start,
                   routed_q, routed_scale, m_indices, output_index)
        expert_ids = torch.bucketize(torch.arange(rows, device=device),
                                     indptr[1:], right=True).clamp_max_(e - 1)
        group_ids = torch.arange(e + 1, dtype=torch.int32, device=device)
        sf_offsets = ((indptr + group_ids * 127) // 128) * 128
        scale_rows = torch.arange(rows, device=device) + \
            (sf_offsets[:-1] - indptr[:-1]).index_select(0, expert_ids)
        if not _GROUPED_FP4_ASYNC:
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
            .reshape(rows, inter // FP4_BLOCK)
        down = gemm(hidden_q, hidden_scale,
                    self._w2.view(torch.uint8), self._s2_sm120)
        output = torch.empty((n, d), dtype=torch.float32, device=device)
        ep_gather(down, adjusted_ids, weights, output_index, output)
        return output
    def _forward_capture_sm120(
        self,
        x,
        weights,
        indices,
        capacity_tokens: int = 0,
        input_scale: Optional[torch.Tensor] = None,
        expert_start_id: int = 0,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Decode / CUDA-graph path: DeepGEMM masked grouped GEMM. Static
        # [E, align(n, 128), K] layout, no host count sync. capacity_tokens is
        # kept for the fixed-EP caller ABI; masked sizing is shape-driven.
        del capacity_tokens
        return self._forward_sm120_deepgemm_masked(
            x,
            weights,
            indices,
            input_scale=input_scale,
            expert_start_id=expert_start_id,
            out=out,
        )
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
