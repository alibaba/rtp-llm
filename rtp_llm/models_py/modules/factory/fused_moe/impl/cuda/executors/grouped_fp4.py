"""Single-rank DeepGEMM FP8-activation x FP4-weight MoE executor.

It routes tokens by expert, runs grouped
``m_grouped_fp8_fp4_gemm_nt_contiguous`` for the
gate/up projection, fuse SiLU+mul+FP8 requant, run grouped down projection,
then gather/reduce by router weight.

Single-card + DeepGEMM ≥ 2.4 + SM100 selects this path by default. Pass
``moe_strategy=local_loop`` to opt out.

Forward is the 4-opt prefill path:
  (1) quant input ONCE pre-permute (vs. ×topk on padded buffer)
  (2) Triton ep_scatter (vs. argsort + bincount + cumsum + index_copy chain)
  (3) Fused silu+mul+fp8 quant kernel (vs. clamp + silu + mul + cast + quant)
  (4) Triton ep_gather with router-weight reduce (vs. index_select +
      fp32 [N,topk,D] materialize + sum)
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import (
    fp8_fp4_gemm_nt,
    m_grouped_fp8_fp4_gemm_nt_contiguous,
)
from rtp_llm.models_py.kernels.cuda.fp8_kernel import sgl_per_token_group_quant_fp8
from rtp_llm.models_py.kernels.cuda.quant_layouts import (
    FP4_BLOCK,
    FP8_BLOCK,
    prepare_fp4_weight_scale_for_deepgemm,
)
from rtp_llm.models_py.triton_kernels.moe.ep_kernels import (
    ep_gather,
    ep_scatter,
    recompute_topk_ids_sum_expert_count,
)
from rtp_llm.models_py.triton_kernels.moe.silu_mul_fp8_quant import (
    silu_mul_fp8_quant_packed,
)
from rtp_llm.models_py.utils.math import align, ceil_div

from .fp8_fp4_base import Fp8Fp4ExecutorBase, normalize_moe_w13_gate_up

# ep_scatter requires m_indices.shape[0] % BLOCK_E == 0 (BLOCK_E=128); also
# DeepGEMM contiguous requires per-expert M to be a multiple of the kernel's
# alignment (128 on SM100). We use the same constant.
_GROUPED_ALIGNMENT = 128

# DeepGEMM's fp8_fp4 kernels require the (packed) inner K dim to be a
# multiple of 128 bytes, i.e. K % 256 == 0 in fp elements. Checkpoints whose
# hidden/intermediate sizes miss that (gpt-oss: 2880) are zero-padded up to
# the next multiple — weight scales carry zero nibbles in the pad region, so
# padded columns contribute exactly zero.
_K_ALIGNMENT = 256


def _ceil_to(x: int, mult: int) -> int:
    return ((x + mult - 1) // mult) * mult


def _pad_k_bytes(packed_w: torch.Tensor, scales_raw: torch.Tensor, target_k: int):
    """Zero-pad the packed FP4 K dim (and matching scale groups) of
    ``[E, N, K/2]`` int8 + ``[E, N, K/32]`` UE8M0 up to ``target_k``.

    Padded weight nibbles are zero (magnitude 0 -> exactly-zero
    contribution); their scale bytes are filled with exponent 127 (scale
    1.0) — a zero scale byte (2^-127) makes deep_gemm's
    ``transform_sf_into_required_layout`` kernel fault (CUDA error 719),
    while any valid scale multiplies the zero nibble into zero.
    """
    k = packed_w.shape[-1] * 2
    if k == target_k:
        return packed_w, scales_raw
    w_pad = torch.zeros(
        (*packed_w.shape[:-1], target_k // 2),
        dtype=packed_w.dtype,
        device=packed_w.device,
    )
    w_pad[..., : packed_w.shape[-1]] = packed_w
    s_view = scales_raw.view(torch.uint8)
    s_pad = torch.full(
        (*s_view.shape[:-1], target_k // 32),
        127,
        dtype=torch.uint8,
        device=s_view.device,
    )
    s_pad[..., : s_view.shape[-1]] = s_view
    return w_pad, s_pad.view(scales_raw.dtype)


def _has_fp8_fp4_grouped_kernel() -> bool:
    """True iff the grouped FP4 routed-expert path should be used.

    Requires deep_gemm ≥ 2.4 (ships ``m_grouped_fp8_fp4_gemm_nt_contiguous``)
    and an SM100 device.

    Selection is based only on runtime support. An explicit public strategy
    argument controls whether this implementation is selected.
    """
    try:
        import deep_gemm
    except Exception:
        return False
    if not hasattr(deep_gemm, "m_grouped_fp8_fp4_gemm_nt_contiguous"):
        return False
    if not hasattr(deep_gemm, "get_mk_alignment_for_contiguous_layout"):
        return False
    if not torch.cuda.is_available():
        return False
    cap = torch.cuda.get_device_capability()
    return cap[0] == 10


class GroupedFp4Executor(Fp8Fp4ExecutorBase):

    @classmethod
    def check_conditions(cls, checker: Any, config) -> None:
        super().check_conditions(checker, config)
        checker.check(config.ep_size == 1)
        # K dims only need 32-divisibility; the _K_ALIGNMENT gap is closed
        # by zero-padding (gpt-oss hidden/inter 2880 -> 3072).
        checker.check(config.dim % FP4_BLOCK == 0)
        checker.check(config.moe_inter_dim % FP4_BLOCK == 0)
        checker.check(_has_fp8_fp4_grouped_kernel())

    def setup_weights(self, layer_weights: Dict) -> None:
        """Stack EP-sliced routed-expert tensors into ``[E, ...]`` int8 +
        UE8M0 SF buffers in the layout DeepGEMM's contiguous kernel reads.

        Pops canonical ``W.moe_w{1,2}`` and ``W.moe_s{1,2}`` keys from
        ``layer_weights`` (each shaped ``[E_local, ...]``).

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
        # DeepGEMM fp8_fp4 K alignment (see _K_ALIGNMENT): pad zero columns
        # when the checkpoint dims miss it (no-op for aligned models like
        # DeepSeek V4). Activation quant groups follow what divides the
        # ORIGINAL dims: 128 when possible (legacy path), else 32 (MXFP8).
        self._k1 = _ceil_to(D, _K_ALIGNMENT)
        self._k2 = _ceil_to(inter, _K_ALIGNMENT)
        self._act_group = FP8_BLOCK if self._k1 % FP8_BLOCK == 0 else FP4_BLOCK
        self._h_group = FP8_BLOCK if inter % FP8_BLOCK == 0 else FP4_BLOCK
        self._w13 = layer_weights.pop(W.moe_w1)
        s13_raw = layer_weights.pop(W.moe_s1)
        # gpt-oss-style checkpoints carry raw U8 scale bytes (declared uint8
        # so the loader never numerically casts them); DeepSeek V4 ships
        # float8_e8m0fnu directly. Bit-view the former so both feed the same
        # UE8M0 consumers below.
        if s13_raw.dtype == torch.uint8:
            s13_raw = s13_raw.view(torch.float8_e8m0fnu)
        self._w13, s13_raw = normalize_moe_w13_gate_up(
            self._w13,
            s13_raw,
            inter,
            cfg.moe_w1_layout,
        )
        self._w2 = layer_weights.pop(W.moe_w2)
        s2_raw = layer_weights.pop(W.moe_s2)
        if s2_raw.dtype == torch.uint8:
            s2_raw = s2_raw.view(torch.float8_e8m0fnu)
        self._w13, s13_raw = _pad_k_bytes(self._w13, s13_raw, self._k1)
        self._w2, s2_raw = _pad_k_bytes(self._w2, s2_raw, self._k2)
        # Optional per-expert biases (gpt-oss): gate_up bias is added after
        # GEMM1 (pre-clamp/activation), down bias after GEMM2 (pre-combine).
        # Bring-up applies them as gathered elementwise adds; folding into the
        # activation/gather kernels is a later fusion step.
        self._b13 = layer_weights.pop(W.moe_b1, None)
        self._b2 = layer_weights.pop(W.moe_b2, None)
        if self._b13 is not None and self._b13.dtype != torch.bfloat16:
            self._b13 = self._b13.to(torch.bfloat16)
        if self._b2 is not None and self._b2.dtype != torch.bfloat16:
            self._b2 = self._b2.to(torch.bfloat16)

        self._s13 = prepare_fp4_weight_scale_for_deepgemm(
            s13_raw, 2 * inter, self._k1, E
        )
        self._s2 = prepare_fp4_weight_scale_for_deepgemm(s2_raw, D, self._k2, E)
        s13_dense = prepare_fp4_weight_scale_for_deepgemm(
            s13_raw.reshape(E * 2 * inter, self._k1 // FP4_BLOCK),
            E * 2 * inter,
            self._k1,
        )
        self._s13_dense_t = s13_dense.as_strided(
            (E, s13_dense.size(1), 2 * inter),
            (2 * inter, E * 2 * inter, 1),
        )
        s2_dense = prepare_fp4_weight_scale_for_deepgemm(
            s2_raw.reshape(E * D, self._k2 // FP4_BLOCK),
            E * D,
            self._k2,
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

    def dense_gemm_warmup_weights(self):
        """Expose the two dense fallback GEMMs to model-level JIT warmup."""
        return (
            ("grouped_w13", self._w13, self._s13_dense_t),
            ("grouped_w2", self._w2, self._s2_dense_t),
        )

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
        if torch.cuda.is_current_stream_capturing():
            return self._forward_capture_topk(x, weights, indices)

        # (1) Quant input ONCE — column-major TMA-aligned UE8M0 packed scale,
        # shape compatible with both ep_scatter input and DeepGEMM contiguous.
        # Zero-pad K to the DeepGEMM fp8_fp4 alignment first when needed
        # (padded columns quantize to 0 and contribute nothing).
        x_quant = x.contiguous()
        if self._k1 != D:
            x_quant = torch.nn.functional.pad(x_quant, (0, self._k1 - D))
        a_fp8, a_scale = sgl_per_token_group_quant_fp8(
            x_quant,
            group_size=self._act_group,
            eps=1e-4,
            column_major_scales=True,
            scale_tma_aligned=True,
            scale_ue8m0=True,
        )
        del x_quant

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
            dtype=torch.int32,
            pin_memory=True,
            device="cpu",
        ).to(device, non_blocking=True)

        # (2) Triton ep_scatter: per-expert padded layout in 1 kernel pair.
        # Output scale is column-major TMA-aligned int32 (matches DeepGEMM
        # contiguous expectation when scale_ue8m0=True) — see framework's
        # deepgemm_hybrid_executor.py:427-432 for the same allocation pattern.
        # Both buffers are zero/127-INITIALIZED: the kernel writes only the
        # valid rows, and the padding rows keep whatever the allocator
        # returned. torch.empty padding has produced NaN activation bytes
        # and 0x00 scale bytes (UE8M0 exponent 0 = 2^-127) in-engine, which
        # poison the grouped GEMM's valid-row outputs (observed rel 0.72 on
        # valid rows); finite-zero padding with scale exponent 127 keeps the
        # wasted padding compute benign.
        scatter_out = torch.zeros(
            (all_tokens, self._k1), dtype=torch.float8_e4m3fn, device=device
        )
        scatter_out_scale = torch.full(
            [ceil_div(self._k1 // self._act_group, 4), ((all_tokens + 3) // 4) * 4],
            0x7F7F7F7F,
            device=device,
            dtype=torch.int,
        ).transpose(0, 1)[:all_tokens, :]
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
            recipe_a=(1, self._act_group),
            recipe_b=(1, FP4_BLOCK),
        )
        del scatter_out, scatter_out_scale

        if self._b13 is not None:
            gate_up += self._b13[m_indices]

        # (3) Fused gated activation (SiLU*up by default; gpt-oss quick-gelu
        # variant via alpha/up_shift) + clamp + per-token-group FP8 quant +
        # UE8M0 packed scale. Router weight is NOT applied here — the
        # ep_gather below folds it into the topk-reduce.
        h_fp8, h_scale = silu_mul_fp8_quant_packed(
            gate_up,
            clamp_limit=cfg.swiglu_limit,
            group_size=self._h_group,
            act_alpha=cfg.swiglu_alpha,
            up_shift=cfg.swiglu_up_shift,
        )
        del gate_up

        if self._k2 != inter:
            h_pad = torch.zeros(
                (h_fp8.shape[0], self._k2),
                dtype=h_fp8.dtype,
                device=device,
            )
            h_pad[:, :inter].copy_(h_fp8)
            h_fp8 = h_pad
            # Rebuild the scale in the SAME column-major TMA-aligned layout
            # silu_mul emits (the GEMM requires sf.stride(-2) == 1): pack
            # rows first, then transpose. Pad scale bytes carry exponent
            # 127 (see _pad_k_bytes: zero scale bytes fault deep_gemm).
            m_rows = h_scale.shape[0]
            packs_new = ceil_div(self._k2 // self._h_group, 4)
            aligned_m = ((m_rows + 3) // 4) * 4
            s_pad = torch.full(
                (packs_new, aligned_m),
                0x7F7F7F7F,
                dtype=h_scale.dtype,
                device=device,
            ).transpose(0, 1)[:m_rows, :]
            s_pad[:, : h_scale.shape[1]].copy_(h_scale)
            h_scale = s_pad

        # GEMM 2: down
        down_out = torch.empty(all_tokens, D, device=device, dtype=torch.bfloat16)
        m_grouped_fp8_fp4_gemm_nt_contiguous(
            (h_fp8, h_scale),
            (self._w2, self._s2),
            down_out,
            m_indices,
            recipe_a=(1, self._h_group),
            recipe_b=(1, FP4_BLOCK),
        )
        del h_fp8, h_scale

        if self._b2 is not None:
            down_out += self._b2[m_indices]

        # (4) Triton ep_gather: per output token accumulates topk source rows
        # × router weight in fp32 register, single BF16 store. No
        # [N, topk, D] fp32 intermediate (legacy materialised ~700 MB at
        # N=4k, topk=6, D=7168).
        gather_out = torch.empty((N, D), dtype=torch.bfloat16, device=device)
        ep_gather(down_out, adjusted_topk_ids, weights, output_index, gather_out)
        return gather_out.float()

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
            if self._k1 != D:
                x_n = torch.nn.functional.pad(x_n, (0, self._k1 - D))
            x_fp8, x_scale = sgl_per_token_group_quant_fp8(
                x_n,
                group_size=self._act_group,
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
                gate_up = torch.empty(1, 2 * inter, device=device, dtype=torch.bfloat16)
                fp8_fp4_gemm_nt(
                    (x_fp8, x_scale),
                    (w13, s13),
                    gate_up,
                    recipe_a=(1, self._act_group),
                    recipe_b=(1, FP4_BLOCK),
                )
                if self._b13 is not None:
                    gate_up += self._b13[eid_t]

                h_fp8, h_scale = silu_mul_fp8_quant_packed(
                    gate_up,
                    clamp_limit=cfg.swiglu_limit,
                    group_size=self._h_group,
                    act_alpha=cfg.swiglu_alpha,
                    up_shift=cfg.swiglu_up_shift,
                )
                if self._k2 != inter:
                    h_pad = torch.zeros(
                        (1, self._k2), dtype=h_fp8.dtype, device=device
                    )
                    h_pad[:, :inter].copy_(h_fp8)
                    h_fp8 = h_pad
                    # column-major TMA-aligned layout (see the forward pad):
                    # [packs, aligned_m] transposed, pad bytes = exponent 127
                    packs_new = ceil_div(self._k2 // self._h_group, 4)
                    s_pad = torch.full(
                        (packs_new, 4),
                        0x7F7F7F7F,
                        dtype=h_scale.dtype,
                        device=device,
                    ).transpose(0, 1)[:1, :]
                    s_pad[:, : h_scale.shape[1]].copy_(h_scale)
                    h_scale = s_pad
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
                    recipe_a=(1, self._h_group),
                    recipe_b=(1, FP4_BLOCK),
                )
                if self._b2 is not None:
                    down_out += self._b2[eid_t]
                y[n : n + 1].add_(down_out.float() * router_w)

        return y
