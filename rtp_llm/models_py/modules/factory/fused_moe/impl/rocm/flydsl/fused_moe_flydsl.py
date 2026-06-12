# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL fused MoE entry point.

This mirrors the current Gluon fused MoE fp8/no-quant call shape but keeps the
implementation on a separate public API so existing Gluon behavior is untouched.
Weights are expected to already be in the same preshuffled layout produced by
`atrex.src.triton.fused_moe.fused_moe_helper.shuffle_weight`.
"""

from typing import Optional

import aiter
import torch

import flydsl.compiler as flyc
from flydsl.expr import Stream as _FlyStream

from .fused_moe_helper import (
    ActivationType,
    QuantType,
    get_block_size_m,
    get_m_align,
)
from .tuning import get_qwen_ptpc_fp8_tuning

_gemm1_cf_cache = {}
_gemm2_cf_cache = {}


def _launch_cached(cache, key, launch_fn, args, stream):
    """AOT-compiled dispatch: first call JITs, subsequent calls use CompiledFunction (~5us vs ~60us)."""
    cf = cache.get(key)
    stream_arg = _FlyStream(stream)
    if cf is None:
        cf = flyc.compile(launch_fn, *args, stream_arg)
        cache[key] = cf
        return
    cf(*args, stream_arg)


def _needs_gfx94_bf16_atomic_workaround(dtype: torch.dtype) -> bool:
    if dtype != torch.bfloat16:
        return False
    try:
        from flydsl.runtime.device import get_rocm_arch
    except ImportError:
        return False
    return str(get_rocm_arch()).startswith("gfx94")


def _dtype_to_moe_input(dtype: torch.dtype) -> str:
    if dtype == torch.float16:
        return "fp16"
    if dtype == torch.bfloat16:
        return "bf16"
    raise ValueError(f"fused_moe_flydsl supports fp16/bf16 hidden states, got {dtype}")


def _dtype_to_out(dtype: torch.dtype) -> str:
    if dtype == torch.float16:
        return "f16"
    if dtype == torch.bfloat16:
        return "bf16"
    raise ValueError(f"fused_moe_flydsl supports fp16/bf16 outputs, got {dtype}")


def _empty_scale_like(hidden_states: torch.Tensor) -> torch.Tensor:
    return torch.empty((0,), dtype=torch.float32, device=hidden_states.device)


def _quantize_per_token(
    x: torch.Tensor,
    scale_shape,
    *,
    row_weights: Optional[torch.Tensor] = None,
    input_cache_modifier: int = 0,
    output_cache_modifier: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    from .fp8_quant import fp8_quantize_per_token

    return fp8_quantize_per_token(
        x,
        scale_shape,
        stream=torch.cuda.current_stream(),
        row_weights=row_weights,
        input_cache_modifier=input_cache_modifier,
        output_cache_modifier=output_cache_modifier,
    )


def _quantize_per_token_aiter(
    x: torch.Tensor,
    scale_shape,
    *,
    row_weights: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if row_weights is not None:
        return _quantize_per_token(x, scale_shape, row_weights=row_weights)
    if not x.is_contiguous():
        x = x.contiguous()

    out = torch.empty(x.shape, dtype=torch.float8_e4m3fnuz, device=x.device)
    scales = torch.empty(scale_shape, dtype=torch.float32, device=x.device)
    aiter.dynamic_per_token_scaled_quant(out, x, scales)
    return out, scales


def _quantize_per_token_with_zero(
    x: torch.Tensor,
    scale_shape,
    zero_tensor: torch.Tensor,
    *,
    row_weights: Optional[torch.Tensor] = None,
    input_cache_modifier: int = 0,
    output_cache_modifier: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    from .fp8_quant import fp8_quantize_per_token_with_zero

    return fp8_quantize_per_token_with_zero(
        x,
        scale_shape,
        zero_tensor,
        stream=torch.cuda.current_stream(),
        row_weights=row_weights,
        input_cache_modifier=input_cache_modifier,
        output_cache_modifier=output_cache_modifier,
    )


def fused_moe_flydsl(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    expert_mask: Optional[torch.Tensor] = None,
    activation=ActivationType.Silu,
    quant_type=QuantType.per_Token,
    doweight_stage1: bool = False,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_size_M: Optional[int] = None,
    num_local_tokens: Optional[torch.Tensor] = None,
    moe_sorting_dispatch_policy: int = 0,
    dtype=None,
    hidden_pad: int = 0,
    intermediate_pad: int = 0,
    bias1=None,
    bias2=None,
    out1_ref=None,
    w2_bf16: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run fused MoE through FlyDSL stage1/stage2 kernels.

    The supported conversion target is the same hot path as the current Gluon
    operator: Silu activation, `QuantType.per_Token` fp8 or `QuantType.No`, and
    preshuffled W1/W2 weights.
    """
    del dtype, hidden_pad, intermediate_pad, bias1, bias2, out1_ref
    if activation != ActivationType.Silu:
        raise NotImplementedError(
            f"fused_moe_flydsl only supports Silu, got {activation}"
        )

    quant_type = QuantType(int(quant_type))
    fp8_per_token = quant_type == QuantType.per_Token
    if not fp8_per_token:
        raise NotImplementedError(
            f"fused_moe_flydsl currently supports only QuantType.per_Token, got {quant_type}"
        )
    if fp8_per_token and (w1_scale is None or w2_scale is None):
        raise ValueError("QuantType.per_Token requires w1_scale and w2_scale")

    B = hidden_states.shape[0]
    E, N1, K1 = w1.shape
    N2, K2 = w2.shape[1], w2.shape[2]
    TOPK = topk_ids.shape[1]
    if N1 != 2 * K2:
        raise ValueError(f"expected w1.shape[1] == 2 * w2.shape[2], got {N1} and {K2}")
    if N2 != K1:
        raise ValueError(f"expected w2.shape[1] == hidden dim {K1}, got {N2}")

    model_dim = K1
    inter_dim = K2

    stage1_tile_n = 128
    stage1_tile_k = 128
    stage2_tile_n = 128
    stage2_tile_k = 128
    stage2_mode_name = "atomic"
    stage2_zero_intermediate = True

    M_ALIGN = get_m_align(B, TOPK, E)
    if block_size_M is not None:
        BLOCK_SIZE_M = block_size_M
    elif (E, TOPK, model_dim, inter_dim) == (128, 8, 5120, 3072):
        # FlyDSL-tuned: block_m=32 wins at all B on 397B. The shared helper returns
        # 64 for B>=1024 (m_per_expert_avg > 32), which is empirically slower here.
        BLOCK_SIZE_M = 32
    else:
        BLOCK_SIZE_M = get_block_size_m(B, TOPK, E)

    # Per-shape weight-load cache policy (SC1 = bypass L2/LLC).
    # Small batch is memory-bound: bypassing LLC frees cache for activations and prevents
    # weight-pollution. Large batch reuses weights across MFMA tiles → keep cached.
    # Tuned via op_test sweep (k3a_bnt_sweep): KEEP entries are the per-B best from
    # `s1_nt∈{0,2,4} × s2_nt∈{0,2,4}` × 50-iter rocprofv3 P50.
    stage1_b_nt = 0
    stage2_b_nt = 0
    # K3b investigation: rocdl.sched_* hints in the K-loop. Isolated sweep showed B=4
    # gain (~6.8%, sweep: /tmp/k3b_sched_sweep), but stacked on top of K3a (LLC bypass)
    # the residual improvement drops to <1% and other batches regress.
    stage1_hotloop_sched = False
    stage2_hotloop_sched = False

    if (E, TOPK, model_dim, inter_dim, BLOCK_SIZE_M) == (128, 8, 5120, 3072, 32):
        if B == 1:
            stage1_tile_n = 64
            stage2_tile_n = 64
            stage2_mode_name = "reduce"
            stage2_zero_intermediate = False
            stage2_b_nt = 2
        elif B == 2:
            stage1_tile_n = 64
            stage2_tile_n = 256
            stage2_mode_name = "reduce"
            stage2_zero_intermediate = False
            stage1_b_nt = 2
        elif B == 256:
            stage2_tile_n = 256
            stage1_b_nt = 2
            stage2_b_nt = 2
        elif B in (4, 128, 512, 1024, 2048):
            stage2_tile_n = 256
        if B in (4, 8, 16, 32, 64):
            stage1_b_nt = 2
            stage2_b_nt = 2
        elif B == 128:
            stage1_b_nt = 2
    elif E == 512 and TOPK == 10 and model_dim == 4096:
        # Qwen3.5-397B-A17B PTPC-FP8 (TP=4: inter=256, TP=8: inter=128).
        # All stages memory-bound. Per-B configs from exhaustive rocprofv3
        # kernel-trace P50 sweep (tune_per_shape.py, 30-iter, MI308X).
        tuning = get_qwen_ptpc_fp8_tuning(inter_dim)
        grouped_route_min_b = tuning.grouped_route_min_b if tuning is not None else 32
        if block_size_M is None and B >= grouped_route_min_b:
            # Qwen grouped routing is sparse at small/medium B, but larger B
            # benefits from wider M tiles. These cut points come from the
            # B=1..2048 torch-correct policy probes.
            default_grouped_tile_m = (
                tuning.grouped_tile_m(B, BLOCK_SIZE_M)
                if tuning is not None
                else BLOCK_SIZE_M
            )
            if default_grouped_tile_m in (16, 32, 64):
                BLOCK_SIZE_M = default_grouped_tile_m
        if B <= 4 and inter_dim <= 128:
            # Small-B TP=8 decode: direct FlyDSL routing plus single-N-tile stage1.
            stage1_tile_n = 64
            stage2_tile_n = 256
            stage1_hotloop_sched = True
        elif B <= 4 and inter_dim == 256:
            # V27 used tile_n=32 for speed, but the TP4 direct path can emit NaN
            # for some seeds at B<=4. Keep the stable tile_n=64 path for torch
            # reference correctness.
            stage1_tile_n = 64
            stage2_tile_n = 256
        else:
            stage2_tile_n = 256
            if inter_dim <= 128:
                if 128 <= B <= 512:
                    stage2_tile_n = 512
                elif B == 1024:
                    # V43: TP8 B1024 benefits slightly from a wider stage2 N
                    # tile; B2048 regresses with wider N and stays at 128.
                    stage2_tile_n = 512
                elif B >= 2048:
                    stage2_tile_n = 128
                stage1_hotloop_sched = True
                if 512 <= B <= 1024:
                    # V48: with stage2 M persistence, non-atomic GEMM2 plus
                    # direct topk reduction wins for TP8 B512/B1024. B2048
                    # still favors the atomic path.
                    stage2_mode_name = "reduce"
                    stage2_zero_intermediate = False
            elif inter_dim == 256 and B >= 8:
                # tile_n=32 was faster for V28 TP4 B>=32, but it is not
                # numerically stable against the torch FP8 reference. Use
                # correctness-safe tile sizes selected from B=1..2048 probes.
                stage1_tile_n = 64 if B >= 2048 or B < 32 else 128
                if B >= 2048:
                    # V47/V57: after tight route max-blocks, B1024 is faster
                    # with tile_n=256 plus persist_m=1; B2048 keeps tile_n=512.
                    stage2_tile_n = 512
                if B >= 256:
                    # Large-B TP4 is dominated by the stage2 atomic path.
                    # SC1/bypass on W2 is consistently better in the V33
                    # rocprofv3 probes. After tight max-blocks, K64 remains
                    # useful only at B256; V63/V65 use K128 for B512/B1024.
                    stage2_b_nt = 2
                    if B >= 2048:
                        # V74/V75: with tight route max-blocks and chunked
                        # persistence, keeping W2 cached is faster at B2048.
                        stage2_b_nt = 0
                if B == 256 and BLOCK_SIZE_M != 16:
                    stage1_tile_k = 64
                if B >= 512:
                    # V42: for TP4 large-B, non-atomic stage2 plus direct
                    # topk reduction is faster than BF16 atomics after V40's
                    # stage2 tile-M split. TP8 regresses and stays atomic.
                    stage2_mode_name = "reduce"
                    stage2_zero_intermediate = False
                if B == 512 and BLOCK_SIZE_M == 16:
                    # V198: under the restored user-facing M512 tile-M=16
                    # policy, atomic mode avoids the reduction launch and
                    # narrowly meets the TP4 aiter+10% gate.
                    stage2_mode_name = "atomic"
                    stage2_zero_intermediate = True
            if B >= 8:
                stage1_b_nt = 2
    if model_dim % stage1_tile_k != 0 or inter_dim % stage2_tile_k != 0:
        raise ValueError(
            f"FlyDSL MoE requires model_dim divisible by stage1_tile_k={stage1_tile_k} "
            f"and inter_dim divisible by stage2_tile_k={stage2_tile_k}, "
            f"got model_dim={model_dim}, inter_dim={inter_dim}"
        )
    if inter_dim % stage1_tile_n != 0 or model_dim % stage2_tile_n != 0:
        raise ValueError(
            f"FlyDSL MoE requires inter_dim divisible by stage1_tile_n={stage1_tile_n} "
            f"and model_dim divisible by stage2_tile_n={stage2_tile_n}, "
            f"got model_dim={model_dim}, inter_dim={inter_dim}"
        )
    stage2_tile_m_default = BLOCK_SIZE_M
    if E == 512 and TOPK == 10 and model_dim == 4096:
        if inter_dim <= 128 and B >= 16:
            # V119: tile_m=16 halves per-CTA LDS/CShuffle/MFMA overhead for the
            # short-K (K=128) Stage2 GEMM. rocprofv3 TP=8 kernel-only:
            # B=16 +33%, B=32 +36%, B=64 +39%, B=128 +38%, B=512 +35%.
            # Correctness PASS all shapes (max_rel<0.012, rel_norm<0.005).
            stage2_tile_m_default = 16
        elif inter_dim == 256 and B >= 2048:
            # V119's tile_m=16 was useful for isolated task66 load-only gates,
            # but it regresses the full TP4 API-vs-aiter path at M<=512. Keep
            # the older tile_m=32 user-facing policy except for the large-B
            # task66/throughput path.
            stage2_tile_m_default = 16
    stage2_tile_m = stage2_tile_m_default
    if BLOCK_SIZE_M % stage2_tile_m != 0:
        raise ValueError(
            f"stage2_tile_m={stage2_tile_m} must divide routing BLOCK_SIZE_M={BLOCK_SIZE_M}"
        )
    stage2_force_f32_atomic = False
    if _needs_gfx94_bf16_atomic_workaround(hidden_states.dtype):
        # gfx94 bf16 CShuffle atomics can corrupt packed bf16 pairs, while the
        # reduce fallback has shown non-finite output for real Qwen3.5 PTPC
        # activations. Accumulate stage2 into an f32 scratch tensor with scalar
        # atomics, then cast back to the bf16 API dtype.
        stage2_mode_name = "atomic"
        stage2_force_f32_atomic = True

    stream = torch.cuda.current_stream()
    use_direct_route = (
        fp8_per_token
        and expert_mask is None
        and num_local_tokens is None
        and moe_sorting_dispatch_policy == 0
        and E == 512
        and TOPK == 10
        and model_dim == 4096
    )
    if not use_direct_route:
        raise NotImplementedError(
            "fused_moe_flydsl is restricted to the Qwen3.5 direct-routing path "
            "(E=512, TOPK=10, hidden=4096, no expert_mask, no num_local_tokens) "
            "to avoid CK/AITER operator fallbacks."
        )
    if use_direct_route:
        if topk_ids.dtype != torch.int32:
            topk_ids = topk_ids.to(torch.int32)
        if topk_weight.dtype != torch.float32:
            topk_weight = topk_weight.float()
        tuning = get_qwen_ptpc_fp8_tuning(inter_dim)
        grouped_route_min_b = tuning.grouped_route_min_b if tuning is not None else 32
        use_route_free = (
            tuning.use_route_free(B) if tuning is not None else False
        ) and not doweight_stage1
        use_grouped_route = (not use_route_free) and B >= grouped_route_min_b
        if use_route_free:
            sorted_ids = torch.empty((0,), dtype=torch.int32, device=topk_ids.device)
            sorted_weights = topk_weight.reshape(B * TOPK).contiguous()
            sorted_expert_ids = topk_ids.reshape(B * TOPK).contiguous()
            num_valid_ids = torch.empty((0,), dtype=torch.int32, device=topk_ids.device)
        elif use_grouped_route:
            from .moe_routing import grouped_moe_route

            sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids = (
                grouped_moe_route(
                    topk_ids,
                    topk_weight,
                    experts=E,
                    topk=TOPK,
                    tile_m=BLOCK_SIZE_M,
                    stream=stream,
                )
            )
        else:
            from .moe_routing import direct_moe_route

            sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids = (
                direct_moe_route(
                    topk_ids,
                    topk_weight,
                    topk=TOPK,
                    tile_m=BLOCK_SIZE_M,
                    stream=stream,
                )
            )
    else:
        use_route_free = False
    del M_ALIGN

    # Grouped routing may allocate inactive fixed expert slots; GEMMs skip slots
    # whose expert id is -1 before issuing weight loads.
    _grid_expert_blocks = sorted_expert_ids.numel()

    quant_in_cache_default = 0
    quant_out_cache_default = 0
    if E == 512 and TOPK == 10 and model_dim == 4096 and inter_dim == 256 and B >= 2048:
        quant_in_cache_default = 4
        quant_out_cache_default = 2
    quant_in_cache = quant_in_cache_default
    quant_out_cache = quant_out_cache_default
    tuning = get_qwen_ptpc_fp8_tuning(inter_dim)
    use_aiter_quant = tuning.use_aiter_quant(B) if tuning is not None else False

    if fp8_per_token:
        if use_aiter_quant:
            a1_qt, a1_scale = _quantize_per_token_aiter(hidden_states, (B, 1))
        else:
            a1_qt, a1_scale = _quantize_per_token(
                hidden_states,
                (B, 1),
                input_cache_modifier=quant_in_cache,
                output_cache_modifier=quant_out_cache,
            )
        stage_dtype = "fp8"
    else:
        a1_qt = hidden_states
        a1_scale = _empty_scale_like(hidden_states)
        w1_scale = _empty_scale_like(hidden_states)
        stage_dtype = _dtype_to_moe_input(hidden_states.dtype)

    out_dtype = _dtype_to_out(hidden_states.dtype)
    # FlyDSL 1-stage fused kernel (opt-in). TP8 uses the original decode path;
    # TP4 B2B can either use predequantized W2 or requantize A2 in-CTA and
    # keep W2 on the standard FP8 path.
    fused_1stage_fp8_w2 = False
    _use_fused_1stage = False

    if _use_fused_1stage:
        from .moe_gemm_1stage import compile_moe_gemm_fused

        stream = torch.cuda.current_stream()
        fused_tile_k = inter_dim
        fused_tile_n_out_default = 256 if inter_dim == 256 else 128
        fused_tile_n_out = fused_tile_n_out_default
        fused_out_dtype = out_dtype
        cast_fused_output = False
        kernel_fused = compile_moe_gemm_fused(
            model_dim=model_dim,
            inter_dim=inter_dim,
            experts=E,
            topk=TOPK,
            tile_m=BLOCK_SIZE_M,
            tile_n_out=fused_tile_n_out,
            tile_k=fused_tile_k,
            out_dtype=fused_out_dtype,
            b_cache_modifier_w1=stage1_b_nt,
            b_cache_modifier_w2=0,
            loop_n_in_block=False,
            stage2_use_fp8_w2=fused_1stage_fp8_w2,
            n_tiles_per_block=0,
            cshuffle_nlane=32,
            total_threads=256,
            fold_route_weight_into_a2_scale=False,
            out_tile_pair=1,
            cshuffle_lds_xor=False,
        )
        w2_fused = w2 if fused_1stage_fp8_w2 else w2_bf16
        output_dtype = torch.float16 if cast_fused_output else hidden_states.dtype
        output = torch.zeros(
            (B, model_dim), dtype=output_dtype, device=hidden_states.device
        )
        kernel_fused(
            output,
            a1_qt,
            w1,
            w2_fused,
            a1_scale,
            w1_scale,
            w2_scale,
            sorted_ids,
            sorted_expert_ids,
            sorted_weights,
            num_valid_ids,
            B,
            model_dim,
            inter_dim,
            _grid_expert_blocks,
            stream,
        )
        return output.to(hidden_states.dtype) if cast_fused_output else output

    from .moe_gemm_2stage import (
        MoeGemm2Mode,
        compile_moe_gemm1,
        compile_moe_gemm1_route_free,
        compile_moe_gemm2_ex,
    )

    # CShuffle epilog requires tile_n divisible by (cshuffle_nlane * e_vec) = 128.
    stage1_cshuffle = False
    # V116/V195: iglp_opt can trigger LLVM AMDGPUIGroupLP assertions on TP4
    # (inter_dim=256) and block_m<32. Keep it only on the validated TP8 path.
    if (inter_dim > 128 or BLOCK_SIZE_M < 32) and stage1_hotloop_sched:
        stage1_hotloop_sched = False
    # V116: fast_barrier + fine_sched synergy for memory-bound shapes.
    stage1_fast_barrier = False
    stage1_fine_sched = False
    use_route_free_stage1 = (
        use_route_free
        and use_direct_route
        and fp8_per_token
        and not stage1_cshuffle
        and stage_dtype == "fp8"
        and model_dim == 4096
        and E == 512
        and TOPK == 10
    )
    if use_route_free_stage1:
        stage1 = compile_moe_gemm1_route_free(
            model_dim=model_dim,
            inter_dim=inter_dim,
            experts=E,
            topk=TOPK,
            sort_tile_m=BLOCK_SIZE_M,
            tile_n=stage1_tile_n,
            tile_k=stage1_tile_k,
            doweight_stage1=doweight_stage1,
            in_dtype=stage_dtype,
            out_dtype=out_dtype,
            b_cache_modifier=stage1_b_nt,
            enable_hotloop_sched=stage1_hotloop_sched,
            fast_barrier=stage1_fast_barrier,
            fine_sched=stage1_fine_sched,
        )
    else:
        stage1 = compile_moe_gemm1(
            model_dim=model_dim,
            inter_dim=inter_dim,
            experts=E,
            topk=TOPK,
            tile_m=BLOCK_SIZE_M,
            tile_n=stage1_tile_n,
            tile_k=stage1_tile_k,
            doweight_stage1=doweight_stage1,
            in_dtype=stage_dtype,
            out_dtype=out_dtype,
            use_cshuffle_epilog=stage1_cshuffle,
            b_cache_modifier=stage1_b_nt,
            enable_hotloop_sched=stage1_hotloop_sched,
            fast_barrier=stage1_fast_barrier,
            fine_sched=stage1_fine_sched,
        )
    stage1_out = torch.empty(
        (B, TOPK, inter_dim),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    _s1_args = (
        stage1_out,
        a1_qt,
        w1,
        a1_scale,
        w1_scale,
        sorted_ids,
        sorted_expert_ids,
        sorted_weights,
        num_valid_ids,
        B,
        inter_dim,
        model_dim,
        _grid_expert_blocks,
    )
    _launch_cached(_gemm1_cf_cache, id(stage1), stage1, _s1_args, stream)

    # Decode optimization: skip intermediate fp8 quantization when w2_bf16 is provided.
    # The caller precomputes w2_bf16 = shuffle(dequant(w2_fp8, w2_scale)) at model load.
    # Only beneficial when inter_dim<=128 (TP=8): bf16 stage2 K=128 is single-pass.
    # For inter_dim=256 (TP=4), bf16 stage2 has 2x K-tiles and larger W2 loads,
    # which outweighs the quant kernel savings at small B.
    skip_a2_quant_max_b = 4
    _skip_a2_quant = (
        w2_bf16 is not None
        and fp8_per_token
        and B <= skip_a2_quant_max_b
        and inter_dim <= 128
    )

    fold_weight_into_a2_scale = False
    _fuse_quant2_zero = False
    _fused_zero_output = None
    if _skip_a2_quant:
        # bf16 path: stage1 output (bf16) → bf16 stage2 directly (no quant)
        a2_for_s2 = stage1_out
        a2_scale_for_s2 = _empty_scale_like(hidden_states)
        w2_for_s2 = w2_bf16
        w2_scale_for_s2 = _empty_scale_like(hidden_states)
        s2_in_dtype = "bf16"
    elif fp8_per_token:
        fold_weight_into_a2_scale = (
            E == 512 and TOPK == 10 and model_dim == 4096 and B >= 512
        )
        a2_row_weights = (
            topk_weight.reshape(B * TOPK)
            if fold_weight_into_a2_scale and not doweight_stage1
            else None
        )
        _fuse_quant2_zero = (
            stage2_mode_name == "atomic"
            and not stage2_force_f32_atomic
            and hidden_states.dtype == torch.bfloat16
            and use_grouped_route
            and E == 512
            and TOPK == 10
            and model_dim == 4096
            and (B * model_dim) % 8 == 0
        )
        if _fuse_quant2_zero:
            _output_for_zero = torch.empty(
                (B, model_dim), dtype=hidden_states.dtype, device=hidden_states.device
            )
            a2_qt, a2_scale = _quantize_per_token_with_zero(
                stage1_out,
                (B, TOPK, 1),
                _output_for_zero,
                row_weights=a2_row_weights,
                input_cache_modifier=quant_in_cache,
                output_cache_modifier=quant_out_cache,
            )
            _fused_zero_output = _output_for_zero
        else:
            if use_aiter_quant:
                a2_qt, a2_scale = _quantize_per_token_aiter(
                    stage1_out,
                    (B, TOPK, 1),
                    row_weights=a2_row_weights,
                )
            else:
                a2_qt, a2_scale = _quantize_per_token(
                    stage1_out,
                    (B, TOPK, 1),
                    row_weights=a2_row_weights,
                    input_cache_modifier=quant_in_cache,
                    output_cache_modifier=quant_out_cache,
                )
            _fused_zero_output = None
        a2_for_s2 = a2_qt
        a2_scale_for_s2 = a2_scale
        w2_for_s2 = w2
        w2_scale_for_s2 = w2_scale
        s2_in_dtype = stage_dtype
    else:
        a2_for_s2 = stage1_out
        a2_scale_for_s2 = _empty_scale_like(hidden_states)
        w2_for_s2 = w2
        w2_scale_for_s2 = _empty_scale_like(hidden_states)
        s2_in_dtype = stage_dtype

    stage2_skip_invalid_lds_write = False
    stage2_lds_sorted_ids = False
    stage2_precompute_row_base = False
    stage2_readfirst_metadata = False
    stage2_m_fast_grid = False
    stage2_persist_m_default = 1
    if (
        E == 512
        and TOPK == 10
        and model_dim == 4096
        and inter_dim <= 128
        and 32 <= B < 512
    ):
        # V119: persist_m=2 amortizes single-K-tile (K=128) startup cost for TP=8.
        # Sweep confirms pm=2 optimal for B=32-256; pm=1 for B<32 and B>=512 (already handled below).
        stage2_persist_m_default = 2
    if E == 512 and TOPK == 10 and model_dim == 4096 and B >= 512:
        if inter_dim == 256:
            if B == 512:
                # V212: M512 is faster with full Stage2 M-tile parallelism.
                # pm16 collapses grid_y to 50 and leaves the memory-bound
                # Stage2 underfilled; pm1 restores grid_y to 800.
                stage2_persist_m_default = 1
            elif B == 1024:
                stage2_persist_m_default = 1
            else:
                stage2_persist_m_default = 32
        elif inter_dim <= 128:
            stage2_persist_m_default = 1 if B == 1024 else 16
    stage2_persist_m = stage2_persist_m_default
    stage2_persistent_grid_y = 0
    stage2_persistent_chunk_y_default = 0
    if E == 512 and TOPK == 10 and model_dim == 4096 and inter_dim == 256 and B == 2048:
        stage2_persistent_chunk_y_default = 64
    stage2_persistent_chunk_y = stage2_persistent_chunk_y_default
    stage2_group_size_m_default = 1
    if E == 512 and TOPK == 10 and model_dim == 4096 and inter_dim == 256 and B == 2048:
        stage2_group_size_m_default = 4
    stage2_group_size_m = stage2_group_size_m_default
    stage2_mode = (
        MoeGemm2Mode.REDUCE if stage2_mode_name == "reduce" else MoeGemm2Mode.ATOMIC
    )
    reduction_in_cache_default = 0
    reduction_out_cache_default = 0
    if E == 512 and TOPK == 10 and model_dim == 4096 and inter_dim == 256 and B >= 2048:
        reduction_in_cache_default = 2
        reduction_out_cache_default = 2
    stage2_out_dtype = hidden_states.dtype
    stage2_out_dtype_name = out_dtype
    cast_stage2_output = False
    stage2_reduce_intermediate_dtype_name = None
    f16_reduce_intermediate_default = (
        E == 512 and TOPK == 10 and model_dim == 4096 and inter_dim == 256 and B >= 2048
    )
    if stage2_force_f32_atomic:
        stage2_out_dtype = torch.float32
        stage2_out_dtype_name = "f32"
        cast_stage2_output = True
    elif (
        stage2_mode == MoeGemm2Mode.REDUCE
        and hidden_states.dtype == torch.bfloat16
        and f16_reduce_intermediate_default
    ):
        # V92: keep the API output in bf16, but let non-atomic GEMM2 write the
        # topk intermediate as f16. This removes bf16 conversion cost from the
        # dominant TP4 B2048 stage2 kernel; the reduction converts directly to
        # bf16 final output without an extra cast kernel.
        stage2_reduce_intermediate_dtype_name = "f16"

    if fp8_per_token and _fuse_quant2_zero and _fused_zero_output is not None:
        output = _fused_zero_output
    else:
        output = torch.empty(
            (B, model_dim), dtype=stage2_out_dtype, device=hidden_states.device
        )
    if stage2_mode == MoeGemm2Mode.ATOMIC:
        if fp8_per_token and _fuse_quant2_zero and _fused_zero_output is not None:
            pass
        else:
            use_custom_zero = (
                hidden_states.dtype == torch.bfloat16
                and not stage2_force_f32_atomic
                and output.is_contiguous()
                and use_grouped_route
                and E == 512
                and TOPK == 10
                and model_dim == 4096
            )
            zeroed = False
            if use_custom_zero:
                from .zero_fill import zero_fill_tensor

                zeroed = zero_fill_tensor(output, stream=stream)
            if not zeroed:
                output.zero_()

    stage2_cshuffle_nlane_default = 32
    if E == 512 and TOPK == 10 and model_dim == 4096 and inter_dim == 256 and B >= 2048:
        # V114: TP4 B2048 reduce-mode stage2 is faster with a narrower CShuffle
        # lane group; other Qwen shapes keep the historical 32-lane geometry.
        stage2_cshuffle_nlane_default = 16
    stage2_cshuffle_nlane = stage2_cshuffle_nlane_default
    stage2 = compile_moe_gemm2_ex(
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=E,
        topk=TOPK,
        tile_m=stage2_tile_m,
        tile_n=stage2_tile_n,
        tile_k=stage2_tile_k,
        doweight_stage2=(not doweight_stage1)
        and not (
            fp8_per_token and s2_in_dtype == stage_dtype and fold_weight_into_a2_scale
        ),
        in_dtype=s2_in_dtype,
        out_dtype=stage2_out_dtype_name,
        use_cshuffle_epilog=not stage2_force_f32_atomic,
        mode=stage2_mode,
        zero_intermediate=stage2_zero_intermediate,
        b_cache_modifier=stage2_b_nt,
        enable_hotloop_sched=stage2_hotloop_sched,
        skip_invalid_lds_write=stage2_skip_invalid_lds_write,
        use_lds_sorted_ids=stage2_lds_sorted_ids,
        precompute_row_base=stage2_precompute_row_base,
        readfirst_metadata=stage2_readfirst_metadata,
        m_fast_grid=stage2_m_fast_grid,
        sort_block_m=BLOCK_SIZE_M,
        persist_m=stage2_persist_m,
        persistent_grid_y=stage2_persistent_grid_y,
        persistent_chunk_y=stage2_persistent_chunk_y,
        reduce_intermediate_dtype=stage2_reduce_intermediate_dtype_name,
        group_size_m=stage2_group_size_m,
        cshuffle_nlane=stage2_cshuffle_nlane,
        reduction_input_cache_modifier=reduction_in_cache_default,
        reduction_output_cache_modifier=reduction_out_cache_default,
        single_token_route=use_route_free,
    )
    _s2_args = (
        output,
        a2_for_s2,
        w2_for_s2,
        a2_scale_for_s2,
        w2_scale_for_s2,
        sorted_ids,
        sorted_expert_ids,
        sorted_weights,
        num_valid_ids,
        B,
        model_dim,
        inter_dim,
        _grid_expert_blocks,
    )
    if hasattr(stage2, "_gemm2_exe"):
        stage2(*_s2_args, stream)
    else:
        _launch_cached(_gemm2_cf_cache, id(stage2), stage2, _s2_args, stream)
    if cast_stage2_output:
        return output.to(hidden_states.dtype)
    return output
