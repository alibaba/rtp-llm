# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""1-Stage Fused MoE GEMM kernel for decode/B2B probes.

Fuses stage1 (X×W1→SiLU→intermediate) and stage2 (intermediate×W2→output) into a
single kernel launch, eliminating intermediate global memory traffic, the quant
kernel, and the zero() kernel.

Constraints:
- FP8 input (X, W1)
- BF16 W2 (precomputed: dequant(W2_fp8, scale_w2) → bf16, preshuffled)
- BF16 output (atomic add)
- inter_dim must equal tile_k (128 for TP8, 256 for TP4) — single-pass Phase 2
- Designed for decode batch sizes (B=1-4)
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, buffer_ops, const_expr, gpu, range_constexpr, rocdl, vector
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

try:
    from flydsl.runtime.device import supports_bf16_global_atomics
except ImportError:
    def supports_bf16_global_atomics(arch: str) -> bool:
        return str(arch).startswith(("gfx94", "gfx95", "gfx12"))

from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm, scf
from flydsl.expr.typing import T
from flydsl.expr.vector import ReductionOp
from .kernels.mfma_epilogues import c_shuffle_epilog, mfma_epilog
from .kernels.mfma_preshuffle_pipeline import (
    buffer_copy_gmem16_dwordx4,
    crd2idx,
    lds_store_16b_xor16,
    lds_store_8b_xor16,
    lds_store_4b_xor16,
    load_b_pack_k32,
    make_preshuffle_b_layout,
    swizzle_xor16,
    tile_chunk_coord_i32,
)

_FP32_MIN_NORMAL = 1.1754943508222875e-38

from contextlib import contextmanager

@contextmanager
def _if_then(if_op):
    with ir.InsertionPoint(if_op.then_block):
        try:
            yield if_op.then_block
        finally:
            blk = if_op.then_block
            if (not blk.operations) or not isinstance(blk.operations[-1], scf.YieldOp):
                scf.YieldOp([])


@functools.lru_cache(maxsize=256)
def compile_moe_gemm_fused(
    *,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    tile_m: int = 32,
    tile_n_out: int = 128,
    tile_k: int = 128,
    out_dtype: str = "bf16",
    b_cache_modifier_w1: int = 0,
    b_cache_modifier_w2: int = 0,
    loop_n_in_block: bool = False,
    stage2_use_fp8_w2: bool = False,
    n_tiles_per_block: int = 0,
    cshuffle_nlane: int = 32,
    total_threads: int = 256,
    fold_route_weight_into_a2_scale: bool = False,
    out_tile_pair: int = 1,
    cshuffle_lds_xor: bool = False,
):
    """Compile 1-stage fused MoE kernel for decode.

    Requirements:
    - inter_dim == tile_k (single-pass Phase 2)
    - FP8 input (X, W1), BF16 W2 (precomputed skip-quant), BF16 output
    """
    if inter_dim != tile_k:
        raise ValueError(
            f"1-stage fused kernel requires inter_dim == tile_k, got {inter_dim} != {tile_k}"
        )
    if model_dim % tile_n_out != 0:
        raise ValueError(f"model_dim ({model_dim}) must be divisible by tile_n_out ({tile_n_out})")
    if model_dim % tile_k != 0:
        raise ValueError(f"model_dim ({model_dim}) must be divisible by tile_k ({tile_k})")
    if stage2_use_fp8_w2 and not loop_n_in_block:
        raise ValueError("stage2_use_fp8_w2 is only implemented for B2B N-loop mode")
    if stage2_use_fp8_w2 and inter_dim != 256:
        raise ValueError("stage2_use_fp8_w2 currently requires TP4 inter_dim=256")
    if fold_route_weight_into_a2_scale and not (
        loop_n_in_block and stage2_use_fp8_w2 and inter_dim == 256
    ):
        raise ValueError(
            "fold_route_weight_into_a2_scale is only validated for TP4 FP8-W2 B2B mode"
        )
    if out_tile_pair not in (1, 2):
        raise ValueError("out_tile_pair currently supports only 1 or 2")
    if out_tile_pair != 1 and not (loop_n_in_block and stage2_use_fp8_w2 and inter_dim == 256):
        raise ValueError("out_tile_pair is only validated for TP4 FP8-W2 B2B mode")
    if out_tile_pair != 1 and fold_route_weight_into_a2_scale:
        raise ValueError("out_tile_pair is validated only with route-weight folding disabled")
    if cshuffle_lds_xor and not (loop_n_in_block and stage2_use_fp8_w2 and inter_dim == 256):
        raise ValueError("cshuffle_lds_xor is only validated for TP4 FP8-W2 B2B mode")
    if cshuffle_lds_xor and (n_tiles_per_block != 0 or out_tile_pair != 1):
        raise ValueError("cshuffle_lds_xor requires the default single-CTA output loop")
    if cshuffle_lds_xor and (tile_n_out != 512 or cshuffle_nlane != 32 or total_threads != 256):
        raise ValueError("cshuffle_lds_xor is validated only for tile_n_out=512, NLane32, 256 threads")
    if total_threads not in (256, 512):
        raise ValueError("total_threads currently supports only 256 or 512")
    if total_threads != 256 and not (loop_n_in_block and stage2_use_fp8_w2 and inter_dim == 256):
        raise ValueError("non-default total_threads is only validated for TP4 FP8-W2 B2B mode")
    num_waves = total_threads // 64
    if inter_dim % num_waves != 0:
        raise ValueError(f"inter_dim ({inter_dim}) must be divisible by num_waves ({num_waves})")
    if tile_n_out % num_waves != 0:
        raise ValueError(f"tile_n_out ({tile_n_out}) must be divisible by num_waves ({num_waves})")
    if (inter_dim // num_waves) % 16 != 0:
        raise ValueError("stage1 n_per_wave must be divisible by 16")
    if (tile_n_out // num_waves) % 16 != 0:
        raise ValueError("stage2 n_per_wave must be divisible by 16")
    if cshuffle_nlane not in (16, 32, 64):
        raise ValueError("cshuffle_nlane currently supports only 16, 32, or 64")
    if total_threads % cshuffle_nlane != 0:
        raise ValueError(
            f"total_threads ({total_threads}) must be divisible by cshuffle_nlane ({cshuffle_nlane})"
        )
    if tile_m % (total_threads // cshuffle_nlane) != 0:
        raise ValueError(
            f"tile_m ({tile_m}) must be divisible by CShuffleMLane "
            f"({total_threads // cshuffle_nlane})"
        )
    if tile_n_out % (cshuffle_nlane * 2) != 0:
        raise ValueError(
            f"tile_n_out ({tile_n_out}) must be divisible by cshuffle_nlane*2 "
            f"({cshuffle_nlane * 2})"
        )
    total_n_tiles = model_dim // tile_n_out
    if out_tile_pair != 1:
        if n_tiles_per_block != 0:
            raise ValueError("out_tile_pair requires n_tiles_per_block=0")
        if total_n_tiles % out_tile_pair != 0:
            raise ValueError(
                f"output N tiles ({total_n_tiles}) must be divisible by out_tile_pair "
                f"({out_tile_pair})"
            )
    if n_tiles_per_block < 0:
        raise ValueError("n_tiles_per_block must be non-negative")
    if n_tiles_per_block != 0:
        if not loop_n_in_block:
            raise ValueError("n_tiles_per_block requires B2B N-loop mode")
        if not stage2_use_fp8_w2:
            raise ValueError("n_tiles_per_block is only validated for FP8-W2 B2B mode")
        if n_tiles_per_block not in (2, 4):
            raise ValueError("n_tiles_per_block currently supports only 2 or 4")
        if total_n_tiles % n_tiles_per_block != 0:
            raise ValueError(
                f"output N tiles ({total_n_tiles}) must be divisible by n_tiles_per_block "
                f"({n_tiles_per_block})"
            )

    gpu_arch = get_hip_arch()
    allocator = SmemAllocator(None, arch=gpu_arch)

    out_is_bf16 = out_dtype == "bf16"
    out_is_f16 = out_dtype in ("f16", "fp16", "half")
    if not (out_is_bf16 or out_is_f16):
        raise ValueError("1-stage fused kernel only supports bf16/f16 output")

    _needs_global_atomic_bf16 = out_is_bf16 and "gfx942" in gpu_arch

    # Phase 1 parameters (FP8)
    elem_bytes_fp8 = 1
    tile_k_bytes_fp8 = tile_k * elem_bytes_fp8  # 128
    lds_stride_stage1 = tile_k  # no padding with XOR16

    # Phase 2 parameters (BF16)
    elem_bytes_bf16 = 2
    tile_k_bytes_bf16 = tile_k * elem_bytes_bf16  # 256
    elem_bytes_w2 = elem_bytes_fp8 if stage2_use_fp8_w2 else elem_bytes_bf16
    tile_k_bytes_w2 = tile_k * elem_bytes_w2

    # Stage1 tile_n covers all gate+up columns
    tile_n_stage1 = inter_dim  # 128 (each wave handles n_per_wave=32 of gate AND 32 of up)

    bytes_x_per_tile = tile_m * tile_k * elem_bytes_fp8
    if bytes_x_per_tile % total_threads != 0:
        raise ValueError(f"tile_m*tile_k must be divisible by {total_threads}")
    bytes_per_thread_x = bytes_x_per_tile // total_threads

    # LDS sizing:
    # Phase 1: ping-pong X tiles = 2 * tile_m * lds_stride_stage1 * 1B = 2*32*128 = 8KB
    # Phase 1.5→2: intermediate bf16 = tile_m * inter_dim * 2B = 32*128*2 = 8KB
    # Phase 3: CShuffle output = tile_m * tile_n_out * 2B = 32*128*2 = 8KB
    lds_pingpong_bytes = 2 * tile_m * lds_stride_stage1 * elem_bytes_fp8
    lds_intermediate_bytes = tile_m * inter_dim * elem_bytes_bf16
    lds_a2_fp8_bytes = tile_m * inter_dim * elem_bytes_fp8
    lds_cshuffle_bytes = tile_m * tile_n_out * elem_bytes_bf16
    lds_out_offset_bytes = 0
    lds_a2_fp8_offset_bytes = 0
    lds_scale_offset_bytes = 0
    lds_red_offset_bytes = 0
    red_slots = total_threads // 64
    if stage2_use_fp8_w2:
        lds_a2_fp8_offset_bytes = ((lds_intermediate_bytes + 15) // 16) * 16
        lds_scale_offset_bytes = ((lds_a2_fp8_offset_bytes + lds_a2_fp8_bytes + 15) // 16) * 16
        lds_red_offset_bytes = ((lds_scale_offset_bytes + tile_m * 4 + 15) // 16) * 16
        lds_out_offset_bytes = ((lds_red_offset_bytes + red_slots * 4 + 15) // 16) * 16
        lds_total_bytes = max(
            lds_pingpong_bytes,
            lds_out_offset_bytes + lds_cshuffle_bytes,
        )
    elif loop_n_in_block:
        lds_out_offset_bytes = ((lds_intermediate_bytes + 15) // 16) * 16
        lds_total_bytes = max(
            lds_pingpong_bytes,
            lds_out_offset_bytes + lds_cshuffle_bytes,
        )
    else:
        lds_total_bytes = max(lds_pingpong_bytes, lds_intermediate_bytes, lds_cshuffle_bytes)
    lds_total_elems = lds_total_bytes  # fp8 elem_bytes=1

    lds_alloc_bytes = lds_total_elems
    lds_alloc_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_alloc_offset + lds_alloc_bytes

    # CShuffle epilogue parameters
    _cshuffle_nlane = cshuffle_nlane
    _e_vec = 2  # atomic mode

    # Module name for cache key
    _bnt_w1_tag = f"_bntw1_{b_cache_modifier_w1}" if b_cache_modifier_w1 != 0 else ""
    _bnt_w2_tag = f"_bntw2_{b_cache_modifier_w2}" if b_cache_modifier_w2 != 0 else ""
    _b2b_tag = "_b2bn" if loop_n_in_block else ""
    _fp8w2_tag = "_fp8w2" if stage2_use_fp8_w2 else ""
    _out_tag = "_outf16" if out_dtype in ("f16", "fp16", "half") else ""
    _ntpb_tag = f"_ntpb{n_tiles_per_block}" if n_tiles_per_block != 0 else ""
    _csnl_tag = f"_csnl{cshuffle_nlane}" if cshuffle_nlane != 32 else ""
    _thr_tag = f"_thr{total_threads}" if total_threads != 256 else ""
    _wina2_tag = "_wina2" if fold_route_weight_into_a2_scale else ""
    _opair_tag = f"_opair{out_tile_pair}" if out_tile_pair != 1 else ""
    _csxor_tag = "_csxor" if cshuffle_lds_xor else ""
    module_name = (
        f"mfma_moe_fused1s_fp8_bf16"
        f"_t{tile_m}x{tile_n_out}x{tile_k}"
        f"_i{inter_dim}{_csnl_tag}"
        f"{_bnt_w1_tag}{_bnt_w2_tag}{_b2b_tag}{_fp8w2_tag}{_out_tag}"
        f"{_ntpb_tag}{_thr_tag}{_wina2_tag}{_opair_tag}{_csxor_tag}"
        f"_abi2"
    ).replace("-", "_")

    kernel_decorator = (
        flyc.kernel(known_block_size=[total_threads, 1, 1])
        if total_threads > 256
        else flyc.kernel
    )

    if True:

        @kernel_decorator
        def moe_gemm_fused(
            arg_out: fx.Tensor,       # [tokens, model_dim] bf16
            arg_x: fx.Tensor,         # [tokens, model_dim] fp8
            arg_w1: fx.Tensor,        # [E, 2*inter_dim, model_dim] fp8 preshuffled
            arg_w2: fx.Tensor,        # [E, model_dim, inter_dim] bf16 preshuffled
            arg_scale_x: fx.Tensor,   # [tokens, 1] f32
            arg_scale_w1: fx.Tensor,  # [E, 2*inter_dim, 1] f32
            arg_scale_w2: fx.Tensor,  # [E, model_dim, 1] f32 for FP8 W2 path
            arg_sorted_token_ids: fx.Tensor,
            arg_expert_ids: fx.Tensor,
            arg_sorted_weights: fx.Tensor,
            arg_num_valid_ids: fx.Tensor,
            i32_tokens_in: fx.Int32,
            i32_model_dim_in: fx.Int32,
            i32_inter_dim_in: fx.Int32,
            i32_size_expert_ids_in: fx.Int32,
        ):
            tokens_in = arith.index_cast(T.index, i32_tokens_in)
            model_dim_in = arith.index_cast(T.index, i32_model_dim_in)
            size_expert_ids_in = arith.index_cast(T.index, i32_size_expert_ids_in)

            # Element types
            x_elem = T.f8
            w1_elem = T.f8
            w2_elem = T.f8 if stage2_use_fp8_w2 else T.bf16
            out_elem = T.bf16 if out_is_bf16 else T.f16
            vec16_elems_fp8 = 16  # 16 fp8 in 16B
            vec8_elems_fp8 = 8
            vec4_elems_fp8 = 4
            vec16_x = T.vec(vec16_elems_fp8, x_elem)
            vec8_x = T.vec(vec8_elems_fp8, x_elem)
            vec4_x = T.vec(vec4_elems_fp8, x_elem)

            acc_init = arith.constant_vector(0.0, T.f32x4)

            # --- Phase 1 layouts ---
            # W1 preshuffle: c_n = E * 2*inter_dim, c_k = model_dim
            c_n_w1 = arith.index(experts * 2 * inter_dim)
            k_in_stage1 = model_dim_in
            kpack_bytes_fp8 = 16
            b_layout_w1 = make_preshuffle_b_layout(
                arith, c_n=c_n_w1, c_k=k_in_stage1, kpack_bytes=kpack_bytes_fp8, elem_bytes=elem_bytes_fp8
            )
            layout_b_w1 = b_layout_w1.layout_b

            # X LDS layout
            shape_lds_s1 = fx.make_shape(tile_m, tile_k)
            stride_lds_s1 = fx.make_stride(lds_stride_stage1, 1)
            layout_lds_s1 = fx.make_layout(shape_lds_s1, stride_lds_s1)

            # --- Phase 2 layouts ---
            # W2 preshuffle: c_n = E * model_dim, c_k = inter_dim
            c_n_w2 = arith.index(experts * model_dim)
            k_in_stage2 = arith.index(inter_dim)
            kpack_bytes_w2 = 16
            b_layout_w2 = make_preshuffle_b_layout(
                arith, c_n=c_n_w2, c_k=k_in_stage2, kpack_bytes=kpack_bytes_w2, elem_bytes=elem_bytes_w2
            )
            layout_b_w2 = b_layout_w2.layout_b

            # Intermediate LDS layout (bf16, [tile_m, inter_dim])
            lds_stride_stage2 = inter_dim
            shape_lds_s2 = fx.make_shape(tile_m, inter_dim)
            stride_lds_s2 = fx.make_stride(lds_stride_stage2, 1)
            layout_lds_s2 = fx.make_layout(shape_lds_s2, stride_lds_s2)
            layout_lds_s2_fp8 = fx.make_layout(
                fx.make_shape(tile_m, inter_dim), fx.make_stride(inter_dim, 1)
            )

            tx = gpu.thread_id("x")
            by = gpu.block_id("x")  # tile along model_dim (Phase 2 output N)
            bx = gpu.block_id("y")  # tile along sorted M (expert block)

            bx_m = bx * fx.Index(tile_m)

            # Early-exit: check num_valid_ids
            numids_rsrc = buffer_ops.create_buffer_resource(
                arg_num_valid_ids, max_size=False, num_records_bytes=fx.Index(4),
            )
            num_valid_i32 = buffer_ops.buffer_load(numids_rsrc, fx.Index(0), vec_width=1, dtype=T.i32)
            bx_m_i32 = arith.index_cast(T.i32, bx_m)
            blk_valid = arith.cmpi(arith.CmpIPredicate.ult, bx_m_i32, num_valid_i32)

            k_blocks16_s1 = arith.index(tile_k_bytes_fp8 // 16)
            layout_tx_wave_lane = fx.make_layout((num_waves, 64), stride=(64, 1))
            layout_lane16 = fx.make_layout((4, 16), stride=(16, 1))

            _if_blk = scf.IfOp(blk_valid)
            with _if_then(_if_blk):
                base_ptr = allocator.get_base()
                lds_x_ptr = SmemPtr(
                    base_ptr, lds_alloc_offset, T.f8, shape=(lds_total_elems,),
                )
                lds_x = lds_x_ptr.get()
                # In B2B-N-loop mode, CShuffle needs a separate LDS slice so
                # each output tile can reuse the stage1 intermediate.
                lds_intermediate = SmemPtr(
                    base_ptr, lds_x_ptr.byte_offset, T.bf16,
                    shape=(tile_m * inter_dim,),
                ).get()
                lds_out = SmemPtr(
                    base_ptr, lds_x_ptr.byte_offset + lds_out_offset_bytes, out_elem,
                    shape=(tile_m * tile_n_out,),
                ).get()
                if const_expr(stage2_use_fp8_w2):
                    lds_a2_q_ptr = SmemPtr(
                        base_ptr, lds_x_ptr.byte_offset + lds_a2_fp8_offset_bytes, T.f8,
                        shape=(tile_m * inter_dim,),
                    )
                    lds_a2_q = lds_a2_q_ptr.get()
                    lds_a2_q_i8 = SmemPtr(
                        base_ptr, lds_x_ptr.byte_offset + lds_a2_fp8_offset_bytes, T.i8,
                        shape=(tile_m * inter_dim,),
                    )
                    lds_a2_q_i8.get()
                    lds_a2_scale = SmemPtr(
                        base_ptr, lds_x_ptr.byte_offset + lds_scale_offset_bytes, T.f32,
                        shape=(tile_m,),
                    )
                    lds_a2_scale.get()
                    lds_red = SmemPtr(
                        base_ptr, lds_x_ptr.byte_offset + lds_red_offset_bytes, T.f32,
                        shape=(red_slots,),
                    )
                    lds_red.get()

                # Buffer resources
                c_topk = fx.Index(topk)
                x_nbytes_idx = tokens_in * model_dim_in * fx.Index(elem_bytes_fp8)
                x_rsrc = buffer_ops.create_buffer_resource(arg_x, max_size=False, num_records_bytes=x_nbytes_idx)
                w1_rsrc = buffer_ops.create_buffer_resource(arg_w1, max_size=False)
                w2_rsrc = buffer_ops.create_buffer_resource(arg_w2, max_size=False)

                out_nbytes_idx = tokens_in * model_dim_in * fx.Index(2)
                out_rsrc = buffer_ops.create_buffer_resource(arg_out, max_size=False, num_records_bytes=out_nbytes_idx)

                sx_nbytes_idx = tokens_in * fx.Index(4)
                sx_rsrc = buffer_ops.create_buffer_resource(arg_scale_x, max_size=False, num_records_bytes=sx_nbytes_idx)
                sw1_rsrc = buffer_ops.create_buffer_resource(arg_scale_w1, max_size=False)
                sw2_rsrc = -1
                if const_expr(stage2_use_fp8_w2):
                    sw2_rsrc = buffer_ops.create_buffer_resource(arg_scale_w2, max_size=False)

                sorted_nbytes_idx = size_expert_ids_in * fx.Index(tile_m) * fx.Index(4)
                sorted_rsrc = buffer_ops.create_buffer_resource(
                    arg_sorted_token_ids, max_size=False, num_records_bytes=sorted_nbytes_idx
                )
                sorted_w_rsrc = buffer_ops.create_buffer_resource(
                    arg_sorted_weights, max_size=False, num_records_bytes=sorted_nbytes_idx
                )
                expert_rsrc = buffer_ops.create_buffer_resource(
                    arg_expert_ids, max_size=False,
                    num_records_bytes=(size_expert_ids_in * fx.Index(4)),
                )

                # Expert id for this M tile
                expert_i32 = buffer_ops.buffer_load(expert_rsrc, bx, vec_width=1, dtype=T.i32)
                expert_idx = arith.index_cast(T.index, expert_i32)

                # W1 expert offset: expert * 2*inter_dim (in N dimension)
                inter2_idx = arith.index(2 * inter_dim)
                expert_off_w1 = expert_idx * inter2_idx

                # W2 expert offset: expert * model_dim (in N dimension)
                model_dim_idx = arith.index(model_dim)
                expert_off_w2 = expert_idx * model_dim_idx

                # ---- X gmem->reg prefetch setup ----
                if const_expr(bytes_per_thread_x % 16 == 0):
                    x_load_bytes = 16
                elif const_expr(bytes_per_thread_x % 8 == 0):
                    x_load_bytes = 8
                elif const_expr(bytes_per_thread_x % 4 == 0):
                    x_load_bytes = 4
                else:
                    raise ValueError(
                        f"bytes_per_thread_x ({bytes_per_thread_x}) must be divisible by 4"
                    )
                num_x_loads = bytes_per_thread_x // x_load_bytes
                chunk_i32 = x_load_bytes // 4

                c_k_div4 = model_dim_in // fx.Index(4)  # fp8: elem_bytes=1, so k/4 = dwords
                tile_k_dwords = tile_k // 4
                layout_x_tile_div4 = fx.make_layout((tile_m, tile_k_dwords), stride=(tile_k_dwords, 1))
                c_chunk_i32 = fx.Index(chunk_i32)
                tx_i32_base = tx * c_chunk_i32
                mask24 = fx.Int32(0xFFFFFF)
                tokens_i32 = arith.index_cast(T.i32, tokens_in)
                topk_i32 = fx.Int32(topk)

                def x_tile_chunk_coord_i32(i: int):
                    return tile_chunk_coord_i32(
                        arith, tx_i32_base=tx_i32_base, i=i,
                        total_threads=total_threads,
                        layout_tile_div4=layout_x_tile_div4,
                        chunk_i32=chunk_i32,
                    )

                # Decode sorted tokens once
                x_row_base_div4 = []
                x_col_local_i32 = []
                x_row_local = []
                for i in range_constexpr(num_x_loads):
                    row_local, col_local_i32 = x_tile_chunk_coord_i32(i)
                    x_row_local.append(row_local)
                    x_col_local_i32.append(col_local_i32)

                    sorted_row_i = bx_m + row_local
                    fused_i = buffer_ops.buffer_load(sorted_rsrc, sorted_row_i, vec_width=1, dtype=T.i32)
                    t_raw = fused_i & mask24
                    t_valid_i32 = arith.cmpi(arith.CmpIPredicate.ult, t_raw, tokens_i32)
                    t_idx = arith.index_cast(T.index, t_raw)
                    t_safe = t_valid_i32.select(t_idx, fx.Index(0))
                    x_row_base_div4.append(t_safe * c_k_div4)

                def load_x(idx_i32):
                    if const_expr(x_load_bytes == 16):
                        idx_elem = idx_i32  # fp8: elem_bytes=1, idx is already in element units
                        return buffer_copy_gmem16_dwordx4(
                            buffer_ops, vector, elem_type=x_elem,
                            idx_i32=idx_elem, rsrc=x_rsrc,
                            vec_elems=vec16_elems_fp8, elem_bytes=elem_bytes_fp8,
                        )
                    if const_expr(x_load_bytes == 8):
                        return buffer_ops.buffer_load(x_rsrc, idx_i32, vec_width=2, dtype=T.i32)
                    return buffer_ops.buffer_load(x_rsrc, idx_i32, vec_width=1, dtype=T.i32)

                def load_x_tile(base_k):
                    base_k_div4 = base_k // fx.Index(4)
                    parts = []
                    for i in range_constexpr(num_x_loads):
                        idx_i32 = x_row_base_div4[i] + base_k_div4 + x_col_local_i32[i]
                        x_vec = load_x(idx_i32)
                        if const_expr(x_load_bytes == 16):
                            parts.append(vector.bitcast(T.i32x4, x_vec))
                        elif const_expr(x_load_bytes == 8):
                            parts.append(x_vec)
                        else:
                            parts.append(x_vec)
                    return parts

                # tx -> wave/lane
                coord_wl = fx.idx2crd(tx, layout_tx_wave_lane)
                wave_id = fx.get(coord_wl, 0)
                lane_id = fx.get(coord_wl, 1)
                coord_l16 = fx.idx2crd(lane_id, layout_lane16)
                lane_div_16 = fx.get(coord_l16, 0)
                lane_mod_16 = fx.get(coord_l16, 1)

                row_a_lds = lane_mod_16
                a_kpack_elems_fp8 = 16  # 16 bytes / 1 byte
                col_offset_base_fp8 = lane_div_16 * arith.index(a_kpack_elems_fp8)
                # col_offset_base_bytes for fp8 = col_offset_base (since elem=1B)
                col_offset_base_bytes_s1 = col_offset_base_fp8

                # Phase 1 N-tiling: covers ALL 2*inter_dim columns
                # n_per_wave_s1 = tile_n_stage1 / 4 = 128/4 = 32
                # num_acc_n_s1 = 32/16 = 2
                n_per_wave_s1 = tile_n_stage1 // num_waves  # 32
                num_acc_n_s1 = n_per_wave_s1 // 16  # 2
                c_n_per_wave_s1 = fx.Index(n_per_wave_s1)
                wave_n_id = wave_id % fx.Index(num_waves)
                n_tile_base_s1 = wave_n_id * c_n_per_wave_s1

                # Precompute n_blk/n_intra for gate and up
                n_intra_gate = []
                n_blk_gate = []
                n_intra_up = []
                n_blk_up = []
                col_g_list_s1 = []
                inter_idx = fx.Index(inter_dim)
                c_n0_s1 = experts * (2 * inter_dim) // 16
                layout_n_blk_intra_s1 = fx.make_layout((c_n0_s1, 16), stride=(16, 1))

                # by_n_s1 = 0 (Phase 1 always processes full inter_dim, no grid tiling)
                by_n_s1 = fx.Index(0)
                for ni in range_constexpr(num_acc_n_s1):
                    offset = arith.index(ni * 16)
                    col_g = by_n_s1 + n_tile_base_s1 + offset + lane_mod_16
                    col_g_list_s1.append(col_g)

                    row_gate = expert_off_w1 + col_g
                    row_up = row_gate + inter_idx

                    coord_gate = fx.idx2crd(row_gate, layout_n_blk_intra_s1)
                    n_blk_gate.append(fx.get(coord_gate, 0))
                    n_intra_gate.append(fx.get(coord_gate, 1))

                    coord_up = fx.idx2crd(row_up, layout_n_blk_intra_s1)
                    n_blk_up.append(fx.get(coord_up, 0))
                    n_intra_up.append(fx.get(coord_up, 1))

                m_repeat = tile_m // 16
                k_unroll_s1 = tile_k_bytes_fp8 // 64  # 128/64 = 2

                # --- Phase 1 B Load ---
                def load_b_pack_w1(base_k, ki_step, ni, blk_list, intra_list):
                    return load_b_pack_k32(
                        buffer_ops, arith, vector,
                        arg_b=arg_w1, b_rsrc=w1_rsrc, layout_b=layout_b_w1,
                        base_k=base_k, ki_step=ki_step,
                        n_blk=blk_list[ni], n_intra=intra_list[ni],
                        lane_div_16=lane_div_16, elem_type=w1_elem,
                        kpack_bytes=kpack_bytes_fp8, elem_bytes=elem_bytes_fp8,
                        unpack_int4=False,
                        **({"cache_modifier": b_cache_modifier_w1} if b_cache_modifier_w1 != 0 else {}),
                    )

                def load_b_tile_w1(base_k, blk_list, intra_list):
                    b_tile = []
                    for ku in range_constexpr(k_unroll_s1):
                        packs0 = []
                        packs1 = []
                        for ni in range_constexpr(num_acc_n_s1):
                            ki0 = (ku * 2) + 0
                            ki1 = (ku * 2) + 1
                            b0 = load_b_pack_w1(base_k, ki0, ni, blk_list, intra_list)
                            b1 = load_b_pack_w1(base_k, ki1, ni, blk_list, intra_list)
                            packs0.append(b0)
                            packs1.append(b1)
                        b_tile.append((packs0, packs1))
                    return b_tile

                # --- LDS helpers ---
                def store_x_tile_to_lds(vec_x_in_parts, lds_base):
                    for i in range_constexpr(num_x_loads):
                        row_local = x_row_local[i]
                        col_local_i32 = x_col_local_i32[i]
                        if const_expr(x_load_bytes == 16):
                            lds_store_16b_xor16(
                                arith, vector,
                                lds_memref=lds_x,
                                vec16_ty=vec16_x,
                                layout_lds=layout_lds_s1,
                                row_local=row_local,
                                col_local_i32=col_local_i32,
                                tx_c4=fx.Index(4),
                                k_blocks16=k_blocks16_s1,
                                lds_base=lds_base,
                                vec_part_i32x4=vec_x_in_parts[i],
                                elem_bytes=elem_bytes_fp8,
                            )
                        elif const_expr(x_load_bytes == 8):
                            lds_store_8b_xor16(
                                arith, vector,
                                lds_memref=lds_x,
                                vec8_ty=vec8_x,
                                layout_lds=layout_lds_s1,
                                row_local=row_local,
                                col_local_i32=col_local_i32,
                                tx_c4=fx.Index(4),
                                k_blocks16=k_blocks16_s1,
                                lds_base=lds_base,
                                vec_part_i32x2=vec_x_in_parts[i],
                                elem_bytes=elem_bytes_fp8,
                            )
                        else:
                            lds_store_4b_xor16(
                                arith, vector,
                                lds_memref=lds_x,
                                vec4_ty=vec4_x,
                                layout_lds=layout_lds_s1,
                                row_local=row_local,
                                col_local_i32=col_local_i32,
                                tx_c4=fx.Index(4),
                                k_blocks16=k_blocks16_s1,
                                lds_base=lds_base,
                                vec_part_i32x1=vec_x_in_parts[i],
                                elem_bytes=elem_bytes_fp8,
                            )

                def lds_load_packs_k64_s1(curr_row_a_lds, col_base_bytes, lds_base):
                    col_base_swz_bytes = swizzle_xor16(curr_row_a_lds, col_base_bytes, k_blocks16_s1)
                    col_base_swz = col_base_swz_bytes  # fp8: elem=1B
                    idx_a16 = crd2idx((curr_row_a_lds, col_base_swz), layout_lds_s1)
                    idx_a16 = idx_a16 + lds_base
                    loaded_a16 = vector.load_op(vec16_x, lds_x, [idx_a16])
                    a_i64x2 = vector.bitcast(T.i64x2, loaded_a16)
                    a0 = vector.extract(a_i64x2, static_position=[0], dynamic_position=[])
                    a1 = vector.extract(a_i64x2, static_position=[1], dynamic_position=[])
                    return a0, a1

                # --- Phase 1 MFMA ---
                def mfma_k64_fp8(acc0, a0, a1, b0, b1):
                    acc1 = rocdl.mfma_f32_16x16x32_fp8_fp8(T.f32x4, [a0, b0, acc0, 0, 0, 0])
                    return rocdl.mfma_f32_16x16x32_fp8_fp8(T.f32x4, [a1, b1, acc1, 0, 0, 0])

                def compute_tile_s1(gate_list, up_list, b_gate_tile, b_up_tile, lds_base, *, a0_prefetch=None):
                    for ku in range_constexpr(k_unroll_s1):
                        b_gate_packs0, b_gate_packs1 = b_gate_tile[ku]
                        b_up_packs0, b_up_packs1 = b_up_tile[ku]
                        ki64 = arith.index(ku * 64)
                        col_base = col_offset_base_bytes_s1 + ki64

                        for mi in range_constexpr(m_repeat):
                            mi_val = arith.index(mi * 16)
                            curr_row_a_lds = row_a_lds + mi_val

                            if const_expr((a0_prefetch is not None) and (ku == 0) and (mi == 0)):
                                a0, a1 = a0_prefetch
                            else:
                                a0, a1 = lds_load_packs_k64_s1(curr_row_a_lds, col_base, lds_base)

                            for ni in range_constexpr(num_acc_n_s1):
                                acc_idx = mi * num_acc_n_s1 + ni
                                gate_list[acc_idx] = mfma_k64_fp8(
                                    gate_list[acc_idx], a0, a1,
                                    b_gate_packs0[ni], b_gate_packs1[ni],
                                )
                                up_list[acc_idx] = mfma_k64_fp8(
                                    up_list[acc_idx], a0, a1,
                                    b_up_packs0[ni], b_up_packs1[ni],
                                )
                    return gate_list, up_list

                # ============ PHASE 1: K-loop over model_dim ============
                acc_gate = [acc_init] * (m_repeat * num_acc_n_s1)
                acc_up = [acc_init] * (m_repeat * num_acc_n_s1)

                lds_tile_elems_s1 = arith.index(tile_m * lds_stride_stage1)
                lds_base_cur = fx.Index(0)
                lds_base_nxt = lds_tile_elems_s1

                rocdl.sched_barrier(0)

                # Prologue: prefetch tile0
                k0 = fx.Index(0)
                x_regs0 = load_x_tile(k0)
                b_gate_cur = load_b_tile_w1(k0, n_blk_gate, n_intra_gate)
                b_up_cur = load_b_tile_w1(k0, n_blk_up, n_intra_up)
                store_x_tile_to_lds(x_regs0, lds_base_cur)
                gpu.barrier()

                lds_base_pong = lds_base_cur
                lds_base_ping = lds_base_nxt

                a0_prefetch_pong = lds_load_packs_k64_s1(row_a_lds, col_offset_base_bytes_s1, lds_base_pong)

                # Ping-pong main loop
                c_tile_k = arith.index(tile_k)
                total_tiles_s1 = model_dim // tile_k
                pair_iters_s1 = max((total_tiles_s1 - 2) // 2, 0)

                _n_acc_s1 = m_repeat * num_acc_n_s1
                _vals_per_b_tile_s1 = k_unroll_s1 * 2 * num_acc_n_s1

                def _flatten_b_s1(b_tile):
                    flat = []
                    for ku_entry in b_tile:
                        flat.extend(ku_entry[0])
                        flat.extend(ku_entry[1])
                    return flat

                def _unflatten_b_s1(vals):
                    b_tile, idx = [], 0
                    for _ in range_constexpr(k_unroll_s1):
                        packs_even = list(vals[idx:idx + num_acc_n_s1])
                        idx += num_acc_n_s1
                        packs_odd = list(vals[idx:idx + num_acc_n_s1])
                        idx += num_acc_n_s1
                        b_tile.append((packs_even, packs_odd))
                    return b_tile

                init_state = (
                    list(acc_gate) + list(acc_up)
                    + _flatten_b_s1(b_gate_cur) + _flatten_b_s1(b_up_cur)
                    + list(a0_prefetch_pong)
                )

                _p_bg = 2 * _n_acc_s1
                _p_bu = _p_bg + _vals_per_b_tile_s1
                _p_a0 = _p_bu + _vals_per_b_tile_s1

                for pair_iv, state in range(0, pair_iters_s1, 1, init=init_state):
                    _ag = list(state[:_n_acc_s1])
                    _au = list(state[_n_acc_s1:_p_bg])
                    _bg = _unflatten_b_s1(list(state[_p_bg:_p_bu]))
                    _bu = _unflatten_b_s1(list(state[_p_bu:_p_a0]))
                    _a0pf = (state[_p_a0], state[_p_a0 + 1])

                    k_iv = pair_iv * (c_tile_k + c_tile_k)

                    # Stage 0: prefetch ping, compute pong
                    next_k1 = k_iv + c_tile_k
                    x_regs_ping = load_x_tile(next_k1)
                    _bg_ping = load_b_tile_w1(next_k1, n_blk_gate, n_intra_gate)
                    _bu_ping = load_b_tile_w1(next_k1, n_blk_up, n_intra_up)

                    _ag, _au = compute_tile_s1(_ag, _au, _bg, _bu, lds_base_pong, a0_prefetch=_a0pf)
                    store_x_tile_to_lds(x_regs_ping, lds_base_ping)
                    rocdl.sched_barrier(0)
                    gpu.barrier()

                    _a0pf_ping = lds_load_packs_k64_s1(row_a_lds, col_offset_base_bytes_s1, lds_base_ping)

                    # Stage 1: prefetch pong, compute ping
                    next_k2 = k_iv + c_tile_k + c_tile_k
                    x_regs_pong = load_x_tile(next_k2)
                    _bg_next = load_b_tile_w1(next_k2, n_blk_gate, n_intra_gate)
                    _bu_next = load_b_tile_w1(next_k2, n_blk_up, n_intra_up)

                    _ag, _au = compute_tile_s1(_ag, _au, _bg_ping, _bu_ping, lds_base_ping, a0_prefetch=_a0pf_ping)
                    store_x_tile_to_lds(x_regs_pong, lds_base_pong)
                    rocdl.sched_barrier(0)
                    gpu.barrier()

                    _a0pf_new = lds_load_packs_k64_s1(row_a_lds, col_offset_base_bytes_s1, lds_base_pong)

                    loop_results = yield (
                        list(_ag) + list(_au)
                        + _flatten_b_s1(_bg_next) + _flatten_b_s1(_bu_next)
                        + list(_a0pf_new)
                    )

                # After loop: extract final state
                SmemPtr._view_cache = None
                if const_expr(pair_iters_s1 > 0):
                    acc_gate = list(loop_results[:_n_acc_s1])
                    acc_up = list(loop_results[_n_acc_s1:_p_bg])
                    b_gate_cur = _unflatten_b_s1(list(loop_results[_p_bg:_p_bu]))
                    b_up_cur = _unflatten_b_s1(list(loop_results[_p_bu:_p_a0]))
                    a0_prefetch_pong = (loop_results[_p_a0], loop_results[_p_a0 + 1])

                # Tail: penultimate tile
                k_tail1 = arith.index(model_dim - tile_k)
                x_regs_ping = load_x_tile(k_tail1)
                b_gate_ping = load_b_tile_w1(k_tail1, n_blk_gate, n_intra_gate)
                b_up_ping = load_b_tile_w1(k_tail1, n_blk_up, n_intra_up)

                acc_gate, acc_up = compute_tile_s1(
                    acc_gate, acc_up, b_gate_cur, b_up_cur, lds_base_pong,
                    a0_prefetch=a0_prefetch_pong,
                )
                store_x_tile_to_lds(x_regs_ping, lds_base_ping)
                rocdl.sched_barrier(0)
                gpu.barrier()

                a0_prefetch_ping = lds_load_packs_k64_s1(row_a_lds, col_offset_base_bytes_s1, lds_base_ping)

                # Final tile
                acc_gate, acc_up = compute_tile_s1(
                    acc_gate, acc_up, b_gate_ping, b_up_ping, lds_base_ping,
                    a0_prefetch=a0_prefetch_ping,
                )

                # ============ PHASE 1.5: SiLU + Store intermediate to LDS ============
                # intermediate[row, col] = silu(gate * sx * sw_gate) * (up * sx * sw_up)
                # Store as bf16 to LDS with XOR16 swizzle for Phase 2 reads.
                #
                # MFMA 16x16 output layout (CDNA3):
                #   Row = lane_div_16*4 + ii (ii is element index within f32x4)
                #   Col = lane_mod_16 (within the ni-th 16-column block)
                # So acc[mi*num_acc_n+ni][ii] corresponds to:
                #   row_in_tile = mi*16 + lane_div_16*4 + ii
                #   col = n_tile_base_s1 + ni*16 + lane_mod_16

                def silu(x):
                    t = x * (-1.4426950408889634)
                    emu = rocdl.exp2(T.f32, t)
                    den = 1.0 + emu
                    sig = rocdl.rcp(T.f32, den)
                    return x * sig

                mask24_i32 = fx.Int32(0xFFFFFF)
                tokens_i32_v = tokens_i32

                # k_blocks16 for intermediate LDS. The stage1 result is staged
                # as BF16 first; the FP8-W2 B2B path requantizes it into a
                # separate FP8 LDS tile for stage2.
                k_blocks16_s2_bf16 = arith.index(tile_k_bytes_bf16 // 16)
                k_blocks16_s2_fp8 = arith.index(tile_k_bytes_fp8 // 16)
                lds_stride_s2_idx = arith.index(lds_stride_stage2)

                # Iterate using default_epilog pattern: for mi, for ii → row
                lane_div_16_mul4 = lane_div_16 * fx.Index(4)
                ii_idx_list = [fx.Index(ii) for ii in range(4)]

                for mi in range_constexpr(m_repeat):
                    mi_base = arith.index(mi * 16)
                    for ii in range_constexpr(4):
                        row_off = lane_div_16_mul4 + ii_idx_list[ii]
                        row_in_tile = mi_base + row_off
                        sorted_row = bx_m + row_in_tile

                        # Decode sorted token info for scale_x
                        fused2 = buffer_ops.buffer_load(sorted_rsrc, sorted_row, vec_width=1, dtype=T.i32)
                        t2 = fused2 & mask24_i32
                        t_valid = arith.cmpi(arith.CmpIPredicate.ult, t2, tokens_i32_v)
                        sx = arith.select(
                            t_valid,
                            buffer_ops.buffer_load(sx_rsrc, t2, vec_width=1, dtype=T.f32),
                            fx.Float32(0.0),
                        )

                        for ni in range_constexpr(num_acc_n_s1):
                            acc_idx = mi * num_acc_n_s1 + ni
                            # W1 scale column: col_g = n_tile_base_s1 + ni*16 + lane_mod_16
                            col_g = col_g_list_s1[ni]
                            row_gate_scale = expert_off_w1 + col_g
                            row_up_scale = row_gate_scale + inter_idx
                            sw_gate = buffer_ops.buffer_load(sw1_rsrc, row_gate_scale, vec_width=1, dtype=T.f32)
                            sw_up = buffer_ops.buffer_load(sw1_rsrc, row_up_scale, vec_width=1, dtype=T.f32)

                            vg = vector.extract(acc_gate[acc_idx], static_position=[ii], dynamic_position=[])
                            vu = vector.extract(acc_up[acc_idx], static_position=[ii], dynamic_position=[])

                            vg = vg * sx * sw_gate
                            vu = vu * sx * sw_up
                            y = silu(vg) * vu
                            y_bf16 = arith.trunc_f(T.bf16, y)

                            # LDS address: intermediate[row_in_tile, col]
                            # col = n_tile_base_s1 + ni*16 + lane_mod_16
                            col_elem = n_tile_base_s1 + arith.index(ni * 16) + lane_mod_16
                            col_bytes = col_elem * fx.Index(2)  # bf16 = 2 bytes
                            # Apply XOR16 swizzle
                            col_swz_bytes = swizzle_xor16(row_in_tile, col_bytes, k_blocks16_s2_bf16)
                            col_swz = col_swz_bytes // fx.Index(2)  # back to element index
                            lds_idx = crd2idx((row_in_tile, col_swz), layout_lds_s2)

                            v1 = vector.from_elements(T.vec(1, T.bf16), [y_bf16])
                            vector.store(v1, lds_intermediate, [lds_idx], alignment=2)

                # Barrier before Phase 2 reads from LDS
                gpu.barrier()

                if const_expr(stage2_use_fp8_w2):
                    tx_i32_v = arith.index_cast(T.i32, tx)
                    lane_i32 = tx_i32_v % arith.constant(64, type=T.i32)
                    width_i32 = arith.constant(64, type=T.i32)
                    abs_mask_i32 = fx.Int32(0x7FFFFFFF)

                    def wave_reduce_max_f32(val):
                        w = val
                        for shift in [32, 16, 8, 4, 2, 1]:
                            peer = w.shuffle_xor(arith.constant(shift, type=T.i32), width_i32)
                            w = w.maximumf(peer)
                        return w

                    abs_mask_vec4 = fx.Vector.filled(4, 0x7FFFFFFF, fx.Int32)
                    for row_group in range_constexpr(tile_m // num_waves):
                        row_q_idx = fx.Index(row_group * num_waves) + wave_id
                        col_q = lane_id * fx.Index(4)
                        col_bf16_bytes = col_q * fx.Index(2)
                        col_bf16_swz_bytes = swizzle_xor16(
                            row_q_idx, col_bf16_bytes, k_blocks16_s2_bf16
                        )
                        col_bf16_swz = col_bf16_swz_bytes // fx.Index(2)
                        idx_bf16 = crd2idx((row_q_idx, col_bf16_swz), layout_lds_s2)
                        loaded_bf16 = vector.load_op(
                            T.vec(4, T.bf16), lds_intermediate, [idx_bf16]
                        )
                        vals_f32 = fx.Vector(loaded_bf16).to(fx.Float32)
                        vals_abs = (vals_f32.bitcast(fx.Int32) & abs_mask_vec4).bitcast(fx.Float32)
                        lane_max = vals_abs.reduce(ReductionOp.MAX).ir_value()
                        row_max = wave_reduce_max_f32(lane_max)
                        scale = row_max / arith.constant(240.0, type=T.f32)
                        safe_nonzero_scale = scale.maximumf(
                            arith.constant(_FP32_MIN_NORMAL, type=T.f32)
                        )
                        final_scale = arith.select(
                            scale == arith.constant(0.0, type=T.f32),
                            arith.constant(1.0, type=T.f32),
                            safe_nonzero_scale,
                        )
                        if lane_i32 == fx.Int32(0):
                            scale_out = final_scale
                            if const_expr(fold_route_weight_into_a2_scale):
                                sorted_row_q = bx_m + row_q_idx
                                fused_q = buffer_ops.buffer_load(
                                    sorted_rsrc, sorted_row_q, vec_width=1, dtype=T.i32
                                )
                                t_q = fused_q & mask24_i32
                                s_q = fused_q >> 24
                                sorted_row_q_i32 = arith.index_cast(T.i32, sorted_row_q)
                                row_q_ok = arith.cmpi(
                                    arith.CmpIPredicate.ult, sorted_row_q_i32, num_valid_i32
                                )
                                t_q_ok = arith.cmpi(arith.CmpIPredicate.ult, t_q, tokens_i32_v)
                                s_q_ok = arith.cmpi(arith.CmpIPredicate.ult, s_q, topk_i32)
                                q_ok = row_q_ok & t_q_ok & s_q_ok
                                tw_q = arith.select(
                                    q_ok,
                                    buffer_ops.buffer_load(
                                        sorted_w_rsrc, sorted_row_q, vec_width=1, dtype=T.f32
                                    ),
                                    fx.Float32(0.0),
                                )
                                scale_out = scale_out * tw_q
                            SmemPtr.store(lds_a2_scale, scale_out, [row_q_idx])
                        inv_scale = arith.constant(1.0, type=T.f32) / final_scale
                        q_vals = vals_f32 * inv_scale
                        lo = rocdl.cvt_pk_fp8_f32(
                            T.i32, q_vals[0], q_vals[1], fx.Int32(0), False
                        )
                        packed = rocdl.cvt_pk_fp8_f32(
                            T.i32, q_vals[2], q_vals[3], lo, True
                        )
                        for qj in range_constexpr(4):
                            q_shift = arith.constant(qj * 8, type=T.i32)
                            q_byte = arith.trunci(T.i8, packed >> q_shift)
                            col_fp8 = col_q + fx.Index(qj)
                            col_fp8_swz = swizzle_xor16(row_q_idx, col_fp8, k_blocks16_s2_fp8)
                            idx_fp8 = crd2idx((row_q_idx, col_fp8_swz), layout_lds_s2_fp8)
                            SmemPtr.store(lds_a2_q_i8, q_byte, [idx_fp8])
                    gpu.barrier()

                def _emit_output_tile(by_tile):
                    # ============ PHASE 2: intermediate × W2 (single K-pass) ============
                    # Output N is either the grid-x tile or a CTA-local B2B loop tile.
                    by_n_s2 = by_tile * fx.Index(tile_n_out)
                    n_per_wave_s2 = tile_n_out // num_waves  # 128/4 = 32
                    num_acc_n_s2 = n_per_wave_s2 // 16  # 2
                    c_n_per_wave_s2 = fx.Index(n_per_wave_s2)
                    n_tile_base_s2 = wave_n_id * c_n_per_wave_s2

                    # Precompute W2 B-tile coordinates
                    n_intra_list_w2 = []
                    n_blk_list_w2 = []
                    col_g_list_s2 = []
                    c_n0_s2 = experts * model_dim // 16
                    layout_n_blk_intra_s2 = fx.make_layout((c_n0_s2, 16), stride=(16, 1))

                    for ni in range_constexpr(num_acc_n_s2):
                        offset = arith.index(ni * 16)
                        col_g = by_n_s2 + n_tile_base_s2 + offset + lane_mod_16
                        col_g_list_s2.append(col_g)

                        row_w = expert_off_w2 + col_g
                        coord_w = fx.idx2crd(row_w, layout_n_blk_intra_s2)
                        n_blk_list_w2.append(fx.get(coord_w, 0))
                        n_intra_list_w2.append(fx.get(coord_w, 1))

                    k_unroll_s2 = tile_k_bytes_w2 // 64

                    # Load W2 tile (single K-pass, all k_unroll steps at once)
                    def load_b_pack_w2(base_k, ki_step, ni):
                        return load_b_pack_k32(
                            buffer_ops, arith, vector,
                            arg_b=arg_w2, b_rsrc=w2_rsrc, layout_b=layout_b_w2,
                            base_k=base_k, ki_step=ki_step,
                            n_blk=n_blk_list_w2[ni], n_intra=n_intra_list_w2[ni],
                            lane_div_16=lane_div_16, elem_type=w2_elem,
                            kpack_bytes=kpack_bytes_w2, elem_bytes=elem_bytes_w2,
                            unpack_int4=False,
                            **({"cache_modifier": b_cache_modifier_w2} if b_cache_modifier_w2 != 0 else {}),
                        )

                    # Load entire W2 B-tile
                    b_w2_tile = []
                    base_k_s2 = fx.Index(0)
                    for ku in range_constexpr(k_unroll_s2):
                        packs0 = []
                        packs1 = []
                        for ni in range_constexpr(num_acc_n_s2):
                            ki0 = (ku * 2) + 0
                            ki1 = (ku * 2) + 1
                            b0 = load_b_pack_w2(base_k_s2, ki0, ni)
                            b1 = load_b_pack_w2(base_k_s2, ki1, ni)
                            packs0.append(b0)
                            packs1.append(b1)
                        b_w2_tile.append((packs0, packs1))

                    if const_expr(stage2_use_fp8_w2):
                        a_kpack_elems_fp8_s2 = 16
                        col_offset_base_fp8_s2 = lane_div_16 * arith.index(a_kpack_elems_fp8_s2)
                        vec16_a2_fp8 = T.vec(16, T.f8)

                        def lds_load_packs_k64_s2(curr_row_a_lds, col_base_bytes, lds_base_s2):
                            col_base_swz = swizzle_xor16(
                                curr_row_a_lds, col_base_bytes, k_blocks16_s2_fp8
                            )
                            idx_a16 = crd2idx((curr_row_a_lds, col_base_swz), layout_lds_s2_fp8)
                            idx_a16 = idx_a16 + lds_base_s2
                            loaded_a16 = vector.load_op(vec16_a2_fp8, lds_a2_q, [idx_a16])
                            a_i64x2 = vector.bitcast(T.i64x2, loaded_a16)
                            a0 = vector.extract(a_i64x2, static_position=[0], dynamic_position=[])
                            a1 = vector.extract(a_i64x2, static_position=[1], dynamic_position=[])
                            return a0, a1

                        def mfma_k64_s2(acc0, a0, a1, b0, b1):
                            return mfma_k64_fp8(acc0, a0, a1, b0, b1)

                        col_offset_base_bytes_s2 = col_offset_base_fp8_s2
                    else:
                        # Phase 2 A-load from LDS (bf16 intermediate)
                        # a_kpack_elems for bf16 = 16/2 = 8
                        a_kpack_elems_bf16 = 8
                        col_offset_base_bf16 = lane_div_16 * arith.index(a_kpack_elems_bf16)
                        col_offset_base_bytes_s2 = col_offset_base_bf16 * fx.Index(2)

                        vec16_bf16 = T.vec(8, T.bf16)  # 16B = 8 bf16

                        def lds_load_packs_k64_s2(curr_row_a_lds, col_base_bytes, lds_base_s2):
                            col_base_swz_bytes = swizzle_xor16(
                                curr_row_a_lds, col_base_bytes, k_blocks16_s2_bf16
                            )
                            col_base_swz = col_base_swz_bytes // fx.Index(2)
                            idx_a16 = crd2idx((curr_row_a_lds, col_base_swz), layout_lds_s2)
                            idx_a16 = idx_a16 + lds_base_s2
                            loaded_a16 = vector.load_op(vec16_bf16, lds_intermediate, [idx_a16])
                            a_i64x2 = vector.bitcast(T.i64x2, loaded_a16)
                            a0 = vector.extract(a_i64x2, static_position=[0], dynamic_position=[])
                            a1 = vector.extract(a_i64x2, static_position=[1], dynamic_position=[])
                            return a0, a1

                        # BF16 MFMA
                        mfma_f32_bf16_k16 = getattr(rocdl, "mfma_f32_16x16x16bf16_1k", None) or getattr(
                            rocdl, "mfma_f32_16x16x16_bf16_1k", None
                        )

                        def _i64_to_v4i16(x_i64):
                            v1 = vector.from_elements(T.vec(1, T.i64), [x_i64])
                            return vector.bitcast(T.i16x4, v1)

                        def mfma_k64_s2(acc0, a0, a1, b0, b1):
                            a0v = _i64_to_v4i16(a0)
                            a1v = _i64_to_v4i16(a1)
                            b0v = _i64_to_v4i16(b0)
                            b1v = _i64_to_v4i16(b1)
                            acc1 = mfma_f32_bf16_k16(T.f32x4, [a0v, b0v, acc0, 0, 0, 0])
                            return mfma_f32_bf16_k16(T.f32x4, [a1v, b1v, acc1, 0, 0, 0])

                    # Phase 2 MFMA compute
                    acc_out = [acc_init] * (m_repeat * num_acc_n_s2)
                    lds_base_s2_val = fx.Index(0)  # intermediate starts at offset 0

                    for ku in range_constexpr(k_unroll_s2):
                        b_packs0, b_packs1 = b_w2_tile[ku]
                        ki64 = arith.index(ku * 64)
                        col_base = col_offset_base_bytes_s2 + ki64

                        for mi in range_constexpr(m_repeat):
                            mi_val = arith.index(mi * 16)
                            curr_row_a_lds = row_a_lds + mi_val
                            a0, a1 = lds_load_packs_k64_s2(curr_row_a_lds, col_base, lds_base_s2_val)

                            for ni in range_constexpr(num_acc_n_s2):
                                acc_idx = mi * num_acc_n_s2 + ni
                                acc_out[acc_idx] = mfma_k64_s2(
                                    acc_out[acc_idx], a0, a1,
                                    b_packs0[ni], b_packs1[ni],
                                )

                    # ============ PHASE 3: Epilog (CShuffle + atomic bf16 output) ============
                    gpu.barrier()  # Ensure Phase 2 LDS reads complete before using CShuffle LDS.

                    model_i32 = fx.Int32(model_dim)
                    topk_i32_v = topk_i32
                    zero_i32 = fx.Int32(0)
                    c2_i32 = fx.Int32(2)
                    mask_even_i32 = fx.Int32(0xFFFFFFFE)
                    cshuffle_k_blocks16 = arith.index((tile_n_out * elem_bytes_bf16) // 16)

                    out_base_idx = fx.Index(0)
                    if const_expr(_needs_global_atomic_bf16):
                        out_base_idx = buffer_ops.extract_base_index(arg_out)

                    def cshuffle_lds_idx(row_local, row_base_lds, col_local):
                        if const_expr(cshuffle_lds_xor):
                            col_bytes = col_local * fx.Index(elem_bytes_bf16)
                            col_swz_bytes = swizzle_xor16(
                                row_local, col_bytes, cshuffle_k_blocks16
                            )
                            return row_base_lds + (col_swz_bytes // fx.Index(elem_bytes_bf16))
                        return row_base_lds + col_local

                    sw_vals_s2 = [fx.Float32(1.0)] * num_acc_n_s2
                    if const_expr(stage2_use_fp8_w2):
                        sw_vals_s2 = []
                        for ni in range_constexpr(num_acc_n_s2):
                            row_w_idx = expert_off_w2 + col_g_list_s2[ni]
                            sw_vals_s2.append(
                                buffer_ops.buffer_load(sw2_rsrc, row_w_idx, vec_width=1, dtype=T.f32)
                            )

                    def write_row_to_lds_out(
                        *, mi: int, ii: int, row_in_tile, row,
                        row_base_lds, col_base_local, num_acc_n: int, lds_out,
                    ):
                        tw = fx.Float32(1.0)
                        if const_expr(not (stage2_use_fp8_w2 and fold_route_weight_into_a2_scale)):
                            fused2 = buffer_ops.buffer_load(sorted_rsrc, row, vec_width=1, dtype=T.i32)
                            t2 = fused2 & mask24_i32
                            s2 = fused2 >> 24
                            t_ok = arith.cmpi(arith.CmpIPredicate.ult, t2, tokens_i32_v)
                            s_ok = arith.cmpi(arith.CmpIPredicate.ult, s2, topk_i32_v)
                            ts_ok = t_ok & s_ok

                            # Sorted weight (topk routing weight)
                            tw = arith.select(
                                ts_ok,
                                buffer_ops.buffer_load(sorted_w_rsrc, row, vec_width=1, dtype=T.f32),
                                fx.Float32(0.0),
                            )

                        for ni in range_constexpr(num_acc_n):
                            col_local = col_base_local + (ni * 16)
                            acc_idx = mi * num_acc_n_s2 + ni
                            v = vector.extract(acc_out[acc_idx], static_position=[ii], dynamic_position=[])
                            if const_expr(stage2_use_fp8_w2):
                                sx2 = SmemPtr.load(lds_a2_scale, [row_in_tile])
                                v = v * sx2 * sw_vals_s2[ni]
                            # BF16 W2 path has scales already baked into W2_bf16.
                            v = v * tw
                            v_out = arith.trunc_f(out_elem, v)
                            lds_idx = cshuffle_lds_idx(row_in_tile, row_base_lds, col_local)
                            v1 = vector.from_elements(T.vec(1, out_elem), [v_out])
                            vector.store(v1, lds_out, [lds_idx], alignment=2)

                    def precompute_row_out(*, row_local, row):
                        fused2 = buffer_ops.buffer_load(sorted_rsrc, row, vec_width=1, dtype=T.i32)
                        row_i32 = arith.index_cast(T.i32, row)
                        row_valid0 = arith.cmpi(arith.CmpIPredicate.ult, row_i32, num_valid_i32)
                        t = fused2 & mask24_i32
                        s = fused2 >> 24
                        t_ok = arith.cmpi(arith.CmpIPredicate.ult, t, tokens_i32_v)
                        s_ok = arith.cmpi(arith.CmpIPredicate.ult, s, topk_i32_v)
                        row_valid = row_valid0 & t_ok & s_ok
                        return (fused2, row_valid)

                    def store_pair_out(*, row_local, row, row_ctx, col_pair0, col_g0, frag):
                        fused = row_ctx
                        t = fused & mask24_i32
                        idx0 = t * model_i32
                        col_i32 = arith.index_cast(T.i32, col_g0)
                        idx_elem = idx0 + col_i32
                        idx_elem_even = idx_elem & mask_even_i32
                        if const_expr(_needs_global_atomic_bf16):
                            byte_off = idx_elem_even * c2_i32
                            byte_off_idx = arith.index_cast(T.index, byte_off)
                            ptr_addr_idx = out_base_idx + byte_off_idx
                            out_ptr = buffer_ops.create_llvm_ptr(ptr_addr_idx, address_space=1)
                            out_ptr_v = out_ptr._value if hasattr(out_ptr, "_value") else out_ptr
                            frag_v = frag._value if hasattr(frag, "_value") else frag
                            llvm.AtomicRMWOp(
                                llvm.AtomicBinOp.fadd,
                                out_ptr_v, frag_v,
                                llvm.AtomicOrdering.monotonic,
                                syncscope="agent", alignment=4,
                            )
                        else:
                            byte_off = idx_elem_even * c2_i32
                            rocdl.raw_ptr_buffer_atomic_fadd(
                                frag, out_rsrc, byte_off, zero_i32, zero_i32,
                            )

                    c_shuffle_epilog(
                        arith=arith,
                        vector=vector,
                        gpu=gpu,
                        scf=scf,
                        range_constexpr=range_constexpr,
                        tile_m=tile_m,
                        tile_n=tile_n_out,
                        e_vec=_e_vec,
                        cshuffle_nlane=_cshuffle_nlane,
                        block_size=total_threads,
                        m_repeat=m_repeat,
                        num_acc_n=num_acc_n_s2,
                        tx=tx,
                        lane_div_16=lane_div_16,
                        lane_mod_16=lane_mod_16,
                        bx_m=bx_m,
                        by_n=by_n_s2,
                        n_tile_base=n_tile_base_s2,
                        lds_out=lds_out,
                        frag_elem_type=out_elem,
                        write_row_to_lds=write_row_to_lds_out,
                        precompute_row=precompute_row_out,
                        store_pair=store_pair_out,
                        lds_index_mapper=cshuffle_lds_idx if cshuffle_lds_xor else None,
                    )
                    if const_expr(loop_n_in_block):
                        gpu.barrier()

                def _emit_output_tile_pair(by_tile0):
                    # Pair two adjacent B2B output tiles while reusing the same A2 LDS loads.
                    by_tile1 = by_tile0 + fx.Index(1)
                    by_n0_s2 = by_tile0 * fx.Index(tile_n_out)
                    by_n1_s2 = by_tile1 * fx.Index(tile_n_out)
                    n_per_wave_s2 = tile_n_out // num_waves
                    num_acc_n_s2 = n_per_wave_s2 // 16
                    c_n_per_wave_s2 = fx.Index(n_per_wave_s2)
                    n_tile_base_s2 = wave_n_id * c_n_per_wave_s2
                    c_n0_s2 = experts * model_dim // 16
                    layout_n_blk_intra_s2 = fx.make_layout((c_n0_s2, 16), stride=(16, 1))

                    def precompute_w2_tile(by_n_s2):
                        n_intra_list_w2 = []
                        n_blk_list_w2 = []
                        col_g_list_s2 = []
                        for ni in range_constexpr(num_acc_n_s2):
                            offset = arith.index(ni * 16)
                            col_g = by_n_s2 + n_tile_base_s2 + offset + lane_mod_16
                            col_g_list_s2.append(col_g)
                            row_w = expert_off_w2 + col_g
                            coord_w = fx.idx2crd(row_w, layout_n_blk_intra_s2)
                            n_blk_list_w2.append(fx.get(coord_w, 0))
                            n_intra_list_w2.append(fx.get(coord_w, 1))
                        return n_blk_list_w2, n_intra_list_w2, col_g_list_s2

                    n_blk0_w2, n_intra0_w2, col_g0_s2 = precompute_w2_tile(by_n0_s2)
                    n_blk1_w2, n_intra1_w2, col_g1_s2 = precompute_w2_tile(by_n1_s2)
                    k_unroll_s2 = tile_k_bytes_w2 // 64
                    base_k_s2 = fx.Index(0)

                    def load_b_pack_w2_pair(base_k, ki_step, ni, n_blk_list, n_intra_list):
                        return load_b_pack_k32(
                            buffer_ops, arith, vector,
                            arg_b=arg_w2, b_rsrc=w2_rsrc, layout_b=layout_b_w2,
                            base_k=base_k, ki_step=ki_step,
                            n_blk=n_blk_list[ni], n_intra=n_intra_list[ni],
                            lane_div_16=lane_div_16, elem_type=w2_elem,
                            kpack_bytes=kpack_bytes_w2, elem_bytes=elem_bytes_w2,
                            unpack_int4=False,
                            **({"cache_modifier": b_cache_modifier_w2} if b_cache_modifier_w2 != 0 else {}),
                        )

                    def load_b_packs_for_tile(ku, n_blk_list, n_intra_list):
                        packs0 = []
                        packs1 = []
                        for ni in range_constexpr(num_acc_n_s2):
                            ki0 = (ku * 2) + 0
                            ki1 = (ku * 2) + 1
                            packs0.append(
                                load_b_pack_w2_pair(base_k_s2, ki0, ni, n_blk_list, n_intra_list)
                            )
                            packs1.append(
                                load_b_pack_w2_pair(base_k_s2, ki1, ni, n_blk_list, n_intra_list)
                            )
                        return packs0, packs1

                    a_kpack_elems_fp8_s2 = 16
                    col_offset_base_fp8_s2 = lane_div_16 * arith.index(a_kpack_elems_fp8_s2)
                    vec16_a2_fp8 = T.vec(16, T.f8)

                    def lds_load_packs_k64_s2_pair(curr_row_a_lds, col_base_bytes, lds_base_s2):
                        col_base_swz = swizzle_xor16(
                            curr_row_a_lds, col_base_bytes, k_blocks16_s2_fp8
                        )
                        idx_a16 = crd2idx((curr_row_a_lds, col_base_swz), layout_lds_s2_fp8)
                        idx_a16 = idx_a16 + lds_base_s2
                        loaded_a16 = vector.load_op(vec16_a2_fp8, lds_a2_q, [idx_a16])
                        a_i64x2 = vector.bitcast(T.i64x2, loaded_a16)
                        a0 = vector.extract(a_i64x2, static_position=[0], dynamic_position=[])
                        a1 = vector.extract(a_i64x2, static_position=[1], dynamic_position=[])
                        return a0, a1

                    def mfma_k64_s2_pair(acc0, a0, a1, b0, b1):
                        return mfma_k64_fp8(acc0, a0, a1, b0, b1)

                    acc_out0 = [acc_init] * (m_repeat * num_acc_n_s2)
                    acc_out1 = [acc_init] * (m_repeat * num_acc_n_s2)
                    lds_base_s2_val = fx.Index(0)

                    for ku in range_constexpr(k_unroll_s2):
                        b0_packs0, b0_packs1 = load_b_packs_for_tile(ku, n_blk0_w2, n_intra0_w2)
                        b1_packs0, b1_packs1 = load_b_packs_for_tile(ku, n_blk1_w2, n_intra1_w2)
                        ki64 = arith.index(ku * 64)
                        col_base = col_offset_base_fp8_s2 + ki64

                        for mi in range_constexpr(m_repeat):
                            mi_val = arith.index(mi * 16)
                            curr_row_a_lds = row_a_lds + mi_val
                            a0, a1 = lds_load_packs_k64_s2_pair(
                                curr_row_a_lds, col_base, lds_base_s2_val
                            )

                            for ni in range_constexpr(num_acc_n_s2):
                                acc_idx = mi * num_acc_n_s2 + ni
                                acc_out0[acc_idx] = mfma_k64_s2_pair(
                                    acc_out0[acc_idx], a0, a1,
                                    b0_packs0[ni], b0_packs1[ni],
                                )
                                acc_out1[acc_idx] = mfma_k64_s2_pair(
                                    acc_out1[acc_idx], a0, a1,
                                    b1_packs0[ni], b1_packs1[ni],
                                )

                    model_i32 = fx.Int32(model_dim)
                    topk_i32_v = topk_i32
                    zero_i32 = fx.Int32(0)
                    c2_i32 = fx.Int32(2)
                    mask_even_i32 = fx.Int32(0xFFFFFFFE)

                    out_base_idx = fx.Index(0)
                    if const_expr(_needs_global_atomic_bf16):
                        out_base_idx = buffer_ops.extract_base_index(arg_out)

                    def _emit_pair_epilog(acc_for_tile, by_n_s2, col_g_list_s2):
                        sw_vals_s2 = []
                        for ni in range_constexpr(num_acc_n_s2):
                            row_w_idx = expert_off_w2 + col_g_list_s2[ni]
                            sw_vals_s2.append(
                                buffer_ops.buffer_load(sw2_rsrc, row_w_idx, vec_width=1, dtype=T.f32)
                            )

                        def write_row_to_lds_out_pair(
                            *, mi: int, ii: int, row_in_tile, row,
                            row_base_lds, col_base_local, num_acc_n: int, lds_out,
                        ):
                            fused2 = buffer_ops.buffer_load(sorted_rsrc, row, vec_width=1, dtype=T.i32)
                            t2 = fused2 & mask24_i32
                            s2 = fused2 >> 24
                            t_ok = arith.cmpi(arith.CmpIPredicate.ult, t2, tokens_i32_v)
                            s_ok = arith.cmpi(arith.CmpIPredicate.ult, s2, topk_i32_v)
                            ts_ok = t_ok & s_ok
                            tw = arith.select(
                                ts_ok,
                                buffer_ops.buffer_load(sorted_w_rsrc, row, vec_width=1, dtype=T.f32),
                                fx.Float32(0.0),
                            )

                            for ni in range_constexpr(num_acc_n):
                                col_local = col_base_local + (ni * 16)
                                acc_idx = mi * num_acc_n_s2 + ni
                                v = vector.extract(
                                    acc_for_tile[acc_idx],
                                    static_position=[ii],
                                    dynamic_position=[],
                                )
                                sx2 = SmemPtr.load(lds_a2_scale, [row_in_tile])
                                v = v * sx2 * sw_vals_s2[ni]
                                v = v * tw
                                v_out = arith.trunc_f(out_elem, v)
                                lds_idx = row_base_lds + col_local
                                v1 = vector.from_elements(T.vec(1, out_elem), [v_out])
                                vector.store(v1, lds_out, [lds_idx], alignment=2)

                        def precompute_row_out_pair(*, row_local, row):
                            fused2 = buffer_ops.buffer_load(sorted_rsrc, row, vec_width=1, dtype=T.i32)
                            row_i32 = arith.index_cast(T.i32, row)
                            row_valid0 = arith.cmpi(arith.CmpIPredicate.ult, row_i32, num_valid_i32)
                            t = fused2 & mask24_i32
                            s = fused2 >> 24
                            t_ok = arith.cmpi(arith.CmpIPredicate.ult, t, tokens_i32_v)
                            s_ok = arith.cmpi(arith.CmpIPredicate.ult, s, topk_i32_v)
                            row_valid = row_valid0 & t_ok & s_ok
                            return (fused2, row_valid)

                        def store_pair_out_pair(*, row_local, row, row_ctx, col_pair0, col_g0, frag):
                            fused = row_ctx
                            t = fused & mask24_i32
                            idx0 = t * model_i32
                            col_i32 = arith.index_cast(T.i32, col_g0)
                            idx_elem = idx0 + col_i32
                            idx_elem_even = idx_elem & mask_even_i32
                            if const_expr(_needs_global_atomic_bf16):
                                byte_off = idx_elem_even * c2_i32
                                byte_off_idx = arith.index_cast(T.index, byte_off)
                                ptr_addr_idx = out_base_idx + byte_off_idx
                                out_ptr = buffer_ops.create_llvm_ptr(ptr_addr_idx, address_space=1)
                                out_ptr_v = out_ptr._value if hasattr(out_ptr, "_value") else out_ptr
                                frag_v = frag._value if hasattr(frag, "_value") else frag
                                llvm.AtomicRMWOp(
                                    llvm.AtomicBinOp.fadd,
                                    out_ptr_v, frag_v,
                                    llvm.AtomicOrdering.monotonic,
                                    syncscope="agent", alignment=4,
                                )
                            else:
                                byte_off = idx_elem_even * c2_i32
                                rocdl.raw_ptr_buffer_atomic_fadd(
                                    frag, out_rsrc, byte_off, zero_i32, zero_i32,
                                )

                        c_shuffle_epilog(
                            arith=arith,
                            vector=vector,
                            gpu=gpu,
                            scf=scf,
                            range_constexpr=range_constexpr,
                            tile_m=tile_m,
                            tile_n=tile_n_out,
                            e_vec=_e_vec,
                            cshuffle_nlane=_cshuffle_nlane,
                            block_size=total_threads,
                            m_repeat=m_repeat,
                            num_acc_n=num_acc_n_s2,
                            tx=tx,
                            lane_div_16=lane_div_16,
                            lane_mod_16=lane_mod_16,
                            bx_m=bx_m,
                            by_n=by_n_s2,
                            n_tile_base=n_tile_base_s2,
                            lds_out=lds_out,
                            frag_elem_type=out_elem,
                            write_row_to_lds=write_row_to_lds_out_pair,
                            precompute_row=precompute_row_out_pair,
                            store_pair=store_pair_out_pair,
                        )
                        gpu.barrier()

                    _emit_pair_epilog(acc_out0, by_n0_s2, col_g0_s2)
                    _emit_pair_epilog(acc_out1, by_n1_s2, col_g1_s2)

                if const_expr(loop_n_in_block):
                    if const_expr(n_tiles_per_block != 0):
                        by_group_start = by * fx.Index(n_tiles_per_block)
                        for by_local in range_constexpr(n_tiles_per_block):
                            _emit_output_tile(by_group_start + fx.Index(by_local))
                    elif const_expr(out_tile_pair == 2):
                        for by_pair in range_constexpr(total_n_tiles // 2):
                            _emit_output_tile_pair(fx.Index(by_pair * 2))
                    else:
                        for by_iter in range_constexpr(total_n_tiles):
                            _emit_output_tile(fx.Index(by_iter))
                else:
                    _emit_output_tile(by)

    # ── Host launcher ────────────────────────────────────────────────────────
    @flyc.jit
    def launch_moe_gemm_fused(
        arg_out: fx.Tensor,
        arg_x: fx.Tensor,
        arg_w1: fx.Tensor,
        arg_w2: fx.Tensor,
        arg_scale_x: fx.Tensor,
        arg_scale_w1: fx.Tensor,
        arg_scale_w2: fx.Tensor,
        arg_sorted_token_ids: fx.Tensor,
        arg_expert_ids: fx.Tensor,
        arg_sorted_weights: fx.Tensor,
        arg_num_valid_ids: fx.Tensor,
        i32_tokens_in: fx.Int32,
        i32_model_dim_in: fx.Int32,
        i32_inter_dim_in: fx.Int32,
        i32_size_expert_ids_in: fx.Int32,
        stream: fx.Stream,
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        model_dim_in = arith.index_cast(T.index, i32_model_dim_in)
        size_expert_ids_in = arith.index_cast(T.index, i32_size_expert_ids_in)
        gx = model_dim_in // fx.Index(tile_n_out)
        if const_expr(loop_n_in_block):
            if const_expr(n_tiles_per_block != 0):
                gx = gx // fx.Index(n_tiles_per_block)
            else:
                gx = fx.Index(1)
        gy = size_expert_ids_in

        moe_gemm_fused(
            arg_out, arg_x, arg_w1, arg_w2,
            arg_scale_x, arg_scale_w1, arg_scale_w2,
            arg_sorted_token_ids, arg_expert_ids, arg_sorted_weights,
            arg_num_valid_ids,
            i32_tokens_in, i32_model_dim_in, i32_inter_dim_in,
            i32_size_expert_ids_in,
        ).launch(
            grid=(gx, gy, 1),
            block=(total_threads, 1, 1),
            stream=stream,
        )

    return launch_moe_gemm_fused
