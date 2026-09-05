"""Triton input packer for the MegaMoE router.

This fuses the hot pre-Mega chain: BF16 activation -> FP8 E4M3, packed UE8M0
group-32 scales, and router tensor copies into DeepGEMM's symmetric-memory
dispatch buffer. It mirrors ``per_token_cast_to_fp8_packed_ue8m0`` but writes
directly into the final buffer instead of materializing temporary tensors.
"""

from __future__ import annotations

import os

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - CPU-only import
    triton = None
    tl = None


if triton is not None:

    @triton.jit(do_not_specialize=["M"])
    def _pack_x_kernel(
        x_ptr,
        out_fp8_ptr,
        out_sf_ptr,
        M,
        N: tl.constexpr,
        x_stride_m: tl.constexpr,
        out_stride_m: tl.constexpr,
        sf_stride_m: tl.constexpr,
        eps: tl.constexpr,
        fp8_max: tl.constexpr,
    ):
        pid_m = tl.program_id(0).to(tl.int64)
        pid_blk = tl.program_id(1)
        offs = tl.arange(0, 128)
        col = pid_blk * 128 + offs
        mask = (pid_m < M) & (col < N)
        x = tl.load(x_ptr + pid_m * x_stride_m + col, mask=mask, other=0.0).to(
            tl.float32
        )
        x_is_finite = tl.abs(x) < float("inf")
        x = tl.where(x_is_finite, x, 0.0)
        x_2d = tl.reshape(tl.abs(x), (4, 32))
        block_absmax = tl.maximum(tl.max(x_2d, axis=1), eps)
        scale = tl.math.exp2(tl.ceil(tl.log2(block_absmax / fp8_max)))
        scale_exp = tl.reshape(
            tl.broadcast_to(tl.reshape(scale, (4, 1)), (4, 32)),
            (128,),
        )
        q = tl.clamp(x / scale_exp, -fp8_max, fp8_max).to(tl.float8e4nv)
        tl.store(out_fp8_ptr + pid_m * out_stride_m + col, q, mask=mask)

        scale_bits = scale.to(tl.int32, bitcast=True)
        group_offsets = tl.arange(0, 4)
        ue8m0 = (scale_bits >> 23) & 0xFF
        packed = tl.sum(ue8m0 << (group_offsets * 8))
        tl.store(out_sf_ptr + pid_m * sf_stride_m + pid_blk, packed, mask=pid_m < M)

    @triton.jit(do_not_specialize=["M"])
    def _pack_router_kernel(
        weights_ptr,
        indices_ptr,
        out_weights_ptr,
        out_indices_ptr,
        M,
        K: tl.constexpr,
        weights_stride_m: tl.constexpr,
        indices_stride_m: tl.constexpr,
        out_weights_stride_m: tl.constexpr,
        out_indices_stride_m: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid = tl.program_id(0).to(tl.int64)
        offs = tl.arange(0, BLOCK_K)
        mask = (pid < M) & (offs < K)
        w = tl.load(
            weights_ptr + pid * weights_stride_m + offs, mask=mask, other=0.0
        ).to(tl.float32)
        idx = tl.load(
            indices_ptr + pid * indices_stride_m + offs, mask=mask, other=0
        ).to(tl.int64)
        tl.store(out_weights_ptr + pid * out_weights_stride_m + offs, w, mask=mask)
        tl.store(out_indices_ptr + pid * out_indices_stride_m + offs, idx, mask=mask)

    @triton.jit(do_not_specialize=["M"])
    def _pack_mega_moe_inputs_optimized_kernel(
        x_ptr,
        weights_ptr,
        indices_ptr,
        out_fp8_ptr,
        out_sf_ptr,
        out_weights_ptr,
        out_indices_ptr,
        M,
        N: tl.constexpr,
        K: tl.constexpr,
        x_stride_m: tl.constexpr,
        weights_stride_m: tl.constexpr,
        indices_stride_m: tl.constexpr,
        out_stride_m: tl.constexpr,
        sf_stride_m: tl.constexpr,
        out_weights_stride_m: tl.constexpr,
        out_indices_stride_m: tl.constexpr,
        eps: tl.constexpr,
        fp8_max: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m_blk = tl.program_id(0).to(tl.int64)
        pid_blk = tl.program_id(1)

        offs_m = pid_m_blk * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
        offs_32 = tl.arange(0, 32)
        row_mask = offs_m < M
        packed = tl.zeros((BLOCK_M,), dtype=tl.int32)

        for pack_idx in tl.static_range(4):
            cols = pid_blk * 128 + pack_idx * 32 + offs_32
            mask = row_mask[:, None] & (cols[None, :] < N)
            x = tl.load(
                x_ptr + offs_m[:, None] * x_stride_m + cols[None, :],
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            x_is_finite = tl.abs(x) < float("inf")
            x = tl.where(x_is_finite, x, 0.0)

            block_absmax = tl.maximum(tl.max(tl.abs(x), axis=1), eps)
            scale_raw = block_absmax / fp8_max
            scale_raw_bits = scale_raw.to(tl.int32, bitcast=True)
            exp = ((scale_raw_bits >> 23) & 0xFF) + ((scale_raw_bits & 0x7FFFFF) != 0)
            exp = tl.minimum(tl.maximum(exp, 1), 254)
            scale_bits = exp << 23
            scale = scale_bits.to(tl.float32, bitcast=True)

            q = tl.clamp(x / scale[:, None], -fp8_max, fp8_max).to(tl.float8e4nv)
            tl.store(
                out_fp8_ptr + offs_m[:, None] * out_stride_m + cols[None, :],
                q,
                mask=mask,
            )
            packed = packed | (exp << (pack_idx * 8))

        tl.store(
            out_sf_ptr + offs_m * sf_stride_m + pid_blk,
            packed,
            mask=row_mask,
        )

        if pid_blk == 0:
            router_offs = tl.arange(0, BLOCK_K)
            router_mask = row_mask[:, None] & (router_offs[None, :] < K)
            w = tl.load(
                weights_ptr + offs_m[:, None] * weights_stride_m + router_offs[None, :],
                mask=router_mask,
                other=0.0,
            ).to(tl.float32)
            idx = tl.load(
                indices_ptr + offs_m[:, None] * indices_stride_m + router_offs[None, :],
                mask=router_mask,
                other=0,
            ).to(tl.int64)
            tl.store(
                out_weights_ptr
                + offs_m[:, None] * out_weights_stride_m
                + router_offs[None, :],
                w,
                mask=router_mask,
            )
            tl.store(
                out_indices_ptr
                + offs_m[:, None] * out_indices_stride_m
                + router_offs[None, :],
                idx,
                mask=router_mask,
            )

    @triton.jit
    def _pack_x_block(
        x_ptr,
        out_fp8_ptr,
        out_sf_ptr,
        offs_m,
        pid_blk: tl.constexpr,
        M,
        N: tl.constexpr,
        x_stride_m: tl.constexpr,
        out_stride_m: tl.constexpr,
        sf_stride_m: tl.constexpr,
        eps: tl.constexpr,
        fp8_max: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        offs_32 = tl.arange(0, 32)
        row_mask = offs_m < M
        packed = tl.zeros((BLOCK_M,), dtype=tl.int32)

        for pack_idx in tl.static_range(4):
            cols = pid_blk * 128 + pack_idx * 32 + offs_32
            mask = row_mask[:, None] & (cols[None, :] < N)
            x = tl.load(
                x_ptr + offs_m[:, None] * x_stride_m + cols[None, :],
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            x_is_finite = tl.abs(x) < float("inf")
            x = tl.where(x_is_finite, x, 0.0)

            block_absmax = tl.maximum(tl.max(tl.abs(x), axis=1), eps)
            scale_raw = block_absmax / fp8_max
            scale_raw_bits = scale_raw.to(tl.int32, bitcast=True)
            exp = ((scale_raw_bits >> 23) & 0xFF) + ((scale_raw_bits & 0x7FFFFF) != 0)
            exp = tl.minimum(tl.maximum(exp, 1), 254)
            scale_bits = exp << 23
            scale = scale_bits.to(tl.float32, bitcast=True)

            q = tl.clamp(x / scale[:, None], -fp8_max, fp8_max).to(tl.float8e4nv)
            tl.store(
                out_fp8_ptr + offs_m[:, None] * out_stride_m + cols[None, :],
                q,
                mask=mask,
            )
            packed = packed | (exp << (pack_idx * 8))

        tl.store(
            out_sf_ptr + offs_m * sf_stride_m + pid_blk,
            packed,
            mask=row_mask,
        )

    @triton.jit(do_not_specialize=["M"])
    def _mega_moe_gate_pack_nonhash_kernel(
        x_ptr,
        scores_ptr,
        bias_ptr,
        out_fp8_ptr,
        out_sf_ptr,
        out_weights_ptr,
        out_indices_ptr,
        M,
        N: tl.constexpr,
        E: tl.constexpr,
        K: tl.constexpr,
        x_stride_m: tl.constexpr,
        scores_stride_m: tl.constexpr,
        out_stride_m: tl.constexpr,
        sf_stride_m: tl.constexpr,
        out_weights_stride_m: tl.constexpr,
        out_indices_stride_m: tl.constexpr,
        route_scale: tl.constexpr,
        norm_eps: tl.constexpr,
        eps: tl.constexpr,
        fp8_max: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_E: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m_blk = tl.program_id(0).to(tl.int64)
        pid_blk = tl.program_id(1)
        offs_m = pid_m_blk * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)

        _pack_x_block(
            x_ptr,
            out_fp8_ptr,
            out_sf_ptr,
            offs_m,
            pid_blk,
            M,
            N,
            x_stride_m,
            out_stride_m,
            sf_stride_m,
            eps,
            fp8_max,
            BLOCK_M,
        )

        if pid_blk == 0:
            offs_e = tl.arange(0, BLOCK_E)
            e_mask = offs_e < E
            bias_row = tl.load(bias_ptr + offs_e, mask=e_mask, other=0.0).to(tl.float32)
            bias_is_finite = tl.abs(bias_row) < float("inf")
            bias_row = tl.where(bias_is_finite, bias_row, 0.0)
            k_offs = tl.arange(0, BLOCK_K)
            k_mask_base = k_offs < K

            for row_i in tl.static_range(BLOCK_M):
                row = pid_m_blk * BLOCK_M + row_i
                row_mask = row < M
                scores = tl.load(
                    scores_ptr + row * scores_stride_m + offs_e,
                    mask=row_mask & e_mask,
                    other=0.0,
                ).to(tl.float32)
                score_is_finite = tl.abs(scores) < float("inf")
                bad_value = row_mask & e_mask & (~score_is_finite | ~bias_is_finite)
                row_is_finite = tl.sum(bad_value.to(tl.int32), axis=0) == 0
                scores = tl.where(score_is_finite, scores, 0.0)

                threshold = tl.full([1], 20.0, dtype=tl.float32)
                softplus = tl.where(
                    scores > threshold, scores, tl.log(1.0 + tl.exp(scores))
                )
                active = tl.sqrt(softplus)
                biased = tl.where(e_mask, active + bias_row, -float("inf"))
                cur = biased
                selected_weights = tl.zeros((BLOCK_K,), dtype=tl.float32)

                for k in tl.static_range(K):
                    idx = tl.argmax(cur, axis=0)
                    weight = tl.sum(tl.where(offs_e == idx, active, 0.0), axis=0)
                    safe_idx = tl.where(row_is_finite, idx, k)
                    tl.store(
                        out_indices_ptr + row * out_indices_stride_m + k,
                        safe_idx.to(tl.int64),
                        mask=row_mask,
                    )
                    selected_weights = tl.where(k_offs == k, weight, selected_weights)
                    cur = tl.where(offs_e == idx, -float("inf"), cur)

                k_mask = row_mask & k_mask_base
                denom = tl.sum(tl.where(k_mask_base, selected_weights, 0.0), axis=0)
                weights = selected_weights / (denom + norm_eps) * route_scale
                weights = tl.where(row_is_finite, weights, route_scale / K)
                tl.store(
                    out_weights_ptr + row * out_weights_stride_m + k_offs,
                    weights,
                    mask=k_mask,
                )

    @triton.jit(do_not_specialize=["M"])
    def _mega_moe_gate_pack_hash_kernel(
        x_ptr,
        scores_ptr,
        input_ids_ptr,
        tid2eid_ptr,
        out_fp8_ptr,
        out_sf_ptr,
        out_weights_ptr,
        out_indices_ptr,
        M,
        N: tl.constexpr,
        E: tl.constexpr,
        K: tl.constexpr,
        x_stride_m: tl.constexpr,
        scores_stride_m: tl.constexpr,
        input_ids_stride: tl.constexpr,
        tid2eid_stride_m: tl.constexpr,
        tid2eid_stride_k: tl.constexpr,
        out_stride_m: tl.constexpr,
        sf_stride_m: tl.constexpr,
        out_weights_stride_m: tl.constexpr,
        out_indices_stride_m: tl.constexpr,
        route_scale: tl.constexpr,
        norm_eps: tl.constexpr,
        eps: tl.constexpr,
        fp8_max: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m_blk = tl.program_id(0).to(tl.int64)
        pid_blk = tl.program_id(1)
        offs_m = pid_m_blk * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)

        _pack_x_block(
            x_ptr,
            out_fp8_ptr,
            out_sf_ptr,
            offs_m,
            pid_blk,
            M,
            N,
            x_stride_m,
            out_stride_m,
            sf_stride_m,
            eps,
            fp8_max,
            BLOCK_M,
        )

        if pid_blk == 0:
            k_offs = tl.arange(0, BLOCK_K)
            k_mask_base = k_offs < K

            for row_i in tl.static_range(BLOCK_M):
                row = pid_m_blk * BLOCK_M + row_i
                row_mask = row < M
                token_id = tl.load(
                    input_ids_ptr + row * input_ids_stride,
                    mask=row_mask,
                    other=0,
                ).to(tl.int64)
                k_mask = row_mask & k_mask_base
                idx = tl.load(
                    tid2eid_ptr
                    + token_id * tid2eid_stride_m
                    + k_offs * tid2eid_stride_k,
                    mask=k_mask,
                    other=0,
                ).to(tl.int64)
                selected = tl.load(
                    scores_ptr + row * scores_stride_m + idx,
                    mask=k_mask,
                    other=0.0,
                ).to(tl.float32)
                selected_is_finite = tl.abs(selected) < float("inf")
                row_is_finite = (
                    tl.sum((k_mask & ~selected_is_finite).to(tl.int32), axis=0) == 0
                )
                selected = tl.where(selected_is_finite, selected, 0.0)
                threshold = tl.full([1], 20.0, dtype=tl.float32)
                softplus = tl.where(
                    selected > threshold,
                    selected,
                    tl.log(1.0 + tl.exp(selected)),
                )
                weights = tl.sqrt(softplus)
                denom = tl.sum(tl.where(k_mask, weights, 0.0), axis=0) + norm_eps
                weights = weights / denom * route_scale
                weights = tl.where(row_is_finite, weights, route_scale / K)
                tl.store(
                    out_indices_ptr + row * out_indices_stride_m + k_offs,
                    idx,
                    mask=k_mask,
                )
                tl.store(
                    out_weights_ptr + row * out_weights_stride_m + k_offs,
                    weights,
                    mask=k_mask,
                )


def _validate_inputs(
    x: torch.Tensor,
    weights: torch.Tensor,
    indices: torch.Tensor,
    out_fp8: torch.Tensor,
    out_sf: torch.Tensor,
    out_indices: torch.Tensor,
    out_weights: torch.Tensor,
) -> tuple[int, int, int]:
    if triton is None:
        raise RuntimeError("triton is unavailable")
    if x.dim() != 2:
        raise ValueError(f"x must be [T,D], got {tuple(x.shape)}")
    T, D = x.shape
    if weights.dim() != 2 or indices.dim() != 2:
        raise ValueError("weights and indices must both be rank-2 [T,topk]")
    if weights.shape != indices.shape or weights.size(0) != T:
        raise ValueError(
            "weights and indices must have identical [T,topk] shape matching x"
        )
    if D % 128 != 0:
        raise ValueError(f"fused MegaMoE packer requires D % 128 == 0, got D={D}")
    topk = weights.size(1)
    specs = (
        ("x", x, (T, D), torch.bfloat16),
        ("weights", weights, (T, topk), torch.float32),
        ("indices", indices, (T, topk), torch.int64),
        ("out_fp8", out_fp8, (T, D), torch.float8_e4m3fn),
        ("out_sf", out_sf, (T, D // 128), torch.int32),
        ("out_indices", out_indices, (T, topk), torch.int64),
        ("out_weights", out_weights, (T, topk), torch.float32),
    )
    for name, tensor, expected_shape, expected_dtype in specs:
        if not tensor.is_cuda:
            raise RuntimeError(
                f"fused MegaMoE input packer requires CUDA tensors; {name} "
                f"is on {tensor.device}"
            )
        if tensor.device != x.device:
            raise ValueError(f"{name} must be on {x.device}, got {tensor.device}")
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"{name} shape mismatch: expected {expected_shape}, "
                f"got {tuple(tensor.shape)}"
            )
        if tensor.dtype != expected_dtype:
            raise ValueError(f"{name} must be {expected_dtype}, got {tensor.dtype}")
        if tensor.dim() > 1 and tensor.stride(-1) != 1:
            raise ValueError(
                f"{name} must be contiguous in its last dimension; "
                f"got stride={tensor.stride()}"
            )
    return T, D, topk


def fused_pack_mega_moe_inputs_legacy(
    x: torch.Tensor,
    weights: torch.Tensor,
    indices: torch.Tensor,
    out_fp8: torch.Tensor,
    out_sf: torch.Tensor,
    out_indices: torch.Tensor,
    out_weights: torch.Tensor,
) -> None:
    T, D, topk = _validate_inputs(
        x, weights, indices, out_fp8, out_sf, out_indices, out_weights
    )
    if T == 0:
        return
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    grid_x = (T, triton.cdiv(D, 128))
    _pack_x_kernel[grid_x](
        x,
        out_fp8,
        out_sf,
        T,
        D,
        x.stride(0),
        out_fp8.stride(0),
        out_sf.stride(0),
        1.0e-4,
        fp8_max,
        num_warps=4,
    )
    block_k = triton.next_power_of_2(topk)
    _pack_router_kernel[(T,)](
        weights,
        indices,
        out_weights,
        out_indices,
        T,
        topk,
        weights.stride(0),
        indices.stride(0),
        out_weights.stride(0),
        out_indices.stride(0),
        BLOCK_K=block_k,
        num_warps=1,
    )


def fused_pack_mega_moe_inputs_optimized(
    x: torch.Tensor,
    weights: torch.Tensor,
    indices: torch.Tensor,
    out_fp8: torch.Tensor,
    out_sf: torch.Tensor,
    out_indices: torch.Tensor,
    out_weights: torch.Tensor,
) -> None:
    T, D, topk = _validate_inputs(
        x, weights, indices, out_fp8, out_sf, out_indices, out_weights
    )
    if T == 0:
        return
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    block_k = triton.next_power_of_2(topk)
    block_m_env = os.environ.get("MEGA_MOE_PACK_BLOCK_M")
    block_m = int(block_m_env) if block_m_env is not None else (8 if T >= 2048 else 2)
    if block_m not in (1, 2, 4, 8):
        raise ValueError(
            f"invalid MEGA_MOE_PACK_BLOCK_M={block_m}; expected 1, 2, 4, or 8"
        )
    grid = (triton.cdiv(T, block_m), triton.cdiv(D, 128))
    _pack_mega_moe_inputs_optimized_kernel[grid](
        x,
        weights,
        indices,
        out_fp8,
        out_sf,
        out_weights,
        out_indices,
        T,
        D,
        topk,
        x.stride(0),
        weights.stride(0),
        indices.stride(0),
        out_fp8.stride(0),
        out_sf.stride(0),
        out_weights.stride(0),
        out_indices.stride(0),
        1.0e-4,
        fp8_max,
        BLOCK_M=block_m,
        BLOCK_K=block_k,
        num_warps=4,
    )


def fused_pack_mega_moe_inputs(
    x: torch.Tensor,
    weights: torch.Tensor,
    indices: torch.Tensor,
    out_fp8: torch.Tensor,
    out_sf: torch.Tensor,
    out_indices: torch.Tensor,
    out_weights: torch.Tensor,
) -> None:
    impl = os.environ.get("MEGA_MOE_INPUT_PACKER_IMPL", "optimized").lower()
    if impl == "legacy":
        return fused_pack_mega_moe_inputs_legacy(
            x, weights, indices, out_fp8, out_sf, out_indices, out_weights
        )
    if impl == "optimized":
        return fused_pack_mega_moe_inputs_optimized(
            x, weights, indices, out_fp8, out_sf, out_indices, out_weights
        )
    raise ValueError(
        f"invalid MEGA_MOE_INPUT_PACKER_IMPL={impl!r}; expected legacy|optimized"
    )


def _validate_gate_pack_inputs(
    x: torch.Tensor,
    scores: torch.Tensor,
    out_fp8: torch.Tensor,
    out_sf: torch.Tensor,
    out_indices: torch.Tensor,
    out_weights: torch.Tensor,
    topk: int,
) -> tuple[int, int, int]:
    if triton is None:
        raise RuntimeError("triton is unavailable")
    if x.dim() != 2 or scores.dim() != 2:
        raise ValueError(
            f"x and scores must be rank 2, got {tuple(x.shape)} / {tuple(scores.shape)}"
        )
    tokens, dim = x.shape
    score_tokens, experts = scores.shape
    if score_tokens != tokens:
        raise ValueError(f"scores rows={score_tokens} must match input rows={tokens}")
    if dim % 128 != 0:
        raise ValueError(f"MegaMoE gate-pack requires D % 128 == 0, got D={dim}")
    topk = int(topk)
    if not 1 <= topk <= 32:
        raise ValueError(f"MegaMoE gate-pack requires 1 <= topk <= 32, got {topk}")
    specs = (
        ("x", x, (tokens, dim), torch.bfloat16),
        ("scores", scores, (tokens, experts), torch.bfloat16),
        ("out_fp8", out_fp8, (tokens, dim), torch.float8_e4m3fn),
        ("out_sf", out_sf, (tokens, dim // 128), torch.int32),
        ("out_indices", out_indices, (tokens, topk), torch.int64),
        ("out_weights", out_weights, (tokens, topk), torch.float32),
    )
    for name, tensor, expected_shape, expected_dtype in specs:
        if not tensor.is_cuda:
            raise RuntimeError(
                f"MegaMoE gate-pack requires CUDA tensors; {name} is on {tensor.device}"
            )
        if tensor.device != x.device:
            raise ValueError(f"{name} must be on {x.device}, got {tensor.device}")
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"{name} shape mismatch: expected {expected_shape}, "
                f"got {tuple(tensor.shape)}"
            )
        if tensor.dtype != expected_dtype:
            raise ValueError(f"{name} must be {expected_dtype}, got {tensor.dtype}")
        if tensor.dim() > 1 and tensor.stride(-1) != 1:
            raise ValueError(
                f"{name} must be contiguous in its last dimension; "
                f"got stride={tensor.stride()}"
            )
    return tokens, dim, experts


def _gate_pack_block_m(tokens: int) -> int:
    value = os.environ.get("MEGA_MOE_PACK_BLOCK_M")
    block_m = (
        int(value)
        if value is not None
        else (8 if tokens >= 2048 else (4 if tokens >= 1024 else 2))
    )
    if block_m not in (1, 2, 4, 8):
        raise ValueError(
            f"invalid MEGA_MOE_PACK_BLOCK_M={block_m}; expected 1, 2, 4, or 8"
        )
    return block_m


def fused_pack_mega_moe_gate_inputs(
    x: torch.Tensor,
    scores: torch.Tensor,
    out_fp8: torch.Tensor,
    out_sf: torch.Tensor,
    out_indices: torch.Tensor,
    out_weights: torch.Tensor,
    *,
    topk: int,
    score_func: str,
    route_scale: float,
    norm_eps: float = 1.0e-12,
    bias: torch.Tensor | None = None,
    input_ids: torch.Tensor | None = None,
    tid2eid: torch.Tensor | None = None,
) -> None:
    """Fuse sqrt-softplus routing with MegaMoE activation/input packing.

    Hash routing is selected by supplying both ``input_ids`` and ``tid2eid``;
    score routing is selected by supplying ``bias``.  The API intentionally
    contains no model-specific configuration or weight names.
    """

    tokens, dim, experts = _validate_gate_pack_inputs(
        x,
        scores,
        out_fp8,
        out_sf,
        out_indices,
        out_weights,
        topk,
    )
    if score_func != "sqrtsoftplus":
        raise ValueError(
            f"MegaMoE gate-pack only supports sqrtsoftplus, got {score_func!r}"
        )
    if tokens == 0:
        return

    block_m = _gate_pack_block_m(tokens)
    block_k = triton.next_power_of_2(int(topk))
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    grid = (triton.cdiv(tokens, block_m), triton.cdiv(dim, 128))
    is_hash = tid2eid is not None
    if is_hash:
        if input_ids is None or tid2eid is None:
            raise ValueError("hash gate-pack requires both input_ids and tid2eid")
        if bias is not None:
            raise ValueError("hash gate-pack does not accept a routing bias")
        if input_ids.device != x.device or tid2eid.device != x.device:
            raise ValueError("input_ids and tid2eid must be on the input device")
        if input_ids.dim() != 1 or input_ids.numel() != tokens:
            raise ValueError(
                f"input_ids must be [T] with T={tokens}, got {tuple(input_ids.shape)}"
            )
        if tid2eid.dim() != 2 or tuple(tid2eid.shape[1:]) != (int(topk),):
            raise ValueError(
                f"tid2eid must be [vocab, topk={int(topk)}], "
                f"got {tuple(tid2eid.shape)}"
            )
        _mega_moe_gate_pack_hash_kernel[grid](
            x,
            scores,
            input_ids,
            tid2eid,
            out_fp8,
            out_sf,
            out_weights,
            out_indices,
            tokens,
            dim,
            experts,
            int(topk),
            x.stride(0),
            scores.stride(0),
            input_ids.stride(0),
            tid2eid.stride(0),
            tid2eid.stride(1),
            out_fp8.stride(0),
            out_sf.stride(0),
            out_weights.stride(0),
            out_indices.stride(0),
            float(route_scale),
            float(norm_eps),
            1.0e-4,
            fp8_max,
            BLOCK_M=block_m,
            BLOCK_K=block_k,
            num_warps=4,
        )
        return

    if bias is None:
        raise ValueError("non-hash gate-pack requires a routing bias")
    if bias.device != x.device:
        raise ValueError(f"bias must be on {x.device}, got {bias.device}")
    if bias.dtype != torch.float32 or bias.dim() != 1 or bias.numel() != experts:
        raise ValueError(
            f"bias must be [E] float32 with E={experts}, "
            f"got {tuple(bias.shape)} {bias.dtype}"
        )
    _mega_moe_gate_pack_nonhash_kernel[grid](
        x,
        scores,
        bias,
        out_fp8,
        out_sf,
        out_weights,
        out_indices,
        tokens,
        dim,
        experts,
        int(topk),
        x.stride(0),
        scores.stride(0),
        out_fp8.stride(0),
        out_sf.stride(0),
        out_weights.stride(0),
        out_indices.stride(0),
        float(route_scale),
        float(norm_eps),
        1.0e-4,
        fp8_max,
        BLOCK_M=block_m,
        BLOCK_E=triton.next_power_of_2(experts),
        BLOCK_K=block_k,
        num_warps=4,
    )
