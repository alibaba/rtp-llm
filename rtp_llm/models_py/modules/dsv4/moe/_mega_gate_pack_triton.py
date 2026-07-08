"""MegaMoE-specific fused router gate + input pack kernels.

The MegaMoE fused DeepGEMM path consumes a symmetric-memory buffer containing:

* BF16 activations quantized to FP8 E4M3.
* Packed UE8M0 group-32 activation scales.
* Router top-k expert ids and weights.

The existing path materializes router weights/indices first, then launches the
MegaMoE input packer.  These kernels keep the router scores in BF16, compute
the FP32 gate epilogue inside Triton, and write the final DeepGEMM buffer
directly.
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
            k_offs = tl.arange(0, BLOCK_K)
            k_mask_base = k_offs < K

            for row_i in tl.static_range(BLOCK_M):
                row = pid_m_blk * BLOCK_M + row_i
                row_mask = row < M
                scores = tl.load(
                    scores_ptr + row * scores_stride_m + offs_e,
                    mask=row_mask & e_mask,
                    other=-float("inf"),
                ).to(tl.float32)

                threshold = tl.full([1], 20.0, dtype=tl.float32)
                softplus = tl.where(scores > threshold, scores, tl.log(1.0 + tl.exp(scores)))
                active = tl.sqrt(softplus)
                biased = tl.where(e_mask, active + bias_row, -float("inf"))
                cur = biased
                selected_weights = tl.zeros((BLOCK_K,), dtype=tl.float32)

                for k in tl.static_range(K):
                    idx = tl.argmax(cur, axis=0)
                    weight = tl.sum(tl.where(offs_e == idx, active, 0.0), axis=0)
                    tl.store(
                        out_indices_ptr + row * out_indices_stride_m + k,
                        idx.to(tl.int64),
                        mask=row_mask,
                    )
                    selected_weights = tl.where(k_offs == k, weight, selected_weights)
                    cur = tl.where(offs_e == idx, -float("inf"), cur)

                k_mask = row_mask & k_mask_base
                denom = tl.sum(tl.where(k_mask_base, selected_weights, 0.0), axis=0)
                weights = selected_weights / (denom + norm_eps) * route_scale
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
                    tid2eid_ptr + token_id * tid2eid_stride_m + k_offs * tid2eid_stride_k,
                    mask=k_mask,
                    other=0,
                ).to(tl.int64)
                selected = tl.load(
                    scores_ptr + row * scores_stride_m + idx,
                    mask=k_mask,
                    other=-float("inf"),
                ).to(tl.float32)
                threshold = tl.full([1], 20.0, dtype=tl.float32)
                softplus = tl.where(
                    selected > threshold, selected, tl.log(1.0 + tl.exp(selected))
                )
                weights = tl.sqrt(softplus)
                denom = tl.sum(tl.where(k_mask, weights, 0.0), axis=0) + norm_eps
                weights = weights / denom * route_scale
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


def _block_m(tokens: int) -> int:
    value = os.environ.get("DSV4_MEGA_MOE_GATE_PACK_BLOCK_M")
    if value is not None:
        block_m = int(value)
    else:
        block_m = 8 if tokens >= 2048 else (4 if tokens >= 1024 else 2)
    if block_m not in (1, 2, 4, 8):
        raise ValueError(
            "invalid DSV4_MEGA_MOE_GATE_PACK_BLOCK_M="
            f"{block_m}; expected 1, 2, 4, or 8"
        )
    return block_m


def _validate_common(
    x: torch.Tensor,
    scores_bf16: torch.Tensor,
    out_fp8: torch.Tensor,
    out_sf: torch.Tensor,
    out_indices: torch.Tensor,
    out_weights: torch.Tensor,
) -> tuple[int, int, int, int]:
    if triton is None:
        raise RuntimeError("triton is unavailable")
    if not x.is_cuda:
        raise RuntimeError("MegaMoE gate-pack requires CUDA tensors")
    if x.dtype != torch.bfloat16:
        raise ValueError(f"x must be bfloat16, got {x.dtype}")
    if scores_bf16.dtype != torch.bfloat16:
        raise ValueError(f"scores_bf16 must be bfloat16, got {scores_bf16.dtype}")
    if x.dim() != 2 or scores_bf16.dim() != 2:
        raise ValueError(
            f"x/scores_bf16 must be 2D, got {tuple(x.shape)} / {tuple(scores_bf16.shape)}"
        )
    tokens, dim = x.shape
    score_tokens, experts = scores_bf16.shape
    if score_tokens != tokens:
        raise ValueError(
            f"scores rows {score_tokens} must match x rows {tokens}"
        )
    if dim % 128 != 0:
        raise ValueError(f"MegaMoE gate-pack requires D % 128 == 0, got D={dim}")
    if out_sf.shape[1] != dim // 128:
        raise ValueError(
            f"out_sf shape mismatch: expected second dim {dim // 128}, got {out_sf.shape}"
        )
    if out_indices.shape != out_weights.shape:
        raise ValueError("out_indices and out_weights must share [T, topk] shape")
    if out_indices.shape[0] != tokens:
        raise ValueError("router output row count must match tokens")
    if out_indices.dtype != torch.int64:
        raise ValueError(f"out_indices must be int64, got {out_indices.dtype}")
    if out_weights.dtype != torch.float32:
        raise ValueError(f"out_weights must be float32, got {out_weights.dtype}")
    return tokens, dim, experts, out_indices.shape[1]


def fused_mega_moe_gate_pack_nonhash(
    x: torch.Tensor,
    scores_bf16: torch.Tensor,
    bias: torch.Tensor,
    out_fp8: torch.Tensor,
    out_sf: torch.Tensor,
    out_indices: torch.Tensor,
    out_weights: torch.Tensor,
    *,
    route_scale: float,
    norm_eps: float = 1.0e-12,
) -> None:
    tokens, dim, experts, topk = _validate_common(
        x, scores_bf16, out_fp8, out_sf, out_indices, out_weights
    )
    if bias.dtype != torch.float32 or bias.dim() != 1 or bias.numel() != experts:
        raise ValueError(
            f"bias must be [E] float32 with E={experts}, got {tuple(bias.shape)} {bias.dtype}"
        )
    if tokens == 0:
        return
    block_m = _block_m(tokens)
    block_e = triton.next_power_of_2(experts)
    block_k = triton.next_power_of_2(topk)
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    grid = (triton.cdiv(tokens, block_m), triton.cdiv(dim, 128))
    _mega_moe_gate_pack_nonhash_kernel[grid](
        x,
        scores_bf16,
        bias,
        out_fp8,
        out_sf,
        out_weights,
        out_indices,
        tokens,
        dim,
        experts,
        topk,
        x.stride(0),
        scores_bf16.stride(0),
        out_fp8.stride(0),
        out_sf.stride(0),
        out_weights.stride(0),
        out_indices.stride(0),
        float(route_scale),
        float(norm_eps),
        1.0e-4,
        fp8_max,
        BLOCK_M=block_m,
        BLOCK_E=block_e,
        BLOCK_K=block_k,
        num_warps=4,
    )


def fused_mega_moe_gate_pack_hash(
    x: torch.Tensor,
    scores_bf16: torch.Tensor,
    input_ids: torch.Tensor,
    tid2eid: torch.Tensor,
    out_fp8: torch.Tensor,
    out_sf: torch.Tensor,
    out_indices: torch.Tensor,
    out_weights: torch.Tensor,
    *,
    route_scale: float,
    norm_eps: float = 1.0e-12,
) -> None:
    tokens, dim, experts, topk = _validate_common(
        x, scores_bf16, out_fp8, out_sf, out_indices, out_weights
    )
    if input_ids.dim() != 1 or input_ids.numel() != tokens:
        raise ValueError(
            f"input_ids must be [T] with T={tokens}, got {tuple(input_ids.shape)}"
        )
    if tid2eid.dim() != 2 or tid2eid.shape[1] != topk:
        raise ValueError(
            f"tid2eid must be [vocab, topk={topk}], got {tuple(tid2eid.shape)}"
        )
    if tokens == 0:
        return
    block_m = _block_m(tokens)
    block_k = triton.next_power_of_2(topk)
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    grid = (triton.cdiv(tokens, block_m), triton.cdiv(dim, 128))
    _mega_moe_gate_pack_hash_kernel[grid](
        x,
        scores_bf16,
        input_ids,
        tid2eid,
        out_fp8,
        out_sf,
        out_weights,
        out_indices,
        tokens,
        dim,
        experts,
        topk,
        x.stride(0),
        scores_bf16.stride(0),
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


def fused_mega_moe_gate_pack_supported(
    *,
    x: torch.Tensor,
    scores_bf16: torch.Tensor,
    topk: int,
    score_func: str,
) -> bool:
    return (
        triton is not None
        and x.is_cuda
        and x.dtype == torch.bfloat16
        and x.dim() == 2
        and x.shape[1] % 128 == 0
        and scores_bf16.is_cuda
        and scores_bf16.dtype == torch.bfloat16
        and scores_bf16.dim() == 2
        and scores_bf16.shape[0] == x.shape[0]
        and 1 <= int(topk) <= 32
        and score_func == "sqrtsoftplus"
    )
