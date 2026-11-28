"""Fused DSV4 Q LoRA RMSNorm + FP8 E4M3 UE8M0 group-128 quantization."""

from __future__ import annotations

from typing import Optional

import torch

try:
    import triton
    import triton.language as tl
except Exception:
    triton = None
    tl = None

_UE8M0_GROUP_SIZE = 128
_DEFAULT_CLAMP_EPS = 1.0e-4
_FP8_DTYPE = torch.float8_e4m3fn
_FP8_MAX = float(torch.finfo(_FP8_DTYPE).max)
_FP8_MIN = float(torch.finfo(_FP8_DTYPE).min)
_SMALL_D_MAX = 2048
_DEFAULT_CHUNK_D = 1024


if triton is not None:

    @triton.jit(
        do_not_specialize=[
            "M",
            "x_stride_m",
            "out_norm_stride_m",
            "out_q_stride_m",
            "out_scale_stride_k",
        ]
    )
    def _fused_qnorm_fp8_ue8m0_small_d_kernel(
        x_ptr,
        w_ptr,
        out_norm_ptr,
        out_q_ptr,
        out_scale_ptr,
        M,
        x_stride_m,
        out_norm_stride_m,
        out_q_stride_m,
        out_scale_stride_k,
        EPS: tl.constexpr,
        CLAMP_EPS: tl.constexpr,
        FP8_MIN: tl.constexpr,
        FP8_MAX: tl.constexpr,
        D: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_M: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
        BLOCK_SCALE_PACKS: tl.constexpr,
        ACTIVE_GROUPS: tl.constexpr,
        ACTIVE_SCALE_PACKS: tl.constexpr,
        EMIT_NORM: tl.constexpr,
        HAS_TAIL_PAD: tl.constexpr,
    ):
        row_base = tl.program_id(0).to(tl.int64) * BLOCK_M
        offs_m = row_base + tl.arange(0, BLOCK_M)
        row_mask = offs_m < M
        offs_d = tl.arange(0, BLOCK_D)
        d_mask = offs_d < D
        io_mask = row_mask[:, None] & d_mask[None, :]

        x = tl.load(
            x_ptr + offs_m[:, None] * x_stride_m + offs_d[None, :],
            mask=io_mask,
            other=0.0,
        ).to(tl.float32)

        inv = tl.rsqrt(tl.sum(x * x, axis=1) / D + EPS)
        w = tl.load(w_ptr + offs_d, mask=d_mask, other=0.0).to(tl.float32)
        # Preserve the two-step BF16 activation boundary before FP8 quantization.
        y = (x * inv[:, None] * w[None, :]).to(tl.bfloat16)
        if EMIT_NORM:
            tl.store(
                out_norm_ptr + offs_m[:, None] * out_norm_stride_m + offs_d[None, :],
                y,
                mask=io_mask,
            )

        y_f32 = y.to(tl.float32)
        y_grouped = tl.reshape(y_f32, (BLOCK_M, BLOCK_SCALE_PACKS, 4, GROUP_SIZE))
        absmax = tl.max(tl.abs(y_grouped), axis=3)
        scale_raw = tl.maximum(absmax, CLAMP_EPS) * (1.0 / FP8_MAX)
        raw_bits = scale_raw.to(tl.int32, bitcast=True)
        mant = raw_bits & 0x7FFFFF
        exp_biased = tl.minimum(
            tl.maximum(((raw_bits >> 23) & 0xFF) + tl.where(mant != 0, 1, 0), 1),
            253,
        )
        inv_scale = ((254 - exp_biased) << 23).to(tl.float32, bitcast=True)
        q = tl.clamp(y_grouped * inv_scale[:, :, :, None], FP8_MIN, FP8_MAX).to(
            out_q_ptr.dtype.element_ty
        )
        tl.store(
            out_q_ptr + offs_m[:, None] * out_q_stride_m + offs_d[None, :],
            tl.reshape(q, (BLOCK_M, BLOCK_D)),
            mask=io_mask,
        )

        if HAS_TAIL_PAD:
            group_ids = (
                tl.arange(0, BLOCK_SCALE_PACKS)[:, None] * 4 + tl.arange(0, 4)[None, :]
            )
            exp_biased = tl.where(group_ids[None, :, :] < ACTIVE_GROUPS, exp_biased, 0)
        pack_shifts = tl.arange(0, 4)[None, None, :] * 8
        packed = tl.sum(exp_biased << pack_shifts, axis=2)

        offs_sp = tl.arange(0, BLOCK_SCALE_PACKS)
        if BLOCK_M == 1:
            packed_1d = tl.reshape(packed, (BLOCK_SCALE_PACKS,))
            tl.store(
                out_scale_ptr + offs_sp.to(tl.int64) * out_scale_stride_k + row_base,
                packed_1d,
                mask=offs_sp < ACTIVE_SCALE_PACKS,
            )
        else:
            # Transpose so the contiguous column-major row index sits on the last axis.
            packed_t = tl.trans(packed)
            scale_mask = (offs_sp[:, None] < ACTIVE_SCALE_PACKS) & row_mask[None, :]
            tl.store(
                out_scale_ptr
                + offs_sp[:, None].to(tl.int64) * out_scale_stride_k
                + offs_m[None, :],
                packed_t,
                mask=scale_mask,
            )

    @triton.jit(
        do_not_specialize=[
            "M",
            "x_stride_m",
            "out_norm_stride_m",
            "out_q_stride_m",
            "out_scale_stride_k",
        ]
    )
    def _fused_qnorm_fp8_ue8m0_chunked_kernel(
        x_ptr,
        w_ptr,
        out_norm_ptr,
        out_q_ptr,
        out_scale_ptr,
        M,
        x_stride_m,
        out_norm_stride_m,
        out_q_stride_m,
        out_scale_stride_k,
        EPS: tl.constexpr,
        CLAMP_EPS: tl.constexpr,
        FP8_MIN: tl.constexpr,
        FP8_MAX: tl.constexpr,
        D: tl.constexpr,
        CHUNK_D: tl.constexpr,
        NUM_CHUNKS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
        PACKS_PER_CHUNK: tl.constexpr,
        ACTIVE_GROUPS: tl.constexpr,
        ACTIVE_SCALE_PACKS: tl.constexpr,
        EMIT_NORM: tl.constexpr,
        HAS_TAIL_PAD: tl.constexpr,
    ):
        row_base = tl.program_id(0).to(tl.int64) * BLOCK_M
        offs_m = row_base + tl.arange(0, BLOCK_M)
        row_mask = offs_m < M
        offs_c = tl.arange(0, CHUNK_D)

        var_vec = tl.zeros((BLOCK_M, CHUNK_D), dtype=tl.float32)
        for chunk_id in range(0, NUM_CHUNKS):
            cols = chunk_id * CHUNK_D + offs_c
            col_mask = cols < D
            load_mask = row_mask[:, None] & col_mask[None, :]
            x_c = tl.load(
                x_ptr + offs_m[:, None] * x_stride_m + cols[None, :],
                mask=load_mask,
                other=0.0,
            ).to(tl.float32)
            var_vec += x_c * x_c

        inv = tl.rsqrt(tl.sum(var_vec, axis=1) / D + EPS)
        pack_shifts = tl.arange(0, 4)[None, None, :] * 8

        for chunk_id in range(0, NUM_CHUNKS):
            cols = chunk_id * CHUNK_D + offs_c
            col_mask = cols < D
            io_mask = row_mask[:, None] & col_mask[None, :]
            x_c = tl.load(
                x_ptr + offs_m[:, None] * x_stride_m + cols[None, :],
                mask=io_mask,
                other=0.0,
            ).to(tl.float32)
            w_c = tl.load(w_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)

            y_c = (x_c * inv[:, None] * w_c[None, :]).to(tl.bfloat16)
            if EMIT_NORM:
                tl.store(
                    out_norm_ptr + offs_m[:, None] * out_norm_stride_m + cols[None, :],
                    y_c,
                    mask=io_mask,
                )

            y_f32 = y_c.to(tl.float32)
            y_grouped = tl.reshape(y_f32, (BLOCK_M, PACKS_PER_CHUNK, 4, GROUP_SIZE))
            absmax = tl.max(tl.abs(y_grouped), axis=3)
            scale_raw = tl.maximum(absmax, CLAMP_EPS) * (1.0 / FP8_MAX)
            raw_bits = scale_raw.to(tl.int32, bitcast=True)
            mant = raw_bits & 0x7FFFFF
            exp_biased = tl.minimum(
                tl.maximum(((raw_bits >> 23) & 0xFF) + tl.where(mant != 0, 1, 0), 1),
                253,
            )
            inv_scale = ((254 - exp_biased) << 23).to(tl.float32, bitcast=True)
            q_c = tl.clamp(y_grouped * inv_scale[:, :, :, None], FP8_MIN, FP8_MAX).to(
                out_q_ptr.dtype.element_ty
            )
            tl.store(
                out_q_ptr + offs_m[:, None] * out_q_stride_m + cols[None, :],
                tl.reshape(q_c, (BLOCK_M, CHUNK_D)),
                mask=io_mask,
            )

            pack_base = chunk_id * PACKS_PER_CHUNK
            if HAS_TAIL_PAD:
                group_ids = (pack_base + tl.arange(0, PACKS_PER_CHUNK))[
                    :, None
                ] * 4 + tl.arange(0, 4)[None, :]
                exp_biased = tl.where(
                    group_ids[None, :, :] < ACTIVE_GROUPS, exp_biased, 0
                )
            packed = tl.sum(exp_biased << pack_shifts, axis=2)

            offs_sp = pack_base + tl.arange(0, PACKS_PER_CHUNK)
            if BLOCK_M == 1:
                packed_1d = tl.reshape(packed, (PACKS_PER_CHUNK,))
                tl.store(
                    out_scale_ptr
                    + offs_sp.to(tl.int64) * out_scale_stride_k
                    + row_base,
                    packed_1d,
                    mask=offs_sp < ACTIVE_SCALE_PACKS,
                )
            else:
                packed_t = tl.trans(packed)
                scale_mask = (offs_sp[:, None] < ACTIVE_SCALE_PACKS) & row_mask[None, :]
                tl.store(
                    out_scale_ptr
                    + offs_sp[:, None].to(tl.int64) * out_scale_stride_k
                    + offs_m[None, :],
                    packed_t,
                    mask=scale_mask,
                )


def _make_ue8m0_scale_like(
    rows: int,
    width: int,
    *,
    device: torch.device,
    group_size: int = _UE8M0_GROUP_SIZE,
) -> torch.Tensor:
    """Allocate DeepGEMM TMA-aligned column-major packed int32 UE8M0 scales."""
    scale_cols = width // group_size
    aligned_m = ((rows + 3) // 4) * 4
    aligned_k = ((scale_cols + 3) // 4) * 4
    return torch.empty(
        (aligned_k // 4, aligned_m),
        device=device,
        dtype=torch.int32,
    ).transpose(-1, -2)[:rows, :]


def _select_small_d_launch(rows: int, block_d: int) -> tuple[int, int]:
    if rows <= 64:
        return 1, 8 if block_d >= 4096 else (4 if block_d >= 1024 else 2)
    if block_d <= 1024:
        return 1, 1
    if block_d <= 4096:
        return 1, 2
    return 1, 4 if rows <= 384 else 2


def _select_chunked_launch(rows: int, chunk_d: int) -> tuple[int, int]:
    if rows <= 64:
        return 1, 4
    return 1, 1 if chunk_d <= 1024 else 2


def _is_row_contiguous(t: torch.Tensor) -> bool:
    if t.dim() == 2:
        return t.stride(-1) == 1 and (t.shape[0] <= 1 or t.stride(0) >= t.shape[1])
    return t.is_contiguous()


def _row_stride(t: torch.Tensor) -> int:
    return int(t.stride(0)) if t.dim() == 2 else int(t.shape[-1])


def _has_16b_aligned_rows(t: torch.Tensor) -> bool:
    width = int(t.shape[-1])
    if width <= 0:
        return False
    row_stride = _row_stride(t)
    if (row_stride * t.element_size()) % 16 != 0:
        return False
    rows = int(t.numel() // width)
    if rows == 0:
        return True
    return int(t.data_ptr()) % 16 == 0


def _tensor_span_bytes(t: torch.Tensor) -> tuple[int, int]:
    if t.numel() == 0:
        return 0, 0
    if t.dim() == 2:
        span_elems = (int(t.shape[0]) - 1) * int(t.stride(0)) + int(t.shape[1])
    else:
        span_elems = int(t.numel())
    start = int(t.data_ptr())
    return start, start + span_elems * t.element_size()


def _scale_span_bytes(scale: torch.Tensor) -> tuple[int, int]:
    rows, packs = int(scale.shape[0]), int(scale.shape[1])
    if rows == 0 or packs == 0:
        return 0, 0
    span_elems = (packs - 1) * int(scale.stride(1)) + rows
    start = int(scale.data_ptr())
    return start, start + span_elems * scale.element_size()


def _spans_overlap(span_a: tuple[int, int], span_b: tuple[int, int]) -> bool:
    a_start, a_end = span_a
    b_start, b_end = span_b
    if a_start == a_end or b_start == b_end:
        return False
    return a_start < b_end and b_start < a_end


def _validate_optional_outputs(
    x: torch.Tensor,
    weight: torch.Tensor,
    rows: int,
    width: int,
    scale_packs: int,
    emit_norm: bool,
    out_norm: Optional[torch.Tensor],
    out_q: Optional[torch.Tensor],
    out_scale: Optional[torch.Tensor],
) -> tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
    if out_norm is not None:
        if not emit_norm:
            raise ValueError("out_norm requires emit_norm=True")
        if out_norm.shape != x.shape:
            raise ValueError(
                f"out_norm must have shape {tuple(x.shape)}, got {tuple(out_norm.shape)}"
            )
        if out_norm.dtype != torch.bfloat16:
            raise ValueError(f"out_norm must be bf16, got {out_norm.dtype}")
        if out_norm.device != x.device or not _is_row_contiguous(out_norm):
            raise ValueError("out_norm must be row-contiguous on the same CUDA device")
    elif emit_norm:
        out_norm = torch.empty(x.shape, device=x.device, dtype=x.dtype)

    if out_q is None:
        out_q = torch.empty(x.shape, device=x.device, dtype=_FP8_DTYPE)
    else:
        if out_q.shape != x.shape:
            raise ValueError(
                f"out_q must have shape {tuple(x.shape)}, got {tuple(out_q.shape)}"
            )
        if out_q.dtype != _FP8_DTYPE:
            raise ValueError(f"out_q must be {_FP8_DTYPE}, got {out_q.dtype}")
        if out_q.device != x.device or not _is_row_contiguous(out_q):
            raise ValueError("out_q must be row-contiguous on the same CUDA device")

    if out_scale is None:
        out_scale = _make_ue8m0_scale_like(
            rows, width, device=x.device, group_size=_UE8M0_GROUP_SIZE
        )
    else:
        expected_shape = (rows, scale_packs)
        if tuple(out_scale.shape) != expected_shape:
            raise ValueError(
                f"out_scale must have shape {expected_shape}, got {tuple(out_scale.shape)}"
            )
        if out_scale.dtype != torch.int32 or out_scale.device != x.device:
            raise ValueError("out_scale must be int32 on the same CUDA device")
        if rows > 0 and (out_scale.stride(0) != 1 or out_scale.stride(1) % 4 != 0):
            raise ValueError(
                "out_scale must be column-major TMA-aligned (stride(0)==1, stride(1)%4==0)"
            )

    if rows > 0:
        x_span = _tensor_span_bytes(x)
        w_span = _tensor_span_bytes(weight)
        q_span = _tensor_span_bytes(out_q)
        s_span = _scale_span_bytes(out_scale)
        if _spans_overlap(q_span, x_span) or _spans_overlap(q_span, w_span):
            raise ValueError("out_q must not overlap x or weight")
        if _spans_overlap(s_span, x_span) or _spans_overlap(s_span, w_span):
            raise ValueError("out_scale must not overlap x or weight")
        if _spans_overlap(q_span, s_span):
            raise ValueError("out_q and out_scale must not overlap")
        if out_norm is not None:
            n_span = _tensor_span_bytes(out_norm)
            if _spans_overlap(n_span, w_span):
                raise ValueError("out_norm must not overlap weight")
            if _spans_overlap(n_span, q_span) or _spans_overlap(n_span, s_span):
                raise ValueError("out_norm must not overlap out_q or out_scale")
            if _spans_overlap(n_span, x_span):
                exact_inplace = out_norm.data_ptr() == x.data_ptr() and (
                    rows <= 1 or _row_stride(out_norm) == _row_stride(x)
                )
                if not exact_inplace:
                    raise ValueError(
                        "out_norm overlapping x must be an exact in-place alias"
                    )
    return out_norm, out_q, out_scale


def can_fuse_qnorm_fp8_ue8m0_quant(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    group_size: int = _UE8M0_GROUP_SIZE,
) -> bool:
    """Return True when fused RMSNorm + UE8M0 group quantization supports ``x``."""
    if triton is None:
        return False
    if group_size != _UE8M0_GROUP_SIZE:
        return False
    if not isinstance(x, torch.Tensor) or not isinstance(weight, torch.Tensor):
        return False
    if torch.is_grad_enabled() and (x.requires_grad or weight.requires_grad):
        return False
    if not x.is_cuda or not weight.is_cuda or x.device != weight.device:
        return False
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        return False
    if x.dim() < 2 or not _is_row_contiguous(x) or not weight.is_contiguous():
        return False
    width = int(x.shape[-1])
    if width <= 0 or width % group_size != 0 or tuple(weight.shape) != (width,):
        return False
    if not _has_16b_aligned_rows(x) or int(weight.data_ptr()) % 16 != 0:
        return False
    return True


def fused_rmsnorm_fp8_ue8m0_quant(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    eps: float,
    group_size: int = _UE8M0_GROUP_SIZE,
    clamp_eps: float = _DEFAULT_CLAMP_EPS,
    emit_norm: bool = False,
    out_norm: Optional[torch.Tensor] = None,
    out_q: Optional[torch.Tensor] = None,
    out_scale: Optional[torch.Tensor] = None,
    _block_m: Optional[int] = None,
    _chunk_d: Optional[int] = None,
    _num_warps: Optional[int] = None,
) -> tuple[Optional[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    """Fuse RMSNorm + DeepGEMM UE8M0 group-128 FP8 quantization for ``[..., D]``.

    Returns ``(norm_bf16_or_None, (fp8_q, packed_ue8m0_scale))`` where
    ``fp8_q`` mirrors ``x.shape`` and ``packed_ue8m0_scale`` has 2D shape
    ``(M, ceil_div(D // group_size, 4))`` in TMA-aligned column-major ``int32``
    layout matching ``fp8_gemm_nt`` / ``CudaFp8GEMMLinear.forward_quantized``.
    """
    if triton is None:
        raise RuntimeError("triton is unavailable")
    if group_size != _UE8M0_GROUP_SIZE:
        raise ValueError(f"group_size must be {_UE8M0_GROUP_SIZE}, got {group_size}")
    if x.dim() < 2:
        raise ValueError(f"x must have at least 2 dims [..., D], got {tuple(x.shape)}")
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise ValueError(f"x and weight must be bf16, got {x.dtype} and {weight.dtype}")
    if not x.is_cuda or not weight.is_cuda or x.device != weight.device:
        raise RuntimeError(
            "fused_rmsnorm_fp8_ue8m0_quant requires CUDA tensors on one device"
        )
    if not _is_row_contiguous(x) or not weight.is_contiguous():
        raise ValueError("x must be row-contiguous and weight must be contiguous")

    width = int(x.shape[-1])
    if width <= 0 or width % group_size != 0:
        raise ValueError(f"D={width} must be a positive multiple of {group_size}")
    if tuple(weight.shape) != (width,):
        raise ValueError(
            f"weight must have shape ({width},), got {tuple(weight.shape)}"
        )

    flat_x = x if x.dim() == 2 else x.view(-1, width)
    rows = int(flat_x.shape[0])
    active_groups = width // group_size
    active_scale_packs = (active_groups + 3) // 4

    out_norm, out_q, out_scale = _validate_optional_outputs(
        x,
        weight,
        rows,
        width,
        active_scale_packs,
        emit_norm,
        out_norm,
        out_q,
        out_scale,
    )
    if _block_m is not None and int(_block_m) <= 0:
        raise ValueError(f"_block_m must be positive, got {_block_m}")
    if _num_warps is not None and int(_num_warps) <= 0:
        raise ValueError(f"_num_warps must be positive, got {_num_warps}")
    if rows == 0:
        return out_norm, (out_q, out_scale)

    flat_q = out_q if out_q.dim() == 2 else out_q.view(rows, width)
    flat_norm = (
        (out_norm if out_norm.dim() == 2 else out_norm.view(rows, width))
        if emit_norm
        else flat_q
    )
    norm_stride_m = int(flat_norm.stride(0)) if emit_norm else 0
    scale_stride_k = int(out_scale.stride(1))

    small_d_limit = 8192 if rows <= 768 else (4096 if rows <= 1536 else _SMALL_D_MAX)
    if _chunk_d is None and width <= small_d_limit:
        block_d = max(4 * group_size, triton.next_power_of_2(width))
        block_scale_packs = block_d // (4 * group_size)
        default_m, default_warps = _select_small_d_launch(rows, block_d)
        block_m = _block_m if _block_m is not None else default_m
        num_warps = _num_warps if _num_warps is not None else default_warps
        grid = (triton.cdiv(rows, block_m),)
        _fused_qnorm_fp8_ue8m0_small_d_kernel[grid](
            flat_x,
            weight,
            flat_norm,
            flat_q,
            out_scale,
            rows,
            int(flat_x.stride(0)),
            norm_stride_m,
            int(flat_q.stride(0)),
            scale_stride_k,
            EPS=float(eps),
            CLAMP_EPS=float(clamp_eps),
            FP8_MIN=_FP8_MIN,
            FP8_MAX=_FP8_MAX,
            D=width,
            BLOCK_D=block_d,
            BLOCK_M=block_m,
            GROUP_SIZE=group_size,
            BLOCK_SCALE_PACKS=block_scale_packs,
            ACTIVE_GROUPS=active_groups,
            ACTIVE_SCALE_PACKS=active_scale_packs,
            EMIT_NORM=emit_norm,
            HAS_TAIL_PAD=block_d != width,
            num_warps=num_warps,
            num_stages=2,
        )
    else:
        chunk_d = _DEFAULT_CHUNK_D if _chunk_d is None else int(_chunk_d)
        if chunk_d <= 0 or chunk_d % (4 * group_size) != 0:
            raise ValueError(
                f"chunk_d must be a positive multiple of {4 * group_size}, got {chunk_d}"
            )
        packs_per_chunk = chunk_d // (4 * group_size)
        num_chunks = (width + chunk_d - 1) // chunk_d
        default_m, default_warps = _select_chunked_launch(rows, chunk_d)
        block_m = _block_m if _block_m is not None else default_m
        num_warps = _num_warps if _num_warps is not None else default_warps
        grid = (triton.cdiv(rows, block_m),)
        _fused_qnorm_fp8_ue8m0_chunked_kernel[grid](
            flat_x,
            weight,
            flat_norm,
            flat_q,
            out_scale,
            rows,
            int(flat_x.stride(0)),
            norm_stride_m,
            int(flat_q.stride(0)),
            scale_stride_k,
            EPS=float(eps),
            CLAMP_EPS=float(clamp_eps),
            FP8_MIN=_FP8_MIN,
            FP8_MAX=_FP8_MAX,
            D=width,
            CHUNK_D=chunk_d,
            NUM_CHUNKS=num_chunks,
            BLOCK_M=block_m,
            GROUP_SIZE=group_size,
            PACKS_PER_CHUNK=packs_per_chunk,
            ACTIVE_GROUPS=active_groups,
            ACTIVE_SCALE_PACKS=active_scale_packs,
            EMIT_NORM=emit_norm,
            HAS_TAIL_PAD=(num_chunks * chunk_d) != width,
            num_warps=num_warps,
            num_stages=2,
        )

    return out_norm, (out_q, out_scale)


__all__ = [
    "can_fuse_qnorm_fp8_ue8m0_quant",
    "fused_rmsnorm_fp8_ue8m0_quant",
]
