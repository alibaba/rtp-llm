import functools
import os
from contextlib import contextmanager
from typing import Any, Callable, Generator, List, NoReturn, Optional, Tuple

import torch
import triton
import triton.language as tl

from rtp_llm.utils.module_util import has_module, resolve_symbol

__all__ = [
    "fp8_gemm_nt",
    "m_grouped_fp8_gemm_nt_contiguous",
    "m_grouped_fp8_gemm_nt_masked",
    "bf16_gemm_nt",
    "m_grouped_bf16_gemm_nt_contiguous",
    "m_grouped_bf16_gemm_nt_masked",
    "fp8_fp4_gemm_nt",
    "m_grouped_fp8_fp4_gemm_nt_contiguous",
    "m_grouped_fp8_fp4_gemm_nt_masked",
    "fp8_fp4_paged_mqa_logits",
    "per_token_cast_to_fp4",
    "cast_back_from_fp4",
    "transpose_packed_fp4",
    "tf32_hc_prenorm_gemm",
    "has_deep_gemm",
    "is_deep_gemm_e8m0_used",
    "configure_deep_gemm_num_sms",
    "maybe_pack_ue8m0_scale",
]

_deep_gemm_impl_new_map = {
    "fp8_gemm_nt": "fp8_gemm_nt",
    "m_grouped_fp8_gemm_nt_contiguous": "m_grouped_fp8_gemm_nt_contiguous",
    "m_grouped_fp8_gemm_nt_masked": "m_grouped_fp8_gemm_nt_masked",
    "bf16_gemm_nt": "bf16_gemm_nt",
    "m_grouped_bf16_gemm_nt_contiguous": "m_grouped_bf16_gemm_nt_contiguous",
    "m_grouped_bf16_gemm_nt_masked": "m_grouped_bf16_gemm_nt_masked",
    "fp8_fp4_gemm_nt": "fp8_fp4_gemm_nt",
    "m_grouped_fp8_fp4_gemm_nt_contiguous": "m_grouped_fp8_fp4_gemm_nt_contiguous",
    "m_grouped_fp8_fp4_gemm_nt_masked": "m_grouped_fp8_fp4_gemm_nt_masked",
    "fp8_fp4_paged_mqa_logits": "fp8_fp4_paged_mqa_logits",
    "per_token_cast_to_fp4": "per_token_cast_to_fp4",
    "cast_back_from_fp4": "cast_back_from_fp4",
    "transpose_packed_fp4": "transpose_packed_fp4",
    "tf32_hc_prenorm_gemm": "tf32_hc_prenorm_gemm",
}

_deep_gemm_impl_old_map = {
    "fp8_gemm_nt": "fp8_gemm_nt",
    "m_grouped_fp8_gemm_nt_contiguous": "m_grouped_fp8_gemm_nt_contiguous",
    "m_grouped_fp8_gemm_nt_masked": "fp8_m_grouped_gemm_nt_masked",
    "bf16_gemm_nt": "bf16_gemm_nt",
    "m_grouped_bf16_gemm_nt_contiguous": "m_grouped_bf16_gemm_nt_contiguous",
    "m_grouped_bf16_gemm_nt_masked": "m_grouped_bf16_gemm_nt_masked",
    "fp8_fp4_gemm_nt": "fp8_fp4_gemm_nt",
    "m_grouped_fp8_fp4_gemm_nt_contiguous": "m_grouped_fp8_fp4_gemm_nt_contiguous",
    "m_grouped_fp8_fp4_gemm_nt_masked": "m_grouped_fp8_fp4_gemm_nt_masked",
    "fp8_fp4_paged_mqa_logits": "fp8_fp4_paged_mqa_logits",
    "per_token_cast_to_fp4": "per_token_cast_to_fp4",
    "cast_back_from_fp4": "cast_back_from_fp4",
    "transpose_packed_fp4": "transpose_packed_fp4",
    "tf32_hc_prenorm_gemm": "tf32_hc_prenorm_gemm",
}


_fp8_gemm_nt_impl: Callable[..., Any] | None = None
_m_grouped_fp8_gemm_nt_contiguous_impl: Callable[..., Any] | None = None
_m_grouped_fp8_gemm_nt_masked_impl: Callable[..., Any] | None = None
_bf16_gemm_nt_impl: Callable[..., Any] | None = None
_m_grouped_bf16_gemm_nt_contiguous_impl: Callable[..., Any] | None = None
_m_grouped_bf16_gemm_nt_masked_impl: Callable[..., Any] | None = None
_fp8_fp4_gemm_nt_impl: Callable[..., Any] | None = None
_m_grouped_fp8_fp4_gemm_nt_contiguous_impl: Callable[..., Any] | None = None
_m_grouped_fp8_fp4_gemm_nt_masked_impl: Callable[..., Any] | None = None
_fp8_fp4_paged_mqa_logits_impl: Callable[..., Any] | None = None
_per_token_cast_to_fp4_impl: Callable[..., Any] | None = None
_cast_back_from_fp4_impl: Callable[..., Any] | None = None
_transpose_packed_fp4_impl: Callable[..., Any] | None = None
_tf32_hc_prenorm_gemm_impl: Callable[..., Any] | None = None


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes", "on")


def _disable_dynamo_capture_for_dsv4_graphfx(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Keep DeepGEMM barrier calls outside DSV4 GraphFX compile regions.

    DeepGEMM kernels may synchronize ranks through device-side NVLink barriers.
    The DSV4 GraphFX fusion pass compiles full Python-model forward with
    dynamic request shapes, so letting Dynamo capture DeepGEMM calls can make
    one rank spend extra time compiling while peer ranks are already waiting in
    the next barrier.  With DSV4 GraphFX enabled, force these wrappers to be
    graph breaks; the FX passes still rewrite the adjacent RMSNorm/RoPE/Quant
    segments, and DeepGEMM remains the real producer/consumer in the timeline.
    """
    if not _env_flag("DSV4_GRAPHFX_FUSION"):
        return fn
    try:
        disable = getattr(getattr(torch, "compiler", None), "disable", None)
        if disable is None:
            import torch._dynamo as dynamo

            disable = dynamo.disable
        return disable(fn, recursive=True)
    except TypeError:
        return disable(fn)  # type: ignore[misc]
    except Exception:
        return fn


@functools.cache
def has_deep_gemm() -> bool:
    """Whether the optional `deep_gemm` package is available."""
    return has_module("deep_gemm")


@functools.cache
def is_deep_gemm_e8m0_used() -> bool:
    return torch.cuda.get_device_capability()[0] in [10, 12]


@contextmanager
def configure_deep_gemm_num_sms(num_sms: int) -> Generator[None, None, None]:
    """Configure the number of sms for deep gemm."""
    if not has_deep_gemm():
        raise RuntimeError(
            "DeepGEMM is not available. Please install the `deep_gemm` package to enable DeepGEMM kernels."
        )
    import deep_gemm

    # get original num sms
    original_num_sms = deep_gemm.get_num_sms()
    # set num sms
    deep_gemm.set_num_sms(num_sms)
    try:
        yield
    finally:
        # restore original num sms
        deep_gemm.set_num_sms(original_num_sms)


def _missing_deep_gemm() -> NoReturn:
    """Placeholder for unavailable DeepGEMM package."""
    raise RuntimeError(
        "DeepGEMM is not available. Please install the `deep_gemm` package to enable DeepGEMM kernels."
    )


def _lazy_init_deep_gemm(symbols: List[str]) -> None:
    """Import deep_gemm and resolve symbols on first use."""
    global _fp8_gemm_nt_impl, _m_grouped_fp8_gemm_nt_contiguous_impl, _m_grouped_fp8_gemm_nt_masked_impl
    global _bf16_gemm_nt_impl, _m_grouped_bf16_gemm_nt_contiguous_impl, _m_grouped_bf16_gemm_nt_masked_impl
    global _fp8_fp4_gemm_nt_impl, _m_grouped_fp8_fp4_gemm_nt_contiguous_impl, _m_grouped_fp8_fp4_gemm_nt_masked_impl
    global _fp8_fp4_paged_mqa_logits_impl, _per_token_cast_to_fp4_impl
    global _cast_back_from_fp4_impl, _transpose_packed_fp4_impl
    global _tf32_hc_prenorm_gemm_impl

    symbol_impls = [f"_{symbol}_impl" for symbol in symbols]
    # check if the symbols are valid
    if any(symbol not in _deep_gemm_impl_new_map for symbol in symbols):
        raise ValueError(f"Invalid symbols: {symbols}")
    if all(
        getattr(globals(), symbol_impl, None) is not None
        for symbol_impl in symbol_impls
    ):
        # already initialized
        return
    if not has_deep_gemm():
        # deep_gemm is not available
        return

    import deep_gemm

    # resolve symbols
    for i, symbol in enumerate(symbols):
        symbol_impl = symbol_impls[i]
        try:
            globals()[symbol_impl] = resolve_symbol(
                deep_gemm,
                _deep_gemm_impl_new_map[symbol],
                _deep_gemm_impl_old_map[symbol],
            )
        except AttributeError:
            raise RuntimeError(
                f"DeepGEMM symbol {_deep_gemm_impl_new_map[symbol]} and "
                f"{_deep_gemm_impl_old_map[symbol]} not found in deep_gemm module"
            )


def _lazy_init_deep_gemm_once():
    _lazy_init_deep_gemm(
        [
            "fp8_gemm_nt",
            "m_grouped_fp8_gemm_nt_contiguous",
            "m_grouped_fp8_gemm_nt_masked",
            "bf16_gemm_nt",
            "m_grouped_bf16_gemm_nt_contiguous",
            "m_grouped_bf16_gemm_nt_masked",
            "fp8_fp4_gemm_nt",
            "m_grouped_fp8_fp4_gemm_nt_contiguous",
            "m_grouped_fp8_fp4_gemm_nt_masked",
            "fp8_fp4_paged_mqa_logits",
            "per_token_cast_to_fp4",
            "cast_back_from_fp4",
            "transpose_packed_fp4",
        ]
    )


_lazy_init_deep_gemm_once()


@triton.jit
def pack_ue8m0_kernel_vectorized(
    scale_ptr,
    output_ptr,
    M,
    K,
    K_packed,
    stride_scale_b,
    stride_scale_m,
    stride_scale_k,
    stride_out_b,
    stride_out_k_packed,
    stride_out_m,
    gran_mn: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K_PACKED: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_b = tl.program_id(2)

    # Compute starting offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k_packed = pid_k * BLOCK_K_PACKED + tl.arange(0, BLOCK_K_PACKED)

    # K offset for loading 4 elements per packed output
    # Shape: (BLOCK_K_PACKED, 4)
    offs_k = offs_k_packed[:, None] * 4 + tl.arange(0, 4)[None, :]

    # Scale row indices
    row_idxs = offs_m // gran_mn

    # Compute scale pointers with shape (BLOCK_M, BLOCK_K_PACKED, 4)
    scale_ptrs = (
        scale_ptr
        + pid_b * stride_scale_b
        + row_idxs[:, None, None] * stride_scale_m
        + offs_k[None, :, :] * stride_scale_k
    )

    # Masks
    mask_m = offs_m < M
    mask_k = offs_k < K
    mask = mask_m[:, None, None] & mask_k[None, :, :]

    # Load scale values
    vals = tl.load(scale_ptrs, mask=mask, other=0.0)

    # Convert to UE8M0 using bitcast and shift
    vals_i32 = vals.to(tl.int32, bitcast=True)
    # Extract exponent (8 bits) and mask to ensure only 8 bits
    exponents = (vals_i32 >> 23) & 0xFF

    # Pack 4 bytes into int32 using vectorized shifts
    # exponents shape: (BLOCK_M, BLOCK_K_PACKED, 4)
    # We want to pack along the last dimension

    # Create shift amounts: [0, 8, 16, 24]
    shifts = tl.arange(0, 4)[None, None, :] * 8

    # Shift each exponent to its position and combine
    shifted = exponents << shifts
    # Sum along the last axis to pack
    packed = tl.sum(shifted, axis=2).to(tl.int32)

    # Compute output pointers
    out_ptrs = (
        output_ptr
        + pid_b * stride_out_b
        + offs_k_packed[None, :] * stride_out_k_packed
        + offs_m[:, None] * stride_out_m
    )

    # Output mask
    mask_out = mask_m[:, None] & (offs_k_packed[None, :] < K_packed)

    tl.store(out_ptrs, packed, mask=mask_out)


@triton.jit
def pack_ue8m0_kernel_gran1(
    scale_ptr,
    output_ptr,
    M,
    K,
    K_packed,
    stride_scale_b,
    stride_scale_m,
    stride_scale_k,
    stride_out_b,
    stride_out_k_packed,
    stride_out_m,
    BLOCK_M: tl.constexpr,
    BLOCK_K_PACKED: tl.constexpr,
):
    """
    Specialized kernel for gran_mn=1 case (most common).

    When gran_mn=1, each M row maps directly to a scale row,
    allowing for simplified and faster memory access.
    """
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_b = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k_packed = pid_k * BLOCK_K_PACKED + tl.arange(0, BLOCK_K_PACKED)

    # For gran_mn=1: row_idxs = offs_m (no division needed)
    offs_k = offs_k_packed[:, None] * 4 + tl.arange(0, 4)[None, :]

    # Direct pointer computation (row = m)
    scale_ptrs = (
        scale_ptr
        + pid_b * stride_scale_b
        + offs_m[:, None, None] * stride_scale_m
        + offs_k[None, :, :] * stride_scale_k
    )

    mask_m = offs_m < M
    mask_k = offs_k < K
    mask = mask_m[:, None, None] & mask_k[None, :, :]

    vals = tl.load(scale_ptrs, mask=mask, other=0.0)

    # Fast UE8M0 conversion and packing
    exponents = (vals.to(tl.int32, bitcast=True) >> 23) & 0xFF
    shifts = tl.arange(0, 4)[None, None, :] * 8
    packed = tl.sum(exponents << shifts, axis=2).to(tl.int32)

    out_ptrs = (
        output_ptr
        + pid_b * stride_out_b
        + offs_k_packed[None, :] * stride_out_k_packed
        + offs_m[:, None] * stride_out_m
    )

    mask_out = mask_m[:, None] & (offs_k_packed[None, :] < K_packed)
    tl.store(out_ptrs, packed, mask=mask_out)


def pack_ue8m0_kernel_launcher(scale: torch.Tensor, gran_mn: int):
    import deep_gemm

    if scale.dim() == 2:
        scale = scale.unsqueeze(0)
        is_2d = True
    else:
        is_2d = False

    B, M_scale, K = scale.shape
    M = M_scale * gran_mn

    # Calculate aligned dimensions
    aligned_mn = deep_gemm.get_tma_aligned_size(M, 4)
    aligned_k = (K + 3) // 4 * 4
    K_packed = aligned_k // 4

    # Allocate output (Column Major)
    # Storage: (B, K_packed, aligned_mn)
    packed_storage = torch.zeros(
        (B, K_packed, aligned_mn), device=scale.device, dtype=torch.int32
    )
    # View as (B, aligned_mn, K_packed)
    packed = packed_storage.transpose(1, 2)

    BLOCK_M = 64
    BLOCK_K_PACKED = 32

    total_elements = BLOCK_M * BLOCK_K_PACKED
    num_warps = min(max(total_elements // 256, 4), 8)

    # Use software pipelining for better memory latency hiding
    num_stages = 2

    grid = (
        triton.cdiv(M, BLOCK_M),
        triton.cdiv(K_packed, BLOCK_K_PACKED),
        B,
    )

    if gran_mn == 1:
        pack_ue8m0_kernel_gran1[grid](
            scale,
            packed,
            M,
            K,
            K_packed,
            scale.stride(0),
            scale.stride(1),
            scale.stride(2),
            packed.stride(0),
            packed.stride(2),
            packed.stride(1),
            BLOCK_M=BLOCK_M,
            BLOCK_K_PACKED=BLOCK_K_PACKED,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    else:
        # Use vectorized kernel for general case
        pack_ue8m0_kernel_vectorized[grid](
            scale,
            packed,
            M,
            K,
            K_packed,
            scale.stride(0),
            scale.stride(1),
            scale.stride(2),
            packed.stride(0),
            packed.stride(2),
            packed.stride(1),
            gran_mn=gran_mn,
            BLOCK_M=BLOCK_M,
            BLOCK_K_PACKED=BLOCK_K_PACKED,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    res = packed[:, :M, :]
    if is_2d:
        return res.squeeze(0)
    return res


@_disable_dynamo_capture_for_dsv4_graphfx
def fp8_gemm_nt(
    a: Tuple[torch.Tensor, torch.Tensor],
    b: Tuple[torch.Tensor, torch.Tensor],
    output: torch.Tensor,
    c: Optional[torch.Tensor] = None,
    compiled_dims: str = "nk",
    disable_ue8m0_cast: Optional[bool] = None,
) -> None:
    """Execute FP8 GEMM (A * B^T).

    Args:
        a (Tuple[torch.Tensor, torch.Tensor]): FP8 data and scales for the first matrix.
        b (Tuple[torch.Tensor, torch.Tensor]): FP8 data and scales for the second matrix.
        output (torch.Tensor): Output tensor.
        c (Optional[torch.Tensor], optional): Optional bias tensor. Defaults to None.
        compiled_dims (str, optional): Compiled dimensions. Defaults to "nk".
        disable_ue8m0_cast (bool, optional): Whether to disable E8M0 type cast for E8M0 scale.
            Defaults to None, which will be set to False if E8M0 scale is used, otherwise True.

    Returns:
        None
    """
    global _fp8_gemm_nt_impl
    if _fp8_gemm_nt_impl is None:
        return _missing_deep_gemm()
    _fp8_gemm_nt_impl(
        a,
        b,
        output,
        c,
        compiled_dims=compiled_dims,
        # normal gemm tmp not use ue8m0 cast default
        disable_ue8m0_cast=(
            disable_ue8m0_cast if disable_ue8m0_cast is not None else True
        ),
    )


@_disable_dynamo_capture_for_dsv4_graphfx
def m_grouped_fp8_gemm_nt_contiguous(
    a: Tuple[torch.Tensor, torch.Tensor],
    b: Tuple[torch.Tensor, torch.Tensor],
    output: torch.Tensor,
    m_indices: torch.Tensor,
    compiled_dims: str = "nk",
    disable_ue8m0_cast: Optional[bool] = None,
) -> None:
    """Execute grouped FP8 GEMM (A * B^T) with contiguous layout.

    Args:
        a (Tuple[torch.Tensor, torch.Tensor]): FP8 data and scales for the first matrix with contiguous layout.
        b (Tuple[torch.Tensor, torch.Tensor]): FP8 data and scales for the second matrix.
        output (torch.Tensor): Output tensor.
        m_indices (torch.Tensor): Grouped indices for valid tokens in each group.
            The length of m_indices is the a[0].shape[0], and the corresponding value of valid tokens is group_idx.
        compiled_dims (str, optional): Compiled dimensions. Defaults to "nk".
        disable_ue8m0_cast (bool, optional): Whether to disable E8M0 type cast for E8M0 scale.
            Defaults to None, which will be set to False if E8M0 scale is used, otherwise True.
    """

    global _m_grouped_fp8_gemm_nt_contiguous_impl
    if _m_grouped_fp8_gemm_nt_contiguous_impl is None:
        return _missing_deep_gemm()
    _m_grouped_fp8_gemm_nt_contiguous_impl(
        a,
        b,
        output,
        m_indices,
        compiled_dims=compiled_dims,
        disable_ue8m0_cast=(
            disable_ue8m0_cast
            if disable_ue8m0_cast is not None
            else not is_deep_gemm_e8m0_used()
        ),
    )


def maybe_pack_ue8m0_scale(
    x: torch.Tensor, scale: torch.Tensor, disable_ue8m0_cast: bool
) -> torch.Tensor:
    # check pack conditions:
    # 1. sm=100
    # 2. sf.scalar_type() == torch::kFloat
    # 3. not disable_ue8m0_cast
    # 4. num_groups > 1
    arch_major, _ = torch.cuda.get_device_capability()
    if arch_major != 10:
        return scale
    if scale.dtype != torch.float32:
        return scale
    if disable_ue8m0_cast:
        return scale
    if scale.dim() != 3 or scale.shape[0] < 2:
        return scale

    gran_mn = x.shape[-2] // scale.shape[-2]
    if gran_mn != 1 and gran_mn != 128:
        return scale

    return pack_ue8m0_kernel_launcher(scale, gran_mn)


@_disable_dynamo_capture_for_dsv4_graphfx
def m_grouped_fp8_gemm_nt_masked(
    a: Tuple[torch.Tensor, torch.Tensor],
    b: Tuple[torch.Tensor, torch.Tensor],
    output: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m: int,
    compiled_dims: str = "nk",
    disable_ue8m0_cast: Optional[bool] = None,
) -> None:
    """Execute grouped FP8 GEMM (A * B^T) with masked layout.

    Args:
        a (Tuple[torch.Tensor, torch.Tensor]): FP8 data and scales for the first matrix with masked layout.
        b (Tuple[torch.Tensor, torch.Tensor]): FP8 data and scales for the second matrix.
        output (torch.Tensor): Output tensor.
        masked_m (torch.Tensor): the number of valid tokens in each group.
        expected_m (int): Expected number of valid tokens in each group.
        compiled_dims (str, optional): Compiled dimensions. Defaults to "nk".
        disable_ue8m0_cast (bool, optional): Whether to disable E8M0 type cast for E8M0 scale.
            Defaults to None, which will be set to False if E8M0 scale is used, otherwise True.
    """
    global _m_grouped_fp8_gemm_nt_masked_impl
    if _m_grouped_fp8_gemm_nt_masked_impl is None:
        return _missing_deep_gemm()

    disable_ue8m0_cast = (
        disable_ue8m0_cast
        if disable_ue8m0_cast is not None
        else not is_deep_gemm_e8m0_used()
    )

    a = (a[0], maybe_pack_ue8m0_scale(a[0], a[1], disable_ue8m0_cast))
    b = (b[0], maybe_pack_ue8m0_scale(b[0], b[1], disable_ue8m0_cast))

    _m_grouped_fp8_gemm_nt_masked_impl(
        a,
        b,
        output,
        masked_m,
        expected_m,
        compiled_dims=compiled_dims,
        disable_ue8m0_cast=disable_ue8m0_cast,
    )


@_disable_dynamo_capture_for_dsv4_graphfx
def bf16_gemm_nt(
    a: torch.Tensor,
    b: torch.Tensor,
    output: torch.Tensor,
    c: Optional[torch.Tensor] = None,
    compiled_dims: str = "nk",
) -> None:
    """Execute BF16 GEMM (A * B^T).

    Args:
        a (torch.Tensor): BF16 data for the first matrix.
        b (torch.Tensor): BF16 data for the second matrix.
        output (torch.Tensor): Output tensor.
        c (Optional[torch.Tensor], optional): Optional bias tensor. Defaults to None.
        compiled_dims (str, optional): Compiled dimensions. Defaults to "nk".
    """
    global _bf16_gemm_nt_impl
    if _bf16_gemm_nt_impl is None:
        return _missing_deep_gemm()
    _bf16_gemm_nt_impl(a, b, output, c, compiled_dims)


@_disable_dynamo_capture_for_dsv4_graphfx
def m_grouped_bf16_gemm_nt_contiguous(
    a: torch.Tensor,
    b: torch.Tensor,
    output: torch.Tensor,
    m_indices: torch.Tensor,
    compiled_dims: str = "nk",
) -> None:
    """Execute grouped BF16 GEMM (A * B^T) with contiguous layout.

    Args:
        a (torch.Tensor): BF16 data for the first matrix with contiguous layout.
        b (torch.Tensor): BF16 data for the second matrix.
        output (torch.Tensor): Output tensor.
        m_indices (torch.Tensor): Grouped indices for valid tokens in each group.
            The length of m_indices is the a.shape[0], and the corresponding value of valid tokens is group_idx.
        compiled_dims (str, optional): Compiled dimensions. Defaults to "nk".
    """
    global _m_grouped_bf16_gemm_nt_contiguous_impl
    if _m_grouped_bf16_gemm_nt_contiguous_impl is None:
        return _missing_deep_gemm()
    _m_grouped_bf16_gemm_nt_contiguous_impl(
        a,
        b,
        output,
        m_indices,
        compiled_dims,
    )


@_disable_dynamo_capture_for_dsv4_graphfx
def m_grouped_bf16_gemm_nt_masked(
    a: torch.Tensor,
    b: torch.Tensor,
    output: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m: int,
    compiled_dims: str = "nk",
) -> None:
    """Execute grouped BF16 GEMM (A * B^T) with masked layout.

    Args:
        a (torch.Tensor): BF16 data for the first matrix with masked layout.
        b (torch.Tensor): BF16 data for the second matrix.
        output (torch.Tensor): Output tensor.
        masked_m (torch.Tensor): the number of valid tokens in each group.
        expected_m (int): Expected number of valid tokens in each group.
        compiled_dims (str, optional): Compiled dimensions. Defaults to "nk".
    """
    global _m_grouped_bf16_gemm_nt_masked_impl
    if _m_grouped_bf16_gemm_nt_masked_impl is None:
        return _missing_deep_gemm()
    _m_grouped_bf16_gemm_nt_masked_impl(
        a,
        b,
        output,
        masked_m,
        expected_m,
        compiled_dims,
    )


def _require_sm100_packed_scale_for_fp8_fp4(
    a: Tuple[torch.Tensor, torch.Tensor],
    b: Tuple[torch.Tensor, torch.Tensor],
) -> None:
    if not torch.cuda.is_available():
        return
    arch_major, _ = torch.cuda.get_device_capability()
    if arch_major != 10 or not is_deep_gemm_e8m0_used():
        return
    if a[1].dtype != torch.int32 or b[1].dtype != torch.int32:
        raise RuntimeError(
            "SM100 FP8xFP4 DeepGEMM calls require prepacked int32 scales; "
            f"got a_scale={a[1].dtype}, b_scale={b[1].dtype}"
        )


# --- FP8 activation × FP4 weight (UE8M0 block-scale) ---
#
# DeepGEMM's fp8_fp4 family consumes packed-int8 FP4 weights (2 FP4/byte)
# with UE8M0 block-32 scale along K, matching the DeepSeek-native FP4
# recipe shipped by V3.2/V4 routed experts. Activation is FP8 e4m3fn with
# per-token block-128 UE8M0 scale. SM100 only.


@_disable_dynamo_capture_for_dsv4_graphfx
def fp8_fp4_gemm_nt(
    a: Tuple[torch.Tensor, torch.Tensor],
    b: Tuple[torch.Tensor, torch.Tensor],
    output: torch.Tensor,
    c: Optional[torch.Tensor] = None,
    recipe: Optional[Tuple[int, int, int]] = None,
    recipe_a: Optional[Tuple[int, int]] = None,
    recipe_b: Optional[Tuple[int, int]] = None,
    compiled_dims: str = "nk",
    disable_ue8m0_cast: Optional[bool] = None,
) -> None:
    """Dense FP8-act × packed-FP4-weight GEMM with UE8M0 block scales."""
    global _fp8_fp4_gemm_nt_impl
    if _fp8_fp4_gemm_nt_impl is None:
        return _missing_deep_gemm()
    _require_sm100_packed_scale_for_fp8_fp4(a, b)
    _fp8_fp4_gemm_nt_impl(
        a,
        b,
        output,
        c,
        recipe=recipe,
        recipe_a=recipe_a,
        recipe_b=recipe_b,
        compiled_dims=compiled_dims,
        disable_ue8m0_cast=(
            disable_ue8m0_cast
            if disable_ue8m0_cast is not None
            else not is_deep_gemm_e8m0_used()
        ),
    )


@_disable_dynamo_capture_for_dsv4_graphfx
def m_grouped_fp8_fp4_gemm_nt_contiguous(
    a: Tuple[torch.Tensor, torch.Tensor],
    b: Tuple[torch.Tensor, torch.Tensor],
    output: torch.Tensor,
    grouped_layout: torch.Tensor,
    recipe: Optional[Tuple[int, int, int]] = None,
    recipe_a: Optional[Tuple[int, int]] = None,
    recipe_b: Optional[Tuple[int, int]] = None,
    compiled_dims: str = "nk",
    disable_ue8m0_cast: Optional[bool] = None,
    use_psum_layout: bool = False,
    expected_m_for_psum_layout: Optional[int] = None,
) -> None:
    """Grouped FP8×FP4 GEMM with contiguous per-expert token layout.

    `grouped_layout` is a `[num_tokens]` int tensor: `grouped_layout[i]` is
    the expert index owning token `i`. Tokens must already be permuted so
    each expert's rows are contiguous.
    """
    global _m_grouped_fp8_fp4_gemm_nt_contiguous_impl
    if _m_grouped_fp8_fp4_gemm_nt_contiguous_impl is None:
        return _missing_deep_gemm()
    _require_sm100_packed_scale_for_fp8_fp4(a, b)
    _m_grouped_fp8_fp4_gemm_nt_contiguous_impl(
        a,
        b,
        output,
        grouped_layout,
        recipe=recipe,
        recipe_a=recipe_a,
        recipe_b=recipe_b,
        compiled_dims=compiled_dims,
        disable_ue8m0_cast=(
            disable_ue8m0_cast
            if disable_ue8m0_cast is not None
            else not is_deep_gemm_e8m0_used()
        ),
        use_psum_layout=use_psum_layout,
        expected_m_for_psum_layout=expected_m_for_psum_layout,
    )


@_disable_dynamo_capture_for_dsv4_graphfx
def m_grouped_fp8_fp4_gemm_nt_masked(
    a: Tuple[torch.Tensor, torch.Tensor],
    b: Tuple[torch.Tensor, torch.Tensor],
    output: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m: int,
    recipe: Optional[Tuple[int, int, int]] = None,
    recipe_a: Optional[Tuple[int, int]] = None,
    recipe_b: Optional[Tuple[int, int]] = None,
    compiled_dims: str = "nk",
    disable_ue8m0_cast: Optional[bool] = None,
) -> None:
    """Grouped FP8×FP4 GEMM with masked layout (data-dependent per-expert
    token counts; avoids a D2H sync, suitable for decode)."""
    global _m_grouped_fp8_fp4_gemm_nt_masked_impl
    if _m_grouped_fp8_fp4_gemm_nt_masked_impl is None:
        return _missing_deep_gemm()
    _require_sm100_packed_scale_for_fp8_fp4(a, b)
    _m_grouped_fp8_fp4_gemm_nt_masked_impl(
        a,
        b,
        output,
        masked_m,
        expected_m,
        recipe=recipe,
        recipe_a=recipe_a,
        recipe_b=recipe_b,
        compiled_dims=compiled_dims,
        disable_ue8m0_cast=(
            disable_ue8m0_cast
            if disable_ue8m0_cast is not None
            else not is_deep_gemm_e8m0_used()
        ),
    )


@_disable_dynamo_capture_for_dsv4_graphfx
def fp8_fp4_paged_mqa_logits(
    q: Tuple[torch.Tensor, Optional[torch.Tensor]],
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_table: torch.Tensor,
    schedule_meta: torch.Tensor,
    max_context_len: int,
    clean_logits: bool = False,
    logits_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """FP8-query × FP4-packed paged-KV MQA logits — same kernel V3.2 DSA
    uses for the lightning indexer score step."""
    global _fp8_fp4_paged_mqa_logits_impl
    if _fp8_fp4_paged_mqa_logits_impl is None:
        return _missing_deep_gemm()
    return _fp8_fp4_paged_mqa_logits_impl(
        q,
        kv_cache,
        weights,
        context_lens,
        block_table,
        schedule_meta,
        max_context_len,
        clean_logits,
        logits_dtype,
    )


def per_token_cast_to_fp4(*args: Any, **kwargs: Any) -> Any:
    """DeepGEMM helper: cast BF16 activations to packed-FP4 + UE8M0 scale.
    Thin passthrough — signature/kwargs owned by deep_gemm."""
    global _per_token_cast_to_fp4_impl
    if _per_token_cast_to_fp4_impl is None:
        return _missing_deep_gemm()
    return _per_token_cast_to_fp4_impl(*args, **kwargs)


def cast_back_from_fp4(*args: Any, **kwargs: Any) -> Any:
    """DeepGEMM helper: dequant packed-FP4 → BF16 (debug/inspection path)."""
    global _cast_back_from_fp4_impl
    if _cast_back_from_fp4_impl is None:
        return _missing_deep_gemm()
    return _cast_back_from_fp4_impl(*args, **kwargs)


def transpose_packed_fp4(*args: Any, **kwargs: Any) -> Any:
    """DeepGEMM helper: transpose a packed-FP4 weight in its int8 storage,
    preserving nibble ordering."""
    global _transpose_packed_fp4_impl
    if _transpose_packed_fp4_impl is None:
        return _missing_deep_gemm()
    return _transpose_packed_fp4_impl(*args, **kwargs)


@_disable_dynamo_capture_for_dsv4_graphfx
def tf32_hc_prenorm_gemm(
    x: torch.Tensor,
    fn: torch.Tensor,
    out: torch.Tensor,
    sqrsum: torch.Tensor,
    num_split: int,
) -> torch.Tensor:
    """DeepGEMM mHC pre helper: out=x.float()@fn.T and sqrsum=x^2.

    This symbol is optional in DeepGEMM, so it is not part of the eager
    wrapper initialization used by the unrelated GEMM paths.
    """
    global _tf32_hc_prenorm_gemm_impl
    if _tf32_hc_prenorm_gemm_impl is None:
        _lazy_init_deep_gemm(["tf32_hc_prenorm_gemm"])
    if _tf32_hc_prenorm_gemm_impl is None:
        return _missing_deep_gemm()
    return _tf32_hc_prenorm_gemm_impl(x, fn, out, sqrsum, num_split)
