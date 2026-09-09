"""M890P MXFP4 activation packing and dense W4A4 GEMM primitives.

This module deliberately owns only the lowest-level W4A4 contract.  Expert
selection, MoE execution policy, and qlinear integration belong to their
respective model modules.

The packed activation uses e2m1 nibbles (even K element in the low nibble).
Its per-32 UE8M0 exponents are emitted directly as prepared uint16 pairs in
DeepGEMM's mn-major layout.  The physical K is rounded up to 64 because
``deep_gemm`` consumes scale pairs.  Callers must pad the matching weight to
the same physical K when logical K is not aligned.
"""

from __future__ import annotations

import importlib
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

MXFP4_BLOCK_SIZE = 32
MXFP4_K_ALIGNMENT = 64
_SUPPORTED_ACTIVATION_DTYPES = (torch.bfloat16, torch.float32)


def _ceil_to_multiple(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _require_m890p(tensor: torch.Tensor) -> None:
    if not tensor.is_cuda:
        raise ValueError("M890P MXFP4 kernels require a CUDA-compatible PPU tensor")
    device_name = torch.cuda.get_device_name(tensor.device)
    if device_name != "ZW-M890P":
        raise RuntimeError(f"M890P MXFP4 kernels cannot run on device {device_name!r}")


def _load_deep_gemm():
    try:
        return importlib.import_module("deep_gemm")
    except ImportError as error:
        raise RuntimeError("M890P W4A4 requires the PPU deep_gemm package") from error


def _require_deep_gemm_symbol(module, name: str):
    symbol = getattr(module, name, None)
    if not callable(symbol):
        raise RuntimeError(
            f"installed PPU deep_gemm does not provide callable {name!r}"
        )
    return symbol


def _expected_mn_major_stride(shape: torch.Size) -> Tuple[int, ...]:
    """Stride of [..., rows, scale_pairs] backed by [..., pairs, rows]."""
    rows, pairs = shape[-2:]
    leading_stride = rows * pairs
    leading = []
    for dimension in reversed(shape[:-2]):
        leading.append(leading_stride)
        leading_stride *= dimension
    return tuple(reversed(leading)) + (1, rows)


def _mn_major_layout_error(tensor: torch.Tensor) -> Optional[str]:
    """Return why ``tensor`` is not an address-safe mn-major view, if any."""
    expected_stride = _expected_mn_major_stride(tensor.shape)
    for dimension, (size, stride, expected) in enumerate(
        zip(tensor.shape, tensor.stride(), expected_stride)
    ):
        if size > 1 and stride != expected:
            return (
                f"dimension {dimension} has stride {stride}, expected {expected} "
                f"for shape {tuple(tensor.shape)}"
            )

    storage_offset = tensor.storage_offset()
    if storage_offset < 0 or any(stride < 0 for stride in tensor.stride()):
        return (
            f"negative storage offset/stride is unsupported: "
            f"offset={storage_offset}, stride={tensor.stride()}"
        )
    max_offset = storage_offset + sum(
        (size - 1) * stride for size, stride in zip(tensor.shape, tensor.stride())
    )
    storage_elements = tensor.untyped_storage().nbytes() // tensor.element_size()
    if max_offset >= storage_elements:
        return (
            f"maximum element offset {max_offset} is outside storage with "
            f"{storage_elements} elements"
        )
    return None


@triton.jit
def _downcast_to_mxfp4_kernel(
    packed_ptr,
    packed_stride_m,
    prepared_scale_ptr,
    scale_stride_pair,
    scale_stride_m,
    src_ptr,
    src_stride_m,
    rows,
    logical_k,
    padded_k: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    tl.static_assert(BLOCK_K % 64 == 0)

    block_m = tl.program_id(0).to(tl.int64)
    block_k = tl.program_id(1).to(tl.int64)
    start_m = block_m * BLOCK_M
    start_k = block_k * BLOCK_K

    offsets_m = tl.arange(0, BLOCK_M)[:, None].to(tl.int64)
    offsets_k = tl.arange(0, BLOCK_K)[None, :].to(tl.int64)
    source_mask = (start_m + offsets_m < rows) & (start_k + offsets_k < logical_k)
    source = tl.load(
        src_ptr + (start_m + offsets_m) * src_stride_m + start_k + offsets_k,
        mask=source_mask,
        other=0.0,
    ).to(tl.float32)

    groups_per_block: tl.constexpr = BLOCK_K // 32
    grouped = tl.reshape(source, [BLOCK_M, groups_per_block, 32])
    group_amax = tl.max(tl.abs(grouped), axis=2, keep_dims=True)
    group_amax = tl.maximum(group_amax, 1.0e-10)

    # 2**ceil(log2(amax/6)): rounding the positive float32 significand upward
    # and keeping only exponent bits is bit-identical to the SGLang PPU path.
    scale_f32 = group_amax / 6.0
    scale_bits = (scale_f32.to(tl.uint32, bitcast=True) + 0x007FFFFF) & 0x7F800000
    rounded_scale = scale_bits.to(tl.float32, bitcast=True)
    quantized = tl.reshape(grouped / rounded_scale, [BLOCK_M, BLOCK_K])

    pair_count: tl.constexpr = BLOCK_K // 2
    pairs = tl.reshape(quantized, [BLOCK_M, pair_count, 2])
    even, odd = tl.split(pairs)
    # PPU PTX puts the second source operand into the low nibble.
    packed = tl.inline_asm_elementwise(
        """
        {
            .reg .b8 r;
            cvt.rn.satfinite.e2m1x2.f32 r, $1, $2;
            mov.b32 $0, {r, r, r, r};
        }
        """,
        constraints="=r,f,f",
        args=[odd.to(tl.float32), even.to(tl.float32)],
        dtype=tl.uint8,
        is_pure=True,
        pack=1,
    )

    packed_offsets = tl.arange(0, pair_count)[None, :].to(tl.int64)
    packed_mask = (start_m + offsets_m < rows) & (
        start_k // 2 + packed_offsets < padded_k // 2
    )
    tl.store(
        packed_ptr
        + (start_m + offsets_m) * packed_stride_m
        + start_k // 2
        + packed_offsets,
        packed,
        mask=packed_mask,
    )

    raw_scale = (tl.reshape(scale_bits, [BLOCK_M, groups_per_block]) >> 23).to(tl.uint8)
    group_offsets = start_k // 32 + tl.arange(0, groups_per_block)[None, :]
    # A partial final group keeps the scale of its real elements.  Groups with
    # no logical elements are explicit zero bytes so padding cannot leak data.
    raw_scale = tl.where(group_offsets < (logical_k + 31) // 32, raw_scale, 0)
    scale_pairs_per_block: tl.constexpr = groups_per_block // 2
    raw_scale_pairs = tl.reshape(raw_scale, [BLOCK_M, scale_pairs_per_block, 2])
    low_scale, high_scale = tl.split(raw_scale_pairs)
    prepared_scale = (low_scale.to(tl.uint32) | (high_scale.to(tl.uint32) << 8)).to(
        tl.uint16
    )
    scale_pair_offsets = tl.arange(0, scale_pairs_per_block)[None, :].to(tl.int64)
    scale_mask = (start_m + offsets_m < rows) & (
        start_k // 64 + scale_pair_offsets < padded_k // 64
    )
    tl.store(
        prepared_scale_ptr
        + (start_k // 64 + scale_pair_offsets) * scale_stride_pair
        + (start_m + offsets_m) * scale_stride_m,
        prepared_scale,
        mask=scale_mask,
    )


def downcast_to_mxfp4(
    activation: torch.Tensor,
    *,
    validate_finite: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pack a 2-D BF16/FP32 activation into M890P MXFP4 storage.

    For input ``[M, logical_k]``, returns packed uint8 data
    ``[M, padded_k / 2]`` and prepared uint16 UE8M0 scale pairs with logical
    shape ``[M, padded_k / 64]`` and physical mn-major stride ``(1, M)``,
    where ``padded_k = ceil(logical_k, 64)``.

    The input must already be contiguous; this hot-path primitive never makes
    an implicit copy.  ``validate_finite`` is intended for validation and
    data-ingress checks.  It synchronizes to report NaN/Inf immediately, so hot
    inference paths keep it disabled to avoid host synchronization.
    """
    if activation.ndim != 2:
        raise ValueError(
            f"activation must be two-dimensional [M, K], got {activation.shape}"
        )
    if activation.dtype not in _SUPPORTED_ACTIVATION_DTYPES:
        raise TypeError(
            "activation must have dtype torch.bfloat16 or torch.float32, "
            f"got {activation.dtype}"
        )
    if activation.shape[0] <= 0 or activation.shape[1] <= 0:
        raise ValueError(
            f"activation dimensions must be positive, got {activation.shape}"
        )
    _require_m890p(activation)
    if not activation.is_contiguous():
        raise ValueError("activation must be contiguous")
    if validate_finite and not bool(torch.isfinite(activation).all().item()):
        raise ValueError("activation contains NaN or Inf")

    source = activation
    rows, logical_k = source.shape
    padded_k = _ceil_to_multiple(logical_k, MXFP4_K_ALIGNMENT)
    packed = torch.empty((rows, padded_k // 2), dtype=torch.uint8, device=source.device)
    scale_storage = torch.empty(
        (padded_k // MXFP4_K_ALIGNMENT, rows),
        dtype=torch.uint16,
        device=source.device,
    )

    block_m = 32
    block_k = 128
    grid = (triton.cdiv(rows, block_m), triton.cdiv(padded_k, block_k))
    _downcast_to_mxfp4_kernel[grid](
        packed,
        packed.stride(0),
        scale_storage,
        scale_storage.stride(0),
        scale_storage.stride(1),
        source,
        source.stride(0),
        rows,
        logical_k,
        padded_k=padded_k,
        BLOCK_M=block_m,
        BLOCK_K=block_k,
        num_warps=4,
    )
    return packed, scale_storage.t()


def prepare_fp4_weight_scale_mxfp4(raw_scale: torch.Tensor) -> torch.Tensor:
    """Preprocess checkpoint UE8M0 scale bytes once for PPU DeepGEMM.

    ``raw_scale`` must be contiguous uint8 (or a bitwise
    ``torch.float8_e8m0fnu`` view) with shape ``[..., N, padded_k/32]``.  The
    returned uint16 tensor has shape ``[..., N, padded_k/64]`` and DeepGEMM's
    mn-major stride.  The returned tensor owns its lifetime and must be retained
    alongside the packed weight; this function is not a GEMM hot-path helper.
    """
    e8m0_dtype = getattr(torch, "float8_e8m0fnu", None)
    if raw_scale.dtype == torch.uint8:
        scale_bytes = raw_scale
    elif e8m0_dtype is not None and raw_scale.dtype == e8m0_dtype:
        scale_bytes = raw_scale.view(torch.uint8)
    else:
        raise TypeError(
            "weight scale must be raw uint8/F8_E8M0 checkpoint data, "
            f"got {raw_scale.dtype}"
        )
    if scale_bytes.ndim < 2:
        raise ValueError(
            f"weight scale must have shape [..., N, K/32], got {scale_bytes.shape}"
        )
    if any(dimension <= 0 for dimension in scale_bytes.shape):
        raise ValueError(
            f"weight scale dimensions must be positive: {scale_bytes.shape}"
        )
    if scale_bytes.shape[-1] % 2:
        raise ValueError(
            "weight scale K/32 dimension must be even so padded K is aligned to 64"
        )
    if not scale_bytes.is_cuda:
        raise ValueError(
            "weight scale preprocessing requires a CUDA-compatible PPU tensor"
        )
    if not scale_bytes.is_contiguous():
        raise ValueError("raw weight scale must be contiguous")
    _require_m890p(scale_bytes)

    deep_gemm = _load_deep_gemm()
    preprocess = _require_deep_gemm_symbol(deep_gemm, "preprocess_mxfp4_scales")
    try:
        prepared = preprocess(scale_bytes)
    except TypeError as error:
        raise RuntimeError(
            "PPU deep_gemm.preprocess_mxfp4_scales has an incompatible signature"
        ) from error

    expected_shape = scale_bytes.shape[:-1] + (scale_bytes.shape[-1] // 2,)
    if not isinstance(prepared, torch.Tensor):
        raise RuntimeError("preprocess_mxfp4_scales did not return a torch.Tensor")
    if prepared.dtype != torch.uint16 or prepared.shape != expected_shape:
        raise RuntimeError(
            "preprocess_mxfp4_scales returned an incompatible tensor: "
            f"dtype={prepared.dtype}, shape={prepared.shape}, expected uint16 {expected_shape}"
        )
    if prepared.device != scale_bytes.device:
        raise RuntimeError("preprocess_mxfp4_scales moved the scale to another device")
    layout_error = _mn_major_layout_error(prepared)
    if layout_error is not None:
        raise RuntimeError(
            "preprocess_mxfp4_scales returned an incompatible non-mn-major layout: "
            f"{layout_error}"
        )
    return prepared


def _validate_packed_matrix(name: str, tensor: torch.Tensor) -> None:
    if tensor.dtype != torch.uint8 or tensor.ndim != 2:
        raise TypeError(
            f"{name} must be packed uint8 [rows, K/2], got "
            f"dtype={tensor.dtype}, shape={tensor.shape}"
        )
    if not tensor.is_cuda or not tensor.is_contiguous():
        raise ValueError(f"{name} must be a contiguous CUDA-compatible PPU tensor")
    if tensor.shape[0] <= 0 or tensor.shape[1] <= 0:
        raise ValueError(f"{name} dimensions must be positive, got {tensor.shape}")
    if tensor.shape[1] % (MXFP4_K_ALIGNMENT // 2):
        raise ValueError(
            f"{name} packed K must represent a K aligned to 64, got {tensor.shape[1]}"
        )


def _validate_prepared_scale(
    name: str, scale: torch.Tensor, rows: int, packed_k: int
) -> None:
    expected_shape = (rows, packed_k // MXFP4_BLOCK_SIZE)
    if scale.dtype != torch.uint16 or scale.shape != expected_shape:
        raise TypeError(
            f"{name} must be prepared uint16 {expected_shape}, got "
            f"dtype={scale.dtype}, shape={scale.shape}"
        )
    if not scale.is_cuda:
        raise ValueError(f"{name} must be on a CUDA-compatible PPU device")
    layout_error = _mn_major_layout_error(scale)
    if layout_error is not None:
        raise ValueError(
            f"{name} must use a non-overlapping DeepGEMM mn-major layout: "
            f"{layout_error}"
        )


def _unpack_tensor_pair(name: str, pair) -> Tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(pair, tuple) or len(pair) != 2:
        raise TypeError(f"{name} must be a tuple of exactly (data, scale)")
    data, scale = pair
    if not isinstance(data, torch.Tensor) or not isinstance(scale, torch.Tensor):
        raise TypeError(f"{name} data and scale must both be torch.Tensor instances")
    return data, scale


def gemm_fp4_fp4_bf16_nt(
    activation: Tuple[torch.Tensor, torch.Tensor],
    weight: Tuple[torch.Tensor, torch.Tensor],
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run one dense M890P packed-FP4 x packed-FP4 -> BF16 NT GEMM.

    ``activation`` is ``(packed_data, prepared_uint16_scale)`` from
    :func:`downcast_to_mxfp4`.  ``weight`` is ``(packed_data,
    prepared_uint16_scale)`` where the scale was prepared once with
    :func:`prepare_fp4_weight_scale_mxfp4`.  The call is asynchronous on the
    current stream.  Callers must retain all input tensors and the returned
    output until that stream has completed their use.
    """
    activation_data, activation_prepared_scale = _unpack_tensor_pair(
        "activation", activation
    )
    weight_data, weight_prepared_scale = _unpack_tensor_pair("weight", weight)
    _validate_packed_matrix("activation data", activation_data)
    _validate_packed_matrix("weight data", weight_data)
    _require_m890p(activation_data)
    if weight_data.device != activation_data.device:
        raise ValueError("activation and weight data must be on the same device")
    if activation_data.shape[1] != weight_data.shape[1]:
        raise ValueError(
            "activation and weight packed K disagree: "
            f"{activation_data.shape[1]} != {weight_data.shape[1]}"
        )

    rows, packed_k = activation_data.shape
    out_features = weight_data.shape[0]
    _validate_prepared_scale(
        "activation scale", activation_prepared_scale, rows, packed_k
    )
    if activation_prepared_scale.device != activation_data.device:
        raise ValueError("activation data and scale must be on the same device")
    _validate_prepared_scale(
        "weight scale", weight_prepared_scale, out_features, packed_k
    )
    if weight_prepared_scale.device != activation_data.device:
        raise ValueError("activation and weight scale must be on the same device")

    expected_output_shape = (rows, out_features)
    output_was_provided = output is not None
    if output is None:
        output = torch.empty(
            expected_output_shape,
            dtype=torch.bfloat16,
            device=activation_data.device,
        )
    elif (
        output.dtype != torch.bfloat16
        or output.shape != expected_output_shape
        or output.device != activation_data.device
        or not output.is_contiguous()
    ):
        raise ValueError(
            f"output must be contiguous BF16 {expected_output_shape} on "
            f"{activation_data.device}, got dtype={output.dtype}, "
            f"shape={output.shape}, device={output.device}"
        )
    if output_was_provided:
        output_storage = output.untyped_storage().data_ptr()
        input_tensors = (
            ("activation data", activation_data),
            ("activation scale", activation_prepared_scale),
            ("weight data", weight_data),
            ("weight scale", weight_prepared_scale),
        )
        for input_name, input_tensor in input_tensors:
            if input_tensor.untyped_storage().data_ptr() == output_storage:
                raise ValueError(f"output must not share storage with {input_name}")

    deep_gemm = _load_deep_gemm()
    gemm = _require_deep_gemm_symbol(deep_gemm, "gemm_fp4_fp4_bf16_nt")

    try:
        gemm(
            (activation_data, activation_prepared_scale),
            (weight_data, weight_prepared_scale),
            None,
            output,
        )
    except TypeError as error:
        raise RuntimeError(
            "PPU deep_gemm.gemm_fp4_fp4_bf16_nt has an incompatible signature"
        ) from error
    return output


__all__ = [
    "MXFP4_BLOCK_SIZE",
    "MXFP4_K_ALIGNMENT",
    "downcast_to_mxfp4",
    "gemm_fp4_fp4_bf16_nt",
    "prepare_fp4_weight_scale_mxfp4",
]
