"""Whole-row E4M3 activation quantization for the PPU PTPC linear ABI.

The CPU/reference expression is deliberately unchanged. The device kernel
combines its elementwise work and row maximum into one launch, including
group-major packing for ``wo_a``. Payload and FP32 scale rounding boundaries
remain separate; this is not the existing group-128 block-FP8 quantizer.
"""

from functools import lru_cache
import os

import torch

from rtp_llm.models_py.modules.dsv41.ppu.quant_conversion import FP8_INV_MAX

MAX_FUSED_WIDTH = 16384
PTPC_FUSION_ENV = "RTP_V41_PPU_PTPC_FUSION"


def ptpc_fusion_enabled():
    """Use validated device fusion by default; explicit ``0`` restores ATen."""
    return os.environ.get(PTPC_FUSION_ENV, "1") == "1"


def quantize_ptpc_activation_reference(x):
    """Original ATen expression, also retained as the portable CPU path."""
    values = x.float()
    amax = values.abs().amax(-1, keepdim=True).clamp_min(1e-10)
    scale = amax * FP8_INV_MAX
    payload = (values * (448.0 / amax)).clamp(-448, 448).to(torch.float8_e4m3fn)
    return payload.contiguous(), scale.contiguous()


@lru_cache(maxsize=1)
def _kernels():
    # Keep CPU imports independent of the accelerator compiler and wheels.
    import triton
    import triton.language as tl

    @triton.jit
    def quantize_rows(
        X,
        Q,
        S,
        ROWS: tl.constexpr,
        GROUPS: tl.constexpr,
        WIDTH: tl.constexpr,
        STRIDE_ROW: tl.constexpr,
        STRIDE_GROUP: tl.constexpr,
        STRIDE_COL: tl.constexpr,
        GROUP_MAJOR: tl.constexpr,
        INV_MAX: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        out_row = tl.program_id(0).to(tl.int64)
        if GROUP_MAJOR:
            row, group = out_row % ROWS, out_row // ROWS
        else:
            row, group = out_row // GROUPS, out_row % GROUPS
        columns = tl.arange(0, BLOCK)
        values = tl.load(
            X + row * STRIDE_ROW + group * STRIDE_GROUP + columns * STRIDE_COL,
            columns < WIDTH,
            0.0,
        ).to(tl.float32)
        amax = tl.maximum(tl.max(tl.abs(values), 0), 1.0e-10)
        scale = amax * INV_MAX
        # Tensor.__rtruediv__ computes reciprocal(amax), then multiplies by
        # 448. Preserve those two FP32 roundings before multiplying values.
        inverse = tl.div_rn(1.0, amax)
        inverse = inverse * 448.0
        quantized = values * inverse
        quantized = tl.minimum(tl.maximum(quantized, -448.0), 448.0)
        tl.store(Q + out_row * WIDTH + columns, quantized, columns < WIDTH)
        tl.store(S + out_row, scale)

    @triton.jit
    def pack_grouped_bytes(
        X,
        Y,
        ROWS: tl.constexpr,
        GROUPS: tl.constexpr,
        ROW_BYTES: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        group = tl.program_id(1)
        columns = tl.program_id(2) * BLOCK + tl.arange(0, BLOCK)
        value = tl.load(
            X + (group * ROWS + row) * ROW_BYTES + columns,
            columns < ROW_BYTES,
            0,
        )
        tl.store(
            Y + (row * GROUPS + group) * ROW_BYTES + columns,
            value,
            columns < ROW_BYTES,
        )

    return triton, quantize_rows, pack_grouped_bytes


def _validate(x):
    if x.ndim < 1 or x.shape[-1] <= 0:
        raise ValueError("PTPC quantization requires a nonempty feature dimension")
    if x.layout != torch.strided:
        raise ValueError("PTPC quantization requires a strided tensor")


def _quantize_device(x, *, group_major):
    if x.ndim == 1:
        rows, groups, width = 1, 1, x.shape[0]
        stride_row, stride_group, stride_col = 0, 0, x.stride(0)
    elif x.ndim == 2:
        rows, groups, width = x.shape[0], 1, x.shape[1]
        stride_row, stride_group, stride_col = x.stride(0), 0, x.stride(1)
    else:
        rows, groups, width = x.shape
        stride_row, stride_group, stride_col = x.stride()
    shape = (groups, rows, width) if group_major else tuple(x.shape)
    scale_shape = (groups, rows, 1) if group_major else (*x.shape[:-1], 1)
    payload = torch.empty(shape, device=x.device, dtype=torch.float8_e4m3fn)
    scale = torch.empty(scale_shape, device=x.device, dtype=torch.float32)
    if rows * groups:
        triton, kernel, _ = _kernels()
        block = triton.next_power_of_2(width)
        kernel[(rows * groups,)](
            x,
            payload,
            scale,
            rows,
            groups,
            width,
            stride_row,
            stride_group,
            stride_col,
            group_major,
            FP8_INV_MAX,
            block,
            num_warps=4 if block <= 4096 else 8,
            num_stages=1,
            enable_fp_fusion=False,
        )
    return payload, scale


def quantize_ptpc_activation(x):
    """Return contiguous E4M3 ``x.shape`` and FP32 ``[..., 1]`` scales.

    Device FP32/BF16/FP16 rank-one through rank-three inputs of width at most
    16384, including noncontiguous views, use one kernel. Other callers retain
    the reference expression. Production Linear passes a rank-two view.
    """
    _validate(x)
    if _device_supported(x):
        return _quantize_device(x, group_major=False)
    return quantize_ptpc_activation_reference(x)


def quantize_ptpc_grouped_activation(x):
    """Pack ``[M,G,K]`` directly into contiguous ``[G,M,K]`` plus row scales."""
    _validate(x)
    if x.ndim != 3:
        raise ValueError("grouped PTPC quantization requires [M, G, K]")
    if _device_supported(x):
        return _quantize_device(x, group_major=True)
    payload, scale = quantize_ptpc_activation_reference(x)
    payload = payload.view(torch.uint8).permute(1, 0, 2).contiguous()
    return payload.view(torch.float8_e4m3fn), scale.permute(1, 0, 2).contiguous()


def _device_supported(x):
    return (
        x.is_cuda
        and ptpc_fusion_enabled()
        and x.ndim <= 3
        and x.shape[-1] <= MAX_FUSED_WIDTH
        and x.dtype in (torch.bfloat16, torch.float16, torch.float32)
    )


def pack_ptpc_grouped_output(groupmajor):
    """Pack contiguous ``[G,M,R]`` GEMM output into contiguous ``[M,G,R]``.

    The native GEMM requires contiguous per-group output. This byte-copy
    kernel preserves signed zeros and NaN payloads without float conversion.
    """
    if groupmajor.ndim != 3 or not groupmajor.is_contiguous():
        raise ValueError("PTPC grouped output must be contiguous [G, M, R]")
    groups, rows, width = groupmajor.shape
    result = torch.empty(
        (rows, groups, width), device=groupmajor.device, dtype=groupmajor.dtype
    )
    source_bytes = groupmajor.view(torch.uint8)
    result_bytes = result.view(torch.uint8)
    if not groupmajor.is_cuda:
        result_bytes.copy_(source_bytes.permute(1, 0, 2))
    elif groupmajor.numel():
        triton, _, kernel = _kernels()
        row_bytes = width * groupmajor.element_size()
        block = 1024
        kernel[(rows, groups, triton.cdiv(row_bytes, block))](
            source_bytes,
            result_bytes,
            rows,
            groups,
            row_bytes,
            block,
            num_warps=4,
            num_stages=1,
        )
    return result
