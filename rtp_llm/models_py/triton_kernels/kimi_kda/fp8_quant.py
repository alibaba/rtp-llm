"""Quantize KDA's strided 128-wide forget latent without a staging copy."""

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["M"])
def _quantize_forget_latent_fp8(
    x,
    output,
    scale,
    M,
    ROW_STRIDE: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, 128)
    values = tl.load(
        x + rows[:, None] * ROW_STRIDE + cols[None, :],
        rows[:, None] < M,
        other=0,
    ).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(values), axis=1), EPS)
    scaling = tl.exp2(tl.ceil(tl.log2(tl.maximum(amax / 448.0, 1.0e-10))))
    quantized = tl.minimum(tl.maximum(values / scaling[:, None], -448.0), 448.0)
    tl.store(output + rows[:, None] * 128 + cols[None, :], quantized, rows[:, None] < M)
    # One real group per row. DeepGEMM packs four UE8M0 bytes per int32;
    # initialize unused K groups and TMA-aligned row padding to scale=1.
    exponent = (scaling.to(tl.int32, bitcast=True) >> 23) & 255
    packed = tl.where(rows < M, exponent | 0x7F7F7F00, 0x7F7F7F7F)
    tl.store(scale + rows, packed, rows < ((M + 3) // 4) * 4)


def quantize_forget_latent_fp8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return contiguous E4M3 values and column-major packed UE8M0 scales.

    This is the existing group128/eps=1e-4 activation policy for f_b. The
    source may be a row-strided slice with a nonzero storage offset. No
    persistent buffers are shared between calls or CUDA streams.
    """
    if x.ndim != 2 or x.shape[1] != 128 or x.stride(1) != 1 or x.stride(0) < 128:
        raise ValueError("KDA forget FP8 quantization requires row-strided [M,128]")
    if not x.is_cuda or x.dtype != torch.bfloat16:
        raise ValueError("KDA forget FP8 quantization requires CUDA BF16 input")
    m = x.shape[0]
    output = torch.empty((m, 128), dtype=torch.float8_e4m3fn, device=x.device)
    scales = torch.empty((1, triton.cdiv(m, 4) * 4), dtype=torch.int32, device=x.device).T
    if m:
        _quantize_forget_latent_fp8[(triton.cdiv(m, 4),)](
            x, output, scales, m, x.stride(0), 1.0e-4, 4, num_warps=4

        )
    return output, scales[:m]
