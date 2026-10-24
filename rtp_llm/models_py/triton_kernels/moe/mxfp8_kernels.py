"""Fused MXFP8 (1x32 microscaling FP8) activation quant for MiniMax-M3.1.

Replaces the eager-PyTorch ``mxfp8_quant_act`` (``abs``/``amax``/``log2``/
``ceil``/``exp2``/``div``/``clamp``/``to(fp8)`` ~= 7 kernels) with a single
Triton kernel: dynamic per-(row, 32-col) max-abs -> UE8M0 power-of-two scale
-> e4m3 cast. The UE8M0 scale is rounded in this model kernel, and the output
(fp32 ``[M, K//32]`` scale) feeds DeepGEMM's native ``pack_mxfp8_scale``.

There is no existing drop-in for this: the FP8 group-quant kernels
(``sgl_per_token_group_quant_fp8`` / ``per_token_group_quant_fp8_v2`` /
``trt_fp8_quantize_128``) are hardwired to the 1x128 recipe and do not emit a
1x32 power-of-two scale.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _ue8m0_pow2_round(value):
    bits = value.to(tl.int32, bitcast=True)
    exponent = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).to(tl.int32)
    return (exponent << 23).to(tl.float32, bitcast=True), exponent & 0xFF


MX_BLOCK = 32
_FP8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max


@triton.jit(do_not_specialize=["M", "ALIGNED_MN"])
def _pack_flashinfer_mxfp8_scale_kernel(
    scale_u8_ptr,
    packed_ptr,
    M,
    K_GROUPS: tl.constexpr,
    K_PACKED: tl.constexpr,
    ALIGNED_MN,
    BLOCK_M: tl.constexpr,
    BLOCK_K_PACKED: tl.constexpr,
):
    # Live row count and aligned output stride vary between Prefill batches;
    # neither changes the byte packing or tile geometry.
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_kp = pid_k * BLOCK_K_PACKED + tl.arange(0, BLOCK_K_PACKED)
    shifts = tl.arange(0, 4) * 8
    offs_g = offs_kp[:, None] * 4 + tl.arange(0, 4)[None, :]
    mask = (offs_m[:, None, None] < M) & (offs_g[None, :, :] < K_GROUPS)
    vals = tl.load(
        scale_u8_ptr + offs_m[:, None, None] * K_GROUPS + offs_g[None, :, :],
        mask=mask,
        other=0,
    ).to(tl.int32)
    packed = tl.sum(vals << shifts[None, None, :], axis=2).to(tl.int32)
    tl.store(
        packed_ptr + offs_m[:, None] + offs_kp[None, :] * ALIGNED_MN,
        packed,
        mask=(offs_m[:, None] < M) & (offs_kp[None, :] < K_PACKED),
    )


def pack_flashinfer_mxfp8_scale_triton(
    scale_u8: torch.Tensor, M: int, K: int
) -> torch.Tensor:
    """Pack FlashInfer uint8 UE8M0 scales into DeepGEMM's int32 TMA layout."""
    assert scale_u8.dtype == torch.uint8
    assert scale_u8.numel() == M * (K // MX_BLOCK)
    import deep_gemm

    k_groups = K // MX_BLOCK
    assert k_groups % 4 == 0
    k_packed = k_groups // 4
    aligned_mn = deep_gemm.get_tma_aligned_size(M, 4)
    storage = torch.empty(
        (k_packed, aligned_mn), device=scale_u8.device, dtype=torch.int32
    )
    packed = storage.transpose(0, 1)
    grid = (triton.cdiv(M, 64), triton.cdiv(k_packed, 32))
    with torch.cuda.device(scale_u8.device):
        _pack_flashinfer_mxfp8_scale_kernel[grid](
            scale_u8,
            packed,
            M=M,
            K_GROUPS=k_groups,
            K_PACKED=k_packed,
            ALIGNED_MN=aligned_mn,
            BLOCK_M=64,
            BLOCK_K_PACKED=32,
            num_warps=8,
        )
    return packed[:M, :]


@triton.jit
def _mxfp8_quant_act_kernel(
    x_ptr,
    q_ptr,
    s_ptr,
    n_groups,
    x_row_stride,
    q_row_stride,
    s_row_stride,
    FP8_MAX: tl.constexpr,
    GROUP: tl.constexpr,
    NG: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    gblk = tl.program_id(1)
    g0 = gblk * NG
    g_idx = tl.arange(0, NG)
    groups = g0 + g_idx
    gmask = groups < n_groups
    col = tl.arange(0, GROUP)
    # [NG, GROUP] element offsets inside the row
    elem = groups[:, None] * GROUP + col[None, :]

    x = tl.load(
        x_ptr + row * x_row_stride + elem,
        mask=gmask[:, None],
        other=0.0,
    ).to(tl.float32)
    amax = tl.max(tl.abs(x), axis=1)
    amax = tl.maximum(amax, 1e-20)
    # UE8M0 power-of-two scale == exp2(ceil(log2(amax / FP8_MAX))), shared with
    # the existing FP8 quant kernels via the bit-hack helper.
    scale, _ = _ue8m0_pow2_round(amax / FP8_MAX)
    q = x / scale[:, None]
    q = tl.minimum(tl.maximum(q, -FP8_MAX), FP8_MAX)

    tl.store(
        q_ptr + row * q_row_stride + elem,
        q.to(q_ptr.dtype.element_ty),
        mask=gmask[:, None],
    )
    tl.store(s_ptr + row * s_row_stride + groups, scale, mask=gmask)


def mxfp8_quant_act_triton(x: torch.Tensor):
    """Fused per-(row, 32) MXFP8 quant. Returns (e4m3 ``[M, K]``, fp32 scale ``[M, K//32]``)."""
    assert x.dim() == 2, f"expected 2D activation, got {tuple(x.shape)}"
    M, K = x.shape
    assert K % MX_BLOCK == 0, f"K={K} must be a multiple of {MX_BLOCK}"
    x = x.contiguous()
    n_groups = K // MX_BLOCK
    q = torch.empty(M, K, device=x.device, dtype=torch.float8_e4m3fn)
    s = torch.empty(M, n_groups, device=x.device, dtype=torch.float32)
    if M == 0:
        return q, s
    NG = 16
    grid = (M, triton.cdiv(n_groups, NG))
    _mxfp8_quant_act_kernel[grid](
        x,
        q,
        s,
        n_groups,
        x.stride(0),
        q.stride(0),
        s.stride(0),
        FP8_MAX=float(_FP8_E4M3_MAX),
        GROUP=MX_BLOCK,
        NG=NG,
        num_warps=4,
    )
    return q, s
