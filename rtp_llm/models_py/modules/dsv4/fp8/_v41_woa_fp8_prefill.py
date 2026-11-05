"""FP8 batched wo_a output projection for V4.1 PREFILL on SM120 (R6-2).

Flag DSV41_WOA_FP8_EINSUM=1 (default off). Replaces the BF16 einsum
("rgd,god->rgo", cuBLASLt sm80 kernel, L1TEX-bound at 95% mem throughput /
175 TF per phase-B ncu) with the in-tree DeepGEMM batched FP8 einsum
("bhr,hdr->bhd", recipe (1,1,128)) — measured 0.290 ms vs 0.674 ms BF16 at
the production shape (m2048 h8 k4096 n1024) on SM120, i.e. the SM100 gate in
fp8/_v41_output_projection.py is policy, not kernel capability.

Recipe note: V4.1 wo_a checkpoint scales are block-32; the stacked weight
buffers (_wo_a_stk_s via get_mn_major_tma_aligned_packed_ue8m0_tensor) carry
32-granularity UE8M0 words, so the einsum runs recipe (1,1,32) and the
activation quantizer uses matching 32-column blocks (the V4 block-128
convention of _fused_inv_rope_fp8_quant_triton would mismatch).

Quantization otherwise mirrors _fused_inv_rope_fp8_quant_triton: per-32
blocks, scale = exp2(ceil(log2(amax/448))), UE8M0 byte = fp32 bits >> 23,
four bytes packed little-endian per int32, group-major TMA scale storage
[H, P, tma_M] addressed (h*P + p)*tma_M + m. NOT byte-exact by design
(FP8 e4m3 mantissa); gated by tolerance + tokens + top-1 on real prompts.
"""

import os

import torch
import triton
import triton.language as tl

WOA_FP8_FLAG = "DSV41_WOA_FP8_EINSUM"
_FP8_MAX = 448.0
_EPS = 1e-10


def woa_fp8_enabled() -> bool:
    # R45 adoption: default ON (qualified on the h32 stack: -30.6ms median,
    # 2/3 pairs, 64/64 tokens, argmax all, logits rel_l2 0.290 <= 0.34,
    # deterministic sha 3e7001beb7742dda). Escape: =0 restores the BF16
    # einsum (reference sha ac4da65061991a9e).
    return os.environ.get(WOA_FP8_FLAG, "1") == "1"


def woa_fp8_supported(weight: torch.Tensor) -> bool:
    return (
        weight.is_cuda
        and torch.cuda.get_device_capability(weight.device) == (12, 0)
        and weight.dtype == torch.bfloat16
        and weight.ndim == 3
        and weight.shape[2] % 512 == 0
        and weight.shape[1] % 4 == 0
        and weight.is_contiguous()
    )


def quantize_woa_weight(w: torch.Tensor):
    """[H,N,K] BF16 -> (fp8 [H,N,K], scale view [H,N,K/512] int32, TMA layout)."""
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise RuntimeError("wo_a FP8 weight preparation must precede capture")
    H, N, K = w.shape
    P = K // 128
    tma_n = (N + 3) // 4 * 4
    blocks = w.float().view(H, N, K // 32, 32)
    amax = blocks.abs().amax(dim=-1).clamp_min(_EPS)
    scale = torch.exp2(torch.ceil(torch.log2(amax / _FP8_MAX)))
    wq = (
        (blocks / scale.unsqueeze(-1))
        .clamp(-_FP8_MAX, _FP8_MAX)
        .to(torch.float8_e4m3fn)
        .view(H, N, K)
    )
    sbits = scale.view(torch.int32)
    sbytes = ((sbits >> 23) & 0xFF).to(torch.int64)  # [H,N,K/128]
    sbytes = sbytes.view(H, N, P, 4)
    packed = (
        sbytes[..., 0]
        | (sbytes[..., 1] << 8)
        | (sbytes[..., 2] << 16)
        | (sbytes[..., 3] << 24)
    ).to(
        torch.int32
    )  # [H,N,P]
    buf = torch.zeros(H * P * tma_n, dtype=torch.int32, device=w.device)
    view = buf.as_strided((H, N, P), (P * tma_n, 1, tma_n))
    view.copy_(packed)
    return wq, buf.as_strided((H, N, P), (P * tma_n, 1, tma_n))


@triton.jit
def _quant_woa_prefill_kernel(
    X,
    FP8,
    SCALE,
    H: tl.constexpr,
    K: tl.constexpr,
    P: tl.constexpr,
    TMA_M: tl.constexpr,
):
    m = tl.program_id(0)
    h = tl.program_id(1)
    offs = tl.arange(0, K)
    base = (m * H + h) * K
    x = tl.load(X + base + offs).to(tl.float32)
    xb = tl.reshape(tl.abs(x), (K // 32, 32))
    amax = tl.maximum(tl.max(xb, axis=1), 1e-10)
    scale = tl.math.exp2(tl.ceil(tl.log2(amax * (1.0 / 448.0))))
    scale_full = tl.reshape(
        tl.broadcast_to(tl.reshape(scale, (K // 32, 1)), (K // 32, 32)),
        (K,),
    )
    xq = tl.clamp(x / scale_full, -448.0, 448.0).to(tl.float8e4nv)
    tl.store(FP8 + base + offs, xq)
    sbytes = (scale.to(tl.int32, bitcast=True) >> 23) & 0xFF
    words = tl.reshape(sbytes, (P, 4))
    block_offsets = tl.arange(0, 4)
    packed = tl.sum(words << (block_offsets[None, :] * 8), axis=1)
    pidx = tl.arange(0, P)
    tl.store(SCALE + h * (P * TMA_M) + pidx * TMA_M + m, packed)


def quantize_activation(x: torch.Tensor):
    """[M,H,K] BF16 -> (fp8 [M,H,K], scale view [M,H,K/128] int32, 32-gran)."""
    M, H, K = x.shape
    P = K // 128
    tma_m = (M + 3) // 4 * 4
    a = torch.empty(M * H * K, dtype=torch.float8_e4m3fn, device=x.device)
    buf = torch.empty(H * P * tma_m, dtype=torch.int32, device=x.device)
    _quant_woa_prefill_kernel[(M, H)](
        x, a, buf, H=H, K=K, P=P, TMA_M=tma_m, num_warps=4
    )
    scale = buf.as_strided((H, M, P), (P * tma_m, 1, tma_m)).transpose(0, 1)
    return a.view(M, H, K), scale


def woa_fp8_forward(unrotated: torch.Tensor, w_fp8, w_scale, out=None):
    """BF16 [M,H,K] -> BF16 [M,H,N] via deep_gemm.fp8_einsum."""
    import deep_gemm

    M, H, K = unrotated.shape
    N = w_fp8.shape[1]
    a, a_scale = quantize_activation(unrotated)
    if out is None:
        out = torch.empty(M, H, N, dtype=torch.bfloat16, device=unrotated.device)
    deep_gemm.fp8_einsum(
        "bhr,hdr->bhd", (a, a_scale), (w_fp8, w_scale), out, recipe=(1, 1, 32)
    )
    return out
