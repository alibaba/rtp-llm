"""M890P Flash Decode normalization over a merged Q-A/KV projection."""

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _qkv_norm(
    X,
    WQ,
    WK,
    FREQ,
    Q,
    KV,
    STRIDE: tl.constexpr,
    QD: tl.constexpr,
    KD: tl.constexpr,
    RD: tl.constexpr,
    EPS: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    kind = tl.program_id(1)
    if kind == 0:
        q_col = tl.arange(0, QD)
        q_x = tl.load(X + row * STRIDE + q_col).to(tl.float32)
        q_w = tl.load(WQ + q_col).to(tl.float32)
        q_inv = tl.rsqrt(tl.sum(q_x * q_x, 0) / QD + EPS)
        tl.store(Q + row * QD + q_col, q_x * q_inv * q_w)
    else:
        k_col = tl.arange(0, KD)
        base = X + row * STRIDE + QD
        k_x = tl.load(base + k_col).to(tl.float32)
        k_w = tl.load(WK + k_col).to(tl.float32)
        k_inv = tl.rsqrt(tl.sum(k_x * k_x, 0) / KD + EPS)
        tl.store(KV + row * KD + k_col, k_x * k_inv * k_w, k_col < KD - RD)
        pair = tl.arange(0, RD // 2)
        real_col = KD - RD + 2 * pair
        imag_col = real_col + 1
        real = (
            tl.load(base + real_col).to(tl.float32)
            * k_inv
            * tl.load(WK + real_col).to(tl.float32)
        )
        imag = (
            tl.load(base + imag_col).to(tl.float32)
            * k_inv
            * tl.load(WK + imag_col).to(tl.float32)
        )
        cosine = tl.load(FREQ + row * RD + pair * 2)
        sine = tl.load(FREQ + row * RD + pair * 2 + 1)
        tl.store(KV + row * KD + real_col, real * cosine - imag * sine)
        tl.store(KV + row * KD + imag_col, real * sine + imag * cosine)


def normalize_decode_qkv(raw, q_weight, kv_weight, freqs, eps):
    """Return contiguous BF16 QR and KV without materializing projection slices.

    The row layout is Flash Q-A 1024 followed by KV 512. The kernel retains
    FP32 RMSNorm/RoPE arithmetic and the original BF16 consumer boundary.
    """
    if raw.ndim != 3 or tuple(raw.shape[1:]) != (1, 1536):
        raise ValueError("Merged Decode QKV must have shape [B,1,1536]")
    batch = raw.shape[0]
    tensors = (raw, q_weight, kv_weight, freqs)
    if any(
        not t.is_cuda or t.device != raw.device or not t.is_contiguous()
        for t in tensors
    ):
        raise ValueError("Merged QKV inputs must be contiguous tensors on one PPU")
    if torch.cuda.get_device_name(raw.device) != "ZW-M890P":
        raise ValueError("Merged Decode QKV requires M890P")
    if (
        raw.dtype != torch.bfloat16
        or q_weight.dtype != raw.dtype
        or kv_weight.dtype != raw.dtype
    ):
        raise TypeError("Merged QKV projection and norm weights must be BF16")
    if q_weight.shape != (1024,) or kv_weight.shape != (512,):
        raise ValueError("Merged QKV requires norm weights [1024] and [512]")
    if freqs.shape != (batch, 32) or freqs.dtype != torch.complex64:
        raise ValueError("Merged QKV frequencies must be complex64 [B,32]")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("Merged QKV norm epsilon must be finite and positive")
    q = torch.empty((batch, 1, 1024), dtype=raw.dtype, device=raw.device)
    kv = torch.empty((batch, 1, 512), dtype=raw.dtype, device=raw.device)
    if batch:
        _qkv_norm[(batch, 2)](
            raw,
            q_weight,
            kv_weight,
            torch.view_as_real(freqs),
            q,
            kv,
            STRIDE=1536,
            QD=1024,
            KD=512,
            RD=64,
            EPS=eps,
            num_warps=4,
        )
    return q, kv
