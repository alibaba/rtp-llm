"""Generic fused E4M3 MLA operand conversion and paged latent insertion."""

import torch
import triton
import triton.language as tl


@triton.jit
def _quant_fp8(x, inverse_scale, ASSUME_UNIT_SCALES: tl.constexpr):
    if ASSUME_UNIT_SCALES:
        return x.to(tl.float8e4nv)
    value = x.to(tl.float32) * inverse_scale
    # Match ordinary E4M3 saturation while preserving NaNs.
    return tl.where(value != value, value,
                    tl.minimum(tl.maximum(value, -448.0), 448.0)).to(tl.float8e4nv)


@triton.jit
def _fused_mla_fp8_epilogue(
    Q, KN, PE, C, V, OQ, OK, OV, CACHE, SLOTS,
    QS, KS, VS, CS,
    T: tl.constexpr, H: tl.constexpr, NOPE: tl.constexpr, ROPE: tl.constexpr,
    VD: tl.constexpr, LATENT: tl.constexpr, CACHE_BLOCK: tl.constexpr,
    Q_T: tl.constexpr, Q_H: tl.constexpr,
    KN_T: tl.constexpr, KN_H: tl.constexpr,
    PE_T: tl.constexpr, C_T: tl.constexpr,
    V_T: tl.constexpr, V_H: tl.constexpr,
    CACHE_B: tl.constexpr, CACHE_T: tl.constexpr,
    BLOCK_T: tl.constexpr, BLOCK_Q: tl.constexpr,
    BLOCK_V: tl.constexpr, BLOCK_C: tl.constexpr,
    ASSUME_UNIT_SCALES: tl.constexpr,
):
    token = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    slot = tl.program_id(1)
    valid = token < T
    if slot < H:
        q_feature = tl.arange(0, BLOCK_Q)
        q_value = tl.load(
            Q + token[:, None] * Q_T + slot * Q_H + q_feature[None, :],
            valid[:, None] & (q_feature[None, :] < NOPE + ROPE), other=0,
        )
        if ASSUME_UNIT_SCALES:
            q_scale = 1.0
        else:
            q_scale = tl.load(QS)
        tl.store(
            OQ + (token[:, None] * H + slot) * (NOPE + ROPE) + q_feature[None, :],
            _quant_fp8(q_value, q_scale, ASSUME_UNIT_SCALES),
            valid[:, None] & (q_feature[None, :] < NOPE + ROPE),
        )

        nope_value = tl.load(
            KN + token[:, None] * KN_T + slot * KN_H + q_feature[None, :],
            valid[:, None] & (q_feature[None, :] < NOPE), other=0,
        )
        key_rope_value = tl.load(
            PE + token[:, None] * PE_T + q_feature[None, :] - NOPE,
            valid[:, None] & (q_feature[None, :] >= NOPE)
            & (q_feature[None, :] < NOPE + ROPE), other=0,
        )
        key_value = tl.where(q_feature[None, :] < NOPE, nope_value,
                             key_rope_value)
        if ASSUME_UNIT_SCALES:
            key_scale = 1.0
        else:
            key_scale = tl.load(KS)
        tl.store(
            OK + (token[:, None] * H + slot) * (NOPE + ROPE) + q_feature[None, :],
            _quant_fp8(key_value, key_scale, ASSUME_UNIT_SCALES),
            valid[:, None] & (q_feature[None, :] < NOPE + ROPE),
        )

        value_feature = tl.arange(0, BLOCK_V)
        value = tl.load(
            V + token[:, None] * V_T + slot * V_H + value_feature[None, :],
            valid[:, None] & (value_feature[None, :] < VD), other=0,
        )
        if ASSUME_UNIT_SCALES:
            value_scale = 1.0
        else:
            value_scale = tl.load(VS)
        tl.store(
            OV + (token[:, None] * H + slot) * VD + value_feature[None, :],
            _quant_fp8(value, value_scale, ASSUME_UNIT_SCALES),
            valid[:, None] & (value_feature[None, :] < VD),
        )
    else:
        cache_feature = tl.arange(0, BLOCK_C)
        mapping = tl.load(SLOTS + token, valid, other=-1)
        cache_valid = valid & (mapping >= 0)
        latent_value = tl.load(
            C + token[:, None] * C_T + cache_feature[None, :],
            cache_valid[:, None] & (cache_feature[None, :] < LATENT), other=0,
        )
        cache_rope_value = tl.load(
            PE + token[:, None] * PE_T + cache_feature[None, :] - LATENT,
            cache_valid[:, None] & (cache_feature[None, :] >= LATENT)
            & (cache_feature[None, :] < LATENT + ROPE), other=0,
        )
        cache_value = tl.where(cache_feature[None, :] < LATENT,
                               latent_value, cache_rope_value)
        if ASSUME_UNIT_SCALES:
            cache_scale = 1.0
        else:
            cache_scale = tl.load(CS)
        tl.store(
            CACHE + (mapping[:, None] // CACHE_BLOCK) * CACHE_B
            + (mapping[:, None] % CACHE_BLOCK) * CACHE_T
            + cache_feature[None, :],
            _quant_fp8(cache_value, cache_scale, ASSUME_UNIT_SCALES),
            cache_valid[:, None] & (cache_feature[None, :] < LATENT + ROPE),
        )


def fused_mla_fp8_epilogue(
    q: torch.Tensor,
    k_nope: torch.Tensor,
    k_pe: torch.Tensor,
    compressed_kv: torch.Tensor,
    v: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    q_scale_inv: torch.Tensor,
    k_scale_inv: torch.Tensor,
    v_scale_inv: torch.Tensor,
    cache_scale_inv: torch.Tensor,
    *,
    block_tokens: int = 32,
    num_warps: int = 4,
    assume_unit_scales: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize MLA Q/K/V and insert the latent into ordinary paged E4M3 cache.

    Scale arguments are inverse dequantization scales. A negative slot skips the
    cache write. The projection views may have independent token/head strides.
    """
    if q.ndim != 3 or k_nope.ndim != 3 or v.ndim != 3:
        raise ValueError("MLA Q, K-nope and V must have [token, head, dim] shape")
    tokens, heads, query_dim = q.shape
    if k_pe.ndim != 2 or compressed_kv.ndim != 2:
        raise ValueError("MLA RoPE key and latent must have [token, dim] shape")
    rope_dim = k_pe.shape[1]
    nope_dim = k_nope.shape[2]
    value_dim = v.shape[2]
    latent_dim = compressed_kv.shape[1]
    if not (k_nope.shape[:2] == v.shape[:2] == (tokens, heads)
            and k_pe.shape[0] == compressed_kv.shape[0] == tokens
            and query_dim == nope_dim + rope_dim):
        raise ValueError("incompatible MLA operand dimensions")
    if (kv_cache.ndim != 3 or kv_cache.shape[2] != latent_dim + rope_dim
            or slot_mapping.shape != (tokens,)):
        raise ValueError("incompatible paged MLA cache or slot mapping")
    if not kv_cache.is_contiguous() or kv_cache.dtype != torch.float8_e4m3fn:
        raise ValueError("ordinary MLA cache must be contiguous E4M3")
    if slot_mapping.dtype != torch.int64:
        raise ValueError("slot mapping must be int64")
    inputs = (q, k_nope, k_pe, compressed_kv, v)
    if any(x.dtype not in (torch.bfloat16, torch.float16) for x in inputs):
        raise ValueError("MLA inputs must be BF16 or FP16")
    if len({x.dtype for x in inputs}) != 1:
        raise ValueError("MLA operands must share one floating dtype")
    scalars = (q_scale_inv, k_scale_inv, v_scale_inv, cache_scale_inv)
    if any(s.dtype != torch.float32 or s.numel() != 1 for s in scalars):
        raise ValueError("inverse scales must be one-element FP32 CUDA tensors")
    tensors = (*inputs, kv_cache, slot_mapping, *scalars)
    if not q.is_cuda or any(x.device != q.device for x in tensors):
        raise ValueError("all MLA operands must be on the same CUDA device")
    if block_tokens not in (8, 16, 32, 64):
        raise ValueError("block_tokens must be 8, 16, 32 or 64")
    if num_warps not in (2, 4, 8):
        raise ValueError("num_warps must be 2, 4 or 8")
    fp8 = torch.float8_e4m3fn
    q_out = torch.empty((tokens, heads, query_dim), device=q.device, dtype=fp8)
    k_out = torch.empty_like(q_out)
    v_out = torch.empty((tokens, heads, value_dim), device=q.device, dtype=fp8)
    if tokens:
        _fused_mla_fp8_epilogue[(triton.cdiv(tokens, block_tokens), heads + 1)](
            q, k_nope, k_pe, compressed_kv, v, q_out, k_out, v_out,
            kv_cache, slot_mapping, *scalars,
            tokens, heads, nope_dim, rope_dim, value_dim, latent_dim,
            kv_cache.shape[1],
            q.stride(0), q.stride(1), k_nope.stride(0), k_nope.stride(1),
            k_pe.stride(0), compressed_kv.stride(0), v.stride(0), v.stride(1),
            kv_cache.stride(0), kv_cache.stride(1),
            block_tokens, triton.next_power_of_2(query_dim),
            triton.next_power_of_2(value_dim),
            triton.next_power_of_2(latent_dim + rope_dim),
            assume_unit_scales,
            num_warps=num_warps,
        )
    return q_out, k_out, v_out
