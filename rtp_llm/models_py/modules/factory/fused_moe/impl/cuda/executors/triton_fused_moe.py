"""Triton FP8 per-block FusedMoE kernel for small-batch fallback."""
from __future__ import annotations

import functools
import logging
from collections import OrderedDict
from typing import List, Optional, Tuple

import torch
import triton
import triton.language as tl
from rtp_kernel import moe_align_block_size

try:
    from triton.tools.tensor_descriptor import TensorDescriptor
    _support_tensor_descriptor = True
except Exception:
    _support_tensor_descriptor = False

logger = logging.getLogger(__name__)

_fp8_dtype = torch.float8_e4m3fn
_FP8_MAX = torch.finfo(_fp8_dtype).max
_FP8_MIN = torch.finfo(_fp8_dtype).min


# ── FP8 per-token-group quantization (triton, supports pre-allocated buffers) ─
@triton.jit
def _per_token_group_quant_8bit(
    y_ptr, y_q_ptr, y_s_ptr, y_stride, N, eps,
    bit8_min, bit8_max, BLOCK: tl.constexpr,
):
    g_id = tl.program_id(0)
    y_ptr += g_id * y_stride
    y_q_ptr += g_id * y_stride
    y_s_ptr += g_id
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    y = tl.load(y_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    _absmax = tl.maximum(tl.max(tl.abs(y)), eps)
    y_s = _absmax / bit8_max
    y_q = tl.clamp(y / y_s, bit8_min, bit8_max).to(y_q_ptr.dtype.element_ty)
    tl.store(y_q_ptr + cols, y_q, mask=mask)
    tl.store(y_s_ptr, y_s)


def per_token_group_quant_fp8(
    x: torch.Tensor, group_size: int, eps: float = 1e-10,
    out_q: Optional[torch.Tensor] = None, out_s: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.shape[-1] % group_size == 0 and x.is_contiguous()
    x_q = out_q if out_q is not None else torch.empty_like(x, dtype=_fp8_dtype)
    x_s = out_s if out_s is not None else torch.empty(
        x.shape[:-1] + (x.shape[-1] // group_size,), device=x.device, dtype=torch.float32,
    )
    M = x.numel() // group_size
    BLOCK = triton.next_power_of_2(group_size)
    _per_token_group_quant_8bit[(M,)](
        x, x_q, x_s, group_size, group_size, eps,
        bit8_min=_FP8_MIN, bit8_max=_FP8_MAX,
        BLOCK=BLOCK, num_warps=min(max(BLOCK // 256, 1), 8), num_stages=1,
    )
    return x_q, x_s


# ── Config ───────────────────────────────────────────────────────────────
def _get_default_config(block_shape):
    bn, bk = (block_shape or [0, 0])
    return {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": bn or 128,
        "BLOCK_SIZE_K": bk or 128,
        "GROUP_SIZE_M": 32,
        "num_warps": 4,
        "num_stages": 3,
    }


# ── Triton fused_moe_kernel ─────────────────────────────────────────────
@functools.lru_cache(maxsize=8)
def _should_swap_ab(bm, bn):
    major, _ = torch.cuda.get_device_capability()
    return major >= 9 and bm < 64 and bn >= 64


@triton.jit
def _write_zeros(c_ptr, stride_cm, stride_cn, pid_n, N, offs_token, token_mask,
                 BLOCK_SIZE_M, BLOCK_SIZE_N, compute_type):
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=compute_type)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    tl.store(c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :],
             acc, mask=token_mask[:, None] & (offs_cn[None, :] < N))


@triton.jit
def fused_moe_kernel(
    a_ptr, a_desc, b_ptr, b_desc, c_ptr, a_scale_ptr, b_scale_ptr,
    topk_weights_ptr, sorted_token_ids_ptr, expert_ids_ptr, num_tokens_post_padded_ptr,
    N, K, EM, num_valid_tokens,
    stride_am, stride_ak, stride_be, stride_bk, stride_bn, stride_cm, stride_cn,
    stride_asm, stride_ask, stride_bse, stride_bsk, stride_bsn,
    group_n: tl.constexpr, group_k: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr, top_k: tl.constexpr, compute_type: tl.constexpr,
    even_Ks: tl.constexpr, c_sorted: tl.constexpr, filter_expert: tl.constexpr,
    swap_ab: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    group_id = pid // (GROUP_SIZE_M * num_pid_n)
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % (GROUP_SIZE_M * num_pid_n)) % group_size_m)
    pid_n = (pid % (GROUP_SIZE_M * num_pid_n)) // group_size_m

    if pid_m * BLOCK_SIZE_M >= tl.load(num_tokens_post_padded_ptr):
        return
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id).to(tl.int64)
    token_mask = offs_token < num_valid_tokens
    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)

    if filter_expert and off_experts == -1:
        _write_zeros(c_ptr, stride_cm, stride_cn, pid_n, N, offs_token, token_mask,
                     BLOCK_SIZE_M, BLOCK_SIZE_N, compute_type)
        return

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    if a_desc is not None:
        start_offs_m = pid_m * BLOCK_SIZE_M
    else:
        a_ptrs = a_ptr + (offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak)
    if b_desc is not None:
        start_offs_n = pid_n * BLOCK_SIZE_N
    else:
        b_ptrs = b_ptr + off_experts * stride_be + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    if a_desc is not None:
        a_scale_ptrs = a_scale_ptr + offs_token_id * stride_asm
    else:
        a_scale_ptrs = a_scale_ptr + (offs_token // top_k) * stride_asm
    if BLOCK_SIZE_N > group_n:
        offs_bsn = offs_bn // group_n
    else:
        offs_bsn = pid_n * BLOCK_SIZE_N // group_n
    b_scale_ptrs = b_scale_ptr + off_experts * stride_bse + offs_bsn * stride_bsn

    if swap_ab:
        accumulator = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
    else:
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_SIZE_K):
        if a_desc is not None:
            a = a_desc.load([start_offs_m, k_start])
        elif even_Ks:
            a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
        else:
            a = tl.load(a_ptrs, mask=token_mask[:, None] & (offs_k[None, :] < K - k_start), other=0.0)
        if b_desc is not None:
            b = b_desc.load([tl.load(expert_ids_ptr + pid_m), start_offs_n, k_start]).reshape(BLOCK_SIZE_N, BLOCK_SIZE_K).T
        elif even_Ks:
            b = tl.load(b_ptrs)
        else:
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k_start, other=0.0)

        offs_ks = k_start // group_k
        a_s = tl.load(a_scale_ptrs + offs_ks * stride_ask, mask=token_mask, other=0.0)
        b_s = tl.load(b_scale_ptrs + offs_ks * stride_bsk)
        if swap_ab:
            a, b = tl.trans(b, (1, 0)), tl.trans(a, (1, 0))
            a_s, b_s = b_s, a_s
        if BLOCK_SIZE_N > group_n:
            accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
        else:
            accumulator += tl.dot(a, b) * (a_s[:, None] * b_s)

        if a_desc is None:
            a_ptrs += BLOCK_SIZE_K * stride_ak
        if b_desc is None:
            b_ptrs += BLOCK_SIZE_K * stride_bk

    if swap_ab:
        accumulator = tl.trans(accumulator, (1, 0))
    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
        accumulator *= moe_weight[:, None]

    accumulator = accumulator.to(compute_type)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    if c_sorted:
        c_ptrs = c_ptr + stride_cm * offs_token_id[:, None] + stride_cn * offs_cn[None, :]
    else:
        c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, accumulator, mask=token_mask[:, None] & (offs_cn[None, :] < N))


# ── TMA helpers (for weight B TMA loading) ───────────────────────────────
_TMA_SET = False

def _set_tma_allocator():
    global _TMA_SET
    if _TMA_SET:
        return
    triton.set_allocator(lambda size, align, stream: torch.empty(size, device="cuda", dtype=torch.int8))
    _TMA_SET = True

_B_DESC_CACHE: OrderedDict = OrderedDict()

def _get_b_desc(B, bn, bk):
    key = (B.data_ptr(), B.shape, B.stride(), str(B.dtype), bn, bk)
    if key in _B_DESC_CACHE:
        _B_DESC_CACHE.move_to_end(key)
        return _B_DESC_CACHE[key]
    desc = TensorDescriptor(B, B.shape, B.stride(), [1, bn, bk])
    _B_DESC_CACHE[key] = desc
    if len(_B_DESC_CACHE) > 64:
        _B_DESC_CACHE.popitem(last=False)
    return desc


# ── Invoke kernel ────────────────────────────────────────────────────────
def _invoke(A, B, C, A_scale, B_scale, topk_weights, topk_ids,
            sorted_ids, expert_ids, num_post_pad,
            mul_weight, topk, config, compute_type, block_shape,
            filter_expert=True, b_use_tma=False,
            a_quant_buf=None, a_scale_buf=None):
    bn, bk = block_shape
    A, A_scale = per_token_group_quant_fp8(A, bk, out_q=a_quant_buf, out_s=a_scale_buf)

    swap = _should_swap_ab(config["BLOCK_SIZE_M"], config["BLOCK_SIZE_N"])
    BM, BN, BK = config["BLOCK_SIZE_M"], config["BLOCK_SIZE_N"], config["BLOCK_SIZE_K"]
    N_dim, K_dim = B.shape[1], B.shape[2]
    grid = lambda META: (triton.cdiv(sorted_ids.shape[0], META["BLOCK_SIZE_M"]) * triton.cdiv(N_dim, META["BLOCK_SIZE_N"]),)

    if b_use_tma:
        _set_tma_allocator()
    b_desc = _get_b_desc(B, BN, BK) if b_use_tma else None

    fused_moe_kernel[grid](
        A, None, B, b_desc, C, A_scale, B_scale,
        topk_weights, sorted_ids, expert_ids, num_post_pad,
        N_dim, K_dim, sorted_ids.shape[0], topk_ids.numel(),
        A.stride(0), A.stride(1), B.stride(0), B.stride(2), B.stride(1),
        C.stride(-2), C.stride(-1),
        A_scale.stride(0) if A_scale.ndim == 2 else 0,
        A_scale.stride(1) if A_scale.ndim == 2 else 0,
        B_scale.stride(0) if B_scale.ndim >= 2 else 0,
        B_scale.stride(2) if B_scale.ndim == 3 else 0,
        B_scale.stride(1) if B_scale.ndim >= 2 else 0,
        bn, bk,
        MUL_ROUTED_WEIGHT=mul_weight, top_k=topk, compute_type=compute_type,
        even_Ks=K_dim % BK == 0, c_sorted=False, filter_expert=filter_expert,
        swap_ab=swap, **config,
    )


# ── Activation ───────────────────────────────────────────────────────────
@triton.jit
def _act_kernel(gateup, down, hidden_size, expert_ids_ptr,
                BLOCK: tl.constexpr, ACT: tl.constexpr):
    half = hidden_size // 2
    pid = tl.program_id(0)
    if tl.load(expert_ids_ptr + pid) == -1:
        return
    base_g = gateup + pid * hidden_size
    base_d = down + pid * half
    for off in tl.range(0, half, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        m = idx < half
        gate = tl.load(base_g + idx, mask=m).to(tl.float32)
        up = tl.load(base_g + half + idx, mask=m)
        if ACT == "silu":
            act = gate * tl.sigmoid(gate)
        else:
            kA = 0.7978845608028654
            act = 0.5 * gate * (1 + 2 * tl.sigmoid(2 * kA * (gate + 0.044715 * gate * gate * gate)) - 1)
        tl.store(base_d + idx, (act.to(up.dtype) * up).to(base_d.dtype.element_ty), mask=m)


def _act_and_mul(gateup, down, topk_ids, activation="silu"):
    half = gateup.shape[1] // 2
    bs = 1024 if half >= 1024 else (512 if half >= 512 else 256)
    _act_kernel[(down.shape[0],)](gateup, down, gateup.shape[1], topk_ids.view(-1),
                                   BLOCK=bs, ACT=activation,
                                   num_warps=8 if bs >= 1024 else 4)


# ── Reduce ───────────────────────────────────────────────────────────────
@triton.jit
def _reduce_kernel(inp, s0, s1, s2, out, os0, os1,
                   T: int, topk: int, D: int, scale: tl.constexpr,
                   BM: tl.constexpr, BD: tl.constexpr, NS: tl.constexpr):
    s0 = tl.cast(s0, tl.int64); s1 = tl.cast(s1, tl.int64); os0 = tl.cast(os0, tl.int64)
    ot = tl.program_id(0) * BM + tl.arange(0, BM)
    od = tl.program_id(1) * BD + tl.arange(0, BD)
    mt, md = ot < T, od < D
    acc = tl.zeros((BM, BD), dtype=tl.float32)
    for i in tl.range(0, topk, num_stages=NS):
        acc += tl.load(inp + ot[:, None] * s0 + i * s1 + od[None, :],
                       mask=mt[:, None] & md[None, :], other=0.0).to(tl.float32)
    tl.store(out + ot[:, None] * os0 + od[None, :],
             (acc * scale).to(inp.dtype.element_ty), mask=mt[:, None] & md[None, :])


def _reduce(inp, out, scale=1.0):
    T, topk, D = inp.shape
    BD = min(triton.next_power_of_2(D), 8192) if D > 2048 else 2048
    BM = 4 if T > 1024 else (2 if T > 64 else 1)
    NS = 2 if T > 1024 else 1
    _reduce_kernel[(triton.cdiv(T, BM), triton.cdiv(D, BD))](
        inp, *inp.stride(), out, *out.stride(),
        T, topk, D, scale, BM=BM, BD=BD, NS=NS, num_warps=16)


# ── Main entry ───────────────────────────────────────────────────────────
def fused_experts_impl(
    hidden_states: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor,
    topk_weights: torch.Tensor, topk_ids: torch.Tensor,
    inplace: bool = False,
    activation: str = "silu",
    use_fp8_w8a8: bool = True,
    w1_scale: Optional[torch.Tensor] = None, w2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[List[int]] = None,
    filter_expert: bool = True,
) -> torch.Tensor:
    assert use_fp8_w8a8 and block_shape is not None
    num_tokens = hidden_states.shape[0]
    K_hidden = hidden_states.shape[1]
    E, N, _ = w1.shape
    w2_out = w2.shape[1]
    N_half = N // 2
    topk = topk_ids.shape[1]
    dev = hidden_states.device
    dtype = hidden_states.dtype
    bk = block_shape[1]

    cfg = _get_default_config(block_shape)
    b_tma = _support_tensor_descriptor
    total = num_tokens * topk

    cache = torch.empty(total * max(N, w2_out), device=dev, dtype=dtype)
    ic3 = cache[:num_tokens * topk * w2_out].view(num_tokens, topk, w2_out)
    cache2 = torch.empty(total * N_half, device=dev, dtype=dtype)

    compute_type = tl.bfloat16 if dtype == torch.bfloat16 else tl.float16
    out = hidden_states if inplace else torch.empty_like(hidden_states)

    g1_aq = torch.empty(num_tokens, K_hidden, device=dev, dtype=_fp8_dtype)
    g1_as = torch.empty(num_tokens, K_hidden // bk, device=dev, dtype=torch.float32)
    g2_aq = torch.empty(total, N_half, device=dev, dtype=_fp8_dtype)
    g2_as = torch.empty(total, N_half // bk, device=dev, dtype=torch.float32)

    ct = total
    ic1 = cache[:ct * N].view(ct, N)
    ic2 = cache2[:ct * N_half].view(ct, N_half)

    sorted_ids, expert_ids, npp = moe_align_block_size(topk_ids, cfg["BLOCK_SIZE_M"], E)

    _invoke(hidden_states, w1, ic1, None, w1_scale, topk_weights, topk_ids,
            sorted_ids, expert_ids, npp, False, topk, cfg, compute_type, block_shape,
            filter_expert=filter_expert, b_use_tma=b_tma,
            a_quant_buf=g1_aq, a_scale_buf=g1_as)

    _act_and_mul(ic1, ic2, topk_ids, activation)

    _invoke(ic2, w2, ic3, None, w2_scale, topk_weights, topk_ids,
            sorted_ids, expert_ids, npp, True, 1, cfg, compute_type, block_shape,
            filter_expert=filter_expert, b_use_tma=b_tma,
            a_quant_buf=g2_aq, a_scale_buf=g2_as)

    if topk == 1:
        out.copy_(ic3.squeeze(1))
    elif topk == 2:
        torch.add(ic3[:, 0], ic3[:, 1], out=out)
    else:
        _reduce(ic3, out)

    return out
