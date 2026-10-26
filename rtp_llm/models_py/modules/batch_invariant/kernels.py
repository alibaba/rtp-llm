"""Canonical CUDA arithmetic for the dense-v1 batch invariant profile.

GEMM constrains reduction algorithms; attention visits fixed logical key blocks
ending at the query's causal position. Packing, future tokens and physical KV
page placement cannot change that order. See UPSTREAM.md for vLLM provenance.
"""

from functools import lru_cache

import torch
import triton
import triton.language as tl

from .persistent_matmul import matmul_kernel_persistent
from .unified_attention import unified_attention_kernel


@lru_cache
def _num_sms(device_index):
    return torch.cuda.get_device_properties(device_index).multi_processor_count


@lru_cache
def _device_major(device_index):
    return torch.cuda.get_device_properties(device_index).major


def linear(x, weight, bias=None, *, output_dtype=None):
    """F.linear layout: weight is [out_features, in_features]."""
    if x.device.type != "cuda" or x.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("batch invariant linear requires CUDA BF16 or FP32")
    if weight.device != x.device or weight.dtype != x.dtype:
        raise ValueError(
            "batch invariant linear requires matching input/weight dtype and device"
        )
    if weight.ndim != 2 or x.shape[-1] != weight.shape[1]:
        raise ValueError("invalid batch invariant linear dimensions")
    if bias is not None and (
        bias.shape != (weight.shape[0],) or bias.device != x.device
    ):
        raise ValueError("invalid batch invariant linear bias")
    shape = x.shape[:-1]
    x = x.reshape(-1, x.shape[-1]).contiguous()
    m, k = x.shape
    n = weight.shape[0]
    dtype = output_dtype or x.dtype
    if (
        x.dtype == torch.bfloat16
        and _device_major(x.device.index) in (9, 10)
        and k % 8 == 0
        and n % 8 == 0
    ):
        from rtp_llm.ops.compute_ops import rtp_llm_ops

        # vLLM's Hopper/Blackwell strategy, expressed as operation-local
        # cuBLASLt descriptors: FP32 accumulation, no split-K or workspace.
        # Native code handles the existing weight strides without a weight copy.
        # Normalize exceptional sliced-buffer alignment. In particular the
        # Blackwell M=1 GEMM workaround is unsupported for a misaligned RHS.
        # Dispatch for odd feature widths is fixed by (N,K), never by batch.
        if x.data_ptr() % 16:
            x = x.clone()
        if weight.data_ptr() % 16:
            weight = weight.clone()
        out = rtp_llm_ops.batch_invariant_cublaslt_gemm(
            x, weight, dtype == torch.float32 or bias is not None
        )
        if bias is not None:
            out = out + bias.float()
        return out.to(dtype=dtype).view(*shape, n)
    out = torch.empty((m, n), dtype=dtype, device=x.device)
    if m:
        # vLLM's persistent GEMM configuration. No shape-dependent autotuning
        # or split-K: changing M changes work assignment, not a row's reduction.
        bm, bn, bk = 128, 128, 32 if x.dtype == torch.float32 else 64
        sms = _num_sms(x.device.index)
        matmul_kernel_persistent[(min(sms, triton.cdiv(m, bm) * triton.cdiv(n, bn)),)](
            x,
            weight,
            out,
            bias,
            m,
            n,
            k,
            x.stride(0),
            x.stride(1),
            weight.stride(1),
            weight.stride(0),
            out.stride(0),
            out.stride(1),
            BLOCK_SIZE_M=bm,
            BLOCK_SIZE_N=bn,
            BLOCK_SIZE_K=bk,
            GROUP_SIZE_M=8,
            NUM_SMS=sms,
            A_LARGE=x.numel() > 2**31,
            B_LARGE=weight.numel() > 2**31,
            C_LARGE=out.numel() > 2**31,
            HAS_BIAS=bias is not None,
            num_warps=8,
            num_stages=3,
        )
    return out.view(*shape, n)


@triton.jit
def _rmsnorm(X, W, Y, H: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    x = tl.load(X + row * H + col, col < H, 0).to(tl.float32)
    w = tl.load(W + col, col < H, 0).to(tl.float32)
    var = tl.sum(x * x, 0) / H
    y = (x * tl.rsqrt(var + EPS)) * w
    tl.store(Y + row * H + col, y, col < H)


def rmsnorm(x, weight, eps):
    x = x.contiguous()
    h = x.shape[-1]
    if weight.shape != (h,) or x.device != weight.device:
        raise ValueError("invalid batch invariant RMSNorm weight")
    out = torch.empty_like(x)
    if x.numel():
        _rmsnorm[(x.numel() // h,)](
            x, weight, out, h, eps, triton.next_power_of_2(h), num_warps=4
        )
    return out


@triton.jit
def _lse_parts(X, P, V: tl.constexpr, PARTS: tl.constexpr, BLOCK: tl.constexpr):
    row, part = tl.program_id(0), tl.program_id(1)
    i = part * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + row * V + i, i < V, -float("inf")).to(tl.float32)
    maximum = tl.max(x, 0)
    total = tl.sum(tl.exp(x - maximum), 0)
    tl.store(P + row * PARTS * 2 + part * 2, maximum)
    tl.store(P + row * PARTS * 2 + part * 2 + 1, total)


@triton.jit
def _lse_merge(P, Y, PARTS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    i = tl.arange(0, BLOCK)
    maximum = tl.load(P + row * PARTS * 2 + i * 2, i < PARTS, -float("inf"))
    total = tl.load(P + row * PARTS * 2 + i * 2 + 1, i < PARTS, 0)
    m = tl.max(maximum, 0)
    s = tl.sum(total * tl.exp(maximum - m), 0)
    tl.store(Y + row, m + tl.log(s))


def logsumexp(logits):
    if logits.ndim != 2 or logits.shape[1] == 0 or not logits.is_cuda:
        raise ValueError(
            "batch invariant logsumexp requires nonempty CUDA vocabulary rows"
        )
    x = logits.contiguous()
    rows, vocab = x.shape
    parts = triton.cdiv(vocab, 1024)
    partials = torch.empty((rows, parts, 2), device=x.device, dtype=torch.float32)
    out = torch.empty((rows, 1), device=x.device, dtype=torch.float32)
    if rows:
        _lse_parts[(rows, parts)](x, partials, vocab, parts, 1024, num_warps=4)
        _lse_merge[(rows,)](
            partials, out, parts, triton.next_power_of_2(parts), num_warps=4
        )
    return out


def selected_logprobs(logits, token_ids):
    if token_ids.numel() != logits.shape[0]:
        raise ValueError("one target token is required per logits row")
    return logits.float().gather(1, token_ids.reshape(-1, 1).long()) - logsumexp(logits)


def paged_attention(q, cache, block_table, request_ids, positions, *, query_start=None):
    """Causal GQA over [pages, 2, kv_heads, page_size, head_dim] KV."""
    if q.dtype != torch.bfloat16 or cache.dtype != q.dtype:
        raise ValueError("dense-v1 attention requires BF16 Q and KV")
    if cache.ndim != 5 or cache.shape[1] != 2 or cache.shape[-1] != q.shape[-1]:
        raise ValueError("invalid batch invariant paged KV layout")
    h, d = q.shape[1:]
    kh, page = cache.shape[2:4]
    if h % kh or d not in (64, 128, 256) or q.stride(-1) != 1:
        raise ValueError("unsupported batch invariant GQA shape")
    if request_ids.numel() != q.shape[0] or positions.numel() != q.shape[0]:
        raise ValueError("one request and position are required per query")
    if query_start is not None and (
        query_start.device != q.device
        or query_start.dtype != torch.int32
        or query_start.ndim != 1
        or query_start.numel() != block_table.shape[0] + 1
    ):
        raise ValueError(
            "packed queries require one CUDA int32 offset per request plus end"
        )
    out = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    if q.shape[0]:
        group = h // kh
        bm = max(16, triton.next_power_of_2(group))
        bq = bm // group
        packed = query_start is not None
        blocks = q.shape[0] // bq + block_table.shape[0] if packed else q.shape[0]
        unified_attention_kernel[(blocks, kh)](
            q,
            cache,
            block_table,
            request_ids,
            positions,
            query_start,
            out,
            block_table.shape[0],
            q.stride(0),
            q.stride(1),
            *cache.stride(),
            block_table.stride(0),
            h,
            group,
            d,
            page,
            d**-0.5,
            BLOCK_M=bm,
            BLOCK_Q=bq,
            TILE_SIZE=32,
            PACKED=packed,
            num_warps=4,
            num_stages=2,
        )
    return out
