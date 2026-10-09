"""Read-only paged KV gather/GQA expansion with original bmm operand strides."""

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["LENGTH", "FIRST_TOKEN"])
def _gather_kernel(
    CACHE,
    PAGES,
    K,
    V,
    S_PAGE: tl.constexpr,
    S_KV: tl.constexpr,
    S_HEAD: tl.constexpr,
    S_TOKEN: tl.constexpr,
    S_DIM: tl.constexpr,
    LENGTH,
    FIRST_TOKEN,
    HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    index = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    output_token = index // (HEADS * D)
    token = output_token + FIRST_TOKEN
    head = (index // D) % HEADS
    channel = index % D
    page = tl.load(PAGES + token // PAGE_SIZE, mask=output_token < LENGTH, other=-1).to(
        tl.int64
    )
    offset = (
        page * S_PAGE
        + (head // (HEADS // KV_HEADS)) * S_HEAD
        + (token % PAGE_SIZE) * S_TOKEN
        + channel * S_DIM
    )
    # The caller validates required-page coverage and physical bounds. Native
    # SWA's expired null pages are exact zeros and never read a reserved page.
    valid = (output_token < LENGTH) & (page >= 0)
    key = tl.load(CACHE + offset, mask=valid, other=0)
    value = tl.load(CACHE + offset + S_KV, mask=valid, other=0)
    tl.store(K + index, key, mask=output_token < LENGTH)
    tl.store(V + index, value, mask=output_token < LENGTH)


def gather(
    cache: torch.Tensor,
    pages: torch.Tensor,
    length: int,
    heads: int,
    first_token: int = 0,
):
    """Caller must reject missing required pages; return None when unsupported."""
    if (
        not cache.is_cuda
        or cache.dtype != torch.bfloat16
        or cache.dim() != 5
        or cache.shape[1] != 2
        or not pages.is_cuda
        or pages.device != cache.device
        or pages.dtype not in (torch.int32, torch.int64)
        or pages.dim() != 1
        or not pages.is_contiguous()
        or length <= 0
        or heads <= 0
        or heads % cache.shape[2]
        or first_token < 0
        or first_token + length > pages.numel() * cache.shape[3]
        or any(stride <= 0 for stride in cache.stride())
    ):
        return None
    dim = cache.shape[-1]
    key = torch.empty((length, heads, dim), dtype=cache.dtype, device=cache.device)
    value = torch.empty_like(key)
    _gather_kernel[(triton.cdiv(length * heads * dim, 2048),)](
        cache,
        pages,
        key,
        value,
        *cache.stride(),
        length,
        first_token,
        heads,
        cache.shape[2],
        cache.shape[3],
        dim,
        2048,
        num_warps=4,
    )
    return key.transpose(0, 1), value.transpose(0, 1)
