"""Graph-safe gated residual branch mixing for Qwen4."""

import torch
import triton
import triton.language as tl
from rtp_llm.models_py.utils.device_capability import (
    cached_device_capability,
    sm90_pdl_enabled,
)


@triton.jit
def _mix_reduce_kernel(
    logits_ptr,
    normed_ptr,
    output_ptr,
    HIDDEN: tl.constexpr,
    BRANCHES: tl.constexpr,
    BLOCK: tl.constexpr,
    PDL: tl.constexpr = False,
):
    row = tl.program_id(0)
    columns = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = columns < HIDDEN
    if PDL:
        base = row * (BRANCHES * HIDDEN) + columns
        n0 = tl.load(normed_ptr + base, mask=mask, other=0).to(tl.float32)
        n1 = tl.load(normed_ptr + base + HIDDEN, mask=mask, other=0).to(tl.float32)
        n2 = tl.load(normed_ptr + base + 2 * HIDDEN, mask=mask, other=0).to(tl.float32)
        n3 = tl.load(normed_ptr + base + 3 * HIDDEN, mask=mask, other=0).to(tl.float32)
        tl.extra.cuda.gdc_wait()
        l0 = tl.load(logits_ptr + base, mask=mask, other=0).to(tl.float32)
        l1 = tl.load(logits_ptr + base + HIDDEN, mask=mask, other=0).to(tl.float32)
        l2 = tl.load(logits_ptr + base + 2 * HIDDEN, mask=mask, other=0).to(tl.float32)
        l3 = tl.load(logits_ptr + base + 3 * HIDDEN, mask=mask, other=0).to(tl.float32)
        g0 = (1.0 / (1.0 + tl.exp(-l0))).to(tl.bfloat16).to(tl.float32)
        g1 = (1.0 / (1.0 + tl.exp(-l1))).to(tl.bfloat16).to(tl.float32)
        g2 = (1.0 / (1.0 + tl.exp(-l2))).to(tl.bfloat16).to(tl.float32)
        g3 = (1.0 / (1.0 + tl.exp(-l3))).to(tl.bfloat16).to(tl.float32)
        acc = (
            (g0 * n0).to(tl.bfloat16).to(tl.float32)
            + (g1 * n1).to(tl.bfloat16).to(tl.float32)
            + (g2 * n2).to(tl.bfloat16).to(tl.float32)
            + (g3 * n3).to(tl.bfloat16).to(tl.float32)
        )
        tl.extra.cuda.gdc_launch_dependents()
        tl.store(output_ptr + row * HIDDEN + columns, acc / BRANCHES, mask=mask)
    else:
        acc = tl.full((BLOCK,), 0, tl.float32)
        for branch in tl.static_range(BRANCHES):
            offset = row * (BRANCHES * HIDDEN) + branch * HIDDEN + columns
            logits = tl.load(logits_ptr + offset, mask=mask, other=0).to(tl.float32)
            normed = tl.load(normed_ptr + offset, mask=mask, other=0).to(tl.float32)
            # The eager path materializes both sigmoid and the product in BF16.
            gate = (1.0 / (1.0 + tl.exp(-logits))).to(tl.bfloat16).to(tl.float32)
            product = (gate * normed).to(tl.bfloat16).to(tl.float32)
            acc += product
        tl.store(output_ptr + row * HIDDEN + columns, acc / BRANCHES, mask=mask)


def is_supported(logits: torch.Tensor, normed: torch.Tensor, branches: int) -> bool:
    return (
        logits.is_cuda
        and torch.version.hip is None
        and logits.dtype == normed.dtype == torch.bfloat16
        and logits.shape == normed.shape
        and logits.device == normed.device
        and logits.is_contiguous()
        and normed.is_contiguous()
        and logits.dim() >= 2
        and branches == 4
        and logits.shape[-1] % branches == 0
    )


def fused_mix_reduce(
    logits: torch.Tensor, normed: torch.Tensor, branches: int
) -> torch.Tensor:
    if not is_supported(logits, normed, branches):
        raise ValueError("unsupported Qwen4 gated residual mix layout")
    hidden = int(logits.shape[-1]) // branches
    rows = logits.numel() // (branches * hidden)
    result = torch.empty(
        (*logits.shape[:-1], hidden), device=logits.device, dtype=logits.dtype
    )
    block, warps = 128, 4
    if hidden == 2560:
        capability = cached_device_capability(logits.device)
        if capability == (9, 0) and rows >= 4096:
            block, warps = 256, 2
        elif capability == (12, 0):
            # Fewer CTAs for the bandwidth-bound prefill consumer.
            if 4093 <= rows <= 4096:
                block = 2048
            elif rows == 12288:
                block, warps = 256, 2
    pdl = sm90_pdl_enabled(logits.device)
    _mix_reduce_kernel[(rows, triton.cdiv(hidden, block))](
        logits,
        normed,
        result,
        HIDDEN=hidden,
        BRANCHES=branches,
        BLOCK=block,
        PDL=pdl,
        num_warps=warps,
        launch_pdl=pdl,
    )
    return result
