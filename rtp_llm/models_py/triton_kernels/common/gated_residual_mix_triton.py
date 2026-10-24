"""Graph-safe gated residual branch mixing for Qwen4."""

import torch
import triton
import triton.language as tl


@triton.jit
def _mix_reduce_kernel(
    logits_ptr,
    normed_ptr,
    output_ptr,
    HIDDEN: tl.constexpr,
    BRANCHES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = columns < HIDDEN
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
    if hidden == 2560 and torch.cuda.get_device_capability(logits.device) == (12, 0):
        # Fewer CTAs for the bandwidth-bound prefill consumer. The four BF16
        # products and their sequential FP32 sum keep the original boundaries.
        if 4093 <= rows <= 4096:
            block = 2048
        elif rows == 12288:
            block, warps = 256, 2
    _mix_reduce_kernel[(rows, triton.cdiv(hidden, block))](
        logits,
        normed,
        result,
        HIDDEN=hidden,
        BRANCHES=branches,
        BLOCK=block,
        num_warps=warps,
    )
    return result
