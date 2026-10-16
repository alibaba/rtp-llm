"""Sort integer token indices and restore invalid padding in one launch."""

import torch
import triton
import triton.language as tl
from rtp_llm.models_py.triton_kernels.common.host_math import next_power_of_2


@triton.jit
def _canonicalize(
    out, WIDTH: tl.constexpr, SENTINEL: tl.constexpr, BLOCK: tl.constexpr
):
    row = tl.program_id(0).to(tl.int64)
    col = tl.arange(0, BLOCK).to(tl.int64)
    value = tl.load(out + row * WIDTH + col, col < WIDTH, SENTINEL)
    value = tl.where(value >= 0, value, SENTINEL)
    value = tl.sort(value, descending=False)
    value = tl.where(value == SENTINEL, -1, value)
    tl.store(out + row * WIDTH + col, value, col < WIDTH)


def is_supported(out):
    return (
        out.is_cuda
        and torch.version.hip is None
        and out.dtype in (torch.int32, torch.int64)
        and out.ndim == 2
        and out.is_contiguous()
        and 0 < out.shape[1] <= 4096
    )


def canonicalize_topk_(out):
    """Return False for unsupported tensors without modifying them."""
    if not is_supported(out):
        return False
    if out.shape[0]:
        block = next_power_of_2(out.shape[1])
        _canonicalize[(out.shape[0],)](
            out,
            out.shape[1],
            torch.iinfo(out.dtype).max,
            block,
            num_warps=(
                16
                if out.dtype == torch.int32 and out.shape[1] == 2048
                else (
                    8
                    if out.dtype == torch.int32 and out.shape[1] == 1024
                    else 4 if block <= 1024 else 8
                )
            ),
        )
    return True
