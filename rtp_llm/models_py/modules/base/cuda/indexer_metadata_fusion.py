"""Copy packed verify page rows without an intermediate int64 request tensor."""

from typing import Optional

import torch
import triton as tr
import triton.language as tl


@tr.jit(do_not_specialize=["COLS", "STRIDE"])
def _expand_request_pages(Pages, Requests, Output, COLS, STRIDE, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    request = tl.load(Requests + row).to(tl.int64)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    page = tl.load(Pages + request * STRIDE + col, col < COLS, other=0)
    tl.store(Output + row * COLS + col, page, col < COLS)


def can_expand_request_pages(pages: torch.Tensor, requests: torch.Tensor) -> bool:
    """Check metadata-only eligibility before allocating capture-owned storage."""
    return (
        pages.is_cuda
        and pages.dtype == torch.int32
        and pages.ndim == 2
        and pages.stride(1) == 1
        and requests.is_cuda
        and requests.device == pages.device
        and requests.dtype == torch.int32
        and requests.ndim == 1
        and requests.is_contiguous()
    )


def try_expand_request_pages(
    pages: torch.Tensor, requests: torch.Tensor, out: Optional[torch.Tensor] = None
) -> Optional[torch.Tensor]:
    """Materialize this invocation's table for each packed verify query.

    Requests are the nonnegative in-range batch IDs from the packed metadata
    producer, including its padded rows. Keep page values untouched; zero or
    missing pages retain the scorer's existing validity contract.
    """
    if not can_expand_request_pages(pages, requests):
        return None
    shape = (requests.numel(), pages.shape[1])
    if out is not None and (
        tuple(out.shape) != shape
        or out.dtype != pages.dtype
        or out.device != pages.device
        or not out.is_contiguous()
    ):
        raise ValueError("packed Indexer page output changed shape/dtype/device")
    output = out if out is not None else torch.empty(shape, dtype=pages.dtype, device=pages.device)
    if output.numel():
        with torch.cuda.device(pages.device):
            _expand_request_pages[(requests.numel(), tr.cdiv(pages.shape[1], 256))](
                pages, requests, output, pages.shape[1], pages.stride(0), 256
            )
    return output
