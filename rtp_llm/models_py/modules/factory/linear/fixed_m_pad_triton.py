import torch
import triton
import triton.language as tl


@triton.jit
def _pad_rows_kernel(source, output, valid, total, BLOCK: tl.constexpr):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(source + offsets, offsets < valid, other=0)
    tl.store(output + offsets, values, offsets < total)


def is_supported(source: torch.Tensor, rows: int) -> bool:
    return (
        torch.version.hip is None
        and source.is_cuda
        and source.ndim == 2
        and source.shape[1] > 0
        and rows > source.shape[0]
        and source.is_contiguous()
        and source.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and not source.requires_grad
    )


def try_pad_rows(source: torch.Tensor, rows: int) -> torch.Tensor | None:
    if not is_supported(source, rows):
        return None
    output = torch.empty(
        (rows, source.shape[1]), dtype=source.dtype, device=source.device
    )
    _pad_rows_kernel[(triton.cdiv(output.numel(), 1024),)](
        source, output, source.numel(), output.numel(), BLOCK=1024
    )
    return output
