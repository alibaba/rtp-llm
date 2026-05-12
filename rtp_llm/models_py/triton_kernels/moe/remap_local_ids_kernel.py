import torch
import triton
import triton.language as tl


@triton.jit
def _remap_to_local_ids_kernel(
    ids_ptr,
    weights_ptr,
    out_ids_ptr,
    out_weights_ptr,
    local_start,
    local_end,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    ids = tl.load(ids_ptr + offsets, mask=mask, other=0)
    weights = tl.load(weights_ptr + offsets, mask=mask, other=0.0)

    is_local = (ids >= local_start) & (ids < local_end)
    local_ids = tl.where(is_local, ids - local_start, 0)
    local_weights = tl.where(is_local, weights, 0.0)

    tl.store(out_ids_ptr + offsets, local_ids, mask=mask)
    tl.store(out_weights_ptr + offsets, local_weights, mask=mask)


def remap_to_local_ids(
    dispatch_ids: torch.Tensor,
    dispatch_weights: torch.Tensor,
    local_start: int,
    local_end: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert dispatch_ids.is_contiguous()
    assert dispatch_weights.is_contiguous()

    N = dispatch_ids.numel()
    local_ids = torch.empty_like(dispatch_ids, dtype=torch.int32)
    local_weights = torch.empty(
        dispatch_ids.shape, dtype=torch.float32, device=dispatch_ids.device
    )

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)

    _remap_to_local_ids_kernel[grid](
        dispatch_ids.view(-1),
        dispatch_weights.to(torch.float32).view(-1),
        local_ids.view(-1),
        local_weights.view(-1),
        local_start,
        local_end,
        N,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return local_ids, local_weights
