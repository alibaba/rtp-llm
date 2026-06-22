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
    original_shape = dispatch_ids.shape
    if dispatch_ids.shape != dispatch_weights.shape:
        raise ValueError(
            f"dispatch_ids shape {list(dispatch_ids.shape)} must match "
            f"dispatch_weights shape {list(dispatch_weights.shape)}"
        )
    if dispatch_ids.device != dispatch_weights.device:
        raise ValueError(
            f"dispatch_ids device {dispatch_ids.device} must match "
            f"dispatch_weights device {dispatch_weights.device}"
        )
    if not dispatch_ids.is_contiguous() or not dispatch_weights.is_contiguous():
        raise ValueError("dispatch_ids and dispatch_weights must be contiguous")
    if dispatch_ids.dtype != torch.int32:
        raise ValueError(
            f"dispatch_ids must be int32, got {dispatch_ids.dtype}"
        )
    if dispatch_weights.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError(
            f"dispatch_weights must be float32/float16/bfloat16, got {dispatch_weights.dtype}"
        )

    N = dispatch_ids.numel()

    if N == 0:
        local_ids = torch.empty_like(dispatch_ids, dtype=torch.int32)
        local_weights = torch.empty(
            dispatch_ids.shape, dtype=torch.float32, device=dispatch_ids.device
        )
        return local_ids, local_weights

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

    if local_ids.shape != original_shape:
        local_ids = local_ids.view(original_shape)
        local_weights = local_weights.view(original_shape)

    return local_ids, local_weights
