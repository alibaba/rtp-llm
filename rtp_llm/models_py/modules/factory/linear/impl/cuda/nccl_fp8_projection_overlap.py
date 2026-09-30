"""Overlap a BF16 NCCL gather with the local rows of an FP8 projection."""

import torch
import torch.distributed as dist


def all_gather_project_local_overlap(
    local_input: torch.Tensor, linear, process_group: dist.ProcessGroup
) -> torch.Tensor:
    """Preserve gather row order while projecting this rank's rows in flight.

    The communication tensor remains BF16. The caller selects a grouped FP8
    projection that accepts a contiguous caller-owned output buffer.
    """
    if local_input.ndim != 2 or not local_input.is_contiguous():
        raise ValueError("local projection overlap requires contiguous [M, K] input")
    world_size = dist.get_world_size(process_group)
    rank = dist.get_rank(process_group)
    if world_size < 2 or not 0 <= rank < world_size:
        raise ValueError("local projection overlap requires a multi-rank group")
    local_rows, width = local_input.shape
    gathered = local_input.new_empty((world_size * local_rows, width))
    output = local_input.new_empty((world_size * local_rows, linear.N))
    pending = dist.all_gather_into_tensor(
        gathered, local_input, group=process_group, async_op=True
    )
    local_start = rank * local_rows
    try:
        linear(local_input, out=output.narrow(0, local_start, local_rows))
    finally:
        pending.wait()
    if local_start:
        linear(gathered.narrow(0, 0, local_start), out=output.narrow(0, 0, local_start))
    remote_start = local_start + local_rows
    remote_rows = world_size * local_rows - remote_start
    if remote_rows:
        linear(
            gathered.narrow(0, remote_start, remote_rows),
            out=output.narrow(0, remote_start, remote_rows),
        )
    return output


def reduce_scatter_project_columns_overlap(
    values: torch.Tensor,
    scales: torch.Tensor,
    linear,
    process_group: dist.ProcessGroup,
    *,
    splits: int = 2,
) -> torch.Tensor:
    """Project FP8 output columns while earlier BF16 stripes reduce-scatter.

    The token dimension and NCCL dtype match an unsplit reduce-scatter. Each
    stripe has its own contiguous input and output, and the returned tensor
    restores the original column order.
    """
    world_size = dist.get_world_size(process_group)
    if (
        world_size < 2 or values.ndim != 2
        or values.shape[0] % world_size
        or splits < 2 or linear.N % splits
    ):
        raise ValueError("column overlap requires divisible tokens and columns")
    columns = linear.N // splits
    local_rows = values.shape[0] // world_size
    sources = []
    outputs = []
    works = []
    try:
        for stripe in range(splits):
            start = stripe * columns
            source = linear.forward_quantized_columns(
                values, scales, start, start + columns
            )
            if source.dtype != torch.bfloat16 or not source.is_contiguous():
                raise ValueError("column projection must produce contiguous BF16")
            output = source.new_empty((local_rows, columns))
            work = dist.reduce_scatter_tensor(
                output, source, group=process_group, async_op=True
            )
            sources.append(source)
            outputs.append(output)
            works.append(work)
    finally:
        for work in works:
            work.wait()
    return torch.cat(outputs, dim=1)
