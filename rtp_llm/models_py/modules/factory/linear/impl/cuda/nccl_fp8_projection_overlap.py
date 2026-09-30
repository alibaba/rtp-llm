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
