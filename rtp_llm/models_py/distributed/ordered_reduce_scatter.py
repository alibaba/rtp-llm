"""Fixed-order FP32 reduce-scatter for the SM120 context-parallel MoE path."""

import torch
import torch.distributed as dist


def ordered_reduce_scatter(
    partial: torch.Tensor, process_group: dist.ProcessGroup
) -> torch.Tensor:
    """Reduce inputs whose destination shards are in bit-reversed rank order.

    The producer must write shard r at bit_reverse(r, log2(world)) before
    calling this function. Exchanges consume rank bits from low to high so
    the FP32 addition tree does not depend on row count or destination rank.
    ``ep_gather(..., output_rank_count=world)`` produces this layout without
    a separate permutation kernel. The returned rows are in ordinary order
    for this rank. All ranks must participate with equally shaped inputs.
    """
    world, rank = dist.get_world_size(process_group), dist.get_rank(process_group)
    if world < 1 or world & (world - 1):
        raise ValueError("ordered_reduce_scatter requires a power-of-two group size")
    if (
        not partial.is_cuda
        or partial.dtype != torch.float32
        or partial.ndim != 2
        or not partial.is_contiguous()
        or partial.shape[0] % world
    ):
        raise ValueError("ordered_reduce_scatter requires contiguous FP32 [world*M, D]")
    if partial.numel() == 0:
        return partial[: partial.shape[0] // world]
    current = partial
    for bit in range(world.bit_length() - 1):
        distance = 1 << bit
        lower, upper = current.chunk(2)
        keep, send = (upper, lower) if rank & distance else (lower, upper)
        received = torch.empty_like(keep)
        output = torch.empty_like(keep)
        peer = dist.get_global_rank(process_group, rank ^ distance)
        transfer_bytes = keep.numel() * keep.element_size()
        if transfer_bytes < 128 * 1024 * 1024:
            handles = dist.batch_isend_irecv(
                [
                    dist.P2POp(dist.isend, send, peer, group=process_group),
                    dist.P2POp(dist.irecv, received, peer, group=process_group),
                ]
            )
            for handle in handles:
                handle.wait()
            torch.add(keep, received, out=output)
            current = output
            continue
        chunk_rows = (keep.shape[0] + 7) // 8
        transfers = []
        # Enqueue all transfers before compute-stream waits so each addition
        # can overlap later receives without changing the rank reduction tree.
        for start in range(0, keep.shape[0], chunk_rows):
            span = slice(start, min(start + chunk_rows, keep.shape[0]))
            handles = dist.batch_isend_irecv(
                [
                    dist.P2POp(dist.isend, send[span], peer, group=process_group),
                    dist.P2POp(dist.irecv, received[span], peer, group=process_group),
                ]
            )
            transfers.append((span, handles))
        for span, handles in transfers:
            for handle in handles:
                handle.wait()
            torch.add(keep[span], received[span], out=output[span])
        current = output
    return current
