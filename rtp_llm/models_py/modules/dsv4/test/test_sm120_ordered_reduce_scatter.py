"""Validate the producer layout and FP32 tree for CP groups and subgroups."""

import gc
import tempfile
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rtp_llm.models_py.distributed.ordered_reduce_scatter import ordered_reduce_scatter
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.sm120_fused_moe import (
    Sm120FusedMoeExecutor,
)
from rtp_llm.models_py.triton_kernels.moe.ep_kernels import ep_gather


def _non_power_of_two_cp(rank):
    collective = "rtp_llm.models_py.distributed.collective_torch"
    executor_module = "rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.sm120_fused_moe"
    for ranks in ([1, 3, 5], list(range(6))):
        group = dist.new_group(ranks, backend="nccl")
        if rank in ranks:
            world, local_rank = len(ranks), ranks.index(rank)
            local_experts = 48 // world

            def local_partial(x, weights, ids, *, output_rank_count=1):
                source = x * (local_rank + 1)
                indices = (
                    torch.arange(x.size(0), device="cuda")
                    .view(-1, 1)
                    .expand_as(ids)
                    .contiguous()
                )
                partial = torch.empty_like(source)
                ep_gather(
                    source,
                    ids,
                    weights,
                    indices,
                    partial,
                    output_rank_count=output_rank_count,
                )
                return partial

            executor = SimpleNamespace(
                cfg=SimpleNamespace(
                    moe_cp_size=world,
                    n_local_experts=local_experts,
                    local_expert_start=local_rank * local_experts,
                ),
                _local=SimpleNamespace(forward_eager=local_partial),
            )
            for rows in (1, 17, 128):
                x = (
                    torch.arange(rows * 128, device="cuda", dtype=torch.float32).view(
                        rows, 128
                    )
                    + local_rank
                )
                weights = torch.ones((rows, world), device="cuda")
                ids = (
                    torch.arange(world, device="cuda")
                    .mul(local_experts)
                    .expand(rows, -1)
                    .contiguous()
                )
                with patch(f"{collective}._get_group", return_value=group), patch(
                    f"{executor_module}._get_group", return_value=group
                ):
                    actual = Sm120FusedMoeExecutor._forward_cp_reduce_scatter(
                        executor, x, weights, ids
                    )
                torch.testing.assert_close(
                    actual, x * (world * (world + 1) // 2), rtol=0, atol=0
                )
            torch.cuda.synchronize()
            dist.destroy_process_group(group)
        dist.barrier()


def _worker(rank, rendezvous):
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=rendezvous,
        rank=rank,
        world_size=8,
        timeout=timedelta(seconds=180),
        device_id=torch.device("cuda", rank),
    )
    try:
        for world in (1, 2, 4, 8):
            # Interleaved groups exercise subgroup-to-global peer translation.
            groups = []
            for offset in range(8 // world):
                ranks = list(range(offset, 8, 8 // world))
                group = dist.new_group(ranks, backend="nccl")
                groups.append(group)
                if rank in ranks:
                    selected, local_rank = group, ranks.index(rank)
            fence = torch.zeros((), device="cuda")
            dist.all_reduce(fence, group=selected)
            shapes = [(rows, 513) for rows in (0, 1, 4, 17, 128)]
            if world > 1:
                # Cross the transfer threshold with uneven chunk tails.
                shapes.append((65536 // world * 2 + 1, 513))
            if world == 8:
                shapes.append((4097, 4096))
            for rows, hidden in shapes:
                torch.manual_seed(912)
                values = (
                    torch.randn(world, 7, hidden)
                    * torch.tensor([1, 0.01, 10, 0.1, 100, 1, 0.001, 0.1])[
                        :world, None, None
                    ]
                )
                tree = values.clone()
                while tree.shape[0] > 1:
                    tree = tree[::2] + tree[1::2]
                tokens = world * rows
                logical_ids = torch.arange(tokens) % 7
                source = values[local_rank].index_select(0, logical_ids).cuda()
                ids = torch.zeros(tokens, 1, device="cuda", dtype=torch.int64)
                weights = torch.ones(tokens, 1, device="cuda")
                indices = torch.arange(tokens, device="cuda").reshape(tokens, 1)
                partial = torch.empty_like(source)
                bits = world.bit_length() - 1
                permutation = [
                    sum(((r >> b) & 1) << (bits - b - 1) for b in range(bits))
                    for r in range(world)
                ]
                expected_layout = source.reshape(world, rows, hidden)[
                    permutation
                ].reshape_as(source)
                expected = (
                    tree[0]
                    .index_select(0, logical_ids)
                    .reshape(world, rows, hidden)[local_rank]
                    .cuda()
                )

                def execute():
                    ep_gather(
                        source, ids, weights, indices, partial, output_rank_count=world
                    )
                    return ordered_reduce_scatter(partial, selected)

                for _ in range(3):
                    actual = execute()
                    torch.testing.assert_close(partial, expected_layout, rtol=0, atol=0)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                if not rows:
                    continue
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        dist.all_reduce(fence, group=selected)
                        execute()
                torch.cuda.current_stream().wait_stream(stream)
                torch.cuda.synchronize()
                dist.barrier()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    dist.all_reduce(fence, group=selected)
                    captured = execute()
                for _ in range(3):
                    partial.fill_(float("nan"))
                    captured.fill_(float("nan"))
                    graph.replay()
                    torch.testing.assert_close(captured, expected, rtol=0, atol=0)
                torch.cuda.synchronize()
                del graph, captured
                gc.collect()
            dist.barrier()
            for group in groups:
                if group != dist.GroupMember.NON_GROUP_MEMBER:
                    dist.destroy_process_group(group)
        _non_power_of_two_cp(rank)
    finally:
        dist.destroy_process_group()


def test_producer_and_subgroup_reduction():
    assert torch.cuda.device_count() >= 8, "Eight GPUs are required"
    with tempfile.TemporaryDirectory() as directory:
        rendezvous = (Path(directory) / "rendezvous").as_uri()
        mp.spawn(_worker, args=(rendezvous,), nprocs=8, join=True)
