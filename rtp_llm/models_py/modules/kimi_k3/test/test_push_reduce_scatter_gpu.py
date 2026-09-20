"""TP8 K3 push RS: rank ownership, changing grids, zeros and graph replay.

Run with eight SM100/SM103 NVLink GPUs; no model weights are required.
"""

import socket
from datetime import timedelta
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rtp_llm.models_py.distributed import push_reduce_scatter as push_rs
from rtp_llm.models_py.modules.kimi_k3 import gemm_reduce_scatter as rs


def ordered_reference(x, group):
    rank = group.rank()
    parts = [torch.empty_like(x) for _ in range(group.size())]
    dist.all_gather(parts, x, group=group)
    result = parts[0].chunk(group.size())[rank].float()
    for part in parts[1:]:
        result.add_(part.chunk(group.size())[rank].float())
    return result.bfloat16()


def worker(rank, port):
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=8,
        timeout=timedelta(minutes=3),
    )
    group = dist.group.WORLD
    rs.configure_gemm_reduce_scatter(group, device, max_m=8192, n=7168, use_fused=False)
    state = rs._STATES[(group, rank)]
    assert state.push is not None, "K3 push must be enabled on this supported setup"
    first_push = state.push
    rs.configure_gemm_reduce_scatter(
        group, device, max_m=8192, n=7168, fp8=True, use_fused=False
    )
    assert state.push is first_push, "BF16/FP8 must share the workspace"
    torch.manual_seed(1337 + rank)
    # Repeated non-monotone shapes exercise both phases and launch changes.
    for m in (0, 8, 128, 512, 1024, 24, 384, 8192, 8, 768, 256):
        x = torch.empty((m, 7168), dtype=torch.bfloat16, device=device)
        for kind in ("zeros", "rank_rows", "random"):
            if kind == "zeros":
                x.zero_()
            elif kind == "rank_rows":
                row = torch.arange(m, device=device)[:, None]
                col = torch.arange(7168, device=device)[None, :]
                x.copy_((row + col + rank) % 9 - 4)
            else:
                x.normal_().mul_(2.0 ** (rank - 4))
            with patch.object(
                rs.dist, "reduce_scatter_tensor", wraps=dist.reduce_scatter_tensor
            ) as nccl:
                out = rs.reduce_scatter(x, group)
                assert nccl.call_count == 0
            if not m:
                assert out.shape == (0, 7168)
                continue
            if kind == "rank_rows":
                expected = (
                    sum((row + col + peer) % 9 - 4 for peer in range(8))
                    .chunk(8)[rank]
                    .bfloat16()
                )
            else:
                expected = ordered_reference(x, group)
            torch.testing.assert_close(out, expected, rtol=0, atol=0)
        if rank == 0:
            print(
                f"PASS push M={m} zero / exact rank-row / random FP32 reference",
                flush=True,
            )

    # A rank-local storage offset must not split push/NCCL dispatch.
    offset = int(rank == 0)
    storage = torch.randn(8 * 7168 + offset, dtype=torch.bfloat16, device=device)
    x = storage[offset:].view(8, 7168)
    actual = rs.reduce_scatter(x, group)
    torch.testing.assert_close(actual, ordered_reference(x, group), rtol=0, atol=0)
    if rank == 0:
        print("PASS rank-local misaligned view keeps push on all ranks", flush=True)

    x = torch.randn(512, 7168, dtype=torch.bfloat16, device=device)
    # Match CudaGraphRunner: warm up on the original stream, synchronize the
    # device, then switch to a fresh stream only for capture.
    for _ in range(3):
        rs.reduce_scatter(x, group)
    torch.cuda.synchronize()
    capture_stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        # Two sizes and an immediate consumer detect stale output / phase state.
        out = rs.reduce_scatter(x, group)
        consumed = out.float() * 3
        small = rs.reduce_scatter(x[:8], group)
    for step in range(8):
        x.normal_().mul_(step + rank + 1)
        eager = rs.reduce_scatter(x, group)
        graph.replay()
        expected = ordered_reference(x, group)
        torch.testing.assert_close(out, expected, rtol=0, atol=0)
        torch.testing.assert_close(eager, expected, rtol=0, atol=0)
        torch.testing.assert_close(consumed, expected.float() * 3, rtol=0, atol=0)
        torch.testing.assert_close(
            small, ordered_reference(x[:8], group), rtol=0, atol=0
        )
    torch.cuda.synchronize()
    graph.reset()
    if rank == 0:
        print(
            "PASS changing-input CUDA Graph replay with immediate consumer", flush=True
        )

    # Equal TP sizes do not imply the same rank domain or workspace.
    second = dist.new_group(list(range(8)))
    rs.configure_gemm_reduce_scatter(second, device, max_m=8, n=7168, use_fused=False)
    assert (
        rs._STATES[(second, rank)].push.storage.data_ptr()
        != first_push.storage.data_ptr()
    )
    actual = rs.reduce_scatter(x[:8], second)
    torch.testing.assert_close(actual, ordered_reference(x[:8], second), rtol=0, atol=0)
    if rank == 0:
        print("PASS separate process groups own separate workspaces", flush=True)
    # One rank declining the topology must send the entire group to NCCL.
    fallback_group = dist.new_group(list(range(8)))
    with patch.object(push_rs, "_nvlink_peers", return_value=rank != 0):
        rs.configure_gemm_reduce_scatter(
            fallback_group, device, max_m=8, n=7168, use_fused=False
        )
    assert rs._STATES[(fallback_group, rank)].push is None
    x.fill_(rank)
    with patch.object(
        rs.dist, "reduce_scatter_tensor", wraps=dist.reduce_scatter_tensor
    ) as nccl:
        actual = rs.reduce_scatter(x[:8], fallback_group)
        assert nccl.call_count == 1
    torch.testing.assert_close(actual, torch.full_like(actual, 28), rtol=0, atol=0)
    if rank == 0:
        print(
            "PASS single-rank topology rejection selects NCCL on all ranks", flush=True
        )
    torch.cuda.synchronize()
    dist.barrier()
    dist.destroy_process_group(second)
    dist.destroy_process_group(fallback_group)
    dist.destroy_process_group()


if __name__ == "__main__":
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(worker, args=(port,), nprocs=8, join=True)
