"""Two-rank SM120 collective MoE eager/CUDA-graph regression."""

from __future__ import annotations

import os
import socket
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rtp_llm.models_py.modules.dsv4.moe.strategies.base import MoeCfg
from rtp_llm.models_py.modules.dsv4.moe.strategies.sm120_fused_moe import (
    Sm120FusedMoeStrategy,
)
from rtp_llm.models_py.utils.arch import is_sm120


class _RankScaledExperts(torch.nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        del weights, indices
        return x.float() * float(dist.get_rank() + 1)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _run_collective_rank(rank: int, world_size: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    try:
        device = torch.device("cuda", rank)
        if not is_sm120(device):
            raise RuntimeError(f"rank {rank} is not running on SM120")
        cfg = MoeCfg(
            layer_id=0,
            dim=8,
            moe_inter_dim=16,
            n_routed_experts=4,
            n_activated_experts=2,
            swiglu_limit=7.0,
            ep_size=world_size,
            ep_rank=rank,
            n_local_experts=2,
            local_expert_start=rank * 2,
            local_expert_end=(rank + 1) * 2,
            max_tokens_per_rank=4,
        )
        strategy = Sm120FusedMoeStrategy.__new__(Sm120FusedMoeStrategy)
        torch.nn.Module.__init__(strategy)
        strategy.cfg = cfg
        strategy._fused_moe = _RankScaledExperts()

        x = torch.full((2, 8), rank + 1.0, dtype=torch.bfloat16, device=device)
        weights = torch.full((2, 2), 0.5, dtype=torch.float32, device=device)
        indices = torch.tensor([[0, 2], [1, 3]], dtype=torch.int64, device=device)
        expected_scale = world_size * (world_size + 1) / 2

        eager = strategy._forward_collective(x, weights, indices)
        torch.testing.assert_close(eager, x.float() * expected_scale)
        dist.barrier()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_output = strategy._forward_collective(x, weights, indices)
        dist.barrier()
        x.fill_(rank + 2.0)
        graph.replay()
        torch.cuda.synchronize(device)
        torch.testing.assert_close(graph_output, x.float() * expected_scale)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


class Sm120CollectiveHardwareTest(unittest.TestCase):
    def test_two_rank_collective_eager_and_cuda_graph(self) -> None:
        if torch.cuda.device_count() < 2:
            self.skipTest("requires two SM120 GPUs")
        mp.spawn(
            _run_collective_rank,
            args=(2, _free_port()),
            nprocs=2,
            join=True,
        )


if __name__ == "__main__":
    unittest.main()
