"""Distributed projection correctness across staging/direct/overlap boundaries."""

import os
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rtp_llm.models_py.modules.factory.linear.impl.cuda.bf16_comm_fusion import (
    Bf16CollectiveProjection,
)
from rtp_llm.models_py.modules.factory.linear.impl.cuda.f16_linear import CudaF16Linear


def _worker(rank, rendezvous):
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=rendezvous, rank=rank, world_size=2)
    try:
        torch.manual_seed(173 + rank)
        device = torch.device("cuda", rank)
        output_weight = torch.randn(128, 7168, device=device, dtype=torch.bfloat16) * 0.02
        projection = CudaF16Linear(output_weight)
        inputs = CudaF16Linear(
            torch.randn(7168, 128, device=device, dtype=torch.bfloat16) * 0.02
        )
        collective = Bf16CollectiveProjection(
            dist.group.WORLD, device, max_m=8192, hidden_size=7168,
            output_projections=[projection],
        )
        for local_rows in (16, 127, 128, 256, 4095, 4096):
            x = torch.randn(local_rows, 7168, device=device, dtype=torch.bfloat16)
            gathered = x.new_empty(local_rows * 2, 7168)
            dist.all_gather_into_tensor(gathered, x)
            actual = collective.all_gather_gemm(x, inputs)
            torch.testing.assert_close(actual, inputs(gathered), atol=0.125, rtol=0.02)

            values = torch.randn(local_rows * 2, 128, device=device, dtype=torch.bfloat16)
            partial = projection(values)
            expected = partial.new_empty(local_rows, 7168)
            dist.reduce_scatter_tensor(expected, partial)
            actual = collective.gemm_reduce_scatter(values, projection)
            torch.testing.assert_close(actual, expected, atol=0.03125, rtol=0.02)
    finally:
        dist.destroy_process_group()


class Bf16CollectiveProjectionTest(unittest.TestCase):
    def test_projection_boundaries(self):
        if torch.cuda.device_count() < 2:
            self.skipTest("requires two CUDA GPUs")
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(_worker, args=("file://" + os.path.join(directory, "rendezvous"),), nprocs=2)


if __name__ == "__main__":
    unittest.main()
