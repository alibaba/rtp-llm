"""Real NCCL dispatch rounds with uneven and empty DP ranks."""

import os
from datetime import timedelta
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors import sm120_decode


def _worker(rank, rendezvous):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    try:
        # Exercise the production capacity boundary with small allocations.
        sm120_decode._A2A_PAYLOAD_TOKEN_BOUND = 16
        os.environ["DSV4_MOE_RETURN_DTYPE"] = "fp32"
        executor = sm120_decode.Sm120DecodeExecutor.__new__(
            sm120_decode.Sm120DecodeExecutor
        )
        torch.nn.Module.__init__(executor)
        executor.cfg = SimpleNamespace(
            n_routed_experts=4, n_local_experts=2, local_expert_start=rank * 2
        )
        executor._chunk_tokens = 6

        def weighted_identity(x, weights, indices, input_scale):
            # Isolate transport from GEMM approximation; quantization and both
            # NCCL exchanges still use their production implementations.
            scale = input_scale.view(torch.float8_e8m0fnu).float()
            values = (x.float().reshape(-1, 4, 32) * scale[..., None]).reshape(-1, 128)
            return values * weights.sum(-1, keepdim=True)

        executor._experts = weighted_identity
        for counts in ((0, 0), (8, 8), (9, 8), (17, 0), (0, 17), (17, 5)):
            rows = counts[rank]
            x = torch.full((rows, 128), 2**rank, dtype=torch.bfloat16, device=rank)
            weights = torch.full((rows, 2), 0.5, device=rank)
            indices = torch.tensor([0, 2], device=rank).expand(rows, 2).contiguous()
            actual = executor._forward_all_to_all(x, weights, indices)
            torch.testing.assert_close(actual, x.float(), rtol=0, atol=0)
        dist.barrier()
    finally:
        dist.destroy_process_group()


def test_bounded_dispatch_with_real_nccl(tmp_path):
    assert torch.cuda.device_count() >= 2
    mp.spawn(_worker, args=((tmp_path / "rendezvous").as_uri(),), nprocs=2, join=True)
