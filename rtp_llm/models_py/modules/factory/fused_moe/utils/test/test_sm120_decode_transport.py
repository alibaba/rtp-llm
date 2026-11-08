"""Real NCCL dispatch rounds with uneven and empty DP ranks."""

from datetime import timedelta
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors import sm120_decode

_CASES = {
    2: ((0, 0), (8, 8), (9, 8), (17, 0), (0, 17), (17, 5)),
    4: (
        (0, 0, 0, 0),
        (8, 8, 8, 8),
        (9, 8, 7, 6),
        (17, 0, 17, 0),
        (17, 5, 3, 0),
        (17, 17, 17, 17),
    ),
}


def _worker(rank, world, rendezvous):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=rendezvous,
        rank=rank,
        world_size=world,
        timeout=timedelta(seconds=60),
    )
    try:
        # Exercise the production capacity boundary with small allocations.
        sm120_decode._A2A_PAYLOAD_TOKEN_BOUND = 16
        executor = sm120_decode.Sm120DecodeExecutor.__new__(
            sm120_decode.Sm120DecodeExecutor
        )
        torch.nn.Module.__init__(executor)
        experts_per_rank = 4 // world
        executor.cfg = SimpleNamespace(
            n_routed_experts=4,
            n_local_experts=experts_per_rank,
            local_expert_start=rank * experts_per_rank,
        )
        executor._chunk_tokens = 6

        def weighted_identity(x, weights, indices, input_scale):
            # Isolate transport from GEMM approximation; quantization and both
            # NCCL exchanges still use their production implementations.
            scale = input_scale.view(torch.float8_e8m0fnu).float()
            values = (x.float().reshape(-1, 4, 32) * scale[..., None]).reshape(-1, 128)
            return values * weights.sum(-1, keepdim=True)

        executor._experts = weighted_identity
        for counts in _CASES[world]:
            rows = counts[rank]
            x = torch.full((rows, 128), 2**rank, dtype=torch.bfloat16, device=rank)
            weights = torch.full((rows, 2), 0.5, device=rank)
            indices = torch.tensor([0, 2], device=rank).expand(rows, 2).contiguous()
            actual = executor._forward_all_to_all(x, weights, indices)
            assert actual.dtype == torch.float32
            torch.testing.assert_close(actual, x.float(), rtol=0, atol=0)
        dist.barrier()
    finally:
        dist.destroy_process_group()


def test_bounded_dispatch_with_real_nccl_2rank(tmp_path):
    assert torch.cuda.device_count() >= 2
    mp.spawn(
        _worker,
        args=(2, (tmp_path / "rendezvous").as_uri()),
        nprocs=2,
        join=True,
    )


def test_bounded_dispatch_with_real_nccl_4rank(tmp_path):
    assert torch.cuda.device_count() >= 4
    mp.spawn(
        _worker,
        args=(4, (tmp_path / "rendezvous").as_uri()),
        nprocs=4,
        join=True,
    )
