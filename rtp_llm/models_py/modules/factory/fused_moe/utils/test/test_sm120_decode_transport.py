"""Real NCCL dispatch rounds with uneven and empty DP ranks."""

from datetime import timedelta
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rtp_llm.models_py.modules.factory.fused_moe.defs.decode_plan import DecodeMoePlan
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors import (
    sm120_decode,
    sm120_graph_decode,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.sm120.graph_decode import (
    Sm120DecodeMoeGeometry,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.sm120.workspace import (
    Sm120MoeWorkspace,
)
from rtp_llm.ops.compute_ops import DecodeExecutionMode, DecodeExecutionPhase

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
    8: (
        (0, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 17, 0),
        (13, 0, 1, 7, 0, 2, 0, 4),
        (3, 3, 3, 3, 3, 3, 3, 3),
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
        experts = max(4, world * 2)
        experts_per_rank = experts // world
        executor.cfg = SimpleNamespace(
            n_routed_experts=experts,
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
            indices = (
                torch.tensor([0, experts - 1], device=rank).expand(rows, 2).contiguous()
            )
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


def test_bounded_dispatch_with_real_nccl_8rank(tmp_path):
    assert torch.cuda.device_count() >= 8
    mp.spawn(
        _worker,
        args=(8, (tmp_path / "rendezvous").as_uri()),
        nprocs=8,
        join=True,
    )


class _IdentityMaskedExperts(torch.nn.Module):
    """Transport oracle independent of grouped GEMM and packed DeepGEMM scales."""

    def _forward_coalesced(self, x, weights, ids, scales, geometry, start, workspace):
        scale = torch.exp2(scales.float() - 127)
        values = (
            x.float().reshape(geometry.rows, -1, 32) * scale.unsqueeze(-1)
        ).reshape_as(workspace.partial)
        owned = (ids >= start) & (ids < start + geometry.local_experts)
        owned_weight = torch.where(owned, weights, 0.0).sum(-1, keepdim=True)
        workspace.partial.copy_(values * owned_weight)


def _fixed_graph_worker(rank, world, rendezvous):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=rendezvous,
        rank=rank,
        world_size=world,
        timeout=timedelta(seconds=60),
    )
    try:
        sm120_graph_decode.require_sm120_decode_graph_providers()
        device = torch.device("cuda", rank)
        executor = sm120_graph_decode.Sm120GraphDecodeExecutor.__new__(
            sm120_graph_decode.Sm120GraphDecodeExecutor
        )
        torch.nn.Module.__init__(executor)
        geometry = Sm120DecodeMoeGeometry(4, world, 640, 512, world * 2, 3)
        executor.cfg = SimpleNamespace(
            local_expert_start=rank * geometry.local_experts,
            sm120_workspace=Sm120MoeWorkspace(),
        )
        executor._eager = SimpleNamespace(experts=_IdentityMaskedExperts())
        executor.cfg.sm120_workspace.reserve_decode_graph_workspace(geometry, device)

        x = torch.empty((4, 640), dtype=torch.bfloat16, device=device)
        weights = torch.empty((4, 3), dtype=torch.float32, device=device)
        ids = torch.empty((4, 3), dtype=torch.int64, device=device)
        validity = torch.zeros(4, dtype=torch.bool, device=device)
        graph_plan = DecodeMoePlan(
            DecodeExecutionMode.GRAPH, DecodeExecutionPhase.TARGET, 4, 1, validity
        )
        fixed_plan = DecodeMoePlan(
            DecodeExecutionMode.FIXED, DecodeExecutionPhase.TARGET, 4, 1, validity
        )

        def set_inputs(active, value):
            x.fill_(value)
            weights.copy_(torch.tensor([0.25, 0.25, 0.5], device=device))
            ids.copy_(torch.tensor([0, 0, world * 2 - 1], device=device))
            validity.copy_(torch.arange(4, device=device) < active)
            # Make stale-tail reuse visible even when the next pass shrinks.
            x.masked_fill_(~validity[:, None], float("nan"))

        set_inputs(4, 1.0)
        executor._forward_fixed(x, weights, ids, graph_plan, geometry)
        torch.cuda.synchronize()
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = executor._forward_fixed(x, weights, ids, graph_plan, geometry)

        for active in (4, int(rank == world - 1), rank % 3, int(rank == 0) * 4):
            set_inputs(active, float(2 ** (rank % 4)))
            fixed = executor._forward_fixed(x, weights, ids, fixed_plan, geometry)
            graph.replay()
            expected = torch.where(validity[:, None], x.float(), 0.0)
            assert captured.dtype == torch.float32
            torch.testing.assert_close(fixed, expected, rtol=0, atol=0)
            torch.testing.assert_close(captured, fixed, rtol=0, atol=0)
        dist.barrier()
    finally:
        dist.destroy_process_group()


def test_fixed_dispatch_graph_with_real_nccl_4rank(tmp_path):
    assert torch.cuda.device_count() >= 4
    mp.spawn(
        _fixed_graph_worker,
        args=(4, (tmp_path / "rendezvous").as_uri()),
        nprocs=4,
        join=True,
    )


def test_fixed_dispatch_graph_with_real_nccl_8rank(tmp_path):
    assert torch.cuda.device_count() >= 8
    mp.spawn(
        _fixed_graph_worker,
        args=(8, (tmp_path / "rendezvous").as_uri()),
        nprocs=8,
        join=True,
    )
