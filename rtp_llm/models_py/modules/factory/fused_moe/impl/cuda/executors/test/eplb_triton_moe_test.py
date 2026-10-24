"""Run logical routes against migrated BF16 weights through a captured graph."""

import os
import tempfile
import unittest
from datetime import timedelta
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from rtp_llm.eplb.runtime import EplbLayer
from rtp_llm.eplb.shuffle import expert_model_parallel_shuffle_inplace
from rtp_llm.models_py.distributed import collective_torch as collective
from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import FusedMoe
from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
    FusedMoEQuantConfig,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.triton_fused_executor import (
    TritonFusedMoeExecutor,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.pure_tp_router import (
    PureTpRouterNoQuant,
)
from rtp_llm.models_py.utils.expert_load_monitor import ExpertLoadContext
from rtp_llm.utils.model_weight import W


def _exercise_migration(rank, world_size):
    torch.manual_seed(20261022)
    device = torch.device("cuda", rank)
    logical_w1 = torch.randn(2, 256, 128, device=device, dtype=torch.bfloat16) * 0.1
    logical_w2 = torch.randn(2, 128, 128, device=device, dtype=torch.bfloat16) * 0.1
    physical = torch.tensor([0, 1, 0, 0], device=device)
    local_experts = 4 // world_size
    local = physical[rank * local_experts : (rank + 1) * local_experts]
    config = SimpleNamespace(
        ep_size=world_size,
        ep_rank=rank,
        tp_size=world_size,
        physical_expert_num=4,
        moe_k=2,
    )
    quant = FusedMoEQuantConfig()
    executor = TritonFusedMoeExecutor(
        config,
        quant,
        {W.moe_w1: logical_w1[local], W.moe_w2: logical_w2[local]},
    )
    state = EplbLayer(
        0,
        torch.tensor([[0, 2, 3], [1, -1, -1]], dtype=torch.int32, device=device),
        torch.tensor([3, 1], dtype=torch.int32, device=device),
        executor.get_eplb_weights(),
        ep_rank=rank,
    )
    moe = FusedMoe(PureTpRouterNoQuant(config, quant), executor, 4, eplb_layer=state)
    stable_tensors = (
        *state.weights,
        state.logical_to_physical,
        state.logical_count,
    )
    pointers = tuple(t.data_ptr() for t in stable_tensors)
    # TP ranks receive identical tokens and must choose identical physical slots.
    x = torch.randn(16, 128, device=device, dtype=torch.bfloat16)
    ids = torch.tensor([[0, 1]] * 16, device=device, dtype=torch.int32)
    routing_weights = torch.rand(16, 2, device=device)
    routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
    counts = torch.zeros(local_experts, device=device, dtype=torch.int32)
    context = ExpertLoadContext(counts)

    # Independent logical expert computation: no physical layout or remap.
    reference_parts = []
    for expert in range(2):
        # The Triton executor consumes w1 with the up rows before the gate rows.
        up, gate = F.linear(x, logical_w1[expert]).chunk(2, dim=-1)
        hidden = (F.silu(gate.float()) * up.float()).to(x.dtype)
        # GEMM2 applies the route weight before rounding its accumulator to BF16.
        output = F.linear(hidden.float(), logical_w2[expert].float())
        reference_parts.append((output * routing_weights[:, expert, None]).to(x.dtype))
    reference = torch.stack(reference_parts, dim=1).sum(dim=1)

    with torch.inference_mode():
        before = moe(x, routing_weights, ids, context).clone()
        torch.testing.assert_close(before, reference, rtol=0.02, atol=0.02)
        warmup = torch.cuda.Stream()
        warmup.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup):
            for _ in range(3):
                moe(x, routing_weights, ids, context)
        torch.cuda.current_stream().wait_stream(warmup)
        torch.cuda.synchronize(device)
        if world_size > 1:
            dist.barrier()
        counts.zero_()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = moe(x, routing_weights, ids, context)

        # Change replica ownership and cycle the only source of expert 1.
        next_physical = torch.tensor([1, 0, 1, 1])
        expert_model_parallel_shuffle_inplace(
            state.physical_to_logical.unsqueeze(0),
            [state.weights],
            next_physical.unsqueeze(0),
            dist.group.WORLD,
        )
        state.logical_to_physical.copy_(
            torch.tensor([[1, -1, -1], [0, 2, 3]], dtype=torch.int32)
        )
        state.logical_count.copy_(torch.tensor([1, 3], dtype=torch.int32))
        state.physical_to_logical.copy_(next_physical)
        captured.fill_(float("nan"))
        counts.zero_()
        if world_size > 1:
            dist.barrier()
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(captured, reference, rtol=0.02, atol=0.02)
        torch.testing.assert_close(captured, before, rtol=0, atol=0)
        expected_counts = torch.bincount(
            state.map_ids(ids).long().flatten(), minlength=4
        )
        expected_counts = expected_counts[
            rank * local_experts : (rank + 1) * local_experts
        ]
        torch.testing.assert_close(counts, expected_counts.int(), rtol=0, atol=0)
        assert pointers == tuple(t.data_ptr() for t in stable_tensors)
        eager = moe(x, routing_weights, ids, None)
        torch.testing.assert_close(eager, captured, rtol=0, atol=0)


def _worker(rank, rendezvous, world_size):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=world_size,
        device_id=torch.device("cuda", rank),
        timeout=timedelta(seconds=180),
    )
    collective._group_map = {collective.Group.DP_AND_TP: dist.group.WORLD}
    collective._parallelism_config = SimpleNamespace(
        tp_size=world_size, dp_size=1, world_size=world_size
    )
    collective._initialized = True
    try:
        _exercise_migration(rank, world_size)
    finally:
        collective._group_map.clear()
        collective._parallelism_config = None
        collective._initialized = False
        dist.destroy_process_group()


class EplbTritonMoeTest(unittest.TestCase):
    def _run(self, world_size):
        with tempfile.TemporaryDirectory(dir=os.environ.get("TEST_TMPDIR")) as tmp:
            mp.spawn(
                _worker,
                args=(os.path.join(tmp, "rendezvous"), world_size),
                nprocs=world_size,
                join=True,
            )

    def test_migration_preserves_logical_result_in_eager_and_graph(self):
        self.assertTrue(torch.cuda.is_available(), "This test requires a CUDA GPU")
        self._run(1)

    def test_tp_ep_replicated_inputs_survive_migration_and_graph_replay(self):
        self.assertGreaterEqual(torch.cuda.device_count(), 2)
        self._run(2)


if __name__ == "__main__":
    unittest.main()
