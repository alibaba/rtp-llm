"""Real two-rank FP4 migration, static replicas, and captured-graph execution."""

import os
import tempfile
import unittest
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rtp_llm.eplb.runtime import _shuffle_weights
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.test.mega_moe_routed_only_execution_test import (
    _DIM,
    _EXPERTS,
    _INTER_DIM,
    _MAX_TOKENS,
    _TOPK,
    _make_global_weights,
    _make_moe,
    _sync_ranks_for_mega,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.fp8_fp4.chunked_layer import (
    ChunkedFp8Fp4MoeLayer,
)
from rtp_llm.models_py.utils.expert_load_monitor import ExpertLoadContext
from rtp_llm.utils.model_weight import W

_WORLD = 2
_PHYSICAL = 2 * _EXPERTS


def _inverse(physical):
    mapping = torch.full((_EXPERTS, _PHYSICAL - _EXPERTS + 1), -1, dtype=torch.int32)
    counts = torch.zeros(_EXPERTS, dtype=torch.int32)
    for slot, expert in enumerate(physical.tolist()):
        mapping[expert, counts[expert]] = slot
        counts[expert] += 1
    return mapping, counts


def _assert_close(actual, expected):
    assert torch.isfinite(actual).all()
    relative = (actual.float() - expected.float()).abs().mean() / (
        expected.float().abs().mean() + 1e-6
    )
    assert relative.item() < 0.05, f"FP4 mean relative error {relative.item()}"


def _worker(rank, rendezvous, hash_routing):
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        backend="nccl",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=_WORLD,
        device_id=device,
        timeout=timedelta(seconds=180),
    )
    try:
        torch.manual_seed(20261022)
        global_weights = _make_global_weights(
            device, include_shared=True, hash_routing=hash_routing
        )
        physical = torch.tensor(list(range(_EXPERTS)) + [0] * _EXPERTS)
        mapping, counts = _inverse(physical)
        local = physical[rank * _EXPERTS : (rank + 1) * _EXPERTS].to(device)
        routed_keys = (W.moe_w1, W.moe_s1, W.moe_w2, W.moe_s2)
        local_weights = {}
        for key, value in global_weights.items():
            # UE8M0 needs byte indexing; its values must not be converted.
            if key in routed_keys:
                local_weights[key] = value.view(torch.uint8)[local].view(value.dtype)
            else:
                local_weights[key] = value.clone()
        local_weights[W.log2phy] = mapping.to(device)
        local_weights[W.logic_expert_cnt] = counts.to(device)
        moe = ChunkedFp8Fp4MoeLayer(
            layer_id=7,
            dim=_DIM,
            moe_inter_dim=_INTER_DIM,
            n_routed_experts=_EXPERTS,
            n_physical_experts=_PHYSICAL,
            n_activated_experts=_TOPK,
            n_shared_experts=1,
            score_func="sqrtsoftplus",
            route_scale=1.5,
            swiglu_limit=10.0,
            n_hash_layers=8 if hash_routing else 0,
            vocab_size=64,
            layer_weights=local_weights,
            ep_size=_WORLD,
            ep_rank=rank,
            max_tokens_per_rank=_MAX_TOKENS,
            strategy="mega_moe_se",
            moe_w1_layout="gate_up",
        )
        reference = _make_moe(
            {key: value.clone() for key, value in global_weights.items()},
            ep_size=1,
            ep_rank=0,
            strategy="local_loop",
            n_shared_experts=1,
            hash_routing=hash_routing,
        )
        state = moe.fused_moe.eplb_layer
        assert not moe.fused_moe.supports_gate_pack
        pointers = tuple(
            t.data_ptr()
            for t in (*state.weights, state.logical_to_physical, state.logical_count)
        )
        torch.manual_seed(20261030 + rank)
        x = torch.randn(3, _DIM, dtype=torch.bfloat16, device=device)
        ids = torch.arange(3, device=device, dtype=torch.int64)
        step_counts = torch.zeros(_EXPERTS, dtype=torch.int32, device=device)
        context = ExpertLoadContext(step_counts)
        with torch.inference_mode():
            expected = reference(x, input_ids=ids, expert_load=None).clone()
            _sync_ranks_for_mega(device)
            before = moe(x, input_ids=ids, expert_load=context).clone()
        _assert_close(before, expected)
        total = step_counts.sum()
        dist.all_reduce(total)
        assert total.item() == _WORLD * x.size(0) * _TOPK
        step_counts.zero_()
        _sync_ranks_for_mega(device)
        graph = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(graph):
            captured = moe(x, input_ids=ids, expert_load=context)

        # Remove extra replicas of 0, introduce replicas of 1, and cycle the
        # first rank's slots. Rank 1 must receive every runtime tensor of 1.
        next_physical = torch.tensor(list(range(1, _EXPERTS)) + [0] + [1] * _EXPERTS)
        next_mapping, next_counts = _inverse(next_physical)
        with torch.inference_mode():
            _shuffle_weights(state, next_physical, _WORLD)
            state.logical_to_physical.copy_(next_mapping)
            state.logical_count.copy_(next_counts)
            state.physical_to_logical.copy_(next_physical)
            captured.fill_(float("nan"))
            step_counts.zero_()
        _sync_ranks_for_mega(device)
        graph.replay()
        torch.cuda.synchronize(device)
        _assert_close(captured, expected)
        _assert_close(captured, before)
        assert pointers == tuple(
            t.data_ptr()
            for t in (*state.weights, state.logical_to_physical, state.logical_count)
        )
        total = step_counts.sum()
        dist.all_reduce(total)
        assert total.item() == _WORLD * x.size(0) * _TOPK
        with torch.inference_mode():
            _sync_ranks_for_mega(device)
            eager = moe(x, input_ids=ids, expert_load=None).clone()
        _assert_close(eager, captured)
    finally:
        dist.destroy_process_group()


class EplbMegaMoeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available() or torch.cuda.device_count() < _WORLD:
            raise AssertionError("EPLB MegaMoE test requires two CUDA GPUs")

    def _run(self, hash_routing):
        with tempfile.TemporaryDirectory(dir=os.environ.get("TEST_TMPDIR")) as tmp:
            mp.spawn(
                _worker,
                args=(os.path.join(tmp, "rendezvous"), hash_routing),
                nprocs=_WORLD,
                join=True,
            )

    def test_score_gate_migration_updates_captured_backend(self):
        self._run(False)

    def test_hash_gate_migration_updates_captured_backend(self):
        self._run(True)


if __name__ == "__main__":
    unittest.main()
