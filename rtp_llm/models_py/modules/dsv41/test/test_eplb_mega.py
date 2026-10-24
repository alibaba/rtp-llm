"""Private V4.1 Mega backends: real counts, runtime migration, and graph replay.

Routed weights use packed FP4/group32; standalone shared weights use V4.1
native group32 FP8, while the selectable fused backend uses group128. Numerical
checks are bounded floating-point comparisons, not a claim of bitwise equality
or a real-checkpoint/full-model precision result.
"""

import hashlib
import os
import tempfile
import unittest
from datetime import timedelta
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rtp_llm.eplb.runtime import RuntimeExpertBalancer
from rtp_llm.models_py.modules.dsv41.moe.moe_layer import MoE
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.test.mega_moe_routed_only_execution_test import (
    _DIM,
    _EXPERTS,
    _INTER_DIM,
    _MAX_TOKENS,
    _TOPK,
    _make_global_weights,
    _sync_ranks_for_mega,
)
from rtp_llm.models_py.utils.expert_load_monitor import ExpertLoadContext
from rtp_llm.utils.model_weight import W

_WORLD = 2
_PHYSICAL = 2 * _EXPERTS
_LAYER = 7


def _inverse(physical):
    mapping = torch.full((_EXPERTS, _PHYSICAL - _EXPERTS + 1), -1, dtype=torch.int32)
    counts = torch.zeros(_EXPERTS, dtype=torch.int32)
    for slot, expert in enumerate(physical.tolist()):
        mapping[expert, counts[expert]] = slot
        counts[expert] += 1
    return mapping, counts


def _private_weights(weights):
    return {
        W.v4_router_w: weights[W.moe_gate],
        W.v4_router_bias: weights[W.moe_gate_bias],
        W.v4_routed_w1_w: weights[W.moe_w1][:, :_INTER_DIM],
        W.v4_routed_w1_s: weights[W.moe_s1][:, :_INTER_DIM],
        W.v4_routed_w3_w: weights[W.moe_w1][:, _INTER_DIM:],
        W.v4_routed_w3_s: weights[W.moe_s1][:, _INTER_DIM:],
        W.v4_routed_w2_w: weights[W.moe_w2],
        W.v4_routed_w2_s: weights[W.moe_s2],
        W.v4_shared_w13_w: weights[W.ffn_w13],
        W.v4_shared_w13_s: weights[W.ffn_s13],
        W.v4_shared_w2_w: weights[W.ffn_w2],
        W.v4_shared_w2_s: weights[W.ffn_s2],
    }


def _make_private(weights, rank, strategy, physical=None):
    return MoE(
        layer_id=_LAYER,
        dim=_DIM,
        moe_inter_dim=_INTER_DIM,
        n_routed_experts=_EXPERTS,
        n_physical_experts=physical,
        n_activated_experts=_TOPK,
        n_shared_experts=1,
        score_func="sqrtsoftplus",
        route_scale=1.5,
        swiglu_limit=10.0,
        n_hash_layers=0,
        vocab_size=64,
        layer_weights=weights,
        ep_size=_WORLD if physical is not None else 1,
        ep_rank=rank,
        max_tokens_per_rank=_MAX_TOKENS,
        is_decode_role=True,
        strategy=strategy,
    )


def _assert_close(actual, expected, max_relative_rmse):
    assert actual.data_ptr() != expected.data_ptr(), "output-cache self comparison"
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
    actual, expected = actual.float(), expected.float()
    error = (actual - expected).square().mean().sqrt()
    relative_rmse = error / expected.square().mean().sqrt().clamp_min(1e-6)
    assert relative_rmse.item() < max_relative_rmse, relative_rmse.item()


def _local_hashes(state):
    # Preserve packed FP4/UE8M0 bytes, including scales with non-contiguous views.
    return [
        tuple(
            hashlib.sha256(
                weight[slot]
                .contiguous()
                .reshape(-1)
                .view(torch.uint8)
                .cpu()
                .numpy()
                .tobytes()
            ).hexdigest()
            for weight in state.weights
        )
        for slot in range(state.num_local_experts)
    ]


def _assert_kernel_counts(moe, x, ids, counts):
    _, logical = moe.gate(x, ids)
    physical = moe.eplb_layer.map_ids(logical)
    expected = torch.bincount(physical.flatten(), minlength=_PHYSICAL)
    dist.all_reduce(expected)
    start = moe.ep_rank * moe.n_local_experts
    torch.testing.assert_close(
        counts[_LAYER].long(),
        expected[start : start + moe.n_local_experts],
        rtol=0,
        atol=0,
    )
    assert expected.sum().item() == _WORLD * x.size(0) * _TOPK
    assert counts[:_LAYER].count_nonzero().item() == 0


def _worker(rank, rendezvous, strategy, shared_block):
    # The explicit private strategy must not be superseded by a worker's env.
    for key in (
        "DSV4_MOE_STRATEGY",
        "DSV4_USE_MEGA_MOE",
        "DSV4_USE_MEGA_MOE_SE",
        "DSV4_USE_GROUPED_FP4",
    ):
        os.environ.pop(key, None)
    os.environ["MOEDBG"] = "0"
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
        canonical = _make_global_weights(device, include_shared=True)
        # Unit exponents retain the same real scale in either shared recipe.
        for scale_key, weight_key in ((W.ffn_s13, W.ffn_w13), (W.ffn_s2, W.ffn_w2)):
            out_dim, in_dim = canonical[weight_key].shape
            canonical[scale_key] = torch.full(
                (out_dim // shared_block, in_dim // shared_block),
                127,
                dtype=torch.uint8,
                device=device,
            ).view(torch.float8_e8m0fnu)
        global_weights = _private_weights(canonical)
        physical = torch.tensor(list(range(_EXPERTS)) + [0] * _EXPERTS)
        mapping, replicas = _inverse(physical)
        local = physical[rank * _EXPERTS : (rank + 1) * _EXPERTS].to(device)
        routed_keys = {
            W.v4_routed_w1_w,
            W.v4_routed_w1_s,
            W.v4_routed_w2_w,
            W.v4_routed_w2_s,
            W.v4_routed_w3_w,
            W.v4_routed_w3_s,
        }
        local_weights = {
            key: (
                value.contiguous().view(torch.uint8)[local].view(value.dtype)
                if key in routed_keys
                else value.clone()
            )
            for key, value in global_weights.items()
        }
        local_weights[W.log2phy] = mapping.to(device)
        local_weights[W.logic_expert_cnt] = replicas.to(device)
        moe = _make_private(local_weights, rank, strategy, _PHYSICAL)
        reference = _make_private(
            {key: value.clone() for key, value in global_weights.items()},
            0,
            "local_loop",
        )
        assert moe._strategy.name == strategy
        assert moe._strategy.cfg.shared_fp8_block_size == shared_block
        assert not moe._use_gate_pack()
        state = moe.eplb_layer
        assert len(state.weights) == 4
        stable = (*state.weights, state.logical_to_physical, state.logical_count)
        pointers = tuple(t.data_ptr() for t in stable)
        hashes = [None] * _WORLD
        dist.all_gather_object(hashes, _local_hashes(state))
        canonical_hashes = {}
        for slot, digest in enumerate(sum(hashes, [])):
            logical = physical[slot].item()
            if logical in canonical_hashes:
                assert canonical_hashes[logical] == digest
            canonical_hashes[logical] = digest
        assert len(canonical_hashes) == _EXPERTS
        torch.manual_seed(20261030 + rank)
        x = torch.randn(3, _DIM, dtype=torch.bfloat16, device=device)
        ids = torch.arange(3, device=device, dtype=torch.int64)
        counts = torch.zeros(_LAYER + 1, _EXPERTS, dtype=torch.int32, device=device)
        context = ExpertLoadContext(counts[_LAYER])
        with torch.inference_mode():
            expected = reference(x, ids, expert_load=None).clone()
            _sync_ranks_for_mega(device)
            before = moe(x, ids, expert_load=context).clone()
            _assert_close(before, expected, 0.05)
            _assert_kernel_counts(moe, x, ids, counts)
            counts.zero_()
            _sync_ranks_for_mega(device)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = moe(x, ids, expert_load=context)
            counts.zero_()
            captured.fill_(float("nan"))
            _sync_ranks_for_mega(device)
            graph.replay()
            torch.cuda.synchronize(device)
            _assert_close(captured, before, 0.01)
            _assert_kernel_counts(moe, x, ids, counts)
            runtime = RuntimeExpertBalancer(
                [state],
                SimpleNamespace(
                    eplb_mode=3,
                    eplb_update_time=1,
                    eplb_stats_window_size=1,
                    eplb_balance_layer_per_step=1,
                    balance_method="round",
                ),
                ep_rank=rank,
                ep_size=_WORLD,
                model_id=0,
            )
            runtime.record_step(counts)
            counts.zero_()
            assert runtime.balance(), "real kernel counts must cause a placement update"
            after_map = state.physical_to_logical.clone()
            assert (after_map != physical).any()
            assert (state.logical_count.cpu() > 1).any()
            # Rank 1 initially owns only copies of expert 0. Its new nonzero
            # experts necessarily cross rank; changed positions also test slots.
            assert (after_map[_EXPERTS:] != 0).any()
            gathered_maps = [None] * _WORLD
            dist.all_gather_object(gathered_maps, after_map.tolist())
            assert gathered_maps[0] == gathered_maps[1]
            for slot, digest in enumerate(_local_hashes(state)):
                logical = after_map[rank * _EXPERTS + slot].item()
                assert digest == canonical_hashes[logical], (rank, slot, logical)
            assert pointers == tuple(
                t.data_ptr()
                for t in (
                    *state.weights,
                    state.logical_to_physical,
                    state.logical_count,
                )
            )
            captured.fill_(float("nan"))
            _sync_ranks_for_mega(device)
            graph.replay()
            torch.cuda.synchronize(device)
            after = captured.clone()
            _assert_close(after, before, 0.01)
            _assert_close(after, expected, 0.05)
            _assert_kernel_counts(moe, x, ids, counts)
            _sync_ranks_for_mega(device)
            eager = moe(x, ids, expert_load=None).clone()
            _assert_close(eager, after, 0.01)
    finally:
        dist.destroy_process_group()


class EplbPrivateMegaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available() or torch.cuda.device_count() < _WORLD:
            raise AssertionError("V4.1 EPLB Mega tests require two CUDA GPUs")

    def _run(self, strategy, shared_block):
        with tempfile.TemporaryDirectory(dir=os.environ.get("TEST_TMPDIR")) as tmp:
            mp.spawn(
                _worker,
                args=(os.path.join(tmp, "rendezvous"), strategy, shared_block),
                nprocs=_WORLD,
                join=True,
            )

    def test_native32_shared_standalone_mega(self):
        self._run("mega", 32)

    def test_block128_shared_fused_mega_se(self):
        # Production selection keeps native32 on standalone Mega. Exercise the
        # private fused strategy with its selectable block128 shared recipe.
        self._run("mega_se", 128)


if __name__ == "__main__":
    unittest.main()
