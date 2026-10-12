"""Eight-rank real NCCL transport and production MoE integration on SM120."""

import os
import socket
import unittest
from dataclasses import replace
from datetime import timedelta
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rtp_llm.models_py.distributed.ep_stage_context import EpStageContext
from rtp_llm.models_py.modules.dsv4.moe.moe_layer import MoE
from rtp_llm.models_py.modules.dsv4.moe.strategies import MoeCfg, select_strategy
from rtp_llm.models_py.modules.dsv4.moe.strategies.nccl_ep_mxfp8 import (
    NcclEpMxfp8Strategy,
)
from rtp_llm.models_py.modules.dsv4.moe.strategies.v41_nccl_ep import V41NcclEpStrategy
from rtp_llm.models_py.modules.dsv4.v41_dispatch import pack_dispatch
from rtp_llm.models_py.modules.dsv4.v41_expert import V41_EXPERT_RECIPE
from rtp_llm.models_py.modules.dsv4.v41_moe import V41LocalMoE
from rtp_llm.models_py.modules.dsv4.v41_routed import V41RoutedExperts
from rtp_llm.utils.model_weight import W

DIM = 128
EXPERTS = 16
TOPK = 3


def config(rank, world=8, context=None, bound=32):
    return MoeCfg(
        layer_id=2,
        dim=DIM,
        moe_inter_dim=DIM,
        n_routed_experts=EXPERTS,
        n_activated_experts=TOPK,
        swiglu_limit=7.0,
        ep_size=world,
        ep_rank=rank,
        n_local_experts=EXPERTS // world,
        local_expert_start=rank * (EXPERTS // world),
        local_expert_end=(rank + 1) * (EXPERTS // world),
        max_tokens_per_rank=bound,
        expert_recipe=V41_EXPERT_RECIPE,
        stage_context=context,
    )


def weights():
    rng = torch.Generator().manual_seed(41810)
    return {
        "w13_w": torch.randint(
            0, 256, (EXPERTS, 2 * DIM, DIM // 2), generator=rng, dtype=torch.uint8
        ),
        "w13_s": torch.randint(
            122, 125, (EXPERTS, 2 * DIM, DIM // 32), generator=rng, dtype=torch.uint8
        ),
        "w2_w": torch.randint(
            0, 256, (EXPERTS, DIM, DIM // 2), generator=rng, dtype=torch.uint8
        ),
        "w2_s": torch.randint(
            122, 125, (EXPERTS, DIM, DIM // 32), generator=rng, dtype=torch.uint8
        ),
    }


def loader_weights(raw, begin, end, device):
    local = {
        name: value[begin:end].to(device).contiguous() for name, value in raw.items()
    }
    return {
        W.v4_routed_w1_w: local["w13_w"][:, :DIM].contiguous(),
        W.v4_routed_w3_w: local["w13_w"][:, DIM:].contiguous(),
        W.v4_routed_w1_s: local["w13_s"][:, :DIM].contiguous(),
        W.v4_routed_w3_s: local["w13_s"][:, DIM:].contiguous(),
        W.v4_routed_w2_w: local["w2_w"],
        W.v4_routed_w2_s: local["w2_s"],
    }


def request(rows, rank, phase, device):
    rng = torch.Generator().manual_seed(41820 + rank + phase * 17)
    x = (torch.randn(rows, DIM, generator=rng) * 0.2).bfloat16()
    routes = torch.rand(rows, TOPK, generator=rng) * 1.5
    ids = (torch.arange(rows * TOPK).reshape(rows, TOPK) + rank * 3 + phase) % EXPERTS
    if rows:
        routes[0] = 0
        ids[0] = -1
        x[0] = float("nan")
    return tuple(value.to(device).contiguous() for value in (x, routes, ids))


def make_strategy(cfg, raw, device):
    assert select_strategy(cfg) is V41NcclEpStrategy
    strategy = V41NcclEpStrategy(cfg)
    owned = loader_weights(raw, cfg.local_expert_start, cfg.local_expert_end, device)
    strategy.setup_weights(owned)
    assert not owned
    return strategy


def check_transport(rank, world, raw, device, context=None):
    cfg = config(rank, world, context)
    strategies = [make_strategy(cfg, raw, device) for _ in range(2)]
    cpu = V41LocalMoE(
        V41RoutedExperts(raw, swiglu_limit=7.0, backend="eager"),
        max_tokens=32,
        masked_token_limit=32,
        max_topk=TOPK,
    )
    # Distinct oracle: ordinary local BF16 execution for every source, then
    # gather FP32 results in rank order. It bypasses dispatch packing and A2A.
    for phase, counts in enumerate(
        (
            [0] * world,
            list(range(world)),
            [1] * world,
            [19 + (r % 3) for r in range(world)],
            [3 if r == world - 1 else 0 for r in range(world)],
        )
    ):
        strategy = strategies[phase % 2]
        x, route, ids = request(counts[rank], rank, phase, device)
        observed = {}
        forward_received = strategy._local.local.forward_received
        all_to_all = strategy._all_to_all

        def record_partial(*args, **kwargs):
            partial = forward_received(*args, **kwargs)
            assert partial.dtype == torch.float32
            observed["partial"] = partial.clone()
            return partial

        def record_return(send, send_counts, recv_counts, width, group):
            if width == DIM * 4:
                torch.testing.assert_close(
                    send,
                    observed["partial"].contiguous().view(torch.uint8),
                    rtol=0,
                    atol=0,
                )
            received = all_to_all(send, send_counts, recv_counts, width, group)
            if width == DIM * 4:
                observed["returned"] = received.view(torch.float32).clone()
            return received

        with (
            patch.dict(os.environ, {"DSV4_MOE_EXTENT_COUNT_FUSE": str(phase % 2)}),
            patch.object(strategy._local.local, "forward_received", record_partial),
            patch.object(strategy, "_all_to_all", record_return),
        ):
            assert strategy.synchronized_chunk_extent(x.shape[0], device) == max(counts)
            actual = strategy(x, route, ids)
        assert actual.dtype == torch.float32
        assert actual.shape == x.shape and bool(torch.isfinite(actual).all())
        assert strategy._pending_counts is None
        expected_sum = torch.zeros_like(actual)
        for partial in observed["returned"].reshape(world, x.shape[0], DIM):
            expected_sum.add_(partial)
        torch.testing.assert_close(actual, expected_sum, rtol=0, atol=0)
        payload = pack_dispatch(
            x,
            route,
            ids,
            torch.ones(x.shape[0], dtype=torch.bool, device=device),
            strategy.layout,
        )
        expected_payload = pack_dispatch(
            x.cpu(),
            route.cpu(),
            ids.cpu(),
            torch.ones(x.shape[0], dtype=torch.bool),
            strategy.layout,
            backend="eager",
        )
        torch.testing.assert_close(payload.cpu(), expected_payload, rtol=0, atol=0)
        references = []
        for peer, count in enumerate(counts):
            px, pw, pi = request(count, peer, phase, device)
            references.append(strategy._local(px, pw, pi))
        all_partial = torch.cat(references)
        gathered = [torch.empty_like(all_partial) for _ in range(world)]
        if sum(counts):
            dist.all_gather(gathered, all_partial, group=strategy._stage()[0])
            start, end = sum(counts[:rank]), sum(counts[: rank + 1])
            expected = torch.zeros_like(actual)
            for partial in gathered:
                expected.add_(partial[start:end])
            # Both layouts use the same BF16 expert GEMMs, but changing GEMM
            # M can cross quantization rounding boundaries. CPU checks below
            # independently cover the checkpoint arithmetic.
            torch.testing.assert_close(actual, expected, rtol=0.025, atol=0.015 * TOPK)
        if x.shape[0]:
            expected_cpu = cpu(x.cpu(), route.cpu(), ids.cpu())
            torch.testing.assert_close(
                actual.cpu(), expected_cpu, rtol=0.025, atol=0.015 * TOPK
            )
            assert bool((actual[0] == 0).all())
    return strategy


def check_layer(rank, raw, device):
    cfg = config(rank, bound=4)
    local = loader_weights(raw, cfg.local_expert_start, cfg.local_expert_end, device)
    rng = torch.Generator().manual_seed(41831)
    local[W.v4_router_w] = (
        torch.randn(EXPERTS, DIM, generator=rng).bfloat16().to(device)
    )
    local[W.v4_router_bias] = torch.linspace(-1, 1, EXPERTS, device=device)
    for weight, scale, n, k in (
        (W.v4_shared_w13_w, W.v4_shared_w13_s, DIM * 2, DIM),
        (W.v4_shared_w2_w, W.v4_shared_w2_s, DIM, DIM),
    ):
        local[weight] = (
            (torch.randn(n, k, generator=rng) * 0.025)
            .to(device)
            .to(torch.float8_e4m3fn)
        )
        local[scale] = torch.ones(n // 32, k // 32, device=device).to(
            torch.float8_e8m0fnu
        )
    layer = MoE(
        layer_id=2,
        dim=DIM,
        moe_inter_dim=DIM,
        n_routed_experts=EXPERTS,
        n_activated_experts=TOPK,
        n_shared_experts=1,
        score_func="sqrtsoftplus",
        route_scale=1.5,
        swiglu_limit=7.0,
        n_hash_layers=0,
        vocab_size=128,
        layer_weights=local,
        ep_size=8,
        ep_rank=rank,
        max_tokens_per_rank=4,
        expert_recipe=V41_EXPERT_RECIPE,
    )
    cpu = V41LocalMoE(
        V41RoutedExperts(raw, swiglu_limit=7.0, backend="eager"), max_topk=TOPK
    )
    for phase, n in enumerate((rank * 2, (7 - rank) * 2, 1)):
        x, _, _ = request(n, rank, phase + 10, device)
        x.nan_to_num_()
        ids = torch.arange(n, device=device)
        routed, selected = layer.gate(x, ids)
        if n:
            expected = (
                cpu(x.cpu(), routed.cpu(), selected.cpu())
                + layer.shared_experts(x).cpu().float()
            ).bfloat16()
        actual = layer(x, ids).clone()
        assert bool(torch.isfinite(actual).all())
        if n:
            torch.testing.assert_close(
                actual.cpu(), expected, rtol=0.025, atol=0.015 * TOPK
            )
        else:
            assert actual.shape == (0, DIM)


def check_peer_rejection(rank, raw, device):
    for variant in ("bounds", "recipe_option", "count_fuse"):
        cfg = config(rank, bound=31 if variant == "bounds" and rank == 0 else 32)
        strategy = make_strategy(cfg, raw, device)
        env = {
            "DSV4_MOE_LOCAL_REPLAY": (
                "1" if variant == "recipe_option" and rank == 0 else "0"
            ),
            "DSV4_MOE_EXTENT_COUNT_FUSE": (
                "1" if variant == "count_fuse" and rank == 0 else "0"
            ),
        }
        with patch.dict(os.environ, env):
            try:
                strategy(*request(1, rank, 80, device))
            except RuntimeError as error:
                assert "peer recipe" in str(error)
                assert strategy._failed
            else:
                raise AssertionError("mismatched peers were admitted")
    strategy = make_strategy(config(rank), raw, device)
    try:
        strategy(*request(33 if rank == 0 else 0, rank, 90, device))
    except ValueError as error:
        assert "per-rank bound" in str(error)
        assert strategy._failed
    else:
        raise AssertionError("oversized peer was admitted")
    strategy = make_strategy(config(rank), raw, device)
    strategy(*request(0, rank, 91, device))
    with patch.dict(os.environ, {"DSV4_MOE_EXTENT_COUNT_FUSE": "1"}):
        try:
            strategy.synchronized_chunk_extent(0, device)
        except RuntimeError as error:
            assert "changed after admission" in str(error)
            assert strategy._failed
        else:
            raise AssertionError("admitted collective options were changed")


def run_rank(rank, world, port):
    torch.cuda.set_device(rank)
    torch.set_num_threads(1)
    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["DSV4_MOE_STRATEGY"] = "auto"
    os.environ["MOEDBG"] = "0"
    os.environ["DSV4_MOE_CHUNK_PREFILL"] = "1"
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world,
        timeout=timedelta(seconds=120),
    )
    try:
        device = torch.device("cuda", rank)
        assert torch.cuda.get_device_capability() == (12, 0)
        raw = weights()
        with torch.inference_mode():
            check_transport(rank, world, raw, device)
            check_layer(rank, raw, device)
            check_peer_rejection(rank, raw, device)
            # Two independent EP4 groups must never exchange across stages.
            groups = [dist.new_group(list(range(start, start + 4))) for start in (0, 4)]
            stage, local_rank = rank // 4, rank % 4
            context = EpStageContext(
                groups[stage],
                tuple(range(stage * 4, stage * 4 + 4)),
                local_rank,
                4,
                stage,
                0,
                "fork_nccl_mxfp8",
            )
            stage_raw = {name: value.clone() for name, value in raw.items()}
            if stage:
                stage_raw["w2_w"].bitwise_xor_(0x88)
            check_transport(local_rank, 4, stage_raw, device, context)
            for group in groups:
                if group != dist.GroupMember.NON_GROUP_MEMBER:
                    dist.destroy_process_group(group)
        dist.barrier()
        if rank == 0:
            print(
                "V4.1 EP8 and two EP4 groups: payload/oracle, empty/uneven rows, chunked MoE, peer rejection PASS",
                flush=True,
            )
    finally:
        dist.destroy_process_group()


class V41NcclEpSM120Test(unittest.TestCase):
    def test_recipe_selection_and_guards(self):
        cfg = config(0)
        self.assertIs(select_strategy(cfg), V41NcclEpStrategy)
        self.assertIs(select_strategy(cfg, forced="v41_nccl_ep"), V41NcclEpStrategy)
        for bad in (replace(cfg, tp_size=2), replace(cfg, local_expert_start=1)):
            self.assertFalse(V41NcclEpStrategy.can_handle(bad))
        with self.assertRaisesRegex(ValueError, "recipe"):
            NcclEpMxfp8Strategy(cfg)
        with self.assertRaisesRegex(RuntimeError, "recipe"):
            select_strategy(cfg, forced="fork_nccl_mxfp8")

    def test_eight_rank_nccl_and_stage_groups(self):
        self.assertEqual(torch.cuda.device_count(), 8)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        mp.spawn(run_rank, args=(8, port), nprocs=8, join=True)


if __name__ == "__main__":
    unittest.main()
