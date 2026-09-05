"""Two-rank production coverage for routed-only, standalone, and fused MegaMoE."""

from __future__ import annotations

import os
import tempfile
import unittest
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rtp_llm.models_py.modules.dsv4.moe_layer import Dsv4MoeLayer as MoE
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.local_loop import (
    LocalLoopExecutor,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe import (
    MegaMoeExecutor,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe_se import (
    MegaMoeSEExecutor,
)
from rtp_llm.utils.model_weight import W

_WORLD_SIZE = 2
_EXPERTS = 16
_TOPK = 4
_DIM = 4096
_INTER_DIM = 2048
_MAX_TOKENS = 32


def _sync_ranks_for_mega(device: torch.device) -> None:
    """Finish NCCL work before DeepGEMM occupies every SM for grid sync."""

    dist.barrier()
    torch.cuda.synchronize(device)


def _assert_outputs_do_not_alias(expected: torch.Tensor, actual: torch.Tensor) -> None:
    """Catch result-cache self-comparisons, including zero-length tensors."""

    assert expected is not actual
    assert not expected.is_set_to(actual)
    if expected.numel() and actual.numel():
        assert expected.data_ptr() != actual.data_ptr()


def _ue8m0_ones(*shape: int, device: torch.device) -> torch.Tensor:
    return torch.full(shape, 127, dtype=torch.uint8, device=device).view(
        torch.float8_e8m0fnu
    )


def _make_global_weights(
    device: torch.device,
    *,
    include_shared: bool = False,
    n_shared_experts: int = 1,
) -> dict:
    def packed_fp4(out_dim: int, in_dim: int) -> torch.Tensor:
        shape = (_EXPERTS, out_dim, in_dim // 2)
        # Build each packed byte from finite E2M1 nibbles. Arbitrary int8
        # bytes can encode reserved/non-finite FP4 patterns and make a
        # numerical regression fail for reasons unrelated to the kernel.
        low = torch.randint(0, 7, shape, dtype=torch.uint8, device=device)
        high = torch.randint(0, 7, shape, dtype=torch.uint8, device=device)
        low |= torch.randint(0, 2, shape, dtype=torch.uint8, device=device) << 3
        high |= torch.randint(0, 2, shape, dtype=torch.uint8, device=device) << 3
        return (low | (high << 4)).view(torch.int8)

    def fp4_scale(out_dim: int, in_dim: int) -> torch.Tensor:
        return _ue8m0_ones(
            _EXPERTS,
            out_dim,
            in_dim // 32,
            device=device,
        )

    weights = {
        W.v4_router_w: torch.randn(_EXPERTS, _DIM, dtype=torch.bfloat16, device=device),
        W.v4_router_bias: torch.zeros(_EXPERTS, dtype=torch.float32, device=device),
        W.v4_routed_w1_w: packed_fp4(_INTER_DIM, _DIM),
        W.v4_routed_w1_s: fp4_scale(_INTER_DIM, _DIM),
        W.v4_routed_w2_w: packed_fp4(_DIM, _INTER_DIM),
        W.v4_routed_w2_s: fp4_scale(_DIM, _INTER_DIM),
        W.v4_routed_w3_w: packed_fp4(_INTER_DIM, _DIM),
        W.v4_routed_w3_s: fp4_scale(_INTER_DIM, _DIM),
    }
    if include_shared:
        shared_inter = _INTER_DIM * n_shared_experts
        weights.update(
            {
                W.v4_shared_w13_w: (
                    torch.randn(
                        2 * shared_inter,
                        _DIM,
                        dtype=torch.float32,
                        device=device,
                    )
                    * 0.05
                ).to(torch.float8_e4m3fn),
                W.v4_shared_w13_s: _ue8m0_ones(
                    2 * shared_inter // 128,
                    _DIM // 128,
                    device=device,
                ),
                W.v4_shared_w2_w: (
                    torch.randn(
                        _DIM,
                        shared_inter,
                        dtype=torch.float32,
                        device=device,
                    )
                    * 0.05
                ).to(torch.float8_e4m3fn),
                W.v4_shared_w2_s: _ue8m0_ones(
                    _DIM // 128,
                    shared_inter // 128,
                    device=device,
                ),
            }
        )
    return weights


def _make_moe(
    layer_weights: dict,
    *,
    ep_size: int,
    ep_rank: int,
    strategy,
    n_shared_experts: int = 0,
) -> MoE:
    return MoE(
        layer_id=7,
        dim=_DIM,
        moe_inter_dim=_INTER_DIM,
        n_routed_experts=_EXPERTS,
        n_activated_experts=_TOPK,
        n_shared_experts=n_shared_experts,
        score_func="sqrtsoftplus",
        route_scale=1.0,
        swiglu_limit=10.0,
        n_hash_layers=0,
        vocab_size=64,
        layer_weights=layer_weights,
        ep_size=ep_size,
        ep_rank=ep_rank,
        max_tokens_per_rank=_MAX_TOKENS,
        strategy=strategy,
    )


def _worker(
    rank: int,
    world_size: int,
    rendezvous_path: str,
    tokens_per_rank: tuple[int, int],
    run_graph: bool,
    include_shared: bool = False,
) -> None:
    os.environ["MODEL_WARM_UP"] = "0"
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        backend="nccl",
        init_method=f"file://{rendezvous_path}",
        rank=rank,
        world_size=world_size,
        device_id=device,
        timeout=timedelta(seconds=120),
    )
    try:
        torch.manual_seed(20260901)
        global_weights = _make_global_weights(device, include_shared=include_shared)
        local_start = rank * (_EXPERTS // world_size)
        local_end = local_start + (_EXPERTS // world_size)
        mega_weights = {
            W.v4_router_w: global_weights[W.v4_router_w],
            W.v4_router_bias: global_weights[W.v4_router_bias],
        }
        for key in (
            W.v4_routed_w1_w,
            W.v4_routed_w1_s,
            W.v4_routed_w2_w,
            W.v4_routed_w2_s,
            W.v4_routed_w3_w,
            W.v4_routed_w3_s,
        ):
            mega_weights[key] = global_weights[key][local_start:local_end].clone()
        if include_shared:
            for key in (
                W.v4_shared_w13_w,
                W.v4_shared_w13_s,
                W.v4_shared_w2_w,
                W.v4_shared_w2_s,
            ):
                mega_weights[key] = global_weights[key].clone()

        # Routed-only uses the production default. The shared-expert variant
        # explicitly selects ordinary MegaMoE to exercise the production
        # fallback used when MegaMoE-SE is unavailable.
        mega = _make_moe(
            mega_weights,
            ep_size=world_size,
            ep_rank=rank,
            strategy="mega_moe" if include_shared else None,
            n_shared_experts=int(include_shared),
        )
        reference = _make_moe(
            dict(global_weights),
            ep_size=1,
            ep_rank=0,
            strategy="local_loop",
            n_shared_experts=int(include_shared),
        )
        assert isinstance(mega.fused_moe.fused_experts, MegaMoeExecutor)
        assert isinstance(reference.fused_moe.fused_experts, LocalLoopExecutor)
        if include_shared:
            assert not mega.fused_moe.includes_shared_expert
            assert mega._moe.shared_experts is not None
            assert mega._moe._shared_executor is not None
        else:
            assert not mega.fused_moe.includes_shared_expert
            assert mega._moe.shared_experts is None
            assert mega._moe._shared_executor is None

        tokens = tokens_per_rank[rank]
        torch.manual_seed(20261000 + rank)
        x = torch.randn(tokens, _DIM, dtype=torch.bfloat16, device=device)
        input_ids = torch.arange(tokens, dtype=torch.long, device=device)
        with torch.inference_mode():
            expected = reference(x, input_ids).clone()
        torch.cuda.synchronize(device)
        _sync_ranks_for_mega(device)
        with torch.inference_mode():
            actual = mega(x, input_ids)
        torch.cuda.synchronize(device)
        dist.barrier()

        _assert_outputs_do_not_alias(expected, actual)
        assert actual.shape == x.shape
        assert torch.isfinite(actual).all()
        if tokens:
            error = (actual.float() - expected.float()).abs()
            baseline = expected.float().abs().mean().item() + 1e-6
            relative_error = error.mean().item() / baseline
            assert relative_error < 0.05, (
                f"rank={rank} MegaMoE relative error {relative_error:.6f} "
                "exceeds the local-loop baseline"
            )

        # Capture and replay the routed-only production path at a fixed
        # decode shape. This catches allocations or host-side decisions that
        # would make the n_shared_experts=0 branch graph-unsafe.
        if not run_graph:
            return
        torch.manual_seed(20261100 + rank)
        graph_x = torch.randn(2, _DIM, dtype=torch.bfloat16, device=device)
        graph_ids = torch.arange(2, dtype=torch.long, device=device)
        with torch.inference_mode():
            reference(graph_x, graph_ids)
            _sync_ranks_for_mega(device)
            mega(graph_x, graph_ids)
        torch.cuda.synchronize(device)
        _sync_ranks_for_mega(device)
        graph = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(graph):
            graph_actual = mega(graph_x, graph_ids)
        torch.manual_seed(20261200 + rank)
        graph_x.copy_(torch.randn_like(graph_x))
        graph_ids.copy_(torch.arange(1, -1, -1, dtype=torch.long, device=device))
        with torch.inference_mode():
            graph_expected = reference(graph_x, graph_ids).clone()
        _sync_ranks_for_mega(device)
        graph.replay()
        torch.cuda.synchronize(device)
        dist.barrier()
        _assert_outputs_do_not_alias(graph_expected, graph_actual)
        graph_error = (graph_actual.float() - graph_expected.float()).abs()
        graph_baseline = graph_expected.float().abs().mean().item() + 1e-6
        assert graph_error.mean().item() / graph_baseline < 0.05
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _se_worker(
    rank: int,
    world_size: int,
    rendezvous_path: str,
    tokens_per_rank: tuple[int, int],
    run_graph: bool,
) -> None:
    os.environ["MODEL_WARM_UP"] = "0"
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        backend="nccl",
        init_method=f"file://{rendezvous_path}",
        rank=rank,
        world_size=world_size,
        device_id=device,
        timeout=timedelta(seconds=120),
    )
    try:
        torch.manual_seed(20263001)
        n_shared_experts = 2
        global_weights = _make_global_weights(
            device, include_shared=True, n_shared_experts=n_shared_experts
        )
        local_start = rank * (_EXPERTS // world_size)
        local_end = local_start + (_EXPERTS // world_size)
        mega_se_weights = {
            W.v4_router_w: global_weights[W.v4_router_w],
            W.v4_router_bias: global_weights[W.v4_router_bias],
            W.v4_shared_w13_w: global_weights[W.v4_shared_w13_w].clone(),
            W.v4_shared_w13_s: global_weights[W.v4_shared_w13_s].clone(),
            W.v4_shared_w2_w: global_weights[W.v4_shared_w2_w].clone(),
            W.v4_shared_w2_s: global_weights[W.v4_shared_w2_s].clone(),
        }
        for key in (
            W.v4_routed_w1_w,
            W.v4_routed_w1_s,
            W.v4_routed_w2_w,
            W.v4_routed_w2_s,
            W.v4_routed_w3_w,
            W.v4_routed_w3_s,
        ):
            mega_se_weights[key] = global_weights[key][local_start:local_end].clone()

        # The production default must consume the shared weights in the
        # fused MegaMoESE strategy. The independent EP=1 reference executes
        # the same shared expert through the standalone path.
        mega_se = _make_moe(
            mega_se_weights,
            ep_size=world_size,
            ep_rank=rank,
            strategy=None,
            n_shared_experts=n_shared_experts,
        )
        reference = _make_moe(
            dict(global_weights),
            ep_size=1,
            ep_rank=0,
            strategy="local_loop",
            n_shared_experts=n_shared_experts,
        )
        assert isinstance(mega_se.fused_moe.fused_experts, MegaMoeSEExecutor)
        assert isinstance(reference.fused_moe.fused_experts, LocalLoopExecutor)
        assert mega_se.fused_moe.includes_shared_expert
        assert mega_se.fused_moe.includes_shared_expert
        assert mega_se._moe.shared_experts is None
        assert mega_se._moe._shared_executor is None
        assert reference.shared_experts is not None
        assert reference._moe._shared_executor is not None

        tokens = tokens_per_rank[rank]
        torch.manual_seed(20263100 + rank)
        x = torch.randn(tokens, _DIM, dtype=torch.bfloat16, device=device)
        input_ids = torch.arange(tokens, dtype=torch.long, device=device)
        with torch.inference_mode():
            expected = reference(x, input_ids).clone()
        _sync_ranks_for_mega(device)
        with torch.inference_mode():
            actual = mega_se(x, input_ids)
        torch.cuda.synchronize(device)
        dist.barrier()

        _assert_outputs_do_not_alias(expected, actual)
        assert actual.shape == x.shape
        assert torch.isfinite(actual).all()
        if tokens:
            error = (actual.float() - expected.float()).abs()
            baseline = expected.float().abs().mean().item() + 1e-6
            relative_error = error.mean().item() / baseline
            assert relative_error < 0.05, (
                f"rank={rank} MegaMoESE relative error {relative_error:.6f} "
                "exceeds the standalone baseline"
            )

        if not run_graph:
            return
        torch.manual_seed(20263200 + rank)
        graph_x = torch.randn(2, _DIM, dtype=torch.bfloat16, device=device)
        graph_ids = torch.arange(2, dtype=torch.long, device=device)
        with torch.inference_mode():
            reference(graph_x, graph_ids)
            _sync_ranks_for_mega(device)
            mega_se(graph_x, graph_ids)
        torch.cuda.synchronize(device)
        _sync_ranks_for_mega(device)
        graph = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(graph):
            graph_actual = mega_se(graph_x, graph_ids)
        torch.manual_seed(20263300 + rank)
        graph_x.copy_(torch.randn_like(graph_x))
        graph_ids.copy_(torch.arange(1, -1, -1, dtype=torch.long, device=device))
        with torch.inference_mode():
            graph_expected = reference(graph_x, graph_ids).clone()
        _sync_ranks_for_mega(device)
        graph.replay()
        torch.cuda.synchronize(device)
        dist.barrier()
        _assert_outputs_do_not_alias(graph_expected, graph_actual)
        graph_error = (graph_actual.float() - graph_expected.float()).abs()
        graph_baseline = graph_expected.float().abs().mean().item() + 1e-6
        assert graph_error.mean().item() / graph_baseline < 0.05
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


class RoutedOnlyMegaMoEExecutionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        if not torch.cuda.is_available():
            raise AssertionError("CUDA is required by this dedicated target")
        if torch.cuda.device_count() < _WORLD_SIZE:
            raise AssertionError(
                f"{_WORLD_SIZE} GPUs are required by this dedicated target"
            )

    def _spawn(self, worker, *args) -> None:
        test_tmpdir = os.environ.get("TEST_TMPDIR")
        with tempfile.TemporaryDirectory(dir=test_tmpdir) as tmpdir:
            rendezvous_path = os.path.join(tmpdir, "torch_dist_rendezvous")
            mp.spawn(
                worker,
                args=(_WORLD_SIZE, rendezvous_path, *args),
                nprocs=_WORLD_SIZE,
                join=True,
            )

    def test_two_rank_default_handles_empty_rank(self):
        self._spawn(_worker, (8, 0), False)

    def test_two_rank_chunking_keeps_collectives_symmetric(self):
        self._spawn(_worker, (65, 1), False)

    def test_two_rank_default_matches_local_reference_and_cuda_graph(self):
        self._spawn(_worker, (3, 2), True)

    def test_two_rank_mega_moe_standalone_shared_expert_fallback(self):
        self._spawn(_worker, (3, 2), True, True)

    def test_two_rank_multi_se_handles_empty_rank(self):
        self._spawn(_se_worker, (8, 0), False)

    def test_two_rank_multi_se_chunking_keeps_collectives_symmetric(self):
        self._spawn(_se_worker, (65, 1), False)

    def test_two_rank_multi_se_matches_standalone_and_cuda_graph(self):
        self._spawn(_se_worker, (3, 2), True)


if __name__ == "__main__":
    unittest.main()
