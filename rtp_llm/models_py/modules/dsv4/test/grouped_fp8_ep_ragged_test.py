"""Real four-rank NCCL regression for non-decode ragged EP exchanges."""

import dataclasses
import datetime
import os
import tempfile
import unittest
from contextlib import nullcontext
from unittest import mock

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from rtp_llm.models_py.modules.dsv4.moe.strategies import grouped_fp8
from rtp_llm.models_py.modules.dsv4.moe.strategies.base import MoeCfg
from rtp_llm.models_py.modules.factory.fused_moe.factory import FusedMoeFactory
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.strategy.fp8_fp4 import (
    CudaGroupedFp8Strategy,
)
from rtp_llm.models_py.modules.factory.fused_moe.strategy_registry import (
    StrategyRegistry,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.fp8_fp4.chunked_layer import (
    ChunkedFp8Fp4MoeLayer,
    resolve_moe_max_tokens_per_rank,
    synchronized_moe_chunk_plan,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.fp8_fp4.layer import (
    Fp8Fp4MoeRuntimeConfig,
)
from rtp_llm.utils.model_weight import W

_DIM = 1024
_INTER = 128
_EXPERTS = 8
_RANKS = 4


def _weights(device, dim, inter, experts, seed=1504):
    generator = torch.Generator().manual_seed(seed)
    result = {}
    for weight_name, scale_name, n, k in (
        (W.v4_routed_w1_w, W.v4_routed_w1_s, inter, dim),
        (W.v4_routed_w3_w, W.v4_routed_w3_s, inter, dim),
        (W.v4_routed_w2_w, W.v4_routed_w2_s, dim, inter),
    ):
        result[weight_name] = (
            (torch.randn(experts, n, k, generator=generator) * 0.1)
            .to(device)
            .to(torch.float8_e4m3fn)
        )
        result[scale_name] = torch.full(
            (experts, n // 128, k // 128), 0.125, device=device
        )
    return result


def _inputs(n, device, dim, experts=_EXPERTS, topk=2):
    generator = torch.Generator().manual_seed(100 + n)
    x = torch.randn(n, dim, generator=generator).to(device).to(torch.bfloat16)
    ids = (
        [0, 3]
        if topk == 2
        else [0, experts // 4, experts // 2, 3 * experts // 4, 1, experts // 4 + 1]
    )
    indices = (
        torch.tensor([ids], dtype=torch.int64, device=device).expand(n, -1).contiguous()
    )
    weights = torch.full((n, topk), 1.0 / topk, device=device)
    return x, weights, indices


class _CountingRoutedLayer(nn.Module):
    """Supply deterministic gates while retaining the production MoE chain."""

    def __init__(self, moe, dim):
        super().__init__()
        self.moe = moe
        self.dim = dim
        self.calls = 0

    def forward(self, x, input_ids, *, out, observer=None):
        self.calls += 1
        _, weights, indices = _inputs(x.shape[0], x.device, self.dim)
        out.copy_(self.moe(x, weights, indices))
        return out


def _worker(rank, init_method, low_latency=False, speculative=False):
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    # LL uses the production Flash dimensions; its transport and masked GEMMs
    # do not support all of the tiny synthetic NCCL fixture's shapes.
    dim, inter = (4096, 2048) if low_latency else (_DIM, _INTER)
    experts, topk = (256, 6) if low_latency else (_EXPERTS, 2)
    decode = low_latency or speculative
    budget = (
        resolve_moe_max_tokens_per_rank(
            8192,
            8192,
            1,
            4,
            is_decode_role=True,
            is_speculative=True,
            gen_num_per_cycle=2,
        )
        if speculative
        else 8 if low_latency else 8192
    )
    dist.init_process_group(
        "nccl",
        init_method=init_method,
        rank=rank,
        world_size=_RANKS,
        timeout=datetime.timedelta(seconds=120),
    )
    try:
        cfg = MoeCfg(
            layer_id=0,
            dim=dim,
            moe_inter_dim=inter,
            n_routed_experts=experts,
            n_activated_experts=topk,
            swiglu_limit=10.0,
            ep_size=1,
            ep_rank=0,
            n_local_experts=experts,
            local_expert_start=0,
            local_expert_end=experts,
            max_tokens_per_rank=budget,
            is_decode_role=decode,
            enable_cuda_graph=low_latency,
            is_speculative=speculative,
        )
        local_experts = experts // _RANKS
        ep_cfg = dataclasses.replace(
            cfg,
            ep_size=_RANKS,
            ep_rank=rank,
            n_local_experts=local_experts,
            local_expert_start=rank * local_experts,
            local_expert_end=(rank + 1) * local_experts,
        )
        if low_latency:
            # Only EP-local weights are needed for graph/eager equivalence.
            local = _weights(device, dim, inter, local_experts, seed=1504 + rank)
        else:
            full_weights = _weights(device, dim, inter, experts)
            reference = grouped_fp8.GroupedFP8Strategy(cfg)
            reference.setup_weights(dict(full_weights))
            local = {
                key: value[rank * local_experts : (rank + 1) * local_experts].clone()
                for key, value in full_weights.items()
            }
        runtime_cfg = Fp8Fp4MoeRuntimeConfig(
            layer_id=0,
            hidden_size=dim,
            moe_inter_dim=inter,
            expert_num=experts,
            moe_k=topk,
            n_shared_experts=0,
            swiglu_limit=10.0,
            ep_size=_RANKS,
            ep_rank=rank,
            max_tokens_per_rank=cfg.max_tokens_per_rank,
            moe_strategy="grouped_fp8",
            expert_dtype="fp8",
            moe_w1_layout="gate_up",
            is_decode_role=decode,
            enable_cuda_graph=low_latency,
            is_speculative=speculative,
        )
        weights = {
            W.moe_w1: torch.cat(
                (
                    local[W.v4_routed_w1_w].view(torch.uint8),
                    local[W.v4_routed_w3_w].view(torch.uint8),
                ),
                dim=1,
            ).view(torch.float8_e4m3fn),
            W.moe_s1: torch.cat(
                (local[W.v4_routed_w1_s], local[W.v4_routed_w3_s]), dim=1
            ),
            W.moe_w2: local[W.v4_routed_w2_w],
            W.moe_s2: local[W.v4_routed_w2_s],
        }
        registry = StrategyRegistry()
        registry.register(CudaGroupedFp8Strategy())
        FusedMoeFactory.set_registry(registry)
        with mock.patch.object(grouped_fp8, "_ep_group", return_value=dist.group.WORLD):
            if speculative and not low_latency:
                # Reject the unsupported combination during weight setup, before
                # either draft wrapper can start a CUDA graph capture. Changing
                # a capture bucket cannot fix the shared-wrapper constraint.
                runtime_cfg.enable_cuda_graph = True
                try:
                    FusedMoeFactory().create_fused_moe(runtime_cfg, dict(weights))
                except ValueError as error:
                    assert "DECODE_CAPTURE_CONFIG cannot" in str(error), error
                    assert "MAX_TOKENS to at least 12" in str(error), error
                else:
                    raise AssertionError("NCCL MTP graphs must fail before capture")
                finally:
                    runtime_cfg.enable_cuda_graph = False
                dist.barrier()
            # Real factory -> FusedMoe -> Router -> Executor -> grouped backend.
            # In particular, empty ranks must survive FusedMoe's fast path.
            moe = FusedMoeFactory().create_fused_moe(runtime_cfg, weights)
            if low_latency:
                assert moe.fused_experts.grouped._ll_ok
                graphs = {}
                sizes = (1, 3, 4, 6, 9, 12) if speculative else (1, 2, 4, 8)
                for n in sizes:
                    inputs = _inputs(n, device, dim, experts, topk)
                    # Compile all specializations before entering capture.
                    expected = moe(*inputs).clone()
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        output = moe(*inputs)
                    graphs[n] = (graph, inputs, output, expected)
                    dist.barrier()
                # Ranks deliberately replay different graph batch sizes. LL
                # carries their token counts in its fixed-capacity protocol.
                cases = (
                    ((1, 3, 6, 12), (4, 6, 9, 3), (12, 9, 4, 1), (3, 3, 3, 3))
                    if speculative
                    else ((1, 2, 4, 8), (8, 4, 2, 1), (1, 1, 1, 1))
                )
                for counts in cases:
                    graph, inputs, output, expected = graphs[counts[rank]]
                    graph.replay()
                    torch.cuda.synchronize()
                    torch.testing.assert_close(output, expected, rtol=0.03, atol=3e-4)
                    dist.barrier()
                return
            if speculative:
                # The actionable eager alternative must serve all draft/verify
                # shapes and empty ranks under the common verify-sized budget.
                for counts in ((0, 1, 4, 12), (12, 0, 3, 1), (4, 3, 2, 1)):
                    inputs = _inputs(counts[rank], device, dim)
                    expected = reference(*inputs)
                    actual = moe(*inputs)
                    torch.cuda.synchronize()
                    torch.testing.assert_close(actual, expected, rtol=0.03, atol=3e-4)
                    dist.barrier()
                return
            for counts in ((0, 1, 2, 0), (0, 0, 0, 0), (257, 1, 2, 0)):
                inputs = _inputs(counts[rank], device, dim)
                expected = reference(*inputs)
                actual = moe(*inputs)
                torch.cuda.synchronize()
                torch.testing.assert_close(actual, expected, rtol=0.02, atol=2e-4)
                dist.barrier()

            # Only one rank needs three chunks. Empty peers still participate
            # three times, as do peers whose own input fits a single chunk.
            moe.fused_experts.grouped.cfg = dataclasses.replace(
                ep_cfg, max_tokens_per_rank=257
            )
            layer = ChunkedFp8Fp4MoeLayer.__new__(ChunkedFp8Fp4MoeLayer)
            nn.Module.__init__(layer)
            layer.layer_id = 0
            layer.dim = dim
            layer.max_tokens_per_rank = 257
            layer._is_decode_role = False
            layer.chunking_enabled = True
            layer.strategy_name = "grouped_fp8"
            layer._observer_factory = None
            layer._record_function_scope = nullcontext
            layer._moe = _CountingRoutedLayer(moe, dim)
            inputs = _inputs((0, 1, 600, 0)[rank], device, dim)
            expected = reference(*inputs).to(torch.bfloat16)
            with synchronized_moe_chunk_plan([layer], inputs[0].shape[0], device):
                actual = layer(inputs[0])
            assert layer._moe.calls == 3, layer._moe.calls
            torch.testing.assert_close(actual, expected, rtol=0.03, atol=3e-4)
            dist.barrier()
    finally:
        dist.destroy_process_group()


class GroupedFP8RaggedNCCLTest(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.device_count() >= _RANKS, "needs four CUDA GPUs")
    def test_non_decode_ragged_empty_and_chunked_ranks(self):
        self._run(low_latency=False)

    @unittest.skipUnless(torch.cuda.device_count() >= _RANKS, "needs four CUDA GPUs")
    def test_low_latency_replays_different_rank_local_graph_sizes(self):
        self._run(low_latency=True)

    @unittest.skipUnless(torch.cuda.device_count() >= _RANKS, "needs four CUDA GPUs")
    def test_mtp_nccl_graphs_fail_early_and_eager_handles_ragged_draft_shapes(self):
        self._run(low_latency=False, speculative=True)

    @unittest.skipUnless(torch.cuda.device_count() >= _RANKS, "needs four CUDA GPUs")
    def test_mtp_low_latency_shared_strategy_captures_draft_and_verify_shapes(self):
        self._run(low_latency=True, speculative=True)

    def _run(self, low_latency, speculative=False):
        with tempfile.TemporaryDirectory() as root, mock.patch.dict(
            os.environ,
            {
                "DSV4_EP_CHECK_SIZES": "1",
                "DSV4_MOE_DEEPEP_LL": "1" if low_latency else "0",
                "DSV4_MOE_LL_MAX_TOKENS": "12" if speculative else "8",
                "DSV4_MOE_MASKED": "0",
                "NCCL_P2P_LEVEL": "NVL",
            },
        ):
            if low_latency:
                # Exercise the backend's NVLink default, without relying on
                # the benchmark launcher's ACCL environment setting.
                os.environ.pop("ACCL_LOW_LATENCY_OPTIMIZE", None)
            mp.spawn(
                _worker,
                args=(
                    "file://" + os.path.join(root, "nccl-init"),
                    low_latency,
                    speculative,
                ),
                nprocs=_RANKS,
            )


if __name__ == "__main__":
    unittest.main()
