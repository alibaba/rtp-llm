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
from rtp_llm.models_py.modules.factory.fused_moe.utils.fp8_fp4.chunked_layer import (
    ChunkedFp8Fp4MoeLayer,
    synchronized_moe_chunk_plan,
)
from rtp_llm.utils.model_weight import W

_DIM = 1024
_INTER = 128
_EXPERTS = 8
_RANKS = 4


def _weights(device):
    generator = torch.Generator().manual_seed(1504)
    result = {}
    for weight_name, scale_name, n, k in (
        (W.v4_routed_w1_w, W.v4_routed_w1_s, _INTER, _DIM),
        (W.v4_routed_w3_w, W.v4_routed_w3_s, _INTER, _DIM),
        (W.v4_routed_w2_w, W.v4_routed_w2_s, _DIM, _INTER),
    ):
        result[weight_name] = (
            (torch.randn(_EXPERTS, n, k, generator=generator) * 0.1)
            .to(device)
            .to(torch.float8_e4m3fn)
        )
        result[scale_name] = torch.full(
            (_EXPERTS, n // 128, k // 128), 0.125, device=device
        )
    return result


def _inputs(n, device):
    generator = torch.Generator().manual_seed(100 + n)
    x = torch.randn(n, _DIM, generator=generator).to(device).to(torch.bfloat16)
    indices = (
        torch.tensor([[0, 3]], dtype=torch.int64, device=device)
        .expand(n, -1)
        .contiguous()
    )
    weights = torch.full((n, 2), 0.5, device=device)
    return x, weights, indices


class _RoutedLayer(nn.Module):
    def __init__(self, strategy):
        super().__init__()
        self.strategy = strategy
        self.calls = 0

    def forward(self, x, input_ids, *, out, observer=None):
        self.calls += 1
        _, weights, indices = _inputs(x.shape[0], x.device)
        out.copy_(self.strategy(x, weights, indices))
        return out


def _worker(rank, init_method):
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
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
            dim=_DIM,
            moe_inter_dim=_INTER,
            n_routed_experts=_EXPERTS,
            n_activated_experts=2,
            swiglu_limit=10.0,
            ep_size=1,
            ep_rank=0,
            n_local_experts=_EXPERTS,
            local_expert_start=0,
            local_expert_end=_EXPERTS,
            max_tokens_per_rank=8192,
            is_decode_role=False,
        )
        full_weights = _weights(device)
        reference = grouped_fp8.GroupedFP8Strategy(cfg)
        reference.setup_weights(dict(full_weights))
        local_experts = _EXPERTS // _RANKS
        ep_cfg = dataclasses.replace(
            cfg,
            ep_size=_RANKS,
            ep_rank=rank,
            n_local_experts=local_experts,
            local_expert_start=rank * local_experts,
            local_expert_end=(rank + 1) * local_experts,
        )
        strategy = grouped_fp8.GroupedFP8Strategy(ep_cfg)
        strategy.setup_weights(
            {
                key: value[rank * local_experts : (rank + 1) * local_experts].clone()
                for key, value in full_weights.items()
            }
        )
        with mock.patch.object(grouped_fp8, "_ep_group", return_value=dist.group.WORLD):
            for counts in ((0, 1, 2, 0), (0, 0, 0, 0), (257, 1, 2, 0)):
                inputs = _inputs(counts[rank], device)
                expected = reference(*inputs)
                actual = strategy(*inputs)
                torch.cuda.synchronize()
                torch.testing.assert_close(actual, expected, rtol=0.02, atol=2e-4)
                dist.barrier()

            # Only one rank needs three chunks. Empty peers still participate
            # three times, as do peers whose own input fits a single chunk.
            strategy.cfg = dataclasses.replace(ep_cfg, max_tokens_per_rank=257)
            layer = ChunkedFp8Fp4MoeLayer.__new__(ChunkedFp8Fp4MoeLayer)
            nn.Module.__init__(layer)
            layer.layer_id = 0
            layer.dim = _DIM
            layer.max_tokens_per_rank = 257
            layer._is_decode_role = False
            layer.chunking_enabled = True
            layer.strategy_name = "grouped_fp8"
            layer._observer_factory = None
            layer._record_function_scope = nullcontext
            layer._moe = _RoutedLayer(strategy)
            inputs = _inputs((0, 1, 600, 0)[rank], device)
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
        with tempfile.TemporaryDirectory() as root, mock.patch.dict(
            os.environ,
            {
                "DSV4_EP_CHECK_SIZES": "1",
                "DSV4_MOE_DEEPEP_LL": "0",
                "DSV4_MOE_MASKED": "0",
                "NCCL_P2P_LEVEL": "NVL",
            },
        ):
            mp.spawn(
                _worker,
                args=("file://" + os.path.join(root, "nccl-init"),),
                nprocs=_RANKS,
            )


if __name__ == "__main__":
    unittest.main()
