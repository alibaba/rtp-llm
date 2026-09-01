"""Two-rank SM120 collective MoE eager/CUDA-graph regression."""

from __future__ import annotations

import os
import socket
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from rtp_llm.models_py.modules.dsv4.moe.strategies.base import MoeCfg
from rtp_llm.models_py.modules.dsv4.moe.strategies.sm120_fused_moe import (
    Sm120FusedMoeStrategy,
)
from rtp_llm.models_py.utils.arch import is_sm120
from rtp_llm.utils.model_weight import W


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _unpack_e2m1(packed: torch.Tensor) -> torch.Tensor:
    """Decode raw MXFP4 bytes without using a FlashInfer MoE/GEMM path."""
    codes = packed.view(torch.uint8)
    low = torch.bitwise_and(codes, 0x0F)
    high = torch.bitwise_right_shift(codes, 4)
    codes = torch.stack((low, high), dim=-1).reshape(*packed.shape[:-1], -1)
    values = torch.tensor(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            -0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ],
        dtype=torch.float32,
        device=packed.device,
    )
    return values.index_select(0, codes.long().reshape(-1)).view(codes.shape)


def _dense_fp4_reference(
    x: torch.Tensor,
    router_weights: torch.Tensor,
    indices: torch.Tensor,
    *,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    swiglu_limit: float,
) -> torch.Tensor:
    """Independent dense oracle for non-uniform raw MXFP4 weights."""
    x_cpu = x.float().cpu()
    weights_cpu = router_weights.float().cpu()
    indices_cpu = indices.long().cpu()
    w1_cpu = w1.float().cpu()
    w2_cpu = w2.float().cpu()
    w3_cpu = w3.float().cpu()
    output = torch.zeros(x.size(0), x.size(1), dtype=torch.float32)
    for token in range(x.size(0)):
        routed_value = torch.zeros(x.size(1), dtype=torch.float32)
        for route in range(indices.size(1)):
            expert = int(indices_cpu[token, route])
            if expert < 0:
                continue
            gate = F.linear(x_cpu[token], w1_cpu[expert])
            up = F.linear(x_cpu[token], w3_cpu[expert])
            if swiglu_limit > 0:
                gate = torch.clamp(gate, max=swiglu_limit)
                up = torch.clamp(up, min=-swiglu_limit, max=swiglu_limit)
            hidden = (F.silu(gate) * up).to(torch.bfloat16).float()
            dense_down = F.linear(hidden, w2_cpu[expert])
            routed_value += weights_cpu[token, route] * dense_down
        output[token].copy_(routed_value)
    return output.to(x.device)


def _run_collective_rank(rank: int, world_size: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    try:
        device = torch.device("cuda", rank)
        if not is_sm120(device):
            raise RuntimeError(f"rank {rank} is not running on SM120")
        cfg = MoeCfg(
            layer_id=0,
            dim=128,
            moe_inter_dim=128,
            n_routed_experts=4,
            n_activated_experts=2,
            swiglu_limit=7.0,
            ep_size=world_size,
            ep_rank=rank,
            n_local_experts=2,
            local_expert_start=rank * 2,
            local_expert_end=(rank + 1) * 2,
            max_tokens_per_rank=4,
            moe_tp_size=world_size,
            cp_size=world_size,
            cp_enabled=True,
        )
        strategy = Sm120FusedMoeStrategy(cfg)
        if strategy._sm120_grouped is None:
            raise RuntimeError(
                "SM120 hardware gate requires the production FlashInfer GroupedFP4 executor"
            )

        def packed_weights(out_dim: int, in_dim: int, salt: int) -> torch.Tensor:
            # Non-uniform experts, rows and columns make any packed-weight
            # permutation visible.  Keep codes positive so the independent
            # E2M1 decoder above remains especially easy to audit.
            expert = torch.arange(4, device=device).view(4, 1, 1)
            row = torch.arange(out_dim, device=device).view(1, out_dim, 1)
            column = torch.arange(in_dim // 2, device=device).view(1, 1, -1)
            low = (expert + row + 3 * column + salt).remainder(7) + 1
            high = (2 * expert + 3 * row + column + 2 * salt).remainder(7) + 1
            return (
                torch.bitwise_or(low, torch.bitwise_left_shift(high, 4))
                .to(torch.uint8)
                .view(torch.int8)
            )

        def scales(out_dim: int, in_dim: int) -> torch.Tensor:
            return torch.ones(
                (cfg.n_local_experts, out_dim, in_dim // 32),
                dtype=torch.float8_e8m0fnu,
                device=device,
            )

        full_w1 = packed_weights(cfg.moe_inter_dim, cfg.dim, salt=1)
        full_w2 = packed_weights(cfg.dim, cfg.moe_inter_dim, salt=3)
        full_w3 = packed_weights(cfg.moe_inter_dim, cfg.dim, salt=5)
        dense_w1 = _unpack_e2m1(full_w1)
        dense_w2 = _unpack_e2m1(full_w2)
        dense_w3 = _unpack_e2m1(full_w3)
        local_slice = slice(cfg.local_expert_start, cfg.local_expert_end)
        strategy.setup_weights(
            {
                W.v4_routed_w1_w: full_w1[local_slice].contiguous(),
                W.v4_routed_w1_s: scales(cfg.moe_inter_dim, cfg.dim),
                W.v4_routed_w2_w: full_w2[local_slice].contiguous(),
                W.v4_routed_w2_s: scales(cfg.dim, cfg.moe_inter_dim),
                W.v4_routed_w3_w: full_w3[local_slice].contiguous(),
                W.v4_routed_w3_s: scales(cfg.moe_inter_dim, cfg.dim),
            }
        )

        def make_x(tokens: int, phase: int) -> torch.Tensor:
            rows = torch.arange(tokens, device=device).view(-1, 1)
            columns = torch.arange(cfg.dim, device=device).view(1, -1)
            return (
                ((3 * columns + 5 * rows + 7 * rank + phase).remainder(23) - 11)
                .to(torch.float32)
                .div_(256.0)
                .to(torch.bfloat16)
            )

        def assert_collective_matches_dense(
            x: torch.Tensor,
            weights: torch.Tensor,
            indices: torch.Tensor,
            *,
            label: str,
        ) -> torch.Tensor:
            actual = strategy._forward_collective(x, weights, indices)
            reference = _dense_fp4_reference(
                x,
                weights,
                indices,
                w1=dense_w1,
                w2=dense_w2,
                w3=dense_w3,
                swiglu_limit=cfg.swiglu_limit,
            )
            torch.testing.assert_close(
                actual,
                reference,
                rtol=8e-2,
                atol=5e-1,
                msg=lambda message: f"{label}: {message}",
            )
            return actual

        # Dynamic eager collectives must agree when ranks contribute different
        # token counts.  This exercises the count exchange and local slicing.
        uneven_tokens = 3 if rank == 0 else 1
        uneven_x = make_x(uneven_tokens, phase=1)
        uneven_indices = torch.tensor(
            [
                [(rank + token) % 4, (rank + token + 2) % 4]
                for token in range(uneven_tokens)
            ],
            dtype=torch.int64,
            device=device,
        )
        uneven_weights = torch.tensor(
            [
                [0.25 + 0.1 * token, 0.75 - 0.1 * token]
                for token in range(uneven_tokens)
            ],
            dtype=torch.float32,
            device=device,
        )
        assert_collective_matches_dense(
            uneven_x, uneven_weights, uneven_indices, label="uneven tokens"
        )
        dist.barrier()

        # A rank with no local tokens must still participate in all
        # collectives and return a correctly shaped empty result.
        zero_rank_tokens = 2 if rank == 0 else 0
        zero_rank_x = torch.full(
            (zero_rank_tokens, cfg.dim),
            1.0 / 64.0,
            dtype=torch.bfloat16,
            device=device,
        )
        zero_rank_weights = torch.full(
            (zero_rank_tokens, 2), 0.5, dtype=torch.float32, device=device
        )
        zero_rank_indices = torch.tensor(
            [[0, 3]] * zero_rank_tokens, dtype=torch.int64, device=device
        ).reshape(zero_rank_tokens, 2)
        zero_rank_output = assert_collective_matches_dense(
            zero_rank_x,
            zero_rank_weights,
            zero_rank_indices,
            label="zero-token rank",
        )
        if tuple(zero_rank_output.shape) != (zero_rank_tokens, cfg.dim):
            raise AssertionError(
                f"zero-token rank returned shape {tuple(zero_rank_output.shape)}"
            )
        dist.barrier()

        x = make_x(2, phase=2)
        weights = torch.full((2, 2), 0.5, dtype=torch.float32, device=device)
        indices = torch.tensor([[0, 2], [1, 3]], dtype=torch.int64, device=device)

        eager = assert_collective_matches_dense(x, weights, indices, label="eager")
        if not torch.isfinite(eager).all() or not eager.abs().max().item() > 0:
            raise AssertionError("real SM120 grouped expert output is invalid")
        dist.barrier()

        # Exercise the public Pure-CP path once before capture.  Capture uses
        # the public entry point too, which deliberately chooses its graph-safe
        # collective implementation while the stream is capturing.
        public_cp = strategy(x, weights, indices)
        public_cp_reference = _dense_fp4_reference(
            x,
            weights,
            indices,
            w1=dense_w1,
            w2=dense_w2,
            w3=dense_w3,
            swiglu_limit=cfg.swiglu_limit,
        )
        torch.testing.assert_close(public_cp, public_cp_reference, rtol=8e-2, atol=5e-1)
        dist.barrier()

        # Materialize the exact graph backend and its fixed-address workspace
        # before capture.  The public entry point still exercises Pure-CP
        # routing here, while the warmup flag makes the local grouped executor
        # select the same graph-safe kernel used during capture.
        with patch.dict(os.environ, {"RTP_LLM_CUDA_GRAPH_WARMUP_FORWARD": "1"}):
            graph_warmup = strategy(x, weights, indices)
        torch.testing.assert_close(
            graph_warmup, public_cp_reference, rtol=8e-2, atol=5e-1
        )
        torch.cuda.synchronize(device)
        dist.barrier()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_output = strategy(x, weights, indices)
        dist.barrier()
        x.copy_(make_x(2, phase=13))
        # Change the per-expert histogram: experts 1 and 2 now receive zero
        # tokens, while every rank must still map global ids to its local slice.
        indices.copy_(torch.tensor([[3, 0], [0, 3]], device=device))
        weights.copy_(torch.tensor([[0.75, 0.25], [0.25, 0.75]], device=device))
        graph.replay()
        torch.cuda.synchronize(device)
        replay = graph_output.clone()
        replay_reference = _dense_fp4_reference(
            x,
            weights,
            indices,
            w1=dense_w1,
            w2=dense_w2,
            w3=dense_w3,
            swiglu_limit=cfg.swiglu_limit,
        )
        torch.testing.assert_close(
            replay,
            replay_reference,
            rtol=8e-2,
            atol=5e-1,
        )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


class Sm120CollectiveHardwareTest(unittest.TestCase):
    def test_two_rank_collective_eager_and_cuda_graph(self) -> None:
        if torch.cuda.device_count() < 2:
            self.skipTest("requires two SM120 GPUs")
        mp.spawn(
            _run_collective_rank,
            args=(2, _free_port()),
            nprocs=2,
            join=True,
        )


if __name__ == "__main__":
    unittest.main()
