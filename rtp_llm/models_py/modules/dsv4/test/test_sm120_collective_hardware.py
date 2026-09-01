"""Two-rank SM120 collective MoE eager/CUDA-graph regression."""

from __future__ import annotations

import os
import socket
import unittest

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


def _dense_constant_fp4_reference(
    x: torch.Tensor,
    weights: torch.Tensor,
    indices: torch.Tensor,
    *,
    inter_dim: int,
    swiglu_limit: float,
) -> torch.Tensor:
    """Independent dense oracle for the constant packed weights below.

    Packed byte ``0x11 * (expert + 1)`` repeats one positive E2M1 value
    across the full matrix: expert 0..3 therefore has dense weight
    0.5, 1.0, 1.5, or 2.0.  Computing the two linear layers and routing on
    CPU avoids using the distributed strategy, ep_gather, or FlashInfer as
    the oracle.  MXFP8 activation quantization is allowed its normal small
    error budget by the caller.
    """
    x_cpu = x.float().cpu()
    weights_cpu = weights.float().cpu()
    indices_cpu = indices.long().cpu()
    output = torch.zeros(x.size(0), x.size(1), dtype=torch.float32)
    for token in range(x.size(0)):
        source_sum = x_cpu[token].sum()
        routed_value = torch.tensor(0.0)
        for route in range(indices.size(1)):
            expert = int(indices_cpu[token, route])
            if expert < 0:
                continue
            expert_weight = 0.5 * (expert + 1)
            gate = source_sum * expert_weight
            up = source_sum * expert_weight
            if swiglu_limit > 0:
                gate = torch.clamp(gate, max=swiglu_limit)
                up = torch.clamp(up, min=-swiglu_limit, max=swiglu_limit)
            hidden = (F.silu(gate) * up).to(torch.bfloat16).float()
            dense_down = hidden * inter_dim * expert_weight
            routed_value += weights_cpu[token, route] * dense_down
        output[token].fill_(routed_value)
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
        )
        strategy = Sm120FusedMoeStrategy(cfg)
        if strategy._sm120_grouped is None:
            raise RuntimeError(
                "SM120 hardware gate requires the production FlashInfer GroupedFP4 executor"
            )

        def packed_weights(out_dim: int, in_dim: int) -> torch.Tensor:
            result = torch.empty(
                (cfg.n_local_experts, out_dim, in_dim // 2),
                dtype=torch.int8,
                device=device,
            )
            for local_id in range(cfg.n_local_experts):
                # Both nibbles encode a positive E2M1 value. Distinct global
                # expert constants make a broken global->local mapping visible.
                global_id = cfg.local_expert_start + local_id
                result[local_id].fill_(0x11 * (global_id + 1))
            return result

        def scales(out_dim: int, in_dim: int) -> torch.Tensor:
            return torch.ones(
                (cfg.n_local_experts, out_dim, in_dim // 32),
                dtype=torch.float8_e8m0fnu,
                device=device,
            )

        strategy.setup_weights(
            {
                W.v4_routed_w1_w: packed_weights(cfg.moe_inter_dim, cfg.dim),
                W.v4_routed_w1_s: scales(cfg.moe_inter_dim, cfg.dim),
                W.v4_routed_w2_w: packed_weights(cfg.dim, cfg.moe_inter_dim),
                W.v4_routed_w2_s: scales(cfg.dim, cfg.moe_inter_dim),
                W.v4_routed_w3_w: packed_weights(cfg.moe_inter_dim, cfg.dim),
                W.v4_routed_w3_s: scales(cfg.moe_inter_dim, cfg.dim),
            }
        )

        def assert_collective_matches_dense(
            x: torch.Tensor,
            weights: torch.Tensor,
            indices: torch.Tensor,
            *,
            label: str,
        ) -> torch.Tensor:
            actual = strategy._forward_collective(x, weights, indices)
            reference = _dense_constant_fp4_reference(
                x,
                weights,
                indices,
                inter_dim=cfg.moe_inter_dim,
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
        uneven_x = torch.stack(
            [
                torch.full(
                    (cfg.dim,),
                    (rank + token + 1) / 128.0,
                    dtype=torch.bfloat16,
                    device=device,
                )
                for token in range(uneven_tokens)
            ]
        )
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

        x = torch.full(
            (2, cfg.dim),
            (rank + 1.0) / 128.0,
            dtype=torch.bfloat16,
            device=device,
        )
        weights = torch.full((2, 2), 0.5, dtype=torch.float32, device=device)
        indices = torch.tensor([[0, 2], [1, 3]], dtype=torch.int64, device=device)

        eager = assert_collective_matches_dense(x, weights, indices, label="eager")
        if not torch.isfinite(eager).all() or not eager.abs().max().item() > 0:
            raise AssertionError("real SM120 grouped expert output is invalid")
        dist.barrier()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_output = strategy._forward_collective(x, weights, indices)
        dist.barrier()
        x.fill_((rank + 2.0) / 128.0)
        # Change the per-expert histogram: experts 1 and 2 now receive zero
        # tokens, while every rank must still map global ids to its local slice.
        indices.copy_(torch.tensor([[3, 0], [0, 3]], device=device))
        weights.copy_(torch.tensor([[0.75, 0.25], [0.25, 0.75]], device=device))
        graph.replay()
        torch.cuda.synchronize(device)
        replay = graph_output.clone()
        replay_reference = _dense_constant_fp4_reference(
            x,
            weights,
            indices,
            inter_dim=cfg.moe_inter_dim,
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
