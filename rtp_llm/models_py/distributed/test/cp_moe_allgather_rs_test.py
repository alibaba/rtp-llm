"""Tests for the allgather + reduce_scatter communication pattern used by
PureCpRouter / PureDpRouter (router-level path B).

The router internally does:
    prepare:  allgather(scattered_tokens) -> full_tokens
    execute:  partial MoE compute (each rank holds 1/N of experts)
    finalize: reduce_scatter(partial_output) -> scattered_output

This file verifies the underlying NCCL roundtrip is correct using a
multi-process simulation.
"""

import logging
import multiprocessing as mp
import os
import unittest

logging.basicConfig(level=logging.INFO)

import torch

from rtp_llm.models_py.distributed.collective_torch import (
    Group,
    all_gather,
    destroy_distributed_environment,
    init_distributed_environment,
    reduce_scatter,
)
from rtp_llm.ops import NcclCommConfig, ParallelismConfig
from rtp_llm.test.utils.port_util import PortManager


def _test_cp_moe_roundtrip_worker(
    rank: int, world_size: int, tp_size: int, dp_size: int, nccl_port: int
):
    """Worker: verify allgather -> partial_compute -> reduce_scatter recovers data.

    Simulates the router-level CP/DP MoE pattern:
    1. Each rank starts with scattered tokens (chunk_size tokens)
    2. allgather: each rank gets full tokens (world_size * chunk_size)
    3. Each rank computes partial result (1/world_size of final answer)
    4. reduce_scatter: sum partials and scatter back
    Expected: output == original scattered chunk (within fp tolerance)
    """
    try:
        parallelism_config = ParallelismConfig()
        base_port = nccl_port + 11
        nccl_comm_config = NcclCommConfig(
            nccl_ip="127.0.0.1",
            tp_nccl_port=base_port - 2,
            dp_tp_nccl_port=base_port - 10,
            ffn_tp_nccl_port=base_port - 5,
        )
        parallelism_config.world_rank = rank
        parallelism_config.world_size = world_size
        parallelism_config.local_rank = (
            rank % torch.cuda.device_count() if torch.cuda.is_available() else 0
        )
        parallelism_config.tp_size = tp_size
        parallelism_config.dp_size = dp_size

        torch.cuda.set_device(parallelism_config.local_rank)
        torch.set_default_device(f"cuda:{parallelism_config.local_rank}")
        init_distributed_environment(
            parallelism_config,
            nccl_comm_config=nccl_comm_config,
            nccl_init_port=nccl_port,
            backend="nccl",
            timeout=60,
        )

        chunk_size = 8
        hidden_dim = 16
        device = f"cuda:{parallelism_config.local_rank}"

        # Simulate scattered hidden_states (each rank has its own chunk)
        torch.manual_seed(42 + rank)
        hidden_scattered = torch.randn(chunk_size, hidden_dim, device=device)
        residual_scattered = torch.randn(chunk_size, hidden_dim, device=device)

        # Step 1: allgather
        hidden_full = all_gather(hidden_scattered, group=Group.TP)
        residual_full = all_gather(residual_scattered, group=Group.TP)
        assert hidden_full.shape[0] == tp_size * chunk_size

        # Step 2: simulate MoE partial compute (each rank does 1/tp_size)
        hidden_partial = hidden_full / tp_size
        residual_partial = residual_full / tp_size

        # Step 3: reduce_scatter
        hidden_out = reduce_scatter(hidden_partial, group=Group.TP)
        residual_out = reduce_scatter(residual_partial, group=Group.TP)
        torch.cuda.synchronize()

        # Verify: output should match original scattered chunk
        assert hidden_out.shape == hidden_scattered.shape, (
            f"Rank {rank}: shape mismatch {hidden_out.shape} vs {hidden_scattered.shape}"
        )
        assert torch.allclose(hidden_out, hidden_scattered, atol=1e-5), (
            f"Rank {rank}: hidden max diff = {(hidden_out - hidden_scattered).abs().max().item()}"
        )
        assert torch.allclose(residual_out, residual_scattered, atol=1e-5), (
            f"Rank {rank}: residual max diff = {(residual_out - residual_scattered).abs().max().item()}"
        )

        torch.distributed.barrier()
        torch.cuda.synchronize()
        destroy_distributed_environment()
    except Exception as e:
        print(f"Rank {rank} error: {e}")
        import traceback
        traceback.print_exc()
        raise


class TestCpMoeRoundtrip(unittest.TestCase):
    """Test allgather+reduce_scatter roundtrip with real NCCL (multi-GPU)."""

    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass
        self.port_manager = PortManager()

    def _run_test(self, worker_func, world_size, tp_size, dp_size):
        ports, locks = self.port_manager.get_consecutive_ports(1)
        nccl_port = ports[0]
        try:
            processes = []
            for rank in range(world_size):
                p = mp.Process(
                    target=worker_func,
                    args=(rank, world_size, tp_size, dp_size, nccl_port),
                    name=f"rank-{rank}",
                )
                p.start()
                processes.append(p)
            for p in processes:
                p.join(timeout=120)
                if p.exitcode != 0:
                    raise RuntimeError(f"Process {p.name} exited with code {p.exitcode}")
        finally:
            for lock in locks:
                lock.__exit__(None, None, None)

    def test_roundtrip_tp4(self):
        """allgather+reduce_scatter roundtrip with tp_size=4"""
        self._run_test(_test_cp_moe_roundtrip_worker, world_size=4, tp_size=4, dp_size=1)

    def test_roundtrip_tp2_dp2(self):
        """allgather+reduce_scatter roundtrip with tp_size=2, dp_size=2"""
        self._run_test(_test_cp_moe_roundtrip_worker, world_size=4, tp_size=2, dp_size=2)


if __name__ == "__main__":
    os.environ["NCCL_DEBUG"] = "INFO"
    unittest.main()
