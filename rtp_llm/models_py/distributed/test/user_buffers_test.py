"""Unit tests for user_buffers.py

This module tests the UserBufferCommunicator class which provides
GPU-to-GPU communication using CUDA IPC shared buffers.
"""

import logging
import multiprocessing as mp
import os
import unittest
from unittest import mock

logging.basicConfig(level=logging.INFO)

import torch
import torch.distributed as dist

from rtp_llm.models_py.distributed.collective_torch import (
    destroy_distributed_environment,
    init_distributed_environment,
    init_user_buffers_environment,
)
from rtp_llm.models_py.distributed.user_buffers import (
    UserBufferCommunicator,
    get_user_buffers_communicator,
)
from rtp_llm.ops import (
    CPRotateMethod,
    NcclCommConfig,
    ParallelismConfig,
    PrefillCPConfig,
)
from rtp_llm.test.utils.port_util import PortManager

BUFFER_SIZE = 128 * 1024 * 1024


def get_parallelism_config(world_rank, world_size, tp_size, dp_size, port):
    parallelism_config = ParallelismConfig()
    parallelism_config.world_rank = world_rank
    parallelism_config.world_size = world_size
    parallelism_config.local_rank = (
        world_rank % torch.cuda.device_count() if torch.cuda.is_available() else 0
    )
    parallelism_config.tp_size = tp_size
    parallelism_config.dp_size = dp_size

    parallelism_config.prefill_cp_config = PrefillCPConfig()
    parallelism_config.prefill_cp_config.method = CPRotateMethod.ALL_GATHER
    parallelism_config.prefill_cp_config.comm_buffer_size = BUFFER_SIZE

    master_port = int(os.getenv("MASTER_PORT", "8376"))
    base_port = master_port + 11
    nccl_comm_config = NcclCommConfig(
        nccl_ip="127.0.0.1",
        tp_nccl_port=base_port - 2,
        dp_tp_nccl_port=base_port - 10,
        ffn_tp_nccl_port=base_port - 5,
    )
    nccl_init_port = base_port - 11

    return parallelism_config, nccl_comm_config, nccl_init_port


# Test functions that operate on a communicator instance
def _test_basic_properties(
    comm: UserBufferCommunicator, rank: int, world_size: int, buffer_size: int
):
    """Test basic properties of the communicator"""
    assert comm.local_rank == rank
    assert comm.world_size == world_size
    assert comm.buffer_size == buffer_size
    expected_device = torch.device(f"cuda:{rank}")
    assert comm.device == expected_device
    logging.info(f"Rank {rank}: basic properties test passed")


def _test_buffer_internals(comm: UserBufferCommunicator, rank: int):
    """Test that communicator maintains buffer references and streams"""
    # Check buffer pointers and handles are initialized
    assert comm._buffer_ptrs is not None
    assert comm._communicator_ptr is not None
    assert comm._ub_handle is not None

    assert len(comm._send_stream_ids) == comm.world_size
    assert comm._current_stream is not None
    assert comm._recv_stream is not None
    logging.info(f"Rank {rank}: buffer internals test passed")


def _test_send_recv_tensor(comm: UserBufferCommunicator, rank: int, world_size: int):
    prev_rank = (rank - 1) % world_size
    next_rank = (rank + 1) % world_size
    src_tensor = rank * torch.ones(
        [1024, 4096], dtype=torch.float32, device=torch.cuda.current_device()
    )
    dst_tensor = torch.empty(
        [1024, 4096], dtype=torch.float32, device=torch.cuda.current_device()
    )
    comm.send(src_tensor, next_rank)
    comm.recv(dst_tensor, prev_rank)
    expect_tensor = prev_rank * torch.ones(
        [1024, 4096], dtype=torch.float32, device=src_tensor.device
    )
    assert torch.equal(expect_tensor, dst_tensor)
    logging.info(f"Rank {rank}: send recv valid tensor test passed")


def _test_all_gather_tensor(comm: UserBufferCommunicator, rank: int, world_size: int):
    src_tensor = rank * torch.ones(
        [1, 4096], dtype=torch.float32, device=torch.cuda.current_device()
    )
    expect_tensor = (
        torch.arange(world_size, dtype=torch.float32, device=src_tensor.device)
        .unsqueeze(1)
        .repeat(1, 4096)
    )

    all_gather_tensor = comm.all_gather(src_tensor)

    assert torch.equal(expect_tensor, all_gather_tensor)
    logging.info(f"Rank {rank}: all_gather returns tensor test passed")


# Worker functions that create communicator and run all tests
def run_user_buffer_test_main(rank: int, world_size: int, port: int):
    """Worker function that creates one communicator and tests all interfaces"""
    logging.info(f"Rank {rank}: starting all interfaces test")

    try:

        parallelism_config, nccl_comm_config, nccl_init_port = get_parallelism_config(
            rank, world_size, world_size, 1, port
        )
        torch.cuda.set_device(parallelism_config.local_rank)
        torch.set_default_device(f"cuda:{parallelism_config.local_rank}")
        init_distributed_environment(
            parallelism_config,
            nccl_comm_config=nccl_comm_config,
            nccl_init_port=nccl_init_port,
            backend="nccl",
            timeout=60,
        )
        # use cp group for test
        ub_communicator = get_user_buffers_communicator()

        _test_basic_properties(ub_communicator, rank, world_size, BUFFER_SIZE)
        _test_buffer_internals(ub_communicator, rank)
        _test_send_recv_tensor(ub_communicator, rank, world_size)
        _test_all_gather_tensor(ub_communicator, rank, world_size)

        logging.info(f"Rank {rank}: all tests passed")

        torch.cuda.synchronize()
        destroy_distributed_environment()

    except Exception as e:
        print(f"Rank {rank} error in collective operations test: {e}")
        raise


class TestUserBufferCommunicator(unittest.TestCase):
    """Test UserBufferCommunicator with single process and multiprocess scenarios"""

    def setUp(self):
        """Set up test fixtures"""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        # Set spawn method for multiprocessing
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass  # Already set
        self.port_manager = PortManager()

    def _run_multi_process_test(self, worker_func, world_size: int, test_name: str):
        """Helper to run a multi-process test"""
        if torch.cuda.device_count() < world_size:
            self.skipTest(f"Need at least {world_size} GPUs")

        ports, locks = self.port_manager.get_consecutive_ports(1)
        master_port = ports[0]

        try:
            processes = []
            for rank in range(world_size):
                p = mp.Process(
                    target=worker_func,
                    args=(rank, world_size, master_port),
                    name=f"rank-{rank}",
                )
                p.start()
                processes.append(p)

            # Wait for all processes to complete
            for p in processes:
                p.join(timeout=120)
                if p.exitcode != 0:
                    raise RuntimeError(
                        f"Process {p.name} exited with code {p.exitcode}"
                    )
        finally:
            # Release port locks
            for lock in locks:
                lock.__exit__(None, None, None)

    def test_user_buffers_worldsize_2(self):
        """Test all interfaces with multiple processes"""
        self._run_multi_process_test(
            run_user_buffer_test_main,
            world_size=2,
            test_name="test_user_buffers_worldsize_2",
        )

    def test_user_buffers_worldsize_4(self):
        """Test all interfaces with multiple processes"""
        self._run_multi_process_test(
            run_user_buffer_test_main,
            world_size=4,
            test_name="test_user_buffers_worldsize_4",
        )


if __name__ == "__main__":
    unittest.main()
