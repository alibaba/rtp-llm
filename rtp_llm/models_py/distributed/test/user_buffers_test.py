"""Unit tests for user_buffers.py

This module tests the UserBufferCommunicator class which provides
GPU-to-GPU communication using CUDA IPC shared buffers.
"""

import logging
import multiprocessing as mp
import unittest
from unittest import mock

logging.basicConfig(level=logging.INFO)

import torch
import torch.distributed as dist

from rtp_llm.models_py.distributed import user_buffers
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
NCCL_PORT_COUNT = 12


def get_parallelism_config(world_rank, world_size, tp_size, dp_size, port):
    parallelism_config = ParallelismConfig()
    parallelism_config.world_rank = world_rank
    parallelism_config.world_size = world_size
    parallelism_config.local_rank = (
        world_rank % torch.cuda.device_count() if torch.cuda.is_available() else 0
    )
    parallelism_config.tp_size = tp_size
    parallelism_config.dp_size = dp_size

    parallelism_config.local_world_size = world_size

    parallelism_config.prefill_cp_config = PrefillCPConfig()
    parallelism_config.prefill_cp_config.method = CPRotateMethod.ALL_GATHER_WITH_OVERLAP
    parallelism_config.prefill_cp_config.comm_buffer_size = BUFFER_SIZE
    parallelism_config.use_ub_comm = True

    base_port = port + 11
    nccl_comm_config = NcclCommConfig(
        nccl_ip="127.0.0.1",
        tp_nccl_port=base_port - 2,
        dp_tp_nccl_port=base_port - 10,
        ffn_tp_nccl_port=base_port - 5,
    )
    nccl_init_port = port

    return parallelism_config, nccl_comm_config, nccl_init_port


# Test functions that operate on a communicator instance
def _test_basic_properties(
    comm: UserBufferCommunicator,
    group_rank: int,
    group_size: int,
    device_index: int,
    buffer_size: int,
):
    """Test basic properties of the communicator."""
    assert comm.group_rank == group_rank
    assert comm.group_size == group_size
    assert comm.device_index == device_index
    assert comm.buffer_size == buffer_size
    assert comm.device == torch.device(f"cuda:{device_index}")
    logging.info(f"Group rank {group_rank}: basic properties test passed")


def _test_buffer_internals(comm: UserBufferCommunicator, group_rank: int):
    """Test that communicator maintains buffer references and streams."""
    assert comm._buffer_ptrs is not None
    assert comm._communicator_ptr is not None
    assert comm._ub_handle is not None

    assert len(comm._send_stream_ids) == comm.group_size
    assert comm._current_stream is not None
    assert comm._recv_stream is not None
    logging.info(f"Group rank {group_rank}: buffer internals test passed")


def _test_send_recv_tensor(
    comm: UserBufferCommunicator, group_rank: int, group_size: int
):
    prev_rank = (group_rank - 1) % group_size
    next_rank = (group_rank + 1) % group_size
    src_tensor = group_rank * torch.ones(
        [1024, 4096], dtype=torch.float32, device=torch.cuda.current_device()
    )
    dst_tensor = torch.empty(
        [1024, 4096], dtype=torch.float32, device=torch.cuda.current_device()
    )
    assert comm.send_recv(src_tensor, next_rank, dst_tensor, prev_rank)
    expect_tensor = prev_rank * torch.ones(
        [1024, 4096], dtype=torch.float32, device=src_tensor.device
    )
    assert torch.equal(expect_tensor, dst_tensor)
    logging.info(f"Group rank {group_rank}: send_recv valid tensor test passed")


def _test_all_gather_tensor(
    comm: UserBufferCommunicator, group_rank: int, group_size: int
):
    src_tensor = group_rank * torch.ones(
        [1, 4096], dtype=torch.float32, device=torch.cuda.current_device()
    )
    expect_tensor = (
        torch.arange(group_size, dtype=torch.float32, device=src_tensor.device)
        .unsqueeze(1)
        .repeat(1, 4096)
    )

    all_gather_tensor = comm.all_gather(src_tensor)

    assert torch.equal(expect_tensor, all_gather_tensor)
    logging.info(f"Group rank {group_rank}: all_gather returns tensor test passed")


# Worker functions that create communicator and run all tests
def _run_user_buffer_test_main(
    rank: int,
    world_size: int,
    tp_size: int,
    dp_size: int,
    port: int,
    test_reinitialization: bool,
):
    logging.info(f"Rank {rank}: starting all interfaces test")

    try:
        parallelism_config, nccl_comm_config, nccl_init_port = get_parallelism_config(
            rank, world_size, tp_size, dp_size, port
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
        ub_communicator = get_user_buffers_communicator()
        assert ub_communicator is not None
        group_rank = dist.get_rank(ub_communicator.group)
        group_size = dist.get_world_size(ub_communicator.group)
        group_start = (rank // tp_size) * tp_size
        expected_devices = [
            peer_rank % torch.cuda.device_count()
            for peer_rank in range(group_start, group_start + tp_size)
        ]
        assert ub_communicator._group_device_indices == expected_devices

        _test_basic_properties(
            ub_communicator,
            group_rank,
            group_size,
            parallelism_config.local_rank,
            BUFFER_SIZE,
        )
        _test_buffer_internals(ub_communicator, group_rank)
        _test_send_recv_tensor(ub_communicator, group_rank, group_size)
        _test_all_gather_tensor(ub_communicator, group_rank, group_size)

        if test_reinitialization:
            dist.barrier(group=ub_communicator.group)
            user_buffers.destroy_user_buffers_communicator()
            assert ub_communicator._communicator_ptr is None
            assert get_user_buffers_communicator() is None
            user_buffers.destroy_user_buffers_communicator()

            init_user_buffers_environment(parallelism_config)
            reinitialized = get_user_buffers_communicator()
            assert reinitialized is not None
            assert reinitialized is not ub_communicator
            _test_send_recv_tensor(reinitialized, group_rank, group_size)
            dist.barrier(group=reinitialized.group)

        logging.info(f"Rank {rank}: all tests passed")

        torch.cuda.synchronize()
        destroy_distributed_environment()

    except Exception as e:
        print(f"Rank {rank} error in collective operations test: {e}")
        raise


def run_user_buffer_test_main(rank: int, world_size: int, port: int):
    _run_user_buffer_test_main(rank, world_size, world_size, 1, port, True)


def run_user_buffer_tp2_dp2_test_main(rank: int, world_size: int, port: int):
    _run_user_buffer_test_main(rank, world_size, 2, 2, port, False)


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

    def test_send_recv_preserves_producer_and_consumer_stream_order(self):
        comm = object.__new__(UserBufferCommunicator)
        comm.group_rank = 0
        comm.per_rank_buffer_size = 1024
        comm._ub_handle = 1
        comm._communicator_ptr = 2
        comm._rank_offsets = {0: 0, 1: 512}
        send_stream = mock.Mock(cuda_stream=11)
        recv_stream = mock.Mock(cuda_stream=12)
        current_stream = mock.Mock()
        comm._send_streams = {1: send_stream}
        comm._recv_stream = recv_stream
        comm.cleanup = mock.Mock()

        timeline = []
        send_stream.wait_stream.side_effect = lambda _: timeline.append("send-ready")
        recv_stream.wait_stream.side_effect = lambda _: timeline.append("recv-ready")
        current_stream.wait_stream.side_effect = lambda stream: timeline.append(
            "send-done" if stream is send_stream else "recv-done"
        )

        with mock.patch.object(
            torch.cuda, "current_stream", return_value=current_stream
        ), mock.patch.object(
            user_buffers,
            "userbuffers_send",
            side_effect=lambda *args: timeline.append("send"),
        ), mock.patch.object(
            user_buffers,
            "userbuffers_recv",
            side_effect=lambda *args: timeline.append("recv"),
        ):
            send_tensor = torch.empty(4, device="cuda")
            recv_tensor = torch.empty_like(send_tensor)
            self.assertTrue(comm.send_recv(send_tensor, 1, recv_tensor, 1))

        self.assertEqual(
            timeline,
            ["send-ready", "send", "recv-ready", "recv", "send-done", "recv-done"],
        )

    def _run_multi_process_test(self, worker_func, world_size: int, test_name: str):
        """Helper to run a multi-process test"""
        if torch.cuda.device_count() < world_size:
            self.skipTest(f"Need at least {world_size} GPUs")

        ports, locks = self.port_manager.get_consecutive_ports(NCCL_PORT_COUNT)
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

            # Wait for all processes to complete, then clean up every child
            # before surfacing failures so later cases do not inherit sockets.
            for p in processes:
                p.join(timeout=120)

            failures = []
            for p in processes:
                if p.is_alive():
                    p.terminate()
                    p.join(timeout=10)
                    failures.append(f"{p.name} timed out")
                elif p.exitcode != 0:
                    failures.append(f"{p.name} exited with code {p.exitcode}")

            if failures:
                raise RuntimeError(f"{test_name} failed: {', '.join(failures)}")
        finally:
            for p in processes:
                if p.is_alive():
                    p.terminate()
                    p.join(timeout=10)
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
        """Test all interfaces with multiple processes."""
        self._run_multi_process_test(
            run_user_buffer_test_main,
            world_size=4,
            test_name="test_user_buffers_worldsize_4",
        )

    def test_user_buffers_tp2_dp2(self):
        """Test two independent TP communicators across four devices."""
        self._run_multi_process_test(
            run_user_buffer_tp2_dp2_test_main,
            world_size=4,
            test_name="test_user_buffers_tp2_dp2",
        )


if __name__ == "__main__":
    unittest.main()
