"""Unit tests for collective_torch.py

This module tests the torch.distributed-based collective operations
using real multiprocessing spawn.
"""

import multiprocessing as mp
import unittest
import logging
import os

import torch

from rtp_llm.models_py.distributed.collective_torch import (
    Group,
    _get_group,
    init_distributed_environment,
    distributed_environment_initialized,
    destroy_distributed_environment,
    send,
    recv,
    broadcast,
    all_reduce,
    all_gather,
)
from rtp_llm.ops import ParallelismConfig
from rtp_llm.test.utils.port_util import PortManager
from pytest import mark


def _calculate_group_ranks(rank: int, world_size: int, tp_size: int, group_type: Group):
    """Calculate ranks in a specific group"""
    if group_type == Group.DP_AND_TP:
        return list(range(world_size))
    elif group_type == Group.DP:
        tp_rank = rank % tp_size
        return [r for r in range(world_size) if r % tp_size == tp_rank]
    else:  # Group.TP
        dp_rank = rank // tp_size
        return [r for r in range(world_size) if r // tp_size == dp_rank]


def _test_all_reduce_collective(
    rank: int,
    parallelism_config: ParallelismConfig,
    world_size: int,
    tp_size: int,
    dp_size: int,
    get_process_group_and_ranks,
):
    """Test all_reduce collective operation across all groups"""
    logging.info(f"Rank {rank} testing all_reduce")
    for group_type in [Group.DP_AND_TP, Group.DP, Group.TP]:
        # Skip if group doesn't make sense for this configuration
        if group_type == Group.DP and dp_size == 1:
            continue
        if group_type == Group.TP and tp_size == 1:
            continue
        
        process_group, group_ranks = get_process_group_and_ranks(group_type)
        
        logging.info(f"Rank {rank} process_group: {process_group}, group_ranks: {group_ranks}")
        # Only test if this rank is in the group
        if rank in group_ranks:
            logging.info(f"Rank {rank} all_reduce: {group_type} {group_ranks}")
            tensor = torch.ones(3, device=f"cuda:{parallelism_config.local_rank}") * (rank + 1)
            all_reduce(tensor, group=group_type)
            torch.cuda.synchronize()
            torch.distributed.barrier(group=process_group)
            
            expected_sum = sum(r + 1 for r in group_ranks)
            expected = torch.ones(3, device=f"cuda:{parallelism_config.local_rank}") * expected_sum
            logging.info(f"Rank {rank} expected: {expected.cpu()}, got {tensor.cpu()}")
            assert torch.allclose(tensor, expected), (
                f"Rank {rank} all_reduce {group_type}: Expected {expected.cpu()}, got {tensor.cpu()}"
            )
    
    # All ranks synchronize before next test
    torch.distributed.barrier()


def _test_broadcast_collective(
    rank: int,
    parallelism_config: ParallelismConfig,
    world_size: int,
    tp_size: int,
    dp_size: int,
    get_process_group_and_ranks,
):
    """Test broadcast collective operation across all groups"""
    logging.info(f"Rank {rank} testing broadcast")
    for group_type in [Group.DP_AND_TP, Group.DP, Group.TP]:
        # Skip if group doesn't make sense for this configuration
        if group_type == Group.DP and dp_size == 1:
            continue
        if group_type == Group.TP and tp_size == 1:
            continue
        
        process_group, group_ranks = get_process_group_and_ranks(group_type)
        
        # Only test if group has at least 2 ranks and this rank is in the group
        if len(group_ranks) >= 2 and rank in group_ranks:
            src_rank = group_ranks[0]
            
            if rank == src_rank:
                tensor = torch.tensor([1.0, 2.0, 3.0], device=f"cuda:{parallelism_config.local_rank}")
            else:
                tensor = torch.zeros(3, device=f"cuda:{parallelism_config.local_rank}")
            
            broadcast(tensor, src=src_rank, group=group_type)
            torch.cuda.synchronize()
            torch.distributed.barrier(group=process_group)
            
            expected = torch.tensor([1.0, 2.0, 3.0], device=f"cuda:{parallelism_config.local_rank}")
            assert torch.allclose(tensor, expected), (
                f"Rank {rank} broadcast {group_type}: Expected {expected.cpu()}, got {tensor.cpu()}"
            )
    
    # All ranks synchronize before next test
    torch.distributed.barrier()


def _test_send_recv_collective(
    rank: int,
    parallelism_config: ParallelismConfig,
    world_size: int,
    tp_size: int,
    dp_size: int,
    get_process_group_and_ranks,
):
    """Test send/recv collective operation across all groups
    
    Simple test: if group has at least 2 ranks, rank 0 sends and rank 1 receives.
    """
    logging.info(f"Rank {rank} testing send/recv")
    for group_type in [Group.DP_AND_TP, Group.DP, Group.TP]:
        # Skip if group doesn't make sense for this configuration
        if group_type == Group.DP and dp_size == 1:
            continue
        if group_type == Group.TP and tp_size == 1:
            continue
        
        process_group, group_ranks = get_process_group_and_ranks(group_type)
        
        # Only test if group has at least 2 ranks
        if len(group_ranks) >= 2:
            src_rank = group_ranks[0]
            dst_rank = group_ranks[1]
            
            if rank == src_rank:
                logging.info(f"Rank {rank} sending to rank {dst_rank}, {group_type} {group_ranks}")
                tensor = torch.tensor([1.0, 2.0, 3.0], device=f"cuda:{parallelism_config.local_rank}")
                send(tensor, dst=dst_rank, group=group_type)
                torch.cuda.synchronize()
            elif rank == dst_rank:
                logging.info(f"Rank {rank} receiving from rank {src_rank}, {group_type} {group_ranks}")
                tensor = torch.zeros(3, device=f"cuda:{parallelism_config.local_rank}")
                recv(tensor, src=src_rank, group=group_type)
                torch.cuda.synchronize()
                expected = torch.tensor([1.0, 2.0, 3.0], device=f"cuda:{parallelism_config.local_rank}")
                assert torch.allclose(tensor, expected), (
                    f"Rank {rank} send/recv {group_type}: Expected [1,2,3], got {tensor.cpu()}"
                )
            
            torch.distributed.barrier(group=process_group)
    
    # All ranks synchronize before next test
    torch.distributed.barrier()


def _test_all_gather_collective(
    rank: int,
    parallelism_config: ParallelismConfig,
    world_size: int,
    tp_size: int,
    dp_size: int,
    get_process_group_and_ranks,
):
    """Test all_gather collective operation across all groups"""
    logging.info(f"Rank {rank} testing all_gather")
    for group_type in [Group.DP_AND_TP, Group.DP, Group.TP]:
        # Skip if group doesn't make sense for this configuration
        if group_type == Group.DP and dp_size == 1:
            continue
        if group_type == Group.TP and tp_size == 1:
            continue
        
        process_group, group_ranks = get_process_group_and_ranks(group_type)
        
        # Only test if this rank is in the group
        if rank in group_ranks:
            tensor = torch.ones(2, device=f"cuda:{parallelism_config.local_rank}") * (rank + 1)
            result = all_gather(tensor, group=group_type)
            torch.cuda.synchronize()
            torch.distributed.barrier(group=process_group)
            
            # Expected: concatenated tensors from all ranks in the group
            expected_list = [torch.ones(2, device=f"cuda:{parallelism_config.local_rank}") * (r + 1) for r in group_ranks]
            expected = torch.cat(expected_list, dim=0)
            assert torch.allclose(result, expected), (
                f"Rank {rank} all_gather {group_type}: Expected {expected.cpu()}, got {result.cpu()}"
            )
    
    # All ranks synchronize before next test
    torch.distributed.barrier()


def _test_all_collectives_worker(rank: int, world_size: int, tp_size: int, dp_size: int, nccl_port: int):
    """Worker function for testing all collective operations across all groups in a single initialization"""
    logging.info("test world_size: %d, tp_size: %d, dp_size: %d", world_size, tp_size, dp_size)
    try:
        # Setup - initialize once for all tests
        parallelism_config = ParallelismConfig()
        parallelism_config.nccl_ip = "127.0.0.1"
        parallelism_config.th_nccl_port = nccl_port
        parallelism_config.world_rank = rank
        parallelism_config.world_size = world_size
        parallelism_config.local_rank = rank % torch.cuda.device_count() if torch.cuda.is_available() else 0
        parallelism_config.tp_size = tp_size
        parallelism_config.dp_size = dp_size
        
        torch.cuda.set_device(parallelism_config.local_rank)
        torch.set_default_device(f"cuda:{parallelism_config.local_rank}")
        init_distributed_environment(parallelism_config, backend="nccl", timeout=60)
        
        # Helper function to get process group and group ranks
        def _get_process_group_and_ranks(group_type: Group):
            """Get process group and ranks for a given group type"""
            if group_type == Group.DP_AND_TP:
                process_group = torch.distributed.group.WORLD
            elif group_type == Group.DP:
                process_group = _get_group(Group.DP)
            else:  # Group.TP
                process_group = _get_group(Group.TP)
            
            group_ranks = _calculate_group_ranks(rank, world_size, tp_size, group_type)
            return process_group, group_ranks
        
        # Test all collective operations
        _test_all_reduce_collective(rank, parallelism_config, world_size, tp_size, dp_size, _get_process_group_and_ranks)
        _test_broadcast_collective(rank, parallelism_config, world_size, tp_size, dp_size, _get_process_group_and_ranks)
        _test_send_recv_collective(rank, parallelism_config, world_size, tp_size, dp_size, _get_process_group_and_ranks)
        _test_all_gather_collective(rank, parallelism_config, world_size, tp_size, dp_size, _get_process_group_and_ranks)
        
        # All ranks synchronize at the end
        torch.distributed.barrier()
        torch.cuda.synchronize()
        destroy_distributed_environment()
    except Exception as e:
        print(f"Rank {rank} error in collective operations test: {e}")
        import traceback
        traceback.print_exc()
        raise

@mark.H20
@mark.gpu(count=4)
@mark.cuda
class TestCollectiveOperations(unittest.TestCase):
    """Test collective operations with real multiprocessing"""
    
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
    
    def tearDown(self):
        """Clean up after tests"""
        pass
    
    def _run_test(self, worker_func, world_size: int, tp_size: int, dp_size: int, test_name: str):
        """Helper to run a test with multiple processes"""
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
            
            # Wait for all processes to complete
            for p in processes:
                p.join(timeout=120)
                if p.exitcode != 0:
                    raise RuntimeError(f"Process {p.name} exited with code {p.exitcode}")
        finally:
            # Release port locks
            for lock in locks:
                lock.__exit__(None, None, None)
    
    # Test configurations: world_size=4, tp_size=2, dp_size=2
    def test_all_collectives_tp2_dp2(self):
        """Test all collective operations (all_reduce, broadcast, send/recv, all_gather) with tp_size=2, dp_size=2"""
        self._run_test(_test_all_collectives_worker, world_size=4, tp_size=2, dp_size=2, test_name="all_collectives_tp2_dp2")
    
    # Test configurations: world_size=4, tp_size=4, dp_size=1
    def test_all_collectives_tp4_dp1(self):
        """Test all collective operations (all_reduce, broadcast, send/recv, all_gather) with tp_size=4, dp_size=1"""
        self._run_test(_test_all_collectives_worker, world_size=4, tp_size=4, dp_size=1, test_name="all_collectives_tp4_dp1")
    
    # Test configurations: world_size=4, tp_size=1, dp_size=4
    def test_all_collectives_tp1_dp4(self):
        """Test all collective operations (all_reduce, broadcast, send/recv, all_gather) with tp_size=1, dp_size=4"""
        self._run_test(_test_all_collectives_worker, world_size=4, tp_size=1, dp_size=4, test_name="all_collectives_tp1_dp4")


@mark.H20
@mark.gpu
@mark.cuda
class TestDistributedEnvironment(unittest.TestCase):
    """Test distributed environment initialization"""
    
    def setUp(self):
        mp.set_start_method("spawn", force=True)
    
    def test_distributed_environment_initialized(self):
        """Test checking if distributed environment is initialized"""
        self.assertFalse(distributed_environment_initialized())


if __name__ == '__main__':
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["NCCL_DEBUG_FILE"] = "nccl.log"
    unittest.main()
