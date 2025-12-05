import itertools
import logging
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional
from unittest import SkipTest, TestCase, main

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed import ProcessGroup

from rtp_llm.models_py.distributed.symm_mem import TorchSymmMemCommunicator
from rtp_llm.test.utils.port_util import PortsContext

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def init_distributed_for_test(
    rank: int, world_size: int, master_addr: str, master_port: str
):
    """Initialize distributed environment for testing."""
    if dist.is_initialized():
        dist.destroy_process_group()

    # Set CUDA device for this rank
    local_rank = rank % torch.cuda.device_count()
    torch.cuda.set_device(local_rank)
    os.environ["LOCAL_RANK"] = str(local_rank)

    try:
        dist.init_process_group(
            backend="nccl",
            init_method=f"tcp://{master_addr}:{master_port}",
            world_size=world_size,
            rank=rank,
        )
    except Exception as e:
        logging.error(f"Failed to initialize process group: {e}")
        return None, None, None

    # Create a process group for testing
    ranks = list(range(world_size))
    group = dist.new_group(ranks, backend="nccl")

    return group, rank, world_size


def cleanup_distributed():
    """Clean up distributed environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


class TorchSymmMemTest(TestCase):
    """Unit tests for TorchSymmMemCommunicator."""

    TENSOR_SIZES = [
        (128, 1024),  # seq_len, hidden_size
        (256, 2048),
        (512, 4096),
        (1024, 2048),
    ]

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            raise SkipTest("CUDA is not available")
        torch.set_default_device(device)
        set_seed(42)

    def tearDown(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _create_communicator(
        self, group: ProcessGroup, device: torch.device
    ) -> Optional[TorchSymmMemCommunicator]:
        """Helper to create a communicator."""
        try:
            comm = TorchSymmMemCommunicator(group, device)
            return comm if not comm.disabled else None
        except Exception as e:
            logging.warning(f"Failed to create communicator: {e}")
            return None

    def _test_all_reduce_correctness(
        self, group: ProcessGroup, rank: int, world_size: int
    ):
        """Test that all_reduce correctly sums tensors across ranks."""
        device_obj = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
        comm = self._create_communicator(group, device_obj)

        if comm is None:
            self.skipTest("TorchSymmMemCommunicator is not available")

        # Create test tensor
        seq_len, hidden_size = 256, 2048
        tensor_size = seq_len * hidden_size
        inp = torch.randn(
            (seq_len, hidden_size), dtype=torch.bfloat16, device=device_obj
        )

        # Check if tensor is eligible for symmetric memory
        if not comm.should_torch_symm_mem_allreduce(inp):
            self.skipTest(
                f"Tensor size {tensor_size} is not eligible for symmetric memory"
            )

        # Initialize each rank with different values
        inp.fill_(float(rank + 1))

        # Perform all_reduce
        out = comm.all_reduce(inp)

        if out is None:
            self.skipTest("all_reduce returned None (communicator disabled)")

        # Verify result: should be sum of all ranks
        expected_sum = sum(range(1, world_size + 1)) * tensor_size
        actual_sum = out.sum().item()

        self.assertAlmostEqual(
            actual_sum,
            expected_sum,
            places=1,
            msg=f"Rank {rank}: Expected sum {expected_sum}, got {actual_sum}",
        )

        # Verify all ranks have the same result
        gathered = [torch.zeros_like(out) for _ in range(world_size)]
        dist.all_gather(gathered, out, group=group)

        if rank == 0:
            for i in range(1, world_size):
                self.assertTrue(
                    torch.allclose(gathered[0], gathered[i], atol=1e-2, rtol=1e-2),
                    f"Rank 0 and rank {i} have different results",
                )

    @staticmethod
    def _run_all_reduce_correctness_test(
        rank: int, world_size: int, master_addr: str, master_port: str
    ):
        """Worker function for all_reduce correctness test."""
        try:
            group, rank, world_size = init_distributed_for_test(
                rank, world_size, master_addr, master_port
            )
            if group is None:
                return

            # Create test instance
            test = TorchSymmMemTest()
            test.setUp()

            try:
                test._test_all_reduce_correctness(group, rank, world_size)
                if dist.is_initialized():
                    dist.barrier(group)
            finally:
                test.tearDown()
                cleanup_distributed()
        except Exception as e:
            logging.error(f"Rank {rank} failed: {e}")
            import traceback

            traceback.print_exc()
            raise

    def test_all_reduce_correctness(self):
        """Test all_reduce correctness using multiprocessing spawn."""
        if not torch.cuda.is_available():
            raise SkipTest("CUDA is not available")

        # Get TP_SIZE from environment or use default
        tp_size = int(os.environ.get("TP_SIZE", "2"))
        if tp_size < 2:
            self.skipTest(f"TP_SIZE={tp_size} is too small, need at least 2")

        # Limit to available GPUs
        num_gpus = torch.cuda.device_count()
        if tp_size > num_gpus:
            logging.warning(
                f"TP_SIZE={tp_size} > available GPUs={num_gpus}, using {num_gpus}"
            )
            tp_size = num_gpus

        with PortsContext(None, 1) as ports:
            master_port = str(ports[0])
            master_addr = "127.0.0.1"

            try:
                mp.spawn(
                    TorchSymmMemTest._run_all_reduce_correctness_test,
                    args=(tp_size, master_addr, master_port),
                    nprocs=tp_size,
                    join=True,
                )
            except Exception as e:
                logging.error(f"Test failed: {e}")
                raise

    @staticmethod
    def _run_all_reduce_different_sizes_test(
        rank: int, world_size: int, master_addr: str, master_port: str
    ):
        """Worker function for all_reduce with different sizes test."""
        try:
            group, rank, world_size = init_distributed_for_test(
                rank, world_size, master_addr, master_port
            )
            if group is None:
                return

            device_obj = torch.device(f"cuda:{rank % torch.cuda.device_count()}")

            # Create test instance
            test = TorchSymmMemTest()
            test.setUp()

            try:
                comm = test._create_communicator(group, device_obj)
                if comm is None:
                    return

                for seq_len, hidden_size in test.TENSOR_SIZES:
                    inp = torch.randn(
                        (seq_len, hidden_size), dtype=torch.bfloat16, device=device_obj
                    )

                    if not comm.should_torch_symm_mem_allreduce(inp):
                        continue  # Skip if not eligible

                    inp.fill_(float(rank + 1))
                    out = comm.all_reduce(inp)

                    if out is not None:
                        expected_sum = sum(range(1, world_size + 1)) * inp.numel()
                        actual_sum = out.sum().item()
                        assert (
                            abs(actual_sum - expected_sum) < 1.0
                        ), f"Size ({seq_len}, {hidden_size}): Expected {expected_sum}, got {actual_sum}"

                if dist.is_initialized():
                    dist.barrier(group)
            finally:
                test.tearDown()
                cleanup_distributed()
        except Exception as e:
            logging.error(f"Rank {rank} failed: {e}")
            import traceback

            traceback.print_exc()
            raise

    def test_all_reduce_with_different_sizes(self):
        """Test all_reduce with different tensor sizes using multiprocessing spawn."""
        if not torch.cuda.is_available():
            raise SkipTest("CUDA is not available")

        tp_size = int(os.environ.get("TP_SIZE", "2"))
        if tp_size < 2:
            self.skipTest(f"TP_SIZE={tp_size} is too small, need at least 2")

        num_gpus = torch.cuda.device_count()
        if tp_size > num_gpus:
            logging.warning(
                f"TP_SIZE={tp_size} > available GPUs={num_gpus}, using {num_gpus}"
            )
            tp_size = num_gpus

        with PortsContext(None, 1) as ports:
            master_port = str(ports[0])
            master_addr = "127.0.0.1"

            try:
                mp.spawn(
                    TorchSymmMemTest._run_all_reduce_different_sizes_test,
                    args=(tp_size, master_addr, master_port),
                    nprocs=tp_size,
                    join=True,
                )
            except Exception as e:
                logging.error(f"Test failed: {e}")
                raise


class TorchSymmMemBenchmark:
    """Benchmark suite for TorchSymmMemCommunicator."""

    def __init__(
        self,
        num_warmup: int = 10,
        num_iterations: int = 100,
        sizes: Optional[List[int]] = None,
    ):
        self.num_warmup = num_warmup
        self.num_iterations = num_iterations
        self.sizes = sizes or self._default_sizes()

    def _default_sizes(self) -> List[int]:
        """Default tensor sizes to benchmark."""
        seq_lens = [128, 256, 512, 1024, 2048, 4096, 8192]
        hidden_sizes = [1024, 2048, 4096]
        sizes = []
        for seq_len in seq_lens:
            for hidden_size in hidden_sizes:
                sizes.append(seq_len * hidden_size)
        return sorted(list(set(sizes)))

    @staticmethod
    def _run_benchmark_worker(
        rank: int,
        world_size: int,
        master_addr: str,
        master_port: str,
        num_warmup: int,
        num_iterations: int,
        sizes: List[int],
    ):
        """Worker function for benchmark."""
        try:
            group, rank, world_size = init_distributed_for_test(
                rank, world_size, master_addr, master_port
            )
            if group is None:
                return

            device_obj = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
            comm = TorchSymmMemCommunicator(group, device_obj)

            if comm.disabled:
                if rank == 0:
                    print("TorchSymmMemCommunicator is disabled, skipping benchmark")
                return

            if rank == 0:
                print(f"\n{'='*80}")
                print(f"TorchSymmMemCommunicator All-Reduce Benchmark")
                print(f"{'='*80}")
                print(f"World size: {world_size}")
                print(f"Device capability: {comm.device_capability}")
                print(f"Max size: {comm.max_size / (1024**2):.2f} MB")
                print(f"Warmup iterations: {num_warmup}")
                print(f"Test iterations: {num_iterations}")
                print(f"{'='*80}\n")

            results = []

            for size in sizes:
                # Create tensor
                seq_len = int(size**0.5)
                hidden_size = (size + seq_len - 1) // seq_len
                actual_size = seq_len * hidden_size

                if actual_size > comm.max_size // torch.bfloat16.itemsize:
                    if rank == 0:
                        print(f"Skipping size {size} (too large)")
                    continue

                inp = torch.randn(
                    (seq_len, hidden_size), dtype=torch.bfloat16, device=device_obj
                )

                if not comm.should_torch_symm_mem_allreduce(inp):
                    if rank == 0:
                        print(f"Skipping size {size} (not eligible)")
                    continue

                # Initialize data
                inp.fill_(float(rank + 1))

                # Warmup
                for _ in range(num_warmup):
                    out = comm.all_reduce(inp)
                    if out is not None:
                        inp.copy_(out)

                torch.cuda.synchronize()
                dist.barrier(group)

                # Benchmark
                start_events = [
                    torch.cuda.Event(enable_timing=True) for _ in range(num_iterations)
                ]
                end_events = [
                    torch.cuda.Event(enable_timing=True) for _ in range(num_iterations)
                ]

                for i in range(num_iterations):
                    start_events[i].record()
                    out = comm.all_reduce(inp)
                    if out is not None:
                        inp.copy_(out)
                    end_events[i].record()

                torch.cuda.synchronize()
                dist.barrier(group)

                # Calculate times
                times = []
                for i in range(num_iterations):
                    elapsed_ms = start_events[i].elapsed_time(end_events[i])
                    elapsed_us = elapsed_ms * 1000.0
                    times.append(elapsed_us)

                avg_time = sum(times) / len(times)
                min_time = min(times)
                max_time = max(times)

                # Calculate bandwidth (GB/s)
                bytes_transferred = actual_size * 2 * 2
                bandwidth_gbps = (bytes_transferred / (avg_time / 1e6)) / (1024**3)

                results.append(
                    {
                        "size": actual_size,
                        "shape": (seq_len, hidden_size),
                        "avg_time_us": avg_time,
                        "min_time_us": min_time,
                        "max_time_us": max_time,
                        "bandwidth_gbps": bandwidth_gbps,
                    }
                )

                if rank == 0:
                    print(f"Size: {actual_size:>10} elements ({seq_len}x{hidden_size})")
                    print(f"  Avg time: {avg_time:>8.3f} us")
                    print(f"  Min time: {min_time:>8.3f} us")
                    print(f"  Max time: {max_time:>8.3f} us")
                    print(f"  Bandwidth: {bandwidth_gbps:>8.2f} GB/s")
                    print()

            if rank == 0:
                print(f"{'='*80}")
                print("Benchmark Summary:")
                print(f"{'='*80}")
                print(
                    f"{'Size':>12} {'Shape':>15} {'Avg (us)':>10} {'Min (us)':>10} {'Max (us)':>10} {'Bandwidth (GB/s)':>18}"
                )
                print(f"{'-'*80}")
                for r in results:
                    shape_str = f"{r['shape'][0]}x{r['shape'][1]}"
                    print(
                        f"{r['size']:>12} {shape_str:>15} {r['avg_time_us']:>10.3f} {r['min_time_us']:>10.3f} {r['max_time_us']:>10.3f} {r['bandwidth_gbps']:>18.2f}"
                    )
                print(f"{'='*80}\n")

            dist.barrier(group)
            cleanup_distributed()
        except Exception as e:
            logging.error(f"Rank {rank} failed: {e}")
            import traceback

            traceback.print_exc()
            raise

    def run_benchmark(self):
        """Run the benchmark."""
        if not torch.cuda.is_available():
            print("CUDA is not available, skipping benchmark")
            return

        # Get TP_SIZE from environment or use default
        tp_size = int(os.environ.get("TP_SIZE", "2"))
        if tp_size < 2:
            print(f"TP_SIZE={tp_size} is too small, need at least 2")
            return

        num_gpus = torch.cuda.device_count()
        if tp_size > num_gpus:
            print(f"TP_SIZE={tp_size} > available GPUs={num_gpus}, using {num_gpus}")
            tp_size = num_gpus

        with PortsContext(None, 1) as ports:
            master_port = str(ports[0])
            master_addr = "127.0.0.1"

            try:
                mp.spawn(
                    TorchSymmMemBenchmark._run_benchmark_worker,
                    args=(
                        tp_size,
                        master_addr,
                        master_port,
                        self.num_warmup,
                        self.num_iterations,
                        self.sizes,
                    ),
                    nprocs=tp_size,
                    join=True,
                )
            except Exception as e:
                print(f"Benchmark failed: {e}")
                import traceback

                traceback.print_exc()


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Test and benchmark TorchSymmMemCommunicator"
    )
    parser.add_argument(
        "--benchmark", action="store_true", help="Run benchmark instead of tests"
    )
    parser.add_argument(
        "--warmup", type=int, default=10, help="Number of warmup iterations"
    )
    parser.add_argument(
        "--iterations", type=int, default=100, help="Number of test iterations"
    )
    parser.add_argument(
        "--sizes",
        type=str,
        default=None,
        help="Comma-separated list of tensor sizes (e.g., '262144,1048576,4194304')",
    )
    parser.add_argument(
        "--tp-size",
        type=int,
        default=None,
        help="Tensor parallelism size (number of processes). Defaults to TP_SIZE env var or 2",
    )

    args = parser.parse_args()

    # Set TP_SIZE from command line if provided
    if args.tp_size is not None:
        os.environ["TP_SIZE"] = str(args.tp_size)

    try:
        if args.benchmark:
            sizes = None
            if args.sizes:
                sizes = [int(s.strip()) for s in args.sizes.split(",")]
            benchmark = TorchSymmMemBenchmark(
                num_warmup=args.warmup, num_iterations=args.iterations, sizes=sizes
            )
            benchmark.run_benchmark()
        else:
            main()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)
