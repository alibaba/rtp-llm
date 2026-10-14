# type: ignore
"""Two-rank destination counts after actual NCCL all-gather dispatch scaffolding.

This checks local expert ownership and global reduction, not DeepEP transport.
"""

import socket
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


def _rank_main(rank, init_method, queue):
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group(
        "nccl", init_method=init_method, rank=rank, world_size=2
    )
    try:
        from expert_load_recorder_test import (
            _make_inputs,
            _make_router,
            _SyntheticModel,
        )

        model = _SyntheticModel([_make_router(ep_size=2, ep_rank=rank)])
        inputs = _make_inputs(experts=2)
        counts = inputs.expert_token_counts
        source_rows = [[2, 3], [2, 3]] if rank == 0 else [[0, 1], [-1, -1]]
        local_ids = torch.tensor(source_rows, dtype=torch.int64, device="cuda")
        destinations = [torch.empty_like(local_ids) for _ in range(2)]
        torch.distributed.all_gather(destinations, local_ids)
        model(torch.cat(destinations, dim=0), counts)
        expert_counts = [torch.empty_like(counts) for _ in range(2)]
        torch.distributed.all_gather(expert_counts, counts)
        local_max = counts.sum()
        local_sum = counts.sum()
        torch.distributed.all_reduce(local_max, op=torch.distributed.ReduceOp.MAX)
        torch.distributed.all_reduce(local_sum, op=torch.distributed.ReduceOp.SUM)
        queue.put(
            (
                rank,
                counts.cpu().tolist(),
                torch.cat(expert_counts, dim=1).cpu().tolist(),
                int(local_max.item()),
                int(local_sum.item()),
            )
        )
    finally:
        torch.distributed.destroy_process_group()


class DestinationRecorderEp2Test(unittest.TestCase):
    def test_destination_ownership_and_global_reduction(self):
        self.assertGreaterEqual(torch.cuda.device_count(), 2, "target requires 2 GPUs")
        import torch.multiprocessing as mp

        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            init_method = f"tcp://127.0.0.1:{sock.getsockname()[1]}"

        ctx = mp.get_context("spawn")
        queue = ctx.Queue()
        processes = [
            ctx.Process(target=_rank_main, args=(rank, init_method, queue))
            for rank in range(2)
        ]
        try:
            for process in processes:
                process.start()
            results = [queue.get(timeout=300) for _ in processes]
            for process in processes:
                process.join(timeout=300)
                self.assertEqual(process.exitcode, 0)
            for rank, local, global_counts, maximum, total in results:
                self.assertEqual(local, [[1, 1]] if rank == 0 else [[2, 2]])
                self.assertEqual(global_counts, [[1, 1, 2, 2]])
                self.assertEqual((maximum, total), (4, 6))
                self.assertAlmostEqual(maximum * 2 / total, 4 / 3)
        finally:
            for process in processes:
                if process.pid is not None:
                    if process.is_alive():
                        process.terminate()
                    process.join(timeout=10)
            queue.close()


if __name__ == "__main__":
    unittest.main()
