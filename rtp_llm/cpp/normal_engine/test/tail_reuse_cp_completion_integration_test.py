import multiprocessing as mp
import os
import time
import unittest

import torch
import torch.distributed as dist

from rtp_llm.test.utils.port_util import PortManager


def _worker(world_rank: int, world_size: int, port: int, results: mp.Queue) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(world_rank)
    dist.init_process_group("nccl", rank=world_rank, world_size=world_size)
    from rtp_llm.cpp.normal_engine.test import (  # noqa: PLC0415
        libth_tail_reuse_cp_completion_integration_test as extension,
    )

    if world_size == 2:
        tp_group = dist.group.WORLD
    else:
        # Every world rank creates both groups in the same order; collectives
        # below involve only the two members of this rank's TP group.
        groups = [
            dist.new_group(ranks=[0, 1], backend="nccl"),
            dist.new_group(ranks=[2, 3], backend="nccl"),
        ]
        tp_group = groups[world_rank // 2]
        assert tp_group is not dist.group.WORLD
    tp_rank = world_rank % 2
    assert dist.get_rank(group=tp_group) == tp_rank
    assert dist.get_world_size(group=tp_group) == 2
    collective_counts = {"broadcast": 0, "allreduce": 0}

    def broadcast(tensors, root, mode):
        assert mode == 0
        assert root == 0
        global_root = dist.get_global_rank(tp_group, root)
        assert global_root == world_rank - tp_rank
        collective_counts["broadcast"] += 1
        for tensor in tensors:
            if tensor.is_cuda:
                dist.broadcast(tensor, src=global_root, group=tp_group)
            else:
                gpu = tensor.cuda()
                dist.broadcast(gpu, src=global_root, group=tp_group)
                tensor.copy_(gpu.cpu())

    def allreduce(tensor, op, mode, dest):
        assert mode == 0
        assert dest is None
        assert op == 0
        collective_counts["allreduce"] += 1
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=tp_group)
        return tensor

    def unexpected_allgather(*_args):
        raise AssertionError("CP completion fixture unexpectedly issued allgather")

    def writer_barrier():
        # Called by both C++ model forwards after their local copy check.
        # Processes have independent GILs, so this NCCL barrier can wait for
        # both model invocations without a Python observer thread.
        dist.barrier(group=tp_group)

    try:
        extension.init_exec_ctx(world_rank, False, False, 0)
        extension.register_comm_ops(broadcast, allreduce, unexpected_allgather)
        dist.barrier(group=tp_group)
        result = extension.run_cp_completion(
            tp_rank, world_rank, world_size, writer_barrier
        )
        result["tp_group_global_root"] = dist.get_global_rank(tp_group, 0)
        result["collective_counts"] = collective_counts
        results.put(dict(result))
    finally:
        extension.clear_comm_ops()
        dist.destroy_process_group()


class TailReuseCpCompletionIntegrationTest(unittest.TestCase):
    def _run_case(self, world_size: int) -> dict:
        if not torch.cuda.is_available() or torch.cuda.device_count() < world_size:
            self.skipTest(f"requires {world_size} CUDA GPUs")

        manager = PortManager()
        ports, locks = manager.get_consecutive_ports(1)
        context = mp.get_context("spawn")
        results = context.Queue()
        processes = [
            context.Process(target=_worker, args=(rank, world_size, ports[0], results))
            for rank in range(world_size)
        ]
        try:
            for process in processes:
                process.start()
            deadline = time.monotonic() + 120
            for process in processes:
                process.join(timeout=max(0, deadline - time.monotonic()))
            exitcodes = [process.exitcode for process in processes]
            self.assertEqual(exitcodes, [0] * world_size)
            reports = {
                report["world_rank"]: report
                for report in (results.get(timeout=5) for _ in processes)
            }
            self.assertEqual(set(reports), set(range(world_size)))
            return reports
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
            for process in processes:
                process.join(timeout=5)
            for process in processes:
                if process.is_alive():
                    process.kill()
                    process.join(timeout=5)
            for lock in locks:
                lock.__exit__(None, None, None)

    def _check_tp_pair(self, reports: dict, global_root: int) -> None:
        owner = reports[global_root]
        worker = reports[global_root + 1]
        self.assertEqual([owner["rank"], worker["rank"]], [0, 1])
        for report in (owner, worker):
            self.assertEqual(report["tp_group_global_root"], global_root)
            self.assertGreater(report["collective_counts"]["broadcast"], 0)
            self.assertGreater(report["collective_counts"]["allreduce"], 0)
        self.assertTrue(owner["copy_rows"])
        self.assertEqual(owner["copy_rows"], worker["copy_rows"])
        self.assertTrue(owner["content_complete"])
        self.assertTrue(owner["source_pins_empty"])
        self.assertTrue(owner["pending_during_writer"])
        self.assertTrue(owner["tail_complete"])
        self.assertTrue(owner["tail_published"])
        self.assertGreaterEqual(owner["elapsed_ms"], 500)

    @unittest.skipIf(
        os.environ.get("TAIL_REUSE_TP_GROUP_COUNT") == "2",
        "covered by two-subgroup target",
    )
    def test_cp2_copy_writer_and_owner_publication_order(self) -> None:
        reports = self._run_case(2)
        self._check_tp_pair(reports, 0)

    @unittest.skipUnless(
        os.environ.get("TAIL_REUSE_TP_GROUP_COUNT") == "2", "requires four-GPU target"
    )
    def test_two_independent_tp2_groups_with_non_world_root(self) -> None:
        reports = self._run_case(4)
        self._check_tp_pair(reports, 0)
        self._check_tp_pair(reports, 2)

    @unittest.skipIf(
        os.environ.get("TAIL_REUSE_TP_GROUP_COUNT") == "2", "covered by two-GPU target"
    )
    def test_tp2_missing_group_rejected(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("requires CUDA")
        from rtp_llm.cpp.normal_engine.test import (  # noqa: PLC0415
            libth_tail_reuse_cp_completion_integration_test as extension,
        )

        torch.cuda.set_device(0)
        extension.init_exec_ctx(0, False, False, 0)
        extension.clear_comm_ops()
        with self.assertRaisesRegex(
            RuntimeError,
            "broadcast callback not registered|allreduce callback not registered",
        ):
            extension.confirm_tp_cache_work_completion(2, 0)


if __name__ == "__main__":
    unittest.main()
