import multiprocessing as mp
import os
import socket
import time
import traceback
import unittest
from datetime import timedelta
from unittest.mock import patch

import torch
import torch.distributed as dist


class FixedPlanBalancer:
    def __init__(self, stage):
        self.calls = 0
        self.stats = None
        # Distinct maps make an incorrect WORLD broadcast visible on stage 1.
        self.plan = (
            torch.tensor([stage * 2], dtype=torch.int32),
            torch.tensor([2, 2], dtype=torch.int32),
            torch.tensor([[0, 2, -1], [1, 3, -1]], dtype=torch.int32).roll(stage, 0),
            torch.tensor([0, 1, 0, 1], dtype=torch.int32).roll(stage, 0),
        )

    def create_balance_plan(self, log_stats, gpu_loads):
        self.calls += 1
        self.stats = (log_stats.clone(), gpu_loads.clone())
        return self.plan


def run_rank(rank, pp_size, port):
    from rtp_llm.cpp.models.eplb.test import libth_eplb_stage_test as test_ops
    from rtp_llm.models_py.distributed import collective_torch as ct

    torch.cuda.set_device(rank)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    world_size = pp_size * 2
    dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=timedelta(seconds=60))
    try:
        config = ct.ParallelismConfig()
        config.world_size, config.world_rank = world_size, rank
        config.local_world_size, config.local_rank = world_size, rank
        config.pp_size, config.dp_size, config.tp_size = pp_size, 2, 1
        config.ep_size, config.ep_rank = 2, rank % 2
        config.pp_stage_layer_counts = [4 // pp_size] * pp_size
        ct._normalize_parallelism_ranks(config)
        ct._parallelism_config = config
        ct._initialized = True
        ct._group_map.clear()
        ct._group_map[ct.Group.WORLD] = dist.group.WORLD
        ct._create_process_groups(config, "nccl", None)
        # This extension links its own ExecOps state; register the production
        # callbacks there so the balancer uses the real NCCL process groups.
        with patch.dict("sys.modules", {"librtp_compute_ops": test_ops}):
            ct._register_process_groups_to_cpp()

        stage = rank // 2
        begin, end = stage * (4 // pp_size), (stage + 1) * (4 // pp_size)
        log_stats = torch.zeros((4, 2), dtype=torch.int32, device="cuda")
        gpu_loads = torch.zeros_like(log_stats)
        log_stats[begin:end] = rank + 1
        gpu_loads[begin:end, rank % 2] = rank + 1
        python_balancer = FixedPlanBalancer(stage)
        result = test_ops.exercise_stage(config, python_balancer, log_stats, gpu_loads)

        assert result["mode"] == (2 if stage == 0 else 3), result
        assert result["update_time"] == 100 + stage, result
        assert result["fake_step_count"] == 1 and result["real_step_count"] == 2, result
        torch.testing.assert_close(result["fake_step_stats"], torch.zeros((4, 2), dtype=torch.int32))
        torch.testing.assert_close(result["fake_step_loads"], torch.zeros((4, 2), dtype=torch.int32))
        expected_stats = torch.zeros((4, 2), dtype=torch.int32)
        expected_stats[begin:end] = 3 + 4 * stage
        expected_loads = torch.zeros_like(expected_stats)
        expected_loads[begin:end] = torch.tensor([2 * stage + 1, 2 * stage + 2])
        torch.testing.assert_close(result["log_stats"], expected_stats)
        torch.testing.assert_close(result["gpu_loads"], expected_loads)
        assert python_balancer.calls == (1 if rank % 2 == 0 else 0)
        if rank % 2 == 0:
            torch.testing.assert_close(python_balancer.stats[0], expected_stats)
            torch.testing.assert_close(python_balancer.stats[1], expected_loads)
        assert result["layer_id"] == stage * 2
        for field, expected in zip(("logic_expert_cnt", "log2phy", "phy2log"), python_balancer.plan[1:]):
            torch.testing.assert_close(result[field], expected)
        assert result["partially_ready"] == (stage == 0), result
        assert result["all_ready"], result
        assert result["metrics_layer_begin"] == begin, result
        assert result["metrics_gpu_loads"] == [rank + 1] * (end - begin), result
    finally:
        test_ops.clear_comm_ops()
        test_ops.clear_pp_ops()
        dist.destroy_process_group()


def worker(rank, pp_size, port, queue):
    try:
        run_rank(rank, pp_size, port)
        queue.put((rank, None))
    except Exception:
        queue.put((rank, traceback.format_exc()))
        raise


class EplbStageTest(unittest.TestCase):
    def run_topology(self, pp_size):
        world_size = pp_size * 2
        self.assertGreaterEqual(torch.cuda.device_count(), world_size, "run with GPU_COUNT=4 and gpu_lock")
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        ctx = mp.get_context("spawn")
        queue = ctx.Queue()
        processes = [ctx.Process(target=worker, args=(rank, pp_size, port, queue)) for rank in range(world_size)]
        for process in processes:
            process.start()
        try:
            deadline = time.monotonic() + 180
            for _ in processes:
                rank, error = queue.get(timeout=max(1, deadline - time.monotonic()))
                self.assertIsNone(error, f"rank {rank}: {error}")
            for process in processes:
                process.join(timeout=max(1, deadline - time.monotonic()))
                self.assertEqual(process.exitcode, 0)
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
                process.join(timeout=5)
            queue.close()

    def test_pp2_ep2_stage_isolation(self):
        self.run_topology(pp_size=2)

    def test_pp1_ep2_regression(self):
        self.run_topology(pp_size=1)


if __name__ == "__main__":
    unittest.main()
