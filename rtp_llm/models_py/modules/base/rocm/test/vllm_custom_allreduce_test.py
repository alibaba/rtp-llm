import socket
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rtp_llm.models_py.modules.base.rocm.vllm_custom_allreduce import (
    RocmVllmCustomAllReduce,
)


def _find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _custom_ar_worker(rank, world_size, port, return_dict):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        world_size=world_size,
        rank=rank,
    )
    metadata_group = dist.new_group(
        ranks=list(range(world_size)),
        backend="gloo",
    )
    manager = None
    try:
        from rtp_llm.models_py.modules.base.rocm.vllm_custom_allreduce import (
            RocmVllmCustomAllReduce,
        )

        manager = RocmVllmCustomAllReduce(metadata_group, device=rank)
        torch.manual_seed(1234 + rank)
        inp = torch.randn((3, 3584), device=f"cuda:{rank}", dtype=torch.float16)
        ref = inp.clone()
        dist.all_reduce(ref, group=dist.group.WORLD)
        out = manager.custom_all_reduce(inp)
        torch.cuda.synchronize()
        torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)
        return_dict[rank] = "ok"
    finally:
        if manager is not None:
            manager.close()
        dist.destroy_process_group(metadata_group)
        dist.destroy_process_group()


class RocmVllmCustomAllReduceUnitTest(unittest.TestCase):
    def test_should_reject_misaligned_tensor_size(self):
        manager = object.__new__(RocmVllmCustomAllReduce)
        manager.disabled = False
        manager.max_size = 1024
        manager.world_size = 2
        manager.fully_connected = True

        tensor = torch.zeros((3,), dtype=torch.float16)
        with patch(
            "rtp_llm.models_py.modules.base.rocm.vllm_custom_allreduce.is_weak_contiguous",
            return_value=True,
        ):
            self.assertFalse(manager.should_custom_ar(tensor))


class RocmVllmCustomAllReduceCorrectnessTest(unittest.TestCase):
    def test_custom_allreduce_matches_torch_allreduce(self):
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            self.skipTest("requires two ROCm GPUs")
        port = _find_free_port()
        ctx = mp.get_context("spawn")
        manager = ctx.Manager()
        return_dict = manager.dict()
        procs = [
            ctx.Process(target=_custom_ar_worker, args=(rank, 2, port, return_dict))
            for rank in range(2)
        ]
        for proc in procs:
            proc.start()
        for proc in procs:
            proc.join(timeout=120)
        for proc in procs:
            self.assertEqual(proc.exitcode, 0)
        self.assertEqual(dict(return_dict), {0: "ok", 1: "ok"})


if __name__ == "__main__":
    unittest.main()
