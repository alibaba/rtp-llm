"""Cross-rank native NVFP4 dispatch/combine and CUDA graph regression."""

import tempfile
import unittest
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rtp_llm.model_loader.test import test_native_mega_moe_nvfp4 as native_test


def _worker(rank, init_path):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method="file://" + init_path,
        rank=rank,
        world_size=4,
        timeout=timedelta(minutes=5),
    )
    try:
        case = native_test.NativeNvfp4KernelTest(
            "test_kernel_reference_and_changed_input_graph_replay"
        )
        case.test_kernel_reference_and_changed_input_graph_replay()
        dist.barrier()
    finally:
        dist.destroy_process_group()


class NativeNvfp4ExpertParallelTest(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.device_count() >= 4, "Four CUDA GPUs required")
    def test_ep4_reference_and_changed_input_graph_replay(self):
        if torch.cuda.get_device_capability()[0] != 10:
            self.skipTest("SM100/SM103 required")
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(_worker, args=(directory + "/init",), nprocs=4, join=True)


if __name__ == "__main__":
    unittest.main()
