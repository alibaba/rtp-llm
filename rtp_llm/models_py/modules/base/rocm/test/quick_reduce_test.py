import socket
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rtp_llm.models_py.modules.base.rocm.quick_reduce import (
    QuickReduceRegime,
    RocmQuickReduce,
)


def _find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class RocmQuickReduceUnitTest(unittest.TestCase):
    def test_quantization_regime_values_match_native_kernel(self):
        self.assertEqual(QuickReduceRegime.FP.value, 0)
        self.assertEqual(QuickReduceRegime.INT8.value, 1)
        self.assertEqual(QuickReduceRegime.INT6.value, 2)
        self.assertEqual(QuickReduceRegime.INT4.value, 3)

    def test_should_reject_unsupported_dtype(self):
        manager = object.__new__(RocmQuickReduce)
        manager.disabled = False
        manager.qr_max_size = 1024 * 1024
        manager.qr_min_size = 0
        manager.qr_quant_level = QuickReduceRegime.FP
        manager.qr_quantization_min_size = None
        manager.use_fp16_kernels = False
        manager.world_size = 2

        tensor = torch.zeros((16,), dtype=torch.float32)
        self.assertFalse(manager.should_quick_allreduce(tensor))


def _quick_reduce_worker(rank, world_size, port, quantization, return_dict):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        world_size=world_size,
        rank=rank,
    )
    metadata_group = dist.new_group(ranks=list(range(world_size)), backend="gloo")
    manager = None
    try:
        from rtp_llm.models_py.modules.base.rocm.quick_reduce import RocmQuickReduce

        manager = RocmQuickReduce(
            metadata_group,
            device=rank,
            quantization=quantization,
            max_size_mb=16,
            min_size_mb=0,
        )
        torch.manual_seed(4321 + rank)
        inp = torch.randn((256, 2048), device=f"cuda:{rank}", dtype=torch.float16)
        ref = inp.clone()
        dist.all_reduce(ref, group=dist.group.WORLD)
        out = manager.quick_all_reduce(inp)
        torch.cuda.synchronize()
        tolerances = {
            "FP": (1e-2, 1e-2),
            "INT8": (8e-2, 8e-2),
            "INT6": (2e-1, 2e-1),
            "INT4": (1.0, 1.0),
        }
        rtol, atol = tolerances[quantization]
        torch.testing.assert_close(out, ref, rtol=rtol, atol=atol)
        return_dict[rank] = "ok"
    finally:
        if manager is not None:
            manager.close()
        dist.destroy_process_group(metadata_group)
        dist.destroy_process_group()


class RocmQuickReduceCorrectnessTest(unittest.TestCase):
    def _run_quantization_case(self, quantization):
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            self.skipTest("requires two ROCm GPUs")
        port = _find_free_port()
        ctx = mp.get_context("spawn")
        manager = ctx.Manager()
        return_dict = manager.dict()
        procs = [
            ctx.Process(
                target=_quick_reduce_worker,
                args=(rank, 2, port, quantization, return_dict),
            )
            for rank in range(2)
        ]
        for proc in procs:
            proc.start()
        for proc in procs:
            proc.join(timeout=120)
        for proc in procs:
            self.assertEqual(proc.exitcode, 0)
        self.assertEqual(dict(return_dict), {0: "ok", 1: "ok"})

    def test_quick_reduce_fp_matches_torch_allreduce(self):
        self._run_quantization_case("FP")

    def test_quick_reduce_int8_smoke(self):
        self._run_quantization_case("INT8")

    def test_quick_reduce_int6_smoke(self):
        self._run_quantization_case("INT6")

    def test_quick_reduce_int4_smoke(self):
        self._run_quantization_case("INT4")


if __name__ == "__main__":
    unittest.main()
