import importlib
import multiprocessing as mp
import tempfile
import time
import unittest
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist


def _assert_rounds(rounds, rank: int) -> None:
    assert len(rounds) == 3
    order = torch.tensor([0, 1] if rank == 0 else [1, 0])
    expected_tags = ["shared", "other"] if rank == 0 else ["other", "shared"]
    for step, result in enumerate(rounds):
        selected = 1 if step == 1 else 0
        assert result["selected_module"] == selected
        assert len(result["modules"]) == 2
        for module, cache in enumerate(result["modules"]):
            width = 2 if module == 0 else 3
            offset = step * 100 + module * 20
            physical = (torch.arange(2 * width, dtype=torch.int32) + offset).reshape(
                2, 1, width
            )
            kernel = (torch.arange(4 * width, dtype=torch.int32) + offset + 10).reshape(
                2, 1, 2 * width
            )
            physical = physical.index_select(0, order)
            kernel = kernel.index_select(0, order)
            assert cache["tags"] == expected_tags
            assert cache["kernel_tokens"] == (4 if module == 0 else 2)
            for name, expected in (("physical", physical), ("kernel", kernel)):
                actual = cache[name]
                assert actual.is_cuda and actual.device.index == rank
                torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
                if selected == module:
                    selected_tensor = result[f"selected_{name}"]
                    assert selected_tensor.device == actual.device
                    torch.testing.assert_close(
                        selected_tensor.cpu(), expected, rtol=0, atol=0
                    )
            types = torch.tensor([1, 0] if module == 0 else [0, 1], dtype=torch.int32)
            mapping = torch.tensor(
                [[0, 7 + offset, 9 + offset], [1, 11 + offset, 13 + offset]],
                dtype=torch.int32,
            )
            mapping[:, 0] = order.to(torch.int32)
            assert cache["types"].device.type == "cpu"
            assert cache["mapping"].device.type == "cpu"
            torch.testing.assert_close(
                cache["types"], types.index_select(0, order), rtol=0, atol=0
            )
            torch.testing.assert_close(cache["mapping"], mapping, rtol=0, atol=0)
        for name, expected in (("tokens", 7 + step), ("lengths", 4 + step)):
            assert result[name].device.type == "cpu"
            torch.testing.assert_close(
                result[name],
                torch.tensor([expected], dtype=torch.int32),
                rtol=0,
                atol=0,
            )


def _worker(rank: int, rendezvous: str, cpu_base: str) -> None:
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=Path(rendezvous).as_uri(),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=90),
    )
    extension = None
    gpu_broadcasts = 0
    try:
        # Load only the test extension: production ops register the same pybind
        # types and would own different C++ communication callback globals.
        extension = importlib.import_module(
            "rtp_llm.cpp.models.test.libth_pywrapped_model_cache_store_integration_test"
        )
        extension.init_cpu_tp_broadcaster(rank, 2, cpu_base)

        def broadcast(tensors, root: int, mode: int) -> None:
            nonlocal gpu_broadcasts
            assert root == 0 and mode == 0
            for tensor in tensors:
                # CPU headers and metadata must use the initialized UDS lane.
                assert tensor.is_cuda and tensor.device.index == rank
                dist.broadcast(tensor, src=root)
                gpu_broadcasts += 1

        def unexpected_collective(*_args):
            raise AssertionError("TP input sync should only broadcast")

        extension.register_comm_ops(
            broadcast, unexpected_collective, unexpected_collective
        )
        rounds = extension.run_domain_tp_round(rank)
        torch.cuda.synchronize()
        assert gpu_broadcasts >= 3
        _assert_rounds(rounds, rank)
        dist.barrier()
    finally:
        if extension is not None:
            extension.clear_comm_ops()
            extension.destroy_cpu_tp_broadcaster()
        dist.destroy_process_group()


class ModelDataGpuTpTest(unittest.TestCase):
    def test_module_tables_survive_nccl_sync_and_module_switches(self):
        self.assertTrue(torch.cuda.is_available(), "requires CUDA")
        self.assertGreaterEqual(torch.cuda.device_count(), 2, "requires two CUDA GPUs")
        context = mp.get_context("spawn")
        # Keep Unix-domain socket paths below sockaddr_un's length limit.
        with tempfile.TemporaryDirectory(
            prefix="model_gpu_tp.", dir="/tmp"
        ) as directory:
            processes = [
                context.Process(
                    target=_worker,
                    args=(rank, f"{directory}/nccl", f"{directory}/cpu"),
                )
                for rank in range(2)
            ]
            exitcodes = []
            try:
                for process in processes:
                    process.start()
                deadline = time.monotonic() + 150
                for process in processes:
                    process.join(timeout=max(0, deadline - time.monotonic()))
                exitcodes = [process.exitcode for process in processes]
            finally:
                for process in processes:
                    if process.is_alive():
                        process.terminate()
                for process in processes:
                    if process.pid is not None:
                        process.join(timeout=5)
                        if process.is_alive():
                            process.kill()
                            process.join(timeout=5)
            self.assertEqual(exitcodes, [0, 0])


if __name__ == "__main__":
    unittest.main()
