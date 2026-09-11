import hashlib
import multiprocessing
import os
import tempfile
from pathlib import Path
from unittest import TestCase, main

import torch
from host_cuda_test_support import (
    cpu_lookup_reference,
    initialized_slices,
    load_host_modules,
)


def _gpu_consumer(source_root, store_root, device, ready, release, mapped):
    shared_module, cuda_module = load_host_modules()
    torch.cuda.set_device(device)
    slices = []
    for layer in (1, 14):
        for kind, dim, dtype in (("weight", 256, "F8_E4M3"), ("scale", 8, "F8_E8M0")):
            name = f"layers.{layer}.engram.embed.{kind}"
            path = Path(source_root) / name
            slices.append(
                shared_module.SharedWeightSlice(
                    name, path, 0, 1024 * dim, (1024, dim), dtype
                )
            )
    shared = shared_module.HostSharedWeightStore(store_root).open_or_publish(
        "a" * 40, slices
    )
    with cuda_module.SharedEngramLookup(shared, device=device) as lookup:
        lookup.warmup()
        ids = torch.tensor(
            [device + 1, 512 + device, 1023 - device], device=device, dtype=torch.int64
        )
        for layer in (1, 14):
            actual = lookup.lookup(layer, ids)
            expected = cpu_lookup_reference(shared, layer, ids)
            torch.testing.assert_close(
                actual.cpu(), expected, rtol=0, atol=0, equal_nan=True
            )
        mapped.wait(timeout=180)
        ready.put(lookup.accounting())
        if not release.wait(180):
            raise TimeoutError("multi-GPU Engram test consumer was not released")


class HostSharedCudaTest(TestCase):
    @classmethod
    def setUpClass(cls):
        if os.geteuid() == 0:
            raise RuntimeError("run Engram GPU tests as a non-root dev-container user")
        cls.devices = [
            int(value)
            for value in os.environ.get("DSV41_TEST_GPU_IDS", "0,1").split(",")
        ]
        if len(set(cls.devices)) < 2 or any(
            device >= torch.cuda.device_count() for device in cls.devices
        ):
            raise RuntimeError(
                "two distinct development GPUs are required for the shared mapping test"
            )
        cls.shared_module, cls.cuda_module = load_host_modules()
        for device in cls.devices:
            supported, reason, _ = cls.cuda_module.is_supported(device)
            if not supported:
                raise RuntimeError(reason)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(
            dir=os.environ.get("DSV41_TEST_SHARED_ROOT")
        )
        self.root = Path(self.temporary.name)
        self.slices = initialized_slices(self.root / "source", self.shared_module)
        self.store = self.shared_module.HostSharedWeightStore(self.root / "shared")

    def tearDown(self):
        self.cuda_module.shutdown_shared_engram()
        self.temporary.cleanup()

    def test_multi_process_gpu_queries_use_same_backing(self):
        context = multiprocessing.get_context("spawn")
        ready, release = context.Queue(), context.Event()
        mapped = context.Barrier(2)
        workers = [
            context.Process(
                target=_gpu_consumer,
                args=(
                    str(self.root / "source"),
                    str(self.root / "shared"),
                    device,
                    ready,
                    release,
                    mapped,
                ),
            )
            for device in self.devices[:2]
        ]
        reports = []
        try:
            for worker in workers:
                worker.start()
            reports = [ready.get(timeout=180) for _ in workers]
            self.assertEqual(
                reports[0]["backing_identity"], reports[1]["backing_identity"]
            )
            self.assertNotEqual(reports[0]["gpu_uuid"], reports[1]["gpu_uuid"])
            self.assertNotIn("unavailable", [report["gpu_uuid"] for report in reports])
            for name in reports[0]["tensors"]:
                first, second = reports[0]["tensors"][name], reports[1]["tensors"][name]
                self.assertEqual(
                    (first["device"], first["inode"], first["sha256"]),
                    (second["device"], second["inode"], second["sha256"]),
                )
                self.assertEqual(
                    hashlib.sha256(Path(first["path"]).read_bytes()).hexdigest(),
                    first["sha256"],
                )
            for report in reports:
                self.assertTrue(report["query_api_read_only"])
                self.assertTrue(
                    all(
                        permission.endswith("s")
                        for permission in report["mapping_permissions"]
                    )
                )
            self.assertFalse(
                self.store.remove_if_unused(reports[0]["backing_identity"])
            )
        finally:
            release.set()
            for worker in workers:
                worker.join(timeout=30)
                if worker.is_alive():
                    worker.terminate()
                    worker.join(timeout=10)
                    if worker.is_alive():
                        worker.kill()
                        worker.join()
        self.assertTrue(all(worker.exitcode == 0 for worker in workers))
        self.assertTrue(self.store.remove_if_unused(reports[0]["backing_identity"]))

    def test_codec_empty_padding_streams_and_graph_lifetime(self):
        device = self.devices[0]
        shared = self.store.open_or_publish("b" * 40, self.slices)
        with self.cuda_module.SharedEngramLookup(shared, device=device) as lookup:
            with shared.view("layers.1.engram.embed.weight") as view:
                self.assertTrue(view.readonly)
                with self.assertRaises(TypeError):
                    view[0] = 0
            stream = torch.cuda.Stream(device=device)
            lookup.warmup(stream)
            with torch.cuda.device(device), torch.cuda.stream(stream):
                ids = torch.tensor(
                    [[0, 511, 1023], [-1, 12, -1]], device=device, dtype=torch.int64
                )
                valid = ids >= 0
                output = lookup.lookup(1, ids, valid_mask=valid)
            stream.synchronize()
            torch.testing.assert_close(
                output.cpu(),
                cpu_lookup_reference(shared, 1, ids, valid),
                rtol=0,
                atol=0,
                equal_nan=True,
            )
            empty = torch.empty((0, 24), device=device, dtype=torch.int64)
            self.assertEqual(lookup.lookup(14, empty).shape, (0, 24, 256))
            static_ids = torch.tensor(
                [[1, 2, 3], [4, 5, 6]], device=device, dtype=torch.int64
            )
            static_out = torch.empty((2, 3, 256), device=device, dtype=torch.bfloat16)
            graph = lookup.graph()
            torch.cuda.synchronize(device)
            with graph.capture(stream):
                lookup.lookup(14, static_ids, out=static_out)
            static_ids.copy_(
                torch.tensor([[6, 5, 4], [3, 2, 1]], device=device, dtype=torch.int64)
            )
            graph.replay()
            torch.cuda.synchronize(device)
            torch.testing.assert_close(
                static_out.cpu(),
                cpu_lookup_reference(shared, 14, static_ids),
                rtol=0,
                atol=0,
                equal_nan=True,
            )
            self.assertFalse(self.store.remove_if_unused(shared.manifest["identity"]))
            lookup.close()
            with self.assertRaisesRegex(RuntimeError, "closed"):
                graph.replay()
        self.assertTrue(self.store.remove_if_unused(shared.manifest["identity"]))


if __name__ == "__main__":
    main()
