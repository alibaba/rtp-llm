import hashlib
import multiprocessing
import os
import tempfile
from pathlib import Path
from unittest import TestCase, main
from unittest.mock import patch

import torch
from host_cuda_test_support import (
    cpu_lookup_reference,
    initialized_slices,
    load_host_modules,
)


def _slices(root, shared_module):
    return [
        shared_module.SharedWeightSlice(
            f"layers.{layer}.engram.embed.{kind}",
            Path(root) / f"layers.{layer}.engram.embed.{kind}",
            0,
            1024 * dim,
            (1024, dim),
            dtype,
        )
        for layer in (1, 14)
        for kind, dim, dtype in (("weight", 256, "F8_E4M3"), ("scale", 8, "F8_E8M0"))
    ]


def _interrupted_loader(source, store, written, release):
    shared_module, _ = load_host_modules()
    original_open = Path.open

    class PausingWriter:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            self.stream.__enter__()
            return self

        def __exit__(self, *arguments):
            return self.stream.__exit__(*arguments)

        def __getattr__(self, name):
            return getattr(self.stream, name)

        def write(self, data):
            result = self.stream.write(data)
            self.stream.flush()
            written.set()
            if not release.wait(60):
                raise TimeoutError("loader fault injection was not released")
            return result

    def open_with_pause(path, mode="r", *args, **kwargs):
        stream = original_open(path, mode, *args, **kwargs)
        if mode == "xb" and ".loading-" in str(path):
            return PausingWriter(stream)
        return stream

    with patch.object(Path, "open", open_with_pause):
        shared_module.HostSharedWeightStore(store).open_or_publish(
            "a" * 40, _slices(source, shared_module), chunk_bytes=256
        )


def _lookup_owner(source, store, ready, release, abrupt):
    shared_module, cuda_module = load_host_modules()
    torch.cuda.set_device(0)
    shared = shared_module.HostSharedWeightStore(store).open_or_publish(
        "a" * 40, _slices(source, shared_module)
    )
    with cuda_module.SharedEngramLookup(shared, device=0) as lookup:
        lookup.warmup()
        ready.put(lookup.accounting())
        if not release.wait(60):
            raise TimeoutError("owner lifetime test was not released")
        if abrupt:
            os._exit(0)


class HostSharedLifecycleTest(TestCase):
    def setUp(self):
        self.assertNotEqual(os.geteuid(), 0)
        self.assertGreaterEqual(torch.cuda.device_count(), 2)
        self.shared_module, self.cuda_module = load_host_modules()
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.slices = initialized_slices(self.root / "source", self.shared_module)
        self.store = self.shared_module.HostSharedWeightStore(self.root / "shared")
        self.context = multiprocessing.get_context("spawn")

    def tearDown(self):
        self.cuda_module.shutdown_shared_engram()
        self.temporary.cleanup()

    def _join(self, process):
        process.join(timeout=20)
        if process.is_alive():
            process.kill()
            process.join(timeout=10)
        self.assertFalse(process.is_alive())

    def test_loader_death_never_publishes_partial_backing(self):
        written, release = self.context.Event(), self.context.Event()
        loader = self.context.Process(
            target=_interrupted_loader,
            args=(str(self.root / "source"), str(self.store.root), written, release),
        )
        loader.start()
        try:
            self.assertTrue(written.wait(30))
            self.assertFalse(list(self.store.root.glob("*/READY.json")))
            self.assertTrue(list(self.store.root.glob("*.loading-*/*.bin")))
        finally:
            # A killed waiter can leave multiprocessing.Event's semaphore locked.
            # Do not signal an object owned by the deliberately killed process.
            if loader.is_alive():
                loader.kill()
            self._join(loader)
        self.assertNotEqual(loader.exitcode, 0)
        with self.store.open_or_publish("a" * 40, self.slices) as shared:
            self.assertFalse(list(self.store.root.glob("*.loading-*")))
            self.assertEqual(len(shared.manifest["tensors"]), 4)
            for name, item in shared.manifest["tensors"].items():
                with shared.view(name) as view:
                    self.assertEqual(hashlib.sha256(view).hexdigest(), item["sha256"])
            with self.cuda_module.SharedEngramLookup(shared, device=0) as lookup:
                lookup.warmup()
                ids = torch.tensor([0, 512, 1023], device=0, dtype=torch.int64)
                for layer in (1, 14):
                    torch.testing.assert_close(
                        lookup.lookup(layer, ids).cpu(),
                        cpu_lookup_reference(shared, layer, ids),
                        rtol=0,
                        atol=0,
                        equal_nan=True,
                    )

    def test_owner_exit_restart_revision_and_live_graph(self):
        ready, release = self.context.Queue(), self.context.Event()
        owner = self.context.Process(
            target=_lookup_owner,
            args=(
                str(self.root / "source"),
                str(self.store.root),
                ready,
                release,
                True,
            ),
        )
        owner.start()
        try:
            original = ready.get(timeout=30)
            shared = self.store.open_or_publish("a" * 40, self.slices)
            with self.cuda_module.SharedEngramLookup(shared, device=1) as lookup:
                lookup.warmup()
                ids = torch.tensor([0, 512, 1023], device=1, dtype=torch.int64)
                out = torch.empty((3, 256), device=1, dtype=torch.bfloat16)
                graph = lookup.graph()
                stream = torch.cuda.Stream(device=1)
                torch.cuda.synchronize(1)
                with graph.capture(stream):
                    lookup.lookup(1, ids, out=out)
                release.set()
                self._join(owner)
                self.assertEqual(owner.exitcode, 0)
                self.assertFalse(
                    self.store.remove_if_unused(original["backing_identity"])
                )
                graph.replay()
                torch.cuda.synchronize(1)
                torch.testing.assert_close(
                    out.cpu(),
                    cpu_lookup_reference(shared, 1, ids),
                    rtol=0,
                    atol=0,
                    equal_nan=True,
                )
                restarted = self.context.Process(
                    target=_lookup_owner,
                    args=(
                        str(self.root / "source"),
                        str(self.store.root),
                        ready,
                        release,
                        False,
                    ),
                )
                restarted.start()
                try:
                    current = ready.get(timeout=30)
                    self.assertEqual(
                        original["backing_identity"], current["backing_identity"]
                    )
                    for name in original["tensors"]:
                        self.assertEqual(
                            original["tensors"][name], current["tensors"][name]
                        )
                finally:
                    self._join(restarted)
                self.assertEqual(restarted.exitcode, 0)
                new = self.store.open_or_publish("b" * 40, self.slices)
                with self.cuda_module.SharedEngramLookup(new, device=1) as replacement:
                    self.assertNotEqual(shared.directory, new.directory)
                    replacement.warmup()
                    torch.testing.assert_close(
                        replacement.lookup(1, ids).cpu(),
                        out.cpu(),
                        rtol=0,
                        atol=0,
                        equal_nan=True,
                    )
                graph.replay()
                torch.cuda.synchronize(1)
                torch.testing.assert_close(
                    out.cpu(),
                    cpu_lookup_reference(shared, 1, ids),
                    rtol=0,
                    atol=0,
                    equal_nan=True,
                )
            self.assertTrue(self.store.remove_if_unused(original["backing_identity"]))
        finally:
            release.set()
            self._join(owner)


if __name__ == "__main__":
    main()
