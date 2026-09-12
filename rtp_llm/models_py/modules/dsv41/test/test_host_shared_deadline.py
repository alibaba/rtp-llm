import fcntl
import hashlib
import multiprocessing
import os
import tempfile
import time
from pathlib import Path
from unittest import TestCase, main

import torch
from host_cuda_test_support import (
    cpu_lookup_reference,
    initialized_slices,
    load_host_modules,
)
from test_host_shared_lifecycle import _interrupted_loader, _slices


def _waiter(source, store, result):
    shared_module, _ = load_host_modules()
    started = time.monotonic()
    try:
        with shared_module.HostSharedWeightStore(store).open_or_publish(
            "a" * 40, _slices(source, shared_module), lock_timeout_seconds=0.15
        ):
            outcome = "opened"
    except Exception as error:
        outcome = type(error).__name__
    result.put((outcome, time.monotonic() - started))


def _hold_lock(path, ready, release):
    with Path(path).open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ready.set()
        if not release.wait(60):
            raise TimeoutError("lock holder was not released")


class HostSharedDeadlineTest(TestCase):
    def setUp(self):
        self.assertNotEqual(os.getuid(), 0)
        self.shared_module, self.cuda_module = load_host_modules()
        self.temporary = tempfile.TemporaryDirectory(
            dir=os.environ.get("DSV41_TEST_SHARED_ROOT")
        )
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        self.slices = initialized_slices(self.source, self.shared_module)
        self.store = self.shared_module.HostSharedWeightStore(self.root / "shared")
        self.key = self.store.identity("a" * 40, self.slices)
        self.context = multiprocessing.get_context("spawn")

    def tearDown(self):
        self.cuda_module.shutdown_shared_engram()
        self.temporary.cleanup()

    def _join(self, process):
        process.join(15)
        if process.is_alive():
            process.kill()
            process.join(10)
        self.assertFalse(process.is_alive())

    def _timeout(self):
        result = self.context.Queue()
        process = self.context.Process(
            target=_waiter, args=(str(self.source), str(self.store.root), result)
        )
        process.start()
        try:
            outcome, elapsed = result.get(timeout=15)
            self.assertEqual(outcome, "TimeoutError")
            self.assertGreaterEqual(elapsed, 0.15)
            self.assertLess(elapsed, 5)
        finally:
            self._join(process)
        self.assertEqual(process.exitcode, 0)

    def test_live_publisher_timeout_does_not_remove_staging(self):
        ready, release = self.context.Event(), self.context.Event()
        publisher = self.context.Process(
            target=_interrupted_loader,
            args=(str(self.source), str(self.store.root), ready, release),
        )
        publisher.start()
        try:
            self.assertTrue(ready.wait(30))
            staging = list(self.store.root.glob(self.key + ".loading-*"))
            self.assertEqual(len(staging), 1)
            partial = staging[0] / "0.bin"
            before = (partial.stat().st_ino, partial.read_bytes())
            lock_inode = (self.store.root / (self.key + ".lock")).stat().st_ino
            self._timeout()
            self.assertTrue(publisher.is_alive())
            self.assertFalse(self.store.remove_if_unused(self.key))
            self.assertEqual(before, (partial.stat().st_ino, partial.read_bytes()))
            self.assertEqual(
                lock_inode, (self.store.root / (self.key + ".lock")).stat().st_ino
            )
            self.assertFalse((self.store.root / self.key / "READY.json").exists())
        finally:
            release.set()
            self._join(publisher)
        self.assertEqual(publisher.exitcode, 0)
        with self.store.open_or_publish("a" * 40, self.slices) as shared:
            self.assertEqual((shared.directory / "0.bin").stat().st_ino, before[0])
            self.assertFalse(list(self.store.root.glob(self.key + ".loading-*")))
            for name, record in shared.manifest["tensors"].items():
                with shared.view(name) as value:
                    self.assertEqual(
                        hashlib.sha256(value).hexdigest(), record["sha256"]
                    )

    def test_timeout_preserves_live_registered_graph_and_ready(self):
        torch.cuda.set_device(0)
        shared = self.store.open_or_publish("a" * 40, self.slices)
        ready_path = shared.directory / "READY.json"
        original = (ready_path.stat().st_ino, ready_path.read_bytes())
        with self.cuda_module.SharedEngramLookup(shared, device=0) as lookup:
            lookup.warmup()
            ids = torch.tensor([0, 512, 1023], device=0, dtype=torch.int64)
            output = torch.empty((3, 256), device=0, dtype=torch.bfloat16)
            graph = lookup.graph()
            stream = torch.cuda.Stream(device=0)
            torch.cuda.synchronize()
            with graph.capture(stream):
                lookup.lookup(1, ids, out=output)
            ready, release = self.context.Event(), self.context.Event()
            holder = self.context.Process(
                target=_hold_lock,
                args=(str(self.store.root / (self.key + ".lock")), ready, release),
            )
            holder.start()
            try:
                self.assertTrue(ready.wait(30))
                self._timeout()
                self.assertTrue(holder.is_alive())
                self.assertFalse(self.store.remove_if_unused(self.key))
                self.assertEqual(
                    original, (ready_path.stat().st_ino, ready_path.read_bytes())
                )
                graph.replay()
                torch.cuda.synchronize()
                torch.testing.assert_close(
                    output.cpu(),
                    cpu_lookup_reference(shared, 1, ids),
                    rtol=0,
                    atol=0,
                    equal_nan=True,
                )
            finally:
                release.set()
                self._join(holder)
            self.assertEqual(holder.exitcode, 0)
            ids.copy_(torch.tensor([1, 2, 3], device=0))
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(
                output.cpu(),
                cpu_lookup_reference(shared, 1, ids),
                rtol=0,
                atol=0,
                equal_nan=True,
            )
        self.assertTrue(self.store.remove_if_unused(self.key))

    def test_invalid_timeout_creates_no_publication(self):
        for timeout in (True, None, "1", 0, -1, float("nan"), float("inf")):
            with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                self.store.open_or_publish(
                    "a" * 40, self.slices, lock_timeout_seconds=timeout
                )
        self.assertEqual(list(self.store.root.iterdir()), [])


if __name__ == "__main__":
    main()
