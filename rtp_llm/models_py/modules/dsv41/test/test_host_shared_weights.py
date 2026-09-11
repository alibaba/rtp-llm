import hashlib
import multiprocessing
import tempfile
import unittest
from pathlib import Path

from standalone_load import load_component

_shared = load_component(
    "rtp_host_shared_weights_test", "model_loader/host_shared_weights.py"
)
HostSharedWeightStore = _shared.HostSharedWeightStore
SharedWeightSlice = _shared.SharedWeightSlice


def _consumer(source, root, ready, release):
    store = HostSharedWeightStore(root)
    slices = [SharedWeightSlice("table", Path(source), 0, 128, (16, 8), "U8")]
    with store.open_or_publish("d" * 40, slices, chunk_bytes=13) as shared:
        view = shared.view("table")
        record = shared.manifest["tensors"]["table"]
        ready.put(
            (
                shared.manifest["identity"],
                (shared.directory / record["file"]).stat().st_ino,
                hashlib.sha256(view).hexdigest(),
            )
        )
        try:
            if not release.wait(20):
                raise TimeoutError("shared-weight test consumer lease was not released")
        finally:
            view.release()


class HostSharedWeightTest(unittest.TestCase):
    def test_spawned_consumers_share_one_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            source.write_bytes(bytes(range(128)))
            root = Path(directory) / "shared"
            context = multiprocessing.get_context("spawn")
            ready, release = context.Queue(), context.Event()
            workers = [
                context.Process(
                    target=_consumer, args=(str(source), str(root), ready, release)
                )
                for _ in range(2)
            ]
            try:
                for worker in workers:
                    worker.start()
                first, second = ready.get(timeout=15), ready.get(timeout=15)
                self.assertEqual(first, second)
                self.assertEqual(
                    first[2], hashlib.sha256(bytes(range(128))).hexdigest()
                )
                self.assertFalse(HostSharedWeightStore(root).remove_if_unused(first[0]))
            finally:
                release.set()
                for worker in workers:
                    worker.join(timeout=15)
                    if worker.is_alive():
                        worker.terminate()
                        worker.join()
            self.assertTrue(all(worker.exitcode == 0 for worker in workers))
            self.assertTrue(HostSharedWeightStore(root).remove_if_unused(first[0]))

    def test_publication_shared_backing_and_live_lease(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.bin"
            source.write_bytes(b"prefix" + bytes(range(128)) + b"suffix")
            slices = [
                SharedWeightSlice("L1.weight", source, 6, 128, (16, 8), "F8_E4M3")
            ]
            store = HostSharedWeightStore(root / "shared")
            first = store.open_or_publish("a" * 40, slices, chunk_bytes=7)
            second = store.open_or_publish("a" * 40, slices)
            self.assertEqual(first.directory, second.directory)
            view = first.view("L1.weight")
            self.assertTrue(view.readonly)
            self.assertEqual(bytes(view), bytes(range(128)))
            self.assertEqual(
                first.manifest["tensors"]["L1.weight"]["sha256"],
                hashlib.sha256(bytes(range(128))).hexdigest(),
            )
            self.assertFalse(store.remove_if_unused(first.manifest["identity"]))
            with self.assertRaises(BufferError):
                first.close()
            view.release()
            first.close()
            self.assertFalse(store.remove_if_unused(second.manifest["identity"]))
            second.close()
            self.assertTrue(store.remove_if_unused(second.manifest["identity"]))

    def test_revisions_and_partial_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "data"
            source.write_bytes(b"1234")
            store = HostSharedWeightStore(Path(directory) / "shared")
            good = [SharedWeightSlice("table", source, 0, 4, (4,), "U8")]
            with store.open_or_publish("a" * 40, good) as old, store.open_or_publish(
                "b" * 40, good
            ) as new:
                self.assertNotEqual(old.directory, new.directory)
            bad = [SharedWeightSlice("table", source, 0, 5, (5,), "U8")]
            with self.assertRaises(ValueError):
                store.open_or_publish("c" * 40, bad)
            self.assertFalse(
                any(
                    path.name.startswith(store.identity("c" * 40, bad))
                    for path in store.root.glob("*/READY.json")
                )
            )


if __name__ == "__main__":
    unittest.main()
