# SPDX-License-Identifier: Apache-2.0
"""CPU-only storage orchestration tests with a resource-accounting native fake."""
import gc
from contextlib import nullcontext
from unittest.mock import patch
import importlib
import json
from pathlib import Path
import sys
import threading
import types
import unittest
from uuid import UUID
import weakref
from rtp_llm.models_py.modules.dsv41 import engram_bootstrap as b

PACKAGE = "engram_storage_cpu_test"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(Path(__file__).resolve().parents[1])]
sys.modules[PACKAGE] = package
m = importlib.import_module(PACKAGE + ".engram_manifest")
s = importlib.import_module(PACKAGE + ".engram_storage")
a = importlib.import_module(PACKAGE + ".engram_model_attach")


def uid(n):
    return str(UUID(int=n))


def fixture():
    document = dict(schema_version=1, model_revision="fixture-v1", generation=uid(1),
                    layout_abi=m.LAYOUT_ABI, hash_abi="sha256:" + "a" * 64,
                    backend="fabric_egm", state="READY", tensors=[])
    for layer in (1, 2):
        for role, width in (("weight", 32), ("scale", 1)):
            document["tensors"].append(dict(
                name=f"layers.{layer}.engram.embed.{role}", shape=[3, width],
                dtype="float8_e4m3fn" if role == "weight" else "float8_e8m0fnu",
                storage_dtype="uint8", row_bytes=width, quant_block=32,
                shards=[dict(shard_id=uid(layer * 10 + width), row_begin=0, row_end=3,
                             logical_bytes=3 * width, byte_offset=0 if width == 32 else 96,
                             allocation_bytes=4096, allocation_granularity=4096,
                             owner_id="owner", owner_incarnation=uid(2), allocation_id=uid(100 + layer),
                             handle_ref=f"capability-{layer}", sha256="sha256:" + "b" * 64)]))
    document["dataset_id"] = m.compute_dataset_id(document)
    layout = m.ExpectedLayout(layers=((1, 3), (2, 3)), head_dim=32,
                              model_revision="fixture-v1", hash_abi=document["hash_abi"])
    return document, layout


def sharded_fixture():
    document = dict(schema_version=1, model_revision="fixture-v1", generation=uid(1),
                    layout_abi=m.LAYOUT_ABI, hash_abi="sha256:" + "a" * 64,
                    backend="fabric_egm", state="READY", tensors=[])
    rows, shard_rows = 8192, 4096
    for layer in (1, 2):
        for role, width in (("weight", 32), ("scale", 1)):
            shards = []
            for index in range(2):
                begin = index * shard_rows
                end = min(begin + shard_rows, rows)
                shards.append(dict(
                    shard_id=uid(1000 + layer * 10 + (role == "scale") * 2 + index),
                    row_begin=begin, row_end=end,
                    logical_bytes=(end - begin) * width, byte_offset=0,
                    allocation_bytes=shard_rows * width, allocation_granularity=4096,
                    owner_id=f"shard-owner-{index}", owner_incarnation=uid(2000 + index),
                    allocation_id=uid(3000 + layer * 10 + (role == "scale") * 2 + index),
                    handle_ref=f"capability-{layer}-{role}-{index}",
                    sha256="sha256:" + "b" * 64))
            document["tensors"].append(dict(
                name=f"layers.{layer}.engram.embed.{role}", shape=[rows, width],
                dtype="float8_e4m3fn" if role == "weight" else "float8_e8m0fnu",
                storage_dtype="uint8", row_bytes=width, quant_block=32, shards=shards))
    document["dataset_id"] = m.compute_dataset_id(document)
    layout = m.ExpectedLayout(layers=((1, rows), (2, rows)), head_dim=32,
                              model_revision="fixture-v1", hash_abi=document["hash_abi"])
    return document, layout


class FakeShardOps:
    def __init__(self):
        self.calls = []
        self._next = 0x1000_0000_0000

    def reserve(self, size, alignment):
        self.calls.append(("reserve", size))
        pointer = self._next
        self._next += 1 << 34
        return pointer

    def import_handle(self, raw):
        self.calls.append(("import", raw))
        return int.from_bytes(raw[:8], "little")

    def map(self, va, size, offset, handle):
        self.calls.append(("map", va, size, offset, handle))

    def set_access(self, va, size):
        self.calls.append(("set_access", va, size))
        if getattr(self, "fail_on", "") == "set_access":
            raise RuntimeError("set_access failed")

    def release(self, handle):
        self.calls.append(("release", handle))

    def synchronize(self):
        self.calls.append(("synchronize",))

    def unmap(self, va, size):
        self.calls.append(("unmap", va, size))

    def free(self, va, size):
        self.calls.append(("free", va, size))

    def view(self, va, shape, owner):
        self.calls.append(("view", va, shape))
        return (va, shape)


class View:
    def __init__(self, mapping):
        self.mapping = mapping


class Native:
    def __init__(self, fail_import=None, fail_view=None, fail_close=False):
        self.fail_import, self.fail_view, self.fail_close = fail_import, fail_view, fail_close
        self.imports, self.views, self.closed = [], 0, []
        self.attempts = 0

    def Consumer(self, handle, size, granularity, device):
        self.attempts += 1
        if self.attempts == self.fail_import:
            raise RuntimeError("injected import failure")
        mapping = Mapping(self, len(self.imports), handle)
        self.imports.append(mapping)
        return mapping


class Mapping:
    def __init__(self, native, index, handle):
        self.native, self.index = native, index
        self.closed = False
        self.refs = []

    def tensor_view(self, shape, offset):
        self.native.views += 1
        if self.native.views == self.native.fail_view:
            raise RuntimeError("injected view failure")
        result = View(self)
        self.refs.append(weakref.ref(result))
        return result

    def info(self):
        return dict(active_views=sum(r() is not None for r in self.refs), closed=self.closed)

    def close(self):
        if self.native.fail_close:
            raise RuntimeError("injected drain failure")
        if self.info()["active_views"]:
            raise RuntimeError("live views")
        if not self.closed:
            self.native.closed.append(self.index)
        self.closed = True


class StorageTest(unittest.TestCase):
    def attach(self, native=None, verify=True, **changes):
        d, layout = fixture()
        arguments = dict(expected_layout=layout, dataset_id=d["dataset_id"], generation=d["generation"],
                         handles={uid(101): b"private-one", uid(102): b"private-two"},
                         device=0, native=native or Native())
        arguments.update(changes)
        result = s.FabricEngramStorage.attach(json.dumps(d).encode(), **arguments)
        if verify:
            result.verify(lambda views, verified: True)
        return result


    def test_two_shard_attach_stitches_two_spans(self):
        d, layout = sharded_fixture()
        handles = {shard["allocation_id"]: b"handle-" + shard["allocation_id"].encode()
                   for tensor in d["tensors"] for shard in tensor["shards"]}
        ops = FakeShardOps()
        storage = s.FabricEngramStorage.attach(
            json.dumps(d).encode(), expected_layout=layout, dataset_id=d["dataset_id"],
            generation=d["generation"], handles=handles, device=0, native=Native(),
            shard_ops=ops)
        self.assertEqual(storage.info()["allocation_count"], 8)
        reserves = [c for c in ops.calls if c[0] == "reserve"]
        maps = [c for c in ops.calls if c[0] == "map"]
        self.assertEqual(len(reserves), 4)  # two layers, weight + scale spans
        self.assertEqual(len(maps), 8)     # two shards per span
        for _, _, size, offset, _ in maps:
            self.assertEqual(offset, 0)
        storage.verify(lambda views, verified: True)
        weight, scale = storage.table_views(1)
        self.assertEqual(weight[1], (8192, 32))
        self.assertEqual(scale[1], (8192, 1))
        storage.close()
        self.assertEqual(storage.info()["state"], "CLOSED")
        self.assertEqual(len([c for c in ops.calls if c[0] == "unmap"]), 8)
        self.assertEqual(len([c for c in ops.calls if c[0] == "free"]), 4)

    def test_deduplicated_imports_and_close_are_idempotent(self):
        native = Native()
        storage = self.attach(native)
        self.assertEqual(native.attempts, 2)
        self.assertEqual(native.views, 4)
        self.assertEqual(storage.info()["physical_bytes"], 8192)
        storage.close()
        storage.close()
        self.assertEqual(native.closed, [1, 0])
        self.assertEqual(storage.info()["state"], "CLOSED")

    def test_wrong_identity_and_handle_set_fail_before_cuda(self):
        for change in (dict(dataset_id="sha256:" + "0" * 64), dict(generation=uid(99)),
                       dict(handles={}), dict(handles={uid(101): b"one"}),
                       dict(handles={uid(101): b"one", uid(102): "not-bytes"}), dict(device=True)):
            native = Native()
            with self.assertRaises(ValueError):
                self.attach(native, **change)
            self.assertEqual(native.attempts, 0)

    def test_second_import_failure_rolls_back_first(self):
        native = Native(fail_import=2)
        with self.assertRaisesRegex(RuntimeError, "import failure"):
            self.attach(native)
        self.assertEqual(native.closed, [0])

    def test_partial_view_failure_drops_views_before_reverse_cleanup(self):
        native = Native(fail_view=3)
        with self.assertRaisesRegex(RuntimeError, "view failure"):
            self.attach(native)
        self.assertEqual(native.closed, [1, 0])
        self.assertTrue(all(m.info()["active_views"] == 0 for m in native.imports))

    def test_failed_rollback_is_not_reported_as_clean(self):
        native = Native(fail_view=1, fail_close=True)
        with self.assertRaises(s.CleanupPendingError) as error:
            self.attach(native)
        pending = error.exception.resource
        self.assertEqual(pending.info()["state"], "FAILED")
        native.fail_close = False
        pending.close()
        self.assertEqual(native.closed, [1, 0])

    def test_live_external_views_prevent_any_partial_unmap(self):
        native = Native()
        storage = self.attach(native)
        weight, scale = storage.table_views(1)
        with self.assertRaises(s.StorageBusyError):
            storage.close()
        self.assertEqual(native.closed, [])
        self.assertEqual(storage.info()["state"], "CLOSING")
        with self.assertRaises(RuntimeError):
            storage.table_views(2)
        del weight, scale
        storage.close()
        self.assertEqual(native.closed, [1, 0])

    def test_graph_pin_rejects_drain_without_changing_ready(self):
        storage = self.attach()
        storage._pin_graph()
        with self.assertRaises(s.StorageBusyError):
            storage.close()
        self.assertEqual(storage.info()["state"], "READY")
        storage._unpin_graph()
        storage.close()
        with self.assertRaises(RuntimeError):
            storage._unpin_graph()

    def test_ready_model_process_pin_blocks_actual_storage_close_and_survives_model_gc(self):
        native = Native()
        storage = self.attach(native)
        retained = []
        model = types.SimpleNamespace(
            _v4_args=types.SimpleNamespace(v41_config={"checkpoint_path": "/local/consumer"}),
            v4=types.SimpleNamespace(embed=types.SimpleNamespace(weight=types.SimpleNamespace(device="cuda:0")),
                                     layers=[types.SimpleNamespace(engram=None) for _ in range(3)]),
            py_hw_kernel_config=types.SimpleNamespace(enable_cuda_graph=True),
        )
        class LayerViews:
            def __init__(self, layer_id):
                self.views = storage.table_views(layer_id)
            def release_external_views(self):
                self.views = None
        b.stage_package({"staged": True})
        self.addCleanup(b.clear_staged_package)
        attachment = a.initialize_model_engram(
            model, layout_factory=lambda config: types.SimpleNamespace(layer_ids=(1, 2)),
            hash_factory=lambda *args, **kwargs: object(),
            layer_factory=lambda config, layer_id, *args, **kwargs: LayerViews(layer_id),
            attach_factory=lambda *args, **kwargs: storage,
            retain=retained.append,
            environ={a.EXTERNAL_GRAPH_MODE_ENV: a.PROCESS_LIFETIME_GRAPH},
        )
        self.assertEqual(attachment.phase, "READY")
        a.require_engram_runtime(model, is_cuda_graph=True)
        before = storage.info()
        self.assertEqual(before["graph_pins"], 1)
        # Even an explicit model-side view release cannot authorize unmapping
        # while the engine may hold/replay a captured raw pointer.
        for _, _, module in attachment.layers:
            module.release_external_views()
        with self.assertRaises(s.StorageBusyError):
            storage.close()
        self.assertEqual(storage.info(), before)
        self.assertEqual(native.closed, [])
        weak_storage = weakref.ref(storage)
        del model, attachment, storage
        gc.collect()
        self.assertIs(weak_storage(), retained[0])
        with self.assertRaises(s.StorageBusyError):
            retained[0].close()
        self.assertEqual(native.closed, [])
        self.assertEqual(retained[0].info()["graph_pins"], 1)

    def test_unknown_layer_and_use_after_close_fail(self):
        storage = self.attach()
        with self.assertRaises(KeyError):
            storage.table_views(99)
        storage.close()
        with self.assertRaises(RuntimeError):
            with storage.use():
                self.fail("closed storage admitted work")

    def test_close_waits_for_submitter_then_rejects_new_work(self):
        storage = self.attach()
        entered, release, attempted, closed = (threading.Event() for _ in range(4))
        failures = []
        def submit():
            try:
                with storage.use():
                    entered.set()
                    self.assertTrue(release.wait(3))
            except BaseException as error:
                failures.append(error)
        def drain():
            attempted.set()
            try:
                storage.close()
                closed.set()
            except BaseException as error:
                failures.append(error)
        a = threading.Thread(target=submit)
        b = threading.Thread(target=drain)
        a.start()
        self.assertTrue(entered.wait(3))
        b.start()
        self.assertTrue(attempted.wait(3))
        self.assertFalse(closed.is_set())
        release.set()
        a.join(3)
        b.join(3)
        self.assertFalse(a.is_alive() or b.is_alive())
        self.assertFalse(failures)
        self.assertTrue(closed.is_set())

    def test_mapped_is_not_ready_without_explicit_content_verification(self):
        storage = self.attach(verify=False)
        self.assertEqual(storage.info()["state"], "MAPPED")
        with self.assertRaises(RuntimeError):
            storage.table_views(1)
        with self.assertRaises(ValueError):
            storage.verify(lambda views, verified: None)
        self.assertEqual(storage.info()["state"], "FAILED")
        storage.close()

    def test_capture_cleanup_failure_retains_retryable_owner_and_pins(self):
        storage = self.attach()
        class FakeStream:
            def wait_stream(self, other): pass
        class Cuda:
            device = staticmethod(lambda device: nullcontext())
            stream = staticmethod(lambda stream: nullcontext())
            current_stream = staticmethod(lambda: None)
            Stream = FakeStream
            calls = 0
            @classmethod
            def synchronize(cls):
                cls.calls += 1
                if cls.calls == 1:
                    raise RuntimeError("injected sync failure")
        fake_torch = types.SimpleNamespace(cuda=Cuda)
        def broken_forward():
            raise ValueError("capture input failure")
        with patch.dict(sys.modules, {"torch": fake_torch}):
            with self.assertRaises(s.CleanupPendingError) as error:
                s.CapturedEngramGraph.capture(broken_forward, [storage])
            self.assertIsInstance(error.exception.__cause__, ValueError)
            pending = error.exception.resource
            with self.assertRaisesRegex(RuntimeError, "not ready for replay"):
                pending.replay()
            self.assertEqual(storage.info()["graph_pins"], 1)
            with self.assertRaises(s.StorageBusyError):
                storage.close()
            pending.close()
        self.assertEqual(storage.info()["graph_pins"], 0)
        storage.close()

    def test_verifier_cannot_reenter_close_or_use(self):
        storage = self.attach(verify=False)
        def verify(views, manifest):
            with self.assertRaises(s.StorageBusyError):
                storage.close()
            with self.assertRaises(RuntimeError):
                storage.verify(lambda *_: True)
            with self.assertRaises(RuntimeError):
                storage.table_views(1)
            return True
        storage.verify(verify)
        self.assertEqual(storage.info()["state"], "READY")
        storage.close()

    def test_drain_failure_keeps_failure_state(self):
        native = Native(fail_close=True)
        storage = self.attach(native)
        with self.assertRaisesRegex(RuntimeError, "drain failure"):
            storage.close()
        self.assertEqual(storage.info()["state"], "FAILED")
        native.fail_close = False
        storage.close()


class FakeStream:
    def wait_stream(self, other):
        pass

    def synchronize(self):
        pass


class FakeGraph:
    def __init__(self, cuda):
        self.cuda = cuda
        self.replays = 0
        self.reset_attempts = 0
        self.reset_successes = 0

    def replay(self):
        self.replays += 1

    def reset(self):
        self.reset_attempts += 1
        if self.cuda.fail == "reset":
            raise RuntimeError("injected reset failure")
        self.reset_successes += 1


class FakeCuda:
    def __init__(self):
        self.fail = None
        self.sync_calls = 0
        self.graph_object = None

    device = staticmethod(lambda device: nullcontext())
    stream = staticmethod(lambda stream: nullcontext())
    current_stream = staticmethod(lambda: None)
    graph = staticmethod(lambda graph, stream: nullcontext())
    Stream = FakeStream

    def CUDAGraph(self):
        self.graph_object = FakeGraph(self)
        return self.graph_object

    def synchronize(self):
        self.sync_calls += 1
        if self.fail == "sync":
            raise RuntimeError("injected sync failure")


class GraphFailureTest(unittest.TestCase):
    def check_close_failure(self, failure):
        # Real storage pin/use/close behavior; native mappings are unnecessary
        # for a CPU test of graph orchestration and reference ownership.
        storage = s.FabricEngramStorage()
        storage._state = "READY"
        storage.device = 0
        cuda = FakeCuda()
        output = object()
        with patch.dict(sys.modules, {"torch": types.SimpleNamespace(cuda=cuda)}):
            graph = s.CapturedEngramGraph.capture(lambda: output, (storage,), warmup=1)
            native_graph = cuda.graph_object
            self.assertEqual(storage._graph_pins, 1)
            self.assertIs(graph.replay(), output)
            self.assertEqual(native_graph.replays, 1)

            cuda.fail = failure
            with self.assertRaisesRegex(RuntimeError, f"injected {failure} failure"):
                graph.close()
            self.assertFalse(graph._closed)
            self.assertEqual(storage._graph_pins, 1)
            self.assertIs(graph._graph, native_graph)
            self.assertEqual(native_graph.reset_successes, 0)
            with self.assertRaisesRegex(RuntimeError, "not ready for replay"):
                graph.replay()
            self.assertEqual(native_graph.replays, 1)
            with self.assertRaises(s.StorageBusyError):
                storage.close()

            cuda.fail = None
            graph.close()
            self.assertTrue(graph._closed)
            self.assertEqual(native_graph.reset_successes, 1)
            self.assertEqual(storage._graph_pins, 0)
            self.assertIsNone(graph._graph)
            self.assertIsNone(graph.output)
            sync_calls = cuda.sync_calls
            graph.close()
            self.assertEqual(cuda.sync_calls, sync_calls)
            with self.assertRaisesRegex(RuntimeError, "not ready for replay"):
                graph.replay()
            storage.close()
            self.assertEqual(storage._state, "CLOSED")

    def test_successful_capture_stops_replay_after_close_sync_failure(self):
        self.check_close_failure("sync")

    def test_successful_capture_stops_replay_after_close_reset_failure(self):
        self.check_close_failure("reset")


if __name__ == "__main__":
    unittest.main()
