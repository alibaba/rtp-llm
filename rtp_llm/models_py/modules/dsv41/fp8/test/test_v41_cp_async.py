"""CP asynchronous handoff ordering and allocator-lifetime contracts."""

import gc
import time
import unittest
import weakref
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv41.fp8 import _v41_cp_async as async_cp
from rtp_llm.models_py.modules.dsv41.fp8._v41_prefill_kv_workspace import (
    PreparedGlobalRows,
)


class FakeTensor:
    def __init__(self, name, log, *, base=None, cuda=True, storage=None):
        self.name, self.log, self._base, self.is_cuda = name, log, base, cuda
        self.device = "cuda:0" if cuda else "cpu"
        self.storage = storage if storage is not None else object()

    def untyped_storage(self):
        return SimpleNamespace(data_ptr=lambda: id(self.storage))

    def record_stream(self, stream):
        self.log.append(("record", self.name, stream.name))


class FakeStream:
    def __init__(self, name, log):
        self.name, self.log = name, log

    def wait_stream(self, stream):
        self.log.append(("wait_stream", self.name, stream.name))

    def wait_event(self, event):
        self.log.append(("wait_event", self.name, event))


class FakeEvent:
    def __init__(self, log):
        self.log = log

    def record(self, stream):
        self.log.append(("event", stream.name, self))


class CPAsyncHostTest(unittest.TestCase):
    def setUp(self):
        self.log = []
        self.main = FakeStream("main", self.log)
        self.comm = FakeStream("comm", self.log)
        self.current = self.main

        @contextmanager
        def use_stream(stream):
            previous, self.current = self.current, stream
            try:
                yield
            finally:
                self.current = previous

        fake = SimpleNamespace(
            Tensor=FakeTensor,
            device=torch.device,
            cuda=SimpleNamespace(
                current_stream=lambda device: self.current,
                stream=use_stream,
                Event=lambda: FakeEvent(self.log),
            ),
        )
        self.addCleanup(patch.stopall)
        patch.object(async_cp, "torch", fake).start()
        patch.object(async_cp, "_communication_stream", return_value=self.comm).start()

    def tensor(self, name, **kwargs):
        return FakeTensor(name, self.log, **kwargs)

    def test_launch_and_each_consumer_have_only_stream_dependencies(self):
        source, result = self.tensor("source"), self.tensor("result")

        def compute():
            self.assertIs(self.current, self.comm)
            self.log.append(("compute",))
            return result

        pending = async_cp.launch("cuda:0", compute, inputs=(source,))
        self.assertEqual(
            self.log[:3],
            [
                ("wait_stream", "comm", "main"),
                ("record", "source", "comm"),
                ("compute",),
            ],
        )
        self.assertEqual(self.log[3], ("event", "comm", pending.event))
        self.assertIs(pending.wait(), result)
        self.assertEqual(
            self.log[-2:],
            [("wait_event", "main", pending.event), ("record", "result", "main")],
        )
        before = len(self.log)
        pending.wait()
        self.assertEqual(len(self.log), before)
        self.current = FakeStream("consumer2", self.log)
        pending.wait()
        self.assertEqual(self.log[-1], ("record", "result", "consumer2"))

    def test_nested_views_and_tuple_workspace_are_retained_once(self):
        owner = self.tensor("owner")
        view = self.tensor("view", base=owner)
        offsets = self.tensor("offsets")
        keys = SimpleNamespace(quant=self.tensor("quant"), scale=self.tensor("scale"))
        workspace = SimpleNamespace(
            output=owner,
            device_offsets=offsets,
            refresh_layout=self.tensor("refresh", base=offsets),
        )
        result = PreparedGlobalRows(((view, keys), (view, keys)), workspace)
        pending = async_cp.launch("cuda:0", lambda: result)
        pending.wait()
        recorded = [entry[1] for entry in self.log if entry[0] == "record"]
        self.assertCountEqual(recorded, ["owner", "offsets", "quant", "scale"])

    def test_exception_records_inputs_and_restores_current_stream(self):
        source = self.tensor("source")

        def failing():
            self.assertEqual(self.log[-1], ("record", "source", "comm"))
            raise ValueError("enqueue failed")

        with self.assertRaisesRegex(ValueError, "enqueue failed"):
            async_cp.launch("cuda:0", failing, inputs={"source": source})
        self.assertIs(self.current, self.main)
        self.assertEqual(self.log[-1], ("wait_stream", "main", "comm"))
        self.assertFalse(any(entry[0] == "event" for entry in self.log))

    def test_event_failure_also_orders_submitted_work_before_propagating(self):
        with patch.object(
            async_cp.torch.cuda, "Event", side_effect=RuntimeError("event failed")
        ):
            with self.assertRaisesRegex(RuntimeError, "event failed"):
                async_cp.launch("cuda:0", lambda: self.tensor("result"))
        self.assertIs(self.current, self.main)
        self.assertEqual(self.log[-1], ("wait_stream", "main", "comm"))

    def test_cpu_fallback_has_no_cuda_calls(self):
        result = object()
        pending = async_cp.launch("cpu", lambda: result)
        self.assertIs(pending.wait(), result)
        self.assertEqual(self.log, [])

    def test_forward_ownership_is_not_cached_or_shared(self):
        first_value = self.tensor("first")
        first_ref = weakref.ref(first_value)
        first = async_cp.launch("cuda:0", lambda: first_value)
        second = async_cp.launch("cuda:0", lambda: self.tensor("second"))
        first.wait()
        second.wait()
        self.assertIsNot(first.event, second.event)
        self.assertIsNot(first._consumers, second._consumers)
        del first, first_value
        gc.collect()
        self.assertIsNone(first_ref())

    def test_cyclic_tree_and_host_tensors_are_safe(self):
        source = self.tensor("source")
        inputs = [source, self.tensor("host", cuda=False)]
        inputs.append(inputs)
        async_cp.launch("cuda:0", lambda: None, inputs=inputs)
        self.assertEqual(
            [entry for entry in self.log if entry[0] == "record"],
            [("record", "source", "comm")],
        )

    def test_inference_aliases_record_each_storage_once(self):
        storage = object()
        aliases = [self.tensor(str(i), storage=storage) for i in range(64)]
        independent = self.tensor("independent")
        pending = async_cp.launch("cuda:0", lambda: (aliases, independent))
        pending.wait()
        recorded = [entry[1] for entry in self.log if entry[0] == "record"]
        self.assertEqual(len(recorded), 2)
        self.assertIn("independent", recorded)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class CPAsyncCudaTest(unittest.TestCase):
    def test_inference_views_share_one_allocator_record(self):
        with torch.inference_mode():
            source = torch.empty((64, 17), device="cuda")
            views = source.split(1)
            self.assertTrue(all(view._base is None for view in views))
            recorded = []
            original = torch.Tensor.record_stream

            def record(tensor, stream):
                recorded.append(tensor.untyped_storage().data_ptr())
                return original(tensor, stream)

            with patch.object(torch.Tensor, "record_stream", record):
                async_cp._record_stream(views, torch.cuda.current_stream())
            self.assertEqual(recorded, [source.untyped_storage().data_ptr()])

    def test_exception_orders_in_place_reuse_after_submitted_writes(self):
        device = torch.device("cuda", torch.cuda.current_device())
        comm = torch.cuda.Stream(device=device)
        pool = torch.zeros(1 << 20, dtype=torch.int32, device=device)
        with torch.cuda.stream(comm):
            torch.cuda._sleep(1)
            pool.fill_(7)
        comm.synchronize()

        def failing():
            torch.cuda._sleep(100_000_000)
            pool.fill_(7)
            raise RuntimeError("after enqueue")

        with patch.object(async_cp, "_communication_stream", return_value=comm):
            with self.assertRaisesRegex(RuntimeError, "after enqueue"):
                async_cp.launch(device, failing, inputs=(pool,))
        # An exception may cause the next request to reuse the same physical
        # KV page. Its main-stream write must follow the abandoned producer.
        pool.fill_(9)
        torch.cuda.synchronize(device)
        self.assertTrue(torch.equal(pool, torch.full_like(pool, 9)))

    def test_pending_work_does_not_block_host_and_survives_input_release(self):
        device = torch.device("cuda", torch.cuda.current_device())
        comm = torch.cuda.Stream(device=device)
        source = torch.full((1 << 20,), 7, device=device, dtype=torch.int32)
        # Warm the exact allocation/op before testing a pending stream.
        with torch.cuda.stream(comm):
            torch.cuda._sleep(1)
            warm = source + 1
        comm.synchronize()
        del warm
        with patch.object(async_cp, "_communication_stream", return_value=comm):

            def compute():
                torch.cuda._sleep(100_000_000)
                return source + 1

            begin = time.monotonic()
            pending = async_cp.launch(device, compute, inputs=(source,))
            elapsed = time.monotonic() - begin
            self.assertFalse(pending.event.query(), "producer already completed")
            del source
            # Exercise the allocator on the original stream while the input is
            # still being read on the communication stream.
            trash = [
                torch.full((1 << 20,), -3, device=device, dtype=torch.int32)
                for _ in range(8)
            ]
            result = pending.wait()
            self.assertLess(elapsed, 0.1, "launch waited for GPU work")
            self.assertTrue(torch.equal(result, torch.full_like(result, 8)))
            del trash

    def test_output_can_be_released_while_another_stream_consumes_it(self):
        device = torch.device("cuda", torch.cuda.current_device())
        comm, consumer = torch.cuda.Stream(device=device), torch.cuda.Stream(
            device=device
        )
        with patch.object(async_cp, "_communication_stream", return_value=comm):
            pending = async_cp.launch(
                device, lambda: torch.full((1 << 20,), 11, device=device)
            )
            with torch.cuda.stream(consumer):
                value = pending.wait()
                torch.cuda._sleep(100_000_000)
                copied = value.clone()
            del value, pending
            with torch.cuda.stream(comm):
                trash = [torch.full((1 << 20,), -4, device=device) for _ in range(8)]
            consumer.synchronize()
            self.assertTrue(torch.equal(copied, torch.full_like(copied, 11)))
            del trash


if __name__ == "__main__":
    unittest.main()
