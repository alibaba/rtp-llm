import ctypes
import json
import os
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from rtp_llm.multimodal.kvcm.registered_transport import (
    RegisteredObjectTransport,
    RegisteredTransportPool,
    RegisteredTransportUnavailableError,
    _enable_sync_memops,
    _new_cuda_stream,
)

from rtp_llm.multimodal.kvcm.tensor_object import plan_object


class CudaRdmaSyncMemopsTest(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_registered_stream_handles_are_distinct_and_reusable(self):
        with patch(
            "rtp_llm.multimodal.kvcm.registered_transport._ISOLATED_CUDA_STREAMS",
            True,
        ):
            streams = [_new_cuda_stream(torch.device("cuda:0")) for _ in range(33)]
            addresses = {stream.cuda_stream for stream in streams}
            self.assertEqual(len(addresses), 33)
            streams.clear()
            streams = [_new_cuda_stream(torch.device("cuda:0")) for _ in range(33)]
            self.assertEqual({stream.cuda_stream for stream in streams}, addresses)

    def test_enables_and_checks_the_owned_allocation(self):
        arena = Mock()
        arena.data_ptr.return_value = 0x12340000
        operations = []

        def set_attribute(value, attribute, pointer):
            self.assertEqual(
                (value._obj.value, attribute, pointer), (1, 6, arena.data_ptr())
            )
            operations.append("set")
            return 0

        def get_attribute(value, attribute, pointer):
            self.assertEqual((attribute, pointer), (6, arena.data_ptr()))
            operations.append("get")
            value._obj.value = 1
            return 0

        with patch("ctypes.CDLL") as load:
            driver = load.return_value
            driver.cuPointerSetAttribute.side_effect = set_attribute
            driver.cuPointerGetAttribute.side_effect = get_attribute
            _enable_sync_memops(arena)
            self.assertEqual(operations, ["set", "get"])

    def test_driver_errors_and_unset_attribute_are_rejected(self):
        for set_result, get_result in ((1, 0), (0, 999), (0, 0)):
            with self.subTest(set_result=set_result, get_result=get_result):
                with patch("ctypes.CDLL") as load:
                    driver = load.return_value
                    driver.cuPointerSetAttribute.return_value = set_result
                    # A successful query that leaves the value zero is unsafe too.
                    driver.cuPointerGetAttribute.return_value = get_result
                    with self.assertRaisesRegex(RuntimeError, "SYNC_MEMOPS"):
                        _enable_sync_memops(Mock(data_ptr=lambda: 0x12340000))
                    if set_result:
                        driver.cuPointerGetAttribute.assert_not_called()


class RegisteredObjectTransportTest(unittest.TestCase):
    def test_invalid_configuration_does_not_construct_client(self):
        def forbidden(**kw):
            self.fail("invalid configuration constructed client")

        for capacity in (0, -1, 1024**3 + 1):
            with self.assertRaises(ValueError):
                RegisteredTransportPool(
                    forbidden, max_object_bytes=capacity, device="cuda:0"
                )
        with self.assertRaises(ValueError):
            RegisteredTransportPool(forbidden, max_object_bytes=1024, device="cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_registered_range_roundtrip_bounds_and_close(self):
        seen = {}

        class Client:
            instance_id = "test"
            instance_group = "test"

            def __init__(self, gpu_tensors):
                (self.arena,) = gpu_tensors
                assert "client" not in seen
                value = ctypes.c_uint()
                driver = ctypes.CDLL("libcuda.so.1")
                driver.cuPointerGetAttribute.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_int,
                    ctypes.c_uint64,
                ]
                assert (
                    driver.cuPointerGetAttribute(
                        ctypes.byref(value), 6, self.arena.data_ptr()
                    )
                    == 0
                )
                assert value.value == 1, "arena exported before SYNC_MEMOPS"
                seen["client"] = self
                self.closed = 0

            def save_one(self, key, value, **kwargs):
                assert value.device.type == "cuda"
                assert value.data_ptr() >= self.arena.data_ptr() + 65536
                assert (
                    value.data_ptr() + value.numel()
                    <= self.arena.data_ptr() + self.arena.numel()
                )
                self.value = value.clone()

            def load_one(self, key, value, **kwargs):
                value.copy_(self.value)

            def close(self):
                self.closed += 1

        pool = RegisteredTransportPool(
            Client, max_object_bytes=1024, device="cuda:0", slots=2
        )
        transport = pool.acquire()
        copies = []

        class ObserveCopies(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                if func == torch.ops.aten.copy_.default:
                    dst, src = args[:2]
                    if dst.is_cuda != src.is_cuda:
                        host = src if dst.is_cuda else dst
                        copies.append((dst.is_cuda, host.is_pinned(), host.data_ptr()))
                return func(*args, **(kwargs or {}))

        reader = pool.acquire()
        self.assertEqual(reader._buffer.data_ptr() - transport._buffer.data_ptr(), 1024)
        host_ptr = transport._host_staging.data_ptr()
        read_ptr = reader._host_staging.data_ptr()
        self.assertNotEqual(host_ptr, read_ptr)
        with ObserveCopies():
            for size in (1024, 17):
                source = torch.arange(size).to(torch.uint8)
                target = torch.empty_like(source)
                transport.save_one("x", source)
                self.assertIs(reader.load_one("x", target), target)
                self.assertTrue(torch.equal(source, target))
        self.assertEqual(copies, [(True, True, host_ptr), (False, True, read_ptr)] * 2)
        for bad in (
            torch.empty(1025, dtype=torch.uint8),
            torch.empty(0, dtype=torch.uint8),
            torch.empty(3),
            torch.empty((2, 2), dtype=torch.uint8),
        ):
            with self.assertRaises(ValueError):
                transport.save_one("x", bad)
        pool.release(reader)
        pool.release(transport)
        pool.close()
        pool.close()
        self.assertEqual(seen["client"].closed, 1)
        self.assertIsNone(transport._buffer)
        self.assertIsNone(transport._host_staging)
        with self.assertRaises(RuntimeError):
            transport.load_one("x", target)
        with self.assertRaises(RuntimeError):
            transport.load_from_meta(object(), target)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_direct_mixed_object_does_not_alias_a_recycled_slot(self):
        storage = {}

        class Client:
            instance_id = instance_group = "test"

            def __init__(self, gpu_tensors):
                (self.arena,) = gpu_tensors

            def save_one(self, key, data, **kwargs):
                storage[key] = data.clone()
                torch.cuda.current_stream().synchronize()

            def load_from_meta(self, meta, target, **kwargs):
                target.copy_(storage[meta.key])
                torch.cuda.current_stream().synchronize()

            def close(self):
                pass

        pool = RegisteredTransportPool(
            Client,
            max_object_bytes=4109,
            device="cuda:0",
            direct_payload=True,
            arena_bytes=16384,
            min_block_bytes=64,
        )
        gpu = torch.arange(42, dtype=torch.float16, device="cuda").reshape(6, 7).t()
        host = torch.tensor([7, 8, 9], dtype=torch.int32)
        value = (gpu, host, {"alias": gpu, "empty": gpu[:0]})
        plan = plan_object(value, checksum=False)
        first = pool.acquire(nbytes=plan.nbytes, for_write=True)
        second = pool.acquire(nbytes=plan.nbytes)
        try:
            self.assertIsNone(first._host_staging)
            self.assertIsNone(second._host_staging)
            self.assertEqual(second._buffer.data_ptr() % 64, 0)
            self.assertEqual(
                second._buffer.data_ptr() - first._buffer.data_ptr(),
                ((plan.nbytes + 63) // 64) * 64,
            )
            payload = first.pack(plan)
            self.assertEqual(payload.data_ptr(), first._buffer.data_ptr())
            first.save_one("test", payload)
            meta = SimpleNamespace(key="test", size_bytes=plan.nbytes)
            restored = second.load_object_from_meta(
                meta, consumer_stream=torch.cuda.current_stream()
            )
            second._buffer.fill_(0xA6)
            torch.cuda.current_stream().synchronize()
            self.assertIs(restored[0], restored[2]["alias"])
            self.assertEqual(restored[2]["empty"].device.type, "cuda")
            self.assertEqual(restored[1].device.type, "cpu")
            torch.testing.assert_close(restored[0], gpu)
            torch.testing.assert_close(restored[1], host)
            self.assertNotEqual(
                restored[0].untyped_storage().data_ptr(),
                second._buffer.untyped_storage().data_ptr(),
            )
        finally:
            pool.release(second)
            pool.release(first)
            pool.close()


class RegisteredTransportFailureTest(unittest.TestCase):
    """Exercise failure ownership without claiming actual CUDA/RDMA coverage."""

    def setUp(self):
        self.memops = patch(
            "rtp_llm.multimodal.kvcm.registered_transport._enable_sync_memops"
        )
        self.enable_memops = self.memops.start()
        self.addCleanup(self.memops.stop)
        self.quarantine = patch.object(
            RegisteredObjectTransport, "_quarantined_transports", []
        )
        self.quarantine.start()
        self.addCleanup(self.quarantine.stop)
        self.device = patch("torch.cuda.device", return_value=nullcontext())
        self.device.start()
        self.addCleanup(self.device.stop)
        stream_factory = patch(
            "rtp_llm.multimodal.kvcm.registered_transport._new_cuda_stream"
        )
        stream_factory.start()
        self.addCleanup(stream_factory.stop)
        stream_context = patch("torch.cuda.stream", return_value=nullcontext())
        stream_context.start()
        self.addCleanup(stream_context.stop)
        self.stream = patch("torch.cuda.current_stream", return_value=Mock())
        self.stream.start()
        self.addCleanup(self.stream.stop)

    def pool(self, slots=1, factory=None, ttl=300, direct_payload=False, **kwargs):
        allocate = torch.empty
        with patch("torch.cuda.MemPool"), patch(
            "torch.cuda.use_mem_pool", return_value=nullcontext()
        ), patch(
            "rtp_llm.multimodal.kvcm.registered_transport._new_cuda_stream",
            return_value=torch.cuda.current_stream(),
        ), patch(
            "torch.empty",
            side_effect=lambda size, *, dtype, **kw: allocate(size, dtype=dtype),
        ):
            pool = RegisteredTransportPool(
                factory or Mock(),
                max_object_bytes=kwargs.pop("max_object_bytes", 16),
                device="cuda:0",
                slots=slots,
                observe_hold=Mock(),
                quarantine_ttl_seconds=ttl,
                direct_payload=direct_payload,
                **kwargs,
            )
        self.addCleanup(pool._gc_stop.set)
        self.addCleanup(pool.close)
        return pool

    def transport(self):
        return self.pool()._slots[0]

    def assert_single_free_range(self, pool, start, size):
        if pool._native_arena is None:
            self.assertEqual(pool._free_ranges, [(start, size)])
        else:
            self.assertEqual(pool._native_arena.free_space(), (size, size))

    def test_one_client_registers_root_and_transfers_disjoint_slices_concurrently(self):
        factory = Mock()
        pool = self.pool(slots=3, factory=factory)
        client = factory.return_value
        factory.assert_called_once_with(gpu_tensors=(pool._arena,))
        self.enable_memops.assert_called_once_with(pool._arena)
        self.assertEqual(pool._arena.storage_offset(), 0)
        self.assertEqual(pool._arena.numel(), pool._GUARD_BYTES + 3 * 16)
        slots = [
            pool.acquire(for_write=True),
            pool.acquire(for_write=True),
            pool.acquire(),
        ]
        gate = threading.Barrier(3)
        payload = torch.arange(16, dtype=torch.uint8)

        def transfer(key, view):
            self.assertEqual(
                view.data_ptr() - pool._arena.data_ptr(), pool._GUARD_BYTES + key * 16
            )
            self.assertEqual(view.numel(), 16 - key)
            gate.wait(timeout=3)
            if key == 2:
                view.fill_(23)
            else:
                torch.testing.assert_close(view, payload[: 16 - key] + key)

        client.save_one.side_effect = client.load_one.side_effect = transfer
        out = torch.empty(14, dtype=torch.uint8)
        try:
            with ThreadPoolExecutor(3) as io:
                jobs = [
                    io.submit(slots[i].save_one, i, payload[: 16 - i] + i)
                    for i in range(2)
                ]
                jobs.append(io.submit(slots[2].load_one, 2, out))
                for job in jobs:
                    job.result(timeout=5)
            torch.testing.assert_close(out, torch.full_like(out, 23))
        finally:
            for slot in slots:
                pool.release(slot)
        pool.close()
        pool.close()
        client.close.assert_called_once()

    def test_lease_metrics_do_not_hold_the_allocator_lock(self):
        reporting_release = threading.Event()
        resume_report = threading.Event()

        def observe_lease(for_write, delta, nbytes):
            if delta == -1 and not reporting_release.is_set():
                reporting_release.set()
                resume_report.wait(4)

        pool = self.pool(observe_lease=observe_lease)
        slot = pool.acquire()
        acquired = None
        with ThreadPoolExecutor(2) as workers:
            released = workers.submit(pool.release, slot)
            try:
                self.assertTrue(reporting_release.wait(1))
                acquired = workers.submit(pool.acquire)
                self.assertIs(acquired.result(timeout=1), slot)
            finally:
                resume_report.set()
                released.result(timeout=1)
                if acquired is not None:
                    next_slot = acquired.result(timeout=1)
                    if next_slot is not None:
                        pool.release(next_slot)

    def test_acquire_deadline_includes_contended_allocator_lock(self):
        observe = Mock()
        pool = self.pool(observe_acquire_failure=observe)
        with ThreadPoolExecutor(1) as workers:
            acquired = None
            try:
                with pool._condition:
                    acquired = workers.submit(pool.acquire, timeout=0.03)
                    self.assertIsNone(acquired.result(timeout=1))
            finally:
                if acquired is not None:
                    slot = acquired.result(timeout=1)
                    if slot is not None:
                        pool.release(slot)
        observe.assert_called_once_with(False, "lock_timeout")
        slot = pool.acquire()
        self.assertIsNotNone(slot)
        try:
            self.assertIsNone(pool.acquire())
            self.assertEqual(
                observe.call_args_list[-1].args, (False, "extent_unavailable")
            )
        finally:
            pool.release(slot)
        pool.stop_acquiring()
        self.assertIsNone(pool.acquire())
        self.assertEqual(observe.call_args_list[-1].args, (False, "closed"))

    def test_lock_hold_sampling_excludes_condition_wait(self):
        samples = []

        def observe(for_write, stage, nanoseconds):
            self.assertFalse(pool._condition._is_owned())
            samples.append((for_write, stage, nanoseconds))

        pool = self.pool(observe_lock_hold=observe)
        with patch.object(pool._lock_hold_rng, "getrandbits", return_value=1):
            slot = pool.acquire(for_write=True)
        with patch.object(pool._lock_hold_rng, "getrandbits", return_value=0):
            with patch.object(pool._condition, "wait", return_value=False), patch.object(
                time, "perf_counter_ns", side_effect=(10, 20, 200, 203)
            ):
                self.assertIsNone(pool.acquire(timeout=1))
            with patch.object(time, "perf_counter_ns", side_effect=(500, 508)):
                pool.release(slot)
        self.assertEqual(
            samples,
            [(False, "acquire", 13), (True, "release", 8)],
        )

    def test_sized_lease_materializes_views_outside_the_allocator_lock(self):
        pool = self.pool(
            direct_payload=True,
            max_object_bytes=128,
            arena_bytes=256,
            min_block_bytes=64,
        )
        if pool._native_arena is not None:
            self.skipTest("native view construction is covered by the ticket tests")
        view_started, resume_view = threading.Event(), threading.Event()
        narrow = pool._arena.narrow

        def create_view(dim, offset, size):
            if offset == pool._GUARD_BYTES:
                view_started.set()
                self.assertTrue(resume_view.wait(3))
                raise RuntimeError("cannot materialize first view")
            return narrow(dim, offset, size)

        with patch.object(pool._arena, "narrow", side_effect=create_view):
            with ThreadPoolExecutor(2) as workers:
                first = workers.submit(pool.acquire, nbytes=64)
                try:
                    self.assertTrue(view_started.wait(1))
                    second = workers.submit(pool.acquire, nbytes=64, timeout=0.1)
                    slot = second.result(timeout=1)
                    self.assertIsNotNone(slot)
                    try:
                        self.assertEqual(
                            slot._buffer.storage_offset(), pool._GUARD_BYTES + 64
                        )
                    finally:
                        pool.release(slot)
                finally:
                    resume_view.set()
                with self.assertRaisesRegex(RuntimeError, "cannot materialize"):
                    first.result(timeout=1)
        self.assert_single_free_range(pool, pool._GUARD_BYTES, 256)
        self.assertFalse(pool._in_use)

    @unittest.skipUnless(
        os.environ.get("MM_EMB_NATIVE_GDR_ARENA") == "1", "native sized arena"
    )
    def test_native_pending_ticket_blocks_close_without_blocking_other_leases(self):
        pool = self.pool(
            direct_payload=True,
            max_object_bytes=128,
            arena_bytes=256,
            min_block_bytes=64,
        )
        client = pool._client
        started = threading.Event()
        resume = threading.Event()
        stopped = threading.Event()
        make_slot = RegisteredObjectTransport
        stop = pool.stop_acquiring

        def delayed_slot(*args, **kwargs):
            if kwargs["extent"][0] == pool._GUARD_BYTES:
                started.set()
                self.assertTrue(resume.wait(3))
            return make_slot(*args, **kwargs)

        def stopping():
            stop()
            stopped.set()

        with patch(
            "rtp_llm.multimodal.kvcm.registered_transport.RegisteredObjectTransport",
            side_effect=delayed_slot,
        ), patch.object(pool, "stop_acquiring", side_effect=stopping):
            with ThreadPoolExecutor(2) as workers:
                first = workers.submit(pool.acquire, nbytes=64)
                try:
                    self.assertTrue(started.wait(1))
                    second = workers.submit(pool.acquire, nbytes=64, timeout=0.025)
                    slot = second.result(timeout=1)
                    self.assertIsNotNone(slot)
                    try:
                        self.assertEqual(
                            slot._buffer.storage_offset(), pool._GUARD_BYTES + 64
                        )
                    finally:
                        pool.release(slot)
                    closing = workers.submit(pool.close)
                    self.assertTrue(stopped.wait(1))
                    self.assertFalse(closing.done())
                    client.close.assert_not_called()
                finally:
                    resume.set()
                self.assertIsNone(first.result(timeout=1))
                closing.result(timeout=1)
        client.close.assert_called_once()

    @unittest.skipUnless(
        os.environ.get("MM_EMB_NATIVE_GDR_ARENA") == "1", "native sized arena"
    )
    def test_native_invalid_view_and_stale_ticket_do_not_consume_extents(self):
        pool = self.pool(
            direct_payload=True,
            max_object_bytes=128,
            arena_bytes=256,
            min_block_bytes=64,
        )
        arena = pool._native_arena
        with self.assertRaises(RuntimeError):
            arena.acquire(torch.empty(1, dtype=torch.uint8), 64, False, 0, False)
        self.assert_single_free_range(pool, pool._GUARD_BYTES, 256)
        original = pool.acquire(nbytes=64)
        pool.release(original)
        replacement = pool.acquire(nbytes=64)
        try:
            self.assertEqual(original._extent, replacement._extent)
            with self.assertRaisesRegex(ValueError, "stale"):
                arena.release(original._native_ticket, original._extent[0], True, False)
        finally:
            pool.release(replacement)
        self.assert_single_free_range(pool, pool._GUARD_BYTES, 256)

    def test_sized_release_destroys_its_view_outside_the_allocator_lock(self):
        pool = self.pool(
            direct_payload=True,
            max_object_bytes=128,
            arena_bytes=256,
            min_block_bytes=64,
        )
        destroyed_under_lock = []

        class View:
            def __del__(self):
                destroyed_under_lock.append(pool._condition._is_owned())

        slot = pool.acquire(nbytes=64)
        slot._buffer = View()
        pool.release(slot)
        self.assertIsNone(slot._buffer)
        self.assertEqual(destroyed_under_lock, [False])
        self.assert_single_free_range(pool, pool._GUARD_BYTES, 256)

    def test_sized_leases_coalesce_and_preserve_a_contiguous_read_range(self):
        pool = self.pool(
            direct_payload=True,
            max_object_bytes=256,
            arena_bytes=512,
            min_block_bytes=64,
        )
        first = pool.acquire(nbytes=65, for_write=True)
        middle = pool.acquire(nbytes=19, for_write=True)
        last = pool.acquire(nbytes=31, for_write=True)
        reader = pool.acquire(nbytes=256)
        try:
            self.assertIsNone(pool.acquire(nbytes=1))
            self.assertIsNone(pool.acquire(nbytes=1, for_write=True))
            self.assertEqual(
                [s._capacity for s in (first, middle, last, reader)],
                [128, 64, 64, 256],
            )
            self.assertEqual(
                [
                    s._buffer.storage_offset() - pool._GUARD_BYTES
                    for s in (first, middle, last, reader)
                ],
                [0, 128, 192, 256],
            )
        finally:
            pool.release(last)
            pool.release(middle)
            pool.release(first)
        try:
            merged = pool.acquire(nbytes=256, for_write=True)
            self.assertIsNotNone(merged)
            pool.release(merged)
        finally:
            pool.release(reader)
        self.assert_single_free_range(pool, pool._GUARD_BYTES, 512)
        with self.assertRaises(ValueError):
            pool.acquire(nbytes=257)

    def test_sized_leases_concurrently_wait_and_never_overlap(self):
        factory = Mock()
        pool = self.pool(
            factory=factory,
            slots=1,
            direct_payload=True,
            max_object_bytes=256,
            arena_bytes=512,
            min_block_bytes=64,
        )
        worker_gate = threading.Barrier(4)
        ranges = []

        def lease(size, for_write):
            slot = pool.acquire(nbytes=size, for_write=for_write, timeout=1)
            self.assertIsNotNone(slot)
            try:
                start = slot._buffer.storage_offset()
                ranges.append((start, start + slot._capacity))
                worker_gate.wait(timeout=2)
            finally:
                pool.release(slot)

        with ThreadPoolExecutor(4) as executor:
            jobs = [
                executor.submit(lease, size, for_write)
                for size, for_write in ((128, True), (64, True), (64, False), (128, False))
            ]
            for job in jobs:
                job.result(timeout=4)
        ranges.sort()
        for (_, end), (start, _) in zip(ranges, ranges[1:]):
            self.assertLessEqual(end, start)
        factory.assert_called_once_with(gpu_tensors=(pool._arena,))
        self.assert_single_free_range(pool, pool._GUARD_BYTES, 512)

    def test_slow_lease_metrics_do_not_block_another_lease(self):
        for delayed_delta in (1, -1):
            with self.subTest(delta=delayed_delta):
                measuring = threading.Event()
                finish = threading.Event()

                def observe(for_write, delta, nbytes):
                    if for_write and delta == delayed_delta:
                        measuring.set()
                        finish.wait(timeout=4)

                pool = self.pool(slots=2, direct_payload=True, observe_lease=observe)

                def write():
                    slot = pool.acquire(for_write=True)
                    self.assertIsNotNone(slot)
                    pool.release(slot)

                with ThreadPoolExecutor(2) as executor:
                    writer = executor.submit(write)
                    try:
                        self.assertTrue(measuring.wait(timeout=2))
                        reader = executor.submit(pool.acquire)
                        slot = reader.result(timeout=1)
                        self.assertIsNotNone(slot)
                        pool.release(slot)
                    finally:
                        finish.set()
                    writer.result(timeout=2)

    def test_sized_quarantine_keeps_only_the_unsafe_range_unavailable(self):
        pool = self.pool(
            direct_payload=True,
            max_object_bytes=128,
            arena_bytes=256,
            min_block_bytes=64,
        )
        slot = pool.acquire(nbytes=33)
        start = slot._buffer.storage_offset()
        client = pool._client
        client.load_from_meta.side_effect = TimeoutError("read deadline")
        client.buffer_safe_after_failure.return_value = False
        with self.assertLogs(level="ERROR"), self.assertRaises(TimeoutError):
            slot.load_object_from_meta(
                SimpleNamespace(size_bytes=33), consumer_stream=None
            )
        pool.release(slot)
        slot._quarantine_started -= 1000
        pool._gc_tick()
        self.assertTrue(slot.quarantined)
        safe = pool.acquire(nbytes=100)
        try:
            self.assertGreaterEqual(safe._buffer.storage_offset(), start + 64)
            client.buffer_safe_after_failure.return_value = True
            with self.assertRaises(TimeoutError):
                safe.load_object_from_meta(
                    SimpleNamespace(size_bytes=100), consumer_stream=None
                )
        finally:
            pool.release(safe)
        self.assert_single_free_range(pool, start + 64, 192)
        pool.close()
        client.close.assert_not_called()

    def test_completed_transfer_after_deadline_skips_restore_without_quarantine(self):
        pool = self.pool(direct_payload=True)
        slot = pool.acquire()
        now = [1.0]

        def finish_load(*args, **kwargs):
            now[0] = 3.0

        pool._client.load_from_meta.side_effect = finish_load
        try:
            with patch.object(time, "monotonic", side_effect=lambda: now[0]), patch(
                "rtp_llm.multimodal.kvcm.tensor_object.unpack_object"
            ) as unpack:
                with self.assertRaisesRegex(TimeoutError, "expired before restore"):
                    slot.load_object_from_meta(
                        SimpleNamespace(size_bytes=16),
                        consumer_stream=None,
                        deadline=2.0,
                    )
                unpack.assert_not_called()
            self.assertFalse(slot.quarantined)
        finally:
            pool.release(slot)
        reused = pool.acquire()
        self.assertIs(reused, slot)
        pool.release(reused)

    def test_blocking_completion_wait_preserves_the_stream_fence(self):
        pool = self.pool(direct_payload=True)
        pool.blocking_cuda_events = True
        slot = pool.acquire()
        try:
            with patch("torch.cuda.Event") as make_event:
                slot._wait_for_stream_completion()
            make_event.assert_called_once_with(blocking=True)
            slot._stream.record_event.assert_called_once_with(make_event.return_value)
            slot._stream.record_event.return_value.synchronize.assert_called_once()
        finally:
            pool.release(slot)

    def test_observes_contention_and_copies_without_extra_cuda_sync(self):
        transport = self.transport()
        stages = {}

        @contextmanager
        def observe(operation, stage):
            started = time.perf_counter()
            yield
            stages[stage] = (time.perf_counter() - started) * 1000

        transport._observe_stage = observe
        transport._client.load_one.side_effect = lambda key, value, **kw: value.fill_(7)
        target = torch.zeros(16, dtype=torch.uint8)
        transport._lock.acquire()
        timer = threading.Timer(0.025, transport._lock.release)
        timer.start()
        sync = torch.cuda.current_stream().synchronize
        before = sync.call_count
        try:
            transport.load_one("key", target)
        finally:
            timer.join()
        self.assertTrue(torch.equal(target, torch.full_like(target, 7)))
        self.assertGreaterEqual(stages["transport_lock_wait"], 10)
        self.assertTrue(
            {"native_load", "gpu_ready_wait", "d2h_staging", "host_copy"} <= set(stages)
        )
        self.assertEqual(sync.call_count - before, 2)
        transport._client.load_one.assert_called_once()
        transport._owner.close()

    def test_gc_returns_expired_tensor_to_registered_pool_after_lease_release(self):
        pool = self.pool()
        slot = pool._slots[0]
        arena, client = slot._buffer, slot._client
        slot._client.load_one.side_effect = TimeoutError("unconfirmed completion")
        with self.assertLogs(level="ERROR"), self.assertRaises(TimeoutError):
            slot.load_one("old-key", torch.zeros(16, dtype=torch.uint8))
        slot._quarantine_started -= 299
        pool._gc_tick()
        self.assertTrue(slot.quarantined)
        slot._quarantine_started -= 2
        pool._in_use[slot] = False
        pool._gc_tick()
        self.assertTrue(slot.quarantined)
        pool.release(slot)
        pool._gc_tick()
        self.assertFalse(slot.quarantined)
        self.assertIs(pool.acquire(), slot)
        pool.release(slot)
        self.assertIs(slot._buffer, arena)
        self.assertIs(slot._client, client)
        client.close.assert_not_called()
        client.load_one.side_effect = None
        slot.load_one("new-key", torch.zeros(16, dtype=torch.uint8))
        self.assertEqual(client.load_one.call_count, 2)
        self.assertEqual(RegisteredObjectTransport._quarantined_transports, [])
        self.assertIn(
            "gc_success", [call.args[1] for call in pool._observe_hold.call_args_list]
        )

    def test_direct_slot_only_reuses_native_confirmed_quiescent_io(self):
        pool = self.pool(direct_payload=True)
        slot = pool.acquire()
        slot._client.buffer_safe_after_failure.return_value = True
        slot._client.load_from_meta.side_effect = TimeoutError("safe SDK failure")
        meta = SimpleNamespace(size_bytes=16)
        with self.assertRaises(TimeoutError):
            slot.load_object_from_meta(meta, consumer_stream=None)
        self.assertFalse(slot.quarantined)
        slot._client.buffer_safe_after_failure.return_value = False
        with self.assertLogs(level="ERROR"), self.assertRaises(TimeoutError):
            slot.load_object_from_meta(meta, consumer_stream=None)
        pool.release(slot)
        slot._quarantine_started -= 3600
        pool._gc_tick()
        self.assertTrue(slot.quarantined)
        self.assertIsNone(pool.acquire())

    def test_gc_does_not_recycle_cuda_failures_or_contended_slots(self):
        pool = self.pool()
        slot = pool._slots[0]
        with self.assertLogs(level="ERROR"):
            slot._quarantine(
                TimeoutError("GPU error"), stage="gpu_copy", operation="read"
            )
        slot.device_failed = True
        slot._quarantine_started -= 301
        pool._gc_tick()
        self.assertTrue(slot.quarantined)
        slot.device_failed = False
        slot._quarantine_stage = "sdk_transfer"
        with slot._lock:
            pool._gc_tick()
        self.assertTrue(slot.quarantined)
        pool._gc_tick()
        self.assertFalse(slot.quarantined)

    def test_closed_pool_gc_unregisters_before_releasing_tensor_and_pool(self):
        pool = self.pool(slots=2)
        slot = pool.acquire()
        client, arena, allocator = slot._client, pool._arena, pool._allocator
        with self.assertLogs(level="ERROR"):
            slot._quarantine(
                TimeoutError("unknown IO"), stage="sdk_transfer", operation="read"
            )
        pool.release(slot)
        other = pool.acquire()
        self.assertIs(other, pool._slots[1])
        with ThreadPoolExecutor(1) as worker:
            closing = worker.submit(pool.close)
            try:
                self.assertFalse(closing.done())
                client.close.assert_not_called()
            finally:
                pool.release(other)
            closing.result(timeout=3)
        client.close.assert_not_called()
        self.assertFalse(pool._gc_stop.is_set())

        def close_native():
            self.assertIs(pool._arena, arena)
            self.assertIs(pool._allocator, allocator)
            self.assertTrue(all(s._buffer is not None for s in pool._slots))

        client.close.side_effect = close_native
        slot._quarantine_started -= 301
        pool._gc_tick()
        client.close.assert_called_once()
        self.assertIsNone(pool._arena)
        self.assertIsNone(pool._allocator)
        self.assertIsNone(slot._client)
        self.assertTrue(all(s._buffer is None for s in pool._slots))

    def test_closed_pool_counts_every_slot_after_staggered_expiry(self):
        pool = self.pool(slots=2)
        client = pool._client
        first, second = pool._slots
        with self.assertLogs(level="ERROR"):
            first._quarantine(
                TimeoutError("first"), stage="sdk_transfer", operation="read"
            )
            second._quarantine(
                TimeoutError("second"), stage="sdk_transfer", operation="write"
            )
        pool.close()
        first._quarantine_started -= 301
        pool._gc_tick()
        client.close.assert_not_called()
        self.assertEqual(pool._pending_gc_success, ["read"])
        self.assertNotIn(
            "gc_success", [c.args[1] for c in pool._observe_hold.call_args_list]
        )
        second._quarantine_started -= 301
        pool._gc_tick()
        client.close.assert_called_once()
        self.assertEqual(pool._pending_gc_success, [])
        self.assertEqual(
            [
                c.args[0]
                for c in pool._observe_hold.call_args_list
                if c.args[1] == "gc_success"
            ],
            ["read", "write"],
        )
        pool._gc_tick()
        self.assertEqual(
            sum(c.args[1] == "gc_success" for c in pool._observe_hold.call_args_list), 2
        )

    def test_background_gc_honors_configured_ttl(self):
        pool = self.pool(ttl=1)
        slot = pool._slots[0]
        with self.assertLogs(level="ERROR"):
            slot._quarantine(
                TimeoutError("unknown IO"), stage="sdk_transfer", operation="read"
            )
        done = threading.Event()

        def observed(*args):
            if args[1] == "gc_success":
                done.set()

        pool._observe_hold.side_effect = observed
        slot._quarantine_started -= 2
        worker = pool._gc_thread
        self.assertTrue(done.wait(3))
        self.assertFalse(slot.quarantined)
        pool.close()
        worker.join(2)
        self.assertFalse(worker.is_alive())

    def test_transfer_failure_quarantines_arena_and_blocks_reuse(self):
        for operation in ("save_one", "load_one"):
            with self.subTest(operation=operation):
                RegisteredObjectTransport._quarantined_transports.clear()
                transport = self.transport()
                native = getattr(transport._client, operation)
                error = TimeoutError("private credential=do-not-log")
                error.code = SimpleNamespace(name="ER_SDK_TIMEOUT")
                error.reason = "sdk_timeout"
                error.data_transfer_may_be_inflight = True
                native.side_effect = error
                buffer = torch.zeros(16, dtype=torch.uint8)
                with self.assertLogs(level="ERROR") as logs:
                    with self.assertRaises(TimeoutError):
                        getattr(transport, operation)(
                            "key", buffer, trace_id="first-gdr-failure"
                        )
                self.assertEqual(len(logs.output), 1)
                first = json.loads(logs.output[0].split("MM_EMB_GDR_QUARANTINE ", 1)[1])
                self.assertEqual(first["action"], "hold_registered_slot")
                self.assertEqual(first["error_code"], "ER_SDK_TIMEOUT")
                self.assertEqual(first["error_reason"], "sdk_timeout")
                self.assertTrue(first["data_transfer_may_be_inflight"])
                self.assertEqual(first["key"], "key")
                self.assertEqual(first["trace_id"], "first-gdr-failure")
                self.assertEqual(first["reason"], "gdr_transport_quarantined")
                self.assertEqual(first["stage"], "sdk_transfer")
                self.assertEqual(first["nbytes"], 16)
                self.assertNotIn("private credential=do-not-log", logs.output[0])
                self.assertTrue(transport._poisoned)
                self.assertIn(
                    transport, RegisteredObjectTransport._quarantined_transports
                )
                arena, pool, client = (
                    transport._buffer,
                    transport._owner._allocator,
                    transport._client,
                )
                with self.assertNoLogs(level="ERROR"):
                    for method in ("load_one", "save_one"):
                        with self.assertRaises(RuntimeError):
                            getattr(transport, method)("key", buffer)
                    with self.assertRaises(
                        RegisteredTransportUnavailableError
                    ) as rejected:
                        transport.object_size("key")
                self.assertEqual(
                    rejected.exception.quarantine_id, first["quarantine_id"]
                )
                self.assertEqual(
                    rejected.exception.initial_failure["error_code"], "ER_SDK_TIMEOUT"
                )
                self.assertTrue(rejected.exception.restart_required)
                native.assert_called_once()
                transport._owner.close()
                transport._owner.close()
                client.close.assert_not_called()
                self.assertIs(transport._buffer, arena)
                self.assertIs(transport._owner._allocator, pool)

    def test_quarantine_distinguishes_gpu_copy_from_post_sdk_wait(self):
        for operation, expected_stage in (
            ("save_one", "gpu_copy"),
            ("load_one", "gpu_ready_wait"),
        ):
            with self.subTest(operation=operation):
                RegisteredObjectTransport._quarantined_transports.clear()
                transport = self.transport()
                torch.cuda.current_stream().synchronize.side_effect = RuntimeError(
                    "CUDA event synchronization failed"
                )
                try:
                    with self.assertLogs(level="ERROR") as logs, self.assertRaises(
                        RuntimeError
                    ):
                        getattr(transport, operation)(
                            "key", torch.zeros(16, dtype=torch.uint8)
                        )
                    first = json.loads(
                        logs.output[0].split("MM_EMB_GDR_QUARANTINE ", 1)[1]
                    )
                    self.assertEqual(first["stage"], expected_stage)
                    self.assertEqual(
                        first["error_message"], "CUDA event synchronization failed"
                    )
                finally:
                    torch.cuda.current_stream().synchronize.side_effect = None

    def test_proven_quiescent_sdk_failure_allows_the_next_operation(self):
        for operation in ("save_one", "load_one"):
            with self.subTest(operation=operation):
                transport = self.transport()
                native = getattr(transport._client, operation)
                native.side_effect = RuntimeError(
                    "metadata RPC failed without pending DMA"
                )
                transport._client.buffer_safe_after_failure.return_value = True
                buffer = torch.zeros(16, dtype=torch.uint8)
                with self.assertRaisesRegex(RuntimeError, "metadata RPC failed"):
                    getattr(transport, operation)("key", buffer)
                self.assertTrue(transport.is_transfer_quiescent(native.side_effect))
                self.assertFalse(transport._poisoned)
                self.assertEqual(RegisteredObjectTransport._quarantined_transports, [])
                native.side_effect = None
                getattr(transport, operation)("key", buffer)
                self.assertEqual(native.call_count, 2)
                client = transport._client
                transport._owner.close()
                client.close.assert_called_once()

    def test_unavailable_reason_distinguishes_closed_from_quarantined(self):
        for poisoned in (False, True):
            with self.subTest(poisoned=poisoned):
                transport = self.transport()
                client = transport._client
                transport._poisoned = poisoned
                transport._owner.close()
                with self.assertRaises(RegisteredTransportUnavailableError) as raised:
                    transport.object_size("key")
                self.assertEqual(
                    raised.exception.reason,
                    "gdr_transport_quarantined" if poisoned else "gdr_transport_closed",
                )
                self.assertEqual(
                    str(raised.exception),
                    "registered transport unavailable; restart worker after GDR failure",
                )
                client.object_size.assert_not_called()

    def test_failed_safety_check_does_not_hide_the_original_error(self):
        transport = self.transport()
        transport._client.save_one.side_effect = TimeoutError(
            "original transfer failure"
        )
        transport._client.buffer_safe_after_failure.side_effect = RuntimeError(
            "bad proof"
        )
        with self.assertLogs(level="ERROR"):
            with self.assertRaisesRegex(TimeoutError, "original transfer failure"):
                transport.save_one("key", torch.zeros(16, dtype=torch.uint8))
        self.assertTrue(transport._poisoned)

    def test_local_validation_and_metadata_error_do_not_poison(self):
        transport = self.transport()
        with self.assertRaises(ValueError):
            transport.load_one("key", torch.empty(17, dtype=torch.uint8))
        transport._client.object_size.side_effect = TimeoutError("metadata unavailable")
        with self.assertRaises(TimeoutError):
            transport.object_size("key")
        self.assertFalse(transport._poisoned)
        self.assertEqual(RegisteredObjectTransport._quarantined_transports, [])
        client = transport._client
        transport._owner.close()
        client.close.assert_called_once()
        self.assertIsNone(transport._buffer)

    def test_safe_failure_reuses_same_arena_for_next_request(self):
        for method in ("load_one", "save_one"):
            with self.subTest(method=method):
                slot = self.transport()
                native = getattr(slot._client, method)
                failure = RuntimeError("safe SDK error")
                native.side_effect = [failure, None]
                slot._client.buffer_safe_after_failure.side_effect = (
                    lambda exc: exc is failure
                )
                payload = torch.arange(16, dtype=torch.uint8)
                with self.assertRaises(RuntimeError):
                    getattr(slot, method)("a", payload)
                getattr(slot, method)("b", payload)
                first, second = (call.args[1] for call in native.call_args_list)
                self.assertEqual(first.data_ptr(), second.data_ptr())
                self.assertFalse(slot.quarantined)
                slot._owner.close()

    def test_failed_close_keeps_allocator_and_registration_owner(self):
        pool = self.pool(slots=2)
        slot = pool._slots[0]
        arena, allocator, client = pool._arena, pool._allocator, pool._client
        with self.assertLogs(level="ERROR"):
            slot._quarantine(TimeoutError("unknown IO"), stage="sdk_transfer")
        pool.close()
        slot._quarantine_started -= 301
        client.close.side_effect = RuntimeError("unregister failed")
        with self.assertLogs(level="ERROR"):
            pool._gc_tick()
        pool._gc_tick()
        pool.close()
        client.close.assert_called_once()
        self.assertIs(pool._arena, arena)
        self.assertIs(pool._allocator, allocator)
        self.assertEqual(RegisteredObjectTransport._quarantined_transports, pool._slots)
        self.assertNotIn(
            "gc_success", [call.args[1] for call in pool._observe_hold.call_args_list]
        )
        self.assertIn(
            "gc_error", [call.args[1] for call in pool._observe_hold.call_args_list]
        )

    def test_constructor_failure_is_retained_and_blocks_new_allocations(self):
        arena = torch.empty(65552, dtype=torch.uint8)
        factory = Mock(side_effect=RuntimeError("registration failed"))
        with patch("torch.cuda.MemPool") as pool, patch(
            "torch.cuda.use_mem_pool", return_value=nullcontext()
        ), patch("torch.empty", return_value=arena) as allocate:
            with self.assertRaises(RuntimeError):
                RegisteredTransportPool(factory, max_object_bytes=16, device="cuda:0")
            retained = RegisteredObjectTransport._quarantined_transports[0]
            self.assertIs(retained._owner._arena, arena)
            self.assertIs(retained._owner._allocator, pool.return_value)
            with self.assertRaisesRegex(RuntimeError, "quarantined") as rejected:
                RegisteredTransportPool(factory, max_object_bytes=16, device="cuda:0")
            self.assertEqual(rejected.exception.reason, "gdr_transport_quarantined")
            self.assertEqual(
                rejected.exception.initial_failure["error_type"], "RuntimeError"
            )
            self.assertEqual(rejected.exception.quarantine_id, retained._quarantine_id)
            self.assertEqual(allocate.call_count, 2)
            factory.assert_called_once()

    def test_sync_memops_failure_prevents_client_registration(self):
        self.enable_memops.side_effect = RuntimeError("SYNC_MEMOPS failed")
        factory = Mock()
        arena = torch.empty(65552, dtype=torch.uint8)
        with patch("torch.cuda.MemPool"), patch(
            "torch.cuda.use_mem_pool", return_value=nullcontext()
        ), patch("torch.empty", return_value=arena):
            with self.assertRaisesRegex(RuntimeError, "SYNC_MEMOPS"):
                RegisteredTransportPool(factory, max_object_bytes=16, device="cuda:0")
        factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
