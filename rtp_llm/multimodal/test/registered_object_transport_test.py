import ctypes
import threading
import time
import unittest
from contextlib import contextmanager, nullcontext
from unittest.mock import Mock, patch

import torch
from rtp_llm.multimodal.kvcm.registered_transport import (
    RegisteredObjectTransport,
    _enable_sync_memops,
)
from torch.utils._python_dispatch import TorchDispatchMode


class CudaRdmaSyncMemopsTest(unittest.TestCase):
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
                RegisteredObjectTransport(
                    forbidden, max_object_bytes=capacity, device="cuda:0"
                )
        with self.assertRaises(ValueError):
            RegisteredObjectTransport(forbidden, max_object_bytes=1024, device="cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_registered_range_roundtrip_bounds_and_close(self):
        seen = {}

        class Client:
            instance_id = "test"
            instance_group = "test"

            def __init__(self, gpu_tensors):
                self.arena = gpu_tensors[0]
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

        transport = RegisteredObjectTransport(
            Client, max_object_bytes=1024, device="cuda:0"
        )
        copies = []

        class ObserveCopies(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                if func == torch.ops.aten.copy_.default:
                    dst, src = args[:2]
                    if dst.is_cuda != src.is_cuda:
                        host = src if dst.is_cuda else dst
                        copies.append((dst.is_cuda, host.is_pinned(), host.data_ptr()))
                return func(*args, **(kwargs or {}))

        host_ptr = transport._host_staging.data_ptr()
        with ObserveCopies():
            for size in (1024, 17):
                source = torch.arange(size).to(torch.uint8)
                target = torch.empty_like(source)
                transport.save_one("x", source)
                self.assertIs(transport.load_one("x", target), target)
                self.assertTrue(torch.equal(source, target))
        self.assertEqual(copies, [(True, True, host_ptr), (False, True, host_ptr)] * 2)
        for bad in (
            torch.empty(1025, dtype=torch.uint8),
            torch.empty(0, dtype=torch.uint8),
            torch.empty(3),
            torch.empty((2, 2), dtype=torch.uint8),
        ):
            with self.assertRaises(ValueError):
                transport.save_one("x", bad)
        transport.close()
        transport.close()
        self.assertEqual(seen["client"].closed, 1)
        self.assertIsNone(transport._arena)
        self.assertIsNone(transport._host_staging)
        with self.assertRaises(RuntimeError):
            transport.load_one("x", target)


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
        self.stream = patch("torch.cuda.current_stream", return_value=Mock())
        self.stream.start()
        self.addCleanup(self.stream.stop)

    def transport(self):
        transport = object.__new__(RegisteredObjectTransport)
        transport._lock = threading.Lock()
        transport._closed = transport._poisoned = False
        transport._device = torch.device("cuda:0")
        transport._capacity = 16
        transport._host_staging = torch.empty(16, dtype=torch.uint8)
        transport._arena = torch.empty(transport._GUARD_BYTES + 16, dtype=torch.uint8)
        transport._pool = object()
        transport._client = Mock()
        return transport

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
        transport.close()

    def test_transfer_failure_quarantines_arena_and_blocks_reuse(self):
        for operation in ("save_one", "load_one"):
            with self.subTest(operation=operation):
                transport = self.transport()
                native = getattr(transport._client, operation)
                native.side_effect = TimeoutError("DMA completion is unknown")
                buffer = torch.zeros(16, dtype=torch.uint8)
                with self.assertLogs(level="ERROR") as logs:
                    with self.assertRaises(TimeoutError):
                        getattr(transport, operation)("key", buffer)
                self.assertEqual(len(logs.output), 1)
                self.assertIn("TimeoutError: DMA completion is unknown", logs.output[0])
                self.assertTrue(transport._poisoned)
                self.assertIn(
                    transport, RegisteredObjectTransport._quarantined_transports
                )
                arena, pool, client = (
                    transport._arena,
                    transport._pool,
                    transport._client,
                )
                for method in ("load_one", "save_one"):
                    with self.assertRaises(RuntimeError):
                        getattr(transport, method)("key", buffer)
                with self.assertRaises(RuntimeError):
                    transport.object_size("key")
                native.assert_called_once()
                transport.close()
                transport.close()
                client.close.assert_not_called()
                self.assertIs(transport._arena, arena)
                self.assertIs(transport._pool, pool)

    def test_proven_quiescent_sdk_failure_allows_the_next_operation(self):
        for operation in ("save_one", "load_one"):
            with self.subTest(operation=operation):
                transport = self.transport()
                native = getattr(transport._client, operation)
                native.side_effect = RuntimeError(
                    "metadata RPC failed without pending DMA"
                )
                transport._client.is_transfer_quiescent.return_value = True
                buffer = torch.zeros(16, dtype=torch.uint8)
                with self.assertRaisesRegex(RuntimeError, "metadata RPC failed"):
                    getattr(transport, operation)("key", buffer)
                self.assertTrue(transport.is_transfer_quiescent(native.side_effect))
                self.assertFalse(transport._poisoned)
                self.assertEqual(RegisteredObjectTransport._quarantined_transports, [])
                native.side_effect = None
                getattr(transport, operation)("key", buffer)
                self.assertEqual(native.call_count, 2)
                transport.close()
                transport._client.close.assert_called_once()

    def test_failed_safety_check_does_not_hide_the_original_error(self):
        transport = self.transport()
        transport._client.save_one.side_effect = TimeoutError(
            "original transfer failure"
        )
        transport._client.is_transfer_quiescent.side_effect = RuntimeError("bad proof")
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
        transport.close()
        transport._client.close.assert_called_once()
        self.assertIsNone(transport._arena)

    def test_failed_close_keeps_allocator_and_registration_owner(self):
        transport = self.transport()
        arena, pool = transport._arena, transport._pool
        transport._client.close.side_effect = RuntimeError("unregister failed")
        with self.assertRaises(RuntimeError):
            transport.close()
        transport.close()
        transport._client.close.assert_called_once()
        self.assertIs(transport._arena, arena)
        self.assertIs(transport._pool, pool)
        self.assertIn(transport, RegisteredObjectTransport._quarantined_transports)

    def test_constructor_failure_is_retained_and_blocks_new_allocations(self):
        arena = torch.empty(65552, dtype=torch.uint8)
        factory = Mock(side_effect=RuntimeError("registration failed"))
        with patch("torch.cuda.MemPool") as pool, patch(
            "torch.cuda.use_mem_pool", return_value=nullcontext()
        ), patch("torch.empty", return_value=arena) as allocate:
            with self.assertRaises(RuntimeError):
                RegisteredObjectTransport(factory, max_object_bytes=16, device="cuda:0")
            retained = RegisteredObjectTransport._quarantined_transports[0]
            self.assertIs(retained._arena, arena)
            self.assertIs(retained._pool, pool.return_value)
            with self.assertRaisesRegex(RuntimeError, "quarantined"):
                RegisteredObjectTransport(factory, max_object_bytes=16, device="cuda:0")
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
                RegisteredObjectTransport(factory, max_object_bytes=16, device="cuda:0")
        factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
