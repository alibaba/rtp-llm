import threading
import unittest
from contextlib import nullcontext
from unittest.mock import Mock, patch

import torch

from rtp_llm.multimodal.kvcm.registered_transport import RegisteredObjectTransport


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
        source = torch.arange(1024).to(torch.uint8)
        target = torch.empty_like(source)
        transport.save_one("x", source)
        transport.load_one("x", target)
        self.assertTrue(torch.equal(source, target))
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
        with self.assertRaises(RuntimeError):
            transport.load_one("x", target)


class RegisteredTransportFailureTest(unittest.TestCase):
    """Exercise failure ownership without claiming actual CUDA/RDMA coverage."""

    def setUp(self):
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
        transport._arena = torch.empty(transport._GUARD_BYTES + 16, dtype=torch.uint8)
        transport._pool = object()
        transport._client = Mock()
        return transport

    def test_transfer_failure_quarantines_arena_and_blocks_reuse(self):
        for operation in ("save_one", "load_one"):
            with self.subTest(operation=operation):
                transport = self.transport()
                native = getattr(transport._client, operation)
                native.side_effect = TimeoutError("DMA completion is unknown")
                buffer = torch.zeros(16, dtype=torch.uint8)
                with self.assertRaises(TimeoutError):
                    getattr(transport, operation)("key", buffer)
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
            allocate.assert_called_once()
            factory.assert_called_once()


if __name__ == "__main__":
    unittest.main()
