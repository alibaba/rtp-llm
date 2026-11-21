import gc
import json
import os
import struct
import unittest
import weakref
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import torch
from rtp_llm.multimodal.kvcm import tensor_object
from rtp_llm.multimodal.kvcm.tensor_object import (
    pack_object,
    plan_object,
    unpack_object,
)


class TensorObjectTest(unittest.TestCase):
    def roundtrip(self, value):
        return unpack_object(pack_object(plan_object(value)))

    def test_observed_codec_roundtrip_and_checksum_failure_boundaries(self):
        seen = []

        @contextmanager
        def observe(operation, stage):
            try:
                yield
            except Exception:
                seen.append((operation, stage, "error"))
                raise
            else:
                seen.append((operation, stage, "success"))

        value = (torch.arange(12).reshape(3, 4), {"x": torch.ones(2)})
        blob = pack_object(plan_object(value), observe_stage=observe)
        restored = unpack_object(blob, observe_stage=observe)
        self.assertTrue(torch.equal(restored[0], value[0]))
        self.assertTrue(torch.equal(restored[1]["x"], value[1]["x"]))
        self.assertIn(("write", "pack_payload", "success"), seen)
        self.assertIn(("read", "checksum", "success"), seen)
        self.assertNotIn(("read", "h2d_restore", "success"), seen)
        seen.clear()
        blob[-1] ^= 1
        with self.assertRaisesRegex(ValueError, "checksum"):
            unpack_object(blob, observe_stage=observe)
        self.assertEqual(
            seen, [("read", "decode_header", "success"), ("read", "checksum", "error")]
        )

    def test_new_format_has_no_full_payload_hash_and_validates_ranges(self):
        value = torch.arange(12, dtype=torch.int32).reshape(3, 4).t()
        with patch(
            "rtp_llm.multimodal.kvcm.tensor_object.hashlib.sha256",
            side_effect=AssertionError("new codec must not hash the payload"),
        ):
            plan = plan_object(
                {"cache_key": "key", "result": (value, value)}, checksum=False
            )
            packed = pack_object(plan)
            restored = unpack_object(packed, expected_key="key")["result"]
            with self.assertRaisesRegex(ValueError, "key mismatch"):
                unpack_object(packed, expected_key="other-key")
            self.assertEqual(bytes(packed[:8]), b"RTPMM002")
            self.assertIs(restored[0], restored[1])
            torch.testing.assert_close(restored[0], value)
            with self.assertRaisesRegex(ValueError, "range"):
                unpack_object(packed[:-1])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_cuda_reads_short_header_once_and_fetches_large_header_suffix(self):
        self._assert_cuda_header_reads(blocking_cuda_events=False)

    @unittest.skipUnless(torch.cuda.is_available() and torch.version.cuda, "CUDA required")
    @unittest.skipUnless(
        os.environ.get("MM_EMB_NATIVE_RESTORE") == "1", "native restore required"
    )
    def test_native_cuda_header_reads_on_current_stream(self):
        import librtp_compute_ops

        with patch.object(
            librtp_compute_ops,
            "read_emb_cuda_object_header",
            wraps=librtp_compute_ops.read_emb_cuda_object_header,
        ) as read_object_header, torch.cuda.stream(torch.cuda.Stream()):
            self._assert_cuda_header_reads(
                blocking_cuda_events=True, single_binding=True
            )
        self.assertEqual(read_object_header.call_count, 3)

        with patch.object(
            librtp_compute_ops,
            "read_emb_cuda_header",
            wraps=librtp_compute_ops.read_emb_cuda_header,
        ) as read_header, patch.object(
            librtp_compute_ops, "read_emb_cuda_object_header", None
        ), torch.cuda.stream(torch.cuda.Stream()):
            self._assert_cuda_header_reads(blocking_cuda_events=True)
        self.assertEqual(read_header.call_count, 4)
        self.assertEqual(
            [call.args[1] for call in read_header.call_args_list], [0, 0, 1024, 0]
        )

    @unittest.skipUnless(torch.cuda.is_available() and torch.version.cuda, "CUDA required")
    @unittest.skipUnless(
        os.environ.get("MM_EMB_NATIVE_RESTORE") == "1", "native restore required"
    )
    def test_native_payload_fence_completes_before_releasing_registered_buffer(self):
        import librtp_compute_ops

        gpu = torch.arange(4096, dtype=torch.int16, device="cuda")
        cpu = torch.arange(8, dtype=torch.int32)
        value = {"cache_key": "key", "result": (gpu, cpu, gpu)}
        plan = plan_object(value, checksum=False)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        consumer = torch.cuda.current_stream()
        with torch.cuda.stream(stream):
            packed = pack_object(
                plan, out=torch.empty(plan.nbytes, dtype=torch.uint8, device="cuda")
            )
            with (
                patch.object(tensor_object, "_USE_NATIVE_OBJECT_RESTORE", False),
                patch.object(
                    librtp_compute_ops,
                    "restore_emb_cuda_tensor_slices_and_wait",
                    wraps=librtp_compute_ops.restore_emb_cuda_tensor_slices_and_wait,
                ) as native_fence,
            ):
                restored_gpu, restored_cpu, alias = unpack_object(
                    packed,
                    expected_key="key",
                    blocking_cuda_events=True,
                    consumer_stream=consumer,
                    payload_fence=lambda: self.fail("Python fence must not run"),
                )["result"]
            native_fence.assert_called_once()
            self.assertTrue(stream.query())
            packed.zero_()
        self.assertIs(restored_gpu, alias)
        torch.testing.assert_close(restored_gpu, gpu)
        torch.testing.assert_close(restored_cpu, cpu)
        stream.synchronize()

    @unittest.skipUnless(torch.cuda.is_available() and torch.version.cuda, "CUDA required")
    @unittest.skipUnless(
        os.environ.get("MM_EMB_NATIVE_RESTORE") == "1", "native restore required"
    )
    def test_complete_native_restore_and_python_scalar_fallback(self):
        import librtp_compute_ops

        gpu = torch.arange(1024, dtype=torch.bfloat16, device="cuda")
        cpu = torch.arange(9, dtype=torch.int32)
        consumer = torch.cuda.current_stream()
        stream = torch.cuda.Stream()
        stream.wait_stream(consumer)
        with (
            patch.object(tensor_object, "_USE_NATIVE_OBJECT_RESTORE", True),
            patch.object(
                librtp_compute_ops,
                "read_emb_cuda_object_header",
                wraps=librtp_compute_ops.read_emb_cuda_object_header,
            ) as separate_header,
            torch.cuda.stream(stream),
        ):
            for scalar in ("code\x00文本", 1 << 100, 0.1):
                value = {"cache_key": "original", "items": [gpu, cpu, gpu, scalar]}
                plan = plan_object(value, checksum=False)
                wire = pack_object(
                    plan, out=torch.empty(plan.nbytes, device="cuda", dtype=torch.uint8)
                )
                native, fallback = librtp_compute_ops.restore_emb_cuda_object(
                    wire, "original", True, consumer.stream_id
                )
                self.assertEqual(fallback is None, type(scalar) is str)
                if native is not None:
                    self.assertTrue(stream.query())
                result = unpack_object(
                    wire,
                    expected_key="original",
                    blocking_cuda_events=True,
                    consumer_stream=consumer,
                    payload_fence=lambda: self.fail("native copies are already fenced"),
                )
                restored_gpu, restored_cpu, alias, marker = result["items"]
                self.assertIs(restored_gpu, alias)
                self.assertIs(type(marker), type(scalar))
                self.assertEqual(marker, scalar)
                torch.testing.assert_close(restored_gpu, gpu)
                torch.testing.assert_close(restored_cpu, cpu)
            with self.assertRaisesRegex(ValueError, "cache key mismatch"):
                unpack_object(
                    wire,
                    expected_key="wrong",
                    blocking_cuda_events=True,
                    payload_fence=lambda: self.fail("invalid key must not copy tensors"),
                )
        separate_header.assert_not_called()

    def _assert_cuda_header_reads(self, *, blocking_cuda_events, single_binding=False):
        stages = []

        @contextmanager
        def observe(operation, stage):
            stages.append((operation, stage))
            yield

        tensor = torch.arange(2048, device="cuda")
        for metadata_size, expected_copies in ((64, 1), (4096, 2)):
            if single_binding:
                expected_copies = 1
            with self.subTest(metadata_size=metadata_size):
                value = {"cache_key": "key", "result": (tensor, "x" * metadata_size)}
                plan = plan_object(value, checksum=False)
                packed = pack_object(
                    plan, out=torch.empty(plan.nbytes, dtype=torch.uint8, device="cuda")
                )
                stages.clear()
                restored = unpack_object(
                    packed,
                    observe_stage=observe,
                    expected_key="key",
                    blocking_cuda_events=blocking_cuda_events,
                )
                actual, metadata = restored["result"]
                torch.testing.assert_close(actual, tensor)
                self.assertEqual(metadata, value["result"][1])
                self.assertEqual(
                    stages[: expected_copies + 1],
                    [("read", "d2h_header")] * expected_copies
                    + [("read", "decode_header")],
                )
                self.assertEqual(stages.count(("read", "d2h_header")), expected_copies)
        stages.clear()
        packed[8:16].fill_(255)
        with self.assertRaisesRegex(ValueError, "object header"):
            unpack_object(
                packed, observe_stage=observe, blocking_cuda_events=blocking_cuda_events
            )
        self.assertEqual(stages, [("read", "d2h_header")])

    def test_pack_into_reused_buffer_matches_zero_initialized_wire_bytes(self):
        value = (
            torch.arange(7, dtype=torch.uint8),
            torch.empty(0, dtype=torch.uint8),
            torch.arange(6, dtype=torch.int32)[::2],
            torch.empty(0, dtype=torch.uint8),
        )
        plan = plan_object(value)
        expected = pack_object(plan)
        dirty = torch.full((plan.nbytes + 8,), 0xA7, dtype=torch.uint8)
        packed = pack_object(plan, out=dirty)
        self.assertEqual(packed.data_ptr(), dirty.data_ptr())
        self.assertTrue(torch.equal(packed, expected))
        self.assertEqual(dirty[plan.nbytes:].tolist(), [0xA7] * 8)
        for restored, original in zip(unpack_object(packed), value):
            self.assertTrue(torch.equal(restored, original))

    def test_plan_releases_tensor_without_cyclic_gc(self):
        was_enabled = gc.isenabled()
        gc.disable()
        try:
            tensor = torch.ones(4)
            reference = weakref.ref(tensor)
            plan = plan_object((tensor, None))
            del tensor
            self.assertIsNotNone(reference())
            del plan
            self.assertIsNone(reference())
        finally:
            if was_enabled:
                gc.enable()
            gc.collect()

    def test_rejected_plan_releases_tensor_without_cyclic_gc(self):
        was_enabled = gc.isenabled()
        gc.disable()
        try:
            for invalid in (object(), "x" * 70000):
                with self.subTest(invalid_type=type(invalid).__name__):
                    tensor = torch.ones(4)
                    reference = weakref.ref(tensor)
                    with self.assertRaises(ValueError):
                        plan_object((tensor, invalid))
                    del tensor
                    self.assertIsNone(reference())
        finally:
            if was_enabled:
                gc.enable()
            gc.collect()

    def test_unpack_releases_tensor_table_without_cyclic_gc(self):
        class TensorTable(list):
            pass

        source = {"cache_key": "key", "result": torch.arange(4)}
        packed = pack_object(plan_object(source, checksum=False))
        tables = []
        loads = json.loads

        def track_table(data):
            header = loads(data)
            header["tensors"] = TensorTable(header["tensors"])
            tables.append(weakref.ref(header["tensors"]))
            return header

        was_enabled = gc.isenabled()
        gc.disable()
        try:
            with patch(
                "rtp_llm.multimodal.kvcm.tensor_object.json.loads",
                side_effect=track_table,
            ):
                restored = unpack_object(packed, expected_key="key")
                torch.testing.assert_close(restored["result"], source["result"])
                with self.assertRaisesRegex(ValueError, "key mismatch"):
                    unpack_object(packed, expected_key="different-key")
            self.assertEqual([table() is None for table in tables], [True, True])
        finally:
            if was_enabled:
                gc.enable()
            gc.collect()

    def test_whole_result_mixed_types_empty_and_alias(self):
        emb = torch.arange(24, dtype=torch.bfloat16).reshape(3, 8)
        pos = torch.arange(9, dtype=torch.int64).reshape(3, 3)
        extra = [
            torch.tensor([True, False]),
            torch.tensor([1.5], dtype=torch.float64),
            torch.empty(0, 2),
            None,
            {"grid": pos},
        ]
        result = self.roundtrip((emb, pos, extra))
        self.assertIsInstance(result, tuple)
        self.assertIsInstance(result[2], list)
        self.assertIs(result[1], result[2][4]["grid"])
        self.assertIsNone(result[2][3])
        self.assertEqual(result[2][2].shape, (0, 2))
        for actual, expected in [
            (result[0], emb),
            (result[1], pos),
            (result[2][0], extra[0]),
            (result[2][1], extra[1]),
        ]:
            self.assertEqual(actual.dtype, expected.dtype)
            self.assertTrue(torch.equal(actual, expected))

    def test_views_scalars_and_optional_position(self):
        tensor = torch.arange(12).reshape(3, 4).t()
        result = self.roundtrip(
            ([tensor, torch.tensor(9)], None, {"count": 3, "name": "video"})
        )
        self.assertTrue(torch.equal(result[0][0], tensor))
        self.assertEqual(result[0][1].shape, ())
        self.assertIsNone(result[1])
        self.assertEqual(result[2], {"count": 3, "name": "video"})

    def test_detached_buffer_owns_restored_views(self):
        result = self.roundtrip((torch.ones(2, 4), None))
        self.assertEqual(result[0].sum().item(), 8)

    def test_size_limit_before_pack(self):
        with self.assertRaises(ValueError):
            plan_object((torch.ones(100), None), max_bytes=20)

    def test_rejects_unsupported_value_and_metadata_explosion(self):
        for value in [
            object(),
            {1: torch.zeros(2)},
            float("nan"),
            [None] * 5000,
            "x" * 70000,
        ]:
            with self.subTest(type=type(value)), self.assertRaises(ValueError):
                plan_object(value)

    def test_payload_corruption_and_truncation(self):
        data = pack_object(plan_object((torch.arange(8), None)))
        for corrupted in [data[:-1], data.clone(), data[:5]]:
            if len(corrupted) == len(data):
                corrupted[-1] ^= 1
            with self.assertRaises(ValueError):
                unpack_object(corrupted)

    def test_invalid_shape_never_allocates_destination(self):
        data = pack_object(plan_object((torch.zeros(2, 3), None)))
        length = struct.unpack_from("<Q", memoryview(data.numpy()), 8)[0]
        header = json.loads(bytes(data[16 : 16 + length].tolist()))
        header["tensors"][0]["shape"] = [9, 3]
        encoded = json.dumps(header, separators=(",", ":")).encode()
        self.assertEqual(len(encoded), length)
        data[16 : 16 + length].copy_(
            torch.frombuffer(bytearray(encoded), dtype=torch.uint8)
        )
        with self.assertRaises(ValueError):
            unpack_object(data)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    @unittest.skipUnless(os.environ.get("MM_EMB_NATIVE_PACK") == "1", "native pack required")
    def test_native_pack_retains_headers_until_their_cuda_stream_completes(self):
        payload = torch.arange(256, dtype=torch.int32, device="cuda")
        plans = [
            plan_object({"key": str(index) * 15, "payload": payload}, checksum=False)
            for index in range(64)
        ]
        expected = [pack_object(plan) for plan in plans]
        streams = [torch.cuda.Stream() for _ in range(4)]
        results = []
        for index, plan in enumerate(plans):
            stream = streams[index % len(streams)]
            with torch.cuda.stream(stream):
                output = torch.empty(plan.nbytes, dtype=torch.uint8, device="cuda")
                results.append(pack_object(plan, out=output))
        for stream in streams:
            stream.synchronize()
        for index, (actual, reference) in enumerate(zip(results, expected)):
            with self.subTest(index=index):
                self.assertTrue(torch.equal(actual.cpu(), reference))

    @unittest.skipUnless(torch.cuda.is_available() and torch.version.cuda, "CUDA required")
    @unittest.skipUnless(os.environ.get("MM_EMB_NATIVE_PACK") == "1", "native pack required")
    def test_native_pack_fence_completes_before_the_sdk_can_read_output(self):
        import librtp_compute_ops

        gpu = torch.arange(4096, dtype=torch.int16, device="cuda")
        cpu = torch.arange(7, dtype=torch.int32)
        plan = plan_object((gpu, cpu), checksum=False)
        expected = pack_object(plan)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            output = torch.full((plan.nbytes + 64,), 255, dtype=torch.uint8, device="cuda")
            with patch.object(
                librtp_compute_ops,
                "pack_emb_cuda_tensor_slices_and_wait",
                wraps=librtp_compute_ops.pack_emb_cuda_tensor_slices_and_wait,
            ) as native_fence:
                packed = pack_object(
                    plan,
                    out=output,
                    blocking_cuda_events=True,
                    payload_fence=lambda: self.fail("Python fence must not run"),
                )
            native_fence.assert_called_once()
            self.assertTrue(stream.query())
        torch.testing.assert_close(packed.cpu(), expected)
        self.assertTrue(torch.all(output[plan.nbytes:] == 255))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_cuda_roundtrip_and_offloaded_device_tree(self):
        device = torch.device("cuda", torch.cuda.current_device())
        value = (
            torch.arange(12, device=device).reshape(3, 4),
            None,
            [torch.tensor([2], dtype=torch.int32)],
        )
        result = self.roundtrip(value)
        self.assertEqual(result[0].device, device)
        self.assertTrue(torch.equal(result[0], value[0]))
        self.assertEqual(result[2][0].device.type, "cpu")
        offloaded = (value[0].cpu(), None, value[2])
        devices = (device, None, [torch.device("cpu")])
        restored = unpack_object(pack_object(plan_object(offloaded, devices=devices)))
        self.assertEqual(restored[0].device, device)
        self.assertTrue(torch.equal(restored[0], value[0]))


class ObjectFacadeTest(unittest.TestCase):
    def test_public_save_load_object_uses_one_key_and_preserves_full_result(self):
        from unittest.mock import MagicMock, patch

        from rtp_llm.multimodal.kvcm import client as facade_module
        from rtp_llm.multimodal.kvcm._config import RtpKvMetaObjectClientConfig

        stored = {}
        backend = MagicMock()

        def save(keys, tensors, **kwargs):
            self.assertEqual(list(keys), ["existing_multimodal_key"])
            self.assertEqual(len(tensors), 1)
            stored[keys[0]] = tensors[0].clone()

        def get_meta(key, timeout_ms, **kwargs):
            return SimpleNamespace(key=key, size_bytes=stored[key].numel())

        def load(meta, buffer, **kwargs):
            buffer.copy_(stored[meta.key])

        backend.save.side_effect = save
        backend.get_object_meta.side_effect = get_meta
        backend.load_from_meta.side_effect = load
        config = RtpKvMetaObjectClientConfig(
            addresses=("127.0.0.1:19001",),
            instance_id="kve_test",
            instance_group="kve_test",
            user_data="",
            transfer_client_config="{}",
            call_timeout_ms=100,
            write_timeout_seconds=30,
            max_object_bytes=65536,
        )
        with patch.object(
            facade_module,
            "_load_kvcm_client_types",
            return_value=(
                lambda _: backend,
                lambda **kw: SimpleNamespace(**kw),
                RuntimeError,
                object(),
            ),
        ):
            client = facade_module.RtpKvMetaObjectClient._from_config(config)
        self.addCleanup(client.close)
        result = (
            torch.arange(12, dtype=torch.bfloat16).reshape(3, 4),
            torch.arange(3),
            [torch.tensor([3.5]), torch.tensor([2], dtype=torch.int32)],
        )
        client.save_object("existing_multimodal_key", result)
        actual = client.load_object("existing_multimodal_key", trace_id="read")
        self.assertEqual(backend.get_object_meta.call_count, 1)
        self.assertEqual(backend.load_from_meta.call_count, 1)
        self.assertEqual(backend.get_object_meta.call_args.kwargs["trace_id"], "read")
        self.assertEqual(backend.load_from_meta.call_args.kwargs["trace_id"], "read")
        backend.load.assert_not_called()
        self.assertTrue(torch.equal(actual[0], result[0]))
        self.assertTrue(torch.equal(actual[1], result[1]))
        self.assertTrue(torch.equal(actual[2][0], result[2][0]))
        self.assertTrue(torch.equal(actual[2][1], result[2][1]))
        backend.remove.assert_not_called()


if __name__ == "__main__":
    unittest.main()
