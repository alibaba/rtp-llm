import gc
import json
import struct
import unittest
import weakref
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import torch
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
