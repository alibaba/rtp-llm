import concurrent.futures
import json
import struct
import unittest
from types import SimpleNamespace

import grpc
import torch

from rtp_llm.multimodal.kvcm import lookup_pb2
from rtp_llm.multimodal.kvcm.lookup import KvMetaLookup, KvMetaLookupError
from rtp_llm.multimodal.kvcm.tensor_object import (
    pack_object,
    plan_object,
    unpack_object,
)


class TensorObjectTest(unittest.TestCase):
    def roundtrip(self, value):
        return unpack_object(pack_object(plan_object(value)))

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


class LookupTest(unittest.TestCase):
    def setUp(self):
        self.requests = []
        self.response = lookup_pb2.GetResponse()
        self.response.header.status.code = 1
        self.response.hit_mask.values.append(True)
        self.response.locations.add(value_size=123)
        self.server = grpc.server(concurrent.futures.ThreadPoolExecutor(max_workers=1))

        def get(request, context):
            self.requests.append(request)
            return self.response

        self.server.add_generic_rpc_handlers(
            (
                grpc.method_handlers_generic_handler(
                    "kv_cache_manager.proto.kv_meta.MetaService",
                    {
                        "Get": grpc.unary_unary_rpc_method_handler(
                            get,
                            request_deserializer=lookup_pb2.GetRequest.FromString,
                            response_serializer=lookup_pb2.GetResponse.SerializeToString,
                        ),
                    },
                ),
            )
        )
        port = self.server.add_insecure_port("127.0.0.1:0")
        self.server.start()
        self.addCleanup(lambda: self.server.stop(0).wait())
        self.client = KvMetaLookup(
            SimpleNamespace(
                addresses=[f"127.0.0.1:{port}"],
                instance_id="kve_model_v1",
                call_timeout_ms=500,
                max_object_bytes=1024,
            )
        )
        self.addCleanup(self.client.close)

    def test_wire_contract_matches_kvmeta_field_numbers(self):
        request = lookup_pb2.GetRequest(instance_id="i", query_type=1, keys=["k"])
        self.assertEqual(request.SerializeToString(), bytes.fromhex("120169180122016b"))
        response = lookup_pb2.GetResponse.FromString(
            bytes.fromhex("0a040a0208011202207b1a030a0101")
        )
        self.assertEqual(response.header.status.code, 1)
        self.assertEqual(response.locations[0].value_size, 123)
        self.assertEqual(list(response.hit_mask.values), [True])

    def test_actual_grpc_get_hit_miss_and_request_identity(self):
        self.assertEqual(
            self.client.size("existing_url_config_key", trace_id="trace"), 123
        )
        request = self.requests[-1]
        self.assertEqual(list(request.keys), ["existing_url_config_key"])
        self.assertEqual(request.instance_id, "kve_model_v1")
        self.assertEqual(request.query_type, 1)
        self.assertEqual(request.trace_id, "trace")
        self.response.hit_mask.values[0] = False
        self.assertIsNone(self.client.size("missing"))

    def test_rejects_wrong_size_count_status_and_closed(self):
        self.response.locations[0].value_size = 10000
        with self.assertRaises(KvMetaLookupError):
            self.client.size("key")
        self.response.ClearField("locations")
        with self.assertRaises(KvMetaLookupError):
            self.client.size("key")
        self.response.header.status.code = 8
        with self.assertRaises(KvMetaLookupError):
            self.client.size("key")
        self.client.close()
        with self.assertRaises(RuntimeError):
            self.client.size("key")


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

        def load(keys, tensors, **kwargs):
            tensors[0].copy_(stored[keys[0]])

        backend.save.side_effect = save
        backend.load.side_effect = load
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
            return_value=(lambda _: backend, lambda **kw: SimpleNamespace(**kw)),
        ):
            client = facade_module.RtpKvMetaObjectClient._from_config(config)
        self.addCleanup(client.close)
        result = (
            torch.arange(12, dtype=torch.bfloat16).reshape(3, 4),
            torch.arange(3),
            [torch.tensor([3.5]), torch.tensor([2], dtype=torch.int32)],
        )
        client.save_object("existing_multimodal_key", result)
        with patch.object(
            client, "object_size", side_effect=lambda key, **kw: stored[key].numel()
        ):
            actual = client.load_object("existing_multimodal_key")
        self.assertTrue(torch.equal(actual[0], result[0]))
        self.assertTrue(torch.equal(actual[1], result[1]))
        self.assertTrue(torch.equal(actual[2][0], result[2][0]))
        self.assertTrue(torch.equal(actual[2][1], result[2][1]))
        backend.remove.assert_not_called()


if __name__ == "__main__":
    unittest.main()
