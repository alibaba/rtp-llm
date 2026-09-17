import asyncio
import struct
import threading
import unittest
from concurrent import futures
from types import SimpleNamespace
from unittest.mock import Mock

import grpc
import torch

from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.config.generate_config import GenerateConfig, RoleAddr, RoleType
from rtp_llm.cpp.model_rpc.model_rpc_client import (
    iter_multimodal_inputs,
    multimodal_cache_keys,
)
from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import (
    MultimodalHashRequestPB,
    MultimodalHashResponsePB,
    MultimodalInputsPB,
)
from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2_grpc import (
    MultimodalRpcServiceStub,
    add_MultimodalRpcServiceServicer_to_server,
)
from rtp_llm.multimodal.mm_embedding_cache import MMEmbeddingCache, MMHashKeyCache
from rtp_llm.multimodal.mm_process_engine import MMEmbeddingRes
from rtp_llm.ops import MMPreprocessConfig, MultimodalInput
from rtp_llm.server.master_client import MasterClient
from rtp_llm.server.mm_cache_metadata import metadata_from_proto
from rtp_llm.server.vit_rpc_server import MultimodalRpcServer
from rtp_llm.utils.base_model_datatypes import GenerateInput


class MMHashRpcTest(unittest.IsolatedAsyncioTestCase):
    """Real loopback gRPC, production handlers/caches, controlled embedding work."""

    async def asyncSetUp(self):
        self.hashes = [
            torch.tensor([-2147483648, -7, 0, 2147483647], dtype=torch.int32)
        ]
        self.engine = SimpleNamespace(
            is_proxy_mode=False,
            _hash_key_cache=MMHashKeyCache(max_bytes=65536),
            _embedding_cache=MMEmbeddingCache(gpu_max_bytes=0, cpu_max_bytes=4096),
            _greennet_enabled=Mock(return_value=False),
            get_embedding_result=Mock(side_effect=self.compute),
            cancel_queued_request=Mock(),
            report_vit_error=Mock(),
        )
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
        add_MultimodalRpcServiceServicer_to_server(
            MultimodalRpcServer(self.engine), self.server
        )
        port = self.server.add_insecure_port("127.0.0.1:0")
        self.server.start()
        self.channel = grpc.aio.insecure_channel(f"127.0.0.1:{port}")
        self.stub = MultimodalRpcServiceStub(self.channel)
        self.address = RoleAddr(
            role=RoleType.VIT, ip="127.0.0.1", http_port=1, grpc_port=port
        )
        self.client = MasterClient()
        self.request = GenerateInput(
            123,
            torch.tensor([1, 99, 2]),
            [
                MultimodalInput(
                    "https://example/image",
                    1,
                    torch.empty(0),
                    MMPreprocessConfig(-1, -1, -1, -1, -1, -1, -1, [], -1),
                )
            ],
            GenerateConfig(),
        )
        self.request.headers = {
            "X-DashScope-Uid": "uid-rpc",
            "X-DashScope-Service": "service-rpc",
        }
        self.keys = multimodal_cache_keys(self.request)

    async def asyncTearDown(self):
        await self.client.close()
        await self.channel.close()
        self.server.stop(0).wait()

    def compute(self, inputs, **kwargs):
        for item in inputs:
            self.engine._hash_key_cache.put(
                item.cache_key(),
                self.hashes,
                greennet_passed=self.engine._greennet_enabled(),
            )
        return [MMEmbeddingRes([], feature_hashes=self.hashes) for _ in inputs]

    def wire_request(self, keys=None):
        inputs = MultimodalInputsPB(request_id=self.request.request_id)
        inputs.multimodal_inputs.extend(
            iter_multimodal_inputs(self.request, self.request.generate_config)
        )
        return MultimodalHashRequestPB(
            keys=keys or self.keys, inputs=inputs, timeout_ms=2000
        )

    async def test_cold_hot_and_evicted_embedding_preserve_hashes_and_approval(self):
        self.engine._greennet_enabled.return_value = True
        # An uninspected historical hash cannot bypass inspection.
        self.engine._hash_key_cache.put(self.keys[0], self.hashes)
        probe = await self.stub.GetMultimodalHashes(
            MultimodalHashRequestPB(keys=self.keys)
        )
        self.assertFalse(probe.entries[0].hash_hit)
        self.assertFalse(probe.entries[0].feature_hashes)
        first = await self.client.get_vit_cache_metadata(
            self.address, self.keys, self.request
        )
        call = self.engine.get_embedding_result.call_args
        self.assertTrue(call.kwargs["hashes_only"])
        self.assertEqual(call.kwargs["user_id"], "uid-rpc")
        self.assertEqual(call.kwargs["service_name"], "service-rpc")
        self.assertIsNotNone(call.kwargs["cancellation_event"])
        self.assertEqual(
            self.request.greennet_verified_vit,
            (self.address.ip, self.address.grpc_port, tuple(self.keys)),
        )
        self.engine._embedding_cache.clear()
        self.engine.get_embedding_result.reset_mock()
        hot = await self.client.get_vit_cache_metadata(
            self.address, self.keys, self.request
        )
        self.engine.get_embedding_result.assert_not_called()
        self.assertEqual(
            first["entries"][0]["feature_hashes"], hot["entries"][0]["feature_hashes"]
        )
        self.assertEqual(
            list(hot["entries"][0]["feature_hashes"]), self.hashes[0].tolist()
        )
        self.assertFalse(hot["entries"][0]["embedding_hit"])

    async def test_cache_disabled_still_returns_hashes_without_embeddings(self):
        self.engine._hash_key_cache.resize(0)
        self.engine._embedding_cache.clear()
        response = await self.stub.GetMultimodalHashes(self.wire_request())
        self.assertTrue(response.entries[0].hash_hit)
        self.assertEqual(
            response.entries[0].feature_hashes,
            struct.pack("<4i", *self.hashes[0].tolist()),
        )
        self.assertEqual(
            set(response.DESCRIPTOR.fields_by_name),
            {"worker_instance", "feature_hash_version", "entries"},
        )
        self.assertFalse(self.engine._hash_key_cache.contains(self.keys[0]))

    async def test_concurrent_probes_match_http_metadata_and_keep_media_order(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from rtp_llm.server.vit_app import register_mm_cache_routes

        for i in range(8):
            self.engine._hash_key_cache.put(
                f"image-{i}",
                [
                    torch.tensor([i, -i], dtype=torch.int32),
                    torch.tensor([i + 100], dtype=torch.int32),
                ],
            )
        _, entry = self.engine._embedding_cache.try_acquire("image-3")
        entry.complete(torch.ones(3, 4))
        keys = ["image-3", "image-1", "absent", "image-3"]
        app = FastAPI()
        register_mm_cache_routes(app, self.engine)
        with TestClient(app) as http:
            expected = http.post("/mm_cache/metadata", json={"keys": keys}).json()

        async def probe():
            response = await self.stub.GetMultimodalHashes(
                MultimodalHashRequestPB(keys=keys)
            )
            actual = metadata_from_proto(response)
            self.assertEqual(actual["worker_instance"], expected["worker_instance"])
            self.assertEqual([e["key"] for e in actual["entries"]], keys)
            for a, b in zip(actual["entries"], expected["entries"]):
                for name in (
                    "hash_hit",
                    "embedding_hit",
                    "embedding_tier",
                    "greennet_passed",
                ):
                    self.assertEqual(a[name], b[name])
                self.assertEqual(a["split_size"], b.get("split_size", []))
                self.assertEqual(list(a["feature_hashes"]), b.get("feature_hashes", []))

        await asyncio.gather(*(probe() for _ in range(16)))
        self.engine.get_embedding_result.assert_not_called()

    async def test_errors_reach_frontend_without_an_http_fallback(self):
        cases = [
            (FtRuntimeException(code, "expected-error"), code)
            for code in (
                ExceptionType.UNSAFE_INPUT_CONTENT,
                ExceptionType.MM_PROCESS_ERROR,
                ExceptionType.CONCURRENCY_LIMIT_ERROR,
                ExceptionType.GENERATE_TIMEOUT,
            )
        ]
        cases.append((TimeoutError("expired"), ExceptionType.GENERATE_TIMEOUT))
        for error, code in cases:
            with self.subTest(error=error):
                self.engine.get_embedding_result.side_effect = error
                with self.assertRaises(FtRuntimeException) as raised:
                    await self.client.get_vit_cache_metadata(
                        self.address, self.keys, self.request
                    )
                self.assertEqual(raised.exception.exception_type, code)
                self.assertIn(
                    getattr(error, "message", str(error)), raised.exception.message
                )

    async def test_invalid_keys_and_tensors_never_start_compute(self):
        tensor_request = self.wire_request()
        tensor_request.inputs.multimodal_inputs[0].multimodal_tensor.int32_data = (
            b"1234"
        )
        for request in (
            self.wire_request(["wrong"]),
            tensor_request,
            MultimodalHashRequestPB(keys=["x"] * 257),
            MultimodalHashRequestPB(keys=["x" * 4097]),
        ):
            with self.assertRaises(grpc.aio.AioRpcError) as raised:
                await self.stub.GetMultimodalHashes(request)
            self.assertEqual(raised.exception.code(), grpc.StatusCode.INVALID_ARGUMENT)
        self.engine.get_embedding_result.assert_not_called()

    async def test_rpc_deadline_cancels_queued_work(self):
        seen = threading.Event()
        cancelled = threading.Event()

        def wait_for_cancel(inputs, **kwargs):
            self.assertLessEqual(kwargs["timeout_ms"], 100)
            seen.set()
            if kwargs["cancellation_event"].wait(2):
                cancelled.set()
            raise FtRuntimeException(ExceptionType.CANCELLED_ERROR, "cancelled")

        self.engine.get_embedding_result.side_effect = wait_for_cancel
        with self.assertRaises(grpc.aio.AioRpcError) as raised:
            await self.stub.GetMultimodalHashes(self.wire_request(), timeout=0.1)
        self.assertEqual(raised.exception.code(), grpc.StatusCode.DEADLINE_EXCEEDED)
        self.assertTrue(seen.is_set())
        for _ in range(100):
            if cancelled.is_set():
                break
            await asyncio.sleep(0.01)
        self.assertTrue(cancelled.is_set())
        self.engine.cancel_queued_request.assert_called_with(self.request.request_id)

    def test_malformed_hash_bytes_and_splits_are_rejected(self):
        for data, sizes in ((b"x", [1]), (b"1234", [2]), (b"1234", [0])):
            response = MultimodalHashResponsePB(feature_hash_version=1)
            response.entries.add(
                key="k", hash_hit=True, feature_hashes=data, split_size=sizes
            )
            with self.assertRaises(ValueError):
                metadata_from_proto(response)


if __name__ == "__main__":
    unittest.main()
