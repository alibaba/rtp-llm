import asyncio
import struct
import threading
import unittest
from concurrent import futures
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import grpc
import torch

from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.config.generate_config import GenerateConfig, RoleAddr, RoleType
from rtp_llm.cpp.model_rpc.model_rpc_client import (
    iter_multimodal_inputs,
    multimodal_cache_keys,
    trans_input,
)
from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import (
    CacheVersionPB,
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
from rtp_llm.server.backend_rpc_server_visitor import BackendRPCServerVisitor
from rtp_llm.server.master_client import FlexlbResponse, MasterClient
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

    async def test_status_poll_does_not_snapshot_keys(self):
        with patch.object(self.engine._hash_key_cache, "keys") as hashes, patch.object(
            self.engine._embedding_cache, "resident_tiers"
        ) as tiers:
            response = await self.stub.GetCacheStatus(CacheVersionPB(), timeout=2)
        self.assertFalse(response.HasField("multimodal_cache"))
        self.assertEqual(response.ByteSize(), 0)
        hashes.assert_not_called()
        tiers.assert_not_called()

    async def test_cache_directory_tracks_hashes_and_embedding_eviction(self):
        self.engine._hash_key_cache.put("hash-only", self.hashes)
        self.engine._hash_key_cache.put("ready", self.hashes)
        _, ready = self.engine._embedding_cache.try_acquire("ready")
        ready.complete(torch.ones(2, 4), self.hashes)
        _, embedding_only = self.engine._embedding_cache.try_acquire("embedding-only")
        embedding_only.complete(torch.ones(1))
        _, pending = self.engine._embedding_cache.try_acquire("pending")
        request = CacheVersionPB(need_cache_keys=True)
        response = await self.stub.GetCacheStatus(request, timeout=2)
        self.assertTrue(response.HasField("multimodal_cache"))
        cache = response.multimodal_cache
        self.assertEqual(cache.worker_instance, self.engine._hash_key_cache.instance_id)
        self.assertEqual(set(cache.keys), {"hash-only", "ready"})
        self.assertEqual(set(cache.cpu_embedding_keys), {"ready", "embedding-only"})
        self.assertEqual(list(cache.gpu_embedding_keys), [])
        self.assertFalse(pending.is_done)
        self.engine.get_embedding_result.assert_not_called()

        self.engine._embedding_cache.remove("ready")
        response = await self.stub.GetCacheStatus(request, timeout=2)
        self.assertIn("ready", response.multimodal_cache.keys)
        self.assertNotIn("ready", response.multimodal_cache.cpu_embedding_keys)
        self.engine._hash_key_cache.clear()
        self.engine._embedding_cache.clear()
        response = await self.stub.GetCacheStatus(request, timeout=2)
        self.assertTrue(response.HasField("multimodal_cache"))
        self.assertTrue(response.multimodal_cache.worker_instance)
        self.assertFalse(response.multimodal_cache.keys)
        self.assertFalse(response.multimodal_cache.cpu_embedding_keys)

    async def test_cache_directory_preserves_gpu_tier_without_reading_tensors(self):
        self.engine._hash_key_cache.put("hash", self.hashes)
        with patch.object(
            self.engine._embedding_cache,
            "resident_tiers",
            return_value={"hash": "gpu", "cpu-only": "cpu"},
        ):
            response = await self.stub.GetCacheStatus(
                CacheVersionPB(need_cache_keys=True), timeout=2
            )
        self.assertEqual(list(response.multimodal_cache.gpu_embedding_keys), ["hash"])
        self.assertEqual(
            list(response.multimodal_cache.cpu_embedding_keys), ["cpu-only"]
        )
        self.engine.get_embedding_result.assert_not_called()

    async def test_cache_directory_rejects_proxy_and_oversized_response(self):
        self.engine.is_proxy_mode = True
        with self.assertRaises(grpc.aio.AioRpcError) as error:
            await self.stub.GetCacheStatus(
                CacheVersionPB(need_cache_keys=True), timeout=2
            )
        self.assertEqual(error.exception.code(), grpc.StatusCode.UNIMPLEMENTED)
        self.engine.is_proxy_mode = False
        with patch("rtp_llm.server.vit_rpc_server.MM_CACHE_SNAPSHOT_MAX_BYTES", 1):
            with self.assertRaises(grpc.aio.AioRpcError) as error:
                await self.stub.GetCacheStatus(
                    CacheVersionPB(need_cache_keys=True), timeout=2
                )
        self.assertEqual(error.exception.code(), grpc.StatusCode.RESOURCE_EXHAUSTED)

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
            {"worker_instance", "entries"},
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

    async def test_frontend_route_to_real_vit_rpc_expands_prefill_input(self):
        # The frontend routing method, MasterClient, gRPC transport, and ViT
        # handler are real. Only FlexLB's route choice and ViT compute are
        # controlled so this test does not need a GPU or model checkpoint.
        visitor = BackendRPCServerVisitor.__new__(BackendRPCServerVisitor)
        visitor._mm_cache_routing = True
        visitor.mm_model_config = SimpleNamespace(
            mm_sep_tokens=[[99]], include_sep_tokens=False
        )
        visitor.seq_size_per_block = 2
        visitor._page_rr_route_cache_keys = False
        visitor._page_rr_cp_size = 1
        visitor.max_seq_len = 100
        visitor._report_recent_cache_key_metrics = lambda keys: None
        visitor.master_client = self.client

        prefill = RoleAddr(
            role=RoleType.PREFILL, ip="127.0.0.2", http_port=8000, grpc_port=8001
        )
        status = {
            "role": "VIT",
            "server_ip": self.address.ip,
            "http_port": self.address.http_port,
            "grpc_port": self.address.grpc_port,
        }
        expected_tokens = [1, -2147483648, -7, 0, 2147483647, 2]

        for _ in range(2):
            self.client.get_backend_role_addrs = AsyncMock(
                side_effect=[
                    FlexlbResponse(
                        role_addrs=[self.address],
                        result={"server_status": [status]},
                    ),
                    FlexlbResponse.ok([prefill, self.address]),
                ]
            )
            self.assertIsNone(await visitor.get_master_route_addrs(self.request))
            calls = self.client.get_backend_role_addrs.call_args_list
            self.assertEqual(len(calls), 2)
            self.assertTrue(calls[0].kwargs["vit_only"])
            self.assertEqual(calls[0].kwargs["media_keys"], self.keys)
            self.assertEqual(calls[1].kwargs["seq_len"], len(expected_tokens))
            self.assertEqual(calls[1].kwargs["selected_vit"], status)
            self.assertTrue(calls[1].kwargs["block_cache_keys"])
            wire = calls[1].kwargs["input_pb"]
            self.assertEqual(list(wire.token_ids), expected_tokens)
            self.assertEqual(
                [(s.offset, s.length) for s in wire.multimodal_token_layout.spans],
                [(1, 4)],
            )
            self.assertEqual(list(trans_input(self.request).token_ids), expected_tokens)
            self.assertEqual(self.request.token_ids.tolist(), [1, 99, 2])
            self.assertEqual(
                self.request.generate_config.role_addrs, [prefill, self.address]
            )
            self.assertEqual(self.engine.get_embedding_result.call_count, 1)
            self.request.generate_config.role_addrs = []

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
            response = MultimodalHashResponsePB()
            response.entries.add(
                key="k", hash_hit=True, feature_hashes=data, split_size=sizes
            )
            with self.assertRaises(ValueError):
                metadata_from_proto(response)


if __name__ == "__main__":
    unittest.main()
