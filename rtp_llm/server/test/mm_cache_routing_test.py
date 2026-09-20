import asyncio
import struct
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import grpc
import torch

from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.config.generate_config import GenerateConfig, RoleAddr, RoleType
from rtp_llm.cpp.model_rpc.model_rpc_client import (
    multimodal_cache_keys,
    trans_input,
    trans_multimodal_input,
)
from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import (
    ErrorDetailsPB,
    GenerateInputPB,
    MultimodalHashRequestPB,
    MultimodalHashResponsePB,
)
from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2_grpc import (
    MultimodalRpcServiceServicer,
    add_MultimodalRpcServiceServicer_to_server,
)
from rtp_llm.multimodal.multimodal_util import trans_config
from rtp_llm.ops import MMPreprocessConfig, MultimodalInput
from rtp_llm.server.backend_rpc_server_visitor import BackendRPCServerVisitor
from rtp_llm.server.master_client import (
    VIT_ROUTE_STALE_CODE,
    FlexlbResponse,
    MasterClient,
)
from rtp_llm.server.mm_cache_routing import multimodal_routing_tokens
from rtp_llm.utils.base_model_datatypes import GenerateInput


def metadata(keys, hashes):
    return {
        "worker_instance": "epoch",
        "entries": [
            {
                "key": key,
                "hit": True,
                "split_size": [len(values)],
                "feature_hashes": values,
            }
            for key, values in zip(keys, hashes)
        ],
    }


def hash_response(keys, hashes):
    result = MultimodalHashResponsePB(worker_instance="epoch")
    for key, values in zip(keys, hashes):
        result.entries.add(
            key=key,
            hash_hit=True,
            split_size=[len(values)],
            feature_hashes=struct.pack(f"<{len(values)}i", *values),
        )
    return result


class MMCacheRoutingTest(unittest.TestCase):
    def test_shared_key_resolves_request_overrides_and_ignores_timeout(self):
        item = MultimodalInput(
            "https://example/image",
            1,
            torch.empty(0),
            MMPreprocessConfig(-1, -1, 10, 2000, 1.2, 1, 8, [0.1, 0.2], 10, 100),
        )
        cfg = GenerateConfig(
            min_pixels=20,
            max_pixels=4000,
            mm_timeout_ms=99,
            fps=2.3,
            crop_positions=[0.3, 0.4],
            max_long_side_pixel=200,
        )
        request = GenerateInput(1, torch.tensor([1]), [item], cfg)
        wire = GenerateInputPB()
        trans_multimodal_input(request, wire, cfg)
        resolved = wire.multimodal_inputs[0]
        expected = MultimodalInput(
            item.url,
            item.mm_type,
            torch.empty(0),
            trans_config(resolved.mm_preprocess_config),
        ).cache_key()
        self.assertEqual(multimodal_cache_keys(request), [expected])
        cfg.mm_timeout_ms = 999
        self.assertEqual(multimodal_cache_keys(request), [expected])
        cfg.min_pixels = 30
        self.assertNotEqual(multimodal_cache_keys(request), [expected])

    def test_full_hit_does_not_change_original_tokens(self):
        tokens = [10, 99, 20, 99, 30]
        result, length = multimodal_routing_tokens(
            tokens, [[99]], False, ["a", "b"], metadata(["a", "b"], [[-1, 2], [3]]), 100
        )
        self.assertEqual(result, [10, -1, 2, 20, 3, 30])
        self.assertEqual(length, 6)
        self.assertEqual(tokens, [10, 99, 20, 99, 30])

    def test_partial_hit_stops_before_unknown_image(self):
        result, length = multimodal_routing_tokens(
            [10, 99, 20, 99, 30],
            [[99]],
            False,
            ["a", "b"],
            metadata(["a"], [[-1, 2]]),
            100,
        )
        self.assertEqual(result, [10, -1, 2, 20])
        self.assertIsNone(length)

    def test_expanded_locations_cover_video_segments_and_repeated_media(self):
        data = metadata(["video"], [[-1, 99, 3]])
        data["entries"][0]["split_size"] = [2, 1]
        spans = []
        tokens, length = multimodal_routing_tokens(
            [1, 90, 5, 91, 2, 90, 6, 91, 90, 7, 91, 90, 8, 91],
            [[90, 91]],
            False,
            ["video", "video"],
            data,
            100,
            compact=True,
            expanded_spans=spans,
        )
        self.assertEqual(
            list(tokens), [1, 90, -1, 99, 91, 2, 90, 3, 91, 90, -1, 99, 91, 90, 3, 91]
        )
        self.assertEqual(length, 16)
        self.assertEqual(spans, [(2, 2), (7, 1), (10, 2), (14, 1)])

    def test_paired_tags_and_repeated_media(self):
        tokens = [1, 90, 5, 91, 2, 90, 6, 91]
        data = metadata(["a"], [[-1, 3]])
        kept, _ = multimodal_routing_tokens(
            tokens, [[90, 91]], False, ["a", "a"], data, 100
        )
        removed, _ = multimodal_routing_tokens(
            tokens, [[90, 91]], True, ["a", "a"], data, 100
        )
        self.assertEqual(kept, [1, 90, -1, 3, 91, 2, 90, -1, 3, 91])
        self.assertEqual(removed, [1, -1, 3, 2, -1, 3])

    def test_missing_metadata_returns_only_safe_prefix(self):
        for data in (None, {}, {"entries": []}):
            self.assertEqual(
                multimodal_routing_tokens([1, 99, 2], [[99]], False, ["a"], data, 100),
                ([1], None),
            )

    def test_invalid_hashes_are_rejected(self):
        data = metadata(["a"], [[1 << 32]])
        with self.assertRaises(ValueError):
            multimodal_routing_tokens([1, 99], [[99]], False, ["a"], data, 100)

    def test_hash_only_hit_expands_tokens_and_embedding_only_hit_does_not(self):
        data = metadata(["a"], [[-10, 11]])
        data["entries"][0].update(
            hash_hit=True, embedding_hit=False, embedding_tier=None
        )
        self.assertEqual(
            multimodal_routing_tokens([1, 99, 2], [[99]], False, ["a"], data, 100),
            ([1, -10, 11, 2], 4),
        )
        data["entries"][0].update(
            hash_hit=False, embedding_hit=True, embedding_tier="cpu"
        )
        self.assertEqual(
            multimodal_routing_tokens([1, 99, 2], [[99]], False, ["a"], data, 100),
            ([1], None),
        )

    def test_compact_expansion_produces_identical_prefill_block_hashes(self):
        from array import array

        from rtp_llm.ops import get_block_cache_keys

        hashes = list(range(-10000, 10000))
        data = metadata(["a"], [hashes])
        tokens, length = multimodal_routing_tokens(
            [1, 99, 2], [[99]], False, ["a"], data, 30000, compact=True
        )
        self.assertIsInstance(tokens, array)
        self.assertEqual(tokens.itemsize, 4)
        expected = [1] + hashes + [2]
        self.assertEqual(list(tokens), expected)
        self.assertEqual(length, len(expected))
        actual_blocks = get_block_cache_keys(tokens, 16)
        self.assertTrue(actual_blocks)
        self.assertEqual(actual_blocks, get_block_cache_keys(expected, 16))


class MMCacheRoutingIntegrationTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
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
        item = MultimodalInput(
            "https://example/image",
            1,
            torch.empty(0),
            MMPreprocessConfig(-1, -1, -1, -1, -1, -1, -1, [], -1),
        )
        request = GenerateInput(
            123, torch.tensor([1, 99, 2], dtype=torch.int32), [item], GenerateConfig()
        )
        vit = RoleAddr(
            role=RoleType.VIT, ip="127.0.0.1", http_port=8000, grpc_port=8001
        )
        prefill = RoleAddr(
            role=RoleType.PREFILL, ip="127.0.0.2", http_port=8000, grpc_port=8001
        )
        status = {
            "role": "VIT",
            "server_ip": vit.ip,
            "http_port": 8000,
            "grpc_port": 8001,
        }
        visitor.master_client = SimpleNamespace(
            get_backend_role_addrs=AsyncMock(
                side_effect=[
                    FlexlbResponse(
                        role_addrs=[vit], result={"server_status": [status]}
                    ),
                    FlexlbResponse.ok([prefill, vit]),
                ]
            ),
            get_vit_cache_metadata=AsyncMock(
                return_value=metadata(multimodal_cache_keys(request), [[-10, 11]])
            ),
        )
        self.visitor, self.request, self.vit, self.prefill, self.status = (
            visitor,
            request,
            vit,
            prefill,
            status,
        )

    async def test_vit_preselection_still_routes_pd_and_sends_expanded_length(self):
        visitor, request, vit, prefill, status = (
            self.visitor,
            self.request,
            self.vit,
            self.prefill,
            self.status,
        )
        await visitor.get_master_route_addrs(request)
        calls = visitor.master_client.get_backend_role_addrs.call_args_list
        self.assertEqual(len(calls), 2)
        self.assertTrue(calls[0].kwargs["vit_only"])
        self.assertEqual(calls[1].kwargs["seq_len"], 4)
        self.assertEqual(calls[1].kwargs["selected_vit"], status)
        self.assertEqual(request.generate_config.role_addrs, [prefill, vit])
        self.assertEqual(request.token_ids.tolist(), [1, 99, 2])
        wire = GenerateInputPB.FromString(trans_input(request).SerializeToString())
        self.assertEqual(list(wire.token_ids), [1, -10, 11, 2])
        self.assertTrue(wire.HasField("multimodal_token_layout"))
        self.assertEqual(
            [(s.offset, s.length) for s in wire.multimodal_token_layout.spans], [(1, 2)]
        )

    async def test_missing_or_invalid_required_hashes_stop_before_prefill_routing(self):
        for data in (None, {"entries": [{}]}):
            await self.asyncSetUp()
            self.visitor.master_client.get_vit_cache_metadata.return_value = data
            with self.assertRaises(FtRuntimeException):
                await self.visitor.get_master_route_addrs(self.request)
            calls = self.visitor.master_client.get_backend_role_addrs.call_args_list
            self.assertEqual(len(calls), 1)
            self.assertEqual(self.request.token_ids.tolist(), [1, 99, 2])

    async def test_required_hash_miss_submits_only_missing_distinct_inputs(self):
        from rtp_llm.multimodal.multimodal_util import trans_mm_input

        item = self.request.mm_inputs[0]
        second = MultimodalInput(
            "https://example/second",
            item.mm_type,
            torch.empty(0),
            item.mm_preprocess_config,
        )
        self.request.headers = {"X-DashScope-Uid": "uid-http"}
        self.request.mm_inputs = [item, second, second]
        self.request.generate_config.max_pixels = 1024
        keys = multimodal_cache_keys(self.request)
        probe = metadata(keys[:1], [[-10, 11]])
        probe["entries"].append({"key": keys[1], "hash_hit": False})
        filled = metadata(keys[1:2], [[21, 22, 23]])
        client = MasterClient()
        client._get_vit_metadata = AsyncMock(side_effect=[probe, filled])
        result = await client.get_vit_cache_metadata(self.vit, keys, input=self.request)
        self.assertEqual(
            [e["feature_hashes"] for e in result["entries"]], [[-10, 11], [21, 22, 23]]
        )
        calls = client._get_vit_metadata.call_args_list
        self.assertEqual(list(calls[0].args[1].keys), keys[:2])
        self.assertFalse(calls[0].args[1].HasField("inputs"))
        payload = calls[1].args[1]
        self.assertEqual(list(payload.keys), keys[1:2])
        self.assertEqual(payload.inputs.request_id, self.request.request_id)
        self.assertEqual(len(payload.inputs.multimodal_inputs), 1)
        resolved = trans_mm_input(payload.inputs)
        self.assertEqual(resolved[0].cache_key(), keys[1])
        self.assertEqual(resolved[0].mm_preprocess_config.max_pixels, 1024)
        self.assertEqual(calls[1].kwargs["headers"]["X-DashScope-Uid"], "uid-http")
        self.assertTrue(calls[1].kwargs["required"])
        self.assertGreater(calls[1].args[2], 0.5)

    async def test_hash_hit_sends_no_media_and_failed_submission_is_not_optional(self):
        client = MasterClient()
        keys = multimodal_cache_keys(self.request)
        ready = metadata(keys, [[-10, 11]])
        client._get_vit_metadata = AsyncMock(return_value=ready)
        self.assertIs(
            await client.get_vit_cache_metadata(self.vit, keys, input=self.request),
            ready,
        )
        client._get_vit_metadata.assert_awaited_once()
        self.assertEqual(list(client._get_vit_metadata.call_args.args[1].keys), keys)
        client._get_vit_metadata = AsyncMock(
            side_effect=[
                {"worker_instance": "epoch", "entries": []},
                {"worker_instance": "epoch", "entries": []},
            ]
        )
        with self.assertRaisesRegex(FtRuntimeException, "incomplete feature hashes"):
            await client.get_vit_cache_metadata(self.vit, keys, input=self.request)

    async def test_frontend_drops_metadata_and_retains_only_compact_expansion(
        self,
    ):
        class TrackedMetadata(dict):
            pass

        import weakref

        saved = []

        async def get_metadata(*args, **kwargs):
            result = TrackedMetadata(
                metadata(multimodal_cache_keys(self.request), [[-10, 11]])
            )
            saved.append(weakref.ref(result))
            return result

        async def route(*args, **kwargs):
            if kwargs.get("vit_only"):
                return FlexlbResponse(
                    role_addrs=[self.vit], result={"server_status": [self.status]}
                )
            self.assertIsNone(saved[0]())
            self.assertTrue(kwargs["block_cache_keys"])
            return FlexlbResponse.ok([self.prefill, self.vit])

        self.visitor.master_client.get_vit_cache_metadata = get_metadata
        self.visitor.master_client.get_backend_role_addrs = route
        await self.visitor.get_master_route_addrs(self.request)
        from array import array

        expansion = self.request.mm_token_expansion
        self.assertIsInstance(expansion.token_ids, array)
        self.assertEqual(expansion.token_ids.itemsize, 4)
        self.assertEqual(expansion.spans, [(1, 2)])

    async def test_evicted_embedding_still_supplies_prefill_cache_routing_hashes(self):
        from rtp_llm.multimodal.mm_embedding_cache import (
            MMEmbeddingCache,
            MMHashKeyCache,
        )

        hashes = MMHashKeyCache(max_bytes=8192)
        embeddings = MMEmbeddingCache(gpu_max_bytes=0, cpu_max_bytes=8)
        key = multimodal_cache_keys(self.request)[0]
        _, entry = embeddings.try_acquire(key)
        entry.complete(torch.ones(2))
        hashes.put(key, [torch.tensor([-10, 11])], entry.generation)
        embeddings.clear()
        self.visitor.master_client.get_vit_cache_metadata.return_value = (
            hashes.metadata([key], embeddings)
        )
        await self.visitor.get_master_route_addrs(self.request)
        route = self.visitor.master_client.get_backend_role_addrs.call_args.kwargs
        self.assertEqual(route["seq_len"], 4)
        self.assertTrue(route["block_cache_keys"])
        self.assertEqual(route["selected_vit"], self.status)
        self.assertEqual(self.request.token_ids.tolist(), [1, 99, 2])

    async def test_old_master_falls_back_to_ordinary_schedule(self):
        self.visitor.master_client.get_backend_role_addrs.side_effect = [
            FlexlbResponse.error_response(404),
            FlexlbResponse(
                role_addrs=[self.prefill, self.vit],
                result={"server_status": [self.status]},
            ),
            FlexlbResponse.ok([self.prefill, self.vit]),
        ]
        self.assertIsNone(await self.visitor.get_master_route_addrs(self.request))
        calls = self.visitor.master_client.get_backend_role_addrs.call_args_list
        self.assertNotIn("selected_vit", calls[1].kwargs)
        self.visitor.master_client.get_vit_cache_metadata.assert_awaited_once()
        self.assertEqual(calls[2].kwargs["selected_vit"], self.status)
        self.assertEqual(calls[2].kwargs["seq_len"], 4)
        self.assertTrue(calls[2].kwargs["block_cache_keys"])

    async def test_master_cannot_change_the_selected_vit(self):
        self.visitor.master_client.get_backend_role_addrs.side_effect = [
            FlexlbResponse(
                role_addrs=[self.vit], result={"server_status": [self.status]}
            ),
            FlexlbResponse.ok([self.prefill]),
        ]
        with self.assertRaisesRegex(Exception, "changed the selected ViT"):
            await self.visitor.get_master_route_addrs(self.request)
        self.assertFalse(self.request.generate_config.role_addrs)
        self.assertIsNone(self.request.mm_token_expansion)

    async def test_stale_vit_reselection_acquires_new_hashes_before_prefill(self):
        new_vit = self.vit.model_copy(update={"ip": "127.0.0.3"})
        new_status = {**self.status, "server_ip": new_vit.ip}
        self.visitor.master_client.get_backend_role_addrs.side_effect = [
            FlexlbResponse(
                role_addrs=[self.vit], result={"server_status": [self.status]}
            ),
            FlexlbResponse.error_response(VIT_ROUTE_STALE_CODE),
            FlexlbResponse(
                role_addrs=[self.prefill, new_vit],
                result={"server_status": [new_status]},
            ),
            FlexlbResponse.ok([self.prefill, new_vit]),
        ]
        keys = multimodal_cache_keys(self.request)
        self.visitor.master_client.get_vit_cache_metadata.side_effect = [
            metadata(keys, [[-10, 11]]),
            metadata(keys, [[31, 32, 33, 34]]),
        ]
        self.assertIsNone(await self.visitor.get_master_route_addrs(self.request))
        last = self.visitor.master_client.get_backend_role_addrs.call_args.kwargs
        self.assertEqual(last["selected_vit"], new_status)
        self.assertTrue(last["block_cache_keys"])
        self.assertEqual(last["seq_len"], 6)
        self.assertEqual(
            self.visitor.master_client.get_vit_cache_metadata.call_args.args[0], new_vit
        )
        self.assertEqual(
            self.request.generate_config.role_addrs, [self.prefill, new_vit]
        )
        self.assertEqual(self.request.token_ids.tolist(), [1, 99, 2])
        wire = trans_input(self.request)
        self.assertEqual(list(wire.token_ids), [1, 31, 32, 33, 34, 2])
        self.assertEqual(
            [(s.offset, s.length) for s in wire.multimodal_token_layout.spans], [(1, 4)]
        )

    async def test_failed_reroute_discards_previous_expansion(self):
        await self.visitor.get_master_route_addrs(self.request)
        self.assertIsNotNone(self.request.mm_token_expansion)
        self.visitor.master_client.get_backend_role_addrs.side_effect = [
            FlexlbResponse.error_response(503)
        ]
        response = await self.visitor.get_master_route_addrs(self.request)
        self.assertEqual(response.error_code, 503)
        self.assertIsNone(self.request.mm_token_expansion)
        self.assertFalse(trans_input(self.request).HasField("multimodal_token_layout"))

    async def test_short_http_timeout_does_not_shorten_pending_placement_ttl(self):
        client = MasterClient(
            host_service=SimpleNamespace(
                get_master_addr=lambda: "master:8000",
                master_vip=SimpleNamespace(domain=""),
            )
        )
        self.request.generate_config.ttft_timeout_ms = 30000
        client._send_schedule_request = AsyncMock(
            return_value=FlexlbResponse.error_response(404)
        )
        await client.get_backend_role_addrs(
            [], 2, self.request, 123, media_keys=["image"], vit_only=True
        )
        call = client._send_schedule_request.call_args
        self.assertEqual(call.args[1]["generate_timeout"], 30000)
        self.assertEqual(call.args[2], 500)
        self.assertEqual(call.kwargs["path"], "/rtp_llm/vit/route")

    async def test_metadata_grpc_failure_and_timeout_are_optional(self):
        class Service(MultimodalRpcServiceServicer):
            async def GetMultimodalHashes(self, request, context):
                if request.keys[0] == "slow":
                    await asyncio.sleep(0.6)
                if request.keys[0] == "ready":
                    return hash_response(["ready"], [[-1, 2]])
                await context.abort(grpc.StatusCode.UNIMPLEMENTED, "old worker")

        server = grpc.aio.server()
        add_MultimodalRpcServiceServicer_to_server(Service(), server)
        port = server.add_insecure_port("127.0.0.1:0")
        await server.start()
        address = self.vit.model_copy(update={"grpc_port": port, "http_port": 1})
        client = MasterClient()
        try:
            result = await client.get_vit_cache_metadata(address, ["ready"])
            self.assertEqual(list(result["entries"][0]["feature_hashes"]), [-1, 2])
            self.assertEqual(result["entries"][0]["feature_hashes"].itemsize, 4)
            self.assertIsNone(
                await client.get_vit_cache_metadata(address, ["old-worker"])
            )
            self.assertIsNone(await client.get_vit_cache_metadata(address, ["slow"]))
        finally:
            await client.close()
            await server.stop(0)

    async def test_grpc_cold_hash_waits_before_prefill_and_preserves_errors(self):
        keys = multimodal_cache_keys(self.request)
        self.request.generate_config.mm_timeout_ms = 2000
        payloads, seen_headers = [], []
        self.request.headers = {
            "X-DashScope-Uid": "uid-cold-submit",
            "X-DashScope-Service": "service-cold",
            "Authorization": "must-not-forward",
        }
        reject = False

        class Service(MultimodalRpcServiceServicer):
            async def GetMultimodalHashes(self, request, context):
                payloads.append(request)
                seen_headers.append(dict(context.invocation_metadata()))
                if not request.HasField("inputs"):
                    result = MultimodalHashResponsePB(worker_instance="epoch")
                    result.entries.add(key=keys[0], hash_hit=False)
                    return result
                if reject:
                    context.set_trailing_metadata(
                        (
                            (
                                "grpc-status-details-bin",
                                ErrorDetailsPB(
                                    error_code=int(
                                        ExceptionType.CONCURRENCY_LIMIT_ERROR
                                    ),
                                    error_message="full",
                                ).SerializeToString(),
                            ),
                        )
                    )
                    await context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, "full")
                await asyncio.sleep(0.6)
                return hash_response(keys, [[-10, 11]])

        server = grpc.aio.server()
        add_MultimodalRpcServiceServicer_to_server(Service(), server)
        port = server.add_insecure_port("127.0.0.1:0")
        await server.start()
        vit = self.vit.model_copy(update={"grpc_port": port, "http_port": 1})
        status = {**self.status, "grpc_port": port, "http_port": 1}
        client = MasterClient()
        client.get_backend_role_addrs = AsyncMock(
            side_effect=[
                FlexlbResponse(role_addrs=[vit], result={"server_status": [status]}),
                FlexlbResponse.ok([self.prefill, vit]),
            ]
        )
        self.visitor.master_client = client
        try:
            await self.visitor.get_master_route_addrs(self.request)
            self.assertEqual(len(payloads), 2)
            self.assertFalse(payloads[0].HasField("inputs"))
            self.assertEqual(len(payloads[1].inputs.multimodal_inputs), 1)
            self.assertEqual(seen_headers[1]["x-dashscope-uid"], "uid-cold-submit")
            self.assertEqual(seen_headers[1]["x-dashscope-service"], "service-cold")
            self.assertNotIn("authorization", seen_headers[1])
            route = client.get_backend_role_addrs.call_args.kwargs
            self.assertEqual(route["seq_len"], 4)
            self.assertTrue(route["block_cache_keys"])
            self.assertEqual(self.request.token_ids.tolist(), [1, 99, 2])
            reject = True
            with self.assertRaises(FtRuntimeException) as raised:
                await client.get_vit_cache_metadata(vit, keys, input=self.request)
            self.assertEqual(
                raised.exception.exception_type, ExceptionType.CONCURRENCY_LIMIT_ERROR
            )
        finally:
            await client.close()
            await server.stop(0)


class MMCacheApiTest(unittest.TestCase):
    def test_submit_on_metadata_miss_returns_hashes_even_with_caches_disabled(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from google.protobuf.json_format import MessageToDict

        from rtp_llm.cpp.model_rpc.model_rpc_client import iter_multimodal_inputs
        from rtp_llm.multimodal.mm_embedding_cache import (
            MMEmbeddingCache,
            MMHashKeyCache,
        )
        from rtp_llm.multimodal.mm_process_engine import MMEmbeddingRes
        from rtp_llm.server.vit_app import register_mm_cache_routes

        item = MultimodalInput(
            "https://example/image",
            1,
            torch.empty(0),
            MMPreprocessConfig(-1, -1, -1, -1, -1, -1, -1, [], -1),
        )
        request = GenerateInput(321, torch.tensor([1, 99]), [item], GenerateConfig())
        key = multimodal_cache_keys(request)[0]
        payload = {
            "keys": [key],
            "request_id": 321,
            "timeout_ms": 1234,
            "inputs": [
                MessageToDict(i, preserving_proto_field_name=True)
                for i in iter_multimodal_inputs(request, request.generate_config)
            ],
        }
        engine = SimpleNamespace(
            is_proxy_mode=False,
            _embedding_cache=MMEmbeddingCache(gpu_max_bytes=0, cpu_max_bytes=0),
            _hash_key_cache=MMHashKeyCache(max_bytes=0),
            get_embedding_result=Mock(
                return_value=[
                    MMEmbeddingRes(
                        [], feature_hashes=[torch.tensor([-7, 8], dtype=torch.int32)]
                    )
                ]
            ),
        )
        app = FastAPI()
        register_mm_cache_routes(app, engine)
        with TestClient(app) as client:
            response = client.post(
                "/mm_cache/metadata",
                json=payload,
                headers={
                    "x-DashScope-uID": "uid-http",
                    "X-DashScope-Service": "service-http",
                },
            )
            self.assertEqual(response.status_code, 200, response.text)
            entry = response.json()["entries"][0]
            self.assertTrue(entry["hash_hit"])
            self.assertFalse(entry["embedding_hit"])
            self.assertEqual(entry["feature_hashes"], [-7, 8])
            self.assertEqual(entry["split_size"], [2])
            self.assertNotIn("embeddings", entry)
            call = engine.get_embedding_result.call_args
            self.assertEqual(call.args[0][0].cache_key(), key)
            self.assertEqual(
                call.kwargs,
                {
                    "request_id": 321,
                    "timeout_ms": 1234,
                    "hashes_only": True,
                    "user_id": "uid-http",
                    "service_name": "service-http",
                },
            )

            engine.get_embedding_result.reset_mock()
            response = client.post(
                "/mm_cache/metadata", json={**payload, "keys": ["wrong-key"]}
            )
            self.assertEqual(response.status_code, 400)
            engine.get_embedding_result.assert_not_called()
            engine.get_embedding_result.side_effect = TimeoutError("timed out")
            self.assertEqual(
                client.post("/mm_cache/metadata", json=payload).status_code, 504
            )
            engine.get_embedding_result.side_effect = FtRuntimeException(
                ExceptionType.CONCURRENCY_LIMIT_ERROR, "full"
            )
            response = client.post("/mm_cache/metadata", json=payload)
            self.assertEqual(response.status_code, 503)
            self.assertEqual(
                response.json()["detail"]["error_code"],
                int(ExceptionType.CONCURRENCY_LIMIT_ERROR),
            )

            # A historical hash hit bypasses submit even when inputs are supplied.
            engine._hash_key_cache.resize(4096)
            engine._hash_key_cache.put(key, [torch.tensor([-7, 8], dtype=torch.int32)])
            engine.get_embedding_result.reset_mock()
            self.assertEqual(
                client.post("/mm_cache/metadata", json=payload).status_code, 200
            )
            engine.get_embedding_result.assert_not_called()

    def test_metadata_endpoint_never_computes_or_waits(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from rtp_llm.multimodal.mm_embedding_cache import (
            MMEmbeddingCache,
            MMHashKeyCache,
        )
        from rtp_llm.server.vit_app import register_mm_cache_routes

        cache = MMEmbeddingCache(gpu_max_bytes=0, cpu_max_bytes=4096)
        _, pending = cache.try_acquire("pending")
        _, ready = cache.try_acquire("ready")
        ready.complete(
            (torch.ones(2, 4), None), [torch.tensor([-11, 12], dtype=torch.int32)]
        )
        hash_keys = MMHashKeyCache(max_bytes=4096)
        hash_keys.put(
            "ready", [torch.tensor([-11, 12], dtype=torch.int32)], ready.generation
        )
        app = FastAPI()
        register_mm_cache_routes(
            app,
            SimpleNamespace(
                _embedding_cache=cache,
                _hash_key_cache=hash_keys,
                is_proxy_mode=False,
            ),
        )
        with TestClient(app) as client:
            snapshot = client.get("/mm_cache/keys").json()
            self.assertEqual(snapshot["keys"], ["ready"])
            self.assertEqual(snapshot["gpu_embedding_keys"], [])
            self.assertEqual(snapshot["cpu_embedding_keys"], ["ready"])
            result = client.post(
                "/mm_cache/metadata", json={"keys": ["ready", "pending", "absent"]}
            )
            self.assertEqual(result.status_code, 200)
            entries = result.json()["entries"]
            self.assertEqual(entries[0]["feature_hashes"], [-11, 12])
            self.assertEqual([e["hit"] for e in entries], [True, False, False])
            self.assertEqual([e["hash_hit"] for e in entries], [True, False, False])
            self.assertEqual(
                [e["embedding_hit"] for e in entries], [True, False, False]
            )
            self.assertEqual(entries[0]["embedding_tier"], "cpu")
            self.assertFalse(pending.is_done)
            self.assertIsNone(cache.peek("absent"))
            # A byte-bounded index can outgrow the directory response limit.
            # Publish a bounded recent snapshot without evicting cached keys.
            hash_keys.put("recent", [torch.tensor([42], dtype=torch.int32)])
            with patch("rtp_llm.server.vit_app.MM_CACHE_SNAPSHOT_MAX_KEYS", 1):
                snapshot = client.get("/mm_cache/keys")
            self.assertEqual(snapshot.status_code, 200)
            self.assertEqual(snapshot.json()["keys"], ["recent"])
            self.assertEqual(hash_keys.keys(), ["ready", "recent"])
            cache.remove("ready")
            history = client.post(
                "/mm_cache/metadata", json={"keys": ["ready"]}
            ).json()["entries"][0]
            self.assertTrue(history["hash_hit"])
            self.assertFalse(history["embedding_hit"])
            self.assertEqual(history["feature_hashes"], [-11, 12])
            _, embedding_only = cache.try_acquire("embedding-only")
            embedding_only.complete(torch.ones(1))
            snapshot = client.get("/mm_cache/keys").json()
            self.assertEqual(snapshot["keys"], ["ready", "recent"])
            self.assertEqual(snapshot["cpu_embedding_keys"], ["embedding-only"])
            self.assertEqual(
                client.post(
                    "/mm_cache/metadata", json={"keys": ["x"] * 257}
                ).status_code,
                422,
            )
            self.assertEqual(
                client.post(
                    "/mm_cache/metadata", json={"keys": ["x" * 4097]}
                ).status_code,
                400,
            )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_rdma_response_keeps_inline_feature_hashes(self):
        from unittest.mock import Mock

        from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import MMRdmaDescPB
        from rtp_llm.multimodal.mm_process_engine import MMEmbeddingRes
        from rtp_llm.server.vit_rpc_server import MultimodalRpcServer
        from rtp_llm.utils.grpc_util import trans_tensor

        server = MultimodalRpcServer.__new__(MultimodalRpcServer)
        server._rdma = Mock()
        server._rdma.export_embedding.return_value = [
            MMRdmaDescPB(handle="handle").SerializeToString()
        ]
        hashes = torch.tensor([-4, 5], dtype=torch.int32)
        result = MMEmbeddingRes(
            [torch.ones(2, 4, device="cuda")], feature_hashes=[hashes]
        )
        response = server._trans_output_rdma(result)
        self.assertEqual(response.output_rdma.handle, "handle")
        self.assertTrue(
            torch.equal(trans_tensor(response.multimodal_feature_hash), hashes)
        )
        self.assertFalse(response.HasField("multimodal_embedding"))


if __name__ == "__main__":
    unittest.main()
