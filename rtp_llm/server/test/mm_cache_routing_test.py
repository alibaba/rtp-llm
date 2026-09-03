import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

import torch

from rtp_llm.config.generate_config import GenerateConfig, RoleAddr, RoleType
from rtp_llm.cpp.model_rpc.model_rpc_client import (
    multimodal_cache_keys,
    trans_multimodal_input,
)
from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import GenerateInputPB
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
        "feature_hash_version": 1,
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

    def test_miss_and_unknown_version_return_only_safe_prefix(self):
        for data in (None, {"feature_hash_version": 2}):
            self.assertEqual(
                multimodal_routing_tokens([1, 99, 2], [[99]], False, ["a"], data, 100),
                ([1], None),
            )

    def test_invalid_hashes_are_rejected(self):
        data = metadata(["a"], [[1 << 32]])
        with self.assertRaises(ValueError):
            multimodal_routing_tokens([1, 99], [[99]], False, ["a"], data, 100)


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

    async def test_cache_probe_miss_or_invalid_metadata_keeps_inference_route(self):
        for data in (None, {"feature_hash_version": 1, "entries": [{}]}):
            await self.asyncSetUp()
            self.visitor.master_client.get_vit_cache_metadata.return_value = data
            self.assertIsNone(await self.visitor.get_master_route_addrs(self.request))
            calls = self.visitor.master_client.get_backend_role_addrs.call_args_list
            self.assertNotIn("seq_len", calls[1].kwargs)
            self.assertEqual(calls[1].kwargs["selected_vit"], self.status)
            self.assertEqual(calls[1].kwargs["block_cache_keys"], [])
            self.assertEqual(self.request.token_ids.tolist(), [1, 99, 2])

    async def test_old_master_falls_back_to_ordinary_schedule(self):
        self.visitor.master_client.get_backend_role_addrs.side_effect = [
            FlexlbResponse.error_response(404),
            FlexlbResponse.ok([self.prefill, self.vit]),
        ]
        self.assertIsNone(await self.visitor.get_master_route_addrs(self.request))
        calls = self.visitor.master_client.get_backend_role_addrs.call_args_list
        self.assertNotIn("selected_vit", calls[1].kwargs)
        self.visitor.master_client.get_vit_cache_metadata.assert_not_awaited()

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

    async def test_stale_vit_reselection_discards_hash_hints(self):
        new_vit = self.vit.model_copy(update={"ip": "127.0.0.3"})
        self.visitor.master_client.get_backend_role_addrs.side_effect = [
            FlexlbResponse(
                role_addrs=[self.vit], result={"server_status": [self.status]}
            ),
            FlexlbResponse.error_response(VIT_ROUTE_STALE_CODE),
            FlexlbResponse.ok([self.prefill, new_vit]),
        ]
        self.assertIsNone(await self.visitor.get_master_route_addrs(self.request))
        last = self.visitor.master_client.get_backend_role_addrs.call_args.kwargs
        self.assertNotIn("selected_vit", last)
        self.assertEqual(last["block_cache_keys"], [])
        self.assertEqual(last["seq_len"], 4)
        self.assertEqual(
            self.request.generate_config.role_addrs, [self.prefill, new_vit]
        )
        self.assertEqual(self.request.token_ids.tolist(), [1, 99, 2])

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

    async def test_metadata_http_failure_and_timeout_are_optional(self):
        import asyncio

        from aiohttp import web

        async def handler(request):
            key = (await request.json())["keys"][0]
            if key == "slow":
                await asyncio.sleep(0.6)
            if key == "ready":
                return web.json_response(metadata(["ready"], [[-1, 2]]))
            return web.Response(status=501)

        app = web.Application()
        app.router.add_post("/mm_cache/metadata", handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        address = RoleAddr(
            role=RoleType.VIT,
            ip="127.0.0.1",
            http_port=runner.addresses[0][1],
            grpc_port=8001,
        )
        client = MasterClient()
        try:
            self.assertEqual(
                (await client.get_vit_cache_metadata(address, ["ready"]))["entries"][0][
                    "feature_hashes"
                ],
                [-1, 2],
            )
            self.assertIsNone(
                await client.get_vit_cache_metadata(address, ["old-worker"])
            )
            self.assertIsNone(await client.get_vit_cache_metadata(address, ["slow"]))
        finally:
            await client.close()
            await runner.cleanup()


class MMCacheApiTest(unittest.TestCase):
    def test_metadata_endpoint_never_computes_or_waits(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from rtp_llm.multimodal.mm_embedding_cache import MMEmbeddingCache
        from rtp_llm.server.vit_app import register_mm_cache_routes

        cache = MMEmbeddingCache(max_size=4)
        _, pending = cache.try_acquire("pending")
        _, ready = cache.try_acquire("ready")
        ready.complete(
            (torch.ones(2, 4), None), [torch.tensor([-11, 12], dtype=torch.int32)]
        )
        app = FastAPI()
        register_mm_cache_routes(
            app, SimpleNamespace(_embedding_cache=cache, is_proxy_mode=False)
        )
        with TestClient(app) as client:
            snapshot = client.get("/mm_cache/keys").json()
            self.assertEqual(snapshot["keys"], ["ready"])
            result = client.post(
                "/mm_cache/metadata", json={"keys": ["ready", "pending", "absent"]}
            )
            self.assertEqual(result.status_code, 200)
            entries = result.json()["entries"]
            self.assertEqual(entries[0]["feature_hashes"], [-11, 12])
            self.assertEqual([e["hit"] for e in entries], [True, False, False])
            self.assertFalse(pending.is_done)
            self.assertIsNone(cache.peek("absent"))
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
        self.assertEqual(response.feature_hash_version, 1)
        self.assertTrue(
            torch.equal(trans_tensor(response.multimodal_feature_hash), hashes)
        )
        self.assertFalse(response.HasField("multimodal_embedding"))


if __name__ == "__main__":
    unittest.main()
