"""Exercise transport admission without loading a device or starting an Agent."""

import ast
import asyncio
import logging
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

ADAPTER = Path(__file__).resolve().parents[2] / "adapters/vllm"
source = ADAPTER / "VLLMEngineTransport.py"
tree = ast.parse(source.read_text())
node = next(
    n
    for n in tree.body
    if isinstance(n, ast.ClassDef) and n.name == "VLLMEngineTransport"
)
namespace = dict(
    asyncio=asyncio,
    logger=logging.getLogger(__name__),
    forward_request_fields=lambda request: request,
    _make_route_failure=lambda error: error,
)
module = ast.parse("from __future__ import annotations\n")
module.body.append(node)
exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
Transport = namespace["VLLMEngineTransport"]


class RoutingTest(unittest.IsolatedAsyncioTestCase):
    def make_transport(self, master=False, hybrid=False):
        t = object.__new__(Transport)
        t._hybrid_cache = hybrid
        t._request_ids, t._external_request_ids, t._remote_cache_stats = {}, {}, {}
        t._client_init_lock = asyncio.Lock()
        t._engine_core = SimpleNamespace(
            add_request_async=AsyncMock(), call_utility_async=AsyncMock(return_value=32)
        )
        client = SimpleNamespace(
            has_master_biz=lambda: master,
            allocate_request_id=lambda: 1,
            report_local_fallback=Mock(),
            route_async=AsyncMock(side_effect=TimeoutError("RPC response lost")),
        )
        t._client = None if hybrid else client

        def initialize(block_size):
            self.assertEqual(block_size, 32)
            t._client = client

        t._initialize_client = initialize
        t._ensure_local_output_task = Mock()
        t._ensure_remote_output_task = Mock()
        t._encoder = SimpleNamespace(encode=lambda request: [b"payload"])
        t._msgspec = SimpleNamespace(
            msgpack=SimpleNamespace(encode=lambda value: b"wire")
        )
        t._adapter_config = SimpleNamespace(
            master_addresses=(), master_vip_domain="", target_biz_name="test"
        )
        return t

    async def test_embeddings_fall_back_for_streaming_and_nonstreaming(self):
        for stream in (False, True):
            t = self.make_transport()
            request = SimpleNamespace(
                request_id=str(stream),
                prompt_token_ids=None,
                prompt_embeds=object(),
                stream=stream,
            )
            await t.add_request_async(request)
            t._engine_core.add_request_async.assert_awaited_once_with(request)
            self.assertTrue(request.navi_sched_local_fallback)
            t._client.route_async.assert_not_awaited()

    async def test_remote_embeddings_still_require_token_ids(self):
        t = self.make_transport(master=True)
        with self.assertRaisesRegex(ValueError, "requires prompt token ids"):
            await t.add_request_async(
                SimpleNamespace(request_id="embed", prompt_token_ids=None)
            )
        t._engine_core.add_request_async.assert_not_awaited()
        t._client.route_async.assert_not_awaited()

    async def test_hybrid_first_request_initializes_then_falls_back(self):
        for native_prefill in (False, True):
            t = self.make_transport(hybrid=True)
            result = (
                {
                    "block_size": 32,
                    "trim_last_prompt_token": False,
                    "prefix_cache_lookup_enabled": True,
                }
                if native_prefill
                else 32
            )
            t._engine_core.call_utility_async.return_value = result
            # _ensure_client looks up the mode in the transport module globals.
            with patch.dict(
                namespace, is_vllm_native_pd_prefill=lambda: native_prefill
            ):
                request = SimpleNamespace(request_id="hybrid", prompt_token_ids=None)
                await t.add_request_async(request)
            method = (
                "get_navi_sched_routing_metadata"
                if native_prefill
                else "get_navi_sched_hash_block_size"
            )
            t._engine_core.call_utility_async.assert_awaited_once_with(method)
            t._engine_core.add_request_async.assert_awaited_once_with(request)

    async def test_rpc_timeout_never_executes_a_second_local_request(self):
        t = self.make_transport(master=True)
        with self.assertRaises(TimeoutError):
            await t.add_request_async(
                SimpleNamespace(request_id="timeout", prompt_token_ids=[1])
            )
        t._client.route_async.assert_awaited_once()
        t._engine_core.add_request_async.assert_not_awaited()
        self.assertEqual(t._request_ids, {})
        self.assertEqual(t._external_request_ids, {})


class PluginOptInTest(unittest.TestCase):
    def test_installed_entrypoint_does_not_enable_itself(self):
        source = ADAPTER.parents[1] / "navi_sched_vllm_plugin.py"
        namespace = {}
        exec(compile(source.read_text(), str(source), "exec"), namespace)
        with patch.dict(os.environ, {}, clear=True):
            # No vLLM import is needed or attempted on this native path.
            namespace["register_vllm_plugin"]()
            self.assertNotIn("PYTHONHASHSEED", os.environ)


if __name__ == "__main__":
    unittest.main()
