import asyncio
import importlib
import sys
import types
import unittest
from unittest import mock


class RoutingInterface:
    pass


routing_logic = types.ModuleType("vllm_router.routers.routing_logic")
routing_logic.RoutingInterface = RoutingInterface
routers = types.ModuleType("vllm_router.routers")
routers.routing_logic = routing_logic
vllm_router = types.ModuleType("vllm_router")
vllm_router.routers = routers
sys.modules["vllm_router"] = vllm_router
sys.modules["vllm_router.routers"] = routers
sys.modules["vllm_router.routers.routing_logic"] = routing_logic


class ClientTimeout:
    def __init__(self, total=None):
        self.total = total


aiohttp = types.ModuleType("aiohttp")
aiohttp.ClientTimeout = ClientTimeout
sys.modules["aiohttp"] = aiohttp

router_module = importlib.import_module(
    "navi_sched.adapters.vllm.NaviSchedProductionStackRouter"
)
NaviSchedProductionStackRouter = (
    router_module.NaviSchedProductionStackRouter
)
NaviSchedAdapterConfig = router_module.NaviSchedAdapterConfig


class AsyncResponse:
    def __init__(self, payload):
        self._payload = payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

    def raise_for_status(self):
        return None

    async def json(self):
        return self._payload


class ProductionStackRouterTest(unittest.TestCase):
    def setUp(self):
        self.config = NaviSchedAdapterConfig(
            master_biz_name="master",
            target_biz_name="model",
            master_addresses=("127.0.0.1:19000",),
        )
        self.client = mock.Mock()
        self.client.route_async = mock.AsyncMock(
            return_value=mock.Mock(
                target=mock.Mock(address="127.0.0.1:28001")
            )
        )
        self.router = NaviSchedProductionStackRouter(
            config=self.config,
            client_factory=lambda unused, **options: self.client,
        )
        self.session = mock.Mock()
        self.request = mock.Mock()
        self.request.headers = {"X-Request-Id": "request_0"}
        self.request.url.path = "/v1/completions"
        self.request.app.state.aiohttp_client_wrapper.return_value = (
            self.session
        )
        self.endpoints = [
            mock.Mock(
                url="http://127.0.0.1:28000",
                model_names=["model"],
            ),
            mock.Mock(
                url="http://127.0.0.1:28001",
                model_names=["model"],
            ),
        ]

    def tearDown(self):
        self.router.close()

    def test_completion_is_tokenized_and_only_endpoint_is_returned(self):
        self.session.post.return_value = AsyncResponse(
            {"tokens": [1, 2, 3]}
        )
        request_json = {
            "model": "model",
            "prompt": "hello",
            "add_special_tokens": False,
            "max_tokens": 16,
            "priority": 3,
        }

        result = asyncio.run(self.router.route_request(
            self.endpoints,
            {},
            {},
            self.request,
            request_json,
        ))

        self.assertEqual("http://127.0.0.1:28001", result)
        self.session.post.assert_called_once()
        post_args = self.session.post.call_args
        self.assertEqual(
            "http://127.0.0.1:28000/tokenize",
            post_args.args[0],
        )
        self.assertEqual(
            {
                "model": "model",
                "prompt": "hello",
                "add_special_tokens": False,
            },
            post_args.kwargs["json"],
        )
        self.client.route_async.assert_awaited_once_with(
            [1, 2, 3],
            request_id=self.router._get_sched_request_id(
                self.request, request_json
            ),
            forward_payload=self.router._get_engine_request_id(
                self.request, request_json
            ).encode("utf-8"),
        )

    def test_chat_tokenize_payload_preserves_template_options(self):
        payload = self.router._make_tokenize_payload({
            "model": "model",
            "messages": [{"role": "user", "content": "hello"}],
            "add_generation_prompt": True,
            "chat_template_kwargs": {"enable_thinking": False},
        })

        self.assertEqual("model", payload["model"])
        self.assertEqual(True, payload["add_generation_prompt"])
        self.assertEqual(
            {"enable_thinking": False},
            payload["chat_template_kwargs"],
        )

    def test_token_ids_skip_tokenize_request(self):
        result = asyncio.run(self.router.route_request(
            self.endpoints,
            {},
            {},
            self.request,
            {
                "model": "model",
                "prompt": [4, 5, 6],
                "max_completion_tokens": 8,
            },
        ))

        self.assertEqual("http://127.0.0.1:28001", result)
        self.session.post.assert_not_called()
        self.client.route_async.assert_awaited_once_with(
            [4, 5, 6],
            request_id=self.router._get_sched_request_id(
                self.request,
                {"model": "model", "prompt": [4, 5, 6]},
            ),
            forward_payload=self.router._get_engine_request_id(
                self.request,
                {"model": "model", "prompt": [4, 5, 6]},
            ).encode("utf-8"),
        )

    def test_generated_router_request_id_is_forwarded_to_master(self):
        self.request.headers = {}
        self.request.state.request_id = "router_request_0"

        asyncio.run(self.router.route_request(
            self.endpoints,
            {},
            {},
            self.request,
            {"model": "model", "prompt": [4, 5, 6]},
        ))

        self.assertEqual(
            self.router._get_sched_request_id(
                self.request,
                {"model": "model", "prompt": [4, 5, 6]},
            ),
            self.client.route_async.await_args.kwargs["request_id"],
        )

    def test_chat_request_id_matches_vllm_engine_request_id(self):
        self.request.url.path = "/v1/chat/completions"
        self.session.post.return_value = AsyncResponse({"tokens": [4, 5, 6]})

        asyncio.run(self.router.route_request(
            self.endpoints,
            {},
            {},
            self.request,
            {
                "model": "model",
                "messages": [{"role": "user", "content": "hello"}],
            },
        ))

        self.assertEqual(
            self.router._get_sched_request_id(
                self.request,
                {
                    "model": "model",
                    "messages": [{"role": "user", "content": "hello"}],
                },
            ),
            self.client.route_async.await_args.kwargs["request_id"],
        )

    def test_sched_request_id_is_a_stable_nonzero_uint64(self):
        request_json = {"model": "model", "prompt": [4, 5, 6]}

        request_id = self.router._get_sched_request_id(
            self.request, request_json
        )

        self.assertIsInstance(request_id, int)
        self.assertGreater(request_id, 0)
        self.assertLessEqual(request_id, (1 << 64) - 1)
        self.assertEqual(
            request_id,
            self.router._get_sched_request_id(self.request, request_json),
        )

    def test_cache_salt_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "cache_salt"):
            asyncio.run(self.router.route_request(
                self.endpoints,
                {},
                {},
                self.request,
                {
                    "model": "model",
                    "prompt": [4, 5, 6],
                    "cache_salt": "private",
                },
            ))

    def test_multiple_served_models_do_not_change_routing_request(self):
        self.endpoints[0].model_names.append("adapter")

        asyncio.run(self.router.route_request(
            self.endpoints,
            {},
            {},
            self.request,
            {"model": "model", "prompt": [4, 5, 6]},
        ))

        self.assertNotIn(
            "enable_cache_match",
            self.client.route_async.await_args.kwargs,
        )

    def test_target_must_be_an_eligible_endpoint(self):
        self.client.route_async.return_value = mock.Mock(
            target=mock.Mock(address="127.0.0.1:29000")
        )

        with self.assertRaisesRegex(RuntimeError, "not an eligible"):
            asyncio.run(self.router.route_request(
                self.endpoints,
                {},
                {},
                self.request,
                {"model": "model", "prompt": [1]},
            ))

    def test_close_releases_client_once(self):
        self.router.close()
        self.router.close()

        self.client.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
