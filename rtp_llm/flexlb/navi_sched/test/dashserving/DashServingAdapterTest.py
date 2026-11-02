import types
import unittest

from navi_sched.adapters.dashserving import (
    DashServingRequestError,
    DashServingRoutingError,
    NaviSchedDashServingRouter,
)
from navi_sched.adapters.vllm import (
    NaviSchedAdapterConfig,
    NaviSchedClient,
    NaviSchedEndpointRouter,
    calculate_vllm_block_hashes,
    stable_request_id,
)


class NativeClient:
    def __init__(self):
        self.calls = []
        self.stopped = False

    def schedule(self, input_length, **options):
        self.calls.append((input_length, options))
        return types.SimpleNamespace(
            target=types.SimpleNamespace(address="127.0.0.1:28001")
        )

    def stop(self):
        self.stopped = True


class DashServingAdapterTest(unittest.TestCase):
    def setUp(self):
        self.config = NaviSchedAdapterConfig(
            master_biz_name="master",
            target_biz_name="prefill",
            master_addresses=("127.0.0.1:19000",),
            block_size=4,
        )
        self.native_client = NativeClient()

        self.client_options = None

        def client_factory(config, **options):
            self.assertIs(config, self.config)
            self.client_options = options
            return NaviSchedClient(config, native_client=self.native_client)

        self.router = NaviSchedDashServingRouter(
            self.config,
            client_factory=client_factory,
        )

    def tearDown(self):
        self.router.close()

    def test_routes_tokens_with_vllm_hash_and_stable_request_id(self):
        token_ids = list(range(10))

        route = self.router.route_sync(
            request_id="chat-request",
            prompt={"prompt_token_ids": token_ids},
        )

        self.assertEqual("127.0.0.1:28001", route.target_address)
        self.assertEqual("127.0.0.1", route.target_host)
        self.assertEqual(28001, route.target_port)
        self.assertEqual(stable_request_id("chat-request"), route.scheduler_request_id)
        self.assertEqual(1, len(self.native_client.calls))
        input_length, options = self.native_client.calls[0]
        self.assertEqual(len(token_ids), input_length)
        self.assertEqual(4, options["block_size"])
        self.assertEqual(
            calculate_vllm_block_hashes(token_ids, 4),
            options["block_hashes"],
        )
        self.assertEqual(route.scheduler_request_id, options["request_id"])
        self.assertEqual("prefill", options["target_biz_name"])
        self.assertEqual("sched_client", self.client_options["name"])

    def test_rejects_hash_inputs_that_do_not_match_plain_vllm_tokens(self):
        invalid_prompts = (
            {"prompt_token_ids": [1], "multi_modal_data": {}},
            {"prompt_token_ids": [1], "cache_salt": "salt"},
            {"prompt_token_ids": [1], "lora_request": object()},
        )

        for prompt in invalid_prompts:
            with (
                self.subTest(prompt=prompt),
                self.assertRaises(DashServingRequestError),
            ):
                self.router.route_sync(
                    request_id="request",
                    prompt=prompt,
                )
        self.assertEqual([], self.native_client.calls)

    def test_rejects_hash_options_from_all_dashserving_parameter_sources(self):
        invalid_options = (
            {"params": {"cache_salt": "salt"}},
            {"params": types.SimpleNamespace(extra={"lora_id": 1})},
            {"extra_params": {"modalities": ["image"]}},
            {"extra_params": {"mm_processor_kwargs": {}}},
            {"params": {"n": 2}},
        )

        for options in invalid_options:
            with (
                self.subTest(options=options),
                self.assertRaises(DashServingRequestError),
            ):
                self.router.route_sync(
                    request_id="request",
                    prompt={"prompt_token_ids": [1]},
                    **options,
                )
        self.assertEqual([], self.native_client.calls)

    def test_rejects_invalid_request_identity_and_tokens(self):
        invalid = (
            ("", {"prompt_token_ids": [1]}),
            (1, {"prompt_token_ids": [1]}),
            ("request", {}),
            ("request", {"prompt_token_ids": [True]}),
            ("request", {"prompt_token_ids": ["1"]}),
        )

        for request_id, prompt in invalid:
            with (
                self.subTest(request_id=request_id, prompt=prompt),
                self.assertRaises(DashServingRequestError),
            ):
                self.router.route_sync(
                    request_id=request_id,
                    prompt=prompt,
                )

    def test_rejects_missing_or_invalid_prefill_target(self):
        class InvalidTargetClient:
            def route(self, unused_tokens, **unused):
                return types.SimpleNamespace(target=types.SimpleNamespace(address="bad"))

            def close(self):
                pass

        router = NaviSchedDashServingRouter(
            self.config,
            client_factory=lambda *args, **kwargs: InvalidTargetClient(),
        )
        with self.assertRaises(DashServingRoutingError):
            router.route_sync(
                request_id="request",
                prompt={"prompt_token_ids": [1]},
            )
        router.close()

    def test_rejects_prefill_target_with_invalid_port(self):
        class InvalidPortClient:
            def route(self, unused_tokens, **unused):
                return types.SimpleNamespace(
                    target=types.SimpleNamespace(address="127.0.0.1:70000")
                )

            def close(self):
                pass

        router = NaviSchedDashServingRouter(
            self.config,
            client_factory=lambda *args, **kwargs: InvalidPortClient(),
        )
        with self.assertRaises(DashServingRoutingError):
            router.route_sync(
                request_id="request",
                prompt={"prompt_token_ids": [1]},
            )
        router.close()

    def test_close_stops_the_navi_sched_client(self):
        self.router.route_sync(
            request_id="request",
            prompt={"prompt_token_ids": [1]},
        )

        self.router.close()
        self.router.close()

        self.assertTrue(self.native_client.stopped)

    def test_route_timeout_override_must_be_positive(self):
        endpoint_router = NaviSchedEndpointRouter(
            self.native_client,
            "prefill",
            timeout_ms=1000,
            block_size=4,
        )

        with self.assertRaisesRegex(ValueError, "timeout_ms must be positive"):
            endpoint_router.route([1], timeout_ms=0)

        self.assertEqual([], self.native_client.calls)


if __name__ == "__main__":
    unittest.main()
