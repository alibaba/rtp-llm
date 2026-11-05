import types
import unittest

from navi_sched.adapters.dashserving import (
    DashServingMasterBizAbsent,
    DashServingMasterUnavailable,
    DashServingRequestError,
    DashServingRoutingError,
    NaviSchedDashServingRouter,
)
from navi_sched.adapters.dashserving.navi_sched_dashserving_adapter import (
    _config_from_env,
)
from navi_sched.adapters.vllm import (
    NaviSchedAdapterConfig,
    NaviSchedClient,
    NaviSchedEndpointRouter,
    calculate_vllm_block_hashes,
    stable_request_id,
)
from navi_sched.sched import NaviSchedScheduleError, decisionFromNative


class NativeClient:
    def __init__(self):
        self.calls = []
        self.stopped = False
        self.master_present = True
        self.fallback_reports = 0
        self.schedule_error = None

    def has_master_biz(self, unused_name):
        return self.master_present

    def report_local_fallback(self):
        self.fallback_reports += 1

    def schedule(self, input_length, **options):
        self.calls.append((input_length, options))
        if self.schedule_error is not None:
            raise self.schedule_error
        return types.SimpleNamespace(
            target=types.SimpleNamespace(address="127.0.0.1:28001")
        )

    def stop(self):
        self.stopped = True


class DashServingAdapterTest(unittest.TestCase):
    def test_spectrum_master_service_id_from_environment(self):
        config = _config_from_env(
            {
                "DS_NAVI_SCHED_MASTER_BIZ_NAME": "master",
                "DS_NAVI_SCHED_TARGET_BIZ_NAME": "prefill",
                "DS_NAVI_SCHED_MASTER_SPECTRUM_SERVICE_ID": "v-c123ec48",
            }
        )
        self.assertEqual("v-c123ec48", config.master_spectrum_service_id)
        with self.assertRaisesRegex(
            DashServingRoutingError, "exactly one NaviSched Master discovery mode"
        ):
            _config_from_env(
                {
                    "DS_NAVI_SCHED_MASTER_BIZ_NAME": "master",
                    "DS_NAVI_SCHED_TARGET_BIZ_NAME": "prefill",
                    "DS_NAVI_SCHED_MASTER_SPECTRUM_SERVICE_ID": "v-c123ec48",
                    "DS_NAVI_SCHED_MASTER_VIP_DOMAIN": "master.vip",
                }
            )

    def test_vip_master_grpc_port_offset_from_environment(self):
        config = _config_from_env(
            {
                "DS_NAVI_SCHED_MASTER_BIZ_NAME": "master",
                "DS_NAVI_SCHED_TARGET_BIZ_NAME": "prefill",
                "DS_NAVI_SCHED_MASTER_VIP_DOMAIN": "master.vip",
                "DS_NAVI_SCHED_MASTER_GRPC_PORT_DIFF": "10999",
            }
        )
        self.assertEqual(10999, config.master_vip_grpc_port_diff)
        with self.assertRaisesRegex(
            DashServingRoutingError,
            "DS_NAVI_SCHED_MASTER_GRPC_PORT_DIFF must be an integer",
        ):
            _config_from_env(
                {
                    "DS_NAVI_SCHED_MASTER_BIZ_NAME": "master",
                    "DS_NAVI_SCHED_TARGET_BIZ_NAME": "prefill",
                    "DS_NAVI_SCHED_MASTER_VIP_DOMAIN": "master.vip",
                    "DS_NAVI_SCHED_MASTER_GRPC_PORT_DIFF": "not-an-integer",
                }
            )

    def test_native_error_retains_code_for_router_fallback(self):
        with self.assertRaises(NaviSchedScheduleError) as caught:
            decisionFromNative(types.SimpleNamespace(error_code=35, error_message=""))
        self.assertEqual(35, caught.exception.error_code)

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
        self.assertTrue(options["route_only"])
        self.assertEqual("prefill", options["target_biz_name"])
        self.assertEqual("sched_client", self.client_options["name"])
        self.assertTrue(self.client_options["kmon_platform_discovery"])

    def test_falls_back_only_when_master_biz_is_absent(self):
        self.native_client.master_present = False
        with self.assertRaises(DashServingMasterBizAbsent):
            self.router.route_sync(
                request_id="request",
                prompt={"prompt_token_ids": [1]},
            )
        self.assertEqual([], self.native_client.calls)
        self.assertEqual(1, self.native_client.fallback_reports)

    def test_unreachable_static_master_uses_fallback(self):
        self.native_client.schedule_error = NaviSchedScheduleError(35, "")
        with self.assertRaises(DashServingMasterUnavailable):
            self.router.route_sync(
                request_id="request",
                prompt={"prompt_token_ids": [1]},
            )
        self.assertEqual(1, len(self.native_client.calls))
        self.assertEqual(1, self.native_client.fallback_reports)

    def test_scheduler_failure_does_not_fallback(self):
        self.native_client.schedule_error = NaviSchedScheduleError(
            32, "scheduler failed"
        )
        with self.assertRaises(DashServingRoutingError):
            self.router.route_sync(
                request_id="request",
                prompt={"prompt_token_ids": [1]},
            )
        self.assertEqual(0, self.native_client.fallback_reports)

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
            def has_master_biz(self):
                return True

            def route(self, unused_tokens, **unused):
                return types.SimpleNamespace(
                    target=types.SimpleNamespace(address="bad")
                )

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
            def has_master_biz(self):
                return True

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
