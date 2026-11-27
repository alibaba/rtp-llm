
import importlib
import unittest
from unittest import mock

from navi_sched import NaviSchedAgent
from navi_sched import _navi_sched_core as core


class NativePPAgentConfigTest(unittest.TestCase):
    def options(self):
        policy = core.TPUPrefillExecutionPolicyConfig()
        policy.pcp_size = 8
        policy.interleave_size = 256
        policy.compile_sizes = [256, 512, 1024]
        policy.max_scheduled_tokens = 8192
        policy.hash_block_size = 4096
        policy.scheduler_block_size = 4096
        policy.async_kv_lookup_enabled = True
        policy.async_kv_lookup_hash_block_size = 4096
        return dict(biz_name="native_config_contract", server_ip="127.0.0.1",
                    server_port=0, total_blocks=128, block_size=4096,
                    linear_step=1, hybrid_tail_required=False,
                    pd_separated=True, latency_prediction_mode="unified",
                    pd_unified_capacity_split=True,
                    native_prefill_pipeline_stages=4,
                    prefill_execution_policy=policy,
                    enable_kmon=False, enable_snapshot=False, wait_no_query=False)

    def test_real_constructor_consumes_and_preserves_observation_flag(self):
        module = importlib.import_module("navi_sched.agent")
        original = module.makeRuntimeConfig
        configs = []
        def capture(*args, **kwargs):
            config = original(*args, **kwargs)
            configs.append(config)
            return config
        for enabled in (False, True):
            with self.subTest(enabled=enabled), mock.patch.object(module, "makeRuntimeConfig", side_effect=capture):
                agent = NaviSchedAgent(**self.options(), native_prefill_observation_only=enabled)
                self.addCleanup(agent.stop)
                self.assertEqual(enabled, configs[-1].native_prefill_observation_only)
                self.assertEqual(4, configs[-1].native_prefill_pipeline_stages)

    def test_real_cpp_runtime_initializes_for_both_native_modes(self):
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                agent = NaviSchedAgent(**self.options(), native_prefill_observation_only=enabled)
                try:
                    self.assertIs(agent, agent.init())
                    self.assertGreater(agent.grpc_port, 0)
                finally:
                    agent.stop()

    def test_ordinary_non_pp_constructor_retains_false_default(self):
        options = self.options()
        options.pop("prefill_execution_policy")
        options["native_prefill_pipeline_stages"] = 0
        agent = NaviSchedAgent(**options)
        self.addCleanup(agent.stop)

    def test_observation_only_requires_pp4(self):
        options = self.options()
        options["native_prefill_pipeline_stages"] = 0
        with self.assertRaisesRegex(ValueError, "observation-only.*PP4"):
            NaviSchedAgent(**options, native_prefill_observation_only=True)

    def test_observation_flag_rejects_non_booleans(self):
        for value in (1, "true", None):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "observation-only"):
                NaviSchedAgent(**self.options(), native_prefill_observation_only=value)


if __name__ == "__main__":
    unittest.main()
