"""SM120 gating tests for automatic DeepEP selection and DSv4 startup validation.

Covers the behavior added with "disable automatic DeepEP on SM120 workers":
automatic DeepEP fallback consults only participating devices, rejects exactly
the SM120 (12, 0) architecture for model workers, and leaves explicit user
configuration, control roles, and dense topologies untouched. All hardware
access is statically mocked; no CUDA device is required.
"""

import os
from contextlib import ExitStack
from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import patch

from rtp_llm.config.py_config_modules import DeepEPConfig
from rtp_llm.config.server_config_setup import (
    _auto_deepep_supported_on_visible_devices,
    auto_configure_deepep,
    setup_and_configure_server,
)
from rtp_llm.ops import MoeConfig, ParallelismConfig, RoleType, SpeculativeType

_TORCH_CUDA = "rtp_llm.config.server_config_setup.torch.cuda"


def _cuda_env(test_case, capabilities, available=True):
    stack = ExitStack()
    test_case.addCleanup(stack.close)
    stack.enter_context(patch(f"{_TORCH_CUDA}.is_available", return_value=available))
    stack.enter_context(
        patch(f"{_TORCH_CUDA}.device_count", return_value=len(capabilities))
    )
    capability = stack.enter_context(
        patch(
            f"{_TORCH_CUDA}.get_device_capability",
            side_effect=capabilities.__getitem__,
        )
    )
    return stack, capability


class Sm120ParticipatingDeviceTest(TestCase):
    def test_only_sm120_is_rejected(self):
        for capabilities, expected in (
            ([(9, 0), (9, 0)], True),
            ([(10, 0), (10, 0)], True),
            ([(12, 1)], True),
            ([(12, 0), (12, 0)], False),
            ([(9, 0), (12, 0)], False),
        ):
            with self.subTest(capabilities=capabilities):
                stack, _ = _cuda_env(self, capabilities)
                with stack:
                    self.assertEqual(
                        _auto_deepep_supported_on_visible_devices(len(capabilities)),
                        expected,
                    )

    def test_devices_outside_local_world_do_not_participate(self):
        # A stray SM120 card excluded by CUDA_VISIBLE_DEVICES must not flip
        # the decision for the participating ranks.
        stack, capability = _cuda_env(self, [(9, 0), (12, 0)])
        with stack:
            self.assertTrue(_auto_deepep_supported_on_visible_devices(1))
            self.assertEqual([call.args[0] for call in capability.call_args_list], [0])
        stack, _ = _cuda_env(self, [(9, 0), (12, 0)])
        with stack:
            self.assertFalse(_auto_deepep_supported_on_visible_devices(0))

    def test_cpu_only_config_parsing_stays_supported(self):
        stack, capability = _cuda_env(self, [], available=False)
        with stack:
            self.assertTrue(_auto_deepep_supported_on_visible_devices(4))
            capability.assert_not_called()


class Sm120AutoConfigureGateTest(TestCase):
    @staticmethod
    def _parallel(world_size, tp_size, ep_size, local_world_size, dp_size=1):
        config = ParallelismConfig()
        config.world_size = world_size
        config.tp_size = tp_size
        config.ep_size = ep_size
        config.dp_size = dp_size
        config.local_world_size = local_world_size
        return config

    def _configure(
        self,
        capabilities,
        parallel,
        role_type,
        deep_ep_config=None,
        model_type="deepseek_v4",
    ):
        moe_config = MoeConfig()
        moe_config.use_all_gather = False
        stack, capability = _cuda_env(self, capabilities)
        with stack:
            auto_configure_deepep(
                moe_config,
                deep_ep_config or DeepEPConfig(),
                parallel,
                role_type,
                model_type=model_type,
            )
        return moe_config, capability

    def test_sm120_model_workers_disable_automatic_deepep(self):
        parallel = self._parallel(
            world_size=8, tp_size=2, ep_size=4, local_world_size=8, dp_size=4
        )
        for role in (RoleType.PREFILL, RoleType.DECODE, RoleType.PDFUSION):
            with self.subTest(role=role):
                moe_config, _ = self._configure([(12, 0)] * 8, parallel, role)
                self.assertFalse(moe_config.use_deepep_moe)
                self.assertFalse(moe_config.use_deepep_low_latency)
                self.assertFalse(moe_config.use_deepep_internode)

    def test_supported_architectures_keep_automatic_selection(self):
        parallel = self._parallel(
            world_size=8, tp_size=2, ep_size=4, local_world_size=8, dp_size=4
        )
        for capability in ((9, 0), (10, 0), (12, 1)):
            with self.subTest(capability=capability):
                moe_config, _ = self._configure(
                    [capability] * 8, parallel, RoleType.DECODE
                )
                self.assertTrue(moe_config.use_deepep_moe)
                self.assertTrue(moe_config.use_deepep_low_latency)
                self.assertFalse(moe_config.use_deepep_internode)

    def test_other_models_and_pipeline_parallel_keep_main_defaults(self):
        parallel = self._parallel(
            world_size=2, tp_size=2, ep_size=2, local_world_size=2, dp_size=2
        )
        for model_type, pp_size in (("qwen_2", 1), ("deepseek_v4", 2)):
            with self.subTest(model_type=model_type, pp_size=pp_size):
                parallel.pp_size = pp_size
                moe_config, capability = self._configure(
                    [(12, 0)] * 2, parallel, RoleType.DECODE, model_type=model_type
                )
                self.assertTrue(moe_config.use_deepep_moe)
                self.assertTrue(moe_config.use_deepep_low_latency)
                self.assertFalse(moe_config.use_deepep_internode)
                capability.assert_not_called()

    def test_dense_topology_never_probes_architecture(self):
        # ep_size == 1 has no DeepEP path on any architecture; the gate must
        # not consume the device probe for dense/nonPP-shaped topologies.
        parallel = self._parallel(
            world_size=8, tp_size=2, ep_size=1, local_world_size=8, dp_size=8
        )
        moe_config, capability = self._configure(
            [(12, 0)] * 8, parallel, RoleType.DECODE
        )
        self.assertTrue(moe_config.use_deepep_moe)
        capability.assert_not_called()

    def test_unnormalized_ep_zero_keeps_main_behavior(self):
        # device_base.py invokes this before set_parallelism_config normalizes
        # ep_size; raw values must flow to the automatic path without probing.
        for capability in ((9, 0), (12, 0)):
            with self.subTest(capability=capability):
                parallel = self._parallel(
                    world_size=2, tp_size=2, ep_size=0, local_world_size=2
                )
                moe_config, probe = self._configure(
                    [capability], parallel, RoleType.DECODE
                )
                self.assertTrue(moe_config.use_deepep_moe)
                self.assertTrue(moe_config.use_deepep_low_latency)
                self.assertFalse(moe_config.use_deepep_internode)
                probe.assert_not_called()

    def test_control_roles_skip_the_gate(self):
        parallel = self._parallel(
            world_size=2, tp_size=1, ep_size=2, local_world_size=2, dp_size=2
        )
        moe_config, capability = self._configure(
            [(12, 0)] * 2, parallel, RoleType.FRONTEND
        )
        self.assertFalse(moe_config.use_deepep_moe)
        capability.assert_not_called()

    def test_explicit_user_selection_bypasses_the_gate(self):
        parallel = self._parallel(
            world_size=8, tp_size=2, ep_size=4, local_world_size=8, dp_size=4
        )
        deep_ep_config = DeepEPConfig()
        deep_ep_config.use_deepep_moe = True
        moe_config, capability = self._configure(
            [(12, 0)] * 8, parallel, RoleType.DECODE, deep_ep_config
        )
        self.assertTrue(moe_config.use_deepep_moe)
        capability.assert_not_called()
