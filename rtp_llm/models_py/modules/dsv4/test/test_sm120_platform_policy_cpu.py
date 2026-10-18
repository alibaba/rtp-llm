"""CPU tests of production platform selection without native-library imports."""

from __future__ import annotations

import ast
import logging
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "rtp_llm/config/server_config_setup.py").is_file()
)


def load_functions(path, names, namespace):
    source = ROOT / path
    nodes = [
        node
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert {node.name for node in nodes} == set(names)
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    module = ast.fix_missing_locations(
        ast.Module(body=[future] + nodes, type_ignores=[])
    )
    exec(compile(module, str(source), "exec"), namespace)
    return namespace


class PlatformPolicyTest(unittest.TestCase):
    def setUp(self):
        self.roles = SimpleNamespace(
            PREFILL="prefill", DECODE="decode", PDFUSION="fusion"
        )
        self.env = load_functions(
            "rtp_llm/config/server_config_setup.py",
            (
                "_auto_deepep_supported_on_visible_devices",
                "auto_configure_deepep",
                "_apply_auto_deepep_config",
            ),
            {"torch": torch, "logging": logging, "RoleType": self.roles},
        )

    def test_only_participating_devices_affect_selection(self):
        guard = self.env["_auto_deepep_supported_on_visible_devices"]
        with patch.object(torch.cuda, "is_available", return_value=True), patch.object(
            torch.cuda, "device_count", return_value=2
        ), patch.object(
            torch.cuda,
            "get_device_capability",
            side_effect=lambda i: [(10, 0), (12, 0)][i],
        ):
            self.assertTrue(guard(1))
            self.assertFalse(guard(2))
            self.assertFalse(guard(0))
            self.assertFalse(guard(8))

    def test_cpu_configuration_does_not_probe_devices(self):
        with patch.object(torch.cuda, "is_available", return_value=False), patch.object(
            torch.cuda, "get_device_capability"
        ) as capability:
            self.assertTrue(self.env["_auto_deepep_supported_on_visible_devices"](8))
        capability.assert_not_called()

    def configure(self, capability, *, ep=8, explicit=None):
        moe = SimpleNamespace(
            moe_strategy="auto",
            use_all_gather=False,
            use_mori_ep=False,
            use_deepep_moe=False,
            use_deepep_low_latency=False,
            use_deepep_internode=False,
        )
        user = SimpleNamespace(
            use_deepep_moe=explicit,
            use_deepep_low_latency=None,
            use_deepep_internode=None,
            use_mori_ep=None,
        )
        parallel = SimpleNamespace(
            tp_size=1,
            ep_size=ep,
            dp_size=8,
            world_size=8,
            local_world_size=8,
            prefill_cp_config=SimpleNamespace(is_enabled=lambda: False),
        )
        with patch.object(torch.cuda, "is_available", return_value=True), patch.object(
            torch.cuda, "device_count", return_value=8
        ), patch.object(
            torch.cuda, "get_device_capability", return_value=capability
        ) as probe:
            self.env["auto_configure_deepep"](moe, user, parallel, self.roles.DECODE)
        return moe, probe

    def test_sm120_disables_automatic_deepep_but_preserves_sm100(self):
        for capability, expected in (
            ((12, 0), False),
            ((12, 1), False),
            ((10, 0), True),
            ((9, 0), True),
        ):
            with self.subTest(capability=capability):
                moe, _ = self.configure(capability)
                self.assertEqual(moe.use_deepep_moe, expected)
                self.assertEqual(moe.use_deepep_low_latency, expected)
                self.assertFalse(moe.use_deepep_internode)

    def test_dense_and_explicit_configuration_do_not_probe_hardware(self):
        moe, probe = self.configure((12, 0), ep=1)
        self.assertFalse(moe.use_deepep_moe)
        probe.assert_not_called()
        moe, probe = self.configure((12, 0), explicit=False)
        self.assertFalse(moe.use_deepep_moe)
        probe.assert_not_called()

    def test_mhc_uses_architecture_default_and_honors_explicit_backend(self):
        env = load_functions(
            "rtp_llm/models_py/utils/arch.py",
            ("mhc_pre_gemm_backend",),
            {
                "os": os,
                "is_sm12x": lambda device: device == 12,
            },
        )
        resolve = env["mhc_pre_gemm_backend"]
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(resolve(12), "tilelang_single")
            self.assertEqual(resolve(10), "deepgemm")
        for value, expected in (
            ("dg", "deepgemm"),
            ("tilelang", "tilelang_single"),
            ("single", "tilelang_single"),
        ):
            with patch.dict(os.environ, {"DSV4_MHC_PRE_GEMM_BACKEND": value}):
                self.assertEqual(resolve(12), expected)


class DecodeStrategyAdmissionTest(unittest.TestCase):
    def setUp(self):
        self.is_sm120 = Mock(return_value=True)
        self.namespace = load_functions(
            "rtp_llm/models_py/model_desc/deepseek_v4_model.py",
            ("_resolve_dsv4_moe_strategy", "_validate_sm120_dp_moe_strategy"),
            {
                "os": os,
                "is_sm120": self.is_sm120,
                "_DSV4_MOE_STRATEGY_ALIASES": {
                    "sm120_decode",
                    "sm120_fused_moe",
                },
            },
        )
        self.validate = self.namespace["_validate_sm120_dp_moe_strategy"]

    def test_dp_ep_rejects_unsynchronized_world_strategies(self):
        for role in ("decode", "prefill", "fusion"):
            for strategy in ("auto", "sm120_fused_moe"):
                with self.subTest(role=role, strategy=strategy), self.assertRaisesRegex(
                    ValueError, "--moe_strategy=sm120_decode"
                ):
                    self.validate(
                        SimpleNamespace(role_type=role, dp_size=4, ep_size=4), strategy
                    )

    def test_decode_accepts_explicit_and_alias_strategy(self):
        resolve = self.namespace["_resolve_dsv4_moe_strategy"]
        for requested, alias in (
            ("sm120_decode", "sm120_fused_moe"),
            ("auto", "sm120_decode"),
        ):
            with patch.dict(os.environ, {"DSV4_MOE_STRATEGY": alias}):
                strategy = resolve(SimpleNamespace(moe_strategy=requested))
            self.validate(
                SimpleNamespace(role_type="decode", dp_size=8, ep_size=8), strategy
            )
        self.is_sm120.assert_not_called()

    def test_cp_prefill_and_independent_dp_keep_strategy_selection(self):
        for dp, ep in ((1, 8), (4, 1), (1, 1)):
            for strategy in ("auto", "sm120_fused_moe"):
                self.validate(SimpleNamespace(dp_size=dp, ep_size=ep), strategy)
        self.is_sm120.assert_not_called()

    def test_other_architectures_keep_decode_strategy_selection(self):
        self.is_sm120.return_value = False
        for strategy in ("auto", "sm120_fused_moe"):
            self.validate(SimpleNamespace(dp_size=8, ep_size=8), strategy)


if __name__ == "__main__":
    unittest.main()
