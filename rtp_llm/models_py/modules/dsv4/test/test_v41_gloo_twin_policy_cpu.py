"""CPU tests for the R20/C1+C2 gloo-twin routing policy.

Hermetic namespace-stub loading as in test_v41_deferred_checks_cpu: the
policy module is import-free by design, so the bare-shell battery can load
it without the rtp_llm.ops .so chain.
"""

import importlib
import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[5]
for suffix in [
    "",
    ".models_py",
    ".models_py.distributed",
]:
    name = "rtp_llm" + suffix
    module = types.ModuleType(name)
    module.__path__ = [str(ROOT.joinpath(*name.split(".")))]
    sys.modules.setdefault(name, module)

policy = importlib.import_module("rtp_llm.models_py.distributed._gloo_twin_policy")


class ValueFlagTest(unittest.TestCase):
    def test_absent_is_off(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            os.environ.pop(policy.CPU_ALLGATHER_GLOO_FLAG, None)
            os.environ.pop(policy.AGREE_DP_LEAN_FLAG, None)
            self.assertFalse(policy.value_flag(policy.CPU_ALLGATHER_GLOO_FLAG))
            self.assertFalse(policy.value_flag(policy.AGREE_DP_LEAN_FLAG))

    def test_only_explicit_one_enables(self):
        for value, expected in (
            ("1", True),
            ("0", False),
            ("true", False),
            ("yes", False),
            ("", False),
            ("2", False),
        ):
            with mock.patch.dict(
                os.environ, {policy.CPU_ALLGATHER_GLOO_FLAG: value}, clear=False
            ):
                self.assertEqual(
                    policy.value_flag(policy.CPU_ALLGATHER_GLOO_FLAG),
                    expected,
                    value,
                )


class EnablementMatrixTest(unittest.TestCase):
    def _env(self, allgather=None, lean=None):
        env = {}
        if allgather is not None:
            env[policy.CPU_ALLGATHER_GLOO_FLAG] = allgather
        if lean is not None:
            env[policy.AGREE_DP_LEAN_FLAG] = lean
        return mock.patch.dict(os.environ, env, clear=True)

    def test_twins_enabled_by_either_flag(self):
        for allgather, lean, expected in (
            (None, None, False),
            ("0", "0", False),
            ("1", None, True),
            (None, "1", True),
            ("1", "1", True),
            ("1", "0", True),
            ("0", "1", True),
        ):
            with self._env(allgather, lean):
                self.assertEqual(
                    policy.cpu_gloo_twins_enabled(), expected, (allgather, lean)
                )

    def test_lean_requires_its_own_flag(self):
        # C1 alone must NOT reroute _agree_dp (each flag owns its A/B).
        with self._env(allgather="1"):
            self.assertTrue(policy.cpu_gloo_twins_enabled())
            self.assertFalse(policy.agree_dp_lean_enabled())
        with self._env(lean="1"):
            self.assertTrue(policy.cpu_gloo_twins_enabled())
            self.assertTrue(policy.agree_dp_lean_enabled())
        with self._env():
            self.assertFalse(policy.agree_dp_lean_enabled())


class RoutingPredicateTest(unittest.TestCase):
    TWIN = object()

    def test_cpu_noninplace_with_twin_routes_to_gloo(self):
        self.assertTrue(
            policy.cpu_allgather_via_gloo(
                self.TWIN, recv_is_cuda=False, send_is_cuda=False, inplace=False
            )
        )

    def test_no_twin_keeps_staged_path(self):
        self.assertFalse(
            policy.cpu_allgather_via_gloo(
                None, recv_is_cuda=False, send_is_cuda=False, inplace=False
            )
        )

    def test_inplace_keeps_staged_path(self):
        # Aliased send slice inside recv is unverified on gloo.
        self.assertFalse(
            policy.cpu_allgather_via_gloo(
                self.TWIN, recv_is_cuda=False, send_is_cuda=False, inplace=True
            )
        )

    def test_any_cuda_side_keeps_nccl(self):
        for recv_cuda, send_cuda in ((True, False), (False, True), (True, True)):
            self.assertFalse(
                policy.cpu_allgather_via_gloo(
                    self.TWIN,
                    recv_is_cuda=recv_cuda,
                    send_is_cuda=send_cuda,
                    inplace=False,
                ),
                (recv_cuda, send_cuda),
            )


if __name__ == "__main__":
    unittest.main(verbosity=0)
    print("OK")
