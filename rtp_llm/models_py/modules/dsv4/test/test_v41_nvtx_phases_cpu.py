"""CPU tests for the R26-TRACE NVTX phase helper (_nvtx_phases.py).

Hermetic namespace-stub loading as in test_v41_deferred_checks_cpu. The
helper is pure python + torch.cuda.nvtx (push/pop are host-side); CUDA is
not required — push/pop are mocked to verify call pairing when the flag is
on and zero CUDA interaction when off. The flag is cached at import time,
so each case loads a fresh module instance under a patched environment.
"""

import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import torch  # noqa: F401  (helper imports it)

ROOT = Path(__file__).resolve().parents[5]
for suffix in ["", ".models_py", ".models_py.modules", ".models_py.modules.dsv4"]:
    name = "rtp_llm" + suffix
    module = types.ModuleType(name)
    module.__path__ = [str(ROOT.joinpath(*name.split(".")))]
    sys.modules.setdefault(name, module)

FLAG = "DSV41_NVTX_PHASES"
HELPER = ROOT / "rtp_llm/models_py/modules/dsv4/_nvtx_phases.py"


def _load(flag_value):
    env = {FLAG: flag_value} if flag_value is not None else {}
    with mock.patch.dict(os.environ, env, clear=True):
        spec = importlib.util.spec_from_file_location("dsv4_nvtx_phases_test", HELPER)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


class FlagTest(unittest.TestCase):
    def test_absent_is_off(self):
        self.assertFalse(_load(None).nvtx_phases_enabled())

    def test_value_based(self):
        for value, expected in (
            ("1", True),
            ("0", False),
            ("true", False),
            ("", False),
            ("2", False),
        ):
            self.assertEqual(_load(value).nvtx_phases_enabled(), expected, value)


class OffPathTest(unittest.TestCase):
    def test_never_touches_cuda(self):
        module = _load("0")
        with mock.patch.object(torch.cuda, "nvtx", create=True) as fake_nvtx:
            with module.nvtx_phase("v41.test"):
                pass
        self.assertFalse(fake_nvtx.range_push.called)
        self.assertFalse(fake_nvtx.range_pop.called)

    def test_runs_body_and_propagates_exceptions(self):
        module = _load("0")
        hit = []
        with module.nvtx_phase("v41.test"):
            hit.append(1)
        self.assertEqual(hit, [1])
        with self.assertRaises(RuntimeError):
            with module.nvtx_phase("v41.test"):
                raise RuntimeError("boom")


class OnPairingTest(unittest.TestCase):
    def test_push_pop_pairing_including_exceptions(self):
        module = _load("1")
        calls = []
        fake = types.SimpleNamespace(
            range_push=lambda name: calls.append(("push", name)),
            range_pop=lambda: calls.append(("pop", None)),
        )
        with mock.patch.object(torch.cuda, "nvtx", fake, create=True):
            with module.nvtx_phase("v41.moe.router"):
                pass
            with self.assertRaises(ValueError):
                with module.nvtx_phase("v41.attn"):
                    raise ValueError("x")
        self.assertEqual(
            calls,
            [
                ("push", "v41.moe.router"),
                ("pop", None),
                ("push", "v41.attn"),
                ("pop", None),
            ],
        )

    def test_free_push_pop(self):
        module = _load("1")
        calls = []
        fake = types.SimpleNamespace(
            range_push=lambda name: calls.append(("push", name)),
            range_pop=lambda: calls.append(("pop", None)),
        )
        with mock.patch.object(torch.cuda, "nvtx", fake, create=True):
            module.nvtx_push("v41.attn")
            module.nvtx_pop()
            module.nvtx_push("v41.moe")
            module.nvtx_pop()
        self.assertEqual(
            calls,
            [("push", "v41.attn"), ("pop", None), ("push", "v41.moe"), ("pop", None)],
        )

    def test_free_push_pop_off_is_noop(self):
        module = _load("0")
        with mock.patch.object(torch.cuda, "nvtx", create=True) as fake_nvtx:
            module.nvtx_push("v41.attn")
            module.nvtx_pop()
        self.assertFalse(fake_nvtx.range_push.called)
        self.assertFalse(fake_nvtx.range_pop.called)


if __name__ == "__main__":
    unittest.main(verbosity=0)
    print("OK")
