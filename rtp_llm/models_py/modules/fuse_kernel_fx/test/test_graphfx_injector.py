from __future__ import annotations

import os
import unittest
from unittest import mock


class _FakePyModel:
    def __init__(self):
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return x


class GraphFXInjectorTest(unittest.TestCase):
    def setUp(self):
        self._old_env = {
            key: os.environ.get(key)
            for key in (
                "QWEN35_GRAPHFX_FUSION",
                "QWEN35_FUSION_REGISTRY_DEBUG",
            )
        }
        os.environ["QWEN35_GRAPHFX_FUSION"] = "1"

    def tearDown(self):
        for key, value in self._old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_install_compiles_model_forward(self):
        from rtp_llm.models_py.modules.fuse_kernel_fx.graphfx_injector import (
            maybe_install_qwen35_graphfx_fusions,
        )

        calls = []

        def fake_compile(fn, **kwargs):
            calls.append(kwargs)

            def compiled(*args, **inner_kwargs):
                return fn(*args, **inner_kwargs)

            return compiled

        py_model = _FakePyModel()
        with mock.patch(
            "rtp_llm.models_py.modules.fuse_kernel_fx.graphfx_injector.compile_with_qwen35_fusions",
            side_effect=fake_compile,
        ):
            self.assertTrue(maybe_install_qwen35_graphfx_fusions(py_model))

        self.assertEqual(len(calls), 1)
        self.assertTrue(all(item.get("dynamic") is True for item in calls))
        self.assertTrue(all(item.get("fullgraph") is False for item in calls))
        self.assertTrue(getattr(py_model.forward, "_qwen35_graphfx_compiled", False))

    def test_install_is_idempotent(self):
        from rtp_llm.models_py.modules.fuse_kernel_fx.graphfx_injector import (
            maybe_install_qwen35_graphfx_fusions,
        )

        calls = []

        def fake_compile(fn, **kwargs):
            calls.append(kwargs)
            return fn

        py_model = _FakePyModel()
        with mock.patch(
            "rtp_llm.models_py.modules.fuse_kernel_fx.graphfx_injector.compile_with_qwen35_fusions",
            side_effect=fake_compile,
        ):
            self.assertTrue(maybe_install_qwen35_graphfx_fusions(py_model))
            self.assertFalse(maybe_install_qwen35_graphfx_fusions(py_model))

        self.assertEqual(len(calls), 1)

    def test_disabled_env_is_noop(self):
        from rtp_llm.models_py.modules.fuse_kernel_fx.graphfx_injector import (
            maybe_install_qwen35_graphfx_fusions,
        )

        os.environ["QWEN35_GRAPHFX_FUSION"] = "0"
        py_model = _FakePyModel()
        self.assertFalse(maybe_install_qwen35_graphfx_fusions(py_model))
        self.assertFalse(getattr(py_model.forward, "_qwen35_graphfx_compiled", False))
        self.assertFalse(getattr(py_model, "_qwen35_graphfx_forward_compiled", False))

    def test_install_handles_missing_forward(self):
        from rtp_llm.models_py.modules.fuse_kernel_fx.graphfx_injector import (
            maybe_install_qwen35_graphfx_fusions,
        )

        class _NoForward:
            pass

        self.assertFalse(maybe_install_qwen35_graphfx_fusions(_NoForward()))
        self.assertFalse(maybe_install_qwen35_graphfx_fusions(None))


if __name__ == "__main__":
    unittest.main()
