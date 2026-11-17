"""Run the actual dispatch classes with mocked backends, including on CPU hosts."""

import ast
import os
import unittest
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
from unittest.mock import Mock, patch

import torch


class FakeLinear(torch.nn.Module):
    def __init__(self, *args):
        super().__init__()


def load_classes(path, names, namespace):
    # Isolate dispatch from the native library/GPU-dependent package initializer.
    tree = ast.parse(path.read_text())
    tree.body = [
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in names
    ]
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace


class F16HotPathDispatchTest(unittest.TestCase):
    def setUp(self):
        self.native, self.legacy, self.initialize = Mock(), Mock(), Mock()
        self.ns = load_classes(
            Path(__file__).resolve().parents[1] / "f16_linear.py",
            {"RocmF16LinearBase", "RocmF16LinearNoSwizzle", "RocmF16LinearWithSwizzle"},
            dict(
                torch=torch,
                os=os,
                Optional=Optional,
                LinearBase=FakeLinear,
                lru_cache=lru_cache,
                HWKernelConfig=object,
                hipb_mm=self.legacy,
                hipb_create_extension=self.initialize,
                rtp_llm_ops=SimpleNamespace(rocm_cached_f16_gemm=self.native),
            ),
        )
        self.x, self.w = torch.ones(2, 4), torch.ones(4, 8)

    def layer(self, swizzle=False):
        return self.ns[
            "RocmF16LinearWithSwizzle" if swizzle else "RocmF16LinearNoSwizzle"
        ](self.w)

    def test_hit_avoids_legacy_wrapper_but_prepares_graph_fallback_once(self):
        for swizzle in (False, True):
            layer = self.layer(swizzle)
            result = layer(self.x)
            self.assertIs(result, self.native.return_value)
            self.native.assert_called_with(self.x, self.w, None, swizzle)
        self.legacy.assert_not_called()
        self.initialize.assert_called_once()

    def test_unsupported_and_capture_fall_back(self):
        self.native.return_value = None
        for swizzle in (False, True):
            layer = self.layer(swizzle)
            self.assertIs(layer(self.x), self.legacy.return_value)
            self.assertEqual(self.legacy.call_args.kwargs["bpreshuffle"], swizzle)
            self.assertEqual(self.legacy.call_args.kwargs["solution_index"], -1)
        self.initialize.assert_called_once()

    def test_native_execution_error_is_not_retried(self):
        self.native.side_effect = RuntimeError("HIP execution failed")
        with self.assertRaisesRegex(RuntimeError, "HIP execution failed"):
            self.layer()(self.x)
        self.legacy.assert_not_called()

    def test_opt_out_is_bound_once_at_construction(self):
        with patch.dict(os.environ, {"RTP_ROCM_F16_GEMM_PLAN_CACHE": "0"}):
            layer = self.layer()
        layer(self.x)
        self.native.assert_not_called()
        self.legacy.assert_called_once()

    def test_older_native_library_uses_legacy(self):
        self.ns["rtp_llm_ops"] = SimpleNamespace()
        self.layer()(self.x)
        self.native.assert_not_called()
        self.legacy.assert_called_once()

    def test_explicit_online_tuning_keeps_aiter_path(self):
        with patch.dict(os.environ, {"HIP_ONLINE_TUNING": "1"}):
            self.layer()(self.x)
        self.native.assert_not_called()
        self.legacy.assert_called_once()


if __name__ == "__main__":
    unittest.main()
