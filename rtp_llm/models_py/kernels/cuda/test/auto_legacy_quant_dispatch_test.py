"""CPU dispatch contracts; real CUDA numerical coverage lives in chain tests."""

import ast
import math
import importlib.util
import sys
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch


class AutoLegacyQuantDispatchTest(unittest.TestCase):
    def setUp(self):
        folder = Path(__file__).resolve().parents[1] / "fp8_kernel"
        self.calls = {name: Mock(name=name) for name in ("v1", "v2", "v3")}
        self.calls["v3"].return_value = ("v3_q", "v3_s")
        # Load the complete module so missing global imports cannot be hidden by
        # the dispatch fixture. Only native/optional dependencies are stubbed.
        spec = importlib.util.spec_from_file_location(
            "fp8_dispatch_under_test", folder / "fp8_kernel.py"
        )
        module = importlib.util.module_from_spec(spec)
        with patch.dict(
            sys.modules,
            {
                "rtp_llm.models_py.kernels.cuda.fp8_kernel.get_best_config": SimpleNamespace(
                    get_cutlass_groupgemm_best_config=Mock()
                ),
                "rtp_llm.models_py.utils.arch": SimpleNamespace(is_cuda=lambda: False),
                "rtp_llm.models_py.utils.math": SimpleNamespace(align=Mock()),
            },
        ):
            spec.loader.exec_module(module)
        self.ns = vars(module)
        self.ns.update(
            create_per_token_group_quant_fp8_output_scale=Mock(return_value="s"),
            per_token_group_quant_fp8=self.calls["v1"],
            per_token_group_quant_fp8_v2=self.calls["v2"],
            sgl_per_token_group_quant_fp8_v3=self.calls["v3"],
        )

    def dispatch(self, m=32, k=8192, policy="auto", cc=(10, 3), attrs=None, **opts):
        for call in self.calls.values():
            call.reset_mock()
        x = SimpleNamespace(
            is_cuda=True,
            dtype=torch.bfloat16,
            ndim=2,
            dim=lambda: 2,
            shape=(m, k),
            device=torch.device("cuda:0"),
            numel=lambda: m * k,
            is_contiguous=lambda: True,
            data_ptr=lambda: 16,
        )
        x.__dict__.update(attrs or {})
        kwargs = dict(
            group_size=32,
            eps=torch.finfo(torch.float32).tiny,
            column_major_scales=True,
            scale_tma_aligned=True,
            scale_ue8m0=True,
        )
        kwargs.update(opts)
        with patch.dict(os.environ, DSV4_FP8_QUANT_KERNEL=policy), patch.object(
            torch.cuda, "get_device_capability", return_value=cc
        ), patch.object(torch, "empty", return_value="q"):
            result = self.ns["sgl_per_token_group_quant_fp8"](x, **kwargs)
        return [name for name, call in self.calls.items() if call.called], result

    def test_auto_uses_v3_on_both_sides_of_old_threshold(self):
        for k in (2304, 4096, 5120, 8192):
            threshold = (4 * 1024 * 1024 + k - 1) // k
            for m in (1, threshold - 1, threshold, 3072):
                with self.subTest(m=m, k=k):
                    calls, result = self.dispatch(m=m, k=k)
                    self.assertEqual(calls, ["v3"])
                    self.assertEqual(result, ("v3_q", "v3_s"))

    def test_auto_v3_does_not_depend_on_device_architecture(self):
        for cc in ((8, 0), (8, 9), (9, 0), (10, 0), (10, 3), (12, 0)):
            with self.subTest(cc=cc):
                self.assertEqual(self.dispatch(m=65536, cc=cc)[0], ["v3"])

    def test_auto_fallback_keeps_legacy_at_large_m(self):
        for overrides in (
            dict(k=96),
            dict(attrs={"dtype": torch.float16}),
            dict(attrs={"dtype": torch.float32}),
            dict(attrs={"data_ptr": lambda: 2}),
            dict(eps=1e-3),
        ):
            with self.subTest(overrides=overrides):
                calls, _ = self.dispatch(m=65536, **overrides)
                self.assertEqual(calls, ["v1"])

    def test_explicit_implementations_and_empty_input(self):
        for policy, expected in (("legacy", "v1"), ("v2", "v2")):
            for m in (1, 3072):
                self.assertEqual(self.dispatch(m=m, policy=policy)[0], [expected])
        self.assertEqual(self.dispatch(m=0)[0], [])
        with self.assertRaises(ValueError):
            self.dispatch(policy="bad")

    def test_fused_rounding_survives_policy_overrides(self):
        for policy in ("auto", "legacy", "v2"):
            self.assertEqual(
                self.dispatch(
                    policy=policy,
                    group_size=128,
                    fuse_silu_and_mul=True,
                    preserve_silu_mul_rounding=True,
                )[0],
                ["v2"],
            )
            self.assertTrue(
                self.calls["v2"].call_args.kwargs["preserve_silu_mul_rounding"]
            )
        self.dispatch(policy="v2")
        self.assertFalse(
            self.calls["v2"].call_args.kwargs["preserve_silu_mul_rounding"]
        )

    def test_legacy_override_never_uses_unfused_geometry_for_fused_input(self):
        self.assertEqual(
            self.dispatch(policy="legacy", fuse_silu_and_mul=True)[0], ["v2"]
        )
        self.assertEqual(self.dispatch(policy="legacy", masked_m=object())[0], ["v2"])

    def test_epsilon_specialization_boundary(self):
        for eps in (0.0, torch.finfo(torch.float32).tiny, 1e-10):
            self.assertEqual(self.dispatch(eps=eps)[0], ["v3"])
        for eps in (math.nextafter(1e-10, math.inf), -1e-10, math.inf, math.nan):
            self.assertEqual(self.dispatch(eps=eps)[0], ["v1"])

    def test_other_quantization_apis_keep_existing_dispatch(self):
        for opts in (
            dict(fuse_silu_and_mul=True),
            dict(masked_m=object()),
            dict(group_size=128),
            dict(scale_ue8m0=False),
        ):
            with self.subTest(opts=opts):
                self.assertEqual(self.dispatch(m=3072, **opts)[0], ["v2"])
        self.assertEqual(
            self.dispatch(m=3072, attrs={"ndim": 3, "shape": (2, 1536, 8192)})[0],
            ["v2"],
        )


class V41ProducerPolicyTest(unittest.TestCase):
    def definition(self, filename, name, **extra):
        root = Path(__file__).resolve().parents[3] / "modules/dsv41"
        path = root / filename
        tree = ast.parse(path.read_text())
        tree.body = [n for n in tree.body if getattr(n, "name", None) == name]
        self.assertEqual(len(tree.body), 1)
        ns = dict(torch=torch, os=os, math=math, **extra)
        exec(compile(tree, str(path), "exec"), ns)
        return ns[name]

    def test_norm_fusion_auto_always_selects_legacy_math(self):
        launch = Mock()

        # Triton's kernel[grid](...) selection is CPU-side dispatch only here.
        class Kernel:
            def __getitem__(self, grid):
                return launch

        norm = self.definition(
            "fp8/_v41_norm_quant.py",
            "rmsnorm_group32_quant",
            is_supported=lambda *args: True,
            triton=SimpleNamespace(cdiv=lambda a, b: (a + b - 1) // b),
            _v41_norm_quant_kernel=Kernel(),
        )
        for m in (819, 820):
            x = torch.empty(m, 5120, dtype=torch.bfloat16)
            for mode in ("auto", "legacy", "v2"):
                norm(x, None, quant_kernel=mode)
                self.assertEqual(launch.call_args.args[-1], mode != "v2")


if __name__ == "__main__":
    unittest.main()
