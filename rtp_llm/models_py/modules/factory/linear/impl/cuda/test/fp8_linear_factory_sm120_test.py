"""Public LinearFactory regression for SM120 FP8/FP4 strategy registration."""

import builtins
import importlib
import sys
import types
import unittest
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import torch

import rtp_llm.models_py.modules.factory.linear.impl.cuda as impl_cuda
from rtp_llm.models_py.modules.factory.linear.factory import LinearFactory
from rtp_llm.models_py.modules.factory.linear.impl.cuda import (
    fp8_gemm_linear as gemm_module,
)

IMPL_CUDA_PREFIX = impl_cuda.__name__ + "."
FP4_MODULE_NAME = impl_cuda.__name__ + ".fp4_linear"
ARCH = "rtp_llm.models_py.utils.arch"


def _drop_impl_cuda_strategies():
    LinearFactory._strategies[:] = [
        cls
        for cls in LinearFactory._strategies
        if not getattr(cls, "__module__", "").startswith(IMPL_CUDA_PREFIX)
    ]


def _register_production_strategy_set(capability=(12, 0), fp4_module="missing"):
    _drop_impl_cuda_strategies()
    fake_fp4 = None
    if fp4_module != "missing":
        fake_fp4 = types.ModuleType(FP4_MODULE_NAME)

        class _FakeFp4GEMMLinear:
            pass

        _FakeFp4GEMMLinear.__module__ = FP4_MODULE_NAME
        fake_fp4.CudaFp4GEMMLinear = _FakeFp4GEMMLinear
        fake_fp4.fp4_backend_available = lambda device=None: fp4_module == "ready"
    with ExitStack() as stack:
        stack.enter_context(patch(f"{ARCH}.is_cuda", return_value=True))
        stack.enter_context(patch(f"{ARCH}.get_sm", return_value=capability))
        stack.enter_context(
            patch(f"{ARCH}.is_sm120", return_value=(capability == (12, 0)))
        )
        stack.enter_context(patch.dict(sys.modules, {FP4_MODULE_NAME: fake_fp4}))
        importlib.reload(impl_cuda)
    return fake_fp4


def _fp8_per_block_weight(n=256, k=128):
    quant_config = MagicMock()
    quant_config.get_method.return_value = "FP8_PER_BLOCK"
    weight = torch.zeros((n, k), dtype=torch.float8_e4m3fn)
    weight_scales = torch.ones((n // 128, k // 128), dtype=torch.float32)
    return quant_config, weight, weight_scales


class Sm120LinearFactoryRegistrationTest(unittest.TestCase):
    def setUp(self):
        self._saved = list(LinearFactory._strategies)

    def tearDown(self):
        LinearFactory._strategies[:] = self._saved

    def test_modelopt_hybrid_fp8_stays_with_the_shared_strategy(self):
        quant_config, weight, weight_scales = _fp8_per_block_weight()
        quant_config.get_method.return_value = "modelopt_fp4"
        quant_config.hybrid_attn_quant_method = "FP8_PER_BLOCK"
        self.assertTrue(
            gemm_module.CudaFp8GEMMLinear.can_handle(
                quant_config, weight, weight_scales
            )
        )
        for keyword in ("weight_scale_2", "input_scale"):
            with self.subTest(keyword=keyword):
                self.assertFalse(
                    gemm_module.CudaFp8GEMMLinear.can_handle(
                        quant_config,
                        weight,
                        weight_scales,
                        **{keyword: torch.ones(1)},
                    )
                )

    def test_legacy_cutlass_fp4_availability_contract(self):
        from rtp_llm.models_py.kernels.cuda import fp4_kernel
        from rtp_llm.models_py.kernels.cuda.fp4_kernel import (
            fp4_kernel as fp4_kernel_mod,
        )
        from rtp_llm.models_py.utils import arch

        with (
            patch.object(arch, "is_cuda", return_value=True),
            patch.object(fp4_kernel_mod, "is_cuda", return_value=True),
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(fp4_kernel_mod, "_NVFP4_READY", True),
            patch.object(
                fp4_kernel_mod, "cutlass_scaled_fp4_mm", lambda *args, **kwargs: None
            ),
        ):
            with patch.object(arch, "get_sm", return_value=(10, 0)):
                self.assertTrue(arch.is_sm10x())
                self.assertTrue(fp4_kernel_mod.is_legacy_cutlass_fp4_available())
            with patch.object(arch, "get_sm", return_value=(12, 0)):
                self.assertFalse(arch.is_sm10x())
                self.assertFalse(fp4_kernel_mod.is_legacy_cutlass_fp4_available())
                self.assertFalse(fp4_kernel.is_legacy_cutlass_fp4_available())
        self.assertIsInstance(fp4_kernel.is_legacy_cutlass_fp4_available(), bool)

    def test_fp4_registration_preserves_main_major_ge_10(self):
        for capability in [
            (9, 0),
            (10, 0),
            (10, 3),
            (11, 0),
            (12, 0),
            (12, 1),
            (13, 0),
        ]:
            with self.subTest(capability=capability, fp4_module="ready"):
                _register_production_strategy_set(capability, fp4_module="ready")
                fp4_registered = any(
                    getattr(cls, "__module__", "") == FP4_MODULE_NAME
                    for cls in LinearFactory._strategies
                )
                self.assertEqual(fp4_registered, capability[0] >= 10)
                self.assertIn(gemm_module.CudaFp8GEMMLinear, LinearFactory._strategies)
            with self.subTest(capability=capability, fp4_module="missing"):
                _register_production_strategy_set(capability, fp4_module="missing")
                fp4_registered = any(
                    getattr(cls, "__module__", "") == FP4_MODULE_NAME
                    for cls in LinearFactory._strategies
                )
                self.assertFalse(fp4_registered)
            self.assertIn(gemm_module.CudaFp8GEMMLinear, LinearFactory._strategies)

    def test_fp4_binary_oserror_still_registers_mandatory_fp8_strategies(self):
        real_import = builtins.__import__

        def fail_fp4_load(name, globals=None, locals=None, fromlist=(), level=0):
            if (name == FP4_MODULE_NAME) or (level == 1 and name == "fp4_linear"):
                raise OSError("broken fp4 shared library")
            return real_import(name, globals, locals, fromlist, level)

        for capability in ((10, 0), (10, 3), (12, 0)):
            with self.subTest(capability=capability):
                _drop_impl_cuda_strategies()
                with ExitStack() as stack:
                    stack.enter_context(patch(f"{ARCH}.is_cuda", return_value=True))
                    stack.enter_context(
                        patch(f"{ARCH}.get_sm", return_value=capability)
                    )
                    stack.enter_context(
                        patch(f"{ARCH}.is_sm120", return_value=(capability == (12, 0)))
                    )
                    stack.enter_context(
                        patch.dict(sys.modules, {FP4_MODULE_NAME: None})
                    )
                    sys.modules.pop(FP4_MODULE_NAME, None)
                    stack.enter_context(
                        patch.object(builtins, "__import__", side_effect=fail_fp4_load)
                    )
                    importlib.reload(impl_cuda)
                self.assertFalse(
                    any(
                        getattr(cls, "__module__", "") == FP4_MODULE_NAME
                        for cls in LinearFactory._strategies
                    )
                )
                self.assertIn(gemm_module.CudaFp8GEMMLinear, LinearFactory._strategies)

    def test_create_linear_no_candidate_preserves_main_errors(self):
        LinearFactory._strategies[:] = []
        weight = torch.zeros((4, 4), dtype=torch.bfloat16)
        quant_config = MagicMock()
        quant_config.get_method.return_value = "W8A8_INT8_PER_CHANNEL_COMPRESSED"
        with self.assertRaises(ValueError) as caught:
            LinearFactory.create_linear(
                weight=weight,
                bias=None,
                quant_config=quant_config,
                weight_scales=None,
            )
        self.assertEqual(
            str(caught.exception),
            "W8A8_INT8_PER_CHANNEL_COMPRESSED weights were loaded, but "
            "no registered Linear compute backend can consume them; "
            "install or register a backend with W8A8 INT8 per-channel "
            "execution support",
        )
        quant_config.get_method.return_value = "FP8_PER_BLOCK"
        with self.assertRaises(ValueError) as caught:
            LinearFactory.create_linear(
                weight=weight,
                bias=None,
                quant_config=quant_config,
                weight_scales=None,
            )
        self.assertEqual(
            str(caught.exception),
            f"No suitable Linear strategy found for:"
            f"weight.dtype={weight.dtype}, "
            f"has_scales=False, "
            f"quant_config={quant_config}",
        )


if __name__ == "__main__":
    unittest.main()
