"""Public LinearFactory regression for the SM120 blockwise FP8 strategy.

At this commit ``CudaFp8VllmBlockwiseLinear`` must be reachable through the
public factory: ``impl/cuda/__init__`` registers it behind the exact-SM120
guard so ``LinearFactory.create_linear`` can select it for an FP8_PER_BLOCK
weight. The tests exercise the real package registration and the real
candidate competition together: the complete CUDA strategy set is
registered the same way ``impl/cuda/__init__`` does on an SM120 host, and
every module that consults the SM120 blockwise backend probe is patched
consistently so the legacy FP8 strategies defer exactly as they do at
runtime. Only hardware-bound leaves (device probes, the CUTLASS op
handle, the CUDA-device layout check) are mocked.

``LinearFactory.register`` is a plain append, so each test first removes
every class the reload is about to append (all classes defined under
``impl.cuda``), then runs the real package init once. That produces the
production registry regardless of whether the module was first imported
on CPU (nothing CUDA registered yet) or on a real SM120 host (the full
set already registered), and ``tearDown`` always restores the original
global state.
"""

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
    fp8_deepgemm_linear as deepgemm_module,
)
from rtp_llm.models_py.modules.factory.linear.impl.cuda import (
    fp8_gemm_linear as gemm_module,
)
from rtp_llm.models_py.modules.factory.linear.impl.cuda import (
    fp8_vllm_blockwise_sm120_linear as strategy_module,
)

STRATEGY = strategy_module.CudaFp8VllmBlockwiseLinear
IMPL_CUDA_PREFIX = impl_cuda.__name__ + "."
FP4_MODULE_NAME = impl_cuda.__name__ + ".fp4_linear"
ARCH = "rtp_llm.models_py.utils.arch"


def _drop_impl_cuda_strategies():
    """Remove every class the impl/cuda package init can register.

    ``register`` appends unconditionally, so leaving any impl.cuda class in
    the registry would double it after the reload below — two identical
    candidates, exactly the bug a real-SM120-host run would hit. Removing
    by defining module covers the whole set the init appends (F16,
    per-tensor FP8, FP8 GEMM, optional FP4, SM120 blockwise) whether or
    not each one happened to register on this host.
    """
    LinearFactory._strategies[:] = [
        cls
        for cls in LinearFactory._strategies
        if not getattr(cls, "__module__", "").startswith(IMPL_CUDA_PREFIX)
    ]


def _register_production_strategy_set(
    capability=(12, 0), fp4_module="ready", blockwise_available=True
):
    """Rebuild the production CUDA registry exactly once.

    Drops the classes the init appends, then runs the real impl-package
    registration under the requested device capability, so each strategy
    lands exactly once on both CPU-initialized and SM120-initialized
    processes. ``fp4_module`` supplies the main FP4 provider class, without
    any later SM120 availability helper, or makes the provider import fail.
    ``blockwise_available`` sets the ``sm120_blockwise_fp8_available``
    probe the init consults before registering the SM120 blockwise
    strategy, so the registration result does not depend on the build of
    the host running the test.
    """
    _drop_impl_cuda_strategies()
    fake_fp4 = None
    if fp4_module != "missing":
        fake_fp4 = types.ModuleType(FP4_MODULE_NAME)

        class _FakeFp4GEMMLinear:
            pass

        _FakeFp4GEMMLinear.__module__ = FP4_MODULE_NAME
        fake_fp4.CudaFp4GEMMLinear = _FakeFp4GEMMLinear
    with ExitStack() as stack:
        stack.enter_context(patch(f"{ARCH}.is_cuda", return_value=True))
        stack.enter_context(patch(f"{ARCH}.get_sm", return_value=capability))
        stack.enter_context(
            patch(f"{ARCH}.is_sm120", return_value=(capability == (12, 0)))
        )
        if fake_fp4 is not None:
            stack.enter_context(patch.dict(sys.modules, {FP4_MODULE_NAME: fake_fp4}))
        else:
            # A None sys.modules entry makes ``from .fp4_linear import ...``
            # raise ImportError, the missing-provider state.
            stack.enter_context(patch.dict(sys.modules, {FP4_MODULE_NAME: None}))
        # The init now gates blockwise registration on the backend probe, so
        # the bound copy the from-import resolves must be patched here or a
        # host without the SM120 build would skip the registration under test.
        stack.enter_context(
            patch.object(
                strategy_module,
                "sm120_blockwise_fp8_available",
                return_value=blockwise_available,
            )
        )
        importlib.reload(impl_cuda)
    return fake_fp4


def _fp8_per_block_weight(n=256, k=128):
    quant_config = MagicMock()
    quant_config.get_method.return_value = "FP8_PER_BLOCK"
    weight = torch.zeros((n, k), dtype=torch.float8_e4m3fn)
    weight_scales = torch.ones((n // 128, k // 128), dtype=torch.float32)
    return quant_config, weight, weight_scales


def _sm120_runtime_patches():
    """Consistent SM120 environment probes across every consumer module.

    ``sm120_blockwise_fp8_available`` is imported by name into the
    blockwise, GEMM, and DeepGEMM modules; each bound copy must agree or
    the legacy strategies and the SM120 strategy see different runtimes.
    """
    return (
        patch.object(strategy_module, "is_sm120", return_value=True),
        patch.object(
            strategy_module, "sm120_blockwise_fp8_available", return_value=True
        ),
        patch.object(gemm_module, "sm120_blockwise_fp8_available", return_value=True),
        patch.object(
            deepgemm_module, "sm120_blockwise_fp8_available", return_value=True
        ),
        patch.object(
            strategy_module,
            "sm120_blockwise_fp8_op",
            return_value=MagicMock(),
        ),
        patch.object(
            STRATEGY,
            "_restore_blockwise_weight_layout",
            staticmethod(
                lambda weight, scales: (
                    weight,
                    scales,
                    *weight.shape[::-1],
                    *scales.shape[::-1],
                )
            ),
        ),
    )


class Sm120BlockwiseFactoryRegistrationTest(unittest.TestCase):
    def setUp(self):
        self._saved = list(LinearFactory._strategies)

    def tearDown(self):
        LinearFactory._strategies[:] = self._saved

    def test_package_init_registers_blockwise_on_sm120(self):
        # Re-register under the same conditions the __init__ guards on:
        # is_cuda + is_sm120, with the existing FP4 provider available.
        _register_production_strategy_set()
        self.assertIn(STRATEGY, LinearFactory._strategies)
        self.assertEqual(LinearFactory._strategies.count(STRATEGY), 1)
        # The legacy FP8 competitors were registered through the same path.
        self.assertIn(gemm_module.CudaFp8GEMMLinear, LinearFactory._strategies)

    def test_create_linear_selects_blockwise_for_fp8_per_block(self):
        # Full production registry present — legacy FP8 candidates compete —
        # with consistent backend probes, so the SM120 strategy is the
        # single match and the public factory constructs it.
        _register_production_strategy_set()
        self.assertEqual(LinearFactory._strategies.count(STRATEGY), 1)
        quant_config, weight, weight_scales = _fp8_per_block_weight()
        with ExitStack() as stack:
            for runtime_patch in _sm120_runtime_patches():
                stack.enter_context(runtime_patch)
            linear = LinearFactory.create_linear(
                weight=weight,
                bias=None,
                quant_config=quant_config,
                weight_scales=weight_scales,
            )
        self.assertIsInstance(linear, STRATEGY)
        self.assertEqual(linear.weight.shape, weight.shape)

    def test_create_linear_rejects_non_fp8_per_block(self):
        # The negative can_handle case needs no registry setup: the strategy
        # defers on the quant method before any device or backend probe runs.
        quant_config, weight, weight_scales = _fp8_per_block_weight()
        quant_config.get_method.return_value = "FP8"
        self.assertFalse(STRATEGY.can_handle(quant_config, weight, weight_scales))

    def test_fp4_registration_preserves_main_major_ge_10(self):
        # FP4's provider import is optional on main for major >= 10: a
        # missing provider only warns and leaves FP4 unregistered.
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
                if capability == (12, 0):
                    self.assertIn(STRATEGY, LinearFactory._strategies)
            with self.subTest(capability=capability, fp4_module="missing"):
                _register_production_strategy_set(capability, fp4_module="missing")
                self.assertFalse(
                    any(
                        getattr(cls, "__module__", "") == FP4_MODULE_NAME
                        for cls in LinearFactory._strategies
                    )
                )
                self.assertIn(gemm_module.CudaFp8GEMMLinear, LinearFactory._strategies)
                if capability == (12, 0):
                    self.assertIn(STRATEGY, LinearFactory._strategies)

    def test_create_linear_no_candidate_preserves_main_errors(self):
        # No registered strategy matches: the existing no-candidate errors
        # are preserved verbatim, with no added diagnostics suffix.
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
