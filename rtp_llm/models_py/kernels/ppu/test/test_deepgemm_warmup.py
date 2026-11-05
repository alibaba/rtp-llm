"""Contract tests for the PPU DeepGEMM warmup scaffolding.

The warmup only exists to fill DeepGEMM's JIT cache, so it must not depend on
the launch being suppressed: ``deep_gemm.set_compile_mode(1)`` is not honoured
by every wheel build, and reading the weight storage as per-expert token counts
made the masked kernel page-walk outside its operands
(``HGGC_ERROR_ILLEGAL_ADDRESS`` on PPU-ZW810E). These tests need no PPU
hardware: a ``deep_gemm`` stub records the launches the warmup performs and
emulates the write a real launch does.
"""

import importlib.util
import os
import sys
import tempfile
import types
import unittest
from typing import Any

import torch

from rtp_llm.models_py.kernels.ppu import (
    deepgemm_warmup,
    deepgemm_wrapper,
)

NUM_GROUPS = 4
N = 64
K = 128
MAX_M = 8
NUM_SMS = 8

# K is 128, so INT8 weights carry exactly one scale per row.
_SCALE_COLUMNS = 1

_WHEEL_SYMBOLS = (
    "get_compile_mode",
    "set_compile_mode",
    "get_num_sms",
    "set_num_sms",
    "m_grouped_bf16_gemm_nt_contiguous",
    "m_grouped_bf16_gemm_nt_masked",
    "m_grouped_bf16_gemm_nt_nopad",
    "gemm_int8_int8_bf16_nt",
    "m_grouped_gemm_int8_int8_bf16_nt_contiguous",
    "m_grouped_gemm_int8_int8_bf16_nt_masked",
    "m_grouped_gemm_int8_int8_bf16_nt_nopad",
)

_WRAPPER_SYMBOLS = (
    "_contiguous_impl",
    "_masked_impl",
    "_nopad_impl",
    "_int8_dense_impl",
    "_int8_contiguous_impl",
    "_int8_masked_impl",
    "_int8_nopad_impl",
)


class _FakeDeepGemm:
    """Stand-in for the wheel: records launches, emulates their writes."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []
        self.compile_modes: list[int] = []
        self.num_sms_calls: list[int] = []
        self._compile_mode = 0
        self._num_sms = NUM_SMS

    def get_compile_mode(self) -> int:
        return self._compile_mode

    def set_compile_mode(self, mode: int) -> None:
        self._compile_mode = mode
        self.compile_modes.append(mode)

    def get_num_sms(self) -> int:
        return self._num_sms

    def set_num_sms(self, num_sms: int) -> None:
        self._num_sms = num_sms
        self.num_sms_calls.append(num_sms)

    def _launch(self, name: str, args: tuple[Any, ...]) -> None:
        # Every entry point the warmup uses takes the output third. A launch
        # that escapes compile-only mode fills it, which is exactly how a
        # write-through into the weight storage used to happen.
        args[2].fill_(1.0)
        self.calls.append((name, args))

    def m_grouped_bf16_gemm_nt_contiguous(self, a, b, output, m_indices, _reserved):
        self._launch("bf16_contiguous", (a, b, output, m_indices))

    def m_grouped_bf16_gemm_nt_masked(
        self, a, b, output, masked_m, expected_m, _reserved
    ):
        self._launch("bf16_masked", (a, b, output, masked_m, expected_m))

    def m_grouped_bf16_gemm_nt_nopad(self, a, b, output, m_indices, m_rows, _reserved):
        self._launch("bf16_nopad", (a, b, output, m_indices, m_rows))

    def gemm_int8_int8_bf16_nt(self, lhs, rhs, output, _reserved):
        self._launch("int8_dense", (lhs, rhs, output))

    def m_grouped_gemm_int8_int8_bf16_nt_contiguous(
        self, lhs, rhs, output, m_indices, _reserved
    ):
        self._launch("int8_contiguous", (lhs, rhs, output, m_indices))

    def m_grouped_gemm_int8_int8_bf16_nt_masked(
        self, lhs, rhs, output, masked_m, expected_m, _reserved
    ):
        self._launch("int8_masked", (lhs, rhs, output, masked_m, expected_m))

    def m_grouped_gemm_int8_int8_bf16_nt_nopad(
        self, lhs, rhs, output, m_indices, m_rows, _reserved
    ):
        self._launch("int8_nopad", (lhs, rhs, output, m_indices, m_rows))


def _fake_module(fake: _FakeDeepGemm) -> Any:
    module = types.ModuleType("deep_gemm")
    # ``deepgemm_wrapper.has_deep_gemm`` probes ``importlib.util.find_spec``,
    # which reads ``__spec__`` off an already imported module.
    module.__spec__ = importlib.util.spec_from_loader("deep_gemm", loader=None)
    for name in _WHEEL_SYMBOLS:
        setattr(module, name, getattr(fake, name))
    return module


def _reset_wrapper_cache() -> None:
    for name in _WRAPPER_SYMBOLS:
        setattr(deepgemm_wrapper, name, None)
    deepgemm_wrapper._initialized = False
    deepgemm_wrapper._available = None


class DeepGemmWarmupTest(unittest.TestCase):
    def setUp(self):
        self.fake = _FakeDeepGemm()
        self._previous_module = sys.modules.get("deep_gemm")
        self._had_previous_module = "deep_gemm" in sys.modules
        sys.modules["deep_gemm"] = _fake_module(self.fake)
        _reset_wrapper_cache()
        deepgemm_warmup._warmup_cache.clear()
        # Keep the warmup lock file out of the shared wheel cache directory.
        self._previous_cache_dir = os.environ.get("DG_CACHE_DIR")
        os.environ["DG_CACHE_DIR"] = tempfile.mkdtemp(prefix="deepgemm-warmup-test-")

    def tearDown(self):
        if self._had_previous_module:
            sys.modules["deep_gemm"] = self._previous_module
        else:
            sys.modules.pop("deep_gemm", None)
        _reset_wrapper_cache()
        deepgemm_warmup._warmup_cache.clear()
        if self._previous_cache_dir is None:
            os.environ.pop("DG_CACHE_DIR", None)
        else:
            os.environ["DG_CACHE_DIR"] = self._previous_cache_dir

    # -- helpers --------------------------------------------------------
    def _bf16_weight(self, num_groups: int) -> torch.Tensor:
        # 1.0 has bit pattern 0x3F80, so the storage reinterpreted as int32 is a
        # huge positive number -- the garbage the masked warmup used to pass as
        # per-expert token counts.
        return torch.full((num_groups, N, K), 1.0, dtype=torch.bfloat16)

    def _int8_weight(self, num_groups: int) -> tuple[torch.Tensor, torch.Tensor]:
        weight = torch.ones((num_groups, N, K), dtype=torch.int8)
        scale = torch.ones((num_groups, N, _SCALE_COLUMNS), dtype=torch.float32)
        return weight, scale

    def _reset_recorder(self) -> None:
        # subTest keeps one recorder across layouts, so start each warmup clean.
        self.fake.calls.clear()
        self.fake.compile_modes.clear()
        self.fake.num_sms_calls.clear()

    def _assert_scratch(
        self, tensor: torch.Tensor, weight: torch.Tensor, label: str
    ) -> None:
        self.assertNotEqual(
            tensor.untyped_storage().data_ptr(),
            weight.untyped_storage().data_ptr(),
            f"{label} must not alias the weight storage",
        )

    def _check_launch(
        self, name: str, args: tuple[Any, ...], layout: str, weight: torch.Tensor
    ) -> None:
        output = args[2]
        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertEqual(output.ndim, 3 if layout == "masked" else 2)
        m = output.shape[1] if layout == "masked" else output.shape[0]
        self.assertEqual(output.shape[-1], N)

        metadata = args[3]
        self.assertEqual(metadata.dtype, torch.int32)
        if layout == "masked":
            # Per-expert token counts: non-negative, never above the padded M,
            # and describing exactly the m rows the kernel is about to walk.
            self.assertEqual(metadata.numel(), NUM_GROUPS)
            self.assertGreaterEqual(int(metadata.min()), 0)
            self.assertLessEqual(int(metadata.max()), m)
            self.assertEqual(int(metadata.sum()), m)
            self.assertEqual(args[4], m)
        else:
            # Per-row expert ids: every row resolves to a group inside `rhs`.
            self.assertEqual(metadata.numel(), m)
            self.assertGreaterEqual(int(metadata.min()), 0)
            self.assertLess(int(metadata.max()), NUM_GROUPS)
            if layout == "nopad":
                m_rows = args[4]
                self.assertEqual(m_rows.numel(), NUM_GROUPS)
                self.assertGreaterEqual(int(m_rows.min()), 0)
                self.assertEqual(int(m_rows.sum()), m)
                self._assert_scratch(m_rows, weight, f"{name} row counts")

        # The warmup runs in compile-only mode, but a launch that slips through
        # must not write its tile over the weights.
        self._assert_scratch(output, weight, f"{name} output")
        self._assert_scratch(metadata, weight, f"{name} group metadata")

    def _assert_weight_untouched(
        self, weight: torch.Tensor, before: torch.Tensor
    ) -> None:
        self.assertTrue(
            torch.equal(weight, before),
            "warmup must not write through to the weight storage",
        )

    def _assert_compile_only_contract(self, covered: int) -> None:
        self.assertEqual(len(self.fake.calls), covered)
        self.assertGreater(covered, 0)
        # Compile-only is still requested, and the wheel's grid width restored.
        self.assertEqual(self.fake.compile_modes, [1, 0])
        self.assertEqual(self.fake.num_sms_calls, [NUM_SMS, NUM_SMS])

    def _run_grouped(self, dtype: str, layout: str) -> None:
        self._reset_recorder()
        if dtype == "bf16":
            weight = self._bf16_weight(NUM_GROUPS)
            before = weight.clone()
            covered = deepgemm_warmup.warmup_grouped_bf16_gemm(
                weight, max_m=MAX_M, layout=layout, mode="relax", num_sms=NUM_SMS
            )
        else:
            weight, scale = self._int8_weight(NUM_GROUPS)
            before = weight.clone()
            covered = deepgemm_warmup.warmup_grouped_int8_gemm(
                (weight, scale),
                max_m=MAX_M,
                layout=layout,
                mode="relax",
                num_sms=NUM_SMS,
            )
        self._assert_compile_only_contract(covered)
        for name, args in self.fake.calls:
            self._check_launch(name, args, layout, weight)
        self._assert_weight_untouched(weight, before)

    # -- tests ----------------------------------------------------------
    def test_bf16_grouped_launches_use_scratch_and_well_formed_metadata(self):
        for layout in ("masked", "contiguous", "nopad"):
            with self.subTest(layout=layout):
                self._run_grouped("bf16", layout)

    def test_int8_grouped_launches_use_scratch_and_well_formed_metadata(self):
        for layout in ("masked", "contiguous", "nopad"):
            with self.subTest(layout=layout):
                self._run_grouped("int8", layout)

    def test_int8_dense_launches_do_not_alias_the_weights(self):
        self._reset_recorder()
        weight, scale = self._int8_weight(1)
        before = weight.clone()
        covered = deepgemm_warmup.warmup_dense_int8_gemm(
            (weight[0], scale[0]), max_m=MAX_M, mode="relax", num_sms=NUM_SMS
        )
        self._assert_compile_only_contract(covered)
        for name, args in self.fake.calls:
            output = args[2]
            self.assertEqual(output.dtype, torch.bfloat16)
            self.assertEqual(output.ndim, 2)
            self.assertEqual(output.shape[-1], N)
            self._assert_scratch(output, weight, f"{name} output")
        self._assert_weight_untouched(weight, before)

    def test_skip_mode_compiles_nothing(self):
        weight = self._bf16_weight(NUM_GROUPS)
        self.assertEqual(
            deepgemm_warmup.warmup_grouped_bf16_gemm(
                weight, max_m=MAX_M, layout="masked", mode="skip", num_sms=NUM_SMS
            ),
            0,
        )
        self.assertEqual(self.fake.calls, [])


if __name__ == "__main__":
    unittest.main()