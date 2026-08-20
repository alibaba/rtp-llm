"""Unit tests for warmup_aiter_batch_prefill gating and call contract.

The real kernel invocation is fully mocked so these tests never trigger an
aiter JIT build; they only verify the warmup switch, one-shot guard and the
arguments handed to aiter.mha_batch_prefill_func.

Skips automatically off ROCm or without ``aiter`` so the suite stays green
on the rest of the fleet.
"""

import unittest
from unittest import mock

import torch

_IS_ROCM_BUILD = torch.version.hip is not None

try:
    import aiter  # noqa: F401

    _AITER_AVAILABLE = True
except ImportError:
    if _IS_ROCM_BUILD:
        raise
    _AITER_AVAILABLE = False

try:
    from rtp_llm.models_py.modules.factory.attention.rocm_impl import (
        aiter as aiter_mod,
    )

    _OPS_IMPORTABLE = True
except ImportError:
    if _IS_ROCM_BUILD:
        raise
    _OPS_IMPORTABLE = False


@unittest.skipUnless(_AITER_AVAILABLE and _OPS_IMPORTABLE, "requires ROCm + aiter")
class AiterBatchPrefillWarmupTest(unittest.TestCase):
    def setUp(self):
        self._orig_done = aiter_mod._batch_prefill_warmup_done
        aiter_mod._batch_prefill_warmup_done = False

    def tearDown(self):
        aiter_mod._batch_prefill_warmup_done = self._orig_done

    def _run_with_mocks(self, warm_up_enabled: bool, fp8_kv_cache: bool = False):
        fake_torch = mock.MagicMock()
        # vs = 16 // element_size(): bf16 -> 8, fp8 -> 16
        element_size = 1 if fp8_kv_cache else 2
        fake_torch.empty.return_value.element_size.return_value = element_size
        with mock.patch.object(aiter_mod, "torch", fake_torch), mock.patch.object(
            aiter_mod.aiter, "mha_batch_prefill_func"
        ) as func, mock.patch.object(
            aiter_mod, "model_warm_up_enabled", return_value=warm_up_enabled
        ):
            aiter_mod.warmup_aiter_batch_prefill(fp8_kv_cache=fp8_kv_cache)
        return func

    def test_skip_when_model_warmup_disabled(self):
        func = self._run_with_mocks(warm_up_enabled=False)
        func.assert_not_called()

    def test_calls_func_once_with_production_flags(self):
        func = self._run_with_mocks(warm_up_enabled=True)
        self.assertEqual(func.call_count, 1)
        kwargs = func.call_args.kwargs
        self.assertTrue(kwargs["causal"])
        self.assertIsNone(kwargs["q_descale"])
        self.assertIsNone(kwargs["k_descale"])
        self.assertIsNone(kwargs["v_descale"])
        self.assertIsNotNone(kwargs["block_table"])
        self.assertIsNotNone(kwargs["seqlen_k"])

        # one-shot guard: a second call must not re-trigger the kernel
        aiter_mod.warmup_aiter_batch_prefill()
        self.assertEqual(func.call_count, 1)

    def test_fp8_kv_cache_passes_descales(self):
        func = self._run_with_mocks(warm_up_enabled=True, fp8_kv_cache=True)
        self.assertEqual(func.call_count, 1)
        kwargs = func.call_args.kwargs
        self.assertIsNotNone(kwargs["q_descale"])
        self.assertIsNotNone(kwargs["k_descale"])
        self.assertIsNotNone(kwargs["v_descale"])

    def test_kernel_failure_is_swallowed(self):
        fake_torch = mock.MagicMock()
        fake_torch.empty.return_value.element_size.return_value = 2
        with mock.patch.object(aiter_mod, "torch", fake_torch), mock.patch.object(
            aiter_mod.aiter,
            "mha_batch_prefill_func",
            side_effect=RuntimeError("build failed"),
        ) as func, mock.patch.object(
            aiter_mod, "model_warm_up_enabled", return_value=True
        ):
            aiter_mod.warmup_aiter_batch_prefill()
        func.assert_called_once()


if __name__ == "__main__":
    unittest.main()
