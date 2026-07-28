"""Factory-level selection tests for the SUPPORTS_NONCAUSAL prefill gate.

Pure-logic tests: the registry is monkeypatched with fake impls, so no GPU
kernels run. Guards the invariant that a non-causal prefill batch never
selects an impl that does not claim SUPPORTS_NONCAUSAL (a hardcoded-causal
kernel would silently drop the semantics).
"""

import unittest
from types import SimpleNamespace
from unittest import mock

from rtp_llm.models_py.modules.factory.attention import attn_factory
from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import FMHAImplBase


class _FakeImplBase(FMHAImplBase):
    """Instantiable fake: records nothing, computes nothing."""

    def __init__(self, attn_configs, attn_inputs, parallelism_config=None):
        pass

    def forward(self, qkv, kv_cache, layer_idx=0):
        raise NotImplementedError

    @staticmethod
    def support(attn_configs, attn_inputs):
        return True


class _HardcodedCausalImpl(_FakeImplBase):
    """Simulates a kernel that hardcodes causal=True (default flag: False)."""


class _NonCausalCapableImpl(_FakeImplBase):
    SUPPORTS_NONCAUSAL = True


def _make_inputs(is_prefill=True):
    return SimpleNamespace(is_prefill=is_prefill)


def _make_configs(is_causal):
    return SimpleNamespace(is_causal=is_causal)


class AttnFactoryNonCausalGateTest(unittest.TestCase):
    def _get_impl(self, impls, is_causal, is_prefill=True):
        with mock.patch.object(attn_factory, "PREFILL_MHA_IMPS", impls), \
                mock.patch.object(attn_factory, "DECODE_MHA_IMPS", impls):
            return attn_factory.get_fmha_impl(
                _make_configs(is_causal), None, _make_inputs(is_prefill)
            )

    def test_noncausal_prefill_skips_hardcoded_causal_impl(self):
        # Hardcoded-causal impl has higher priority but must be skipped.
        impl = self._get_impl(
            [_HardcodedCausalImpl, _NonCausalCapableImpl], is_causal=False
        )
        self.assertIsInstance(impl, _NonCausalCapableImpl)
        self.assertTrue(type(impl).SUPPORTS_NONCAUSAL)

    def test_causal_prefill_keeps_priority_order(self):
        # The gate must be a no-op for causal batches: priority wins.
        impl = self._get_impl(
            [_HardcodedCausalImpl, _NonCausalCapableImpl], is_causal=True
        )
        self.assertIsInstance(impl, _HardcodedCausalImpl)

    def test_noncausal_decode_is_not_gated(self):
        # The gate only applies to prefill; decode selection is unchanged.
        impl = self._get_impl(
            [_HardcodedCausalImpl, _NonCausalCapableImpl],
            is_causal=False,
            is_prefill=False,
        )
        self.assertIsInstance(impl, _HardcodedCausalImpl)

    def test_noncausal_prefill_with_no_capable_impl_raises_with_reason(self):
        with self.assertRaises(Exception) as ctx:
            self._get_impl([_HardcodedCausalImpl], is_causal=False)
        msg = str(ctx.exception)
        self.assertIn("is_causal=False", msg)
        self.assertIn("_HardcodedCausalImpl: no SUPPORTS_NONCAUSAL", msg)


if __name__ == "__main__":
    unittest.main()
