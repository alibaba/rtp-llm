"""Unit test for ``dsv4/moe/strategies/base.py::select_strategy``.

Covers the priority matrix in the strategy module docstring + ``forced``
override + legacy env-toggle resolution + the explicit-fail-on-mismatch
contract. Pure-Python, no CUDA / DeepGEMM / dist required — runs on host.
"""

from __future__ import annotations

import dataclasses
import os
import sys
import types
import unittest
from contextlib import contextmanager
from unittest import mock

# Importing strategies populates the registry via ``register_strategy``.
from rtp_llm.models_py.modules.dsv4.moe.strategies import (
    DeepEPStrategy,
    GroupedFP4Strategy,
    GroupedFP8Strategy,
    LocalLoopStrategy,
    MegaMoEStrategy,
    MegaMoEStrategySE,
    MoeCfg,
    _has_fp8_fp4_grouped_kernel,
    _has_grouped_fp8_kernel,
    select_strategy,
)
from rtp_llm.models_py.modules.dsv4.moe.strategies.base import _resolve_forced


def _cfg(ep_size: int = 1) -> MoeCfg:
    """A minimal MoeCfg sufficient for ``can_handle`` checks."""
    n_local = 256 // max(ep_size, 1)
    return MoeCfg(
        layer_id=2,
        dim=7168,
        moe_inter_dim=2048,
        n_routed_experts=256,
        n_activated_experts=6,
        swiglu_limit=10.0,
        ep_size=ep_size,
        ep_rank=0,
        n_local_experts=n_local,
        local_expert_start=0,
        local_expert_end=n_local,
        max_tokens_per_rank=8192,
    )


@contextmanager
def _env(**kw):
    """Temporarily set env vars; ``None`` value pops the var."""
    saved = {k: os.environ.get(k) for k in kw}
    try:
        for k, v in kw.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = str(v)
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class StrategySelectTest(unittest.TestCase):
    """Cover the (ep_size, kernel_avail, mega_avail) matrix."""

    def setUp(self):
        # Ensure clean env baseline for every test.
        for k in (
            "DSV4_MOE_STRATEGY",
            "DSV4_USE_MEGA_MOE",
            "DSV4_USE_MEGA_MOE_SE",
            "DSV4_USE_MEGA_MOE_FUSED",
            "DSV4_USE_GROUPED_FP4",
            "DSV4_USE_GROUPED_FP8",
        ):
            os.environ.pop(k, None)

    # --- auto-pick matrix --------------------------------------------------

    def test_ep1_with_grouped_kernel_picks_grouped(self):
        with mock.patch.object(
            GroupedFP4Strategy, "can_handle", return_value=True
        ), mock.patch.object(MegaMoEStrategy, "can_handle", return_value=False):
            self.assertIs(select_strategy(_cfg(ep_size=1)), GroupedFP4Strategy)

    def test_grouped_selection_is_gated_by_ep_size(self):
        cfg = _cfg(ep_size=2)
        with mock.patch(
            "rtp_llm.models_py.modules.dsv4.moe.strategies.grouped_fp4."
            "_has_fp8_fp4_grouped_kernel",
            return_value=True,
        ):
            self.assertFalse(GroupedFP4Strategy.can_handle(cfg))

    def test_grouped_kernel_probe_requires_sm100(self):
        fake_deep_gemm = types.SimpleNamespace(
            m_grouped_fp8_fp4_gemm_nt_contiguous=object(),
            get_mk_alignment_for_contiguous_layout=lambda: (128, 128),
        )
        with mock.patch.dict(sys.modules, {"deep_gemm": fake_deep_gemm}), mock.patch(
            "rtp_llm.models_py.modules.dsv4.moe.strategies.grouped_fp4."
            "torch.cuda.is_available",
            return_value=True,
        ), mock.patch(
            "rtp_llm.models_py.modules.dsv4.moe.strategies.grouped_fp4."
            "torch.cuda.get_device_capability",
            return_value=(12, 0),
        ):
            self.assertFalse(_has_fp8_fp4_grouped_kernel())

        with mock.patch.dict(sys.modules, {"deep_gemm": fake_deep_gemm}), mock.patch(
            "rtp_llm.models_py.modules.dsv4.moe.strategies.grouped_fp4."
            "torch.cuda.is_available",
            return_value=True,
        ), mock.patch(
            "rtp_llm.models_py.modules.dsv4.moe.strategies.grouped_fp4."
            "torch.cuda.get_device_capability",
            return_value=(10, 0),
        ):
            self.assertTrue(_has_fp8_fp4_grouped_kernel())

    def test_ep1_no_grouped_falls_to_local(self):
        # Both grouped strategies have to be out of the way: grouped_fp8 sits
        # above local_loop in the priority list and is capable wherever the SM90
        # FP8 grouped kernel resolves.
        with mock.patch.object(
            GroupedFP4Strategy, "can_handle", return_value=False
        ), mock.patch.object(
            GroupedFP8Strategy, "can_handle", return_value=False
        ), mock.patch.object(
            MegaMoEStrategy, "can_handle", return_value=False
        ), mock.patch.object(
            DeepEPStrategy, "can_handle", return_value=False
        ):
            self.assertIs(select_strategy(_cfg(ep_size=1)), LocalLoopStrategy)

    def test_ep_gt1_with_mega_picks_mega(self):
        with mock.patch.object(MegaMoEStrategy, "can_handle", return_value=True):
            self.assertIs(select_strategy(_cfg(ep_size=4)), MegaMoEStrategy)

    def test_ep_gt1_default_stays_mega_when_se_is_capable(self):
        with mock.patch.object(
            MegaMoEStrategy, "can_handle", return_value=True
        ), mock.patch.object(MegaMoEStrategySE, "can_handle", return_value=True):
            self.assertIs(select_strategy(_cfg(ep_size=4)), MegaMoEStrategy)

    def test_ep_gt1_no_ep_capable_raises(self):
        # grouped_fp8 joined mega/mega_fused as EP-capable, so the raise now
        # requires every one of them to decline.
        with mock.patch.object(MegaMoEStrategy, "can_handle", return_value=False), \
             mock.patch.object(GroupedFP8Strategy, "can_handle", return_value=False):
            with self.assertRaises(RuntimeError) as cm:
                select_strategy(_cfg(ep_size=4))
        self.assertIn("requires one of", str(cm.exception))
        self.assertIn("grouped_fp8", str(cm.exception))
        self.assertIn("fallback to DeepEP/LocalLoop is disabled", str(cm.exception))

    # --- grouped_fp8: the SM90 EP path ------------------------------------

    def test_ep_gt1_sm90_picks_grouped_fp8(self):
        """Mega is SM100-only; on SM90 grouped_fp8 is what serves ep_size > 1."""
        with mock.patch.object(MegaMoEStrategy, "can_handle", return_value=False), \
             mock.patch.object(GroupedFP8Strategy, "can_handle", return_value=True):
            self.assertIs(select_strategy(_cfg(ep_size=4)), GroupedFP8Strategy)

    def test_ep1_prefers_grouped_fp8_over_local_loop(self):
        """grouped_fp8 outranks local_loop, which hardcodes FP4 expert storage."""
        with mock.patch.object(GroupedFP4Strategy, "can_handle", return_value=False), \
             mock.patch.object(MegaMoEStrategy, "can_handle", return_value=False), \
             mock.patch.object(GroupedFP8Strategy, "can_handle", return_value=True):
            self.assertIs(select_strategy(_cfg(ep_size=1)), GroupedFP8Strategy)

    def test_forced_grouped_fp8_at_ep_gt1_is_allowed(self):
        """It carries its own EP combine, so forcing it under EP must not raise."""
        with mock.patch.object(GroupedFP8Strategy, "can_handle", return_value=True):
            self.assertIs(
                select_strategy(_cfg(ep_size=4), forced="grouped_fp8"),
                GroupedFP8Strategy,
            )

    def test_grouped_fp8_kernel_probe(self):
        """env off / no CUDA / kernel absent / non-Hopper each disable it.

        The predicate is ``@functools.cache``d -- it is called once per layer, so
        caching the import and the capability query is deliberate -- which means
        every case here has to clear it first.
        """
        impl = (
            "rtp_llm.models_py.kernels.cuda.deepgemm_wrapper."
            "_m_grouped_fp8_gemm_nt_contiguous_impl"
        )
        cuda = (
            "rtp_llm.models_py.modules.dsv4.moe.strategies.grouped_fp8.torch.cuda"
        )

        def probe():
            _has_grouped_fp8_kernel.cache_clear()
            return _has_grouped_fp8_kernel()

        with _env(DSV4_USE_GROUPED_FP8="0"):
            self.assertFalse(probe())
        with mock.patch(f"{cuda}.is_available", return_value=False):
            self.assertFalse(probe())
        with mock.patch(f"{cuda}.is_available", return_value=True), \
             mock.patch(impl, None):
            self.assertFalse(probe())
        # Gated to Hopper rather than "not SM100" on purpose: SM100 has the FP4
        # kernels, which are faster, so this path would only regress there.
        with mock.patch(f"{cuda}.is_available", return_value=True), \
             mock.patch(impl, object()), \
             mock.patch(f"{cuda}.get_device_capability", return_value=(10, 0)):
            self.assertFalse(probe())
        with mock.patch(f"{cuda}.is_available", return_value=True), \
             mock.patch(impl, object()), \
             mock.patch(f"{cuda}.get_device_capability", return_value=(9, 0)):
            self.assertTrue(probe())
        _has_grouped_fp8_kernel.cache_clear()

    # --- forced override ---------------------------------------------------

    def test_forced_known_and_capable_returns_it(self):
        self.assertIs(
            select_strategy(_cfg(ep_size=1), forced="local_loop"),
            LocalLoopStrategy,
        )

    def test_forced_known_but_incapable_raises(self):
        # Force grouped_fp4 with grouped kernel mocked unavailable.
        with mock.patch.object(GroupedFP4Strategy, "can_handle", return_value=False):
            with self.assertRaises(RuntimeError) as cm:
                select_strategy(_cfg(ep_size=1), forced="grouped_fp4")
        self.assertIn("Forced MoE strategy 'grouped_fp4'", str(cm.exception))
        self.assertIn("cannot handle", str(cm.exception))

    def test_forced_ep_gt1_non_mega_raises_even_if_capable(self):
        with mock.patch.object(DeepEPStrategy, "can_handle", return_value=True):
            with self.assertRaises(RuntimeError) as cm:
                select_strategy(_cfg(ep_size=4), forced="deepep")
        self.assertIn("requires one of", str(cm.exception))
        self.assertIn("has no EP combine", str(cm.exception))

    def test_forced_unknown_raises(self):
        with self.assertRaises(RuntimeError) as cm:
            select_strategy(_cfg(), forced="bogus")
        self.assertIn("Unknown MoE strategy 'bogus'", str(cm.exception))
        self.assertIn("Available", str(cm.exception))

    # --- env resolution ----------------------------------------------------

    def test_env_dsv4_moe_strategy_overrides_ctor(self):
        with _env(DSV4_MOE_STRATEGY="local_loop"):
            self.assertEqual(_resolve_forced(None), ("local_loop", True))
            self.assertEqual(_resolve_forced("mega"), ("local_loop", True))

    def test_env_dsv4_moe_strategy_auto_falls_through(self):
        with _env(DSV4_MOE_STRATEGY="auto"):
            self.assertEqual(_resolve_forced(None), (None, False))
            self.assertEqual(_resolve_forced("mega"), ("mega", True))

    def test_legacy_use_mega_moe_1_translates_to_mega_nonstrict(self):
        # Legacy toggle is non-strict: ``select_strategy`` falls through to
        # auto-pick when the named strategy can't handle the cfg (e.g.
        # ep_size=1 + Mega). Smokes commonly leave DSV4_USE_MEGA_MOE=1
        # ON across configs that include ep_size=1.
        with _env(DSV4_USE_MEGA_MOE="1"):
            self.assertEqual(_resolve_forced(None), ("mega", False))

    def test_mega_moe_se_opt_in_is_strict(self):
        with _env(DSV4_USE_MEGA_MOE_SE="1"):
            self.assertEqual(_resolve_forced(None), ("mega_se", True))

    def test_mega_moe_se_opt_in_accepts_generic_mega_hint(self):
        with _env(DSV4_USE_MEGA_MOE_SE="1", DSV4_USE_MEGA_MOE="1"):
            self.assertEqual(_resolve_forced(None), ("mega_se", True))

    def test_mega_moe_se_opt_in_accepts_generic_mega_ctor(self):
        with _env(DSV4_USE_MEGA_MOE_SE="1"):
            self.assertEqual(_resolve_forced("mega"), ("mega_se", True))

    def test_mega_moe_se_and_grouped_conflict(self):
        with _env(
            DSV4_USE_MEGA_MOE_SE="1",
            DSV4_USE_GROUPED_FP4="1",
        ):
            with self.assertRaises(RuntimeError) as cm:
                _resolve_forced(None)
        self.assertIn("Conflicting", str(cm.exception))

    def test_mega_moe_se_opt_in_selects_se(self):
        with _env(DSV4_USE_MEGA_MOE_SE="1"), mock.patch.object(
            MegaMoEStrategySE, "can_handle", return_value=True
        ):
            forced, strict = _resolve_forced(None)
            self.assertIs(
                select_strategy(_cfg(ep_size=2), forced=forced, strict=strict),
                MegaMoEStrategySE,
            )

    def test_mega_moe_se_unavailable_fails_loudly(self):
        with _env(DSV4_USE_MEGA_MOE_SE="1"), mock.patch.object(
            MegaMoEStrategySE, "can_handle", return_value=False
        ):
            forced, strict = _resolve_forced(None)
            with self.assertRaises(RuntimeError) as cm:
                select_strategy(_cfg(ep_size=2), forced=forced, strict=strict)
        self.assertIn("Forced MoE strategy 'mega_se'", str(cm.exception))

    def test_mega_moe_se_and_old_fused_conflict(self):
        with _env(
            DSV4_USE_MEGA_MOE_SE="1",
            DSV4_USE_MEGA_MOE_FUSED="1",
        ):
            with self.assertRaises(RuntimeError) as cm:
                select_strategy(_cfg(ep_size=2))
        self.assertIn("select exactly one Mega variant", str(cm.exception))

    def test_legacy_use_grouped_fp4_1_translates_to_grouped_nonstrict(self):
        with _env(DSV4_USE_GROUPED_FP4="1"):
            self.assertEqual(_resolve_forced(None), ("grouped_fp4", False))

    def test_legacy_conflicting_positives_raise(self):
        with _env(DSV4_USE_MEGA_MOE="1", DSV4_USE_GROUPED_FP4="1"):
            with self.assertRaises(RuntimeError) as cm:
                _resolve_forced(None)
            self.assertIn("Conflicting", str(cm.exception))

    def test_legacy_conflicting_with_ctor_raises(self):
        with _env(DSV4_USE_MEGA_MOE="1"):
            with self.assertRaises(RuntimeError) as cm:
                _resolve_forced("grouped_fp4")
            self.assertIn("Conflicting MoE strategy", str(cm.exception))

    def test_legacy_negation_does_not_force_alternative(self):
        # DSV4_USE_MEGA_MOE=0 should NOT force a different strategy. EP>1
        # select_strategy() treats disabled Mega as a fatal config error.
        with _env(DSV4_USE_MEGA_MOE="0"):
            self.assertEqual(_resolve_forced(None), (None, False))

    def test_legacy_negation_ep_gt1_raises(self):
        with _env(DSV4_USE_MEGA_MOE="0"):
            with self.assertRaises(RuntimeError) as cm:
                select_strategy(_cfg(ep_size=4))
        self.assertIn("DSV4_USE_MEGA_MOE=0 disables Mega MoE", str(cm.exception))

    def test_legacy_force_nonstrict_falls_through_when_incapable(self):
        # Legacy DSV4_USE_MEGA_MOE=1 + ep_size=1 cfg: Mega.can_handle False
        # because ep_size=1; should silently fall through to LocalLoop
        # (NOT raise — that's the strict-mode behaviour). Mirrors the
        # 64k_cp4_ep1 smoke that has ep_size=1 + DSV4_USE_MEGA_MOE=1.
        with mock.patch.object(
            MegaMoEStrategy, "can_handle", return_value=False
        ), mock.patch.object(
            GroupedFP4Strategy, "can_handle", return_value=False
        ), mock.patch.object(
            GroupedFP8Strategy, "can_handle", return_value=False
        ):
            self.assertIs(
                select_strategy(_cfg(ep_size=1), forced="mega", strict=False),
                LocalLoopStrategy,
            )


class GroupedFP8CaptureGuardTest(unittest.TestCase):
    """``_assert_one_captured_size``: turn a replay-time NCCL hang into a startup error.

    The hazard is that ``CudaGraphRunner`` selects a graph per rank from that rank's
    own batch size while this strategy's EP collectives sit inside the graph, so two
    ranks on different graphs hang. Neither condition is observable from inside the
    strategy at replay time, but both are observable at capture time: more than one
    captured size, or a captured size the scheduler can exceed.
    """

    CAPTURING = (
        "rtp_llm.models_py.modules.dsv4.moe.strategies.grouped_fp8."
        "torch.cuda.is_current_stream_capturing"
    )

    def _strategy(self, ep_size: int, max_tokens_per_rank: int) -> GroupedFP8Strategy:
        # MoeCfg is immutable, so the bound is set by rebuilding it.
        cfg = dataclasses.replace(
            _cfg(ep_size=ep_size), max_tokens_per_rank=max_tokens_per_rank
        )
        strat = GroupedFP8Strategy(cfg)
        strat._captured_ns = set()
        return strat

    def test_single_size_at_the_scheduler_bound_is_accepted(self):
        strat = self._strategy(ep_size=4, max_tokens_per_rank=8)
        with mock.patch(self.CAPTURING, return_value=True):
            strat._assert_one_captured_size(8)
            strat._assert_one_captured_size(8)  # same size again is fine

    def test_second_captured_size_raises(self):
        strat = self._strategy(ep_size=4, max_tokens_per_rank=8)
        with mock.patch(self.CAPTURING, return_value=True):
            strat._assert_one_captured_size(8)
            with self.assertRaises(RuntimeError) as cm:
                strat._assert_one_captured_size(16)
        self.assertIn("more than one batch size", str(cm.exception))
        self.assertIn("concurrency_limit", str(cm.exception))

    def test_captured_size_below_scheduler_bound_raises(self):
        # A rank whose batch exceeds the graph falls back to eager while the others
        # replay -- the same divergence, reached the other way.
        strat = self._strategy(ep_size=4, max_tokens_per_rank=64)
        with mock.patch(self.CAPTURING, return_value=True):
            with self.assertRaises(RuntimeError) as cm:
                strat._assert_one_captured_size(8)
        self.assertIn("may hand it up to", str(cm.exception))

    def test_ep1_and_non_capturing_are_exempt(self):
        # ep_size == 1 has no EP collectives in the graph, and outside capture the
        # runner is not selecting graphs at all.
        strat = self._strategy(ep_size=1, max_tokens_per_rank=8)
        with mock.patch(self.CAPTURING, return_value=True):
            strat._assert_one_captured_size(8)
            strat._assert_one_captured_size(999)
        strat = self._strategy(ep_size=4, max_tokens_per_rank=8)
        with mock.patch(self.CAPTURING, return_value=False):
            strat._assert_one_captured_size(8)
            strat._assert_one_captured_size(999)


class GroupedFP8LowLatencyGateTest(unittest.TestCase):
    """The low-latency buffer must only be built where the role can reach it.

    ``_ll_max_tokens`` rounds the requested tokens-per-rank up to the alignment the
    masked GEMM needs at this ``ep``: the 64-row floor is on ``M_MAX * ep``, not on
    ``M_MAX``, so the unit is ``64 // gcd(64, ep)``.
    """

    def test_ll_max_tokens_alignment_unit_follows_ep(self):
        from rtp_llm.models_py.modules.dsv4.moe.strategies.grouped_fp8 import (
            _ll_max_tokens,
        )

        with _env(DSV4_MOE_LL_MAX_TOKENS="8"):
            self.assertEqual(_ll_max_tokens(4), 16)   # 64 // gcd(64, 4) = 16
            self.assertEqual(_ll_max_tokens(8), 8)    # 64 // gcd(64, 8) = 8
            self.assertEqual(_ll_max_tokens(1), 64)   # single rank: the full floor
        with _env(DSV4_MOE_LL_MAX_TOKENS="0"):
            self.assertEqual(_ll_max_tokens(4), 16)   # empty/0 still clears the floor

    def test_gate_predicate_separates_the_roles(self):
        # resolve_moe_max_tokens_per_rank gives a decode role
        # max_generate_batch_size * tokens_per_batch, and any other role a >= 4096
        # budget, so the predicate below is exactly "is this a decode role".
        from rtp_llm.models_py.modules.dsv4.moe.strategies.grouped_fp8 import (
            _ll_max_tokens,
        )

        with _env(DSV4_MOE_LL_MAX_TOKENS="8"):
            self.assertLessEqual(8, _ll_max_tokens(4))       # decode, conc 8
            self.assertGreater(8192, _ll_max_tokens(4))      # prefill budget


if __name__ == "__main__":
    unittest.main()
