"""Unit test for ``dsv4/moe/strategies/base.py::select_strategy``.

Covers the priority matrix in the strategy module docstring + ``forced``
override + legacy env-toggle resolution + the explicit-fail-on-mismatch
contract. Pure-Python, no CUDA / DeepGEMM / dist required — runs on host.
"""

from __future__ import annotations

import os
import unittest
from contextlib import contextmanager
from unittest import mock

# Importing strategies populates the registry via ``register_strategy``.
from rtp_llm.models_py.modules.dsv4.moe.strategies import (
    DeepEPStrategy,
    GroupedFP4Strategy,
    LocalLoopStrategy,
    MegaMoEStrategy,
    MoeCfg,
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
            "DSV4_USE_GROUPED_FP4",
        ):
            os.environ.pop(k, None)

    # --- auto-pick matrix --------------------------------------------------

    def test_ep1_with_grouped_kernel_picks_grouped(self):
        with mock.patch.object(GroupedFP4Strategy, "can_handle", return_value=True), \
             mock.patch.object(MegaMoEStrategy, "can_handle", return_value=False):
            self.assertIs(select_strategy(_cfg(ep_size=1)),
                          GroupedFP4Strategy)

    def test_ep1_no_grouped_falls_to_local(self):
        with mock.patch.object(GroupedFP4Strategy, "can_handle", return_value=False), \
             mock.patch.object(MegaMoEStrategy, "can_handle", return_value=False), \
             mock.patch.object(DeepEPStrategy, "can_handle", return_value=False):
            self.assertIs(select_strategy(_cfg(ep_size=1)),
                          LocalLoopStrategy)

    def test_ep_gt1_with_mega_picks_mega(self):
        with mock.patch.object(MegaMoEStrategy, "can_handle", return_value=True):
            self.assertIs(select_strategy(_cfg(ep_size=4)),
                          MegaMoEStrategy)

    def test_ep_gt1_no_mega_picks_deepep(self):
        with mock.patch.object(MegaMoEStrategy, "can_handle", return_value=False):
            self.assertIs(select_strategy(_cfg(ep_size=4)),
                          DeepEPStrategy)

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
        # DSV4_USE_MEGA_MOE=0 should NOT force a different strategy — it just
        # disables mega via the strategy's own can_handle (mega_buf checks
        # the same env var). _resolve_forced returns (None, False) so
        # auto-pick runs.
        with _env(DSV4_USE_MEGA_MOE="0"):
            self.assertEqual(_resolve_forced(None), (None, False))

    def test_legacy_force_nonstrict_falls_through_when_incapable(self):
        # Legacy DSV4_USE_MEGA_MOE=1 + ep_size=1 cfg: Mega.can_handle False
        # because ep_size=1; should silently fall through to LocalLoop
        # (NOT raise — that's the strict-mode behaviour). Mirrors the
        # 64k_cp4_ep1 smoke that has ep_size=1 + DSV4_USE_MEGA_MOE=1.
        with mock.patch.object(MegaMoEStrategy, "can_handle", return_value=False), \
             mock.patch.object(GroupedFP4Strategy, "can_handle", return_value=False):
            self.assertIs(
                select_strategy(_cfg(ep_size=1), forced="mega", strict=False),
                LocalLoopStrategy,
            )


if __name__ == "__main__":
    unittest.main()
