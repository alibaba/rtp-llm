"""Unit test for ``dsv4/moe/strategies/base.py::select_strategy``.

Covers the priority matrix in the strategy module docstring + ``forced``
override + legacy env-toggle resolution + the explicit-fail-on-mismatch
contract. Pure-Python, no CUDA / DeepGEMM / dist required — runs on host.
"""

from __future__ import annotations

import inspect
import os
import sys
import types
import unittest
from contextlib import contextmanager
from dataclasses import replace
from unittest import mock

from rtp_llm.models_py.modules.dsv4.moe import mega_se_buf

# Importing strategies populates the registry via ``register_strategy``.
from rtp_llm.models_py.modules.dsv4.moe.strategies import (
    DeepEPStrategy,
    GroupedFP4Strategy,
    LocalLoopStrategy,
    MegaMoEStrategy,
    MegaMoEStrategySE,
    MoeCfg,
    _has_fp8_fp4_grouped_kernel,
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


def _fake_shared32_module(*, exact_combine: bool):
    def mega(
        y,
        l1_weights,
        l2_weights,
        sym_buffer,
        shared_l1_weights=None,
        shared_l2_weights=None,
        recipe=(1, 1, 32),
        activation_clamp=None,
        round_swiglu_to_bf16=False,
        torch_sum_combine=False,
    ):
        pass

    if not exact_combine:
        signature = inspect.signature(mega)
        mega.__signature__ = signature.replace(
            parameters=[
                parameter
                for name, parameter in signature.parameters.items()
                if name != "torch_sum_combine"
            ]
        )

    def buffer(group, num_shared_experts=0, mma_type="fp8xfp4"):
        pass

    return types.SimpleNamespace(
        fp8_fp4_mega_moe=mega,
        get_symm_buffer_for_mega_moe=buffer,
        get_block_m_for_mega_moe=object(),
        transform_weights_for_mega_moe=object(),
        transform_sf_into_required_layout=object(),
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
        ):
            os.environ.pop(k, None)
        # Most selection-matrix tests exercise the routed-only baseline.
        # Tests for the new default explicitly remove this override.
        os.environ["DSV4_USE_MEGA_MOE_SE"] = "0"

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
        with mock.patch.object(
            GroupedFP4Strategy, "can_handle", return_value=False
        ), mock.patch.object(
            MegaMoEStrategy, "can_handle", return_value=False
        ), mock.patch.object(
            DeepEPStrategy, "can_handle", return_value=False
        ):
            self.assertIs(select_strategy(_cfg(ep_size=1)), LocalLoopStrategy)

    def test_ep_gt1_with_mega_picks_mega(self):
        with mock.patch.object(MegaMoEStrategy, "can_handle", return_value=True):
            self.assertIs(select_strategy(_cfg(ep_size=4)), MegaMoEStrategy)

    def test_ep_gt1_default_picks_mega_se_when_capable(self):
        with _env(DSV4_USE_MEGA_MOE_SE=None), mock.patch.object(
            MegaMoEStrategy, "can_handle", return_value=True
        ), mock.patch.object(MegaMoEStrategySE, "can_handle", return_value=True):
            self.assertIs(select_strategy(_cfg(ep_size=4)), MegaMoEStrategySE)

    def test_ep_gt1_explicit_se_zero_picks_non_fused_mega(self):
        with _env(DSV4_USE_MEGA_MOE_SE="0"), mock.patch.object(
            MegaMoEStrategy, "can_handle", return_value=True
        ), mock.patch.object(MegaMoEStrategySE, "can_handle", return_value=True):
            self.assertIs(select_strategy(_cfg(ep_size=4)), MegaMoEStrategy)

    def test_new_api_default_shared128_uses_routed_mega(self):
        with _env(DSV4_USE_MEGA_MOE_SE=None), mock.patch.object(
            MegaMoEStrategy, "can_handle", return_value=True
        ), mock.patch.object(
            MegaMoEStrategySE, "can_handle", return_value=False
        ), mock.patch(
            "rtp_llm.models_py.modules.dsv4.moe.mega_se_buf.mega_moe_se_requires_shared32",
            return_value=True,
        ):
            self.assertIs(select_strategy(_cfg(ep_size=8)), MegaMoEStrategy)

    def test_new_api_explicit_shared128_fusion_is_rejected(self):
        with _env(DSV4_USE_MEGA_MOE_SE="1"), mock.patch.object(
            MegaMoEStrategySE, "can_handle", return_value=False
        ), self.assertRaisesRegex(RuntimeError, "cannot handle"):
            select_strategy(_cfg(ep_size=8))

    def test_shared32_requires_fused_strategy(self):
        cfg = replace(_cfg(ep_size=8), shared_fp8_block_size=32)
        with _env(DSV4_USE_MEGA_MOE_SE=None), mock.patch.object(
            MegaMoEStrategySE, "can_handle", return_value=True
        ):
            self.assertIs(select_strategy(cfg), MegaMoEStrategySE)
        with _env(DSV4_USE_MEGA_MOE_SE="0"), self.assertRaisesRegex(
            RuntimeError, "requires the fused"
        ):
            select_strategy(cfg)
        with _env(DSV4_MOE_STRATEGY="mega"), self.assertRaisesRegex(
            RuntimeError, "requires the fused"
        ):
            select_strategy(cfg, forced="mega")

    def test_shared32_clamp_is_explicit(self):
        with self.assertRaisesRegex(ValueError, "clamp 10.0"):
            replace(_cfg(ep_size=8), shared_fp8_block_size=32, swiglu_limit=0.0)

    def test_shared32_capability_rejects_rounding_only_wheel_without_fallback(self):
        cfg = replace(_cfg(ep_size=8), shared_fp8_block_size=32)
        module = _fake_shared32_module(exact_combine=False)
        with _env(DSV4_USE_MEGA_MOE_SE=None), mock.patch.dict(
            sys.modules, {"deep_gemm": module}
        ), mock.patch.object(
            mega_se_buf, "_mega_moe_unavailable_reason", return_value=None
        ):
            self.assertIn(
                "torch_sum_combine patch",
                mega_se_buf._mega_moe_se_unavailable_reason(32),
            )
            self.assertFalse(MegaMoEStrategySE.can_handle(cfg))
            with self.assertRaisesRegex(RuntimeError, "cannot handle"):
                select_strategy(cfg)

    def test_shared32_complete_wheel_is_selected_without_changing_defaults(self):
        cfg = replace(_cfg(ep_size=8), shared_fp8_block_size=32)
        module = _fake_shared32_module(exact_combine=True)
        with _env(DSV4_USE_MEGA_MOE_SE=None), mock.patch.dict(
            sys.modules, {"deep_gemm": module}
        ), mock.patch.object(
            mega_se_buf, "_mega_moe_unavailable_reason", return_value=None
        ):
            self.assertIsNone(mega_se_buf._mega_moe_se_unavailable_reason(32))
            self.assertIs(select_strategy(cfg), MegaMoEStrategySE)
            parameters = inspect.signature(module.fp8_fp4_mega_moe).parameters
            self.assertIs(parameters["round_swiglu_to_bf16"].default, False)
            self.assertIs(parameters["torch_sum_combine"].default, False)

    def test_shared32_weight_setup_rejects_old_wheel_before_consuming_weights(self):
        cfg = replace(_cfg(ep_size=8), shared_fp8_block_size=32)
        module = _fake_shared32_module(exact_combine=False)
        for strategy in (MegaMoEStrategySE, MegaMoEStrategy):
            with self.subTest(strategy=strategy.name), mock.patch.dict(
                sys.modules, {"deep_gemm": module}
            ):
                weights = {"unconsumed": object()}
                before = dict(weights)
                with self.assertRaisesRegex(RuntimeError, "torch_sum_combine patch"):
                    strategy(cfg).setup_weights(weights)
                self.assertEqual(weights, before)

    def test_ep_gt1_no_mega_raises(self):
        with mock.patch.object(MegaMoEStrategy, "can_handle", return_value=False):
            with self.assertRaises(RuntimeError) as cm:
                select_strategy(_cfg(ep_size=4))
        self.assertIn("requires MegaMoEStrategy", str(cm.exception))
        self.assertIn("fallback to DeepEP/LocalLoop is disabled", str(cm.exception))

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
        self.assertIn("requires MegaMoEStrategy", str(cm.exception))
        self.assertIn("bypass Mega", str(cm.exception))

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
        self.assertIn("conflicts with DSV4_USE_MEGA_MOE_FUSED=1", str(cm.exception))

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
        ), mock.patch.object(GroupedFP4Strategy, "can_handle", return_value=False):
            self.assertIs(
                select_strategy(_cfg(ep_size=1), forced="mega", strict=False),
                LocalLoopStrategy,
            )


if __name__ == "__main__":
    unittest.main()
