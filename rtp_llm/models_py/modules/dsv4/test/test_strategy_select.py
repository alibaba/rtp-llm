"""Unit test for ``dsv4/moe/strategies/base.py::select_strategy``.

Covers the priority matrix in the strategy module docstring + ``forced``
override + legacy env-toggle resolution + the explicit-fail-on-mismatch
contract. Pure-Python, no CUDA / DeepGEMM / dist required — runs on host.
"""

from __future__ import annotations

import os
import sys
import types
import unittest
from contextlib import contextmanager
from unittest import mock

# Importing strategies populates the registry via ``register_strategy``.
from rtp_llm.models_py.modules.dsv4.moe.mega_se_buf import (
    _mega_moe_se_unavailable_reason,
)
from rtp_llm.models_py.modules.dsv4.moe.strategies import (
    DeepEPStrategy,
    GroupedFP4Strategy,
    LocalLoopStrategy,
    MegaMoEFusedStrategy,
    MegaMoEStrategy,
    MegaMoEStrategySE,
    MoeCfg,
    _has_fp8_fp4_grouped_kernel,
    select_strategy,
)
from rtp_llm.models_py.modules.dsv4.moe.strategies import base as strategy_base
from rtp_llm.models_py.modules.dsv4.moe.strategies.base import _resolve_forced


_STRATEGY_ENV_NAMES = (
    "DSV4_MOE_STRATEGY",
    "DSV4_USE_MEGA_MOE",
    "DSV4_USE_MEGA_MOE_SE",
    "DSV4_USE_MEGA_MOE_FUSED",
    "DSV4_USE_GROUPED_FP4",
)


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
        self._saved_env = {
            name: os.environ.pop(name, None) for name in _STRATEGY_ENV_NAMES
        }
        # Most tests retain the old routed-only baseline. Tests for automatic
        # SE selection explicitly remove this opt-out.
        os.environ["DSV4_USE_MEGA_MOE_SE"] = "0"
        strategy_base._MEGA_SE_AUTO_FALLBACK_WARNED = False

    def tearDown(self):
        for name, value in self._saved_env.items():
            os.environ.pop(name, None)
            if value is not None:
                os.environ[name] = value

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

    def test_ep_gt1_with_se_disabled_picks_mega(self):
        with mock.patch.object(MegaMoEStrategy, "can_handle", return_value=True):
            self.assertIs(select_strategy(_cfg(ep_size=4)), MegaMoEStrategy)

    def test_ep_gt1_default_picks_mega_se_when_capable(self):
        with _env(DSV4_USE_MEGA_MOE_SE=None), mock.patch.object(
            MegaMoEStrategySE, "can_handle", return_value=True
        ), mock.patch.object(MegaMoEStrategy, "can_handle") as mega_can_handle:
            self.assertIs(select_strategy(_cfg(ep_size=4)), MegaMoEStrategySE)
        mega_can_handle.assert_not_called()

    def test_ep_gt1_default_warns_and_uses_mega_when_se_is_incapable(self):
        with _env(DSV4_USE_MEGA_MOE_SE=None), mock.patch.object(
            MegaMoEStrategySE, "can_handle", return_value=False
        ), mock.patch.object(
            MegaMoEStrategy, "can_handle", return_value=True
        ), mock.patch(
            "rtp_llm.models_py.modules.dsv4.moe.mega_se_buf."
            "_mega_moe_se_unavailable_reason",
            return_value="missing shared-expert API",
        ), mock.patch.object(strategy_base.logging, "warning") as warning:
            self.assertIs(select_strategy(_cfg(ep_size=4)), MegaMoEStrategy)
        warning.assert_called_once()
        self.assertIn("missing shared-expert API", warning.call_args.args)

    def test_ep1_default_never_probes_mega_se(self):
        with _env(DSV4_USE_MEGA_MOE_SE=None), mock.patch.object(
            MegaMoEStrategySE, "can_handle"
        ) as se_can_handle, mock.patch.object(
            GroupedFP4Strategy, "can_handle", return_value=True
        ):
            self.assertIs(select_strategy(_cfg(ep_size=1)), GroupedFP4Strategy)
        se_can_handle.assert_not_called()

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

    def test_explicit_env_strategy_beats_default_se_and_legacy_fused(self):
        with _env(
            DSV4_MOE_STRATEGY="mega",
            DSV4_USE_MEGA_MOE_SE=None,
            DSV4_USE_MEGA_MOE_FUSED="1",
        ), mock.patch.object(
            MegaMoEStrategy, "can_handle", return_value=True
        ), mock.patch.object(
            MegaMoEStrategySE, "can_handle"
        ) as se_can_handle, mock.patch.object(
            MegaMoEFusedStrategy, "can_handle"
        ) as fused_can_handle:
            forced, strict = _resolve_forced(None)
            self.assertEqual((forced, strict), ("mega", True))
            self.assertIs(
                select_strategy(_cfg(ep_size=4), forced=forced, strict=strict),
                MegaMoEStrategy,
            )
        se_can_handle.assert_not_called()
        fused_can_handle.assert_not_called()

    def test_family_disable_rejects_named_mega_variants(self):
        variants = (
            ("mega", MegaMoEStrategy),
            ("mega_se", MegaMoEStrategySE),
            ("mega_fused", MegaMoEFusedStrategy),
        )
        for strategy_name, strategy_cls in variants:
            for ep_size in (1, 2):
                with self.subTest(strategy=strategy_name, ep_size=ep_size), _env(
                    DSV4_MOE_STRATEGY=strategy_name,
                    DSV4_USE_MEGA_MOE="0",
                ), mock.patch.object(strategy_cls, "can_handle") as can_handle:
                    forced, strict = _resolve_forced(None)
                    with self.assertRaises(RuntimeError) as cm:
                        select_strategy(
                            _cfg(ep_size=ep_size), forced=forced, strict=strict
                        )
                can_handle.assert_not_called()
                self.assertIn(
                    "DSV4_USE_MEGA_MOE=0 disables the Mega MoE family",
                    str(cm.exception),
                )
                self.assertIn(strategy_name, str(cm.exception))

    def test_family_disable_rejects_constructor_forced_mega_variants(self):
        variants = (
            ("mega", MegaMoEStrategy),
            ("mega_se", MegaMoEStrategySE),
            ("mega_fused", MegaMoEFusedStrategy),
        )
        for strategy_name, strategy_cls in variants:
            with self.subTest(strategy=strategy_name), _env(
                DSV4_USE_MEGA_MOE="0"
            ), mock.patch.object(strategy_cls, "can_handle") as can_handle:
                with self.assertRaises(RuntimeError) as cm:
                    select_strategy(
                        _cfg(ep_size=2), forced=strategy_name, strict=True
                    )
            can_handle.assert_not_called()
            self.assertIn(
                "DSV4_USE_MEGA_MOE=0 disables the Mega MoE family",
                str(cm.exception),
            )

    def test_family_disable_allows_named_non_mega_on_ep1(self):
        with _env(
            DSV4_MOE_STRATEGY="local_loop",
            DSV4_USE_MEGA_MOE="0",
        ), mock.patch.object(LocalLoopStrategy, "can_handle", return_value=True):
            forced, strict = _resolve_forced(None)
            self.assertIs(
                select_strategy(_cfg(ep_size=1), forced=forced, strict=strict),
                LocalLoopStrategy,
            )

    def test_explicit_variant_does_not_require_legacy_opt_in(self):
        variants = (
            (
                "mega_se",
                MegaMoEStrategySE,
                "rtp_llm.models_py.modules.dsv4.moe.strategies.mega_se."
                "_mega_moe_se_available",
            ),
            (
                "mega_fused",
                MegaMoEFusedStrategy,
                "rtp_llm.models_py.modules.dsv4.moe.strategies.mega_fused."
                "_mega_moe_fused_available",
            ),
        )
        for strategy_name, strategy_cls, availability_probe in variants:
            for source in ("env", "ctor"):
                with self.subTest(strategy=strategy_name, source=source), _env(
                    DSV4_MOE_STRATEGY=strategy_name if source == "env" else None,
                    DSV4_USE_MEGA_MOE_SE=(
                        None
                        if strategy_name == "mega_fused" and source == "ctor"
                        else "0"
                    ),
                    DSV4_USE_MEGA_MOE_FUSED="0",
                ), mock.patch(availability_probe, return_value=True):
                    forced, strict = _resolve_forced(
                        strategy_name if source == "ctor" else None
                    )
                    self.assertIs(
                        select_strategy(
                            _cfg(ep_size=2), forced=forced, strict=strict
                        ),
                        strategy_cls,
                    )

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
        ), mock.patch(
            "rtp_llm.models_py.modules.dsv4.moe.mega_se_buf."
            "_mega_moe_se_disabled_or_unavailable_reason",
            return_value="torch.distributed is not initialized",
        ):
            forced, strict = _resolve_forced(None)
            with self.assertRaises(RuntimeError) as cm:
                select_strategy(_cfg(ep_size=2), forced=forced, strict=strict)
        self.assertIn("Forced MoE strategy 'mega_se'", str(cm.exception))
        self.assertIn("torch.distributed is not initialized", str(cm.exception))

    def test_mega_moe_se_explicit_enable_is_invalid_on_ep1(self):
        with _env(DSV4_USE_MEGA_MOE_SE="1"):
            forced, strict = _resolve_forced(None)
            with self.assertRaisesRegex(RuntimeError, "requires ep_size > 1"):
                select_strategy(_cfg(ep_size=1), forced=forced, strict=strict)

    def test_default_or_explicit_mega_moe_se_and_old_fused_conflict(self):
        for se_value in (None, "1"):
            with self.subTest(se_value=se_value), _env(
                DSV4_USE_MEGA_MOE_SE=se_value,
                DSV4_USE_MEGA_MOE_FUSED="1",
            ):
                forced, strict = _resolve_forced(None)
                with self.assertRaises(RuntimeError) as cm:
                    select_strategy(_cfg(ep_size=2), forced=forced, strict=strict)
            self.assertIn("set DSV4_USE_MEGA_MOE_SE=0", str(cm.exception))

    def test_explicit_se_zero_allows_old_fused(self):
        with _env(
            DSV4_USE_MEGA_MOE_SE="0",
            DSV4_USE_MEGA_MOE_FUSED="1",
        ), mock.patch.object(
            MegaMoEFusedStrategy, "can_handle", return_value=True
        ):
            self.assertIs(select_strategy(_cfg(ep_size=2)), MegaMoEFusedStrategy)

    def test_family_disable_rejects_old_fused(self):
        for ep_size in (1, 2):
            with self.subTest(ep_size=ep_size), _env(
                DSV4_USE_MEGA_MOE="0",
                DSV4_USE_MEGA_MOE_SE="0",
                DSV4_USE_MEGA_MOE_FUSED="1",
            ):
                with self.assertRaises(RuntimeError) as cm:
                    select_strategy(_cfg(ep_size=ep_size))
            self.assertIn(
                "DSV4_USE_MEGA_MOE=0 disables the Mega MoE family",
                str(cm.exception),
            )
            self.assertIn("DSV4_USE_MEGA_MOE_FUSED=1", str(cm.exception))

    def test_old_fused_is_invalid_on_ep1(self):
        with _env(
            DSV4_USE_MEGA_MOE_SE="0",
            DSV4_USE_MEGA_MOE_FUSED="1",
        ):
            with self.assertRaisesRegex(RuntimeError, "requires ep_size > 1"):
                select_strategy(_cfg(ep_size=1))

    def test_grouped_fp4_is_invalid_on_ep_topology(self):
        for se_value in (None, "0"):
            with self.subTest(se_value=se_value), _env(
                DSV4_USE_MEGA_MOE_SE=se_value,
                DSV4_USE_GROUPED_FP4="1",
            ):
                forced, strict = _resolve_forced(None)
                with self.assertRaisesRegex(
                    RuntimeError, "incompatible with ep_size > 1"
                ):
                    select_strategy(_cfg(ep_size=2), forced=forced, strict=strict)

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

    def test_legacy_negation_ep_gt1_raises_before_mega_probes(self):
        with _env(
            DSV4_USE_MEGA_MOE="0",
            DSV4_USE_MEGA_MOE_SE=None,
        ), mock.patch.object(
            MegaMoEStrategySE, "can_handle"
        ) as se_can_handle, mock.patch.object(
            MegaMoEStrategy, "can_handle"
        ) as mega_can_handle:
            with self.assertRaises(RuntimeError) as cm:
                select_strategy(_cfg(ep_size=4))
        se_can_handle.assert_not_called()
        mega_can_handle.assert_not_called()
        self.assertIn(
            "DSV4_USE_MEGA_MOE=0 disables the Mega MoE family", str(cm.exception)
        )

    def test_legacy_negation_ep1_skips_all_mega_probes(self):
        with _env(
            DSV4_USE_MEGA_MOE="0",
            DSV4_USE_MEGA_MOE_SE=None,
        ), mock.patch.object(
            MegaMoEStrategySE, "can_handle"
        ) as se_can_handle, mock.patch.object(
            MegaMoEFusedStrategy, "can_handle"
        ) as fused_can_handle, mock.patch.object(
            MegaMoEStrategy, "can_handle"
        ) as mega_can_handle, mock.patch.object(
            GroupedFP4Strategy, "can_handle", return_value=True
        ):
            self.assertIs(select_strategy(_cfg(ep_size=1)), GroupedFP4Strategy)
        se_can_handle.assert_not_called()
        fused_can_handle.assert_not_called()
        mega_can_handle.assert_not_called()

    def test_explicit_se_enable_conflicts_with_mega_family_disable(self):
        with _env(DSV4_USE_MEGA_MOE="0", DSV4_USE_MEGA_MOE_SE="1"):
            with self.assertRaisesRegex(RuntimeError, "conflicts"):
                _resolve_forced(None)

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

    def test_mega_se_capability_requires_distributed_initialization(self):
        fake_deep_gemm = types.SimpleNamespace(fp8_fp4_mega_moe=object())
        with _env(DSV4_USE_MEGA_MOE_SE=None), mock.patch.dict(
            sys.modules, {"deep_gemm": fake_deep_gemm}
        ), mock.patch("torch.distributed.is_initialized", return_value=False):
            self.assertFalse(MegaMoEStrategySE.can_handle(_cfg(ep_size=2)))
            self.assertEqual(
                _mega_moe_se_unavailable_reason(),
                "torch.distributed is not initialized",
            )

    def test_mega_se_capability_accepts_initialized_supported_runtime(self):
        def fp8_fp4_mega_moe(
            *args,
            shared_l1_weights=None,
            shared_l2_weights=None,
            shared_recipe=None,
        ):
            pass

        def get_symm_buffer_for_mega_moe(*args, num_shared_experts=None):
            pass

        fake_deep_gemm = types.SimpleNamespace(
            fp8_fp4_mega_moe=fp8_fp4_mega_moe,
            get_symm_buffer_for_mega_moe=get_symm_buffer_for_mega_moe,
            get_block_m_for_mega_moe=object(),
            transform_weights_for_mega_moe=object(),
            transform_sf_into_required_layout=object(),
        )
        with _env(DSV4_USE_MEGA_MOE_SE=None), mock.patch.dict(
            sys.modules, {"deep_gemm": fake_deep_gemm}
        ), mock.patch(
            "torch.distributed.is_initialized", return_value=True
        ), mock.patch(
            "torch.distributed.get_world_size", return_value=2
        ), mock.patch(
            "torch.cuda.is_available", return_value=True
        ), mock.patch(
            "torch.cuda.get_device_capability", return_value=(10, 0)
        ):
            self.assertIsNone(_mega_moe_se_unavailable_reason())
            self.assertTrue(MegaMoEStrategySE.can_handle(_cfg(ep_size=2)))


if __name__ == "__main__":
    unittest.main()
