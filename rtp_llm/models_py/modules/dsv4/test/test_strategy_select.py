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

import torch

from rtp_llm.device.device_type import DeviceType
from rtp_llm.models_py.distributed import deepep_wrapper

# Importing strategies populates the registry via ``register_strategy``.
from rtp_llm.models_py.modules.dsv4.moe.strategies import (
    DeepEPStrategy,
    GroupedFP4Strategy,
    LocalLoopStrategy,
    MegaMoEFusedStrategy,
    MegaMoEStrategy,
    MegaMoEStrategySE,
    MoeCfg,
    Sm120FusedMoeStrategy,
    _has_fp8_fp4_grouped_kernel,
)
from rtp_llm.models_py.modules.dsv4.moe.strategies import (
    grouped_fp4 as grouped_fp4_module,
)
from rtp_llm.models_py.modules.dsv4.moe.strategies import select_strategy
from rtp_llm.models_py.modules.dsv4.moe.strategies.base import _resolve_forced
from rtp_llm.utils.model_weight import W


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

    def test_grouped_kernel_probe_supports_sm100_and_sm120(self):
        fake_flashinfer = types.ModuleType("flashinfer")
        fake_flashinfer.block_scale_interleave = lambda value: value
        fake_flashinfer.mxfp8_quantize = lambda value, **_: (value, value)
        fake_gemm = types.ModuleType("flashinfer.gemm")
        fake_gemm.group_gemm_mxfp4_nt_groupwise = lambda *args, **kwargs: None
        fake_fused_moe = types.ModuleType("flashinfer.fused_moe")
        fake_fused_moe.cutlass_fused_moe = lambda *args, **kwargs: None
        fake_fused_moe.cutlass_fused_moe_workspace_size = lambda *args, **kwargs: 1
        fake_fused_moe_core = types.ModuleType("flashinfer.fused_moe.core")
        fake_fused_moe_core.ActivationType = types.SimpleNamespace(Swiglu=object())
        fake_deep_gemm = types.SimpleNamespace(
            m_grouped_fp8_fp4_gemm_nt_contiguous=object(),
            get_mk_alignment_for_contiguous_layout=lambda: (128, 128),
        )
        with mock.patch.dict(
            sys.modules,
            {
                "flashinfer": fake_flashinfer,
                "flashinfer.gemm": fake_gemm,
                "flashinfer.fused_moe": fake_fused_moe,
                "flashinfer.fused_moe.core": fake_fused_moe_core,
            },
        ), mock.patch(
            "rtp_llm.models_py.modules.dsv4.moe.strategies.grouped_fp4."
            "torch.cuda.is_available",
            return_value=True,
        ), mock.patch(
            "rtp_llm.models_py.modules.dsv4.moe.strategies.grouped_fp4."
            "torch.cuda.get_device_capability",
            return_value=(12, 0),
        ), mock.patch(
            "rtp_llm.models_py.modules.dsv4.moe.strategies.grouped_fp4.is_sm120",
            return_value=True,
        ):
            self.assertTrue(_has_fp8_fp4_grouped_kernel())

        with mock.patch.dict(sys.modules, {"deep_gemm": fake_deep_gemm}), mock.patch(
            "rtp_llm.models_py.modules.dsv4.moe.strategies.grouped_fp4."
            "torch.cuda.is_available",
            return_value=True,
        ), mock.patch(
            "rtp_llm.models_py.modules.dsv4.moe.strategies.grouped_fp4."
            "torch.cuda.get_device_capability",
            return_value=(10, 0),
        ), mock.patch(
            "rtp_llm.models_py.modules.dsv4.moe.strategies.grouped_fp4.is_sm120",
            return_value=False,
        ):
            self.assertTrue(_has_fp8_fp4_grouped_kernel())

    def test_sm120_grouped_probe_requires_cuda_graph_apis(self):
        fake_flashinfer = types.ModuleType("flashinfer")
        fake_flashinfer.block_scale_interleave = lambda value: value
        fake_flashinfer.mxfp8_quantize = lambda value, **_: (value, value)
        fake_gemm = types.ModuleType("flashinfer.gemm")
        fake_gemm.group_gemm_mxfp4_nt_groupwise = lambda *args, **kwargs: None
        fake_fused_moe = types.ModuleType("flashinfer.fused_moe")
        # Deliberately omit cutlass_fused_moe: eager grouped GEMM exists, but
        # graph capture must be rejected during strategy selection.
        fake_fused_moe.cutlass_fused_moe_workspace_size = lambda *args, **kwargs: 1
        fake_core = types.ModuleType("flashinfer.fused_moe.core")
        fake_core.ActivationType = types.SimpleNamespace(Swiglu=object())
        with mock.patch.dict(
            sys.modules,
            {
                "flashinfer": fake_flashinfer,
                "flashinfer.gemm": fake_gemm,
                "flashinfer.fused_moe": fake_fused_moe,
                "flashinfer.fused_moe.core": fake_core,
            },
        ), mock.patch(
            "rtp_llm.models_py.modules.dsv4.moe.strategies.grouped_fp4."
            "torch.cuda.is_available",
            return_value=True,
        ), mock.patch(
            "rtp_llm.models_py.modules.dsv4.moe.strategies.grouped_fp4.is_sm120",
            return_value=True,
        ):
            self.assertFalse(_has_fp8_fp4_grouped_kernel())

    def test_ep1_no_grouped_falls_to_local(self):
        with mock.patch.object(
            GroupedFP4Strategy, "can_handle", return_value=False
        ), mock.patch.object(
            MegaMoEStrategy, "can_handle", return_value=False
        ), mock.patch.object(
            DeepEPStrategy, "can_handle", return_value=False
        ):
            self.assertIs(select_strategy(_cfg(ep_size=1)), LocalLoopStrategy)

    def test_sm120_local_loop_setup_never_calls_deepgemm_packer(self):
        cfg = MoeCfg(
            layer_id=0,
            dim=8,
            moe_inter_dim=4,
            n_routed_experts=2,
            n_activated_experts=1,
            swiglu_limit=0.0,
            ep_size=1,
            ep_rank=0,
            n_local_experts=2,
            local_expert_start=0,
            local_expert_end=2,
            max_tokens_per_rank=4,
        )
        strategy = LocalLoopStrategy(cfg)
        weights = {
            W.v4_routed_w1_w: torch.zeros(2, 1),
            W.v4_routed_w1_s: torch.ones(2, 1),
            W.v4_routed_w2_w: torch.zeros(2, 1),
            W.v4_routed_w2_s: torch.ones(2, 1),
            W.v4_routed_w3_w: torch.zeros(2, 1),
            W.v4_routed_w3_s: torch.ones(2, 1),
        }

        class _FakeExpert(torch.nn.Module):
            def __init__(self, *args, expert_weights, **kwargs):
                super().__init__()
                self.expert_weights = expert_weights

        with mock.patch(
            "rtp_llm.models_py.modules.dsv4.moe.strategies.local_loop."
            "_uses_sm120_local_loop",
            return_value=True,
        ), mock.patch(
            "rtp_llm.models_py.modules.dsv4.moe.strategies.local_loop."
            "prepare_fp4_weight_scale_for_deepgemm",
            side_effect=AssertionError("DeepGEMM packer must not run on SM120"),
        ), mock.patch(
            "rtp_llm.models_py.modules.dsv4.moe.strategies.local_loop.Expert",
            _FakeExpert,
        ):
            strategy.setup_weights(weights)

        self.assertFalse(strategy._deepgemm_topk_available)
        self.assertIsNone(strategy._W1_s_gemm)
        self.assertIsNone(strategy._W2_s_gemm)
        self.assertIsNone(strategy._W3_s_gemm)
        for expert in strategy.experts:
            self.assertIsNone(expert.expert_weights["w1_s_gemm"])
            self.assertIsNone(expert.expert_weights["w2_s_gemm"])
            self.assertIsNone(expert.expert_weights["w3_s_gemm"])

    def test_sm120_local_loop_rejects_cuda_graph_capture(self):
        strategy = LocalLoopStrategy(_cfg(ep_size=1))
        strategy._deepgemm_topk_available = False
        with mock.patch(
            "rtp_llm.models_py.modules.dsv4.moe.strategies.local_loop."
            "torch.cuda.is_current_stream_capturing",
            return_value=True,
        ):
            with self.assertRaisesRegex(RuntimeError, "eager-only"):
                strategy.forward_local_range(
                    torch.zeros(1, 1),
                    torch.ones(1, 1),
                    torch.zeros(1, 1, dtype=torch.long),
                    0,
                    1,
                )

    def test_sm120_workspace_growth_reuses_capacity_without_dropping_old_buffer(self):
        strategy = GroupedFP4Strategy(_cfg(ep_size=1))
        fake_fused_moe = types.ModuleType("flashinfer.fused_moe")
        fake_fused_moe.cutlass_fused_moe_workspace_size = (
            lambda tokens, *_args, **_kwargs: tokens * 10
        )
        fake_core = types.ModuleType("flashinfer.fused_moe.core")
        fake_core.ActivationType = types.SimpleNamespace(Swiglu=object())

        grouped_fp4_module._SM120_FUSED_MOE_WORKSPACES.clear()
        try:
            with mock.patch.dict(
                sys.modules,
                {
                    "flashinfer.fused_moe": fake_fused_moe,
                    "flashinfer.fused_moe.core": fake_core,
                },
            ):
                first = strategy._get_sm120_fused_moe_workspace(torch.device("cpu"), 3)
                grown = strategy._get_sm120_fused_moe_workspace(torch.device("cpu"), 9)
                reused = strategy._get_sm120_fused_moe_workspace(torch.device("cpu"), 2)

            self.assertNotEqual(first.data_ptr(), grown.data_ptr())
            self.assertEqual(first.data_ptr(), reused.data_ptr())
            generations = next(
                iter(grouped_fp4_module._SM120_FUSED_MOE_WORKSPACES.values())
            )
            self.assertEqual([capacity for capacity, _ in generations], [4, 16])
        finally:
            grouped_fp4_module._SM120_FUSED_MOE_WORKSPACES.clear()

    def test_ep_gt1_with_mega_picks_mega(self):
        with mock.patch.object(MegaMoEStrategy, "can_handle", return_value=True):
            self.assertIs(select_strategy(_cfg(ep_size=4)), MegaMoEStrategy)

    def test_ep_gt1_default_stays_mega_when_se_is_capable(self):
        with mock.patch.object(
            MegaMoEStrategy, "can_handle", return_value=True
        ), mock.patch.object(MegaMoEStrategySE, "can_handle", return_value=True):
            self.assertIs(select_strategy(_cfg(ep_size=4)), MegaMoEStrategy)

    def test_mega_child_strategies_share_exact_architecture_gate(self):
        with mock.patch.object(
            MegaMoEStrategy, "_architecture_supported", return_value=False
        ), mock.patch(
            "rtp_llm.models_py.modules.dsv4.moe.strategies.mega_se."
            "_mega_moe_se_enabled",
            return_value=True,
        ), mock.patch(
            "rtp_llm.models_py.modules.dsv4.moe.strategies.mega_fused."
            "_mega_moe_fused_enabled",
            return_value=True,
        ):
            self.assertFalse(MegaMoEStrategySE.can_handle(_cfg(ep_size=4)))
            self.assertFalse(MegaMoEFusedStrategy.can_handle(_cfg(ep_size=4)))

    def test_accl_ep_policy_is_device_type_aware(self):
        cases = (
            (DeviceType.Cuda, False, True),
            (DeviceType.Cuda, True, False),
            (DeviceType.ROCm, False, False),
            (DeviceType.Ppu, False, True),
            (DeviceType.Cpu, False, False),
        )
        for device_type, sm12x, expected in cases:
            with self.subTest(device_type=device_type, sm12x=sm12x), mock.patch.object(
                deepep_wrapper, "get_device_type", return_value=device_type
            ), mock.patch.object(deepep_wrapper, "is_sm12x", return_value=sm12x):
                self.assertEqual(deepep_wrapper.use_accl_ep(), expected)

    def test_explicit_unsupported_deepep_request_fails_closed(self):
        with mock.patch.object(
            deepep_wrapper.DeepEPWrapper, "supported", return_value=False
        ):
            with self.assertRaisesRegex(RuntimeError, "was requested"):
                deepep_wrapper.init_deepep_wrapper(None, None)

    def test_ep_gt1_no_mega_fails_instead_of_silently_using_deepep(self):
        with mock.patch.object(
            MegaMoEStrategy, "can_handle", return_value=False
        ), mock.patch.object(
            Sm120FusedMoeStrategy, "can_handle", return_value=False
        ), mock.patch.object(
            DeepEPStrategy, "can_handle", return_value=True
        ), mock.patch(
            "rtp_llm.models_py.modules.dsv4.moe.mega_buf."
            "_mega_moe_disabled_or_unavailable_reason",
            return_value="Mega unavailable in test",
        ):
            with self.assertRaisesRegex(RuntimeError, "Mega unavailable in test"):
                select_strategy(_cfg(ep_size=4))

    def test_sm120_ep_gt1_uses_explicit_fused_moe_strategy(self):
        with mock.patch.object(
            MegaMoEStrategy, "can_handle", return_value=False
        ), mock.patch(
            "rtp_llm.models_py.modules.dsv4.moe.strategies.sm120_fused_moe.is_sm120",
            return_value=True,
        ):
            self.assertIs(
                select_strategy(_cfg(ep_size=4)),
                Sm120FusedMoeStrategy,
            )

    def test_sm120_strategy_rejects_other_sm12x_devices(self):
        with mock.patch(
            "rtp_llm.models_py.modules.dsv4.moe.strategies.sm120_fused_moe.is_sm120",
            return_value=False,
        ):
            self.assertFalse(Sm120FusedMoeStrategy.can_handle(_cfg(ep_size=4)))

    def test_deepep_probe_rejects_sm12x_without_mocking_strategy_result(self):
        with mock.patch(
            "rtp_llm.models_py.modules.dsv4.moe.strategies.deepep.is_sm12x",
            return_value=True,
        ):
            self.assertFalse(DeepEPStrategy.can_handle(_cfg(ep_size=4)))

        with mock.patch(
            "rtp_llm.models_py.modules.dsv4.moe.strategies.deepep.is_sm12x",
            return_value=False,
        ):
            self.assertTrue(DeepEPStrategy.can_handle(_cfg(ep_size=4)))

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

    def test_forced_ep_gt1_deepep_is_allowed(self):
        with mock.patch.object(DeepEPStrategy, "can_handle", return_value=True):
            self.assertIs(
                select_strategy(_cfg(ep_size=4), forced="deepep"), DeepEPStrategy
            )

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
        with _env(DSV4_USE_MEGA_MOE="0"):
            self.assertEqual(_resolve_forced(None), (None, False))

    def test_legacy_negation_ep_gt1_fails_closed(self):
        with _env(DSV4_USE_MEGA_MOE="0"), mock.patch.object(
            Sm120FusedMoeStrategy, "can_handle", return_value=False
        ), mock.patch.object(DeepEPStrategy, "can_handle", return_value=True):
            with self.assertRaisesRegex(RuntimeError, "DSV4_USE_MEGA_MOE=0"):
                select_strategy(_cfg(ep_size=4))

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
