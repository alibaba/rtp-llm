"""CPU-only contracts for the SM90 fused MegaMoE strategy wiring.

The kernel itself needs eight Hopper GPUs, but the opt-in gate, the geometry
validation that mirrors ``SymmBufferSM90``, and the activation-scale math are
all host-side and must stay covered by ordinary CI.
"""

import os
import unittest
from unittest import mock

import torch

from rtp_llm.models_py.modules.dsv41.moe.mega_sm90_buf import (
    _sm90_mega_moe_enabled,
    _sm90_mega_moe_requested,
    _sm90_shape_supported,
)
from rtp_llm.models_py.modules.dsv41.moe.mega_sm90_input_packer import (
    SM90_ACT_SF_GRAN_K,
    per_token_cast_to_fp8_fp32_scale,
)
from rtp_llm.models_py.modules.dsv41.moe.strategies import SM90MegaMoEStrategy


class Sm90MegaMoEGateTest(unittest.TestCase):
    def test_opt_in_env_is_required(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DSV4_USE_MEGA_MOE_SM90", None)
            self.assertFalse(_sm90_mega_moe_requested())
            # Even with the switch off, the enabled gate must stay closed.
            self.assertFalse(_sm90_mega_moe_enabled())
        for value in ("1", "true", "YES", "on"):
            with mock.patch.dict(os.environ, {"DSV4_USE_MEGA_MOE_SM90": value}):
                self.assertTrue(_sm90_mega_moe_requested())
        for value in ("0", "", "false", "off"):
            with mock.patch.dict(os.environ, {"DSV4_USE_MEGA_MOE_SM90": value}):
                self.assertFalse(_sm90_mega_moe_requested())

    def test_switch_alone_does_not_enable_without_the_kernel(self):
        # This host has no SM90 device, so the kernel probe must keep the gate
        # closed even when the operator sets the switch.
        with mock.patch.dict(os.environ, {"DSV4_USE_MEGA_MOE_SM90": "1"}):
            self.assertFalse(_sm90_mega_moe_enabled())

    def test_shape_gate_matches_the_kernel_limits(self):
        # V4.1 Flash geometry: hidden 5120, inter 2304, 384 experts, EP8, topk 6.
        self.assertTrue(_sm90_shape_supported(5120, 2304, 384, 8, 6))
        # hidden must be a multiple of 512 and at most 8192
        self.assertFalse(_sm90_shape_supported(4600, 2304, 384, 8, 6))
        self.assertFalse(_sm90_shape_supported(8704, 2304, 384, 8, 6))
        self.assertTrue(_sm90_shape_supported(8192, 2304, 384, 8, 6))
        # intermediate must be a multiple of 256 and at most 4096
        self.assertFalse(_sm90_shape_supported(5120, 2300, 384, 8, 6))
        self.assertFalse(_sm90_shape_supported(4352, 2304, 384, 8, 6))
        self.assertTrue(_sm90_shape_supported(5120, 4096, 384, 8, 6))
        # experts must divide the group; topk must be in [1, min(32, experts)]
        self.assertFalse(_sm90_shape_supported(5120, 2304, 385, 8, 6))
        self.assertFalse(_sm90_shape_supported(5120, 2304, 384, 8, 33))
        self.assertFalse(_sm90_shape_supported(5120, 2304, 384, 33, 6))


class Sm90MegaMoEStrategyTest(unittest.TestCase):
    def test_sm90_auto_selects_native_without_legacy_opt_in(self):
        from rtp_llm.models_py.modules.dsv41.moe.strategies.base import select_strategy

        cfg = mock.Mock()
        base = "rtp_llm.models_py.modules.dsv41.moe.strategies.base"
        with (
            mock.patch(f"{base}.is_cuda", return_value=True),
            mock.patch("torch.cuda.get_device_capability", return_value=(9, 0)),
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch.object(SM90MegaMoEStrategy, "can_handle", return_value=True) as gate,
        ):
            self.assertIs(select_strategy(cfg), SM90MegaMoEStrategy)
            gate.assert_called_once_with(cfg, explicit=True)
            with self.assertRaisesRegex(RuntimeError, "requires mega_sm90"):
                select_strategy(cfg, forced="sm90_flashinfer")

    def test_native_unavailable_does_not_restore_retired_backend(self):
        from rtp_llm.models_py.modules.dsv41.moe.strategies.base import select_strategy

        base = "rtp_llm.models_py.modules.dsv41.moe.strategies.base"
        mega = "rtp_llm.models_py.modules.dsv41.moe.strategies.mega_sm90"
        with (
            mock.patch(f"{base}.is_cuda", return_value=True),
            mock.patch("torch.cuda.get_device_capability", return_value=(9, 0)),
            mock.patch.object(SM90MegaMoEStrategy, "can_handle", return_value=False),
            mock.patch(f"{mega}.sm90_mega_moe_unavailable_reason", return_value="missing kernel"),
        ):
            with self.assertRaisesRegex(RuntimeError, "missing kernel"):
                select_strategy(mock.Mock())

    def test_registered_as_a_routed_only_strategy(self):
        self.assertEqual(SM90MegaMoEStrategy.name, "mega_sm90")
        # SymmBufferSM90 has no shared-expert fusion, so the MoE layer must
        # keep running its standalone shared expert.
        self.assertFalse(SM90MegaMoEStrategy.routed_includes_shared)

    def test_cannot_handle_without_the_opt_in(self):
        cfg = mock.Mock(
            dim=5120,
            moe_inter_dim=2304,
            n_routed_experts=384,
            ep_size=8,
            n_activated_experts=6,
        )
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DSV4_USE_MEGA_MOE_SM90", None)
            self.assertFalse(SM90MegaMoEStrategy.can_handle(cfg))

    def test_explicit_selection_counts_as_the_opt_in(self):
        cfg = mock.Mock(
            dim=5120,
            moe_inter_dim=2304,
            n_routed_experts=384,
            ep_size=8,
            n_activated_experts=6,
        )
        module = "rtp_llm.models_py.modules.dsv41.moe.strategies.mega_sm90"
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DSV4_USE_MEGA_MOE_SM90", None)
            with mock.patch(f"{module}._sm90_mega_moe_available", return_value=True):
                # The automatic path still needs the switch ...
                self.assertFalse(SM90MegaMoEStrategy.can_handle(cfg))
                # ... while naming the strategy is itself the opt-in.
                self.assertTrue(SM90MegaMoEStrategy.can_handle(cfg, explicit=True))
            # A missing kernel keeps even the explicit path closed.
            with mock.patch(f"{module}._sm90_mega_moe_available", return_value=False):
                self.assertFalse(SM90MegaMoEStrategy.can_handle(cfg, explicit=True))


class Sm90MegaMoEPackerTest(unittest.TestCase):
    def test_scale_is_amax_over_448_per_128_columns(self):
        torch.manual_seed(0)
        x = torch.randn(4, 512, dtype=torch.bfloat16)
        fp8, scale = per_token_cast_to_fp8_fp32_scale(x)
        self.assertEqual(tuple(fp8.shape), tuple(x.shape))
        self.assertEqual(fp8.dtype, torch.float8_e4m3fn)
        self.assertEqual(tuple(scale.shape), (4, 512 // SM90_ACT_SF_GRAN_K))
        self.assertEqual(scale.dtype, torch.float32)
        expected = (
            x.view(4, 512 // SM90_ACT_SF_GRAN_K, SM90_ACT_SF_GRAN_K)
            .abs()
            .float()
            .amax(dim=2)
            / 448.0
        )
        torch.testing.assert_close(scale, expected, rtol=0, atol=0)

    def test_round_trip_stays_within_fp8_relative_error(self):
        # E4M3 keeps three mantissa bits, so one step is 2**-3 of the value and
        # rounding stays within half a step (2**-4) of it.
        torch.manual_seed(1)
        x = torch.randn(8, 512, dtype=torch.bfloat16)
        fp8, scale = per_token_cast_to_fp8_fp32_scale(x)
        dequant = (
            fp8.float()
            .view(8, 512 // SM90_ACT_SF_GRAN_K, SM90_ACT_SF_GRAN_K)
            * scale.unsqueeze(2)
        ).view(8, 512)
        reference = x.float()
        relative = (dequant - reference).abs() / reference.abs().clamp_min(1e-3)
        self.assertLess(relative.max().item(), 0.0625)

    def test_columns_are_padded_to_the_scale_granularity(self):
        # An odd width proves the padded view still returns the input columns.
        x = torch.randn(2, 200, dtype=torch.bfloat16)
        fp8, scale = per_token_cast_to_fp8_fp32_scale(x)
        self.assertEqual(tuple(fp8.shape), (2, 200))
        self.assertEqual(tuple(scale.shape), (2, 2))


if __name__ == "__main__":
    unittest.main()
