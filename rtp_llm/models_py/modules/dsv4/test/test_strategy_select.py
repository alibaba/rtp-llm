"""PPU selection stays explicit after removal of the DSv4 global registry."""

import unittest
from unittest.mock import patch

from rtp_llm.platforms.ppu.models.dsv4.ppu_provider import M890PDsv4Provider
from rtp_llm.platforms.ppu.models.dsv4.ppu_legacy_deepep import PpuLegacyDeepEPStrategy


class PpuMoeSelectionTest(unittest.TestCase):
    def test_shared_expert_allows_ep_to_require_sglang_contract(self):
        provider = M890PDsv4Provider({})
        with patch(
            "rtp_llm.platforms.ppu.models.dsv4.ppu_shared_expert.PpuSharedExpert"
        ) as ctor:
            provider.build_shared_expert(4, 8, sglang_moe=True)

        self.assertTrue(ctor.call_args.kwargs["sglang_moe"])

    def test_verified_platform_can_force_deepep(self):
        provider = M890PDsv4Provider({"DSV4_PPU_GROUPED_FP4": "1"})
        for tp, ep in ((1, 2), (2, 4), (1, 8), (1, 16)):
            with self.subTest(tp=tp, ep=ep):
                self._check_deepep_selection(provider, tp, ep)

    def _check_deepep_selection(self, provider, tp, ep):
        with patch("rtp_llm.platforms.ppu.models.dsv4.ppu_ep_moe.PpuEPMoE") as ctor:
            result = provider.build_moe(None, tp_size=tp, ep_size=ep, strategy="deepep")
        self.assertIs(result, ctor.return_value)
        self.assertIs(ctor.call_args.kwargs["platform_provider"], provider)
        self.assertIs(ctor.call_args.kwargs["strategy_type"], PpuLegacyDeepEPStrategy)
        self.assertEqual(
            ctor.call_args.kwargs["strategy_kwargs"]["options"],
            provider.execution_options,
        )

    def test_public_auto_default_selects_explicit_legacy_ppu_implementation(self):
        provider = M890PDsv4Provider({"DSV4_PPU_GROUPED_FP4": "1"})
        with patch("rtp_llm.platforms.ppu.models.dsv4.ppu_ep_moe.PpuEPMoE") as ctor:
            provider.build_moe(None, tp_size=1, ep_size=8, strategy="auto")
        self.assertIs(ctor.call_args.kwargs["strategy_type"], PpuLegacyDeepEPStrategy)

    def test_unselected_ep8_does_not_fall_back_to_cuda_factory(self):
        provider = M890PDsv4Provider({})
        with self.assertRaises(RuntimeError):
            provider.build_moe(
                lambda **kwargs: self.fail("CUDA fallback"), tp_size=1, ep_size=8
            )

    def test_ppu_tp_group_selects_the_complete_tp_module(self):
        provider = M890PDsv4Provider({})
        for tp in (1, 2, 4, 8):
            with self.subTest(tp=tp), patch(
                "rtp_llm.platforms.ppu.models.dsv4.ppu_tp_moe.PpuTPMoE"
            ) as ctor:
                result = provider.build_moe(None, tp_size=tp, tp_rank=tp - 1, ep_size=1)
                self.assertIs(result, ctor.return_value)
                self.assertEqual(ctor.call_args.kwargs["tp_rank"], tp - 1)
                self.assertIs(ctor.call_args.kwargs["platform_provider"], provider)


if __name__ == "__main__":
    unittest.main()
