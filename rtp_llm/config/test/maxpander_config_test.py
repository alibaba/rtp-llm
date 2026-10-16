import os
import pickle
import unittest
from dataclasses import FrozenInstanceError
from unittest.mock import patch

from rtp_llm.config.maxpander_config import MaxPanderConfig, MaxPanderRuntimeConfig


class MaxPanderConfigTest(unittest.TestCase):
    def test_default_passes_and_explicit_override(self):
        self.assertEqual(MaxPanderConfig().wf_passes, 2)
        self.assertEqual(MaxPanderRuntimeConfig().wf_passes, 2)
        for passes in (2, 8):
            requested = (
                MaxPanderConfig(enable=True)
                if passes == 2
                else MaxPanderConfig(enable=True, wf_passes=passes)
            )
            self.assertEqual(requested.resolve("PREFILL").wf_passes, passes)
            self.assertFalse(requested.resolve("DECODE").enabled)

    def test_prefill_only_and_immutable_snapshot(self):
        requested = MaxPanderConfig(enable=True, seed=17, wf_passes=4)
        prefill = requested.resolve("PREFILL")
        decode = requested.resolve("DECODE")
        self.assertTrue(prefill.enabled)
        self.assertTrue(decode.requested)
        self.assertFalse(decode.enabled)
        requested.seed = 99
        with patch.dict(os.environ, {"MAXPANDER_ENABLE": "0", "MAXPANDER_SEED": "42"}):
            self.assertEqual(prefill.seed, 17)
            self.assertTrue(prefill.enabled)
        self.assertEqual(pickle.loads(pickle.dumps(prefill)), prefill)
        with self.assertRaises(FrozenInstanceError):
            prefill.enabled = False

    def test_disabled_keeps_all_roles_unchanged(self):
        for role in ("PREFILL", "DECODE", "FRONTEND", "PDFUSION", "VIT"):
            for seed, passes in ((0, 8), (-1, 0), (17, 65)):
                with self.subTest(role=role, seed=seed, passes=passes):
                    resolved = MaxPanderConfig(seed=seed, wf_passes=passes).resolve(
                        role
                    )
                    self.assertFalse(resolved.requested)
                    self.assertFalse(resolved.enabled)
                    self.assertEqual(
                        (resolved.seed, resolved.wf_passes), (seed, passes)
                    )
                    resolved.validate_placement(
                        expert_num=0, physical_expert_num=0, ep_size=1
                    )

    def test_decode_ignores_inactive_tuning(self):
        for seed, passes in ((-1, 8), (0, 0), (0, 65), (-1, 65)):
            with self.subTest(seed=seed, passes=passes):
                resolved = MaxPanderConfig(
                    enable=True, seed=seed, wf_passes=passes
                ).resolve("decode")
                self.assertTrue(resolved.requested)
                self.assertFalse(resolved.enabled)
                self.assertEqual(resolved.role, "DECODE")

    def test_frontend_inherits_switch_without_enabling_or_validating_tuning(self):
        config = MaxPanderConfig(enable=True, seed=-1, wf_passes=65).resolve("frontend")
        self.assertTrue(config.requested)
        self.assertFalse(config.enabled)
        self.assertEqual(config.role, "FRONTEND")

    def test_ft_style_restriction_only_applies_when_effectively_enabled(self):
        for role in ("PREFILL", "DECODE", "FRONTEND"):
            for requested in (False, True):
                config = MaxPanderConfig(enable=requested).resolve(role)
                config.validate_weight_format(is_ft_style=False)
                if config.enabled:
                    with self.assertRaisesRegex(ValueError, "ft-style"):
                        config.validate_weight_format(is_ft_style=True)
                else:
                    config.validate_weight_format(is_ft_style=True)

    def test_invalid_startup_settings(self):
        for role in ("PDFUSION", "VIT"):
            with self.subTest(role=role), self.assertRaisesRegex(
                ValueError, "dedicated"
            ):
                MaxPanderConfig(enable=True).resolve(role)
        for passes in (0, 65):
            with self.assertRaisesRegex(ValueError, "wf_passes"):
                MaxPanderConfig(enable=True, wf_passes=passes).resolve("PREFILL")
        with self.assertRaisesRegex(ValueError, "seed"):
            MaxPanderConfig(enable=True, seed=-1).resolve("PREFILL")

    def test_placement_bounds_and_conflicts(self):
        config = MaxPanderConfig(enable=True).resolve("PREFILL")
        for physical in (256, 320, 384, 512):
            config.validate_placement(
                expert_num=256, physical_expert_num=physical, ep_size=2
            )
        for kwargs in (
            {"ep_size": 1},
            {"expert_num": 255},
            {"physical_expert_num": 321},
            {"physical_expert_num": 514},
            {"dynamic_eplb": True},
            {"phy2log_path": "/not/read.json"},
        ):
            settings = dict(expert_num=256, physical_expert_num=320, ep_size=2)
            settings.update(kwargs)
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                config.validate_placement(**settings)


if __name__ == "__main__":
    unittest.main()
