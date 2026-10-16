import os
import pickle
import unittest
from dataclasses import FrozenInstanceError
from unittest.mock import patch

from rtp_llm.config.maxpander_config import MaxPanderConfig


class MaxPanderConfigTest(unittest.TestCase):
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
        for role in ("PREFILL", "DECODE", "PDFUSION", "VIT"):
            self.assertFalse(MaxPanderConfig().resolve(role).enabled)

    def test_invalid_startup_settings(self):
        for role in ("PDFUSION", "VIT"):
            with self.subTest(role=role), self.assertRaisesRegex(
                ValueError, "dedicated"
            ):
                MaxPanderConfig(enable=True).resolve(role)
        for passes in (0, 65):
            with self.assertRaisesRegex(ValueError, "wf_passes"):
                MaxPanderConfig(wf_passes=passes).resolve("PREFILL")
        with self.assertRaisesRegex(ValueError, "seed"):
            MaxPanderConfig(seed=-1).resolve("PREFILL")

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
