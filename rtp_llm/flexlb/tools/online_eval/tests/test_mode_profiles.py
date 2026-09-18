import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mode_profiles import load_mode_tables, resolve_address_plan, resolve_mode


class ModeProfilesTest(unittest.TestCase):
    def test_four_master_modes_and_runtime_defaults(self):
        tables = load_mode_tables()
        self.assertEqual(set(tables["master_modes"]), {"sb", "sn", "wb", "wn"})
        for mode, decision, dispatcher in (
            ("sb", "SINGLE", "BATCH"), ("sn", "SINGLE", "NON_BATCH"),
            ("wb", "FIXED_WINDOW", "BATCH"),
            ("wn", "FIXED_WINDOW", "NON_BATCH"),
        ):
            plan = resolve_mode("scenario", mode)
            self.assertEqual((plan["decision"], plan["dispatcher"]),
                             (decision, dispatcher))
        self.assertEqual(resolve_mode("scenario", "sn")["observation"]["jsonl"], "bounded")
        self.assertEqual(resolve_mode("scenario", "sn")["features"]["orchestration"], "scenario")
        self.assertFalse(resolve_mode("whale_embedded", "sb", pod_ip="10.0.0.1")
                         ["observation"]["jsonl"])

    def test_reachability_is_independent_of_master_mode(self):
        self.assertTrue(resolve_address_plan("whale_embedded", pod_ip="10.0.0.1")
                        ["unique_engine_ips"])
        self.assertFalse(resolve_address_plan(
            "whale_embedded", external_engine_clients=True, pod_ip="10.0.0.1"
        )["unique_engine_ips"])
        with self.assertRaisesRegex(ValueError, "external engine clients"):
            resolve_address_plan("stress", external_engine_clients=True)
        with self.assertRaisesRegex(ValueError, "routable Pod IP"):
            resolve_address_plan("whale_independent", pod_ip="127.1.0.1")


if __name__ == "__main__":
    unittest.main()
